# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function


import pdb
import argparse
import glob
import logging

import os
import pickle
import random

import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn import manifold
import matplotlib.pyplot as plt
# from azure.cosmosdb.table.tableservice import TableService
# from azure.cosmosdb.table.models import Entity
from datetime import datetime




# import sys
# sys.path.append('./')
# cwd = os.getcwd()
# pt_path = os.path.join( cwd[:-4], 'pytorch_transformers')
# sys.path.append(pt_path)
# print(f"Pytorch Transformer {pt_path}")



from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForLatentConnector, BertTokenizer,
                                  GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

from utils import (calc_iwnll, calc_mi, calc_au, Dialog_BucketingDataLoader, TextDataset_Split, TextDataset_2Tokenizers, frange_cycle_linear, frange_cycle_zero_linear)


from modules import SpaceFusion
from eval_dialog_response import eval_dialog_response
from eval_dialog_multi_response import eval_multi_ref

# logging.getLogger("azure").setLevel(logging.WARNING)
# logging.getLogger("TableService").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}

    
storage_name="textae"
key=r"6yBCXlblof8DVFJ4BD3eNFTrGQCej6cKfCf5z308cKnevyHaG+yl/m+ITVErB9yt0kvN3ToqxLIh0knJEfFmPA=="
# ts = TableService(account_name=storage_name, account_key=key)


def build_dataload_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        if not evaluate:
            args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            file_path=args.train_data_file
            use_shuffle = True
            bucket_size = 100
        else:
            args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  
            file_path=args.eval_data_file
            use_shuffle = False
            bucket_size = 1

        dataloader = Dialog_BucketingDataLoader(file_path, args.batch_size, args.max_seq_length, tokenizer, args, bucket=bucket_size, shuffle=use_shuffle)
    else:
        pass 
    return dataloader




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)




def dist_mat(x):
    return euclidean_distances(x, x)
    #return cosine_similarity(x, x)

def euc_dist_mat(x):
    n = x.shape[0]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum(np.power(x[i, :] - x[j, :], 2)))
            mat[i, j] = d
            mat[j, i] = d
    return mat


def visual2D(args, model_sf, inputs_src, inputs_tgt, n=200, method='MDS', path_prefix='vis_'):
    
    print('>'*10 + ' calculating z, n=%i'%n)
    model_sf.eval()
    with torch.no_grad():
        z_AE, z_S2S = model_sf(inputs_src[:n,:], inputs_tgt[:n,:], None, return_vec=True)
        z = torch.cat([z_AE, z_S2S], dim=0)
        latent = z.cpu().detach().numpy()
    labels = ['AE','S2S']
    
    colors = {
        'AE': 'r',
        'S2S': 'b',
        }

    print('>'*10 + ' calculating dist mat')
    # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    cmap = 'bwr' #, True:'hot'}#cubehelix'#'gnuplot2'# 
    dmat = dist_mat(latent)
    suffix = '_dist.png'
    f, ax = plt.subplots(figsize=(3*len(labels),2*len(labels)))
    cax = ax.imshow(dmat, cmap=cmap)
    f.colorbar(cax)

    """
    ticks = []
    ticklabels = []
    n_prev = 0
    for i in range(n_labels):
        ticks.append(n_prev + n/2)
        ticklabels.append(labels[i]+'\n')
        ticks.append(n_prev + n)
        ticklabels.append('%i'%(n * (i+1)))
        n_prev = n_prev + n
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.xaxis.tick_top()
    ax.set_yticks(ticks)
    ax.set_yticklabels([s.strip('\n') for s in ticklabels])
    """
    path_prefix = os.path.join(args.output_dir, path_prefix)
    plt.savefig(path_prefix + suffix)
    plt.close()

    print('>'*10 + ' runnning %s'%method)
    if method == 'tSNE':
        approx = manifold.TSNE(init='pca', verbose=1).fit_transform(latent)
    elif method == 'MDS':
        approx = manifold.MDS(2, verbose=1, max_iter=500, n_init=1).fit_transform(latent)
    elif method == 'isomap':
        approx = manifold.Isomap().fit_transform(latent)
    else:
        raise ValueError

    f, ax = plt.subplots()
    for k in labels:
        ax.plot(np.nan, np.nan, colors[k] + '.', label=k)
    
    i0 = 0
    for k in labels:
        i1 = i0 + n
        ax.plot(approx[i0:i1, 0], approx[i0:i1, 1], colors[k]+'.', alpha=0.5)
        i0 = i1
    
    plt.legend(loc='best')
    plt.savefig(path_prefix+'_%s.png'%method)


def train(args, train_dataloader, model_sf, encoder_tokenizer, decoder_tokenizer, table_name):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)


    # model_encoder, model_decoder, model_connector = model_sf.encoder,  model_sf.decoder, model_sf.linear
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_sf.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model_sf.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)


    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model_sf, optimizer = amp.initialize(model_sf, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model_sf = torch.nn.DataParallel(model_sf, device_ids=range(args.n_gpu)).to(args.device)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_sf = torch.nn.parallel.DistributedDataParallel(model_sf, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataloader.num_examples)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0


    model_sf.zero_grad()
   
    # model_sf = model_sf.module if hasattr(model_sf, 'module') else model_sf  # Take care of distributed/parallel training   
    
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    n_iter = int(args.num_train_epochs) * len(train_dataloader)
    beta_t_list = frange_cycle_zero_linear(n_iter, start=args.beta, stop=args.beta,  n_cycle=1, ratio_increase=args.ratio_increase, ratio_zero=args.ratio_zero)

    tmp_list = []
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # if step > 5:
            #     break

            input_ids_bert_ctx, input_ids_bert, input_ids_gpt, token_lengths = batch

            # if token_lengths[0,0]>512:
            #     input_ids_bert_ctx = input_ids_bert_ctx[0,:512].unsqueeze(0)

            # if token_lengths[0,1]>512:
            #     input_ids_bert_ctx = input_ids_bert_ctx[0,:512].unsqueeze(0)


            #logger.info(f'Conxtext in Bert, Length {token_lengths[0]} ; Tokens: {input_ids_bert_ctx}')
            #logger.info(f'Response in Bert, Length {token_lengths[1]} ; Tokens: {input_ids_bert}')
            #logger.info(f'Response in GPT2, Length {token_lengths[2]} ; Tokens: {input_ids_gpt}')

            #pdb.set_trace()
            model_sf.train()
            beta_t = beta_t_list[step +  epoch*len(epoch_iterator)]
            model_sf.module.args.beta = beta_t


            """
            xiag: not sure about fb_mode yet

            if beta_t == 0.0:
                model_sf.args.fb_mode = 0
            else:
                model_sf.args.fb_mode = 1
            
            if args.use_deterministic_connect:
                model_sf.args.fb_mode = 2
                """
        
            input_ids_bert_ctx = input_ids_bert_ctx.to(args.device)
            input_ids_bert = input_ids_bert.to(args.device)
            input_ids_gpt = input_ids_gpt.to(args.device)

            loss_rec, loss_kl, loss = model_sf(input_ids_bert_ctx, input_ids_bert, input_ids_gpt)
            

            # the following is copied from run_lm_vae_pretraining.py

            # Chunyuan: loss_rec size is [4], while latent_z size is [12]
            if args.n_gpu > 1:
                loss_rec = loss_rec.mean()  # mean() to average on multi-gpu parallel training
                loss_kl = loss_kl.mean()
                loss = loss.mean()

            if args.use_philly:
                print("PROGRESS: {}%".format(round(100 * (step +  epoch*len(epoch_iterator) ) /(int(args.num_train_epochs) *  len(epoch_iterator)) , 4))) 
                print("EVALERR: {}%".format(loss_rec)) 

            epoch_iterator.set_description(
                (
                    f'iter: {step +  epoch*len(epoch_iterator) }; loss: {loss.mean().item():.3f}; '
                    f'loss_rec: {loss_rec.mean().item():.3f}; loss_kl: {loss_kl.mean().item():.3f}; '
                    f'beta: {model_sf.module.args.beta:.3f}'
                )
            )

            if global_step % 5 == 0:
                row = {
                        'PartitionKey': 'MILU_Rule_Rule_Template',
                        'RowKey': str(datetime.now()),
                        'ExpName' : args.ExpName, 
                        'iter': str( step +  epoch*len(epoch_iterator) ),
                        'loss': str( loss.mean().item()),
                        'loss_rec': str(loss_rec.mean().item()),
                        'loss_kl': str(loss_kl.mean().item()),
                        'beta': str(model_sf.module.args.beta)
                    }
                # pdb.set_trace()
                #ts.insert_entity(table_name, row)

            # pdb.set_trace()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()                                   
            else:
                loss = loss.mean()
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model_sf.parameters(), args.max_grad_norm)

                optimizer.step()

                scheduler.step()  # Update learning rate schedule

                model_sf.zero_grad()

                global_step += 1


                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model_sf, encoder_tokenizer, decoder_tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    # Save encoder model checkpoint
                    output_encoder_dir = os.path.join(args.output_dir, 'checkpoint-encoder-{}'.format(global_step))

                    if not os.path.exists(output_encoder_dir):
                        os.makedirs(output_encoder_dir)

                    model_encoder_to_save = model_sf.module.encoder if hasattr(model_sf, 'module') else model_sf.encoder  # Take care of distributed/parallel training
                    if args.use_philly:
                        save_solid = False
                        while not save_solid:
                            try:
                                model_encoder_to_save.save_pretrained(output_encoder_dir)
                                torch.save(args, os.path.join(output_encoder_dir, 'training_args.bin'))
                                logger.info("Saving model checkpoint to %s", output_encoder_dir)
                                save_solid = True
                            except:
                                pass
                    else:
                        model_encoder_to_save.save_pretrained(output_encoder_dir)
                        torch.save(args, os.path.join(output_encoder_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_encoder_dir)

                    # Save decoder model checkpoint
                    output_decoder_dir = os.path.join(args.output_dir, 'checkpoint-decoder-{}'.format(global_step))

                    if not os.path.exists(output_decoder_dir):
                        os.makedirs(output_decoder_dir)

                    model_decoder_to_save = model_sf.module.decoder if hasattr(model_sf, 'module') else model_sf.decoder  # Take care of distributed/parallel training
                    if args.use_philly:
                        save_solid = False
                        while not save_solid:
                            try:
                                model_decoder_to_save.save_pretrained(output_decoder_dir)
                                torch.save(args, os.path.join(output_decoder_dir, 'training_args.bin'))
                                logger.info("Saving model checkpoint to %s", output_decoder_dir)
                                save_solid = True
                            except:
                                pass
                    else:
                        model_decoder_to_save.save_pretrained(output_decoder_dir)
                        torch.save(args, os.path.join(output_decoder_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_decoder_dir)


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

            
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step#, tr_loss / global_step


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits



def top_k_top_p_filtering_mb(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def sample_sequence_conditional(model, length, context, past=None, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu', decoder_tokenizer=None):
    
    generated = context
    with torch.no_grad():
        while True:
        # for _ in trange(length):
            inputs = {'input_ids': generated, 'past': past}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering_mb(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

            # pdb.set_trace()
            if next_token.unsqueeze(0)[0,0].item() == decoder_tokenizer.encode('<EOS>')[0] or generated.shape[1] > length :
                break

        # gpt_eos_id = decoder_tokenizer.encode('<EOS>')[0]
        # idx = (generated == gpt_eos_id).nonzero().squeeze()

    # pdb.set_trace()
    return generated


def evaluate(args, model_sf, encoder_tokenizer, decoder_tokenizer, table_name, prefix="", subset="test"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    logger.info("***** Running evaluation on {} dataset *****".format(subset))

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    # args.per_gpu_eval_batch_size = 1
    args.n_gpu = 1
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model_sf.eval()

    count = 0
    result = []

    epoch_iterator = tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        input_ids_bert_ctx, input_ids_bert, input_ids_gpt, token_lengths = batch

        input_ids_bert_ctx = input_ids_bert_ctx.to(args.device)
        input_ids_bert = input_ids_bert.to(args.device)
        input_ids_gpt = input_ids_gpt.to(args.device)

        if len(input_ids_bert_ctx[0,:])>512:
            input_ids_bert_ctx = input_ids_bert_ctx[0,-512:].unsqueeze(0)
        
        # else: 
        #     continue

        # pdb.set_trace()

        # if step == 0:
        #     input_ids_bert_ctx_previous = input_ids_bert_ctx
        # else:
        #     # pdb.set_trace()
        #     if (input_ids_bert_ctx_previous.shape == input_ids_bert_ctx.shape) and torch.eq(input_ids_bert_ctx_previous, input_ids_bert_ctx)[0].type(torch.float).mean().item() == 1.0:
        #         continue
        #     else:
        #         input_ids_bert_ctx_previous = input_ids_bert_ctx
        #         print(step)

        
        context_tokens = decoder_tokenizer.encode('<BOS>')
        context_tokens = torch.tensor(context_tokens, dtype=torch.long, device=args.device)
        context_tokens = context_tokens.unsqueeze(0).repeat(token_lengths.shape[0], 1)

        with torch.no_grad():

            text_src = encoder_tokenizer.decode(input_ids_bert_ctx[0,:].tolist(), clean_up_tokenization_spaces=False)
            text_src = "".join(text_src)

            text_ref = encoder_tokenizer.decode(input_ids_bert[0,:].tolist(), clean_up_tokenization_spaces=False)
            text_ref = "".join(text_ref)

            for i in range(args.sents_per_cxt):
                latent_z = model_sf.sent2latent(input_ids_bert_ctx)

                out = sample_sequence_conditional(
                    model=model_sf.decoder,
                    context=context_tokens,
                    past=latent_z,
                    length=256, # Chunyuan: Fix length; or use <EOS> to complete a sentence
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=args.device,
                    decoder_tokenizer = decoder_tokenizer
                )
                text_hpy = decoder_tokenizer.decode(out[0,:].tolist(), clean_up_tokenization_spaces=False)
                
                text_hpy = text_hpy.split()[1:-1]
                text_hpy = ' '.join(text_hpy) + '\n'

                textline = "\t".join([text_src, text_ref, text_hpy])
                # pdb.set_trace()
                result.append(textline)


            epoch_iterator.set_description(
                (
                    f'step: {step}'
                )
            )           

        count += 1
        if args.total_sents>0 and count>args.total_sents:
            break   


    output_eval_file = os.path.join(eval_output_dir, "eval_text_generation_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for res in result:
            # logger.info("%s \n" % res)
            writer.write("%s \n" % res)

    return result






def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")                    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default=None, type=str, help="The dataset.")
    parser.add_argument("--checkpoint_dir", default=None, type=str, required=True,
                        help="The directory where checkpoints are saved.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to run text generation.")
    parser.add_argument("--eval_generated_text_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a generated text file).")                    
    parser.add_argument("--ExpName", default="", type=str,
                        help="The experiment name used in Azure Table.")

    ## Encoder options
    parser.add_argument("--encoder_model_type", default="bert", type=str,
                        help="The encoder model architecture to be fine-tuned.")
    parser.add_argument("--encoder_model_name_or_path", default="bert-base-uncased", type=str,
                        help="The encoder model checkpoint for weights initialization.")
    parser.add_argument("--encoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--encoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Decoder options
    parser.add_argument("--decoder_model_type", default="gpt2", type=str,
                        help="The decoder model architecture to be fine-tuned.")
    parser.add_argument("--decoder_model_name_or_path", default="gpt2", type=str,
                        help="The decoder model checkpoint for weights initialization.")
    parser.add_argument("--decoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--decoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Space Fusion
    parser.add_argument("--latent_size", default=32, type=int, help="Latent space dimension.")
    parser.add_argument("--use_deterministic_connect", action='store_true',
                        help="Use deterministic inference to generate latent codes, i.e., standard auto-encoders.")
    parser.add_argument("--use_pretrained_model", action='store_true',
                        help="Use pre-trained auto-encoder models as the initialization")
    parser.add_argument("--use_pretrained_vae", action='store_true',
                        help="Use use_pretrained_vae as initialization, where beta value is specified in the folder")
    parser.add_argument("--num_s2s_bert_layer", default=1, type=int, help="Number of BERT layer used for S2S loass in space fusion.")
    parser.add_argument("--num_frozen_bert_layer", default=11, type=int, help="Number of BERT layer used for S2S loass in space fusion")
                        
    parser.add_argument('--freeze_bert', action='store_true')
    parser.add_argument('--n_pnt', type=int, default=200)
    parser.add_argument('--path_ids', type=str, default='dailydialog_data_1000.pt')


    ## Objective functions
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="The weighting hyper-parameter of the KL term in VAE")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Optional input sequence length before tokenization. The sequence will be dropped if it is longer the max_seq_length")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_generation", action='store_true',
                        help="Whether to run text generation on the dev set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_vis", action='store_true',
                        help="Whether to run visualization on the latent space.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")


    # Training Schedules
    parser.add_argument("--ratio_increase", default=0.25, type=float,
                        help="Learning schedule, the percentage for the annealing stage.") 
    parser.add_argument("--ratio_zero", default=0.25, type=float,
                        help="Learning schedule, the percentage for the pure auto-encoding stage.")     
    parser.add_argument("--fb_mode", default=0, type=int,
                        help="free bit training mode.")   
    parser.add_argument("--dim_target_kl", default=3.0, type=float,
                        help="dim_target_kl free bit training mode.")                            
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--use_philly", action='store_true',
                        help="Use Philly for computing.")

    ## IO: Logging and Saving
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gloabl_step_eval', type=int, default=661,
                        help="Evaluate the results at the given global step")

    # Text Generation
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--total_sents", default=10, type=int, help="Total sentences to test recontruction.")
    parser.add_argument("--sents_per_cxt", default=10, type=int, help="Number of responses to generate for a given context.")


    # Precision & Distributed Training 
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if args.decoder_model_type in ["bert", "roberta"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    args.ExpName = 'Vae_' + args.dataset + '_Nz_' + str(args.latent_size)  + '_Beta_'  + str(args.beta) + '_Dkl_' + str(args.dim_target_kl) + '_Ra_' + str(args.ratio_increase) + '_R0_' + str(args.ratio_zero) 
    table_name = 'Vae' + args.dataset + 'Nz' + str(args.latent_size) 
    try: 
        ts.create_table(table_name)
    except:
        pass


    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.do_train or args.do_generation or args.do_vis: 
        if args.use_pretrained_model:

            args.encoder_model_type = args.encoder_model_type.lower()
            args.decoder_model_type = args.decoder_model_type.lower()

            global_step = args.gloabl_step_eval
            if args.use_pretrained_vae:
                output_encoder_dir = os.path.join(args.checkpoint_dir, 'checkpoint-encoder-{}-1.0'.format(global_step))
                output_decoder_dir = os.path.join(args.checkpoint_dir, 'checkpoint-decoder-{}-1.0'.format(global_step)) 
            else:
                output_encoder_dir = os.path.join(args.checkpoint_dir, 'checkpoint-encoder-{}'.format(global_step))
                output_decoder_dir = os.path.join(args.checkpoint_dir, 'checkpoint-decoder-{}'.format(global_step)) 

            checkpoints = [ [output_encoder_dir, output_decoder_dir] ]
            logger.info("Evaluate the following checkpoints: %s", checkpoints)

            # Load a trained Encoder model and vocabulary
            encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES[args.encoder_model_type]
            model_encoder = encoder_model_class.from_pretrained(output_encoder_dir, latent_size=args.latent_size)
            tokenizer_encoder = encoder_tokenizer_class.from_pretrained(args.encoder_tokenizer_name if args.encoder_tokenizer_name else args.encoder_model_name_or_path, do_lower_case=args.do_lower_case)

            model_encoder.to(args.device)
            if args.block_size <= 0:
                args.block_size = tokenizer_encoder.max_len_single_sentence  # Our input block size will be the max possible for the model
            args.block_size = min(args.block_size, tokenizer_encoder.max_len_single_sentence)

            # Load a trained Decoder model and vocabulary
            decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[args.decoder_model_type]
            model_decoder = decoder_model_class.from_pretrained(output_decoder_dir, latent_size=args.latent_size)
            tokenizer_decoder = decoder_tokenizer_class.from_pretrained(args.decoder_tokenizer_name if args.decoder_tokenizer_name else args.decoder_model_name_or_path, do_lower_case=args.do_lower_case)
            model_decoder.to(args.device)
            if args.block_size <= 0:
                args.block_size = tokenizer_decoder.max_len_single_sentence  # Our input block size will be the max possible for the model
            args.block_size = min(args.block_size, tokenizer_decoder.max_len_single_sentence)

        else:
            ## Encoder 
            encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES[args.encoder_model_type]
            encoder_config = encoder_config_class.from_pretrained(args.encoder_config_name if args.encoder_config_name else args.encoder_model_name_or_path)
            tokenizer_encoder = encoder_tokenizer_class.from_pretrained(args.encoder_tokenizer_name if args.encoder_tokenizer_name else args.encoder_model_name_or_path, do_lower_case=args.do_lower_case)
            if args.block_size <= 0:
                args.block_size = tokenizer_encoder.max_len_single_sentence  # Our input block size will be the max possible for the model
            args.block_size = min(args.block_size, tokenizer_encoder.max_len_single_sentence)
            model_encoder = encoder_model_class.from_pretrained(args.encoder_model_name_or_path, from_tf=bool('.ckpt' in args.encoder_model_name_or_path), config=encoder_config, latent_size=args.latent_size)
            # model_encoder.to(args.device)

            ## Decoder 
            decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[args.decoder_model_type]
            decoder_config = decoder_config_class.from_pretrained(args.decoder_config_name if args.decoder_config_name else args.decoder_model_name_or_path)
            tokenizer_decoder = decoder_tokenizer_class.from_pretrained(args.decoder_tokenizer_name if args.decoder_tokenizer_name else args.decoder_model_name_or_path, do_lower_case=args.do_lower_case)
            if args.block_size <= 0:
                args.block_size = tokenizer_decoder.max_len_single_sentence  # Our input block size will be the max possible for the model
            args.block_size = min(args.block_size, tokenizer_decoder.max_len_single_sentence)
            setattr(decoder_config, "latent_size", args.latent_size)
            model_decoder = decoder_model_class.from_pretrained(args.decoder_model_name_or_path, from_tf=bool('.ckpt' in args.decoder_model_name_or_path), config=decoder_config, latent_size=args.latent_size, latent_as_gpt_emb=False)
            
        # Chunyuan: Add Padding token to GPT2
        special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
        print('We have added', num_added_toks, 'tokens to GPT2')
        model_decoder.resize_token_embeddings(len(tokenizer_decoder))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        assert tokenizer_decoder.pad_token == '<PAD>'

        model_sf = SpaceFusion(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args).to(args.device) # 


    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    
    # Training
    if args.do_train:
        global_step= 0
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataloader = build_dataload_and_cache_examples(args, [tokenizer_encoder, tokenizer_decoder], evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step = train(args, train_dataloader, model_sf, tokenizer_encoder, tokenizer_decoder, table_name)
        logger.info(" global_step = %s", global_step)

    # Text Generation based on a trained model
    if args.do_generation and args.local_rank in [-1, 0]:
        results = {}
        model_sf = SpaceFusion(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args).to(args.device) # 
        result = evaluate(args, model_sf, tokenizer_encoder, tokenizer_decoder, table_name, prefix=global_step, subset='test')

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:

        if args.dataset == "dailydialog":
            results = eval_dialog_response(args.eval_generated_text_file)
        else:
            results = eval_multi_ref(args.eval_generated_text_file, args.eval_data_file)


        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(results.keys()):
                logger.info("%s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))

    # Visualization of the latent space
    if args.do_vis and args.local_rank in [-1, 0]:

        print('>'*10 + ' loading ids')
        ids = torch.load(args.path_ids)
        inputs_src = ids['input_ids_bert_ctx']
        inputs_tgt = ids['input_ids_bert']
        model_sf = SpaceFusion(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args).to(args.device) # 
        visual2D(args, model_sf, inputs_src, inputs_tgt, n=args.n_pnt)


if __name__ == "__main__":
    main()