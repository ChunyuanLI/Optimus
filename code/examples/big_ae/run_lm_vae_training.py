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

import numpy as np
import torch
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from collections import defaultdict

# from azure.cosmosdb.table.tableservice import TableService
# from azure.cosmosdb.table.models import Entity
from datetime import datetime



from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForLatentConnector, BertTokenizer,
                                  GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

from utils import (weight_init, calc_iwnll, calc_rec, calc_mi, calc_au, BucketingDataLoader, TextDataset_Split, TextDataset_2Tokenizers, frange_cycle_linear, frange_cycle_zero_linear)


from modules import VAE


# logging.getLogger("azure").setLevel(logging.WARNING)
# logging.getLogger("TableService").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}

    

def load_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        dataset = TextDataset_2Tokenizers(tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    else:
        dataset = TextDataset_Split(tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    return dataset

def build_dataload_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        if not evaluate:
            args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            file_path=args.train_data_file
        else:
            args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  
            file_path=args.eval_data_file
        dataloader = BucketingDataLoader(file_path, args.batch_size, args.max_seq_length, tokenizer, args, bucket=100, shuffle=False)
    else:
        pass 
    return dataloader




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    
    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).to(torch.uint8)
    labels[masked_indices==1] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(torch.uint8) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(torch.uint8) & masked_indices & ~indices_replaced
    indices_random = indices_random
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def weights_init_rondom(model):
    model = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        pdb.set_trace()
        if 'encoder' in key:
            init.normal_(model_state_dict[key].data)  
        # weight_init(item)

def save_checkpoint(model_vae, optimizer, global_step, args):

    # Create output directory if needed
    # Save model checkpoint
    output_encoder_dir = os.path.join(args.output_dir, 'checkpoint-encoder-{}'.format(global_step))
    output_decoder_dir = os.path.join(args.output_dir, 'checkpoint-decoder-{}'.format(global_step))
    if not os.path.exists(output_encoder_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_encoder_dir)
    if not os.path.exists(output_decoder_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_decoder_dir)

    logger.info("Saving encoder model checkpoint to %s", output_encoder_dir)
    logger.info("Saving decoder model checkpoint to %s", output_decoder_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`

    model_encoder_to_save = model_vae.module.encoder if hasattr(model_vae, 'module') else model_vae.encoder  # Take care of distributed/parallel training
    model_decoder_to_save = model_vae.module.decoder if hasattr(model_vae, 'module') else model_vae.decoder  # Take care of distributed/parallel training

    # Good practice: save your training arguments together with the trained model
    if args.use_philly:
        save_solid = False
        while not save_solid:
            try:
                model_encoder_to_save.save_pretrained(output_encoder_dir)
                torch.save(args, os.path.join(output_encoder_dir, 'training_encoder_args.bin'))
                save_solid = True
            except:
                pass
    else:
        model_encoder_to_save.save_pretrained(output_encoder_dir)
        torch.save(args, os.path.join(output_encoder_dir, 'training_encoder_args.bin'))


    if args.use_philly:
        save_solid = False
        while not save_solid:
            try:
                model_decoder_to_save.save_pretrained(output_decoder_dir)
                torch.save(args, os.path.join(output_decoder_dir, 'training_decoder_args.bin'))
                save_solid = True
            except:
                pass
    else:
        model_decoder_to_save.save_pretrained(output_decoder_dir)
        torch.save(args, os.path.join(output_decoder_dir, 'training_encoder_args.bin'))


    # save the full model and optmizer into a checkpoint
    model_to_save = model_vae.module if hasattr(model_vae, 'module') else model_vae  # Take care of distributed/parallel training

    checkpoint = {
    'iter': global_step,
    'model_state_dict': model_to_save.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'beta': model_to_save.args.beta,
    'args': args
    }

    output_full_dir = os.path.join(args.output_dir, 'checkpoint-full-{}'.format(global_step))
    if not os.path.exists(output_full_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_full_dir)

    logger.info("Start saving full model checkpoint to %s", output_full_dir)
    if args.use_philly:
        save_solid = False
        n_save_attempts = 0
        while not save_solid:
            try:
                n_save_attempts += 1
                logger.info(f"Saving full checkpoint: {n_save_attempts} attempts made")
                torch.save(checkpoint, os.path.join(output_full_dir, 'training.bin'))
                logger.info("Saving full checkpoint to %s,", output_full_dir)
                save_solid = True
            except:
                pass
    else:
        torch.save(checkpoint, os.path.join(output_full_dir, 'training.bin'))
        logger.info("Saving full checkpoint to %s", output_full_dir)



def load_checkpoint(args, loading_step=None):

    args.encoder_model_type = args.encoder_model_type.lower()
    args.decoder_model_type = args.decoder_model_type.lower()
    if loading_step:
        global_step = args.gloabl_step_eval
    else:
        global_step = args.gloabl_step_eval

    output_encoder_dir = os.path.join(args.checkpoint_dir, 'checkpoint-encoder-{}'.format(global_step))
    output_decoder_dir = os.path.join(args.checkpoint_dir, 'checkpoint-decoder-{}'.format(global_step)) 
    output_full_dir    = os.path.join(args.checkpoint_dir, 'checkpoint-full-{}'.format(global_step)) 

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

    # Load full model
    checkpoint = torch.load(os.path.join(output_full_dir, 'training.bin'))




def train(args, train_dataloader, model_vae, encoder_tokenizer, decoder_tokenizer, table_name):
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


    # model_encoder, model_decoder, model_connector = model_vae.encoder,  model_vae.decoder, model_vae.linear
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_vae.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model_vae.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)


    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model_vae, optimizer = amp.initialize(model_vae, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model_vae = torch.nn.DataParallel(model_vae, device_ids=range(args.n_gpu)).to(args.device)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_vae = torch.nn.parallel.DistributedDataParallel(model_vae, device_ids=[args.local_rank],
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
    model_vae.zero_grad()
   
    # model_vae = model_vae.module if hasattr(model_vae, 'module') else model_vae  # Take care of distributed/parallel training   
    
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    n_iter = int(args.num_train_epochs) * len(train_dataloader)
    beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=args.beta,  n_cycle=1, ratio_increase=args.ratio_increase, ratio_zero=args.ratio_zero)

    tmp_list = []
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            tokenized_text0, tokenized_text1, tokenized_text_lengths = batch
            # tokenized_text0 = tokenized_text0.to(args.device)
            # tokenized_text1 = tokenized_text1.to(args.device)
            # prepare input-output data for reconstruction

            # if (tokenized_text0>len(encoder_tokenizer)).sum().item()>0.0 or (tokenized_text1>len(decoder_tokenizer)).sum().item()>0.0: 
            #     pdb.set_trace()
            #     continue

            inputs, labels = mask_tokens(tokenized_text0, encoder_tokenizer, args) if args.mlm else (tokenized_text0, tokenized_text1)
            labels = tokenized_text1

            tokenized_text1 = tokenized_text1.to(args.device)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model_vae.train()

            beta_t = beta_t_list[step +  epoch*len(epoch_iterator)]
            model_vae.module.args.beta = beta_t

            if beta_t == 0.0:
                model_vae.module.args.fb_mode = 0
            else:
                model_vae.module.args.fb_mode = 1
            
            if args.use_deterministic_connect:
                model_vae.module.args.fb_mode = 2


            loss_rec, loss_kl, loss = model_vae(inputs, labels)


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
                    f'iter: {step +  epoch*len(epoch_iterator) }; loss: {loss.item():.3f}; '
                    f'loss_rec: {loss_rec.item():.3f}; loss_kl: {loss_kl.item():.3f}; '
                    f'beta: {model_vae.module.args.beta:.3f}'
                )
            )

            # if global_step % 5 == 0:
            #     row = {
            #             'PartitionKey': 'MILU_Rule_Rule_Template',
            #             'RowKey': str(datetime.now()),
            #             'ExpName' : args.ExpName, 
            #             'iter': str( step +  epoch*len(epoch_iterator) ),
            #             'loss': str( loss.item()),
            #             'loss_rec': str(loss_rec.item()),
            #             'loss_kl': str(loss_kl.item()),
            #             'beta': str(model_vae.args.beta)
            #         }
            #     # pdb.set_trace()
            #     ts.insert_entity(table_name, row)

            # pdb.set_trace()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()                                   
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model_vae.parameters(), args.max_grad_norm)

                optimizer.step()

                scheduler.step()  # Update learning rate schedule

                model_vae.zero_grad()

                global_step += 1


                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model_vae, encoder_tokenizer, decoder_tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(model_vae, optimizer, global_step, args)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

            
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, optimizer


def evaluate(args, model_vae, encoder_tokenizer, decoder_tokenizer, table_name, prefix="", subset="test"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    # if subset == 'test':
    #     eval_dataset = load_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    # elif subset == 'train':
    #     eval_dataset = load_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=False)
    logger.info("***** Running evaluation on {} dataset *****".format(subset))

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.per_gpu_eval_batch_size = 1
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model_vae.eval()

    model_vae =  model_vae.module if hasattr(model_vae, 'module') else model_vae  # Take care of distributed/parallel training

    mi = calc_mi(model_vae, eval_dataloader, args)
    au = calc_au(model_vae, eval_dataloader, delta=0.01, args=args)[0]
    ppl, elbo, nll, kl = calc_iwnll(model_vae, eval_dataloader, args, ns=100)

    result = {
        "perplexity": ppl, "elbo": elbo, "kl": kl, "nll": nll, "au": au, "mi": mi
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    row = {
            'PartitionKey': 'MILU_Rule_Rule_Template',
            'RowKey': str(datetime.now()),
            'ExpName' : args.ExpName, 
            'test_perplexity': str( ppl ),
            'test_elbo': str( elbo ),
            'test_nll': str(nll),
            'test_au': str(au),
            'test_mi': str(mi)
        }
    # pdb.set_trace()
    # ts.insert_entity(table_name, row)


    return result




def evaluate_rec(args, model_vae, encoder_tokenizer, decoder_tokenizer, table_name, prefix="", subset="test"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    if subset == 'test':
        eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    elif subset == 'train':
        eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=False)
    logger.info("***** Running evaluation on {} dataset *****".format(subset))

    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model_vae.eval()
    model_vae =  model_vae.module if hasattr(model_vae, 'module') else model_vae  # Take care of distributed/parallel training
    nll_s, nll_w = calc_rec(model_vae, eval_dataloader, args, ns=1)

    result = {
        "rec_w": nll_w, "rec_s": nll_s
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("%s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--checkpoint_dir", default=None, type=str,
                        help="The directory where checkpoints are saved.")                        
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default=None, type=str, help="The dataset.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--ExpName", default="", type=str,
                        help="The experiment name used in Azure Table.")
    parser.add_argument("--save_bert_gpt_init", action='store_true',
                        help="Use Philly for computing.")
    parser.add_argument("--length_weighted_loss", action='store_true',
                        help="Use sentence length re-weight the reconstruction loss.")


    ## Encoder options
    parser.add_argument("--encoder_model_type", default="bert", type=str,
                        help="The encoder model architecture to be fine-tuned.")
    parser.add_argument("--encoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The encoder model checkpoint for weights initialization.")
    parser.add_argument("--encoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--encoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Decoder options
    parser.add_argument("--decoder_model_type", default="gpt2", type=str,
                        help="The decoder model architecture to be fine-tuned.")
    parser.add_argument("--decoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The decoder model checkpoint for weights initialization.")
    parser.add_argument("--decoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--decoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Variational auto-encoder
    parser.add_argument("--latent_size", default=32, type=int, help="Latent space dimension.")
    parser.add_argument("--use_deterministic_connect", action='store_true',
                        help="Use deterministic inference to generate latent codes, i.e., standard auto-encoders.")
    parser.add_argument("--use_pretrained_model", action='store_true',
                        help="Use pre-trained auto-encoder models as the initialization")
    parser.add_argument("--latent_as_gpt_memory", default=1, type=int, help="Latent vector as memery for GPT2 to attend.")
    parser.add_argument("--latent_as_gpt_emb", default=1, type=int, help="Latent vector as embeddings for GPT2.")
    
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
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_rec", action='store_true',
                        help="Whether to run eval reconstruction on a set of models.")   
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
    parser.add_argument("--use_pretrained_vae", action='store_true',
                        help="Use use_pretrained_vae as initialization, where beta value is specified in the folder")
    parser.add_argument("--use_random_weight", action='store_true',
                        help="Use random weights as initialization")


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

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    # Load Optimius pre-trained model and tokenizer
    if args.use_pretrained_model:
        args.encoder_model_type = args.encoder_model_type.lower()
        args.decoder_model_type = args.decoder_model_type.lower()

        global_step = args.gloabl_step_eval

        output_encoder_dir = os.path.join(args.checkpoint_dir, 'checkpoint-encoder-{}'.format(global_step))
        output_decoder_dir = os.path.join(args.checkpoint_dir, 'checkpoint-decoder-{}'.format(global_step)) 
        output_full_dir    = os.path.join(args.checkpoint_dir, 'checkpoint-full-{}'.format(global_step)) 

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

        # Load full model
        checkpoint = torch.load(os.path.join(output_full_dir, 'training.bin'))

        
    else:
        # Load BERT and GPT weights (As an alternaive, one may train a VAE for this small)

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

        
        if args.latent_as_gpt_emb + args.latent_as_gpt_memory == 0:
            return # latent vector should pass into GPT to decode 
        else: 
            latent_as_gpt_emb = True if args.latent_as_gpt_emb == 1 else False
            latent_as_gpt_memory = True if args.latent_as_gpt_memory == 1 else False

        setattr(decoder_config, "latent_size", args.latent_size)
        model_decoder = decoder_model_class.from_pretrained(args.decoder_model_name_or_path, from_tf=bool('.ckpt' in args.decoder_model_name_or_path), config=decoder_config, latent_size=args.latent_size, latent_as_gpt_emb=latent_as_gpt_emb, latent_as_gpt_memory=latent_as_gpt_memory)
        
    # Save the init weights of BERT and GPT-2, so that we can load from local (Some infra requires so)
    if args.save_bert_gpt_init:
        encoder_path = os.path.join(args.output_dir, f"initial-models-tokenization-enoder-{args.latent_size}")
        if not os.path.exists(encoder_path): os.makedirs(encoder_path)
        model_encoder.save_pretrained(encoder_path)
        tokenizer_encoder.save_pretrained(encoder_path)

        decoder_path = os.path.join(args.output_dir, f"initial-models-tokenization-decoder-{args.latent_size}")
        if not os.path.exists(decoder_path): os.makedirs(decoder_path)
        model_decoder.save_pretrained(decoder_path)
        tokenizer_decoder.save_pretrained(decoder_path)

        return 



    # Chunyuan: Add Padding token to GPT2
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    model_decoder.resize_token_embeddings(len(tokenizer_decoder))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    assert tokenizer_decoder.pad_token == '<PAD>'

    # model_decoder.to(args.device)

    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)

    # pdb.set_trace()
    if args.use_random_weight:
        model_vae.apply(weights_init_rondom)

    if args.use_pretrained_model:
        model_vae.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Pre-trained Optimus is successfully loaded")
    model_vae.to(args.device) # 

    # on_gpu = next(model_vae.parameters()).is_cuda

    

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    
    ##############################
    # Training
    global_step= 0
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataloader = build_dataload_and_cache_examples(args, [tokenizer_encoder, tokenizer_decoder], evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss, optimizer = train(args, train_dataloader, model_vae, tokenizer_encoder, tokenizer_decoder, table_name)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        save_checkpoint(model_vae, optimizer, global_step, args)

        
    ##############################
    # Evaluation the metrics of VAE models, including PPL, MI, AU
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if global_step == 0:
            global_step = args.gloabl_step_eval

        output_encoder_dir = os.path.join(args.output_dir, 'checkpoint-encoder-{}'.format(global_step))
        output_decoder_dir = os.path.join(args.output_dir, 'checkpoint-decoder-{}'.format(global_step))
        output_full_dir    = os.path.join(args.output_dir, 'checkpoint-full-{}'.format(global_step))
        checkpoint_dir = [output_encoder_dir, output_decoder_dir, output_full_dir]

        logger.info("Evaluate the following checkpoint: %s", checkpoint_dir[-1])
        global_step = checkpoint_dir[-1].split('-')[-1] if len(checkpoint_dir) > 1 else ""

        checkpoint = torch.load(os.path.join(output_full_dir, 'training.bin'))
        model_vae.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Pre-trained Optimus is successfully loaded: {output_full_dir}")
        model_vae.to(args.device)

        result = evaluate(args, model_vae, tokenizer_encoder, tokenizer_decoder, table_name, prefix=global_step, subset='test')
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_vae_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(results.keys()):
                logger.info("%s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))
        logger.info(f"The testing results are successfully saved: {output_eval_file}")

    ##############################
    #  Evaluate the reconstruction loss for each checkpoints; 
    # This is used in studying two different latent vector injection schemes
    results = {}
    if args.do_eval_rec and args.local_rank in [-1, 0]:
        if global_step == 0:
            global_step = args.gloabl_step_eval
            # eval_steps = range(500, 13500, 500)
            # eval_steps = range(1000, 2000, 500)
            eval_steps = range(2000, 32000, 2000)

        checkpoints = []
        for e in eval_steps:
            output_encoder_dir = os.path.join(args.output_dir, 'checkpoint-encoder-{}'.format(e))
            output_decoder_dir = os.path.join(args.output_dir, 'checkpoint-decoder-{}'.format(e))
            checkpoints.append([output_encoder_dir, output_decoder_dir])

        

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint[0].split('-')[-1] if len(checkpoints) > 1 else ""

            model_encoder = encoder_model_class.from_pretrained(checkpoint[0], latent_size=args.latent_size)
            model_encoder.to(args.device)     
     
            model_decoder = decoder_model_class.from_pretrained(checkpoint[1])
            model_decoder.to(args.device)

            model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args).to(args.device)

            result = evaluate_rec(args, model_vae, tokenizer_encoder, tokenizer_decoder, table_name, prefix=global_step, subset='test')
            result = dict((k + '_test_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

            result = evaluate_rec(args, model_vae, tokenizer_encoder, tokenizer_decoder, table_name, prefix=global_step, subset='train')
            result = dict((k + '_train_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
            
            # pdb.set_trace()

        output_eval_file = os.path.join(args.output_dir, "eval_rec_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(results.keys()):
                logger.info("%s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))
        logger.info(f"The testing results are successfully saved: {output_eval_file}")


    return results


if __name__ == "__main__":
    main()
