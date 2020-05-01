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
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from collections import defaultdict
# from azure.cosmosdb.table.tableservice import TableService
# from azure.cosmosdb.table.models import Entity
from datetime import datetime
import sys
import json
import nltk
nltk.download('punkt')

sys.path.append('../../')
from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForLatentConnector, BertTokenizer,
                                  GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
from utils import (TextDataset_Split, TextDataset_2Tokenizers_LCtrlG,
                   frange_cycle_linear, frange_cycle_zero_linear, AverageValueMeter)
# from modules import ARAE
from modules import CARA
# logging.getLogger("azure").setLevel(logging.WARNING)
# logging.getLogger("TableService").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
import time
def get_time_str():
    return time.ctime().replace(' ', '_').replace(':', '-')

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}


storage_name="textae"
key=r"6yBCXlblof8DVFJ4BD3eNFTrGQCej6cKfCf5z308cKnevyHaG+yl/m+ITVErB9yt0kvN3ToqxLIh0knJEfFmPA=="
# ts = TableService(account_name=storage_name, account_key=key)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        dataset = TextDataset_2Tokenizers_LCtrlG(tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file,
                                                 block_size=args.block_size, create_new=args.create_new)
    else:
        raise NotImplementedError
        # dataset = TextDataset_Split(tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    return dataset

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

def train(args, train_dataset, model_vae, encoder_tokenizer, decoder_tokenizer, table_name, logff):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
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
        model_vae = torch.nn.parallel.DistributedDataParallel(model_vae, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    # model_vae = model_vae.module if hasattr(model_vae, 'module') else model_vae  # Take care of distributed/parallel training

    # Train!
    logger.info("***** Running training *****")
    logff.write("***** Running training *****\n")
    logger.info("  Num examples = {}".format(len(train_dataset)))
    logff.write("  Num examples = {}\n".format(len(train_dataset)))
    logger.info("  Num Epochs = {}".format(args.num_train_epochs))
    logff.write("  Num Epochs = {}\n".format(args.num_train_epochs))
    logger.info("  Instantaneous batch size per GPU = {}".format(args.per_gpu_train_batch_size))
    logff.write("  Instantaneous batch size per GPU = {}\n".format(args.per_gpu_train_batch_size))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logff.write("  Total train batch size (w. parallel, distributed & accumulation) = {}\n".format(
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)))
    logger.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    logff.write("  Gradient Accumulation steps = {}\n".format(args.gradient_accumulation_steps))
    logger.info("  Total optimization steps = {}".format( t_total))
    logff.write("  Total optimization steps = {}\n".format(t_total))
    logff.flush()
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model_vae.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    n_iter = int(args.num_train_epochs) * len(train_dataloader)
    beta_t_list = frange_cycle_zero_linear(n_iter, start=1.0, stop=args.beta_cls,  n_cycle=1, ratio_increase=args.ratio_increase, ratio_zero=args.ratio_zero)

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    accmeter = {
        'acc_encode_z_dis': AverageValueMeter(),
        'acc_gen_z_dis': AverageValueMeter(),
        'acc_encode_z_cls': AverageValueMeter(),
        'acc_cls': AverageValueMeter(),
        # 'acc_at_soft_cls': AverageValueMeter(),
    }
    lossmeter = {
        'loss': AverageValueMeter(),
        'loss_rec': AverageValueMeter(),
        'loss_encoder': AverageValueMeter(),
        'loss_lsc': AverageValueMeter(),
        'loss_lsd': AverageValueMeter(),
        'loss_lsg': AverageValueMeter(),
        'loss_cls': AverageValueMeter(),
        # 'loss_at_soft_cls': AverageValueMeter(),
    }
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # pbar = tqdm(total=(len(train_dataloader)+1) // args.gradient_accumulation_steps)
        for step, batch in enumerate(train_dataloader):

            # if step > 100:
            #     break

            # Data
            input_seq_ids, tgt_seq_ids, tokenized_text_lengths, cond_labels = batch
            max_len_values, _ = tokenized_text_lengths.max(0)
            input_seq_ids = input_seq_ids[:,:max_len_values[0]]
            tgt_seq_ids = tgt_seq_ids[:,:max_len_values[1]]
            input_seq_ids, tgt_seq_ids = mask_tokens(input_seq_ids, encoder_tokenizer, args) if args.mlm else (input_seq_ids, tgt_seq_ids)
            input_seq_ids = input_seq_ids.to(args.device)
            tgt_seq_ids = tgt_seq_ids.to(args.device)
            cond_labels = cond_labels.to(args.device)
            input_mask = torch.where(torch.arange(max_len_values[0].item()).unsqueeze(0).repeat(input_seq_ids.size(0), 1).type_as(tokenized_text_lengths).to(args.device)
                                     < tokenized_text_lengths[:, 0].unsqueeze(1).to(args.device), torch.ones_like(input_seq_ids), torch.zeros_like(input_seq_ids))

            # Configs
            model_vae.train()
            beta_t = beta_t_list[step +  epoch*len(epoch_iterator)]
            model_vae.module.args.beta_cls = beta_t
            # if beta_t == 0.0:
            #     model_vae.args.fb_mode = 0
            # else:
            #     model_vae.args.fb_mode = 1
            # if args.use_deterministic_connect:
            #     model_vae.args.fb_mode = 2

            # Model
            loss_dict, acc_dict = model_vae(input_seq_ids=input_seq_ids, tgt_seq_ids=tgt_seq_ids, cond_labels=cond_labels, attention_mask=input_mask)

            # Loss
            for key, value in loss_dict.items():
                loss_dict[key] = value.mean()

            loss = loss_dict['loss']
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()

            # Log
            for key, value in loss_dict.items():
                lossmeter[key].add(value.item())

            for key, value in acc_dict.items():
                value = value.cpu().tolist()
                for v in value:
                    accmeter[key].add(float(v))

            # Optimize
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Optimize
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model_vae.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model_vae.zero_grad()
                global_step += 1
                # pbar.update(1)

                # Log
                if global_step % args.logging_steps == 0:
                    logger.info("\n")
                    logger.info("global_step: {}, avg loss: {:3f}".format(global_step, tr_loss/global_step))
                    logff.write("global_step: {}, avg loss: {:3f}\n".format(global_step, tr_loss/global_step))
                    logger.info("loss: {}".format(', '.join(key + ': ' + str(round(meter.mean, 3)) for key, meter in lossmeter.items())))
                    logff.write("loss: {}\n".format(', '.join(key + ': ' + str(round(meter.mean, 3)) for key, meter in lossmeter.items())))
                    logger.info("acc: {}".format(', '.join(key + ': ' + str(round(meter.mean, 3)) for key, meter in accmeter.items())))
                    logff.write("acc: {}\n".format(', '.join(key + ': ' + str(round(meter.mean, 3)) for key, meter in accmeter.items())))
                    logff.flush()


                if args.use_philly:
                    #if args.local_rank in [-1, 0]:
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logger.info("PROGRESS: {}%".format(round(100 * (step +  epoch*len(train_dataloader) ) /(int(args.num_train_epochs) *  len(train_dataloader)) , 4)))
                        logger.info("EVALERR: {}%".format(tr_loss / global_step))


                if args.local_rank in [-1, 0] and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model_vae, encoder_tokenizer, decoder_tokenizer, table_name, epoch=epoch)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.eval_steps, global_step)
                    logging_loss = tr_loss

                # Save checkpoints
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save encoder model checkpoint
                    output_encoder_dir = os.path.join(args.output_dir, 'checkpoint-encoder-{}'.format(global_step))
                    if not os.path.exists(output_encoder_dir):
                        os.makedirs(output_encoder_dir)
                    model_encoder_to_save = model_vae.module.encoder if hasattr(model_vae, 'module') else model_vae.encoder  # Take care of distributed/parallel training
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
                    model_decoder_to_save = model_vae.module.decoder if hasattr(model_vae, 'module') else model_vae.decoder  # Take care of distributed/parallel training
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
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model_vae, encoder_tokenizer, decoder_tokenizer, table_name, prefix="", subset="test", epoch=None):

    eval_output_dir = args.output_dir

    if subset == 'test':
        eval_dataset = load_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    elif subset == 'train':
        eval_dataset = load_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=False)
    else:
        raise ValueError
        
    args.label_size = len(eval_dataset.get_labels())

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Num steps = %d", len(eval_dataset) // args.eval_batch_size)
    logger.info("  eval_output_dir = %s", eval_output_dir)

    model_vae.eval()
    model_vae_module =  model_vae.module if hasattr(model_vae, 'module') else model_vae  # Take care of distributed/parallel training

    outputs = {
        'sampled_cond_labels': None,
        'cond_labels': None,
        'tgt_seq_ids': None,
        'generated': None,
        'at_generated': None,
        'cg_generated': None,
        'pred_cls': None,
        'pred_ge_cls': None,
        'pred_at_cls': None,
        'pred_cg_cls': None,
    }

    for bi, batch in enumerate(tqdm(eval_dataloader, desc="#Sentences", disable=args.local_rank not in [-1, 0]) ):
        # if bi == 3:
        #     break

        # Data
        input_seq_ids, tgt_seq_ids, tokenized_text_lengths, cond_labels = batch
        max_len_values, _ = tokenized_text_lengths.max(0)
        input_seq_ids = input_seq_ids[:,:max_len_values[0]]
        tgt_seq_ids = tgt_seq_ids[:,:max_len_values[1]]
        input_seq_ids = input_seq_ids.to(args.device)
        tgt_seq_ids = tgt_seq_ids.to(args.device)
        cond_labels = cond_labels.to(args.device)
        input_mask = torch.where(torch.arange(max_len_values[0].item()).unsqueeze(0).repeat(input_seq_ids.size(0), 1).type_as(tokenized_text_lengths).to(args.device)
                                     < tokenized_text_lengths[:, 0].unsqueeze(1).to(args.device), torch.ones_like(input_seq_ids), torch.zeros_like(input_seq_ids))

        # Model
        with torch.no_grad():
            result = model_vae(input_seq_ids=input_seq_ids, tgt_seq_ids=tgt_seq_ids, cond_labels=cond_labels, attention_mask=input_mask)
        if bi == 0:
            for key in outputs.keys():
                outputs[key] = result[key].cpu().tolist()
        else:
            for key in outputs.keys():
                outputs[key].extend(result[key].cpu().tolist())

    # compute accuracies and store in results
    acc = np.mean(np.array(np.array(outputs['pred_cls']) == np.array(outputs['cond_labels']), dtype=np.float))
    acc_ge = np.mean(np.array(np.array(outputs['pred_ge_cls']) == np.array(outputs['cond_labels']), dtype=np.float))
    acc_at = np.mean(np.array(np.array(outputs['pred_at_cls']) == np.array(outputs['sampled_cond_labels']), dtype=np.float))
    acc_cg = np.mean(np.array(np.array(outputs['pred_cg_cls']) == np.array(outputs['sampled_cond_labels']), dtype=np.float))
    metrics = {'acc': acc, 'acc_ge': acc_ge, 'acc_at': acc_at, 'acc_cg': acc_cg}

    # dump generated outputs to file.
    json.dump(outputs, open(os.path.join(eval_output_dir, "outputs_{}.json".format(epoch) if epoch is not None else "outputs.json"), 'w'))

    # compute BLEU
    bos_token_id = model_vae_module.tokenizer_decoder.encode('<BOS>')[0]
    eos_token_id = model_vae_module.tokenizer_decoder.encode('<EOS>')[0]
    pad_token_id = model_vae_module.tokenizer_decoder.encode('<PAD>')[0]

    generated_ids = []
    generated_text = []
    for g in outputs['generated']:
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        g = g[:g.index(eos_token_id)] if eos_token_id in g else g
        g = g[:g.index(pad_token_id)] if pad_token_id in g else g
        g_text = model_vae_module.tokenizer_decoder.decode(g, clean_up_tokenization_spaces=True)
        generated_ids.append(g)
        generated_text.append(g_text)

    tgt_seq_ids = []
    tgt_seq_text = []
    for g in outputs['tgt_seq_ids']:
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        g = g[:g.index(eos_token_id)] if eos_token_id in g else g
        g = g[:g.index(pad_token_id)] if pad_token_id in g else g
        g_text = model_vae_module.tokenizer_decoder.decode(g, clean_up_tokenization_spaces=True)
        tgt_seq_ids.append(g)
        tgt_seq_text.append(g_text)

    at_generated_ids = []
    at_generated_text = []
    for g in outputs['at_generated']:
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        g = g[:g.index(eos_token_id)] if eos_token_id in g else g
        g = g[:g.index(pad_token_id)] if pad_token_id in g else g
        g_text = model_vae_module.tokenizer_decoder.decode(g, clean_up_tokenization_spaces=True)
        at_generated_ids.append(g)
        at_generated_text.append(g_text)

    cg_generated_ids = []
    cg_generated_text = []
    for g in outputs['cg_generated']:
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        g = g[:g.index(eos_token_id)] if eos_token_id in g else g
        g = g[:g.index(pad_token_id)] if pad_token_id in g else g
        g_text = model_vae_module.tokenizer_decoder.decode(g, clean_up_tokenization_spaces=True)
        cg_generated_ids.append(g)
        cg_generated_text.append(g_text)

    f = open(os.path.join(eval_output_dir, "reconstruction{}.txt".format(('_'+str(epoch)) if epoch is not None else '')), 'w')
    f.write('\n'.join([g + '\n' + t for g, t in zip(generated_text, tgt_seq_text)]))
    fat = open(os.path.join(eval_output_dir, "attribute_transfer{}.txt".format(('_'+str(epoch)) if epoch is not None else '')), 'w')
    fat.write('\n'.join([g + '\n' + t for g, t in zip(at_generated_text, tgt_seq_text)]))
    fcg = open(os.path.join(eval_output_dir, "conditional_generation{}.txt".format(('_'+str(epoch)) if epoch is not None else '')), 'w')
    fcg.write('\n'.join(cg_generated_text))

    rec_bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references=[[nltk.word_tokenize(t)] for t in tgt_seq_text],
                                                     hypotheses=[nltk.word_tokenize(g) for g in generated_text])

    at_bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references=[[nltk.word_tokenize(t)] for t in tgt_seq_text],
                                                    hypotheses=[nltk.word_tokenize(g) for g in at_generated_text])

    cg_generated_text_subset = cg_generated_text[:500]  # use a subset, otherwise it takes a long time to compute.
    cg_bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references=[[nltk.word_tokenize(t) for t in tgt_seq_text] for _ in range(len(cg_generated_text_subset))],
                                                    hypotheses=[nltk.word_tokenize(g) for g in cg_generated_text_subset])

    cg_self_bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references=[[nltk.word_tokenize(t) for t in cg_generated_text_subset[:i]+cg_generated_text_subset[i+1:]]
                                                         for i in range(len(cg_generated_text_subset))],
                                                         hypotheses=[nltk.word_tokenize(g) for g in cg_generated_text_subset])

    metrics['rec_bleu'] = rec_bleu
    metrics['at_bleu'] = at_bleu
    metrics['cg_bleu'] = cg_bleu
    metrics['cg_self_bleu'] = cg_self_bleu

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    writer = open(output_eval_file, "w")
    logger.info("***** Eval results, global steps: {} *****".format(prefix))
    for key, value in metrics.items():
        logger.info("  %s = %s", key, str(value))
        writer.write("%s = %s\n" % (key, str(value)))

    return metrics

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default='results_cara', type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--soft_temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--lambda", default=0, type=float, help="")

    ## Data parameters
    parser.add_argument("--dataset", default='yelp', type=str, help="The dataset.")
    # parser.add_argument("--train_data_file", default='../../../data/yelp/sentiment.train.tiny.text', type=str, help="The input training data file (a text file).")
    parser.add_argument("--train_data_file", default='../../../data/yelp/sentiment.train.text', type=str, help="The input training data file (a text file).")
    # parser.add_argument("--eval_data_file", default='../../../data/yelp/sentiment.dev.tiny.text', type=str, help="")
    parser.add_argument("--eval_data_file", default='../../../data/yelp/sentiment.dev.small.text', type=str, help="2000 samples.")
    parser.add_argument("--ExpName", default="local_lctrlg_yelp", type=str, help="The experiment name used in Azure Table.")
    parser.add_argument("--create_new", default=0, type=int, help="")

    # Training parameters
    parser.add_argument("--checkpoint_dir", default='results_arae/checkpoint-47501/pytorch_model.bin', type=str, help='results/checkpoint-1212/pytorch_model.bin')
    # parser.add_argument("--checkpoint", default='', type=str, help='results/checkpoint-1212/pytorch_model.bin')
    parser.add_argument("--start_global_step", default=1001, type=int, help='')
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Run evaluation during training at each logging step.")
    parser.add_argument('--gloabl_step_eval', type=int, default=0, help="Evaluate the results at the given global step")
    # parser.add_argument('--logging_steps', type=int, default=2000, help="ARAE")
    parser.add_argument('--logging_steps', type=int, default=10, help="CARA")
    parser.add_argument('--eval_steps', type=int, default=500, help="CARA")
    # parser.add_argument('--save_steps', type=int, default=5000, help="ARAE")
    parser.add_argument('--save_steps', type=int, default=1000, help="CARA")
    parser.add_argument("--eval_all_checkpoints", action='store_true', help="")

    ## Encoder options
    # parser.add_argument("--encoder_model_name_or_path", default="bert-base-uncased", type=str, )
    parser.add_argument("--encoder_model_name_or_path", default="results_cara/checkpoint-encoder-1000", type=str)
    # parser.add_argument("--encoder_model_name_or_path", default="results/checkpoint-encoder-55000", type=str")
    parser.add_argument("--encoder_config_name", default="", type=str, help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--encoder_tokenizer_name", default="", type=str, help="Keep empty. Will default to decoder_model_name_or_path")
    parser.add_argument("--encoder_model_type", default="bert", type=str, help="The encoder model architecture to be fine-tuned.")

    ## Decoder options
    # parser.add_argument("--decoder_model_name_or_path", default="gpt2", type=str)
    parser.add_argument("--decoder_model_name_or_path", default="results_cara/checkpoint-decoder-1000", type=str)
    # parser.add_argument("--decoder_model_name_or_path", default="results/checkpoint-decoder-55000", type=str)
    parser.add_argument("--decoder_config_name", default="", type=str, help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--decoder_tokenizer_name", default="", type=str, help="Keep empty. Will default to decoder_model_name_or_path")
    parser.add_argument("--decoder_model_type", default="gpt2", type=str, help="The decoder model architecture to be fine-tuned.")

    ## Variational auto-encoder
    parser.add_argument("--latent_size", default=32, type=int, help="Latent space dimension.")
    parser.add_argument("--use_deterministic_connect", action='store_true', help="Use deterministic inference to generate latent codes, i.e., standard auto-encoders.")

    ## Objective functions
    parser.add_argument("--mlm", action='store_true', help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--cache_dir", default="", type=str, help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=21, type=int, help="21 for Yelp and Yahoo on label-conditional text generation")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    # Training Schedules
    parser.add_argument("--ratio_increase", default=0.25, type=float, help="Learning schedule, the percentage for the annealing stage.")
    parser.add_argument("--ratio_zero", default=0.5, type=float, help="Learning schedule, the percentage for the pure auto-encoding stage.")
    parser.add_argument("--fb_mode", default=1, type=int, help="free bit training mode.")
    parser.add_argument("--dim_target_kl", default=3.0, type=float, help="dim_target_kl free bit training mode.")
    parser.add_argument("--learning_rate", default=5e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--use_philly", action='store_true', help="Use Philly for computing.")
    parser.add_argument("--use_pretrained_model", action='store_true',
                        help="Use pre-trained auto-encoder models as the initialization")
    parser.add_argument("--use_pretrained_vae", action='store_true',
                        help="Use use_pretrained_vae as initialization, where beta value is specified in the folder")

    parser.add_argument("--beta", type=float, default=1.0, help="The weighting hyper-parameter of the KL term in VAE")
    parser.add_argument("--beta_cls", type=float, default=1.0, help="The weighting hyper-parameter for the classifier on the generated sentences")

    ## IO: Logging and Saving
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', type=int, default=1, help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # Precision & Distributed Training
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # New parameters
    parser.add_argument('--label_size', type=int, default=2, help="This depends on which dataset is used.")
    args = parser.parse_args()
    if args.decoder_model_type in ["bert", "roberta"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file or remove the --do_eval argument.")
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logger.info("Waiting for debugger attach")
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
    # pdb.set_trace()
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    args.ExpName = 'Vae_' + args.dataset + '_Nz_' + str(args.latent_size)  + '_Beta_'  + str(args.beta) + '_Dkl_' + str(args.dim_target_kl) + \
                    '_Ra_' + str(args.ratio_increase) + '_R0_' + str(args.ratio_zero)
    table_name = 'Vae' + args.dataset + 'Nz' + str(args.latent_size)
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab




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
        # model_encoder = encoder_model_class(config=encoder_config, latent_size=args.latent_size)

        ## Decoder
        decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[args.decoder_model_type]
        decoder_config = decoder_config_class.from_pretrained(args.decoder_config_name if args.decoder_config_name else args.decoder_model_name_or_path)
        tokenizer_decoder = decoder_tokenizer_class.from_pretrained(args.decoder_tokenizer_name if args.decoder_tokenizer_name else args.decoder_model_name_or_path, do_lower_case=args.do_lower_case)
        if args.block_size <= 0:
            args.block_size = tokenizer_decoder.max_len_single_sentence  # Our input block size will be the max possible for the model
        args.block_size = min(args.block_size, tokenizer_decoder.max_len_single_sentence)
        setattr(decoder_config, "latent_size", args.latent_size)
        model_decoder = decoder_model_class.from_pretrained(args.decoder_model_name_or_path, from_tf=bool('.ckpt' in args.decoder_model_name_or_path), config=decoder_config, latent_size=args.latent_size)
        # model_decoder = decoder_model_class(config=decoder_config, latent_size=args.latent_size)

    # Chunyuan: Add Padding token to GPT2
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    logger.info('We have added {} tokens to GPT2'.format(num_added_toks))
    model_decoder.resize_token_embeddings(len(tokenizer_decoder))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    assert tokenizer_decoder.pad_token == '<PAD>'


    # on_gpu = next(model_vae.parameters()).is_cuda
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab
    logger.info("Training/evaluation parameters %s", args)

    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    # Training
    
    logff = open(os.path.join(args.output_dir, 'log_{}'.format(get_time_str())), 'a')

    if args.do_train:
        global_step = args.start_global_step
        model_vae = CARA(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args).to(args.device)

        # if args.checkpoint:
        #     logger.info("Loading checkpoint from {}".format(args.checkpoint))
        #     model_vae.load_state_dict(torch.load(args.checkpoint))

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        if args.local_rank == 0:
            torch.distributed.barrier()

        train_dataset = load_and_cache_examples(args, [tokenizer_encoder, tokenizer_decoder], evaluate=False)

        # logger.info("Test evaluate before training.")
        # evaluate(args, model_vae, tokenizer_encoder, tokenizer_decoder, table_name, prefix=0, subset='test')

        # Train
        global_step, tr_loss = train(args, train_dataset, model_vae, tokenizer_encoder, tokenizer_decoder, table_name, logff=logff)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
        output_encoder_dir = os.path.join(args.output_dir, 'checkpoint-encoder-{}'.format(global_step))
        output_decoder_dir = os.path.join(args.output_dir, 'checkpoint-decoder-{}'.format(global_step))
        if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(output_dir)
        if not os.path.exists(output_encoder_dir) and args.local_rank in [-1, 0]:
            os.makedirs(output_encoder_dir)
        if not os.path.exists(output_decoder_dir) and args.local_rank in [-1, 0]:
            os.makedirs(output_decoder_dir)

        logger.info("Saving encoder model checkpoint to %s", output_encoder_dir)
        logger.info("Saving decoder model checkpoint to %s", output_decoder_dir)

        model_encoder_to_save = model_vae.module.encoder if hasattr(model_vae, 'module') else model_vae.encoder  # Take care of distributed/parallel training
        model_decoder_to_save = model_vae.module.decoder if hasattr(model_vae, 'module') else model_vae.decoder  # Take care of distributed/parallel training
        model_to_save = model_vae.module if hasattr(model_vae, "module") else model_vae

        # Good practice: save your training arguments together with the trained model
        if args.use_philly:
            save_solid = False
            while not save_solid:
                try:
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                    save_solid = True
                except:
                    pass
        else:
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        args.checkpoint = os.path.join(output_dir, 'pytorch_model.bin')

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
            torch.save(args, os.path.join(output_decoder_dir, 'training_decoder_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        # model_encoder = encoder_model_class.from_pretrained(output_encoder_dir, latent_size=args.latent_size)
        # model_encoder.to(args.device)
        #
        # # Load a trained model and vocabulary that you have fine-tuned
        # model_decoder = decoder_model_class.from_pretrained(output_decoder_dir, latent_size=args.latent_size)
        # model_decoder.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # if global_step == 0:
        #     global_step = args.gloabl_step_eval

        # output_encoder_dir = os.path.join(args.output_dir, 'checkpoint-encoder-{}'.format(global_step))
        # output_decoder_dir = os.path.join(args.output_dir, 'checkpoint-decoder-{}'.format(global_step))
        # checkpoints = [ [output_encoder_dir, output_decoder_dir] ]

        # logger.info("Evaluate the following checkpoints: %s", checkpoints)
        # for checkpoint in checkpoints:

        # global_step = args.checkpoint_dir.split('/')[-2].split('-')[-1] if args.checkpoint_dir else ""

        # model_encoder = encoder_model_class.from_pretrained(checkpoint[0], latent_size=args.latent_size)
        # model_encoder.to(args.device)
        # model_decoder = decoder_model_class.from_pretrained(checkpoint[1], latent_size=args.latent_size)
        # model_decoder.to(args.device)

        model_vae = CARA(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args).to(args.device)

        if args.gloabl_step_eval < 1:
            args.gloabl_step_eval = global_step
            args.checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(args.gloabl_step_eval))
        else:
            global_step = args.gloabl_step_eval
            args.checkpoint_dir = os.path.join(args.checkpoint_dir, 'checkpoint-{}/pytorch_model.bin'.format(args.gloabl_step_eval))


        # if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        #     logger.info("Loading checkpoint from {}".format(args.checkpoint_dir))
        #     model_vae.load_state_dict(torch.load(args.checkpoint_dir))
        # else:
        #     raise ValueError("Cannot find checkpoint at: {}".format(args.checkpoint))

        metrics = evaluate(args, model_vae, tokenizer_encoder, tokenizer_decoder, table_name, prefix=global_step, subset='test')
        metrics = dict((k + '_{}'.format(global_step), v) for k, v in metrics.items())
        results.update(metrics)

        # result = evaluate(args, model_vae, tokenizer_encoder, tokenizer_decoder, table_name, prefix=global_step, subset='train')
        # result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        # results.update(result)

    return results


if __name__ == "__main__":
    main()