import math
import torch
import torch.nn as nn
from .utils import log_sum_exp
import pdb
import sys
sys.path.append('../../')
from pytorch_transformers.modeling_bert import BertEmbeddings
import torch.nn.functional as F


class Ctrl_Gen(nn.Module):
    def __init__(self, encoder, decoder, tokenizer_encoder, tokenizer_decoder, args): #
        super(Ctrl_Gen, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder

        self.args = args
        self.nz = args.latent_size

        self.bos_token_id_list = self.tokenizer_decoder.encode(self.tokenizer_decoder.bos_token)
        self.pad_token_id = self.tokenizer_decoder.encode(self.tokenizer_decoder.pad_token)[0]

        # connector: from Bert hidden units to the latent space
        self.linear = nn.Linear(encoder.config.hidden_size, self.nz, bias=False)

        # # Standard Normal prior
        # loc = torch.zeros(self.nz, device=args.device)
        # scale = torch.ones(self.nz, device=args.device)
        # self.prior = torch.distributions.normal.Normal(loc, scale)

        self.label_embedding = nn.Embedding(args.label_size, self.nz, padding_idx=0)    # use the same size as latent_z so as to use the same decoder.linear()
        self.latent_generator = nn.Linear(self.nz, self.nz)
        self.latent_classifier = nn.Linear(self.nz, args.label_size if args.label_size > 2 else 1)
        self.latent_discriminator = nn.Linear(self.nz, 1)

        self.gpt_embeddings = nn.Embedding(self.decoder.config.vocab_size, self.decoder.config.n_embd)
        self.gpt_embeddings.weight.data = decoder.transformer.wte.weight.data

        self.conv1 = nn.Conv1d(self.encoder.config.hidden_size, self.encoder.config.hidden_size, 3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1 if args.label_size <= 2 else args.label_size)

        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        self.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_seq_ids, tgt_seq_ids, cond_labels, attention_mask):
        # inputs: (B, seq_len)
        # labels: (B, seq_len)
        # cond_labels: (B), conditional labels.

        ones_label = torch.ones_like(cond_labels).to(dtype=torch.float32)
        zeros_label = torch.zeros_like(cond_labels).to(dtype=torch.float32)
        random_noise = torch.nn.init.normal_(torch.empty(input_seq_ids.size(0), self.nz)).to(device=input_seq_ids.device, dtype=torch.float32)

        # Encode inputs
        outputs = self.encoder(input_seq_ids, attention_mask=attention_mask)
        pooled_hidden_fea = outputs[1]  # (B, dim_h)

        # Encode z
        latent_z = self.linear(pooled_hidden_fea)    # (B, nz)

        # Generate z
        gen_z = self.latent_generator(random_noise)  # (B, nz)

        # Latent discriminator
        prob_encode_z_dis = self.latent_discriminator(latent_z).squeeze(1).float()  # (B)
        prob_gen_z_dis = self.latent_discriminator(gen_z).squeeze(1).float()  # (B)
        # Train latent discriminator
        loss_lsd = self.BCEWithLogitsLoss(prob_gen_z_dis, zeros_label) + self.BCEWithLogitsLoss(prob_encode_z_dis, ones_label)
        acc_encode_z_dis = ((prob_encode_z_dis >= 0).float() == ones_label).float()
        acc_gen_z_dis = ((prob_gen_z_dis >= 0).float() == zeros_label).float()
        # Train sampler adversarially
        loss_lsg = self.BCEWithLogitsLoss(prob_gen_z_dis, ones_label)

        # Latent classifier
        prob_encode_z_cls = self.latent_classifier(latent_z)  # (B, n_labels)
        if self.args.label_size <= 2:
            prob_encode_z_cls = prob_encode_z_cls.squeeze(1)  # (B)
            # Train latent classifier
            loss_lsc = self.BCEWithLogitsLoss(prob_encode_z_cls, cond_labels.float())
            acc_encode_z_cls = ((prob_encode_z_cls >= 0).float() == cond_labels.float()).float()
            # Train encoder adversarially
            loss_encoder = 1 - self.BCEWithLogitsLoss(prob_encode_z_cls, cond_labels.float())
        else:
            # Train latent classifier
            loss_lsc = self.CrossEntropyLoss(prob_encode_z_cls, cond_labels)
            acc_encode_z_cls = (torch.argmax(prob_encode_z_cls, dim=-1) == cond_labels).float()
            # Train encoder adversarially
            loss_encoder = 1 - self.CrossEntropyLoss(prob_encode_z_cls, cond_labels)

        # Embed labels
        label_emb = self.label_embedding(cond_labels)  # (B, hidden_size)
        # past_label = self.decoder.linear(label_emb)    # (B, n_blocks * hidden_size)  # todo: use the same linear layer for latent_z for now.
        if self.args.label_size <= 2:
            sampled_cond_labels = 1 - cond_labels
        else:
            raise NotImplementedError    # todo: currently only implemented for binary labels. need to change for multi-class labels.
        sampled_label_emb = self.label_embedding(sampled_cond_labels)  # (B, hidden_size)
        # past_sampled_label = self.decoder.linear(sampled_label_emb)    # (B, n_blocks * hidden_size)  # todo: use the same linear layer for latent_z for now.
        past_sampled_label = sampled_label_emb

        # Generate based on encoded z and gt labels. (reconstruction)
        # past_z = self.decoder.linear(latent_z)    # (B, n_blocks * hidden_size)
        past_z = latent_z
        # gen_past_z = self.decoder.linear(gen_z)    # (B, n_blocks * hidden_size)
        gen_past_z = gen_z    # (B, n_blocks * hidden_size)

        # past = torch.cat([past_z.unsqueeze(1), past_label.unsqueeze(1)], dim=1) # (B, 2, n_blocks * hidden_size)

        past = latent_z + label_emb # (B, n_blocks * hidden_size)

        outputs = self.decoder(input_ids=tgt_seq_ids, past=past, labels=tgt_seq_ids, label_ignore=self.pad_token_id)
        loss_rec = outputs[0]

        # Train a classifier in the observation space
        tgt_emb = self.gpt_embeddings(tgt_seq_ids)
        tgt_encode = self.conv1(tgt_emb.transpose(1, 2))    # (B, dim_h, seq_len)
        tgt_encode = torch.mean(tgt_encode, dim=-1) # (B, dim_h)
        prob_cls = self.classifier(tgt_encode)   # (B, n_labels)
        if self.args.label_size <= 2:
            prob_cls = prob_cls.squeeze(1)
            loss_cls = self.BCEWithLogitsLoss(prob_cls, cond_labels.float())
            pred_cls = (prob_cls >= 0).to(dtype=torch.long)
        else:
            loss_cls = self.CrossEntropyLoss(prob_cls, cond_labels)
            pred_cls = torch.argmax(prob_cls, dim=-1)
        acc_cls = (pred_cls == cond_labels).float()

        # Generate based on encoded z and sampled labels (attribute transfer)
        # at_past = torch.cat([past_z.unsqueeze(1), past_sampled_label.unsqueeze(1)], dim=1) # (B, 2, n_blocks * hidden_size)
        # at_generated_soft = self.sample_sequence_conditional_batch_soft(past=at_past, context=self.bos_token_id_list) # (B, seq_len, vocab_size)

        # # Classifier on attribute transfer generated sentences. Train Generator on attribute transfer.
        # at_soft_emb = torch.matmul(at_generated_soft, self.gpt_embeddings.weight)
        # at_soft_encode = self.conv1(at_soft_emb.transpose(1, 2))    # (B, dim_h, seq_len)
        # at_soft_encode = torch.mean(at_soft_encode, dim=-1)   # (B, dim_h)
        # prob_at_soft_cls = self.classifier(at_soft_encode)    # (B, 1)
        # if self.args.label_size <= 2:
        #     prob_at_soft_cls = prob_at_soft_cls.squeeze(1)
        #     loss_at_soft_cls = self.BCEWithLogitsLoss(prob_at_soft_cls, sampled_cond_labels.float())
        #     pred_at_soft_cls = (prob_at_soft_cls >= 0).to(torch.long)
        # else:
        #     loss_at_soft_cls = self.CrossEntropyLoss(prob_at_soft_cls, sampled_cond_labels)
        #     pred_at_soft_cls = torch.argmax(prob_at_soft_cls, dim=-1)
        # acc_at_soft_cls = (pred_at_soft_cls == sampled_cond_labels).float()

        # Loss
        loss = loss_rec + loss_encoder + loss_lsc + loss_lsd + loss_lsg + self.args.beta_cls * loss_cls # + loss_at_soft_cls

        if not self.training:
            # Generate based on encoded z and gt labels
            generated = self.sample_sequence_conditional_batch(past=past, context=self.bos_token_id_list)

            # Generate based on encoded z and sampled labels (attribute transfer)
            # at_past = torch.cat([past_z.unsqueeze(1), past_sampled_label.unsqueeze(1)], dim=1) # (B, 2, n_blocks * hidden_size)
            at_past = past_z + past_sampled_label # (B, n_blocks * hidden_size)
            at_generated = self.sample_sequence_conditional_batch(past=at_past, context=self.bos_token_id_list) # (B, seq_len)

            # Generate based on sampled z and sampled labels. (conditional generation)
            # cg_past = torch.cat([gen_past_z.unsqueeze(1), past_sampled_label.unsqueeze(1)], dim=1) # (B, 2, n_blocks * hidden_size)
            cg_past = gen_past_z +  past_sampled_label # (B, n_blocks * hidden_size)
            cg_generated = self.sample_sequence_conditional_batch(past=cg_past, context=self.bos_token_id_list) # (B, seq_len)

            # classifier on gt generated sentences.
            ge_emb = self.gpt_embeddings(generated)
            ge_encode = self.conv1(ge_emb.transpose(1, 2))    # (B, dim_h, seq_len)
            ge_encode = torch.mean(ge_encode, dim=-1)   # (B, dim_h)
            prob_ge_cls = self.classifier(ge_encode)    # (B, 1)

            if self.args.label_size <= 2:
                pred_ge_cls = (prob_ge_cls.squeeze(1) >= 0).to(torch.long)
            else:
                pred_ge_cls = torch.argmax(prob_ge_cls, dim=-1)
            acc_ge_cls = (pred_ge_cls == cond_labels).float()

            # classifier on attribute transfer generated sentences.
            at_emb = self.gpt_embeddings(at_generated)
            at_encode = self.conv1(at_emb.transpose(1, 2))    # (B, dim_h, seq_len)
            at_encode = torch.mean(at_encode, dim=-1)   # (B, dim_h)
            prob_at_cls = self.classifier(at_encode)    # (B, 1)
            if self.args.label_size <= 2:
                pred_at_cls = (prob_at_cls.squeeze(1) >= 0).to(torch.long)
            else:
                pred_at_cls = torch.argmax(prob_at_cls, dim=-1)
            acc_at_cls = (pred_at_cls == sampled_cond_labels).float()

            # classifier on conditional generated sentences.
            cg_emb = self.gpt_embeddings(cg_generated)
            cg_encode = self.conv1(cg_emb.transpose(1, 2))    # (B, dim_h, seq_len)
            cg_encode = torch.mean(cg_encode, dim=-1)   # (B, dim_h)
            prob_cg_cls = self.classifier(cg_encode)    # (B, 1)
            if self.args.label_size <= 2:
                pred_cg_cls = (prob_cg_cls.squeeze(1) >= 0).to(torch.long)
            else:
                pred_cg_cls = torch.argmax(prob_cg_cls, dim=-1)
            acc_cg_cls = (pred_cg_cls == sampled_cond_labels).float()

            result = {
                    'sampled_cond_labels': sampled_cond_labels,
                    'cond_labels': cond_labels,

                    'tgt_seq_ids': tgt_seq_ids,
                    'generated': generated,
                    'at_generated': at_generated,
                    'cg_generated': cg_generated,

                    'acc_encode_z_dis': acc_encode_z_dis,
                    'acc_gen_z_dis': acc_gen_z_dis,
                    'acc_encode_z_cls': acc_encode_z_cls,
                    'acc_cls': acc_cls,
                    'acc_ge_cls': acc_ge_cls,
                    'acc_at_cls': acc_at_cls,
                    'acc_cg_cls': acc_cg_cls,

                    'pred_cls': pred_cls,
                    'pred_ge_cls': pred_ge_cls,
                    'pred_at_cls': pred_at_cls,
                    'pred_cg_cls': pred_cg_cls,
                    }

            return result

        loss_dict = {
                'loss': loss,
                'loss_rec': loss_rec,
                'loss_encoder': loss_encoder,
                'loss_lsc': loss_lsc,
                'loss_lsd': loss_lsd,
                'loss_lsg': loss_lsg,
                'loss_cls': loss_cls,
                # 'loss_at_soft_cls': loss_at_soft_cls,
        }
        acc_dict = {
                'acc_encode_z_dis': acc_encode_z_dis,
                'acc_gen_z_dis': acc_gen_z_dis,
                'acc_encode_z_cls': acc_encode_z_cls,
                'acc_cls': acc_cls,
                # 'acc_at_soft_cls': acc_at_soft_cls,
        }
        return loss_dict, acc_dict

    def sample_sequence_conditional_batch(self, past, context):
        # context: a single id of <BOS>
        # past: (B, past_seq_len dim_h)
        num_samples = past.size(0)
        context = torch.tensor(context, dtype=torch.long, device=past.device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context # (B, 1)

        # with torch.no_grad():
        while generated.size(-1) < self.args.block_size:
            inputs = {'input_ids': generated, 'past': past}
            outputs = self.decoder(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            lm_logits = outputs[0]

            # softmax sample
            next_tokens_logits = lm_logits[:, -1, :] / self.args.temperature  # (B, 1, vocab_size)
            filtered_logits = self.top_k_top_p_filtering_batch(next_tokens_logits, top_k=self.args.top_k, top_p=self.args.top_p)  # (B, 1, vocab_size)
            filtered_logits = F.softmax(filtered_logits, dim=-1)
            next_tokens = torch.multinomial(filtered_logits, num_samples=1)   # (B, 1)
            generated = torch.cat((generated, next_tokens), dim=1)  # (B, seq_len+1)

            not_finished = next_tokens != self.tokenizer_decoder.encode('<EOS>')[0]
            if torch.sum(not_finished) == 0:
                break

        return generated    # (B, seq_len)

    def top_k_top_p_filtering_batch(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear

        top_k = min(top_k, logits.size(-1))  # Safety check

        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            threshold = torch.topk(logits, top_k, dim=-1)[0][:, -1, None]
            logits.masked_fill_(logits < threshold, filter_value)   #  (B, vocab_size)

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)         # (B, vocab_size)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)   # (B, vocab_size)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]

            logits.masked_fill_(indices_to_remove, filter_value)

        return logits

    def sample_sequence_conditional_batch_soft(self, past, context):
        # context: a single id of <BOS>
        # past: (B, past_seq_len dim_h)
        num_samples = past.size(0)
        context = torch.tensor(context, dtype=torch.long, device=past.device).unsqueeze(0).repeat(num_samples, 1)     # (B, 1)
        context_soft = torch.FloatTensor(num_samples, self.decoder.config.vocab_size).zero_().to(device=past.device)    # (B, vocab_size)
        context_soft.scatter_(1, context, 1)  # (B, vocab_size)
        generated_soft = context_soft.unsqueeze(1) # (B, 1, vocab_size)

        # with torch.no_grad():
        while generated_soft.size(1) < self.args.block_size:    # generated_soft: (B, seq_len, vocab_size)
            inputs = {'soft_ids': generated_soft, 'past': past}
            outputs = self.decoder(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            lm_logits = outputs[0]  # (B, seq_len, vocab_size)

            # Gumbel softmax sample
            next_tokens_soft = gumbel_softmax(logits=lm_logits[:, -1:, :], temperature=self.args.soft_temperature, hard=False)  # (B, 1, vocab_size)
            generated_soft = torch.cat((generated_soft, next_tokens_soft), dim=1)   # (B, seq_len+1, vocab_size)

            # # softmax sample
            # next_tokens_logits = lm_logits[:, -1, :] / self.args.temperature  # (B, 1, vocab_size)
            # filtered_logits = self.top_k_top_p_filtering_batch(next_tokens_logits, top_k=self.args.top_k, top_p=self.args.top_p)  # (B, 1, vocab_size)
            # filtered_logits = F.softmax(filtered_logits, dim=-1)
            # next_tokens = torch.multinomial(filtered_logits, num_samples=1)   # (B, 1)
            # generated = torch.cat((generated, next_tokens), dim=1)  # (B, seq_len+1)

            next_tokens = torch.argmax(next_tokens_soft, dim=-1)    # (B, 1)
            not_finished = next_tokens != self.tokenizer_decoder.encode('<EOS>')[0]
            if torch.sum(not_finished) == 0:
                break

        return generated_soft    # (B, seq_len, vocab_size)


### Gumbel Softmax
def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
            logits: [..., n_class] unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
            [..., n_class] sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, otherwise it will be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)  # (..., n_class)

    if hard:    # return onehot
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)  # one hot
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y = (y_hard - y).detach() + y

    return y    # (..., n_class)

from torch.nn import functional as F
def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), logits.device)
    return F.softmax(y / temperature, dim=-1)

def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape).to(device=device)
    return -torch.log(-torch.log(U + eps) + eps)
