from .vae import VAE
import numpy as np
import torch, copy, pdb
import torch.nn.functional as F

from torch import nn

import pdb


def set_trainable(module, value):
    for param in module.parameters():
        param.requires_grad = value

class SpaceFusion(VAE):
    def __init__(self, encoder, decoder,  tokenizer_encoder, tokenizer_decoder, args): 
        super(SpaceFusion, self).__init__(encoder, decoder,  tokenizer_encoder, tokenizer_decoder, args)
        children = [v for v in encoder.encoder.layer.children()]    # list of 12 BertLayer

        self.num_s2s_bert_layer = args.num_s2s_bert_layer
        self.S2S_layers = nn.ModuleList([copy.deepcopy(c) for c in children[-args.num_s2s_bert_layer:] ])    # the last layer of encoder
        self.S2S_pooler = copy.deepcopy(encoder.pooler)
        self.ix_turn_sep = tokenizer_encoder.convert_tokens_to_ids('[SEP]')
        if args.freeze_bert:
            print('@'*20 + f' freezing BERT {args.num_frozen_bert_layer} layers')
            for child in children[:args.num_frozen_bert_layer]:
                set_trainable(child, False)



    def ids2speaker(self, ids):
        # 0 for speaker A, 1 for speaker B
        N, T = ids.shape
        speaker = np.zeros((N, T))
        sep = ids == self.ix_turn_sep
        for i in range(N):
            is_B = False    # start with speaker A
            for t in range(T):
                speaker[i,t] = int(is_B)
                if sep[i,t].item():
                    is_B = not is_B

        # make sure the final speaker is speaker B (so response is always speaker A)
        if not is_B:
            speaker = 1 - speaker

        return torch.LongTensor(speaker).to(ids.device)

    def forward(self, inputs_src, inputs_tgt, labels_tgt, return_vec=False):  # [batch, time]
        # toggle config to get desired encoder output
        self.encoder.encoder.output_attentions = False
        self.encoder.encoder.output_hidden_states = True

        
        # AE encoder
        mask = (inputs_tgt > 0).float().to(inputs_src.device)
        outputs = self.encoder(inputs_tgt, attention_mask=mask)
        z_AE, _ = self.connect(outputs[1])
        z_AE = z_AE.squeeze(1)

        # S2S encoder
        mask = (inputs_src > 0).float()
        speaker = self.ids2speaker(inputs_src)
        outputs = self.encoder(inputs_src, attention_mask=mask, token_type_ids=speaker)
        _, _, all_layer_attn = outputs      # last_layer_attn, pooled, all_layer_attn = outputs
        seq_z_prev = all_layer_attn[-self.num_s2s_bert_layer-1]     # seq of z at layer 11 ()

        for s2s in self.S2S_layers: 
            layer_outputs = s2s(seq_z_prev, attention_mask=mask.unsqueeze(1).unsqueeze(1))
            seq_z_prev = layer_outputs[0]

        z_S2S = self.encoder.pooler(layer_outputs[0])
        z_S2S, _ = self.connect(z_S2S)
        z_S2S = z_S2S.squeeze(1)

        if return_vec:
            return z_AE, z_S2S

        # interpolation/smoothness
        u = torch.FloatTensor(np.random.random((z_AE.shape[0], 1))).to(inputs_tgt.device)
        z_interp = u * z_AE + (1 - u) * z_S2S
        std = 0.1
        noise = torch.FloatTensor(np.random.normal(size=z_interp.shape) * std).to(z_interp.device)
        z_interp = z_interp + noise

        loss_rec = 0
        z_idx = 0
        for z in [z_AE, z_S2S, z_interp]:
            #pdb.set_trace()
            past = z # past = self.decoder.linear(z)
            outputs = self.decoder(input_ids=labels_tgt, past=past, labels=labels_tgt, label_ignore=self.pad_token_id)
            if z_idx == 1:
                loss_rec = loss_rec + 1.0 * outputs[0]
            else:
                loss_rec = loss_rec + outputs[0]
            z_idx += 1
        loss_rec = loss_rec/3
        
        # fusion/regularization
        L_pull = self.dist_pair(z_AE, z_S2S)
        L_push = torch.stack([self.dist_batch(z) for z in [z_AE, z_S2S]]).min()
        loss_reg = (L_pull - L_push * 2) / np.sqrt(z.shape[-1])
        
        loss = loss_rec + self.args.beta * loss_reg
        return loss_rec, loss_reg, loss

    def sent2latent(self, inputs_src):
        # toggle config to get desired encoder output
        self.encoder.encoder.output_attentions = False
        self.encoder.encoder.output_hidden_states = True

        # S2S encoder
        mask = (inputs_src > 0).float()
        speaker = self.ids2speaker(inputs_src)
        outputs = self.encoder(inputs_src, attention_mask=mask, token_type_ids=speaker)

        _, _, all_layer_attn = outputs      # last_layer_attn, pooled, all_layer_attn = outputs
        # seq_z_prev = all_layer_attn[-2]     # seq of z at layer 11 ()
        # layer_outputs = self.S2S_layer(seq_z_prev, attention_mask=mask.unsqueeze(1).unsqueeze(1))

        seq_z_prev = all_layer_attn[-self.num_s2s_bert_layer-1]     # seq of z at layer 11 ()
        for s2s in self.S2S_layers: 
            layer_outputs = s2s(seq_z_prev, attention_mask=mask.unsqueeze(1).unsqueeze(1))
            seq_z_prev = layer_outputs[0]

        z_S2S = self.encoder.pooler(layer_outputs[0])
        z_S2S, _ = self.connect(z_S2S)
        z_S2S = z_S2S.squeeze(1)
        
        return z_S2S


    def dist_pair(self, a, b):
        return F.pairwise_distance(a, b).mean()


    def dist_batch(self, vec):
        n = vec.shape[0]
        dmin = []
        for i in range(n):
            dd = F.pairwise_distance(vec[i:i+1,:].repeat(n,1), vec)
            dmin.append(dd.min())
        return torch.stack(dmin).mean()