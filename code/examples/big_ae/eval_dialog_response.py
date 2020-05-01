import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity as cosine
from collections import Counter
import os, pickle

class Metrics:
    # based on https://raw.githubusercontent.com/guxd/DialogWAE/29f206af05bfe5fe28fec4448e208310a7c9258d/experiments/metrics.py
    
    def __init__(self, path_word2vec='../data/datasets/dailydialog_data/glove.twitter.27B.200d.txt'):
        """
        :param word2vec - a numpy array of word2vec with shape [vocab_size x emb_size]
        """
        self.path_word2vec = path_word2vec
        super(Metrics, self).__init__()
        self.load_word2vec(path_word2vec)

    def load_word2vec(self, path_word2vec):
        path_pkl = path_word2vec + '.pkl'
        if os.path.exists(path_pkl):
            print('loading word2vec from '+path_pkl)
            self.word2vec = pickle.load(open(path_pkl, 'rb'))
        else:
            self.word2vec = dict()
            for i, line in enumerate(open(path_word2vec, encoding='utf-8')):
                ss = line.strip('\n').split() 
                self.word2vec[ss[0]] = [float(v) for v in ss[1:]]
                if i % 1e4 == 0:
                    print('processed %ik word2vec'%(i/1e3))
            print('dumping word2vec to '+path_pkl)
            pickle.dump(self.word2vec, open(path_pkl, 'wb'))
        # pdb.set_trace()
        self.embed_dim = len(self.word2vec["."]) # len(self.word2vec.values()[0])
        print('loaded %i word2vec of dim %i'%(len(self.word2vec), self.embed_dim))

    def embedding(self, seqs): 
        # note: different from original implementation
        batch_size, seqlen = seqs.shape
        embs = np.zeros([batch_size, seqlen, self.embed_dim])
        for i in range(batch_size):
            for j in range(seqlen):
                w = seqs[i,j] 
                if w != '' and w in self.word2vec:
                    embs[i, j, :] = self.word2vec[w]
        return embs

    
    def extrema(self, embs, lens): # embs: [batch_size x seq_len x emb_size]  lens: [batch_size]
        """
        computes the value of every single dimension in the word vectors which has the greatest
        difference from zero.
        :param seq: sequence
        :param seqlen: length of sequence
        """
        # Find minimum and maximum value for every dimension in predictions
        batch_size, seq_len, emb_size = embs.shape
        max_mask = np.zeros((batch_size, seq_len, emb_size), dtype=np.int)
        for i,length in enumerate(lens):
            max_mask[i,:length,:]=1
        min_mask = 1-max_mask
        seq_max = (embs*max_mask).max(1) # [batch_sz x emb_sz]
        seq_min = (embs+min_mask).min(1)
        # Find the maximum absolute value in min and max data
        comp_mask = seq_max >= np.abs(seq_min)# [batch_sz x emb_sz]
        # Add vectors for finding final sequence representation for predictions
        extrema_emb = seq_max* comp_mask + seq_min* np.logical_not(comp_mask)
        return extrema_emb
    
    def mean(self, embs, lens):
        batch_size, seq_len, emb_size=embs.shape
        mask = np.zeros((batch_size, seq_len, emb_size), dtype=np.int)
        for i,length in enumerate(lens):
            mask[i,:length,:]=1
        return (embs*mask).sum(1)/(mask.sum(1)+1e-8)

    def sim_bleu(self, hyps, ref):
        """
        :param ref - a list of tokens of the reference
        :param hyps - a list of tokens of the hypothesis
    
        :return maxbleu - recall bleu
        :return avgbleu - precision bleu
        """
        scores = []
        for hyp in hyps:
            try:
                scores.append(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7,
                                        weights=[1./3, 1./3, 1./3]))
            except:
                scores.append(0.0)
        return np.max(scores), np.mean(scores)


    def sim_bow(self, pred, pred_lens, ref, ref_lens):
        """
        :param pred - ndarray [batch_size x seqlen]
        :param pred_lens - list of integers
        :param ref - ndarray [batch_size x seqlen]
        """
        # look up word embeddings for prediction and reference
        emb_pred = self.embedding(pred) # [batch_sz x seqlen1 x emb_sz]
        emb_ref = self.embedding(ref) # [batch_sz x seqlen2 x emb_sz]
        
        ext_emb_pred=self.extrema(emb_pred, pred_lens)
        ext_emb_ref=self.extrema(emb_ref, ref_lens)
        bow_extrema=cosine(ext_emb_pred, ext_emb_ref) # [batch_sz_pred x batch_sz_ref]
        
        avg_emb_pred = self.mean(emb_pred, pred_lens) # Calculate mean over seq
        avg_emb_ref = self.mean(emb_ref, ref_lens) 
        bow_avg = cosine(avg_emb_pred, avg_emb_ref) # [batch_sz_pred x batch_sz_ref]

        
        batch_pred, seqlen_pred, emb_size=emb_pred.shape
        batch_ref, seqlen_ref, emb_size=emb_ref.shape
        cos_sim = cosine(emb_pred.reshape((-1, emb_size)), emb_ref.reshape((-1, emb_size))) # [(batch_sz*seqlen1)x(batch_sz*seqlen2)]
        cos_sim = cos_sim.reshape((batch_pred, seqlen_pred, batch_ref, seqlen_ref))
        # Find words with max cosine similarity
        max12 = cos_sim.max(1).mean(2) # max over seqlen_pred
        max21 = cos_sim.max(3).mean(1) # max over seqlen_ref
        bow_greedy=(max12+max21)/2 # [batch_pred x batch_ref(1)]
        return np.max(bow_extrema), np.max(bow_avg), np.max(bow_greedy)
    
    def div_distinct(self, seqs, seq_lens):
        """
        distinct-1 distinct-2 metrics for diversity measure proposed 
        by Li et al. "A Diversity-Promoting Objective Function for Neural Conversation Models"
        we counted numbers of distinct unigrams and bigrams in the generated responses 
        and divide the numbers by total number of unigrams and bigrams. 
        The two metrics measure how informative and diverse the generated responses are. 
        High numbers and high ratios mean that there is much content in the generated responses, 
        and high numbers further indicate that the generated responses are long
        """
        batch_size = seqs.shape[0]
        intra_dist1, intra_dist2=np.zeros(batch_size), np.zeros(batch_size)
        
        n_unigrams, n_bigrams, n_unigrams_total , n_bigrams_total = 0. ,0., 0., 0.
        unigrams_all, bigrams_all = Counter(), Counter()
        for b in range(batch_size):
            unigrams= Counter([tuple(seqs[b,i:i+1]) for i in range(seq_lens[b])])
            bigrams = Counter([tuple(seqs[b,i:i+2]) for i in range(seq_lens[b]-1)])
            intra_dist1[b]=(len(unigrams.items())+1e-12)/(seq_lens[b]+1e-5)
            intra_dist2[b]=(len(bigrams.items())+1e-12)/(max(0, seq_lens[b]-1)+1e-5)
            
            unigrams_all.update([tuple(seqs[b,i:i+1]) for i in range(seq_lens[b])])
            bigrams_all.update([tuple(seqs[b,i:i+2]) for i in range(seq_lens[b]-1)])
            n_unigrams_total += seq_lens[b]
            n_bigrams_total += max(0, seq_lens[b]-1)

        inter_dist1 = (len(unigrams_all.items())+1e-12)/(n_unigrams_total+1e-5)
        inter_dist2 = (len(bigrams_all.items())+1e-12)/(n_bigrams_total+1e-5)
        return intra_dist1, intra_dist2, inter_dist1, inter_dist2

import pdb

def eval_dialog_response(generated_text_file_path):
    """
    based on: https://github.com/guxd/DialogWAE/blob/29f206af05bfe5fe28fec4448e208310a7c9258d/sample.py
    quoted from the DialogWAE paper: https://arxiv.org/pdf/1805.12352.pdf
    * "For each test context, we sample 10 responses from the models and compute their BLEU scores"
    * "We use Glove vectors" "For each test context, we report the maximum BOW embedding score among the 10 sampled responses."
    * "intra-dist as the average of distinct values within each sampled response"
    " "inter-dist as the distinct value among all sampled responses."
    """
    metrics = Metrics()
    d_ref = dict()
    d_hyp = dict()
    for line in open(generated_text_file_path, encoding='utf-8'):
        line = line.strip('\n').strip()
        if len(line) == 0:
            continue
        src, ref, hyp = line.split('\t')
        src = src.strip()
        ref = ref.strip().split()
        hyp = hyp.strip().split()
        if src not in d_ref:
            d_ref[src] = ref
            d_hyp[src] = [hyp]
        else:
            d_hyp[src].append(hyp)
    
    n = len(d_ref)
    print(generated_text_file_path)
    print('n_src\t%i'%n)

    avg_lens = 0
    maxbleu = 0
    avgbleu = 0
    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = 0,0,0,0
    bow_extrema, bow_avg, bow_greedy = 0,0,0
    for src in d_ref:
        m, a = metrics.sim_bleu(d_hyp[src], d_ref[src])
        maxbleu += m
        avgbleu += a
        
        seq_len = [len(hyp) for hyp in d_hyp[src]]
        max_len = max(seq_len)
        seqs = []
        for hyp in d_hyp[src]:
            padded = hyp + [''] * (max_len - len(hyp))
            seqs.append(np.reshape(padded, [1, -1]))
        seqs = np.concatenate(seqs, axis=0)
        intra1, intra2, inter1, inter2 = metrics.div_distinct(seqs, seq_len)
        intra_dist1 += np.mean(intra1)
        intra_dist2 += np.mean(intra2)
        inter_dist1 += inter1
        inter_dist2 += inter2
        
        n_hyp = len(d_hyp[src])
        seqs_ref = np.concatenate([np.reshape(d_ref[src], [1,-1])] * n_hyp, axis=0)
        seq_len_ref = [len(d_ref[src])] * n_hyp
        if metrics.word2vec is not None:
            extrema, avg, greedy = metrics.sim_bow(seqs, seq_len, seqs_ref, seq_len_ref)
            bow_extrema += extrema
            bow_avg += avg
            bow_greedy += greedy

        avg_lens += np.mean(seq_len)

    recall_bleu = maxbleu/n
    prec_bleu = avgbleu/n
    f1 = 2*(prec_bleu*recall_bleu) / (prec_bleu+recall_bleu+10e-12)
    
    print('BLEU')
    print('  R\t%.3f'%recall_bleu)
    print('  P\t%.3f'%prec_bleu)
    print('  F1\t%.3f'%f1)
    print('BOW')
    print('  A\t%.3f'%(bow_avg/n))
    print('  E\t%.3f'%(bow_extrema/n))
    print('  G\t%.3f'%(bow_greedy/n))
    print('intra_dist')
    print('  1\t%.3f'%(intra_dist1/n))
    print('  2\t%.3f'%(intra_dist2/n))
    print('inter_dist')
    print('  1\t%.3f'%(inter_dist1/n))
    print('  2\t%.3f'%(inter_dist2/n))
    print('avg_L\t%.1f'%(avg_lens/n))

    results = {
        "BLEU_R": recall_bleu, "BLEU_P": prec_bleu, "BLEU_F1": f1, "BOW_A": bow_avg/n, "BOW_E": bow_extrema/n, "BOW_G": bow_greedy/n, "intra_dist1": intra_dist1/n, "intra_dist2": intra_dist2/n, "inter_dist1": inter_dist1/n, "inter_dist2": inter_dist2/n, "avg_L": avg_lens/n
    }

    return results



def create_rand_baseline():
    path = 'data/datasets/dailydialog_data/test.txt'
    srcs = []
    refs = []
    for line in open(path, encoding='utf-8'):
        src, ref = line.strip('\n').split('\t')
        srcs.append(src.strip())
        refs.append(ref.strip())
    
    hyps = set()
    path = 'data/datasets/dailydialog_data/train.txt'
    for line in open(path, encoding='utf-8'):
        _, ref = line.strip('\n').split('\t')
        hyps.add(ref)
        if len(hyps) == len(srcs) *10:
            print('collected training ref')
            break
    
    hyps = list(hyps)
    lines = []
    j = 0
    for i in range(len(srcs)):
        lines += ['\t'.join([srcs[i], refs[i], hyp]) for hyp in hyps[j:j+10]]
        j = j + 10
    with open('out/rand.tsv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def create_human_baseline():
    path = 'data/datasets/dailydialog_data/test.txt'
    lines = []
    for line in open(path, encoding='utf-8'):
        src, ref = line.strip('\n').split('\t')
        src = src.strip()
        ref = ref.strip()
        lines.append('\t'.join([src, ref, ref]))
        
    with open('out/human.tsv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    #create_rand_baseline()
    #create_human_baseline()
    eval_dialog_response('out/eval_text_generation_results (1).txt')
    #eval('out/rand.tsv')