import os
from multiprocessing import Pool
import pdb
import numpy as np
import nltk
nltk.download('punkt')

from nltk.translate.bleu_score import SmoothingFunction

try: 
    from multiprocessing import cpu_count
except: 
    from os import cpu_count

class Metrics(object):
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_score(self):
        pass


class Bleu(Metrics):
    def __init__(self, test_text='', real_text='', gram=3, num_real_sentences=500, num_fake_sentences=10000):
        super(Bleu, self).__init__()
        self.name = 'Bleu'
        self.test_data = test_text
        self.real_data = real_text
        self.gram = gram
        self.sample_size = num_real_sentences
        self.reference = None
        self.is_first = True
        self.num_sentences = num_fake_sentences


    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    # fetch REAL DATA
    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.real_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        raise Exception('make sure you call BLEU paralell')
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(cpu_count())
        result = list()
        maxx = self.num_sentences
        with open(self.test_data) as test_data:
            for i, hypothesis in enumerate(test_data):
                #print('i : {}'.format(i))
                hypothesis = nltk.word_tokenize(hypothesis)
                result.append(pool.apply_async(self.calc_bleu, args=(reference, hypothesis, weight)))
                if i > maxx : break
        score = 0.0
        cnt = 0
        for it, i in enumerate(result):
            #print('i : {}'.format(it))
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt




class SelfBleu(Metrics):
    def __init__(self, test_text='', gram=3, model_path='', num_sentences=500):
        super(SelfBleu, self).__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = num_sentences
        self.reference = None
        self.is_first = True


    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.test_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            #genious:
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt
