# From Salesforce awd-lm repo, mostly

import numpy as np
import os
import pickle
import torch
import random

from collections import Counter

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.get_index(word)
        self.counter[token_id] += 1
        self.total += 1
        return self.get_index(word)

    def __len__(self):
        return len(self.idx2word)

    def get_index(self, word):
        return self.word2idx.get(word, 0)

    def in_freq_order(self):
        new_idx2word = []
        in_order = self.counter.most_common(len(self.idx2word))
        for i in range(len(in_order)):
            idx = in_order[i][0]
            word = self.idx2word[idx]
            self.word2idx[word] = i
            new_idx2word.append(word)
        self.idx2word = new_idx2word

class Corpus(object):
    def __init__(self, path=None, data_type=None):
        self.dictionary = Dictionary()
        self.train = None
        self.valid = None
        self.test = None

    def cull_vocab(self, size=50000):
        if len(self.dictionary.idx2word) <= size:
            print(f"No need to cull! Vocab has length {len(self.dictionary.idx2word)}")
            return
        most_common = self.dictionary.counter.most_common(size)
        new_idx2word = ["<unk>"]
        new_word2idx = {"<unk>": 0}
        new_counter = Counter()

        # start at 1 to accommodate unk
        # TODO doesn't this make it so that we miss the most common word?!
        for i in range(1, len(most_common)):
            old_index, freq = most_common[i]
            word = self.dictionary.idx2word[old_index]
            new_idx2word.append(word)
            new_word2idx[word] = i
            new_counter[i] = freq

        new_train = torch.LongTensor(len(self.train))
        new_valid = torch.LongTensor(len(self.valid))
        new_test = torch.LongTensor(len(self.test))

        for i in range(len(self.train)):
            word = self.dictionary.idx2word[self.train[i]]
            new_index = new_word2idx.get(word,0)
            new_train[i] = new_index
            if new_index == 0:
                new_counter[0] += 1

        for i in range(len(self.valid)):
            word = self.dictionary.idx2word[self.valid[i]]
            new_index = new_word2idx.get(word, 0)
            new_valid[i] = new_index
            if new_index == 0:
                new_counter[0] += 1

        for i in range(len(self.test)):
            word = self.dictionary.idx2word[self.test[i]]
            new_index = new_word2idx.get(word,0)
            new_test[i] = new_index
            if new_index == 0:
                new_counter[0] += 1

        self.dictionary.idx2word = new_idx2word
        self.dictionary.word2idx = new_word2idx
        self.dictionary.counter = new_counter
        self.train = new_train
        self.valid = new_valid
        self.test = new_test

    def shuffle(self):
        new_idx2word = self.dictionary.idx2word.copy()
        random.shuffle(new_idx2word)
        new_word2idx = {}
        new_counter = Counter()
        for i in range(len(new_idx2word)):
            word = new_idx2word[i]
            new_word2idx[word] = i
            old_idx = self.dictionary.word2idx[word]
            new_counter[i] = self.dictionary.counter[old_idx]
        for i in range(len(self.train)):
            self.train[i] = new_word2idx[self.dictionary.idx2word[self.train[i]]]
        for i in range(len(self.valid)):
            self.valid[i] = new_word2idx[self.dictionary.idx2word[self.valid[i]]]
        for i in range(len(self.test)):
            self.test[i] = new_word2idx[self.dictionary.idx2word[self.test[i]]]
        self.dictionary.idx2word = new_idx2word
        self.dictionary.word2idx = new_word2idx
        self.dictionary.counter = new_counter
