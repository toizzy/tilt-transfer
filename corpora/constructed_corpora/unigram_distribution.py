import collections
import hashlib
import numpy as np
import torch
import sys
sys.path.insert(0,'/u/scr/isabelvp/multilingual-transfer/salesforce-awd-lm')

import data

# Language to base the distribution on. Should not be that important.
lang = "es"

lang_location = f"/u/scr/nlp/isabelvp/wiki-multilingual/{lang}/{lang}wiki-corpus-"
fn = '/u/scr/isabelvp/multilingual-transfer/salesforce-awd-lm/sf_corpora/onelang/corpus.{}.culldata'.format(hashlib.md5(lang_location.encode()).hexdigest())
lang_corpus = torch.load(fn)

uni_corpus = data.Corpus()
uni_corpus.dictionary = lang_corpus.dictionary
word_indices, freq = list(zip(*lang_corpus.dictionary.counter.items()))
freq = np.array(freq)
ps = freq / sum(freq)

uni_corpus.train = torch.LongTensor(
    np.random.choice(word_indices, len(lang_corpus.train), p=ps))
uni_corpus.valid = torch.LongTensor(
    np.random.choice(word_indices, len(lang_corpus.valid), p=ps))
uni_corpus.test = torch.LongTensor(
    np.random.choice(word_indices, len(lang_corpus.test), p=ps))

print(f"Made train/valid/test, train_length is {len(uni_corpus.train)}")

save_fn = '/u/scr/isabelvp/multilingual-transfer/salesforce-awd-lm/sf_corpora/onelang/corpus.{}.culldata'.format(hashlib.md5(f"unigram-{lang}".encode()).hexdigest())
torch.save(uni_corpus, save_fn)
