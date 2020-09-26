import argparse
import collections
import hashlib
import numpy as np
import torch

import sys
sys.path.insert(0,'/u/scr/isabelvp/multilingual-transfer/salesforce-awd-lm')

import data

default_vocab_size = 50000
parser = argparse.ArgumentParser(description="Create a corpus from a uniform distribution")
parser.add_argument('--vocab-size', type=int, default=default_vocab_size)
args = parser.parse_args()

# Language to base the corpus length on. Should not be that important.
lang = "es"
lang_location = f"/u/scr/nlp/isabelvp/wiki-multilingual/{lang}/{lang}wiki-corpus-"
fn = '/u/scr/isabelvp/multilingual-transfer/salesforce-awd-lm/sf_corpora/onelang/corpus.{}.culldata'.format(hashlib.md5(lang_location.encode()).hexdigest())
lang_corpus = torch.load(fn)

uni_corpus = data.Corpus()
uni_corpus.dictionary = lang_corpus.dictionary

uni_corpus.train = torch.LongTensor(
    np.random.randint(args.vocab_size, size=len(lang_corpus.train)))
uni_corpus.valid = torch.LongTensor(
    np.random.randint(args.vocab_size, size=len(lang_corpus.valid)))
uni_corpus.test = torch.LongTensor(
    np.random.randint(args.vocab_size, size=len(lang_corpus.test)))

print(f"Made train/valid/test, train_length is {len(uni_corpus.train)}")

if args.vocab_size == default_vocab_size:
    save_fn = '/u/scr/isabelvp/multilingual-transfer/salesforce-awd-lm/sf_corpora/onelang/corpus.{}.culldata'.format(hashlib.md5(f"random".encode()).hexdigest())
else:
    save_fn = '/u/scr/isabelvp/multilingual-transfer/salesforce-awd-lm/sf_corpora/onelang/corpus.{}.culldata'.format(hashlib.md5(f"random{args.vocab_size}".encode()).hexdigest())
print(f"Saving to {save_fn}")
torch.save(uni_corpus, save_fn)
