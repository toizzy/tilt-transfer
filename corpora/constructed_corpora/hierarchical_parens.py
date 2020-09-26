import argparse
from collections import deque
import hashlib
import numpy as np
import torch

import os
import sys
this_file_path = os.path.join(os.getcwd(), __file__)
project_path = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]
print(project_path)
sys.path.insert(0, project_path)

from corpora.data import Corpus

parser = argparse.ArgumentParser()
parser.add_argument("--zipf", action="store_true",
                    help="whether to sample characters from a zipfian distribution (uniform if flag is not set)")
parser.add_argument("--lang", type=str, default="es", help="corpus to base this one off of (length, vocab size etc)")
args = parser.parse_args()
print(args)

lang_fn = os.path.join(project_path, "corpora", "pickled_files", f"corpus-{args.lang}.cull")
print(f"loading original lang corpus from {lang_fn}")
lang_corpus = torch.load(lang_fn)
word_indices, freq = list(zip(*lang_corpus.dictionary.counter.items()))
freq = np.array(freq)
ps = freq / sum(freq)

if args.zipf:
    save_fn = os.path.join(project_path, "corpora", "pickled_files", f"corpus-paren-zipf.cull") 
else:
    save_fn = os.path.join(project_path, "corpora", "pickled_files", f"corpus-paren.cull") 

open_prob = 0.4
paren_corpus = Corpus()

print("Loaded and initialized everything")
paren_corpus.valid = torch.LongTensor(len(lang_corpus.valid))
paren_corpus.test = torch.LongTensor(len(lang_corpus.test))
paren_corpus.train = torch.LongTensor(len(lang_corpus.train))

for data in [paren_corpus.valid, paren_corpus.test, paren_corpus.train]:
    print("Sampling open chars")
    open_deque = deque()
    open_decision = np.random.choice([0, 1], len(data))
    if args.zipf:
        samples = np.random.choice(word_indices, len(data), p=ps)
    else:
        samples = np.random.randint(0, len(lang_corpus.dictionary), len(data))
    print("Finished sampling, starting construction")
    for i in range(len(data)):
        if i % 1000000 == 0:
            print(f"i is {i}")
        if open_decision[i] or len(open_deque) == 0:
            data[i] = torch.tensor(samples[i])
            open_deque.append(data[i])
        else:
            last_open = open_deque.pop()
            data[i] = last_open
    torch.save(paren_corpus, save_fn)

