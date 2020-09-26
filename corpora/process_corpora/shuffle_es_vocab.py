import argparse
import hashlib
import os
import random
import sys
import torch

this_file_path = os.path.join(os.getcwd(), __file__)
project_base_path = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]
print(project_base_path)
sys.path.insert(0, project_base_path)

from corpora.data import Corpus

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--lang', type=str, default="es")
args = parser.parse_args()

if args.seed >= 0:
    print(f"Setting random seed to {args.seed}")
    random.seed(args.seed)

corpus_fn = os.path.join(project_base_path, "corpora", "pickled_files", f"corpus-{args.lang}.cull")
if args.seed >= 0:
    shuffle_fn = os.path.join(project_base_path, "corpora", "pickled_files", f"corpus-{args.lang}.cull-shuf{args.seed}")
else:
    shuffle_fn = os.path.join(project_base_path, "corpora", "pickled_files", f"corpus-{args.lang}.cull-shuf")
corpus = torch.load(corpus_fn)
print("Shuffling corpus vocab to make sure there are not shared vocab effects. This can take some time")
corpus.shuffle()
print("Done shuffling!")
print("saving")
torch.save(corpus, shuffle_fn)
