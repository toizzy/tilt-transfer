import argparse
import numpy as np
import os
import pickle
import torch

import sys
this_file_path = os.path.join(os.getcwd(), __file__)
project_path = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]
print(f"Project base path is {project_path}. If that's not right, you might be running this from the wrong directory")
sys.path.insert(0, project_path)

from corpora.data import Corpus

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, default="es", help="corpus to base this one off of (length, vocab size etc)")
parser.add_argument("--deplength-counter", type=str, default="deplength-counter-gsd", help="Filename of the deplength counter dictionary. Expected to be in this directory")
parser.add_argument("--save-name", type=str, default=None)
args = parser.parse_args()
print(args)

lang_fn = os.path.join(
    project_path, "corpora", "pickled_files", f"corpus-{args.lang}.cull")
print(f"loading original lang corpus from {lang_fn}")
lang_corpus = torch.load(lang_fn)
word_indices, freq = list(zip(*lang_corpus.dictionary.counter.items()))
freq = np.array(freq)
vocab_ps = freq / sum(freq)

if args.save_name:
    save_fn = os.path.join(
        project_path, "corpora", "pickled_files", f"corpus-{args.save_name}.cull")
else:
    save_fn = os.path.join(
        project_path, "corpora", "pickled_files", f"corpus-valence.cull")

valence_corpus = Corpus()
print("Loaded and initialized everything")
valence_corpus.dictionary = lang_corpus.dictionary
valence_corpus.valid = torch.zeros(len(lang_corpus.valid), dtype=torch.long) - 1
valence_corpus.test = torch.zeros(len(lang_corpus.test), dtype=torch.long) - 1
valence_corpus.train = torch.zeros(len(lang_corpus.train), dtype=torch.long) - 1

dep_length_dict = pickle.load(open(os.path.join(
    project_path, "corpora", "constructed_corpora", args.deplength_counter), "rb"))
dep_lengths, length_freq = list(zip(*dep_length_dict.items()))
length_freq = np.array(length_freq)
dep_length_ps = length_freq / sum(length_freq)


for data in [valence_corpus.valid, valence_corpus.test, valence_corpus.train]:
    print("Sampling")
    vocab_samples = np.random.choice(word_indices, len(data), p=vocab_ps)
    dep_length_samples = np.random.choice(dep_lengths, len(data), p=dep_length_ps)
    print("Done sampling")
    for i in range(len(data)):
        if i % 1000000 == 0:
            print(f"i is {i}")
        if data[i] >= 0:
            # This means that this index is already taken by the closing
            # parenthesis of an earlier open.
            continue
        chosen_word = torch.tensor(vocab_samples[i])
        data[i] = chosen_word 
        dep_length = dep_length_samples[i]
        closing_index = i + dep_length
        if closing_index >= len(data):
            continue
        if data[closing_index] < 0:
            data[closing_index] = chosen_word 
        else:
            # Look around the original sampled index to find the closest open
            # spot.
            displacement = 1
            found_spot = False
            while not found_spot:
                if data[closing_index - displacement] < 0:
                    data[closing_index - displacement] = chosen_word
                    found_spot = True
                elif closing_index + displacement < len(data) and data[closing_index + displacement] < 0:
                    data[closing_index + displacement] = chosen_word
                    found_spot = True
                displacement += 1
        if i < 3:
            print(data[:20])
    torch.save(valence_corpus, save_fn)


