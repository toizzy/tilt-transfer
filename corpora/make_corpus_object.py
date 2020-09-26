# Make a Corpus object from a clean, tokenized corpus file. If you have some 
# downloaded dump, first run a script from raw_to_tokens and then come here.
import argparse
import os
import pickle
import torch

import sys
this_file_path = os.path.join(os.getcwd(), __file__)
project_path = os.path.split(os.path.split(this_file_path)[0])[0]
print(project_path)
sys.path.insert(0, project_path)

from corpora.data import Corpus

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help="Path of where the data is, minus the train/val/test part of the filename")
parser.add_argument('--name', type=str, help="The shorthand name for this corpus, such as 'unigram' or 'pt'")
parser.add_argument('--reading-type', type=str, default="language", choices=("language", "music", "code"), 
                    help="What kind of data is stored. 'language' (almost everything), 'code' or 'music'")

def main(args):
    corpus = Corpus()
    corpus.train = tokenize(corpus, args.path + 'train', args.reading_type)
    corpus.valid = tokenize(corpus, args.path + 'val', args.reading_type)
    corpus.test = tokenize(corpus, args.path + 'test', args.reading_type)
    torch.save(corpus, os.path.join(project_path, "corpora", "pickled_files", f"corpus-{args.name}"))
    print("Finished and saved!")

def tokenize(corpus, path, reading_type):
    if reading_type == "language":
        return lang_tokenize(corpus, path)
    if reading_type =="code":
        return code_tokenize(corpus, path)
    if reading_type == "music":
        return music_tokenize(corpus, path)
    print(f"ERROR: reading_type should be language, code or music, but is {reading_type}")

def lang_tokenize(corpus, path):
    """Tokenizes a text file."""
    assert os.path.exists(path)
    # Add words to the dictionary
    with open(path, 'r') as f:
        tokens = 0
        for line in f:
            words = line.split() + ['<eos>']
            tokens += len(words)
            for word in words:
                corpus.dictionary.add_word(word)

    # Tokenize file content
    with open(path, 'r') as f:
        ids = torch.LongTensor(tokens)
        token = 0
        for line in f:
            words = line.split() + ['<eos>']
            for word in words:
                ids[token] = corpus.dictionary.get_index(word)
                token += 1
    return ids

def code_tokenize(corpus, path):
    """The code corpus is stored as a pickled list of tokens"""
    tokens = pickle.load(open(path, "rb"))
    ids = torch.LongTensor(len(tokens))
    for token in tokens:
        corpus.dictionary.add_word(token)
    for i in range(len(tokens)):
        ids[i] = corpus.dictionary.get_index(tokens[i])
    return ids

def music_tokenize(corpus, path):
    """ 
    The music corpus is stored as numpy array of token ids. I guess I'll
    use the numbers as the "words" as well? Crazy...
    """
    files = [os.path.join(path, f) for f in os.listdir(path) if not os.path.isdir(f)]
    num_tokens = 0
    for f in files:
        arr = np.load(f)
        num_tokens += len(arr)
        print(f"This file has length {len(arr)}")
        for token in arr:
            corpus.dictionary.add_word(str(token))

    ids = torch.LongTensor(num_tokens)
    i = 0
    for f in files:
        arr = np.load(f)
        for token in arr:
            ids[i] = corpus.dictionary.get_index(str(token))
            i += 1
    return ids

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
