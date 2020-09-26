import argparse
import os
import sys
import torch

this_file_path = os.path.join(os.getcwd(), __file__)
project_base_path = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]
print(project_base_path)
sys.path.insert(0, project_base_path)

parser = argparse.ArgumentParser()
parser.add_argument('--corpus-name', type=str, default=None)
args = parser.parse_args()

corpus_fn = os.path.join(project_base_path, "corpora", "pickled_files",
                         f"corpus-{args.corpus_name}")
cull_fn = f"{corpus_fn}.cull"

print("loading corpus")
corpus = torch.load(corpus_fn)
print("culling vocab")
corpus.cull_vocab()
print("saving")
torch.save(corpus, cull_fn)

