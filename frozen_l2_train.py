import argparse
import hashlib
import numpy as np
import os
import pickle
import torch

from corpora import data
from paths import project_base_path
from training.l2_train import l2_train
from training.utils import batchify, get_batch, repackage_hidden, get_slice

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', type=str, help="Which pretrained model, look at possible_pretrains for the options")
parser.add_argument('--run-name', type=str, default="last_run", help="How to call the save file for this run, within the results dir of this pretrain type")
parser.add_argument("--finetune", type=str, default="es", help="What kind of L2 data to train on. Will assume that it exists in a shuffled vocab corpus")
parser.add_argument('--trials', nargs="+", type=int, default=[0, 5],
                    help='Lowest and highest trial num (of pretrained models)')
parser.add_argument('--seed', type=int, default=4, help="Seed will be args.seed*100 + the pretrain_index")
args = parser.parse_args()
args.cuda = True
print(args)

batch_size = 80
eval_batch_size = 10
test_batch_size = 1

def run():
    # Need to update this list with every experiment so that each one is
    # associated with an index for seed purposes.
    possible_pretrains = \
        ["pt", "ja", "code", "music", "unigram", "random", "es", "en", "ru", \
        "random310", "repetition", "paren", "paren-zipf", "valence", "de", "eu", \
        "fi", "ro", "tr", "it", "ko", "fa"]
    assert args.pretrain in possible_pretrains

    pretrain_type = args.pretrain if args.pretrain in ["music", "code"] else "language"
    pretrain_path = os.path.join(project_base_path, "models", "pretrained_models", args.pretrain)
    pretrain_idx = possible_pretrains.index(args.pretrain)
    save_dir = os.path.join(project_base_path, "models", "l2_results", args.pretrain)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, args.run_name)
    print(f"Will save to {save_path}")

    l2_data_type = args.finetune if args.finetune in ["music", "code"] else "language"
    l2_data_location = os.path.join(project_base_path, "corpora", "pickled_files", f"corpus-{args.finetune}")
    corpus = load_corpus(l2_data_location, l2_data_type, shuffle_vocab=True)
    train_data = batchify(corpus.train, batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    l1_data_location = os.path.join(project_base_path, "corpora", "pickled_files", f"corpus-{args.pretrain}")
    l1_corpus = load_corpus(l1_data_location, pretrain_type)
    l1_test = batchify(l1_corpus.test, test_batch_size, args)

    if os.path.exists(save_path):
        # We've already run this once, so we should add to the results of a killed job
        results = pickle.load(open(save_path, "rb"))
        print(f"Loaded results {results}")
    else:
        results = {"pretrain": [], "pretrain_trial": [], "train_length": [], "frozen": [], "seed_hundred": [], "trajectory": [], "val_at_convergence": [], "test_at_convergence": [], "train_at_convergence": [],"num_batches_for_convergence": [], "val_at_epoch": [], "test_at_epoch": [], "train_at_epoch": [], "pret_num_batches": [], "pret_final_val": [], "l1_test": [], "embeddings": []}

    print("Path, L1 Path and Index are:")
    print(pretrain_path, l1_data_location, pretrain_idx)
    # NOTE: there may be a cleaner way to do this with hashes. Keeping the hand-set
    # index for reproducability from earlier experiments.
    seed = args.seed*100 + pretrain_idx
    np.random.seed(seed)
    for trial in range(args.trials[0], args.trials[1]):
        print(f"Starting {args.pretrain}, trial {trial}")
        model_path = os.path.join(pretrain_path, f"trial{str(trial)}")
        with open(model_path, 'rb') as f:
            model, criterion, optimizer, scheduler, run_data = torch.load(f)
        val_loss_list, test_loss, train_loss, overall_batch, num_epochs, \
        loss_at_epoch, test_loss_at_epoch, train_loss_at_epoch, zero_shot_test, \
        l1_test_loss, embeddings = \
            l2_train((train_data, val_data, test_data),  model, criterion, 
                     l1_test, seed, freeze_net=True, check_epoch=1)
        results["frozen"].append(True)
        results["pretrain"].append(args.pretrain)
        results["pretrain_trial"].append(trial)
        results["train_length"].append(len(corpus.train))
        results["seed_hundred"].append(seed)
        results["trajectory"].append(val_loss_list)
        results["val_at_convergence"].append(val_loss_list[-1])
        results["test_at_convergence"].append(test_loss)
        results["train_at_convergence"].append(train_loss)
        results["num_batches_for_convergence"].append(overall_batch)
        results["val_at_epoch"].append(loss_at_epoch)
        results["test_at_epoch"].append(test_loss_at_epoch)
        results["train_at_epoch"].append(train_loss_at_epoch)
        results["pret_num_batches"].append(run_data[5])
        results["pret_final_val"].append(run_data[4][-1])
        results["l1_test"].append(l1_test_loss)
        results["embeddings"].append(embeddings)
        pickle.dump(results, open(save_path, "wb"))

def load_corpus(data_path, cull_vocab=True, shuffle_vocab=False):
    if cull_vocab:
        data_path = data_path + ".cull"
    if shuffle_vocab:
        assert cull_vocab, "Usually don't have unculled shuffled corpora"
        data_path = data_path + "-shuf"
    corpus = torch.load(data_path)
    return corpus

if __name__ == "__main__":
    run()
