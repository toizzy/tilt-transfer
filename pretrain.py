# This is the main.py file from the Salesforce awd-lm repository, adapted slightly
import argparse
import time
import math
import numpy as np
import os
import torch
import torch.nn as nn

from corpora import data
from paths import project_base_path
from training import model
from training.utils import batchify, get_batch, repackage_hidden, get_slice

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
########
# Arguments we actually use in our experiments.
parser.add_argument('--data', type=str, default='es',
                    help='Short name of the corpus')
parser.add_argument('--data-type', type=str, default="language",
                    help="What type of data this is: language, code, music")
parser.add_argument('--save', type=str, default="",
                    help='If present, append this to args.data to make the directory name, eg "models/pretrained_models/en-NEWRUN" if args.save is set to NEWRUN')
parser.add_argument('--trial', type=int, default=0,
                    help="We always assume the pretraining is one of a series of trials, so if it is not it is labelled trial0. The model is saved at pretrain_dir/trial{args.trial}. Look at the save argument to see more about what pretrained_dir is. The random seed is set to args.seed + args.trial")
parser.add_argument("--cull-vocab", action='store_true',
                    help="Cull the vocab to 50000 words, and make the rest unks")
parser.add_argument("--corpus-change", type=str, default="nothing",
                    help="What change to do to the corpus object's vocab (shuffle or freq)")
parser.add_argument("--trim-corpus", type=int, default=0,
                    help="How long to trim the corpus to. 0 means no trimming")
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--small', action='store_true', help='Make a small model')

########
########
# Arguments always set to the default.
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--lr-patience', type=int, default=5,
                    help='How many times to wait in same learning rate for no improvement')
parser.add_argument('--max-lr-decreases', type=int, default=5,
                    help='How many times to decrease the learning rate before stopping')
parser.add_argument('--stop-condition', type=str, default="convergence",
                    help="Whether to stop at 'convergence' or 'epochs' (when max epochs are reached)")
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--valid-interval', type=int, default=1000,
                    help='At how many batches to check validation error and save/etc')
###
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--num-embs', type=int, default=50000,
                    help="Number of word embeddings (size of vocab, essentially, but if vocab is smaller some will be unused)")
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--batch-size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
args = parser.parse_args()
args.tied = True
#############
#############
#############
if args.small:
    print("Making a small model!")
    args.nlayers = 1
    args.emsize = 100
    args.nhid = 128

# Set the random seed manually for reproducibility.
seed = args.seed + args.trial
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(seed)
print(f"Set the sed to {seed}")

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer, scheduler,
                    (epoch, num_lr_decreases, lr, stored_loss, val_loss_list,
                     overall_batch, epoch_batch, epoch_data_index)],
                   f)

def model_load(fn):
    global model, criterion, optimizer, scheduler, run_data
    with open(fn, 'rb') as f:
        model, criterion, optimizer, scheduler, run_data = torch.load(f)

save_dir = os.path.join(project_base_path, "models", "pretrained_models", args.data)
if args.save is not "":
    save_dir = f"{save_dir}-{args.save}"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_fn = os.path.join(save_dir, f"trial{args.trial}")
import os
import hashlib

fn = os.path.join(project_base_path, "corpora", "pickled_files",
     f"corpus-{args.data}")
fn_cull = os.path.join(project_base_path, "corpora", "pickled_files", 
     f"corpus-{args.data}.cull")
fn_shuffle = os.path.join(project_base_path, "corpora", "pickled_files", 
     f"corpus-{args.data}.cull-shuf")
loaded_file = False
vocab_culled = False
if args.corpus_change == "shuffle":
    assert args.cull_vocab, "We only shuffle the culled corpora"
    assert os.path.exists(fn_shuffle), \
        "No shuffled corpus file. Look in the corpora/process_corpora dir for how to make one"
    corpus = torch.load(fn_shuffle)
elif args.cull_vocab:
    assert os.path.exists(fn_cull), \
        "No culled corpus file. Look in the corpora/process_corpora dir for how to make one"
    corpus = torch.load(fn_cull)
else:
    corpus = torch.load(fn)

print(f"Just got corpus! First 20 of idx2word are {corpus.dictionary.idx2word[:20]}")
if args.trim_corpus > 0:
    print(f"trimming training corpus to {args.trim_corpus}")
    corpus.train = corpus.train[:args.trim_corpus]
print(f"Length of training corpus is {len(corpus.train)}")

eval_batch_size = 10
test_batch_size = 1
print("Training normally on train/val/test")
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

from training.splitcross import SplitCrossEntropyLoss
criterion = None
optimizer = None
scheduler = None
run_data = None
ntokens = len(corpus.dictionary)
assert ntokens <= args.num_embs, "Vocab can't be bigger than number of embeddings"
model = model.RNNModel(args.model, args.num_embs, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
###
save_path = os.path.join(project_base_path, "models", "pretrained_models", args.data)
if os.path.exists(save_fn):
    print(f"Model already started training! Resuming from {save_fn}")
    model_load(save_fn)
    model.cuda()
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

def train():
    # Turn on training mode which enables dropout.
    global overall_batch, epoch_batch, epoch_data_index, valid_time, best_val_loss, val_loss_list, stored_loss, scheduler, num_lr_decreases
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    while epoch_data_index < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, epoch_data_index, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if epoch_batch % args.log_interval == 0 and epoch_batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'tr loss {:5.2f} | tr ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, epoch_batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()

        if overall_batch % args.valid_interval == 0 and overall_batch > 0:
            elapsed = time.time() - valid_time
            val_loss = evaluate(val_data, eval_batch_size)
            val_loss_list.append(val_loss)
            scheduler.step(val_loss)
            if scheduler.in_cooldown:
                num_lr_decreases += 1
                print(f"Just decreased learning rate! Have decreased {num_lr_decreases} times.")
            print('-' * 89)
            print('| validating at batch {:3d} | time: {:5.2f}m | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              overall_batch, elapsed / 60, val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)
            valid_time = time.time()
            if val_loss < stored_loss:
                model_save(save_fn)
                print('Saving model (new best validation)')
                stored_loss = val_loss
            best_val_loss.append(val_loss)
        ###
        epoch_batch += 1
        overall_batch += 1
        epoch_data_index += seq_len
    epoch_batch = 0
    epoch_data_index = 0

# Loop over epochs.
epoch = 0
num_lr_decreases = 0
lr = args.lr
stored_loss = 100000000
best_val_loss = []
val_loss_list = []
overall_batch, epoch_batch, epoch_data_index = 0, 0, 0
valid_time = time.time()
if run_data is not None:
    print(f"Resuming with run_data {run_data}")
    epoch = run_data[0]
    num_lr_decreases = run_data[1]
    lr = run_data[2]
    stored_loss = run_data[3]
    val_loss_list = run_data[4]
    overal_batch = run_data[5]
    epoch_batch = run_data[6]
    epoch_data_index = run_data[7]

# At any point you can hit Ctrl + C to break out of training early.
try:
    if optimizer is None:
        # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
        if args.optimizer == 'sgd':
            optimizer = \
                torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            optimizer = \
                torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    if scheduler is None:
        # LR dropping scheduler. Make cooldown 1 since checking cooldown is the
        # only way to check if it actually decayed. 
        #TODO Consider making eps=1e-3, default seems a bit overkill and slows
        # convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.lr_patience, cooldown=1)
    stop_condition_met = False
    while not stop_condition_met:
        epoch_start_time = time.time()
        train()
        epoch += 1
        print(f"Epoch {epoch}, overall batch is {overall_batch}")
        print(f"Epoch took {(time.time() - epoch_start_time) / 60} minutes")
        print(f"Have decreased learning rate {num_lr_decreases} times")
        if args.stop_condition == "convergence":
            print("Checking for convergence")
            if num_lr_decreases >= args.max_lr_decreases:
                stop_condition_met = True
            if stop_condition_met == True:
                print("Stopping due to convergence")
        if args.stop_condition == "epochs":
            if epoch >= args.epochs:
                stop_condition_met = True
            if stop_condition_met == True:
                print("Stopping, reached max epochs")
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(save_fn)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
