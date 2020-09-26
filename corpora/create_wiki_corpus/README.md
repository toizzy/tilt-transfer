Scripts to easily create sentence-segmented, tokenized corpus in any language from Wikipedia dumps.

The pipeline is in `make_lang_corpus`. The script downloads a wikipedia dump and runs `process_wiki_dump.py` to process the dump into a file that has one sentence per line, with tokens separated by spaces. It then runs `split_corpus.sh` to split the corpus into train/val/test sets

`./make_lang_corpus.sh LANG` makes a corpus in language LANG

LANG has to be the code that Wikipedia uses for that language. So, for example, the Greek wikipedia is at el.wikipedia.org, so if I wanted to make a Greek corpus I would run `./make_lang_corpus.sh el` from this directory

`split_corpus.sh` splits the corpus into five fifths, and takes one fifth of the val and test set from each fifth. This is to avoid cases where the whole test or validation set is from one strange document in the original wikipedia corpus, that would affect evaluation somehow.

##Parameters that you could change

- As it holds now, `process_wiki_dump.py` stops when the corpus length exceeds 100,000,000 tokens. You can change this by changing the `max_tokens` parameter.
- `split_corpus.sh` makes a test corpus of length `20,000` sentences, a `val` corpus of length 2,000 sentences, and puts the rest in `train`. You can change this by changing the numbers

## Python requirements

The python script `process_wiki_dump.py`requires stanfordnlp, gensim and xml.etree, all available in pip.
