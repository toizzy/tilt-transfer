#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from gensim.corpora import wikicorpus
import pickle
import stanfordnlp
import sys
import xml.etree.cElementTree as etree

parser = argparse.ArgumentParser(description='Scrape your own wikicorpora!')
parser.add_argument('--dump-file', type=str, \
                    help='location of the wikipedia dump (uncompressed)')
parser.add_argument('--output-file', type=str, \
                    help='Where to store the output corpus')
parser.add_argument('--max-tokens', type=int, default=100000000, \
                    help='After how many tokens to stop')
parser.add_argument('--min-tokens-for-article', type=int, default=300,
                    help="How many tokens does an article need to be considered")
args = parser.parse_args()

def add_ns(tag):
    return "{http://www.mediawiki.org/xml/export-0.10/}" + tag

def make_corpus(args):
    # Create new, empty file for the corpus
    wikidump_file = args.dump_file
    lang_code = wikidump_file.split("/")[-1][:2]
    output_filename = args.output_file
    f = open(output_filename, "w+")
    f.close()
    nlp = stanfordnlp.Pipeline(processors="tokenize", lang=lang_code,
        models_dir="/u/nlp/data/stanfordnlp_resources/")
    total_tokens = 0
    checkpoint = 100000
    for event, elem in etree.iterparse(
            args.dump_file, events=('start', 'end', 'start-ns', 'end-ns')):
        if event == 'end':
            if elem.tag == add_ns("page"):
                ns = elem.find(add_ns("ns"))
                if ns is not None and ns.text == "0":
                    revision = elem.find(add_ns("revision"))
                    if revision is None:
                        continue
                    text_elem = revision.find(add_ns("text"))
                    if text_elem is None:
                       continue
                    text = text_elem.text
                    if text is None:
                        continue
                    text = wikicorpus.filter_wiki(text)
                    text = text.lower()
                    try:
                        sentences = nlp(text).sentences
                    except Exception:
                        continue
                    article_len = sum(
                        [len(sent.words) for sent in sentences])
                    if article_len > int(args.min_tokens_for_article):
                        for sentence in sentences:
                            words = [word.text for word in sentence.words]
                            # Take out heading words, that usually appear as ==heading==
                            words = [word for word in words if "==" not in word]
                            if len(words) > 5:
                                total_tokens += len(words)
                                line = " ".join(words)
                                with open(output_filename, "a+") as outfile:
                                    outfile.write(line + "\n")
        if total_tokens >= checkpoint:
            print(f"At {total_tokens} tokens")
            checkpoint += 100000
        if total_tokens >= int(args.max_tokens):
            print("Reached max tokens! We're at {0}.".format(total_tokens))
            return
    print(f"Finished corpus with {total_tokens} tokens!")

def main():
    make_corpus(args)

if __name__ == '__main__':
    main()
