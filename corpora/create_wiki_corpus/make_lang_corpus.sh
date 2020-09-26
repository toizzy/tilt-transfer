#!/bin/bash

echo "Make sure you're in a good conda environment!"
echo "This whole script can take hours, make sure that's ok!"
lang=$1

mkdir $lang ; cd $lang
echo "Getting and unzipping wikipedia dumps for $lang"
curl -O https://dumps.wikimedia.org/"$lang"wiki/latest/"$lang"wiki-latest-pages-articles.xml.bz2
bunzip2 "$lang"wiki-latest-pages-articles.xml.bz2
cd ..

# Download the trained language models you want for the language, and save them
# to the common location on the cluster # Make sure you have the conda env you want activated when you do this
echo "Downloading the pretrained models for stanfordnlp"
printf "yes\n/u/nlp/data/stanfordnlp_resources/" | python -c "import stanfordnlp; stanfordnlp.download(\"$lang\")"

# Now we are ready to run a corpus creation job.
# This creates a file where every line is a sentence (sentence segmentation done
# by stanfordnlp) and every token is separated by a space (tokenisation also by
# stanfordnlp)
echo "Parsing the wikipedia dump into a corpus. This is the part that really takes a while"
echo $lang
echo "$lang"/parselog
python process_wiki_dump.py \
--dump-file "$lang"/"$lang"wiki-latest-pages-articles.xml \
--output-file "$lang"/"$lang"wiki-corpus > \
"$lang"/parselog

echo "Splitting the corpus into train/val/test"
# Split the corpus into train-dev-test
./split_corpus.sh $lang > "$lang"/splitlog
