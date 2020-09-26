#!/bin/bash

# Get the val and test sets from five different random points in the corpus
# Then put all the rest in the training set
corpus=$1/"$1"wiki-corpus
echo $corpus
len=$(wc -l < $corpus)
fifth=$((len / 5))
echo $fifth
s1=$(shuf -i 1-$((fifth-400)) -n1)
e1=$((s1 + 400))
s2=$(shuf -i $((fifth))-$((fifth*2-400)) -n1)
e2=$((s2 + 400))
s3=$(shuf -i $((fifth*2))-$((fifth*3-400)) -n1)
e3=$((s3 + 400))
s4=$(shuf -i $((fifth*3))-$((fifth*4-400)) -n1)
e4=$((s4 + 400))
s5=$(shuf -i $((fifth*4))-$((fifth*5-400)) -n1)
e5=$((s5 + 400))

echo val starts $s1 $s2 $s3 $s4 $s5
echo val ends $e1 $e2 $e3 $e4 $e5

val=$corpus
val+="-val"
head -n $e1 $corpus | tail -n 400 > $val
head -n $e2 $corpus | tail -n 400 >> $val
head -n $e3 $corpus | tail -n 400 >> $val
head -n $e4 $corpus | tail -n 400 >> $val
head -n $e5 $corpus | tail -n 400 >> $val

# A file that has everything except the vals
temp=$corpus
temp+="-temp"
head -n $((s1-1)) $corpus  > $temp
head -n $((s2-1)) $corpus | tail -n +$e1 >> $temp
head -n $((s3-1)) $corpus | tail -n +$e2 >> $temp
head -n $((s4-1)) $corpus | tail -n +$e3 >> $temp
head -n $((s5-1)) $corpus | tail -n +$e4 >> $temp
tail -n +$e5 $corpus >> $temp

echo temp length
wc -l $temp

tst=$corpus
tst+="-test"

len=$(wc -l < $temp)
fifth=$((len / 5))
s1=$(shuf -i 1-$((fifth-4000)) -n1)
e1=$((s1 + 4000))
s2=$(shuf -i $((fifth))-$((fifth*2-4000)) -n1)
e2=$((s2 + 4000))
s3=$(shuf -i $((fifth*2))-$((fifth*3-4000)) -n1)
e3=$((s3 + 4000))
s4=$(shuf -i $((fifth*3))-$((fifth*4-4000)) -n1)
e4=$((s4 + 4000))
s5=$(shuf -i $((fifth*4))-$((fifth*5-4000)) -n1)
e5=$((s5 + 4000))

echo tst starts $s1 $s2 $s3 $s4 $s5
echo tst ends $e1 $e2 $e3 $e4 $e5

head -n $e2 $corpus | tail -n 4000 > $tst
head -n $e2 $corpus | tail -n 4000 >> $tst
head -n $e3 $corpus | tail -n 4000 >> $tst
head -n $e4 $corpus | tail -n 4000 >> $tst
head -n $e5 $corpus | tail -n 4000 >> $tst

train=$corpus
train+='-train'
head -n $((s1-1)) $temp  > $train
head -n $((s2-1)) $temp | tail -n +$e1 >> $train
head -n $((s3-1)) $temp | tail -n +$e2 >> $train
head -n $((s4-1)) $temp | tail -n +$e3 >> $train
head -n $((s5-1)) $temp | tail -n +$e4 >> $train
tail -n +$e5 $temp >> $train

rm $temp
