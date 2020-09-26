import csv
import numpy as np
import os
import pickle

corpus_root = "/u/scr/isabelvp/data/habeascorpus-data-withComments/habeascorpus_tokens"

def read_from_tokens():
    jfiles = []
    for root, dirs, files in os.walk(corpus_root):
        j = [os.path.join(root, f) for f in files if f[-5:] == ".java"]
        jfiles.extend(j)
    jfile = jfiles[0]
    lst = []
    for i in range(len(jfiles)):
        if i % 1000 == 0:
            print(f"At file {i}")
        jfile = jfiles[i]
        reader = csv.reader(open(jfile, "rt"),delimiter='\t' )
        try:
            for row in reader:
                if len(row) != 3:
                    continue
                if "COMMENT" not in row[1]:
                    if row[2] == "":
                        if row[0] is not "":
                            lst.append(row[0])
                    else:
                        words = row[2].split(" ")
                        lst.extend([word for word in words if word is not ""])
        except:
            print(f"Error at file {i}")
            continue


    pickle.dump(lst, open(os.path.join(corpus_root, "all_tokens_list"), "wb"))

def split_corpus(lst, valid_size, test_size, seed=400):
    np.random.seed(seed)
    v_fifth = valid_size // 5
    test_fifth = test_size // 5

    valid = []
    rest_list = lst
    for _ in range(5):
        start = np.random.randint(len(rest_list) - v_fifth)
        valid += rest_list[start:start + v_fifth]
        rest_list = rest_list[:start] + rest_list[start + v_fifth:]
    test = []
    for _ in range(5):
        start = np.random.randint(len(rest_list) - test_fifth)
        test += rest_list[start:start + test_fifth]
        rest_list = rest_list[:start] + rest_list[start + test_fifth:]
    return rest_list, valid, test

     
