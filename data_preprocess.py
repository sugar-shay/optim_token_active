# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 17:34:14 2021

@author: Shadow
"""


import re
import os
import pandas as pd
import pathlib
import random
    
file_path = 'data/memc_train.txt'

def load_text_file(filename):
    if os.path.exists(filename) == False:
        return None

    with open(filename, "r") as fp:
        return fp.readlines()

def get_single_ner(category, train=None, test=None):
    
    base_path = 'data'
    
    def parse_line(line):
        result = re.split(r"\s+", line)
        # record of interest
        token, label  = result[0], result[1]

        return token, label
    
    filename = None
    
    if train and (not test): filename = base_path / pathlib.Path("%s_train.txt" % category)
    if (not train) and test: filename = base_path / pathlib.Path("%s_test.txt" % category)
    if (not train) and (not test): filename = base_path / pathlib.Path("%s_valid.txt" % category)
    
    if filename == None or os.path.exists(filename) == False: 
        return None

    block_list = list()
    block_string = ""
    for line in load_text_file(filename):
        # equivalent to block_string = block_string + line (rather than block_string = line + block_string)
        block_string += line
        # when encountering empty line, append block_string to block_list, and empty block_string
        if not line.split():
            block_list.append((category, block_string))
            block_string = ""
            continue
    
    record_list = list()
    for category, block_string in block_list:
        record_dict = {"category": None, "sentence": "", "label": "", "tuple": list()}
        for line in block_string.split("\n"):
            if not line.strip("\n"): continue
            token, label = parse_line(line)
        
            record_dict["sentence"] += token + " "
            record_dict["label"] += label + " "
            record_dict["tuple"].append((token, label))
        
        record_dict["category"] = category
        record_list.append(record_dict)

    return pd.DataFrame(record_list)

#This function takes the output of 'get_single_ner()
def reformat_data(df, get_unique_labels = False):
    seq = df.iloc[:,1].tolist()
    labels = df.iloc[:,2].tolist()
    labels = [label.split() for label in labels]
    data = list(tuple(zip(seq, labels)))
    
    if get_unique_labels==True:
        unique_labels = list({x for l in labels for x in l})
        unique_labels.sort()
        
        return data, unique_labels
    else:
        return data
    
    
def process_data(data, return_unique = False):
    
    #where data is a pandas df that has columns sentence and label not split
    
    total_text, total_labels, num_slots = [], [], []
    
    df_text, df_labels = data['sentence'].to_list(), data['label'].to_list()
    
    for sent,label in zip(df_text, df_labels):
        
        total_text.append(sent.split())
        total_labels.append(label.split())
        num_slots.append(len(label.split()))
    
    process_data = pd.DataFrame({'text':total_text,
                                 'labels':total_labels,
                                 'num_slots':num_slots})
    
    if return_unique == True:
        unique_labels = sorted(list(set([label for labels in total_labels for label in labels])))
        return process_data, unique_labels

    else:
        return process_data
    