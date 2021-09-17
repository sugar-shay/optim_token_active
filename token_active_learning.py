# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 10:41:21 2021

@author: Shadow
"""


import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForTokenClassification, AutoConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, classification_report
from pytorch_lightning.callbacks import EarlyStopping
import os 
import itertools
import pickle

from tokenize_data import *
from active_lit_ner import *

def get_optim_training_data(train_data):
    
    train_grouped_seq = train_data.groupby(by ='sentence_idxs', axis=0)
    
    optim_training_data = {'input_ids':[],
                           'attention_mask':[],
                           'token_labels':[],
                           'token_idxs':[],
                           'token_label_masks':[]}
    
    num_groups = 0
    for group in train_grouped_seq.groups:
        num_groups += 1
        sub_df = train_grouped_seq.get_group(group)
        token_idxs = sub_df['token_idxs'].to_numpy()
        token_labels = sub_df['labels'].to_numpy()
        
        mask = sub_df['attention_mask'].iloc[0]
        input_ids = sub_df['input_ids'].iloc[0]
        label_mask = sub_df['token_label_maks'].iloc[0]
        
        optim_training_data['attention_mask'].append(mask)
        optim_training_data['input_ids'].append(input_ids)
        optim_training_data['token_label_masks'].append(label_mask)
        optim_training_data['token_idxs'].append(token_idxs)
        optim_training_data['token_labels'].append(token_labels)
        
    optim_training_data = pd.DataFrame(optim_training_data)
    return optim_training_data

def main(data_dir, data_split, category='memc'):
    
    val_data = get_single_ner(category)
    test_data = get_single_ner(category, test = True)        
    
    val_data, unique_labels = process_data(val_data, return_unique=True)
    test_data = process_data(test_data)
    
    if data_split == 'random':
        train_data_dir = data_dir + '/sample_data.pkl'    
    elif data_split == 'easy':
        train_data_dir = data_dir + '/easy_data.pkl'
    elif data_split == 'ambig':
        train_data_dir = data_dir + '/ambig_data.pkl'
    elif data_split == 'hard':
        train_data_dir = data_dir + '/hard_data.pkl'
    
    
    with open(train_data_dir, 'rb') as f:
        train_data = pickle.load(f)
        
    print('number of unique sentence idxs: ', len(np.unique(train_data['sentence_idxs'].to_numpy())))
    
    optim_training_data = get_optim_training_data(train_data)
        
    encoder_name = 'bert-base-uncased'
    
    
    MAX_LEN = 64
    
    tokenizer = NER_tokenizer(unique_labels, max_length=64, tokenizer_name = encoder_name)
    
    '''
    # here, pad = (padding_left, padding_right, padding_top, padding_bottom)
    source_pad = F.pad(source, pad=(0, 0, 0, 70 - source.shape[0]))
    '''
    def encode_pad_token_labels(arr):
        
        return (torch.nn.functional.pad(torch.tensor([tokenizer.tag2id[label] for label in arr]), pad = (MAX_LEN-arr.shape[0]-1, 1), value = -100)).numpy()
    
    def encode_pad_token_idxs(arr):
        return (torch.nn.functional.pad(torch.tensor(arr), pad = (MAX_LEN-1-arr.shape[0], 1), value = -100)).numpy()
    
    optim_training_data['token_labels'] = optim_training_data['token_labels'].apply(lambda x: encode_pad_token_labels(x))
    optim_training_data['token_idxs'] = optim_training_data['token_idxs'].apply(lambda x: encode_pad_token_idxs(x))
    
    
    
    
    #HERE ON WE NEED TO WORK ON 
    train_dataset = Token_Level_Dataset(input_ids = optim_training_data['input_ids'], 
                                        attention_mask = optim_training_data['attention_mask'], 
                                        token_idxs = np.vstack(optim_training_data['token_labels']),
                                        token_label_masks= optim_training_data['token_label_masks'], 
                                        labels=np.vstack(optim_training_data['token_labels']))
    
    val_dataset = tokenizer.tokenize_and_encode_labels(val_data)
    test_dataset = tokenizer.tokenize_and_encode_labels(test_data)
    
    model = ACTIVE_LIT_NER(num_classes = len(tokenizer.id2tag), 
                     id2tag = tokenizer.id2tag,
                     tag2id = tokenizer.tag2id,
                     hidden_dropout_prob=.1,
                     attention_probs_dropout_prob=.1,
                     encoder_name = encoder_name,
                     save_fp='bert_token_memc.pt')
    
    BATCH_SIZE = 64#64*32
    
    model = train_LitModel(model, train_dataset, val_dataset, max_epochs=10, batch_size=BATCH_SIZE, patience = 3, num_gpu=1)
    
    complete_save_path = save_dir+'/memc/token_results/' +data_split
    if not os.path.exists(complete_save_path):
        os.makedirs(complete_save_path)
         
    #saving train stats
    with open(complete_save_path+'/bert_train_stats.pkl', 'wb') as f:
        pickle.dump(model.training_stats, f)
        
    
    
    #reloading the model for testing
    model = PRETRAIN_LIT_NER(num_classes = len(tokenizer.id2tag), 
                     id2tag = tokenizer.id2tag,
                     tag2id = tokenizer.tag2id,
                     hidden_dropout_prob=.1,
                     attention_probs_dropout_prob=.1,
                     encoder_name = encoder_name,
                     save_fp='best_model.pt')
    
    model.load_state_dict(torch.load('bert_token_memc.pt'))
    
    cr = model_testing(model, test_dataset, output_dict=True)
    
    print(cr)
    
    with open(complete_save_path+'/bert_test_stats.pkl', 'wb') as f:
            pickle.dump(cr, f)

if __name__ == "__main__":
    data_directory = 'results/memc/token_data' 
    main(data_dir=data_directory, data_split = 'random', category='memc')
    