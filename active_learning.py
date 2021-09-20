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
from data_preprocess import *

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

def main(data_dir, data_split, category='memc', save_dir = 'results'):
    
    train_data = get_single_ner(category, train = True)
    
    test_data = get_single_ner(category, test = True)        
        
    train_data, unique_labels = process_data(train_data, return_unique=True)
    print('train data columns: ', train_data.columns)

    
    if category == 'memc':
        val_data = get_single_ner(category)
        val_data = process_data(val_data)
        val_data, unique_labels = process_data(val_data, return_unique=True)

    else:
        num_val = np.floor(.2*train_data.shape[0])
        val_data = train_data.loc[:num_val, :]
        train_data = train_data.loc[num_val:, :]
    
    
    
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
        total_train_data = pickle.load(f)
            
    encoder_name = 'bert-base-uncased'
    
    active_learning_iterations = 15
    
    # num tokens we sample
    init_train_size = 250 
    
    MAX_LEN = 64
    
    tokenizer = NER_tokenizer(unique_labels, max_length=64, tokenizer_name = encoder_name)
    
    val_dataset = tokenizer.tokenize_and_encode_labels(val_data)
    test_dataset = tokenizer.tokenize_and_encode_labels(test_data)
    
    '''
    # here, pad = (padding_left, padding_right, padding_top, padding_bottom)
    source_pad = F.pad(source, pad=(0, 0, 0, 70 - source.shape[0]))
    '''
    def encode_pad_token_labels(arr):
        
        return (torch.nn.functional.pad(torch.tensor([tokenizer.tag2id[label] for label in arr]), pad = (MAX_LEN-arr.shape[0]-1, 1), value = -100)).numpy()
    
    def encode_pad_token_idxs(arr):
        return (torch.nn.functional.pad(torch.tensor(arr), pad = (MAX_LEN-1-arr.shape[0], 1), value = -100)).numpy()
    
    complete_save_path = save_dir+'/bypass/token_results/' +data_split
    if not os.path.exists(complete_save_path):
        os.makedirs(complete_save_path)
        
    print('Size of token train pool: ', total_train_data.shape[0])
    
    init_train_data = total_train_data.sample(n=init_train_size, random_state = 0, replace = False)
    
    #performing active learning 
    cr_reports = []
    for iteration in range(active_learning_iterations):
        
    
        optim_train_data = get_optim_training_data(init_train_data)
        
        print()
        print('Number of tokens sampled: ', init_train_data.shape[0])
        print('number of unique sentence idxs: ', len(np.unique(init_train_data['sentence_idxs'].to_numpy())))
        print()
    
        
        token_labels = optim_train_data['token_labels'].apply(lambda x: encode_pad_token_labels(x))
        token_idxs = optim_train_data['token_idxs'].apply(lambda x: encode_pad_token_idxs(x))
        
        
        
        #HERE ON WE NEED TO WORK ON 
        train_dataset = Token_Level_Dataset(input_ids = np.vstack(list(optim_train_data['input_ids'])), 
                                            attention_mask = np.vstack(list(optim_train_data['attention_mask'])), 
                                            token_idxs = np.vstack(list(token_idxs)),
                                            token_label_masks= np.vstack(list(optim_train_data['token_label_masks'])), 
                                            labels=np.vstack(list(token_labels)))
        
    
        
        model = ACTIVE_LIT_NER(num_classes = len(tokenizer.id2tag), 
                         id2tag = tokenizer.id2tag,
                         tag2id = tokenizer.tag2id,
                         hidden_dropout_prob=.1,
                         attention_probs_dropout_prob=.1,
                         encoder_name = encoder_name,
                         save_fp='bert_token_memc.pt')
        
        BATCH_SIZE = 32#64*32
        
        model = train_LitModel(model, train_dataset, val_dataset, max_epochs=15, batch_size=BATCH_SIZE, patience = 3, num_gpu=1)
        

            
        #save_file = 'bert_'+str(len(init_train))
        #saving train stats
        with open(complete_save_path+'/bert_train_stats.pkl', 'wb') as f:
            pickle.dump(model.training_stats, f)
            
        
        
        #reloading the model for testing
        model = ACTIVE_LIT_NER(num_classes = len(tokenizer.id2tag), 
                         id2tag = tokenizer.id2tag,
                         tag2id = tokenizer.tag2id,
                         hidden_dropout_prob=.1,
                         attention_probs_dropout_prob=.1,
                         encoder_name = encoder_name,
                         save_fp='best_model.pt')
        
        model.load_state_dict(torch.load('bert_token_memc.pt'))
        
        cr = model_testing(model, test_dataset)
        
        print()
        print('Active Learning Iteration: ', iteration+1)
        print('Accuracy: ', cr['accuracy'])
        print()
        
        cr_reports.append(cr)
        
        #getting samples from oracle 
        oracle_samples = total_train_data.sample(n=init_train_size, replace = False)
        
        init_train_data = pd.concat([init_train_data, oracle_samples], ignore_index=True)
    
    with open(complete_save_path+'/cr_reports.pkl', 'wb') as f:
            pickle.dump(cr_reports, f)
    

if __name__ == "__main__":
    data_directory = 'results/bypass/token_data' 
    main(data_dir=data_directory, data_split = 'easy', category='bypass')
    