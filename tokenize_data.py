# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:34:41 2021

@author: Shadow
"""

from data_preprocess import *

import numpy as np
import torch


from transformers import AutoTokenizer

class NER_tokenizer():
    
    def __init__(self, unique_labels, max_length, tokenizer_name = None):
        self.unique_tags = unique_labels
        self.tag2id = {tag: id for id, tag in enumerate(self.unique_tags)}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True) if tokenizer_name is not None else AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
        self.max_len = max_length
    
    
    def tokenize_and_encode_labels(self, data):
        
        '''
        sequences = []
        labels = []
        for data_point in data:
            seq = data_point[0]
            seq = seq.split()
            sequences.append(seq)
            labels.append(data_point[1])
        '''
        encodings = self.tokenizer(data['text'].to_list(), is_split_into_words=True, return_offsets_mapping=True, max_length = self.max_len, padding=True, truncation=True)
        
        label_enc = self.encode_tags(data['labels'], encodings)
        encodings.pop("offset_mapping") 
        
        #print('lablel enc shape: ', len(label_enc))
        #print('Num slot shape: ', len(data['num_slots'].to_list()))
        dataset = NER_Dataset(encodings, label_enc, data['num_slots'].to_list())
        
        return dataset
    
    def tokenize_unlabeled_data(self, data):
        
        sequences = []
        for data_point in data:
            seq = data_point[0]
            seq = seq.split()
            sequences.append(seq)
        
        encodings = self.tokenizer(sequences, is_split_into_words=True, max_length = self.max_len, padding=True, truncation=True)

        return encodings
        
        
    def encode_tags(self, tags, encodings):
        
        labels = [[self.tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            #print('len doc labels: ', len(doc_labels))
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            #print('doc enc labels shape: ', len(doc_enc_labels))
            arr_offset = np.array(doc_offset)
            #print('arr offset shaoe: ', arr_offset.shape)
        
            # set labels whose first offset position is 0 and the second is not 0
            #doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            counter = 0
            for i in range(arr_offset.shape[0]):
                if (arr_offset[i,0] == 0) and (arr_offset[i,1] != 0):
                    doc_enc_labels[i] = doc_labels[counter]
                    counter += 1
            #print('counter: ', counter)
            
            encoded_labels.append(doc_enc_labels.tolist())
        
        return encoded_labels

  
class NER_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, num_slots):
        self.encodings = encodings
        self.labels = labels
        self.num_slots = num_slots
        self.sentence_idxs = np.array([x for x in range(0, len(self.encodings['input_ids']))])

        
    
    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        #item['num_slots'] = torch.LongTensor(self.num_slots[idx])
        
        item['sentence_idxs'] = torch.tensor(self.sentence_idxs[idx])
        if self.labels is not None:
            item['labels'] = torch.LongTensor(self.labels[idx])
            
        if self.num_slots is not None:
            item['num_slots'] = torch.tensor(self.num_slots[idx])
            
        #print('Encoding Shape: ', item['input_ids'].shape)
        #print('Labels Shape: ', item['labels'].shape)
        #print('Num Slot Shape: ', item['num_slots'].shape)
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
class Token_Level_Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, token_idxs, token_label_masks, labels=None):
        self.input_ids = torch.tensor(input_ids)
        self.attention_mask = torch.tensor(attention_mask)
        self.token_idxs = token_idxs
        self.token_label_masks = torch.tensor(token_label_masks)
        self.labels = labels
        
        print('labels shape: ', self.labels.shape)
        print('token idxs shape: ', self.token_idxs.shape)
        
    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = self.input_ids[idx, :]
        item['attention_mask'] = self.attention_mask[idx, :]
        
        #we have to apply padding to make sure token idxs and labels are all the same shape
        item['token_idxs'] = torch.IntTensor(self.token_idxs[idx, :])
        #item['token_idxs'] = torch.nn.functional.pad(torch.tensor(self.token_idxs[idx]), pad =(64,1), value=-100)
        item['token_label_masks'] = self.token_label_masks[idx,:]
        if self.labels is not None:
            item['token_labels'] = torch.IntTensor(self.labels[idx, :])
            #item['token_labels'] = torch.nn.functional.pad(torch.tensor(self.labels[idx]), pad=(64,1), value=-100)
        return item

    def __len__(self):
        return self.input_ids.shape[0]
        


