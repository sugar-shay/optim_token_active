# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:24:47 2021

@author: Shadow
"""

import pandas as pd

from data_preprocess import *
from tokenize_data import *
from pretrain_lit_ner import *
import pickle 


def main(category = 'memc', save_dir = 'results'):
        
    train_data = get_single_ner(category, train = True)    
    test_data = get_single_ner(category, test = True)        
    
    
    encoder_name = 'bert-base-uncased'
    
    train_data, unique_labels = process_data(train_data, return_unique=True)
    test_data = process_data(test_data)
    
    if category == 'memc':
        val_data = get_single_ner(category)
        val_data = process_data(val_data)

    else:
        num_val = np.floor(.2*train_data.shape[0])
        val_data = train_data.loc[:num_val, :]
        train_data = train_data.loc[num_val:, :]
    
    print()
    print('# of Training Examples: ', train_data.shape[0])
    print('# of Val Examples: ', val_data.shape[0])
    print('# of Test Examples: ', test_data.shape[0])

    
    tokenizer = NER_tokenizer(unique_labels, max_length=64, tokenizer_name = encoder_name)
    
    train_dataset = tokenizer.tokenize_and_encode_labels(train_data)
    val_dataset = tokenizer.tokenize_and_encode_labels(val_data)
    test_dataset = tokenizer.tokenize_and_encode_labels(test_data)
    
    model = PRETRAIN_LIT_NER(num_classes = len(tokenizer.id2tag), 
                     id2tag = tokenizer.id2tag,
                     tag2id = tokenizer.tag2id,
                     hidden_dropout_prob=.1,
                     attention_probs_dropout_prob=.1,
                     encoder_name = encoder_name,
                     save_fp='bert_memc.pt')
    
    
    model = train_LitModel(model, train_dataset, val_dataset, max_epochs=2, batch_size=32, patience = 2, num_gpu=1)
    
    complete_save_path = save_dir+'/'+category
    if not os.path.exists(complete_save_path):
        os.makedirs(complete_save_path)
         
    #saving train stats
    with open(complete_save_path+'/bert_train_stats.pkl', 'wb') as f:
        pickle.dump(model.training_stats, f)
        
    with open(complete_save_path+'/token_inputs.pkl', 'wb') as f:
        pickle.dump(model.token_inputs, f)
    
    #reloading the model for testing
    model = PRETRAIN_LIT_NER(num_classes = len(tokenizer.id2tag), 
                     id2tag = tokenizer.id2tag,
                     tag2id = tokenizer.tag2id,
                     hidden_dropout_prob=.1,
                     attention_probs_dropout_prob=.1,
                     encoder_name = encoder_name,
                     save_fp='best_model.pt')
    
    model.load_state_dict(torch.load('bert_memc.pt'))
    
    cr = model_testing(model, test_dataset)
    
    cr_df = pd.DataFrame(cr).transpose()
    print(cr_df)
    
    with open(complete_save_path+'/bert_test_stats.pkl', 'wb') as f:
            pickle.dump(cr, f)

if __name__ == "__main__":
    
    categories = ['memc']#, 'bypass']#, 'dirtra', 'httprs']
    for cat in categories:
        main(category=cat)
    
    
    
    