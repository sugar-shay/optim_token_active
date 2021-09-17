# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:43:50 2021

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

class PRETRAIN_LIT_NER(pl.LightningModule):
    def __init__(self, 
                 num_classes, 
                 id2tag,
                 tag2id,
                 hidden_dropout_prob=.5,
                 attention_probs_dropout_prob=.2,
                 encoder_name = 'bert-base-uncased',
                 save_fp='best_model.pt'):
       
        super(PRETRAIN_LIT_NER, self).__init__()
        
        self.num_classes = num_classes
        self.id2tag = id2tag
        self.tag2id = tag2id
        
        self.build_model(hidden_dropout_prob, attention_probs_dropout_prob, encoder_name)
        
        self.training_stats = {'train_losses':[],
                               'val_losses':[],
                               'train_accs':[],
                               'val_accs':[],
                               'gt_probs':[],
                               'correctness':[]}
        
        self.token_inputs = {'input_ids':[],
                             'attention_mask':[],
                             'token_idxs':[],
                             'sentence_idxs':[],
                             'train_labels':[],
                             'token_label_masks': []}
        

        self.save_fp = save_fp
        self.loss_func = nn.CrossEntropyLoss()
    
    def build_model(self, hidden_dropout_prob, attention_probs_dropout_prob, encoder_name):
        config = AutoConfig.from_pretrained(encoder_name, num_labels=self.num_classes)
        #These are the only two dropouts that we can set
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.encoder = AutoModelForTokenClassification.from_pretrained(encoder_name, config=config)
        
    def save_model(self):
        torch.save(self.state_dict(), self.save_fp)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        return outputs

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-6)
        return optimizer

    def training_step(self, batch, batch_idx):
        
        #batch['labels'] has shape [batch_size, MAX_LEN]
        #batch['num_slot'] has shape [batch_size]
        
        
        # Run Forward Pass
        outputs = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels = batch['labels'])

        # Compute Loss (Cross-Entropy)
        
        logits = outputs.logits
        
        logits = torch.nn.functional.softmax(logits, dim=-1)
        
        #print()
        #print('Labels Shape: ', batch['labels'].shape)
        #print('Num Slots Shape: ', batch['num_slots'].shape)
        
        #labels has shape [batch_size, MAX_LEN] => WE HAVE TO APPLY A MASK
        #print('batch labels shape: ', batch['labels'].shape)
        
        
        label_masks = batch['labels'] != -100
        #print('Label Mask Shape: ', label_masks.shape)
        #print('Label Mask: ', label_masks)
        
        active_logits, active_labels = [], []
        input_ids, attention_masks, token_label_masks = [], [], []
        token_idxs, sentence_idxs = [], []
        
        
        for idx in range(label_masks.shape[0]):
            #print('logits[idx,:] shape: ', logits[idx,:].shape)
            #print('label_masks[idx,:] shape: ', label_masks[idx,:].shape)
            
            #THIS WORKS
            label_mask = label_masks[idx,:]
            seq_active_logits = logits[idx,label_mask,:]
            seq_active_labels = batch['labels'][idx, label_mask]
            
            input_id = batch['input_ids'][idx, :]
            attention_mask = batch['attention_mask'][idx, :]
            sent_idx = batch['sentence_idxs'][idx]
            #print()
            #seq_active_logits has shape [true # of tokens, # classes]
            #print('seq_active_logits shape: ', seq_active_logits.shape)
            #print('active_labels shape: ', seq_active_labels.shape)
            #print('num slots: ', batch['num_slots'][idx])
            
            input_id = np.array([input_id.detach().cpu().numpy() for i in range(seq_active_labels.shape[0])])
            attention_mask = np.array([attention_mask.detach().cpu().numpy() for i in range(seq_active_labels.shape[0])])
            label_mask = np.array([label_mask.detach().cpu().numpy() for i in range(seq_active_labels.shape[0])])
            token_idx = np.array([i for i in range(seq_active_labels.shape[0])])
            
            sentence_idxs.append([sent_idx.detach().cpu().numpy() for i in range(seq_active_labels.shape[0])])
            
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_idxs.append(token_idx)
            token_label_masks.append(label_mask)
            
            
            active_logits.append(seq_active_logits)
            active_labels.append(seq_active_labels)
        
        
        active_logits = torch.cat(active_logits, dim=0)
        active_labels = torch.cat(active_labels, dim=0)
        
        input_ids = np.concatenate(input_ids, axis=0)
        attention_masks = np.concatenate(attention_masks, axis=0)
        token_idxs = np.concatenate(token_idxs, axis=0)
        token_label_masks = np.concatenate(token_label_masks, axis=0)
        sentence_idxs = np.concatenate(sentence_idxs, axis=0)

                
        #EVERYTHING BELOW HAS SHAPE [TOTAL # TOKENS, :]
        #print()
        #print('Active Logits shape: ', active_logits.shape)
        #print('Active Labels shape: ', active_labels.shape)
        
        #we need a really small learning rate to make this work 
        loss = self.loss_func(active_logits, active_labels)
        
        
        #were getting the ground truth probs. using amax()
        active_gt_probs = torch.amax(active_logits, dim = -1)
        
        active_preds = torch.argmax(active_logits, dim = -1)
        #print('Active GT Probs Shape: ', active_gt_probs.shape)
        #print('Active Preds Probs Shape: ', active_preds.shape)

        active_gt_probs = active_gt_probs.detach().cpu().numpy()
        active_preds = active_preds.detach().cpu().numpy()
        active_labels = active_labels.detach().cpu().numpy()
        
        
        correct = np.array([True if pred == label else False for pred,label in zip(active_preds, active_labels)])
        
        acc = accuracy_score(active_labels, active_preds)


    
        if self.current_epoch == 0:
            active_labels = np.array([self.id2tag[label] for label in active_labels])
            self.token_inputs['input_ids'].append(input_ids)
            self.token_inputs['attention_mask'].append(attention_masks)
            self.token_inputs['token_idxs'].append(token_idxs)
            self.token_inputs['train_labels'].append(active_labels)
            self.token_inputs['token_label_masks'].append(token_label_masks)
            self.token_inputs['sentence_idxs'].append(sentence_idxs)

            
        return {"loss": loss, 'train_loss': loss, "gt_probs": active_gt_probs, "correct": correct, 'train_acc':acc}
        
    def training_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        self.training_stats['train_losses'].append(avg_loss.detach().cpu())
        
        avg_acc = np.stack([x["train_acc"] for x in outputs]).mean()
        self.training_stats['train_accs'].append(avg_acc)
        
        #both of these have shape [# examples]
        gt_probs = np.concatenate([x['gt_probs'] for x in outputs])
        
        correctness = np.concatenate([x['correct'] for x in outputs])
        
        #print('GT Probs shape: ', gt_probs.shape)
        #print('Correctness shape: ', correctness.shape)
        
        print('Train Loss: ', avg_loss.detach().cpu())
        
        self.training_stats['gt_probs'].append(gt_probs)
        self.training_stats['correctness'].append(correctness)
        
        # We need to get the token inputs 
        if self.current_epoch == 0:
            self.token_inputs['input_ids'] =  np.concatenate([x for x in self.token_inputs['input_ids']])
            self.token_inputs['attention_mask'] =  np.concatenate([x for x in self.token_inputs['attention_mask']])
            self.token_inputs['token_idxs'] =  np.concatenate([x for x in self.token_inputs['token_idxs']])
            self.token_inputs['train_labels'] =  np.concatenate([x for x in self.token_inputs['train_labels']])
            self.token_inputs['token_label_masks'] =  np.concatenate([x for x in self.token_inputs['token_label_masks']])
            self.token_inputs['sentence_idxs'] = np.concatenate([x for x in self.token_inputs['sentence_idxs']])
            
            print()
            print('train_labels shape: ', self.token_inputs['train_labels'].shape)
            print('mask shape: ', self.token_inputs['attention_mask'].shape)
            print('input id shape: ', self.token_inputs['input_ids'].shape)
            print('token idx shape: ', self.token_inputs['token_idxs'].shape)
            print('token_label_masks shape: ', self.token_inputs['token_label_masks'].shape)
            print('sentence idxs shape: ', self.token_inputs['sentence_idxs'].shape)

            print()
        self.log('train_loss', avg_loss)
        
    def validation_step(self, batch, batch_idx):

        # Run Forward Pass
        outputs = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels = batch['labels'])
        
        # Compute Loss (Cross-Entropy)
        loss = outputs.loss
        
        logits = outputs.logits
        
        logits = torch.nn.functional.softmax(logits, dim=-1)
        
        preds = torch.argmax(logits, dim = -1)
        
        labels, preds = batch['labels'].detach().cpu().numpy(), preds.detach().cpu().numpy()
        
        active_preds = [[self.id2tag[p] for (p, l) in zip(pred, label) if l != -100] 
              for pred, label in zip(preds, labels)]
    
        active_labels = [[self.id2tag[l] for (p, l) in zip(pred, label) if l != -100]
                  for pred, label in zip(preds, labels)]
        
        acc = accuracy_score(list(itertools.chain(*active_labels)), list(itertools.chain(*active_preds)))
        
       
        return {"val_loss": loss, 'val_acc': acc}

        
    def validation_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        
        print('Val Loss: ', avg_loss.detach().cpu().numpy())
        
        avg_loss_cpu = avg_loss.detach().cpu().numpy()
        if len(self.training_stats['val_losses']) == 0 or avg_loss_cpu<np.min(self.training_stats['val_losses']):
            self.save_model()
            
        self.training_stats['val_losses'].append(avg_loss_cpu)
        
        avg_acc =  np.stack([x["val_acc"] for x in outputs]).mean()
        self.training_stats['val_accs'].append(avg_acc)
        
        self.log('val_loss', avg_loss)

        


def train_LitModel(model, train_data, val_data, max_epochs, batch_size, patience = 3, num_gpu=1):
    
    #
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle=False)#, num_workers=8)#, num_workers=16)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle = False)
    
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=patience, verbose=False, mode="min")
    
    trainer = pl.Trainer(gpus=num_gpu, max_epochs = max_epochs)
    trainer.fit(model, train_dataloader, val_dataloader)
    
    
    model.training_stats['gt_probs'], model.training_stats['correctness'] = (np.array(model.training_stats['gt_probs'])).T, (np.array(model.training_stats['correctness'])).T
    model.training_stats['train_losses'], model.training_stats['val_losses'] = np.array(model.training_stats['train_losses']), np.array(model.training_stats['val_losses'])
    
    return model


def model_testing(model, test_dataset):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = model.to(device)
    
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    total_preds, total_labels = [], []
    
    model.eval()
    for idx, batch in enumerate(test_dataloader):
        
        seq = (batch['input_ids']).to(device)
        mask = (batch['attention_mask']).to(device)
        labels = batch['labels']
        
        outputs = model(input_ids=seq, attention_mask=mask, labels=None)
        
        logits = outputs.logits
        logits = torch.nn.functional.softmax(logits, dim=-1)
        
        preds = torch.argmax(logits, dim=-1)
        preds = preds.detach().cpu().numpy()
        
        labels = labels.detach().cpu().numpy()
        
        active_preds = [[model.id2tag[p] for (p, l) in zip(pred, label) if l != -100] 
              for pred, label in zip(preds, labels)]
    
        active_labels = [[model.id2tag[l] for (p, l) in zip(pred, label) if l != -100]
                  for pred, label in zip(preds, labels)]
        
        total_preds.extend(active_preds)
        total_labels.extend(active_labels)
    
    #Total Preds is list of lists with length [# sequences] by [# tokens]
    #print('Len of Total Preds: ', len(total_preds))
        
        
    cr = classification_report(list(itertools.chain(*total_labels)), list(itertools.chain(*total_preds)))
    return cr
        

