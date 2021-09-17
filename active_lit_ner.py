# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 10:36:39 2021

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

class ACTIVE_LIT_NER(pl.LightningModule):
    def __init__(self, 
                 num_classes, 
                 id2tag,
                 tag2id,
                 hidden_dropout_prob=.5,
                 attention_probs_dropout_prob=.2,
                 encoder_name = 'bert-base-uncased',
                 save_fp='best_model.pt'):
       
        super(ACTIVE_LIT_NER, self).__init__()
        
        self.num_classes = num_classes
        self.id2tag = id2tag
        self.tag2id = tag2id
        
        self.build_model(hidden_dropout_prob, attention_probs_dropout_prob, encoder_name)
        
        self.training_stats = {'train_losses':[],
                               'val_losses':[],
                               'train_accs':[],
                               'val_accs':[]}
        


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
        optimizer = AdamW(self.parameters(), lr=1e-6)
        return optimizer

    def training_step(self, batch, batch_idx):
        
        #batch['labels'] has shape [batch_size, MAX_LEN]
        #batch['num_slot'] has shape [batch_size]
        
        
        # Run Forward Pass
        outputs = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels = None)

        # Compute Loss (Cross-Entropy)
        
        logits = outputs.logits
        
        #logits has shape [batch_size, MAX_LEN, # classes]
        logits = torch.nn.functional.softmax(logits, dim=-1)
        
        #WE STILL NEED TO APPLY THE LABEL MASK TO GET [batch size, true # tokens, # classes]
        
        #print()
        #print('Labels Shape: ', batch['labels'].shape)
        #print('Num Slots Shape: ', batch['num_slots'].shape)
        
        #labels has shape [batch_size, MAX_LEN] => WE HAVE TO APPLY A MASK
        #print('batch labels shape: ', batch['labels'].shape)
        
        
        label_masks = batch['token_label_masks']
        #print('Label Mask Shape: ', label_masks.shape)
        #print('Label Mask: ', label_masks)
        
        active_token_logits, active_token_labels = [], []
        
        
        for idx in range(label_masks.shape[0]):
            #print('logits[idx,:] shape: ', logits[idx,:].shape)
            #print('label_masks[idx,:] shape: ', label_masks[idx,:].shape)
            
            token_labels = batch['token_labels'][idx, :]
            token_labels = (token_labels[token_labels!=-100]).type(torch.LongTensor)
            active_token_labels.append(token_labels)
            
            label_mask = label_masks[idx,:]
            seq_active_logits = logits[idx, label_mask,:]
            
            token_idxs = batch['token_idxs'][idx,:]
            token_idxs = (token_idxs[token_idxs != -100]).type(torch.LongTensor)
            
            #print('token_idx shape: ', token_idxs.shape)
            #print('token labels shape: ', token_labels.shape)
            #print('token_label: ', token_labels)
            #print('active logits shape: ', seq_active_logits.shape)
            
            #token logits has shape [1, # classes]
            token_logits = seq_active_logits[token_idxs,:]
            #print('token logits shape: ', token_logits.shape)
            active_token_logits.append(token_logits)
        
        active_token_logits = torch.cat(active_token_logits, dim=0)
        active_token_labels = torch.cat(active_token_labels, dim=0)
        
        #print('active token logits shape: ', active_token_logits.shape)
        #print('active label shape: ', active_token_labels.shape)
        
        
        
        #EVERYTHING BELOW HAS SHAPE [TOTAL # TOKENS, :]
        #print()
        #print('Active Logits shape: ', active_logits.shape)
        #print('Active Labels shape: ', active_labels.shape)
        
        #print('Active Logits type: ', active_token_logits.is_cuda)
        #print('Active Labels type: ', active_token_labels.is_cuda)
        
        active_token_labels = active_token_labels.to(self.device)
        #we need a really small learning rate to make this work 
        loss = self.loss_func(active_token_logits, active_token_labels)
        
        active_preds = torch.argmax(active_token_logits, dim = -1)
        #print('Active GT Probs Shape: ', active_gt_probs.shape)
        #print('Active Preds Probs Shape: ', active_preds.shape)

        active_token_preds = active_preds.detach().cpu().numpy()
        active_token_labels = active_token_labels.detach().cpu().numpy()
        

        acc = accuracy_score(active_token_labels, active_token_preds)


            
        return {"loss": loss, 'train_loss': loss, 'train_acc':acc}
        
    def training_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        self.training_stats['train_losses'].append(avg_loss.detach().cpu())
        
        avg_acc = np.stack([x["train_acc"] for x in outputs]).mean()
        self.training_stats['train_accs'].append(avg_acc)
        
        print('Train Loss: ', avg_loss.detach().cpu())
        
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