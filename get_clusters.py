# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 12:29:53 2021

@author: Shadow
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering


import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from collections import Counter
import pandas as pd

import pickle 
import os
from collections import Counter 

def get_clusters(cluster_data):
    
    #clustering 2d does not include correctnesss as a feature
    #this is how the clusters were created in datamaps and mind your outliers 
    #the first two columns correspond to variance and confidence
    cluster_train_data = cluster_data.iloc[:,0:2]
    
    print('Cluster train data shape: ', cluster_train_data.shape)
    print('cluster train data head: ', cluster_train_data.head())
    clustering_2d = SpectralClustering(n_clusters=3,
                                       assign_labels='discretize',
                                       random_state=0).fit(cluster_train_data)
    
    return {'cluster2d_labels':clustering_2d.labels_}

def get_cluster_regions(cluster_labels, cluster_data_df, token_input_data):
    
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = [i for i in range(0, len(cluster_labels))]
    cluster_map['cluster'] = cluster_labels
    
     #Cluster 1
    cluster1_mask = np.array(cluster_map[cluster_map.cluster == 0].data_index)
    #print('cluster mask shape: ', cluster1_mask.shape)
    cluster1 = cluster_data_df.iloc[cluster1_mask,:]
    cluster1_avg_conf = np.mean(np.array(cluster1['confidence']))
    cluster1_token_input = token_input_data.iloc[cluster1_mask, :]
    
    #Cluster 2
    cluster2_mask = np.array(cluster_map[cluster_map.cluster == 1].data_index)
    #print('cluster mask shape: ', cluster1_mask.shape)
    cluster2 = cluster_data_df.iloc[cluster2_mask,:]
    cluster2_avg_conf = np.mean(np.array(cluster2['confidence']))
    cluster2_token_input = token_input_data.iloc[cluster2_mask, :]
 
    #Cluster 3
    cluster3_mask = np.array(cluster_map[cluster_map.cluster == 2].data_index)
    #print('cluster mask shape: ', cluster1_mask.shape)
    cluster3 = cluster_data_df.iloc[cluster3_mask,:]
    cluster3_avg_conf = np.mean(np.array(cluster3['confidence']))
    cluster3_token_input = token_input_data.iloc[cluster3_mask, :]

    cluster_info = [(cluster1_avg_conf, cluster1_token_input, cluster1), 
                    (cluster2_avg_conf, cluster2_token_input, cluster2), 
                    (cluster3_avg_conf, cluster3_token_input, cluster3)]
    
    sorted_clusters = sorted(cluster_info, key=lambda x: x[0])
    
    cluster_regions = {}
    cluster_regions['hard'] = sorted_clusters[0]
    cluster_regions['ambig'] = sorted_clusters[1]
    cluster_regions['easy'] = sorted_clusters[2]
    
    return cluster_regions

def main(directory, sample_size=100000):
    
    
    with open(directory+'/'+'bert_train_stats.pkl', 'rb') as f:
        log = pickle.load(f)
        
    with open(directory+'/'+'bert_test_stats.pkl', 'rb') as f:
        test_log = pickle.load(f)
        
    with open(directory+'/'+'token_inputs.pkl', 'rb') as f:
        token_inputs = pickle.load(f) 
        
    #Calculating the variance, confidence, and correctness of the training samples
    confidence = np.mean(log['gt_probs'], axis=-1)
    variance = np.var(log['gt_probs'], axis=-1)
    #correctness = np.mean(log['correctness'], axis = -1)
    
    correctness = []
    for idx in range(log['correctness'].shape[0]):
        
        correct_bool = log['correctness'][idx,:]
        correctness.append(np.count_nonzero(correct_bool))
    
    correctness = np.array(correctness)
    
    #Performing our cluster analysis
    
    
    #this includes correctness
    cluster_data_df = pd.DataFrame(data={'variance': variance, 
                                         'confidence':confidence, 
                                         'correctness':correctness, 
                                         'labels': token_inputs['train_labels']})
    
    
    
    token_input_data = pd.DataFrame(data={'token_label_maks': [token_inputs['token_label_masks'][idx, :] for idx in range(token_inputs['token_label_masks'].shape[0])],
                                         'labels': token_inputs['train_labels'],
                                         'attention_mask': [token_inputs['attention_mask'][idx, :] for idx in range(token_inputs['attention_mask'].shape[0])],
                                         'input_ids': [token_inputs['input_ids'][idx, :] for idx in range(token_inputs['input_ids'].shape[0])],
                                         'sentence_idxs': token_inputs['sentence_idxs'],
                                         'token_idxs':token_inputs['token_idxs'],
                                         })
    
    cluster_sample_mask = np.random.RandomState(seed=21).permutation(cluster_data_df.shape[0])
    cluster_sample_mask = cluster_sample_mask[0:sample_size]
    
    cluster_data_sample = cluster_data_df.iloc[cluster_sample_mask, :]
    token_input_sample = token_input_data.iloc[cluster_sample_mask, :]
    
    clusters = get_clusters(cluster_data_sample)
    with open(directory+'/'+'clusters.pkl', 'wb') as f:
        pickle.dump(clusters, f)
    
    if not os.path.exists(directory+'/token_data'):
        os.makedirs(directory+'/token_data')

    cluster_regions = get_cluster_regions(clusters['cluster2d_labels'], cluster_data_sample, token_input_sample)
    
    tag_count = Counter(cluster_regions['easy'][2]['labels'].to_list())
    print()
    print('The # of "O" tags in Easy Cluster: ', tag_count['O'])
    print('The # of "SN" tags in Easy Cluster: ', tag_count['SN'])
    print('The # of "SV" tags in Easy Cluster: ', tag_count['SV'])
    
    tag_count = Counter(cluster_regions['ambig'][2]['labels'].to_list())
    print()
    print('The # of "O" tags in Ambigious Cluster: ', tag_count['O'])
    print('The # of "SN" tags in Ambigious Cluster: ', tag_count['SN'])
    print('The # of "SV" tags in Ambigious Cluster: ', tag_count['SV'])
    
    tag_count = Counter(cluster_regions['hard'][2]['labels'].to_list())
    print()
    print('The # of "O" tags in Hard Cluster: ', tag_count['O'])
    print('The # of "SN" tags in Hard Cluster: ', tag_count['SN'])
    print('The # of "SV" tags in Hard Cluster: ', tag_count['SV'])


    with open(directory+'/token_data/sample_data.pkl', 'wb') as f:
        pickle.dump(token_input_sample, f)
    
    with open(directory+'/token_data/easy_data.pkl', 'wb') as f:
        pickle.dump(cluster_regions['easy'][1], f)
    
    with open(directory+'/token_data/ambig_data.pkl', 'wb') as f:
        pickle.dump(cluster_regions['ambig'][1], f)
    
    with open(directory+'/token_data/hard_data.pkl', 'wb') as f:
        pickle.dump(cluster_regions['hard'][1], f)
        
        


if __name__ == "__main__":
    
    main(directory = 'results/memc', sample_size=100000)
        