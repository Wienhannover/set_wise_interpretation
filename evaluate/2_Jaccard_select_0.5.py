#!/usr/bin/env python
# coding: utf-8

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import pytrec_eval
import json
import sys
import argparse
import time
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--feat_num', type=int)
parser.add_argument('--model_path', type=str)
args_in = parser.parse_args()

activation_dict = {"relu": nn.ReLU(),"sigmoid": nn.Sigmoid(),"softmax": nn.Softmax(),"selu": nn.SELU()}
#---------------------------------------------------------------------------------------------------------------------------------------
# group data by query id
def group_data(txt_path, dataset):
    test_content = np.genfromtxt("{}{}.txt".format(txt_path, dataset),dtype=np.dtype(str))

    #data in test set grouped by qid
    x_y = {}
    for i in range(test_content.shape[0]):
        qid = np.int(test_content[i][1][4:])

        features = []
        for j in range(2, test_content.shape[1]):
            #features.append(np.float(test_content[i][j][-8:])) #specific on mq2008
            features.append(np.float(test_content[i][j].split(':')[1]))

        #labels as last column
        label = np.float(test_content[i][0])
        features.append(label)


        if qid in x_y.keys():
            x_y[qid].append(features)
        else:
            x_y[qid] = []
            x_y[qid].append(features)
    for key in x_y.keys():
        x_y[key] = torch.tensor(x_y[key])
        
    return x_y
#---------------------------------------------------------------------------------------------------------------------------------------
#Use feature as the input and output selection probability
class Actor(nn.Module):
    
    def __init__(self, input_dim, h_dim, output_dim, layer_num, activation):
        super(Actor, self).__init__()
        #add regularization term in loss in pytroch, not every layer in keras
        layer_list = []
        layer_list.append(nn.Linear(input_dim, h_dim))
        layer_list.append(activation_dict[activation])
        for _ in range(layer_num - 2):
            layer_list.append(nn.Linear(h_dim, h_dim))
            layer_list.append(activation_dict[activation])
        layer_list.append(nn.Linear(h_dim, output_dim))       
        layer_list.append(activation_dict["sigmoid"])
        self.linears = nn.Sequential(*layer_list)
        
    def forward(self, x):
        return self.linears(x)
          
#---------------------------------------------------------------------------------------------------------------------------------------

final_output = []


model_para = {'actor_h_dim':300,
              'critic_h_dim':200,
              'baseline_h_dim':200,
              'actor_output' :args_in.feat_num,
              'critic_output':1,
              'baseline_output':1,
              'n_layer':10,
              'activation':'selu'}

actor_list = []

for k in range(1):
    
    PATH = args_in.model_path
    checkpoint = torch.load(PATH, map_location='cpu')
    
    in_feature_num = model_para['actor_output']
    actor_model = Actor(in_feature_num, model_para['actor_h_dim'], model_para['actor_output'], model_para['n_layer'], model_para['activation'])
    
    actor_model.load_state_dict(checkpoint['actor_model'])
    
    actor_list.append(actor_model)




def jaccard_sim(a, b):
    a_tmp = []
    b_tmp = []
    for item in torch.nonzero(a, as_tuple =False):
        a_tmp.append(item.item())
        
    for item in torch.nonzero(b, as_tuple =False):
        b_tmp.append(item.item())
    
    unions = len(set(a_tmp).union(set(b_tmp)))
    intersections = len(set(a_tmp).intersection(set(b_tmp)))
    return intersections / unions



    
#----------------------------------------within one query---------------------------------------------------------
Selection_k_fold = []
for k in range(1):
    actor_model = actor_list[k]
    txt_path = args_in.data_path + '/Fold{}/'.format(k+1)
    dataset_ = "test"
    x_y = group_data(txt_path, dataset_)

    Selection_fold = {}
    for qid in x_y.keys():
        
        if x_y[qid].shape[0] <=1 :#*****************MSLR-specific***********************************
            print("*************************************", qid)
            continue
        
        actor_output = actor_model(x_y[qid][:,:-1].float())                
        selection = actor_output.ge(0.5).int()
        Selection_fold[qid] = selection
    Selection_k_fold.append(Selection_fold)
    
print('selection finished')
#---------------------------------------selection finished------------------------------
for k in range(1):
    S = Selection_k_fold[k]
    S_j_score = {}
    for key in S.keys():
        tmp = S[key]
        tmp_j = []
        
        #MSLR-WEB10K is too big
        if args_in.data_path == "MSLR-WEB10K":#*****************MSLR-specific***********************************
            for i in range(1, tmp.shape[0]):
                if tmp[i].sum() == 0:
                    print("first 0")
                    continue
                tmp_j.append(jaccard_sim(tmp[0],tmp[i]))
        
        else:
            for i in range(tmp.shape[0]):
                if tmp[i].sum() == 0:
                    continue
                for j in range(i+1, tmp.shape[0], 1):     
                    if tmp[j].sum() == 0:
                        continue
                    tmp_j.append(jaccard_sim(tmp[i],tmp[j]))

            if len(tmp_j) > 0:
                S_j_score[key] = tmp_j
            else:
                print(key,"all zeros or only one doc has selected feautres")

    X = []
    Y = []
    i = 0
    for key in S_j_score.keys():
        X.append(i)
        i = i + 1
        Y.append(np.mean(S_j_score[key]))
    X = np.array(X)
    Y = np.array(Y)

    final_output.append(np.mean(Y))


#-------------------------------------different queries----------------------------------------------------------------

for k in range(1):
    S = Selection_k_fold[k]

    sample_keys = []
    for key in S.keys():
        sample_keys.append(key)
    sample_keys = np.array(sample_keys)  

    num_points = 10000
    X = np.arange(num_points)
    Y = []
    for i in range(num_points):
        two_keys_sample = np.random.choice(sample_keys, 2, replace=False)           
        i_index = np.random.choice(np.arange(S[two_keys_sample[0]].shape[0]), 1)
        j_index = np.random.choice(np.arange(S[two_keys_sample[1]].shape[0]), 1)

        while (S[two_keys_sample[0]][i_index.item()].sum == 0 or S[two_keys_sample[1]][j_index.item()].sum() == 0):
            two_keys_sample = np.random.choice(sample_keys, 2, replace=False)           
            i_index = np.random.choice(np.arange(S[two_keys_sample[0]].shape[0]), 1)
            j_index = np.random.choice(np.arange(S[two_keys_sample[1]].shape[0]), 1)

        Y.append(jaccard_sim(S[two_keys_sample[0]][i_index.item()],S[two_keys_sample[1]][j_index.item()]))
    Y = np.array(Y)

    final_output.append(np.mean(Y))


print(final_output)
tmp_s = ''
for item in final_output:
    
    tmp_s = tmp_s + '& ' + str(round(item, 4))
    
print(tmp_s)



