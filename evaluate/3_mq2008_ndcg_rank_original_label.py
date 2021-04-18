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
# 0: prob > 0.5
parser.add_argument('--ndcg_num', type=int)
parser.add_argument('--ndcg_rank_num', type=str)

args_in = parser.parse_args()
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


def get_qrel_run_predictor(k, option):
    # k start from 0 in loop
    
    actor_model = actor_list[k]
    critic_model = critic_list[k]
    
    qrel = {}
    run = {}
    selection_dic = {}

    txt_path = args_in.data_path + '/Fold{}/'.format(k+1)
    dataset_ = "test"
    x_y = group_data(txt_path, dataset_)

    for qid in x_y.keys():
        
        actor_output = actor_model(x_y[qid][:,:-1].float())
        
        if option == 0:
            selection = actor_output.ge(0.5).float()
            
        else:
            index = torch.topk(actor_output, option)[1]
            selection = torch.zeros(actor_output.shape)
            for i in range(index.shape[0]):
                selection[i,index[i]] = 1
            
        selection_dic[str(qid)] = selection
        
        critic_output = critic_model.predict(x_y[qid][:,:-1].float(), selection)

        y_ture = x_y[qid][:,-1].type(torch.float).numpy()
        y_score = critic_output.reshape(1, -1)[0].detach().numpy()
        
        qid_true_relevance = {}
        for i in range(len(y_ture)):
            key = 'd' + str(i+1)
            qid_true_relevance[key] = int(y_ture[i])
        qrel[str(qid)] = qid_true_relevance

        qid_score_relevace = {}
        for i in range(len(y_score)):
            key = 'd' + str(i+1)
            #*****************************class numpy.int16 --->class int****
            qid_score_relevace[key] = y_score[i].item()
        run[str(qid)] = qid_score_relevace
    
    #return qrel, run, selection_dic
    return qrel, run


def get_qrel_run_baseline(k):
    # k start from 0 in loop
    
    baseline_model = baseline_list[k]
    
    qrel = {}
    run = {}

    txt_path = args_in.data_path + '/Fold{}/'.format(k+1)
    dataset_ = "test"
    x_y = group_data(txt_path, dataset_)

    for qid in x_y.keys():
        
        baseline_output = baseline_model.predict(x_y[qid][:,:-1].float())

        y_ture = x_y[qid][:,-1].type(torch.float).numpy()
        y_score = baseline_output.reshape(1, -1)[0].detach().numpy()

        qid_true_relevance = {}
        for i in range(len(y_ture)):
            key = 'd' + str(i+1)
            qid_true_relevance[key] = int(y_ture[i])
        qrel[str(qid)] = qid_true_relevance

        qid_score_relevace = {}
        for i in range(len(y_score)):
            key = 'd' + str(i+1)
            qid_score_relevace[key] = y_score[i].item()
        run[str(qid)] = qid_score_relevace
    
    return qrel,run


activation_dict = {"relu": nn.ReLU(),"sigmoid": nn.Sigmoid(),"softmax": nn.Softmax(),"selu": nn.SELU()}

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
        
#Use selected feature as the input and predict labels    
class Critic_RankNet(nn.Module):
    def __init__(self, inputs, hidden_size, outputs):
        super(Critic_RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, outputs),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, selection_1, input_2, selection_2):
        
        input_1 = np.array(input_1) * np.array(selection_1)
        result_1 = self.model(torch.from_numpy(input_1)) 
        
        input_2 = np.array(input_2) * np.array(selection_2)
        result_2 = self.model(torch.from_numpy(input_2)) 
        
        pred = self.sigmoid(result_1 - result_2)
        return pred

    def predict(self, input, selection):
        
        input = np.array(input) * np.array(selection)
        result = self.model(torch.from_numpy(input))
        return result   

#Use the original feature as the input and predict labels
class Baseline_RankNet(nn.Module):
    
    def __init__(self, inputs, hidden_size, outputs):
        super(Baseline_RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, outputs),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        
        result_1 = self.model(input_1)
        result_2 = self.model(input_2)
        pred = self.sigmoid(result_1 - result_2)
        return pred

    def predict(self, input):
        result = self.model(input)
        return result   


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
critic_list = []
baseline_list = []

for k in range(1):
    
    PATH = args_in.model_path
    checkpoint = torch.load(PATH, map_location='cpu')
    
    in_feature_num = model_para['actor_output']
    actor_model = Actor(in_feature_num, model_para['actor_h_dim'], model_para['actor_output'], model_para['n_layer'], model_para['activation'])
    critic_model = Critic_RankNet(in_feature_num, model_para['critic_h_dim'], model_para['critic_output'])
    baseline_model = Baseline_RankNet(in_feature_num, model_para['baseline_h_dim'], model_para['baseline_output'])
    
    actor_model.load_state_dict(checkpoint['actor_model'])
    critic_model.load_state_dict(checkpoint['critic_model'])
    baseline_model.load_state_dict(checkpoint['baseline_model'])
    
    actor_list.append(actor_model)
    critic_list.append(critic_model)
    baseline_list.append(baseline_model)





NDCG_k_fold = []

num = args_in.ndcg_num
ndcg_rank = args_in.ndcg_rank_num
#------------------------------------------------------critic---------------------------------------------------------------------------------

for k in range(1):
    
    qrel,run = get_qrel_run_predictor(k, num)
        
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {ndcg_rank})    
    tmp = json.dumps(evaluator.evaluate(run), indent=1)
    result = json.loads(tmp)
    
    NDCG = []
    for key in result.keys():
        NDCG.append(result[key][ndcg_rank])
    NDCG_k_fold.append(np.mean(NDCG))   

final_output.append(np.mean(NDCG_k_fold))
#----------------------------------------------------------baseline----------------------------------------------------------------------------

NDCG_k_fold = []

for k in range(1):
    qrel,run = get_qrel_run_baseline(k)
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {ndcg_rank})    
    tmp = json.dumps(evaluator.evaluate(run), indent=1)
    result = json.loads(tmp)
    
    NDCG = []
    for key in result.keys():
        NDCG.append(result[key][ndcg_rank])
    NDCG_k_fold.append(np.mean(NDCG))
        
final_output.append(np.mean(NDCG_k_fold))
#---------------------------------------------------------------------------------------------------------------------------------------
print(final_output)



tmp_s = ''
for item in final_output:
    
    tmp_s = tmp_s + '& ' + str(round(item, 4))
    
print(tmp_s)



