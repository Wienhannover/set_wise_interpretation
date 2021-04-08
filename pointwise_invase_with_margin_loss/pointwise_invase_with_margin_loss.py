#!/usr/bin/env python
# coding: utf-8

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F

import sys
import argparse
import time
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from MQ2008_paired.utils_wei.pytorchtools import EarlyStopping

parser = argparse.ArgumentParser()

parser.add_argument('--lamb', type=float)
parser.add_argument('--margin', type=float)
parser.add_argument('--patience', type=int)
parser.add_argument('--start_stop', type=int)
parser.add_argument('--no_sample', type=int)


args_in = parser.parse_args()

activation_dict = {"relu": nn.ReLU(),"sigmoid": nn.Sigmoid(),"softmax": nn.Softmax(),"selu": nn.SELU()}



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, actor_model, critic_model, baseline_model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, actor_model,critic_model, baseline_model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, actor_model,critic_model, baseline_model)
            self.counter = 0

    def save_checkpoint(self, val_loss, actor_model, critic_model, baseline_model):
        
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...'.format(self.val_loss_min, val_loss))
        torch.save({'actor_model': actor_model.state_dict(),\
                    'critic_model': critic_model.state_dict(),\
                    'baseline_model': baseline_model.state_dict()}, self.path)
        self.val_loss_min = val_loss


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
        
        #layer_embedding = layer_list
        #self.linears_embedding = nn.Sequential(*layer_embedding)

        layer_list.append(activation_dict["sigmoid"])
        self.linears = nn.Sequential(*layer_list)
        
    def forward(self, x):
        return self.linears(x)

    #def embedding(self, x):
        #return self.linears_embedding(x)
        
#Use selected feature as the input and predict labels    
class Critic_RankNet(nn.Module):
    def __init__(self, inputs, hidden_size, outputs):
        super(Critic_RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            #nn.Dropout(0.5),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2,  inplace=True),#inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
            #nn.SELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, outputs),
            #nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, selection_1):
        
        input_1 = input_1 * selection_1
        result_1 = self.model(input_1) #预测input_1得分
        
        return result_1

    def predict(self, input, selection):
        
        input = input * selection
        result = self.model(input)
        return result   

#Use the original feature as the input and predict labels
class Baseline_RankNet(nn.Module):
    
    def __init__(self, inputs, hidden_size, outputs):
        super(Baseline_RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            #nn.Dropout(0.5),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2,  inplace=True),#inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
            #nn.SELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, outputs),
            #nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1):
        
        result_1 = self.model(input_1) #预测input_1得分
        return result_1

    def predict(self, input):
        result = self.model(input)
        return result   


class Dataset(data.Dataset):

    def __init__(self, data_path, num_triples):
                
        x_y = self.group_data(data_path)
        
        # (x0,x1) from same query; (x0/x1,x2) from different query
        self.x0, self.y0, self.x1, self.y1, self.x2, self.y2 = self.generate_triple(x_y, num_triples)
        
       
    def __getitem__(self, index):
        
        data1 = self.x0[index].float()
        y1 = self.y0[index].float()
        
        data2 = self.x1[index].float()
        y2 = self.y1[index].float()
        
        data3 = self.x2[index].float()
        y3 = self.y2[index].float()
        
        return data1, y1, data2, y2, data3, y3
    
    def __len__(self):
        return len(self.x0)
    
    def generate_triple(self, x_y, num_triples):
    
        sample_keys = []
        for key in x_y.keys():
            sample_keys.append(key)
        sample_keys = np.array(sample_keys)

        tmp_x0 = []
        tmp_y0 = []
        tmp_x1 = []
        tmp_y1 = []
        tmp_x2 = []
        tmp_y2 = []

        for i in range(num_triples):

            keys_sample = np.random.choice(sample_keys, args_in.no_sample, replace=False)

            i_index = np.random.choice(np.arange(x_y[keys_sample[0]].shape[0]), 2, replace=False)

            tmp_x0.append(x_y[keys_sample[0]][i_index[0], :-1])
            tmp_y0.append(x_y[keys_sample[0]][i_index[0], -1])
            tmp_x1.append(x_y[keys_sample[0]][i_index[1], :-1])
            tmp_y1.append(x_y[keys_sample[0]][i_index[1], -1])

            tmp_negative_x = []
            tmp_negative_y = []

            for j in range(1, args_in.no_sample):

                j_index = np.random.choice(np.arange(x_y[keys_sample[j]].shape[0]), 1)
               
                tmp_negative_x.append(x_y[keys_sample[j]][j_index.item(), :-1].tolist())
                tmp_negative_y.append(x_y[keys_sample[j]][j_index.item(), -1].tolist())

            tmp_x2.append(torch.tensor(tmp_negative_x))
            tmp_y2.append(torch.tensor(tmp_negative_y))

        return tmp_x0, tmp_y0, tmp_x1, tmp_y1, tmp_x2, tmp_y2
    
    # group data by query id
    def group_data(self, data_path):
        test_content = np.genfromtxt(data_path, dtype=np.dtype(str))

        #data in test set grouped by qid
        x_y = {}
        for i in range(test_content.shape[0]):
            qid = np.int(test_content[i][1][4:])

            features = []
            for j in range(2, test_content.shape[1]):
                features.append(np.float(test_content[i][j][-8:]))

            #labels as last column
            label = np.float(test_content[i][0])
    #         if(label > 1):
    #             label = 1  
            features.append(label)
            #注：原始文档中label 为0，1，2 

            if qid in x_y.keys():
                x_y[qid].append(features)
            else:
                x_y[qid] = []
                x_y[qid].append(features)
        for key in x_y.keys():
            x_y[key] = torch.tensor(x_y[key])

        return x_y

def get_loader(data_path, num_triples, batch_size, shuffle, drop_last):
    
    dataset = Dataset(data_path, num_triples)
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last=drop_last
    )
    return data_loader


def pair_actor_loss(actor_output_1, actor_output_2, actor_output_3, selection_1, selection_2, critic_loss_1, critic_loss_2, baseline_loss_1, baseline_loss_2, lamda, margin):
    
    Reward_1 = critic_loss_1.detach() - baseline_loss_1.detach()
    Pi_1 = (selection_1 * torch.log(actor_output_1 + 1e-8) + (1-selection_1) * torch.log(1-actor_output_1 + 1e-8)).sum(1)
    L0_1 = actor_output_1.mean(1)
    custom_actor_loss_1 = Pi_1 * Reward_1 + lamda * L0_1
    #*************************************************************************
    Reward_2 = critic_loss_2.detach() - baseline_loss_2.detach()
    Pi_2 = (selection_2 * torch.log(actor_output_2 + 1e-8) + (1-selection_2) * torch.log(1-actor_output_2 + 1e-8)).sum(1)
    L0_2 = actor_output_2.mean(1)
    custom_actor_loss_2 = Pi_2 * Reward_2 + lamda * L0_2
    #***************************************************************************
    
    embedding_1 = torch.nn.functional.softmax(actor_output_1, dim=1)
    embedding_2 = torch.nn.functional.softmax(actor_output_2, dim=1)
    embedding_3 = torch.nn.functional.softmax(actor_output_3, dim=2)

    distance_0 = -(embedding_1 * torch.log(embedding_2 + 1e-8) + (1-embedding_1) * torch.log(1-embedding_2+1e-8)).sum(1)
    distance_1 = -(embedding_1.unsqueeze(1) * torch.log(embedding_3 + 1e-8) + (1-embedding_1.unsqueeze(1)) * torch.log(1-embedding_3+1e-8)).sum(2).mean(1)
    final_selection_loss = F.relu(distance_0 - distance_1 + margin)
    
    #embedding_1 = F.normalize(actor_output_1, 2, 1)
    #embedding_2 = F.normalize(actor_output_2, 2, 1)
    #embedding_3 = F.normalize(actor_output_3, 2, 2)

    #distance_0 = (embedding_1 - embedding_2).pow(2).sum(1)
    #distance_1 = (embedding_1.unsqueeze(1) - embedding_3).pow(2).sum(2).mean(1)

    #final_selection_loss = F.relu(distance_0 - args_in.w_n*distance_1 + margin)


    value1 = (custom_actor_loss_1.mean() + custom_actor_loss_2.mean())/float(2)
    value2 = final_selection_loss.mean()

    loss_final = value1 + value2
    loss_part1 = value1
    loss_part2 = value2
    
    return loss_final, loss_part1, loss_part2, distance_0.mean(), distance_1.mean()

def train_model(actor_model, critic_model, baseline_model, patience, epoch_start_early_stopping, saved_path, epochs, lamda, margin):
        
    #actor_optimizer = torch.optim.Adam(actor_model.parameters(),lr = 1e-5, weight_decay=1e-5)
    actor_optimizer = torch.optim.Adam(actor_model.parameters(),lr = 1e-6, weight_decay=1e-5)
    #actor_optimizer = torch.optim.Adam(actor_model.parameters(),lr = 1e-7, weight_decay=1e-5)
    #actor_optimizer = torch.optim.Adam(actor_model.parameters(),lr = 1e-8, weight_decay=1e-5)

    critic_optimizer = torch.optim.Adam(critic_model.parameters(),lr = 1e-4, weight_decay=1e-5)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(),lr = 1e-4, weight_decay=1e-5)
    
    critic_criterion = nn.CrossEntropyLoss()
    baseline_criterion = nn.CrossEntropyLoss()
    
    m = torch.nn.Softmax(dim=1)
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, path=saved_path,verbose=True)

    plot_train_loss = []
    plot_train_acc = []
    plot_vali_loss = []
    plot_vali_acc = []
    
    for epoch in range(epochs):
        
        epoch_train_actor_loss_output = []
        epoch_train_actor_loss_output_part1 = []
        epoch_train_actor_loss_output_part2 = []

        epoch_train_actor_loss_output_distance1 = []
        epoch_train_actor_loss_output_distance2 = []

        epoch_train_critic_acc = []
        
        actor_model.train()
        critic_model.train()
        baseline_model.train()

        for batch, (data1, y1, data2, y2, data3, y3) in enumerate(train_loader):

            y1_one_hot = torch.zeros(batch_size, 3).scatter_(1, y1.long().view(-1, 1) ,1)
            y2_one_hot = torch.zeros(batch_size, 3).scatter_(1, y2.long().view(-1, 1) ,1)

            y1_one_hot = y1_one_hot.to(device)
            y2_one_hot = y2_one_hot.to(device)

            data1 = data1.float().to(device)
            y1 = y1.to(device)
            data2 = data2.float().to(device)
            y2 = y2.to(device)
            data3 = data3.float().to(device)
            y3 = y3.to(device)
            
            
            actor_output_1 = actor_model(data1)
            selection_1 = torch.bernoulli(actor_output_1)
            
            actor_output_2 = actor_model(data2)
            selection_2 = torch.bernoulli(actor_output_2)
            
            actor_output_3 = actor_model(data3)
            
            # train critic model
            critic_output_01 = critic_model(data1, selection_1)
            label_1 = y1.long()
            critic_loss_output_1 = critic_criterion(critic_output_01, label_1)

            critic_output_02 = critic_model(data2, selection_2)
            label_2 = y2.long()
            critic_loss_output_2 = critic_criterion(critic_output_02, label_2)

            critic_loss_output = (critic_loss_output_1 + critic_loss_output_2)/float(2)
            
            critic_optimizer.zero_grad()
            critic_loss_output.backward(retain_graph = True)
            critic_optimizer.step()
            
            critic_output_1 = critic_model.predict(data1, selection_1)
            critic_output_2 = critic_model.predict(data2, selection_2)
                        
            #--------Performance of predictor------------------------------------------------------
            critic_acc_1 = torch.eq(torch.max(critic_output_1, dim=1)[1], y1.long()).sum().item() / float(batch_size)
            critic_acc_2 = torch.eq(torch.max(critic_output_2, dim=1)[1], y2.long()).sum().item() / float(batch_size)
            epoch_train_critic_acc.append((critic_acc_1 + critic_acc_2) / float(2))
            #--------------------------------------------------------------------------------------
            
            # train basseline model
            baseline_output_01 = baseline_model(data1)
            label_1 = y1.long()
            baseline_loss_output_1 = baseline_criterion(baseline_output_01, label_1)

            baseline_output_02 = baseline_model(data2)
            label_2 = y2.long()
            baseline_loss_output_2 = baseline_criterion(baseline_output_02, label_2)
 
            baseline_loss_output = (baseline_loss_output_1 + baseline_loss_output_2)/float(2)
            
            baseline_optimizer.zero_grad()
            baseline_loss_output.backward(retain_graph = True)
            baseline_optimizer.step()
            
            baseline_output_1 = baseline_model.predict(data1)
            baseline_output_2 = baseline_model.predict(data2)

            #y1_one_hot = F.one_hot(y1.long(), num_classes=3)
            #y2_one_hot = F.one_hot(y2.long(), num_classes=3)

            critic_loss_1 = -(y1_one_hot.float()  * torch.log(m(critic_output_1) + 1e-8)).sum(1)
            critic_loss_2 = -(y2_one_hot.float()  * torch.log(m(critic_output_2) + 1e-8)).sum(1)
                        
            baseline_loss_1 = -(y1_one_hot.float() * torch.log(m(baseline_output_1) + 1e-8)).sum(1)
            baseline_loss_2 = -(y2_one_hot.float() * torch.log(m(baseline_output_2) + 1e-8)).sum(1) 

            # update selector network
            actor_loss_output = pair_actor_loss(actor_output_1, actor_output_2, actor_output_3, selection_1, selection_2, critic_loss_1, critic_loss_2, baseline_loss_1, baseline_loss_2, lamda, margin)
                        
            actor_optimizer.zero_grad()
            actor_loss_output[0].backward()
            actor_optimizer.step()
                        
            epoch_train_actor_loss_output.append(actor_loss_output[0].item())
            epoch_train_actor_loss_output_part1.append(actor_loss_output[1].item())
            epoch_train_actor_loss_output_part2.append(actor_loss_output[2].item())

            epoch_train_actor_loss_output_distance1.append(actor_loss_output[3].item())
            epoch_train_actor_loss_output_distance2.append(actor_loss_output[4].item())

            
        print(epoch+1,"***********************************************************************")
        print("---------------train actor loss-------------", np.mean(epoch_train_actor_loss_output), np.mean(epoch_train_actor_loss_output_part1), np.mean(epoch_train_actor_loss_output_part2), np.mean(epoch_train_actor_loss_output_distance1), np.mean(epoch_train_actor_loss_output_distance2))
        print("---------------train critic acc-------------", np.mean(epoch_train_critic_acc))

        plot_train_loss.append(np.mean(epoch_train_actor_loss_output))
        plot_train_acc.append(np.mean(epoch_train_critic_acc))
            
        epoch_vali_actor_loss_output = []
        epoch_vali_actor_loss_output_part1 = []
        epoch_vali_actor_loss_output_part2 = []

        epoch_vali_actor_loss_output_distance1 = []
        epoch_vali_actor_loss_output_distance2 = []

        epoch_vali_critic_acc = []

        actor_model.eval()
        critic_model.eval()
        baseline_model.eval()  
        
        with torch.no_grad():   
            for batch, (data1, y1, data2, y2, data3, y3) in enumerate(vali_loader):

                y1_one_hot = torch.zeros(batch_size, 3).scatter_(1, y1.long().view(-1, 1) ,1)
                y2_one_hot = torch.zeros(batch_size, 3).scatter_(1, y2.long().view(-1, 1) ,1)

                y1_one_hot = y1_one_hot.to(device)
                y2_one_hot = y2_one_hot.to(device)

                data1 = data1.float().to(device)
                y1 = y1.to(device)
                data2 = data2.float().to(device)
                y2 = y2.to(device)
                data3 = data3.float().to(device)
                y3 = y3.to(device)

                                                
                vali_actor_output_1 = actor_model(data1)
                vali_selection_1 = vali_actor_output_1.ge(0.5).type(torch.float)
                                          
                vali_actor_output_2 = actor_model(data2)
                vali_selection_2 = vali_actor_output_2.ge(0.5).type(torch.float)
                   
                vali_actor_output_3 = actor_model(data3)

                                                                          
                vali_critic_output_1 = critic_model.predict(data1, vali_selection_1)
                vali_critic_output_2 = critic_model.predict(data2, vali_selection_2)
                               
                vali_baseline_output_1 = baseline_model.predict(data1)
                vali_baseline_output_2 = baseline_model.predict(data2)

                #----------------------------------------------------------
                #y1_one_hot = F.one_hot(y1.long(), num_classes=3)
                #y2_one_hot = F.one_hot(y2.long(), num_classes=3)
                
                vali_critic_loss_1 = -(y1_one_hot.float()  * torch.log(m(vali_critic_output_1) + 1e-8)).sum(1)
                vali_critic_loss_2 = -(y2_one_hot.float()  * torch.log(m(vali_critic_output_2) + 1e-8)).sum(1)
                        
                vali_baseline_loss_1 = -(y1_one_hot.float() * torch.log(m(vali_baseline_output_1) + 1e-8)).sum(1)
                vali_baseline_loss_2 = -(y2_one_hot.float() * torch.log(m(vali_baseline_output_2) + 1e-8)).sum(1) 
                #----------------------------------------------------------
                
                vali_actor_loss_output = pair_actor_loss(vali_actor_output_1, vali_actor_output_2, vali_actor_output_3, vali_selection_1, vali_selection_2, vali_critic_loss_1, vali_critic_loss_2, vali_baseline_loss_1, vali_baseline_loss_2, lamda, margin)

                epoch_vali_actor_loss_output.append(vali_actor_loss_output[0].item())
                epoch_vali_actor_loss_output_part1.append(vali_actor_loss_output[1].item())
                epoch_vali_actor_loss_output_part2.append(vali_actor_loss_output[2].item())

                epoch_vali_actor_loss_output_distance1.append(vali_actor_loss_output[3].item())
                epoch_vali_actor_loss_output_distance2.append(vali_actor_loss_output[4].item())

                
                #--------Performance of predictor------------------------------------------------------
                vali_critic_acc_1 = torch.eq(torch.max(vali_critic_output_1, dim=1)[1], y1.long()).sum().item() / float(batch_size)
                vali_critic_acc_2 = torch.eq(torch.max(vali_critic_output_2, dim=1)[1], y2.long()).sum().item() / float(batch_size)
                epoch_vali_critic_acc.append((vali_critic_acc_1 + vali_critic_acc_2) / float(2))
                #--------------------------------------------------------------------------------------
                
        print("---------------Vali actor loss-------------", np.mean(epoch_vali_actor_loss_output), np.mean(epoch_vali_actor_loss_output_part1), np.mean(epoch_vali_actor_loss_output_part2), np.mean(epoch_vali_actor_loss_output_distance1), np.mean(epoch_vali_actor_loss_output_distance2))
        print("---------------Vali critic acc-------------", np.mean(epoch_vali_critic_acc))

        plot_vali_loss.append(np.mean(epoch_vali_actor_loss_output))
        plot_vali_acc.append(np.mean(epoch_vali_critic_acc))
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        
        if epoch > epoch_start_early_stopping:
            valid_loss = np.mean(epoch_vali_actor_loss_output)
            early_stopping(valid_loss, actor_model, critic_model, baseline_model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    
    # plot out
    x = np.arange(len(plot_train_loss))

    plt.subplot(2,2,1)
    plt.plot(x, np.array(plot_train_loss))
    plt.xlabel('epochs')
    plt.ylabel('training loss')

    plt.subplot(2,2,2)
    plt.plot(x, np.array(plot_train_acc))
    plt.ylim((0, 1))
    plt.xlabel('epochs')
    plt.ylabel('training acc')

    plt.subplot(2,2,3)
    plt.plot(x, np.array(plot_vali_loss))
    plt.xlabel('epochs')
    plt.ylabel('validation loss')

    plt.subplot(2,2,4)
    plt.plot(x, np.array(plot_vali_acc))
    plt.ylim((0, 1))
    plt.xlabel('epochs')
    plt.ylabel('validation acc')
    
    plt.tight_layout()
    #labels = ['training loss', 'training acc', 'validation loss', 'validation acc']
    #plt.legend(labels)
    plt.savefig('plot.png')

    checkpoint = torch.load(saved_path)

    actor_model.load_state_dict(checkpoint['actor_model'])
    critic_model.load_state_dict(checkpoint['critic_model'])
    baseline_model.load_state_dict(checkpoint['baseline_model'])
        
    return actor_model,critic_model,baseline_model


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


model_para = {'lambda':args_in.lamb,
              'actor_h_dim':300,
              'critic_h_dim':200,
              'baseline_h_dim':200,
              'actor_output' :46,
              'critic_output':3,
              'baseline_output':3,
              'n_layer':10,
              'activation':'selu',
              'learning_rate':0.0001}
batch_size = 128

actor_list = []
critic_list = []
baseline_list = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

margin = args_in.margin

patience = args_in.patience
epoch_start_early_stopping = args_in.start_stop

train_num = 12800
vali_num = 12800
test_num = 12800

for k in range(1):

    y_train = []
    x_train = []
    query_id = []
    array_train_x1 = []
    array_train_x0 = []

    path = "./MQ2008/Fold{}/".format(k+1)

    train_path = path + 'train.txt'
    train_loader = get_loader(train_path, train_num, batch_size, shuffle=True, drop_last=True)

    vali_path = path + 'vali.txt'
    vali_loader = get_loader(vali_path, vali_num, batch_size, shuffle=True, drop_last=True)

    test_path = path + 'test.txt'
    test_loader = get_loader(test_path, test_num, batch_size, shuffle=True, drop_last=True)

    actor = Actor(46, model_para['actor_h_dim'], model_para['actor_output'], model_para['n_layer'], model_para['activation']).to(device)
    critic = Critic_RankNet(46, model_para['critic_h_dim'], model_para['critic_output']).to(device)
    baseline = Baseline_RankNet(46, model_para['baseline_h_dim'], model_para['baseline_output']).to(device)

    actor.apply(init_weights)
    critic.apply(init_weights)
    baseline.apply(init_weights)
    
    tmp_saved_path = 'checkpoint.pt'
    
    trained_model_list = train_model(actor, critic, baseline, patience, epoch_start_early_stopping, tmp_saved_path, 30000, model_para['lambda'], margin)

    actor_list.append(trained_model_list[0])
    critic_list.append(trained_model_list[1])
    baseline_list.append(trained_model_list[2])


for k in range(1):    
    torch.save({'actor_model': actor_list[k].state_dict(),\
            'critic_model': critic_list[k].state_dict(),\
            'baseline_model': baseline_list[k].state_dict()}, 'models_dict.pt')


