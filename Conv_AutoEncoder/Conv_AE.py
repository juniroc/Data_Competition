#!/usr/bin/env python
# coding: utf-8

# ### Import Library

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.dates as md
from matplotlib import pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
import random


# ### get_seed

# In[3]:


random_seed = 42
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(random_seed)
random.seed(random_seed)

torch.cuda.manual_seed(random_seed)


# ### Encoder

# In[4]:


class Encoder(nn.Module):
    def __init__(self,  in_channel, out_channel_lst, kernel_size, stride):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel_lst = out_channel_lst 
        self.kernel_size = kernel_size 
        self.stride = stride 


        self.conv1d_1 = nn.Conv1d(self.in_channel, self.out_channel_lst[0], self.kernel_size, self.stride)

        self.conv1d_2 = nn.Conv1d(self.out_channel_lst[0], self.out_channel_lst[1], self.kernel_size, self.stride)
        
        self.bn1 = nn.BatchNorm1d(self.out_channel_lst[0])
        self.bn2 = nn.BatchNorm1d(self.out_channel_lst[1])
        
        self.after_1 = int((2000-self.kernel_size)//self.stride+1)
        self.after_2 = int((self.after_1-self.kernel_size)//self.stride+1)
        
        self.fc1 = nn.Linear(self.after_2,self.after_2//2)
        self.fc2 = nn.Linear(self.after_2//2,self.after_2//4)
        

    def forward(self, x):
#         print('input_shape',x.shape)
        
        w_1 = self.bn1(self.conv1d_1(x))
#         print('w_1',w_1.shape)
        
        w_2 = self.bn2(self.conv1d_2(w_1))
#         print('w_2',w_2.shape)
        
        w_3 = w_2.view(w_2.shape[0],-1)
#         print('w_3', w_3.shape)
        
        l_1 = self.fc1(w_3)
#         print('l_1',l_1.shape)

        l_2 = self.fc2(l_1)
#         print('l_2',l_2.shape)


        return l_2


# ### Decoder

# In[6]:


class Decoder(nn.Module):
    def __init__(self,  in_channel, out_channel_lst, kernel_size, stride):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel_lst = out_channel_lst 
        self.kernel_size = kernel_size 
        self.stride = stride


        self.conv1d_1 = nn.ConvTranspose1d( self.out_channel_lst[1], self.out_channel_lst[0], self.kernel_size, self.stride)
        self.conv1d_2 = nn.ConvTranspose1d(self.out_channel_lst[0], self.in_channel, self.kernel_size, self.stride)
        
        self.bn1 = nn.BatchNorm1d(self.out_channel_lst[0])
        self.bn2 = nn.BatchNorm1d(self.in_channel)

        self.after_1 = int((2000-self.kernel_size)//self.stride+1)
        self.after_2 = int((self.after_1-self.kernel_size)//self.stride+1)
        
        self.fc1 = nn.Linear(self.after_2//4,self.after_2//2)
        self.fc2 = nn.Linear(self.after_2//2,self.after_2)

    def forward(self, z):
        
        l_1 = self.fc1(z)
#         print('l_1',l_1.shape)
        
        l_2 = self.fc2(l_1)
#         print('l_2',l_2.shape)
        
        l_ = l_2.view(l_1.shape[0],1,-1)
#         print('l_',l_.shape)
        
        z_1 = self.bn1(self.conv1d_1(l_))
#         print('z_1', z_1.shape)
        
        z_2 = self.bn2(self.conv1d_2(z_1))
#         print('z_2', z_2.shape)
        


        return z_2


# ### CONV_AE

# In[7]:


class UsadModel_CONV_AE(nn.Module):
    def __init__(self,  in_channel, out_channel_lst, kernel_size, stride):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel_lst = out_channel_lst 
        self.kernel_size = kernel_size 
        self.stride = stride
        
        self.encoder = Encoder(self.in_channel, self.out_channel_lst, self.kernel_size, self.stride)
        self.decoder = Decoder(self.in_channel, self.out_channel_lst, self.kernel_size, self.stride)
        
    def forward(self, x):
        out_1 = self.encoder(x)        
        
        out_2 = self.decoder(out_1)
        
        mse_ = (x - out_2)**2
        
        return out_1, out_2, mse_
    
device = 'cuda'
model_conv_ae = UsadModel_CONV_AE(3,[2,1],500,1)
optimizer = torch.optim.Adam(list(model_conv_ae.encoder.parameters())+list(model_conv_ae.decoder.parameters()))
model_conv_ae.to(device)


# ### Train_function

# In[ ]:


def train(model, train_loader, epoch, optimizer, device='cuda'):

    model.train()

    losses_train = []
    for batch in train_loader:
        out_1, out_2, mse_ = model(batch.type(torch.FloatTensor).to(device))
        loss = torch.mean(mse_)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        losses_train.append(loss.item())

    losses_train = np.mean(losses_train)


    return losses_train


# ### testing_function

# In[ ]:


def testing(model, test_loader, device = 'cuda'):
    latent_space_ = np.empty((0, 250))
    mse_arr = np.empty((0,3,2000))
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            out_1, out_2, mse_ = model(batch.type(torch.FloatTensor).to(device))

            out_1 = out_1.cpu()
            out_2 = out_2.cpu()
            mse_ = mse_.cpu()
            
            latent_space_ = np.concatenate([latent_space_,out_1])
            mse_arr = np.concatenate([mse_arr,mse_])

        return latent_space_, mse_arr

