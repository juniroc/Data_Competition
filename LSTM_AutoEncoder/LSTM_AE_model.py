#!/usr/bin/env python
# coding: utf-8

# ### Import Library

# In[ ]:


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
import torch.utils.data as data_utils
import random


# ### get_seed

# In[ ]:


random_seed = 42
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(random_seed)
random.seed(random_seed)

torch.cuda.manual_seed(random_seed)


# ### Encoder

# In[ ]:


class Encoder(nn.Module):
    def __init__(self,  input_size, hidden_size_1, hidden_size_2, num_layers = 1):
        super().__init__()
        self.input_size = input_size #input size
        self.hidden_size_1 = hidden_size_1 #hidden state == output_vector
        self.hidden_size_2 = hidden_size_2 #hidden state == output_vector
        self.num_layers = num_layers #number of layers == 몇층


        self.lstm_1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size_1,
                      num_layers=self.num_layers, batch_first=True)

        self.lstm_2 = nn.LSTM(input_size=self.hidden_size_1, hidden_size=self.hidden_size_2,
                      num_layers=self.num_layers, batch_first=True)


    def forward(self, w):
        out_1, _ = self.lstm_1(w)

        return self.lstm_2(out_1)


# ### Decoder

# In[ ]:


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_layers = 1):
        super().__init__()
        self.input_size = input_size #input size
        self.hidden_size_1 = hidden_size_1 #hidden state == output_vector
        self.hidden_size_2 = hidden_size_2 #hidden state == output_vector
        self.num_layers = num_layers #number of layers == 몇층

        self.lstm_1 = nn.LSTM(input_size=self.hidden_size_2, hidden_size=self.hidden_size_1,
                      num_layers=self.num_layers, batch_first=True)
        
        self.lstm_2 = nn.LSTM(input_size=self.hidden_size_1, hidden_size=self.input_size,
                      num_layers=self.num_layers, batch_first=True)


    def forward(self, z):
        out_1, _ = self.lstm_1(z)

        return self.lstm_2(out_1)


# ### LSTM_AE

# In[ ]:


class UsadModel_LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_layers = 1):
        super().__init__()
        self.input_size = input_size #input size
        self.hidden_size_1 = hidden_size_1 #hidden state == output_vector
        self.hidden_size_2 = hidden_size_2 #hidden state == output_vector
        self.num_layers = num_layers #number of layers == 몇층
        
        self.encoder = Encoder(input_size, hidden_size_1, hidden_size_2, num_layers = 1)
        self.decoder1 = Decoder(input_size, hidden_size_1, hidden_size_2, num_layers = 1)
        
    def forward(self, x, n):
        out_, _ = self.encoder(x)        
        out_.to(device)
        w1, _= self.decoder1(out_)
        w1.to(device)
        
        mse_ = (x-w1)**2

        return mse_, w1, out_


# ### Train_function

# In[ ]:


def train(model, train_loader, epoch, optimizer, device='cuda'):

    model.train()

    losses_train = []
    for batch in train_loader:
        mse_, w1, out_ = model(batch.type(torch.FloatTensor).to(device),epoch+1)
        loss = torch.mean(mse_)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        losses_train.append(loss.item())

    losses_train = np.mean(losses_train)


    return losses_train


# ### Testing_function

# In[ ]:


def testing(model, test_loader, device = 'cuda'):
    results = np.empty([0, 3])
    with torch.no_grad():
        for batch in test_loader:
            mse_, w1, out_, = model(batch.type(torch.FloatTensor).to(device),epoch+1)
            mse_ = mse_.cpu()
            w1 = w1.cpu()
            out_ = out_.cpu()
            mse_ = np.array(mse_)
            sum_mse = mse_.sum(axis = 1)
            results = np.concatenate([results,sum_mse])
        return results

