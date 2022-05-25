#!/usr/bin/env python
# coding: utf-8
# %%
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


# %%
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


# %%
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


# %%
class UsadModel(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_layers = 1):
        super().__init__()
        self.input_size = input_size #input size
        self.hidden_size_1 = hidden_size_1 #hidden state == output_vector
        self.hidden_size_2 = hidden_size_2 #hidden state == output_vector
        self.num_layers = num_layers #number of layers == 몇층
        
        self.encoder = Encoder(input_size, hidden_size_1, hidden_size_2, num_layers = 1)
        self.decoder1 = Decoder(input_size, hidden_size_1, hidden_size_2, num_layers = 1)
        self.decoder2 = Decoder(input_size, hidden_size_1, hidden_size_2, num_layers = 1)
        
    def forward(self, x, n):
        out_, _ = self.encoder(x)        
        out_.to(device)
        w1, _= self.decoder1(out_)
        w1.to(device)
        w2, _ = self.decoder2(out_)
        w2.to(device)
        self.encoder(w1)
        w3, _ = self.decoder2(self.encoder(w1)[0])
        w3.to(device)

        loss1 = 1/n*torch.mean((x-w1)**2)+(1-1/n)*torch.mean((x-w3)**2)
        loss2 = 1/n*torch.mean((x-w2)**2)-(1-1/n)*torch.mean((x-w3)**2)
        
        return loss1, loss2, out_, w2, w3


# %%
def train(model, train_loader, epoch, optimizer1, optimizer2, device='cuda'):

    model.train()

    losses_train = []
    for batch in train_loader:
        loss1, loss2, out_,w2,w3 = model(batch.type(torch.FloatTensor).to(device),epoch+1)
        loss1.backward(retain_graph=True)
        loss2.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        losses_train.append([loss1.item(),loss2.item()])

    losses_train = np.array(losses_train)
    train_loss_1 = np.mean(losses_train[:,0])
    train_loss_2 = np.mean(losses_train[:,1])


    return train_loss_1, train_loss_2


# %%
def testing(model, test_loader, alpha=.5, beta=.5, device = 'cuda'):
    results = np.empty([0, 9])
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.type(torch.FloatTensor).to(device)
            out_, _ = model.encoder(batch)
            w1, _ = model.decoder1(out_)
            w2, _ = model.decoder2(model.encoder(w1)[0])
            
            batch = batch.cpu()
            w1 = w1.cpu()
            w2 = w2.cpu()

            re_loss = alpha*torch.mean((batch-w1)**2, axis=1) + beta*torch.mean((batch-w2)**2, axis=1)
            re_loss = np.array(re_loss)
            
            results = np.concatenate([results,re_loss])

        return results

