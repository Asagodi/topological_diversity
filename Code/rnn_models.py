import os, sys
import glob
current_dir = os.path.dirname(os.path.realpath('__file__'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.utils.parametrize as P
import numpy as np


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers=1, transform_function='relu',
                 dropout_prob=0., hidden_initial_activations='random', hidden_initial_variance=0.001,
                 output_activation='identity', device='cpu', constrain_spectrum=False):
        super(RNNModel, self).__init__()
        self.input_size = input_size
        self.output_size= output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.hidden_initial_activations = hidden_initial_activations
        self.hidden_initial_variance = hidden_initial_variance

        
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True,
                         nonlinearity=transform_function, dropout=dropout_prob)   

        #hidden unit offset (activity initialization at t=0)
        self.hidden_offset = nn.Parameter(self.hidden_initial_variance*torch.randn((hidden_dim)), requires_grad=True)
        
        # output
        self.fc = nn.Linear(hidden_dim, output_size)         
        if output_activation=='sigmoid':
            self.out_act = nn.Sigmoid()
        elif output_activation=='identity':
            self.out_act = nn.Identity()
        self.device = device
        
        if constrain_spectrum:
            torch.nn.utils.parametrizations.spectral_norm(self.rnn, name='weight_hh_l0') 
    
    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.size(0)
        if self.hidden_initial_activations == 'zero':
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        elif self.hidden_initial_activations == 'offset':
            hidden = self.hidden_offset.repeat(self.n_layers,batch_size,1).reshape((self.n_layers, batch_size, self.hidden_dim))
        elif self.hidden_initial_activations == 'random':
            hidden = torch.normal(mean=0, std=self.hidden_initial_variance, size=(self.n_layers, batch_size, self.hidden_dim)).to(self.device)
        out, hidden = self.rnn(x, hidden)
        out = self.out_act(self.fc(out))
        
        return out, hidden
    


class RNN_interactivation(nn.Module):
    def __init__(self, input_size, hidden_size, inter_hidden_size, output_size, hidden_initial_variance=0.001, device='cpu'):
        super(RNN_interactivation, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.hidden_initial_variance = hidden_initial_variance
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2inter = nn.Linear(hidden_size, inter_hidden_size)
        self.inter2h = nn.Linear(inter_hidden_size, hidden_size) #or should the same matrix be used the other way?
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.out_act = nn.Sigmoid()
    
    def forward(self, input, hidden):
        hidden = self.i2h(input) + hidden
        hidden_inter = self.h2inter(hidden)
        hidden_inter = self.relu(hidden_inter)
        hidden = self.inter2h(hidden_inter)
        out = self.fc(hidden)

        return out, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.normal(mean=0, std=self.hidden_initial_variance, size=(batch_size, self.hidden_size)))


class LSTM(nn.Module):
    "All the weights and biases are initialized from U(-a,a) with a = sqrt(1/hidden_size)"
    def __init__(self, input_size, output_size, hidden_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        out, hidden = self.lstm(x)
        x = self.linear(out)
        return x
    
    
class GRU(nn.Module):
    "All the weights and biases are initialized from U(-a,a) with a = sqrt(1/hidden_size)"
    def __init__(self, input_size, output_size, hidden_dim):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        out, hidden = self.gru(x)
        x = self.linear(out)
        return x
    
    
#parametrization
class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)