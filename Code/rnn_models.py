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
    def __init__(self, input_size, output_size, hidden_dim, forget_gate=True):
        super(LSTM, self).__init__()
        if forget_gate:
            self.lstm = CustomLSTM(input_size, hidden_dim)
        else:
            self.lstm = LSTM_noforget(input_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        out, hidden = self.lstm(x)
        x = self.linear(out)
        return x, hidden
    
    
class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, 
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
    
    
class LSTM_noforget(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 3))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 3))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 3))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, 
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.tanh(gates[:, HS:HS*2]),
                torch.sigmoid(gates[:, HS*2:]), # output
            )
            c_t = i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
    
    
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