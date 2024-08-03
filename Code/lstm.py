# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:53:54 2024

@author: 
"""

import os, sys

from tasks import angularintegration_task, angularintegration_task_constant, double_angularintegration_task

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt; from matplotlib.ticker import MaxNLocator

from analysis_functions import db

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

tanh = nn.Tanh()

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.output_to_hidden = nn.Linear(output_size, hidden_size)
        self.output_to_hidden = nn.Parameter(torch.Tensor(output_size, hidden_size))
        #self.output_to_cell = nn.Linear(output_size, hidden_size)
        self.output_to_cell = nn.Parameter(torch.Tensor(output_size, hidden_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, target):
        
        batch_size = x.shape[0]
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        h_init_torch = nn.Parameter(torch.Tensor(self.num_layers, batch_size, self.hidden_size))
        h_init_torch.requires_grad = False
        h_init_torch = target[:,0,:].matmul(self.output_to_hidden)
        h_init_torch = tanh(h_init_torch)
            
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_init_torch = nn.Parameter(torch.Tensor(self.num_layers, batch_size, self.hidden_size))
        c_init_torch.requires_grad = False
        c_init_torch = target[:,0,:].matmul(self.output_to_cell)
        c_init_torch = tanh(h_init_torch)
        
        h0 = h_init_torch.unsqueeze(0).expand(self.num_layers, -1, -1)  # (num_layers, batch_size, hidden_size)
        c0 = c_init_torch.unsqueeze(0).expand(self.num_layers, -1, -1)  # (num_layers, batch_size, hidden_size)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out)
        return out, hn, cn
    
    def sequence(self, x, target):
            batch_size = x.shape[0]
            
            # Initialize h_init_torch and c_init_torch
            h_init_torch = target[:, 0, :].matmul(self.output_to_hidden)
            h_init_torch = tanh(h_init_torch)
            
            c_init_torch = target[:, 0, :].matmul(self.output_to_cell)
            c_init_torch = tanh(c_init_torch)
            
            # Ensure h0 and c0 have the correct dimensions
            h0 = h_init_torch.unsqueeze(0).expand(self.num_layers, batch_size, self.hidden_size).contiguous()
            c0 = c_init_torch.unsqueeze(0).expand(self.num_layers, batch_size, self.hidden_size).contiguous()
            
            # Initialize lists to store all hidden and cell states
            all_out = []
            all_hidden_states = []
            all_cell_states = []
        
            # Get the sequence length
            seq_len = x.size(1)  # Assuming x is of shape (batch_size, seq_len, input_size)
        
            # Iterate through the sequence
            for t in range(seq_len):
                # Get the input at time step t
                xt = x[:, t, :].unsqueeze(1)
                
                # Apply LSTM cell
                out, (h0, c0) = self.lstm(xt, (h0, c0))
                
                # Store the hidden and cell states
                all_out.append(out)
                all_hidden_states.append(h0[-1].unsqueeze(1))
                all_cell_states.append(c0[-1].unsqueeze(1))
    
            # Stack the states to form tensors in the correct shape (batch_size, T, N)
            all_out = torch.cat(all_out, dim=1)
            all_hidden_states = torch.cat(all_hidden_states, dim=1)
            all_cell_states = torch.cat(all_cell_states, dim=1)

            return all_out, all_hidden_states, all_cell_states
        
def extract_lstm_parameters(model):
    hidden_size = model.hidden_size
    params = {}

    # Extract weights and biases for input, forget, cell, and output gates
    params['W_i'] = model.weight_ih_l0[:hidden_size, :].detach().numpy()
    params['W_f'] = model.weight_ih_l0[hidden_size:2*hidden_size, :].detach().numpy()
    params['W_c'] = model.weight_ih_l0[2*hidden_size:3*hidden_size, :].detach().numpy()
    params['W_o'] = model.weight_ih_l0[3*hidden_size:, :].detach().numpy()

    params['U_i'] = model.weight_hh_l0[:hidden_size, :].detach().numpy()
    params['U_f'] = model.weight_hh_l0[hidden_size:2*hidden_size, :].detach().numpy()
    params['U_c'] = model.weight_hh_l0[2*hidden_size:3*hidden_size, :].detach().numpy()
    params['U_o'] = model.weight_hh_l0[3*hidden_size:, :].detach().numpy()

    params['b_i'] = model.bias_ih_l0[:hidden_size].detach().numpy() + model.bias_hh_l0[:hidden_size].detach().numpy()
    params['b_f'] = model.bias_ih_l0[hidden_size:2*hidden_size].detach().numpy() + model.bias_hh_l0[hidden_size:2*hidden_size].detach().numpy()
    params['b_c'] = model.bias_ih_l0[2*hidden_size:3*hidden_size].detach().numpy() + model.bias_hh_l0[2*hidden_size:3*hidden_size].detach().numpy()
    params['b_o'] = model.bias_ih_l0[3*hidden_size:].detach().numpy() + model.bias_hh_l0[3*hidden_size:].detach().numpy()

    return params

def test_model(model, task, batch_size):
    with torch.no_grad():
        inputs, targets, mask = task(batch_size)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        outputs, _, _ = model(inputs, targets);
        _, hs, cs = model.sequence(inputs, targets);
    targets = targets.detach().numpy()
    outputs = outputs.detach().numpy(); hs = hs.detach().numpy(); cs = cs.detach().numpy(); hs=hs.squeeze(); cs=cs.squeeze();
    trajectories = np.concatenate((hs,cs),axis=-1); trajectories = np.concatenate((hs,cs),axis=-1); 
    from_t=0; to_t=None; target_power = np.mean(targets[:,from_t:to_t,:]**2)
    mse = np.mean((targets[:,from_t:to_t,:] - outputs[:,from_t:to_t,:])**2)
    mse_normalized = mse/target_power
    return db(mse_normalized), outputs, trajectories

def train_model(model, task, num_epochs=100, batch_size=32, learning_rate=0.001, clip_norm=0.,output_noise_level=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        inputs, targets, mask = task(batch_size)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        output_noise = torch.randn(batch_size, inputs.shape[1], targets.shape[-1])*output_noise_level
        targets += output_noise
        mask = torch.tensor(mask, dtype=torch.float32)

        outputs, _, _ = model(inputs, targets)
        loss = criterion(outputs * mask, targets * mask)
        
        optimizer.zero_grad()
        loss.backward()
        if clip_norm>0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def train_n(n, task, input_size = 1, hidden_size = 64,  output_size = 2,
            num_epochs=5000, batch_size=64, learning_rate=0.001, clip_norm=1,
            exp_path='C:/Users/abel_/Documents/Lab/projects/topological_diversity/experiments/angular_integration_old/N128_T128_noisy/lstm/'):
    makedirs(exp_path)
    for i in range(n):
        model = LSTMModel(input_size, hidden_size, output_size,dropout=0.)
        train_model(model, task, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, clip_norm=clip_norm);
        torch.save(model.state_dict(), exp_path+f'/model_{i}.pth')

# def run():
#     T = 12.8
#     dt = 0.1
#     task = angularintegration_task(T, dt, sparsity='variable', random_angle_init=True)
    # long_task = angularintegration_task_constant(T, dt, speed_range=[0,0], random_angle_init='equally_spaced')
#     input_size = 1
#     hidden_size = 50
#     batch_size = 64
#     output_size = 2; model = LSTMModel(input_size, hidden_size, output_size)
#     train_model(model, task, num_epochs=5000, batch_size=batch_size)
    
    
#     batch_size=256; 
#     inputs, targets, mask = task(batch_size)
#     inputs = torch.tensor(inputs, dtype=torch.float32)
#     targets = torch.tensor(targets, dtype=torch.float32)
#     outputs, hs, cs = model.sequence(inputs, targets);
#     outputs = outputs.detach().numpy(); hs = hs.detach().numpy(); cs = cs.detach().numpy(); hs=hs.squeeze(); cs=cs.squeeze(); trajectories = np.concatenate((hs,cs),axis=-1); trajectories = np.concatenate((hs,cs),axis=-1)
    
    
#     fig = plt.figure(figsize=(3, 3));
#     ax = fig.add_subplot(111)
#     xs = np.linspace(0,2*np.pi,100)
#     ax.plot(np.cos(xs), np.sin(xs))
#     for i in range(batch_size):
#         ax.plot(outputs[i,:,0], outputs[i,:,1]);
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     ax.yaxis.set_major_locator(MaxNLocator(integer=True))


# for i in range(0,10):
#     model = LSTMModel(input_size, hidden_size, output_size)
#     train_model(model, task, num_epochs=5000, batch_size=batch_size, learning_rate=0.001,clip_norm=100);
#     torch.save(model.state_dict(), f'C:\\Users\\abel_\\Documents\\Lab\\Projects\\topological_diversity/experiments/angular_integration_old/N128_T128_noisy/lstm/model_{i}.pth')