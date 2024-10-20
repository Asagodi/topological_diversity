# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:53:54 2024

@author: 
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from tasks import angularintegration_task, angularintegration_task_constant, double_angularintegration_task


# from analysis_functions import db
def db(x):
    return 10 * np.log10(x)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

tanh = nn.Tanh()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.5, init_weight_radius_scaling=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_to_hidden = nn.Parameter(torch.Tensor(output_size, hidden_size))
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weight_radius_scaling = init_weight_radius_scaling
        self._initialize_weights()
    
    def _initialize_weights(self):
        stdv = self.init_weight_radius_scaling / np.sqrt(self.hidden_size)
        for name, param in self.gru.named_parameters():
            if 'weight_hh' in name:  # Only initialize recurrent weights
                nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x, target):
        batch_size = x.shape[0]
        h_init_torch = nn.Parameter(torch.Tensor(self.num_layers, batch_size, self.hidden_size)).to(x.device)
        h_init_torch.requires_grad = False
        h_init_torch = target[:, 0, :].matmul(self.output_to_hidden)
        h_init_torch = tanh(h_init_torch)
        h0 = h_init_torch.unsqueeze(0).expand(self.num_layers, -1, -1)  # (num_layers, batch_size, hidden_size)
        out, hn = self.gru(x, h0)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hn
    
    def sequence(self, x, target):
        batch_size = x.shape[0]
        h_init_torch = target[:, 0, :].matmul(self.output_to_hidden)
        h_init_torch = tanh(h_init_torch)
        h0 = h_init_torch.unsqueeze(0).expand(self.num_layers, batch_size, self.hidden_size).contiguous()
        all_out = []
        all_hidden_states = []
        seq_len = x.size(1)  # Assuming x is of shape (batch_size, seq_len, input_size)
        for t in range(seq_len):
            xt = x[:, t, :].unsqueeze(1)
            out, h0 = self.gru(xt, h0)
            all_out.append(out)
            all_hidden_states.append(h0[-1].unsqueeze(1))
        all_out = torch.cat(all_out, dim=1)
        all_hidden_states = torch.cat(all_hidden_states, dim=1)
        return all_out, all_hidden_states

def extract_gru_parameters(model):
    hidden_size = model.hidden_size
    params = {}
    params['W_r'] = model.gru.weight_ih_l0[:hidden_size, :].detach().numpy()
    params['W_z'] = model.gru.weight_ih_l0[hidden_size:2*hidden_size, :].detach().numpy()
    params['W_h'] = model.gru.weight_ih_l0[2*hidden_size:, :].detach().numpy()
    params['U_r'] = model.gru.weight_hh_l0[:hidden_size, :].detach().numpy()
    params['U_z'] = model.gru.weight_hh_l0[hidden_size:2*hidden_size, :].detach().numpy()
    params['U_h'] = model.gru.weight_hh_l0[2*hidden_size:, :].detach().numpy()
    params['b_r'] = model.gru.bias_ih_l0[:hidden_size].detach().numpy() + model.gru.bias_hh_l0[:hidden_size].detach().numpy()
    params['b_z'] = model.gru.bias_ih_l0[hidden_size:2*hidden_size].detach().numpy() + model.gru.bias_hh_l0[hidden_size:2*hidden_size].detach().numpy()
    params['b_h'] = model.gru.bias_ih_l0[2*hidden_size:].detach().numpy() + model.gru.bias_hh_l0[2*hidden_size:].detach().numpy()
    return params

def test_model(model, task, batch_size):
    with torch.no_grad():
        inputs, targets, mask = task(batch_size)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
        outputs, _ = model(inputs, targets)
        _, hs = model.sequence(inputs, targets)
    targets = targets.detach().numpy()
    outputs = outputs.detach().numpy()
    hs = hs.detach().numpy()
    hs = hs.squeeze()
    trajectories = hs
    from_t = 0
    to_t = None
    target_power = np.mean(targets[:, from_t:to_t, :] ** 2)
    mse = np.mean((targets[:, from_t:to_t, :] - outputs[:, from_t:to_t, :]) ** 2)
    mse_normalized = mse / target_power
    return db(mse_normalized), outputs, trajectories



    return inputs, targets, outputs, trajectories

def train_model(model, task, num_epochs=100, batch_size=32, learning_rate=0.001, 
                clip_norm=0., output_noise_level=0.01, weight_decay=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
                            {'params': model.gru.parameters(), 'weight_decay': weight_decay},
                            {'params': model.fc.parameters(), 'weight_decay': 0.0},
                            {'params': model.output_to_hidden, 'weight_decay': 0.0}
                        ], lr=learning_rate)
    
    model.to(device)
    criterion.to(device)

    # for epoch in range(num_epochs):
    epoch = 0
    losses = [] 
    while epoch < num_epochs:

        inputs, targets, mask = task(batch_size)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
        output_noise = torch.randn(batch_size, inputs.shape[1], targets.shape[-1]).to(device) * output_noise_level
        targets += output_noise
        mask = torch.tensor(mask, dtype=torch.float32).to(device)
        
        # Save model and optimizer state
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

        outputs, _ = model(inputs, targets)
        
        # Check for inf values in outputs
        if torch.isinf(outputs).any():
            print(f'Inf detected in outputs at epoch {epoch}. Scaling down weights.')
            with torch.no_grad():
                for param in model.parameters():
                    param *= model.init_weight_radius_scaling
        
        loss = criterion(outputs * mask, targets * mask)
        
        # Check for NaNs in loss
        if torch.isnan(loss):
            return False
            # if epoch == 0:
            #     print(f'NaN detected in loss at epoch {epoch}. Reinitializing model parameters.')
            #     model._initialize_weights()
            #     optimizer = optim.Adam([
            #                 {'params': model.gru.parameters(), 'weight_decay': weight_decay},
            #                 {'params': model.fc.parameters(), 'weight_decay': 0.0},
            #                 {'params': model.output_to_hidden, 'weight_decay': 0.0}
            #             ], lr=learning_rate)
            #     optimizer.zero_grad()

                
            # else:
                # print(f'NaN detected in loss at epoch {epoch}. Rolling back to previous state.')
                # model.load_state_dict(model_state_dict)
                # optimizer.load_state_dict(optimizer_state_dict)
                # with torch.no_grad():
                #     for param in model.parameters():
                #         param *= model.init_weight_radius_scaling
                # optimizer.zero_grad()
                # epoch+=1
                # continue
        
        optimizer.zero_grad()
        loss.backward()
        if clip_norm > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        
        losses.append(loss.item())  # Save the loss value

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        epoch+=1
        
    losses_array = np.array(losses)
    return losses_array

def train_n(n, task, input_size=1, hidden_size=64, output_size=2,
            num_epochs=5000, batch_size=64, learning_rate=0.001, clip_norm=1, dropout=0.,
            exp_path='C:/Users/abel_/Documents/Lab/projects/topological_diversity/experiments/angular_integration_old/N128_T128_noisy/gru/'):
    makedirs(exp_path)
    i=0
    while i<n:
        model = GRUModel(input_size, hidden_size, output_size, dropout=0.5, init_weight_radius_scaling=0.25)
        losses = train_model(model, task, num_epochs=5000, batch_size=batch_size, output_noise_level=0.01,
                             weight_decay=0.0001, learning_rate=0.01, clip_norm=100);
        if not losses:
            continue
        torch.save(model.state_dict(), exp_path+f'/model_{i}.pth')
        np.save(exp_path+f'/losses_{i}.npy', losses)
        i+=1
        
        
# T = 12.8
# dt = 0.1
# task = angularintegration_task(T, dt, sparsity='variable', random_angle_init=True)
# long_task = angularintegration_task_constant(T, dt, speed_range=[0,0], random_angle_init='equally_spaced')
# input_size = 1
# hidden_size = 64
# batch_size = 64
# output_size = 2;

# for i in range(1,10):
#     model = GRUModel(input_size, hidden_size, output_size, dropout=0.5)
#     losses = train_model(model, task, num_epochs=5000, batch_size=batch_size, learning_rate=0.01, clip_norm=1000,scale_factor=.5);
#     torch.save(model.state_dict(), f'C:\\Users\\abel_\\Documents\\Lab\\Projects\\topological_diversity/experiments/angular_integration_old/N{hidden_size}_T128_noisy/gru/model_{i}.pth')
#     np.save(exp_path+f'/losses_{i}.npy', losses)
        
        
def grid_search(task, input_size=1, hidden_size=64, output_size=2,
                num_epochs=5000, batch_size=64, learning_rate=0.001, dropout=0.,
                exp_path=''):
    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    scale_factors = [0.8, .9, 0.99, 1.]
    weight_decays = np.logspace(-5, -1).tolist() + [0]
    clip_norms = np.logspace(-2, 3)
    dropouts = [0, .1, .25, .5]
    
    # Iterate over all combinations of hyperparameters
    for scale_factor in scale_factors:
        for weight_decay in weight_decays:
            for clip_norm in clip_norms:
                for dropout in dropouts:
                    # Initialize the model
                    model = GRUModel(input_size, hidden_size, output_size, dropout=dropout)
                    
                    # Train the model
                    train_model(model, task, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, clip_norm=clip_norm, weight_decay=weight_decay, scale_factor=scale_factor)
                    
                    # Save the model with a name that reflects the hyperparameters
                    model_name = f'model_sf{scale_factor}_wd{weight_decay}_cn{clip_norm}_do{dropout}.pth'
                    torch.save(model.state_dict(), os.path.join(exp_path, model_name))

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.GRU):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)
        
        
        
        
        
        
#######################ANALYSIS
from lstm import nmse, angluar_error

def test_gru(model, task, batch_size=256):
    with torch.no_grad():
        inputs, targets, mask = task(batch_size)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
        outputs, _ = model(inputs, targets)
        _, hs = model.sequence(inputs, targets)
    targets = targets.detach().numpy()
    outputs = outputs.detach().numpy()
    hs = hs.detach().numpy()
    hs = hs.squeeze()
    trajectories = hs
    
    return inputs, targets, outputs, trajectories