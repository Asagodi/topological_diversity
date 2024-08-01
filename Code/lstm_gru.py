# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:53:54 2024

@author: 
"""

from tasks import angularintegration_task

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

tanh = nn.Tanh()

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.output_to_hidden = nn.Linear(output_size, hidden_size)
        self.output_to_hidden = nn.Parameter(torch.Tensor(output_size, hidden_size))
        #self.output_to_cell = nn.Linear(output_size, hidden_size)
        self.output_to_cell = nn.Parameter(torch.Tensor(output_size, hidden_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

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

def train_model(model, task, num_epochs=100, batch_size=32, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        inputs, targets, mask = task(batch_size)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        outputs, _, _ = model(inputs, targets)
        loss = criterion(outputs * mask, targets * mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')



def run():
    T = 10
    dt = 0.1
    task = angularintegration_task(T, dt)
    input_size = 1
    hidden_size = 50
    output_size = 2
    model = LSTMModel(input_size, hidden_size, output_size)
    train_model(model, task, num_epochs=5000, batch_size=64)