# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:44:47 2023

@author: abel_
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from warnings import warn
import time



        
def mse_loss_masked(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: idem -- or torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    loss = (mask * (target - output).pow(2)).sum() / mask.sum()
    return loss


def get_optimizer(model, optimizer, learning_rate, weight_decay=0, momentum=0, adam_betas=(0.9, 0.999), adam_eps=1e-08):
    #initialize optimizer
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=adam_betas, eps=adam_eps)
    else:
        raise Exception("Optimizer not known.")
    return optimizer

    
def get_loss_function(model, loss_function):
    #initialize loss function
    if loss_function == "mse_loss_masked":
        loss_fn = mse_loss_masked
    elif loss_function == "mse":
        loss_fn = nn.MSELoss()
    elif loss_function == "ce":
        loss_fn = nn.CrossEntropyLoss()
    elif loss_function == "bce":
        loss_fn = nn.BCELoss()
    elif loss_function == "bcewll":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise Exception("Loss function not known.")
    
    return loss_fn

def get_scheduler(model, optimizer, scheduler_name, scheduler_step_size, scheduler_gamma, max_epochs):
    #initialize learning rate scheduler
    if scheduler_name == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    elif scheduler_name == "cosineannealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler == 'reduceonplateau':
        scheduler = torch.optim.lr_scheduler.REDUCELRONPLATEAU(optimizer, mode='min', factor=scheduler_gamma, patience=10,
                                                               threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0)
    elif scheduler_name == None:
        pass
    else:
        raise Exception("Scheduler not known.")
    return scheduler    

class EarlyStopper:
    """
    Stop training when the best validation error is at least N epochs (patience) past
    """
    
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class RNN(nn.Module):
    def __init__(self, dims, noise_std=0., dt=0.5, 
                 nonlinearity='tanh', readout_nonlinearity='id',
                 g=None, g_in=1, wi_init=None, wrec_init=None, wo_init=None, brec_init=None, h0_init=None, oth_init=None,
                 train_wi=True, train_wrec=True, train_wo=True, train_brec=True, train_h0=True, save_inputs=False,
                 ML_RNN=True, map_output_to_hidden=False):
        """
        :param dims: list = [input_size, hidden_size, output_size]
        :param noise_std: float
        :param dt: float, integration time step
        :param nonlinearity: str, nonlinearity. Choose 'tanh' or 'id'
        :param readout_nonlinearity: str, nonlinearity. Choose 'tanh' or 'id'
        :param g: float, std of gaussian distribution for initialization
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param wrec_init: torch tensor of shape (hidden_size, hidden_size)
        :param brec_init: torch tensor of shape (hidden_size)
        :param h0_init: torch tensor of shape (hidden_size)
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_brec: bool
        :param train_h0: bool
        :param ML_RNN: bool; whether forward pass is ML convention f(Wr)
        """
        super(RNN, self).__init__()
        self.dims = dims
        input_size, hidden_size, output_size = dims
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.dt = dt
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_brec = train_brec
        self.train_h0 = train_h0
        self.ML_RNN = ML_RNN
        self.save_inputs = save_inputs
        self.map_output_to_hidden = map_output_to_hidden #oth
        
        # Either set g or choose initial parameters. Otherwise, there's a conflict!
        assert (g is not None) or (wrec_init is not None), "Choose g or initial wrec!"

        self.g = g
        
        # Nonlinearity
        if nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'id':
            self.nonlinearity = lambda x: x
            if g is not None:
                if g > 1:
                    warn("g > 1. For a linear network, we need stable dynamics!")
        elif nonlinearity.lower() == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity.lower() == 'rect_tanh':
            # if self.noise_std!=0.:
            self.nonlinearity = nn.ReLU(torch.tanh)                 ##works with noise
            # self.nonlinearity = lambda x: 1.*(torch.tanh(x)>0)   #works with ml_rnn False

        elif nonlinearity.lower() == 'talu':
            self.nonlinearity = lambda x: torch.tanh(x) if x<0 else x
        elif nonlinearity == 'softplus':
            softplus_scale = 1 # Note that scale 1 is quite far from relu
            self.nonlinearity = lambda x: torch.log(1. + torch.exp(softplus_scale * x)) / softplus_scale
        elif type(nonlinearity) == str:
            raise NotImplementedError("Nonlinearity not yet implemented.")
        else:
            self.nonlinearity = nonlinearity
            
            
        # Readout nonlinearity
        if readout_nonlinearity is None:
            # Same as recurrent nonlinearity
            self.readout_nonlinearity = self.nonlinearity
        elif readout_nonlinearity == 'tanh':
            self.readout_nonlinearity = torch.tanh
        elif readout_nonlinearity == 'logistic':
            # Note that the range is [0, 1]. otherwise, 'logistic' is a scaled and shifted tanh
            self.readout_nonlinearity = lambda x: 1. / (1. + torch.exp(-x))
        elif readout_nonlinearity == 'id':
            self.readout_nonlinearity = lambda x: x
        elif type(readout_nonlinearity) == str:
            raise NotImplementedError("readout_nonlinearity not yet implemented.")
        else:
            self.readout_nonlinearity = readout_nonlinearity

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        if not train_wi:
            self.wi.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        if not train_wo:
            self.wo.requires_grad = False
        self.brec = nn.Parameter(torch.Tensor(hidden_size))
        if not train_brec:
            self.brec.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_(std=g_in /np.sqrt(hidden_size))
            else:
                if type(wi_init) == np.ndarray:
                    wi_init = torch.from_numpy(wi_init)
                self.wi.copy_(wi_init)
            if wrec_init is None:
                self.wrec.normal_(std=g / np.sqrt(hidden_size))
            else:
                if type(wrec_init) == np.ndarray:
                    wrec_init = torch.from_numpy(wrec_init)
                self.wrec.copy_(wrec_init)
            if wo_init is None:
                self.wo.normal_(std=1 / np.sqrt(hidden_size))
            else:
                if type(wo_init) == np.ndarray:
                    wo_init = torch.from_numpy(wo_init)
                self.wo.copy_(wo_init)
            if brec_init is None:
                self.brec.zero_()
            else:
                if type(brec_init) == np.ndarray:
                    brec_init = torch.from_numpy(brec_init)
                self.brec.copy_(brec_init)
            if h0_init is None:
                torch.nn.init.uniform_(self.h0, a=-1, b=1)
                # self.h0.zero_()
            else:
                if type(h0_init) == np.ndarray:
                    h0_init = torch.from_numpy(h0_init)
                self.h0.copy_(h0_init)
                
        if map_output_to_hidden:
            self.h0.requires_grad = False
            self.output_to_hidden = nn.Parameter(torch.Tensor(output_size, hidden_size))

            if oth_init is None:
                with torch.no_grad():
                    self.output_to_hidden.normal_(std=1 / np.sqrt(hidden_size))
            else:
                with torch.no_grad():
                    oth_init = torch.from_numpy(oth_init)
                    self.output_to_hidden.copy_(oth_init)
            # self.oth_nonlinearity = nonlinearity
            self.oth_nonlinearity = lambda x: x

        
            
    def forward(self, input, return_dynamics=False, h_init=None, target=None):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]

        if self.map_output_to_hidden:
            h_init_torch = nn.Parameter(torch.Tensor(batch_size, self.hidden_size))
            h_init_torch.requires_grad = False
            h_init_torch = target[:,0,:].matmul(self.output_to_hidden)

            with torch.no_grad():
                h = h_init_torch#.copy_(h_init_torch)
        elif h_init is None:
            h = self.h0
        else:
            h_init_torch = nn.Parameter(torch.Tensor(batch_size, self.hidden_size))
            h_init_torch.requires_grad = False
            # Initialize parameters
            with torch.no_grad():
                h = h_init_torch.copy_(torch.from_numpy(h_init))
                
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.wrec.device)

        # simulation loop
        for i in range(seq_len):
            if self.ML_RNN:
                rec_input = self.nonlinearity(
                    h.matmul(self.wrec.t()) 
                    + input[:, i, :].matmul(self.wi)
                    + self.brec)
                     # Note that if noise is added inside the nonlinearity, the amplitude should be adapted to the slope...
                     # + np.sqrt(2. / self.dt) * self.noise_std * noise[:, i, :])
                h = ((1 - self.dt) * h 
                     + self.dt * rec_input
                     + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])
                out_i = self.readout_nonlinearity(h.matmul(self.wo))
                
            else:
                rec_input = (
                    self.nonlinearity(h).matmul(self.wrec.t()) 
                    + input[:, i, :].matmul(self.wi) 
                    + self.brec)
                h = ((1 - self.dt) * h 
                     + self.dt * rec_input
                     + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])
                out_i = self.readout_nonlinearity(h).matmul(self.wo)

            output[:, i, :] = out_i

            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    
class LSTM_noforget(nn.Module):
    def __init__(self, dims, readout_nonlinearity='id', w_init=None, u_init=None, wo_init=None, bias_init=None, dropout=0.):
        super(LSTM_noforget, self).__init__()
        self.dims = dims
        input_size, hidden_size, output_size = dims
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.readout_nonlinearity = readout_nonlinearity
        
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 3))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 3))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 3))
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        
        self.drop = nn.Dropout(p=dropout)
        
        # Initialize parameters
        with torch.no_grad():
            k = np.sqrt(1/hidden_size)
            if w_init is None:
                torch.nn.init.uniform_(self.W, a=-k, b=k)
                # torch.nn.init.normal_(self.W, std=1/hidden_size)
            else:
                if type(w_init) == np.ndarray:
                    w_init = torch.from_numpy(w_init)
                self.W.copy_(w_init)
            if u_init is None:
               torch.nn.init.uniform_(self.U, a=-k, b=k)
            else:
                if type(u_init) == np.ndarray:
                    u_init = torch.from_numpy(u_init)
            if bias_init is None:
                # self.bias.normal_(std=1 / hidden_size)
                torch.nn.init.uniform_(self.bias, a=-k, b=k)
            else:
                if type(bias_init) == np.ndarray:
                    bias_init = torch.from_numpy(bias_init)
                self.bias.copy_(bias_init)
            
            if wo_init is None:
                # self.wo.normal_(std=1 / hidden_size)
                torch.nn.init.uniform_(self.wo, a=-k, b=k)

            else:
                if type(wo_init) == np.ndarray:
                    wo_init = torch.from_numpy(wo_init)
                self.wo.copy_(wo_init)

        
        # Readout nonlinearity
        if readout_nonlinearity == 'tanh':
            self.readout_nonlinearity = torch.tanh
        elif readout_nonlinearity == 'logistic':
            # Note that the range is [0, 1]. otherwise, 'logistic' is a scaled and shifted tanh
            self.readout_nonlinearity = lambda x: 1. / (1. + torch.exp(-x))
        elif readout_nonlinearity == 'id':
            self.readout_nonlinearity = lambda x: x
        elif type(readout_nonlinearity) == str:
            raise NotImplementedError("readout_nonlinearity not yet implemented.")
        else:
            self.readout_nonlinearity = readout_nonlinearity
                
         
    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_size, sequence_length, _ = x.size()
        hidden_seq = []
        out_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device), 
                        torch.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(sequence_length):
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
            
            # print(i_t, g_t, o_t, h_t)
            
            # out_t = self.readout_nonlinearity(h_t.matmul(self.wo))
            # out_seq.append(o_t.unsqueeze(0))
            
            
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # hidden_seq = self.drop(hidden_seq)
        out_seq = self.readout_nonlinearity(hidden_seq.matmul(self.wo))

        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        out_seq = out_seq.transpose(0, 1).contiguous()

        # print(hidden_seq)
        return out_seq, hidden_seq, (h_t, c_t, 0)
    
    
class LSTM_noforget2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        bs, sequence_length, _ = x.size()
        h_t = torch.zeros(1, self.hidden_size)
        hidden_seq = []
        for t in range(sequence_length):
            combined = torch.cat((x[:,t,:], h_t), dim=1)
            
            i_t = torch.sigmoid(self.input_gate(combined))
            g_t = torch.tanh(self.cell_gate(combined))
            o_t = torch.sigmoid(self.output_gate(combined))
            c_t = i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
            output = self.output_layer(h_t)
            hidden_seq.append(h_t.unsqueeze(0))
        
        return output, hidden

class LSTM(nn.Module):
    def __init__(self, dims, readout_nonlinearity='id', dropout=0.):
        super(LSTM, self).__init__()
        self.dims = dims
        input_size, hidden_size, output_size = dims
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.readout_nonlinearity = readout_nonlinearity

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        h, _ = self.lstm(x)
        h = self.dropout(h)
        y = self.linear(h)

        return y
    

    
class GRU(nn.Module):
    "All the weights and biases are initialized from U(-a,a) with a = sqrt(1/hidden_size)"
    def __init__(self, dims):
        super(GRU, self).__init__()
        self.dims = dims
        input_size, hidden_size, output_size = dims
        self.gru = nn.GRU(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.gru(x)
        x = self.linear(out)
        return x    



def train(net, task=None, data=None, n_epochs=10, batch_size=32, learning_rate=1e-2, clip_gradient=None, cuda=False, record_step=1, h_init=None,
          loss_function='mse_loss_masked', final_loss=True, last_mses=None, act_reg_lambda=0.,
          optimizer='sgd', momentum=0, weight_decay=.0, adam_betas=(0.9, 0.999), adam_eps=1e-8, #optimizers 
          scheduler=None, scheduler_step_size=100, scheduler_gamma=0.3, 
          stop_patience=10, stop_min_delta=0, 
          verbose=True):
    """
    Train a network
    :param net: nn.Module
    :param task: function; generates input, target, mask for a single batch
    :param n_epochs: int
    :param batch_size: int
    :param learning_rate: float
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param cuda: bool
    :param record_step: int; record weights after these steps
    :return: res
    """
    assert (task is not None) or (data is not None), "Choose a task or a dataset!"
 
    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    
    # Optimizer
    optimizer = get_optimizer(net, optimizer, learning_rate, weight_decay, momentum, adam_betas, adam_eps)
    
    scheduler = get_scheduler(net, optimizer, scheduler, scheduler_step_size, scheduler_gamma, n_epochs)

    loss_function = get_loss_function(net, loss_function)
    
    early_stopper = EarlyStopper(patience=stop_patience, min_delta=stop_min_delta)
    
    if data:
        train_input, train_target, train_mask = data['train_input'], data['train_target'], data['train_mask']
        validation_input, validation_target, validation_mask = data['validation_input'], data['validation_target'], data['validation_mask']
        # Convert training data to pytorch tensors
        validation_input = torch.from_numpy(validation_input)
        validation_target = torch.from_numpy(validation_target)
        validation_mask = torch.from_numpy(validation_mask)
        # Allocate
        validation_input = validation_input.to(device=device).float() 
        validation_target = validation_target.to(device=device).float() 
        validation_mask = validation_mask.to(device=device).float() 
        dataset = TensorDataset(torch.tensor(train_input, dtype=torch.float, device=device),
                                torch.tensor(train_target, dtype=torch.float, device=device),
                                torch.tensor(train_mask, dtype=torch.float, device=device))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
    # Save initial weights
    wi_init = net.wi.cpu().detach().numpy().copy()
    wrec_init = net.wrec.cpu().detach().numpy().copy()
    wo_init = net.wo.cpu().detach().numpy().copy()
    brec_init = net.brec.cpu().detach().numpy().copy()
    h0_init = net.h0.cpu().detach().numpy().copy()
    weights_init = [wi_init, wrec_init, wo_init, brec_init, h0_init]
    
    # Record
    dim_rec = net.hidden_size
    dim_in = net.input_size
    dim_out = net.output_size
    n_rec_epochs = n_epochs // record_step
    
    losses = np.zeros((n_epochs), dtype=np.float32)
    validation_losses = np.zeros((n_epochs), dtype=np.float32)
    gradient_norm_sqs = np.zeros((n_epochs), dtype=np.float32)
    epochs = np.zeros((n_epochs))
    rec_epochs = np.zeros((n_rec_epochs))
    if net.train_wi:
        wis = np.zeros((n_rec_epochs, dim_in, dim_rec), dtype=np.float32)
    if net.train_wrec:
        wrecs = np.zeros((n_rec_epochs, dim_rec, dim_rec), dtype=np.float32)
    if net.train_wo:
        wos = np.zeros((n_rec_epochs, dim_rec, dim_out), dtype=np.float32)
    if net.train_brec:
        brecs = np.zeros((n_rec_epochs, dim_rec), dtype=np.float32)
    if net.train_h0:
        h0s = np.zeros((n_rec_epochs, dim_rec), dtype=np.float32)
    if net.map_output_to_hidden:
        oths = np.zeros((n_rec_epochs, dim_out, dim_rec), dtype=np.float32)
    if net.save_inputs:
        all_inputs = np.zeros((n_rec_epochs, dim_out, dim_rec), dtype=np.float32)

    time0 = time.time()
    if verbose:
        print("Training...")
    for i in range(n_epochs):
        # Save weights (before update)
        if i % record_step == 0:
            k = i // record_step
            rec_epochs[k] = i
            if net.train_wi:
                wis[k] = net.wi.cpu().detach().numpy()
            if net.train_wrec:
                wrecs[k] = net.wrec.cpu().detach().numpy()
            if net.train_wo:
                wos[k] = net.wo.cpu().detach().numpy()
            if net.train_brec:
                brecs[k] = net.brec.cpu().detach().numpy()
            if net.train_h0:
                h0s[k] = net.h0.cpu().detach().numpy()
            if net.map_output_to_hidden:
                oths[k] = net.output_to_hidden.cpu().detach().numpy()
            
                
        
        if not data:
            # Generate batch
            _input, _target, _mask = task(batch_size)
            if net.save_inputs:
                all_inputs[k] = _input
            # Convert training data to pytorch tensors
            _input = torch.from_numpy(_input)
            _target = torch.from_numpy(_target)
            # Allocate
            input = _input.to(device=device).float() 
            target = _target.to(device=device).float() 
        
            optimizer.zero_grad()
            output, trajectories = net(input, h_init=h_init, return_dynamics=True, target=target)
            if np.any(_mask):
                _mask = torch.from_numpy(_mask)
                mask = _mask.to(device=device).float() 
                loss = loss_function(output, target, mask)
            else: 
                loss = loss_function(output.view(-1,2), target.view(-1,2))
                # for t_id in range(x.shape[1]):
                    
            
            act_reg = 0.
            if act_reg_lambda != 0.: 
                # trajectories.detach()
                act_reg += torch.mean(torch.linalg.norm(trajectories[:,:,:], dim=(1,2)))
                # for t_id in range(trajectories.shape[1]):  #all states through time
                #     h = trajectories[:,t_id,:]
                #     h.detach()
                #     act_reg += torch.mean(torch.linalg.vector_norm(h, dim=1))
                act_reg /= trajectories.shape[1]
                # print(act_reg)
                loss += act_reg_lambda*act_reg
            
            # Gradient descent
            loss.backward()
            if clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
            gradient_norm_sq = sum([(p.grad ** 2).sum() for p in net.parameters() if p.requires_grad])
            
            # Update weights
            optimizer.step()
            
            scheduler.step()
            
            # Important to prevent memory leaks:
            loss.detach()
            output.detach()
        else:
            for data_idx, (input, target, mask) in enumerate(data_loader):
            
                optimizer.zero_grad()
                output = net(input, h_init=h_init)
                loss = loss_function(output, target, mask)
                
                # Gradient descent
                loss.backward()
                if clip_gradient is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
                gradient_norm_sq = sum([(p.grad ** 2).sum() for p in net.parameters() if p.requires_grad])
                
                # Update weights
                optimizer.step()
                
                scheduler.step()
                
                # These 2 lines important to prevent memory leaks
                loss.detach()
                output.detach()
        
        if data:
            output = net(input, h_init=h_init)
            loss = loss_function(output, target, mask)
            #test on validation dataset
            output = net(validation_input, h_init=h_init)
            validation_loss = loss_function(output, validation_target, validation_mask)
            validation_losses[i] = validation_loss.item()
            
        if early_stopper.early_stop(loss.item()):             
            break

        # Save
        epochs[i] = i
        losses[i] = loss.item()
        gradient_norm_sqs[i] = gradient_norm_sq
        
        if verbose:
            if data:
                print("epoch %d / %d:  loss=%.6f, run.loss=%.6f, val.loss=%.6f \n" % (i+1, n_epochs, losses[i], np.mean(losses[:i]), validation_loss))
            else:
                print("epoch %d / %d:  loss=%.6f, run.loss=%.6f \n" % (i+1, n_epochs, losses[i], np.mean(losses[:i])))
            
    if verbose:
        print("\nDone. Training took %.1f sec." % (time.time() - time0))
    
    # Obtain gradient norm
    gradient_norms = np.sqrt(gradient_norm_sqs)
    
    # Final weights
    wi_last = net.wi.cpu().detach().numpy().copy()
    wrec_last = net.wrec.cpu().detach().numpy().copy()
    wo_last = net.wo.cpu().detach().numpy().copy()
    brec_last = net.brec.cpu().detach().numpy().copy()
    h0_last = net.h0.cpu().detach().numpy().copy()
    weights_last = [wi_last, wrec_last, wo_last, brec_last, h0_last]
    if net.map_output_to_hidden:
        oth_last = net.output_to_hidden.cpu().detach().numpy()
        weights_last = [wi_last, wrec_last, wo_last, brec_last, h0_last, oth_last]
    
    # Weights throughout training: 
    weights_train = {}
    if net.train_wi:
        weights_train["wi"] = wis
    if net.train_wrec:
        weights_train["wrec"] = wrecs
    if net.train_wo:
        weights_train["wo"] = wos
    if net.train_brec:
        weights_train["brec"] = brecs
    if net.train_h0:
        weights_train["h0"] = h0s
    if net.map_output_to_hidden:
        weights_train["oths"] = oths
    
    res = [losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs]
    res_dict = {"losses":losses, "validation_losses":validation_losses, "gradient_norms":gradient_norms,
                "weights_init":weights_init, "weights_last":weights_last, "weights_train":weights_train,
                "epochs":epochs, "rec_epochs":rec_epochs, "all_inputs":all_inputs}

    return res, 

def run_net(net, task, batch_size=32, return_dynamics=False, h_init=None):
    # Generate batch
    input, target, mask = task(batch_size)
    # Convert training data to pytorch tensors
    input = torch.from_numpy(input).float() 
    target = torch.from_numpy(target).float() 
    mask = torch.from_numpy(mask).float() 

    with torch.no_grad():
        # Run dynamics
        if return_dynamics:
            output, trajectories = net(input, return_dynamics, h_init=h_init, target=target)
        else:
            output = net(input, h_init=h_init, target=target)
        loss = mse_loss_masked(output, target, mask)
    res = [input, target, mask, output, loss]
    if return_dynamics:
        res.append(trajectories)
    res = [r.numpy() for r in res]
    return res


def run_net_dataset(net, input, target, mask, batch_size=32, return_dynamics=False, h_init=None):
    with torch.no_grad():
        # Run dynamics
        if return_dynamics:
            output, trajectories = net(input, return_dynamics, h_init=h_init)
        else:
            output = net(input, h_init=h_init)
        loss = mse_loss_masked(output, target, mask)
    res = [input, target, mask, output, loss]
    if return_dynamics:
        res.append(trajectories)
    res = [r.numpy() for r in res]
    return res

def train_lstm(net, task, n_epochs, batch_size=32, learning_rate=1e-2, clip_gradient=None, cuda=False,
          loss_function='mse', init_states=None, final_loss=True, last_mses=None,
          optimizer='sgd', momentum=0, weight_decay=.0, adam_betas=(0.9, 0.999), adam_eps=1e-8, #optimizers 
          scheduler=None, scheduler_step_size=100, scheduler_gamma=0.3, 
          verbose=True, record_step=1):
    
    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    
    # Optimizer
    optimizer = get_optimizer(net, optimizer, learning_rate, weight_decay, momentum, adam_betas, adam_eps)
    scheduler = get_scheduler(net, optimizer, scheduler, scheduler_step_size, scheduler_gamma, n_epochs)
    loss_function = get_loss_function(net, loss_function)
    
    # Save initial weights
    # w_init = net.W.cpu().detach().numpy().copy()
    # u_init = net.U.cpu().detach().numpy().copy()
    # bias_init = net.bias.cpu().detach().numpy().copy()
    # wo_init = net.wo.cpu().detach().numpy().copy()
    # weights_init = [w_init, u_init, bias_init, wo_init]
    weights_init = []
    for name, param in net.named_parameters():
        weights_init.append(param.cpu().detach().numpy().copy())
    
    
    # Record
    losses = np.zeros((n_epochs), dtype=np.float32)
    gradient_norm_sqs = np.zeros((n_epochs), dtype=np.float32)
    epochs = np.zeros((n_epochs))
    
    dim_rec = net.hidden_size
    dim_in = net.input_size
    dim_out = net.output_size
    n_rec_epochs = n_epochs // record_step
    
    #TODO: write in general form (loop over params)
    rec_epochs = np.zeros((n_rec_epochs))
    # ws = np.zeros((n_rec_epochs, dim_in, dim_rec*3), dtype=np.float32)
    # us = np.zeros((n_rec_epochs, dim_rec, dim_rec*3), dtype=np.float32)
    # biases = np.zeros((n_rec_epochs, dim_rec*3), dtype=np.float32)
    # wos = np.zeros((n_rec_epochs, dim_rec, dim_out), dtype=np.float32)
    
    time0 = time.time()
    if verbose:
        print("Training...")
    for i in range(n_epochs):
        # Save weights (before update)
        # if i % record_step == 0:
        #     k = i // record_step
        #     rec_epochs[k] = i
        #     ws[k] = net.W.cpu().detach().numpy()
        #     us[k] = net.U.cpu().detach().numpy()
        #     biases[k] = net.bias.cpu().detach().numpy()
        #     wos[k] = net.wo.cpu().detach().numpy()
        
        # Generate batch
        _input, _target, _mask = task(batch_size)
        # Convert training data to pytorch tensors
        _input = torch.from_numpy(_input)
        _target = torch.from_numpy(_target)
        _mask = torch.from_numpy(_mask)
        # Allocate
        input = _input.to(device=device).float() 
        target = _target.to(device=device).float() 
        # mask = _mask.to(device=device).float() 
        
        optimizer.zero_grad()
        try:        #TODO: write general forward pass
            output, _, _ = net(input, init_states=init_states)
        except:
            output = net(input)
        if final_loss:
            if not last_mses:
                last_mses = output.shape[1]
            fin_int = np.random.randint(1,last_mses,size=batch_size)
            loss = loss_function(output[:,-fin_int,:], target[:,-fin_int,:])
        else:
            loss = loss_function(output, target)
              
        # Gradient descent
        loss.backward()
        if clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
        gradient_norm_sq = sum([(p.grad ** 2).sum() for p in net.parameters() if p.requires_grad])
        
        # Update weights
        optimizer.step()
        
        scheduler.step()
        
        #To prevent memory leaks
        loss.detach()
        output.detach()
        
        # Save
        epochs[i] = i
        losses[i] = loss.item()
        gradient_norm_sqs[i] = gradient_norm_sq
        
        if verbose:
            print("epoch %d / %d:  loss=%.6f \n" % (i+1, n_epochs, np.mean(losses[i])))
            
    if verbose:
        print("\nDone. Training took %.1f sec." % (time.time() - time0))
    
    # Obtain gradient norm
    gradient_norms = np.sqrt(gradient_norm_sqs)
    
    # Weights throughout training: 
    weights_train = {}
    # weights_train["w"] = ws
    # weights_train["u"] = us
    # weights_train["wo"] = wos
    # weights_last =  [net.W, net.U, net.bias, net.wo]
    weights_last = []
    for name, param in net.named_parameters():
        weights_last.append(param.cpu().detach().numpy().copy())
    
    # res = [losses, gradient_norms, weights_last, epochs]
    # return res
    res = [losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs]
    return res
        


def run_lstm(net, task, batch_size=32, return_dynamics=False, init_states=None):
    loss_fn = nn.MSELoss()

    # Generate batch
    input, target, mask = task(batch_size)
    # Convert training data to pytorch tensors
    input = torch.from_numpy(input).float() 
    target = torch.from_numpy(target).float() 
    mask = torch.from_numpy(mask).float() 
    with torch.no_grad():
        # Run dynamics
        output, hidden_seq, _ = net(input, init_states=init_states)

        loss = loss_fn(output, target)
    res = [input, target, mask, output, loss]
    if return_dynamics:
        res.append(hidden_seq)
    res = [r.numpy() for r in res]
    return res
