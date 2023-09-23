# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 20:06:13 2023

@author: 
"""
import os, sys
import glob
current_dir = os.path.dirname(os.path.realpath('__file__'))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
import pickle
import time

import yaml
import shutil
from pathlib import Path
from scipy.linalg import qr, block_diag
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from models import mse_loss_masked
from tasks import bernouilli_noisy_integration_task, bernouilli_integration_task, contbernouilli_noisy_integration_task

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        

class RNN(nn.Module):
    def __init__(self, dims, noise_std=0., dt=0.5, 
                 nonlinearity='tanh', readout_nonlinearity='id',
                 wi_init=None, wrec_init=None, wo_init=None, brec_init=None, h0_init=None, bwo_init=None,
                 train_wi=True, train_wrec=True, train_wo=True, train_brec=True, train_h0=True, 
                 ML_RNN=True):
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
        
        # Nonlinearity
        if nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'id':
            self.nonlinearity = lambda x: x

        elif nonlinearity.lower() == 'relu':
            self.nonlinearity = nn.ReLU()
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
        self.bwo = nn.Parameter(torch.Tensor(output_size))
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_(std=1 /np.sqrt(hidden_size))
            else:
                if type(wi_init) == np.ndarray:
                    wi_init = torch.from_numpy(wi_init)
                self.wi.copy_(wi_init)
            if wrec_init is None:
                self.wrec.normal_(std=1/ np.sqrt(hidden_size))
            else:
                if type(wrec_init) == np.ndarray:
                    wrec_init = torch.from_numpy(wrec_init)
                self.wrec.copy_(wrec_init)
            if wo_init is None:
                self.wo.normal_(std=1 / np.sqrt(hidden_size))
            else:
                if type(wo_init) == np.ndarray:
                    wo_init = torch.from_numpy(wo_init.T)
                self.wo.copy_(wo_init)
            if brec_init is None:
                self.brec.zero_()
            else:
                if type(brec_init) == np.ndarray:
                    brec_init = torch.from_numpy(brec_init)
                self.brec.copy_(brec_init)
            if h0_init is None:
                self.h0.zero_()
            else:
                if type(h0_init) == np.ndarray:
                    h0_init = torch.from_numpy(h0_init)
                self.h0.copy_(h0_init)
            if bwo_init is None:
                self.bwo.zero_()
            else:
                if type(bwo_init) == np.ndarray:
                    bwo_init = torch.from_numpy(bwo_init)
                self.bwo.copy_(bwo_init)
            
    def forward(self, input, return_dynamics=False, h_init=None):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        if h_init is None:
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
                out_i = self.readout_nonlinearity(h.matmul(self.wo))+self.bwo
                
            else:
                rec_input = (
                    self.nonlinearity(h).matmul(self.wrec.t()) 
                    + input[:, i, :].matmul(self.wi) 
                    + self.brec)
                h = ((1 - self.dt) * h 
                     + self.dt * rec_input
                     + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])
                out_i = self.readout_nonlinearity(h).matmul(self.wo)+self.bwo

            output[:, i, :] = out_i

            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories
        
        

def train(net, task=None, data=None, n_epochs=10, batch_size=32, learning_rate=1e-2, clip_gradient=None, cuda=False, record_step=1, h_init=None,
          loss_function='mse_loss_masked', final_loss=True, last_mses=None, act_norm_lambda=0.,
          optimizer='sgd', momentum=0, weight_decay=.0, 
          perturb_weights=False, weight_sigma=1e-6, noise_step=1, fix_seed=None, trial=0,
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
 
    if fix_seed:
        torch.manual_seed(trial)
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
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    loss_function = nn.MSELoss()
                
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
    wis = np.zeros((n_rec_epochs, dim_in, dim_rec), dtype=np.float32)
    wrecs = np.zeros((n_rec_epochs, dim_rec, dim_rec), dtype=np.float32)
    wos = np.zeros((n_rec_epochs, dim_rec, dim_out), dtype=np.float32)
    brecs = np.zeros((n_rec_epochs, dim_rec), dtype=np.float32)
    h0s = np.zeros((n_rec_epochs, dim_rec), dtype=np.float32)
    
    wrecs_pp = np.zeros((n_rec_epochs, dim_rec, dim_rec), dtype=np.float32)

    time0 = time.time()
    if verbose:
        print("Training...")
    for i in range(n_epochs):
        # Save weights (before update)
        if i % record_step == 0:
            k = i // record_step
            rec_epochs[k] = i
            wis[k] = net.wi.cpu().detach().numpy()
            wrecs[k] = net.wrec.cpu().detach().numpy()
            wos[k] = net.wo.cpu().detach().numpy()
            brecs[k] = net.brec.cpu().detach().numpy()
            h0s[k] = net.h0.cpu().detach().numpy()
                
        if perturb_weights and (i==1 or (i+1) % noise_step == 0):
            with torch.no_grad():
                net.wrec += torch.normal(0., weight_sigma, net.wrec.shape)
                
        if i % record_step == 0:
            wrecs_pp[k] = net.wrec.cpu().detach().numpy()
        
        if not data:
            # Generate batch
            _input, _target, _mask = task(batch_size)
            # Convert training data to pytorch tensors
            _input = torch.from_numpy(_input)
            _target = torch.from_numpy(_target)
            _mask = torch.from_numpy(_mask)
            # Allocate
            input = _input.to(device=device).float() 
            target = _target.to(device=device).float() 
            mask = _mask.to(device=device).float() 
        
            optimizer.zero_grad()
            output, trajectories = net(input, h_init=h_init, return_dynamics=True)
            try:
                loss = loss_function(output, target, mask)
            except:
                loss = loss_function(output, target)

            
            act_norm = 0.
            if act_norm_lambda != 0.: 
                for t_id in range(trajectories.shape[1]):  #all states through time
                    act_norm += torch.mean(torch.linalg.vector_norm(trajectories[:,t_id,:], dim=1))
                act_norm /= trajectories.shape[1]
                # print(act_norm)
                loss += act_norm_lambda*act_norm
            
            # Gradient descent
            loss.backward()
            # print([p for p in net.parameters() if p.requires_grad])
            # gradient_norm_sq = sum([(p.grad ** 2).sum() for p in net.parameters() if p.requires_grad])
            
            # Update weights
            optimizer.step()
        
            
            # These 2 lines important to prevent memory leaks
            loss.detach()
            output.detach()
     

        # Save
        epochs[i] = i
        losses[i] = loss.item()
        # gradient_norm_sqs[i] = gradient_norm_sq
        
        if verbose:
            print("epoch %d / %d:  loss=%.6f, run.loss=%.6f \n" % (i+1, n_epochs, np.log(losses[i]), np.mean(losses[:i])))
            
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
    
    # Weights throughout training: 
    weights_train = {}
    weights_train["wi"] = wis
    weights_train["wrec"] = wrecs
    weights_train["wo"] = wos
    weights_train["brec"] = brecs
    weights_train["h0"] = h0s
    weights_train["wrec_pp"] = wrecs_pp
    
    res = [losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs]
    return res

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
            output, trajectories = net(input, return_dynamics, h_init=h_init)
        else:
            output = net(input, h_init=h_init)
        loss = mse_loss_masked(output, target, mask)
    res = [input, target, mask, output, loss]
    if return_dynamics:
        res.append(trajectories)
    res = [r.numpy() for r in res]
    return res

    
def run_noisy_training(experiment_folder, trial=None, training_kwargs={}):
    # timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    

    
    with open(experiment_folder+'/parameters.yml', 'w') as outfile:
        yaml.dump(training_kwargs, outfile, default_flow_style=False)
    
    if training_kwargs['cont']:
        task = contbernouilli_noisy_integration_task(T=training_kwargs['T'],
                                                  input_length=training_kwargs['input_length'],
                                                  sigma=training_kwargs['task_noise_sigma'])
    else:
        task = bernouilli_noisy_integration_task(T=training_kwargs['T'],
                                                  input_length=training_kwargs['input_length'],
                                                  sigma=training_kwargs['task_noise_sigma'])

    
    dims = (2,2,1)
    ouput_bias_value = training_kwargs['ouput_bias_value']
    a = 1
    if training_kwargs['version']=='irnn': # V1: plane attractor
        wi_init = np.array([[1,0],[0,1]], dtype=float)
        wrec_init = np.array([[1,0],[0,1]])
        brec_init = np.array([0,0])
        wo_init = np.array([[-1,1]])
        bwo_init = np.array([0])
        h0_init = np.array([0,0])
        
    elif  training_kwargs['version']=='ubla': # V2
        wi_init = a*np.array([[-1,1],[-1,1]], dtype=float).T
        wrec_init = np.array([[0,1],[1,0]])
        brec_init = np.array([0,0])
        wo_init = np.array([[1,1]])/(2*a)
        bwo_init = np.array([-ouput_bias_value])/a
        h0_init = ouput_bias_value*np.array([1,1])
    
    elif  training_kwargs['version']=='bla': # V3
        wi_init = a*np.array([[-1,1],[1,-1]], dtype=float)
        wrec_init = np.array([[0,-1],[-1,0]])
        brec_init = ouput_bias_value*np.array([1,1])
        wo_init = np.array([[1,-1]])/(2*a)
        bwo_init = np.array([.0])
        h0_init = ouput_bias_value/2*np.array([1,1])
        
    net = RNN(dims=dims, noise_std=training_kwargs['internal_noise_std'], dt=training_kwargs['dt'],
              nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
              wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, bwo_init=bwo_init,
              h0_init=h0_init, ML_RNN=True)
    
    
    result = train(net, task=task, n_epochs=training_kwargs['n_epochs'], batch_size=training_kwargs['batch_size'],
              learning_rate=training_kwargs['learning_rate'], clip_gradient=None, cuda=False, record_step=1, h_init=h0_init,
              loss_function='mse_loss_masked', final_loss=True, last_mses=None, act_norm_lambda=0.,
              optimizer='sgd', momentum=0, weight_decay=training_kwargs['weight_decay'], fix_seed=training_kwargs['fix_seed'], trial=trial,
              perturb_weights=training_kwargs['perturb_weights'], weight_sigma=training_kwargs['weight_sigma'], noise_step=training_kwargs['noise_step'],
              verbose=True)

    result.append(training_kwargs)

    with open(experiment_folder + '/results_%s.pickle'%trial, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return result
    
if __name__ == "__main__":
    # print(current_dir)
    
    training_kwargs = {}
    
    models = ['irnn', 'ubla', 'bla']
    sigmas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    
    training_kwargs['verbose'] = False
    training_kwargs['fix_seed'] = True
    training_kwargs['cont'] = True
    training_kwargs['perturb_weights'] = False
    training_kwargs['task_noise_sigma'] = 0.
    training_kwargs['internal_noise_std'] = 0.
    training_kwargs['weight_sigma'] = 0.
    training_kwargs['weight_decay'] = 0.
    
    training_kwargs['dt'] = 1 
    training_kwargs['nonlinearity'] = 'relu'
    training_kwargs['readout_nonlinearity'] = 'id'
    training_kwargs['T'] = 100 
    training_kwargs['noise_step'] = 1
    training_kwargs['n_epochs'] = 30
    
    training_kwargs['batch_size'] = 1024
    training_kwargs['input_length'] = 10
    training_kwargs['ouput_bias_value'] = 20
    exp_info = training_kwargs
    
    learning_rates = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 0]
    noise_in_list = ['weights', 'input', 'internal']
    factors = [1000, 100, 10, 1, .1, .01]

    exp_info['learning_rates'] = learning_rates
    exp_info['noise_in_list'] = noise_in_list
    with open(parent_dir+f"/experiments/cnoisy/matching_single_T{training_kwargs['T']}_threshold{1e-5}_input{training_kwargs['input_length']}.pickle", 'rb') as handle:
        all_alpha_stars = pickle.load(handle)
        exp_info['all_alpha_stars'] = all_alpha_stars

    for factor in factors:
        
        for n_i, noise_in in enumerate(noise_in_list):
            main_exp_folder = parent_dir + f"/experiments/cnoisy_rs/T{training_kwargs['T']}/gradstep{training_kwargs['noise_step']}/alpha_star_factor{factor}/{noise_in}/input{training_kwargs['input_length']}"
    
            makedirs(main_exp_folder) 
            
            training_kwargs['weight_sigma'] = 0.
            training_kwargs['task_noise_sigma'] = 0.
            training_kwargs['internal_noise_std'] = 0.
            training_kwargs['weight_decay'] = 0.
            training_kwargs['perturb_weights'] = False
    
            with open(main_exp_folder + '/exp_info.pickle', 'wb') as handle:
                pickle.dump(exp_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            for model_i, model in tqdm(enumerate(models)):
                training_kwargs['version'] = model
                for learning_rate in learning_rates:
                    training_kwargs['learning_rate'] = learning_rate
                    experiment_folder = main_exp_folder+f"/lr{learning_rate}/"+model
                    if noise_in == 'weights':
                        training_kwargs['perturb_weights'] = True
                        training_kwargs['weight_sigma'] = all_alpha_stars[noise_in][model_i]*factor
                    elif noise_in == 'input':
                        training_kwargs['task_noise_sigma'] = all_alpha_stars[noise_in][model_i]*factor
                    elif noise_in == 'internal':
                        training_kwargs['internal_noise_std'] = all_alpha_stars[noise_in][model_i]*factor
                    else:
                        training_kwargs['weight_decay'] =  all_alpha_stars[noise_in][model_i]*factor
                    makedirs(experiment_folder) 
        
                    for i in range(10):
                        run_noisy_training(experiment_folder, trial=i, training_kwargs=training_kwargs)