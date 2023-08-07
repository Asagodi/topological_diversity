# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:03:47 2023

@author: abel_
"""

import os, sys
currentdir = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.insert(0, currentdir + "\Code") 

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import qr
from warnings import warn
import time

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

from schuessler_model import RNN
from tasks import *
from network_initialization import *
from qpta_initializers import _qpta_tanh_hh


def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: idem -- or torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    # If mask has the same shape as output:
    if output.shape == mask.shape:
        loss = (mask * (target - output).pow(2)).sum() / mask.sum()
    else:
        raise Exception("This is problematic...")
        output_dim = output.shape[-1]
        loss = (mask * (target - output).pow(2)).sum() / (mask.sum() * output_dim)
    # Take half:
    loss = 0.5 * loss
    return loss

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
        loss = loss_mse(output, target, mask)
    res = [input, target, mask, output, loss]
    if return_dynamics:
        res.append(trajectories)
    res = [r.numpy() for r in res]
    return res
    

def train(config):
    """
    Train a network
    :param net: nn.Module
    :param task: function; generates input, target, mask for a single batch
    :param n_epochs: int
    :param batch_size: int
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param cuda: bool
    :param record_step: int; record weights after these steps
    :return: res
    """
    # CUDA management
    device='cpu'
    
    n_epochs = 1000
    batch_size = config['batch_size']
    clip_gradient = config['clip_gradient']
    N_rec = config['N_rec']

    scheduler_step_size = config['scheduler_step_size']
    scheduler_gamma = config['scheduler_gamma'] 
    task_name = 'angular'
    task_T = config['T']
    g = config['g']
    
    if task_name == 'eyeblink':
        N_in, N_out = 1, 1
        task =  eyeblink_task(input_length=task_T, t_delay=50)

    elif task_name == 'angular':
        N_in, N_out = 1, 2
        task =  angularintegration_task(T=task_T, dt=.1, length_scale=1, sparsity='variable')
    

    if config['initialization_type'] == 'qpta':
        brec_init = np.zeros(N_rec)
        wrec_init = _qpta_tanh_hh()((N_rec,N_rec))
        
    elif config['initialization_type'] == 'ortho':
        H = np.random.randn(N_rec, N_rec)
        wrec_init, _ = qr(H)
        brec_init = np.array([0]*N_rec)
        
    elif config['initialization_type'] == 'gain':
        brec_init, wrec_init = None, None
    # wrec_init, brec_init = qpta_rec_weights(N_in, N_blas, N_out)
    # a=100
    # wrec_init, brec_init = bla_rec_weights(N_in, N_blas, N_out, a)

    
    net = RNN(dims=(N_in, N_rec, N_out), noise_std=0, dt=1, g=g,
              nonlinearity='tanh', readout_nonlinearity='id',
              train_wi=True, train_wrec=True, train_wo=True, train_brec=True, train_h0=True,
              wrec_init=wrec_init, brec_init=brec_init, h0_init=None, ML_RNN=True)
    
    net.to(device=device)
    h_init=None
    
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], betas=(config["beta1"], config["beta2"]), eps=config["eps"])
    if config["scheduler_step_size"] != 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        
    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        net.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
    
    losses = np.zeros((n_epochs), dtype=np.float32)
    gradient_norm_sqs = np.zeros((n_epochs), dtype=np.float32)

    time0 = time.time()

    for epoch  in range(n_epochs):
        
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
        output = net(input, h_init=h_init)
        loss = loss_mse(output, target, mask)
        
        # Gradient descent
        loss.backward()
        if clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
        gradient_norm_sq = sum([(p.grad ** 2).sum() for p in net.parameters() if p.requires_grad])
        
        # Update weights
        optimizer.step()
        
        if config["scheduler_step_size"] != 0:
            scheduler.step()
        
        # These 2 lines important to prevent memory leaks
        loss.detach_()
        output.detach_()
        
        # Save
        # epochs[epoch] = epoch 
        # losses[epoch] = loss.item()
        # gradient_norm_sqs[epoch] = gradient_norm_sq
        
            
    #     checkpoint_data = {
    #         "epoch":  epoch,
    #         "net_state_dict": net.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #     }
    #     checkpoint = Checkpoint.from_dict(checkpoint_data)
        _, _, _, _, val_loss = run_net(net, task, batch_size=batch_size)
    #     session.report(
    #     {"loss": val_loss},
    #     checkpoint=checkpoint,
    # )
        # os.makedirs("my_model", exist_ok=True)
        # torch.save(
        #     (net.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        # checkpoint = Checkpoint.from_directory("my_model")
        session.report({"loss": val_loss}, checkpoint=checkpoint)

    
    # # Obtain gradient norm
    # gradient_norms = np.sqrt(gradient_norm_sqs)

    # res = [losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs]
    # return res



def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2, grace_period=10):
    config = {
        "T": 10, 
        "initialization_type": 'ortho',
        "g": .5,
        "N_rec": 200, #tune.choice([2**i for i in range(9)]),
        "clip_gradient": None, #tune.choice([.1, 1, 10, 100, None]),
        "lr": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
        "batch_size": tune.choice([128]),
        "beta1":  tune.choice([.7, 0.9, .99]),
        "beta2":  tune.choice([.99, .999, .9999]),
        "eps": 1e-08,
        # "g": tune.choice([0., 0.25, 0.5, 1., 1.5]),
        "scheduler_step_size": 0, #tune.choice([50, 100, 200]),
        "scheduler_gamma": 1., #tune.choice([.5, .75]),
    }
    
    asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='loss',
    mode='min',
    max_t=max_num_epochs,
    grace_period=grace_period,
    reduction_factor=3,
    brackets=1)
    
    tuner = tune.Tuner(
        train,
        tune_config=tune.TuneConfig(scheduler=asha_scheduler),
        param_space=config)
    
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))   


if __name__ == "__main__":
    main(num_samples=200, max_num_epochs=500, gpus_per_trial=1, grace_period=50)