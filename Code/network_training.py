# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 08:30:17 2023

@author: abel_
"""
import os, sys
import glob
current_dir = os.path.dirname(os.path.realpath('__file__'))
import time 
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.utils.parametrize as P


def get_optimizer(model, optimizer, learning_rate, weight_decay, momentum, adam_betas):
    #initialize optimizer
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=adam_betas)
    else:
        raise Exception("Optimizer not known.")
    return optimizer

    
def get_loss_function(model, loss_function, device):
    #initialize loss function
    if loss_function == "mse":
        loss_fn = nn.MSELoss().to(device)
    elif loss_function == "ce":
        loss_fn = nn.CrossEntropyLoss().to(device)
    elif loss_function == "bce":
        loss_fn = nn.BCELoss().to(device)
    elif loss_function == "bcewll":
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        raise Exception("Loss function not known.")
    
    return loss_fn

def get_scheduler(model, optimizer, scheduler_name, scheduler_step_size, scheduler_gamma, max_epochs):
    #initialize learning rate scheduler
    if scheduler_name == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    elif scheduler_name == "cosineannealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_name == None:
        scheduler = None
    return scheduler    

    

def train(model, task, n_epochs, batch_size=32, optimizer='sgd', loss_function='mse', device='cpu', scheduler_name=None, rec_step=1,
          learning_rate=0.001, weight_decay=0., momentum=0., adam_betas=(0.9, 0.999), verbose=False):
    """
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    task : 
    n_epochs : 
    batch_size :
    optimizer : TYPE, optional
        DESCRIPTION. The default is 'sgd'.
    rec_step : int, optional
        record weights every 'rec_step' steps. Default is 1.
    learning_rate : TYPE, optional
        DESCRIPTION. The default is 0.001.
    weight_decay : TYPE, optional
        DESCRIPTION. The default is 0..
    momentum : TYPE, optional
        DESCRIPTION. The default is 0..
    adam_betas : TYPE, optional
        DESCRIPTION. The default is (0.9, 0.999).

    Returns
    -------
    None.

    """
    
    optimizer = get_optimizer(model, optimizer, learning_rate, weight_decay, momentum, adam_betas)
    
    loss_fn = get_loss_function(model, loss_function, device)
    scheduler = get_scheduler(model, optimizer, scheduler_name)
    current_lr = learning_rate

    n_rec_epochs = n_epochs // rec_step
    losses = np.zeros((n_epochs), dtype=np.float32)
    gradient_norm_sqs = np.zeros((n_epochs), dtype=np.float32)
    epochs = np.zeros((n_epochs))
    rec_epochs = np.zeros((n_rec_epochs))

    time0 = time.time()
    if verbose:
        print("Training...")
       
    for i in range(n_epochs):
        # Save weights (before update)
        if i % rec_step == 0:
            k = i // rec_step
            
        
    if verbose:
        print("\nDone. Training took %.1f sec." % (time.time() - time0))

