# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:45:29 2023

@author: abel_
"""
import glob
import os, sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir) 

import torch
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.colors as mplcolors
import matplotlib.cm as cmx
import matplotlib as mpl

from models import mse_loss_masked
from perturbed_training import RNN
from tasks import bernouilli_integration_task


device = 'cpu'

def get_params_perfectintegrator(version, ouput_bias_value=1):
    
    a = 1
    
    if version=='irnn': # V1: plane attractor
        wi_init = np.array([[1,0],[0,1]], dtype=float)
        wrec_init = np.array([[1,0],[0,1]], dtype=float)
        brec_init = np.array([0,0])
        wo_init = np.array([[-1,1]])
        bwo_init = np.array([0])
        h0_init = np.array([0,0])
        
    elif version=='ubla': # V2
        wi_init = a*np.array([[-1,1],[-1,1]], dtype=float).T
        wrec_init = np.array([[0,1],[1,0]], dtype=float)
        brec_init = np.array([0,0])
        wo_init = np.array([[1,1]])/(2*a)
        bwo_init = np.array([-ouput_bias_value])/a
        h0_init = ouput_bias_value*np.array([1,1])
    
    elif version=='bla': # V3
        wi_init = a*np.array([[-1,1],[1,-1]], dtype=float)
        wrec_init = np.array([[0,-1],[-1,0]], dtype=float)
        brec_init = ouput_bias_value*np.array([1,1])
        wo_init = np.array([[1,-1]])/(2*a)
        bwo_init = np.array([.0])
        h0_init = ouput_bias_value/2*np.array([1,1])
        

        
    return wi_init, wrec_init, brec_init, wo_init, bwo_init, h0_init

def loss_landscape(T, input_length, batch_size, ouput_bias_value=1):
    np.random.seed(10)
    
    thetas = np.logspace(-5,-2, 30)
    thetas = np.concatenate([-thetas[::-1],[0], thetas])
    loss_theta = np.zeros((thetas.shape[0], 3))
    
    task =  bernouilli_integration_task(T=T,input_length=input_length)
    
    _input, _target, _mask = task(batch_size)
    _input = torch.from_numpy(_input)
    _target = torch.from_numpy(_target)
    _mask = torch.from_numpy(_mask)
    # Allocate
    input = _input.to(device=device).float() 
    target = _target.to(device=device).float() 
    mask = _mask.to(device=device).float() 
    
    
    model_names = ['irnn', 'ubla', 'bla']
    for model_name_j, model_name in enumerate(model_names):
        wi_init, wrec_init, brec_init, wo_init, bwo_init, h0_init = get_params_perfectintegrator(model_name, ouput_bias_value=ouput_bias_value)
        

        for theta_i,theta in enumerate(thetas):
            wrec_init_p = wrec_init.copy()
            wrec_init_p[0,0] += theta
            
            net = RNN(dims=(2,2,1), noise_std=0, dt=1,
                      nonlinearity='relu', readout_nonlinearity='id',
                      wi_init=wi_init, wrec_init=wrec_init_p, wo_init=wo_init, brec_init=brec_init, bwo_init=bwo_init,
                      h0_init=h0_init, ML_RNN=True)

            output = net(input)
            
            loss = mse_loss_masked(output, target, mask).item()
            loss_theta[theta_i,model_name_j] = loss
            
            
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 1})
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    ax.set_prop_cycle(color=['k', 'crimson', 'b'])
    
    ax.plot(thetas, loss_theta, '.-', label=['iRNN', 'UBLA', 'BLA'] )
    # plt.xscale('symlog')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\Delta\theta$')
    ax.set_ylabel('loss')
    ax.legend(title='Network')
    ax.grid()
    plt.savefig(parent_dir+f"/Stability/figures/loss_landscape_bias{ouput_bias_value}.pdf", bbox_inches="tight")
            
    return loss_theta

if __name__ == "__main__":
    print(current_dir)
    
    loss_landscape(T=1000, input_length=50, batch_size=128*4*4, ouput_bias_value=25)