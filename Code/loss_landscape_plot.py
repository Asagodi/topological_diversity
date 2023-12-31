# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:45:29 2023

@author: 
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
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.colors as mplcolors
import matplotlib.cm as cmx
import matplotlib as mpl

from intersect import intersection

from models import mse_loss_masked
from perturbed_training import RNN
from tasks import bernouilli_integration_task, bernouilli_noisy_integration_task, contbernouilli_noisy_integration_task


device = 'cpu'

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 1})

def get_params_perfectintegrator(version, ouput_bias=1):
    
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
        bwo_init = np.array([-ouput_bias])/a
        h0_init = ouput_bias*np.array([1,1])
    
    elif version=='bla': # V3
        wi_init = a*np.array([[-1,1],[1,-1]], dtype=float)
        wrec_init = np.array([[0,-1],[-1,0]], dtype=float)
        brec_init = ouput_bias*np.array([1,1])
        wo_init = np.array([[1,-1]])/(2*a)
        bwo_init = np.array([.0])
        h0_init = ouput_bias/2*np.array([1,1])
        

        
    return wi_init, wrec_init, brec_init, wo_init, bwo_init, h0_init

def loss_landscape(T, input_length, batch_size=128, ouput_bias=1, noise_in='weights', cont=False):
    """
    

    Parameters
    ----------
    T : TYPE
        DESCRIPTION.
    input_length : TYPE
        DESCRIPTION.
    batch_size :  int, optional
    ouput_bias_value : int, optional
        output bias for the LAs. The default is 1.
    noise_in : str, optional
        noise_in determines the type of noise. The default is 'weights', which corresponds to one source of D-type noise.
        Other sources are 
        - 'internal', which corresponds to another source of D-type noise.
        - 'input', which corresponds to one source of S-type noise.

    Returns
    -------
    loss_theta : TYPE
        DESCRIPTION.

    """
    np.random.seed(10)
    
    thetas = np.logspace(-5,-1, 30)
    thetas = np.concatenate([-thetas[::-1],[0], thetas])
    loss_theta = np.zeros((thetas.shape[0], 3))
    
    if cont:
        task = contbernouilli_noisy_integration_task(T=T,input_length=input_length, sigma=0)
    else:
        task = bernouilli_integration_task(T=T,input_length=input_length)
    
    _input, _target, _mask = task(batch_size)
    _input = torch.from_numpy(_input)
    _target = torch.from_numpy(_target)
    _mask = torch.from_numpy(_mask)
    # Allocate
    input = _input.to(device=device).float() 
    target = _target.to(device=device).float() 
    mask = _mask.to(device=device).float() 
    
    noise_std = 0
    model_names = ['irnn', 'ubla', 'bla']
    for model_name_j, model_name in enumerate(model_names):
        wi_init, wrec_init, brec_init, wo_init, bwo_init, h0_init = get_params_perfectintegrator(model_name, ouput_bias=ouput_bias)
        
        
        for theta_i,theta in enumerate(thetas):
            
            wrec_init_p = wrec_init.copy()
            if noise_in=='weights':
                wrec_init_p[0,0] += theta
            
            if noise_in=='internal':
                noise_std=theta
                
            if noise_in=='input':
                if cont:
                    task = contbernouilli_noisy_integration_task(T=T,input_length=input_length, sigma=theta)
                else:
                    task = bernouilli_noisy_integration_task(T=T,input_length=input_length, sigma=theta)
                _input, _target, _mask = task(batch_size)
                input = torch.from_numpy(_input).to(device=device).float() 
                target = torch.from_numpy(_target).to(device=device).float() 
                mask =  torch.from_numpy(_mask).to(device=device).float() 
            
            net = RNN(dims=(2,2,1), noise_std=noise_std, dt=1,
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
    ax.set_yscale('log')
    ax.set_xlabel(r'$\Delta\theta$')
    ax.set_ylabel('loss')
    ax.legend(title='Network')
    ax.grid()
    plt.savefig(parent_dir+f"/Stability/figures/loss_landscape_i{input_length}_b{ouput_bias}_{noise_in}.pdf", bbox_inches="tight")
            
    return loss_theta


def loss_landscape_fixednoise(Ts, input_length, thetas, batch_size=128, ouput_bias=1, noise_in='weights', weight_samples=200, cont=False):
    """
    

    Parameters
    ----------
    T : TYPE
        DESCRIPTION.
    input_length : TYPE
        DESCRIPTION.
    batch_size :  int, optional
    ouput_bias_value : int, optional
        output bias for the LAs. The default is 1.
    noise_in : str, optional
        noise_in determines the type of noise. The default is 'weights', which corresponds to one source of D-type noise.
        Other sources are 
        - 'internal', which corresponds to another source of D-type noise.
        - 'input', which corresponds to one source of S-type noise.

    Returns
    -------
    losses : TYPE
        DESCRIPTION.

    """
    np.random.seed(10)
    
    losses = np.zeros((Ts.shape[0], 3))
    
    noise_std = 0
    sigma = 0
    model_names = ['irnn', 'ubla', 'bla']
    for model_name_j, model_name in enumerate(model_names):
        wi_init, wrec_init, brec_init, wo_init, bwo_init, h0_init = get_params_perfectintegrator(model_name, ouput_bias=ouput_bias)
        theta = thetas[model_name_j]
        
        for T_i, T in enumerate(Ts):
            for j in range(weight_samples):
                if j==1 and noise_in!='weights':
                    break
                wrec_init_p = wrec_init.copy()
                if noise_in=='weights':
                    # wrec_init_p[0,0] += (-1)**j*theta
                    wrec_init_p += np.random.normal(0,theta, (2,2))
                    
                
                if noise_in=='internal':
                    noise_std=theta
                    
                if noise_in=='input':
                    sigma=theta
                    
                if noise_in=='weight_decay':
                    wrec_init_p -= theta*wrec_init_p
                if cont:
                    task = contbernouilli_noisy_integration_task(T=T,input_length=input_length, sigma=theta)
                else:
                    task = bernouilli_noisy_integration_task(T=T,input_length=input_length, sigma=theta)
                _input, _target, _mask = task(batch_size)
                input = torch.from_numpy(_input).to(device=device).float() 
                target = torch.from_numpy(_target).to(device=device).float() 
                mask =  torch.from_numpy(_mask).to(device=device).float() 
                
                net = RNN(dims=(2,2,1), noise_std=noise_std, dt=1,
                          nonlinearity='relu', readout_nonlinearity='id',
                          wi_init=wi_init, wrec_init=wrec_init_p, wo_init=wo_init, brec_init=brec_init, bwo_init=bwo_init,
                          h0_init=h0_init, ML_RNN=True)
    
                output = net(input)
                loss = mse_loss_masked(output, target, mask).item()
                losses[T_i,model_name_j] += loss
            
    if noise_in=='weights':
        losses /= weight_samples
    return losses

def plot_loss_landscape(losses, ouput_bias, noise_in):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 1})
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    ax.set_prop_cycle(color=['k', 'crimson', 'b'])
    
    labels=['iRNN', 'UBLA', 'BLA']
    markers = ['o-', '^-', 's-']
    markersizes = [5,5,6]
    alphas = [.5, .5, .5]
    # ax.plot(Ts*thetas[0], losses, '.-', label=labels)
    for i in range(3):
        ax.plot(Ts*thetas[0], losses[:,i], markers[i], label=labels[i], markersize=markersizes[i], alpha=alphas[i], zorder=-i)
    
    ax.set_yscale('log')
    ax.set_xlabel(r'$|\Delta\theta|$')
    ax.set_ylabel('loss')
    ax.legend(title='Network')
    ax.grid()
    plt.savefig(parent_dir+f"/Stability/figures/matched_loss_landscape_bias{ouput_bias}_{noise_in}.pdf", bbox_inches="tight")
    plt.show()

def calculate_losses(thetas, Ts, input_length, batch_size=128, ouput_bias=1, noise_in='weights', cont=False):
    
    mean_losses = np.zeros((len(thetas), 3))
    for theta_i, theta in tqdm(enumerate(thetas)):
        losses = loss_landscape_fixednoise(Ts=Ts, input_length=input_length, thetas=[theta]*3,
                                           batch_size=batch_size, ouput_bias=ouput_bias, noise_in=noise_in, cont=cont)
    
        mean_losses[theta_i, :] =  np.mean(losses, axis=0)
    
    return mean_losses
    
def plot_losses(mean_losses, thetas, thresholds, input_length, noise_in):
    "Plots matching curves and calculates corresponding alphas for the thresholds"
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    colors=['k', 'crimson', 'b']
    models = ['irnn', 'ubla', 'bla']
    alpha_stars = {}
    for i in range(3):
        alpha_stars[models[i]] = {}
        ax.plot(thetas, mean_losses[:,i], markers[i], color=colors[i], label=labels[i], markersize=markersizes[i], alpha=alphas[i], zorder=-i)
        for threshold in thresholds:
            x, y = intersection(thetas, mean_losses[:,i], thetas, [threshold]*len(thetas))
            alpha_stars[models[i]][threshold] = x    
            ax.plot(x, y, 'x', color=colors[i])
            ax.axhline(y=threshold, linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$|\sigma|$')
    ax.set_ylabel('loss')
    ax.legend(title='Network')
    ax.grid()
    if Ts.shape[0]==1:
        plt.savefig(parent_dir+f"/experiments/noisy/matching_singleT{Ts[0]}_thresholds_input{input_length}_{noise_in}.pdf", bbox_inches="tight")
    else:
        plt.savefig(parent_dir+f"/experiments/noisy/matching_T{Ts[0]}to{Ts[-1]}_thresholds_input{input_length}_{noise_in}.pdf", bbox_inches="tight")

    plt.show()
    return alpha_stars

if __name__ == "__main__":
    print(current_dir)
    
    labels=['iRNN', 'UBLA', 'BLA']
    markers = ['o-', '^-', 's-']
    markersizes = [5,5,6]
    alphas = [.5, .5, .5]
    
    cont = True
    Ts = np.array([1000])
    input_length = 10
    ouput_bias = 20
    batch_size = 1024
    
    logspaces = [np.logspace(-1, -9, 17), np.logspace(4, -9, 17), np.logspace(4, -9, 17)]
    thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    noise_in_list = ['weights', 'input', 'internal']

    # all_alpha_stars =  {} 
    # for n_i, noise_in in enumerate(noise_in_list):
    #     thetas = logspaces[n_i]
    #     mean_losses = calculate_losses(thetas, Ts, input_length, batch_size=batch_size, ouput_bias=ouput_bias, noise_in=noise_in, cont=cont)

    #     alpha_stars = plot_losses(mean_losses, thetas, thresholds, input_length, noise_in)
    #     all_alpha_stars[noise_in] = alpha_stars
    
    # if cont:   
    #     with open(parent_dir+f'/experiments/cnoisy_thrs/matching_single_T{Ts[0]}_input{input_length}.pickle', 'wb') as handle:
    #         pickle.dump(all_alpha_stars, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # else:
    #     with open(parent_dir+f'/experiments/noisy_thrs/matching_single_T{Ts[0]}_input{input_length}.pickle', 'wb') as handle:
    #         pickle.dump(all_alpha_stars, handle, protocol=pickle.HIGHEST_PROTOCOL)