# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 22:21:29 2023

@author: abel_
"""

import glob
import os, sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir) 

import torch
import pickle
import yaml
from pathlib import Path, PurePath
import numpy as np
import pandas as pd
from numpy.linalg import svd 
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.colors as mplcolors
import matplotlib.cm as cmx
import matplotlib as mpl
from matplotlib.lines import Line2D
import pylab as pl

from analysis_functions import find_analytic_fixed_points, find_stabilities



def plot_losses(main_exp_names):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 1}) 
    labels = ['iRNN', 'UBLA', 'BLA']
    colors = ['k', 'red', 'b']
    model_names = ['irnn', 'ubla', 'bla']
    markers = ['-', '--']
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    # ax2 = ax1.twinx()
    # axes = [ax1, ax2]
    # params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')[0]
    # training_kwargs = yaml.safe_load(Path(params_path).read_text())
    
    lines = []
    for men_i, main_exp_name in enumerate(main_exp_names):
        # ax = axes[men_i]
        for model_i, model_name in enumerate(model_names):
            # print(main_exp_name)
            exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")
            exp_list = glob.glob(main_exp_name +'/'+ model_name + "/result*")

            all_losses = np.zeros((len(exp_list), 30))
            # print(exp_list)

            for exp_i, exp in enumerate(exp_list):
                print(exp)
                with open(exp, 'rb') as handle:
                    result = pickle.load(handle)
                
                losses = result[0]
                all_losses[exp_i, :len(losses)] = losses
                
                lines.append(ax.plot(losses, markers[0], alpha=0.2, color=colors[model_i]))
        
            mean = np.mean(all_losses, axis=0)
            ax.plot(range(mean.shape[0]), mean, markers[0], label=labels[model_i], color=colors[model_i])
        
    # ax1.set_yscale('log')
    # ax2.set_yscale('log')
    # ax1.set_ylim(1e-6, 1e3)
    # ax2.set_ylim(1e-6, 1e3)
    # ax2.set_axis_off()
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1e3)

    lines = [Line2D([0], [0], color='k', linewidth=3, linestyle=marker) for marker in markers]
    labels = ['with learning', 'no learning']
    # ax2.legend(lines, labels, loc='upper left', bbox_to_anchor=(1, 0.5))

    # ax1.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
    
    # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/losses_wawlearning_T10000.pdf', bbox_inches="tight")
    fig.savefig(main_exp_name+'/losses.pdf', bbox_inches="tight")

    # plt.show()
    return fig

def plot_losses_grid(main_exp_folder):
    with open(main_exp_folder + '/exp_info.pickle', 'rb') as handle:
        exp_info = pickle.load(handle)
        
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    plt.rcParams['pdf.fonttype'] = 42
    
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 1}) 
    labels = ['iRNN', 'UBLA', 'BLA']
    colors = ['k', 'red', 'b']
    model_names = ['irnn', 'ubla', 'bla']
    markers = ['-', '--']
    
    nsigmas = len(exp_info['weight_sigmas'])
    nlrs = len(exp_info['learning_rates'])
    fig, axes = plt.subplots(nsigmas, nlrs, figsize=(4*nsigmas, 2*nlrs), sharex=True, sharey=False, constrained_layout=True)
    for ws_i, weight_sigma in enumerate(exp_info['weight_sigmas']):
        axes[ws_i][0].set_ylabel(weight_sigma, fontsize=25)
        for lr_i, learning_rate in enumerate(exp_info['learning_rates']):
            axes[0][lr_i].set_title(learning_rate)
            
            ax = axes[ws_i][lr_i]
            # ax.set_axis_off()
            ax.tick_params(rotation=90, labelsize=25)

            exp_name = f"/wsigma{weight_sigma}_lr{learning_rate}/"
            
            for model_i, model_name in enumerate(model_names):
                # print(main_exp_folder + exp_name +'/'+ model_name)
                exp_list = glob.glob(main_exp_folder + exp_name +'/'+ model_name + "/result*")
                all_losses = np.zeros((len(exp_list), exp_info['n_epochs']))
        
                for exp_i, exp in enumerate(exp_list):
                    with open(exp, 'rb') as handle:
                        result = pickle.load(handle)
                        
                    losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs = result
                    all_losses[exp_i, :len(losses)] = losses
                    ax.plot(losses, alpha=0.2, color=colors[model_i])
                    
                    first_nan_index = np.where(np.isnan(losses))[0]
                    ax.plot(first_nan_index, losses[first_nan_index-1]*100, 'x', color=colors[model_i])
                    
                    first_nan_index = np.where(np.isinf(losses))[0]
                    ax.plot(first_nan_index, losses[first_nan_index-1]*100, 'x', color=colors[model_i])
                    
                    # if ws_i==2 and lr_i==1:
                    #     print(model_name, losses)
                    
                    print(losses)
            
                mean = np.mean(all_losses, axis=0)
                ax.plot(range(mean.shape[0]), mean, label=labels[model_i], color=colors[model_i])
        
                ax.set_yscale('log')



    # fig.supxlabel('Year')
    fig.supylabel('$\sigma_W$', fontsize=35)
    fig.suptitle('learning rate', fontsize=35)
    colorlines = [Line2D([0], [0], color=color, linewidth=3, linestyle='-') for color in colors]
    axes[1][-1].legend(colorlines, labels, loc='lower left', bbox_to_anchor=(1, 0.5))
    # markerlines = [Line2D([0], [0], color='k', linewidth=3, linestyle=marker) for marker in markers]
    # axes[4][-1].legend(markerlines, ['with learning', 'no learning'], loc='upper left', bbox_to_anchor=(1, 0.5))
    
    fig.tight_layout()
    plt.savefig(main_exp_folder+'/losses_grid.pdf', bbox_inches="tight")
    
    
def plot_fixedpoints_training(sub_exp_folder, trial=0):
    
    with open(sub_exp_folder+'/results_%s.pickle'%trial, 'rb') as handle:
        result = pickle.load(handle)
        
    losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs = result

    n_steps = len(weights_train["wrec"])
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    start_epoch=5
    norm = mplcolors.Normalize(vmin=0, vmax=n_steps)
    norm = norm(np.linspace(0, n_steps, num=n_steps, endpoint=True))
    cmap1 = cmx.get_cmap("Greens_r")
    cmap2 = cmx.get_cmap("Reds_r")
    for i in range(start_epoch,n_steps):
        W_hh = weights_train["wrec"][i]
        b = weights_train["brec"][i]
        W_in = np.array([[0,0],[0,0]])
        I = np.array([0,0])
        fixed_point_list, stabilist, unstabledimensions = find_analytic_fixed_points(W_hh, b, W_in, I, tol=10**-4)
        # print(fixed_point_list)
        # ax.plot(fixed_point_list)
        for fxdpnt in fixed_point_list:
            ax.plot(fxdpnt[0], fxdpnt[1], '_', color=cmap1(norm[i]), markersize=10, zorder=-i)
            ax.text(fxdpnt[0], fxdpnt[1], str(i), color="black", fontsize=12)
            
        W_hh = weights_train["wrec_pp"][i]
        fixed_point_list, stabilist, unstabledimensions = find_analytic_fixed_points(W_hh, b, W_in, I, tol=10**-4)
        for fxdpnt in fixed_point_list:
            ax.plot(fxdpnt[0], fxdpnt[1], '_', color=cmap2(norm[i]), markersize=10, zorder=-i)
            ax.text(fxdpnt[0], fxdpnt[1], str(i), color="black", fontsize=12)



if __name__ == "__main__":
    print(current_dir)
    
    # main_exp_names = ['noisy/perturbed_weights/T10000','noisy/perturbed_weights/T10000_nol']
    # plot_losses(main_exp_names=main_exp_names)
    
    main_exp_folder = parent_dir + f"/experiments/noisy/weight_decay/grid/T{1000}/input{10}"
    # plot_losses_grid(main_exp_folder)

    training_kwargs = {}
    training_kwargs['T'] = 1000
    training_kwargs['input_length'] = 10
    models = ['irnn', 'ubla', 'bla']
    noise_in_list = ['weights', 'input', 'internal',  'weight_decay']
    learning_rates = [1e-4, 1e-5, 1e-6, 1e-7]

    for n_i, noise_in in enumerate(noise_in_list):
            for learning_rate in learning_rates:

                main_exp_names = []

                main_exp_names.append(parent_dir + f"/experiments/noisy/alpha_star2/{noise_in}/T{training_kwargs['T']}/input{training_kwargs['input_length']}/lr{learning_rate}")

                fig = plot_losses(main_exp_names)
                
                fig.savefig(parent_dir + f"/experiments/noisy/alpha_star2/losses_{noise_in}_T{training_kwargs['T']}_input{training_kwargs['input_length']}_lr{learning_rate}.pdf", bbox_inches="tight")

    
    # sub_exp_folder = parent_dir + f"/experiments/noisy/perturbed_weights/grid/T{100}/input{5}/wsigma1e-06_lr1e-06/bla"
    # plot_fixedpoints_training(sub_exp_folder)