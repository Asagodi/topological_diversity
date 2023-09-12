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


def plot_losses(main_exp_names):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 1}) 
    labels = ['iRNN', 'UBLA', 'BLA']
    colors = ['k', 'red', 'b']
    model_names = ['irnn', 'ubla', 'bla']
    markers = ['-', '--']
    fig, ax1 = plt.subplots(1, 1, figsize=(7.5, 5))
    ax2 = ax1.twinx()
    axes = [ax1, ax2]
    # params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')[0]
    # training_kwargs = yaml.safe_load(Path(params_path).read_text())
    
    lines = []
    for men_i, main_exp_name in enumerate(main_exp_names):
        ax = axes[men_i]
        for model_i, model_name in enumerate(model_names):
            exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")
            all_losses = np.zeros((len(exp_list), 30))
        
            for exp_i, exp in enumerate(exp_list):
                with open(exp, 'rb') as handle:
                    result = pickle.load(handle)
                
                losses = result[0]
                all_losses[exp_i, :len(losses)] = losses
                
                lines.append(ax.plot(losses, markers[men_i], alpha=0.2, color=colors[model_i]))
        
            mean = np.mean(all_losses, axis=0)
            ax.plot(range(mean.shape[0]), mean, markers[men_i], label=labels[model_i], color=colors[model_i])
        
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_ylim(1e-6, 1e3)
    ax2.set_ylim(1e-6, 1e3)
    ax2.set_axis_off()

    lines = [Line2D([0], [0], color='k', linewidth=3, linestyle=marker) for marker in markers]
    labels = ['with learning', 'no learning']
    ax2.legend(lines, labels, loc='upper left', bbox_to_anchor=(1, 0.5))

    ax1.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/losses_wawlearning_T10000.pdf', bbox_inches="tight")


if __name__ == "__main__":
    print(current_dir)
    
    main_exp_names = ['noisy/perturbed_weights/T10000','noisy/perturbed_weights/T10000_nol']

    plot_losses(main_exp_names=main_exp_names)