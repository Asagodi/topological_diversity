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

def plot_losses(main_exp_names):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 1}) 
    labels = ['iRNN', 'UBLA', 'BLA']
    colors = ['k', 'red', 'b']
    model_names = ['irnn', 'ubla', 'bla']
    markers = ['-', '--']
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    
    # params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')[0]
    # training_kwargs = yaml.safe_load(Path(params_path).read_text())
    
    for men_i, main_exp_name in enumerate(main_exp_names):
        for model_i, model_name in enumerate(model_names):
            exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")
            all_losses = np.zeros((len(exp_list), 30))
        
            for exp_i, exp in enumerate(exp_list):
                with open(exp, 'rb') as handle:
                    result = pickle.load(handle)
                
                losses = result[0]
                all_losses[exp_i, :len(losses)] = losses
            
                ax.plot(losses, markers[men_i], alpha=0.2, color=colors[model_i])
        
            mean = np.mean(all_losses, axis=0)
            ax.plot(range(mean.shape[0]), mean, markers[men_i], label=labels[model_i], color=colors[model_i])
        
    ax.set_yscale('log')
    # ax.legend()
    plt.savefig(parent_dir+'/experiments/'+main_exp_name + '/losses.pdf')
    plt.show()
        
    return all_losses

if __name__ == "__main__":
    main_exp_names = ['perturbed_weights_bb','perturbed_weights_bbnol']

    plot_losses(main_exp_names=main_exp_names)
