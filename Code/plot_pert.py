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



def plot_losses(main_exp_names, sigma=None, ax=None, sharey=False, y_lim=None):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 1}) 
    labels = ['iRNN', 'UBLA', 'BLA']
    colors = ['k', 'red', 'b']
    model_names = ['irnn', 'ubla', 'bla']
    markers = ['-', '--']
    
    if ax==None:
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
                # print(exp)
                with open(exp, 'rb') as handle:
                    result = pickle.load(handle)
                
                losses = result[0]
                all_losses[exp_i, :len(losses)] = losses
                
                lines.append(ax.plot(losses, markers[0], alpha=0.2, color=colors[model_i]))
                
                first_nan_index = np.where(np.isnan(losses))[0]
                if sharey:
                    if np.any(first_nan_index):
                        ax.plot(first_nan_index[0], [1e8], 'x', color=colors[model_i])
                else:
                    ax.plot(first_nan_index, losses[first_nan_index-1]*1e4, 'x', color=colors[model_i])
        
            mean = np.mean(all_losses, axis=0)
            ax.plot(range(mean.shape[0]), mean, markers[0], label=labels[model_i], color=colors[model_i])
            
            
            # if model_i==1:
                # print(all_losses[:,-1])
    # ax1.set_yscale('log')
    # ax2.set_yscale('log')
    # ax1.set_ylim(1e-6, 1e3)
    # ax2.set_ylim(1e-6, 1e3)
    ax.tick_params(rotation=90, labelsize=15)
    ax.xaxis.grid(False, which='major')
    ax.axhline(1.6, linestyle='--', color='k') 
    ax.axhline(1e-5, linestyle='--', color='k') 
    ax.axhline(1e5, linestyle='--', color='k') 

    ax.set_yscale('log')

    if sharey:
        min_y = y_lim[0]
        max_y = y_lim[1]
    else:
        try:
            if np.ceil(np.log10(np.max(all_losses)))-np.floor(np.log10(np.min(all_losses)))<3:
                min_y = np.power(10,np.floor(np.log10(np.min(all_losses))))
                max_y = np.power(10, np.ceil(np.log10(np.max(all_losses))))
            else:
                min_y = np.max([1e-12,np.power(10,np.floor(np.log10(np.min(all_losses))))])
                max_y = np.power(10, 4+np.ceil(np.log10(np.max(all_losses))))
        except:
            min_y = 1e-12
            max_y = 1e8

    ax.set_ylim([min_y, max_y])
    ax.set_yticks([min_y, max_y], [int(np.log10(min_y)), int(np.log10(max_y))])
    ax.set_yticks([min_y,1.6, max_y], [int(np.log10(min_y)),np.round(np.log10(1.6),1), np.round(np.log10(max_y),1)])
    # ax.set_yticks([min_y, sigma, 1.6, 1e5, max_y], [int(np.log10(min_y)),int(np.log10(sigma)),np.round(np.log10(1.6),1),5,int(np.log10(max_y))])


    lines = [Line2D([0], [0], color='k', linewidth=3, linestyle=marker) for marker in markers]
    labels = ['with learning', 'no learning']
    # ax2.legend(lines, labels, loc='upper left', bbox_to_anchor=(1, 0.5))

    # ax1.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
    
    # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/losses_wawlearning_T10000.pdf', bbox_inches="tight")
    # fig.savefig(main_exp_name+'/losses.pdf', bbox_inches="tight")

    # plt.show()
    # return fig
    return all_losses


def plot_losses_box(main_exp_names, sigma=None, ax=None, y_lim=[1e-10,1e4]):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 1}) 
    labels = ['iRNN', 'UBLA', 'BLA']
    colors = ['k', 'red', 'b']
    model_names = ['irnn', 'ubla', 'bla']
    if ax==None:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    
    for men_i, main_exp_name in enumerate(main_exp_names):

        for model_i, model_name in enumerate(model_names):

            exp_list = glob.glob(main_exp_name +'/'+ model_name + "/result*")

            all_losses = np.zeros((len(exp_list), 30))

            for exp_i, exp in enumerate(exp_list):
                # print(exp)
                with open(exp, 'rb') as handle:
                    result = pickle.load(handle)
                
                losses = result[0]
                all_losses[exp_i, :len(losses)] = losses
            bp = ax.boxplot(all_losses[:,-10:].flatten(), positions=[1+.5*model_i], patch_artist=True)  
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=colors[model_i])
            for patch in bp['boxes']:
                patch.set(facecolor='white')       
        
    ax.tick_params(rotation=90, labelsize=15)
    ax.set_yscale('log')
    min_y = y_lim[0]
    max_y = y_lim[1]
    ax.set_ylim([min_y, max_y])
    ax.set_xlim([.75,2.25])

    ax.set_yticks([min_y, max_y], [int(np.log10(min_y)), int(np.log10(max_y))])
    ax.set_yticks([min_y, 1e-5, 1.6,max_y], [int(np.log10(min_y)),-5,np.round(np.log10(1.6),1),int(np.log10(max_y))])

    ax.set_xticks([1,1.5,2], labels[:3])
    # ax.set_ylabel("log(MSE)", family='Computer Modern');

    ax.axhline(1.6, linestyle='--', color='k') 
    ax.axhline(1e-5, linestyle='--', color='k') 

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
    training_kwargs = {}
    training_kwargs['T'] = 100
    training_kwargs['input_length'] = 10
    gradstep = 1
    models = ['irnn', 'ubla', 'bla']
    noise_in_list = ['weights', 'input', 'internal']
    learning_rates = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 0]
    learning_rates = [0]

    factors = [10, 1, .1, .01]
    n_lrs = len(learning_rates)
    n_fs = len(factors)
    plot_box = True
    for n_i, noise_in in enumerate(noise_in_list):
        fig, axes = plt.subplots(n_lrs, n_fs, figsize=(2*n_fs, 4*n_lrs), sharex=True, sharey=True)
        fig.supylabel('log(MSE)', fontsize=35)
        fig.suptitle(noise_in, fontsize=35)
        fig.text(x=0.5, y=.8, s= 'log($\sigma_W$)', fontsize=12, ha="center", transform=fig.transFigure)

        for f_i,factor  in enumerate(factors):
            ax=axes[f_i]
            axes[f_i].set_title(int(np.log10(factor*1e-5)), fontsize=25)
            main_exp_names = []
            main_exp_names.append(parent_dir + f"/experiments/noisy/T{training_kwargs['T']}/gradstep{gradstep}/alpha_star_factor{factor}/{noise_in}/input{training_kwargs['input_length']}/lr0")
            if plot_box:
                plot_losses_box(main_exp_names, sigma=factor*1e-5, ax=ax, y_lim=[1e-10,1.6])
            else:
                plot_losses(main_exp_names, sigma=factor*1e-5, ax=ax, sharey=True, y_lim=[1e-10,1.6])


        fig.tight_layout()
        if plot_box:
            fig.savefig(parent_dir + f"/experiments/noisy/T{training_kwargs['T']}/gradstep{gradstep}/nolearning_losses_{noise_in}_T{training_kwargs['T']}_box.pdf", bbox_inches="tight")
        else:
            fig.savefig(parent_dir + f"/experiments/noisy/T{training_kwargs['T']}/gradstep{gradstep}/nolearning_losses_{noise_in}_T{training_kwargs['T']}.pdf", bbox_inches="tight")


    # plot_box = True
    # sharey = True
    # for n_i, noise_in in enumerate(noise_in_list):
    #     fig, axes = plt.subplots(n_fs, n_lrs, figsize=(4*n_fs, 2*n_lrs), sharex=True, sharey=sharey)
    #     fig.supylabel('log($\sigma_W$)', fontsize=35)
    #     fig.suptitle(noise_in, fontsize=35)
    #     fig.text(x=0.1, y=.95, s= 'log(learning rate)', fontsize=12, ha="center", transform=fig.transFigure)

    #     for f_i,factor  in enumerate(factors):
            
    #         for lr_i,learning_rate in enumerate(learning_rates):
    #             ax=axes[f_i,lr_i]
    #             if learning_rate == 0:
    #                 lr_title = 'no learning'
    #             else:
    #                 lr_title = int(np.log10(learning_rate))
    #             axes[0][lr_i].set_title(lr_title)
    #             axes[f_i][0].set_ylabel(int(np.log10(factor*1e-5)), fontsize=25)

    #             main_exp_names = []

    #             main_exp_names.append(parent_dir + f"/experiments/noisy/T{training_kwargs['T']}/gradstep{gradstep}/alpha_star_factor{factor}/{noise_in}/input{training_kwargs['input_length']}/lr{learning_rate}")
    #             if not plot_box:
    #                 all_losses = plot_losses(main_exp_names, sigma=factor*1e-5, ax=ax, sharey=sharey)
    #             else:
    #                 plot_losses_box(main_exp_names, sigma=factor*1e-5, ax=ax)


    #     fig.tight_layout()

    #     if not plot_box:
    #         if sharey:
    #             fig.savefig(parent_dir + f"/experiments/noisy/T{training_kwargs['T']}/losses_{noise_in}_T{training_kwargs['T']}_samescale.pdf", bbox_inches="tight")
    #         else:
    #             fig.savefig(parent_dir + f"/experiments/noisy/T{training_kwargs['T']}/losses_{noise_in}_T{training_kwargs['T']}.pdf", bbox_inches="tight")

    #     else:
    #         if sharey:
    #             fig.savefig(parent_dir + f"/experiments/noisy/T{training_kwargs['T']}/losses_{noise_in}_T{training_kwargs['T']}_box_samescale.pdf", bbox_inches="tight")
    #         else:
    #             fig.savefig(parent_dir + f"/experiments/noisy/T{training_kwargs['T']}/losses_{noise_in}_T{training_kwargs['T']}_box.pdf", bbox_inches="tight")

    #     plt.show()
    #     plt.close()