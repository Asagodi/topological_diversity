# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 22:21:29 2023

@author: 
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

from math import floor, log10
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.colors as mplcolors
import matplotlib.cm as cmx
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pylab as pl

from analysis_functions import find_analytic_fixed_points, find_stabilities

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
def ReLU(x):
    return np.where(x<0,0,x)

def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


def plot_losses_pair(main_exp_name_lists, plot_legend=False):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1}) 
    labels = ['iRNN', 'UBLA', 'BLA', None, None, None]
    colors = ['k', 'red', 'b']
    model_names = ['irnn', 'ubla', 'bla']
    markers = ['-', '--']
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    ax2 = ax.twinx()

    lines = []
    label_i=-1
    all_losses = np.zeros((2, 3, 10, 30))
    for men_i, main_exp_name_list in enumerate(main_exp_name_lists):
        for model_i, model_name in enumerate(model_names):
            label_i+=1
            main_exp_name = main_exp_name_list[model_i]

            exp_list = glob.glob(main_exp_name + "/result*")
            all_losses_men = np.zeros((len(exp_list), 30))

            for exp_i, exp in enumerate(exp_list):
                with open(exp, 'rb') as handle:
                    result = pickle.load(handle)

                losses = result[0]
                all_losses_men[exp_i, :len(losses)] = losses[:30]

                lines.append(ax.plot(losses[:30], markers[men_i], alpha=0.2, color=colors[model_i]))

            mean = np.mean(all_losses_men, axis=0)
            ax.plot(range(mean.shape[0]), mean, markers[men_i], label=labels[label_i], color=colors[model_i])

            all_losses[men_i,model_i,:,:] = all_losses_men

    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("MSE", fontsize=15)
    ax.set_yscale('log')
    ax2.set_yscale('log')
    min_y = np.power(10,np.floor(np.log10(np.min(all_losses))))
    max_y = np.power(10,np.ceil(np.log10(np.max(all_losses))))
    ax.set_ylim([min_y, max_y])
    ax2.set_ylim([min_y, max_y])
    ax2.set_yticks([])
    if plot_legend:
        labels = ['learning', 'no learning']

        lines = [Line2D([0], [0], color='k', linewidth=3, linestyle=marker) for marker in markers]
        ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5))

    ax2.legend(lines, labels, loc='upper left', bbox_to_anchor=(1, 0.5))

    return fig

    return fig
    


def plot_losses(main_exp_name, sigma=None, ax=None, sharey=False, ylim=None, gradstep=1):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    
    sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 1}) 
    labels = ['iRNN', 'UBLA', 'BLA']
    colors = ['k', 'red', 'b']
    model_names = ['irnn', 'ubla', 'bla']
    markers = ['-', '--']
    
    if ax==None:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    if sharey:
        min_y = ylim[0]
        max_y = ylim[1]
    
    lines = []
    for model_i, model_name in enumerate(model_names):
        exp_list = glob.glob(main_exp_name +'/'+ model_name + "/result*")

        all_losses = np.zeros((len(exp_list), gradstep*30))

        for exp_i, exp in enumerate(exp_list):
            with open(exp, 'rb') as handle:
                result = pickle.load(handle)
            
            losses = result[0]
            
            all_losses[exp_i, :len(losses)] = losses
            
            lines.append(ax.plot(losses, markers[0], alpha=0.2, color=colors[model_i]))
            
            first_nan_index = np.where(np.isnan(losses))[0]
            first_max_index = np.where(losses>max_y)[0]

            if sharey:
                if np.any(first_nan_index):
                    ax.plot(first_nan_index[0], [max_y], 'x', color=colors[model_i])
                if np.any(first_max_index):
                    ax.plot(first_max_index[0], [max_y], 'x', color=colors[model_i])
            else:
                ax.plot(first_nan_index, losses[first_nan_index-1]*1e4, 'x', color=colors[model_i])
    
        mean = np.mean(all_losses, axis=0)
        ax.plot(range(mean.shape[0]), mean, markers[0], label=labels[model_i], color=colors[model_i])

    ax.tick_params(rotation=90, labelsize=15)
    ax.xaxis.grid(False, which='major')
    ax.axhline(1.6, linestyle='--', color='k') 
    ax.axhline(sigma, linestyle='--', color='k') 
    ax.axhline(1e5, linestyle='--', color='k') 

    ax.set_yscale('log')

    if sharey:
        min_y = ylim[0]
        max_y = ylim[1]
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
            max_y = 1e5

    ax.set_xticks([0, 30*gradstep], [0, 30])
    ax.set_ylim([min_y, max_y])
    ax.set_yticks([min_y, max_y], [int(np.log10(min_y)), int(np.log10(max_y))])
    ax.set_yticks([min_y,1.6, max_y], [int(np.log10(min_y)),np.round(np.log10(1.6),1), np.round(np.log10(max_y),1)])
    ax.set_yticks([min_y, sigma, 1.6, 1e5, max_y], [int(np.log10(min_y)),int(np.log10(sigma)),np.round(np.log10(1.6),1),5,int(np.log10(max_y))])



def plot_losses_box(main_exp_name, sigma=None, ax=None, ylim=[1e-10,1e4]):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    
    sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 1}) 
    labels = ['iRNN', 'UBLA', 'BLA']
    colors = ['k', 'red', 'b']
    model_names = ['irnn', 'ubla', 'bla']
    if ax==None:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    
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
    min_y = ylim[0]
    max_y = ylim[1]
    ax.set_ylim([min_y, max_y])
    ax.set_xlim([.75,2.25])

    ax.set_yticks([min_y, max_y], [int(np.log10(min_y)), int(np.log10(max_y))])
    ax.set_yticks([min_y, 1e-5, 1.6,max_y], [int(np.log10(min_y)),-5,np.round(np.log10(1.6),1),int(np.log10(max_y))])

    ax.set_xticks([1,1.5,2], labels[:3])
    # ax.set_ylabel("log(MSE)", family='Computer Modern');

    ax.axhline(1.6, linestyle='--', color='k') 
    ax.axhline(sigma, linestyle='--', color='k') 

# def plot_losses_grid(main_exp_folder):
#     with open(main_exp_folder + '/exp_info.pickle', 'rb') as handle:
#         exp_info = pickle.load(handle)
        
#     rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#     rc('text', usetex=True)
#     plt.rcParams['pdf.fonttype'] = 42
    
#     sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 1}) 
#     labels = ['iRNN', 'UBLA', 'BLA']
#     colors = ['k', 'red', 'b']
#     model_names = ['irnn', 'ubla', 'bla']
#     markers = ['-', '--']
    
    
#     nsigmas = len(exp_info['weight_sigmas'])
#     nlrs = len(exp_info['learning_rates'])
#     fig, axes = plt.subplots(nsigmas, nlrs, figsize=(4*nsigmas, 2*nlrs), sharex=True, sharey=False, constrained_layout=True)
#     for ws_i, weight_sigma in enumerate(exp_info['weight_sigmas']):
#         axes[ws_i][0].set_ylabel(weight_sigma, fontsize=25)
#         for lr_i, learning_rate in enumerate(exp_info['learning_rates']):
#             axes[0][lr_i].set_title(learning_rate)
            
#             ax = axes[ws_i][lr_i]
#             # ax.set_axis_off()
#             ax.tick_params(rotation=90, labelsize=25)

#             exp_name = f"/wsigma{weight_sigma}_lr{learning_rate}/"
            
#             for model_i, model_name in enumerate(model_names):
#                 # print(main_exp_folder + exp_name +'/'+ model_name)
#                 exp_list = glob.glob(main_exp_folder + exp_name +'/'+ model_name + "/result*")
#                 all_losses = np.zeros((len(exp_list), exp_info['n_epochs']))
        
#                 for exp_i, exp in enumerate(exp_list):
#                     with open(exp, 'rb') as handle:
#                         result = pickle.load(handle)
                        
#                     losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs = result
#                     all_losses[exp_i, :len(losses)] = losses
#                     ax.plot(losses, alpha=0.2, color=colors[model_i])
                    
#                     first_nan_index = np.where(np.isnan(losses))[0]
#                     ax.plot(first_nan_index, losses[first_nan_index-1]*100, 'x', color=colors[model_i])
                    
#                     # if ws_i==2 and lr_i==1:
#                     #     print(model_name, losses)
                    
#                     print(losses)
            
#                 mean = np.mean(all_losses, axis=0)
#                 ax.plot(range(mean.shape[0]), mean, label=labels[model_i], color=colors[model_i])
        
#                 ax.set_yscale('log')



#     # fig.supxlabel('Year')
#     fig.supylabel('$\sigma_W$', fontsize=35)
#     fig.suptitle('learning rate', fontsize=35)
#     colorlines = [Line2D([0], [0], color=color, linewidth=3, linestyle='-') for color in colors]
#     axes[1][-1].legend(colorlines, labels, loc='lower left', bbox_to_anchor=(1, 0.5))
#     # markerlines = [Line2D([0], [0], color='k', linewidth=3, linestyle=marker) for marker in markers]
#     # axes[4][-1].legend(markerlines, ['with learning', 'no learning'], loc='upper left', bbox_to_anchor=(1, 0.5))
    
#     fig.tight_layout()
#     plt.savefig(main_exp_folder+'/losses_grid.pdf', bbox_inches="tight")
    
    
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


def track_changes_during_learning(main_exp_name, model_name = 'irnn'):
    exp_list = glob.glob(main_exp_name +'/'+ model_name + "/result*");
   
    
    #first vs last
    dists = np.zeros((len(exp_list)))
    for exp_i, exp in enumerate(exp_list):
        with open(exp, 'rb') as handle:
            result = pickle.load(handle)
            losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs = result
            weights_last
            wi_last, wrec_last, wo_last, brec_last, h0_las = weights_last
            for i in range(len(weights_init)):
                dists[exp_i] += np.sum(weights_last[i] - weights_init[i])
                
    #evolution during training
    w00s = np.zeros((len(rec_epochs)-1, len(exp_list)))
    w00pps = np.zeros((len(rec_epochs)-1, len(exp_list)))
    all_losses = np.zeros((len(rec_epochs), len(exp_list)))
    for exp_i, exp in enumerate(exp_list):
        with open(exp, 'rb') as handle:
            result = pickle.load(handle)
            losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs = result
            
            wi_last, wrec_last, wo_last, brec_last, h0_las = weights_last
            for t in range(1,len(rec_epochs)):
                # for i in range(len(weights_init)):
                w00s[t-1,exp_i] = weights_train["wrec"][t][0,0] - weights_train["wrec_pp"][t][0,0]
                w00pps[t-1,exp_i] = weights_train["wrec_pp"][t][0,0]-weights_train["wrec"][t-1][0,0]
            all_losses[:,exp_i] = losses
    return dists, w00s, w00pps, all_losses


def plot_vector_field(W, b, fig=None, ax=None):
    
    if fig==None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    cmap = 'jet'
    w = 1.9
    v = -.3
    Y, X = np.mgrid[v:w:100j, v:w:100j]
    U = ReLU(W[0,0]*X + W[0,1]*Y + b[0]) - X
    V = ReLU(W[1,0]*X + W[1,1]*Y + b[1]) - Y
    speed = np.log(np.sqrt(U**2 + V**2))
    # U *= 1e5
    # V *= 1e5
    # ax.quiver(X,Y,U,V)
    strm = ax.streamplot(X, Y, U, V, density=.45, broken_streamlines=False, color='b') 
    contourf_ = ax.contourf(X, Y, speed, levels=range(-6,3), cmap=cmap)
    ax.set(xlim=(v, w), ylim=(v, w))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    cb_ax = fig.add_axes([0.925,.124,.02,.754])
    fig.colorbar(contourf_, cax=cb_ax)
    cb_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cb_ax.set_ylabel("log(speed)")
    
    # cbar  = fig.colorbar(contourf_, cax=axs)
    # plt.ylabel("log(speed)", family='serif')
    # plt.yticks(rotation=90, family='serif')


def plot_loss_zoom(noise_in_list, training_kwargs):
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
                plot_losses_box(main_exp_names, sigma=factor*1e-5, ax=ax, ylim=[1e-10,1.6])
            else:
                plot_losses(main_exp_names, sigma=factor*1e-5, ax=ax, sharey=True, ylim=[1e-10,1.6])


        fig.tight_layout()
        if plot_box:
            fig.savefig(parent_dir + f"/experiments/noisy/T{training_kwargs['T']}/gradstep{gradstep}/nolearning_losses_{noise_in}_T{training_kwargs['T']}_box.pdf", bbox_inches="tight")
        else:
            fig.savefig(parent_dir + f"/experiments/noisy/T{training_kwargs['T']}/gradstep{gradstep}/nolearning_losses_{noise_in}_T{training_kwargs['T']}.pdf", bbox_inches="tight")

def plot_vfs(main_exp_name, n_epochs=6, model_i=0, noise_in='weights', exp_i=0):
    
    vfs_figs_path = main_exp_name + "/vf_figs"
    makedirs(vfs_figs_path)
    
    fig, axes = plt.subplots(1, n_epochs-1, figsize=(2*(n_epochs-1)+1, 2), sharex=True, sharey=True)
    
    model_name = models[model_i]
    exp_list = glob.glob(main_exp_name +'/'+ model_name + "/result*");
    # for exp_i, exp in enumerate(exp_list):
    exp = exp_list[exp_i]
    with open(exp, 'rb') as handle:
        result = pickle.load(handle)
        losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs = result
        
        wi_last, wrec_last, wo_last, brec_last, h0_las = weights_last
        
        for t in range(1,n_epochs):
            ax = axes[t-1]
            ax.set_xlabel(t)
            W = weights_train["wrec"][t]
            b = weights_train["brec"][t]
            wo = weights_train["wo"][t]
            ax.arrow(0, 0, wo[0][0], wo[1][0], width=.1, head_width=0.05, head_length=0.1, fc='k', ec='k', zorder=100)

            plot_vector_field(W, b, fig=fig, ax=ax)
            # if np.isnan(W[0,0]):
            #     ax.text(x=1./n_epochs*t, y=.5, s= 'nan', fontsize=12, ha="center", transform=fig.transFigure)
    ax.text(x=.05, y=-.1, s= 'Epoch', fontsize=20, ha="center", transform=fig.transFigure)
    fig.savefig(vfs_figs_path + f"/stream_{model_name}_{exp_i}.pdf", bbox_inches="tight")
    plt.show()

def plot_weight_changes(main_exp_name, T, exp_i=0):
    
    weight_figs_path = main_exp_name + "/weight_figs"
    makedirs(weight_figs_path)
    
    fig, ax = plt.subplots(3, 1, figsize=(5, 10), sharex=True)
    # for exp_i, exp in enumerate(exp_list):
    for model_i, model_name in enumerate(models):
        dists, w00s, w00pps, all_losses = track_changes_during_learning(main_exp_name, model_name=model_name)
        
        ax[0].plot(w00s[:,exp_i], 'x', color=colors[model_i])
        ax[0].set_ylabel("Perturbation")
        ax[1].plot(w00pps[:,exp_i], 'o', color=colors[model_i])
        ax[1].set_ylabel("Grad Step")
        ax[2].plot(all_losses[:,exp_i], '-', color=colors[model_i])
        ax[2].set_ylabel("Loss")
    ax[2].set_xlabel("Epoch")

    fig.savefig(weight_figs_path + f"/weight_changes_{exp_i}.pdf", bbox_inches="tight")
    plt.show()
    
    
def plot_losses_grid(noise_in_list, T, gradstep, sharey=False, ylim=None, plot_box=False):
    
    n_lrs = len(learning_rates)
    n_fs = len(factors)
    for n_i, noise_in in enumerate(noise_in_list):
        fig, axes = plt.subplots(n_fs, n_lrs, figsize=(2*n_fs, 2*n_lrs), sharex=True, sharey=sharey)
        fig.supylabel('$\sigma$', fontsize=35)
        fig.suptitle(noise_in, fontsize=35)

        for f_i,factor  in enumerate(factors):
            ylim[0] = 1e-8*factor**2
            for lr_i,learning_rate in enumerate(learning_rates):
                ax=axes[f_i,lr_i]
                if learning_rate == 0:
                    lr_title = 'no learning'
                else:
                    lr_title = r"$10^{%02d}$" % int(np.log10(learning_rate)) 
                axes[0][lr_i].set_title(lr_title)
                axes[f_i][0].set_ylabel(r"$10^{%02d}$" % int(np.log10(factor*1e-5)), fontsize=25)

                main_exp_name = parent_dir + f"/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor{factor}/{noise_in}/input{input_length}/lr{learning_rate}"
                if not plot_box:
                    plot_losses(main_exp_name, sigma=factor*1e-5, ax=ax, sharey=sharey, ylim=ylim, gradstep=gradstep)
                else:
                    plot_losses_box(main_exp_name, sigma=factor*1e-5, ax=ax)


        fig.tight_layout()

        if not plot_box:
            if sharey:
                fig.savefig(parent_dir + f"/experiments/noisy/T{T}/losses_{noise_in}_T{T}_samescale.pdf", bbox_inches="tight")
            else:
                fig.savefig(parent_dir + f"/experiments/noisy/T{T}/losses_{noise_in}_T{T}.pdf", bbox_inches="tight")

        else:
            if sharey:
                fig.savefig(parent_dir + f"/experiments/noisy/T{T}/losses_{noise_in}_T{T}_box_samescale.pdf", bbox_inches="tight")
            else:
                fig.savefig(parent_dir + f"/experiments/noisy/T{T}/losses_{noise_in}_T{T}_box.pdf", bbox_inches="tight")

        plt.show()
        plt.close()

# if __name__ == "__main__":
#     factor = 1
#     noise_in = 'weights'
#     input_length = 10
#     learning_rate = 1e-5
#     T = 1000
#     gradstep = 1
#     main_exp_name = parent_dir + f"/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor{factor}/{noise_in}/input{input_length}/lr{learning_rate}"

#     models = ['irnn', 'ubla', 'bla']
#     colors = ['k', 'red', 'b']

#     plot_vfs(main_exp_name, n_epochs=6, model_i=0, noise_in='weights', exp_i=0)

#     plot_weight_changes(main_exp_name, exp_i=0)
    


        
if __name__ == "__main__":
    training_kwargs = {}
    T = 100
    input_length = 10
    gradstep = 1
    models = ['irnn', 'ubla', 'bla']
    noise_in_list = ['weights', 'input', 'internal']
    # learning_rates = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 0]
    learning_rates = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 0]
    learning_rates = [1e-5, 1e-6, 1e-7, 0]

    factors = [10, 1, .1, .01]

    # plot_losses_grid(noise_in_list, T, gradstep, sharey='row', ylim=[1e-12, 1e5], plot_box=False)
    
    parent_dir
    gradstep = 1
    T = 1000

    if T==1000  and gradstep==1:    #1000, GS1
        main_exp_name_lists = [[parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr1e-07/irnn',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr1e-09/ubla',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr1e-08/bla'],
          [parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr0/irnn',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr0/ubla',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr0/bla']]
    
    #100, GS1
    if T==100  and gradstep==1:
        main_exp_name_lists = [[parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr1e-06/irnn',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr1e-07/ubla',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr1e-06/bla'],
          [parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr0/irnn',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr0/ubla',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr0/bla']]
    
    if T==100  and gradstep==5: #100, GS5
        main_exp_name_lists = [[parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr1e-05/irnn',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr1e-07/ubla',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr1e-06/bla'],
          [parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr0/irnn',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr0/ubla',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr0/bla']]

    if T==500  and gradstep==1: #500, GS1
        main_exp_name_lists = [[parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr1e-07/irnn',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr1e-08/ubla',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr1e-07/bla'],
          [parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr0/irnn',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr0/ubla',
          parent_dir+f'/experiments/noisy/T{T}/gradstep{gradstep}/alpha_star_factor1/weights/input10/lr0/bla']]
    
    fig = plot_losses_pair(main_exp_name_lists, plot_legend=True)
    
    fig.savefig(parent_dir + f"/experiments/noisy/T{T}/gradstep{gradstep}/optimal_lrs.pdf", bbox_inches="tight")