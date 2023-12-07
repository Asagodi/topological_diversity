# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 18:58:02 2023

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
from scipy.optimize import minimize
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.colors as mplcolors
import matplotlib.cm as cmx
import matplotlib as mpl
from matplotlib import transforms
from pylab import rcParams
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize

from models import RNN, run_net, LSTM_noforget, run_lstm
from tasks import angularintegration_task, angularintegration_delta_task, simplestep_integration_task, poisson_clicks_task
from analysis_functions import calculate_lyapunov_spectrum, tanh_jacobian, participation_ratio


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
        
# def load_net_from_weights(weights, dt=.1):
#     wi_init, wrec_init, wo_init, brec_init, h0_init = weights
#     dims=(wi_init.shape[0],wi_init.shape[1],wo_init.shape[1])
#     net = RNN(dims=dims, dt=dt,
#               wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, h0_init=h0_init)
#     return net

def load_net_from_weights(wi, wrec, wo, brec, h0, oth, training_kwargs):

    if oth is None:
        training_kwargs['map_output_to_hidden'] = False

    dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
    net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
              nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
              wi_init=wi, wrec_init=wrec, wo_init=wo, brec_init=brec, h0_init=h0, oth_init=oth,
              ML_RNN=training_kwargs['ml_rnn'], 
              map_output_to_hidden=training_kwargs['map_output_to_hidden'])
    return net
    
def plot_io_net(net, task, ax=None, ax2=None):
    np.random.seed(12345)
    input, target, mask, output, loss = run_net(net, task, batch_size=1, return_dynamics=False, h_init=None)
    
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax2 = ax.twinx()
    ax2.plot(output[0,:,0], '--', color='deepskyblue', alpha=0.5)
    ax2.plot(output[0,:,1], '--', color='lightcoral', alpha=0.5)
    ax.plot(input[0,...], 'k', label='Angular valocity')
    ax2.plot(target[0,:,0], 'b', label='Target cos')
    ax2.plot(target[0,:,1], 'r', label='Target sin')
    ax2.plot(output[0,:,0], '--', color='deepskyblue', alpha=0.5, label='Output cos')
    ax2.plot(output[0,:,1], '--', color='lightcoral', alpha=0.5, label='Output sin')
    
    ax.set_ylabel("Angular velocity")
    ax.set_xlabel("Time")
    ax2.set_ylabel("Head direction")
    # ax2.legend()
    
    return ax, ax2

def plot_io(ax, ax2, input, output, target):
    ax.plot(input[0,...], 'k', label='Angular valocity')

    ax2.plot(target[0,:,0], 'b', label='Target cos')
    ax2.plot(target[0,:,1], 'r', label='Target sin')
    ax2.plot(output[0,:,0], '--', color='deepskyblue', alpha=0.5, label='Output cos')
    ax2.plot(output[0,:,1], '--', color='lightcoral', alpha=0.5, label='Output sin')
    
    ax.set_ylabel("Angular velocity")
    ax.set_xlabel("Time")
    ax2.set_ylabel("Head direction")
    ax2.legend()

    return ax, ax2


def plot_losses_and_trajectories(plot_losses_or_trajectories='losses', main_exp_name='angular_integration', model_name='low_gain', sparsity=.1, task=None):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    
    
    params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")
    all_losses = np.zeros((len(exp_list), training_kwargs['n_epochs']))
    all_vallosses = np.zeros((len(exp_list), training_kwargs['n_epochs']))

    if not task:
        task = angularintegration_task(T=training_kwargs['T'], dt=training_kwargs['dt_task'], sparsity=sparsity)

    if plot_losses_or_trajectories:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax2 = ax.twinx()
    for exp_i, exp in enumerate(exp_list):
        # print(exp)
        
        with open(exp, 'rb') as handle:
            result = pickle.load(handle)
        
        # losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
        losses = result[0]
        all_losses[exp_i, :len(losses)] = losses
        
        if plot_losses_or_trajectories == 'trajectories':
            try:
                losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
            except:
                losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs_ = result

            wi_init, wrec_init, wo_init, brec_init, h0_init = weights_last
            dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
            net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
                      nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
                      wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, h0_init=h0_init, ML_RNN=training_kwargs['ml_rnn'])

            np.random.seed(12345)
            input, target, mask, output, loss = run_net(net, task, batch_size=1, return_dynamics=False, h_init=None)

            ax2.plot(output[0,:,0], '--', color='deepskyblue', alpha=0.5)
            ax2.plot(output[0,:,1], '--', color='lightcoral', alpha=0.5)
    
        elif plot_losses_or_trajectories == 'losses':
            ax.plot(losses, color='b', alpha=0.2, label='')
            # training_kwargs_ = result[-1]
            # print(training_kwargs_['g_in'], losses[-1])
            # print(result)
            if len(result)>7:
                # training_kwargs_['dataset_filename']
                validation_losses = result[1]
                all_vallosses[exp_i, :len(validation_losses)] = validation_losses
                # print(validation_losses)
                ax.plot(validation_losses, color='r', alpha=0.9)

    
    if plot_losses_or_trajectories == 'trajectories':
        ax, ax2 = plot_io(ax, ax2, input, output, target)
        # ax.set_ylim([0,2])
        # ax.set_ylim([0,2])

        plt.savefig(parent_dir+'/experiments/'+main_exp_name +'/'+ model_name + f'/trajectories{sparsity}.pdf')
        plt.show()
    elif plot_losses_or_trajectories == 'losses':
        mean = np.mean(all_losses, axis=0)
        var = np.var(all_losses, axis=0)
        
        ax.plot(range(mean.shape[0]), mean, label='Training loss')
        ax.plot(range(mean.shape[0]), np.mean(all_vallosses, axis=0),  label='Validation loss')

        # ax.errorbar(range(mean.shape[0]), mean, var, label='Training loss')
        # ax.errorbar(range(mean.shape[0]), np.mean(all_vallosses, axis=0), np.var(all_vallosses, axis=0), label='Validation loss')

        ax.set_yscale('log')
        ax.legend()
        plt.savefig(parent_dir+'/experiments/'+main_exp_name +'/'+ model_name + '/losses.pdf')
        plt.show()
        
    return all_losses

def plot_losses_experiment(plot_losses_or_trajectories='losses', main_exp_name='angularintegration',
                           model_name='low_gain', training_label=None):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    colors = ['k', 'r',  'darkorange', 'g', 'b']

    
    params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")
    all_losses = np.zeros((len(exp_list), training_kwargs['n_epochs']))
    all_vallosses = np.zeros((len(exp_list), training_kwargs['n_epochs']))


    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_lines = []
    parameters = []
    for exp_i, exp in enumerate(exp_list):
        
        
        with open(exp, 'rb') as handle:
            result = pickle.load(handle)
        training_kwargs = result[-1]
        parameters.append(training_kwargs[training_label])
        
        losses = result[0]
        all_losses[exp_i, :len(losses)] = losses
        
        l1, = ax.plot(losses, '', color=colors[exp_i])
        
        try:
            
            # training_kwargs_['dataset_filename']
            validation_losses = result[1]
            all_vallosses[exp_i, :len(validation_losses)] = validation_losses
            # print(validation_losses)
            l2, = ax.plot(validation_losses, '--', color=colors[exp_i])
        except:
            0
        plot_lines.append([l1, l2])


    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.set_yscale('log')
    # ax.legend(title=training_label.replace('_',' '))
    legend1 = plt.legend(plot_lines[0], ["Training", "Validation"], loc=5)
    ax.legend([l[0] for l in plot_lines], parameters)
    ax.add_artist(legend1)
    
    plt.savefig(parent_dir+'/experiments/'+main_exp_name +'/'+ model_name + '/losses.pdf')
    plt.show()
        
    return all_losses

def plot_all(model_names, main_exp_name='angularintegration'):
    print(parent_dir+'/experiments/' + main_exp_name +'/'+ model_names[0] + '/param*.yml')
    params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_names[0] + '/param*.yml')[0]
    # params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())

    mean_colors = ['k', 'r',  'darkorange', 'g', 'b']
    trial_colors = ['gray', 'lightcoral', 'moccasin', 'lime', 'deepskyblue']
    labels=['LSTM', r'$g=.5$', r'$g=1.5$', 'Orthogonal', 'QPTA']
    # plt.style.use('ggplot')
    plt.style.use('seaborn-dark-palette')


    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for model_i, model_name in enumerate(model_names):
        all_losses = plot_losses_and_trajectories(plot_losses_or_trajectories=None, main_exp_name=main_exp_name, model_name=model_name)
        for i in range(all_losses.shape[0]):
            ax.plot(all_losses[i,...], color=trial_colors[model_i], alpha=0.1)
        mean = np.mean(all_losses, axis=0)
        ax.plot(range(training_kwargs['n_epochs']), mean, color=mean_colors[model_i], label=labels[model_i])
        ax.set_yscale('log')
    ax.legend()
    plt.gca().set_ylim(top=1)
    plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/angular_losses.pdf')
    plt.show()
    
    
def plot_bestmodel_traj(model_names, main_exp_name='angularintegration', sparsity=1, i=0):
    params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_names[0] + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    
    mean_colors = ['k', 'r',  'darkorange', 'g', 'b']
    trial_colors = ['gray', 'lightcoral', 'moccasin', 'lime', 'deepskyblue']
    labels=['LSTM', r'$g=.5$', r'$g=1.5$', 'Orthogonal', 'QPTA']
    network_types = ['lstm_noforget', 'rnn', 'rnn', 'rnn', 'rnn']
    plt.style.use('seaborn-dark-palette')
    
    if len(model_names)==1:
        mean_colors=[mean_colors[i]]
        trial_colors=[trial_colors[i]]
        labels=[labels[i]]
        network_types = [network_types[i]]

    task = angularintegration_task(T=training_kwargs['T'], length_scale=training_kwargs['T']/12.8, dt=training_kwargs['dt_task'], sparsity=sparsity)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax2 = ax.twinx()
    for model_i, model_name in enumerate(model_names):
        params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_names[model_i] + '/param*.yml')[0]
        training_kwargs = yaml.safe_load(Path(params_path).read_text())
        training_kwargs['network_type'] = network_types[model_i]
        final_losses = plot_losses_and_trajectories(plot_losses_or_trajectories=None, main_exp_name=main_exp_name, model_name=model_name)[:,-1]
        minidx = np.nanargmin(final_losses)

        bestexp = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")[minidx]
        print(model_name, minidx, final_losses[minidx])
        with open(bestexp, 'rb') as handle:
            result = pickle.load(handle)

        if training_kwargs['network_type'] =='lstm_noforget':
            losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
            w_init, u_init, bias_init, wo_init = weights_last

            dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
            net = LSTM_noforget(dims=dims,  readout_nonlinearity=training_kwargs['readout_nonlinearity'],
                      w_init=w_init, u_init=u_init, wo_init=wo_init, bias_init=bias_init)
            np.random.seed(123)
            input, target, mask, output, loss = run_lstm(net, task, batch_size=32, return_dynamics=False, init_states=None)
            
        else:
            losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
            wi_init, wrec_init, wo_init, brec_init, h0_init = weights_last
            dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
            net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
                      nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
                      wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, h0_init=h0_init, ML_RNN=training_kwargs['ml_rnn'])
            
            np.random.seed(123)
            input, target, mask, output, loss = run_net(net, task, batch_size=32, return_dynamics=False, h_init=None)

        ax2.plot(output[0,:,0], '--', color=mean_colors[model_i], label=labels[model_i])
        ax2.plot(output[0,:,1], '--', color=mean_colors[model_i])

    ax.plot(input[0,...], 'k', label='Angular valocity')
    ax.set_ylabel("Angular velocity")
    ax2.plot(target[0,:,0], 'b')
    ax2.plot(target[0,:,1], 'r')
    plt.xlabel("Time")
    ax2.set_ylabel("Head direction (sin, cos)")
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=6)
    plt.savefig(parent_dir+'/experiments/'+main_exp_name+f'/trajectories{sparsity}.pdf', bbox_inches="tight")
    plt.show()
    
def load_best_model_from_grid(experiment_folder, model_name, task):
    # df = pd.read_csv(parent_dir+'/experiments/'+experiment_folder+'/grid_search_'+model_name+'.csv', sep=',')
    
    df = pd.read_pickle(parent_dir+'/experiments/'+experiment_folder+'/grid_search_'+model_name+'.pickle')
    
    losses = np.vstack(df['losses'])

    idx_min_meanloss = np.argmin(np.mean(losses, axis=1))
    idx_min_finalloss = df['final_loss'].idxmin()

    plt.plot(losses[np.argmin(np.mean(losses, axis=1))], 'b');
    plt.plot(losses[df['final_loss'].idxmin()], 'r');
    plt.plot(losses.T, 'k', alpha=.1);
    plt.yscale('log')
    plt.show()
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax2 = ax.twinx()
    weights = df['weights_last'][idx_min_finalloss]
    net = load_net_from_weights(weights, dt=.1)
    plot_io_net(net, task, ax=ax, ax2=ax2)
    
    weights = df['weights_last'][idx_min_meanloss]
    net = load_net_from_weights(weights, dt=.1)
    plot_io_net(net, task, ax=ax, ax2=ax2)
    
    df_final = df.drop(['losses', 'weights_last'], axis=1)
    df_final = df_final.fillna("None")
    
    param_keys = df_final.columns
    param_keys = param_keys.drop(['trial', 'final_loss'])
    param_keys=param_keys.tolist()
    min_final_loss = df.iloc[df['final_loss'].idxmin()]
    print(df['final_loss'].min())
    print(min_final_loss)
    min_final_meanloss = df_final.groupby(param_keys).mean()['final_loss'].idxmin()
    
    print(min_final_meanloss)
    # print(min_final_meanloss)
    # print("LLL", len(min_final_loss['weights_last']))
    # wi_init, wrec_init, wo_init, brec_init, h0_init = min_final_loss['weights_last']
    # dims = (1, 200, 2)
    # net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
    #           nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
    #           wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, h0_init=h0_init, ML_RNN=training_kwargs['ml_rnn'])

    # np.random.seed(12345)
    # input, target, mask, output, loss = run_net(net, task, batch_size=32, return_dynamics=False, h_init=None)

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax2 = ax.twinx()
    # ax2.plot(output[0,:,0], '--', color='deepskyblue', alpha=0.5)
    # ax2.plot(output[0,:,1], '--', color='lightcoral', alpha=0.5)
    # ax, ax2 = plot_io(ax, ax2, input, output, target)
    # # plt.savefig(parent_dir+'/experiments/'+main_exp_name +'/'+ model_name + '/trajectories{sparsity}.pdf')
    # plt.show()
    
    return df
    
    
def calc_LEs(net, T=10, from_t_step=0):
    task = angularintegration_task(T=T, dt=.1, sparsity=0); #just creates zero input batches
    input, target, mask, output, loss, trajectories = run_net(net, task, batch_size=1, return_dynamics=True, h_init=None);
    lyap_spec, lyaps = calculate_lyapunov_spectrum(act_fun=tanh_jacobian, W=net.wrec.cpu().detach().numpy(), b=net.brec.cpu().detach().numpy(),
                                                   tau=1, x_solved=trajectories[0,...], delta_t=net.dt, from_t_step=from_t_step)

    return sorted(lyap_spec, reverse=True), trajectories

def plot_allLEs_model(main_exp_name, model_name, which='last', T=10, from_t_step=0, mean_color='b', trial_color='b', label='', ax=None, save=False, hidden_i=0):
    print(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')
    params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")
    
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    lyap_specs = np.zeros((len(exp_list), training_kwargs['N_rec']))
    for exp_i, exp in enumerate(exp_list):
        print(exp)
        with open(exp, 'rb') as handle:
            result = pickle.load(handle)
            
        losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
        if which=='post':
            wi_init, wrec_init, wo_init, brec_init, h0_init = weights_last
        elif which=='pre':
            wi_init, wrec_init, wo_init, brec_init, h0_init = weights_init
        dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
        net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
                  nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
                  wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, h0_init=h0_init, ML_RNN=training_kwargs['ml_rnn'])
        
        lyap_spec, trajectories = calc_LEs(net, T=T, from_t_step=from_t_step)
        lyap_specs[exp_i, :] = lyap_spec
        ax.plot(lyap_spec, color=trial_color, alpha=.1)
    
    ax.plot(np.mean(lyap_specs, axis=0), color=mean_color, label=label)
    # plt.show()
    # if save:
    #     plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/les_{which}.pdf', bbox_inches="tight")
    #     plt.close()
    return ax



def plot_all_trajs_model(main_exp_name, model_name, T=128, which='post', hidden_i=0, input_length=10,
                         timepart='all', num_of_inputs=51, plot_from_to=(0,None), pca_from_to=(0,None),
                         plot_output=False, input_range=(-3,3)):
    pca_before_t, pca_after_t = pca_from_to
    before_t, after_t = plot_from_to
    
    exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")
    
    norm = mplcolors.Normalize(vmin=-.5,vmax=.5)
    norm = norm(np.linspace(-.5, .5, num=num_of_inputs, endpoint=True))
    cmap = cmx.get_cmap("coolwarm")
    
    cmap2 = plt.get_cmap('hsv')
    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    
    # fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), sharex=False, sharey=False)
    axes1 = axes[:,:3].flatten()
    axes2 = axes[:,3:].flatten()

    for exp_i, exp in enumerate(exp_list[:9]):
        trajectories, start, target, output = get_hidden_trajs(main_exp_name, model_name, exp, T=T, which=which, input_length=input_length,
                                                               timepart=timepart, num_of_inputs=num_of_inputs, pca_from_to=pca_from_to, input_range=input_range)

        for trial_i in range(trajectories.shape[0]):
            axes1[exp_i].plot(trajectories[trial_i,before_t:after_t,0], trajectories[trial_i,before_t:after_t,1], '-',
                              color=cmap(norm[trial_i]))
            if np.linalg.norm(trajectories[trial_i,-2,:]-trajectories[trial_i,-1,:])  < 1e-4:
                axes1[exp_i].scatter(trajectories[trial_i,-1,0], trajectories[trial_i,-1,1], marker='.', s=100, color=cmap(norm[trial_i]), zorder=100)

        axes1[exp_i].set_axis_off()
        axes1[exp_i].scatter(start[0], start[1], marker='.', s=100, color='k', zorder=100)
        
        if plot_output:
            x = np.linspace(0, 2*np.pi, 1000)
            axes2[exp_i].plot(np.cos(x), np.sin(x), 'k', alpha=.5, linewidth=5, zorder=-1)
            for trial_i in range(output.shape[0]):
                # if trial_i<output.shape[0]-1:
                #     axes2[exp_i].plot([target[trial_i,-1,0], target[trial_i+1,-1,0]], [target[trial_i,-1,1], target[trial_i+1,-1,1]], '--', color=cmap(norm[trial_i]), alpha=.5)
                # axes2[exp_i].scatter(target[trial_i,-1,0], target[trial_i,-1,1], color=cmap(norm[trial_i]), alpha=.1)
                # axes2[exp_i].plot(output[trial_i,:,0], output[trial_i,:,1], '-', color=cmap(norm[trial_i]))
                if np.linalg.norm(trajectories[trial_i,-2,:]-trajectories[trial_i,-1,:])  < 1e-4:
                    axes2[exp_i].scatter(output[trial_i,-1,0], output[trial_i,-1,1], marker='.', s=100, color=cmap(norm[trial_i]), zorder=100)
            
            for t in range(output.shape[1]):
                axes2[exp_i].plot([target[trial_i,t,0], output[trial_i,t,0]],
                                 [target[trial_i,t,1], output[trial_i,t,1]], '-', 
                                 color=cmap2(norm2(np.arctan2(target[trial_i,t,1], target[trial_i,t,0]))))
                # ax.scatter(xval, yval, c=xval, s=300, cmap=colormap, norm=norm, linewidths=0)

            axes2[exp_i].set_axis_off()
            
    if not timepart:
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/trajpca_{which}_{after_t}to{before_t}.png', bbox_inches="tight")
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/trajpca_{which}_{after_t}to{before_t}.pdf', bbox_inches="tight")
    else:
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/trajpca_{which}_{timepart}.png', bbox_inches="tight")
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/trajpca_{which}_{timepart}.pdf', bbox_inches="tight")
    
    if plot_output:
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/output_{which}__{after_t}to{before_t}.png', bbox_inches="tight")
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/output_{which}__{after_t}to{before_t}.pdf', bbox_inches="tight")
            
            
def get_params_exp(params_folder, exp_i=0, which='post'):
    # params_folder = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name
    exp_list = glob.glob(params_folder + "/result*")
    exp = exp_list[exp_i]
    params_path = glob.glob(params_folder + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    with open(exp, 'rb') as handle:
        result = pickle.load(handle)
    
    try:
        losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs = result
        
        
    except:
        losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result


    if len(weights_last) == 6:
        if which=='post':
            wi, wrec, wo, brec, h0, oth = weights_last
        elif which=='pre':
            wi, wrec, wo, brec, h0, oth = weights_init
            
        else:
            return weights_train["wi"][which], weights_train["wrec"][which], weights_train["wo"][which], weights_train["brec"][which], weights_train["h0"][which], weights_train["oths"][which], training_kwargs
        return wi, wrec, wo, brec, h0, oth, training_kwargs

    else:
        if which=='post':
            wi, wrec, wo, brec, h0 = weights_last
        elif which=='pre':
            wi, wrec, wo, brec, h0 = weights_init
        else:
            return weights_train["wi"][which], weights_train["wrec"][which], weights_train["wo"][which], weights_train["brec"][which], weights_train["h0"][which], None,  training_kwargs

        return wi, wrec, wo, brec, h0, None, training_kwargs

    
        
def get_traininginfo_exp(params_folder, exp_i=0, which='post'):
    exp_list = glob.glob(params_folder + "/result*")
    exp = exp_list[exp_i]
    params_path = glob.glob(params_folder + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    with open(exp, 'rb') as handle:
        result = pickle.load(handle)
    
    try:
        losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs = result
        
        
    except:
        losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result


    return losses, gradient_norms, epochs, rec_epochs, weights_train
    

from scipy.spatial.distance import cdist

def identify_limit_cycle(time_series, tol=1e-6):
    d = cdist(time_series[-1,:].reshape((1,-1)),time_series[:-100])
    mind = np.min(d)
    idx = np.argmin(d)
    
    if mind < tol:
        return idx, mind
    else:
        return False, mind
    
def find_periodic_orbits(traj, traj_pca, limcyctol=1e-2, mindtol=1e-10):
    recurrences = []
    recurrences_pca = []
    for trial_i in range(traj.shape[0]):
        idx, mind = identify_limit_cycle(traj[trial_i,:,:], tol=limcyctol) #find recurrence
        # print(idx, mind)
        if mind<mindtol: #for fixed point
            recurrences.append([traj[trial_i,-1,:]])
            recurrences_pca.append([traj_pca[trial_i,-1,:]])
            
        elif idx: #for closed orbit
            recurrences.append(traj[trial_i,idx:,:].numpy())
            recurrences_pca.append(traj_pca[trial_i,idx:,:])

    return recurrences, recurrences_pca


def plot_input_driven_trajectory_2d(traj, traj_pca, wo,
                                    plot_asymp=False, limcyctol=1e-2, mindtol=1e-4,
                                    fxd_points = None, ops_fxd_points=None, 
                                    h_stabilities=None, o_stabilities=None, 
                                    ax=None, plot_epoch=False):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    norm = mplcolors.Normalize(vmin=-.5,vmax=.5)
    num_of_inputs = traj.shape[0]
    norm = norm(np.linspace(-.5, .5, num=num_of_inputs, endpoint=True))
    cmap = cmx.get_cmap("coolwarm")
    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    cmap2 = plt.get_cmap('hsv')
    stab_colors = np.array(['g', 'pink', 'red'])
    
    output = np.dot(traj, wo)
    for trial_i in range(traj.shape[0]):
        ax.plot(traj_pca[trial_i,:,0], traj_pca[trial_i,:,1], color=cmap(norm[trial_i]))
        if plot_asymp:
            target_angle = np.arctan2(output[trial_i,-1,1], output[trial_i,-1,0])
            idx, mind = identify_limit_cycle(traj[trial_i,:,:], tol=limcyctol) #find recurrence
            if idx:
                ax.plot(traj_pca[trial_i,idx:,0], traj_pca[trial_i,idx:,1], '-', linewidth=3, color=cmap2(norm2(target_angle)), zorder=100)
                
                if mind<mindtol:
                    ax.scatter(traj_pca[trial_i,-1,0], traj_pca[trial_i,-1,1], marker='.', s=100, color=cmap2(norm2(target_angle)), zorder=110)
    
    ax.set_axis_off()
    if np.any(fxd_points):
        pca_fxd_points = pca.transform(fxd_points)
        ax.scatter(pca_fxd_points[:,0], pca_fxd_points[:,1], marker='s',
                   c=stab_colors[h_stabilities], alpha=.5, zorder=101)
    if np.any(ops_fxd_points):
        pca_ops_fxd_points = pca.transform(ops_fxd_points)
        if o_stabilities==None:
            ax.scatter(pca_ops_fxd_points[:,0], pca_ops_fxd_points[:,1],
                       marker='x', c='k', alpha=.5, zorder=101)
        else:
            ax.scatter(pca_ops_fxd_points[:,0], pca_ops_fxd_points[:,1],
                   marker='x', c=stab_colors[o_stabilities], alpha=.5, zorder=101)
            
    if plot_epoch:
        plt.text(.0, 0., s=plot_epoch, fontsize=10)

        
def plot_input_driven_trajectory_3d(traj, input_length, plot_traj=True,
                                    recurrences=None, recurrences_pca=None, wo=None,
                                    fxd_points=None, ops_fxd_points=None,
                                    h_stabilities=None,
                                    elev=20., azim=-35, roll=0,
                                    lims=[], plot_epoch=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    norm = mplcolors.Normalize(vmin=-.5,vmax=.5)
    norm = norm(np.linspace(-.5, .5, num=num_of_inputs, endpoint=True))
    cmap = cmx.get_cmap("coolwarm")

    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    cmap2 = plt.get_cmap('hsv')

    norm3 = Normalize(-np.pi, np.pi)
    stab_colors = np.array(['g', 'pink', 'red'])
    
    if plot_traj:
        for trial_i in range(traj.shape[0]):
            ax.plot(traj[trial_i,:input_length,0], traj[trial_i,:input_length,1], zs=traj[trial_i,:input_length,2],
                    zdir='z', color=cmap(norm[trial_i]))
            ax.plot(traj[trial_i,input_length:,0], traj[trial_i,input_length:,1], zs=traj[trial_i,input_length:,2],
                    linestyle='--', zdir='z', color=cmap(norm[trial_i]))

    if recurrences is not None:

        for r_i, recurrence in enumerate(recurrences):
            # print(np.array(recurrences_pca[r_i]).shape)
            
            recurrence_pca = np.array(recurrences_pca[r_i]).T

            if np.array(recurrences_pca[r_i]).shape[0]>1:
                recurrence_pca = recurrences_pca[r_i][:,:3]
                # print(recurrences[r_i].numpy().shape)
                output = np.dot(recurrences[r_i], wo)
                output_angle = np.arctan2(output[...,1], output[...,0])
                
                segments = np.stack([recurrence_pca[:-1], recurrence_pca[1:]], axis=1)
                lc = Line3DCollection(segments, cmap=cmap2, norm=norm3)
                lc.set_array(output_angle)
                
                # Plot the line segments in 3D
                ax.add_collection3d(lc)

            else:
            # for t, point in enumerate(recurrence):
                point = recurrence[0]
                # print("P", point)
                output = np.dot(point, wo)
                output_angle = np.arctan2(output[...,1], output[...,0])

                point_pca = recurrence_pca
                ax.scatter(point_pca[0], point_pca[1], point_pca[2],
                            marker='.', s=100, color=cmap2(norm2(output_angle)), zorder=100)


    if np.any(fxd_points):
        pca_fxd_points = pca.transform(fxd_points)
        if h_stabilities==None:
            ax.scatter(pca_fxd_points[:,0], pca_fxd_points[:,1], pca_fxd_points[:,2],
                       marker='s', color='k',     alpha=.5, zorder=101)
        else:
            ax.scatter(pca_fxd_points[:,0], pca_fxd_points[:,1], pca_fxd_points[:,2],
                       marker='s', color=stab_colors[h_stabilities],     alpha=.5, zorder=101)
    if np.any(ops_fxd_points):
        pca_ops_fxd_points = pca.transform(ops_fxd_points)
        ax.scatter(pca_ops_fxd_points[:,0], pca_ops_fxd_points[:,1], pca_ops_fxd_points[:,2],
                   marker='x', color='k', alpha=.5, zorder=101)
        
    # if plot_epoch:
    #     plt.text(.0, 0., 0, s=plot_epoch, fontsize=10)

    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if lims!=[]:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])
    
    
def plot_output_trajectory(traj, wo, input_length, plot_traj=True,
                           fxd_points=None, ops_fxd_points=None,
                           h_stabilities=None, o_stabilities=None,
                           plot_asymp=False, limcyctol=1e-2, mindtol=1e-4, ax=None,
                           xylims=[-1.2,1.2], plot_epoch=False):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    num_of_inputs = traj.shape[0]
    norm = mplcolors.Normalize(vmin=-.5,vmax=.5)
    norm = norm(np.linspace(-.5, .5, num=num_of_inputs, endpoint=True))
    cmap = cmx.get_cmap("coolwarm")
    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    cmap2 = plt.get_cmap('hsv')
    norm3 = Normalize(-np.pi, np.pi)

    stab_colors = np.array(['g', 'pink', 'red'])
    x = np.linspace(-np.pi, np.pi, 1000)
    # axes[1].plot(np.cos(x), np.sin(x), 'k', alpha=.5, linewidth=5, zorder=-1)
    ax.scatter(np.cos(x), np.sin(x), color=cmap2(norm2(x)), alpha=.1, s=5, zorder=100) #ring
    
    output = np.dot(traj, wo)
    for trial_i in range(traj.shape[0]):
        if plot_traj:
            ax.plot(output[trial_i,0,0], output[trial_i,0,1], '.', c='k', zorder=100) #starting point
            ax.plot(output[trial_i,:input_length,0], output[trial_i,:input_length,1], '-', color=cmap(norm[trial_i]), alpha=.5) #all trajectories
            ax.plot(output[trial_i,input_length:,0], output[trial_i,input_length:,1], '--', color=cmap(norm[trial_i]), alpha=.5) #all trajectories
    
        if plot_asymp:
            target_angle = np.arctan2(output[trial_i,-1,1], output[trial_i,-1,0])
            idx, mind = identify_limit_cycle(traj[trial_i,:,:], tol=limcyctol) #find recurrence
            if idx and not mind<mindtol:
                # ax.plot(output[trial_i,idx:,0], output[trial_i,idx:,1], '-', linewidth=3, color=cmap2(norm2(target_angle)), zorder=100)
                out_i = output[trial_i,idx:,:]
                out_i = out_i.reshape(-1, 1, 2)
                # print(out_i[:-1].shape)
                output_angle = np.arctan2(out_i[:,0,1], out_i[:,0,0])
                segments = np.concatenate([out_i[:-1], out_i[1:]], axis=1)
                lc = LineCollection(segments, cmap=cmap2, norm=norm3)
                lc.set_array(output_angle)
                ax.add_collection(lc)

                
            if mind<mindtol:
                ax.scatter(output[trial_i,-1,0], output[trial_i,-1,1], marker='.', s=100, color=cmap2(norm2(target_angle)), zorder=110)
    if np.any(fxd_points):
        proj_fxd_points = np.dot(fxd_points, wo)
        ax.scatter(proj_fxd_points[:,0], proj_fxd_points[:,1],  marker='s',
                   c=stab_colors[h_stabilities], alpha=.5, zorder=101)    
    if np.any(ops_fxd_points):
        proj_ops_fxd_points = np.dot(ops_fxd_points, wo)
        ax.scatter(proj_ops_fxd_points[:,0], proj_ops_fxd_points[:,1],  marker='x', color='k', alpha=.3, zorder=101)    

    if plot_epoch:
        plt.text(-1.4, 1.4, s=plot_epoch, fontsize=10)

    ax.set_axis_off()
    ax.set_xlim(xylims)
    ax.set_ylim(xylims)
        
def plot_trajs_model(main_exp_name, model_name, exp_i, T=128, which='post',  hidden_i=0,
                     input_length=10, timepart='all',  num_of_inputs=51, plot_from_to=(0,None),
                     pca_from_to=(0,None), input_range=(-3,3), axes=None, limcyctol=1e-2,
                     slowpointtol=1e-8, slowpointmethod='L-BFGS-B'):
    pca_before_t, pca_after_t = pca_from_to
    after_t, before_t = plot_from_to
    norm = mplcolors.Normalize(vmin=-.5,vmax=.5)
    norm = norm(np.linspace(-.5, .5, num=num_of_inputs, endpoint=True))
    cmap = cmx.get_cmap("coolwarm")
    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    cmap2 = plt.get_cmap('hsv')
    params_folder = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name
    wi, wrec, wo, brec, h0, training_kwargs = get_params_exp(params_folder, exp_i=exp_i)
    
    trajectories, traj_pca, start, target, output, input_proj, pca, explained_variance = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs,
                                                                       T=T, which=which,
                                                                       input_length=input_length, timepart=timepart,
                                                                       num_of_inputs=num_of_inputs, pca_from_to=pca_from_to,
                                                                       input_range=input_range)

    if not axes:
        # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=False, sharey=False)
        # fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=False, sharey=False)
        fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=False, sharey=False)
        # fig, axes = plt.subplots(1, 6, figsize=(18, 3), sharex=False, sharey=False)

    for trial_i in range(trajectories.shape[0]):
        axes[0].plot(traj_pca[trial_i,:,0], traj_pca[trial_i,:,1], color=cmap(norm[trial_i]))

        target_angle = np.arctan2(output[trial_i,-1,1], output[trial_i,-1,0])
        axes[2].plot(output[trial_i,0,0], output[trial_i,0,1], '.', c='k', zorder=100) #starting point
        axes[2].plot(output[trial_i,:input_length,0], output[trial_i,:input_length,1], '-', color=cmap(norm[trial_i]), alpha=.25) #all trajectories
        axes[2].plot(output[trial_i,input_length:,0], output[trial_i,input_length:,1], '--', color=cmap(norm[trial_i]), alpha=.25) #all trajectories
        idx, mind = identify_limit_cycle(trajectories[trial_i,:,:], tol=limcyctol) #find recurrence
        if idx:
            axes[1].plot(traj_pca[trial_i,idx:,0], traj_pca[trial_i,idx:,1], '-', linewidth=3, color=cmap2(norm2(target_angle)), zorder=100)
            axes[2].plot(output[trial_i,idx:,0], output[trial_i,idx:,1], '-', linewidth=3, color=cmap2(norm2(target_angle)), zorder=100)
            axes[3].plot(input_proj[trial_i,after_t:before_t,0], '-', color=cmap2(norm2(target_angle)))

            if mind<1e-4:
                axes[1].scatter(traj_pca[trial_i,-1,0], traj_pca[trial_i,-1,1],  marker='.', s=100, color=cmap2(norm2(target_angle)), zorder=110)
                axes[2].scatter(output[trial_i,-1,0], output[trial_i,-1,1], marker='.', s=100, color=cmap2(norm2(target_angle)), zorder=110)

    axes[1].set_axis_off()
    axes[1].scatter(start[0], start[1], marker='.', s=100, color='k', zorder=100)
        
    traj = trajectories[:,:input_length:1500,:].reshape((-1,training_kwargs['N_rec']))
    print('Finding slow points in hidden space')
    fxd_points, speeds = find_slow_points(wrec, brec, dt=training_kwargs['dt_rnn'], trajectory=traj, tol=slowpointtol, method=slowpointmethod)
    print('Finding slow points in output space')
    ops_fxd_points, speeds = find_slow_points(wrec, brec, wo=wo, dt=training_kwargs['dt_rnn'], trajectory=traj,
                                              outputspace=True, tol=slowpointtol, method=slowpointmethod)

    print(fxd_points.shape)
    if fxd_points.shape[0]>0:
        pca_fxd_points = pca.transform(fxd_points)
        pca_ops_fxd_points = pca.transform(ops_fxd_points)
        axes[1].scatter(pca_fxd_points[:,0], pca_fxd_points[:,1], marker='s', color='k', alpha=.5, zorder=101)    
        axes[0].scatter(pca_fxd_points[:,0], pca_fxd_points[:,1],  marker='s', color='k', alpha=.5, zorder=101)    
        axes[1].scatter(pca_ops_fxd_points[:,0], pca_ops_fxd_points[:,1],  marker='x', color='k', alpha=.5, zorder=101)    
        axes[0].scatter(pca_ops_fxd_points[:,0], pca_ops_fxd_points[:,1],  marker='x', color='k', alpha=.5, zorder=101)    
        proj_fxd_points = np.dot(fxd_points, wo)
        proj_ops_fxd_points = np.dot(ops_fxd_points, wo)
        axes[2].scatter(proj_fxd_points[:,0], proj_fxd_points[:,1],  marker='s', color='k', alpha=.5, zorder=101)    
        axes[2].scatter(proj_ops_fxd_points[:,0], proj_ops_fxd_points[:,1],  marker='x', color='k', alpha=.5, zorder=101)    

    
    x = np.linspace(-np.pi, np.pi, 1000)
    # axes[1].plot(np.cos(x), np.sin(x), 'k', alpha=.5, linewidth=5, zorder=-1)
    axes[2].scatter(np.cos(x), np.sin(x), color=cmap2(norm2(x)), alpha=.5, s=2, zorder=-1) #ring

    #Speed
    x_lim = np.max(np.abs(output.numpy()))
    num_x_points = 21
    # output_logspeeds, all_logspeeds = average_logspeed(wrec, wo, brec, trajectories[:,input_length:,:], x_min=-x_lim, x_max=x_lim, num_x_points=num_x_points)
    # im=axes[4].imshow(np.rot90(output_logspeeds), cmap='inferno')
    # cbar = fig.colorbar(im, ax=axes[3])
    # cbar.set_label("log(speed)")
    
    #Vector field
    plot_average_vf(trajectories[:1,:input_length,:], wi, wrec, brec, wo, input_length=input_length, 
                    num_of_inputs=num_of_inputs, input_range=(input_range[0],input_range[0]), x_lim=x_lim,
                    num_x_points=num_x_points, color='red', ax=axes[4], change_from_outtraj=False)
    
    plot_average_vf(trajectories[-1:,:input_length,:], wi, wrec, brec, wo, input_length=input_length, 
                    num_of_inputs=num_of_inputs, input_range=(input_range[1],input_range[1]), x_lim=x_lim,
                    num_x_points=num_x_points, color='blue', ax=axes[4], change_from_outtraj=False)
    axes[2].set_xlim([-x_lim, x_lim])
    axes[2].set_ylim([-x_lim, x_lim])

    axes[0].set_axis_off()
    axes[1].set_axis_off()
    axes[2].set_axis_off()
    axes[3].set_xticks([])
    axes[4].set_axis_off()

    plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/mss_output_{which}_{exp_i}.pdf', bbox_inches="tight")
    plt.show()
    
    

def plot_average_vf(trajectories, wi, wrec, brec, wo, input_length=10,
                     num_of_inputs=51, input_range=(-3,3), change_from_outtraj=False,
                     ax=None, x_lim=1.2, num_x_points=21, color='k'):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=False, sharey=False)

    x_values = np.linspace(-x_lim, x_lim, num_x_points)
    y_values = np.linspace(-x_lim, x_lim, num_x_points)
    
    # Generate all grid points
    grid_points = np.array([(np.round(x,5), np.round(y,5)) for x in x_values for y in y_values])+(2*x_lim)/num_x_points
    average_vf = np.zeros((num_x_points, num_x_points, 2))
    if change_from_outtraj:
        for trajectory in trajectories:
            outtrajectories = np.dot(trajectory, wo)    
            average_vf += average_output_changevectors(wi, wrec, brec, wo, 0, outtrajectories, x_min=-x_lim, x_max=x_lim, num_x_points=num_x_points)

    elif input_length==0:
        for trajectory in trajectories:
            average_vf += average_input_vectorfield(wi, wrec, brec, wo, 0, trajectory, x_min=-x_lim, x_max=x_lim, num_x_points=num_x_points)
    else:
        for I, trajectory in zip(np.linspace(input_range[0], input_range[1], num=num_of_inputs, endpoint=True), trajectories[:,:input_length,:]):
            # print("I", I)
            average_vf += average_input_vectorfield(wi, wrec, brec, wo, I, trajectory, x_min=-x_lim, x_max=x_lim, num_x_points=num_x_points)
    
    average_vf = np.round(average_vf, 2)
    average_vf /= num_of_inputs
    # print(average_vf)
    theta = np.radians(90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    average_vf = np.dot(average_vf, R)
    ax.quiver(grid_points[:,0], grid_points[:,1],
                   average_vf.reshape((num_x_points**2,2))[:,0], average_vf.reshape((num_x_points**2,2))[:,1], color=color)
    ax.set_axis_off()


    
def get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs, oth=None, T=128, which='post', input_length=10, timepart='all',
                     num_of_inputs=51, pca_from_to=(0,None), input_range=(-3,3), random_angle_init=False):
    pca_after_t, pca_before_t = pca_from_to

    dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
    net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
              nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
              wi_init=wi, wrec_init=wrec, wo_init=wo, brec_init=brec, h0_init=h0, oth_init=oth,
              ML_RNN=training_kwargs['ml_rnn'], map_output_to_hidden=training_kwargs['map_output_to_hidden'])
    
    min_input, max_input = input_range

    input = np.zeros((num_of_inputs,T,training_kwargs['N_in']))
    stim = np.linspace(min_input, max_input, num=num_of_inputs, endpoint=True)
    input[:,:input_length,0] = np.repeat(stim,input_length).reshape((num_of_inputs,input_length))
    input = torch.from_numpy(input).float() 
    
    outputs_1d = np.cumsum(input, axis=1)*training_kwargs['dt_task']
    if random_angle_init:
        random_angles = np.random.uniform(-np.pi, np.pi, size=num_of_inputs).astype('f')
        outputs_1d += random_angles[:, np.newaxis, np.newaxis]
    target = np.stack((np.cos(outputs_1d), np.sin(outputs_1d)), axis=-1).reshape((num_of_inputs, T, training_kwargs['N_out']))
    h = h0 # np.zeros((num_of_inputs,training_kwargs['N_rec'])) 
    
    # print(net.map_output_to_hidden)
    # if training_kwargs['random_angle_init']:
    #     with torch.no_grad():
    #         h =  np.dot(target[:,0,:],net.output_to_hidden)

    _target = torch.from_numpy(target)
    # print(net.output_to_hidden.dtype)

    # _target = _target.to(device=training_kwargs['device']).float() 
    with torch.no_grad():
        output, trajectories = net(input, return_dynamics=True, h_init=h, target=_target)
    
    input_proj = np.dot(trajectories, wi.T)
    
    pca = PCA(n_components=10)
    pca.fit(trajectories.reshape((-1,training_kwargs['N_rec'])))
    explained_variance = pca.explained_variance_ratio_.cumsum()

    # pca = PCA(n_components=2)   
    trajectories_tofit = trajectories[:,pca_after_t:pca_before_t,:].numpy().reshape((-1,training_kwargs['N_rec']))
    pca.fit(trajectories_tofit)
    traj_pca = pca.transform(trajectories.numpy().reshape((-1,training_kwargs['N_rec']))).reshape((num_of_inputs,-1,10))
    start= pca.transform(h0.reshape((1,-1))).T
    traj_pca = traj_pca[:,:,:]

    # makedirs(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+'/hidden'+exp[-21:-7])
    return trajectories, traj_pca, start, target, output, input_proj, pca, explained_variance

def plot_output_vs_target(target, output, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    cmap2 = plt.get_cmap('hsv')
    x = np.linspace(-np.pi, np.pi, 1000)
    ax.plot(np.cos(x), np.sin(x), 'k', alpha=.5, linewidth=5, zorder=-1)
    # ax.scatter(np.cos(x), np.sin(x), color=cmap2(norm2(x)), alpha=.5, s=2, zorder=-1)
    ax.scatter(output[0,0], output[0,1], c='k', zorder=10)
    for t in range(output.shape[0]):
                ax.plot([target[t,0], output[t,0]],
                                 [target[t,1], output[t,1]], '-', 
                                 color=cmap2(norm2(np.arctan2(target[t,1], target[t,0]))))
                if t>0:
                    ax.plot([output[t,0], output[t-1,0]],
                                     [output[t,1], output[t-1,1]], '-', 
                                     color=cmap2(norm2(np.arctan2(target[t,1], target[t,0]))))
    ax.set_axis_off()
    plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/output_vs_target.pdf', bbox_inches="tight")

def average_output_changevectors(wi, wrec, brec, wo, I, trajectories, x_min=-2, x_max=2, num_x_points=11):
    # Create a regular grid
    x_values = np.linspace(x_min, x_max, num_x_points)
    y_values = np.linspace(x_min, x_max, num_x_points)
    
    # Generate all grid points
    # grid_points = [(np.round(x,5), np.round(y,5)) for x in x_values for y in y_values]
    average_vector_field = np.zeros((num_x_points, num_x_points, 2))
    bin_counts = np.zeros((num_x_points, num_x_points))
     
    for i,x in enumerate(trajectories[:-1,:]):
        cell_containing_point = find_grid_cell(x, x_values, y_values, num_x_points=num_x_points)
        full_vf_at_x = x-trajectories[i+1,:]
        average_vector_field[cell_containing_point] += full_vf_at_x.squeeze()
        bin_counts[cell_containing_point] += 1
    
    avf = average_vector_field/bin_counts.reshape((num_x_points,num_x_points,1))
    avf = np.where(avf==np.nan, 0, avf)
    return avf

def average_input_vectorfield(wi, wrec, brec, wo, I, trajectory, x_min=-2, x_max=2, num_x_points=11):
    """
    Calculates average over x for  g(x,I) = tanh(Wx+b+W_inI) - tanh(Wx+b)
    for a fixed I

    """
    # Create a regular grid
    x_values = np.linspace(x_min, x_max, num_x_points)
    y_values = np.linspace(x_min, x_max, num_x_points)
    
    # Generate all grid points
    # grid_points = [(np.round(x,5), np.round(y,5)) for x in x_values for y in y_values]
    average_vector_field = np.zeros((num_x_points, num_x_points, 2))
    bin_counts = np.ones((num_x_points, num_x_points))

    for t, x in enumerate(trajectory):
        target_point = np.dot(wo.T, x)

        cell_containing_point = find_grid_cell(target_point, x_values, y_values, num_x_points=num_x_points)

        if I==0:
            full_vf_at_x = np.tanh(np.dot(wrec, x)+brec).squeeze()
        else:
            full_vf_at_x = input_vectorfield(x, wi, wrec, brec, I).squeeze()
        # print(t, target_point, cell_containing_point, np.dot(wo.T, full_vf_at_x).squeeze())

        average_vector_field[cell_containing_point] += np.dot(wo.T, full_vf_at_x).squeeze()
        bin_counts[cell_containing_point] += 1

    avf = average_vector_field/bin_counts.reshape((num_x_points,num_x_points,1))
    # avf = np.where(np.isnan(avf), 0, avf)
    return avf



def input_vectorfield(x, wi, wrec, brec, I):
    """
    Calculates g(x, I) in the split RNN equation \dot x = f(x)+g(x,I)
    for \dot x = tanh(Wx+b+W_inI)
    i.e. g(x,I) = tanh(Wx+b+W_inI) - tanh(Wx+b)

    Parameters
    ----------
    wi : TYPE
        DESCRIPTION.
    wrec : TYPE
        DESCRIPTION.
    wo : TYPE
        DESCRIPTION.
    brec : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    dotx = np.tanh(np.dot(wrec, x)+brec+np.dot(wi,I))
    fx = np.tanh(np.dot(wrec, x)+brec)
    return dotx - fx



def ReLU(x):
    return np.where(x<0,0,x)

def relu_ode(t,x,W):
    return ReLU(np.dot(W,x)) - x 

# def tanh_ode(t,x,W):
#     return np.tanh(np.dot(W,x)) - x 


    
def find_grid_cell(target_point, x_values, y_values, num_x_points=11):
    
    # Iterate through grid cells and find the cell containing the target point
    for i in range(num_x_points - 1):
        for j in range(num_x_points - 1):
            x1, x2 = x_values[i], x_values[i + 1]
            y1, y2 = y_values[j], y_values[j + 1]
            
            if x1 <= target_point[0] <= x2 and y1 <= target_point[1] <= y2:
                cell_containing_point = (i, j)
                break
    return cell_containing_point
    
def average_logspeed(wrec, wo, brec, trajectories, x_min=-2, x_max=2, num_x_points=11):
    """
    Calculate the average speed over the trajectories per grid element in output space

    Parameters
    ----------
    wrec : TYPE
        DESCRIPTION.
    wo : TYPE
        DESCRIPTION.
    brec : TYPE
        DESCRIPTION.
    trajectories : TYPE
        DESCRIPTION.

    Returns
    -------
    all_logspeeds : TYPE
        DESCRIPTION.

    """
    
    #discretize output space
    # Create a regular grid
    x_values = np.linspace(x_min, x_max, num_x_points)
    y_values = np.linspace(x_min, x_max, num_x_points)
    
    # Generate all grid points
    grid_points = [(np.round(x,5), np.round(y,5)) for x in x_values for y in y_values]
    output_logspeeds = np.zeros((num_x_points, num_x_points))
    bin_counts = np.zeros((num_x_points, num_x_points))
    
    #calculate speed along trajectories
    all_logspeeds = np.zeros((trajectories.shape[0], trajectories.shape[1]-1))
    for trial_i in range(trajectories.shape[0]):
        # speeds = np.linalg.norm(trajectories[trial_i, :-1, :]-trajectories[trial_i, 1:, :])
        # all_logspeeds[trial_i,:] = np.log(speeds)

        for t in range(trajectories.shape[1]-1):
            x = trajectories[trial_i, t, :]
            # speed = np.linalg.norm(np.dot(wrec,np.tanh(x))+brec - x)
            
            speed = np.linalg.norm(trajectories[trial_i, t+1, :]-trajectories[trial_i, t, :])
            
            all_logspeeds[trial_i,t] = np.log(speed)
            
            target_point = np.dot(wo.T, x)
            cell_containing_point = find_grid_cell(target_point, x_values, y_values, num_x_points=num_x_points)
            output_logspeeds[cell_containing_point] += np.log(speed)
            bin_counts[cell_containing_point] += 1
    return output_logspeeds/bin_counts, all_logspeeds

def plot_allLEs(main_exp_name, mean_colors, trial_colors, labels, T=10, from_t_step=0, which='post'):
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    for model_i, model_name in enumerate(model_names):
        model_i += 1
        plot_allLEs_model(main_exp_name, model_name, first_or_last=which, from_t_step=from_t_step, T=T,
                          ax=ax, mean_color=mean_colors[model_i], trial_color=trial_colors[model_i], label=labels[model_i])
    ax.set_ylabel("Lyapunov exponent")

    plt.legend()
    plt.savefig(parent_dir+'/experiments/'+main_exp_name+f'/les_{which}.pdf', bbox_inches="tight")
    plt.show()

def plot_final_losses(main_exp_name, exp="triallength", mean_colors=['b'], trial_colors=['b'], labels=['']):
    sub_exp_names = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/*' + os.path.sep)
    
    if exp=="triallength":
        xlabel="Trial length"
        sub_exp_names = [float(PurePath(sub_exp_name).parts[-1]) for sub_exp_name in sub_exp_names]
    elif exp=="size":
        xlabel="Network size"
        sub_exp_names = [int(PurePath(sub_exp_name).parts[-1]) for sub_exp_name in sub_exp_names]
    sub_exp_names = sorted(sub_exp_names)
    print(sub_exp_names)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    xticks, positions = [], []
    for sub_exp_i, sub_exp_name in enumerate(sub_exp_names):
        print(sub_exp_name)
        sub_exp_name = str(sub_exp_name)
        for model_i, model_name in enumerate(model_names):
            print(model_name)
            print(parent_dir+'/experiments/' + main_exp_name +'/'+ sub_exp_name +'/'+ model_names[model_i])
            params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ sub_exp_name +'/'+ model_names[model_i] + '/param*.yml')[0]
            training_kwargs = yaml.safe_load(Path(params_path).read_text())
            all_losses = plot_losses_and_trajectories(plot_losses_or_trajectories=None, main_exp_name=main_exp_name+'/'+ sub_exp_name, model_name=model_name)
            final_losses = all_losses[:,-10:]
            final_losses = np.mean(final_losses, axis=1)
            final_losses_nonan = final_losses[~np.isnan(final_losses)]
            
            # ax.scatter([training_kwargs['T']/training_kwargs['dt_task']]*final_losses.shape[0], final_losses, marker='_', color=mean_colors[model_i], alpha=.9)
            # plt.bar(bins1[:-1], hist_data1, width=bar_width, alpha=0.5, label='Data 1')
            if exp=="triallength":
                bplot = ax.boxplot(final_losses_nonan, widths=25,
                                   positions=[np.log(training_kwargs['T']/training_kwargs['dt_task'])*sub_exp_i*50+35*model_i-70],
                                   patch_artist=True)
            
            elif exp=="size":
                bplot = ax.boxplot(final_losses_nonan, widths=10,
                                   positions=[sub_exp_i*100+15*model_i-35],
                                   patch_artist=True)

            bplot['boxes'][0].set_facecolor(mean_colors[model_i])

        
        if exp=="triallength":
            xticks.append(int(training_kwargs['T']/training_kwargs['dt_task']))
            positions.append(np.log(training_kwargs['T']/training_kwargs['dt_task'])*sub_exp_i*50)
        elif exp=="size":
            positions.append(sub_exp_i*100)
            xticks.append(int(training_kwargs['N_rec']))

    
    ax.set_xticks(np.array(positions),xticks)
    ax.set_xlim([positions[0]-100,positions[-1]+100])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final loss")
    handles = [mpatches.Patch(color=color, label=f'Label {i}') for i, color in enumerate(mean_colors)]

    plt.legend(handles, labels)
    if exp=="triallength":
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/alltimes_finallosses.pdf', bbox_inches="tight")
    elif exp=="size":
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/allsizes_finallosses.pdf', bbox_inches="tight")

    plt.show()  
    
def plot_participatioratio(main_exp_name, model_name, task, first_or_last='last'):
    params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")[:1]
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    for exp_i, exp in enumerate(exp_list):
        with open(exp, 'rb') as handle:
            result = pickle.load(handle)
            
        losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs_ = result
        if first_or_last=='last':
            wi_init, wrec_init, wo_init, brec_init, h0_init = weights_last
        elif first_or_last=='first':
            wi_init, wrec_init, wo_init, brec_init, h0_init = weights_init
        dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
        net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'],
                  nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
                  wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, h0_init=h0_init, ML_RNN=training_kwargs['ml_rnn'])
        
        input, target, mask, output, loss, trajectories = run_net(net, task, batch_size=1, return_dynamics=True, h_init=None);
        
        trajectories = trajectories.reshape((-1, training_kwargs['N_rec']))
        trajectories -= np.mean(trajectories, axis=0) 
        U, S, Vt = svd(trajectories, full_matrices=True)
        print(participation_ratio(S))
        
def talu(x):
    y = np.where(x<0,np.tanh(x),x)
    return y

def rect_tanh(x):
    y = np.where(x>0,x,0)
    return y

def rnn_step(x, wrec, brec, dt, nonlinearity):
    act = np.dot(wrec, x) + brec
    rec_input = nonlinearity(act)
    dx =  - dt * x  + dt * rec_input
    return dx

def rect_tanh_step(x, wrec, brec, dt):
    tanh_act = np.tanh(np.dot(wrec, x) + brec)
    rec_input = np.where(tanh_act>0,tanh_act,0)
    dx =  - dt * x  + dt * rec_input
    return dx

def talu_step(x, wrec, brec, dt):
    act = np.dot(wrec, x) + brec
    rec_input = np.where(act<0,np.tanh(act),act)
    dx =  - dt * x  + dt * rec_input
    return dx
        
def tanh_ode(t,x,wrec, brec):
    return np.tanh(np.dot(wrec,x)+brec) - x 

def rnn_speed_function(x, wrec, brec, dt, nolinearity):
    return np.linalg.norm(rnn_step(x, wrec, brec, dt, nolinearity))**2

def rnn_speed_function_in_outputspace(x, wrec, brec, wo, dt, nonlinearity):
    return np.linalg.norm(np.dot(wo.T, rnn_step(x, wrec, brec, dt, nonlinearity)))**2


# def tanh_rnn_speed(x, wrec, brec):
#     f_x = tanh_ode(0,x,wrec, brec)
#     return np.linalg.norm(f_x)

def tanh_jacobian(x,W,b,tau, mlrnn=True):

    if mlrnn:
        return (-np.eye(W.shape[0]) + np.multiply(W,1/np.cosh(np.dot(W,x)+b)**2))/tau
    else:
        return (-np.eye(W.shape[0]) + np.multiply(W,1/np.cosh(x)**2))/tau

# def tanh_jacobian(x, wrec, brec, dt, mlrnn=True):
#       # = 2*tanh_step(x, wrec, brec, dt)
#     if mlrnn:
#         return np.sum(dt*(-np.eye(wrec.shape[0]) + np.multiply(wrec,1/np.cosh(np.dot(wrec,x)+brec)**2)), axis=1)
#     else:
#         return dt*(-np.eye(wrec.shape[0]) + np.multiply(wrec,1/np.cosh(x)**2))
        
def find_slow_points(wrec, brec, wo=None, dt=1, outputspace=False, trajectory=None, n_points=100,
                     method='L-BFGS-B', tol=1e-9, nonlinearity='tanh'):
    
    if nonlinearity=='tanh':
        nonlinearity_function = np.tanh
    elif nonlinearity=='talu':
        nonlinearity_function = talu
    elif nonlinearity=='rect_tanh': 
        nonlinearity_function = rect_tanh
    
    N = wrec.shape[0]
    fxd_points = []
    speeds = []
    if not np.any(trajectory.numpy()):
        for p_i in tqdm(range(n_points)):    
            x0 = np.random.uniform(-1,1,N)
            if outputspace:
                res = minimize(rnn_speed_function_in_outputspace, x0, method=method, tol=tol, args=tuple([wrec, brec, wo, dt, nonlinearity_function]))
            else:
                res = minimize(rnn_speed_function, x0, method=method, tol=tol, args=tuple([wrec, brec, dt, nonlinearity_function]))
            if res.success: # and res.fun<tol: 
                fxd_points.append(res.x)
                speeds.append(res.fun)
    else:
        for x0 in tqdm(trajectory):
            if outputspace:
                res = minimize(rnn_speed_function_in_outputspace, x0, method=method, tol=tol, args=tuple([wrec, brec, wo, dt, nonlinearity_function])) #jac=tanh_jacobian
            else:
                res = minimize(rnn_speed_function, x0, method=method, tol=tol,  args=tuple([wrec, brec, dt, nonlinearity_function]))
            if res.success: # and res.fun<tol: 
                fxd_points.append(res.x)
                speeds.append(res.fun)
        
    return np.array(fxd_points), np.array(speeds)


def plot_error_over_time(params_folder, exp_i, T=None, sparsity=0.2, batch_size=2**8, ax=None, label=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3));
    params_folder = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name
    wi, wrec, wo, brec, h0, training_kwargs = get_params_exp(params_folder, exp_i=exp_i)

    dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
    net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
              nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
              wi_init=wi, wrec_init=wrec, wo_init=wo, brec_init=brec, h0_init=h0, ML_RNN=training_kwargs['ml_rnn'])
    if not T:
        T = training_kwargs['T']
    # task = angularintegration_task(T=T, dt=training_kwargs['dt_task'], sparsity=sparsity)
    # input, target, mask, output, loss = run_net(net, task, batch_size=batch_size, return_dynamics=False)
    trajectories, traj_pca, start, target, output, input_proj, pca, explained_variance = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs,
                                                                                      T=T, input_length=T, which=which,
                                                                                      pca_from_to=pca_from_to,
                                                                                      num_of_inputs=num_of_inputs, input_range=(-.5,.5))
    #plot error over time
    errors = np.linalg.norm(output-target, axis=(2))
    ax.plot(np.mean(errors,axis=0), label=label)
    # error = np.mean(errors)

def plot_error_over_time_list(main_exp_name, model_name, T=None, batch_size=2**8, 
                              act_lambdas = [0.01, 0.001, 1e-05, 1e-07, 1e-09, 0]):
    colors = [plt.cm.jet(i) for i in np.linspace(.5, 1, len(act_lambdas))]
    fig, ax = plt.subplots(1, 1, figsize=(3, 3));
    ax.set_prop_cycle('color', colors)
    for act_lambda in tqdm(act_lambdas):
        np.random.seed(0)
        main_exp_name_an = main_exp_name + f'/{act_lambda}'
        exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name_an +'/'+ model_name + "/result*")
        exp = exp_list[0]
        # print(main_exp_name_an)
        if act_lambda==0:
            label = 0
        else:
            label=r'$10^{%02d}$' %np.log10(act_lambda)
        plot_error_over_time(main_exp_name_an, model_name, exp, T=T, batch_size=batch_size, ax=ax, label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    plt.legend(title='Reg. param.', bbox_to_anchor=(1., 1.))
    plt.savefig(parent_dir+f"/experiments/angular_integration/act_norm/error_T{T}.pdf", bbox_inches="tight")

def plot_explained_variances(main_exp_name, model_name, input_length, batch_size=2**8, 
                              act_lambdas = [0.01, 0.001, 1e-05, 1e-07, 1e-09, 0]):
    colors = [plt.cm.jet(i) for i in np.linspace(.5, 1, len(act_lambdas))]
    fig, ax = plt.subplots(1, 1, figsize=(3, 3));
    ax.set_prop_cycle('color', colors)
    for act_lambda in tqdm(act_lambdas):
        np.random.seed(0)
        main_exp_name_an = main_exp_name + f'/{act_lambda}'
        print(parent_dir+"/experiments/" + main_exp_name_an +'/'+ model_name + "/result*")
        exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name_an +'/'+ model_name + "/result*")
        exp = exp_list[0]
        if act_lambda==0:
            label = 0
        else:
            label=r'$10^{%02d}$' %np.log10(act_lambda)
        # explained_variance
        params_folder = parent_dir+'/experiments/' + main_exp_name_an +'/'+ model_name
        wi, wrec, wo, brec, h0, training_kwargs = get_params_exp(params_folder, exp_i=0)
        trajectories, traj_pca, start, target, output, input_proj, pca, explained_variance = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs,
                                                                                          T=2*input_length, input_length=input_length, which=which,
                                                                                          pca_from_to=pca_from_to,
                                                                                          num_of_inputs=num_of_inputs, input_range=(-.5,.5))
        ax.plot(explained_variance, label=label)
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained variance")
        plt.legend(title='Reg. param.', bbox_to_anchor=(1., 1.))
        plt.savefig(parent_dir+"/experiments/angular_integration/act_norm/exp_vars.pdf", bbox_inches="tight")

def plot_slowpoints(params_folder, exp_i):
    params_folder = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name
    wi, wrec, wo, brec, h0, training_kwargs = get_params_exp(params_folder, exp_i=exp_i)
    trajectories, traj_pca, start, target, output, input_proj, pca, explained_variance = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs,
                                                                                      T=2*input_length, input_length=input_length, which=which,
                                                                                      pca_from_to=pca_from_to,
                                                                                      num_of_inputs=num_of_inputs, input_range=input_range)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3)); 
    # # plot_average_vf(trajectories, wi, wrec, brec, wo, input_length=input_length, 
    # #                 num_of_inputs=2, input_range=(-.51,-.5), ax=ax)
    # # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/vf_{which}.pdf', bbox_inches="tight")
    traj = trajectories[:10,::50,:].reshape((-1,training_kwargs['N_rec']))
    fxd_points, speeds = find_slow_points(wrec, brec, dt=training_kwargs['dt_rnn'], trajectory=traj)
    # #pca projection
    # proj_fxd_points = pca.transform(fxd_points.reshape(-1, training_kwargs['N_rec']))
    # proj_h0 = pca.transform(h0.reshape(1, -1))
    # #output projection
    proj_fxd_points = np.dot(fxd_points, wo)

    proj_h0 = np.dot(h0, wo)
    ax.scatter(proj_fxd_points[:,0], proj_fxd_points[:,1])    
    # ax.scatter(proj_trajectory[:,0], proj_trajectory[:,1])    

    ax.scatter(proj_h0[0], proj_h0[1], c='k')
    
    cmap2 = plt.get_cmap('hsv')
    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    x = np.linspace(-np.pi, np.pi, 1000)
    ax.scatter(np.cos(x), np.sin(x), color=cmap2(norm2(x)), alpha=.5, s=2, zorder=-1)

def get_stabilities(fxd, wrec, brec, tau):
    h_stabilities = []
    for fxd in fxd_points:
        J = tanh_jacobian(fxd, wrec, brec, tau=tau)
        eigvals, _ = np.linalg.eig(J)
        maxeig = np.max(np.real(eigvals))
        numpos = np.where(np.real(eigvals)>0)[0].shape[0]
        # print(maxeig, numpos)
        h_stabilities.append(numpos)
    return h_stabilities


from itertools import chain, combinations, permutations
from analysis_functions import find_analytic_fixed_points, powerset, relu_step_input
def fixed_point_analysis():
    T = 2000
    batch_size = 1000
    for exp_i in range(10):
        task = contbernouilli_noisy_integration_task(T=T,
                                                  input_length=training_kwargs['input_length'],
                                                  sigma=training_kwargs['task_noise_sigma'],
                                                  final_loss=training_kwargs['final_loss'])
        # if exp_i==8:
        #     continue
        print("EXP:", exp_i)
        main_exp_name='integration/N20_long'
        params_folder = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name
        net =  load_net_from_weights(params_folder, exp_i)
        wi, wrec, wo, brec, h0, training_kwargs = get_params_exp(params_folder, exp_i)
        input, target, mask, output, loss, trajectories = run_net(net, task, batch_size=batch_size, return_dynamics=True, h_init=None);
    
        plt.show()
        fixed_point_list, stabilist, unstabledimensions, eigenvalues_list = find_analytic_fixed_points(wrec, brec)
    
        trajectories_tofit = trajectories[:,:200,:].reshape((-1,training_kwargs['N_rec']))
        pca.fit(trajectories_tofit)
        traj_pca = pca.transform(trajectories[:,:2000,:].reshape((-1,training_kwargs['N_rec']))).reshape((batch_size,-1,10))
        fixed_point_pca = pca.transform(np.array(fixed_point_list).reshape(1, -1))
    
        for i in range(batch_size):
            plt.plot(traj_pca[i,200:, 0],traj_pca[i,200:, 1])
            # for t in range(0,T,50):
            #     plt.scatter(traj_pca[i,t, 0],traj_pca[i,t, 1], color=cmap(norm[t]), s=1)
        # plt.scatter(fixed_point_pca[:,0], fixed_point_pca[:,1], zorder=100, c='k')
        
        for i in range(len(fixed_point_list)):
            plt.scatter(fixed_point_pca[i,0], fixed_point_pca[i,1], zorder=100, c=colors[stabilist[i]])
            
        minx = np.min(fixed_point_pca[:,0])-2
        maxx = np.max(fixed_point_pca[:,0])+20
        miny = np.min(fixed_point_pca[:,1])-2
        maxy = np.max(fixed_point_pca[:,1])+20
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        plt.show()
    
        for i in range(len(fixed_point_list)):
            #print(np.real(eigenvalues_list[i][0]))
            nearest_to_zero = min(np.real(eigenvalues_list[i]), key=lambda x:abs(x))
            print(nearest_to_zero)
        
        

if __name__ == "__main__":

    model_names=['lstm', 'low','high', 'ortho', 'qpta']
    mean_colors = ['k', 'r',  'darkorange', 'g', 'b']
    trial_colors = ['gray', 'lightcoral', 'moccasin', 'lime', 'deepskyblue']
    labels=['LSTM', r'$g=.5$', r'$g=1.5$', 'Orthogonal', 'QPTA']
    
    # main_exp_name='angularintegration/lambda_T2'
    # plot_final_losses(main_exp_name, exp="triallength", mean_colors=mean_colors, trial_colors=trial_colors, labels=labels)
    # main_exp_name='angularintegration/sizes2'
    # plot_final_losses(main_exp_name, exp="size", mean_colors=mean_colors, trial_colors=trial_colors, labels=labels)
    
    # T = 20
    # from_t_step = 90
    # plot_allLEs_model(main_exp_name, 'qpta', which='pre', T=10, from_t_step=0, mean_color='b', trial_color='b', label='', ax=None, save=True)
    T = 128*32*2
    num_of_inputs = 11
    input_length = int(128)*2
    which =  3# 'post'
    plot_from_to = (T-1*input_length,T)
    pca_from_to = (0,T)
    slowpointmethod= 'L-BFGS-B' #'Newton-CG' #

    model_name = ''
    main_exp_name='angular_integration/hidden/25.6'
    # main_exp_name='angular_integration/gains/1'
    main_exp_name='angular_integration/N30'
    # main_exp_name='angular_integration/long/100'
    main_exp_name='angular_integration/rect_tanh_N100_fixedstart/' #training_fullest
    
    # main_exp_name='angular_integration/act_norm/1e-07'
    # main_exp_name='angular_integration/act_reg_from10'

    # plot_explained_variances(main_exp_name, model_name, input_length=input_length, batch_size=2**8, 
    #                               act_lambdas = [1e-05, 1e-07, 1e-09, 0])
    
    exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")


    if True:    
        exp_i = 0
        input_range = (-.5, .5)   

        params_folder = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name
        which = 'post'
        losses, gradient_norms, epochs, rec_epochs, weights_train = get_traininginfo_exp(params_folder, exp_i, which)
        print("Epochs trained: ", np.argmin(losses))
        # plt.plot(np.log(losses))
        # plt.close()
        try:
            wi, wrec, wo, brec, h0, oth, training_kwargs = get_params_exp(params_folder, exp_i, which)
        except:
            wi, wrec, wo, brec, h0, training_kwargs = get_params_exp(params_folder, exp_i, which)
            oth = None

        
        trajectories, traj_pca, start, target, output, input_proj, pca, explained_variance = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs, oth,
                                                                                            T=T, input_length=input_length, which=which,
                                                                                            pca_from_to=pca_from_to,
          num_of_inputs=num_of_inputs, input_range=input_range, random_angle_init=False)
        
        xlim = [np.min(traj_pca[...,0]), np.max(traj_pca[...,0])]
        ylim = [np.min(traj_pca[...,1]), np.max(traj_pca[...,1])]
        zlim = [np.min(traj_pca[...,2]), np.max(traj_pca[...,2])]
        
    rcParams["figure.dpi"] = 500
    plt.rcParams["figure.figsize"] = (5,5)
    xylims=[-1.5,1.5]
    makedirs(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +'/inputdriven3d_asymp')
    makedirs(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +'/output_asymp')
    if training_kwargs['nonlinearity'] == 'relu' and training_kwargs['N_rec'] <= 25:
        makedirs(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +'/inputdriven3d_analytic')
        makedirs(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +'/output_analytic')
        
    # for which in range(500, 5500, 25):
    for which in range(100, 500, 5):
    # for which in range(0, 100, 1):

        params_folder = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name
        # training_kwargs['map_output_to_hidden'] = False
        wi, wrec, wo, brec, h0, oth, training_kwargs = get_params_exp(params_folder, exp_i, which)

        # net =  load_net_from_weights(wi, wrec, wo, brec, h0, oth, training_kwargs)
        # exp = exp_list[exp_i]   
        
        trajectories, traj_pca, start, target, output, input_proj, _, explained_variance = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs, oth,
                                                                                            T=T, input_length=input_length, which=which,
                                                                                            pca_from_to=pca_from_to,
          num_of_inputs=num_of_inputs, input_range=input_range, random_angle_init=False)
        traj_pca = pca.transform(trajectories.reshape((-1,training_kwargs['N_rec']))).reshape((num_of_inputs,-1,10))
        # plot_input_driven_trajectory_2d(trajectories, traj_pca, wo, ax=None)
        # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/inputdriven2d_{which}_{exp_i}.pdf', bbox_inches="tight")
        
        # plot_input_driven_trajectory_2d(trajectories, traj_pca, wo, plot_asymp=True, ax=None);
        # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/inputdriven2d_asymp_{which}_{exp_i}.pdf', bbox_inches="tight")
        
        # plot_input_driven_trajectory_3d(traj_pca, input_length, elev=45, azim=135)
        # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/inputdriven3d_{which}_{exp_i}.pdf', bbox_inches="tight")
        
        recurrences, recurrences_pca = find_periodic_orbits(trajectories, traj_pca, limcyctol=1e-2, mindtol=1e-4)
        plot_input_driven_trajectory_3d(traj_pca, input_length, plot_traj=True,
                                            recurrences=recurrences, recurrences_pca=recurrences_pca, wo=wo,
                                            elev=45, azim=135, lims=[xlim, ylim, zlim], plot_epoch=which)
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/inputdriven3d_asymp/{which:04d}_{exp_i}.png', bbox_inches="tight")
        plt.close()
        
        plot_output_trajectory(trajectories, wo, input_length, plot_traj=True,
                                   fxd_points=None, ops_fxd_points=None,
                                   plot_asymp=True, limcyctol=1e-2, mindtol=1e-4, ax=None, xylims=xylims, plot_epoch=which)
        plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/output_asymp/{which:04d}_{exp_i}.png', bbox_inches="tight")
        plt.close()

        # plot_output_trajectory(trajectories, wo, input_length, plot_traj=False,
        #                            fxd_points=None, ops_fxd_points=None,
        #                            plot_asymp=True, limcyctol=1e-2, mindtol=1e-4, ax=None)
        # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/output_asymp_notraj_{which}_{exp_i}.pdf', bbox_inches="tight")

        if training_kwargs['nonlinearity'] == 'relu' and training_kwargs['N_rec'] <= 25:
        
            fixed_point_list, stabilist, unstabledimensions, eigenvalues_list = find_analytic_fixed_points(wrec, brec)
            
            plot_input_driven_trajectory_3d(traj_pca, input_length, plot_traj=True,
                                                recurrences=recurrences, recurrences_pca=recurrences_pca, wo=wo,
                                                fxd_points=fixed_point_list,
                                                h_stabilities=unstabledimensions,
                                                elev=45, azim=135, lims=[xlim, ylim, zlim], plot_epoch=which)
            plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/inputdriven3d_analytic/{which:04d}_{exp_i}.png', bbox_inches="tight")
            plt.close()

            
            plot_output_trajectory(trajectories[:,:,:], wo, input_length, plot_traj=True,
                                       fxd_points=fixed_point_list,
                                       h_stabilities=unstabledimensions,
                                       plot_asymp=True, limcyctol=1e-2, mindtol=1e-4, ax=None, xylims=xylims, plot_epoch=which)
            plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/output_analytic/{which:04d}_{exp_i}.png', bbox_inches="tight")
            plt.close()
        
        if False:
            traj = trajectories[:,:input_length:20,:].reshape((-1,training_kwargs['N_rec']));
            slowpointtol=1e-27
            fxd_points, speeds = find_slow_points(wrec, brec, dt=training_kwargs['dt_rnn'], trajectory=traj, tol=slowpointtol, method=slowpointmethod)
            
            ops_fxd_points, ops_speeds = find_slow_points(wrec, brec, wo=wo, dt=training_kwargs['dt_rnn'], trajectory=traj,
                                                      outputspace=True, tol=slowpointtol, method=slowpointmethod,
                                                      nonlinearity=training_kwargs['nonlinearity'])

    
            h_stabilities=get_stabilities(fxd_points, wrec, brec, tau=1/training_kwargs['dt_rnn'])
            h_stabilities = np.where(np.array(h_stabilities)>2,2,h_stabilities)
            plot_input_driven_trajectory_2d(trajectories, traj_pca, wo, fxd_points=fxd_points, 
                                            ops_fxd_points=ops_fxd_points,
                                            h_stabilities=h_stabilities, plot_asymp=True, ax=None);
            plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/inputdriven2d_asymp_slow_{which}_{exp_i}.pdf', bbox_inches="tight")
            
            plot_output_trajectory(trajectories, wo, input_length, plot_traj=True,
                                       fxd_points=fxd_points, ops_fxd_points=ops_fxd_points,
                                       h_stabilities=h_stabilities,
                                       plot_asymp=True, limcyctol=1e-2, mindtol=1e-4, ax=None)
            plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/output_slow_{which}_{exp_i}.pdf', bbox_inches="tight")
    
            plot_output_trajectory(trajectories, wo, input_length, plot_traj=False,
                                       fxd_points=fxd_points, ops_fxd_points=ops_fxd_points,
                                       h_stabilities=h_stabilities,
                                       plot_asymp=True, limcyctol=1e-2, mindtol=1e-4, ax=None)
            plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/output_slow_notraj_{which}_{exp_i}.pdf', bbox_inches="tight")

        # plot_trajs_model(main_exp_name, model_name, exp_i, T=T, which='post', input_length=input_length,
        #                           timepart='all', num_of_inputs=num_of_inputs,
        #                           plot_from_to=plot_from_to, pca_from_to=pca_from_to,
        #                           input_range=input_range, slowpointtol=1e-15, slowpointmethod=slowpointmethod)
        
        # #Vector field plotting
        # params_folder = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name
        # wi, wrec, wo, brec, h0, training_kwargs = get_params_exp(params_folder)

            #Both directions
            x_lim = 1.4
            num_x_points = 21
            num_of_inputs = 11
            # input_range = (-.25,.25)
            # trajectories, traj_pca, start, target, output, input_proj, pca, explained_variance = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs,
            #                                                                                   T=2*input_length, input_length=input_length, which=which,
            #                                                                                   pca_from_to=pca_from_to,
            #                                                                                   num_of_inputs=num_of_inputs, input_range=input_range)
            # plot_average_vf(trajectories[:,input_length:,:], wi, wrec, brec, wo, exp_i, input_length=0, 
            #                 num_of_inputs=num_of_inputs, input_range=input_range, x_lim=x_lim,
            #                 num_x_points=num_x_points)
            # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/vf_noinput_{which}_{exp_i}.pdf', bbox_inches="tight")
            # plt.show()
            
            #Single direction
            fig, ax = plt.subplots(1, 1, figsize=(3, 3));
            input_range = (.5, .51)
            num_of_inputs = 1
            trajectories, traj_pca, start, target, output, input_proj, pca, explained_variance = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs,
                                                                                              T=input_length, input_length=input_length, which=which,
                                                                                              pca_from_to=pca_from_to,
                                                                                              num_of_inputs=num_of_inputs, input_range=input_range)
            plot_average_vf(trajectories, wi, wrec, brec, wo, input_length=T, 
                            num_of_inputs=num_of_inputs, input_range=input_range, x_lim=x_lim,
                            num_x_points=num_x_points, color='blue', ax=ax)
            
            input_range = (-.51, -.5)
            num_of_inputs = 1
            trajectories, traj_pca, start, target, output, input_proj, pca, explained_variance = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs,
                                                                                              T=input_length, input_length=input_length, which=which,
                                                                                              pca_from_to=pca_from_to,
                                                                                              num_of_inputs=num_of_inputs, input_range=input_range)
            plot_average_vf(trajectories, wi, wrec, brec, wo, input_length=T, 
                            num_of_inputs=num_of_inputs, input_range=input_range, x_lim=x_lim,
                            num_x_points=num_x_points, color='red', ax=ax)
            # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/vf_onedir_{which}_{exp_i}.pdf', bbox_inches="tight")
            plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f'/vf_both_{which}_{exp_i}.pdf', bbox_inches="tight")
            plt.show()
    
        # fig, ax = plt.subplots(1, 1, figsize=(3, 3));
        # # for i in range(10):
        # plot_output_vs_target(target[0,...], output[0,...], ax=ax)
        # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+ model_name +f"/output_vs_target_{exp_i}.pdf")

    #plot target vs output
    # plot_output_vs_target(target[0,...], output[0,...])

    
    # for trial_i in range(trajectories.shape[0]):
    #     target_angle = np.arctan2(output[trial_i,-1,1], output[trial_i,-1,0])

    #     ax.plot(traj_pca[trial_i,after_t:before_t,0], traj_pca[trial_i,after_t:before_t,1], '-', color=cmap2(norm2(target_angle)))



    # for hidden_i in range(20):
    #     plot_all_trajs_model(main_exp_name, model_name='high', T=4096, which='post', hidden_i=hidden_i)
        # plot_trajs_model(main_exp_name, 'ortho', which='post', T=10, from_t_step=0, mean_color='b', trial_color='b', label='', ax=None, save=True, hidden_i=i)
    
    # plot_allLEs(main_exp_name,  mean_colors, trial_colors, labels, T=T, from_t_step=from_t_step, first_or_last='first')
    # plot_allLEs(main_exp_name,  mean_colors, trial_colors, labels, T=T, from_t_step=from_t_step, first_or_last='last')

    # task = simplestep_integration_task(T=10, dt=.1, amplitude=-1, pulse_time=1.5, delay=2)    
    # df = load_best_model_from_grid("angularintegration/adam2", "low", task=task)
    
    # task = angularintegration_task(T=10, dt=.1, sparsity=0.)
    # main_exp_name='angularintegration/lambda_T2/12.8'
    # plot_participatioratio(main_exp_name, model_name, task, first_or_last='last')

    # main_exp_name='poisson_clicks/relu_mse'
    # model_name = "high"
    # task = poisson_clicks_task(T=20, dt=.1)
    # plot_losses_and_trajectories(plot_losses_or_trajectories='trajectories', main_exp_name=main_exp_name, model_name=model_name)
    # plot_losses_and_trajectories(plot_losses_or_trajectories='losses', main_exp_name=main_exp_name, model_name=model_name)
    # plot_losses_experiment(plot_losses_or_trajectories='losses',
    #                              main_exp_name='angularintegration', model_name=model_name, training_label='weight_decay')

# plot_losses_and_trajectories(plot_losses_or_trajectories='losses', main_exp_name='angularintegration', model_name='high_gain')
    
    # model_names=['lstm', 'low','high', 'ortho', 'qpta']
    # i, model_names = 3, ['ortho']

    # plot_all(model_names=model_names,                   main_exp_name='angularintegration/lambda_T2/12.8')
    # for sparsity in [0.01, 0.1, 1.]:
    #     plot_bestmodel_traj(model_names=model_names, main_exp_name=main_exp_name, sparsity=sparsity, i=i)
    
    # experiment_folder = 'lambda_grid_2'
    # for model_name in model_names:
    #     load_best_model_from_grid(experiment_folder, model_name)
