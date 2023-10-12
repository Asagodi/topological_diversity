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


import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.colors as mplcolors
import matplotlib.cm as cmx
import matplotlib as mpl

from models import RNN, run_net, LSTM_noforget, run_lstm
from tasks import angularintegration_task, angularintegration_delta_task, simplestep_integration_task, poisson_clicks_task
from analysis_functions import calculate_lyapunov_spectrum, tanh_jacobian, participation_ratio

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
        
def load_net_from_weights(weights, dt=.1):
    wi_init, wrec_init, wo_init, brec_init, h0_init = weights
    dims=(wi_init.shape[0],wi_init.shape[1],wo_init.shape[1])
    net = RNN(dims=dims, dt=dt,
              wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, h0_init=h0_init)
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


def plot_losses_and_trajectories(plot_losses_or_trajectories='losses', main_exp_name='angularintegration', model_name='low_gain', sparsity=.1, task=None):
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

    # task = angularintegration_task(T=training_kwargs['T'], dt=training_kwargs['dt_task'])
    task = angularintegration_task(T=training_kwargs['T'], dt=training_kwargs['dt_task'], sparsity=sparsity)
# 
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax2 = ax.twinx()
    for model_i, model_name in enumerate(model_names):
        print(model_name)
        params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_names[model_i] + '/param*.yml')[0]
        training_kwargs = yaml.safe_load(Path(params_path).read_text())
        training_kwargs['network_type'] = network_types[model_i]
        final_losses = plot_losses_and_trajectories(plot_losses_or_trajectories=None, main_exp_name=main_exp_name, model_name=model_name)[:,-1]
        minidx = np.nanargmin(final_losses)

        bestexp = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")[minidx]
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
                         plotpca=True, timepart='all', num_of_inputs=51, plot_from_to=(0,None), pca_from_to=(0,None),
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
        trajectories, start, target, output = get_hidden_trajs(main_exp_name, model_name, exp, T=T, which=which, hidden_i=hidden_i, input_length=input_length,
                         plotpca=plotpca, timepart=timepart, num_of_inputs=num_of_inputs, plot_from_to=plot_from_to, pca_from_to=pca_from_to, input_range=input_range)

        for trial_i in range(trajectories.shape[0]):
            axes1[exp_i].plot(trajectories[trial_i,before_t:after_t,0], trajectories[trial_i,before_t:after_t,1], '-', c=cmap(norm[trial_i]))
            if np.linalg.norm(trajectories[trial_i,-2,:]-trajectories[trial_i,-1,:])  < 1e-4:
                axes1[exp_i].scatter(trajectories[trial_i,-1,0], trajectories[trial_i,-1,1], marker='.', s=100, color=cmap(norm[trial_i]), zorder=100)

        axes1[exp_i].set_axis_off()
        axes1[exp_i].scatter(start[0], start[1], marker='.', s=100, color='k', zorder=100)
        
        if plot_output:
            x = np.linspace(0, 2*np.pi, 1000)
            axes2[exp_i].plot(np.cos(x), np.sin(x), 'k', alpha=.5, linewidth=5, zorder=-1)
            for trial_i in range(output.shape[0]):
                # if trial_i<output.shape[0]-1:
                #     axes2[exp_i].plot([target[trial_i,-1,0], target[trial_i+1,-1,0]], [target[trial_i,-1,1], target[trial_i+1,-1,1]], '--', c=cmap(norm[trial_i]), alpha=.5)
                # axes2[exp_i].scatter(target[trial_i,-1,0], target[trial_i,-1,1], color=cmap(norm[trial_i]), alpha=.1)
                # axes2[exp_i].plot(output[trial_i,:,0], output[trial_i,:,1], '-', c=cmap(norm[trial_i]))
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
            
            

    

def plot_trajs_model(main_exp_name, model_name, exp, T=128, which='post',  hidden_i=0, input_length=10,
                     plotpca=True, timepart='all',  num_of_inputs=51, plot_from_to=(0,None), pca_from_to=(0,None), input_range=(-3,3), axes=None):
    pca_before_t, pca_after_t = pca_from_to
    after_t, before_t = plot_from_to
    norm = mplcolors.Normalize(vmin=-.5,vmax=.5)
    norm = norm(np.linspace(-.5, .5, num=num_of_inputs, endpoint=True))
    cmap = cmx.get_cmap("coolwarm")
    norm2 = mpl.colors.Normalize(-np.pi, np.pi)
    # norm2 = norm2(np.linspace(-np.pi, np.pi, endpoint=True))
    cmap2 = plt.get_cmap('hsv')
    
    
    # print(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')
    params_path = glob.glob(parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    # exp = parent_dir+'/experiments/' + main_exp_name +'/'+ model_name + '/results_2023-08-05-13-02-03.pickle'
    with open(exp, 'rb') as handle:
        result = pickle.load(handle)
    
    try:
        losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs = result
    except:
        losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result

    if which=='post':
        wi, wrec, wo, brec, h0 = weights_last
    elif which=='pre':
        wi, wrec, wo, brec, h0 = weights_init
    
    trajectories, traj_pca, start, target, output, input_proj, pca = get_hidden_trajs(wi, wrec, wo, brec, h0, training_kwargs,
                                                                       T=T, which=which, hidden_i=hidden_i, input_length=input_length,
                     plotpca=plotpca, timepart=timepart, num_of_inputs=num_of_inputs, plot_from_to=plot_from_to, pca_from_to=pca_from_to,
                     input_range=input_range)
    print("F",trajectories.shape)

    if not axes:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=False, sharey=False)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=False, sharey=False)
        fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=False, sharey=False)

    for trial_i in range(trajectories.shape[0]):
        target_angle = np.arctan2(output[trial_i,-1,1], output[trial_i,-1,0])

        # axes[0].plot(trajectories[trial_i,after_t:before_t,0], trajectories[trial_i,after_t:before_t,1], '-', c=cmap(norm[trial_i]))
        axes[0].plot(traj_pca[trial_i,after_t:before_t,0], traj_pca[trial_i,after_t:before_t,1], '-', c=cmap2(norm2(target_angle)))

        if np.linalg.norm(trajectories[trial_i,-5,:]-trajectories[trial_i,-1,:])  < 1e-6:
            axes[0].scatter(traj_pca[trial_i,-1,0], traj_pca[trial_i,-1,1], marker='.', s=100, c=cmap2(norm2(target_angle)), zorder=100)

    axes[0].set_axis_off()
    axes[0].scatter(start[0], start[1], marker='.', s=100, color='k', zorder=100)
    
    x = np.linspace(-np.pi, np.pi, 1000)
    # axes[1].plot(np.cos(x), np.sin(x), 'k', alpha=.5, linewidth=5, zorder=-1)
    axes[1].scatter(np.cos(x), np.sin(x), c=cmap2(norm2(x)), alpha=.5, s=2, zorder=-1)

    for trial_i in range(output.shape[0]):
        target_angle = np.arctan2(output[trial_i,-1,1], output[trial_i,-1,0])
        axes[1].plot(output[trial_i,0,0], output[trial_i,0,1], '.', c='k', zorder=100)
        axes[1].plot(output[trial_i,:,0], output[trial_i,:,1], '-', alpha=.25)
        axes[1].plot(output[trial_i,after_t:before_t,0], output[trial_i,after_t:before_t,1], '-', c=cmap2(norm2(target_angle)))

    # for trial_i in range(output.shape[0]):
    #     axes[1].plot(output[trial_i,after_t:before_t,0], output[trial_i,after_t:before_t,1], '-', c=cmap(norm[trial_i]))

        if np.linalg.norm(trajectories[trial_i,-5,:]-trajectories[trial_i,-1,:])  < 1e-6:
            axes[1].scatter(output[trial_i,-1,0], output[trial_i,-1,1], marker='.', s=100, color=cmap2(norm2(target_angle)), zorder=100)
        axes[2].plot(input_proj[trial_i,after_t:before_t,0], '-', c=cmap2(norm2(target_angle)))

    #     for t in range(output.shape[1]):
    #         axes[1].plot([target[trial_i,t,0], output[trial_i,t,0]],
    #                          [target[trial_i,t,1], output[trial_i,t,1]], '-', 
                             # color=cmap2(norm2(np.arctan2(target[trial_i,t,1], target[trial_i,t,0]))))

        
    x_lim = 1.2 #np.max(np.abs(output))
    num_x_points = 21
    output_logspeeds, all_logspeeds = average_logspeed(wrec, wo, brec, trajectories[:,128:,:], x_min=-x_lim, x_max=x_lim, num_x_points=num_x_points)
    im=axes[3].imshow(output_logspeeds, cmap='inferno')
    cbar = fig.colorbar(im, ax=axes[3])
    cbar.set_label("log(speed)")
    
    # Create a regular grid
    x_min=-2
    x_max=2
    num_x_points=11
    x_values = np.linspace(x_min, x_max, num_x_points)
    y_values = np.linspace(x_min, x_max, num_x_points)
    
    # Generate all grid points
    grid_points = np.array([(np.round(x,5), np.round(y,5)) for x in x_values for y in y_values])
    average_ivf = np.zeros((num_x_points, num_x_points, 2))
    for I, trajectory in zip(input_range, trajectories[:,:input_length,:]):
        average_ivf += average_input_vectorfield(wi, wrec, brec, wo, I, trajectory, pca, x_min=-x_lim, x_max=x_lim, num_x_points=num_x_points)
        
    average_ivf /= num_of_inputs
    print(grid_points.shape, average_ivf.shape)
    axes[4].quiver(grid_points[:,0], grid_points[:,1],
                   average_ivf.reshape((num_x_points**2,2))[:,0], average_ivf.reshape((num_x_points**2,2))[:,1])
    
    axes[1].set_axis_off()
    axes[2].set_xticks([])
    axes[3].set_axis_off()

    plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+f'/mss_output_{which}.pdf', bbox_inches="tight")

    
    
def get_hidden_trajs(wi_init, wrec_init, wo_init, brec_init, h0_init, training_kwargs, T=128, which='post',  hidden_i=0, input_length=10,
                     plotpca=True, timepart='all',  num_of_inputs=51, plot_from_to=(0,None), pca_from_to=(0,None), input_range=(-3,3)):
    pca_after_t, pca_before_t = pca_from_to
    after_t, before_t = plot_from_to

    dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
    net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
              nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
              wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, h0_init=h0_init, ML_RNN=training_kwargs['ml_rnn'])
    
    min_input, max_input = input_range
    norm = mplcolors.Normalize(vmin=min_input, vmax=max_input)
    norm = norm(np.linspace(min_input, max_input, num=num_of_inputs, endpoint=True))
    cmap = cmx.get_cmap("coolwarm")
    
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    
    input = np.zeros((num_of_inputs,T,training_kwargs['N_in']))
    # stim = np.zeros((num_of_inputs,T))
    stim = np.linspace(min_input, max_input, num=num_of_inputs, endpoint=True)
    # stim = np.linspace(0, 5, num=num_of_inputs, endpoint=True)
    input[:,:input_length,0] = np.repeat(stim,input_length).reshape((num_of_inputs,input_length))
    # input[:,:10,:] = np.repeat(np.linspace(-.5, .5, num=num_of_inputs, endpoint=True),10).reshape((num_of_inputs,10,training_kwargs['N_in']))
    input = torch.from_numpy(input).float() 
    
    outputs_1d = np.cumsum(input, axis=1)*training_kwargs['dt_task']
    target = np.stack((np.cos(outputs_1d), np.sin(outputs_1d)), axis=-1).reshape((num_of_inputs, T, training_kwargs['N_out']))
    h_init = h0_init # np.zeros((num_of_inputs,training_kwargs['N_rec'])) 

    with torch.no_grad():
        output, trajectories = net(input, return_dynamics=True, h_init=h_init)
    
    print(trajectories.shape, wi_init.shape)
    input_proj = np.dot(trajectories, wi_init.T)
    if not plotpca:
        start=trajectories[0,0,:]
        for trial_i in range(trajectories.shape[0]):
            if timepart=='all' or not timepart:
                traj = trajectories[:,:,:]
            else:
                traj = trajectories[:,after_t:before_t,:]
    else:
        pca = PCA(n_components=2)   

        trajectories_tofit = trajectories[:,pca_after_t:pca_before_t,:].numpy().reshape((-1,training_kwargs['N_rec']))
        pca.fit(trajectories_tofit)
        traj_pca = pca.transform(trajectories.numpy().reshape((-1,training_kwargs['N_rec']))).reshape((num_of_inputs,-1,2))

        start=traj_pca[0,0,:]
        for trial_i in range(trajectories.shape[0]):
            if timepart=='all' or not timepart:
                traj_pca = traj_pca[:,:,:]
            else:
                traj_pca = traj_pca[:,after_t:before_t,:]
        

    # ax.set_axis_off()
    makedirs(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+'/hidden'+exp[-21:-7])
    # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+'/hidden'+exp[-21:-7]+f'/trajpca_{which}_{timepart}_{after_t}to{before_t}.pdf', bbox_inches="tight")
    # plt.savefig(parent_dir+'/experiments/'+main_exp_name+'/'+model_name+'/hidden'+exp[-21:-7]+f'/trajpca_{which}_{timepart}_{after_t}to{before_t}.png', bbox_inches="tight")
    # plt.close()

    return trajectories, traj_pca, start, target, output, input_proj, pca



def average_input_vectorfield(wi, wrec, brec, wo, I, trajectories, pca, x_min=-2, x_max=2, num_x_points=11):
    """
    Calculates average over x for  g(x,I) = tanh(Wx+b+W_inI) - tanh(Wx+b)
    for a fixed I

    """
    # Create a regular grid
    x_values = np.linspace(x_min, x_max, num_x_points)
    y_values = np.linspace(x_min, x_max, num_x_points)
    
    # Generate all grid points
    grid_points = [(np.round(x,5), np.round(y,5)) for x in x_values for y in y_values]
    average_vector_field = np.zeros((num_x_points, num_x_points, 2))
    bin_counts = np.zeros((num_x_points, num_x_points))
     
    for x in trajectories:
        target_point = np.dot(wo.T, x)
        cell_containing_point = find_grid_cell(target_point, x_values, y_values, num_x_points=num_x_points)
        full_vf_at_x = input_vectorfield(x, wi, wrec, brec, I)
        
        average_vector_field[cell_containing_point] += pca.transform(full_vf_at_x).squeeze()
    return average_vector_field

def input_vectorfield(x, wi, wrec, brec, I):
    """
    Calculates g(x, I) in the split RNN equation \dot x = f(x)+g(x,I)
    for \dot x = tanh(Wx+b+W_inI)
    i.e. g(x,I) = tanh(Wx+b+W_inI) - tanh(Wx+b)

    Parameters
    ----------
    wi_init : TYPE
        DESCRIPTION.
    wrec_init : TYPE
        DESCRIPTION.
    wo_init : TYPE
        DESCRIPTION.
    brec_init : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    dotx = np.tanh(np.dot(wrec, x)+brec+np.dot(wi,I))
    fx = np.tanh(np.dot(wrec, x)+brec+np.dot(wi,I))

    return dotx- fx



def ReLU(x):
    return np.where(x<0,0,x)

def relu_ode(t,x,W):
    return ReLU(np.dot(W,x)) - x 

def tanh_ode(t,x,W):
    return np.tanh(np.dot(W,x)) - x 


# def speed(wrec_init):
    
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
    print("A", trajectories.shape[0], trajectories.shape[1]-1)

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

    main_exp_name='angular_integration/hidden/51.2'
    # main_exp_name='angularintegration/hidden'
    # main_exp_name='angularintegration/all_mse/'

    # main_exp_name='poisson_clicks/relu_mse'
    model_name = 'high'

    T = 256*32
    num_of_inputs = 101
    input_range = (-.5,.5)
    input_length = int(T/64)
    # input_range = (-.1,.1)
    which='post'
    plot_from_to = (T-input_length,T)
    pca_from_to = (0,T)
    # for timepart in ['all', 'beginning', 'end']:
    #     plot_all_trajs_model(main_exp_name, model_name=model_name, T=T, which=which, plotpca=True, timepart=timepart, num_of_inputs=num_of_inputs, input_range=input_range)
    
    exp_list = glob.glob(parent_dir+"/experiments/" + main_exp_name +'/'+ model_name + "/result*")
    exp = exp_list[7]
    plot_trajs_model(main_exp_name, model_name, exp, T=T, which='post', hidden_i=0, input_length=input_length,
                              plotpca=True, timepart='all', num_of_inputs=num_of_inputs,
                              plot_from_to=plot_from_to, pca_from_to=pca_from_to,
                              input_range=input_range)
    
    # plot_all_trajs_model(main_exp_name, model_name, T=T, which='post', hidden_i=0, input_length=input_length,
    #                          plotpca=True, timepart='all', num_of_inputs=num_of_inputs,
    #                          plot_from_to=plot_from_to, pca_from_to=pca_from_to,
    #                          plot_output=True, input_range=input_range)
    
    # plot_all_trajs_model(main_exp_name, model_name=model_name, T=T, which=which,
    #                       plotpca=False, num_of_inputs=num_of_inputs, timepart='end',
    #                       plot_output=True, input_range=input_range,
    #                        input_length=input_length, plot_from_to=plot_from_to)

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
