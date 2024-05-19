# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:44:55 2024

@author: abel_
"""


import os, sys
import glob
import pickle
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir) 
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

from models import RNN 
from analysis_functions import simulate_rnn_with_input, simulate_rnn_with_task
from tasks import angularintegration_task

# =============================================================================
# TODO
# =============================================================================
#convergence of trajectories?

def load_net_from_weights(wi, wrec, wo, brec, h0, oth, training_kwargs):

    if oth is None:
        training_kwargs['map_output_to_hidden'] = False

    dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
    net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
              nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
              wi_init=wi, wrec_init=wrec, wo_init=wo, brec_init=brec, h0_init=h0, oth_init=oth,
              ML_RNN=training_kwargs['ml_rnn'], 
              map_output_to_hidden=training_kwargs['map_output_to_hidden'], input_nonlinearity=training_kwargs['input_nonlinearity'])
    return net

def load_net(path, which='post'):
    #main_exp_name='center_out/act_reg_gui/'
    #folder = parent_dir+"/experiments/" + main_exp_name
    #exp_list = glob.glob(folder + "/res*")
    #exp = exp_list[exp_i]
    with open(path, 'rb') as handle:
        result = pickle.load(handle)
    if which=='post':
        try:
            wi, wrec, wo, brec, oth, h0 = result['weights_last']
        except:
            oth = None
            wi, wrec, wo, brec, h0 = result['weights_last']

    elif which=='pre':
        try:
            wi, wrec, wo, brec, oth, h0 = result['weights_init']
        except:
            oth = None
            wi, wrec, wo, brec, h0 = result['weights_init']
    
    h0 = h0[0,:]
    net = load_net_from_weights(wi, wrec, wo, brec, h0, oth, result['training_kwargs'])

    return net, result


def plot_losses(folder):
    paths = glob.glob(folder + "/result*")
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 3));
    all_losses = np.empty((100,5000))
    all_losses[:] = np.nan
    for exp_i, path in tqdm(enumerate(paths)):
        net, result = load_net(path)
        losses = result['losses']
        ax.plot(losses[:np.argmin(losses)], 'b', alpha=.05)
        all_losses[exp_i, :np.argmin(losses)-1] = losses[:np.argmin(losses)-1]
    ax.plot(np.nanmean(all_losses,axis=0), 'b', zorder=1000, label='tanh')



def get_manifold_from_closest_projections(trajectories, wo, npoints=128):
    n_rec = wo.shape[0]
    xs = np.arange(-np.pi, np.pi, 2*np.pi/npoints)
    xs = np.append(xs, -np.pi)
    trajectories_flat = trajectories.reshape((-1,n_rec));
    ys = np.dot(trajectories_flat.reshape((-1,n_rec)), wo)
    circle_points = np.array([np.cos(xs), np.sin(xs)]).T
    dists = cdist(circle_points.reshape((-1,2)), ys)
    csx2 = []
    for i in range(xs.shape[0]):
        csx2.append(trajectories_flat[np.argmin(dists[i,:]),:])
    csx2 = np.array(csx2)
    csx2_proj2 = np.dot(csx2, wo)
    return xs, csx2, csx2_proj2



# task = angularintegration_task(T=T, dt=dt, sparsity=1, random_angle_init=False)]
def angular_loss_angint(net, task, h_init, T, batch_size=128, dt=0.1, random_seed=100, noise_std=0.):
    
    if h_init!='random':
        assert h_init.shape[0]==batch_size, "h_init must have same number of points as batch_size"
    
    np.random.seed(random_seed)
    net.noise_std = noise_std
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, h_init=h_init, batch_size=batch_size)
    output_angle = np.arctan2(output[:,:,1], output[:,:,0]);
    target_angle = np.arctan2(target[:,:,1], target[:,:,0]);
    angle_error = np.abs(output_angle - target_angle)
    angle_error[np.where(angle_error>np.pi)] = 2*np.pi-angle_error[np.where(angle_error>np.pi)]
    mean_error = np.mean(angle_error,axis=0)
    max_error = np.max(angle_error,axis=0)
    
    return mean_error, max_error


def angular_loss_noinput(net, angle_init, h_init, T, batch_size=128, dt=0.1, random_seed=100, noise_std=0.):
    
    if h_init!='random':
        assert h_init.shape[0]==batch_size, "h_init must have same number of points as batch_size"
    
    np.random.seed(random_seed)
    net.noise_std = noise_std
    input = np.zeros((batch_size, T, net.dims[0]))
    output, trajectories = simulate_rnn_with_input(net, input, h_init=h_init)
    output_angle = np.arctan2(output[:,:,1], output[:,:,0]);
    target_angle = np.arctan2(angle_init[:,1], angle_init[:,0]);
    angle_error = np.abs(output_angle - target_angle[:, np.newaxis])
    angle_error[np.where(angle_error>np.pi)] = 2*np.pi-angle_error[np.where(angle_error>np.pi)]
    mean_error = np.mean(angle_error,axis=0)
    max_error = np.max(angle_error,axis=0)
    min_error = np.min(angle_error,axis=0)
    return mean_error, max_error, min_error


def angular_loss_with_input():
    task = angularintegration_task(T=10, dt=dt, sparsity=1, random_angle_init=False)
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, 100, h_init='random', batch_size=batch_size)
    xs, csx2, csx2_proj2 = get_manifold_from_closest_projections(trajectories, net.wo.detach().numpy(), npoints=batch_size)

    T1 = result['training_kwargs']['T']/result['training_kwargs']['dt_rnn'];
    T=int(16*T1)

    net.map_output_to_hidden = False
    net.noise_std = 0.
    anlge_init = xs[:-1]
    input = np.zeros((batch_size, T, 1))
    stim = np.linspace(-.1, .1, num=batch_size, endpoint=True)
    input[:,:T,0] = np.repeat(stim,T).reshape((batch_size,T))
    output, trajectories = simulate_rnn_with_input(net, input, h_init=csx2[:-1,:])
    outputs_1d = np.cumsum(input, axis=1)*result['training_kwargs']['dt_task']
    output_angle = np.arctan2(output[:,:,1], output[:,:,0]);
    target_angle = xs[:-1][:, np.newaxis] + outputs_1d.squeeze()
    target_angle = (target_angle + np.pi) % (2 * np.pi) - np.pi
    angle_error = np.abs(output_angle - target_angle)
    angle_error[np.where(angle_error>np.pi)] = 2*np.pi-angle_error[np.where(angle_error>np.pi)]

    mean_error = np.mean(angle_error,axis=0)
    max_error = np.max(angle_error,axis=0)
    min_error = np.min(angle_error,axis=0)


def angle_analysis_on_net(net, T, input_or_task='input',
                          batch_size=128,
                          dt=0.1, 
                          T_inv_man=1e2):
    
    task = angularintegration_task(T=T_inv_man, dt=dt, sparsity=1, random_angle_init=False)
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T_inv_man, h_init='random', batch_size=batch_size)
    xs, csx2, csx2_proj2 = get_manifold_from_closest_projections(trajectories, net.wo.detach().numpy(), npoints=batch_size)

    net.map_output_to_hidden = False

    #T1 = result['training_kwargs']['T']/result['training_kwargs']['dt_rnn'];
    #T=int(16*T1)
    task = angularintegration_task(T=T, dt=dt, sparsity=1, random_angle_init=False)
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, h_init='random', batch_size=batch_size)
    output_angle = np.arctan2(output[:,:,1], output[:,:,0]);
    target_angle = np.arctan2(target[:,:,1], target[:,:,0]);
    angle_error = np.abs(output_angle - target_angle)
    angle_error[np.where(angle_error>np.pi)] = 2*np.pi-angle_error[np.where(angle_error>np.pi)]
    
    mean_error = np.mean(angle_error,axis=0)
    max_error = np.max(angle_error,axis=0)
    min_error = np.min(angle_error,axis=0)

    eps_mean_int = np.cumsum(mean_error) / np.arange(mean_error.shape[0])
    eps_plus_int = np.cumsum(max_error) / np.arange(mean_error.shape[0])
    eps_min_int = np.cumsum(min_error) / np.arange(mean_error.shape[0])
    
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3));
    # plt.plot(eps_plus_int[:int(T/2)], color='r', label='$\epsilon^+$');
    # plt.plot(eps_min_int[:int(T/2)], color='g', label='$\epsilon^-$');
    # plt.plot(eps_mean_int[:int(T/2)], color='b', label='$\int_0^T\epsilon^{mean}(t)dt$'); 
    # plt.plot(8.5*T1, eps_plus_int[-1], '.', color='r');
    # plt.plot(8.5*T1, eps_min_int[-1], '.', color='g');
    # plt.plot(8.5*T1, eps_mean_int[-1], '.', color='b');

    # ax.plot(np.arange(0,int(T/2),1), np.arange(0,int(T/2),1)*max_error[0])
    # ax.set_ylim([-.1,1.2*np.pi/2.])

    # T1 = result['training_kwargs']['T']/result['training_kwargs']['dt_rnn']
    # ax.axvline(T1, linestyle='--', color='r')
    # #ax.text(T1*1.15, .8, r'$T_1$',color='r')
    # ax.set_xticks(np.arange(0,9*T1,T1),[0,'$T_1$']+[f'${i}T_1$' for i in range(2,9)])
    # plt.legend(); plt.ylabel("loss"); plt.xlabel("t");
    # #fig.savefig(folder+"/angle_error.pdf", bbox_inches="tight");
    
    
    
    

