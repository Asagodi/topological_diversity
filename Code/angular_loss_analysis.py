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
import scipy
import skdim
from ripser import ripser
from persim import plot_diagrams

import matplotlib.pyplot as plt
import matplotlib as mpl

from models import RNN 
from analysis_functions import simulate_rnn_with_input, simulate_rnn_with_task
from tasks import angularintegration_task, angularintegration_task_constant, center_out_reaching_task
from odes import tanh_ode, recttanh_ode, relu_ode

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

def load_net_path(path, which='post'):
    # main_exp_name='center_out/act_reg_gui/'
    # folder = parent_dir+"/experiments/" + main_exp_name
    # exp_list = glob.glob(folder + "/res*")
    # exp = exp_list[exp_i]
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
    
    
    try:    
        net = load_net_from_weights(wi, wrec, wo, brec, h0, oth, result['training_kwargs'])
    except:
        h0 = h0[0,:]
        net = load_net_from_weights(wi, wrec, wo, brec, h0, oth, result['training_kwargs'])

    return net, result

def get_rnn_ode(nonlinearity):
    if nonlinearity == 'tanh':
        return tanh_ode
    elif nonlinearity == 'relu':
        return relu_ode
    elif nonlinearity == 'recttanh':
        return recttanh_ode
    elif nonlinearity == 'softplus':
        return softplus_ode

def get_weights_from_net(net):
    wi = net.wi.detach().numpy()
    wrec = net.wrec.detach().numpy()
    brec = net.brec.detach().numpy()
    wo = net.wo.detach().numpy()
    try:
        oth = net.oth.detach().numpy()
    except:
        oth=None
    return wi, wrec, brec, wo, oth


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










##############ANGULAR
def get_manifold_from_closest_projections(trajectories_flat, wo, npoints=128):
    n_rec = wo.shape[0]
    xs = np.arange(-np.pi, np.pi, 2*np.pi/npoints)
    xs = np.append(xs, -np.pi)
    #trajectories_flat = trajectories.reshape((-1,n_rec));
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


def angular_loss_angvel_noinput(net, angle_init, h_init, T, batch_size=128, dt=0.1, random_seed=100, noise_std=0.):
    
    if h_init!='random':
        assert h_init.shape[0]==batch_size, "h_init must have same number of points as batch_size"
    
    np.random.seed(random_seed)
    net.noise_std = noise_std
    input = np.zeros((batch_size, T, net.dims[0]))
    output, trajectories = simulate_rnn_with_input(net, input, h_init=h_init)
    output_angle = np.arctan2(output[:,:,1], output[:,:,0]);
    #target_angle = np.arctan2(angle_init[:,1], angle_init[:,0]);
    angle_error = np.abs(output_angle - angle_init[:, np.newaxis])
    angle_error[np.where(angle_error>np.pi)] = 2*np.pi-angle_error[np.where(angle_error>np.pi)]
    mean_error = np.mean(angle_error,axis=0)
    max_error = np.max(angle_error,axis=0)
    min_error = np.min(angle_error,axis=0)
    
    eps_mean_int = np.cumsum(mean_error) / np.arange(1,mean_error.shape[0]+1)
    eps_plus_int = np.cumsum(max_error) / np.arange(1,mean_error.shape[0]+1)
    eps_min_int = np.cumsum(min_error) / np.arange(1,mean_error.shape[0]+1)
    
    return eps_min_int, eps_mean_int, eps_plus_int


def angular_loss_angvel_with_input():
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
    target_angle = anlge_init[:, np.newaxis] + outputs_1d.squeeze()
    target_angle = (target_angle + np.pi) % (2 * np.pi) - np.pi
    angle_error = np.abs(output_angle - target_angle)
    angle_error[np.where(angle_error>np.pi)] = 2*np.pi-angle_error[np.where(angle_error>np.pi)]

    mean_error = np.mean(angle_error,axis=0)
    max_error = np.max(angle_error,axis=0)
    min_error = np.min(angle_error,axis=0)
    
    eps_mean_int = np.cumsum(mean_error) / np.arange(mean_error.shape[0])
    eps_plus_int = np.cumsum(max_error) / np.arange(mean_error.shape[0])
    eps_min_int = np.cumsum(min_error) / np.arange(mean_error.shape[0])
    
    return eps_min_int, eps_mean_int, eps_plus_int


def angle_analysis_on_net(net, T, input_or_task='input',
                          batch_size=128,
                          dt=0.1, 
                          T_inv_man=1e2):
    np.random.seed(100)
    task = angularintegration_task(T=T_inv_man, dt=dt, sparsity=1, random_angle_init=False)
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T_inv_man, h_init='random', batch_size=batch_size)
    xs, csx2, csx2_proj2 = get_manifold_from_closest_projections(trajectories, net.wo.detach().numpy(), npoints=batch_size)

    net.map_output_to_hidden = False

    #T1 = result['training_kwargs']['T']/result['training_kwargs']['dt_rnn'];
    #T=int(16*T1)
    np.random.seed(100)
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
    
    
    
    
def angular_loss_msg(net, cue_range, T1_multiple=16, dt=.1, batch_size=128):
    t1_pos = 0 
    time_until_input = 5
    cue_duration = 5
    time_until_cue_range_min, time_until_cue_range_max = cue_range
    min_time = time_until_input + cue_duration + time_until_cue_range_min
    time_until_measured_response = 5
    time_after_response = 2
    add_t = time_until_measured_response+time_after_response
    T1 = time_until_cue_range_max #training_kwargs['time_until_cue_range'][1]
    T = T1*T1_multiple
    tstep=1
    timepoints=int(T)#-15-min_time_until_cue
    print(timepoints)
    
    tstep = 1 #int(T*dt_rnn//10)

    net.noise_std = 0
    net.map_output_to_hidden=False
    np.random.seed(100)
    
    T_tot = T/dt+min_time*10+10
    task = center_out_reaching_task(T=T_tot, dt=dt, time_until_cue_range=[T, T+1], angles_random=False);
    input, target, mask, output, trajectories_full = simulate_rnn_with_task(net, task, T_tot, h_init='random', batch_size=batch_size)
    target = input[:,time_until_input,:]
    target_angle = np.arctan2(target[:,1], target[:,0]);
    target_angle = (target_angle + np.pi) % (2 * np.pi) - np.pi

    angle_errors = np.zeros((batch_size, (timepoints+min_time)//tstep))
    ts = []
    for t in range(min_time,timepoints+min_time,tstep):
        ts.append(t)
        input = np.zeros((batch_size,cue_duration+add_t,3))
        input[:,:cue_duration,2]=1.
        h_init = trajectories_full[:,t,:]
        output, trajectories = simulate_rnn_with_input(net, input, h_init=h_init)
        #ax.plot(output[:,-1,0].T, output[:,-1,1].T, 'b')
        #ax.plot(target[:,0].T, target[:,1].T, 'r')
        output_angle = np.arctan2(output[:,-1,1], output[:,-1,0]);
        
        angle_error = np.abs(output_angle - target_angle)
        angle_error[np.where(angle_error>np.pi)] = 2*np.pi-angle_error[np.where(angle_error>np.pi)]
        mean_error = np.mean(angle_error,axis=0)
        angle_errors[:,(t-min_time)//tstep] = angle_error
        t1_pos=np.min([t1_pos,np.min(mean_error)])

    mean_error = np.mean(angle_errors,axis=0)
    max_error = np.max(angle_errors,axis=0)
    min_error = np.min(angle_errors,axis=0)
    
    eps_mean_int = np.cumsum(mean_error) / np.arange(1, mean_error.shape[0]+1)
    eps_plus_int = np.cumsum(max_error) / np.arange(1, mean_error.shape[0]+1)
    eps_min_int = np.cumsum(min_error) / np.arange(1, mean_error.shape[0]+1)
    
    return ts, eps_min_int, eps_mean_int, eps_plus_int


##########SPEED
def get_speed_and_acceleration(trajectory):
    speeds = []
    accelerations = []
    for t in range(trajectory.shape[0]-1):
        speed = np.linalg.norm(trajectory[t+1, :]-trajectory[t, :])
        speeds.append(speed)
    for t in range(trajectory.shape[0]-2):
        acceleration = np.abs(speeds[t+1]-speeds[t])
        accelerations.append(acceleration)
    speeds = np.array(speeds)
    accelerations = np.array(accelerations)
    return speeds, accelerations

def get_speed_and_acceleration_batch(trajectories):
    all_speeds = []
    all_accs = []
    for trajectory in trajectories:
        speeds, accelerations = get_speed_and_acceleration(trajectory)
        all_speeds.append(speeds)
        all_accs.append(accelerations)
    return np.array(all_speeds), np.array(all_accs)



###slow man
def detect_slow_manifold(trajectories, subsample=100, tol=1e-4):
    all_speeds, all_accs = get_speed_and_acceleration_batch(trajectories)
    idx = np.argmax(all_speeds - all_speeds[:,-1][:,np.newaxis] < tol,axis=1)
    inv_man = np.empty((0,trajectories.shape[-1]))
    for i,index in enumerate(idx):
        inv_man = np.vstack([inv_man, trajectories[i,index::subsample,:]])
        
    return inv_man


##########VFs
def vf_on_ring(trajectories_flat,  wo,  wrec, brec, cs, rnn_ode=tanh_ode,
               fxd_pnt_thetas=None, stabilities=None, method='closest', npoints=128, 
               fig_folder=None, fig_ext=''):
    
    X = []
    Y = []
    U = []
    V = []
    if method=='spline':
        xs = np.arange(-np.pi, np.pi, 2*np.pi/npoints)
        xs = np.append(xs, -np.pi)
        csx=cs(xs);
        # fig, ax = plt.subplots(1, 1, figsize=(3, 3)); 
        diff = np.zeros((csx.shape[0]))

        for i in range(csx.shape[0]):
            x,y=np.cos(xs[i]),np.sin(xs[i])
            x,y=np.dot(wo.T,csx[i])
            u,v=np.dot(wo.T, rnn_ode(0, csx[i], wrec, brec, tau=10))
            diff[i] = np.linalg.norm(np.array([u,v]))

            # plt.quiver(x,y,u,v)
            X.append(x)
            Y.append(y)
            U.append(u)
            V.append(v)

        return diff, np.array(X), np.array(Y), np.array(U), np.array(V)
        # for i,theta in enumerate(fxd_pnt_thetas):
        #     x,y=np.cos(theta),np.sin(theta)
        #     if stabilities[i]==1:
        #         plt.plot(x,y,'.g')
        #     else:
        #         plt.plot(x,y,'.r')
        # plt.axis('off'); fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None);
        # if fig_folder:
        #     plt.savefig(fig_folder+f'/vf_on_ring_{fig_ext}.pdf', bbox_inches="tight")
    
    elif method=='closest':
        #trajectories_flat = trajectories.reshape((-1,trajectories[-1]));
        xs, csx2, csx2_proj2 = get_manifold_from_closest_projections(trajectories_flat, wo, npoints=npoints)
    
        diff = np.zeros((csx2.shape[0]))
        for i in range(csx2.shape[0]):
            x,y=np.cos(xs[i]),np.sin(xs[i])
            x,y=np.dot(csx2[i], wo)
            u,v=np.dot(wo.T, rnn_ode(0, csx2[i], wrec, brec, tau=10))
            diff[i] = np.linalg.norm(np.array([u,v]))
            
            X.append(x)
            Y.append(y)
            U.append(u)
            V.append(v)

        return diff, np.array(X), np.array(Y), np.array(U), np.array(V)
    
    elif method=='points':
        diff = np.zeros((trajectories_flat.shape[0]))

        for i in range(trajectories_flat.shape[0]):
            x,y=np.dot(trajectories_flat[i],wo)
            #x,y=np.cos(xs[i]),np.sin(xs[i])
            #x,y=np.dot(wo.T,csx[i])
            u,v=np.dot(wo.T, rnn_ode(0, trajectories_flat[i], wrec, brec, tau=10))
            diff[i] = np.linalg.norm(np.array([u,v]))

            X.append(x)
            Y.append(y)
            U.append(u)
            V.append(v)
        return diff, np.array(X), np.array(Y), np.array(U), np.array(V)




def plot_vf_on_ring(X,Y,U,V,fxd_pnt_thetas=None,stabilities=None,fig_folder=None,fig_ext='',ax=None,fig=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3)); 
        plt.axis('off'); fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None);

    for x,y,u,v in zip(X,Y,U,V):
        plt.quiver(x,y,u,v)
        
    if np.any(fxd_pnt_thetas):
        for i,theta in enumerate(fxd_pnt_thetas):
            x,y=np.cos(theta),np.sin(theta)
            if stabilities[i]==1:
                plt.plot(x,y,'.g')
            else:
                plt.plot(x,y,'.r')
    if fig_folder:
        #     plt.savefig(fig_folder+f'/vf_on_ring_{fig_ext}.pdf', bbox_inches="tight")
        plt.savefig(fig_folder+f'/vf_on_ring_closest_{fig_ext}.pdf', bbox_inches="tight")
        
        
def detect_fixed_points_from_flow_on_ring(trajectories_proj2, cs=None):
    thetas = np.arctan2(trajectories_proj2[:,:,0], trajectories_proj2[:,:,1]);

    thetas_init = thetas[:,0] #np.arange(-np.pi, np.pi, np.pi/batch_size*2);
    idx = np.argsort(thetas_init)
    thetas_init = thetas_init[idx]

    thetas_sorted = thetas[idx]
    theta_unwrapped = np.unwrap(thetas_sorted, period=2*np.pi);
    theta_unwrapped = np.roll(theta_unwrapped, -1, axis=0);
    arr = np.sign(theta_unwrapped[:,-1]-theta_unwrapped[:,0]);
    idx=[i for i, t in enumerate(zip(arr, arr[1:])) if t[0] != t[1]]; 
    stabilities=-arr[idx].astype(int)
    fxd_pnt_thetas = thetas_init[idx]
    stab_idx = np.where(stabilities==-1); 
    saddle_idx = np.where(stabilities==1)

    if cs:
        fxd_pnts = cs(fxd_pnt_thetas)
        return fxd_pnt_thetas, stabilities, stab_idx, saddle_idx, fxd_pnts
    else:
        return fxd_pnt_thetas, stabilities, stab_idx, saddle_idx
        
def get_cubic_spline_ring(thetas, invariant_manifold):

    thetas_unique, idx_unique = np.unique(thetas, return_index=True);
    idx_sorted = np.argsort(thetas_unique);
    
    invariant_manifold_unique = invariant_manifold[idx_unique,:];
    invariant_manifold_sorted = invariant_manifold_unique[idx_sorted,:];
    invariant_manifold_sorted[-1,:] = invariant_manifold_sorted[0,:];
    cs = scipy.interpolate.CubicSpline(thetas_unique, invariant_manifold_sorted, bc_type='periodic')
        
    return cs    
        
def inv_man_analysis(inv_man, wo, wrec, brec, cs):
    
    #get vfs
    vf_diff_spline, X, Y, U, V = vf_on_ring(None, wo,  wrec, brec, cs, method='spline')
    vf_diff_closest, X, Y, U, V = vf_on_ring(inv_man, wo,  wrec, brec, None, method='closest')
    return vf_diff_spline, vf_diff_closest

def get_dimension(inv_man, function=skdim.id.MOM()):
    dim=function.fit(inv_man).dimension_
    return dim

def tda_trajectories(inv_man, maxdim=1, show=False):
    diagrams = ripser(inv_man, maxdim=maxdim)['dgms']

    if show:
        plot_diagrams(diagrams, show=show)
    return diagrams

def analyse_net(net, task, batch_size=128, T=2e4, dt=.1):
    wi, wrec, brec, wo, oth = get_weights_from_net(net)
    # task = center_out_reaching_task(T=T, dt=dt, time_until_cue_range=[T, T+1], angles_random=False);
    
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, 'random', batch_size=batch_size)
    
    inv_man = detect_slow_manifold(trajectories, subsample=100, tol=1e-4)
    inv_man_proj2 = np.dot(inv_man, wo)
    dim = get_dimension(inv_man)
    
    inv_man_proj2 = np.dot(inv_man,wo)
    #get angles from invariant manifold
    inv_man_thetas = np.arctan2(inv_man_proj2[:,0], inv_man_proj2[:,1]);

    #fit spline
    cs = get_cubic_spline_ring(inv_man_thetas, inv_man)
    
    xs, csx2, csx2_proj2 = get_manifold_from_closest_projections(inv_man, wo, npoints=batch_size);
    
    vf_diff_spline, vf_diff_closest = inv_man_analysis(inv_man, wo, wrec, brec, cs)
    
    
def s_type_perturbations(net, h_init, timesteps, perturbations_per_h0=1, noise_std=1e-3):
    
    net.map_output_to_hidden = False
    s_perts = np.random.normal(0, noise_std, (h_init.shape[0]*perturbations_per_h0, net.dims[1]))
    h_init_ = np.tile(h_init, (perturbations_per_h0, 1, 1)).reshape((-1, net.dims[1]))
    h_init_ += s_perts
    
    input = np.zeros((h_init_.shape[0], timesteps, net.dims[0]))
    output, trajectories = simulate_rnn_with_input(net, input, h_init_)
    
    trajectories_flat = trajectories.reshape((-1,net.dims[1]));

    d = cdist(trajectories_flat,h_init)
    dist_to_inv_man = np.min(d,axis=1).reshape((timesteps*perturbations_per_h0,-1));
    
    return output, trajectories, dist_to_inv_man


def do_all_analysis(folder, batch_size=128, T1_multiple=16, speed_range=[0.05,0.05]):
    
    #main_exp_name='/angular_integration/T25_Mml_rnn_Stanh_N512_Ihighgain_R0/'
    #folder = parent_dir+"/experiments/" + main_exp_name
    
    params_path = glob.glob(folder + '/param*.txt')[0]
    with open(params_path, 'rb') as f:
        training_kwargs = pickle.load(f)
    T1 = training_kwargs['T']/training_kwargs['dt_task']
    T=int(T1); dt=training_kwargs['dt_rnn'];
    exp_list = glob.glob(folder + "/res*")
    nexps = len(exp_list)
    all_eps = np.empty((nexps,3,T))
    for exp_i in range(len(exp_list)):
        path = exp_list[exp_i]
        net, result = load_net_path(path)
        net.map_output_to_hidden = False
        wi, wrec, brec, wo, oth = get_weights_from_net(net);
        #training_kwargs = result['training_kwargs'] #

        task = angularintegration_task_constant(T=T, dt=dt, speed_range=speed_range, sparsity=1, random_angle_init='equally_spaced');
        input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, 'random', batch_size=batch_size)
        xs, csx2, csx2_proj2 = get_manifold_from_closest_projections(trajectories.reshape((-1,net.dims[1])), net.wo.detach().numpy(), npoints=batch_size);
        #eps_min_int, eps_mean_int, eps_plus_int 
        all_eps[exp_i] = angular_loss_angvel_noinput(net, xs, csx2, T*T1_multiple, batch_size=csx2.shape[0])
        
        
    np.save(folder+'all_eps.npy', all_eps)

##################
def plot_losses_in_folder(folder):
    #main_exp_name='center_out/variable_N100_T250_Tr100/tanh/'
    #folder = parent_dir+"/experiments/" + main_exp_name
    exp_list = glob.glob(folder + "/res*")
    nexps = len(exp_list)
    all_losses = np.empty((nexps,10000))
    all_losses[:] = np.nan
    fig, axs = plt.subplots(1, 2, figsize=(3, 3))
    for exp_i in range(nexps):
        print(exp_i)
        path = exp_list[exp_i]
        net, result = load_net_path(path)
        # wi, wrec, brec, wo, oth = get_weights_from_net(net);
        losses = result['losses']
        all_losses[exp_i,:np.argmin(losses)] = losses[:np.argmin(losses)].copy()
        axs[0].plot(losses[:np.argmin(losses)].copy(), 'b',  alpha=.01); 
        #print(np.nanmin(losses), np.argmin(losses))
    axs[0].plot(np.nanmean(all_losses,axis=0), 'b', zorder=1000)
    last_non_nan_idx = np.argmax(np.isnan(all_losses), axis=1)
    last_non_nan_idx = np.maximum(0, last_non_nan_idx - 1)
    last_non_nan = all_losses[np.arange(all_losses.shape[0]), last_non_nan_idx]

    bins = np.logspace(np.log10(np.min(last_non_nan)), np.log10(np.max(last_non_nan)), 20)
    axs[1].hist(last_non_nan, bins=bins, orientation='horizontal')

    axs[0].set_yscale('log'); axs[1].set_yscale('log')
    axs[0].set_ylim([np.nanmin(all_losses), np.nanmax(all_losses)])
    axs[1].set_ylim([np.nanmin(all_losses), np.nanmax(all_losses)])
    axs[1].axis('off')


def plot_example_trajectories_msg(trajectories_proj2, xs, folder):
    
    cmap = plt.get_cmap('hsv')
    norm = mpl.colors.Normalize(-np.pi, np.pi)
    
    fig, ax = plt.subplots(1, 1, figsize=(3, 3));
    for i in range(trajectories_proj2.shape[0]):
        plt.plot(trajectories_proj2[i,:,0].T, trajectories_proj2[i,:,1].T, color=cmap(norm(xs[i])));
    plt.scatter(np.cos(xs), np.sin(xs), s=100, facecolors='none', edgecolors=cmap(norm(xs)))
    ax.axis('off')
    plt.savefig(folder+'example_trials_2d.pdf', bbox_inches="tight");


def plot_vf_ang_int():
    
    T=250; dt=.1; batch_size=2;
    net.map_output_to_hidden = False
    task = angularintegration_task_constant(T=T, dt=dt, speed_range=[0.01,0.01], sparsity=1, random_angle_init='equally_spaced');
    input, target, mask, output, trajectories_0 = simulate_rnn_with_task(net, task, T, 'random', batch_size=batch_size)
    wo = net.wo.detach().numpy();

    inv_man = detect_slow_manifold(trajectories_0, subsample=1, tol=1e-3)
    inv_man_proj2 = np.dot(inv_man, wo)
    xs, csx2, csx2_proj2 = get_manifold_from_closest_projections(inv_man, wo, npoints=batch_size)
    output, trajectories_1 = simulate_rnn_with_input(net, np.zeros((inv_man.shape[0],150,1)), inv_man)

    diff, X, Y, U, V = vf_on_ring(trajectories_1[:,-1,:], wo,  wrec, brec, None, rnn_ode=recttanh_ode, method='points')
    plot_vf_on_ring(X,Y,U,V);