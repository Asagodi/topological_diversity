import os, sys
import glob
current_dir = os.path.dirname(os.path.realpath('__file__'))
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
# import torch.nn.init as weight_init

import scipy
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.integrate import odeint, DOP853, solve_ivp

from functools import partial
import numpy as np
import numpy.ma as ma
from math import isclose
import pickle
import re
import time
from itertools import chain, combinations, permutations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.cm as cmx

import conley_functions as cf
import networkx as nx
import subprocess
from tqdm import tqdm


# from psychrnn.tasks.perceptual_discrimination import PerceptualDiscrimination
# from tasks import PerceptualDiscrimination, PoissonClicks   
#analysis
def pd_accuracy_function(y, yhat, output_mask):
    chosen = np.argmax(np.mean(yhat*output_mask, axis=1), axis = 1)
    truth = np.argmax(np.mean(y*output_mask, axis = 1), axis = 1)
    response_correctness = np.mean(np.equal(truth, chosen))
    return response_correctness
    

def get_correctness_curve(y, yhat, output_mask, trial_params):
    coherence_array = np.array([trial_params[i]['coherence'] for i in range(trial_params.size)])
    direction_array = np.array([trial_params[i]['direction'] for i in range(trial_params.size)])
    directed_coherence_array = coherence_array * (-1.)**direction_array

    chosen = np.argmax(np.mean(yhat*output_mask, axis=1), axis = 1)
    truth = np.argmax(np.mean(y*output_mask, axis = 1), axis = 1)
    response_correctness = np.equal(truth, chosen)
    all_coherences = np.unique(coherence_array)
    average_accuracy_per_coherence = np.array([np.mean(response_correctness[np.where(coherence_array == coherence)]) for coherence in all_coherences])
    
    return all_coherences, average_accuracy_per_coherence


def get_psycho_curve(y, yhat, output_mask, trial_params):
    coherence_array = np.array([trial_params[i]['coherence'] for i in range(trial_params.size)])
    direction_array = np.array([trial_params[i]['direction'] for i in range(trial_params.size)])
    directed_coherence_array = coherence_array * (-1.)**direction_array

    chosen = np.argmax(np.mean(yhat*output_mask, axis=1), axis = 1)
    truth = np.argmax(np.mean(y*output_mask, axis = 1), axis = 1)
    response_correctness = np.equal(truth, chosen)

    all_coherences = np.unique(coherence_array)
    all_directed_coherences = np.unique(directed_coherence_array)
    average_accuracy_per_coherence = np.array([np.mean(chosen[np.where(directed_coherence_array == coherence)]) for coherence in all_directed_coherences])
    
    return all_directed_coherences, average_accuracy_per_coherence



def get_accuracy_poissonclicks(inputs, outputs, output_mask, trial_params):
    N_clicks = np.sum(inputs, axis=1)
    highest_click_count_index = np.argmax(N_clicks, axis=1)
    excludeequals = np.where(N_clicks[:,0]==N_clicks[:,1],False,True)

    ratio_array = np.array([trial_params[i]['ratio'] for i in range(trial_params.size)])
    ratios = np.unique(ratio_array)
    
    chosen = np.argmax(np.mean(outputs*output_mask, axis=1), axis = 1)*excludeequals
    truth = highest_click_count_index*excludeequals

    response_correctness = np.equal(truth, chosen)

    choice_proportion_per_ratio = np.array([np.mean(chosen[np.where(ratio_array == coherence)]) for coherence in ratios])
    average_accuracy_per_ratio = np.array([np.mean(response_correctness[np.where(ratio_array == coherence)]) for coherence in ratios])

    accuracy = np.mean(response_correctness)
    
    return accuracy, response_correctness, choice_proportion_per_ratio, average_accuracy_per_ratio, N_clicks,  highest_click_count_index, chosen


def get_accuracy_poissonclicks_w(inputs, outputs, output_mask):
    N_clicks = np.sum(inputs[...,:2], axis=1)
    highest_click_count_index = np.argmax(N_clicks, axis=1)
    excludeequals = np.where(N_clicks[:,0]==N_clicks[:,1],False,True)
    
    chosen = np.argmax(np.mean(outputs*output_mask, axis=1), axis = 1)[excludeequals]
    truth = highest_click_count_index[excludeequals]

    response_correctness = np.equal(truth, chosen)
    accuracy = np.mean(response_correctness)
    return accuracy, response_correctness, N_clicks, highest_click_count_index, chosen, excludeequals


def get_accuracy_perceptualdiscrimination(y, yhat, output_mask, trial_params):
    coherence_array = np.array([trial_params[i]['coherence'] for i in range(trial_params.size)])
    direction_array = np.array([trial_params[i]['direction'] for i in range(trial_params.size)])
    directed_coherence_array = coherence_array * (-1.)**direction_array

    chosen = np.argmax(np.mean(yhat*output_mask, axis=1), axis = 1)
    truth = np.argmax(np.mean(y*output_mask, axis = 1), axis = 1)
    response_correctness = np.equal(truth, chosen)
    all_coherences = np.unique(coherence_array)
    
    choice_proportion_per_coherence = np.array([np.mean(chosen[np.where(coherence_array == coherence)]) for coherence in all_coherences])
    average_accuracy_per_coherence = np.array([np.mean(response_correctness[np.where(coherence_array == coherence)]) for coherence in all_coherences])
    
    accuracy = np.mean(response_correctness)

    return accuracy, response_correctness, choice_proportion_per_coherence, average_accuracy_per_coherence



def get_wrong_trials(N_clicks, chosen):
    """Determines the wrong trials based on the number of clicks and the choices"""
    truth = np.argmax(N_clicks, axis=1)
    wrong_and_equal_trials = np.where(chosen!=truth)[0]
    equal_trials = np.where(N_clicks[:,0]==N_clicks[:,1])[0]
    wrong_trials = []
    for trial in wrong_and_equal_trials:
        if trial in equal_trials:
            continue
        wrong_trials.append(trial)
    return wrong_trials


def run_model_chunked(x, model, hidden_init='offset', hidden_offset = 0., hidden_initial_variance=0.0001, n_chunks=10):
    """Returns """
    chunks = np.split(x, indices_or_sections=n_chunks, axis=0)
    all_yhat = torch.empty((0, x.shape[1], model.output_size))
    all_hidden_states = torch.empty((0, x.shape[1], model.hidden_dim))
    with torch.no_grad():
        for chunk_x in chunks:
            inputs = torch.tensor(chunk_x, dtype=torch.float).to(model.device)
            if np.all(hidden_init == 'random'):
                hidden = model.hidden_offset+hidden_offset+torch.normal(0, hidden_initial_variance, (model.n_layers, inputs.shape[0], model.hidden_dim)).to(model.device) 
            elif np.all(hidden_init == 'offset'):
                hidden = model.hidden_offset.repeat(model.n_layers,inputs.shape[0],1).reshape((model.n_layers, inputs.shape[0], model.hidden_dim))
            else:
                hidden = hidden_init
            hidden_states, hidden_last = model.rnn(inputs, hidden)
            all_yhat = torch.vstack([all_yhat, model.out_act(model.fc(hidden_states))])
            all_hidden_states = torch.vstack([all_hidden_states, hidden_states])
    
    return all_yhat, all_hidden_states


def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)


####FIXED POINT
def find_stabilities(fixed_points, W_hh, tol = 10**-4):
    """
    Argument: fixed point
    tol: is used as to what level of tolarance to use to determine support for fixed point
    Returns: stabilist which includes the stabilities of the fixed points 
    Note: What to do if support is [] (empty)?
        -Add note: zero support to stabilist?
    """
    stabilist = []
    unstabledimensions = []
    for x in fixed_points:
        #calculate stability of fixed points
        support = np.where(x>tol)[0]
        if len(support) == 0:
            eigenvalues = np.diagonal(W_hh) - 1 
        if len(support) == 1:
            eigenvalues = W_hh[support,support]-1 # display(Math(r"$x_*$=" + str(np.round(x,2)) +"   Eigenvalues:" + str(np.round(W_hh[support,support][0][0]-1,2))))

        elif len(support) > 1:
            r = np.array(support)
            eigenvalues = np.linalg.eigvals(W_hh[r[:,None], r]-np.eye(len(support)))
            # print(r"$x_*$=", x, "Eigenvalues:", eigenvalues)
        try:
            if np.any(np.real(eigenvalues)>0.):
                stabilist.append(0)                # print("Unstable")
            else:
                stabilist.append(1)    # print("Stable")
        except:
            stabilist.append(-1)
        unstabledimensions.append(np.where(np.real(eigenvalues)<0)[0])
    return stabilist, unstabledimensions




def find_fixed_points_grid(fun, Nrec, max_grid=3, step=1, tol = 10**-4,  maxiter = 10000,
                      method='Nelder-Mead', verbose=False):
    options = {"disp": verbose, "maxiter": maxiter}
    
    #create grid to start search
    Ngrid = int(max_grid/step)
    x0grid = np.mgrid[tuple(slice(0, max_grid, step) for _ in range(Nrec))].reshape((int((Ngrid)**Nrec), Nrec))

    # constraint = {LinearConstraint([1], [-tol], [tol])} #use contraint to find real fixed point, not just minimum
    bounds = [(0., np.Inf)]
    results = []
    for x0 in x0grid:
        opt = minimize(
            lambda x: np.linalg.norm(fun(x) - x),
            x0=x0,
            method=method,
            constraints=None, 
            bounds=bounds,
            tol=tol,
            callback=None,
            options=options)
        if opt.success:
            results.append(opt.x)

    results = np.array(results)
    return results


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def find_analytic_fixed_points(W_hh, b, W_ih=None, I=None, tol=10**-4):
    """
    Takes as argument all the parameters of the recurrent part of the model (W_hh, b) with a possible input I that connects into the RNN throught the weight matrix W_ih
    """
    fixed_point_list = []
    stabilist = []
    unstabledimensions = []
    Nrec = W_hh.shape[0]
    eigenvalues_list = []
    
    subsets = powerset(range(Nrec))
    # length = sum(1 for _ in subsets)
    # print("Number of supports to check", length)
    for support in tqdm(subsets):

        if support == ():
            continue
        r = np.array(support)
        
        #invert equation
        fixed_point = np.zeros(Nrec)
        if I:
            fpnt_nonzero = -np.dot(np.linalg.inv(W_hh[r[:,None], r]-np.eye(len(support))), b[r] + np.dot(W_ih[r,:], I))
        else:
            fpnt_nonzero = -np.dot(np.linalg.inv(W_hh[r[:,None], r]-np.eye(len(support))), b[r])
        fixed_point[r] = fpnt_nonzero
        
        #check whether it is a fixed point: zero derivative
        # if fnpt_i > 0 for all i=1, ..., Nrec
        if np.all(fixed_point >= 0) and np.all(np.abs(fixed_point-relu_step_input(fixed_point, W_hh, b, W_ih, I))<tol):
            fixed_point_list.append(fixed_point)
        
            #check stability
            eigenvalues = np.linalg.eigvals(W_hh[r[:,None], r]-np.eye(len(support)))
            if np.any(eigenvalues>0.):
                # print("Unstable")
                stabilist.append(0)
            else:
                # print("Stable")
                stabilist.append(1)
            unstabledimensions.append(np.where(np.real(eigenvalues)>0)[0].shape[0])
            eigenvalues_list.append(eigenvalues)
            
    return fixed_point_list, stabilist, unstabledimensions, eigenvalues_list


def lu_step(x, W, b):
    return x*W+b

def relu_step(x, W, b):
    res = np.array(np.dot(W,x)+b)
    res[res < 0] = 0
    return res
 
def relu_step_input(x, W, b, W_ih=None, I=None):
    if I:
        res = np.array(np.dot(W,x) + b + np.dot(W_ih, I))
    else:
        res = np.array(np.dot(W,x) + b)
    res[res < 0] = 0
    return res



def tanh_ode(t,x,W,b,tau, mlrnn=True):
    
    if mlrnn:
        return (-x + np.tanh(np.dot(W,x)+b))/tau
    else:
        return (-x + np.dot(W,np.tanh(x))+b)/tau

#Jacobians
#include versions for x_solved being from a (scipy) ode solver?
def linear_jacobian(t,W,b,tau,x_solved):
    return W/tau


def tanh_jacobian(t,W,b,tau,x_solved, mlrnn=True):
    
    if mlrnn:
        return (-np.eye(W.shape[0]) + np.multiply(W,1/np.cosh(np.dot(W,x_solved[t])+b)**2))/tau
    else:
        return (-np.eye(W.shape[0]) + np.multiply(W,1/np.cosh(x_solved[t])**2))/tau

def relu_jacobian(t,W,b,tau,x_solved):
    return (-np.eye(W.shape[0]) + np.multiply(W, np.where(np.dot(W,x_solved[t])+b>0,1,0)))/tau


#To calculate Lyapunov Exponents
def calculate_lyapunov_spectrum(act_fun,W,b,tau,x_solved,delta_t,from_t_step=0):
    #Benettin 1980: Lyapunov Characteristic Exponents for smooth dynamical 
    #Echmann and Ruelle 1985: Ergodic theory of chaos and strange attractors
    N = W.shape[0]
    Q_n = np.eye(N)
    lyap_spec = np.zeros(N)
    lyaps = []
    N_t = x_solved.shape[0]
    for n in range(from_t_step,N_t):
        M_n = np.eye(N) + act_fun(n,W,b,tau,x_solved)*delta_t
        Q_n, R_n = np.linalg.qr(np.dot(M_n, Q_n))
        lyap_spec += np.log(np.abs(np.diag(R_n)))/(N_t*delta_t)
        lyaps.append(lyap_spec)
    return lyap_spec, lyaps



######sampling dynamics
def sample_hidden_trajs(model, hidden, maxT):
    """Sample hidden trajectories given initial hidden states (hidden) without input
    hidden shape = (batchsize, Nrec)
    """
    batch_size = hidden.shape[1]
    inputs =  torch.zeros((batch_size, maxT, 2), dtype=torch.float)
    hidden_states, hidden_last = model.rnn(inputs, hidden)
    hidden_states = hidden_states.detach().numpy()
    return hidden_states

#sample trajectories starting around the fixed points to determine connecting orbits (should be enough to do this around unstable fixed points)
def sample_trajs_fxdpnts(model, Nrec, fixed_points, max_grid=0.01, Nsteps=3, maxT=5000):
    maxT = 5000
    fixed_points =  np.array(fixed_points)
    step = max_grid/Nsteps
    Ngrid = int(max_grid/step)

    grid = np.mgrid[tuple(slice(-max_grid, max_grid, 2*step) for _ in range(Nrec))].reshape((1, int((Ngrid)**Nrec), Nrec))
    grid = np.where(grid>=0,grid,0)    #make grid positive (introduces double sampled points)
    batch_size = grid.shape[1] 
    all_hidden_stack = np.empty((0, maxT, Nrec))

    for fxdi,fxd in enumerate(fixed_points[:]):
        grid= np.mgrid[tuple(slice(-max_grid, max_grid, 2*step) for _ in range(Nrec))].reshape((1, int((Ngrid)**Nrec), Nrec))
        grid = np.where(grid>=0,grid,0)
        grid = grid + fxd
        hidden = torch.tensor(grid, dtype=torch.float)
        hidden_states = sample_hidden_trajs(model, hidden, maxT)
        all_hidden_stack = np.concatenate([all_hidden_stack, hidden_states])
    return all_hidden_stack, grid



###analysis of dynamics


def participation_ratio(cov_mat_eigenvalues):
    # https://ganguli-gang.stanford.edu/pdf/17.theory.measurement.pdf
    # The eigenvalues of this matrix, µ1 ≥ µ2 ≥, . . . , ≥ µM, reflect neural population variance in each eigen-direction in firing rate space. 
    # PR = (\sum_i µi)^2/(\sum_i µi^2),
    # 1<PR<M
    # for a wide variety of uneven spectra, the PR corresponds to the number of dimensions required to explain about 80% of the total population variance 
    pr = np.sum(cov_mat_eigenvalues)**2/np.sum(cov_mat_eigenvalues**2)
    return pr

def ReLU(x):
    return np.where(x<0,0,x)

def relu_ode(t,x,W,b,tau, mlrnn=True):

    if mlrnn:
        return (-x + ReLU(np.dot(W,x)+b))/tau
    else:
        return (-x + np.dot(W,ReLU(x))+b)/tau

def simulate_from_y0s(y0s, W, b, tau=1, 
                   maxT = 25, tsteps=501):

    N = W.shape[0]
    t = np.linspace(0, maxT, tsteps)
    sols = np.zeros((y0s.shape[1], t.shape[0], N))
    for yi,y0 in enumerate(y0s.T):
        sol = solve_ivp(relu_ode, y0=y0, t_span=[0,maxT],
                        args=tuple([W, b, tau]),
                        dense_output=True)
        sols[yi,...] = sol.sol(t).T

    return sols

from scipy.signal import argrelextrema

def find_bla_persisten_manifold(W, b):
    
    step=.01;x = np.arange(0,1+step,step); ca = np.vstack([x, np.flip(x)])
    ca = np.vstack([x, np.flip(x)])
    sols = simulate_from_y0s(ca, W, b, tau=10, maxT=20000, tsteps=20001);
    idxx = np.argmax(sols[1:,-1,0]-sols[:-1,-1,0]);

    lowspeed_idx = []
    all_speeds = []
    for trial_i in range(sols.shape[0]):
        speeds = []
        for t in range(sols.shape[1]-1):
            
            speed = np.linalg.norm(sols[trial_i, t+1, :]-sols[trial_i, t, :])
            speeds.append(speed)
        all_speeds.append(speeds)
        try:
            # lowspeed_idx.append(np.min(np.where(np.array(speeds)<1e-5)))
            lowspeed_idx.append(argrelextrema(all_speeds[trial_i,:lowspeed_idx[idxx]], np.less)[0][0])
            
        except:
            lowspeed_idx.append(20)
    all_speeds = np.array(all_speeds)
    
    invariant_manifold = np.concatenate([np.flip(sols[idxx,lowspeed_idx[idxx]:,:],axis=0), sols[idxx+1,lowspeed_idx[idxx+1]:,:]])

    plt.plot(invariant_manifold[:,0], invariant_manifold[:,1]);
    
    
def digitize_manifold(time_series, num_bins_x = 100, num_bins_y = 100):
    
    
    x_grid = np.linspace(0, max(time_series[:, 0])*1.1, num_bins_x)
    y_grid = np.linspace(0, max(time_series[:, 1])*1.1, num_bins_y)
    X, Y = np.meshgrid(x_grid, y_grid)
    # Digitize the time series
    digitized_x = np.digitize(time_series[:, 0], bins=x_grid)-1
    digitized_y = np.digitize(time_series[:, 1], bins=y_grid)-1;
    digitized = np.array([digitized_x, digitized_y]);
    u_dig = np.unique(digitized,axis=1);
    
    plt.scatter(X[u_dig[0], u_dig[1]]+1/200., Y[u_dig[0], u_dig[1]]+1/200., marker='s', s=1);
    plt.plot(time_series[:,0], time_series[:,1], 'r.', alpha=.01)
    
    return u_dig


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Define function to calculate geodesic distance along the 1D manifold
def geodesic_distance(manifold_points, point1_index, point2_index):
    # Calculate the Euclidean distance between the points along the manifold
    distance_along_manifold = 0
    for i in range(point1_index, point2_index):
        distance_along_manifold += euclidean_distance(manifold_points[i], manifold_points[i + 1])
    return distance_along_manifold

def geodesic_cumdist(invariant_manifold):
    cumdist = 0
    cumdist_list = []
    for i in range(invariant_manifold.shape[0]-1):
        distance_along_manifold = geodesic_distance(invariant_manifold, i, i+1)
        cumdist += distance_along_manifold
        cumdist_list.append(cumdist)
    return cumdist_list

def get_uniformly_spaced_points_from_manifold(invariant_manifold, npoints):
    cd = geodesic_cumdist(invariant_manifold)
    max_dist = cd[-1]
    points = [invariant_manifold[0]]
    for dist in np.arange(max_dist/npoints, max_dist, max_dist/npoints):
        idx = np.min(np.where(cd>dist)[0])

        points.append(invariant_manifold[idx])
    points.append(invariant_manifold[-1])
    return np.array(points)


def get_uniformly_spaced_points_from_ringmanifold(invariant_manifold, npoints):
    cd = geodesic_cumdist(invariant_manifold)
    max_dist = cd[-1]
    points = [invariant_manifold[0]] #start from a fixed point?
    for dist in np.arange(max_dist/npoints, max_dist, max_dist/npoints):
        idx = np.min(np.where(cd>dist)[0])

        points.append(invariant_manifold[idx])
    points.append(invariant_manifold[-1])
    return np.array(points)


def digitize_trajectories(trajectories, nbins=1000):
    npoints = trajectories.shape[0]
    dims = trajectories.shape[1]
    bins = np.zeros((npoints,dims),np.int64)
    all_edges = np.zeros((nbins+1,dims))
    for r in range(dims):
        edges = np.histogram_bin_edges(trajectories[:,r], bins=nbins)
        all_edges[:,r] = edges
        bins[:,r] = np.digitize(trajectories[:,r], edges)
    binsizes = np.array([all_edges[0,i]-all_edges[1,i] for i in range(dims)])
    dig_bin_idx = np.unique(bins, axis=0)
    
    all_bin_locs = np.zeros(dig_bin_idx.shape)
    for j in range(dig_bin_idx.shape[0]):

        bin_loc = np.array([all_edges[dig_bin_idx[j,i]-1,i] for i in range(dims)])
        bin_loc=bin_loc-binsizes/2.
        all_bin_locs[j, :] = bin_loc
    
    return all_bin_locs

from scipy.ndimage import gaussian_filter1d
def get_cubic_spline_ring(thetas, invariant_manifold):
    """

    Parameters
    ----------
    all_bin_locs : TYPE
        DESCRIPTION.

    Returns
    -------
    cs : TYPE
        DESCRIPTION.

    """

    thetas_unique, idx_unique = np.unique(thetas, return_index=True);
    idx_sorted = np.argsort(thetas_unique);
    
    invariant_manifold_unique = invariant_manifold[idx_unique,:];
    invariant_manifold_sorted = invariant_manifold_unique[idx_sorted,:];
    invariant_manifold_sorted[-1,:] = invariant_manifold_sorted[0,:];
    cs = scipy.interpolate.CubicSpline(thetas_unique, invariant_manifold_sorted, bc_type='periodic')
    
    # smoothed = gaussian_filter1d(all_bin_locs_sorted, 20, axis=0, mode='wrap')
    
    return cs    

def simulate_rnn(net, task, T, batch_size = 256):

    input, target, mask = task(batch_size); input = torch.from_numpy(input).float();
    output, trajectories = net(input, return_dynamics=True); 
    output = output.detach().numpy();
    trajectories = trajectories.detach().numpy()
    return input, target, mask, output, trajectories


from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
def identify_limit_cycle(time_series, skip_first=10, tol=1e-6):
    d = cdist(time_series[-1,:].reshape((1,-1)),time_series[:-skip_first])
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
            recurrences.append(traj[trial_i,idx:,:])
            recurrences_pca.append(traj_pca[trial_i,idx:,:])

    return recurrences, recurrences_pca

def get_slow_manifold(net, task, T, from_t=300, batch_size=256, n_components=3, nbins=1000):
    n_rec = net.dims[1]
    
    input, target, mask, output, trajectories = simulate_rnn(net, task, T, batch_size)
    
    pca = PCA(n_components=n_components)
    invariant_manifold = trajectories[:,from_t:,:].reshape((-1,n_rec))
    pca.fit(invariant_manifold)
    traj_pca = pca.transform(invariant_manifold).reshape((batch_size,-1,n_components))
    recurrences, recurrences_pca = find_periodic_orbits(trajectories, traj_pca, limcyctol=1e-2, mindtol=1e-4)
    fxd_pnts = np.array([recurrence for recurrence in recurrences if len(recurrence)==1]).squeeze()
    
    traj_pca_flat = traj_pca.reshape((-1,n_components))
    # all_bin_locs = digitize_trajectories(invariant_manifold, nbins=nbins)
    all_bin_locs_pca = digitize_trajectories(traj_pca_flat, nbins=nbins)
    thetas = np.arctan2(traj_pca_flat[:,1],traj_pca_flat[:,0]);
    cs = get_cubic_spline_ring(thetas, invariant_manifold)
    cs_pca = get_cubic_spline_ring(thetas, traj_pca_flat)
    
    saddles = get_saddle_locations_from_theta(thetas, cs)
    
    return trajectories, saddles, pca, cs, cs_pca, fxd_pnts, recurrences, recurrences_pca, all_bin_locs_pca


def get_saddle_locations_from_theta(thetas, cs, cutoff=0.005):
    thetas_unique, idx_unique = np.unique(thetas, return_index=True);
    th_s = np.sort(thetas_unique)
    thetas_jump = th_s[1:] - th_s[:-1]
    idx = np.where(thetas_jump>cutoff)[0];
    saddles = cs((th_s[idx]+th_s[idx-1])/2.);

    return saddles


def get_saddles_from_simulated_trajectories(net, fxd_pnts, pca, cs):
    thetas_saddles = []
    fxd_pnts_u, idx = np.unique(np.round(fxd_pnts), axis=0, return_index=True); 
    fxd_pnts_u = fxd_pnts[idx,:];
    fxd_pnts_u_pca = pca.transform(fxd_pnts_u)
    thetas_fxd = np.arctan2(fxd_pnts_u_pca[:,1],fxd_pnts_u_pca[:,0]);
    thetas_fxd = np.sort(thetas_fxd)
    thetas_fxd_u, idx = np.unique(np.round(thetas_fxd,2), axis=0, return_index=True);
    thetas_fxd = thetas_fxd[idx]
    fxd_pnts_u = fxd_pnts_u[idx,:]
    for i in range(1, thetas_fxd.shape[0]):
        theta_step = -(thetas_fxd[i]- thetas_fxd[i-1])/100
        xs = np.arange(thetas_fxd[i]+25*theta_step, thetas_fxd[i-1]-25*theta_step, theta_step)
        
        starting_points = cs(xs)
        input_zero = torch.from_numpy(np.zeros((starting_points.shape[0], 10000, 3))).float()
        output, trajectories_from_slow = net(input_zero, return_dynamics=True, h_init=starting_points);
        output = output.detach().numpy();
        trajectories_from_slow = trajectories_from_slow.detach().numpy()
        thetas_final = np.arctan2(trajectories_from_slow[:,-1,1],trajectories_from_slow[:,-1,0]);
        thetas_fj = thetas_final[1:]-thetas_final[:-1]; 
        idx = np.argmax(thetas_fj)
        theta_saddle = xs[idx]
        
        thetas_saddles.append(theta_saddle)
        print(theta_saddle, thetas_fxd[i]+5*theta_step, thetas_fxd[i-1]-5*theta_step)
    thetas_saddles.append(np.pi)
    saddles = cs(thetas_saddles)
    
    return saddles

# def get_uniformly_spaced_points_from_manifold(invariant_manifold, npoints):
#     cd = geodesic_cumdist(invariant_manifold)
#     max_dist = cd[-1]
#     points = [invariant_manifold[0]]
#     for dist in np.arange(max_dist/npoints, max_dist, max_dist/npoints):
#         idx = np.min(np.where(cd>dist)[0])
        
#         idx = argrelextrema(np.abs(cd-dist), np.less)[0][0]
#         act_dist = cd[idx]-dist
#         if act_dist>dist_eps:
#             next_point = 
#                 invariant_manifold[idx]
#         else:
#             points.append(invariant_manifold[idx])
#     points.append(invariant_manifold[-1])
#     return np.array(points)


def get_speed_and_acceleration(trajectory):
    speeds = []
    accelerations = []
    for t in range(trajectory.shape[0]-1):
        speed = np.linalg.norm(trajectory[t+1, :]-trajectory[t, :])
        speeds.append(speed)
    for t in range(trajectory.shape[0]-2):
        acceleration = speeds[t+1]-speeds[t]
        accelerations.append(acceleration)
    speeds = np.array(speeds)
    accelerations = np.array(accelerations)
    return speeds, accelerations


#identify slow manifolds

#1fp with full support
from matplotlib.ticker import MaxNLocator
import glob
import os, sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
def get_invman_fullsupp(W, b, tau, maxT=2000, tsteps=2001, fig_folder=parent_dir+'/Stability/figures/inv_man'):
    
    np.random.seed(100)
    eps = 0.1
    W_pert = W + np.random.normal(0,scale=eps,size=(2,2))
    
    ts = np.linspace(0, maxT, tsteps)
    x = np.arange(0,1,1/100.); 
    ca_dense = np.vstack([x, np.flip(x)])
    
    fixed_point_list, stabilist, unstabledimensions, eigenvalues_list = find_analytic_fixed_points(W, b)
    eigenvalues, eigenvectors = np.linalg.eig(W-np.eye(2))
    
    sol_3 = solve_ivp(relu_ode, y0=[fixed_point_list[0][0]-eigenvectors[0,0], fixed_point_list[0][1]-eigenvectors[0,1]],  t_span=[0,maxT],
                    args=tuple([W, b, tau]),
                    dense_output=True); 
    speeds3, accelerations3 = get_speed_and_acceleration(trajectory=sol_3.sol(ts).T)
    idx3 = argrelextrema(accelerations3, np.less)[0][0]

    sol_4 = solve_ivp(relu_ode, y0=[fixed_point_list[0][0]+eigenvectors[0,0], fixed_point_list[0][1]+eigenvectors[0,1]],  t_span=[0,maxT],
                    args=tuple([W, b, tau]),
                    dense_output=True);
    speeds4, accelerations4 = get_speed_and_acceleration(trajectory=sol_4.sol(ts).T)
    idx4 = argrelextrema(accelerations4, np.less)[0][0]
    
    # Define the range for the plot
    x_vf = np.linspace(-.5, 2, 20)
    y_vf = np.linspace(-.5, 2, 20)
    
    X, Y = np.meshgrid(x_vf, y_vf)
    # Compute the derivatives at each point in the meshgrid
    U, V = np.zeros_like(X), np.zeros_like(Y)
    for i in range(len(x_vf)):
        for j in range(len(y_vf)):
            yprime = relu_ode(0, np.array([X[i, j], Y[i, j]]), W, b, tau)
            U[i,j] = yprime[0]
            V[i,j] = yprime[1]

    # Plot the vector field
    fig,ax=plt.subplots(1,1);
    ax.quiver(X, Y, U, V, scale=2)
    ax.set_xlabel(r'$x_1$', size=15)
    ax.set_ylabel(r'$x_2$', size=15)
    ax.set_xlim([-0.2, 1.5])
    ax.set_ylim([-0.2, 1.5])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.plot(sol_3.sol(ts)[0,idx3:], sol_3.sol(ts)[1,idx3:], 'g', alpha=0.5);
    plt.plot(sol_4.sol(ts)[0,idx4:], sol_4.sol(ts)[1,idx4:], 'g', alpha=0.5);
    plt.plot(ca_dense[0,:], ca_dense[1,:], 'b', alpha=0.5);
    plt.quiver(fixed_point_list[0][0], fixed_point_list[0][1], eigenvectors[0,0], eigenvectors[0,1]);
    plt.quiver(fixed_point_list[0][0], fixed_point_list[0][1], eigenvectors[1,0], eigenvectors[1,1]);
    plt.plot(fixed_point_list[0][0]+eigenvectors[0,0], fixed_point_list[0][1]+eigenvectors[0,1], 'x')
    plt.savefig(fig_folder+"/1fp.pdf")
    
def get_invman_3fps(W, b, tau, eps_=0.001):
    
    
    fixed_point_list, stabilist, unstabledimensions, eigenvalues_list = find_analytic_fixed_points(W, b)

    eigenvalues, eigenvectors = np.linalg.eig(W-np.eye(2))
    y0s = np.array([fixed_point_list[0]-2*eigenvectors[:,0]]).T
    sols_2 = simulate_from_y0s(y0s, W, b, tau=10, maxT=20000, tsteps=20001)
    speeds, accelerations = get_speed_and_acceleration(trajectory=sols_2[0,...])
    idx = np.where(np.abs(accelerations[:])<2e-5)[0][0]
    invariant_manifold = np.concatenate([np.flip(sols_2[0,idx:,:],axis=0)])
    idx = np.where(np.abs(accelerations[:])<2e-5)[0][0]
    points_on_invman = get_uniformly_spaced_points_from_manifold(invariant_manifold, npoints=100); plt.plot(*points_on_invman.T, '-g'); 
    x = np.arange(0,1,1/100.); 
    ca_dense = np.vstack([x, np.flip(x)])
    plt.plot(ca_dense[0,:], ca_dense[1,:], 'b', alpha=0.5);
    plt.quiver(fixed_point_list[0][0], fixed_point_list[0][1], -1, 0);
    plt.quiver(fixed_point_list[0][0], fixed_point_list[0][1], 0, 1);
    for i, fxd_pnt in enumerate(fixed_point_list):
        plt.plot(fxd_pnt[0], fxd_pnt[1], 'rx');
    
    plt.savefig(fig_folder+"/1fp_side.pdf")
    
    

    
def get_invman_3fps(W, b, tau, eps_=0.001):
    
    np.random.seed(104)
    eps = 0.1
    W_pert = W + np.random.normal(0,scale=eps,size=(2,2))
    
    fixed_point_list, stabilist, unstabledimensions, eigenvalues_list = find_analytic_fixed_points(W, b)
    eigenvalues, eigenvectors = np.linalg.eig(W-np.eye(2))
    y0s = np.array([fixed_point_list[2]+eps_*eigenvectors[:,1],fixed_point_list[2]-eps_*eigenvectors[:,1]]).T
    sols_2 = simulate_from_y0s(y0s, W, b, tau=10, maxT=20000, tsteps=20001)
    invariant_manifold = np.concatenate([np.flip(sols_2[0,:,:],axis=0), sols_2[1,:,:]])
    points_on_invman = get_uniformly_spaced_points_from_manifold(invariant_manifold, npoints=200); plt.plot(*points_on_invman.T, '.'); 
    
    plt.plot(*points_on_invman.T, '-g'); 
    plt.quiver(fixed_point_list[2][0], fixed_point_list[2][1], eigenvectors[0,0], eigenvectors[0,1]);
    plt.quiver(fixed_point_list[2][0], fixed_point_list[2][1], eigenvectors[1,0], eigenvectors[1,1]);
    for i, fxd_pnt in enumerate(fixed_point_list):
        plt.plot(fxd_pnt[0], fxd_pnt[1], 'rx');


sys.path.append("C:/Users/abel_/Documents/Lab/Software/fixed-point-finder"); 
from FixedPointFinderTorch import *
from plot_utils import plot_fps
#Fixed point finder
def get_fps_fpf():
    main_exp_name='center_out/N200_T500_noisy_hinitlast/tanh/'
    #from plot_losses import *; from analysis_functions import *
    import sys

    folder = parent_dir+"/experiments/" + main_exp_name

    sys.path.append("C:/Users/abel_/Documents/Lab/Software/fixed-point-finder")
    exp_i=0
    which = 'post'
    net, wi, wrec, wo, brec, h0, oth, training_kwargs = load_all(main_exp_name, exp_i, which=which);
    net.noise_std = 0
    rnn = nn.RNN(net.dims[0], net.dims[1], batch_first=True)  # batch_first parameter 
    with torch.no_grad():
        rnn.weight_ih_l0 = nn.Parameter(torch.from_numpy(wi.T))  # input-to-hidden weights
        rnn.weight_hh_l0 = nn.Parameter(torch.from_numpy(wrec))# hidden-to-hidden weights
        rnn.bias_ih_l0 = nn.Parameter(torch.from_numpy(brec))
        rnn.bias_hh_l0.fill_(0.)


    batch_size =  2**10
    np.random.seed(100);
    initial_states = np.random.uniform(-1,1, (batch_size, net.dims[1]));  #(n, n_states) numpy array, where n is the number of initializations and n_states is the dimensionality of the RNN state;
    T=1e4; dt=.1; batch_size=128;
    task = center_out_reaching_task(T=T, dt=dt, time_until_cue_range=[50, 51], angles_random=False);
    input, target, mask, output, trajectories = simulate_rnn(net, task, T, batch_size)
    #initial_states = trajectories[:,500,:]
    inputs = np.zeros((1,3))#1, n_inputs) where n_inputs is an int specifying the depth of the inputs expected by your_rnn_cell
    fpf = FixedPointFinderTorch(rnn);
    fps = fpf.find_fixed_points(initial_states, inputs)
    fig = plot_fps(fps[1]); fig.savefig(folder+'/fpf.pdf', bbox_inches="tight")

    from_t = 0
    invariant_manifold = trajectories[:,from_t:,:]
    fig = plot_fps(fps[1], state_traj=invariant_manifold); fig.savefig(folder+'/fpf_traj.pdf', bbox_inches="tight")
    
    
def is_nonnormal(A):
    A_star = A.conj().T
    return not np.allclose(np.dot(A, A_star), np.dot(A_star, A))

#MORSE
def get_connection_matrix(fixed_point_cubes, RCs, cds_full):
    #if there is a non-zero entry at [j,i] then there is a path from i to j
    #i.e. if [i,j]=1 then i<j
    connection_matrix = np.zeros((len(fixed_point_cubes)+len(RCs),len(fixed_point_cubes)+len(RCs))) #has all path information

    for i in range(len(fixed_point_cubes)):
        for j in range(len(RCs)):
            try:
                pathij = nx.shortest_path(cds_full.G, source=fixed_point_cubes[i], target=RCs[j][0])               
                connection_matrix[j,len(RCs)+i]=1
            except nx.NetworkXNoPath:
                0

        for j in range(len(fixed_point_cubes)):
            if i!=j:
                try:
                    pathij = nx.shortest_path(cds_full.G, source=fixed_point_cubes[i], target=fixed_point_cubes[j])          
                    connection_matrix[len(RCs)+j,len(RCs)+i]=1
                except nx.NetworkXNoPath:
                    0
                    
    #sort rcs based on connection matrix: so that they follow partial ordering
    sum_of_outgoing_trajectories = np.sum(connection_matrix, axis=0)
    sum_of_incoming_trajectories = np.sum(connection_matrix, axis=1)
    order_idx = np.argsort(sum_of_outgoing_trajectories)
    connection_matrix = connection_matrix[order_idx[:,None], order_idx]

    # fxdpnt_idx = len(RCs) + np.array(range(len(RCs_fxd)))
    # sum_of_outgoing_trajectories_from_fixed_points = np.sum(fxdpnt_submatrix, axis=1)
    # fxd_order_idx = np.argsort(sum_of_outgoing_trajectories_from_fixed_points)
    # fxdpnt_submatrix = connection_matrix[fxdpnt_idx[:, None], fxdpnt_idx]
    # connection_matrix = fxdpnt_submatrix[fxd_order_idx[:,None], fxd_order_idx]
                    
    return connection_matrix, order_idx


def find_graph_isomorphism(connection_matrix, morse_dict):
    """Brute force graph ismomorphism by going through all permutations of adjacency matrix"""
    p=permutations(range(morse_dict['number_of_fixed_points']))
    for j in list(p):
        r = np.array(j)+len(morse_dict['rcs'])
        if np.all(connection_matrix[r[:,None], r] == morse_dict['connection_matrix']):
            return r
    return False


#CONLEY (move to conley_functions.py)
def get_indexpair_and_conleyindex(cds, RCs, verbosity=False):
    """
    Computes the index      and the conley index for a given list of recurrent sets.
    
    TODO: allow logging/verbosity
    """

    RPcubes = {}
    index_success = []
    for i in range(0,len(RCs)):#range(0,1)
        RPcubes[i] = []
        # print("Component", i+1)
        RPcubes[i].append(RCs[i])

        #Get (isolating) neighbourhood (candidate) around recurrent set
        Nbhd = RPcubes[i][0]
        # print("Finding isolating neighbourhood")
        S = cds.invariantPart(Nbhd)
        M = cds.cubical_wrap(S).intersection(cds.G.nodes())
    #     for mstep in range(1):
    #         M = cds.cubical_wrap(M).intersection(cds.G.nodes())

        #calculate index pair
        try:
            # print("Calculating index pair")
            P1, P0, Pbar1, Pbar0 = cds.index_pair(M)

            #write index pairs to file for Conley index calcualtion with CHomP

            P1graph = nx.subgraph(cds.G, P1)
            cubefile, mapfile = cf.write_mapandcubes(P1graph, cds.delta, cds)
            with open('rc%s_P1_map.map'%(i+1), 'w') as f:
                f.writelines(mapfile)
            with open('rc%s_P1_cubes.cub'%(i+1), 'w') as f:
                f.writelines(cubefile)

            P0graph = nx.subgraph(cds.G, P0)
            cubefile, mapfile = cf.write_mapandcubes(P0graph, cds.delta, cds)
            with open('rc%s_P0_cubes.cub'%(i+1), 'w') as f:
                f.writelines(cubefile)

            print("Calculating homology")
            proc = subprocess.Popen([r'C:\Users\abel_\Downloads\chompfull_win32\bin\homcubes', '-i',
                                     'rc%s_P1_map.map'%(i+1), 
                                     'rc%s_P1_cubes.cub'%(i+1),
                                     'rc%s_P0_cubes.cub'%(i+1),
                                      "--log rc%s.log"%(i)], stdout=subprocess.PIPE)
            
            index_success.append(True)
            if verbosity:
                linenum = 0
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    print("           "+line.rstrip().decode('UTF-8'))

        except Exception as e:
            index_success.append(False)
            if verbosity:
                if e=='too many values to unpack (expected 4)':
                    print('Failed')
                else:
                    print(e)
    return index_success