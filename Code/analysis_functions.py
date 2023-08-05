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


def find_analytic_fixed_points(W_hh, b, W_ih, I, tol=10**-4):
    """
    Takes as argument all the parameters of the recurrent part of the model (W_hh, b) with a possible input I that connects into the RNN throught the weight matrix W_ih
    """
    fixed_point_list = []
    stabilist = []
    unstabledimensions = []
    Nrec = W_hh.shape[0]
    
    subsets = powerset(range(Nrec))
    for support in subsets:

        if support == ():
            continue
        r = np.array(support)
        
        #invert equation
        fixed_point = np.zeros(Nrec)
        fpnt_nonzero = -np.dot(np.linalg.inv(W_hh[r[:,None], r]-np.eye(len(support))), b[r] + np.dot(W_ih[r,:], I))
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
    return fixed_point_list, stabilist, unstabledimensions



def lu_step(x, W, b):
    return x*W+b

def relu_step(x, W, b):
    res = np.array(np.dot(W,x)+b)
    res[res < 0] = 0
    return res
 
def relu_step_input(x, W, b, W_ih, I):
    res = np.array(np.dot(W,x) + b + np.dot(W_ih, I))
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
    #b is unused, but there for consistency with relu jac
    
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
    Computes the index pair and the conley index for a given list of recurrent sets.
    
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