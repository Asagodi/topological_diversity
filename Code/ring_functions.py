# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:50:25 2023

@author: 
"""
import glob
import os, sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir) 

import numpy as np

import math
import scipy
from scipy.integrate import odeint, DOP853, solve_ivp
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA

import sklearn
import sklearn.decomposition
from scipy.stats import special_ortho_group
from scipy.optimize import minimize
from itertools import chain, combinations, permutations

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, rc
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines
from matplotlib.ticker import LinearLocator
import matplotlib.colors as mplcolors
import matplotlib.cm as cmx

from plot_losses import get_hidden_trajs, plot_output_trajectory
from odes import *


def ReLU(x):
    return np.where(x<0,0,x)

def sigmoid(x,g=1):
    return 1/(1+np.exp(-g*x))

def get_theory_weights(N):
    #𝑤𝑖𝑗~ cos(𝜃𝑖 − 𝜃𝑗 )
    x = np.arange(0,N,1)
    row = np.cos(2*np.pi*x/N)
    W = scipy.linalg.circulant(row)
    return W

def sigmoid_ode(t,x,W,b,tau,g=1):
    return (-x+sigmoid(np.dot(W,x)+b,g=g))/tau


def lax_ode(t,x):
    #two crossing line attractors
    xy = x[0]*x[1]
    gxy = g_lax(x[0],x[1])
    x1 = gxy[0]*xy
    x2 = gxy[1]*xy
    return [x1,x2]

def g_lax(x,y):
    if x>0 and y>0:
        return [-1,-1]
    elif x>0 and y<0:
        return [-1,1]
    elif x<0 and y>0:
        return [1,-1]
    elif x<0 and y<0:
        return [1,1]
    else:
        return [0,0]
        
    
    


#Noorman ring


# symmetric cosine weight matrix W sym jk = JI + JE cos(theta_j - theta_K)
# where JE and JI respectively control the strength of the tuned and untuned components of recurrent connectivity between neurons with preferred headings theta_j and theta_k.

# For a network of size N , there are N 3 such “optimal” values of local excitation J*E

# The parameters (JI, JE) can be set such that this system will generate a population profile that qualitatively looks like a discretely sampled “bump” of activity.
# (JI, JE) are within the subset  \Omega = \OmegaJI\times\OmegaJE \subset (−1, 1) \times (2,1)

def get_noorman_symmetric_weights(N, J_I = 1, J_E = 1):
    # W sym jk = JI + JE cos(theta_j - theta_K)
    x = np.arange(0,N,1)
    row = J_I + J_E*np.cos(2*np.pi*x/N)
    W = scipy.linalg.circulant(row)
    return W


# W asym jk =sin(theta_j - theta_k)
def get_noorman_asymmetric_weights(N):
    # W asym jk =sin(theta_j - theta_k)
    x = np.arange(0,N,1)
    row = np.sin(2*np.pi*x/N)
    W = scipy.linalg.circulant(row)
    return W


def noorman_ode(t,x,tau,transfer_function,W_sym,W_asym,c_ff,N,v_in):
    """Differential equation of head direction network in Noorman et al., 2022. 
    tau: integration constant
    transfer_function: each neuron transforms its inputs via a nonlinear transfer function
    W_sym, W_asym: symmetric and asymmetric weight matrices
    v_in: input
    c_ff: a constant feedforward input to all neurons in the network
    N: number of neurons in the network
    """
    return (-x + np.dot(W_sym+v_in(t)*W_asym, transfer_function(x))/N + c_ff)/tau


#Bump perturbations
def noorman_ode_pert(t,x,tau,transfer_function,W_sym,W_asym,c_ff,N,center,rotation_mat,amplitude,b):
    """
    create ODE for Noorman ring attractor with a local bump perturbation
    center,rotation_mat,amplitude,b are set
    """
    vector_bump = bump_perturbation(x, center, rotation_mat, amplitude, b)
    noor = noorman_ode(t,x,tau,transfer_function,W_sym,W_asym,c_ff,N)
    return noor + vector_bump


def noorman_ode_Npert(t,x,tau,transfer_function,W_sym,W_asym,c_ff,N,Nbumps,bumps):
    """
    create ODE for Noorman ring attractor with Nbumps local bump perturbations
    for each bump: center,rotation_mat,amplitude,b are random
    """
    noorode = noorman_ode(t,x,tau,transfer_function,W_sym,W_asym,c_ff,N)
    for bi in range(Nbumps):
        bump_i = np.random.randint(bumps.shape[0]) 
        roll_j = np.random.randint(N)
        center = np.roll(bumps[:,bump_i], roll_j).copy()
        rotation_mat = special_ortho_group.rvs(N)
        amplitude = np.random.rand()
        b = np.random.rand()
        noorode += bump_perturbation(x, center, rotation_mat, amplitude, b)

    return noorode


# Fixed points and their stabilities
def noorman_jacobian(x, W_sym):
    N = W_sym.shape[0]
    
    r = np.where(x>0)
    W_sub = np.zeros((N,N))
    W_sub[:,r] = W_sym[:,r]
    J = -np.eye(N)
    J += W_sub/N
    return J


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def noorman_fixed_points(W_sym, c_ff):
    """
    Takes as argument all the parameters of the recurrent part of the model (W_sym, c_ff)
    \dot x = -x + 1/N W_sym ReLU(x) + c_ff = 0
    """
    fixed_point_list = []

    N = W_sym.shape[0]
    subsets = powerset(range(N))
    for support in subsets:
        if support == ():
            continue
        r = np.array(support)
        
        W_sub = np.zeros((N,N))
        W_sub[:,r] = W_sym[:,r]
        A = W_sub/N - np.eye(N)
        fixed_point = -np.dot(np.linalg.inv(A), np.ones(N)*c_ff)
        
        #check true fixed point
        negativity_condition = True
        for i in r:
            if fixed_point[i] <= 0:
                negativity_condition = False
        for i in [item for item in range(N) if item not in r]:
            if fixed_point[i] >= 0:
                negativity_condition = False
        
        if negativity_condition:
            fixed_point_list.append(fixed_point)
        
    fixed_point_array = np.array(fixed_point_list)
    return fixed_point_array

def noorman_speed(x,tau,transfer_function,W_sym,W_asym,c_ff,N):
    f_x = noorman_ode(0,x,tau,transfer_function,W_sym,W_asym,c_ff,N)
    return np.linalg.norm(f_x)



def bump_perturbation(x, center, rotation_mat, amplitude, b=1):
    """
    Perturbation is composed of parallel vector field 
    with the location given by center, 
    the norm of the vectors determined by a bump function
    and the orientation given by theta
    
    x.shape = (Numberofpoints,N)
    rotation_mat: orientation of perturbation
    implemented for N-dimensional systems
    """
    N = x.shape[0]
    vector_bump = np.zeros(N)
    vector_bump[0] = 1.
    # rotation_mat = special_ortho_group.rvs(N)
    vector_bump = np.dot(vector_bump, rotation_mat)
    vector_bump = np.multiply(vector_bump, bump_function(x, center=center, amplitude=amplitude, b=b))
    
    return vector_bump

# we will take phi(·) to be threshold linear



################
#Ring, bumps and corners
def get_corners(N, m):
    #works for even N
    corners = []
    corner_0 = np.array([m]*N)
    corner_0[int(N/2):] *= -1
    corner_0[int(N/2)-int(N/4):int(N/2)] = 0
    corner_0[N-int(N/4):] = 0
    for support_j in range(N):
        corners.append(np.roll(corner_0, support_j))
    corners = np.array(corners)
    return corners

def get_bumps_along_oneside_ring(N, m, corners, step_size=0.1):
    x = np.arange(0, m+step_size, step_size)
    n_xs = x.shape[0]
    bumps = np.zeros((N, n_xs))
    for i, x_i in enumerate(x):
        for j in range(N):
            bumps[j,i] = np.interp(x_i, [0,m], [corners[0][j],corners[1][j]])
    return bumps

def get_all_bumps(N, bumps):
    all_bumps = []
    for support_j in range(N):
        for bump_i in range(bumps.shape[1]):
            all_bumps.append(np.roll(bumps[:,bump_i], support_j))
    all_bumps = np.array(all_bumps)
    return all_bumps




# 2D Ring attractor

def ring_ode(t,x):
    r = x[0]
    return [r*(1-r), 0]

def ring_ode_xy(t,x):
    x1, x2 = x
    hypot = np.sqrt(x1**2+x2**2)
    theta = np.arctan2(x2, x1)
    return [hypot*(1-hypot)*np.cos(theta), hypot*(1-hypot)*np.sin(theta)]

def ring_ode_xy_globalpert(t,x,W):
    x1, x2 = x
    hypot = np.sqrt(x1**2+x2**2)
    theta = np.arctan2(x2, x1)
    return [hypot*(1-hypot)*np.cos(theta)+W[0,0]*x1+W[0,1]*x2, hypot*(1-hypot)*np.sin(theta)+W[1,0]*x1+W[1,1]*x2]

def bump_function(x, center=0, amplitude=1, b=1):
    x_ = x-center

    if x.ndim == 1:
        if np.linalg.norm(x_)<b:
            return amplitude*np.exp(b**2/(np.linalg.norm(x_)**2-b**2))
        else:
            return 0
    else:
        bump = np.zeros(x.shape[0])
        support = np.where(np.linalg.norm(x_, axis=1)<b)[0]
        bump[support] = amplitude*np.exp(b**2/(np.linalg.norm(x_[support], axis=1)**2-b**2))
        return bump


def ring_ode_xy_pert(t,x,center,theta_bump,amplitude,b):
    x1, x2 = x
    hypot = np.sqrt(x1**2+x2**2)
    theta = np.arctan2(x2, x1)
    vector_bump = bump_perturbation_2d(x, center, theta_bump, amplitude, b=b)
    return [hypot*(1-hypot)*np.cos(theta)+vector_bump[0], hypot*(1-hypot)*np.sin(theta)+vector_bump[1]]


    
def bump_perturbation_2d(x, center, theta, amplitude, b=1):
    """
    Perturbation is composed of parallel vector field 
    with the location given by center, 
    the norm of the vectors determined by a bump function
    and the orientation given by theta
    
    theta: orientation of perturbation
    implemented for 2-dimensional systems
    """
    vector_bump = np.array([1, 0])
    vector_bump = np.multiply(np.dot(vector_bump, get_rotation_matrix(theta)), bump_function(x, center=center, amplitude=amplitude, b=b))
    
    return vector_bump


def get_rotation_matrix(theta):
    return np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])

def bump_perturbation(x, center, theta, amplitude, b=1):
    """
    Perturbation is composed of parallel vector field 
    with the location given by center, 
    the norm of the vectors determined by a bump function
    and the orientation given by theta
    
    theta: orientation of perturbation
    implemented for 2-dimensional systems
    """
    vector_bump = np.tile(np.array([1, 0]),(x.shape[0],1))

    vector_bump = np.multiply(np.dot(vector_bump, get_rotation_matrix(theta)), bump_function(x, center=center, amplitude=amplitude, b=b).reshape((-1,1)))
    
    return vector_bump


def ring_ode_xy_globalpert_speed(x, W):
    x1, x2 = x
    hypot = np.sqrt(x1**2+x2**2)
    theta = np.arctan2(x2, x1)
    return np.linalg.norm([hypot*(1-hypot)*np.cos(theta)+W[0,0]*x1+W[0,1]*x2, hypot*(1-hypot)*np.sin(theta)+W[1,0]*x1+W[1,1]*x2])

def cartesian_to_spherical(x, y):
    r = np.sqrt(x**2 + y**2)  # radial distance
    theta = np.arctan2(y, x)    # angle in radians
    return r, theta

def spherical_to_cartesian(r, theta):
    x1 = r*np.cos(theta)  
    x2 = r*np.sin(theta)    
    return np.array([x1, x2])


def define_ring(N, je = 4, ji = -2.4, c_ff=1):
    W_sym = get_noorman_symmetric_weights(N, ji, je)
    W_asym = get_noorman_asymmetric_weights(N)
    
    return W_sym, W_asym


def v_constant(value=1):
    #constant input
    return  lambda t : value

def v_zero(t):
    #zero input
    return 0

def v_switch(values=[0,1], t_switch=10):
      return  lambda t : values[0] if t<t_switch else values[1]

    
def get_v(v_name='zero', value=1, values=[0,1], t_switch=10):
    if v_name=='zero':
        return v_zero
    elif v_name=='constant':
        return v_constant(value=value)
    elif v_name=='switch':
        return v_switch(t_switch=t_switch, value=value)

def simulate_ring(W_sym, W_asym, c_ff, tau=1, y0=None, transfer_function=ReLU, v_in=v_zero,
                  maxT=25, tsteps=501):
    
    N = W_sym.shape[0]
    if not np.any(y0):
        y0 = np.random.uniform(0,1,N)
    t = np.linspace(0, maxT, tsteps)
    sol = solve_ivp(noorman_ode, y0=y0,  t_span=[0,maxT],
                    args=tuple([tau, transfer_function, W_sym, W_asym, c_ff, N, v_in]),
                    dense_output=True)
    
    return sol.sol(t),t

def simulate_from_y0s(y0s, W_sym, W_asym, c_ff, tau=1, 
                   transfer_function=ReLU,  v_in=v_zero,
                   maxT = 25, tsteps=501):
    
    N = W_sym.shape[0]
    t = np.linspace(0, maxT, tsteps)
    sols = np.zeros((y0s.shape[1], t.shape[0], N))
    for yi,y0 in enumerate(y0s.T):
        sol = solve_ivp(noorman_ode, y0=y0,  t_span=[0,maxT],
                        args=tuple([tau, transfer_function, W_sym, W_asym, c_ff, N, v_in]),
                        dense_output=True)
        sols[yi,...] = sol.sol(t).T
            
    return sols

def simulate_bumps(bumps_oneside, W_sym, W_asym, c_ff, tau=1, 
                   transfer_function=ReLU,  v_in=v_zero,
                   maxT = 25, tsteps=501):
    
    N = bumps_oneside.shape[0]
    t = np.linspace(0, maxT, tsteps)
    sols = np.zeros((bumps_oneside.shape[1], N, t.shape[0], N))
    for bump_i in range(bumps_oneside.shape[1]):
        for support_j in range(N):
            y0 = np.roll(bumps_oneside[:,bump_i], support_j)
            sol = solve_ivp(noorman_ode, y0=y0,  t_span=[0,maxT],
                            args=tuple([tau, transfer_function, W_sym, W_asym, c_ff, N, v_in]),
                            dense_output=True)
            sols[bump_i,support_j,...] = sol.sol(t).T
            
    return sols

def plot_ring(sols, corners, pca=None, lims = 3.):
    """
    Plots ring (from corners) and solutions (initialized close to ring)

    Parameters
    ----------
    sols : (#bumps along side, N, #time points, N) 
        solutions that with initial values (close to) ring
    corners : 
        corners of the ring.
    lims : float, optional
        lims of plot. The default is 3

    Returns
    -------
    pca : sklearn.decomposition.PCA
        pca of the sols. First 2 components

    """
    
    N = sols.shape[-1]
    sols = sols.reshape((-1, sols.shape[-2], N))
    if not pca:
        pca = sklearn.decomposition.PCA(n_components=2)
        X_proj2 = pca.fit_transform(sols[:,-1,:]) 
    else:
        X_proj2 = pca.transform(sols[:,-1,:]) 

    corners_proj2 = pca.transform(corners)
    # all_bumps_proj2 = pca.fit_transform(all_bumps) 
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_ring_from_corners(N, corners_proj2, ax)
        

    ax.plot(X_proj2[:,0], X_proj2[:,1], '.r', zorder=10, alpha=1., markersize=20)
    # ax.plot(all_bumps_proj2[:,0], all_bumps_proj2[:,1], '.r', zorder=10, alpha=1., markersize=20)

    ax.set(xlim=(-lims, lims), ylim=(-lims,lims))
    ax.set_axis_off()

    return pca

def plot_ring_from_corners(N, corners_proj2, ax, color='k', zorder=0, alpha=1., linewidth=10):
    for i in range(N):
        ax.plot([corners_proj2[i-1,0], corners_proj2[i,0]],
                [corners_proj2[i-1,1], corners_proj2[i,1]],
                color=color, zorder=zorder, alpha=alpha, linewidth=linewidth, 
                solid_capstyle='round')
    
    
def plot_solution(sol, corners, pca, lims = 3.):
    """
    Plots ring (from corners) and solution 
    
    Parameters
    ----------
    sol : (#time points, N) 
        solutions that with initial values (close to) ring
    corners : 
        corners of the ring.
    pca : sklearn.decomposition.PCA
        pca of the sols. First 2 components
    lims : float, optional
        lims of plot. The default is 3

    Returns
    -------
    """
    tsteps=sol.shape[0]
    N = sol.shape[-1]
    X_proj2 = pca.transform(sol) 
    corners_proj2 = pca.transform(corners)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_ring_from_corners(N, corners_proj2, ax, alpha=.2)
        
    cmap = cm.get_cmap("jet");
    norm = mplcolors.Normalize(vmin=0, vmax=tsteps)
    norm = norm(np.linspace(0, tsteps, num=tsteps, endpoint=False))
    ax.scatter(X_proj2[:,0], X_proj2[:,1], c=norm[np.arange(tsteps)], cmap=cmap, zorder=10, alpha=.5, s=5)
    ax.set(xlim=(-lims, lims), ylim=(-lims,lims))
    ax.set_axis_off()
    
    theta = np.arctan2(X_proj2[:,1], X_proj2[:,0])
    
    return fig, ax, theta


def plot_ring_and_fixedpoints(W_sym, pca, eps, c_ff, corners, lims=3, ax=None,  markersize=20):
    
    N = W_sym.shape[0]
    fixed_points = noorman_fixed_points(W_sym+eps, c_ff)
    fixed_points_proj2 = pca.transform(fixed_points) 
    corners_proj2 = pca.transform(corners)

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    n_stab=0
    n_sadd=0
    for f_i, fixed_point_p in enumerate(fixed_points_proj2[:-1]):
        fixed_point = fixed_points[f_i]
        eigenvalues, _ = scipy.linalg.eig(noorman_jacobian(fixed_point, W_sym+eps))
        if np.all(np.real(eigenvalues)<0):
            ax.plot(fixed_point_p[0], fixed_point_p[1], '.g', label="Analytical", zorder=99, alpha=1., markersize= markersize)
            n_stab+=1
        else:
            ax.plot(fixed_point_p[0], fixed_point_p[1], '.', color='darkorange', label="Analytical", zorder=99, alpha=1., markersize= markersize)
            n_sadd+=1

    for i in range(N):
        ax.plot([corners_proj2[i-1,0], corners_proj2[i,0]],
                [corners_proj2[i-1,1], corners_proj2[i,1]],
                'k', label="Original attractor", zorder=0, alpha=1., linewidth=10, 
                solid_capstyle='round')

    ax.set(xlim=(-lims, lims), ylim=(-lims,lims))
    ax.set_axis_off()

    ax.set_axis_off()
    if not ax:
        plt.savefig(current_dir+f"/Stability/figures/noorman_ring_N{N}_pert_{n_stab}stab_{n_sadd-1}sadd.pdf", bbox_inches="tight")


#lowrank
gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(200)
gauss_points = gauss_points*np.sqrt(2)

def phi_prime(mu, delta0):
    integrand = 1 - (np.tanh(mu+np.sqrt(delta0)*gauss_points))**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

def transf(K):
    return(K*phi_prime(0, np.dot(K.T, K)))

def ode_lowrank(t, x, Sigma):
    "Mastrogiuseppe 2018 (ring:Sigma=sigma[[1,0],[0,1]]"
    return - x + np.dot(Sigma, transf(x))


def ring_nef(N, j0, j1):
    "Barak 2021"
    x = np.arange(0,N,1)
    row = np.cos(-2*np.pi*x/N-np.pi)
    W = j0 + j1*scipy.linalg.circulant(row)
    
    return W


def simulate_nefring(W, I_e, y0=None, maxT=25, tsteps=501, tau=1):
    
    N = W.shape[0]
    if not np.any(y0):
        y0 = np.random.uniform(0,10,N)
    t = np.linspace(0, maxT, tsteps)
    sol = solve_ivp(relu_ode, y0=y0,  t_span=[0,maxT],
                    args=tuple([W, I_e, tau]),
                    dense_output=True)
    return sol.sol(t)



# row = sol.sol(t)[:,-1]; sols = scipy.linalg.circulant(row)
# for i in range(N):
#     new_sols[i,:] = simulate_nefring(W, I_e, y0=sols[i,:], maxT=25, tsteps=501)[:,-1]


def pca_transform(X, number_of_pcs):
    X_standardized = (X - np.mean(X, axis=0))
    covariance_matrix = np.cov(X_standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    top_eigenvectors = eigenvectors[:, :number_of_pcs]
    transformed_data = np.dot(X_standardized, top_eigenvectors)
    
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    return top_eigenvectors, transformed_data

from perturbed_training import RNN
###########RNNs and learning
def get_noormanring_rnn(N, je=4, ji=-2.4, c_ff=1, dt=1, internal_noise_std=0):
    
    num_of_inputs = 11
    
    wrec_init, wi_init = define_ring(N, je=je, ji=ji, c_ff=c_ff)     # W_sym, W_asym 
    bwo_init = np.zeros((2))
    corners = get_corners(N, m=1.2512167690384801)
    h0_init = corners[2] #along the ring, e.g. corner
    # h0_init = np.broadcast_to(h0_init, (num_of_inputs,) + h0_init.shape)
    top_eigenvectors, X_transformed = pca_transform(corners, number_of_pcs=2)
    wo_init = np.real(top_eigenvectors.T)

    dims = (1,N,2)
    net = RNN(dims=dims, noise_std=internal_noise_std, dt=dt,
              nonlinearity='relu', readout_nonlinearity='id',
              wi_init=wi_init/N, wrec_init=wrec_init/N, wo_init=wo_init, brec_init=np.array([c_ff]*N), bwo_init=bwo_init,
              h0_init=h0_init, ML_RNN='noorman')
    net.map_output_to_hidden = False
    
    input_range=(-1.,1.)
    T = 500
    #TODO: update to current def
    # res = get_hidden_trajs(net, T=T, dt_task=dt, input_length=T, 
    #                      num_of_inputs=num_of_inputs, input_range=input_range,
    #                      input_type='constant', random_angle_init=False, task=None)
    
    trajectories, traj_pca, h0_pca, input, target, output, input_proj, pca, explained_variance = res
    xylims=[-1.5,1.5]
    cmap = cmx.get_cmap("coolwarm")
    plot_output_trajectory(trajectories, wo_init.T, input_length=T, plot_traj=True,
                               fxd_points=None, ops_fxd_points=None,
                               plot_asymp=True, limcyctol=1e-2, mindtol=1e-4, ax=None, xylims=xylims,
                               cmap=cmap)
    
    

    
def make_connection_matrix(N, W0, R, ell):
    #construct a connection matrix as in Couey 2013
    theta = np.zeros([N])
    theta[0:N:2] = 0
    theta[1:N:2] = 1
    # theta[2:N:4] = 2
    # theta[3:N:4] = 3

    theta = 0.5*np.pi*theta
#    theta = arange(N)*2.*pi/float(N) - pi
    theta = scipy.ravel(theta)
    xes = np.zeros(N)
    for x in range(N):
      xes[x] = x
    
    Wgen = np.zeros([N,N], dtype=bool)
    for x in range(N):
      xdiff = abs(xes-x-ell*np.cos(theta))
      xdiff = np.minimum(xdiff, N-xdiff)
      Wgen[xdiff<R,x] = 1
    W = np.zeros([N,N])
    W[Wgen>0.7] = -W0
    return W, theta

def sim_dyn_one_d_wo(N, extinp, inh, R, dtinv,
              tau, time, ell, alpha, data, dt):
    W, theta = make_connection_matrix(N, inh, R, ell)
    W = scipy.sparse.csc_matrix(W)
    S = np.zeros(N)
    for i in range(N):
      if(np.random.rand()<0.5):
        S[i] = np.random.rand()  
    activities = np.zeros([N,time])
    #Stemp = np.zeros(N)
    vs = []
    thetas = []
    angle_i = 0
    for t in range(0, time, 1):
        if t % 25 == 0 and angle_i+1 < len(data): 
            theta_t = data[angle_i] #- data[angle_i-1] + .5*pi
            v = np.abs(data[angle_i-1]  - data[angle_i+1])
            if v > np.pi:
                v = 2*np.pi - v
            v *= 40
            angle_i += 1 
            vs.append(v)
            thetas.append(np.cos(theta_t - theta))
        S = S + 1./(dtinv+tau) * (-S + np.maximum(0., extinp+S*W + alpha * v * np.cos(theta_t - theta)))
        S[S<0.00001] = 0.
        activities[:,t] = S
    return activities, vs, thetas

def sim_dyn_one_d(N, extinp, inh, R, dtinv,
              tau, time, ell, alpha, data, dt):
    """perform simulation as in Couey 2013
    N: number of neurons
    extinp: external input
    inh: inhibitory connection strength
    R: radius of inhibitory connections
    dtinv: inverse of step size
    tau: 
    time: to perform the simulation
    ell: shift of neuron preference for direction
    alpha: coupling to head direction
    data: positions: posx, posy
    dt: timestep size to calculate head direction and velocity 
    """
    W, theta = make_connection_matrix(N, inh, R, ell)
    W = sparse.csc_matrix(W)
#    posx = data[0]
#    posy = data[1]

    S = zeros(N)
    for i in range(N):
      if(rand()<0.5):
        S[i] = rand()
        
    activities = zeros([N,time])
    Stemp = zeros(N)
    vs = []
    thetas = []
    angle_i = 0
    
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=15, metadata=dict(title=''))
    fig = plt.figure(2)
    with writer.saving(fig, 'mav2.mp4', 300):
        for t in range(0, time, 1):
        ##use if animal position is given:
#        if t % 100 == 0 and t + 500 < time:
#            tn = t / 5
#            v = 10*np.sqrt((posy[tn+50]-posy[tn-50])**2 + (posx[tn+50]-posx[tn-50])**2)
#            theta_t = arctan2( posy[tn+dt]-posy[tn-dt], posx[tn+dt]-posx[tn-dt]) + 0.5*pi*theta
            
            ##use if animal head direction is given:
            if t % 25 == 0 and angle_i+1 < len(data): 
                theta_t = data[angle_i] #- data[angle_i-1] + .5*pi
                v = np.abs(data[angle_i-1]  - data[angle_i+1])
                if v > pi:
                    v = 2*pi - v
                v *= 40
                angle_i += 1 
                vs.append(v)
                thetas.append(cos(theta_t - theta))
    
            S = S + 1./(dtinv+tau) * (-S + maximum(0., extinp+S*W + alpha * v * cos(theta_t - theta)))
            S[S<0.00001] = 0.
            activities[:,t] = S
#            if (10*t) % (time) == 0:
#                print("Process:" + str(100*t/time) + '%')
#            
            if(t<5000 and mod(t,10)==0):
                plt.clf()
                ax = plt.subplot(2,2,1)
                ax.plot(S, '-')
#                plt.plot([ni, ni], [min(S),max(S)], '-', color='red')
#                plt.ylabel('neural activity')
#                plt.xlabel('neurons')
                ax = fig.add_subplot(2,2,2)
                ang=theta_t
                x0 = cos(ang)*0.5
                y0 = sin(ang)*0.5
                ax.plot([0,x0], [0,y0])
                ax.axis([-0.5, 0.5, -0.5, 0.5])
                writer.grab_frame()
                plt.xlabel('heading direction')
    return activities, vs, thetas


def make_burak_1d(N, inh, R, ell, lambda_net):
    theta = np.zeros([N])
    theta[0:N:2] = 0
    theta[1:N:2] = 2
    theta = 0.5*np.pi*theta
    theta = scipy.ravel(theta)
    
    beta = 3./(lambda_net**2)
    gamma = 1.05*beta
    W = np.zeros([N,N])
    for x in range(N):
        for y in range(N):
            xdiff = abs(x-y-ell*np.cos(theta[y]))
#            print(x, y, xdiff)
            W[x, y] = np.exp(-gamma*xdiff) - np.exp(-beta*xdiff)
#            print(W[x, y])
#
    return W, theta

def make_burak(n, inh, R, ell, lambda_net=13, a=1):
    N = n**2
    submatrix = np.array([[0, 1], [2, 3]])*0.5*np.pi
    theta = np.kron(np.ones((N//2, N//2)), submatrix)
    theta = np.ravel(theta)
    
    beta = 3./(lambda_net**2)
    gamma = 1.05*beta
    W = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            x_i = np.array([i%n,i//n])
            x_j = np.array([j%n,j//n])
            xdiff = x_i-x_j-ell*np.cos(theta[j])
#            print(x, y, xdiff)
            W[i, j] = a*np.exp(-gamma*np.linalg.norm(xdiff)) - np.exp(-beta*np.linalg.norm(xdiff))
#            print(W[x, y])
#
    return W, theta

def sim_burak(N, extinp, inh, R, umax, dtinv,
              tau, time, ell, alpha, lambda_net):
    W, theta = make_burak(N, inh, R, ell, lambda_net)
    W = scipy.sparse.csc_matrix(W)

    S = np.zeros(N)
    ## generate random activity (doesn't matter much)
    for i in range(N):
      if(np.random.rand()<0.5):
        S[i] = np.random.rand()
        
    activities = np.zeros([N,time])
    #Stemp = np.zeros(N)
    for t in range(0, time, 1):
        v = .0
        if t % 10000 == 0:
            theta_t = 0.5*np.pi*np.random.choice([0])
#            print(theta_t,  alpha * v* cos(theta_t - theta))
        S = S + 1./(dtinv+tau) * (-S + np.maximum(0., extinp+S*W + alpha * v * np.cos(theta_t - theta)))
        S[S<0.00001] = 0.
        activities[:,t] = S
        if 10*t % time == 0:
            print("Process:" + str(100*t/time) + '%')
    S = scipy.ravel(S)
    return activities


def random_rank_1_matrix(N):
    m, n = N, N
    
    # Generate two random vectors
    v1 = np.random.rand(m)
    v2 = np.random.rand(n)
    
    # Compute the outer product of the two vectors
    matrix = np.outer(v1, v2)
    return matrix

def random_rank_r_matrix(N, r):
    if r > N:
        raise ValueError("Argument 'r' must not be smaller than 'N'")
    m, n = N, N
    
    # Generate two random matrices
    A = np.random.rand(m, r)
    B = np.random.rand(r, n)

    # Compute the matrix product of the two matrices
    matrix = np.dot(A, B)
    
    return matrix


def make_gaussian_connection_matrix_goodridge(N, sigma):
    #GOODRIDGE et al  2000
    #Modeling Attractor Deformation in the Rodent Head-Direction System
    x = np.arange(-N/2,N/2,1)*2*np.pi/N
    x = np.arange(-180,180,180/(1/2*N))
    row = np.exp(-x**2/sigma**2)
    W = scipy.linalg.circulant(row)
    return W


def make_gaussian_connection_matrix_compte(N, sigma, jmin, jplus):
    #Compte et al  2000
    #Synaptic Mechanisms and Network Dynamics Underlying Spatial Working Memory in a Cortical Network Model
    x = np.arange(-N/2,N/2,1)*2*np.pi/N
    x = np.arange(-180,180,180/(1/2*N))
    row = jmin + (jplus-jmin)*np.exp(-x**2/(2*sigma**2))
    W = scipy.linalg.circulant(row)
    return W

# N=10; W = make_gaussian_connection_matrix(N, sigma=28.5); W*=-1; b=1
# N=6; W = make_gaussian_connection_matrix(N, sigma=28.5); W*=-1; b=.1
# maxT = 10000
# tsteps = maxT+1
# t = np.linspace(0, maxT, tsteps)
# sols = np.zeros((Nsim, t.shape[0], N))
# for i in range(Nsim):
#     y0 = np.random.uniform(-1,1,N)

#     sol = solve_ivp(tanh_ode, y0=y0,  t_span=[0,maxT],
#                     args=tuple([W, b, tau]),
#                     dense_output=True)

#     trajectories = sol.sol(t)
#     sols[i,...] = trajectories.T
# plt.plot(sols[:,-1,:].T)
# velocity = np.abs(np.diff(sols, axis=1))
# speed = np.linalg.norm(velocity, axis=2)
# idx = np.argmin(speed[:,-1])
# x = np.arange(0, 1, 1/step_size)
# n_xs = x.shape[0]
# bumps = np.zeros((n_xs, N))
# for i, x_i in enumerate(x):
#     for j in range(N):
#         bumps[i,j] = (1-x_i)*sols[idx,-1,j] + x_i*np.roll(sols[idx,-1,:],1)[j]
    
# for i in range(N):
#     plt.plot(bumps[i,:])
    
# all_bumps = []
# for support_j in range(N):
# for bump_i in range(bumps.shape[0]):   
#         all_bumps.append(np.roll(bumps[bump_i,:], support_j))
# all_bumps = np.array(all_bumps)


def run_init(W, b, nonlin=tanh_ode, maxT=1000, tsteps=1001,  tau=10, Ntrials=100, n_components=2):
    N=W.shape[0]
    t = np.linspace(0, maxT, tsteps)
    sols = np.zeros((Ntrials, tsteps, N))
    for i in range(Ntrials):
        y0 = np.random.uniform(0,1,N)
    
        sol = solve_ivp(nonlin, y0=y0,  t_span=[0,maxT],
                        args=tuple([W, b, tau]),
                        dense_output=True)
        trajectories = sol.sol(t)
        sols[i,...] = trajectories.T
        
    pca = PCA(n_components=n_components)
    invariant_manifold = sols[:,-4:,:].reshape((-1,N))
    pca.fit(invariant_manifold)
    invariant_manifold = sols[:,-1,:].reshape((-1,N))
    sols_pca = pca.transform(invariant_manifold).reshape((Ntrials,-1,n_components))
    output_angle = np.arctan2(sols_pca[...,1], sols_pca[...,0])
    thetas = np.ravel(output_angle)
    idx = np.argsort(thetas)
    thetas_sorted = thetas[idx]
    thetas_sorted[-1] = np.pi
    vals = sols[:,-1,:].reshape((Ntrials,N))[idx]
    vals[-1] = vals[0]
    cs = scipy.interpolate.CubicSpline(thetas_sorted, vals, bc_type='periodic')
    thetas_init = np.arange(-np.pi, np.pi, np.pi/Ntrials*2);
    ring_points=cs(thetas_init)
    xs = np.arange(-np.pi, np.pi, np.pi/1000)
    csxapca=pca.transform(cs(xs))

    return ring_points, thetas_init, pca, csxapca

    

def perturb_and_simulate(W, b, nonlin=tanh_ode, tau=10,
                         rank=1, pert_type='rank', epsilon_step=1e-3,
                         Ntrials=100,
                         maxT=1000, tsteps=1001, 
                         Nsim=100, n_components=2,
                         y0s=None,
                         seed=1000,
                         theta_t=10,
                         patience=50,
                         plot=False,
                         folder=None
):
    
    assert folder is not None, "Provide 'folder' to save figure"
    # folder="C:/Users/abel_/Documents/Lab/Projects/topological_diversity/Stability/ring_perturbations/goodridge/pert"
    colors = colors=np.array(['k', 'r', 'g'])
    N=W.shape[0]
    t = np.linspace(0, maxT, tsteps)
    Nfxd_pnt_list = []
    perf_list = []
    eps_list = []
    Wpert_list = []
    fxd_pnt_thetas_list=[]
    stabilist=[]
    perf = 1
    if y0s is None:
        ring_points, thetas_init, pca, csxapca = run_init(W, b, nonlin=nonlin, tau=tau, maxT=maxT, tsteps=tsteps,
                                                          Ntrials=Ntrials, n_components=n_components)
    else:
        Ntrials = y0s.shape[0]
        pca = PCA(n_components=n_components)
        sols_pca = pca.fit_transform(y0s.reshape((-1,N)))
        thetas_init = np.arctan2(sols_pca[:,1], sols_pca[:,0]);
        thetas = np.ravel(thetas_init)
        idx = np.argsort(thetas)
        thetas_sorted = thetas[idx]
        thetas_sorted[-1] = np.pi
        vals = y0s[idx]
        vals[-1] = vals[0]
        cs = scipy.interpolate.CubicSpline(thetas_sorted, vals, bc_type='periodic')
        print(vals.shape)
        thetas_init = np.arange(-np.pi, np.pi, np.pi/Ntrials*2);
        ring_points=cs(thetas_init)
        xs = np.arange(-np.pi, np.pi, np.pi/1000)
        csxapca=pca.transform(cs(xs))
        
    np.random.seed(seed); 
    if pert_type=='rank':
        Wepsilon=random_rank_r_matrix(N, rank) 
    else:
        Wepsilon=np.random.normal(0,1,((N,N)))
    # Wepsilon/=np.linalg.norm(Wepsilon)

    t = np.linspace(0, theta_t, theta_t+11)
    same_Nfp_t=0
    prev_Nfp=12
    for j in range(0, Nsim):
        if perf==0:
            break
        epsilon=epsilon_step*j; 
        Wpert = W+epsilon*Wepsilon
        # Wpert = W+epsilon*random_rank_1_matrix(N)
        eps_list.append(epsilon)
        Wpert_list.append(Wpert)
        #for non-persistence bifurcation:
            #second positive eigenvalue?
        #np.random.seed(20000+20); epsilon=.000002*200+.01*j; Wepsilon = W+epsilon*np.random.normal(0,1,((N,N)))
        
        sols_pert = np.zeros((Ntrials, t.shape[0], N))
        for i in range(Ntrials):
            if y0s is None:
                y0 = ring_points[i,:]
            else:
                y0 = y0s[i,:]

            sol = solve_ivp(nonlin, y0=y0,  t_span=[0,theta_t], args=tuple([Wpert, b, tau]), dense_output=True)
            trajectories = sol.sol(t)
            sols_pert[i,...] = trajectories.T
        sols_pert_pca = pca.transform(sols_pert.reshape((-1,N))).reshape((Ntrials,-1,n_components))
        
        pca2 = PCA(n_components=n_components)
        pca2.fit(sols_pert[:,:,:].reshape((-1,N)))
        dim=1-np.sum(pca2.explained_variance_ratio_[:2])
        
        thetas = np.arctan2(sols_pert_pca[:,:,1], sols_pert_pca[:,:,0]);
        theta_unwrapped = np.unwrap(thetas, period=2*np.pi);
        theta_unwrapped = np.roll(theta_unwrapped, -1, axis=0);
        arr = np.sign(theta_unwrapped[:,-1]-theta_unwrapped[:,0]);
        idx=np.array([i for i, t in enumerate(zip(arr, arr[1:])) if t[0] != t[1]])
        stabilities=-arr[idx].astype(int)
        fxd_pnt_thetas = thetas_init[idx]
        fxd_pnts = np.array([np.cos(fxd_pnt_thetas), np.sin(fxd_pnt_thetas)]).T
        # fxd_pnts_pca = sols_pert_pca[idx+1, 0, :]
        fxd_pnts_pca = sols_pert_pca[idx+1, 0, :]

        if fxd_pnts.shape[0]%2==1:
            fxd_pnts=np.vstack([fxd_pnts,[-1,0]])
            stabilities=np.append(stabilities, -np.sum(stabilities))
            fxd_pnt_thetas = np.append(fxd_pnt_thetas, 0)
            fxd_pnts_pca=np.vstack([fxd_pnts_pca,pca.transform(cs(np.pi/3).reshape(1, -1))]) 
            
        Nfxd_pnts = fxd_pnts.shape[0]
        print(Nfxd_pnts)
        Nfxd_pnt_list.append(Nfxd_pnts)
        fxd_pnt_thetas_list.append(fxd_pnt_thetas)
        stabilist.append(stabilities)
        # print(Nfxd_pnts)
        
        boas = []
        for i in range(0,Nfxd_pnts,2):
            boa = (fxd_pnt_thetas[i+1]-fxd_pnt_thetas[i-1])% (2*np.pi)
            boas.append(boa/np.pi/2)
        if Nfxd_pnts>2:
            perf = -np.sum([boa*np.log(boa) for boa in boas])
        else:
            perf=0
        perf_list.append(perf)
        
        # print("Simulation ", j, " #fps: ", Nfxd_pnts, "Perf:", np.round(perf,3), "D: ", dim)
        # if dim<0.01 or dim>0.05 or Nfxd_pnts>12:
        #     break
        if same_Nfp_t > patience:
            break
        if j>2: 
            if Nfxd_pnts!=Nfxd_pnt_list[-2] and perf!=perf_list[-2]:
                same_Nfp_t=0
            else:
                same_Nfp_t+=1
        
        if plot:
            makedirs(folder)
            fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.4));
            # plt.scatter(csxapca[:,0]/np.linalg.norm(csxapca,axis=1), csxapca[:,1]/np.linalg.norm(csxapca,axis=1), s=.1);
            # plt.plot(csxapca[:,0]/np.linalg.norm(csxapca,axis=1), csxapca[:,1]/np.linalg.norm(csxapca,axis=1));
    
            # plt.scatter(fxd_pnts[:,0], fxd_pnts[:,1], color=colors[stabilities], zorder=1000)
            plt.scatter(fxd_pnts_pca[:,0], fxd_pnts_pca[:,1], color=colors[stabilities], zorder=1000)
            
            sols_pert_pca_ring = np.vstack([sols_pert_pca[:,-1,:], sols_pert_pca[0,-1,:]])
            plt.plot(sols_pert_pca_ring[:,0], sols_pert_pca_ring[:,1], 'b', alpha=1)

            # for i in range(Ntrials):
            #     # plt.plot(sols_pert_pca[i,:,0], sols_pert_pca[i,:,1], 'k',alpha=.1)

            #     plt.plot(sols_pert_pca[i,:,0]/np.linalg.norm(sols_pert_pca[i,:,:],axis=1),
            #              sols_pert_pca[i,:,1]/np.linalg.norm(sols_pert_pca[i,:,:],axis=1), 'k',alpha=.1)
            #     plt.plot(sols_pert_pca[i,-1,0]/np.linalg.norm(sols_pert_pca[i,-1,:],axis=0),
            #              sols_pert_pca[i,-1,1]/np.linalg.norm(sols_pert_pca[i,-1,:],axis=0), '.k', alpha=1)
            ax.set_axis_off()
            # ax.set_xlim([-1.1,1.1])
            # ax.set_ylim([-1.1,1.1])
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
            fig.savefig(folder+f"/ring_pert_eps{epsilon:.10f}_Nfp{Nfxd_pnts}.pdf")
            plt.show()
    
    return Nfxd_pnt_list, perf_list, fxd_pnt_thetas_list, Wpert_list, eps_list

# fig, ax = plt.subplots(1, 1, figsize=(4, 3)); ax.plot(eps_list, perf_list); plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)); ax.set_xlabel(r"$\epsilon$"); ax.set_ylabel("memory capacity");fig.savefig(folder+f"/performance.pdf", bbox_inches="tight")

from tqdm import tqdm
def N_perturb(W, b, Npert, rank, epsilon_step=1e-3, 
              nonlin=tanh_ode, tau=10, y0s=None,
              Ntrials=100, maxT=1000, tsteps=1001, Nsim=100, n_components=2, theta_t=10, patience=50,
              folder=None):
    # folder = "C:/Users/abel_/Documents/Lab/Projects/topological_diversity/Stability/ring_perturbations/goodridge/pert"
    all_lists = []
    for pert_i in tqdm(range(Npert)):
        # print(pert_i)
        Nfxd_pnt_list, perf_list, fxd_pnt_thetas_list, Wpert_list, eps_list = perturb_and_simulate(W, b, epsilon_step=epsilon_step,
                             nonlin=nonlin, tau=tau, y0s=y0s,
                             Ntrials=Ntrials, maxT=maxT, tsteps=tsteps, Nsim=Nsim, theta_t=theta_t, patience=patience,
                             rank=rank, seed=1000*pert_i+1000, folder=folder, plot=False)
        all_lists.append([Nfxd_pnt_list, perf_list, fxd_pnt_thetas_list, Wpert_list, eps_list])
    return all_lists

import pickle
def plot_perfs(ax=None, xmax=.5):
    
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3));
    folder = 'C:/Users/abel_/Documents/Lab/Projects/topological_diversity/Stability/ring_perturbations/goodridge/N6/rank_2'
    # with open(folder + '/all_lists.pickle', 'wb') as handle:
        # pickle.dump(all_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
    with open(folder + '/all_lists.pickle', 'rb') as handle:
        all_lists = pickle.load(handle)
        
    nsims = len(all_lists)
    perf_lists = [all_lists[i][1] for i in range(nsims)]
    eps_lists = [all_lists[i][4] for i in range(nsims)]
    max_len = max(len(lst) for lst in perf_lists)
    eps_list = np.arange(max_len)*eps_lists[0][1]
    # Pad each sublist with zeros until it reaches the maximum length:
    perf_array = np.array([lst + [0]*(max_len - len(lst)) for lst in perf_lists])
        
    mean_perf = np.mean(perf_array,axis=0)
    var_perf = np.var(perf_array,axis=0)
    
    plt.plot(eps_list, mean_perf); plt.fill_between(eps_list, mean_perf-var_perf, mean_perf+var_perf, alpha=.5);
    plt.xlabel(r"$\epsilon$"); plt.ylabel("Performance"); 
    plt.xlim([0,xmax]);
    
    #fig.savefig(folder+f"/performance.pdf", bbox_inches="tight")
    
    return ax


#############################LOW-RANK
sys.path.append("C:/Users/abel_/Documents/Lab/Software/LowRank/6_ContinuousAttractor")
import fct_mf as mf
import fct_simulations as sim
import fct_facilities as fac
import fct_integrals as integ
import time as tm
def make_low_rank_ring(Si=2, rho=1.6, g=2.1, N=4000):
    #ostojic, mastrogiuseppe
    R = g * sim.GetBulk (N)

    y1 = sim.GetGaussianVector( 0, 1, N) # Unit vectors required for the ran    k-two structure (see Methods)
    y2 = sim.GetGaussianVector( 0, 1, N)
    
    x1 = sim.GetGaussianVector( 0, 1, N)
    x2 = sim.GetGaussianVector( 0, 1, N)
    x3 = sim.GetGaussianVector( 0, 1, N)
    x4 = sim.GetGaussianVector( 0, 1, N)
    
    m1 = np.sqrt(Si**2 - rho**2)*x1 + rho*y1
    m2 = np.sqrt(Si**2 - rho**2)*x2 + rho*y2
    n1 = np.sqrt(Si**2 - rho**2)*x3 + rho*y1
    n2 = np.sqrt(Si**2 - rho**2)*x4 + rho*y2
    
    M = ( np.outer( m1 , n1 ) + np.outer( m2 , n2 ) ) / N
    J = R+M
    return J, n1, n2


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def simulate_low_rank_ring(Nsims=100, Ntrials=128, Si=2, rho=1.6, g=2.1, N=4000, init_type='gaussian', Ngrid=10):
    folder = f'C:/Users/abel_/Documents/Lab/Projects/topological_diversity/Stability/ring_perturbations/ostojic/N{N}_si{Si}_rho{rho}_g{g}/'
    makedirs(folder)
    stab_colors = ['k', 'r', 'g']
    Nfxpnts_list=[]
    np.random.seed(1111)
    Nsample = N
    
    ### Set parameters
    ParVec = [rho, Si]
    ### Compute DMF prediction
    # Stationary solution
    K1, K2, delta0_s = mf.SolveStatic ( [5., 5., 5.], g, ParVec )
    ic_0 = [ K1, K2, 1.*delta0_s, 0.5*delta0_s] 
    
    # Chaotic solution
    K1, K2, delta0_c, deltainf_c = mf.SolveChaotic ( ic_0, g, ParVec )
    radius_attractor = np.sqrt( K1**2 + K2**2 )

    x = np.linspace(-1, 1, Ngrid)
    y = np.linspace(-1, 1, Ngrid)
    # Create the 2D grid
    X, Y = np.meshgrid(x, y)
    inits = np.stack([X,Y]).T.reshape((x.shape[0]*y.shape[0], 2))

    for simi in range(Nsims):
        #Sample connection matrix
        J, n1, n2 = make_low_rank_ring(Si=Si, rho=rho, g=g, N=N)
        n_matrix = np.array([n1,n2])
        left_inverse = np.linalg.inv(n_matrix.T @ n_matrix) @ n_matrix.T
        
        # Simulation
        
        T = 7        # Total time of integration, expressed in time constants o    f single units
        deltat = 0.1
        t = np.linspace( 0, T, int(T/deltat) )
        x = np.linspace(0, 10, int(1e3))
        
        fg = plt.figure(figsize=(3,3))
        ax0 = plt.axes(frameon=True)
        K1_sim = np.zeros (( Ntrials, len(t) ))
        K2_sim = np.zeros (( Ntrials, len(t) ))
        Z_sample = np.zeros (( Ntrials, len(t), Nsample ))
        jumps = []
        speed_min_index_list = []
        for j in range(Ntrials):    
            if init_type == 'grid':
                h_init = inits[j]@n_matrix
            else:
                h_init = sim.GetGaussianVector( 0, 1, N)
            Z = sim.SimulateActivity ( t, h_init, J, I    =0 )
            Z_sample[j,:,:] = Z[:, 0:Nsample]
            K1_sim[j,:] = np.dot(np.tanh(Z), n1) / N
            K2_sim[j,:] = np.dot(np.tanh(Z), n2) / N
            
            thetas = np.arctan2(K2_sim[j,:10], K1_sim[j,:10]);
            rs = np.hypot(K1_sim[j,-19:], K2_sim[j,-19:]);

            sorted_indices=np.argsort(thetas);
            thetas_sorted = thetas[sorted_indices]
            theta_unwrapped = np.unwrap(thetas_sorted, period=2*np.pi);
            theta_unwrapped = np.roll(theta_unwrapped, -1, axis=0); 
            K = np.array([K1_sim[j,:], K2_sim[j,:]]).T
            # print(K.shape)
            velocity = np.abs(np.diff(K, axis=0))
            speed = np.linalg.norm(velocity, axis=1)
            value = np.max(speed[-1])
            plt.plot(velocity)
            # print(theta_unwrapped)|
            # index, value = find_biggest_acceleration(theta_unwrapped)
            jumps.append(value)
            index, _ = find_first_local_minimum(speed)
            speed_min_index_list.append(index)
        
        speed_min_index_list = np.array(speed_min_index_list)
        jidx = np.argsort(jumps)[::-1]
        
        # fg = plt.figure(figsize=(3,3))
        # ax0 = plt.axes(frameon=True)
        # for j in range(Ntrials):
        #     # plt.plot(K1_sim[jidx,:][j,:], K2_sim[jidx,:][j,:], color='k', alpha=1)  # Trajectory
        #     K = np.array([K1_sim[jidx,:][j,:], K2_sim[jidx,:][j,:]]).T
        #     velocity = np.diff(K, axis=0)
        #     speed = np.linalg.norm(velocity, axis=1)
        #     rs = np.hypot(K1_sim[j,:], K2_sim[j,:]);
        #     speed = np.linalg.norm(rs, axis=0)

        #     plt.plot(rs)
            
            
        T = 50
        deltat = 0.1
        t = np.linspace( 0, T, int(T/deltat) )  
        K1_sim_long = np.zeros (( Ntrials, len(t) ))
        K2_sim_long = np.zeros (( Ntrials, len(t) ))
        Z_sample_long = np.zeros (( Ntrials, len(t), Nsample ))  
        for j in range(Ntrials):
            Z = sim.SimulateActivity ( t, Z_sample[j,-1,:], J, I=0 )
            Z_sample_long[j,:,:] = Z[:, 0:Nsample]
            K1_sim_long[j,:] = np.dot(np.tanh(Z), n1) / N
            K2_sim_long[j,:] = np.dot(np.tanh(Z), n2) / N
        
        thetas = np.arctan2(K1_sim_long[:,:], K2_sim_long[:,:]);
        sorted_indices=np.argsort(thetas[:,0]);
        thetas_sorted = thetas[sorted_indices,:]
        theta_unwrapped = np.unwrap(thetas_sorted, period=2*np.pi);
        theta_unwrapped = np.roll(theta_unwrapped, -1, axis=0); 
        arr = np.sign(theta_unwrapped[:,-1]-theta_unwrapped[:,0]);
        
        
        fxd_pnt_idx=np.array([i for i, t in enumerate(zip(arr, arr[1:])) if t[0] != t[1]]);
        if fxd_pnt_idx.shape[0]>1:
            fxd_pnts_reduced = np.array([K1_sim_long[sorted_indices,0][fxd_pnt_idx+1], K2_sim_long[sorted_indices,0][fxd_pnt_idx+1]]).T
            fxd_pnts_full = np.array([Z_sample_long[sorted_indices,0,:][fxd_pnt_idx+1,:]]).squeeze()
            stabilities=-arr[fxd_pnt_idx].astype(int)
            Nfxpnts = fxd_pnts_reduced.shape[0]
            Nfxpnts_list.append(Nfxpnts)
        else:
            Nfxpnts = 0
            fxd_pnts = np.empty((0))
        print("Simulation #: ", simi, "#fps: ", Nfxpnts)
        

        
        fg = plt.figure(figsize=(3,3))
        ax0 = plt.axes(frameon=True)

        # T = 10
        # for fp in fxd_pnts_full:
        #     jacobian = compute_jacobian(fp, J)
        #     eigenvalues, eigenvectors = np.linalg.eig(jacobian)
        #     if np.max(np.real(eigenvalues)) > 0 :
        #         idx = np.argmax(np.real(eigenvalues))
        #         unstable = eigenvectors.T[idx]
        #         fp-unstable
        #         for sign in [-1,1]:
        #             # print(fp+sign*unstable)
        #             Z = sim.SimulateActivity ( t, np.real(fp+sign*unstable), J, I=0 )
        #             K1 = np.dot(np.tanh(Z), n1) / N
        #             K2 = np.dot(np.tanh(Z), n2) / N
        #             plt.plot(K1, K2, color='b')  # Trajectory
                
        
        # for j in range(Nfxpnts*2):
        #     # print(speed_min_index_list[jidx])
        #     plt.plot(K1_sim[jidx,:][j,speed_min_index_list[jidx][j]:],
        #              K2_sim[jidx,:][j,speed_min_index_list[jidx][j]:],
        #              color='k', alpha=1)  # Trajectory
        
        plt.plot(radius_attractor*np.cos(x), radius_attractor*np.sin(x), color = '0.8', linewidth = 3.5)

        sorted_indices = np.append(sorted_indices, sorted_indices[0])
        for j in range(Ntrials):
            plt.plot(K1_sim[j,:], K2_sim[j,:], color='b', alpha=.1)  # Trajectory
            # plt.plot(K1_sim_long[j,:], K2_sim_long[j,:], color='b')  # Trajectory
            plt.plot(K1_sim_long[sorted_indices,0], K2_sim_long[sorted_indices,0], color='b')  # Trajectory

            
       # plt.plot(K1_sim_long[j,0], K2_sim_long[j,0], 'o', color = '0' )  # In    itial condition
            #plt.plot(K1_sim_long[j,-1], K2_sim_long[j,-1], 'o', color = 'g' )  #     Final condition
            
        for i,fp in enumerate(fxd_pnts_reduced):
            if Nfxpnts==0:
                break
            plt.scatter(fp[0], fp[1],color=stab_colors[stabilities[i]], zorder=1000    )
        
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.yaxis.set_ticks_position('left')
        ax0.xaxis.set_ticks_position('bottom')
        plt.locator_params(nbins=5)
        
        #plt.xlim(-0.8, 0.8)
        #plt.ylim(-0.8, 0.8)
        
        plt.xticks([-0.8, 0, 0.8])
        plt.yticks([-0.8, 0, 0.8])
        
        plt.ylabel(r'$m^{(2)}$')
        plt.xlabel(r'$m^{(1)}$')
        #plt.legend(loc=2)
        now=tm.time()
        plt.savefig(folder+f'plane_m1m2_{Nfxpnts}fp_{int(now)}.pdf', bbox_inches='tight')
        plt.show()
    return Nfxpnts_list



def make_biswas_ring(w1, w4):
    assert w1<1 and w1>0.5, "w1>1 or w1<0.5"
    # assert w4<f(w1)
    w2=1-2*w1**2
    w3=w1*(4*w1**2-3)
    row = np.array([0, w1, w2, w3, w4, w3, w2, w1])
    W = scipy.linalg.circulant(row) 
    return W 
    


###########EMPJ
# import json
# folder='C:/Users/abel_/Documents/Lab/Projects/topological_diversity/Stability/ring_perturbations/martin/'
# with open(folder+'W_miracle.json', 'r') as f:
#     data = json.load(f)
#     W = np.array(data).T #need to transpose for consistency
# with open(folder+'D_miracle.json', 'r') as f:
#     data = json.load(f)
#     D = np.array(data)
    
def get_fps():
    xstars, xstars_2, stabilist = find_fixed_points_newton(W, 0, D, tau=10)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3));
    plt.scatter(xstars_2[:,0], xstars_2[:,1], color=colors[stabilist])
    plt.plot(ys[:,0], ys[:,1], '-b')
    plt.xlim([-1.2,1.2]);plt.ylim([-1.2,1.2]);plt.axis('off'); plt.savefig(figfolder+"/saddle_fps.pdf", bbox_inches="tight");
    
def get_empj(W, D, Nsims=100):
    
    thetas_init = np.arange(-np.pi, np.pi, 2*np.pi/Nsims)
    points_init_2d = np.array([np.cos(thetas_init), np.sin(thetas_init)])
    points_init_2_random = points_init_2d+np.random.normal(0,.01,(2,Nsims))
    points_init_nd_random = np.dot(D, points_init_2_random).T
    trajectories = simulate_network_ntimes(Nsims, W.T, 0, nonlinearity_ode=tanh_ode, tau=0.05, maxT=200, tsteps=50001,
                                           mlrnn=False, y0s=points_init_nd_random);
    traj_2 = np.dot(trajectories, D)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    for i in range(100):
        plt.plot(traj_2[i, :,0], traj_2[i,:,1], 'k');
        plt.plot(traj_2[i, 0,0], traj_2[i,0,1], '.r');
        plt.plot(traj_2[i, -1,0], traj_2[i,-1,1], '.b');
    #plt.savefig(folder+'/miracle/on_manifold_multi.pdf', bbox_inches="tight")
    
    traj = simulate_network(W, 0, nonlinearity_ode=tanh_ode, y0=None, maxT=20, tsteps=5001, tau=1000);
    traj_2 = np.dot(traj.T, D)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3)); 
    plt.plot(np.cos(np.append(thetas_init,np.pi)), np.sin(np.append(thetas_init,np.pi))); plt.xlim([-2,2]);
    plt.plot(traj_2[:,0], traj_2[:,1], 'k'); plt.plot(traj_2[0,0], traj_2[0,1], '.r');plt.plot(traj_2[-1,0], traj_2[-1,1], '.b');
    plt.ylim([-2,2]); plt.axis('off'); plt.savefig(folder+'/miracle/off_manifold_single.pdf', bbox_inches="tight")
    
def plot_weight_matrix(mat_to_plot, fig_name):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    vmax = np.max(np.abs(mat_to_plot))
    im = ax.imshow(mat_to_plot, cmap='seismic', vmin=-vmax, vmax=vmax)
    plt.axis('off'); 
    ax.set_position([0, 0, 1, 1]);
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.01)
    
    
    
def newton_method(x0, system, jacobian, W, b, tau, mlrnn, tol=1e-6, max_iter=100):
    """
    Search for (approximate) fixed points with Newton-Raphson.
    Parameters
    ----------
    x0 : starting point
    system : ode of rnn
    jacobian : jacobian of ode
    W : connection matrix
    b : bias
    tau : time constant
    mlrnn : Wf(x) or f(Wx)
    tol : float, optional
        tolerance level at which to stop the search. The default is 1e-6.
    max_iter : int, optional
        maximum number of iterations. The default is 100.

    Returns
    -------
    x : fixed point / slow point.
    """
    x = x0
    for _ in range(max_iter):
        dx = np.linalg.solve(jacobian(W,b,tau,x,mlrnn), system(0,x,W,b,tau,mlrnn))
        x -= dx
        if np.linalg.norm(dx) < tol:
            break
    return x
    
def find_fixed_points_newton(W, b, wo, 
                             tau=1, rnn_ode=tanh_ode, rnn_jacobian=tanh_jacobian, mlrnn=True,
                             npoints=1000, points_init_nd=None,
                             max_iter=1000, tol=1e-9):
    if not np.any(points_init_nd):
        thetas_init = np.arange(-np.pi, np.pi, 2*np.pi/npoints);
        points_init_2d = np.array([np.cos(thetas_init), np.sin(thetas_init)]);
        points_init_nd = np.dot(wo, points_init_2d).T;
    xstars = [];
    eigenvalues_list=[]
    stabilist=[]
    for x0 in tqdm(points_init_nd):
        # print(x0)
        xstar = newton_method(x0, rnn_ode, rnn_jacobian, W, b, tau, mlrnn=mlrnn, tol=tol, max_iter=max_iter)
        # print(np.all(x0==xstar))
        J=rnn_jacobian(W,b,tau,xstar)
        eigenvalues, eigenvectors = np.linalg.eig(J)
        xstars.append(xstar)
        eigenvalues_list.append(eigenvalues)
        if np.all(eigenvalues<0):
            stabilist.append(-1)
        else:
            stabilist.append(1)
    xstars_2 = np.dot(np.array(xstars), wo)
    return xstars, xstars_2, stabilist
    

# xstars, xstars_2 = find_fixed_points_newton(wrec, brec, wo, tau=10,
 # rnn_ode=tanh_ode, rnn_jacobian=tanh_jacobian, mlrnn=True, points_init_nd=csx2, max_iter=1000, tol=1e-6)]
 
def load_and_find(folder, num_points=100, tau=10,
                  max_iter=100, tol=1e-5):
    w_path = glob.glob(folder + '/W*.json')[0]
    with open(w_path, 'r') as f:
        data = json.load(f)
        W = np.array(data).T #need to transpose for consistency
    d_path = glob.glob(folder + '/D*.json')[0]
    with open(d_path, 'r') as f:
        data = json.load(f)
        D = np.array(data)
        
    points = sample_points_on_sphere(num_points)
    points_init_nd = np.dot(D, points.T).T;
    xstars, xstars_2, stabilist = find_fixed_points_newton(W, 0, D, tau=tau, points_init_nd=points_init_nd, max_iter=max_iter, tol=tol)
    
 
    
def plot_fps(xstars_proj, folder):
    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    ax.view_init(elev=0,azim=0)
    ax.scatter(xstars_proj[:,0],xstars_proj[:,1],xstars_proj[:,2], color=stab_colors[stabilist],alpha=.5)
    ax.set_xlim([-1.2,1.2]);ax.set_ylim([-1.2,1.2]);ax.set_zlim([-1.2,1.2]);#plt.axis('off');
    ax.set_xticklabels([]); ax.set_yticklabels([1,-1]); ax.set_yticks([1,-1]);ax.set_zticklabels([1,-1]); ax.set_zticks([1,-1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z");

    ax = fig.add_subplot(122, projection="3d")
    ax.view_init(elev=30,azim=45)
    ax.scatter(xstars_proj[:,0],xstars_proj[:,1],xstars_proj[:,2], color=stab_colors[stabilist],alpha=.5)
    ax.set_xlim([-1.2,1.2]);ax.set_ylim([-1.2,1.2]);ax.set_zlim([-1.2,1.2]); 
    ax.set_xticklabels([1,-1]); ax.set_xticks([1,-1]); ax.set_yticklabels([1,-1]); ax.set_yticks([1,-1]);ax.set_zticklabels([1,-1]); ax.set_zticks([1,-1]);
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z");
    plt.savefig(folder+'/sphere_fps.pdf') #plt.axis('off');
    

# if __name__ == "__main__": 
#     # get_noormanring_rnn(N=8, je=4, ji=-2.4, c_ff=1, dt=.01, internal_noise_std=0)
    
#     N = 10
#     j0 = 0
#     j1 = 3
#     W = ring_nef(N, j0, j1)
#     I_e = -1
#     tsteps=501
#     traj = simulate_nefring(W, I_e, y0=None, maxT=25, tsteps=tsteps)
    
#     batch_size = 100
#     new_sols = np.zeros((batch_size, tsteps, N))
#     for i in range(batch_size):
#         new_sols[i,:] = simulate_nefring(W, I_e, y0=None, maxT=25, tsteps=tsteps).T#[:,-1]