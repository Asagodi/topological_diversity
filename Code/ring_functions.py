# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:50:25 2023

@author: 
"""

import numpy as np

import math
import scipy
from scipy.integrate import odeint, DOP853, solve_ivp
from matplotlib.ticker import MaxNLocator

import sklearn
import sklearn.decomposition
from scipy.stats import special_ortho_group
from scipy.optimize import minimize
from itertools import chain, combinations, permutations

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from matplotlib import rc
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.ticker import LinearLocator

def ReLU(x):
    return np.where(x<0,0,x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_theory_weights(N):
    #𝑤𝑖𝑗~ cos(𝜃𝑖 − 𝜃𝑗 )
    x = np.arange(0,N,1)
    row = np.cos(2*np.pi*x/N)
    W = scipy.linalg.circulant(row)
    return W

def relu_ode(t,x,W):
    return ReLU(np.dot(W,x)) - x 

def sigmoid_ode(t,x,W):
    return sigmoid(np.dot(W,x)) - x 

def tanh_ode(t,x,W):
    return np.tanh(np.dot(W,x)) - x 


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
        # print(r, [item for item in range(N) if item not in r])
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
    for bump_i in range(bumps.shape[1]):
        for support_j in range(N):
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




# def v_constant(t, value=1):
#     #constant input
#     return value

def v_constant(value=1):
      return  lambda x : value

def v_zero(t):
    #zero input
    return 0



def v_switch(t, t_switch=10, value=1):
    if t<t_switch:
        return value
    else:
        return 0
    
def get_v(v_name='zero'):
    if v_name=='zero':
        return v_zero
    elif v_name=='constant':
        return v_constant(value=1)

def simulate_ring(W_sym, W_asym, c_ff, y0=None, tau=1, transfer_function=ReLU, v_in=v_zero,
                  maxT=25, tsteps=501):
    
    N = W_sym.shape[0]
    if not np.any(y0):
        y0 = np.random.uniform(0,1,N)
    t = np.linspace(0, maxT, tsteps)
    sol = solve_ivp(noorman_ode, y0=y0,  t_span=[0,maxT],
                    args=tuple([tau, transfer_function, W_sym, W_asym, c_ff, N, v_in]),
                    dense_output=True)
    
    return sol.sol(t),t

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

def plot_ring(sols, corners, lims = 3.):
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
    
    N = sols.shape[3]
    sols = sols.reshape((-1, sols.shape[2], N))
    pca = sklearn.decomposition.PCA(n_components=2)
    X_proj2 = pca.fit_transform(sols[:,-1,:]) 
    corners_proj2 = pca.transform(corners)
    # all_bumps_proj2 = pca.fit_transform(all_bumps) 
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_ring_from_corners(N, corners_proj2, ax)
        

    ax.plot(X_proj2[:,0], X_proj2[:,1], '.r', zorder=10, alpha=1., markersize=20)
    # ax.plot(all_bumps_proj2[:,0], all_bumps_proj2[:,1], '.r', zorder=10, alpha=1., markersize=20)

    ax.set(xlim=(-lims, lims), ylim=(-lims,lims))
    ax.set_axis_off()

    return pca

def plot_ring_from_corners(N, corners_proj2, ax):
    for i in range(N):
        ax.plot([corners_proj2[i-1,0], corners_proj2[i,0]],
                [corners_proj2[i-1,1], corners_proj2[i,1]],
                'k', zorder=0, alpha=1., linewidth=10, 
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
    plot_ring_from_corners(N, corners_proj2, ax)
        
    cmap = cmx.get_cmap("Blues_r");
    norm = mplcolors.Normalize(vmin=0, vmax=30)
    norm = norm(np.linspace(0, 30, num=30, endpoint=False))
    ax.plot(X_proj2[:,0], X_proj2[:,1], '.r', zorder=10, alpha=1., markersize=5)
    ax.set(xlim=(-lims, lims), ylim=(-lims,lims))
    ax.set_axis_off()

    return pca


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
        plt.savefig(currentdir+f"/Stability/figures/noorman_ring_N{N}_pert_{n_stab}stab_{n_sadd-1}sadd.pdf", bbox_inches="tight")
        # plt.savefig(currentdir+f"/Stability/figures/noorman_ring_N{N}_pert_{n_stab}stab_{n_sadd-1}sadd.png", bbox_inches="tight")
