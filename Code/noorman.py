# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:51:22 2023

@author: abel_
"""

import os, sys
currentdir = os.path.dirname(os.path.abspath(os.getcwd()))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir) 
sys.path.insert(0, parentdir) 
sys.path.insert(0, currentdir + "\Code") 

import math
import numpy as np
import sklearn.decomposition

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, rc
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines
from matplotlib.ticker import LinearLocator
import matplotlib.colors as mplcolors

from tqdm import tqdm
from scipy.integrate import odeint, DOP853, solve_ivp
from itertools import chain, combinations, permutations

from ring_functions import define_ring, ReLU, simulate_ring, get_corners, plot_solution
from ring_functions import get_bumps_along_oneside_ring, get_all_bumps, simulate_bumps, plot_ring, v_zero, v_constant, v_switch

def plot_unfolded():
    
    0
    
def plot_unfolded_for_inputs():
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    for i in [0.05, .1, .5]:
        sol,t = simulate_ring(W_sym, W_asym, c_ff, y0=corners[4], v_in=v_constant(value=i),
                              maxT=277, tsteps=1001)
        X_proj2 = pca.transform(sol.T) 
        theta = np.arctan2(X_proj2[:,1], X_proj2[:,0])
        ax.plot(t, theta, ',', label=i)
    ax.set_xlabel("Time")
    ax.set_yticks([-np.pi,np.pi],[r"$-\pi$",r"$\pi$"])
    ax.set_ylabel("Angle (rad)")
    plt.legend(title='input', loc='lower left', bbox_to_anchor=(1, 0.5))
    plt.savefig(currentdir+f"/Stability/figures/noorman/N{N}_unfolded_inputs.pdf", bbox_inches="tight")

cmap = 'gray'
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

if __name__ == "__main__":
    np.random.seed(1) #makes pca project to hexagon sitting
    N = 6    
    tau = 1
    c_ff = 1

    transfer_function = ReLU
    W_sym, W_asym = define_ring(N, je = 4, ji = -2.4, c_ff=c_ff)
    sol,t = simulate_ring(W_sym, W_asym, c_ff, v_in=v_zero, maxT=1000, tsteps=10001)
    
    m = np.max(np.abs(sol))
    corners = get_corners(N, m)
    bumps_oneside = get_bumps_along_oneside_ring(N, m, corners, step_size=0.05)
    all_bumps = get_all_bumps(N, bumps_oneside)

    sols = simulate_bumps(bumps_oneside, W_sym, W_asym, c_ff)
    # pca = plot_ring(sols, corners, lims = 3.)
    sols = sols.reshape((-1, sols.shape[2], N))
    pca = sklearn.decomposition.PCA(n_components=2)
    X_proj2 = pca.fit_transform(sols[:,-1,:]) 
    
    value = .1
    # corners[4] is on the right edge
    sol,t = simulate_ring(W_sym, W_asym, c_ff, y0=corners[4], v_in=v_constant(value=value),
                          maxT=27.7/value, tsteps=1001)
    fig, ax, theta = plot_solution(sol.T, corners, pca, lims = 3.)
    # plt.savefig(currentdir+f"/Stability/figures/noorman/N{N}_input{value}.pdf", bbox_inches="tight")



    # fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    # for i in np.logspace():
    #     sol,t = simulate_ring(W_sym, W_asym, c_ff, y0=corners[4], v_in=v_constant(value=i),
    #                           maxT=277, tsteps=1001)
    #     X_proj2 = pca.transform(sol.T) 
    #     theta = np.arctan2(X_proj2[:,1], X_proj2[:,0])
    #     ax.plot(t, theta, ',', label=i)