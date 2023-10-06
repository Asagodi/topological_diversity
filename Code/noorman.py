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

def plot_unfolded(sol):
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    X_proj2 = pca.transform(sol) 
    theta = np.arctan2(X_proj2[:,1], X_proj2[:,0])
    ax.plot(t, theta, '-')
    ax.set_xlabel("Time")
    # ax.set_ylim([np.pi/2-.1,np.pi/2+.1])
    # ax.set_yticks([-np.pi,np.pi],[r"$-\pi$",r"$\pi$"])
    ax.set_ylabel("Angle (rad)")
    
def plot_unfolded_for_inputs(input_list, maxT=25, tsteps=501):
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    all_thetas = np.zeros((len(input_list), tsteps))
    for vi, input_velocity in enumerate(input_list):
        sol,t = simulate_ring(W_sym, W_asym, c_ff, y0=corners[4], v_in=v_constant(value=input_velocity),
                              maxT=maxT, tsteps=tsteps)
        X_proj2 = pca.transform(sol.T) 
        theta = np.arctan2(X_proj2[:,1], X_proj2[:,0])
        all_thetas[vi,:] = theta
        ax.plot(t, theta, ',', label=input_velocity)
    ax.set_xlabel("Time")
    ax.set_yticks([-np.pi,np.pi],[r"$-\pi$",r"$\pi$"])
    ax.set_ylabel("Angle (rad)")
    plt.legend(title='input', loc='lower left', bbox_to_anchor=(1, 0.5))
    plt.savefig(currentdir+f"/Stability/figures/noorman/N{N}_unfolded_inputs.pdf", bbox_inches="tight")
    return all_thetas



def get_period(sol):
    """
    Estimate the period of the integrated limit cycle through the Fast Fourier Transform

    Parameters
    ----------
    sol : TYPE
        DESCRIPTION.

    Returns
    -------
    estimated_period : TYPE
        DESCRIPTION.

    """
    # Perform FFT
    fft_result = np.fft.fft(sol)
    
    # Find the dominant frequency
    # The dominant frequency corresponds to the inverse of the period
    frequencies = np.fft.fftfreq(sol.shape[0])
    dominant_frequency = frequencies[np.argmax(np.abs(fft_result))]
    
    # Calculate the period from the dominant frequency
    estimated_period = 1 / dominant_frequency
    
    return estimated_period
    

def get_period_poincaremap(sol, y0):
    np.where(np.abs(sol[:,0]-y0[0])<.0000001)

cmap = 'gray'
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

if __name__ == "__main__":
    np.random.seed(1) #makes pca project to hexagon sitting
    N = 6    
    tau = .1
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
    
    value = 10
    maxT= 10/4
    t_switch= maxT/12
    tsteps=100001
    # corners[4] is on the right edge
    sol,t = simulate_ring(W_sym, W_asym, c_ff, tau=tau, y0=corners[4], v_in=v_switch(values=[value,0], t_switch=t_switch),
                          maxT=maxT, tsteps=tsteps)

    plot_unfolded(sol.T)
    # plt.savefig(currentdir+f"/Stability/figures/noorman/N{N}_input{value}_until{np.round(t_switch,2)}.pdf", bbox_inches="tight")

    # sol,t = simulate_ring(W_sym, W_asym, c_ff, y0=corners[4], v_in=v_constant(value=value),
    #                       maxT=277, tsteps=1001)
    fig, ax, theta = plot_solution(sol.T, corners, pca, lims = 3.)
    plt.savefig(currentdir+f"/Stability/figures/noorman/N{N}_input{value}_until{np.round(t_switch,2)}.pdf", bbox_inches="tight")

    # plt.savefig(currentdir+f"/Stability/figures/noorman/N{N}_input{value}.pdf", bbox_inches="tight")
    
    # input_list=[0.5, 1, 2]
    # all_thetas = plot_unfolded_for_inputs(input_list, maxT=50, tsteps=501)

    # period = get_period(sol.T)
    # print(period)

    # fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    # for i in np.logspace():
    #     sol,t = simulate_ring(W_sym, W_asym, c_ff, y0=corners[4], v_in=v_constant(value=i),
    #                           maxT=277, tsteps=1001)
    #     X_proj2 = pca.transform(sol.T) 
    #     theta = np.arctan2(X_proj2[:,1], X_proj2[:,0])
    #     ax.plot(t, theta, ',', label=i)