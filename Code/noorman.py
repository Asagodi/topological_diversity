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
import sklearn.decomposition
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from matplotlib import rc
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.ticker import LinearLocator
from tqdm import tqdm
from scipy.integrate import odeint, DOP853, solve_ivp
from itertools import chain, combinations, permutations

from ring_functions import * 

cmap = 'gray'
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def define_ring(N, je = 4, ji = -2.4, c_ff=1, tau=1):
    W_sym = get_noorman_symmetric_weights(N, je, ji)
    W_asym = get_noorman_asymmetric_weights(N)
    
    
    return W_sym, W_aym

def simulate_ring(tau, transfer_function, W_sym, W_asym, c_ff, N, 
                  maxT = 25, tsteps=501):
    sol = solve_ivp(noorman_ode, y0=y0,  t_span=[0,maxT], args=tuple([tau, transfer_function, W_sym, W_asym, c_ff, N]),dense_output=True)
    
if __main__:
    
    tau = 1
    transfer_function = ReLU
    define_ring(N, je = 4, ji = -2.4, c_ff=1, tau=1)