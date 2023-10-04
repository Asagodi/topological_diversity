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


if __name__ == "__main__":

    N = 6    
    tau = 1
    c_ff = 1

    transfer_function = ReLU
    W_sym, W_asym = define_ring(N, je = 4, ji = -2.4, c_ff=c_ff)
    sol,t = simulate_ring(W_sym, W_asym, c_ff)
    
    m = np.max(sol.sol(t)) # m #round? what should the maximum be according to the paper?
    
    corners = get_corners(N, m)
    bumps_oneside = get_bumps_along_oneside_ring(N, m, corners, step_size=0.05)
    all_bumps = get_all_bumps(N, bumps)

    sols = simulate_bumps(bumps_oneside, W_sym, W_asym, c_ff)