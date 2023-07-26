# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:50:25 2023

@author: abel_
"""

import numpy as np

import scipy
from scipy.integrate import odeint, DOP853, solve_ivp
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import sklearn
import sklearn.decomposition
import umap
from sklearn.datasets import load_digits
from scipy.stats import special_ortho_group
from scipy.optimize import minimize



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


