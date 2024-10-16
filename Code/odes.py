# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:37:38 2024

@author: 
"""
import numpy as np
from scipy.integrate import solve_ivp

def ReLU(x):
    return np.where(x<0,0,x)

def ReTanh(x):
    return np.where(x<0,0,np.tanh(x))


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

def relu_ode(t,x,W,b,tau,mlrnn=True):

    if mlrnn:
        return (-x + ReLU(np.dot(W,x)+b))/tau
    else:
        return (-x + np.dot(W,ReLU(x))+b)/tau

def tanh_ode(t,x,W,b,tau,mlrnn=True):
    if mlrnn:
        return (-x + np.tanh(np.dot(W,x)+b))/tau
    else:
        return (-x + np.dot(W,np.tanh(x))+b)/tau

def recttanh_ode(t,x,W,b,tau,mlrnn=True):
    if mlrnn:
        return (-x + ReTanh(np.dot(W,x)+b))/tau
    else:
        return (-x + np.dot(W,ReTanh(x))+b)/tau


######numerical integration
def simulate_network(W, b, nonlinearity_ode=relu_ode, y0=None, maxT=25, tsteps=501, tau=1, mlrnn=True):
    
    N = W.shape[0]
    if not np.any(y0):
        y0 = np.random.uniform(0,1,N)
    t = np.linspace(0, maxT, tsteps)
    sol = solve_ivp(nonlinearity_ode, y0=y0,  t_span=[0,maxT],
                    args=tuple([W, b, tau, mlrnn]),
                    dense_output=True)
    return sol.sol(t)


def simulate_network_ntimes(Nsims, W, b, nonlinearity_ode=relu_ode, mlrnn=True,
                            y0s=None, y0_dist="uniform", #todo: y0_dist=
                            maxT=25, tsteps=501, tau=1):
    t = np.linspace(0, maxT, tsteps)
    N = W.shape[0]
    sols = np.zeros((Nsims, tsteps, N))
    for ni in range(Nsims):
        if not np.any(y0s):
            if y0_dist=='uniform':
                y0 = np.random.uniform(-1,1,N)
            else:
                y0 = np.random.normal(0,1,N)
        else:
            y0 = y0s[ni]
        sol = solve_ivp(nonlinearity_ode, y0=y0,  t_span=[0,maxT],
                        args=tuple([W, b, tau, mlrnn]),
                        dense_output=True)
        sols[ni,...] = sol.sol(t).T.copy()
    return sols


##############linearization
#Jacobians

def relu_jacobian(W,b,tau,x):
    return (-np.eye(W.shape[0]) + np.multiply(W, np.where(np.dot(W,x)+b>0,1,0)))/tau

def tanh_jacobian(W,b,tau,x,mlrnn=True):
    #b is unused, but there for consistency with relu jac
    if mlrnn:
        #return (-np.eye(W.shape[0]) + np.multiply(W,1/np.cosh(np.dot(W,x)+b)**2))/tau
        dtanh = 1 - np.tanh(np.dot(W, x) + b) ** 2
        df_dx = -np.eye(len(x))/tau + (np.dot(np.diag(dtanh), W))/tau
        return df_dx
    else:
        return (-np.eye(W.shape[0]) + np.multiply(W,1/np.cosh(x)**2))/tau


def recttanh_jacobian_point(W,b,tau,x):
    #b is unused, but there for consistency with relu jac
    return (-np.eye(W.shape[0]) + np.multiply(np.multiply(W, np.where(np.dot(W,np.tanh(x))+b>0,1,0)),  np.multiply(W,1/np.cosh(x)**2)))/tau



def compute_jacobian(X, J):
    N = len(X)
    Jacobian = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                Jacobian[i, j] = -1 + J[i, j] * (1 - np.tanh(X[j])**2)
            else:
                Jacobian[i, j] = J[i, j] * (1 - np.tanh(X[j])**2)

    return Jacobian


#include versions for x_solved being from a (scipy) ode solver?
def linear_jacobian(t,W,b,tau,x_solved):
    return W/tau

def relu_jacobian_sequence(t,W,b,tau,x_solved):
    return (-np.eye(W.shape[0]) + np.multiply(W, np.where(np.dot(W,x_solved[t])+b>0,1,0)))/tau

def tanh_jacobian_sequence(t,W,b,tau,x_solved, mlrnn=True):
    
    if mlrnn:
        return (-np.eye(W.shape[0]) + np.multiply(W,1/np.cosh(np.dot(W,x_solved[t])+b)**2))/tau
    else:
        return (-np.eye(W.shape[0]) + np.multiply(W,1/np.cosh(x_solved[t])**2))/tau



#########finding fixed points
def newton_method(x0, system, jacobian, W, b, tau, mlrnn, tol=1e-6, max_iter=100):
    # Implement the Newton method
    x = x0
    for _ in range(max_iter):
        dx = np.linalg.solve(jacobian(W,b,tau,x,mlrnn), system(0,x,W,b,tau,mlrnn))
        x -= dx
        if np.linalg.norm(dx) < tol:
            break
    return x
