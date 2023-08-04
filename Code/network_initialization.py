import os, sys
currentdir = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.insert(0, currentdir + "\Code") 

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import qr

import rnn_models

def make_rnn_from_networkparameters(W_in, W_hh, W_out, b_hh, b_out, hidden_offset=None, nonlinearity ='relu', output_activation='identity', hidden_initial_activations='offset'):
    """
    Input, recurrent and output weight and recurrent and output biases need to be given
    returns rnn
    """
    N_in = W_in.shape[1]
    N_out = W_out.shape[0]
    N_rec = W_hh.shape[0]

    rnn_model = rnn_models.RNNModel(N_in, N_out, N_rec, nonlinearity=nonlinearity , output_activation=output_activation, hidden_initial_activations=hidden_initial_activations)

    with torch.no_grad():
        rnn_model.rnn.all_weights[0][0][:] = torch.tensor(W_in, dtype=torch.float)
        rnn_model.rnn.all_weights[0][1][:] = torch.tensor(W_hh, dtype=torch.float)
        rnn_model.rnn.all_weights[0][2][:] =  torch.tensor(b_hh, dtype=torch.float)
        rnn_model.rnn.all_weights[0][3][:] =  torch.zeros((N_rec), dtype=torch.float)
        rnn_model.fc.weight = nn.Parameter(torch.tensor(W_out, dtype=torch.float))
        rnn_model.fc.bias = nn.Parameter(torch.tensor(b_out, dtype=torch.float)) 
        if np.all(hidden_offset!=None):
            rnn_model.hidden_offset = nn.Parameter(torch.tensor(hidden_offset, dtype=torch.float)) 

    return rnn_model

def add_parameter_noise(rnn_model, std):
    """
    Adds a small perturbation to the parameters of the RNN.
    returns perturbed rnn
    """

    with torch.no_grad():
        rnn_model.rnn.all_weights[0][0][:] += std*torch.randn(rnn_model.rnn.all_weights[0][0][:].shape)
        rnn_model.rnn.all_weights[0][1][:] += std*torch.randn(rnn_model.rnn.all_weights[0][1][:].shape)
        rnn_model.rnn.all_weights[0][2][:] += std*torch.randn(rnn_model.rnn.all_weights[0][2][:].shape)
        rnn_model.rnn.all_weights[0][3][:] += std*torch.randn(rnn_model.rnn.all_weights[0][3][:].shape)
        rnn_model.fc.weight += std*torch.randn(rnn_model.fc.weight.shape)
        rnn_model.fc.bias += std*torch.randn(rnn_model.fc.bias.shape) 

    return rnn_model

def perfect_initialization(version, output_dim=1, random_winout=False, ouput_bias_value=100, a=1, eps=.01):
    """
    Returns RNN model with recurrent weights suitable for perfect integration.
    Can be initialized with random input and output weights.
    2D version (i.e. a single line (or plane) attractor)
    """
    N_in = 2
    N_rec = 2
    N_out = output_dim

    if version==1: # V1: plane attractor
        W_in = np.array([[1,0],[0,1]], dtype=float)
        W_hh = np.array([[1,0],[0,1]])
        b_hh = np.array([0,0])
        #1D output (theory/proof in this form)
        W_out = np.array([[-1,1]])
        b_out = np.array([0])
        hidden_offset = np.array([0,0])
        
    elif version==2: # V2
        W_in = a*np.array([[-1,1],[-1,1]], dtype=float)
        W_hh = np.array([[0,1],[1,0]])
        b_hh = np.array([0,0])
        W_out = 1/a*np.array([[1,1]])/2
        b_out = np.array([-ouput_bias_value])/a
        hidden_offset = ouput_bias_value*np.array([1,1])
    
    elif version==3: # V3
        W_in = a*np.array([[-1,1],[1,-1]], dtype=float)
        W_hh = np.array([[0,-1],[-1,0]])
        b_hh = ouput_bias_value*np.array([1,1])
        W_out = np.array([[1,-1]])/(2*a)
        b_out = np.array([.0])
        hidden_offset = ouput_bias_value/2*np.array([1,1])

    else:
        print("Version is not defined. Use verions 1,2 or 3.")
        
    if random_winout=="win":
        W_in = np.random.normal(0, eps/4., (2,2))
        
    elif random_winout=="small_win":
        W_in += np.random.normal(0, eps/4., (2,2))

    elif random_winout=="winout":
        W_in = np.random.normal(0, eps/4., (2,2))
        W_out = np.random.normal(0, eps/2., (1,2))
        b_out = np.random.normal(0, eps/2., (1,2))
        
    elif random_winout=="small_winout":
        W_in += np.random.normal(0, eps/4., (2,2))

    rnn_model = make_rnn_from_networkparameters(W_in, W_hh, W_out, b_hh, b_out, hidden_offset, nonlinearity='relu', output_activation='identity', hidden_initial_activations="offset")
    return rnn_model



def identity_initialization(N_in, N_rec, N_out, weight_init_variance=0.001, hidden_initial_activations='offset'):
    """
    Identity for recurrent weights, no bias for recurrent units
    all other parameters random with He initialization (normal with 1/numberofparams variance)
    Le-2015-A Simple Way to Initialize Recurrent Networks of Rectified Linear Units
      fixed variance for random weight initialization at 0.001
    N_in=#input dimensions
    N_rec=# of neurons in recurrent network
    N_out=# of output dimensions
    
    returns rnn_model with identity for recurrent weights
    """
    W_hh = np.identity(N_rec)
    b_hh = np.zeros((N_rec))
    
    W_in = np.random.normal(0, weight_init_variance, (N_rec, N_in))
    W_out = np.random.normal(0, weight_init_variance, (N_out, N_rec))
    b_out = np.random.normal(0, weight_init_variance, (N_out))
    
    rnn_model = make_rnn_from_networkparameters(W_in, W_hh, W_out, b_hh, b_out, transform_function='relu', output_activation='identity', hidden_initial_activations=hidden_initial_activations)
    return rnn_model

def bla_rec_weights(N_in, N_blas, N_out, a):
    N_rec = 2*N_blas
    bla_mat = np.array(([[0,-1],[-1,0]]))
    W_hh = block_diag(*[bla_mat]*N_blas)
    b_hh = 2*a*np.array([1]*N_rec)

    return W_hh, b_hh

def bla_initialization(N_in, N_blas, N_out, a, weight_init_variance=0.001, hidden_initial_activations='offset', nonlinearity='relu'):
    """
    Pairwise Bounded Line Attractor (BLA) initialization for the recurrent weights of the network
    i.e., [[0,-1],[-1,0]] and a bias a*[1,1] 
    all other parameters random with He initialization (normal with 1/numberofparams variance)
    in Le-A Simple Way: variance = 0.001
    
    N_in=#input dimensions
    N_blas= number of bounded line attractors
    N_rec = 2*N_blas
    N_rec=# of neurons in recurrent network
    N_out=# of output dimensions
    
    returns rnn_model with identity for recurrent weights
    """
    N_rec = 2*N_blas
    W_hh, b_hh = bla_rec_weights(N_in, N_blas, N_out, a)
    
    W_in = np.random.normal(0, weight_init_variance, (N_rec, N_in))
    W_out = np.random.normal(0, weight_init_variance, (N_out, N_rec))
    b_out = np.random.normal(0, weight_init_variance, (N_out))
    # W_in = np.random.normal(0, 1/np.sqrt(N_in*N_rec), (N_rec, N_in))
    # W_out = np.random.normal(0, 1/np.sqrt(N_in*N_rec), (N_out, N_rec))
    # b_out = np.random.normal(0, 1/np.sqrt(N_out), (N_out))
    
    rnn_model = make_rnn_from_networkparameters(W_in, W_hh, W_out, b_hh, b_out, nonlinearity=nonlinearity , output_activation='identity', hidden_initial_activations=hidden_initial_activations)
    return rnn_model

def ubla_initialization(N_in, N_blas, N_out, a, weight_init_variance=0.001, hidden_initial_activations='offset', nonlinearity='relu'):
    """
    Pairwise Bounded Line Attractor (BLA) initialization for the recurrent weights of the network
    i.e., [[0,-1],[-1,0]] and a bias a*[1,1] 
    all other parameters random with He initialization (normal with 1/numberofparams variance)
    in Le-A Simple Way: variance = 0.001
    
    N_in=#input dimensions
    N_blas= number of bounded line attractors
    N_rec = 2*N_blas
    N_rec=# of neurons in recurrent network
    N_out=# of output dimensions
    
    returns rnn_model with identity for recurrent weights
    """
    N_rec = 2*N_blas
    ubla_mat = np.array(([[0,1],[1,0]]))
    W_hh = block_diag(*[ubla_mat]*N_blas)
    b_hh = np.array([0]*N_rec)
    
    W_in = np.random.normal(0, weight_init_variance, (N_rec, N_in))
    W_out = np.random.normal(0, weight_init_variance, (N_out, N_rec))
    b_out = np.random.normal(0, weight_init_variance, (N_out))

    rnn_model = make_rnn_from_networkparameters(W_in, W_hh, W_out, b_hh, b_out, nonlinearity=nonlinearity, output_activation='identity', hidden_initial_activations=hidden_initial_activations)
    return rnn_model

def qpta_rec_weights(N_in, N_blas, N_out):
    
    N_rec = 2*N_blas
    thetas = np.random.uniform(-np.pi, np.pi, N_blas) 
    alpha = np.random.uniform(1, 2, N_blas) 
    # make N_blas harmonic oscillators with thetas[i]
    W_hh = np.zeros((2*N_blas, 2*N_blas))
    ho_mat = np.array(([[np.cos(thetas),-np.sin(thetas)],[np.sin(thetas),np.cos(thetas)]]))
    for i in range(0, N_blas):
        W_hh[2*i:2*i+2, 2*i:2*i+2] = alpha[i]*ho_mat[..., i]
    b_hh = np.array([1]*N_rec)
    
    return W_hh, b_hh

def qpta_initialization(N_in, N_blas, N_out, weight_init_variance=0.001, hidden_initial_activations='offset', nonlinearity='tanh', return_weights=False):
    """
    pairwise QPTAs for the recurrent weights of the network
    i.e., [[0,-1],[-1,0]] and a bias a*[1,1] 
    all other parameters random with He initialization (normal with 1/numberofparams variance)
    
    
    N_in=#input dimensions
    N_blas= number of QPTAs
    N_rec = 2*N_blas
    N_rec=# of neurons in recurrent network
    N_out=# of output dimensions
    
    returns rnn_model with identity for recurrent weights
    """
    N_rec = 2*N_blas
    
    W_hh, b_hh = qpta_rec_weights(N_in, N_blas, N_out)
    
    W_in = np.random.normal(0, weight_init_variance, (N_rec, N_in))
    W_out = np.random.normal(0, weight_init_variance, (N_out, N_rec))
    b_out = np.random.normal(0, weight_init_variance, (N_out))
        
    rnn_model = make_rnn_from_networkparameters(W_in, W_hh, W_out, b_hh, b_out, nonlinearity=nonlinearity, output_activation='identity', hidden_initial_activations=hidden_initial_activations)
    return rnn_model




def ortho_initialization(N_in, N_rec, N_out, weight_init_variance=0.001, hidden_initial_activations='offset', nonlinearity='tanh'):
    """
    random orthogonal initial weights
    
    Saxe-2014-Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
    """
    
    H = np.random.randn(N_rec, N_rec)
    W_hh, _ = qr(H)
    b_hh = np.array([0]*N_rec)

    W_in = np.random.normal(0, weight_init_variance, (N_rec, N_in))
    W_out = np.random.normal(0, weight_init_variance, (N_out, N_rec))
    b_out = np.random.normal(0, weight_init_variance, (N_out))

    rnn_model = make_rnn_from_networkparameters(W_in, W_hh, W_out, b_hh, b_out, nonlinearity=nonlinearity, output_activation='identity', hidden_initial_activations=hidden_initial_activations)
    return rnn_model


def gain_initialization(N_in, N_rec, N_out, gain=1, weight_init_variance=0.001, hidden_initial_activations='offset', nonlinearity='tanh'):
    """
    random initial from N(0,g^2/N_rec)
    
    """
    W_hh =np.random.normal(0, gain/np.sqrt(N_rec), (N_rec, N_rec))
    b_hh = np.array([0]*N_rec)

    W_in = np.random.normal(0, weight_init_variance, (N_rec, N_in))
    W_out = np.random.normal(0, weight_init_variance, (N_out, N_rec))
    b_out = np.random.normal(0, weight_init_variance, (N_out))

    rnn_model = make_rnn_from_networkparameters(W_in, W_hh, W_out, b_hh, b_out, nonlinearity=nonlinearity, output_activation='identity', hidden_initial_activations=hidden_initial_activations)
    return rnn_model
