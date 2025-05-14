import os
import pickle
import glob
import yaml
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd

import torch
import torch.nn as nn



# Loading parameters and RNN
def load_net_from_weights(wi, wrec, wo, brec, h0, oth, training_kwargs):
    if oth is None:
        training_kwargs['map_output_to_hidden'] = False

    dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
    net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'],
              nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
              wi_init=wi, wrec_init=wrec, wo_init=wo, brec_init=brec, h0_init=h0, oth_init=oth,
              ML_RNN=training_kwargs['ml_rnn'], 
              map_output_to_hidden=training_kwargs['map_output_to_hidden'], input_nonlinearity=training_kwargs['input_nonlinearity'])
    return net



def load_net_path(path, which='post'):
    # folder = parent_dir+"/experiments/" + main_exp_name
    # exp_list = glob.glob(folder + "/res*")
    # exp = exp_list[exp_i]
    with open(path, 'rb') as handle:
        result = pickle.load(handle)
    if which=='post':
        try:
            wi, wrec, wo, brec, h0, oth = result['weights_last']
        except:
            oth = None
            wi, wrec, wo, brec, h0 = result['weights_last']

    elif which=='pre':
        try:
            wi, wrec, wo, brec, h0, oth = result['weights_init']
        except:
            oth = None
            wi, wrec, wo, brec, h0 = result['weights_init']
    
    
    try:    
        net = load_net_from_weights(wi, wrec, wo, brec, h0, oth, result['training_kwargs'])
    except:
        
        net = load_net_from_weights(wi, wrec, wo, brec, h0, oth, result['training_kwargs'])

    return net, result

def load_losses(path):
    net, result = load_net_path(path)
    losses = result['losses']
    return losses


def load_all_losses_folder(folder, df):
    exp_list = glob.glob(folder + "/res*")
    nexps = len(exp_list)
    all_losses = np.empty((nexps,10000))
    all_losses[:] = np.nan
    # if not np.any(df):
    #     df = pd.DataFrame(columns=['T', 'N', 'I', 'S', 'R', 'M', 'trial'])

    for exp_i in range(nexps):
        path = exp_list[exp_i]
        net, result = load_net_path(path)
        losses = load_losses(path)
        losses[np.argmin(losses):] = np.nan
        training_kwargs = result['training_kwargs']
        T, N, I, S, R, M, clip_gradient = get_tr_par(training_kwargs)
        df = df.append({'T': T, 'N': N, 'I': I, 'S': S, 'R': R, 'M': M, 'clip_gradient':clip_gradient,
                        'trial': exp_i,
                        'losses': losses}, ignore_index=True)

    return df


##### Simulating the RNN
def simulate_rnn(net, task, T, batch_size):
    input, target, mask = task(batch_size); input = torch.from_numpy(input).float();
    output, trajectories = net(input, return_dynamics=True); 
    output = output.detach().numpy();
    trajectories = trajectories.detach().numpy()
    return input, target, mask, output, trajectories

def simulate_rnn_with_input(net, input, h_init):
    input = torch.from_numpy(input).float();
    output, trajectories = net(input, return_dynamics=True, h_init=h_init); 
    output = output.detach().numpy();
    trajectories = trajectories.detach().numpy()
    return output, trajectories

def simulate_rnn_with_task(net, task, T, h_init, batch_size=256):
    input, target, mask = task(batch_size);
    input_ = torch.from_numpy(input).float();
    target_ = torch.from_numpy(target).float();
    output, trajectories = net(input_, return_dynamics=True, h_init=h_init, target=target_); 
    
    output = output.detach().numpy();
    trajectories = trajectories.detach().numpy()
    return input, target, mask, output, trajectories
  
def test_network(net,T=25.6, dt=.1, batch_size=4, from_t=0, to_t=None, random_seed=100):
    #test network on angular integration task
    np.random.seed(random_seed)
    task = angularintegration_task(T=T, dt=dt, sparsity=1, random_angle_init='equally_spaced');
    #task = angularintegration_task_constant(T=T, dt=dt, speed_range=[0.1,0.1], sparsity=1, random_angle_init='equally_spaced');
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, 'random', batch_size=batch_size)
    target_power = np.mean(target[:,from_t:to_t,:]**2)
    mse = np.mean((target[:,from_t:to_t,:] - output[:,from_t:to_t,:])**2)
    mse_normalized = mse/target_power
    return mse, mse_normalized

def get_autonomous_dynamics(net, T=128, dt=.1, batch_size=32):
    task = angularintegration_task_constant(T=T, dt=dt, speed_range=[0.,0.], sparsity=1, random_angle_init='equally_spaced');
    input, target, mask, output, trajectories = simulate_rnn_with_task(net, task, T, 'random', batch_size=batch_size)
    return output, trajectories

def get_autonomous_dynamics_from_hinit(net, h_init, T=128):
    input = np.zeros((h_init.shape[0], T, net.dims[0]))
    output, trajectories = simulate_rnn_with_input(net, input, h_init=h_init)
    return output, trajectories
    

def test_networks_in_folder(folder, df=None):
    exp_list = glob.glob(folder + "/res*")
    nexps = len(exp_list)

    for exp_i in range(nexps):
        path = exp_list[exp_i]
        print(path)        
        net, result = load_net_path(path)
        mse = test_network(net)
        training_kwargs = result['training_kwargs']
        T, N, I, S, R, M, clip_gradient = get_tr_par(training_kwargs)
        df = df.append({'T': T, 'N': N, 'I': I, 'S': S, 'R': R, 'M': M, 'clip_gradient':clip_gradient,
                        'trial': exp_i,
                        'mse': mse}, ignore_index=True)
    return df

def test_all_networks(folder):
    df = pd.DataFrame(columns=['T', 'N', 'I', 'S', 'R', 'M', 'trial', 'mse'])

    for dirName, subdirList, fileList in os.walk(folder):
        print(dirName)
        df = test_networks_in_folder(dirName, df=df)

    return df


def get_tr_par(training_kwargs):
    #load training parameters
    T = training_kwargs['T']/training_kwargs['dt_task']
    N = training_kwargs['N_rec']
    if training_kwargs['initialization_type']=='gain':
        I = training_kwargs['rnn_init_gain']
    else:
        I = 'irnn'    
    S = training_kwargs['nonlinearity']
    R = training_kwargs['act_reg_lambda']
    M = training_kwargs['ml_rnn']
    clip_gradient = training_kwargs['clip_gradient']
    return T, N, I, S, R, M, clip_gradient




###RNN
class RectifiedTanh(nn.Module):
    def __init__(self):
        super(RectifiedTanh, self).__init__()

    def forward(self, x):
        return torch.max(torch.tanh(x), torch.zeros_like(x))

class RNN(nn.Module):
    def __init__(self, dims, noise_std=0., dt=0.5, 
                 nonlinearity='tanh', readout_nonlinearity='id',
                 g=None, g_in=1, wi_init=None, wrec_init=None, wo_init=None, brec_init=None, bo_init=None,
                 h0_init=None, hidden_initial_variance=0., oth_init=None,
                 train_wi=True, train_wrec=True, train_wo=True, train_brec=True, train_bo=True, train_h0=True,
                 ML_RNN=True, map_output_to_hidden=False, input_nonlinearity=None, save_inputs=False):
        """
        :param dims: list = [input_size, hidden_size, output_size]
        :param noise_std: float
        :param dt: float, integration time step
        :param nonlinearity: str, nonlinearity. Choose 'tanh' or 'id'
        :param readout_nonlinearity: str, nonlinearity. Choose 'tanh' or 'id'
        :param g: float, std of gaussian distribution for initialization
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param wrec_init: torch tensor of shape (hidden_size, hidden_size)
        :param brec_init: torch tensor of shape (hidden_size)
        :param h0_init: torch tensor of shape (hidden_size)
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_brec: bool
        :param train_h0: bool
        :param ML_RNN: bool; whether forward pass is ML convention f(Wr)
        """
        super(RNN, self).__init__()
        self.dims = dims
        input_size, hidden_size, output_size = dims
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.hidden_initial_variance = hidden_initial_variance
        self.dt = dt
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_brec = train_brec
        self.train_h0 = train_h0
        self.ML_RNN = ML_RNN
        self.save_inputs = save_inputs
        self.map_output_to_hidden = map_output_to_hidden #oth
        
        # Either set g or choose initial parameters. Otherwise, there's a conflict!
        assert (g is not None) or (wrec_init is not None), "Choose g or initial wrec!"

        self.g = g
        
        # Nonlinearity
        if nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'id':
            self.nonlinearity = lambda x: x
            if g is not None:
                if g > 1:
                    warn("g > 1. For a linear network, we need stable dynamics!")
        elif nonlinearity.lower() == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity.lower() == 'rect_tanh':
            # if self.noise_std!=0.:
            # self.nonlinearity = nn.ReLU(torch.tanh)                 ##works with noise
            # self.nonlinearity = lambda x: 1.*(torch.tanh(x)>0)   #works with ml_rnn False
            self.nonlinearity = RectifiedTanh()

        elif nonlinearity.lower() == 'talu':
            self.nonlinearity = lambda x: torch.tanh(x) if x<0 else x
        elif nonlinearity == 'softplus':
            softplus_scale = 1 # Note that scale 1 is quite far from relu
            self.nonlinearity = lambda x: torch.log(1. + torch.exp(softplus_scale * x)) / softplus_scale
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = lambda x: 1 / (1 + torch.exp(-x))
        elif type(nonlinearity) == str:
            raise NotImplementedError("Nonlinearity not yet implemented.")
        else:
            self.nonlinearity = nonlinearity
            
            
        # Readout nonlinearity
        if readout_nonlinearity is None:
            # Same as recurrent nonlinearity
            self.readout_nonlinearity = self.nonlinearity
        elif readout_nonlinearity == 'tanh':
            self.readout_nonlinearity = torch.tanh
        elif readout_nonlinearity == 'logistic':
            # Note that the range is [0, 1]. otherwise, 'logistic' is a scaled and shifted tanh
            self.readout_nonlinearity = lambda x: 1. / (1. + torch.exp(-x))
        elif readout_nonlinearity == 'id':
            self.readout_nonlinearity = lambda x: x
        elif type(readout_nonlinearity) == str:
            raise NotImplementedError("readout_nonlinearity not yet implemented.")
        else:
            self.readout_nonlinearity = readout_nonlinearity
            
        # Input mapping nonlinearity
        if input_nonlinearity is None or input_nonlinearity == 'id':
            self.input_nonlinearity = lambda x: x
        elif input_nonlinearity == 'recurrent':
            self.input_nonlinearity = self.nonlinearity
            

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        if self.ML_RNN=='noorman':
            self.wi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wi:
            self.wi.requires_grad = False
            
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.brec = nn.Parameter(torch.Tensor(hidden_size))
        if not train_brec:
            self.brec.requires_grad = False
            
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        if not train_wo:
            self.wo.requires_grad = False
        self.bo = nn.Parameter(torch.Tensor(output_size))
        if not train_bo:
                self.bo.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_(std=g_in /np.sqrt(hidden_size))
            else:
                if type(wi_init) == np.ndarray:
                    wi_init = torch.from_numpy(wi_init)
                self.wi.copy_(wi_init)
            if wrec_init is None:
                self.wrec.normal_(std=g / np.sqrt(hidden_size))
            else:
                if type(wrec_init) == np.ndarray:
                    wrec_init = torch.from_numpy(wrec_init)
                self.wrec.copy_(wrec_init)
            if wo_init is None:
                self.wo.normal_(std=1 / np.sqrt(hidden_size))
            else:
                if type(wo_init) == np.ndarray:
                    wo_init = torch.from_numpy(wo_init)
                self.wo.copy_(wo_init)
            if brec_init is None:
                self.brec.zero_()
                torch.nn.init.uniform_(self.brec, a=-np.sqrt(hidden_size), b=np.sqrt(hidden_size))
            else:
                if type(brec_init) == np.ndarray:
                    brec_init = torch.from_numpy(brec_init)
                self.brec.copy_(brec_init)
            if bo_init is None:
                self.bo.zero_()
            else:
                if type(bo_init) == np.ndarray:
                    bo_init = torch.from_numpy(bo_init)
                self.bo.copy_(bo_init)
            if h0_init is None:
                torch.nn.init.uniform_(self.h0, a=-1, b=1)
            else:
                if type(h0_init) == np.ndarray:
                    h0_init = torch.from_numpy(h0_init)
                    self.h0.copy_(h0_init)
                
        if map_output_to_hidden:
            self.h0.requires_grad = False
            self.output_to_hidden = nn.Parameter(torch.Tensor(output_size, hidden_size))

            if oth_init is None:
                with torch.no_grad():
                    self.output_to_hidden.normal_(std=1 / np.sqrt(hidden_size))
            else:
                with torch.no_grad():
                    oth_init = torch.from_numpy(oth_init)
                    self.output_to_hidden.copy_(oth_init)
            # self.oth_nonlinearity = nonlinearity
            self.oth_nonlinearity = lambda x: x

        
            
    def forward(self, input, return_dynamics=False, h_init=None, target=None):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        
        # assert not self.map_output_to_hidden or target
        
        batch_size = input.shape[0]
        seq_len = input.shape[1]

        if self.map_output_to_hidden:
            h_init_torch = nn.Parameter(torch.Tensor(batch_size, self.hidden_size))
            h_init_torch.requires_grad = False
            h_init_torch = target[:,0,:].matmul(self.output_to_hidden)
            h_init_torch = self.input_nonlinearity(h_init_torch)
            with torch.no_grad():
                h = h_init_torch
                
        elif type(h_init) == np.ndarray or isinstance(h_init, torch.Tensor):
            h_init_torch = nn.Parameter(torch.Tensor(batch_size, self.hidden_size))
            h_init_torch.requires_grad = False
            # Initialize parameters 
            with torch.no_grad():
                if type(h_init) == np.ndarray:
                    h = h_init_torch.copy_(torch.from_numpy(h_init))
                else:
                    h = h_init_torch.copy_(h_init)
                
        elif h_init == 'random':
            h = torch.normal(mean=0, std=self.hidden_initial_variance, size=(1, batch_size, self.hidden_size)).to(self.wrec.device)
            self.h0.requires_grad = False

        elif h_init == 'self_h':
             h = self.h0
                
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.wrec.device)

        # simulation loop
        for i in range(seq_len):

            if self.ML_RNN:
                rec_input = self.nonlinearity(
                    h.matmul(self.wrec.t()) 
                    + input[:, i, :].matmul(self.wi)
                    + self.brec)
                h = ((1 - self.dt) * h 
                     + self.dt * rec_input
                     + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])
                out_i = self.readout_nonlinearity(h.matmul(self.wo))+self.bo
                
            else:
                rec_input = (
                    self.nonlinearity(h).matmul(self.wrec.t()) 
                    + input[:, i, :].matmul(self.wi) 
                    + self.brec)
                h = ((1 - self.dt) * h 
                     + self.dt * rec_input
                     + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])
                out_i = self.readout_nonlinearity(h).matmul(self.wo)+self.bo

            output[:, i, :] = out_i

            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories


##ANGULAR INTEGRATION
import scipy
from scipy.stats import truncexpon


def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


def angularintegration_task(T, dt, length_scale=1, sparsity=1, last_mses=False, random_angle_init=False, max_input=None):
    """
    Creates N_batch trials of the angular integration task with Guassian Process angular velocity inputs.
    Inputs is angular velocity (postive: right, negative:left) and 
    target output is sine and cosine of integrated angular velocity.
    Returns inputs, outputs, mask
    -------
    """
    input_length = int(T/dt)
    
    def task(batch_size):
        
        X = np.expand_dims(np.linspace(-length_scale, length_scale, input_length), 1)
        sigma = exponentiated_quadratic(X, X)  
        if sparsity =='variable':
            sparsities = np.random.uniform(0, 2, batch_size)
            mask_input = np.random.random(size=(batch_size, input_length))<1-sparsities[:,None]
        elif sparsity:
            mask_input = np.random.random(size=(batch_size, input_length)) < 1-sparsity
        inputs = np.random.multivariate_normal(mean=np.zeros(input_length), cov=sigma, size=batch_size)
        if max_input:
            inputs = np.where(np.abs(inputs)>max_input, np.sign(inputs), inputs)
        if sparsity:
            inputs[mask_input] = 0.
        outputs_1d = np.cumsum(inputs, axis=1)*dt
        if random_angle_init=='equally_spaced':
            outputs_1d += np.arange(-np.pi, np.pi, 2*np.pi/batch_size)[:, np.newaxis]
        elif random_angle_init:
            random_angles = np.random.uniform(-np.pi, np.pi, size=batch_size)
            outputs_1d += random_angles[:, np.newaxis]

        outputs = np.stack((np.cos(outputs_1d), np.sin(outputs_1d)), axis=-1)

        if last_mses:
            fin_int = np.random.randint(1,last_mses,size=batch_size)
            mask = np.zeros((batch_size, input_length, 2))
            mask[np.arange(batch_size), -fin_int, :] = 1

        else:
            mask = np.ones((batch_size, input_length, 2))

        return inputs.reshape((batch_size, input_length, 1)), outputs, mask
    
    return task



def angularintegration_task_constant(T, dt, speed_range=[-1,1], sparsity=1, last_mses=False, random_angle_init=False, max_input=None):
    """
    Creates N_batch trials of the angular integration task with Guassian Process angular velocity inputs.
    Inputs is angular velocity (postive: right, negative:left) and 
    target output is sine and cosine of integrated angular velocity.
    Returns inputs, outputs, mask
    -------
    """
    input_length = int(T/dt)
    
    def task(batch_size):
        
        if sparsity =='variable':
            sparsities = np.random.uniform(0, 2, batch_size)
            mask_input = np.random.random(size=(batch_size, input_length))<1-sparsities[:,None]
        elif sparsity:
            mask_input = np.random.random(size=(batch_size, input_length)) < 1-sparsity
        inputs_0 = np.random.uniform(low=speed_range[0], high=speed_range[1], size=batch_size)
        inputs = inputs_0.reshape(-1, 1) * np.ones((batch_size, input_length))
        if max_input:
            inputs = np.where(np.abs(inputs)>max_input, np.sign(inputs), inputs)
        if sparsity:
            inputs[mask_input] = 0.
        outputs_1d = np.cumsum(inputs, axis=1)*dt
        if random_angle_init=='equally_spaced':
            outputs_1d += np.arange(-np.pi, np.pi, 2*np.pi/batch_size)[:, np.newaxis]
        elif random_angle_init:
            random_angles = np.random.uniform(-np.pi, np.pi, size=batch_size)
            outputs_1d += random_angles[:, np.newaxis]

        outputs = np.stack((np.cos(outputs_1d), np.sin(outputs_1d)), axis=-1)

        if last_mses:
            fin_int = np.random.randint(1,last_mses,size=batch_size)
            mask = np.zeros((batch_size, input_length, 2))
            mask[np.arange(batch_size), -fin_int, :] = 1

        else:
            mask = np.ones((batch_size, input_length, 2))

        return inputs.reshape((batch_size, input_length, 1)), outputs, mask
    
    return task