# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 18:31:06 2023

@author: abel_
"""
import os, sys
import glob
current_dir = os.path.dirname(os.path.realpath('__file__'))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

import yaml
from pathlib import Path

from network_training import *
from rnn_models import *
from network_initialization import *

def load_model(load_model_name, N_in, N_rec, N_out, N_blas, hidden_offset=0, hidden_initial_activations='random',
               rnn_init_gain=1., random_winout=False, perfect_inout_variance=0., device='cpu'):
    """
    Loads RNN model from file or a specified intialization.

    Parameters
    ----------
    load_model_name : str
        Initialization method.
        
    N_in : int
        DESCRIPTION.
    N_rec : int
        DESCRIPTION.
    N_out : int
        DESCRIPTION.
    N_blas : int
        DESCRIPTION.
    hidden_offset : float
        DESCRIPTION.
    hidden_initial_activations : float
        DESCRIPTION.
    rnn_init_gain : float
        DESCRIPTION.

    random_winout : TYPE, optional
        DESCRIPTION. The default is False.
    perfect_inout_variance : TYPE, optional
        DESCRIPTION. The default is 0.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.

    Returns
    -------
    rnn_model : TYPE
        DESCRIPTION.

    """
        
        
    if load_model_name[0]=='v': #if in form v{i}, e.g. v1, v2 ,v3
        rnn_model = perfect_initialization(int(load_model_name[1]), random_winout=random_winout, eps=perfect_inout_variance)#.double()
        
    elif load_model_name=='blas':
        N_rec = 2*N_blas
        rnn_model = bla_initialization(N_in, N_blas, N_out, hidden_offset, hidden_initial_activations=hidden_initial_activations)
    
    elif load_model_name=='ublas':
        N_rec = 2*N_blas
        rnn_model = ubla_initialization(N_in, N_blas, N_out, hidden_offset, hidden_initial_activations=hidden_initial_activations)
    
    elif load_model_name=='identity':
        rnn_model = identity_initialization(N_in, N_rec, N_out, hidden_initial_activations=hidden_initial_activations)
        
    elif load_model_name=='qpta':
        N_rec = 2*N_blas
        rnn_model = qpta_initialization(N_in, N_blas, N_out, hidden_initial_activations=hidden_initial_activations)
        
    elif load_model_name=='ortho':
        rnn_model = ortho_initialization(N_in, N_rec, N_out, hidden_initial_activations=hidden_initial_activations)
        
    elif load_model_name=='gain':
        rnn_model = gain_initialization(N_in, N_rec, N_out, gain=rnn_init_gain,  hidden_initial_activations=hidden_initial_activations)

                
    else:
        rnn_model.load_state_dict(torch.load(load_model_name))
    return rnn_model

def initialize_model(load_model_name, nonlinearity, N_in, N_rec, N_out, N_blas, model_type='rnn', hidden_offset=0.,
                     hidden_initial_variance=0., hidden_initial_activations='zero', rnn_init_gain=1.,
                     random_winout=False, perfect_inout_variance=0., output_activation='linear', N_inter=5, device='cpu'):
    """
    Initializes RNN randomly (possibly from an initialization method) or loads model from file.

    Parameters
    ----------
    load_model_name : TYPE
        DESCRIPTION.
    nonlinearity : TYPE
        DESCRIPTION.
    N_in : TYPE
        DESCRIPTION.
    N_rec : TYPE
        DESCRIPTION.
    N_out : TYPE
        DESCRIPTION.
    N_blas : TYPE
        DESCRIPTION.
    hidden_offset : TYPE
        DESCRIPTION.
    hidden_initial_activations : TYPE
        DESCRIPTION.
    rnn_init_gain : TYPE
        DESCRIPTION.
    random_winout : TYPE, optional
        DESCRIPTION. The default is False.
    perfect_inout_variance : TYPE, optional
        DESCRIPTION. The default is 0..
    output_activation : TYPE, optional
        DESCRIPTION. The default is 'linear'.
    N_inter : TYPE, optional
        DESCRIPTION. The default is 5.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.

    Returns
    -------
    rnn_model : TYPE
        DESCRIPTION.

    """
    
    if load_model_name!="":
        rnn_model = load_model(load_model_name, N_in, N_rec, N_out, N_blas, hidden_offset, hidden_initial_activations,
                       rnn_init_gain, random_winout=False, perfect_inout_variance=0., device='cpu')
    else:
        if model_type == 'rnn':

            rnn_model = RNNModel(N_in, N_out, N_rec, transform_function=transform_functionain,
                                             hidden_initial_variance=hidden_initial_variance,
                                             output_activation=output_activation, device=device).to(device)

        elif model_type == 'rnn_interactivation':
            rnn_model = RNN_interactivation(N_in, N_rec, N_inter, N_out, 
                                            hidden_initial_activations=hidden_initial_activations,
                                            hidden_initial_variance=hidden_initial_variance,
                                            output_activation=output_activation, device=device).to(device)

        # elif model_type == 'rnn_cued':
        #     N_in = 4
        #     if training_kwargs['task'] == "dynamic_poisson_clicks":
        #         N_in = 3
        #     rnn_model = RNNModel(N_in, N_out, N_rec, transform_function=transform_functionain,
        #                                      hidden_initial_activations=hidden_initial_activations,
        #                                      hidden_initial_variance=hidden_initial_variance,
        #                          output_activation=output_activation, device=device, constrain_spectrum=training_kwargs['constrain_spectrum']).to(device)
            
        elif model_type == 'lstm':
            rnn_model = LSTM(N_in, N_out, N_rec).to(device)
            
        elif model_type == 'gru':
            rnn_model = GRU(N_in, N_out, N_rec).to(device)
            

    return rnn_model

def main():
    training_kwargs = yaml.safe_load(Path(parent_dir + "\\experiments\\parameter_files\\parameters000.yaml").read_text())

    print(training_kwargs)

    model = initialize_model(training_kwargs['load_model_name'], training_kwargs['nonlinearity'],
                                 training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'], training_kwargs['N_blas'])

    # torch.save(model.state_dict(), exp_path + '/starting_weights.pth')    
    
    train_x, train_y, train_output_mask, _ = data_lists[0]
    test_x, test_y, test_output_mask, _ = data_lists[1]
    dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float, device=training_kwargs['device']),
                            torch.tensor(train_y, dtype=torch.float, device=training_kwargs['device']),
                            torch.tensor(train_output_mask, dtype=torch.float))
    data_loader = DataLoader(dataset, batch_size=training_kwargs['dataloader_batch_size'], shuffle=True)
    
    train(model, optimizer='sgd', loss_function='mse', device='cpu', scheduler_name=None,
              learning_rate=0.001, weight_decay=0., momentum=0., adam_betas=(0.9, 0.999))
    
    
if __name__ == "__main__":
    main()
    
