# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:25:35 2023

@author: 
"""

import os, sys
import glob 
current_dir = os.path.dirname(os.path.realpath('__file__')) 
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
import pickle
import time

import yaml
import shutil
from pathlib import Path
from scipy.linalg import qr, block_diag

import numpy as np
import pandas as pd
from itertools import product
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from models import train, mse_loss_masked, get_optimizer, get_loss_function, get_scheduler, RNN, train_lstm, LSTM_noforget, LSTM_noforget2, LSTM
from network_initialization import qpta_rec_weights, bla_rec_weights
from tasks import *
from qpta_initializers import _qpta_tanh_hh
from plot_losses import get_params_exp
from run_training_s import *

  
if __name__ == "__main__":
    print(current_dir)
    parameter_file_name = 'params.yml'
    training_kwargs = yaml.safe_load(Path(parent_dir + '/experiments/expgrad/'+ parameter_file_name).read_text())
        
    main_exp_name = '/expgrad/bernoulli_integration/' 
    
    training_kwargs['T'] = 1024 * 4
    training_kwargs['N_in'] = 2
    training_kwargs['N_out'] = 1
    training_kwargs['N_rec'] = 2
    training_kwargs['b_a'] = 5
    training_kwargs['alpha'] = 1
    training_kwargs['h0_init'] = 'self_h'
    
    training_kwargs['input_length'] = 1
    training_kwargs['input_noise_level'] = 0.1
    training_kwargs['last_mses'] = 1
    training_kwargs['dt_rnn'] = .1
    training_kwargs['dt_task'] = .1
    
    training_kwargs['task_noise_sigma'] = 0 
    training_kwargs['hidden_initial_variance'] = 0
    
    training_kwargs['batch_size'] = 32
    training_kwargs['learning_rate'] = 0.01
    training_kwargs['n_epochs'] = 100
    training_kwargs['record_step'] = 1

    training_kwargs['initialization_type'] = 'perfect_irnn'
    training_kwargs['verbose'] = True
    training_kwargs['fix_seed'] = True
    
    sub_exp_name = f"/T{training_kwargs['T']}/{training_kwargs['initialization_type']}"
    run_experiment('/parameter_files/'+parameter_file_name, main_exp_name=main_exp_name,
                                                            sub_exp_name=sub_exp_name,
                                                          model_name="", trials=1, training_kwargs=training_kwargs)

