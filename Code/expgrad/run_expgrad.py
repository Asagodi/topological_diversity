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

from models import train, mse_loss_masked, get_optimizer, get_loss_function, get_scheduler, RNN
from network_initialization import qpta_rec_weights, bla_rec_weights
from tasks import *
from qpta_initializers import _qpta_tanh_hh
from plot_losses import get_params_exp
from run_training_s import *

print(current_dir)
parameter_file_name = 'params.yml'
training_kwargs = yaml.safe_load(Path(parent_dir + '/experiments/expgrad/'+ parameter_file_name).read_text())
    
training_kwargs['wrec_init_std'] =  1e-1

main_exp_name = '/expgrad/single_pert/single_pulse/' 
training_kwargs['wrec_perturbation'] = True

training_kwargs['task_name'] = 'single_pulse'
training_kwargs['T'] = 1024 * 4
training_kwargs['N_in'] = 2
training_kwargs['N_out'] = 1
training_kwargs['N_rec'] = 2
training_kwargs['b_a'] = 5
training_kwargs['alpha'] = 1
training_kwargs['h0_init'] = 'self_h'

training_kwargs['input_length'] = 1
training_kwargs['input_noise_level'] = 0.
training_kwargs['step_size'] = 1
training_kwargs['fixed_step'] = True

training_kwargs['last_mses'] = 1
training_kwargs['dt_rnn'] = 1
training_kwargs['dt_task'] = 1

training_kwargs['task_noise_sigma'] = 0 
training_kwargs['hidden_initial_variance'] = 0

training_kwargs['batch_size'] = 32
training_kwargs['n_epochs'] = 100
training_kwargs['record_step'] = 1

training_kwargs['verbose'] = True
training_kwargs['fix_seed'] = True



def run_all(ntrials=10):
    
    learning_rates = np.logspace(-9, -2, 8)
    initialization_types = ['perfect_irnn', 'perfect_ubla', 'perfect_bla']
    
    for lr in learning_rates:
        training_kwargs['learning_rate'] = lr
        for it in initialization_types:
            training_kwargs['initialization_type'] = it
            lr
            sub_exp_name = f"/T{training_kwargs['T']}/wstd{training_kwargs['wrec_init_std']}/{it[8:]}_lr{-int(np.log10(lr))}"
            run_experiment('/parameter_files/'+parameter_file_name, main_exp_name=main_exp_name,
                                                                    sub_exp_name=sub_exp_name,
                                                                  model_name="", trials=ntrials, training_kwargs=training_kwargs)

  
if __name__ == "__main__":

    run_all(ntrials=10)
