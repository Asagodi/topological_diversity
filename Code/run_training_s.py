# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:25:35 2023

@author: abel_
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
from scipy.linalg import qr
import numpy as np
import pandas as pd
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim

from schuessler_model import train, mse_loss_masked, get_optimizer, get_loss_function, get_scheduler, RNN
from network_initialization import qpta_rec_weights
from tasks import angularintegration_task, eyeblink_task

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_task(task_name = 'angularintegration', T=10, dt=.1, t_delay=50, sparsity=1):
    
    if task_name == 'eyeblink':
        task =  eyeblink_task(input_length=T, t_delay=t_delay)

    elif task_name == 'angularintegration':
        task =  angularintegration_task(T=T, dt=dt, sparsity=sparsity)
    else:
        raise Exception("Task not known.")
        
    return task


def run_experiment(parameter_file_name, main_exp_name='', sub_exp_name='', model_name='', trials=10, training_kwargs={}):
    experiment_folder = parent_dir + '\\experiments\\' + main_exp_name +'\\'+ sub_exp_name +'\\'+ model_name
    print(experiment_folder)
    
    makedirs(experiment_folder) 
    
    for trial in range(trials):
        run_single_training(parameter_file_name, exp_name= main_exp_name +'\\'+ sub_exp_name +'\\'+ model_name, trial=trial, training_kwargs=training_kwargs)

def run_single_training(parameter_file_name, exp_name='', trial=None, save=True, training_kwargs={}):
    
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    parameter_path = parent_dir + '\\experiments\\' + parameter_file_name
    # if not training_kwargs['record_step']:
    #     training_kwargs['record_step'] = training_kwargs['n_epochs'] 
    
    experiment_folder = parent_dir + '\\experiments\\' + exp_name 
    
    if training_kwargs=={}:
        training_kwargs = yaml.safe_load(Path(parameter_path).read_text())
        if save:
            shutil.copy(parameter_path, experiment_folder)
    else:
        if save:
            with open(experiment_folder+'//parameters.yml', 'w') as outfile:
                yaml.dump(training_kwargs, outfile, default_flow_style=False)
        
        
        
    
    dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
    
    if training_kwargs['initialization_type'] == 'gain':
        wrec_init, brec_init = None, None
        
    elif training_kwargs['initialization_type'] == 'qpta':
        N_blas = int(training_kwargs['N_rec']/2)
        wrec_init, brec_init = wrec_init, brec_init = qpta_rec_weights(N_in=training_kwargs['N_in'], N_blas=N_blas, N_out=training_kwargs['N_in']);

        
    elif training_kwargs['initialization_type'] == 'ortho':
        H = np.random.randn(training_kwargs['N_rec'], training_kwargs['N_rec'])
        wrec_init, _ = qr(H)
        brec_init = np.array([0]*training_kwargs['N_rec'])

    else:
        raise Exception("Recurrent weight initialization not known.")
        
    net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'], 
              nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
              wi_init=None, wrec_init=wrec_init, wo_init=None, brec_init=brec_init, h0_init=None, ML_RNN=training_kwargs['ml_rnn'])

    
    task = get_task(task_name=training_kwargs['task'], T=training_kwargs['T'], dt=training_kwargs['dt_task'], sparsity=training_kwargs['task_sparsity'])
    
    result = train(net, task=task, n_epochs=training_kwargs['n_epochs'],
          batch_size=training_kwargs['batch_size'], learning_rate=training_kwargs['learning_rate'],
          clip_gradient=training_kwargs['clip_gradient'], cuda=False, h_init=None,
          loss_function=training_kwargs['loss_function'],
          optimizer=training_kwargs['optimizer'], momentum=training_kwargs['adam_momentum'], weight_decay=training_kwargs['weight_decay'], adam_betas=training_kwargs['adam_betas'], adam_eps=1e-8, #optimizers 
          scheduler=training_kwargs['scheduler'], scheduler_step_size=training_kwargs['scheduler_step_size'], scheduler_gamma=training_kwargs['scheduler_gamma'], 
          verbose=training_kwargs['verbose'], record_step=training_kwargs['record_step'])
    
    
    #save result (losses, weights, etc)
    if save:
        with open(experiment_folder + '\\results_%s.pickle'%timestamp, 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return result
    
def grid_search(parameter_file_name, param_grid, experiment_folder, parameter_path='', sub_exp_name='', trials=1):
    """Perform a grid search for the optimal hyperparameters for training"""

    makedirs(parent_dir +  '\\experiments\\' + experiment_folder)
    assert trials>1
    if not parameter_path:
        parameter_path = parent_dir +  '\\experiments\\' + experiment_folder +'//'+ parameter_file_name
    training_kwargs = yaml.safe_load(Path(parameter_path).read_text())
    
    #create grid of parameters
    keys, values = zip(*param_grid.items())
    all_param_combs = [dict(zip(keys, p)) for p in product(*values)]
    
    # columns = [i for i in range(training_kwargs['n_epochs'])]
    # columns.extend(['final_loss'])
    # columns.extend(keys)
    # columns.extend(['trial'])
    
    columns = ['losses', 'final_loss', 'trial', 'weights_last']
    columns.extend(keys)
    
    L = []
    for param_i, param_comb in enumerate(all_param_combs):
        print(param_comb, "   #", param_i+1, "/", len(all_param_combs))

        for key in param_comb:
            training_kwargs[key] = param_comb[key]
        
        for trial in range(trials):
            print("Trial", trial)

            losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = run_single_training(parameter_file_name,
                                                                                                                        exp_name='',
                                                                                                                        trial=None,
                                                                                                                        save=False,
                                                                                                                        training_kwargs=training_kwargs)
            dat = [losses]
            dat.append(losses[-1])
            dat.append(trial)
            dat.append(weights_last)
            dat.extend(param_comb.values())
            L.append(dat)
            
    df = pd.DataFrame(L, columns = columns)
    param_keys = list(param_grid.keys())
    all_keys = param_keys
    all_keys.extend(['final_loss', 'trial'])
    df_final = df[all_keys]
    df.to_csv(parent_dir+'//experiments//'+experiment_folder+'/grid_search_'+sub_exp_name+'.csv', index=False)
    
    min_final_loss = df.iloc[df['final_loss'].idxmin()]
    df_final = df_final.fillna("None")
    min_final_meanloss = df_final.groupby(param_keys).mean().idxmin()
    return df, df_final, min_final_loss, min_final_meanloss
    
if __name__ == "__main__":
    # parameter_file_name = '\\params_ortho_28-07-23.yml'

    parameter_file_name = '\\params_ang_sparse.yml'
    # run_single_training('\\parameter_files\\'+parameter_file_name)
    
    parameter_path = parent_dir + '\\experiments\\' +'\\parameter_files\\'+ parameter_file_name

    training_kwargs = yaml.safe_load(Path(parameter_path).read_text())
    
    model_names = ['low', 'high', 'ortho', 'qpta']
    initialization_type_list =['gain','gain', 'ortho', 'qpta']
    mlrnn_list = [True, True, True, True]
    # mlrnn_list = [False, False, False, False]
    g_list = [.5, 1.5, 1., 1.]
    scheduler_step_sizes = [200, 500, 500, 100]
    gammas = [0.85, 0.85, 0.85, .75]
    for model_i, model_name in enumerate(model_names):
    # if True:
        # model_i, model_name = 3, 'qpta'
        training_kwargs['initialization_type'] = initialization_type_list[model_i]
        training_kwargs['ml_rnn'] = mlrnn_list[model_i]
        training_kwargs['rnn_init_gain'] = g_list[model_i]
        training_kwargs['scheduler_step_size'] = scheduler_step_sizes[model_i]
        training_kwargs['scheduler_gamma'] = gammas[model_i]
        run_experiment('\\parameter_files\\'+parameter_file_name, main_exp_name='angularintegration',
                                                              sub_exp_name='final_posthyp51',
                                                              model_name=model_name, trials=3, training_kwargs=training_kwargs)
    
    model_names = ['qpta']
    # param_grid = {'learning_rate':[1e-2,1e-3],
    #               'batch_size': [100],
    #               'optimizer': ['adam'],
    #               'scheduler': ['steplr'],
    #               'n_epochs': [1000],
    #               'scheduler_step_size':[200,500],
    #               'scheduler_gamma':[0.5, 0.85],
    #               'clip_gradient': [None]}
    # for model_i, model_name in enumerate(model_names):
    #     print(model_name)
    #     training_kwargs['initialization_type'] = initialization_type_list[model_i]
    #     training_kwargs['ml_rnn'] = mlrnn_list[model_i]
    #     training_kwargs['rnn_init_gain'] = g_list[model_i]
    #     df, df_final, min_final_loss, min_final_meanloss = grid_search(parameter_file_name, param_grid=param_grid,
    #                                                                     experiment_folder='angularintegration//final_grid//',
    #                                                                     sub_exp_name=model_name,
    #                                                                     parameter_path=parameter_path, trials=2)
