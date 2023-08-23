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
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from models import train, mse_loss_masked, get_optimizer, get_loss_function, get_scheduler, RNN, train_lstm, LSTM_noforget, LSTM_noforget2, LSTM
from network_initialization import qpta_rec_weights
from tasks import angularintegration_task, eyeblink_task
from qpta_initializers import _qpta_tanh_hh

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_task(task_name = 'angularintegration', T=10, dt=.1, t_delay=50, sparsity=1, last_mses=None):
    
    if task_name == 'eyeblink':
        task =  eyeblink_task(input_length=T, t_delay=t_delay)

    elif task_name == 'angularintegration':
        task =  angularintegration_task(T=T, dt=dt, sparsity=sparsity, last_mses=last_mses)
    else:
        raise Exception("Task not known.")
        
    return task


def run_experiment(parameter_file_name, main_exp_name='', sub_exp_name='', model_name='', trials=10, training_kwargs={}):
    experiment_folder = parent_dir + '/experiments/' + main_exp_name +'/'+ sub_exp_name +'/'+ model_name
    print(experiment_folder)
    
    makedirs(experiment_folder) 
    
    for trial in tqdm.tqdm(range(trials)):
        run_single_training(parameter_file_name, exp_name= main_exp_name +'/'+ sub_exp_name +'/'+ model_name, trial=trial, training_kwargs=training_kwargs)

def run_single_training(parameter_file_name, exp_name='', trial=None, save=True, training_kwargs={}):
    
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    parameter_path = parent_dir + '/experiments/' + parameter_file_name
    if not training_kwargs['record_step']:
        training_kwargs['record_step'] = training_kwargs['n_epochs'] 
    
    experiment_folder = parent_dir + '/experiments/' + exp_name 
    
    if training_kwargs=={}:
        training_kwargs = yaml.safe_load(Path(parameter_path).read_text())
        if save:
            shutil.copy(parameter_path, experiment_folder)
    else:
        if save:
            with open(experiment_folder+'/parameters.yml', 'w') as outfile:
                yaml.dump(training_kwargs, outfile, default_flow_style=False)
        

    if training_kwargs['task']==None:
        task = None
        data = np.load(parent_dir+"/experiments/datasets/"+training_kwargs['dataset_filename'])
    else:
        data = None
        task = get_task(task_name=training_kwargs['task'], T=training_kwargs['T'], dt=training_kwargs['dt_task'],
                        sparsity=training_kwargs['task_sparsity'], last_mses=training_kwargs['last_mses'])
    # if training_kwargs['dataset_filename']!='':

    
    dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'])
    
    if training_kwargs['network_type'] == 'lstm_noforget':
        net = LSTM_noforget((training_kwargs['N_in'],training_kwargs['N_rec'],training_kwargs['N_out']),
                            readout_nonlinearity=training_kwargs['readout_nonlinearity'], dropout=training_kwargs['drouput'])
        
        result = train_lstm(net, task=task, n_epochs=training_kwargs['n_epochs'],
              batch_size=training_kwargs['batch_size'], learning_rate=training_kwargs['learning_rate'],
              clip_gradient=training_kwargs['clip_gradient'], cuda=training_kwargs['cuda'], init_states=None,
              loss_function=training_kwargs['loss_function'], final_loss=training_kwargs['final_loss'], last_mses=training_kwargs['last_mses'], 
              optimizer=training_kwargs['optimizer'], momentum=training_kwargs['adam_momentum'], weight_decay=training_kwargs['weight_decay'],
              adam_betas=(training_kwargs['adam_beta1'],training_kwargs['adam_beta2']), adam_eps=1e-8,  #optimizers 
              scheduler=training_kwargs['scheduler'], scheduler_step_size=training_kwargs['scheduler_step_size'], scheduler_gamma=training_kwargs['scheduler_gamma'], 
              verbose=training_kwargs['verbose'], record_step=training_kwargs['record_step'])
        
    elif training_kwargs['network_type'] == 'lstm':
        net = LSTM((training_kwargs['N_in'],training_kwargs['N_rec'],training_kwargs['N_out']), readout_nonlinearity=training_kwargs['readout_nonlinearity'])
        
        result = train_lstm(net, task=task, n_epochs=training_kwargs['n_epochs'],
              batch_size=training_kwargs['batch_size'], learning_rate=training_kwargs['learning_rate'],
              clip_gradient=training_kwargs['clip_gradient'], cuda=training_kwargs['cuda'], init_states=None,
              loss_function=training_kwargs['loss_function'], final_loss=training_kwargs['final_loss'], last_mses=training_kwargs['last_mses'], 
              optimizer=training_kwargs['optimizer'], momentum=training_kwargs['adam_momentum'], weight_decay=training_kwargs['weight_decay'],
              adam_betas=(training_kwargs['adam_beta1'],training_kwargs['adam_beta2']), adam_eps=1e-8, #optimizers 
              scheduler=training_kwargs['scheduler'], scheduler_step_size=training_kwargs['scheduler_step_size'], scheduler_gamma=training_kwargs['scheduler_gamma'], 
              verbose=training_kwargs['verbose'], record_step=training_kwargs['record_step'])
    else:
        if training_kwargs['initialization_type'] == 'gain':
            wrec_init, brec_init = None, None
            
        elif training_kwargs['initialization_type'] == 'qpta':
            # N_blas = int(training_kwargs['N_rec']/2)
            # wrec_init, brec_init = qpta_rec_weights(N_in=training_kwargs['N_in'], N_blas=N_blas, N_out=training_kwargs['N_in']);
            brec_init = np.zeros(training_kwargs['N_rec'])
            wrec_init = _qpta_tanh_hh()((training_kwargs['N_rec'],training_kwargs['N_rec']))
            
        elif training_kwargs['initialization_type'] == 'ortho':
            H = np.random.randn(training_kwargs['N_rec'], training_kwargs['N_rec'])
            wrec_init, _ = qr(H)
            brec_init = np.array([0]*training_kwargs['N_rec'])
    
        else:
            raise Exception("Recurrent weight initialization not known.")
            
        net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt_rnn'], g=training_kwargs['rnn_init_gain'], g_in=training_kwargs['g_in'],
                  nonlinearity=training_kwargs['nonlinearity'], readout_nonlinearity=training_kwargs['readout_nonlinearity'],
                  wi_init=None, wrec_init=wrec_init, wo_init=None, brec_init=brec_init, h0_init=None, ML_RNN=training_kwargs['ml_rnn'])

        result = train(net, task=task, data=data, n_epochs=training_kwargs['n_epochs'],
              batch_size=training_kwargs['batch_size'], learning_rate=training_kwargs['learning_rate'],
              clip_gradient=training_kwargs['clip_gradient'], cuda=training_kwargs['cuda'], h_init=None,
              loss_function=training_kwargs['loss_function'], act_norm_lambda=training_kwargs['act_norm_lambda'],
              optimizer=training_kwargs['optimizer'], momentum=training_kwargs['adam_momentum'], weight_decay=training_kwargs['weight_decay'],
              adam_betas=(training_kwargs['adam_beta1'],training_kwargs['adam_beta2']), adam_eps=1e-8, #optimizers 
              scheduler=training_kwargs['scheduler'], scheduler_step_size=training_kwargs['scheduler_step_size'], scheduler_gamma=training_kwargs['scheduler_gamma'], 
              verbose=training_kwargs['verbose'], record_step=training_kwargs['record_step'])
    
    
    #save result (losses, weights, etc)
    if save:
        result.append(training_kwargs)
        with open(experiment_folder + '/results_%s.pickle'%timestamp, 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return result
    
def grid_search(parameter_file_name, param_grid, experiment_folder, parameter_path='', sub_exp_name='', trials=1):
    """Perform a grid search for the optimal hyperparameters for training"""

    experiment_path = parent_dir +  '/experiments/' + experiment_folder
    makedirs(experiment_path)
    makedirs(experiment_path + '/' + sub_exp_name)

    assert trials>=1
    if not parameter_path:
        parameter_path = parent_dir +  '/experiments/' + experiment_folder +'/'+ parameter_file_name
    training_kwargs = yaml.safe_load(Path(parameter_path).read_text())
    
    with open(parent_dir +  '/experiments/' + experiment_folder+'/parameters.yml', 'w') as outfile:
        yaml.dump(training_kwargs, outfile, default_flow_style=False)
    
    #create grid of parameters
    keys, values = zip(*param_grid.items())
    all_param_combs = [dict(zip(keys, p)) for p in product(*values)]
    
    # columns = [i for i in range(training_kwargs['n_epochs'])]
    # columns.extend(['final_loss'])
    # columns.extend(keys)
    # columns.extend(['trial'])
    
    columns = ['losses', 'final_loss', 'trial', 'weights_last']
    columns.extend(keys)
    
    #find params that are varied
    varied_params = [key for key in param_grid.keys() if len(param_grid[key])>1]
    
    L = []
    for param_i, param_comb in tqdm.tqdm(enumerate(all_param_combs)):
        print(param_comb, "   #", param_i+1, "/", len(all_param_combs))

        exp_name = ''
        for i in range(len(varied_params)):
            exp_name += '_' + varied_params[i] + str(param_comb[varied_params[i]]) 
        for key in param_comb:
            training_kwargs[key] = param_comb[key]
        
        for trial in tqdm.tqdm(range(trials)):
            print("Trial", trial)
            result = run_single_training(parameter_file_name,
                                        exp_name='',
                                        trial=trial,
                                        save=False,
                                        training_kwargs=training_kwargs)
            losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
            dat = [losses]
            dat.append(losses[-1])
            dat.append(trial)
            dat.append(weights_last)
            dat.extend(param_comb.values())
            L.append(dat)
            result.append(training_kwargs)
            with open(experiment_path+"/"+sub_exp_name+"/" + 'results%s.pickle'%exp_name+'_'+str(trial), 'wb') as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    df = pd.DataFrame(L, columns = columns)
    param_keys = list(param_grid.keys())
    all_keys = param_keys
    all_keys.extend(['final_loss', 'trial'])
    df_final = df[all_keys]
    
    save_path = parent_dir+'/experiments/'+experiment_folder+'/grid_search_'+sub_exp_name
    # with np.printoptions(linewidth=10000):
    #     df.to_csv(save_path+'.csv', index=False)
        
    df.to_pickle(save_path+'.pickle')
    
    min_final_loss = df.iloc[df['final_loss'].idxmin()]
    df_final = df_final.fillna("None")
    min_final_meanloss = df_final.groupby(param_keys).mean().idxmin()
    return df, df_final, min_final_loss, min_final_meanloss

def grid_search_all_models(experiment_folder, model_names, param_grid):
    for model_i, model_name in enumerate(model_names):
        training_kwargs['network_type'] = network_types[model_i]
        training_kwargs['N_rec'] = nrecs[model_i]
        training_kwargs['loss_function'] = loss_functions[model_i]
        training_kwargs['initialization_type'] = initialization_type_list[model_i]
        training_kwargs['rnn_init_gain'] = g_list[model_i]
        df, df_final, min_final_loss, min_final_meanloss = grid_search(parameter_file_name, param_grid=param_grid,
                                                                        experiment_folder=experiment_folder,
                                                                        sub_exp_name=model_name,
                                                                        parameter_path=parameter_path, trials=2)


def trial_length_experiment(main_exp_name, sub_exp_name):
    
    scheduler_step_sizes = [100, 300, 100, 100, 300]
    gammas = [0.75, 0.5, 0.75, 0.75, .5]
    nrecs = [115, 200, 200, 200, 200]
    trial_lengths = [12.8, 51.2, 102.4, 409.6]
    for T in tqdm.tqdm(trial_lengths):
        training_kwargs['T'] = T
        for model_i, model_name in tqdm.tqdm(enumerate(model_names)):
            training_kwargs['network_type'] = network_types[model_i]
            training_kwargs['initialization_type'] = initialization_type_list[model_i]
            training_kwargs['N_rec'] = nrecs[model_i]
            training_kwargs['loss_function'] = loss_functions[model_i]
            training_kwargs['rnn_init_gain'] = g_list[model_i]
            training_kwargs['scheduler_step_size'] = scheduler_step_sizes[model_i]
            training_kwargs['scheduler_gamma'] = gammas[model_i]
            run_experiment('/parameter_files/'+parameter_file_name, main_exp_name=main_exp_name,
                                                                    sub_exp_name=sub_exp_name+f'/T{T}',
                                                                  model_name=model_name, trials=11, training_kwargs=training_kwargs)

def size_experiment(main_exp_name, sub_exp_name):
    
    network_types = ['lstm_noforget', 'rnn', 'rnn', 'rnn', 'rnn']
    model_names = ['lstm', 'low', 'high', 'ortho', 'qpta']
    initialization_type_list =['', 'gain','gain', 'ortho', 'qpta']
    loss_functions = ['mse', 'mse_loss_masked', 'mse_loss_masked', 'mse_loss_masked', 'mse_loss_masked']
    g_list = [0., .5, 1.5, 0., 0.]
    scheduler_step_sizes = [100, 300, 100, 100, 300]
    gammas = [0.75, 0.5, 0.75, 0.75, .75]
    learning_rates = [1e-3, 1e-3, 1e-3, 1e-3, 1e-4]
    
    nrecs_lists = [[2, 6, 6, 6, 6],
                   [8, 16, 16, 16, 16],
                   [32, 58, 58, 58, 58], 
                   [115, 200, 200, 200, 200],
                   [128, 224, 224, 224, 224]]
    
    training_kwargs['T'] = 25.6
    for nrecs_list in tqdm.tqdm(nrecs_lists):

        for model_i, model_name in tqdm.tqdm(enumerate(model_names)):
            training_kwargs['learning_rate'] = learning_rates[model_i]
            training_kwargs['network_type'] = network_types[model_i]
            training_kwargs['initialization_type'] = initialization_type_list[model_i]
            training_kwargs['N_rec'] = nrecs_list[model_i]
            training_kwargs['loss_function'] = loss_functions[model_i]
            training_kwargs['rnn_init_gain'] = g_list[model_i]
            training_kwargs['scheduler_step_size'] = scheduler_step_sizes[model_i]
            training_kwargs['scheduler_gamma'] = gammas[model_i]
            run_experiment('/parameter_files/'+parameter_file_name, main_exp_name=main_exp_name,
                                                                    sub_exp_name=sub_exp_name+f'/{nrecs_list[0]}',
                                                                  model_name=model_name, trials=11, training_kwargs=training_kwargs)
    
    

    
if __name__ == "__main__":
    print(current_dir)
    parameter_file_name = 'params_ang_sparse.yml'
    parameter_path = parent_dir + '/experiments/parameter_files/'+ parameter_file_name
    training_kwargs = yaml.safe_load(Path(parameter_path).read_text())
    
    network_types = ['lstm_noforget', 'rnn', 'rnn', 'rnn', 'rnn']
    model_names = ['lstm', 'low', 'high', 'ortho', 'qpta']
    initialization_type_list =['', 'gain','gain', 'ortho', 'qpta']
    loss_functions = ['mse', 'mse_loss_masked', 'mse_loss_masked', 'mse_loss_masked', 'mse_loss_masked']
    g_list = [0., .5, 1.5, 0., 0.]
    scheduler_step_sizes = [100, 300, 100, 100, 300]
    gammas = [1., 0.5, 0.75, 0.75, 1.]
    nrecs = [115, 200, 200, 200, 200]
    
    # size_experiment(main_exp_name='angularintegration', sub_exp_name='lambda')
    
    main_exp_name = 'angularintegration'
    sub_exp_name  = 'act_norm'

    model_i, model_name = 2, 'high'
    # model_i, model_name = 3, 'ortho'
    # model_i, model_name = 4, 'qpta'
    # model_i, model_name = 0, 'lstm'

    # training_kwargs['clip_gradient'] = 
    training_kwargs['task'] = 'angularintegration'
    # training_kwargs['nonlinearity'] = 'relu'
    training_kwargs['act_norm_lambda'] = 1e-5

    # training_kwargs['dataset_filename'] = 'dataset_T256_BS1024.npz'
    training_kwargs['batch_size'] = 128
    training_kwargs['weight_decay'] = 0.
    training_kwargs['drouput'] = .0
    training_kwargs['g_in'] = 14.142135623730951 #np.sqrt(nrecs[model_i])
    training_kwargs['verbose'] = True
    training_kwargs['learning_rate'] = 1e-3
    training_kwargs['n_epochs'] = 1000
    training_kwargs['T'] = 12.8 #
    training_kwargs['dt_rnn'] = .1
    training_kwargs['adam_beta1'] = 0.9
    training_kwargs['adam_beta2'] = 0.999
    training_kwargs['network_type'] = network_types[model_i]
    training_kwargs['initialization_type'] = initialization_type_list[model_i]
    # training_kwargs['N_rec'] = 200
    training_kwargs['loss_function'] = loss_functions[model_i]
    training_kwargs['rnn_init_gain'] = g_list[model_i]        ##########
    training_kwargs['scheduler_step_size'] = scheduler_step_sizes[model_i]
    training_kwargs['scheduler_gamma'] = gammas[model_i]

    run_experiment('/parameter_files/'+parameter_file_name, main_exp_name=main_exp_name,
                                                            sub_exp_name=sub_exp_name,
                                                          model_name=model_name, trials=11, training_kwargs=training_kwargs)

    
    param_grid = {'initialization_type': ['qpta'],
                  'g': [.5],
                'T': [25.6],
                  'dt_rnn': [.1],
                  'g_in': [1e-3, 1e-2],
                  'learning_rate':[1e-2],
                  'batch_size': [128],
                  'optimizer': ['adam'],
                  'n_epochs': [1000],
                  'scheduler_step_size':[1000],
                  'adam_beta1': [.8], #[.7, .8, .9, .95],
                  'adam_beta2': [0.99], # [.9, .99, .999, .9999],
                  'scheduler_gamma':[1.],
                  'scheduler': ['steplr'],
                  'clip_gradient': [None]}
    # df, df_final, min_final_loss, min_final_meanloss = grid_search(parameter_file_name, param_grid=param_grid,
    #                                                                 experiment_folder='angularintegration/gin256_lr1e-2',
    #                                                                 sub_exp_name='qpta',
    #                                                                 parameter_path=parameter_path, trials=3)
