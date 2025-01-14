# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:22:57 2025

@author: 
"""

import pickle
import glob
import yaml
from pathlib import Path

from models import RNN

def get_params_exp(params_folder, exp_i=0, which='post'):
    
    exp_list = glob.glob(params_folder + "/res*")
    exp = exp_list[exp_i]
    params_path = glob.glob(params_folder + '/param*.yml')[0]
    training_kwargs = yaml.safe_load(Path(params_path).read_text())
    with open(exp, 'rb') as handle:
        result = pickle.load(handle)
        
        if len(result) == 9:
            losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs, training_kwargs = result
        elif len(result) == 8:
            losses, validation_losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result
        elif len(result) == 7:
            losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs = result


    if len(weights_last) == 6:
        if which=='post':
            wi, wrec, wo, brec, h0, oth = weights_last
        elif which=='pre':
            wi, wrec, wo, brec, h0, oth = weights_init
            
        else:
            return weights_train["wi"][which], weights_train["wrec"][which], weights_train["wo"][which], weights_train["brec"][which], weights_train["h0"][which], weights_train["oths"][which], training_kwargs, losses
        return wi, wrec, wo, brec, h0, oth, training_kwargs, losses

    else:
        if which=='post':
            wi, wrec, wo, brec, h0 = weights_last
        elif which=='pre':
            wi, wrec, wo, brec, h0 = weights_init
        else:
            return weights_train["wi"][which], weights_train["wrec"][which], weights_train["wo"][which], weights_train["brec"][which], weights_train["h0"][which], None,  training_kwargs, losses

        return wi, wrec, wo, brec, h0, None, training_kwargs, losses

    

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

