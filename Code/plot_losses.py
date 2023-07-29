# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 18:58:02 2023

@author: abel_
"""

import glob
import os, sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir) 

import pickle
import yaml

import matplotlib.pyplot as plt

with open('C:\\Users\\abel_\\Documents\\Lab\\Projects\\limit_cycle_document\\ALE\\experiments\\results_ortho_2023-07-28-18-24-41.pickle', 'rb') as handle: result = pickle.load(handle)

exp_list = glob.glob(parent_dir+"\\experiments\\\high_gain*")
for exp in exp_list:
    print(exp)

# input, target, mask, output, loss = run_net(net, task, batch_size=32, return_dynamics=False, h_init=None); plt.plot(input[0,...]); plt.plot(target[0,...]); plt.plot(output[0,...])

# weights_last = result[3]; wi_init, wrec_init, wo_init, brec_init, h0_init = weights_last; training_kwargs = yaml.safe_load(Path('params_ortho_28-07-23.yml').read_text()); dims = (training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out']); net = RNN(dims=dims, noise_std=training_kwargs['noise_std'], dt=training_kwargs['dt'], g=training_kwargs['rnn_init_gain'], nonlinearity=training_kwargs['nonlinearity'],
 # readout_nonlinearity=training_kwargs['readout_nonlinearity'], wi_init=wi_init, wrec_init=wrec_init, wo_init=wo_init, brec_init=brec_init, h0_init=h0_init, ML_RNN=training_kwargs['ml_rnn'])