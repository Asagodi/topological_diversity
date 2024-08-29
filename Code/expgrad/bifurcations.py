# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:51:36 2024

@author: abel_
"""
import os, sys
import glob 
current_dir = os.path.dirname(os.path.realpath('__file__')) 
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import RNN
from tasks import singlepulse_integration_task
from network_initialization import perfect_params
from angular_loss_analysis import simulate_rnn_with_task

def remove_nans(data):
    return [group[~np.isnan(group)] for group in data]

def function():
    alpha, beta = 1,5 
    rnn_noise_std = 0.
    dims = (2, 2, 1)
    T=int(1e3)
    dt=.1
    task = singlepulse_integration_task(T, dt, final_loss=True,steps_size=1)
    np.random.seed(100)
    eps=1e-1
    all_mses_01 = []
    for v in tqdm([1,2,3]):

        wi_init, wrec_init, wo_init, brec_init, bo_init, h0_init  = perfect_params(v, ouput_bias_value=1, a=alpha, output_bias_value=beta); 
        oth_init=None
        
        mses = []
        for i in range(100):
            wrec_init_p = wrec_init+ np.random.normal(0,eps, (2,2))
            net = RNN(dims=dims, noise_std=rnn_noise_std, dt=dt,g=1, g_in=1,
                  nonlinearity='relu', readout_nonlinearity='id',
                  wi_init=wi_init, wrec_init=wrec_init_p, wo_init=wo_init, brec_init=brec_init, bo_init=bo_init, h0_init=h0_init, oth_init=oth_init,
                  ML_RNN=True, save_inputs=False,
                  map_output_to_hidden=False, input_nonlinearity='');


            input, target, _, output, trajectories = simulate_rnn_with_task(net, task, T, 'self_h', batch_size=64);
            mses.append(np.mean((target[:,-1,:]-output[:,-1,:])**2))
        all_mses_01.append(mses)
    clean_data = remove_nans(np.log(all_mses_01));
    prob_exploding = np.sum(np.isnan(all_mses_01)| np.isinf(all_mses_01),axis=1)/len(all_mses_01[-1])
    
    
    
    labels = ['irnn', 'ubla', 'bla']
    exp_path = parent_dir+'/experiments/expgrad/'
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111);
    ax.violinplot(clean_data); ax.xticks([1, 2, 3], labels); # Create a plot

    # Add text annotations
    #for x, text in zip([1, 2, 3], prob_exploding):
    #    plt.text(x, 5e2, text, ha='center')
    #plt.text(0.5, 5e2, 'P(exploding)', ha='center')
    ax.set_ylabel("log mse"); #plt.yscale('symlog');
    plt.savefig(exp_path+'/logmse_dt.1.pdf');
    
    