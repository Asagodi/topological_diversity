# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:36:51 2025

@author: abel_
"""

import torch

def simulate_rnn(net, task, T, batch_size):
    input, target, mask = task(batch_size); input = torch.from_numpy(input).float();
    output, trajectories = net(input, return_dynamics=True); 
    output = output.detach().numpy();
    trajectories = trajectories.detach().numpy()
    return input, target, mask, output, trajectories


