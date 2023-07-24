# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:06:58 2023

@author: abel_
"""

import numpy as np

import scipy
from scipy.stats import truncexpon




def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)



#########################################
def eyeblink_task(input_length, t_delay, t_stim=1, t_target=1, min_us_time=5, max_us_time=20):
    """
    Creates N_batch trials of the eyeblick conditioning task.
    
    Parameters
    ----------
    N_batch : int
        Number of trials
    input_length : int
        Number of time steps for each trial
    t_stim : int
        Length of unconditioned stimulus (US)
    t_delay : int
        Length of delay
    t_target: int
        Length of target output
    Returns inputs, outputs, mask
    -------
    The US timings are taken from uniform U(min_us_time,max_us_time)
    
    
    """
    assert input_length > max_us_time+t_delay+t_stim, "input_length needs to be bigger than max_us_time+t_delay+t_stim!"
    
    def task(batch_size):
    
        inputs = np.zeros((batch_size, input_length, 1))
        outputs = np.zeros((batch_size, input_length, 1))
        
        cs_timings = np.random.randint(min_us_time,max_us_time, batch_size)
        for i in range(batch_size):
            inputs[i,cs_timings[i]:cs_timings[i]+t_stim,0] = 1
            outputs[i,cs_timings[i]+t_delay+t_stim:cs_timings[i]+t_stim+t_delay+t_target,0] = 1
        
        mask = np.ones((batch_size, input_length, 1))
        return inputs, outputs, mask

    return task



#########################################
def angularintegration_task(T, dt, length_scale=1):
    """
    Creates N_batch trials of the angular integration task.
    Inputs are left and right angular velocity and 
    target output is sine and cosine of integrated angular velocity.
    Returns inputs, outputs, mask, 0
    -------
    """
    input_length = int(T/dt)
    
    def task(batch_size):
        
        X = np.expand_dims(np.linspace(-length_scale, length_scale, input_length), 1)
        sigma = exponentiated_quadratic(X, X)  
        inputs = np.random.multivariate_normal(mean=np.zeros(input_length), cov=sigma, size=batch_size)
        outputs_1d =  np.cumsum(inputs, axis=1)*dt
        outputs = np.stack((np.cos(outputs_1d), np.sin(outputs_1d)), axis=-1)
        mask = np.ones((batch_size, input_length, 2))

        return inputs.reshape((batch_size, input_length, 1)), outputs, mask
    
    return task