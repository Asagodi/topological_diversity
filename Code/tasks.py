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
def angularintegration_task(T, dt, length_scale=1, sparsity=1, last_mses=None):
    """
    Creates N_batch trials of the angular integration task with Guassian Process angular velocity inputs.
    Inputs are left and right angular velocity and 
    target output is sine and cosine of integrated angular velocity.
    Returns inputs, outputs, mask
    -------
    """
    input_length = int(T/dt)
    
    def task(batch_size):
        
        X = np.expand_dims(np.linspace(-length_scale, length_scale, input_length), 1)
        sigma = exponentiated_quadratic(X, X)  
        if sparsity =='variable':
            sparsities = np.random.uniform(0, 2, batch_size)
            mask_input = np.random.random(size=(batch_size, input_length))<1-sparsities[:,None]
        elif sparsity:
            mask_input = np.random.random(size=(batch_size, input_length)) < 1-sparsity
        inputs = np.random.multivariate_normal(mean=np.zeros(input_length), cov=sigma, size=batch_size)
        if sparsity:
            inputs[mask_input] = 0.
        outputs_1d = np.cumsum(inputs, axis=1)*dt
        outputs = np.stack((np.cos(outputs_1d), np.sin(outputs_1d)), axis=-1)
        
        if last_mses:
            fin_int = np.random.randint(1,last_mses,size=batch_size)
            mask = np.zeros((batch_size, input_length, 2))
            mask[np.arange(batch_size), -fin_int, :] = 1

        else:
            mask = np.ones((batch_size, input_length, 2))

        return inputs.reshape((batch_size, input_length, 1)), outputs, mask
    
    return task


def angularintegration_delta_task(T, dt, p=.1, amplitude=1):
    """
    Creates N_batch trials of the angular integration task with dela pulses.
    Inputs are left and right angular velocity and 
    target output is sine and cosine of integrated angular velocity.
    Returns inputs, outputs, mask
    -------
    """
    input_length = int(T/dt)
    
    def task(batch_size):
        
        inputs = amplitude*np.random.choice([-1,0,1], p=[p, 1-2*p, p], size=(batch_size,input_length))
        outputs_1d =  np.cumsum(inputs, axis=1)*dt
        outputs = np.stack((np.cos(outputs_1d), np.sin(outputs_1d)), axis=-1)
        mask = np.ones((batch_size, input_length, 2))

        return inputs.reshape((batch_size, input_length, 1)), outputs, mask
    
    return task

def simplestep_integration_task(T, dt, amplitude=1, pulse_time=1, delay=1):
    """
    Creates a trial with a positive and negative step with length step_length and amplitude
    Inputs are left and right angular velocity and 
    target output is sine and cosine of integrated angular velocity.
    Returns inputs, outputs
    -------
    """
    input_length = int(T/dt)
    pulse_length = int(pulse_time/dt)
    delay_length = int(delay/dt)

    def task(batch_size):
        
        inputs = np.zeros((batch_size,input_length,1))
        inputs[:,delay_length:delay_length+pulse_length,:] = amplitude
        inputs[:,2*delay_length+pulse_length:2*delay_length+2*pulse_length,:] = -amplitude
        outputs_1d =  np.cumsum(inputs, axis=1)*dt
        outputs = np.stack((np.cos(outputs_1d), np.sin(outputs_1d)), axis=-1)
        mask = np.ones((batch_size, input_length, 2))

        return inputs, outputs.reshape((batch_size, input_length, 2)), mask

    return task

def flipflop(dims, dt,
    t_max=50,
    fixation_duration=1,
    stimulus_duration=1,
    decision_delay_duration=5,
    stim_delay_duration_min=5,
    stim_delay_duration_max=25,
    input_amp=1.,
    target_amp=0.5,
    fixate=False,
    choices=None,
    return_ts=False,
    test=False,
    ):
    """ 
    Flipflop task
    """
    dim_in, _, dim_out = dims
    
    if choices is None:
        choices = np.arange(dim_in)
    n_choices = len(choices)

    # Checks
    assert dim_out == dim_in, "Output and input dimensions must agree:    dim_out != dim_in."
    assert np.max(choices) <= (dim_in - 1), "The max choice must agree with input dimension!"

    # Task times
    fixation_duration_discrete = int(fixation_duration / dt)
    stimulus_duration_discrete = int(stimulus_duration / dt)
    decision_delay_duration_discrete = int(decision_delay_duration / dt)
    mean_stim_delay = 0.5 * (stim_delay_duration_min + stim_delay_duration_max)
    n_t_max = int(t_max / dt)
    
    def task(batch_size):
        # Input and target sequences
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)

        for b_idx in range(batch_size):
            input_samp = np.zeros((n_t_max, dim_in))
            target_samp = np.zeros((n_t_max, dim_out))
            mask_samp = np.zeros((n_t_max, dim_out))

            idx_t = fixation_duration_discrete
            if fixate:
                # Mask
                mask_samp[:idx_t] = 1
            
            i_interval = 0
            test_intervals = np.array([16.55, 9.35, 14.80, 11.73, 12.17,  6.50, 13.06, 19.08, 13.19])
            test_choices = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1])
            test_signs = np.array([-1, 1, 1, 1, -1, -1, -1, -1, 1])
            while True:
                # New interval between pulses
                if test and b_idx == 0 and i_interval < len(test_choices):
                    interval = test_intervals[i_interval]
                else:
                    interval = np.random.uniform(stim_delay_duration_min, stim_delay_duration_max)
                # Add the decision delay
                interval += decision_delay_duration
                # New index
                n_t_interval = int(interval / dt)
                idx_tp1 = idx_t + n_t_interval

                # Choose input. 
                if test and b_idx == 0 and i_interval < len(test_choices):
                    choice = test_choices[i_interval]
                    sign = test_signs[i_interval]
                else:
                    choice = np.random.choice(choices)
                    sign = np.random.choice([1, -1])
                
                # Input
                input_samp[idx_t : idx_t + stimulus_duration_discrete, choice] = sign
                # Target
                target_samp[idx_t + decision_delay_duration_discrete : idx_tp1, choice] = sign
                # Mask
                mask_samp[idx_t + decision_delay_duration_discrete : idx_tp1] = 1
                # Update
                idx_t = idx_tp1
                i_interval += 1
                # Break
                if idx_t > n_t_max: break
                    
            # Join
            input_batch[b_idx] = input_samp
            target_batch[b_idx] = target_samp
            mask_batch[b_idx] = mask_samp

        # Scale by input and target amplitude
        input_batch *= input_amp
        target_batch *= target_amp

        return input_batch, target_batch, mask_batch
    
    if return_ts:
        # Array of times
        ts = np.arange(0, t_max, dt)
        return task, ts
    else:
        return task