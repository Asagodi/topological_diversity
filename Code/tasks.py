# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:06:58 2023

@author: 
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
def angularintegration_task(T, dt, length_scale=1, sparsity=1, last_mses=False, random_angle_init=False, max_input=None):
    """
    Creates N_batch trials of the angular integration task with Guassian Process angular velocity inputs.
    Inputs is angular velocity (postive: right, negative:left) and 
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
        if max_input:
            inputs = np.where(np.abs(inputs)>max_input, np.sign(inputs), inputs)
        if sparsity:
            inputs[mask_input] = 0.
        outputs_1d = np.cumsum(inputs, axis=1)*dt
        if random_angle_init=='equally_spaced':
            outputs_1d += np.arange(-np.pi, np.pi, 2*np.pi/batch_size)[:, np.newaxis]
        elif random_angle_init:
            random_angles = np.random.uniform(-np.pi, np.pi, size=batch_size)
            outputs_1d += random_angles[:, np.newaxis]

        outputs = np.stack((np.cos(outputs_1d), np.sin(outputs_1d)), axis=-1)

        if last_mses:
            fin_int = np.random.randint(1,last_mses,size=batch_size)
            mask = np.zeros((batch_size, input_length, 2))
            mask[np.arange(batch_size), -fin_int, :] = 1

        else:
            mask = np.ones((batch_size, input_length, 2))

        return inputs.reshape((batch_size, input_length, 1)), outputs, mask
    
    return task

def double_angularintegration_task(T, dt, length_scale=1, sparsity=1, last_mses=False, random_angle_init=False, max_input=None):
    input_length = int(T/dt)
    
    def task(batch_size):
        
        X = np.expand_dims(np.linspace(-length_scale, length_scale, input_length), 1)
        sigma = exponentiated_quadratic(X, X)  
        if sparsity =='variable':
            sparsities = np.random.uniform(0, 2, batch_size)
            mask_input = np.random.random(size=(batch_size, input_length))<1-sparsities[:,None]
        elif sparsity:
            mask_input = np.random.random(size=(batch_size, input_length)) < 1-sparsity
        inputs1 = np.random.multivariate_normal(mean=np.zeros(input_length), cov=sigma, size=batch_size)
        inputs2 = np.random.multivariate_normal(mean=np.zeros(input_length), cov=sigma, size=batch_size)
        if max_input:
            inputs1 = np.where(np.abs(inputs1)>max_input, np.sign(inputs1), inputs1)
            inputs2 = np.where(np.abs(inputs2)>max_input, np.sign(inputs2), inputs2)
        if sparsity:
            inputs1[mask_input] = 0.
            inputs2[mask_input] = 0.
        outputs1_1d = np.cumsum(inputs1, axis=1)*dt
        outputs2_1d = np.cumsum(inputs2, axis=1)*dt
        if random_angle_init=='equally_spaced':
            outputs1_1d += np.arange(-np.pi, np.pi, 2*np.pi/batch_size)[:, np.newaxis]
            outputs2_1d += np.arange(-np.pi, np.pi, 2*np.pi/batch_size)[:, np.newaxis]
        elif random_angle_init:
            random_angles1 = np.random.uniform(-np.pi, np.pi, size=batch_size)
            outputs1_1d += random_angles1[:, np.newaxis]
            
            random_angles2 = np.random.uniform(-np.pi, np.pi, size=batch_size)
            outputs2_1d += random_angles2[:, np.newaxis]

        outputs1 = np.stack((np.cos(outputs1_1d), np.sin(outputs1_1d)), axis=-1)
        outputs2 = np.stack((np.cos(outputs2_1d), np.sin(outputs2_1d)), axis=-1)
        outputs = np.concatenate((outputs1, outputs2), axis=-1)
        
        if last_mses:
            fin_int = np.random.randint(1,last_mses,size=batch_size)
            mask = np.zeros((batch_size, input_length, 4))
            mask[np.arange(batch_size), -fin_int, :] = 1

        else:
            mask = np.ones((batch_size, input_length, 4))

        inputs = np.stack((inputs1, inputs2), axis=-1)
        return inputs.reshape((batch_size, input_length, 2)), outputs, mask
    
    return task

def angularintegration_task_constant(T, dt, speed_range=[-1,1], sparsity=1, last_mses=False, random_angle_init=False, max_input=None):
    """
    Creates N_batch trials of the angular integration task with Guassian Process angular velocity inputs.
    Inputs is angular velocity (postive: right, negative:left) and 
    target output is sine and cosine of integrated angular velocity.
    Returns inputs, outputs, mask
    -------
    """
    input_length = int(T/dt)
    
    def task(batch_size):
        
        if sparsity =='variable':
            sparsities = np.random.uniform(0, 2, batch_size)
            mask_input = np.random.random(size=(batch_size, input_length))<1-sparsities[:,None]
        elif sparsity:
            mask_input = np.random.random(size=(batch_size, input_length)) < 1-sparsity
        inputs_0 = np.random.uniform(low=speed_range[0], high=speed_range[1], size=batch_size)
        inputs = inputs_0.reshape(-1, 1) * np.ones((batch_size, input_length))
        if max_input:
            inputs = np.where(np.abs(inputs)>max_input, np.sign(inputs), inputs)
        if sparsity:
            inputs[mask_input] = 0.
        outputs_1d = np.cumsum(inputs, axis=1)*dt
        if random_angle_init=='equally_spaced':
            outputs_1d += np.arange(-np.pi, np.pi, 2*np.pi/batch_size)[:, np.newaxis]
        elif random_angle_init:
            random_angles = np.random.uniform(-np.pi, np.pi, size=batch_size)
            outputs_1d += random_angles[:, np.newaxis]

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
    Inputs is angular velocity (postive: right, negative:left) and 
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

def bernouilli_integration_task(T, input_length, final_loss):
    """
    Creates a trial with a positive and negative step with length step_length and amplitude
    Inputs are left and right angular velocity and 
    target output is sine and cosine of integrated angular velocity.
    Returns inputs, outputs
    -------
    """

    def task(batch_size):
        
        inputs = np.zeros((batch_size,T,2))
        inputs[:,:input_length,:] = np.random.binomial(1, p=.2, size=(batch_size,input_length,2))
        outputs = np.cumsum(inputs[:,:,1], axis=1) - np.cumsum(inputs[:,:,0], axis=1)
        mask = np.zeros((batch_size, T, 1))
        if final_loss:
            mask[:,-1,:] = 1
        else:
            mask[:,:,:] = 1


        return inputs, outputs.reshape((batch_size,T,1)), mask

    return task

def bernouilli_noisy_integration_task(T, input_length, sigma, final_loss):
    """
    Creates a trial with a positive and negative step with length step_length and amplitude
    Inputs are left and right angular velocity and 
    target output is sine and cosine of integrated angular velocity.
    Returns inputs, outputs
    -------
    """

    def task(batch_size):
        
        inputs = np.zeros((batch_size,T,2))
        inputs[:,:input_length,:] = np.random.binomial(1, p=.2, size=(batch_size,input_length,2))
        outputs = np.cumsum(inputs[:,:,1], axis=1) - np.cumsum(inputs[:,:,0], axis=1)
        inputs[:,:input_length,:] += np.random.uniform(-sigma, sigma, size=(batch_size,input_length,2))

        mask = np.zeros((batch_size, T, 1))
        if final_loss:
            mask[:,-1,:] = 1
        else:
            mask[:,:,:] = 1

        return inputs, outputs.reshape((batch_size,T,1)), mask

    return task

def contbernouilli_noisy_integration_task(T, input_length, sigma, final_loss):
    """
    Creates a trial with a positive and negative step with length step_length and amplitude
    Inputs are left and right angular velocity and 
    target output is sine and cosine of integrated angular velocity.
    Returns inputs, outputs
    -------
    """

    def task(batch_size):
        
        inputs = np.zeros((batch_size,T,2))
        inputs[:,:input_length,:] = np.random.binomial(1, p=.2, size=(batch_size,input_length,2))
        inputs[:,:input_length,:] *= np.random.uniform(0., 1., size=(batch_size,input_length,2))
        outputs = np.cumsum(inputs[:,:,1], axis=1) - np.cumsum(inputs[:,:,0], axis=1)
        inputs[:,:input_length,:] += np.random.uniform(-sigma, sigma, size=(batch_size,input_length,2))

        mask = np.zeros((batch_size, T, 1))
        if final_loss:
            mask[:,-1,:] = 1
        else:
            mask[:,:,:] = 1

        return inputs, outputs.reshape((batch_size,T,1)), mask

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
    
def poisson_clicks_task(T, dt, set_stim_duration=None,
                        cue_output_durations = [10,5,10,5,1], 
                        ratios=[-39,-37/3,-31/9,-26/14,26/14,31/9,37/3,39],  sum_of_rates=40,
                        exp_trunc_params={'b':1,'scale':100,'loc':50}, 
                        clicks_capped=True, equal_clicks='addone_random'
                        ):
    
    
    input_length = int(T*dt)
    
    stim_cue_delay, stim_cue_duration, output_cue_delay, output_cue_duration, output_duration = cue_output_durations
    delay_to_stim = stim_cue_delay + stim_cue_duration + 1
    
    def task(batch_size):

        input =  np.zeros([batch_size, input_length, 4])
        stimulus =  np.zeros([batch_size, input_length, 2])
        target = np.zeros([batch_size, input_length, 2])
        mask = np.ones([batch_size, input_length, 2])
        
        stim_durations = []
        if not set_stim_duration:
            stim_durations = truncexpon.rvs(b=exp_trunc_params['b'],
                                                 scale=exp_trunc_params['scale'],
                                                 loc=exp_trunc_params['loc'],
                                                 size=batch_size)*dt

            stim_durations = [int(x) for x in stim_durations]
            
            #cap to T
            stim_durations = [x if (delay_to_stim+x+output_cue_delay+output_cue_duration+output_duration<input_length) else input_length-(output_cue_delay+output_cue_duration+output_duration+2) for x in stim_durations ]

        else: 
            stim_durations = [set_stim_duration]*batch_size
        
        for batch_i in range(batch_size): 
            stim_duration = stim_durations[batch_i]

            # delay_to_cue = max(1,int(np.random.random() * self.T *.1 *self.dt)) #duration is tenth of the trial
            
            ratio = np.random.choice(ratios)
            rates = [sum_of_rates/(1+1/abs(ratio))]
            rates.insert(int(np.sign(ratio)), sum_of_rates - rates[0])
            rates = np.array(rates)
            stimulus[batch_i, delay_to_stim:delay_to_stim+stim_duration, :] = np.random.poisson(lam=rates/dt/1000, size=(stim_duration,2))

            if clicks_capped == True:
                stimulus[batch_i,...] = np.where(stimulus[batch_i,...]<2, stimulus[batch_i,...], 1)
    
            N_clicks = np.sum(stimulus[batch_i,...], axis=0)
            #determine N_1<N_2
            if N_clicks[0] == N_clicks[1]:
                highest_click_count_index = np.random.choice([0,1]) #if N_1=N_2 choose reward randomly
                if equal_clicks == 'addone_random':
                    elapsed_time = delay_to_stim+stim_duration+1 
                    stimulus[batch_i, elapsed_time, highest_click_count_index] = 1.
                    N_clicks[highest_click_count_index] += 1
                    
            else:
                highest_click_count_index = np.argmax(N_clicks)
    

            output_cue = stim_cue_delay + stim_cue_duration + stim_duration + output_cue_delay
            input[batch_i, output_cue-output_cue_duration:output_cue, 3] = 1.
            target[batch_i, output_cue:output_cue+output_duration, highest_click_count_index] = 1.
            target[batch_i, output_cue:, highest_click_count_index] = 1.

            output_end = output_cue+output_duration+1
            # mask[batch_i, output_end:, :] = 0.
        input[:, :, :2] = stimulus
        input[:, stim_cue_delay:stim_cue_delay+stim_cue_duration, 2] = 1.
        
        return input, target, mask
    
    return task





def center_out_reaching_task(T, dt, 
                             cue_output_durations = [5,5,75,5,5],
                             time_until_cue_range=[50, 75], angles_random=True):
    # cue_output_durations = [time_until_input, stim_duration, time_until_cue, cue_duration, time_until_measured_response]
    
    input_length = int(T*dt)
    time_until_input, stim_duration, time_until_cue, cue_duration, time_until_measured_response = cue_output_durations
    
    def task(batch_size):
        input = np.zeros((batch_size, input_length, 3))
        target = np.zeros((batch_size, input_length, 2))
        mask = np.ones((batch_size, input_length, 2))

        if angles_random:
            angles = np.pi * np.random.uniform(0, 2, (batch_size))
        else:
            angles = np.arange(-np.pi, np.pi, 2*np.pi/batch_size)

        x = np.cos(angles)
        y = np.sin(angles)
        
        for trial_i in range(batch_size):
            if time_until_cue_range == None:
                time_until_cue_i = time_until_cue
            else:
                time_until_cue_i = np.random.randint(time_until_cue_range[0], time_until_cue_range[1])

            input[trial_i, time_until_input:time_until_input+stim_duration,0] = x[trial_i]
            input[trial_i, time_until_input:time_until_input+stim_duration,1] = y[trial_i]
            total_time_until_cue = time_until_input+stim_duration+time_until_cue_i
            input[trial_i, total_time_until_cue:total_time_until_cue+cue_duration,2] = 1
                
            total_time_until_measured_response = total_time_until_cue + cue_duration + time_until_measured_response
            target[trial_i, total_time_until_measured_response:,0] = x[trial_i]
            target[trial_i, total_time_until_measured_response:,1] = y[trial_i]
            
            mask[trial_i, total_time_until_cue+cue_duration:total_time_until_measured_response,:] = 0.
            
        
        return input, target, mask

    return task