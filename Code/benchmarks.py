import os, sys
import numpy as np



def benchmark1(N2, N1=None, N=0, side=0, Nbatch=1, Tstim=100, maxT=200, Tprestim=10, cuetime=1, Tpreclick=1, Tpulse=10, Tpoststim=10):
    """
    
    Structure of trials:
    -Tprestim time steps before the stimulus cue 
    -then Tpreclick time steps before the first click pulse starts
    -the second click pulse starts Tstim-N2 time steps after this
    
    Returns a set of trials: 
    -inputs x
    -targets y
    -output mask
    
    This benchmark should reflect the time constant of the memory of the model.
    """
    if not N1:
        N1=N2+1
    x = np.zeros((Nbatch,maxT,4))
    y = np.zeros((Nbatch,maxT,2))
    output_mask = np.zeros((Nbatch,maxT,2))

    x[:,Tprestim:Tprestim+cuetime,2] = 1. #stim cue
    
    #first pulse (left)
    for click_i in range(N1):
        x[:,Tprestim+cuetime+Tpreclick+click_i,side] = 1.
        
    pulse_time = Tprestim+cuetime+Tpreclick+N1+Tpulse
    for click_i in range(N1):
        x[:,pulse_time:pulse_time+N,:2] = 1.
        
    #second pulse (right)
    for click_i in range(N2):
        x[:,Tprestim+cuetime+Tpreclick+Tstim-click_i,(side+1)%2] = 1.    
    
    output_cue_time = Tprestim+cuetime+Tpreclick+Tstim+Tpoststim
    x[:,output_cue_time:output_cue_time+cuetime,3] = 1. #output cue
    
    #target
    y[:,output_cue_time+1,side] = 1. 
    
    output_mask[:,output_cue_time+1,:] = 1. 
    return x, y, output_mask



def benchmark2(N, N1=1, N2=0, side=0, Nbatch=1, Tstim=100, maxT=200, Tprestim=10, cuetime=1, Tpreclick=1, Tpoststim=10, random=False):
    """
    
    Structure of trials:
    -Tprestim time steps before the stimulus cue 
    -N1 click on (left/right) side
    -then Tpreclick time steps before the two click pulses start
    -N2 click on other side (right/left)
    
    This benchmark should reflect the memory limit of the model and the symmetry of how the two inputs get integrated.
    """

    x = np.zeros((Nbatch,maxT,4))
    y = np.zeros((Nbatch,maxT,2))
    output_mask = np.zeros((Nbatch,maxT,2))

    x[:,Tprestim:Tprestim+cuetime,2] = 1. #stim cue
    
    #side
    stim_time1 = Tprestim+cuetime+Tpreclick
    x[:,stim_time1:stim_time1+N1,side] = 1.
    
    simpulse_time = stim_time1+N1
    #pulse (left)
    for click_i in range(N):
        x[:,simpulse_time+click_i,:2] = 1.
    
    #other side
    stim_time2 = simpulse_time+N
    x[:,stim_time2:stim_time2+N2,(side+1)%2] = 1.
    
    #randomize inputs
    if random:
        input_times = np.arange(stim_time1, stim_time2)
        permutated_times = np.arange(stim_time1, stim_time2)
        for batch_i in range(Nbatch):
            np.random.shuffle(permutated_times)
            x[batch_i, np.arange(stim_time1, stim_time2), :] = x[batch_i, permutated_times, :]
    
    #randomize inputs
    if random:
        input_times = np.arange(stim_time1, stim_time1+Tstim)
        for batch_i in range(Nbatch):
            np.random.shuffle(input_times)
            x[batch_i, np.arange(stim_time1, stim_time1+Tstim), :] = x[batch_i, input_times, :]
    
    output_cue_time = Tprestim+cuetime+Tpreclick+Tstim+Tpoststim
    x[:,output_cue_time:output_cue_time+cuetime,3] = 1. #output cue
    
    #target
    target_side = np.where(N1>N2,0,1)
    y[:,output_cue_time+1,target_side] = 1. 
    
    output_mask[:,output_cue_time+1,:] = 1. 
    return x, y, output_mask




def benchmark3(N, side=0, Nbatch=1, Tstim=100, maxT=200, Tprestim=10, cuetime=1, Tpreclick=1, Tpoststim=10, random=False):
    """
    
    Structure of trials:
    -Tprestim time steps before the stimulus cue 
    -1 click on (left/right) side
    -then Tpreclick time steps before the two click pulses start
    
    If random, shuffle both channels independently.
    
    Purpose: symmetry.
    """

    x = np.zeros((Nbatch,maxT,4))
    y = np.zeros((Nbatch,maxT,2))
    output_mask = np.zeros((Nbatch,maxT,2))

    x[:,Tprestim:Tprestim+cuetime,2] = 1. #stim cue
    
    #one click on side
    stim_time = Tprestim+cuetime+Tpreclick
    x[:,stim_time,side] = 1.
    
    #second pulse (right)
    for click_i in range(N):
        x[:,Tprestim+cuetime+Tpreclick+Tstim-click_i,:2] = 1.    
    
    #randomize inputs
    if random:
        input_times = np.arange(stim_time, stim_time+Tstim)
        for batch_i in range(Nbatch):
            np.random.shuffle(input_times)
            x[batch_i, np.arange(stim_time, stim_time+Tstim), 0] = x[batch_i, input_times, 0]
            
            np.random.shuffle(input_times)
            x[batch_i, np.arange(stim_time, stim_time+Tstim), 1] = x[batch_i, input_times, 1]
    
    output_cue_time = Tprestim+cuetime+Tpreclick+Tstim+Tpoststim
    x[:,output_cue_time:output_cue_time+cuetime,3] = 1. #output cue
    
    #target
    y[:,output_cue_time+1,side] = 1. 
    
    output_mask[:,output_cue_time+1,:] = 1. 
    return x, y, output_mask


def benchmark4(N, M=0, side=0, Nbatch=1, Tstim=100, maxT=200, Tprestim=10, cuetime=1, Tpreclick=1, Tpoststim=10, random=False):
    """
    
    Structure of trials:
    -Tprestim time steps before the stimulus cue 
    -N-M clicks on (left/right) side
    -Tpulse time steps later N+1 clicks on other side
    -M click first side
    
    Purpose: 
    """
    
    x = np.zeros((Nbatch,maxT,4))
    y = np.zeros((Nbatch,maxT,2))
    output_mask = np.zeros((Nbatch,maxT,2))

    x[:,Tprestim:Tprestim+cuetime,2] = 1. #stim cue
    
    #side
    stim_time1 = Tprestim+cuetime+Tpreclick
    x[:,stim_time1:stim_time1+N-M,side] = 1.
    
    simpulse_time = stim_time1+N
    #other side
    for click_i in range(N+1):
        x[:,simpulse_time+click_i,(side+1)%2] = 1.
    
    #first side
    stim_time2 = simpulse_time+N
    x[:,Tstim-M:Tstim,side] = 1.  
    
    #randomize inputs
    if random:
        input_times = np.arange(stim_time1, stim_time1+Tstim)
        for batch_i in range(Nbatch):
            np.random.shuffle(input_times)
            x[batch_i, np.arange(stim_time1, stim_time1+Tstim), :] = x[batch_i, input_times, :]
    
    output_cue_time = Tprestim+cuetime+Tpreclick+Tstim+Tpoststim
    x[:,output_cue_time:output_cue_time+cuetime,3] = 1. #output cue
    
    #target
    y[:,output_cue_time+1,side] = 1. 
    
    output_mask[:,output_cue_time+1,:] = 1. 
    return x, y, output_mask