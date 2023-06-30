import os, sys
import glob
current_dir = os.path.dirname(os.path.realpath('__file__'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.utils.parametrize as P
# import torch.nn.init as weight_init

import scipy
from scipy.optimize import minimize
from functools import partial
import numpy as np
import numpy.ma as ma
from math import isclose
import pickle
import re
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.markers as markers

from tasks import PerceptualDiscrimination, PoissonClicks, DynamicPoissonClicks, create_copy_memory_trials_onehot, create_eyeblink_trials, create_flipflop_trials, create_angularintegration_trials
from network_initialization import *
from analysis_functions import *
from rnn_models import *
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        

class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []
    
    
def get_task(task_name):
    if task_name == 'perceptual_discrimination':
        task = PerceptualDiscrimination
    elif task_name == 'poisson_clicks':
        task = PoissonClicks
    elif task_name == 'dynamic_poisson_clicks':
        task = DynamicPoissonClicks
    elif task_name == 'motion_pulse':
        task = MotionPulse

    else:
        print("Task not defined")
        sys.exit()

    return task


#datasets
def make_train_test_sets(training_kwargs):
    """Create dataset for desired task.
    
    Tasks: 
    -Copy memory
    -Perceptual discrimination
    -Poisson clicks
    -Motion pulse
    
    Returns:
    -train set
    -test set
    -train and test generator functions
    """
    print(training_kwargs['task'])
    train_size = int(training_kwargs['total_data_size']*training_kwargs['train_proportion'])
    test_size = training_kwargs['total_data_size']-train_size
    if training_kwargs['task'] == 'copy_memory':
        N_symbols = training_kwargs['N_symbols']
        input_length = training_kwargs['input_length']
        delay = training_kwargs['delay']
        train_set = create_copy_memory_trials_onehot(train_size, N_symbols, input_length, delay)
        test_set = create_copy_memory_trials_onehot(test_size, N_symbols, input_length, delay)
        return train_set, test_set, 0, 0
    elif training_kwargs['task'] == 'eyeblink':
        train_set = create_eyeblink_trials(train_size, training_kwargs['input_length'], t_stim=training_kwargs['fixed_stim_duration'], t_delay=training_kwargs['delay'], t_target=training_kwargs['target_duration'])
        test_set = create_eyeblink_trials(test_size, training_kwargs['input_length'], t_stim=training_kwargs['fixed_stim_duration'], t_delay=training_kwargs['delay'], t_target=training_kwargs['target_duration'])
        return train_set, test_set, 0, 0
    elif training_kwargs['task'] == 'flipflop':
        train_set = create_flipflop_trials(train_size, training_kwargs['input_length'], t_stim=training_kwargs['fixed_stim_duration'], t_delay=training_kwargs['delay'], input_amp=1., target_amp=0.5)
        test_set = create_flipflop_trials(test_size, training_kwargs['input_length'], t_stim=training_kwargs['fixed_stim_duration'], t_delay=training_kwargs['delay'], input_amp=1., target_amp=0.5)
        return train_set, test_set, 0, 0
    elif training_kwargs['task'] == 'angularintegration':
        T = training_kwargs['input_length']
        dt = 1
        train_set = create_angularintegration_trials(train_size, T, dt)
        test_set = create_angularintegration_trials(test_size, T, dt)
        print(train_set[0].shape)
        return train_set, test_set, 0, 0
    else:
        task = get_task(task_name=training_kwargs['task'])
    task_train = task(N_batch=train_size, training_kwargs=training_kwargs)
    train_set = task_train.get_trial_batch() 
    task_test = task(N_batch=test_size, training_kwargs=training_kwargs)
    test_set = task_test.get_trial_batch() 
    return train_set, test_set, task_train, task_test



def make_curriculum_datasets(training_kwargs):
    """Returns curriculum and base task parameters"""
    task = get_task(task_name=training_kwargs['task'])
        
       
    curriculum_dataset = {} #key: coherence, value: [train_x, train_y, train_output_mask, train_trial_params]
    training_subkwargs = training_kwargs
    training_subkwargs['coherence'] = 1
    task_base =  task(N_batch=training_subkwargs['data_size_per_coherence'], training_kwargs=training_kwargs)
    train_size = int(training_subkwargs['data_size_per_coherence']*training_subkwargs['train_proportion'])
    for coherence in training_kwargs['coherence_list']:
        training_subkwargs['ratios'] = [-coherence, coherence]
        task_train = task(N_batch=train_size, training_kwargs=training_subkwargs)
        train_set = task_train.get_trial_batch() 
        task_test = task(N_batch=training_subkwargs['data_size_per_coherence']-train_size, training_kwargs=training_subkwargs)
        test_set = task_test.get_trial_batch() 
        curriculum_dataset[str(coherence)] = [train_set, test_set]
        
    return curriculum_dataset, task_base.get_task_params()



#training
def get_optimizer(model, training_kwargs):
    #initialize optimizer
    if training_kwargs['optimizer'] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=training_kwargs['learning_rate'], momentum=training_kwargs['sgd_momentum'], weight_decay=training_kwargs['weight_decay'])
    elif training_kwargs['optimizer'] == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=training_kwargs['learning_rate'], weight_decay=training_kwargs['weight_decay'])
    elif training_kwargs['optimizer'] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=training_kwargs['learning_rate'], weight_decay=training_kwargs['weight_decay'])
    return optimizer

def get_loss_function(model, training_kwargs):
    #initialize loss function
    if training_kwargs['loss_function'] == "mse":
        loss_fn = nn.MSELoss().to(training_kwargs['device'])
    elif training_kwargs['loss_function'] == "ce":
        loss_fn = nn.CrossEntropyLoss().to(training_kwargs['device'])
    elif training_kwargs['loss_function'] == "bce":
        loss_fn = nn.BCELoss().to(training_kwargs['device'])
    elif training_kwargs['loss_function'] == "bcewll":
        loss_fn = nn.BCEWithLogitsLoss().to(training_kwargs['device'])
    return loss_fn

def get_scheduler(model, optimizer, training_kwargs):
    #initialize learning rate scheduler
    if training_kwargs['scheduler'] == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training_kwargs['scheduler_step_size'], gamma=training_kwargs['steplr_gamma'])
    elif training_kwargs['scheduler'] == "cosineannealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_kwargs['max_epochs'])
    elif training_kwargs['scheduler'] == None:
        scheduler = None
    return scheduler

#custom scheduler
def adjust_learning_rate_poly(optimizer, initial_lr, iteration, max_iter):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = initial_lr * ( 1 - (iteration / max_iter)) * ( 1 - (iteration / max_iter))
    if ( lr < 1.0e-7 ):
        lr = 1.0e-7

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

#training functions
def recover_the_dead(model, x, training_kwargs):
    """Determines dead neurons and then resets (randomizes) weights and biases of dead neurons
    Returns reset model"""
    with torch.no_grad():
        hidden = torch.zeros(model.n_layers, x.shape[0], model.hidden_dim).to(training_kwargs['device'])
        hidden_states, hidden_last = model.rnn(x, hidden)
        hidden_states = hidden_states.detach().numpy()
        dead_neurons = np.where(np.sum(hidden_states, axis=(0,1))==0)[0]
        for neuron_idx in dead_neurons:
            # print(hidden_states.shape, dead_neurons, neuron_idx, model.rnn.all_weights[0][1].shape)
            #input to recurrent
            model.rnn.all_weights[0][0][neuron_idx, :] = torch.normal(0, training_kwargs['weight_reinitialization_variance'], size=(1,training_kwargs['N_in'])) #0.
            #recurrent layer
            model.rnn.all_weights[0][1][neuron_idx, :] = torch.normal(0, training_kwargs['weight_reinitialization_variance'], size=(1,training_kwargs['N_rec'])) #outgoing
            model.rnn.all_weights[0][1][:, neuron_idx] = torch.normal(0, training_kwargs['weight_reinitialization_variance'], size=(1,training_kwargs['N_rec'])) #incoming
            #biases of hidden neuron
            model.rnn.all_weights[0][2][neuron_idx] = torch.normal(0, training_kwargs['weight_reinitialization_variance'], size=(1,1))
            model.rnn.all_weights[0][3][neuron_idx] = torch.normal(0, training_kwargs['weight_reinitialization_variance'], size=(1,1))
            #fully connected layers
            model.fc.weight[:,neuron_idx] = torch.normal(0, training_kwargs['weight_reinitialization_variance'], size=(1,training_kwargs['N_out']))
    return model, len(dead_neurons)


def save_model(model, training_kwargs, epoch, coherence=None):
    """Saves model state dictionary"""
    if epoch=='start':
        torch.save(model.state_dict(), training_kwargs['exp_path'] + '/weights_-1.pth')
    elif epoch=='final':
        torch.save(model.state_dict(), training_kwargs['exp_path'] + '/weights%s.pth'%training_kwargs['idx'])
    else:
        if training_kwargs['single_or_curriculum']=='single':
            torch.save(model.state_dict(), training_kwargs['training_weights_path'] + '/weights_epoch%s_%s.pth'%(epoch, training_kwargs['idx']))
        else:
            torch.save(model.state_dict(), training_kwargs['training_weights_path'] + '/weights_coh%s_epoch%s_%s.pth'%(coherence, epoch, training_kwargs['idx']))


def save_info(learning_info, epoch_losses, epoch_val_losses, accuracies, grads, training_kwargs, epoch):
    """Saves epoch losses for training and test set, accuracies, gradients and learning information"""
   
    learning_info['training_loss'] = epoch_losses
    learning_info['validation_loss'] = epoch_val_losses
    learning_info['accuracies'] = accuracies
    learning_info['grads'] = grads

    with open(training_kwargs['exp_path'] + '/learning_info' + str(training_kwargs['idx']) + '.pickle', 'wb') as handle:
        pickle.dump(learning_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return learning_info

def test_model(model, loss_fn, test_x, test_y, test_output_mask, training_kwargs):
    loss = 0
    if training_kwargs['model_type'] == 'rnn' or training_kwargs['model_type'] == 'rnn_cued':
        _, hidden_states = run_model_chunked(test_x, model, hidden_offset=training_kwargs['hidden_offset'],
                                             hidden_init = training_kwargs['hidden_initial_activations'], 
                                             hidden_initial_variance=training_kwargs['hidden_initial_variance'], n_chunks=10)
        yhat = model.fc(hidden_states)
    elif training_kwargs['model_type'] == 'rnn_interactivation':
        x = torch.tensor(test_x,  dtype=torch.float)
        yhat = torch.zeros(test_y.shape)
        hidden = model.init_hidden(test_x.shape[0])
        for i in range(x.shape[1]):
            yhat[:, i, :], hidden = model(x[:,i,:], hidden)
    elif training_kwargs['model_type'] in ['lstm', 'gru']:
        x = torch.tensor(test_x,  dtype=torch.float)
        yhat = model(x)
        
    if training_kwargs['count_out'] == "1D_uncued" or training_kwargs['task'] == "eyeblink":
        for t_id in range(test_x.shape[1]):
            loss += loss_fn.forward(yhat[:,t_id,0], test_y[:,t_id,0])
        loss/= test_x.shape[1]
    else:
        # print(yhat.shape, test_y.shape)
        loss += loss_fn.forward(yhat.view(-1,training_kwargs['N_out']), test_y.view(-1,training_kwargs['N_out']))
        # for t_id in range(test_x.shape[1]):
        #     # loss += loss_fn.forward(yhat[:,t_id,:], torch.tensor(test_y, dtype=torch.float, device=training_kwargs['device'])[:,t_id,:])
        #     loss += loss_fn.forward(yhat[:,t_id,0], test_y[:,t_id,0])
        #     loss += loss_fn.forward(yhat[:,t_id,1], test_y[:,t_id,1])
        # loss /= test_x.shape[1] / test_x.shape[0]
    loss = loss.cpu().detach().numpy()
    
    yhat_np = yhat.cpu().detach().numpy()
    if training_kwargs["task"] == "perceptual_discrimination":
        accuracy = pd_accuracy_function(test_y, yhat_np, test_output_mask)
    elif training_kwargs["task"] == "poisson_clicks":
        accuracy, response_correctness, N_clicks, highest_click_count_index, chosen, excludeequals = get_accuracy_poissonclicks_w(test_x, yhat_np, test_output_mask)
    elif training_kwargs["task"] in ["copy_memory", "eyeblink", "angularintegration"]:
        accuracy = 0
    return loss, accuracy
    


def train_model(model, data_lists, training_kwargs, coherence=None, ):
    """ 
    Trains an RNN model on a single dataset.
    Note: Some additions only work for a one layer RNN (e.g. resetting dead units).
    
    Args:
        -model: pytorch model
        -data_lists is a list of two lists
            -[train_x, train_y, train_output_mask, train_trial_params] for training set
            -[test_x, test_y, test_output_mask, test_trial_params] for test set
        train_x.shape = (batch_size, T, input_dim)
        -training_kwargs: has all the necessary parameters
        -coherence: can be declared if called by curriculum training
    
    Returns model and learning information.
    """
    #make folder for saving and figures
    makedirs(training_kwargs['exp_path']) 
    makedirs(training_kwargs['training_weights_path']) 
    
    #initialize learning functions
    optimizer = get_optimizer(model, training_kwargs)
    loss_fn = get_loss_function(model, training_kwargs)
    scheduler = get_scheduler(model, optimizer, training_kwargs)
    current_lr = training_kwargs['learning_rate']
    
    train_x, train_y, train_output_mask, _ = data_lists[0]
    test_x, test_y, test_output_mask, _ = data_lists[1]
    dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float, device=training_kwargs['device']),
                            torch.tensor(train_y, dtype=torch.float, device=training_kwargs['device']),
                            torch.tensor(train_output_mask, dtype=torch.float))
    data_loader = DataLoader(dataset, batch_size=training_kwargs['dataloader_batch_size'], shuffle=True)
        
    #initialize lists and dictionaries to store learning progress

    with torch.no_grad():
        yhat, _ = model(torch.Tensor(train_x))
        print(yhat.shape, train_y.shape)
        epoch_losses = [loss_fn(yhat[...,0], torch.Tensor(train_y[...,0]))]
    epoch_val_losses = []
    accuracies = []
    grads = []
    learning_info = {} #stores all information about the learning process: errors (test/validation set), etc.
    learning_info['number_of_dead'] = []
    #initialize stopping criteria
    min_loss = np.inf #set minimum loss per curriculum to track early stopping
    trigger_times = 0
    start = time.time()

    # if training_kwargs['constrain_spectrum']:
    #     torch.nn.utils.parametrizations.spectral_norm(model.rnn, name='weight_hh_l0') 
            
    save_model(model, training_kwargs, epoch='start', coherence=coherence)
    learning_info = save_info(learning_info, [], [], [], [], training_kwargs, 'start')
    
    if training_kwargs['stationary_data'] == False: 
        task = get_task(task_name=training_kwargs['task'])
        task_train = task(N_batch=training_kwargs['dataloader_batch_size'], training_kwargs=training_kwargs)
    for epoch in range(0, training_kwargs['max_epochs']):
        batch_losses = []
        
        if training_kwargs['parameter_noise_variance'] and epoch % training_kwargs['perturbation_per_epoch'] == 1:
            model = add_parameter_noise(model, training_kwargs['parameter_noise_variance'])
            save_model(model, training_kwargs, epoch=-epoch, coherence=coherence)

        for data_idx, (x, y, output_mask) in enumerate(data_loader):
            #generate random data
            if training_kwargs['stationary_data'] == False:
                train_set = task_train.get_trial_batch() 
                x, y, output_mask, _ = train_set
                
                
            optimizer.zero_grad()            # zero the parameter gradients
            if training_kwargs['model_type'] == 'rnn' or training_kwargs['model_type'] == 'rnn_cued':
                
                #random hidden init for discontinual learning, hidden activations of previous batch for continual learning
                if training_kwargs['continual_learning'] and data_idx != 0:
                    hidden = hidden_last.detach()
                elif training_kwargs['task'] == 'copy_memory' or  training_kwargs['task'] == 'poisson_clicks':
                    yhat, hidden = model(x)

                else:
                    hidden = model.hidden_offset+torch.normal(0, training_kwargs['hidden_initial_variance'], (model.n_layers, x.shape[0], model.hidden_dim)).to(training_kwargs['device']) 
                    hidden_states, hidden_last = model.rnn(x, hidden)
                    yhat = model.fc(hidden_states)
                # yhat =  model.out_act(yhat) #this should be only used for loss functions that use outputs which are already put through the sigmoid
                
            elif training_kwargs['model_type'] == 'rnn_interactivation':
                yhat = torch.zeros(y.shape)
                hidden = model.init_hidden(training_kwargs['dataloader_batch_size']).to(training_kwargs['device'])
                for i in range(x.shape[1]):
                    yhat[:, i, :], hidden = model(x[:,i,:], hidden)
            elif training_kwargs['model_type'] in ['lstm', 'gru']:
                yhat = model(x)
            #calculate loss
            loss = 0
            if training_kwargs['count_out'] == "1D_uncued":
                
                for t_id in range(x.shape[1]):
                    loss += loss_fn.forward(yhat[:,t_id,0], y[:,t_id])
                loss/=x.shape[1]
            elif training_kwargs['masked_training']:
                average_unmasked_time = torch.mean(torch.sum(output_mask[...,0], axis=1))
                for t_id in range(x.shape[1]):
                    loss += loss_fn.forward(yhat[:,t_id,:]*output_mask[:,t_id,:], y[:,t_id]*output_mask[:,t_id])
                loss /=  average_unmasked_time #average over time
            else:
                loss += loss_fn.forward(yhat, y)
                # for t_id in range(x.shape[1]):
                #     loss += loss_fn.forward(yhat[:,t_id,0], y[:,t_id,0])
                #     loss += loss_fn.forward(yhat[:,t_id,1], y[:,t_id,1])
                # loss /= x.shape[1]  #average over time

            #regularize:
                #--weights
            if training_kwargs['norm'] == 'l1':
                norm = sum(torch.linalg.norm(p, 1) for p in model.parameters())
            elif training_kwargs['norm'] == 'l2':
                norm = sum(torch.linalg.norm(p, 2) for p in model.parameters())
            else:
                norm = 0.
                
            #--activity
            act_norm = 0.
            if training_kwargs['act_norm_lambda'] != 0.: 
                # act_norm += torch.linalg.vector_norm(hidden, 1) #only last state
                for t_id in range(hidden_states.shape[1]):  #all states through time
                    act_norm += torch.mean(torch.linalg.vector_norm(hidden_states[:,t_id,:], dim=1))
                act_norm /= hidden_states.shape[1]

            #add regularization to loss
            loss = loss + training_kwargs['norm_lambda'] * norm + training_kwargs['act_norm_lambda']*act_norm
            loss.backward()   # backward propagation

            if training_kwargs['clip_grads']:
                clipping_value = training_kwargs['clip_grads_value']
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

            optimizer.step()  #change parameters through gradient step
            
            #constrain parameters
            if training_kwargs['constrain_diagonal']:
                with torch.no_grad():
                    new_diag = torch.where(model.rnn.all_weights[0][1].diagonal()>1., 1., model.rnn.all_weights[0][1].diagonal())
                    model.rnn.all_weights[0][1].diagonal().copy_(new_diag)


            batch_loss = loss.cpu().detach().numpy() 
            batch_losses.append(batch_loss)

        if scheduler: #update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        #check for zero/dead neurons
        if training_kwargs['check_the_dead'] and epoch % training_kwargs['check_the_dead_per_epoch'] == 0 and training_kwargs['check_the_dead_per_epoch']>0:
            model, number_of_dead = recover_the_dead(model, x, training_kwargs)
            learning_info['number_of_dead'].append(number_of_dead)

            
        epoch_train_loss, train_acc = test_model(model, loss_fn, train_x, torch.tensor(train_y, dtype=torch.float, device=training_kwargs['device']), train_output_mask, training_kwargs) #test model with training data
        epoch_val_loss, val_acc = test_model(model, loss_fn, test_x, torch.tensor(test_y, dtype=torch.float, device=training_kwargs['device']), test_output_mask, training_kwargs) #test model with valudation data

        #store losses and accuracy
        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)
        epoch_val_losses.append(epoch_val_loss)
        accuracies.append([train_acc, val_acc])
        grads.append(torch.norm(model.hidden_offset.grad).cpu().detach().numpy())

        #print progress if verbosity
        learning_info = save_info(learning_info, epoch_losses, epoch_val_losses, accuracies, grads, training_kwargs, epoch)
        if training_kwargs['verbosity'] and epoch % training_kwargs['print_epoch']==0:
            print(f'Epoch{epoch:4d} loss: {np.log10(epoch_loss) :.4f}, validation loss: {np.log10(epoch_val_loss) :.4f}, min val loss: {np.log10(min_loss) :.4f}, accuracy: {train_acc :.6f}, trigger time: {trigger_times :2d}, LR {np.log10(current_lr) :.2f}, |Grad|: {torch.norm(model.hidden_offset.grad)}')

        if epoch % training_kwargs['save_training_weights_epoch']==0:        #save weights during training at each epoch
            save_model(model, training_kwargs, epoch=epoch, coherence=coherence)

        
        #check loss covergence
        if epoch_val_loss > min_loss or isclose(epoch_val_loss, min_loss, abs_tol=training_kwargs['difference_tolerance']):
            trigger_times += 1
        else:
            trigger_times = 0
            
        # update minimum loss
        min_loss = min([epoch_val_loss, min_loss])
            
        #similar for accuracy_trigger_times with training_kwargs['accuracy_stopping_criterion_epochs']
        if trigger_times >= training_kwargs['early_stopping_criterion_epochs'] and training_kwargs['early_stopping']:
            trigger_times = 0
            learning_info['early_stopping'] = "Early stopping"
            # if training_kwargs['verbosity']:
                # print('Early stopping!')
            break #attempt next curriculum


        #check to switch to more difficult task
        #checks: accuracy and loss #TODO: implement loss check
        if train_acc > training_kwargs['accuracy_goal'] and training_kwargs['accuracy_goal'] != None and epoch>training_kwargs['min_epochs_per_curriculum'] and trigger_times>training_kwargs['curriculum_switching_criterion_epochs'] or train_acc>=training_kwargs['high_accuracy_stop']:
            trigger_times = 0
            break
        
    #save model and losses
    end = time.time()
    learning_info['early_stopping'] = "Early stopping"
    learning_info['training_duration'] = end-start
    save_model(model, training_kwargs, epoch='final', coherence=coherence)
    save_info(learning_info, epoch_losses, epoch_val_losses, accuracies, grads, training_kwargs, epoch)

    if training_kwargs['verbosity']:
        print("Duration of training: ", end-start)
    print("Final loss: ", str(np.log10(epoch_loss)), "Validation: ", np.log10(epoch_val_loss))
    return model, learning_info



def curriculum_learning(model, curriculum_dataset, training_kwargs):
    """
    Trains an RNN model on a dataset with curriculum learning.
    curriculum_dataset.keys() should be set with total ordering
    Return model and information about curriculum learning process (nested dictionary)
    """
    #make folder for saving and figures
    makedirs(training_kwargs['exp_path']) 
    makedirs(training_kwargs['training_weights_path']) 
    
    #initialize learning functions
    optimizer = get_optimizer(model, training_kwargs)
    loss_fn = get_loss_function(model, training_kwargs)
    scheduler = get_scheduler(model, optimizer, training_kwargs)
    
    curriculum = list(curriculum_dataset.keys())
    float_keys = [float(key) for key in curriculum]
    curr_order = np.argsort(float_keys)[::-1]
    print(np.array(curriculum)[curr_order])
    
    #initializer lists and dictionaries to store learning progress
    epoch_losses = []
    epoch_val_losses = []
    accuracies = []
    learning_info = {} #stores all information about the learning process: errors (test/validation set), stopping for curriculum learning, etc.
    trigger_times = 0
    for coherence in np.array(curriculum)[curr_order]:
        if training_kwargs['verbosity']:
            print("Starting curriculum with coherence %s"%coherence)
        min_loss = np.inf #set minimum loss per curriculum to track early stopping
        data_lists = curriculum_dataset[str(coherence)]
        model, learning_info_coh = train_model(model, data_lists, training_kwargs, coherence=coherence)
        learning_info[str(coherence)] = learning_info_coh
        # learning_info[str(coherence)] = {curriculum_stop_info:"Desired accuracy not reached for curriculum part"}
        
    #save model and losses
    save_model_and_info(model, learning_info, epoch_losses, epoch_val_losses, accuracies, training_kwargs, epoch='final')

    return model, learning_info


#loading models
def load_model(model, training_kwargs):
    """Loads model from either after learning finished or from during training if training didn't finish."""
    try:
        model.load_state_dict(torch.load(training_kwargs['exp_path']+'\\weights%s.pth'%training_kwargs['idx'], map_location=torch.device(training_kwargs['device'])))
        model_name = training_kwargs['exp_path']+'\\weights%s.pth'%training_kwargs['idx']
    except:
        weight_list = glob.glob(training_kwargs['training_weights_path']+'\\*')
        file = weight_list[0]
        idx = int(re.search(r'\d+', file).group())
        coh_epoch_dict = {}
        epochs = []
        for file in weight_list:
            if 'final' in file:
                continue
            if training_kwargs['single_or_curriculum'] == 'curriculum':
                coh = float(file.split("coh",1)[1].split("_",1)[0])
                lastbit = file.split("weights",1)[1].split("_",1)[-1]
                epoch = lastbit.split("epoch",1)[1].split("_",1)[0]
                try:
                    coh_epoch_dict[coh].append(epoch)
                except:
                    coh_epoch_dict[coh] = [epoch]
            else:
                lastbit = file.split("weights",1)[1].split("_",1)[-1]
                epoch = lastbit.split("epoch",1)[1].split("_",1)[0]
                epochs.append(epoch)
                
        if training_kwargs['single_or_curriculum'] == 'curriculum':
            coherence = sorted(list(coh_epoch_dict.keys()))[0]
            epoch = sorted(coh_epoch_dict[coherence])[-1]
            model_name = training_kwargs['training_weights_path'] + '/weights_coh%s_epoch%s_%s.pth'%(coherence, epoch, training_kwargs['idx'])
            model.load_state_dict(torch.load(model_name, map_location=torch.device(training_kwargs['device'])))
        else:
            epoch = sorted(epochs)[-1]
            model_name = training_kwargs['training_weights_path'] + '/weights_epoch%s_%s.pth'%(epoch, training_kwargs['idx'])
            model.load_state_dict(torch.load(model_name, map_location=torch.device(training_kwargs['device'])))
            
    print(model_name)
    return model
