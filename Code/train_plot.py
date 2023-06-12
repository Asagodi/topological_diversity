import os, sys
os.path.abspath(os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader

import numpy as np
import numpy.ma as ma
from math import isclose
import pickle
import argparse
import time
from datetime import datetime
from fractions import Fraction

from tasks import PerceptualDiscrimination, PoissonClicks, DynamicPoissonClicks
from utils import *

current_dir = os.path.dirname(os.path.realpath('__file__'))
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


parser = argparse.ArgumentParser(
    description="""
    A script to run training of the models on the perceptual discrimination task.
    """)
parser.add_argument(
    "--savedir",
    type=str,
    default="experiments",
    help="""
    Folder to save plots in. Individual experiments have a prefix folder of 
    the datetime.
    """)
parser.add_argument(
    '--disable_cuda',
    type=bool,
    default=False,
    help='Disable CUDA')

#data
parser.add_argument(
    "--dt", type=int,
    default=10,
    help="The simulation timestep.")
parser.add_argument(
    "--T", type=int,
    default=2000,
    help="Trial length.")
parser.add_argument(
    "--tau", type=int,
    default=100,
    help="The intrinsic time constant of neural state decay.")
parser.add_argument(
    "--single_or_curriculum", type=str,
    default="single",
    help="Single (mixed) dataset or curriculum dataset.")
parser.add_argument(
    "--dataset_file", type=str,
    default=None, #"C:\\Users\\abel_\\Documents\\Rotations\\CIT\\experiments\\curriculum_dataset_dense.pickle"
    help="If None, make dataset, else load file as dataset.")
# training_kwargs['dataset_idx'] = ""
parser.add_argument(
    "--coherence_list",
    type=str,
    default="0.7,0.5,0.3,0.1,0.05,0.025",
    help="Coherences used to make curriculum or single dataset.")
parser.add_argument(
    "--coherence",
    type=float,
    default=None,
    help="Coherence used to make dataset.")
parser.add_argument(
    "--data_size_per_coherence", type=int,
    default=5000,
    help="Curriculum dataset with different coherences. Write list without spaces, just commas.")
parser.add_argument(
    "--total_data_size", type=int,
    default=10000,
    help="Single dataset size.")
parser.add_argument(
    "--train_proportion", type=float,
    default=0.8,
    help="Single dataset train proportion.")
parser.add_argument(
    "--masked_training", type=bool,
    default=False,
    help="Use output_maks to determine which outputs are used for the learning process.")
parser.add_argument(
    "--onset_times", type=bool,
    default=None,
    help="Time of stimulus onset with a fixed value. Otherwise a random value is selected from a uniform distribution. Defaults to random.")
parser.add_argument(
    "--count_out", type=str,
    default=None,
    help="If not None, then target will be cumulative sum of clicks, instead of one-hot encoding (after cue). If '1D_uncued', then 1D output left-right clicks, If '2D_uncued', two sides separately.")
parser.add_argument(
    "--input_noise", type=float,
    default=0.,
    help="Adds noise to the input if nonzero.")
parser.add_argument(
    "--input_noise_startpulse_steps", type=int,
    default=0.,
    help="Adds noise for input_noise_startpulse_steps timesteps to the input if nonzero. Uses input_noise to sample from Uniform(-input_noise,input_noise).")
parser.add_argument(
    "--stim_durations",
    type=str,
    default=None,
    help="Stimulus durations list used to make curriculum or dataset. If None, random stimulus durations are used (from a uniform distribution for PD and from a truncated exponential for PC). If a list specified randomly chosen durations from list.")
task_keys = ['perceptual_discrimination', 'poisson_clicks', 'dynamic_poisson_clicks', 'motion_pulse', 'copy_memory']
parser.add_argument(
    "--task",
    type=str,
    default=None,
    choices=task_keys,
    help="Task to train model.")

#Task
##PoissonClicks
parser.add_argument(
    "--ratio", type=float,
    default=None,
    help="Ratio of rates of click pulses.")
parser.add_argument(
    "--sum_of_rates", type=float,
    default=40,
    help="Sum of rates of click pulses.")
parser.add_argument(
    "--ratios", type=str,
    default="-39,-37/3,-31/9,-26/14,26/14,31/9,37/3,39",
    help="Ratio list to pick random ratios for trials for dataset.")
parser.add_argument(
    "--fixed_cue_onsetandduration", type=str,
    default="10,5,20", #set to T=2000
    help="None for random cue onset and duration. Cues both stimulus and output.") 
parser.add_argument(
    "--fixed_stim_duration", type=int,
    default=None,
    help="None for random stimulus duration.")
parser.add_argument(
    "--fixed_stim_duration_list", type=str,
    default=None,
    help="List of stimulus durations and pauses betweeen them.") #TODO: incorporate fixed_stim_duration into this parameter.
#for exp_trunc_params
parser.add_argument(
    "--exp_trunc_params_b", type=float,
    default=1,
    help="Shape parameter for truncated exponential continuous random variable.")
parser.add_argument(
    "--exp_trunc_params_loc", type=float,
    default=200,
    help="Location parameter for truncated exponential continuous random variable.")
parser.add_argument(
    "--exp_trunc_params_scale", type=float,
    default=1000, #set to T/2.
    help="Scale parameter for truncated exponential continuous random variable.")
parser.add_argument(
    "--zero_target", type=float,
    default=.0,
    help="Target outside of desired output.")
parser.add_argument(
    "--clicks_capped", type=float,
    default=True,
    help="Cap clicks to be at most 1.")
parser.add_argument(
    "--equal_clicks", type=str,
    default="",
    help="What to do with trials with equal clicks? If 'mask' mask these trials.")
parser.add_argument(
    "--target_withcue", type=bool,
    default=False,
    help="Let target output start together with output cue.")
parser.add_argument(
    "--high_accuracy_stop", type=float,
    default=1.1,
    help="Stop learning of accuracy is reached.")
parser.add_argument(
    "--stationary_data", type=bool,
    default=True,
    help="Regenerate data after epoch?")
parser.add_argument(
    "--continuous_clicks", type=bool,
    default=False,
    help="Clicks are a real value.")

##DynamicPoissonClicks: hazard_rate, min_stim_start, max_stim_start, min_stim, max_stim, target_duration
parser.add_argument(
    "--hazard_rate", type=float,
    default=1,
    help="Hazard rate of switching of context (flip of rates).")
parser.add_argument(
    "--min_stim_start", type=int,
    default=20,
    help="Minimum of stimulus start.")
parser.add_argument(
    "--max_stim_start", type=int,
    default=50,
    help="Maximum of stimulus start.")
parser.add_argument(
    "--min_stim", type=int,
    default=500,
    help="Minimum of stimulus duration.")
parser.add_argument(
    "--max_stim", type=int,
    default=2000,
    help="Minimum of stimulus duration.")
parser.add_argument(
    "--cue_duration", type=int,
    default=50,
    help="Duration cue.")
parser.add_argument(
    "--target_duration", type=int,
    default=50,
    help="Duration target output.")


## Copy memory
parser.add_argument(
    "--N_symbols", type=int,
    default=8,
    help="Number of sysmbols to randomly pick from.")
parser.add_argument(
    "--input_length", type=int,
    default=10,
    help="Number of symbols input symbols to be remembered during input phase.")
parser.add_argument(
    "--delay", type=int,
    default=10,
    help="Number of timesteps to before output cue.")
    
    
    
    
#train params
parser.add_argument(
    "--dataloader_batch_size",
    type=int,
    default=50,
    help="Batch size for training.")
parser.add_argument(
    "--learning_rate", type=float,
    default=1e-3,
    help="Learning rate for optimizer. No other parameters are exposed. Default: 1e-3")
loss_function_keys = ['ce', 'mse', 'bce', 'bcewll']
parser.add_argument(
    "--loss_function", type=str,
    default=loss_function_keys[0],
    choices=loss_function_keys,
    help="Loss function for fitting parameters.")
optimizer_keys = ['rmsprop', 'adam', 'sgd']
parser.add_argument(
    "--optimizer", type=str,
    default=optimizer_keys[0],
    help="Optimizer for fitting parameters.")
parser.add_argument(
    "--sgd_momentum", type=float,
    default=0.9,
    help="Momentum parameter for SGD.")
scheduler_keys = [None, 'steplr', 'cosineannealing']
parser.add_argument(
    "--scheduler", type=str,
    default=scheduler_keys[0],
    choices=scheduler_keys,
    help="Scheduler for learning rate adaptation.")
parser.add_argument(
    "--steplr_gamma", type=float,
    default=0.1,
    help="Multiply learning rate by this factor every training_kwargs['scheduler_step_size'] epochs.")
parser.add_argument(
    "--scheduler_step_size", type=int,
    default=40,
    help="Change learning rate every scheduler_step_size epochs.")
parser.add_argument(
    "--clip_grads", type=bool,
    default=False,
    help="If true, clip gradients by norm training_kwargs['clip_grads_value']. Default: True.")
parser.add_argument(
    "--clip_grads_value", type=float,
    default=1,
    help="Value for gradient clipping. Default: 1.")
parser.add_argument(
    "--weight_decay", type=float,
    default=0.,
    help="The weight_decay parameter adds an L2 penalty to the cost.")
parser.add_argument(
    "--norm", type=str,
    default=None,
    help="Adds L1 or L2 penalty to the cost.")
parser.add_argument(
    "--norm_lambda", type=float,
    default=1e-3,
    help="Lambda hyperparameter determines how severe the added penalty to the cost is for weights sizes.")
parser.add_argument(
    "--act_norm_lambda", type=float,
    default=0,
    help="Parameter to regularize hiddden activity of network.")
parser.add_argument(
    "--continual_learning", type=bool,
    default=False,
    help="Learning with disjoint trials or as continual learning where the hidden activations of the last timestep of the last batch are used as the hidden activations of the first timestep of the current batch.")

#dead units
parser.add_argument(
    "--check_the_dead", type=bool,
    default=False,
    help="If true, then reset dead units. Default: False.")
parser.add_argument(
    "--check_the_dead_per_epoch", type=int,
    default=1,
    help="Frequency by which to check for dead units.")
parser.add_argument(
    "--weight_reinitialization_variance", type=float,
    default=1e-3,
    help="Variance of normal distribution with which to recplace dead units.")

#early stopping params
parser.add_argument(
    "--early_stopping_criterion_epochs", type=int,
    default=200,
    help="If loss doesn't decrease for early_stopping_criterion_epochs, stop training.")
parser.add_argument(
    "--difference_tolerance", type=float,
    default=1e-5,
    help="Tolerance for the difference between the minimum validation loss and the validation loss at current epoch.")

#curriculum params
parser.add_argument(
    "--curriculum_switching_criterion_epochs", type=int,
    default=100,
    help="If loss doesn't decrease for curriculum_switching_criterion_epochs, stop learning at this stadium of the curriculum.")
parser.add_argument(
    "--accuracy_goal", type=float,
    default=0.99,
    help="Accuracy that needs to be achived to go to next curriculum.")
parser.add_argument(
    "--max_epochs", type=int,
    default=200,
    help="Maximum epochs per curriculum.")
parser.add_argument(
    "--min_epochs_per_curriculum", type=int,
    default=5,
    help="Minimum epochs per curriculum.")
##parameter perturbations
parser.add_argument(
    "--parameter_noise_variance", type=float,
    default=0,
    help="Add noise with parameter_noise_variance to the parameters at the beginning of an epoch.")
parser.add_argument(
    "--perturbation_per_epoch", type=int,
    default=10,
    help="Add noise to the parameters at the beginning of every perturbation_per_epoch.")

#model
neural_keys = ['rnn', 'rnn_interactivation', 'rnn_cued', 'lstm', 'gru']
parser.add_argument(
    "--model_type",
    type=str, default=neural_keys[0], 
    choices=neural_keys,
    help="Type of model for neural network.")
parser.add_argument(
    "--hidden_layers",
    type=int, nargs="+",
    default=[64], #TODO: generalize rnn model arch.
    help="Sizes of hidden layers.")
parser.add_argument(
    "--load_model",
    type=str,
    default="",
    help="If given, where to load the pre-trained or pre-initialized model.")
parser.add_argument(
    "--random_winout",
    type=str,
    default="",
    help="If False, use perfect weights for input and output. If 'win' use random weights for input projection. If 'winout' use random for both input and output weights.")
parser.add_argument(
    "--N_in",
    type=int,
    default=2,
    help="Number of inputs.")
parser.add_argument(
    "--N_out",
    type=int,
    default=2,
    help="Number of outputs.")
parser.add_argument(
    "--N_rec",
    type=int,
    default=20,
    help="Number of recurrent units.")
parser.add_argument(
    "--N_inter",
    type=int,
    default=10,
    help="Number of intermediate units.")
transform_functions = ['relu', 'tanh', 'leaky_relu']
parser.add_argument(
    "--transform_function",
    type=str,
    default=transform_functions[0],
    choices=transform_functions,
    help="Activation function for hidden states.")
parser.add_argument(
    "--n_layers",
    type=int,
    default=1,
    help="Number of hidden layers.")
hidden_initial_activations_keys = ['random', 'zero', 'offset'] 
parser.add_argument(
    "--hidden_initial_activations",
    type=str,
    default=hidden_initial_activations_keys[0],
    help="Number of hidden layers.")
parser.add_argument(
    "--hidden_initial_variance",
    type=float,
    default=0.,
    help="Variance of hidden unit activations at t=0.")
parser.add_argument(
    "--hidden_offset",
    type=float,
    default=0.,
    help="Offset for hidden unit activations at t=0.")
parser.add_argument(
    "--output_activation",
    type=str,
    default='identity',
    help="Function to apply after linear output mapping. Implemented: sigmoid and identity")
parser.add_argument(
    "--constrain_diagonal",
    type=bool,
    default=False,
    help="If True restrict diagonal values of the recurrent weights to be less or equal to 1.")
parser.add_argument(
    "--constrain_spectrum",
    type=bool,
    default=False,
    help="If True restrict the spectrum of the recurrent weights to be less or equal to 1.")
parser.add_argument(
    "--N_blas",
    type=int,
    default=1,
    help="Number of bounded line attractors for initialization with loadmodel==blas.")

#perfect params
parser.add_argument(
    "--perfect_inout_variance",
    type=float,
    default=.0001,
    help="Variance (*sqrt(N)) of initial weights of random noise added to input and output weights of parfect model.")

#saving and printing
time_idx = int(time.time())
#add something to distinguis models that are trained in parallel
parser.add_argument(
    "--exp_path",
    type=str,
    default=False,
    help="Where to save the model after training.")
parser.add_argument(
    "--print_epoch",
    type=int,
    default=5000,
    help="Save training weights every 'print_epoch' epochs. Default: 5000.")
parser.add_argument(
    "--save_training_weights_epoch",
    type=int,
    default=5000,
    help="Save training weights every 'save_training_weights_epoch' epochs. Default: 5000.")
parser.add_argument(
    "--callback_every", type=int,
    default=50,
    help="Frequency by which to run callbacks.")
parser.add_argument(
    "--idx",
    type=int,
    default=time_idx,
    help="Index")
parser.add_argument(
    "--verbosity",
    type=bool,
    default=False,
    help="Print during training?")


args = parser.parse_args()

#choose device
if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

training_kwargs = args.__dict__
if training_kwargs['task'] == None:
    print("No task defined. Use --task= to define task. Options: 'perceptual_discrimination', 'poisson_clicks', 'motion_pulse'.")
else:
    print("Training for " + training_kwargs['task'])
    
# if trainin_kwargs['idx'] != time_idx
#     trainin_kwargs['idx'] = idx + str(time_idx)
    
#make lists from parameters: ratio, coherence, stimulus duration list
if "coherence_list" in training_kwargs.keys():
    training_kwargs["coherence_list"] = [float(s.strip()) for s in training_kwargs["coherence_list"].split(",")]
if "ratios" in training_kwargs.keys():
    training_kwargs["ratios"] = [float(Fraction(s.strip())) for s in training_kwargs["ratios"].split(",")]
if training_kwargs["fixed_stim_duration_list"] != None:
    training_kwargs["fixed_stim_duration_list"] = [int(s.strip()) for s in training_kwargs["fixed_stim_duration_list"].split(",")]
if training_kwargs["stim_durations"] != None:
    training_kwargs["stim_durations"] = [float(s.strip()) for s in training_kwargs["stim_durations"].split(",")]
if training_kwargs["fixed_cue_onsetandduration"] != None:
    training_kwargs["fixed_cue_onsetandduration"] = [int(s.strip()) for s in training_kwargs["fixed_cue_onsetandduration"].split(",")]
# fixed_startcue_onsetandduration
    
exp_trunc_params={}
exp_trunc_params['b'] = training_kwargs['exp_trunc_params_b']
exp_trunc_params['loc'] = training_kwargs['exp_trunc_params_loc']
exp_trunc_params['scale'] = training_kwargs['exp_trunc_params_scale']
training_kwargs['exp_trunc_params'] = exp_trunc_params

#compare args to get non-default args
# Arguments from command line and default values
alt_args = vars(parser.parse_args())
# Only default values
defaults = vars(parser.parse_args([]))
altered_args_list = [(key, alt_args[key]) for key in alt_args.keys() if alt_args[key]!=defaults[key] and key!='verbosity']
    
# exp_path = current_dir + '/experiments/exp_%s/'%time_idx
ignore_key_list = ['model_type', 'loss_function', 'single_or_curriculum', 'optimizer', 'load_model', 'transform_function']
ignore_value_list = ['coherence_list', 'fixed_stim_duration_list', 'target_withcue', 'equal_clicks', 'input_noise', 'continuous_clicks']
ignore_both = ['high_accuracy_stop', 'disable_cuda', 'dataloader_batch_size', 'max_epochs', 'T', 'N_in', 'accuracy_goal', 'dataset_file', 'fixed_cue_onsetandduration', 'task', 'coherence_list', 'ratios', 'stim_durations', 'early_stopping_criterion_epochs', 'exp_trunc_params_scale', 'N_rec', 'count_out', 'hidden_initial_variance', 'output_activation', 'equal_clicks', 'norm_lambda', 'N_out', 'hidden_offset', 'random_winout', 'perfect_inout_variance', 'input_noise_startpulse_steps', 'N_blas', 'input_length', 'N_symbols', 'total_data_size', 'steplr_gamma', 'scheduler_step_size', 'scheduler', 'hidden_initial_activations', 'save_training_weights_epoch', 'print_epoch']
if args.exp_path == False:
    # exp_path = current_dir + '/experiments/'+training_kwargs['optimizer']+'_'+training_kwargs['loss_function']+'_exp_%s/'%time_idx
    exp_path = current_dir   +  '/experiments/' + training_kwargs['task'] + '/exp'
    for (k, v) in altered_args_list:
        if k in ignore_both:
            continue
        elif k in ignore_value_list:
            exp_path += "_" + k.replace("_","")
        elif k in ignore_key_list:
            exp_path += "_" + str(v)
        else:
            exp_path += "_" + k.replace("_","") + str(v)
    exp_path += "_" + str(time_idx) + "/"
        
print(exp_path)
training_kwargs['device'] = device
training_kwargs['exp_path'] = exp_path
training_kwargs['figures_path'] = exp_path + '/figures/'   
training_kwargs['training_weights_path'] = exp_path + '/training/' #to save weights during training

makedirs(training_kwargs['exp_path'])
makedirs(training_kwargs['figures_path'])
makedirs(training_kwargs['training_weights_path'])


    
if training_kwargs['task'] == 'copy_memory':
    training_kwargs['N_in'] = training_kwargs['N_symbols']+2
    training_kwargs['N_out'] = training_kwargs['N_symbols']
    
for key in training_kwargs:
    print(key, ' : ', training_kwargs[key])


#load model if specified
if training_kwargs['load_model']!="":
    if training_kwargs['load_model'][1]=='1':
        training_kwargs['hidden_offset'] = 0
    elif training_kwargs['load_model'][1]=='2' or training_kwargs['load_model'][1]=='3':
        training_kwargs['hidden_offset'] = 100
    if training_kwargs['load_model'][0]=='v': #if in form v{i}, e.g. v1, v2 ,v3
        rnn_model = perfect_initialization(int(training_kwargs['load_model'][1]),random_winout=training_kwargs['random_winout'], eps=training_kwargs['perfect_inout_variance'])#.double()
        torch.save(rnn_model.state_dict(), training_kwargs['exp_path'] + '/starting_weights.pth')
    elif training_kwargs['load_model']=='blas':
        training_kwargs['N_rec'] = 2*training_kwargs['N_blas']
        rnn_model = bla_initialization(training_kwargs['N_in'], training_kwargs['N_blas'], training_kwargs['N_out'], training_kwargs['hidden_offset'], hidden_initial_activations=training_kwargs['hidden_initial_activations'])
    elif training_kwargs['load_model']=='ublas':
        training_kwargs['N_rec'] = 2*training_kwargs['N_blas']
        rnn_model = ubla_initialization(training_kwargs['N_in'], training_kwargs['N_blas'], training_kwargs['N_out'], training_kwargs['hidden_offset'], hidden_initial_activations=training_kwargs['hidden_initial_activations'])
    elif training_kwargs['load_model']=='identity':
        rnn_model = identity_initialization(training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'], hidden_initial_activations=training_kwargs['hidden_initial_activations'])
        
    elif training_kwargs['load_model']=='qpta':
        rnn_model = qpta_initialization(training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'], hidden_initial_activations=training_kwargs['hidden_initial_activations'])
        
    elif training_kwargs['load_model']=='ortho':
        rnn_model = ortho_initialization(training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_out'], hidden_initial_activations=training_kwargs['hidden_initial_activations'])

                
    else:
        rnn_model.load_state_dict(torch.load(training_kwargs['load_model']))
else:
    
    if training_kwargs['model_type'] == 'rnn':

        rnn_model = RNNModel(training_kwargs['N_in'], training_kwargs['N_out'], training_kwargs['N_rec'],
                         n_layers=training_kwargs['n_layers'], transform_function=training_kwargs['transform_function'],
                                         hidden_initial_variance=training_kwargs['hidden_initial_variance'],
                                         output_activation=training_kwargs['output_activation'], device=device).to(device)

    elif training_kwargs['model_type'] == 'rnn_interactivation':
        rnn_model = RNN_interactivation(training_kwargs['N_in'], training_kwargs['N_rec'], training_kwargs['N_inter'], training_kwargs['N_out'], 
                                        hidden_initial_activations=training_kwargs['hidden_initial_activations'],
                                        hidden_initial_variance=training_kwargs['hidden_initial_variance'],
                                        output_activation=training_kwargs['output_activation'], device=device).to(device)

    elif training_kwargs['model_type'] == 'rnn_cued':
        training_kwargs['N_in'] = 4
        if training_kwargs['task'] == "dynamic_poisson_clicks":
            training_kwargs['N_in'] = 3
        rnn_model = RNNModel(training_kwargs['N_in'], training_kwargs['N_out'], training_kwargs['N_rec'],
                         n_layers=training_kwargs['n_layers'], transform_function=training_kwargs['transform_function'],
                                         hidden_initial_activations=training_kwargs['hidden_initial_activations'],
                                         hidden_initial_variance=training_kwargs['hidden_initial_variance'],
                             output_activation=training_kwargs['output_activation'], device=device, constrain_spectrum=training_kwargs['constrain_spectrum']).to(device)
        
    elif training_kwargs['model_type'] == 'lstm':
        rnn_model = LSTM(training_kwargs['N_in'], training_kwargs['N_out'], training_kwargs['N_rec']).to(device)
        
    elif training_kwargs['model_type'] == 'gru':
        rnn_model = GRU(training_kwargs['N_in'], training_kwargs['N_out'], training_kwargs['N_rec']).to(device)


        
#save parameters, network and training
with open(training_kwargs['exp_path'] + '/training_kwargs.pickle', 'wb') as handle:
    pickle.dump(training_kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#load dataset if specified
if training_kwargs['dataset_file']!=None: 
    with open(training_kwargs['dataset_file'], 'rb') as handle:
        dataset = pickle.load(handle)

else:
    print("Creating dataset")
    if training_kwargs['single_or_curriculum'] == 'single':
        ###one dataset
        train_set, test_set, task_train, task_t = make_train_test_sets(training_kwargs)
        dataset = [train_set, test_set]
        with open(training_kwargs['exp_path'] + "/dataset.pickle", 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        ###curriculum dataset
        dataset, data_params = make_curriculum_datasets(training_kwargs)
        with open(training_kwargs['exp_path'] + "/curriculum_dataset.pickle", 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
start = time.time()
if training_kwargs['single_or_curriculum'] == 'single':
    rnn_model, learning_info = train_model(model=rnn_model,
                                          data_lists=dataset,
                                          training_kwargs=training_kwargs
                                         )
    print("Finished training " + str(time_idx))
else:
    rnn_model, learning_info = curriculum_learning(model=rnn_model,
                                                  curriculum_dataset=dataset,
                                                  training_kwargs=training_kwargs
                                                 )

