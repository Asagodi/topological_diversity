import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
import time, os, sys, pickle
import numpy as np
from scripts.utils import set_seed
from scripts.ds_class import *
from scripts.homeos import *
from scripts.plotting import *
from scripts.fit_motif import *
from scripts.time_series import *
from scripts.ra import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import seaborn as sns


#available motifs
motifs_2d = ['lds', 'lc', 'ring', 'bla', 'bistable', 'bibla']
motifs_3d = ['sphere', 'torus_attractor', 'torus_lc', 'cylinder']


set_seed(313)
from pathlib import Path
#exp_dir = Path('../experiments')
#data_dir = exp_dir / 'all_targets'
#save_dir = data_dir / 'motif_fits'

def split_data(trajectories_target, train_ratio = 0.8):
    B = trajectories_target.shape[0]
    n_train = int(train_ratio * B)
    n_test = B - n_train
    train_set, test_set = random_split(trajectories_target, [n_train, n_test])
    trajectories_target_train = trajectories_target[train_set.indices]
    trajectories_target_test = trajectories_target[test_set.indices]
    return trajectories_target_train, trajectories_target_test

def run_on_target(target_name, save_dir, data_dir, ds_motif = 'ring', analytic = False, canonical = True, maxT = 5,
                    alpha_init = None, velocity_init = None, vf_on_ring_enabled = False, #if analytic then not used
                    homeo_type = 'node', layer_sizes = 1*[64], 
                    train_ratio = 0.8, training_pairs = False, 
                    lr = 0.01, num_epochs = 200, jac_lambda_reg = 0., 
                    random_seed = 313):

    
    save_dir = os.path.join(save_dir, ds_motif)
    os.makedirs(save_dir, exist_ok=True)
    set_seed(random_seed)
     
    trajectories_target = np.load(data_dir / f'{target_name}.npy')
    dim = trajectories_target.shape[2]
    tsteps = trajectories_target.shape[1]
    B = trajectories_target.shape[0]

    dt = maxT / tsteps
    time_span = torch.tensor([0.0, maxT])
    
    if training_pairs:
        time_span = torch.tensor([0.0, dt])
    ds_params = {'ds_motif': ds_motif, 'dim': dim, 'dt': dt, 'time_span': time_span, 'analytic': analytic, 'canonical': canonical, 'vf_on_ring_enabled': vf_on_ring_enabled, 'alpha_init': alpha_init, 'velocity_init': velocity_init}
    init_type = None
    homeo_params = {'homeo_type': homeo_type, 'dim': dim, 'layer_sizes': layer_sizes, 'activation': nn.ReLU, 'init_type': init_type}
    annealing_params = {'dynamic': False, 'initial_std': .0, 'final_std': 0.}
    training_params = {'lr': lr, 'num_epochs': num_epochs, 'annealing_params': annealing_params, 'early_stopping_patience': 1000,
                        "batch_size": 32, 'use_inverse_formulation': True, 'jac_lambda_reg': jac_lambda_reg}
    all_parameters = { 'homeo_params': homeo_params, 'training_params': training_params, "ds_params": ds_params}
    with open(f"{save_dir}/parameters_{target_name}.pkl", "wb") as f:
        pickle.dump(all_parameters, f)

    trajectories_target = torch.tensor(trajectories_target, dtype=torch.float32).to(device)
    trajectories_target_full, trajectories_target, mean, std = normalize_scale_pair(trajectories_target, training_pairs)
    trajectories_target_train, trajectories_target_test = split_data(trajectories_target, train_ratio = train_ratio)

    #build homeo_ds_net
    homeo = build_homeomorphism(homeo_params)
    source_system_ra = build_ds_motif(**ds_params)
    homeo_ds_net = Homeo_DS_Net(homeo, source_system_ra)
    homeo_ds_net.to(device)
    #train homeo_ds_net
    homeo_ds_net, losses, grad_norms = train_homeo_ds_net_batched(homeo_ds_net=homeo_ds_net, trajectories_target=trajectories_target_train, **training_params)
    homeo_ds_net.eval()
    
    #test
    _, _, training_loss = test_single_homeo_ds_net(homeo_ds_net=homeo_ds_net, trajectories_target=trajectories_target_train)
    _, _, test_loss = test_single_homeo_ds_net(homeo_ds_net=homeo_ds_net, trajectories_target=trajectories_target_test)
    traj_src_np, traj_trans_np, _ = test_single_homeo_ds_net(homeo_ds_net=homeo_ds_net, trajectories_target=trajectories_target)
    traj_trans = torch.tensor(traj_trans_np, dtype=torch.float32)

    inv_man = homeo_ds_net.invariant_manifold(100).detach().numpy()
    jac_norm_frobenius = jacobian_norm_over_batch(homeo_ds_net.homeo_network, traj_trans.reshape(-1,dim), norm_type='fro').detach().numpy()
    jac_norm_spectral = jacobian_norm_over_batch(homeo_ds_net.homeo_network, traj_trans.reshape(-1,dim), norm_type='spectral').detach().numpy()

    np.savez(
    f"{save_dir}/results_{target_name}.npz",
    jac_fro=jac_norm_frobenius,
    jac_spec=jac_norm_spectral,
    training_loss=training_loss,
    test_loss=test_loss,
    losses=np.array(losses),  
    grad_norms=np.array(grad_norms),
    inv_man=inv_man
)

    save_homeo_ds_net(homeo_ds_net, f"{save_dir}/homeo_{target_name}.pth")
    np.save(f"{save_dir}/traj_motif_transformed_{target_name}.npy", traj_trans_np) 
    np.save(f"{save_dir}/traj_motif_source_{target_name}.npy", traj_src_np) 