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
from pathlib import Path

#available motifs
motifs_2d = ['lds', 'lc', 'ring', 'bla', 'bistable']
motifs_3d = ['sphere', 'torus_attractor', 'torus_lc', 'cylinder']

set_seed(313)

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
                    homeo_type = 'node', layer_sizes = 1*[64], quick_jac = False, rescale_trajs = True,
                    train_ratio = 0.8, training_pairs = False, load_hdsnet_path = None,
                    lr = 0.01, num_epochs = 200, jac_lambda_reg = 0., 
                    random_seed = 313):

    
    save_dir = os.path.join(save_dir, ds_motif)
    os.makedirs(save_dir, exist_ok=True)
    set_seed(random_seed)
     
    trajectories_target = np.load(data_dir / f'{target_name}.npy')
    trajectories_target = trajectories_target
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
    if rescale_trajs:
        trajectories_target_full, trajectories_target, mean, std = normalize_scale_pair(trajectories_target, training_pairs)
    trajectories_target_train, trajectories_target_test = split_data(trajectories_target, train_ratio = train_ratio)

    #build homeo_ds_net
    homeo = build_homeomorphism(homeo_params)
    source_system = build_ds_motif(**ds_params)
    if load_hdsnet_path is not None:
        homeo_ds_net = load_homeo_ds_net(load_hdsnet_path, homeo, source_system)
    homeo_ds_net = Homeo_DS_Net(homeo, source_system)
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
    if quick_jac:
        jac_norm_frobenius = jacobian_frobenius_norm(homeo_ds_net.homeo_network, traj_trans).detach().numpy()
        jac_norm_spectral = jacobian_spectral_norm(homeo_ds_net.homeo_network, traj_trans).detach().numpy()
    else:
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


def perthomeo_exp(base_save_dir="homeopert_ring", ds_motif='ring', noise_std=0.0, layer_sizes=[64], num_epochs=200,
                 random_seed_target=313, random_seed_run=0):
    exp_folder = 'experiments/'
    if noise_std > 0:
        save_dir = os.path.join(exp_folder, base_save_dir, f"noise_{noise_std}", ds_motif)
    else:
        save_dir = os.path.join(exp_folder, base_save_dir, ds_motif)
    os.makedirs(save_dir, exist_ok=True)

    set_seed(random_seed_target)
    # Homeomorphism target
    homeo_type = 'node'
    target_layer_sizes = 1*[128]
    homeo_mean = 0.02
    homeo_std = 0.5
    dim = 2
    target_homeo_params = {
        'homeo_type': homeo_type, 'dim': dim, 'layer_sizes': target_layer_sizes,
        'init_type': 'small', 'activation': nn.ReLU, 'init_mean': homeo_mean, 'init_std': homeo_std
    }
    target_homeo = build_homeomorphism(target_homeo_params)
    save_homeo_ds_net(target_homeo, f"{save_dir}/target_homeo.pth")

    set_seed(random_seed_run)
    # Dataset and simulation settings
    dt = .2
    time_span = torch.tensor([0.0, 2.])
    train_ratio = 0.8
    analytic = True
    vf_on_ring_enabled = True
    training_pairs = False
    alpha_init = None
    velocity_init = None
    if training_pairs:
        time_span = torch.tensor([0.0, dt])

    ds_params = {
        'ds_motif': ds_motif, 'dim': dim, 'dt': dt, 'time_span': time_span,
        'analytic': analytic, 'vf_on_ring_enabled': vf_on_ring_enabled,
        'alpha_init': alpha_init, 'velocity_init': velocity_init
    }

    simulation_params = {
        'initial_conditions_mode': 'random',
        'number_of_target_trajectories': 50,
        'time_span': time_span,
        'dt': dt,
        'noise_std': 0.0,
        'training_pairs': training_pairs,
        'margin': 0.5,
        'seed': 42,
        'train_ratio': train_ratio,
        'ds_params': ds_params
    }

    homeo_params = {
        'homeo_type': homeo_type, 'dim': dim, 'layer_sizes': layer_sizes,
        'activation': nn.ReLU, 'init_type': None
    }

    training_params = {
        'lr': 0.01,
        'num_epochs': num_epochs,
        'annealing_params': {'dynamic': False, 'initial_std': 0.0, 'final_std': 0.0},
        'early_stopping_patience': 1000,
        'batch_size': 32,
        'use_inverse_formulation': True
    }

    all_parameters = {
        'target_homeo_params': target_homeo_params,
        'homeo_params': homeo_params,
        'training_params': training_params,
        'simulation_params': simulation_params
    }
    with open(os.path.join(save_dir, "parameters.pkl"), "wb") as f:
        pickle.dump(all_parameters, f)

    # Generate ring attractor trajectories
    if noise_std > 0:
        generator_ra = LearnableNDRingAttractor(dim=dim, dt=dt, time_span=time_span, noise_std=noise_std)
    else:
        generator_ra = AnalyticalRingAttractor(dim=dim, dt=dt, time_span=time_span)
    init_conds = prepare_initial_conditions(
        mode=simulation_params['initial_conditions_mode'],
        num_points=simulation_params['number_of_target_trajectories'],
        margin=simulation_params['margin'],
        seed=simulation_params['seed']
    )
    ra_trajs = generator_ra.compute_trajectory(torch.tensor(init_conds, dtype=torch.float32))

    B = ra_trajs.shape[0]
    n_train = int(train_ratio * B)
    n_test = B - n_train

    results = []
    for interpol_value_i, interpol_value in enumerate(np.linspace(0, 1, 11)):
        interpol_value = round(interpol_value, 2)
        print(f"Interpol value: {interpol_value}")

        interpolated_homeo = rescale_node_vf(target_homeo, interpol_value)
        target_jacobian_norm_fro = jacobian_norm_over_batch(interpolated_homeo, ra_trajs.reshape(-1, dim), norm_type='fro').detach().numpy()
        target_jacobian_norm_spec = jacobian_norm_over_batch(interpolated_homeo, ra_trajs.reshape(-1, dim), norm_type='spectral').detach().numpy()

        trajectories_target_full = interpolated_homeo(ra_trajs)
        trajectories_target_full, trajectories_target, mean, std = normalize_scale_pair(trajectories_target_full, simulation_params['training_pairs'])

        train_set, test_set = random_split(trajectories_target_full, [n_train, n_test])
        trajectories_target_train = trajectories_target[train_set.indices]
        trajectories_target_test = trajectories_target[test_set.indices]

        target_ra_points = get_homeo_invman(interpolated_homeo)
        target_ra_points = (target_ra_points - mean.detach().numpy()) / std.detach().numpy()

        homeo = build_homeomorphism(homeo_params)
        source_system = build_ds_motif(**ds_params)
        homeo_ds_net = Homeo_DS_Net(homeo, source_system)
        jac_norm_frobenius_pre = jacobian_norm_over_batch(homeo_ds_net.homeo_network, trajectories_target.reshape(-1,dim), norm_type='fro').detach().numpy()
        jac_norm_spectral_pre = jacobian_norm_over_batch(homeo_ds_net.homeo_network, trajectories_target.reshape(-1,dim), norm_type='spectral').detach().numpy()
        homeo_ds_net, losses, grad_norms = train_homeo_ds_net_batched(
            homeo_ds_net=homeo_ds_net,
            trajectories_target=trajectories_target_train,
            **training_params
        )
        homeo_ds_net.eval()

        _, _, training_loss = test_single_homeo_ds_net(homeo_ds_net, trajectories_target_train)
        _, _, test_loss = test_single_homeo_ds_net(homeo_ds_net, trajectories_target_test)
        traj_src_np, traj_trans_np, _ = test_single_homeo_ds_net(homeo_ds_net, trajectories_target)

        fit_ra_points = get_homeo_invman(homeo_ds_net.homeo_network)
        jac_norm_frobenius = jacobian_norm_over_batch(homeo_ds_net.homeo_network, trajectories_target.reshape(-1,dim), norm_type='fro').detach().numpy()
        jac_norm_spectral = jacobian_norm_over_batch(homeo_ds_net.homeo_network, trajectories_target.reshape(-1,dim), norm_type='spectral').detach().numpy()

        results.append({
            "interpol_value": interpol_value,
            "train_loss": training_loss,
            "test_loss": test_loss,
            "jac_norm_frobenius_pre":jac_norm_frobenius_pre,
            "jac_norm_spectral_pre":jac_norm_spectral_pre,
            "jac_norm_frobenius": jac_norm_frobenius,
            "jac_norm_spectral": jac_norm_spectral,
            "target_jacobian_norm_fro": target_jacobian_norm_fro,
            "target_jacobian_norm_spec": target_jacobian_norm_spec,
            "losses": losses,
            "grad_norms": grad_norms,
            "fit_ra_points": fit_ra_points,
            "target_ra_points": target_ra_points[0],
        })

        save_homeo_ds_net(homeo_ds_net, f"{save_dir}/homeo_{interpol_value}_rs{random_seed_run}.pth")
        np.save(f"{save_dir}/traj_motif_transformed_{interpol_value}_rs{random_seed_run}.npy", traj_trans_np)
        np.save(f"{save_dir}/traj_motif_source_{interpol_value}_rs{random_seed_run}.npy", traj_src_np)

    df = pd.DataFrame(results)
    df.to_pickle(f"{save_dir}/summary_df_rs{random_seed_run}.pkl")



def run_pert_ra_experiment(base_save_dir="vf_pert_ring/simple", ds_motif='ring', 
                           layer_sizes=[64], num_epochs=200, random_seed_vf=313, random_seed_run=0):
    exp_folder = 'experiments/'
    save_dir = os.path.join(exp_folder, base_save_dir, ds_motif)
    os.makedirs(save_dir, exist_ok=True)
    set_seed(random_seed_vf)

    dim = 2
    maxT = 5
    tsteps = maxT*20
    max_perturbation_norm = .15
    pring_simulation_params = {"maxT": maxT, "tsteps": tsteps, "perturbation_norm": 0.01,
        "random_seed": random_seed_vf, "min_val_sim": 3, "n_grid": 40,
        "add_limit_cycle": False,
        "num_points_invman": 200,
        "number_of_target_trajectories": 50, "initial_conditions_mode": "random", "init_margin": 0.25}
    
    dt = maxT / tsteps
    time_span = torch.tensor([0.0, maxT])
    analytic = True
    vf_on_ring_enabled = False #if analytic then not used
    training_pairs = False
    alpha_init = None
    velocity_init = None
    if training_pairs:
        time_span = torch.tensor([0.0, dt])
    train_ratio = 0.8
    ds_params = {'ds_motif': ds_motif, 'dim': dim, 'dt': dt, 'time_span': time_span, 'analytic': analytic, 'vf_on_ring_enabled': vf_on_ring_enabled, 'alpha_init': alpha_init, 'velocity_init': velocity_init}
    simulation_params = {'initial_conditions_mode': 'random', 'number_of_target_trajectories': 50, 'time_span': time_span, 'dt': dt, 'noise_std': .0, 'max_perturbation_norm': max_perturbation_norm,
                        'training_pairs': training_pairs, 'margin': 0.5, 'seed': 42, 'train_ratio': train_ratio, 'ds_params': ds_params, 'pring_simulation_params': pring_simulation_params}
    
    homeo_type = 'node'
    init_type = None
    homeo_params = {'homeo_type': homeo_type, 'dim': dim, 'layer_sizes': layer_sizes, 'activation': nn.ReLU, 'init_type': init_type}

    lr = 0.01
    annealing_params = {'dynamic': False, 'initial_std': .0, 'final_std': 0.}
    training_params = {'lr': lr, 'num_epochs': num_epochs, 'annealing_params': annealing_params, 'early_stopping_patience': 1000, "batch_size": 32, 'use_inverse_formulation': True}
    
    all_parameters = { 'homeo_params': homeo_params, 'training_params': training_params, 'simulation_params': simulation_params}
    with open(f"{save_dir}/parameters.pkl", "wb") as f:
        pickle.dump(all_parameters, f)

    set_seed(random_seed_run)
    results = []
    for p_norm in np.arange(0.0, max_perturbation_norm + 0.001, 0.01):  # steps of 0.01
        p_norm = round(p_norm, 2)
        pring_simulation_params['perturbation_norm'] = p_norm
        print(f"Perturbation norm: {p_norm}")
        print("Creating perturbed ring attractor...")
        X, Y, U_pert, V_pert, grid_u, grid_v, perturb_grid_u, perturb_grid_v, full_grid_u, full_grid_v, inv_man, trajectories_pertring = build_perturbed_ringattractor(**pring_simulation_params) 
        trajectories_target = torch.tensor(trajectories_pertring, dtype=torch.float32)

        #init homeo_ds_net
        trajectories_target_full, trajectories_target, mean, std = normalize_scale_pair(trajectories_target, False)
        inv_man = (inv_man - mean.detach().numpy()) / std.detach().numpy()

        # Plot the trajectories and vector field
        # plot_vector_field_fixedquivernorm_speedcontour(ax, X, Y, U_pert, V_pert, trajectories_pertring, title=f"$\|{p_norm}\|$",
        #                                                scale=1.0, color='teal', alpha=0.5)
        # plot_vector_field_coloredquivernorm(ax, X, Y, U_pert, V_pert, trajectories_pertring, title=f"$\|{p_norm}\|$",
        #                                    traj_color='darkblue', cmap='inferno', save_name=f"{save_dir}/vf_{p_norm}.pdf")
        np.save(f"{save_dir}/trajectories_target_{p_norm}.npy", trajectories_target.detach().numpy())

        B = trajectories_target.shape[0]
        n_train = int(train_ratio * B)
        n_test = B - n_train
        train_set, test_set = random_split(trajectories_target_full, [n_train, n_test])
        trajectories_target_train = trajectories_target[train_set.indices]
        trajectories_target_test = trajectories_target[test_set.indices]

        #train homeo_ds_net
        homeo = build_homeomorphism(homeo_params)
        source_system = build_ds_motif(**ds_params)
        homeo_ds_net = Homeo_DS_Net(homeo, source_system)
        homeo_ds_net, losses, grad_norms = train_homeo_ds_net_batched(homeo_ds_net=homeo_ds_net, trajectories_target=trajectories_target_train, **training_params)
        homeo_ds_net.eval()
        
        #test
        _, _, training_loss = test_single_homeo_ds_net(homeo_ds_net=homeo_ds_net, trajectories_target=trajectories_target_train)
        _, _, test_loss = test_single_homeo_ds_net(homeo_ds_net=homeo_ds_net, trajectories_target=trajectories_target_test)
        traj_src_np, traj_trans_np, _ = test_single_homeo_ds_net(homeo_ds_net=homeo_ds_net, trajectories_target=trajectories_target)

        fit_ra_points = get_homeo_invman(homeo_ds_net.homeo_network)
        jac_norm_frobenius = jacobian_norm_over_batch(homeo_ds_net.homeo_network, trajectories_target.reshape(-1,dim), norm_type='fro').detach().numpy()
        jac_norm_spectral = jacobian_norm_over_batch(homeo_ds_net.homeo_network, trajectories_target.reshape(-1,dim), norm_type='spectral').detach().numpy()
        
        results.append({
            "p_norm": p_norm,
            "train_loss": training_loss,
            "test_loss": test_loss,
            "jac_norm_frobenius": jac_norm_frobenius,
            "jac_norm_spectral": jac_norm_spectral,
            "losses": losses,
            "grad_norms": grad_norms,
            "fit_ra_points": fit_ra_points,
            "target_ra_points": inv_man[0],
        })

        save_homeo_ds_net(homeo_ds_net, f"{save_dir}/homeo_{p_norm}.pth")
        np.save(f"{save_dir}/traj_motif_transformed_{p_norm}.npy", traj_trans_np) #save trajectories_motif 
        np.save(f"{save_dir}/traj_motif_source_{p_norm}.npy", traj_src_np) #save trajectories_motif directly from the source 

    df = pd.DataFrame(results)
    df.to_pickle(f"{save_dir}/summary_df.pkl")

#run_pert_ra_experiment() #run time: 6h 43m
