import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
import time
from typing import Optional
from .diffeoms import *
from .ds_class import *


def get_annealed_noise_std(epoch: int, total_epochs: int, initial_std: float, final_std: float, anneal_epochs: Optional[int] = None) -> float:
    if anneal_epochs is None:
        anneal_epochs = total_epochs
    progress = min(epoch / anneal_epochs, 1.0)
    log_std = (1 - progress) * np.log(initial_std + 1e-8) + progress * np.log(final_std + 1e-8)
    return float(np.exp(log_std))

def train_diffeomorphism(
    diffeo_net: nn.Module,
    trajectories_target: torch.Tensor,
    source_system: nn.Module,
    use_transformed_system: bool,
    initial_conditions_target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 100,
    lambda_reg:  Optional[float] = 0.,  # Default: no regularization
    max_grad_norm: Optional[float] = None,  # Default: no clipping
    annealing_params: Optional[dict] = None,  # Default: no annealing
):
    """
    Train the diffeomorphism network while tracking training time and epoch time.
    """
    loss_fn = nn.MSELoss()
    num_points = len(trajectories_target)
    
    start_time = time.time()  # Track total training time

    device = next(diffeo_net.parameters()).device
    trajectories_target = [traj.to(device) for traj in trajectories_target]  # Move to device
    initial_conditions_target = initial_conditions_target.to(device)  # Move to device  

    losses = []  # Store losses for each epoch
    grad_norms = []
    for epoch in range(num_epochs):
        epoch_start = time.time()  # Track time per epoch
        optimizer.zero_grad()

        if annealing_params is not None:
            noise_std = get_annealed_noise_std(epoch, num_epochs, **annealing_params)
        else:
            noise_std = 0.0
        transformed_trajectories = generate_trajectories_for_training(diffeo_net, source_system, use_transformed_system, initial_conditions_target, )
        trajectories_target_detached = [traj.detach() for traj in trajectories_target]

        loss = sum(loss_fn(x_t, phi_y_t) for x_t, phi_y_t in zip(trajectories_target_detached, transformed_trajectories)) / num_points

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"NaN or Inf detected in loss at epoch {epoch}.")
            break

        if hasattr(diffeo_net, 'jacobian_regularization'):
            loss += lambda_reg * diffeo_net.jacobian_regularization()

        loss.backward()
        total_norm = torch.norm(torch.stack([p.grad.norm(2) for p in diffeo_net.parameters() if p.grad is not None]), 2)
        grad_norms.append(total_norm.item())
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(diffeo_net.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(loss.item())  # Store loss for this epoch
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    total_time = time.time() - start_time  # Compute total training time
    print(f"Total training time: {total_time:.2f} seconds, Avg time per epoch: {total_time / num_epochs:.4f} sec")
    diffeo_net.losses = losses  # Store losses 
    diffeo_net.grad_norms = grad_norms  
    return diffeo_net


def train_all_motifs(motif_library, diffeo_networks, trajectories_target, initial_conditions_target,lr=0.001, num_epochs=100, use_transformed_system=False, 
    annealing_params=None):
    """
    Train all motifs in the library using the same target trajectories and diffeomorphism networks.
    """
    for motif, diffeo_net in zip(motif_library, diffeo_networks):
        print("Training diffeomorphism for motif:", motif.__class__.__name__)
        diffeo_net = train_diffeomorphism(
            diffeo_net, source_system=motif,
            trajectories_target=trajectories_target,
            initial_conditions_target=initial_conditions_target,
            optimizer=optim.Adam(diffeo_net.parameters(), lr=lr),
            num_epochs=num_epochs, use_transformed_system=use_transformed_system,
            annealing_params=annealing_params
        )
    return diffeo_networks



