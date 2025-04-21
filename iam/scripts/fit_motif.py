import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
import time
from .diffeoms import *
from .ds_class import *


def train_diffeomorphism(
    diffeo_net: nn.Module,
    trajectories_target: torch.Tensor,
    source_system: nn.Module,
    use_transformed_system: bool,
    initial_conditions_target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 100
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
    for epoch in range(num_epochs):
        epoch_start = time.time()  # Track time per epoch
        optimizer.zero_grad()

        transformed_trajectories = generate_trajectories_for_training(diffeo_net, source_system, use_transformed_system, initial_conditions_target)
        trajectories_target_detached = [traj.detach() for traj in trajectories_target]

        loss = sum(loss_fn(x_t, phi_y_t) for x_t, phi_y_t in zip(trajectories_target_detached, transformed_trajectories)) / num_points

        #target_batch = torch.stack(trajectories_target_detached)  # [N, T, d]
        #transformed_batch = torch.stack(transformed_trajectories)  # [N, T, d]
        #loss = loss_fn(transformed_batch, target_batch)
        if hasattr(diffeo_net, 'jacobian_regularization'):
            loss += lambda_reg * diffeo_net.jacobian_regularization()

        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # Store loss for this epoch
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    total_time = time.time() - start_time  # Compute total training time
    print(f"Total training time: {total_time:.2f} seconds, Avg time per epoch: {total_time / num_epochs:.4f} sec")
    diffeo_net.losses = losses  # Store losses in the network for later analysis
    return diffeo_net


def train_all_motifs(motif_library, diffeo_networks, trajectories_target, initial_conditions_target,lr=0.001, num_epochs=100, use_transformed_system=False):
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
            num_epochs=num_epochs, use_transformed_system=use_transformed_system
        )
    return diffeo_networks