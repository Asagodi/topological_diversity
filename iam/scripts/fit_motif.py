import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
import time
from typing import Optional
from .homeos import *
from .ds_class import *
from .torch_utils import *

def update_leaky_running_avg(current_value: float, running_avg: float, alpha: float = 0.9) -> float:
    """
    Update the exponentially weighted moving average.
    :param current_value: The current value (e.g., loss in this case).
    :param running_avg: The previous running average.
    :param alpha: The smoothing factor (leakiness).    """
    return (1 - alpha) * current_value + alpha * running_avg


def get_annealed_noise_std(
    epoch: int,
    total_epochs: int,
    initial_std: float,
    final_std: float,
    dynamic: bool = False,
    anneal_epochs: Optional[int] = None,
    current_loss: Optional[float] = None,
    running_loss: Optional[float] = None,
    loss_threshold: Optional[float] = None,
    variance_threshold: Optional[float] = None,
    noise_increase_factor: Optional[float] = 1.1,
    window_size: Optional[int] = 10,
    alpha: float = 0.1  # Leaky factor for exponential smoothing
) -> float:
    """
    Combine epoch-based annealing and loss-based annealing for dynamic noise adjustment.
    
    :param epoch: Current training epoch.
    :param total_epochs: Total number of epochs.
    :param initial_std: Initial noise standard deviation.
    :param final_std: Final noise standard deviation.
    :param anneal_epochs: Number of epochs to use for annealing (optional, defaults to `total_epochs`).
    :param current_loss: Current epoch's loss (required for dynamic annealing based on loss).
    :param running_loss: Running average of the loss (required for dynamic annealing based on loss).
    :param loss_threshold: Threshold below which we consider the loss to have improved sufficiently.
    :param variance_threshold: Threshold of loss variance below which we consider the loss stable.
    :param noise_increase_factor: Factor by which the noise increases if conditions are met (optional, default is 1.1).
    :param window_size: Number of previous epochs to consider for variance calculation (optional, default is 10).
    :param alpha: Smoothing factor for the exponentially weighted moving average (default is 0.1).
    
    :return: Adjusted noise standard deviation.
    """
    
    if anneal_epochs is None:
        anneal_epochs = total_epochs
    
    # If loss-based annealing is requested (current_loss and running_loss provided)
    if current_loss is not None and running_loss is not None and loss_threshold is not None and variance_threshold is not None:
        # Update the running average of the loss using the leaky running average
        running_loss = update_leaky_running_avg(current_loss, running_loss, alpha)
        
        # Calculate loss variance over the window (e.g., last 10 losses)
        if epoch >= window_size:
            recent_losses = [current_loss]  # Store recent losses in the window
            variance = np.var(recent_losses)
        else:
            variance = 0  # Not enough data for variance calculation
        
        # Check if the loss is improving enough (based on variance and running average)
        if current_loss > running_loss * loss_threshold and variance > variance_threshold:
            # If the loss is not improving (loss above threshold and variance is too high), increase the noise
            print("Loss not improving, increasing noise...")
            noise_std = initial_std * noise_increase_factor
        else:
            # Else, continue annealing normally
            noise_std = initial_std - (epoch / anneal_epochs) * (initial_std - final_std)
        
        # Ensure the noise doesn't go below the final standard deviation
        noise_std = max(noise_std, final_std)
    
    else:
        # Standard epoch-based annealing
        progress = min(epoch / anneal_epochs, 1.0)
        log_std = (1 - progress) * np.log(initial_std + 1e-8) + progress * np.log(final_std + 1e-8)
        noise_std = float(np.exp(log_std))
    
    return noise_std



def train_homeomorphism(
    homeo_net: nn.Module,
    lr: float,
    trajectories_target: torch.Tensor,
    source_system: nn.Module,
    use_transformed_system: bool,
    initial_conditions_target: torch.Tensor,
    #optimizer: Optional[torch.optim.Optimizer] = None,
    num_epochs: int = 100,
    lambda_reg:  Optional[float] = 0.,  # Default: no regularization
    max_grad_norm: Optional[float] = None,  # Default: no clipping
    annealing_params: Optional[dict] = None,  # Default: no annealing
    early_stopping_patience: Optional[int] = 50,
):
    """
    Train the homeomorphism network while tracking training time and epoch time.
    """
    loss_fn = nn.MSELoss()
    num_points = len(trajectories_target)
    
    start_time = time.time()  # Track total training time

    device = next(homeo_net.parameters()).device
    trajectories_target = [traj.to(device) for traj in trajectories_target]  
    initial_conditions_target = initial_conditions_target.to(device)  
    source_system = source_system.to(device)

    params_to_optimize = list(homeo_net.parameters())
    if isinstance(source_system, (LearnableDynamicalSystem, AnalyticalLimitCycle)):
        params_to_optimize += list(source_system.parameters())

    optimizer=optim.Adam(params_to_optimize, lr=lr)
    early_stopper = EarlyStopping(patience=early_stopping_patience)
    best_model_saver = BestModelSaver(model=)

    losses = []  # Store losses for each epoch
    grad_norms = []
    for epoch in range(num_epochs):
        epoch_start = time.time()  # Track time per epoch
        optimizer.zero_grad()

        running_loss = update_leaky_running_avg
        if annealing_params is not None:
            if annealing_params['dynamic']:
                current_loss = losses[-1] if losses else 0.0
                noise_std = get_annealed_noise_std(epoch, num_epochs, **annealing_params, current_loss=current_loss, running_loss=running_loss)
            noise_std = get_annealed_noise_std(epoch, num_epochs, **annealing_params)
        else:
            noise_std = 0.0
        transformed_trajectories = generate_trajectories_for_training(homeo_net, source_system, use_transformed_system, initial_conditions_target, noise_std=noise_std)
        trajectories_target_detached = [traj.detach() for traj in trajectories_target]

        loss = sum(loss_fn(x_t, phi_y_t) for x_t, phi_y_t in zip(trajectories_target_detached, transformed_trajectories)) / num_points

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"NaN or Inf detected in loss at epoch {epoch}.")
            break

        # if hasattr(homeo_net, 'jacobian_regularization'):
        #     loss += lambda_reg * homeo_net.jacobian_regularization()

        loss.backward()
        total_norm = torch.norm(torch.stack([
            p.grad.norm(2) for p in params_to_optimize if p.grad is not None
        ]), 2)
        grad_norms.append(total_norm.item())
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(homeo_net.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(loss.item())  # Store loss for this epoch
        saver(homeo_net, source_system, loss.item())  # Track best model

        if epoch % 10 == 0:
            if hasattr(source_system, 'velocity'):
                print(f"Epoch {epoch}, log(Loss)= {np.log(loss.item()):.4f}", "Velocity: ", np.round(source_system.velocity.detach().cpu().numpy(),3)) # " Lambda: ", source_system.lambda_.detach().cpu().numpy())
            else:
                print(f"Epoch {epoch}, log(Loss)= {np.log(loss.item()):.4f}") 

        early_stopper.step(loss.item())
        if early_stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch}. log(Loss)= {np.log(loss.item()):.4f}")
            break

    total_time = time.time() - start_time  # Compute total training time
    print(f"Total training time: {total_time:.2f} seconds, Avg time per epoch: {total_time / num_epochs:.4f} sec")
    homeo_net.losses = losses  # Store losses 
    homeo_net.grad_norms = grad_norms  
    saver.restore(homeo_net, source_system)  #  Restore best model 
    return Homeo_DS_Net(homeo_network=homeo_net, dynamical_system=source_system)


def train_all_motifs(motif_library, homeo_networks, trajectories_target, initial_conditions_target,
    lr=0.001, num_epochs=100, use_transformed_system=False, 
    annealing_params=None):
    """
    Train all motifs in the library using the same target trajectories and homeomorphism networks.
    """
    homeo_ds_nets = []

    for motif, homeo_net in zip(motif_library, homeo_networks):
        print("Training homeomorphism for motif:", motif.__class__.__name__)
        homeo_ds_net = train_homeomorphism(
            homeo_net, source_system=motif,
            lr=lr,
            trajectories_target=trajectories_target,
            initial_conditions_target=initial_conditions_target,
            num_epochs=num_epochs, use_transformed_system=use_transformed_system,
            annealing_params=annealing_params
        )
        homeo_ds_nets.append(homeo_ds_net)
    return homeo_ds_nets




