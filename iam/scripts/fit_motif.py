import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scipy.integrate import solve_ivp
import time
from typing import Optional
from .homeos import *
from .ds_class import *
from .torch_utils import *

def make_transition_pairs(trajectories: torch.Tensor) -> torch.Tensor:
    """
    Given trajectories of shape (B, T, N), return (B*(T-1), 2, N)
    where each entry is (x_t, x_{t+1}).
    """
    B, T, N = trajectories.shape
    x_t   = trajectories[:, :-1, :]    # (B, T-1, N)
    x_t1  = trajectories[:, 1:, :]     # (B, T-1, N)

    # Stack along new axis for the pair: (B, T-1, 2, N)
    pairs = torch.stack((x_t, x_t1), dim=2)

    # Flatten batch and time: (B*(T-1), 2, N)
    pairs = pairs.reshape(-1, 2, N)

    return pairs

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
            recent_losses = [current_loss]  # Extend window??
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


def compute_homeo_ds_loss(
    homeo_ds_net: nn.Module,
    batch_target: torch.Tensor,
    initial_conditions_target: torch.Tensor,
    loss_fn: Callable,
    noise_std: float = 0.0,
    use_inverse_formulation: bool = True
) -> torch.Tensor:
    """
    Compute the training loss for the homeomorphism-dynamical system network.

    Args:
        homeo_ds_net: The combined homeomorphism and dynamical system module.
        batch_target: Ground truth trajectories of shape (B, T, N).
        initial_conditions_target: Initial conditions y_0 of shape (B, N).
        loss_fn: A torch.nn loss function (e.g., MSELoss).
        noise_std: Optional noise for trajectory generation.
        use_inverse_formulation: Whether to use the inverse loss form.

    Returns:
        A scalar loss tensor.
    """
    batch_target_detached = batch_target.detach()
    B, T, N = batch_target_detached.shape

    if use_inverse_formulation:
        # Trajectory: Φ(f(Φ⁻¹(y0))) for all timesteps
        source_trajectories, transformed_trajectories = homeo_ds_net(initial_conditions_target, noise_std)

        # Loss: f(Φ(x_t)) vs Φ(x_{t+1})
        return source_trajectories, loss_fn(transformed_trajectories[:, 1:, :], batch_target_detached[:, 1:, :])

    else:
        # hat_x_0 = Φ(y_t)
        hat_x_0 = homeo_ds_net.homeo_network(batch_target_detached[:, 0, :])

        # x_{t+1} = f(x_t)
        if isinstance(homeo_ds_net.dynamical_system, AnalyticDynamicalSystem):
            source_trajectories = homeo_ds_net.dynamical_system.compute_trajectory(hat_x_0)[:, 1:, :]
        else:
            _, x_traj, _ = generate_trajectories(
                homeo_ds_net.dynamical_system,
                predefined_initial_conditions=hat_x_0,
                noise_std=noise_std
            )
            source_trajectories = x_traj[:, 1:, :]  # skip x0

        # Φ(f(x_t))
        phi_x_t1 = homeo_ds_net.homeo_network(source_trajectories)

        # Loss: Φ(f(Φ⁻¹(x_t))) vs x_{t+1}
        return source_trajectories, loss_fn(phi_x_t1, batch_target_detached[:, 1:, :])

def train_homeo_ds_net_batched(
    homeo_ds_net: nn.Module,  
    lr: float,
    trajectories_target: torch.Tensor,
    batch_size: int = 0,  # 0 or None means full-batch
    use_inverse_formulation: bool = True,
    num_epochs: int = 100,
    max_grad_norm: Optional[float] = None,
    annealing_params: Optional[dict] = None,
    early_stopping_patience: Optional[int] = None,
    early_stop_loss_explosion_factor: Optional[float] = 1e3,  # stop if loss increases 100x
    jac_lambda_reg:  Optional[float] = 0.,  # Default: no regularization
):
    """
    Train the homeomorphism network with automatic full-batch or mini-batch support.
    """
    if jac_lambda_reg is not None:
        assert jac_lambda_reg >= 0, "jac_lambda_reg must be non-negative"


    loss_fn = nn.MSELoss(reduction='mean')
    device = next(homeo_ds_net.parameters()).device
    trajectories_target = trajectories_target.to(device)
    B, T, N = trajectories_target.shape

    # Decide batching mode
    if batch_size is None or batch_size <= 0 or batch_size >= B:
        dataloader = [(trajectories_target,)]
    else:
        dataset = TensorDataset(trajectories_target)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    params_to_optimize = list(homeo_ds_net.homeo_network.parameters())
    if isinstance(homeo_ds_net.dynamical_system, (LearnableDynamicalSystem, AnalyticDynamicalSystem)):
        params_to_optimize += list(homeo_ds_net.dynamical_system.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=lr)
    if early_stopping_patience is None:
        early_stopping_patience = num_epochs 
    early_stopper = EarlyStopping(patience=early_stopping_patience)
    best_model_saver = BestModelSaver(homeo_net=homeo_ds_net.homeo_network, source_system=homeo_ds_net.dynamical_system)
    best_loss = float('inf')

    losses, grad_norms = [], []
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_start = time.time()

        for batch in dataloader:
            optimizer.zero_grad()
            batch_target = batch[0]  # (B, T, D)
            initial_conditions_target = batch_target[:, 0, :].detach().clone()

            running_loss = update_leaky_running_avg
            if annealing_params is not None:
                if annealing_params['dynamic']:
                    current_loss = losses[-1] if losses else 0.0
                    noise_std = get_annealed_noise_std(epoch, num_epochs, **annealing_params, current_loss=current_loss, running_loss=running_loss)
                else:
                    noise_std = get_annealed_noise_std(epoch, num_epochs, **annealing_params)
            else:
                noise_std = 0.0

            # Calculate loss using the compute_homeo_ds_loss function
            source_trajectories, loss = compute_homeo_ds_loss(
                homeo_ds_net,
                batch_target,
                initial_conditions_target,
                loss_fn,
                noise_std=noise_std,
                use_inverse_formulation=use_inverse_formulation,
            )

            if jac_lambda_reg is not None:
                jacobian_reg = jacobian_spectral_norm(homeo_ds_net.homeo_network, source_trajectories)
                loss += jac_lambda_reg * jacobian_reg 

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                break

            loss.backward()
            total_norm = torch.norm(torch.stack([p.grad.norm(2) for p in params_to_optimize if p.grad is not None]), 2)
            grad_norms.append(total_norm.item())

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(homeo_ds_net.parameters(), max_grad_norm)

            optimizer.step()
            epoch_losses.append(loss.item())

        mean_epoch_loss = np.mean(epoch_losses)
        if mean_epoch_loss > early_stop_loss_explosion_factor * best_loss:
            print(f"Loss exploded at epoch {epoch}. Previous best loss: {best_loss:.4e}, Current loss: {mean_epoch_loss:.4e}")
            break
        best_loss = min(best_loss, mean_epoch_loss)
        losses.append(mean_epoch_loss)
        best_model_saver.step(mean_epoch_loss)

        if epoch % 10 == 0:
            # if hasattr(homeo_ds_net.dynamical_system, 'velocity') and homeo_ds_net.dynamical_system.velocity isinstance(nn.Parameter):
            #     print(f"Epoch {epoch}, log(Loss)= {np.log10(mean_epoch_loss):.4f}", "Velocity: ", np.round(homeo_ds_net.dynamical_system.velocity.detach().cpu().numpy(), 3))
            #else:
            print(f"Epoch {epoch}, log(Loss)= {np.log10(mean_epoch_loss):.4f}")

        early_stopper.step(mean_epoch_loss)
        if early_stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch}. log(Loss)= {np.log10(mean_epoch_loss):.4f}")
            break

    total_time = time.time() - start_time
    print(f"Final log(Loss)= {np.log10(mean_epoch_loss):.4f}, Total training time: {total_time:.2f} sec, Avg epoch time: {total_time / num_epochs:.4f} sec")

    homeo_ds_net.grad_norms = grad_norms
    #best_model_saver.restore()
    return homeo_ds_net, losses, grad_norms

def train_diffeo_ds_net_batched_alternating(
    diffeo_ds_net: nn.Module,
    lr: float,
    trajectories_target: torch.Tensor,
    num_epochs: int,
    batch_size: int = 0,
    use_inverse_formulation: bool = True,
    max_grad_norm: Optional[float] = None,
    jac_lambda_reg: Optional[float] = 0.0,
    annealing_params: Optional[dict] = None,
    early_stopping_patience: Optional[int] = None,
    early_stop_loss_explosion_factor: Optional[float] = 1e3,
):
    affine_params = [p for name, p in diffeo_ds_net.homeo_network.named_parameters() if 'affine' in name]
    diffeo_params = [p for name, p in diffeo_ds_net.homeo_network.named_parameters() if 'affine' not in name]

    optimizer_diffeo = torch.optim.Adam(diffeo_params, lr=lr)

    losses, affine_losses, diffeo_losses, grad_norms = [], [], [], []
    best_loss = float("inf")
    early_stopper = EarlyStopping(patience=early_stopping_patience or num_epochs)
    best_model_saver = BestModelSaver(
        homeo_net=diffeo_ds_net.homeo_network,
        source_system=diffeo_ds_net.dynamical_system
    )

    if batch_size <= 0 or batch_size >= trajectories_target.shape[0]:
        dataloader = [(trajectories_target,)]
    else:
        dataset = TensorDataset(trajectories_target)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # === Initial loss before training ===
    with torch.no_grad():
        x0 = trajectories_target[:, 0, :].detach().clone()
        source_traj, initial_loss = compute_homeo_ds_loss(diffeo_ds_net, trajectories_target, x0,loss_fn=nn.MSELoss(),noise_std=0.0,use_inverse_formulation=use_inverse_formulation)
        # if jac_lambda_reg:
        #     jacobian_reg = jacobian_spectral_norm(diffeo_ds_net.homeo_network.node, source_traj)
        #     initial_loss += jac_lambda_reg * jacobian_reg
        print(f"[Initial] log(Loss): {np.log10(initial_loss.item()):.4f}")

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch in dataloader:
            batch_target = batch[0]
            x0 = batch_target[:, 0, :].detach().clone()

            with torch.no_grad():
                source_traj, initial_loss = compute_homeo_ds_loss(diffeo_ds_net, trajectories_target, x0, loss_fn=nn.MSELoss(),noise_std=0.0,use_inverse_formulation=use_inverse_formulation)
                print(f"[Epoch {epoch}] before LS log(Loss): {np.log10(initial_loss.item()):.4f}")

            # === Affine phase ===
            for p in affine_params: p.requires_grad = False
            for p in diffeo_params: p.requires_grad = False

            with torch.no_grad():
                source_traj = diffeo_ds_net.dynamical_system.compute_trajectory(x0)
                transformed = diffeo_ds_net.homeo_network.node(source_traj)

            # Solve least squares: fit affine map from transformed -> batch_target
            x = transformed.reshape(-1, transformed.shape[-1])  # (B*T, d)
            y = batch_target.reshape(-1, batch_target.shape[-1])  # (B*T, d)

            # Add bias term to x
            x_aug = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)  # (B*T, d+1)

            # Solve (XᵗX)⁻¹XᵗY
            XtX = x_aug.T @ x_aug
            XtY = x_aug.T @ y
            theta = torch.linalg.lstsq(x_aug, y).solution
            #theta = torch.linalg.lstsq(XtX, XtY).solution

            # Update affine parameters
            with torch.no_grad():
                A_param = diffeo_ds_net.homeo_network.affine.A  # assumes shape (d, d)
                b_param = diffeo_ds_net.homeo_network.affine.b  # assumes shape (d,)

                A_param.copy_(theta[:-1].T)
                b_param.copy_(theta[-1])

            with torch.no_grad():
                source_traj, initial_loss = compute_homeo_ds_loss(diffeo_ds_net, trajectories_target, x0,loss_fn=nn.MSELoss(),noise_std=0.0,use_inverse_formulation=use_inverse_formulation)
                print(f"[Epoch {epoch}] after LS log(Loss): {np.log10(initial_loss.item()):.4f}")

            # === Diffeo (NODE) phase ===
            for p in affine_params: p.requires_grad = False
            for p in diffeo_params: p.requires_grad = True

            optimizer_diffeo.zero_grad()
            noise_std = 0.0
            if annealing_params is not None:
                noise_std = get_annealed_noise_std(
                    epoch, num_epochs,
                    **annealing_params,
                    current_loss=losses[-1] if losses else 0.0,
                    running_loss=update_leaky_running_avg,
                ) if annealing_params.get("dynamic", False) else get_annealed_noise_std(epoch, num_epochs, **annealing_params)

            source_traj, loss_diffeo_grad = compute_homeo_ds_loss(
                diffeo_ds_net, batch_target, x0,
                loss_fn=nn.MSELoss(),
                noise_std=noise_std,
                use_inverse_formulation=use_inverse_formulation
            )

            if jac_lambda_reg:
                jacobian_reg = jacobian_spectral_norm(diffeo_ds_net.homeo_network.node, source_traj)
                loss_diffeo = loss_diffeo_grad + jac_lambda_reg * jacobian_reg
            else:
                jacobian_reg = torch.tensor(0.0, device=loss_diffeo.device)

            if torch.isnan(loss_diffeo).any() or torch.isinf(loss_diffeo).any():
                print("NaN or Inf detected. Aborting.")
                return diffeo_ds_net, losses, grad_norms

            loss_diffeo.backward()
            total_norm = torch.norm(torch.stack([
                p.grad.norm(2) for p in diffeo_params if p.grad is not None
            ]), 2)
            grad_norms.append(total_norm.item())

            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(diffeo_params, max_grad_norm)
            optimizer_diffeo.step()

            epoch_losses.append((loss_diffeo.item()))

            with torch.no_grad():
                source_traj, initial_loss = compute_homeo_ds_loss(diffeo_ds_net, trajectories_target, x0,loss_fn=nn.MSELoss(),noise_std=0.0,use_inverse_formulation=use_inverse_formulation)
                print(f"[Epoch {epoch}] after diffeom log(Loss): {np.log10(initial_loss.item()):.4f}")

        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)
        best_model_saver.step(mean_loss)
        best_loss = min(best_loss, mean_loss)

        if mean_loss > early_stop_loss_explosion_factor * best_loss:
            print(f"Loss exploded. Stopping early. Epoch {epoch}, loss: {mean_loss:.4e}")
            break

        early_stopper.step(mean_loss)
        if early_stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] | log(Loss Diffeo): {np.log10(loss_diffeo_grad.item()):.4f} | Jacobian Norm: {jacobian_reg.item():.4e}")

    #best_model_saver.restore()
    return diffeo_ds_net, losses, grad_norms





def train_all_hdsns(homeo_ds_nets, trajectories_target, initial_conditions_target,
    lr: float, num_epochs=100, use_inverse_formulation=True, 
    annealing_params=None, early_stopping_patience=None):
    """
    Train all Homeo-DS-Nets in the library using the same target trajectories.
    """
    all_losses = []
    all_grad_norms = []
    for motif, homeo_net in homeo_ds_nets:
        print("Training homeomorphism for motif:", motif.__class__.__name__)
        homeo_ds_net, losses, all_grad_norms = train_homeomorphism(homeo_net, source_system=motif, lr=lr,
            trajectories_target=trajectories_target, initial_conditions_target=initial_conditions_target,
            num_epochs=num_epochs, use_inverse_formulation=use_inverse_formulation,
            annealing_params=annealing_params, early_stopping_patience=early_stopping_patience
        )
        homeo_ds_nets.append(homeo_ds_net)
        all_losses.append(losses)
        all_grad_norms.append(all_grad_norms)
    return homeo_ds_nets, all_losses, all_grad_norms



#old 
def train_homeomorphism(
    homeo_net: nn.Module,
    source_system: nn.Module,
    trajectories_target: torch.Tensor,
    lr: float,
    use_transformed_system: bool=False,
    num_epochs: int = 100,
    jac_lambda_reg:  Optional[float] = 0.,  # Default: no regularization
    max_grad_norm: Optional[float] = None,  # Default: no clipping
    annealing_params: Optional[dict] = None,  # Default: no annealing
    early_stopping_patience: Optional[int] = 50,
):
    """
    Train the homeomorphism network while tracking training time and epoch time.
    """
    loss_fn = nn.MSELoss(reduction='mean')
    T = trajectories_target.shape[1]  
    
    start_time = time.time()  # Track total training time

    device = next(homeo_net.parameters()).device
    trajectories_target = trajectories_target.to(device)  
    initial_conditions_target = trajectories_target[:, 0, :].detach().clone()  # Get initial conditions from the first trajectory
    #initial_conditions_target = initial_conditions_target.to(device)  
    source_system = source_system.to(device)

    params_to_optimize = list(homeo_net.parameters())
    if isinstance(source_system, (LearnableDynamicalSystem, AnalyticDynamicalSystem)):
        params_to_optimize += list(source_system.parameters())
    optimizer=optim.Adam(params_to_optimize, lr=lr)
    early_stopper = EarlyStopping(patience=early_stopping_patience)
    best_model_saver = BestModelSaver(homeo_net=homeo_net, source_system=source_system)

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
        transformed_trajectories = generate_trajectories_for_training(homeo_net=homeo_net, source_system=source_system, initial_conditions_target=initial_conditions_target, noise_std=noise_std)
        trajectories_target_detached = [traj.detach() for traj in trajectories_target]

        loss = sum(loss_fn(x_t, phi_y_t) for x_t, phi_y_t in zip(trajectories_target_detached, transformed_trajectories)) / T

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"NaN or Inf detected in loss at epoch {epoch}.")
            break

        # if hasattr(homeo_net, 'jacobian_regularization'):
        #     loss += jac_lambda_reg * homeo_net.jacobian_regularization()

        loss.backward()
        total_norm = torch.norm(torch.stack([
            p.grad.norm(2) for p in params_to_optimize if p.grad is not None
        ]), 2)
        grad_norms.append(total_norm.item())
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(homeo_net.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(loss.item())  # Store loss for this epoch
        best_model_saver.step(loss.item())  # Track best model

        if epoch % 10 == 0:
            if hasattr(source_system, 'velocity'):
                print(f"Epoch {epoch}, log(Loss)= {np.log10(loss.item()):.4f}", "Velocity: ", np.round(source_system.velocity.detach().cpu().numpy(),3)) # " Lambda: ", source_system.lambda_.detach().cpu().numpy())
            else:
                print(f"Epoch {epoch}, log(Loss)= {np.log10(loss.item()):.4f}") 

        early_stopper.step(loss.item())
        if early_stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch}. log(Loss)= {np.log(loss.item()):.4f}")
            break

    total_time = time.time() - start_time  # Compute total training time
    print(f"Final log(Loss)= {np.log10(loss.item()):.4f}, Total training time: {total_time:.2f} seconds, Avg time per epoch: {total_time / num_epochs:.4f} sec")
    homeo_net.losses = losses  # Store losses 
    homeo_net.grad_norms = grad_norms  
    best_model_saver.restore()  #  Restore best model 
    return Homeo_DS_Net(homeo_network=homeo_net, dynamical_system=source_system)


def train_all_motifs(motif_library, homeo_networks, trajectories_target, initial_conditions_target,
    lr: float, num_epochs=100, use_transformed_system=False, 
    annealing_params=None, early_stopping_patience=None):
    """
    Train all motifs in the library using the same target trajectories and homeomorphism networks.
    """
    homeo_ds_nets = []

    for motif, homeo_net in zip(motif_library, homeo_networks):
        print("Training homeomorphism for motif:", motif.__class__.__name__)
        homeo_ds_net = train_homeomorphism(homeo_net, source_system=motif, lr=lr,
            trajectories_target=trajectories_target, initial_conditions_target=initial_conditions_target,
            num_epochs=num_epochs, use_transformed_system=use_transformed_system,
            annealing_params=annealing_params, early_stopping_patience=early_stopping_patience
        )
        homeo_ds_nets.append(homeo_ds_net)
    return homeo_ds_nets




