import math
import numpy as np
import itertools
import torch
import torch.nn as nn
from typing import Callable, Tuple, List, Optional
import torchdiffeq
import time

from .ds_class import *



class PeriodicActivation(nn.Module):
    """Implements a periodic activation function."""
    def __init__(self, function: str = "sin"):
        super().__init__()
        if function == "sin":
            self.func = torch.sin
        elif function == "cos":
            self.func = torch.cos
        else:
            raise ValueError(f"Unsupported periodic activation: {function}. Choose 'sin' or 'cos'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)


class HomeomorphismNetwork(nn.Module):
    """Represents a homeomorphic transformation
    Can be initialized as a perturbation-based transformation with a fixed normalization.
    """
    def __init__(self, dim: int, layer_sizes: list=3*[64], activation: str = "tanh", epsilon: float = None, grid_size: int = 10, grid_bound: float = 1) -> None:
        """
        Args:
            dim (int): Input dimension.
            layer_sizes (list): List of integers specifying the sizes of each hidden layer.
            activation (str): Activation function ('tanh', 'relu', etc.).
            epsilon (float, optional): If set, the transformation is scaled to this magnitude.
            grid_size (int): Number of points to use for computing normalization if epsilon is set.
        """
        super().__init__()
        self.epsilon = epsilon
        self.normalization_factor = None

        # Create the layers based on the provided layer sizes
        layers = []
        input_dim = dim
        activation_fn = self._get_activation(activation)
        for hidden_dim in layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn)
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, dim))
        self.layers = nn.Sequential(*layers)

        if epsilon is not None:
            self._compute_normalization(dim, grid_size, grid_bound)


    def _get_activation(self, activation: str) -> nn.Module:
        """Returns the corresponding activation function."""
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),  # Also known as Swish
            "sin": PeriodicActivation("sin"),
            "cos": PeriodicActivation("cos"),
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}. Choose from {list(activations.keys())}.")
        return activations[activation]

    def _compute_normalization(self, dim: int, grid_points: int, grid_bound: float) -> None:
        """Precompute the normalization factor based on a structured grid of points."""
        with torch.no_grad():
            # Generate an evenly spaced grid in each dimension
            linspace = torch.linspace(-grid_bound, grid_bound, grid_points)
            grid = torch.tensor(list(itertools.product(*[linspace] * dim)))  # Cartesian product to form the grid

            Hx = self.layers(grid)
            self.normalization_factor = torch.max(Hx).item()  # Compute the maximum norm over the grid
            # norm_Hx = torch.norm(Hx, dim=1, keepdim=True)
            # mean_norm = norm_Hx.mean()  # Compute average norm over the grid
            # self.normalization_factor = mean_norm.clamp(min=1e-6).item()  # Avoid division by zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Hx = self.layers(x)

        if self.epsilon is not None and self.normalization_factor is not None:
            Hx = (Hx / self.normalization_factor) * self.epsilon

            return x + Hx
        else:
            return Hx
        
    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Efficiently computes the Jacobian matrix of the transformation with respect to the input.
        
        Args:
            x (torch.Tensor): The input tensor for which the Jacobian is computed.
        
        Returns:
            jacobian (torch.Tensor): The Jacobian matrix.
        """
        #x.requires_grad_(True)  # Ensure that gradients can be computed w.r.t. the input.
        output = self.forward(x)

        jacobian = []
        for i in range(output.shape[1]):  # Loop over each output dimension
            grad_output = torch.zeros_like(output)
            grad_output[:, i] = 1  # Set the gradient for the i-th output component to 1

            # Compute the gradients of the output with respect to the input
            jacobian_i = torch.autograd.grad(output, x, grad_outputs=grad_output, create_graph=False)[0]
            jacobian.append(jacobian_i.unsqueeze(1))  # Add jacobian column

        jacobian = torch.cat(jacobian, dim=1)  # Concatenate all columns to form the Jacobian matrix
        return jacobian

    def inverse(self, x: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """Approximate inverse using fixed-point iteration."""
        y = x.clone()
        for _ in range(steps):
            y = x - self.forward(y) + y
        return y

# Function to generate homeomorphism with grid points
def generate_random_homeomorphism(dim: int, num_samples: int = 10, epsilon: float = 0.01, grid_points: bool = False, bounds: tuple = (-2,2), activation: str= 'tanh') -> torch.Tensor:
    """
    Generate homeomorphisms of the form Phi(x) = x + epsilon * H(x) where epsilon is small.
    Can visualize with grid points or random points.
    """
    homeomorphism_network = HomeomorphismNetwork(dim=dim, epsilon=epsilon,    activation=activation)

    # If grid_points is True, generate grid of points
    if grid_points:
        x_vals = np.linspace(bounds[0], bounds[1], num_samples)
        y_vals = np.linspace(bounds[0], bounds[1], num_samples)
        grid_points = np.array(np.meshgrid(x_vals, y_vals)).T.reshape(-1, 2)
        random_samples = torch.tensor(grid_points, dtype=torch.float32)
    else:
        # Otherwise, generate random points
        random_samples = torch.randn(num_samples, dim)  # Random samples from standard normal

    # Apply the homeomorphism (identity + epsilon * H(x))
    transformed_samples = homeomorphism_network(random_samples)

    return homeomorphism_network, random_samples, transformed_samples

###Affine
class AffineTransform(nn.Module):
    def __init__(self, dim: int, init_W: torch.Tensor = None, init_b: torch.Tensor = None, learnable: bool = False):
        super().__init__()
        if init_W is None:
            init_W = torch.eye(dim)
        if init_b is None:
            init_b = torch.zeros(dim)

        self.W = nn.Parameter(init_W, requires_grad=learnable)
        self.b = nn.Parameter(init_b, requires_grad=learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T + self.b
 

#### NODEs
class NeuralODE(nn.Module):
    def __init__(
        self, 
        dim: int, 
        layer_sizes: list[int], 
        activation: Callable[[], nn.Module] = nn.Tanh
    ) -> None:
        super().__init__()
        self.dim = dim
        
        layers = []
        input_dim = dim
        for hidden_dim in layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())  
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, dim))  # Final layer back to dim
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class NODEHomeomorphism(nn.Module):
    """Represents a homeomorphic transformation using a Neural ODE."""
    def __init__(self, dim: int, layer_sizes: list[int] = [64, 64],
                 activation: Callable[[], nn.Module] = nn.Tanh,
                 init_type: str = False, init_mean: float = 0.0, init_std: float = 1e-3, scale: float = 1.0,
                 t_span: tuple = (0.0, 1.0)) -> None:
        super().__init__()
        self.t_span = torch.tensor(t_span)  # Default integration time span
        self.neural_ode = NeuralODE(dim, layer_sizes, activation=activation)
        self.dim = dim
        self.layer_sizes = layer_sizes
        if init_type == 'identity':
            self._initialize_identity_weights()  # Initialize the layers as identity mapping
        elif init_type == 'small':
            self._initialize_small_weights(mean=init_mean, std=init_std)
        elif init_type == 'scaled':
            self._initialize_scaled_weights(scale=scale, mean=init_mean, std=init_std)

    def _initialize_identity_weights(self) -> None:
        """Initialize the neural ODE layers to represent the identity map."""
        for layer in self.neural_ode.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)  

    def _initialize_small_weights(self, mean: float = 0.0, std: float = 1e-3) -> None:
        """Initialize the neural ODE layers with random values."""
        for layer in self.neural_ode.mlp:
            if isinstance(layer, nn.Linear):              # Use small values for weights, ensuring it's close to identity
                nn.init.normal_(layer.weight, mean=mean, std=std) 
                #nn.init.zeros_(layer.bias)  # Initialize bias ?

    def _initialize_scaled_weights(self, scale: float = 1.0, mean: float = 0.0, std: float = 1e-3) -> None:
        """Initialize MLP with small weights and rescale only the last layer to control vector field norm."""
        last_linear = None

        for layer in self.neural_ode.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=mean, std=std)
                nn.init.zeros_(layer.bias)
                last_linear = layer  # Keep track of the last Linear layer

        if last_linear is not None:
            last_linear.weight.data.mul_(scale)
            last_linear.bias.data.mul_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map from source to target space."""
        t = self.t_span.to(x.device)
        y = torchdiffeq.odeint(self.neural_ode, x, t)
        return y[-1]  # Final state

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Map from target to source space."""
        t = self.t_span.flip(0).to(y.device)
        x = torchdiffeq.odeint(self.neural_ode, y, t)
        return x[-1]

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        output = self.forward(x)

        jacobian = []
        for i in range(output.shape[1]):
            grad_output = torch.zeros_like(output)
            grad_output[:, i] = 1
            grad_i = torch.autograd.grad(
                output, x, grad_outputs=grad_output,
                create_graph=True, retain_graph=True
            )[0]
            jacobian.append(grad_i.unsqueeze(1))
        return torch.cat(jacobian, dim=1)


###Invertible Residual Networks (iResNet)
class LipschitzMLP(nn.Module):
    """A simple 2-layer Lipschitz-constrained MLP using spectral norm with choosable activation."""
    def __init__(self, dim: int, hidden: int, activation: Callable[[], nn.Module] = nn.ELU):
        super().__init__()
        self.fc1 = nn.utils.spectral_norm(nn.Linear(dim, hidden))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(hidden, dim))
        self.activation = activation()  # Call the activation function to create an instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)  # Apply first linear transformation
        x = self.activation(x)  # Apply the chosen activation function
        x = self.fc2(x)  # Apply second linear transformation
        return x  # Return the output

class iResBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, activation: Callable[[], nn.Module] = nn.ELU, n_inverse_iter: int = 10, use_identity_init: bool = False):
        super().__init__()
        self.f = LipschitzMLP(dim, hidden, activation)
        self.n_inverse_iter = n_inverse_iter  # # of steps for fixed-point inversion

        if use_identity_init:
            self._initialize_identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        x = y.clone().detach()
        for _ in range(self.n_inverse_iter):
            x = y - self.f(x)
        return x

    def _initialize_identity(self):
        # Initialize the network parameters for identity mapping (if required)
        nn.init.zeros_(self.f.fc1.weight)
        nn.init.zeros_(self.f.fc2.weight)
        nn.init.zeros_(self.f.fc1.bias)
        nn.init.zeros_(self.f.fc2.bias)

class iResNet(nn.Module):
    def __init__(self, dim: int, layer_sizes: list[int], activation: Callable[[], nn.Module] = nn.ELU, use_identity_init: bool = False):
        super().__init__()
        n_layers = len(layer_sizes)
        self.blocks = nn.ModuleList([
            iResBlock(dim, layer_sizes[i], activation, use_identity_init=use_identity_init) for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y




###Invertible Residual Networks (iResNet)
# class LipschitzMLP(nn.Module):
#     def __init__(self, dim: int, hidden: int, activation: Callable[[], nn.Module] = nn.ELU,
#                  init_type: str = None, init_mean: float = 0.0, init_std: float = 1e-3, bias_value: float = 0.0):
#         super().__init__()

#         fc1 = nn.Linear(dim, hidden)
#         fc2 = nn.Linear(hidden, dim)

#         if init_type == 'identity':
#             nn.init.zeros_(fc1.weight)
#             nn.init.zeros_(fc2.weight)
#             nn.init.constant_(fc1.bias, bias_value)
#             nn.init.constant_(fc2.bias, bias_value)
#         elif init_type == 'small':
#             nn.init.normal_(fc1.weight, mean=init_mean, std=init_std)
#             nn.init.normal_(fc2.weight, mean=init_mean, std=init_std)
#             # Leave biases alone or initialize as you wish

#         # Now apply spectral norm
#         self.fc1 = nn.utils.spectral_norm(fc1)
#         self.fc2 = nn.utils.spectral_norm(fc2)
#         self.activation = activation()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc1(x)
#         x = self.activation(x)
#         x = self.fc2(x)
#         return x



# class iResBlock(nn.Module):
#     def __init__(self, dim: int, hidden: int,
#                  activation: Callable[[], nn.Module] = nn.ELU,
#                  n_inverse_iter: int = 10,
#                  init_type: str = None,
#                  init_mean: float = 0.0, init_std: float = 1e-3, bias_value: float = 0.0):
#         super().__init__()
#         self.f = LipschitzMLP(dim, hidden, activation,
#                               init_type=init_type,
#                               init_mean=init_mean, init_std=init_std,
#                               bias_value=bias_value)
#         self.n_inverse_iter = n_inverse_iter

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x + self.f(x)

#     def inverse(self, y: torch.Tensor) -> torch.Tensor:
#         x = y.clone().detach()
#         for _ in range(self.n_inverse_iter):
#             x = y - self.f(x)
#         return x


# class iResNet(nn.Module):
#     def __init__(self, dim: int, layer_sizes: list[int] = [64, 64],
#                  activation: Callable[[], nn.Module] = nn.ELU,
#                  init_type: str = 'identity', n_inverse_iter: int = 10,
#                  init_mean: float = 0.0, init_std: float = 1e-3, bias_value: float = 0.0):
#         super().__init__()
#         self.blocks = nn.ModuleList([
#             iResBlock(dim, hidden_size, activation,
#                       init_type=init_type,
#                       init_mean=init_mean, init_std=init_std, bias_value=bias_value)
#             for hidden_size in layer_sizes
#         ])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for block in self.blocks:
#             x = block(x)
#         return x

#     def inverse(self, y: torch.Tensor) -> torch.Tensor:
#         for block in reversed(self.blocks):
#             y = block.inverse(y)
#         return y


##old
class InvertibleResNetBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, use_identity_init: bool = False):
        super(InvertibleResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.split_size = in_channels # // 2

        self.fc1 = nn.Linear(self.split_size, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, self.split_size)
        self.activation = nn.ELU()

        if use_identity_init:
            self._initialize_identity()
        else:
            self._initialize_default()
    
    def _initialize_identity(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        if self.fc1.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.fc1.bias, -bound, bound)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _initialize_default(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        if self.fc1.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.fc1.bias, -bound, bound)
        if self.fc2.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.fc2.bias, -bound, bound)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into (split_size, remaining)
        x1 = x[:, :self.split_size]
        x2 = x[:, self.split_size:]
        z = self.activation(self.fc1(x1))
        x1_out = x1 + self.fc2(z)
        return torch.cat([x1_out, x2], dim=1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1 = y[:, :self.split_size]
        y2 = y[:, self.split_size:]
        z = self.activation(self.fc1(y1))
        y1_inv = y1 - self.fc2(z)
        return torch.cat([y1_inv, y2], dim=1)


class InvertibleResNet(nn.Module):
    """Represents a homeomorphic transformation using an invertible Residual Network."""
    def __init__(self, dim: int, layer_sizes: List[int], use_identity_init: bool = False):
        super(InvertibleResNet, self).__init__()
        self.dim = dim
        self.layer_sizes = layer_sizes
        self.blocks = nn.ModuleList([
            InvertibleResNetBlock(dim, hidden_size, use_identity_init=use_identity_init)
            for hidden_size in layer_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y



#####NORMALIZING FLOWS#####
# import normflows as nf

# class NormFlowHomeomorphism(nn.Module):
#     """Represents a homeomorphic transformation using a Normalizing Flow."""

#     def __init__(self, dim: int = 2, layer_sizes: list[int] = [64, 64], num_layers: int = 32):
#         super().__init__()
#         self.dim = dim
#         self.flow = self._build_flow(dim, layer_sizes, num_layers)

#     def _build_flow(self, dim: int, layer_sizes: list[int], num_layers: int):
#         base = nf.distributions.base.DiagGaussian(dim)
#         flows = []
#         for _ in range(num_layers):
#             param_map = nf.nets.MLP([dim // 2] + layer_sizes + [dim], init_zeros=True)
#             flows.append(nf.flows.AffineCouplingBlock(param_map))
#             flows.append(nf.flows.Permute(dim, mode='swap'))
#         return nf.NormalizingFlow(base, flows)

#     def _flatten_time(self, x: torch.Tensor) -> torch.Tensor:
#         return x.view(-1, x.shape[-1])

#     def _restore_time(self, x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
#         return x.view(shape)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Map from source system to target space (x -> y)."""
#         original_shape = x.shape
#         x_flat = self._flatten_time(x)
#         y_flat = self.flow.forward(x_flat)  # Returns (z, log_det); we discard log_det
#         #y_flat = torch.clamp(y_flat, -10, 10)
#         return self._restore_time(y_flat, original_shape)

#     def inverse(self, y: torch.Tensor) -> torch.Tensor:
#         """Map from target space to source system (y -> x)."""
#         original_shape = y.shape
#         y_flat = self._flatten_time(y)
#         x_flat = self.flow.inverse(y_flat)
#         #x_flat = torch.clamp(x_flat, -10, 10)
#         return self._restore_time(x_flat, original_shape)


#building
def build_homeomorphism(params: dict) -> nn.Module:
    homeo_type = params['homeo_type']
    
    if homeo_type == 'iresnet':
        cls = iResNet
        allowed_keys = {'dim', 'layer_sizes', 'init_type', 'activation', 'init_std', 'init_mean'}
        filtered_args = {k: v for k, v in params.items() if k in allowed_keys}
        return cls(**filtered_args)

    elif homeo_type == 'node':
        cls = NODEHomeomorphism
        allowed_keys = {'dim', 'layer_sizes', 'init_type', 'activation', 'init_std', 'init_mean', 'scale'}
        filtered_args = {k: v for k, v in params.items() if k in allowed_keys}
        return cls(**filtered_args)

    elif homeo_type == 'affine_node':
        # Split the flat params into those for the NODE and for AffineAfterNODE
        node_keys = {'dim', 'layer_sizes', 'init_type', 'activation', 'init_std', 'init_mean', 'scale'}
        affine_keys = {'dim', 'learnable_affine'}

        node_args = {k: v for k, v in params.items() if k in node_keys}
        affine_args = {k: v for k, v in params.items() if k in affine_keys}

        node = NODEHomeomorphism(**node_args)
        return AffineAfterNODE(node=node, **affine_args)

    else:
        raise ValueError(f"Unknown architecture: {homeo_type}")




############## testing homeomorphism-dynamical system networks
def test_single_homeo_ds_net(
    homeo_ds_net: nn.Module,
    trajectories_target: torch.Tensor,
    time_span: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Test a single homeo_ds_net by mapping target trajectories back, regenerating them, and comparing.

    Returns:
        - np array of source trajectories
        - np array of transformed trajectories
        - scalar MSE loss
    """
    loss_fn = nn.MSELoss(reduction='mean')
    num_points = trajectories_target.shape[0]
    source_system = homeo_ds_net.dynamical_system
    homeo_net = homeo_ds_net.homeo_network

    if time_span is None:
        time_span = homeo_ds_net.dynamical_system.time_span

    # Get initial conditions from first n trajectories
    initial_conditions_target = trajectories_target[:,0, :].clone().detach().requires_grad_(True)

    # Transform to source domain
    initial_conditions_source = homeo_ds_net.homeo_network.inverse(initial_conditions_target)
    #transformed_trajectories = homeo_ds_net(initial_conditions_target)

    # # Generate source trajectories
    if isinstance(source_system, AnalyticDynamicalSystem):
        trajectories_source = source_system.compute_trajectory(initial_conditions_source, time_span=time_span)
    else:
        _, trajectories_source, _ = generate_trajectories(source_system, predefined_initial_conditions=initial_conditions_source, time_span=time_span)
    
    transformed_trajectories = homeo_net(trajectories_source)

    # Compute loss
    loss = sum(loss_fn(x_t, phi_y_t) for x_t, phi_y_t in zip(trajectories_target, transformed_trajectories)) / num_points

    trajectories_source_np = np.array([traj.detach().cpu().numpy() for traj in trajectories_source])
    transformed_trajectories_np = np.array([traj.detach().cpu().numpy() for traj in transformed_trajectories])
    return trajectories_source_np, transformed_trajectories_np, loss.item()


def test_homeo_ds_nets(
    trajectories_target: List[torch.Tensor],
    homeo_ds_nets: List[nn.Module], 
    plot_first_n: int = 5,
    time_span: Optional[torch.Tensor] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Evaluate each homeo_ds_net on the target trajectories.
    """
    trajectories_source_list = []
    transformed_trajectories_list = []
    losses = []

    for homeo_ds_net in homeo_ds_nets:
        traj_src_np, traj_trans_np, loss = test_single_homeo_ds_net(
            homeo_ds_net,
            trajectories_target=trajectories_target,
            time_span=time_span
        )
        trajectories_source_list.append(traj_src_np)
        transformed_trajectories_list.append(traj_trans_np)
        losses.append(loss)

    return trajectories_source_list, transformed_trajectories_list, losses



### JACOBIAN ###
from typing import Callable, Literal

def jacobian_norm_over_batch(
    phi: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    p: float = 2.0,
    norm_type: Literal["fro", "spectral"] = "spectral",
    normalize: bool = True,
    subtract_identity: bool = True,
    inverse : bool = False,
) -> torch.Tensor:
    """
    Computes empirical L^p norm of Jacobian norms of phi over a batch of points.

    Args:
        phi: R^n -> R^m function
        x: Tensor of shape (B, D): batch of input points
        p: Exponent in L^p norm
        norm_type: 'fro' or 'spectral'
        normalize: If True, divide each norm by sqrt(D)
        subtract_identity: If True, compute ||J_phi(x) - I|| instead of ||J_phi(x)||

    Returns:
        Scalar tensor: empirical L^p norm of the chosen Jacobian expression across x in batch
    """
    B, D = x.shape
    print(f"Computing Jacobian norms ({norm_type})...")
    start = time.time()

    def compute_norm(xi: torch.Tensor) -> torch.Tensor:
        xi = xi.detach().unsqueeze(0).requires_grad_(True)  # shape (1, D)
        if inverse:
            xi = phi.inverse(xi)
        else:
            y = phi(xi).squeeze(0)  # shape (D_out,)
        assert y.ndim == 1, "phi(x) must return a 1D output (R^m)"
        grads = [torch.autograd.grad(y[i], xi, retain_graph=True, create_graph=False)[0].squeeze(0) for i in range(y.shape[0])]
        J = torch.stack(grads)  # shape (m, n)

        if subtract_identity:
            if J.shape[0] != J.shape[1]:
                raise ValueError("Cannot subtract identity from a non-square Jacobian.")
            J = J - torch.eye(J.shape[0], device=J.device, dtype=J.dtype)

        if norm_type == "fro":
            norm = torch.norm(J, p='fro')
        elif norm_type == "spectral":
            norm = torch.linalg.svdvals(J)[0]
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        return norm / D**0.5 if normalize else norm

    norms = torch.stack([compute_norm(x[i]) for i in range(B)])
    end = time.time()
    #print("Time elapsed for Jacobian: ", end - start)

    return norms.pow(p).mean().pow(1 / p)


def jacobian_spectral_norm(
    phi: Callable,
    x: torch.Tensor,
    p: float = 2.0,
    n_iter: int = 10,
    subtract_identity: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Approximate the L^p norm of spectral norms of J_phi(x) using power iteration.

    Args:
        phi: A function from R^N to R^M (batched).
        x: Tensor of shape (B, T, N) — batch of trajectories.
        p: Exponent for L^p norm across (B*T) samples.
        n_iter: Number of power iterations.
        subtract_identity: If True, estimate ||J - I|| instead of ||J||.
        normalize: If True, divide each norm by sqrt(N).

    Returns:
        Scalar tensor: empirical L^p norm of the estimated spectral norms.
    """
    B, T, N = x.shape
    x_flat = x.view(-1, N).detach().requires_grad_(True)  # (B*T, N)

    # Initialize random direction v
    v = torch.randn_like(x_flat)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-6)

    for _ in range(n_iter):
        # Compute Jv
        _, jv = torch.autograd.functional.jvp(phi, (x_flat,), (v,), create_graph=True)


        u = jv / (jv.norm(dim=1, keepdim=True) + 1e-6)

        # Compute J^T u
        jtv = torch.autograd.grad(
            outputs=phi(x_flat),
            inputs=x_flat,
            grad_outputs=u,
            create_graph=True,
            retain_graph=True
        )[0]

        v = jtv / (jtv.norm(dim=1, keepdim=True) + 1e-6)

    # Final spectral norm approximation: ||Jv||
    _, jv = torch.autograd.functional.jvp(phi, (x_flat,), (v,), create_graph=True)
    if subtract_identity:
        jv = jv - v
    norms = jv.norm(dim=1)  # shape: (B*T,)

    if normalize:
        norms = norms / N**0.5

    return norms.pow(p).mean().pow(1.0 / p)




def jacobian_frobenius_norm(
    phi: Callable, 
    x: torch.Tensor, 
    p: float = 2.0, 
    subtract_identity: bool = True,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute L^p norm of Frobenius norms of Jacobians over a batch of trajectory points.

    Args:
        phi: A function from R^N to R^M (batched).
        x: Tensor of shape (B, T, N).
        p: Exponent in L^p norm.
        subtract_identity: Whether to compute ||J - I|| instead of ||J||.
        normalize: If True, divide each Frobenius norm by sqrt(N).

    Returns:
        Scalar tensor: empirical L^p norm of the Frobenius norms.
    """
    B, T, N = x.shape
    x_flat = x.view(-1, N).detach().requires_grad_(True)
    fro_sq = 0.0

    for i in range(N):
        v = torch.zeros_like(x_flat)
        v[:, i] = 1.0
        _, jvp_result = torch.autograd.functional.jvp(phi, (x_flat,), (v,), create_graph=True)
        if subtract_identity:
            jvp_result = jvp_result - v
        fro_sq += (jvp_result ** 2).sum(dim=1)  # ||column_i||^2

    fro_norms = fro_sq.sqrt()  # shape: (B*T,)
    if normalize:
        fro_norms = fro_norms / N**0.5

    return fro_norms.pow(p).mean().pow(1.0 / p)





############
def transform_vector_field(homeo_ds_net, xlim=(-2, 2), ylim=(-2, 2), num_points=15):
    """
    Transforms the vector field using the homeomorphism and its Jacobian.
    homeo_net: trained homeomorphism network (function `h(x)`)
    dynamical_system: the dynamical system defining the vector field `g(x)`
    grid_points: points at which to evaluate the vector field
    """
        # Create aligned grid
    x = np.linspace(xlim[0], xlim[1], num_points)
    y = np.linspace(ylim[0], ylim[1], num_points)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    dynamical_system = homeo_ds_net.dynamical_system
    homeo_net = homeo_ds_net.homeo_network

    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # 1. Get the vector field `g(x)` at the points in the source space
    vector_field = dynamical_system.forward(None, grid_tensor)  # <--- FIXED

    # 2. Apply the homeomorphism to the grid points
    transformed_points = homeo_net.forward(grid_tensor)
    jacobian = jacobian_matrix_pointwise(homeo_net.forward, grid_tensor)  # shape (batch_size, 2, 2)

    # Now invert each Jacobian matrix
    jacobian_inv = torch.linalg.inv(jacobian)  # still (batch_size, 2, 2)

    # vector_field is (batch_size, 2)
    # Apply inverse Jacobian to vector_field pointwise
    transformed_vector_field = torch.bmm(jacobian_inv, vector_field.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)

    return transformed_points, transformed_vector_field


def interpolate_node_homeomorphisms(model_1: NODEHomeomorphism, model_2: NODEHomeomorphism, alpha: float) -> NODEHomeomorphism:
    """Interpolates between two NODEHomeomorphisms using the given alpha."""
    
    # Ensure both models have the same dimension and layer sizes
    assert model_1.dim == model_2.dim
    assert model_1.layer_sizes == model_2.layer_sizes

    # Create a new NODEHomeomorphism to hold the interpolated model
    interpolated_model = NODEHomeomorphism(
        dim=model_1.dim, 
        layer_sizes=model_1.layer_sizes,
        activation=model_1.neural_ode.mlp[1].__class__,  # Same activation function
        t_span=model_1.t_span.tolist()  # Same time span
    )
    
    # Interpolate weights and biases of the neural ODE layers
    for layer_1, layer_2, layer_interp in zip(model_1.neural_ode.mlp, model_2.neural_ode.mlp, interpolated_model.neural_ode.mlp):
        if isinstance(layer_1, nn.Linear):  # Check if it's a Linear layer
            # Interpolate weights
            with torch.no_grad():
                layer_interp.weight.data = (1 - alpha) * layer_1.weight.data + alpha * layer_2.weight.data
                layer_interp.bias.data = (1 - alpha) * layer_1.bias.data + alpha * layer_2.bias.data

    return interpolated_model



def rescale_node_vf(model: NODEHomeomorphism, scale: float) -> NODEHomeomorphism:
    """Returns a new NODEHomeomorphism with the final layer of the vector field scaled by a scalar."""
    
    # Clone model config
    rescaled_model = NODEHomeomorphism(
        dim=model.dim,
        layer_sizes=model.layer_sizes,
        activation=model.neural_ode.mlp[1].__class__,
        t_span=model.t_span.tolist()
    )

    # Copy parameters from model
    for src_layer, dst_layer in zip(model.neural_ode.mlp, rescaled_model.neural_ode.mlp):
        if isinstance(src_layer, nn.Linear):
            with torch.no_grad():
                dst_layer.weight.data = src_layer.weight.data.clone()
                dst_layer.bias.data = src_layer.bias.data.clone()

    # Rescale only the last linear layer
    for layer in reversed(rescaled_model.neural_ode.mlp):
        if isinstance(layer, nn.Linear):
            with torch.no_grad():
                layer.weight.data.mul_(scale)
                layer.bias.data.mul_(scale)
            break  # only rescale the final Linear layer

    return rescaled_model



### testing 
def get_homeo_invman(homeo_network, dim: int = 2, num_points: int = 100) -> np.ndarray:
    """
    Generates points on a 2D ring embedded in the first two dimensions of a higher-dimensional space,
    applies the homeomorphism, and returns the transformed ring points.

    Args:
        homeo_network: A PyTorch model representing the homeomorphism.
        dim: Dimensionality of the input space (>=2).
        num_points: Number of points to sample around the ring.

    Returns:
        Transformed ring points as a NumPy array of shape (num_points + 1, dim).
    """
    assert dim >= 2, "Dimension must be at least 2"

    # Sample angles around the unit circle
    radius = 1.0
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    ring_2d = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    # Embed in higher-dimensional space by padding with zeros
    ring_nd = np.zeros((num_points, dim), dtype=np.float32)
    ring_nd[:, :2] = ring_2d

    # Apply homeomorphism
    fit_nd = homeo_network(torch.tensor(ring_nd)).detach().numpy()

    # Close the ring by appending the first point at the end
    fit_nd = np.vstack([fit_nd, fit_nd[0]])

    return fit_nd



#######Link
class AffineAfterNODE(nn.Module):
    def __init__(self, dim: int, node: NODEHomeomorphism, learnable_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.node = node
        self.affine = AffineTransform(dim, learnable=learnable_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.node(x)
        return self.affine(z)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        z = self._inverse_affine(y)
        return self.node.inverse(z)

    def _inverse_affine(self, y: torch.Tensor) -> torch.Tensor:
        W_inv = torch.inverse(self.affine.W)
        return (y - self.affine.b) @ W_inv.T




class Homeo_DS_Net(nn.Module):
    def __init__(self, homeo_network: nn.Module, dynamical_system: nn.Module):
        """
        A class to link a homeomorphism network and a learnable dynamical system.
        
        :param homeo_network: The homeomorphism network module (e.g., a feed-forward neural network).
        :param dynamical_system: The learnable dynamical system (e.g., LearnableDynamicalSystem).
        """
        super(Homeo_DS_Net, self).__init__()
        self.homeo_network = homeo_network
        self.dynamical_system = dynamical_system

    def forward(self, y: torch.Tensor, noise_std: float=0) -> torch.Tensor:
        """Apply inverse homeomorphism to map y to source, then evolve, then re-apply homeo."""
        x0 = self.homeo_network.inverse(y)
        if isinstance(self.dynamical_system, AnalyticDynamicalSystem):
            traj = self.dynamical_system.compute_trajectory(x0)
        else:
            _, traj, _ = generate_trajectories(self.dynamical_system, predefined_initial_conditions=x0, noise_std=noise_std)
        return traj, self.homeo_network(traj) 

    def trajectory(self, y0: torch.Tensor, noise_std: float=0) -> list[torch.Tensor]:
        return self.forward(y0, noise_std)

    def invariant_manifold(self, num_points: int = 100) -> np.ndarray:
        """
        Generate points on the invariant manifold of the homeomorphism.
        
        :param num_points: Number of points to sample on the manifold.
        :return: Transformed points on the invariant manifold.
        """
        return self.homeo_network(self.dynamical_system.invariant_manifold(num_points))



def save_homeo_ds_net(model: Homeo_DS_Net, file_path: str):
    """
    Save the Homeo_DS_Net model to a file.
    
    :param model: The Homeo_DS_Net model to save.
    :param file_path: Path to the file where the model should be saved.
    """
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def load_homeo_ds_net(file_path: str, homeo_network: nn.Module, dynamical_system: nn.Module) -> Homeo_DS_Net:
    """
    Load a Diffeo_DS_Net model from a file.

    :param file_path: Path to the file from which to load the model.
    :param diffeo_network: The diffeomorphism network architecture.
    :param dynamical_system: The source dynamical system.
    :return: A Diffeo_DS_Net model with loaded weights.
    """
    model = Homeo_DS_Net(homeo_network, dynamical_system)
    state_dict = torch.load(file_path)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {file_path}")
    return model


def save_diffeo_ds_net_compact(model: nn.Module, file_path: str, meta: dict) -> None:
    torch.save({
        'state_dict': model.state_dict(),
        'meta': meta  # this should include homeo_params and ds_params
    }, file_path)
    print(f"Model and metadata saved to {file_path}")

def load_diffeo_ds_net_compact(file_path: str) -> Homeo_DS_Net:
    checkpoint = torch.load(file_path)

    homeo_params = checkpoint['meta']['homeo_params']
    ds_params = checkpoint['meta']['ds_params']

    diffeo_network = build_diffeomorphism(homeo_params)
    dynamical_system = build_ds_motif(**ds_params)

    model = Diffeo_DS_Net(diffeo_network, dynamical_system)
    model.load_state_dict(checkpoint['state_dict'])

    print(f"Model and metadata loaded from {file_path}")
    return model
