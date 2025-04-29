import numpy as np
import itertools
import torch
import torch.nn as nn
from typing import Callable, Tuple, List
from .irevnet import *

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
    """Represents a homeomorphic transformation close to the identity.
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


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class iResNet(nn.Module):
    def __init__(self, dim: int, n_blocks: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(dim, hidden_dim) for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, y: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
        """
        Fixed-point iteration for inverting the residual network.
        Assumes that each block is contractive (Lipschitz constant < 1).
        """
        x = y.clone()
        for _ in range(max_iter):
            for block in reversed(self.blocks):
                x = x - (block(x) - x)
        return x
    




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




#####NORMALIZING FLOWS#####
import normflows as nf

class NormFlowHomeomorphism(nn.Module):
    def __init__(self, dim: int = 2, layer_sizes: list[int] = [64, 64], num_layers: int = 32):
        super().__init__()
        self.dim = dim
        self.flow = self._build_flow(dim, layer_sizes, num_layers)

    def _build_flow(self, dim: int, layer_sizes: list[int], num_layers: int):
        base = nf.distributions.base.DiagGaussian(dim)
        flows = []
        for _ in range(num_layers):
            param_map = nf.nets.MLP([dim // 2] + layer_sizes + [dim], init_zeros=True)
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            flows.append(nf.flows.Permute(dim, mode='swap'))
        return nf.NormalizingFlow(base, flows)

    def _flatten_time(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, x.shape[-1])

    def _restore_time(self, x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        return x.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map from source system to target space (x -> y)."""
        original_shape = x.shape
        x_flat = self._flatten_time(x)
        y_flat = self.flow.forward(x_flat)  # Returns (z, log_det); we discard log_det
        #y_flat = torch.clamp(y_flat, -10, 10)
        return self._restore_time(y_flat, original_shape)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Map from target space to source system (y -> x)."""
        original_shape = y.shape
        y_flat = self._flatten_time(y)
        x_flat = self.flow.inverse(y_flat)
        #x_flat = torch.clamp(x_flat, -10, 10)
        return self._restore_time(x_flat, original_shape)


#### NODEs
import torchdiffeq

import torch
import torch.nn as nn
from typing import Callable

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
            layers.append(activation())  # <-- use the passed activation function
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, dim))  # Final layer back to dim
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class NODEHomeomorphism(nn.Module):
    """Represents a homeomorphic transformation using a Neural ODE."""
    def __init__(self, dim: int, layer_sizes: list[int] = [64, 64],
                activation: Callable[[], nn.Module] = nn.Tanh,
                t_span: tuple = (0.0, 1.0)) -> None:
        super().__init__()
        self.t_span = torch.tensor(t_span)  # Default integration time span
        self.neural_ode = NeuralODE(dim, layer_sizes, activation=activation)

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
        x.requires_grad_(True)
        output = self.forward(x)

        jacobian = []
        for i in range(output.shape[1]):
            grad_output = torch.zeros_like(output)
            grad_output[:, i] = 1
            grad_i = torch.autograd.grad(output, x, grad_outputs=grad_output, create_graph=False)[0]
            jacobian.append(grad_i.unsqueeze(1))
        return torch.cat(jacobian, dim=1)

    # def inverse_approximation(self, x: torch.Tensor, steps: int = 5) -> torch.Tensor:
    #     y = x.clone()
    #     for _ in range(steps):
    #         y = x - self.forward(y) + y
    #     return y



###Invertible Residual Networks (iResNet)###
class InvertibleResNetBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(InvertibleResNetBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels // 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, in_channels // 2)
        self.activation = nn.ELU()  # invertible activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        z = self.activation(self.fc1(x1))
        x1_out = x1 + self.fc2(z)
        return torch.cat([x1_out, x2], dim=1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = y.chunk(2, dim=1)
        z = self.activation(self.fc1(y1))
        y1_inv = y1 - self.fc2(z)
        return torch.cat([y1_inv, y2], dim=1)

class InvertibleResNet(nn.Module):
    def __init__(self, in_channels: int, layer_sizes: List[int]):
        super(InvertibleResNet, self).__init__()
        self.blocks = nn.ModuleList([
            InvertibleResNetBlock(in_channels, hidden_size)
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






############## testing homeomorphisms ##############
def test_homeo_networks(
    trajectories_target: torch.Tensor,
    homeo_ds_networks: List[Homeo_DS_Net], 
    plot_first_n: int = 5,
    time_span: torch.Tensor=None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Transforms target trajectories back to the source domain using inverse homeomorphisms,
    generates source trajectories, and maps them forward again for comparison.

    Args:
        trajectories_target: List of target domain trajectories (each a torch.Tensor).
        motif_library: List of systems/motifs used to generate source trajectories.
        homeo_networks: List of homeomorphism networks 
        num_points: Total number of points/trajectories available.
        plot_first_n: Number of trajectories to process (default is 5).

    Returns:
        Tuple of two lists:
            - trajectories_source_list: Source domain trajectories.
            - transformed_trajectories_list: Re-transformed trajectories using homeomorphisms.
    """
    # Get initial conditions from target trajectories


    initial_conditions = torch.stack([traj[0] for traj in trajectories_target[:plot_first_n]])

    trajectories_source_list = []
    transformed_trajectories_list = []

    for homeo_ds_net in homeo_ds_networks:
        source_system = homeo_ds_net.dynamical_system
        homeo_net = homeo_ds_net.homeo_network

        if time_span is None:
            time_span = source_system.time_span
        # Transform initial conditions to source space
        initial_conditions_src = homeo_net.inverse(initial_conditions)

        # Generate source trajectories
        if isinstance(source_system, AnalyticalLimitCycle):
            trajectories_source = source_system.compute_trajectory(initial_conditions_src, time_span=time_span)
        else:
            t_values, trajectories_source, _ = generate_trajectories_scipy(
                system=source_system,
                predefined_initial_conditions=initial_conditions_src,
                time_span=time_span
            )

        # Map trajectories back to target space using homeomorphism
        with torch.no_grad():
            transformed_trajectories = np.array([homeo_net(traj).detach().numpy() for traj in trajectories_source])
        # Convert source trajectories to numpy
        trajectories_source_np = np.array([traj.detach().numpy() for traj in trajectories_source])

        trajectories_source_list.append(trajectories_source_np)
        transformed_trajectories_list.append(transformed_trajectories)

    transformed = torch.stack([
        torch.tensor(traj, dtype=torch.float32) for traj in transformed_trajectories_list
    ])
    target = torch.stack([
        traj if isinstance(traj, torch.Tensor) else torch.tensor(traj, dtype=torch.float32)
        for traj in trajectories_target[:len(transformed_trajectories_list)]
    ])

    loss_fn = nn.MSELoss(reduction='mean')
    loss = loss_fn(transformed, target)
    return trajectories_source_list, transformed_trajectories_list, loss.item()



###
def jacobian_matrix(h, x):
    """
    Computes the Jacobian of the homeomorphism `h(x)` at the batch of points `x`.
    h: homeomorphism function
    x: tensor of shape (batch_size, dim)
    Returns:
        jacobian: (batch_size, output_dim, input_dim)
    """
    x = x.requires_grad_(True)
    def h_sum(x):
        return h(x)

    jac = torch.autograd.functional.jacobian(h_sum, x, create_graph=False)
    # jac shape: (batch_size, output_dim, input_dim)
    return jac

def jacobian_matrix_pointwise(h, x_batch):
    """
    Computes the Jacobians of h(x) for a batch of points x_batch.
    Returns a tensor of shape (batch_size, dim_out, dim_in).
    """
    batch_size, input_dim = x_batch.shape
    jacobians = []
    for i in range(batch_size):
        x = x_batch[i].unsqueeze(0).detach().requires_grad_(True)
        y = h(x)  # shape (1, output_dim)

        jac = []
        for j in range(y.shape[1]):
            grad = torch.autograd.grad(y[0, j], x, retain_graph=True, create_graph=False)[0]
            jac.append(grad.squeeze(0))  # shape (input_dim,)
        jac = torch.stack(jac, dim=0)  # shape (output_dim, input_dim)
        jacobians.append(jac)

    jacobians = torch.stack(jacobians, dim=0)  # (batch_size, output_dim, input_dim)
    return jacobians


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


#Link
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



def save_homeo_ds_net(model: Homeo_DS_Net, file_path: str):
    """
    Save the Homeo_DS_Net model to a file.
    
    :param model: The Homeo_DS_Net model to save.
    :param file_path: Path to the file where the model should be saved.
    """
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def load_homeo_ds_net(homeo_network: nn.Module, dynamical_system: nn.Module, file_path: str) -> Homeo_DS_Net:
    """
    Load the Homeo_DS_Net model from a saved file.
    
    :param homeo_network: The homeomorphism network model to use.
    :param dynamical_system: The dynamical system model to use.
    :param file_path: Path to the file where the model is saved.
    :return: The loaded Homeo_DS_Net model.
    """
    model = Homeo_DS_Net(homeo_network=homeo_network, dynamical_system=dynamical_system)
    model.load_state_dict(torch.load(file_path))
    print(f"Model loaded from {file_path}")
    return model


