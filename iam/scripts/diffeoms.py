import numpy as np
import itertools
import torch
import torch.nn as nn
from typing import Callable, Tuple, List


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

class DiffeomorphismNetwork(nn.Module):
    """Represents a diffeomorphic transformation close to the identity.
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
    




# Function to generate diffeomorphism with grid points
def generate_random_diffeomorphism(dim: int, num_samples: int = 10, epsilon: float = 0.01, grid_points: bool = False, bounds: tuple = (-2,2), activation: str= 'tanh') -> torch.Tensor:
    """
    Generate diffeomorphisms of the form Phi(x) = x + epsilon * H(x) where epsilon is small.
    Can visualize with grid points or random points.
    """
    diffeomorphism_network = DiffeomorphismNetwork(dim=dim, epsilon=epsilon,    activation=activation)

    # If grid_points is True, generate grid of points
    if grid_points:
        x_vals = np.linspace(bounds[0], bounds[1], num_samples)
        y_vals = np.linspace(bounds[0], bounds[1], num_samples)
        grid_points = np.array(np.meshgrid(x_vals, y_vals)).T.reshape(-1, 2)
        random_samples = torch.tensor(grid_points, dtype=torch.float32)
    else:
        # Otherwise, generate random points
        random_samples = torch.randn(num_samples, dim)  # Random samples from standard normal

    # Apply the diffeomorphism (identity + epsilon * H(x))
    transformed_samples = diffeomorphism_network(random_samples)

    return diffeomorphism_network, random_samples, transformed_samples







def test_diffeom_networks(
    trajectories_target: List[torch.Tensor],
    motif_library: List,
    diffeo_networks: List[torch.nn.Module],
    generate_trajectories_scipy,
    num_points: int,
    plot_first_n: int = 5
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Transforms target trajectories back to the source domain using inverse diffeomorphisms,
    generates source trajectories, and maps them forward again for comparison.

    Args:
        trajectories_target: List of target domain trajectories (each a torch.Tensor).
        motif_library: List of systems/motifs used to generate source trajectories.
        diffeo_networks: List of diffeomorphism networks (with inverse method).
        generate_trajectories_scipy: Function to generate trajectories for a given system.
        num_points: Total number of points/trajectories available.
        plot_first_n: Number of trajectories to process (default is 5).

    Returns:
        Tuple of two lists:
            - trajectories_source_list: Source domain trajectories.
            - transformed_trajectories_list: Re-transformed trajectories using diffeomorphisms.
    """
    # Get initial conditions from target trajectories
    initial_conditions_np = np.array([
        trajectories_target[i][0].detach().numpy() for i in range(num_points)
    ])
    initial_conditions = torch.tensor(initial_conditions_np, dtype=torch.float32)[:plot_first_n, :]

    trajectories_source_list = []
    transformed_trajectories_list = []

    for motif, diffeo_net in zip(motif_library, diffeo_networks):
        # Transform initial conditions to source space
        initial_conditions_src = diffeo_net.inverse(initial_conditions)

        # Generate source trajectories
        t_values, trajectories_source, _ = generate_trajectories_scipy(
            system=motif,
            predefined_initial_conditions=initial_conditions_src
        )

        # Map trajectories back to target space using diffeomorphism
        with torch.no_grad():
            transformed_trajectories = np.array([
                diffeo_net(traj.requires_grad_()).detach().numpy() for traj in trajectories_source
            ])

        # Convert source trajectories to numpy
        trajectories_source_np = np.array([
            traj.requires_grad_().detach().numpy() for traj in trajectories_source
        ])

        trajectories_source_list.append(trajectories_source_np)
        transformed_trajectories_list.append(transformed_trajectories)

    return trajectories_source_list, transformed_trajectories_list