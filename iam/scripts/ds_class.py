import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Tuple
from torchdiffeq import odeint
from scipy.integrate import solve_ivp

class DynamicalSystem(nn.Module):
    """
    Base class for a dynamical system. 
    Any subclass should implement the `forward` method defining dx/dt = f(t, x).
    """
    def __init__(self, dim: int = 2, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5), noise_std: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.time_span = time_span
        self.noise_std = noise_std

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the time derivative dx/dt.
        
        :param t: Scalar tensor representing time.
        :param x: Tensor representing the system state, shape (batch_size, state_dim).
        :return: Tensor of shape (batch_size, state_dim) representing dx/dt.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class LearnableDynamicalSystem(nn.Module):
    """
    A base class for dynamical systems with learnable parameters.
    Subclasses should define the specific system dynamics.
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes the time derivatives for a given state.
        This should be overridden by subclasses to define specific dynamics.
        """
        raise NotImplementedError


    
class LimitCycle(DynamicalSystem):
    """
    A simple limit cycle system with dynamics defined in polar coordinates.
    """

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension
        
        x_val, y_val = x[:, 0], x[:, 1]
        r = torch.sqrt(x_val**2 + y_val**2)
        theta = torch.atan2(y_val, x_val)
        
        dx_dt = -r * ((r - 1) * torch.cos(theta) - torch.sin(theta))
        dy_dt = -r * ((r - 1) * torch.sin(theta) + torch.cos(theta))
        
        return torch.stack([dx_dt, dy_dt], dim=1)

class NDLimitCycle(DynamicalSystem):
    """
    N-dimensional limit cycle system.
    The first two coordinates define a planar limit cycle.
    All remaining dimensions are attracted toward the 2D plane.
    """

    def __init__(self, dim: int, attraction_rate: float = 1.0):
        super().__init__()
        assert dim >= 2, "Dimension must be at least 2."
        self.dim = dim
        self.attraction_rate = attraction_rate

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension

        x1, x2 = x[:, 0], x[:, 1]
        r = torch.sqrt(x1**2 + x2**2)
        theta = torch.atan2(x2, x1)

        dx1_dt = -r * ((r - 1) * torch.cos(theta) - torch.sin(theta))
        dx2_dt = -r * ((r - 1) * torch.sin(theta) + torch.cos(theta))

        dx_dt = [dx1_dt, dx2_dt]

        # Remaining dimensions flow toward 0 (attractive dynamics)
        if self.dim > 2:
            residual = x[:, 2:]
            d_residual_dt = -self.attraction_rate * residual
            dx_dt.append(d_residual_dt)

        return torch.cat(dx_dt, dim=1)

class LearnableLimitCycle(LearnableDynamicalSystem):
    """
    A simple limit cycle system with dynamics defined in polar coordinates.
    The system includes learnable speed and lambda parameters.
    """
    def __init__(self, dim: int = 2, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5), noise_std: float = 0.0, speed_init: float = 1.0, lambda_init: float = 1.0):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.time_span = time_span
        self.noise_std = noise_std
        # Initialize learnable parameters for speed and lambda
        self.speed = nn.Parameter(torch.tensor(speed_init, dtype=torch.float32))
        self.lambda_ = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension
        
        # Extract the polar coordinates
        x_val, y_val = x[:, 0], x[:, 1]
        r = torch.sqrt(x_val**2 + y_val**2)
        theta = torch.atan2(y_val, x_val)

        # Define the dynamics in polar coordinates
        dtheta_dt = self.speed  # Constant speed
        dr_dt = - self.lambda_ * r * (1 - r)  #  radial dynamics
        
        # Convert the dynamics to Euclidean coordinates
        dx_dt = dr_dt * torch.cos(theta) - r * torch.sin(theta) * dtheta_dt
        dy_dt = dr_dt * torch.sin(theta) + r * torch.cos(theta) * dtheta_dt
        
        return torch.stack([dx_dt, dy_dt], dim=1)
    

class RingAttractor(DynamicalSystem):
    """
    A simple ring attractor system with dynamics defined in polar coordinates.
    """

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension
        
        x_val, y_val = x[:, 0], x[:, 1]
        r = torch.sqrt(x_val**2 + y_val**2)
        theta = torch.atan2(y_val, x_val)
        
        dx_dt = -r * ((r - 1) * torch.cos(theta))
        dy_dt = -r * ((r - 1) * torch.sin(theta))
        
        return torch.stack([dx_dt, dy_dt], dim=1)


class LinearSystem(DynamicalSystem):
    """
    A linear dynamical system of the form dx/dt = Ax.
    """

    def __init__(self, A: torch.Tensor, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5)):
        """
        :param A: Square matrix defining the linear system.
        """
        super().__init__()
        self.A = A
        self.time_span = time_span
        self.dt = dt

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.A.T)  # Ensure correct matrix multiplication

class BoundedLineAttractor(DynamicalSystem):
    """
    A nonlinear dynamical system of the form dx/dt = -x + ReLU(Wx + b).
    """

    def __init__(self, W: torch.Tensor, b: torch.Tensor, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5)):
        """
        :param W: Weight matrix.
        :param b: Bias vector.
        """
        super().__init__()
        self.W = W
        self.b = b
        self.relu = nn.ReLU()
        self.time_span = time_span
        self.dt = dt

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        linear_part = torch.matmul(x, self.W.T) + self.b
        return -x + self.relu(linear_part)

class NonlinearSystem(DynamicalSystem):
    """
    A general nonlinear system where the dynamics are defined by an arbitrary function.
    """

    def __init__(self, f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], dt: float = 0.05, time_span: Tuple[float, float] = (0, 5)):
        """
        :param f: Function that computes dx/dt given (t, x).
        """
        super().__init__()
        self.f = f
        self.time_span = time_span
        self.dt = dt

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.f(t, x)


class PhiSystemPhiInv(DynamicalSystem):
    """
    A transformed dynamical system where the action of the diffeomorphism is applied to the vector field of the base system. 
    """
    def __init__(self, base_system: nn.Module, diffeomorphism_network: nn.Module) -> None:
        super().__init__()
        self.base_system = base_system
        self.diffeomorphism_network = diffeomorphism_network

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure x is at least (1, dim)

        transformed_x = self.diffeomorphism_network.inverse(x)
        dx_dt_transformed = self.base_system(t, transformed_x)
        dx_dt_transformed = self.diffeomorphism_network(dx_dt_transformed)
        return dx_dt_transformed.squeeze(0)  # Restore original shape if needed
    
class PhiInvSystemPhi(DynamicalSystem):
    """
    A transformed dynamical system where the inverse action of the diffeomorphism is applied to the vector field of the base system. 
    """
    def __init__(self, base_system: nn.Module, diffeomorphism_network: nn.Module) -> None:
        super().__init__()
        self.base_system = base_system
        self.diffeomorphism_network = diffeomorphism_network

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure x is at least (1, dim)

        transformed_x = self.diffeomorphism_network(x)
        dx_dt_transformed = self.base_system(t, transformed_x)
        dx_dt_transformed = self.diffeomorphism_network.inverse(dx_dt_transformed)
        return dx_dt_transformed.squeeze(0)  # Restore original shape if needed

class PhiSystem(DynamicalSystem):
    """
    A transformed dynamical system where the diffeomorphism is applied to the vector field of the base system. 
    """
    def __init__(self, base_system: nn.Module, diffeomorphism_network: nn.Module) -> None:
        super().__init__()
        self.base_system = base_system
        self.diffeomorphism_network = diffeomorphism_network

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure x is at least (1, dim)

        dx_dt_transformed = self.base_system(t, x)
        dx_dt_transformed = self.diffeomorphism_network(dx_dt_transformed)
        return dx_dt_transformed.squeeze(0)  # Restore original shape if needed
    

class PhiJacSystem(DynamicalSystem):

    def __init__(self, base_system: nn.Module, diffeomorphism_network: nn.Module) -> None:
        super().__init__()
        self.base_system = base_system
        self.diffeomorphism_network = diffeomorphism_network

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure x is at least (1, dim)

        dx_dt_transformed = self.base_system(t, x)
        jacobian = self.diffeomorphism_network.jacobian(dx_dt_transformed)
        dx_dt_transformed = torch.matmul(jacobian, dx_dt_transformed.unsqueeze(-1)).squeeze(-1)
        return dx_dt_transformed.squeeze(0)  # Restore original shape if needed
    

#Target systems for testing

#2D systems
# Van der Pol oscillator as an example target system
class VanDerPol(DynamicalSystem):
    def __init__(self, mu: float = 1.0, dim: int = 2, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5), noise_std: float=0.) -> None:
        super().__init__(dim=dim, dt=dt, time_span=time_span)
        self.mu = mu
        self.noise_std = noise_std
        self.time_span = time_span
        self.dt = dt

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Ensure that x is a 2D tensor, even if batch_size = 1
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add a batch dimension if it's missing

        dxdt = torch.zeros_like(x)
        dxdt[:, 0] = x[:, 1]
        dxdt[:, 1] = self.mu * (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0]
        return dxdt


class FitzHughNagumo(DynamicalSystem):
    def __init__(self, a: float = 0.7, b: float = 0.8, c: float = 0.8, dim: int = 2, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5), noise_std: float = 0.0) -> None:
        super().__init__(dim=dim, dt=dt, time_span=time_span)
        self.a = a
        self.b = b
        self.c = c
        self.noise_std = noise_std
        self.time_span = time_span
        self.dt = dt

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Ensure that x is a 2D tensor, even if batch_size = 1
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add a batch dimension if it's missing
        
        # Initialize the derivative tensor
        dxdt = torch.zeros_like(x)

        # FitzHugh-Nagumo equations
        dxdt[:, 0] = self.c * (x[:, 0] - x[:, 0] ** 3 / 3 - x[:, 1])  # dx/dt
        dxdt[:, 1] = - (x[:, 0] - self.a + self.b * x[:, 1]) / self.c  # dy/dt

        return dxdt


#3D systems
class LorenzSystem(DynamicalSystem):
    def __init__(self, sigma: float = 10.0, beta: float = 8.0 / 3.0, rho: float = 28.0, dim: int = 3, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5), noise_std: float = 0.0) -> None:
        super().__init__(dim=dim, dt=dt, time_span=time_span)
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.noise_std = noise_std
        self.time_span = time_span
        self.dt = dt

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Ensure that x is a 3D tensor, even if batch_size = 1
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add a batch dimension if it's missing
        
        # Initialize the derivative tensor
        dxdt = torch.zeros_like(x)

        # Lorenz system equations
        dxdt[:, 0] = self.sigma * (x[:, 1] - x[:, 0])  # dx/dt
        dxdt[:, 1] = x[:, 0] * (self.rho - x[:, 2]) - x[:, 1]  # dy/dt
        dxdt[:, 2] = x[:, 0] * x[:, 1] - self.beta * x[:, 2]  # dz/dt

        return dxdt

class MayLeonardSystem(DynamicalSystem):
    def __init__(self, a: float = 1.2, b: float = 0.8, 
                 dim: int = 3, 
                 dt: float = 0.05, 
                 time_span: Tuple[float, float] = (0, 50), 
                 noise_std: float = 0.0) -> None:
        """
        Initializes the 3-species May-Leonard system (generalized Lotka-Volterra).

        :param a: Competition coefficient against the next species in cyclic order.
        :param b: Competition coefficient against the following species in cyclic order.
        :param dim: Number of species (should be 3 for May-Leonard system).
        :param dt: Time step.
        :param time_span: Simulation time span.
        :param noise_std: Standard deviation of Gaussian noise (optional).
        """
        super().__init__(dim=dim, dt=dt, time_span=time_span)
        assert dim == 3, "May-Leonard model is defined for 3 species."
        self.a = a
        self.b = b
        self.noise_std = noise_std
        self.dt = dt
        self.time_span = time_span

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the May-Leonard dynamics.
        
        :param t: Time tensor (not used; system is autonomous).
        :param x: Tensor of shape (batch_size, dim) representing the populations.
        :return: Time derivatives of populations.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        dxdt = torch.zeros_like(x)

        # Dynamics with cyclic interactions
        dxdt[:, 0] = x[:, 0] * (1.0 - x[:, 0] - self.a * x[:, 1] - self.b * x[:, 2])
        dxdt[:, 1] = x[:, 1] * (1.0 - x[:, 1] - self.a * x[:, 2] - self.b * x[:, 0])
        dxdt[:, 2] = x[:, 2] * (1.0 - x[:, 2] - self.a * x[:, 0] - self.b * x[:, 1])

        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            dxdt += noise

        return dxdt

#ND systems
class LotkaVolterraSystem(DynamicalSystem):
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor, dim: int, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5), noise_std: float = 0.0) -> None:
        """
        Initializes the N-dimensional Lotka-Volterra system.

        :param alpha: Tensor of size (dim,) containing the intrinsic growth rates of the species.
        :param beta: Tensor of size (dim, dim) containing the interaction coefficients.
        :param dim: The number of species (dimension of the system).
        :param dt: Time step for the simulation.
        :param time_span: Tuple specifying the start and end time.
        :param noise_std: Standard deviation of Gaussian noise to be added to the system (optional).
        """
        super().__init__(dim=dim, dt=dt, time_span=time_span)
        self.alpha = alpha
        self.beta = beta
        self.noise_std = noise_std
        self.time_span = time_span
        self.dt = dt

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the derivatives for the Lotka-Volterra system.

        :param t: Time tensor (not used in this case since the system is autonomous).
        :param x: Tensor of size (batch_size, dim) representing the current populations.
        :return: Tensor of size (batch_size, dim) representing the population growth rates.
        """
        # Ensure that x is at least 2D (batch_size, dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        dxdt = torch.zeros_like(x)

        # Lotka-Volterra equations
        for i in range(self.dim):
            interaction_term = torch.sum(self.beta[i] * x, dim=1)
            dxdt[:, i] = x[:, i] * (self.alpha[i] - interaction_term)

        return dxdt


###integrating and generating trajectories
def generate_initial_conditions(
    sampling_method: str,
    bounds: tuple,
    num_points: int,
    kernel_fn=None,
    predefined_initial_conditions=None
) -> torch.Tensor:
    """
    Generates initial conditions for an N-dimensional dynamical system.

    :param sampling_method: Method for sampling initial conditions ('uniform', 'grid', 'density').
    :param bounds: Tuple of tuples, each containing the (min, max) bounds for each dimension.
    :param num_points: Number of initial points to sample.
    :param kernel_fn: Kernel function used to determine the density of sampling (if 'density' sampling is used).
    :param predefined_initial_conditions: Predefined list of initial conditions for trajectories.
    :return: Tensor of initial conditions with shape (num_points, N).
    """
    # Number of dimensions
    N = len(bounds)

    # Handle predefined initial conditions if provided
    if predefined_initial_conditions is not None:
        # Ensure predefined initial conditions are in tensor format
        if isinstance(predefined_initial_conditions, torch.Tensor):
            initial_conditions = predefined_initial_conditions.clone().float()
        else:
            initial_conditions = torch.tensor(predefined_initial_conditions, dtype=torch.float32)
        return initial_conditions

    initial_conditions = []

    # Sample initial points based on the chosen method
    if sampling_method == 'uniform':
        # Sample random points uniformly within the bounds for each dimension
        for _ in range(num_points):
            point = [np.random.uniform(min_bound, max_bound) for min_bound, max_bound in bounds]
            initial_conditions.append(point)

    elif sampling_method == 'grid':
        # Sample points on a grid for each dimension
        grid_points = [np.linspace(min_bound, max_bound, int(np.cbrt(num_points))) for min_bound, max_bound in bounds]
        grid_combinations = np.array(np.meshgrid(*grid_points)).T.reshape(-1, N)
        
        # Select a subset of grid points if the total grid size exceeds num_points
        initial_conditions = grid_combinations[:num_points].tolist()

    elif sampling_method == 'density' and kernel_fn is not None:
        # Sample based on a custom probability density (using a given kernel function)
        points = []
        weights = []
        for _ in range(num_points):
            point = torch.tensor([np.random.uniform(min_bound, max_bound) for min_bound, max_bound in bounds], dtype=torch.float32)
            weight = kernel_fn(point)  # Use the kernel function directly
            points.append(point)
            weights.append(weight)

        # Normalize the weights so they sum to 1
        weights = torch.tensor(weights)
        weights = weights / torch.sum(weights)

        # Sample initial points based on the kernel weights
        indices = torch.multinomial(weights, num_samples=num_points, replacement=True)
        initial_conditions = [points[i].numpy() for i in indices]

    # Convert initial conditions to a tensor
    initial_conditions = torch.tensor(np.array(initial_conditions), dtype=torch.float32)

    return initial_conditions


def generate_trajectories(
    system: DynamicalSystem,
    noise_std: float=None,
    sampling_method: str=None,
    init_points_bounds: tuple=None,
    num_points: int=None,
    time_span: torch.Tensor=None,
    dt: float=None,
    kernel_fn=None,
    predefined_initial_conditions=None
):
    """
    Generates trajectories using the given sampling method and kernel function, or uses predefined initial conditions.

    :param sampling_method: Method for sampling initial conditions ('random', 'grid', 'density').
    :param bounds: Tuple (max_x, max_y) to define the bounds of sampling space.
    :param time_span: Time span for integration as a tensor [t_start, t_end].
    :param dt: Time step for integration.
    :param num_points: Number of initial points to sample.
    :param kernel_fn: Kernel function used to determine the density of sampling (if 'density' sampling is used).
    :param system: Dynamical system (LimitCycle or VanDerPol).
    :param predefined_initial_conditions: Predefined list of initial conditions for trajectories.
    :return: Time values, trajectories, and sampled initial conditions.
    """
    if not dt:
        dt = system.dt
    if time_span is None or not torch.any(time_span):
        time_span = system.time_span
    #t_eval = np.arange(time_span.cpu().numpy()[0], time_span.cpu().numpy()[1], dt)
    t_values = torch.arange(time_span[0], time_span[1], dt)

    if predefined_initial_conditions is not None:
        # Ensure predefined initial conditions are in tensor format
        if isinstance(predefined_initial_conditions, torch.Tensor):
            initial_conditions = predefined_initial_conditions.clone().float() #.detach().float()
            initial_conditions.requires_grad_(True)
        else:
            initial_conditions = torch.tensor(predefined_initial_conditions, dtype=torch.float32)
    else:
       initial_conditions = generate_initial_conditions(sampling_method=sampling_method, bounds=init_points_bounds, num_points=num_points, kernel_fn=kernel_fn) 

    #initial_conditions.requires_grad_(False)
    # Integrate the system for each initial condition
    trajectories = []
    def system_with_noise(t, y):
            dydt = system(t, y)
            noise = torch.randn_like(y) * system.noise_std
            return dydt + noise
        
    for initial_condition in initial_conditions:
        #trajectory = odeint(system, initial_condition, t_values, method='rk4')
        trajectory = odeint(system_with_noise, initial_condition, t_values, method='rk4')

        trajectories.append(trajectory)

    # Convert trajectories to a tensor
    trajectories = torch.stack(trajectories)
    
    return t_values, trajectories, initial_conditions







def integrate_system_scipy(system, initial_condition, time_span, dt):
    """
    Solve the dynamical system using SciPy's `solve_ivp`.

    :param system: A callable dynamical system f(t, x).
    :param initial_condition: Initial state as a numpy array.
    :param time_span: Tuple (t_start, t_end).
    :param dt: Time step for integration.
    :return: Time values and trajectory.
    """
    t_eval = np.arange(time_span[0], time_span[1], dt)

    def system_np(t, x):
        """Wrapper to convert PyTorch tensors to NumPy and back."""
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # Shape (1, dim)
        dx_dt = system(torch.tensor(t, dtype=torch.float32), x_tensor)
        return dx_dt.squeeze(0).detach().numpy()  # Convert back to NumPy

    sol = solve_ivp(system_np, time_span, initial_condition.numpy(), t_eval=t_eval, method='RK45')

    return torch.tensor(sol.t, dtype=torch.float32), torch.tensor(sol.y.T, dtype=torch.float32)

def generate_trajectories_scipy(
    system: DynamicalSystem,
    time_span: torch.Tensor=None,
    dt: float=None,
    sampling_method: str=None,
    num_points: int=None,
    init_points_bounds: tuple=None,
    predefined_initial_conditions=None,
    noise_std: float=0.0
):
    """
    Generates trajectories using SciPy's solve_ivp instead of torchdiffeq.odeint.

    :param sampling_method: Sampling method ('random', 'grid', etc.).
    :param bounds: Tuple (max_x, max_y) for initial conditions.
    :param time_span: Time span as a PyTorch tensor [t_start, t_end].
    :param dt: Time step.
    :param num_points: Number of trajectories to generate.
    :param kernel_fn: Optional kernel function for density-based sampling.
    :param system: Dynamical system function.
    :param predefined_initial_conditions: Predefined list of initial conditions.
    :return: t_values (torch.Tensor), trajectories (list of torch.Tensors), initial_conditions (torch.Tensor).
    """
    if not dt:
        dt = system.dt
    if time_span is None or not np.any(time_span):
        time_span = system.time_span
    t_eval = np.arange(time_span[0], time_span[1], dt)

    if predefined_initial_conditions is not None:
        if isinstance(predefined_initial_conditions, torch.Tensor):
            initial_conditions = predefined_initial_conditions.clone().float() #.detach().float()
            initial_conditions.requires_grad_(True)
        else:
            initial_conditions = torch.tensor(predefined_initial_conditions, dtype=torch.float32)
    else:
        if sampling_method == 'uniform':
            min_x, max_x = init_points_bounds
            initial_conditions = np.random.uniform(min_x, max_x, (num_points, 2))
            initial_conditions = torch.tensor(initial_conditions, dtype=torch.float32)

    def scipy_rhs(t, x):
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        dx_dt = system.forward(torch.tensor([t], dtype=torch.float32), x_tensor).squeeze(0)  # Compute dynamics
        noise = torch.randn_like(dx_dt) * noise_std  # Gaussian noise (could also be more complex)
        return dx_dt.detach().numpy()

    trajectories = []
    for x0 in initial_conditions:
        sol = solve_ivp(scipy_rhs, (t_eval[0], t_eval[-1]), x0.detach().numpy(), t_eval=t_eval, method='RK45')
        trajectories.append(torch.tensor(sol.y.T, dtype=torch.float32).detach())  # Detach computation graph

    trajectories = torch.stack(trajectories)
    return torch.tensor(t_eval, dtype=torch.float32), trajectories, initial_conditions

import time

# def generate_trajectories_for_training(diffeo_net, source_system, use_transformed_system, initial_conditions_target, noise_std=0.0):
#     initial_conditions_source = diffeo_net.inverse(initial_conditions_target)  

#     if use_transformed_system:
#         transformed_system = PhiSystemPhiInv(source_system, diffeo_net)
#         _, transformed_trajectories, _ = generate_trajectories(
#                 system=transformed_system,
#                 predefined_initial_conditions=initial_conditions_source
#             )
#     else:
#         with torch.no_grad():  
#             _, trajectories_source, _ = generate_trajectories_scipy(
#                 system=source_system,
#                 predefined_initial_conditions=initial_conditions_source,
#                 noise_std=noise_std
#             )
#         transformed_trajectories = [diffeo_net(traj.requires_grad_()) for traj in trajectories_source]
#         transformed_trajectories = torch.stack(transformed_trajectories)
#     return transformed_trajectories


def generate_trajectories_for_training(
    diffeo_net, 
    source_system, 
    use_transformed_system, 
    initial_conditions_target, 
    noise_std=0.0
):
    initial_conditions_source = diffeo_net.inverse(initial_conditions_target)

    if isinstance(source_system, LearnableDynamicalSystem):
        # If the source system is a learnable dynamical system, we need to forward it without torch.no_grad()
        # This will allow gradients to flow through the learnable system
        _, trajectories_source, _ = generate_trajectories(
            system=source_system, 
            predefined_initial_conditions=initial_conditions_source, 
            noise_std=noise_std
        )
        # Apply the diffeomorphism network to the trajectories generated by the learnable system
        transformed_trajectories = [diffeo_net(traj.requires_grad_()) for traj in trajectories_source]
        transformed_trajectories = torch.stack(transformed_trajectories)
        
    else:
        # If the source system is not learnable, use the existing method for generating trajectories
        if use_transformed_system:
            transformed_system = PhiSystemPhiInv(source_system, diffeo_net)
            _, transformed_trajectories, _ = generate_trajectories(
                system=transformed_system,
                predefined_initial_conditions=initial_conditions_source, noise_std=noise_std

            )
        else:
            with torch.no_grad():  # Only use no_grad for fixed systems (non-learnable systems)
                _, trajectories_source, _ = generate_trajectories_scipy(
                    system=source_system,
                    predefined_initial_conditions=initial_conditions_source,
                    noise_std=noise_std
                )
            # Apply the diffeomorphism network to the fixed system's trajectories
            transformed_trajectories = [diffeo_net(traj.requires_grad_()) for traj in trajectories_source]
            transformed_trajectories = torch.stack(transformed_trajectories)

    return transformed_trajectories
