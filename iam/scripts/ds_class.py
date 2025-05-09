import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Tuple, Optional, List, Literal, Union
from torchdiffeq import odeint
from scipy.integrate import solve_ivp
import inspect
import warnings
from scripts.utils import set_seed

class TrainablePeriodicFunction(nn.Module):
    """Implements a periodic function."""
    def __init__(self, num_terms: int = 5):
        super().__init__()
        self.num_terms = num_terms
        self.a = nn.Parameter(torch.randn(num_terms)*0.01)  #small cosine coefficients
        self.b = nn.Parameter(torch.randn(num_terms)*0.01)  #small sine coefficients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assumes x in [0, 2pi]
        result = torch.zeros_like(x)
        for n in range(1, self.num_terms + 1):
            result += self.a[n-1] * torch.cos(n * x) + self.b[n-1] * torch.sin(n * x)
        return result

class CircleNN(nn.Module):
    """Implements a near periodic function."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map scalar x to (cos(x), sin(x))
        x_embed = torch.stack([torch.cos(x), torch.sin(x)], dim=-1)
        return self.net(x_embed).squeeze(-1)


### Base class for dynamical systems
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


    def compute_trajectory(self,
        initial_conditions: torch.Tensor,
        noise_std: Optional[float] = None,
        dt: Optional[float] = None,
        time_span: Optional[Tuple[float, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the trajectory of the system starting from given initial conditions.

        :param initial_conditions: Tensor of shape (batch_size, dim) or (dim,).
        :param noise_std: Optional override for noise standard deviation.
        :param dt: Optional override for time step.
        :param time_span: Optional override for time span (start, end).
        :return: Tuple (t_values, trajectories) where:
                 - t_values is a tensor of shape (T,)
                 - trajectories is a tensor of shape (batch_size, T, dim)
        """
        if initial_conditions.dim() == 1:
            initial_conditions = initial_conditions.unsqueeze(0)

        batch_size = initial_conditions.shape[0]

        if time_span is None:
            time_span = self.time_span
        if dt is None:
            dt = self.dt
        if noise_std is None:
            noise_std = self.noise_std

        t_values = torch.arange(time_span[0], time_span[1], dt)

        def system_with_noise(t, y):
            dydt = self(t, y)
            if noise_std > 0:
                dydt += torch.randn_like(y) * noise_std
            return dydt

        trajectories = []
        for i in range(batch_size):
            trajectory = odeint(system_with_noise, initial_conditions[i], t_values, method='rk4')
            trajectories.append(trajectory)

        return torch.stack(trajectories)

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

    def __init__(self, dim: int, attraction_rate: float = 1.0, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5)):
        super().__init__()
        assert dim >= 2, "Dimension must be at least 2."
        self.dim = dim
        self.attraction_rate = attraction_rate

        self.dt = dt
        self.time_span = time_span

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension

        x1, x2 = x[:, 0], x[:, 1]
        r = torch.sqrt(x1**2 + x2**2)
        theta = torch.atan2(x2, x1)

        dx1_dt = (-r * ((r - 1) * torch.cos(theta) - torch.sin(theta))).unsqueeze(1)
        dx2_dt = (-r * ((r - 1) * torch.sin(theta) + torch.cos(theta))).unsqueeze(1)

        dx_dt = [dx1_dt, dx2_dt]

        # Remaining dimensions flow toward 0 (attractive dynamics)
        if self.dim > 2:
            residual = x[:, 2:]
            d_residual_dt = -self.attraction_rate * residual
            dx_dt.append(d_residual_dt)

        return torch.cat(dx_dt, dim=1)



###########learnable time parametrization systems
class LearnableDynamicalSystem(DynamicalSystem):
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
    

class LearnableNDLinearSystem(LearnableDynamicalSystem):
    """
    A linear dynamical system dx/dt = Ax, where A is either a full or diagonal learnable matrix.
    """

    def __init__(
        self,
        dim: int,
        dt: float = 0.05,
        time_span: Tuple[float, float] = (0, 5),
        A_init: Optional[Union[torch.Tensor, np.ndarray, str]] = None,
        diagonal_only: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.time_span = time_span
        self.diagonal_only = diagonal_only

        # Default initialization
        if A_init is None or A_init == 'stable':
            A_init = -torch.ones(dim) if diagonal_only else -torch.eye(dim)

        elif isinstance(A_init, np.ndarray):
            A_init = torch.tensor(A_init, dtype=torch.float32)

        # Register parameter
        if diagonal_only:
            assert A_init.shape == (dim,), "For diagonal_only=True, A_init must be a vector of shape (dim,)"
            self.A_diag = nn.Parameter(A_init)
        else:
            assert A_init.shape == (dim, dim), "For diagonal_only=False, A_init must be a matrix of shape (dim, dim)"
            self.A = nn.Parameter(A_init)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.diagonal_only:
            return x * self.A_diag  # Elementwise
        else:
            return torch.matmul(x, self.A.T)  # Standard matrix multiplication

#BLA
class LearnableNDBoundedLineAttractor(LearnableDynamicalSystem):
    """
    A learnable bounded line attractor: dx/dt = alpha * (-x + ReLU(Wx + b)).
    alpha is a learnable scalar controlling global speed.
    """

    def __init__(
        self,
        W: torch.Tensor,
        b: torch.Tensor,
        dt: float = 0.05,
        time_span: Tuple[float, float] = (0, 5),
        alpha_init: float = 1.0,
    ):
        """
        :param W: Weight matrix (fixed, not learnable).
        :param b: Bias vector (fixed, not learnable).
        :param alpha_init: Initial value for the learnable scaling factor alpha.
        """
        super().__init__()
        self.W = W
        self.b = b
        self.relu = nn.ReLU()
        self.dt = dt
        self.time_span = time_span
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        linear_part = torch.matmul(x, self.W.T) + self.b
        dx_dt = -x + self.relu(linear_part)
        return self.alpha * dx_dt

#LC 
class LearnableNDLimitCycle(LearnableDynamicalSystem):
    """
    N-dimensional limit cycle system with learnable velocity and alpha parameters.
    The first two coordinates define a planar limit cycle (polar coordinates dynamics),
    while the remaining dimensions are attracted toward the 2D plane using the same alpha.
    """

    def __init__(
        self,
        dim: int,
        dt: float = 0.05,
        time_span: Tuple[float, float] = (0, 5),
        noise_std: float = 0.0,
        radius: float = 1.0,  
        velocity_init: float = -1.0,
        alpha_init: float = -1.0,
        use_theta_modulation: bool = False,
        theta_modulation_num_terms: int = 5,
    ):
        super().__init__()
        assert dim >= 2, "Dimension must be at least 2."
        self.dim = dim
        self.dt = dt
        self.time_span = time_span
        self.noise_std = noise_std
        self.radius = radius   
        self.use_theta_modulation = use_theta_modulation

        # Learnable modulation of angular velocity
        if use_theta_modulation:
            self.theta_modulator = TrainablePeriodicFunction(num_terms=theta_modulation_num_terms)
        # Learnable parameters
        if velocity_init is None:
            velocity_init = 1.0
        elif not use_theta_modulation: #only if not using theta modulation
            self.velocity = nn.Parameter(torch.tensor(velocity_init, dtype=torch.float32))
        if alpha_init is None:
            alpha_init = -1.0
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension

        # Planar (r, theta) dynamics
        x_val, y_val = x[:, 0], x[:, 1]
        r = torch.sqrt(x_val**2 + y_val**2)
        theta = torch.atan2(y_val, x_val)

        if self.use_theta_modulation:
            dtheta_dt = self.theta_modulator(theta)  # purely theta-based modulation
        else:
            dtheta_dt = self.velocity  # Learnable velocity
        dr_dt =  - self.alpha * r * (self.radius - r)  # Radial dynamics

        # Convert polar derivatives back to Cartesian
        dx_dt = dr_dt * torch.cos(theta) - r * torch.sin(theta) * dtheta_dt
        dy_dt = dr_dt * torch.sin(theta) + r * torch.cos(theta) * dtheta_dt

        dx_dt = dx_dt.unsqueeze(1)
        dy_dt = dy_dt.unsqueeze(1)

        derivatives = [dx_dt, dy_dt]

        # Higher dimensions: attraction toward (0, 0) with the same alpha
        if self.dim > 2:
            residual = x[:, 2:]
            d_residual_dt = self.alpha * residual
            derivatives.append(d_residual_dt)

        return torch.cat(derivatives, dim=1)

#RA
class LearnableNDRingAttractor(LearnableDynamicalSystem):
    """
    N-dimensional ring attractor system with learnable alpha parameters.
    The first two coordinates define a planar limit cycle (polar coordinates dynamics),
    while the remaining dimensions are attracted toward the 2D plane using the same alpha.
    """

    def __init__(
        self,
        dim: int,
        dt: float = 0.05,
        time_span: Tuple[float, float] = (0, 5),
        noise_std: float = 0.0,
        radius: float = 1.0,  
        alpha_init: float = -1.0,
        sigma_init: float = 0.05,
        vf_on_ring_enabled: bool = False,
        vf_on_ring_num_terms: int = 5
    ):
        super().__init__()
        assert dim >= 2, "Dimension must be at least 2."
        self.dim = dim
        self.dt = dt
        self.time_span = time_span
        self.noise_std = noise_std
        self.radius = radius  # Fixed radius for the limit cycle

        # Learnable parameters
        if alpha_init is None:
            alpha_init = -1.0
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        if vf_on_ring_enabled:
            self.vf_on_ring = TrainablePeriodicFunction(num_terms=vf_on_ring_num_terms)
        self.vf_on_ring_enabled = vf_on_ring_enabled
        self.sigma = nn.Parameter(torch.tensor(sigma_init, dtype=torch.float32))  # width of radial band around the ring

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension

        # Planar (r, theta) dynamics
        x_val, y_val = x[:, 0], x[:, 1]
        r = torch.sqrt(x_val**2 + y_val**2)
        theta = torch.atan2(y_val, x_val)

        dtheta_dt = 0 
        if self.vf_on_ring_enabled:
            #with gaussian envelope
            angular_perturb = self.vf_on_ring(theta)
            bump = torch.exp(-((r - self.radius) ** 2) / (2 * self.sigma**2))
            dtheta_dt = bump * angular_perturb

        dr_dt =  - self.alpha * r * (self.radius - r)  # Radial dynamics

        # Convert polar derivatives back to Cartesian
        dx_dt = dr_dt * torch.cos(theta) - r * torch.sin(theta) * dtheta_dt
        dy_dt = dr_dt * torch.sin(theta) + r * torch.cos(theta) * dtheta_dt

        dx_dt = dx_dt.unsqueeze(1)
        dy_dt = dy_dt.unsqueeze(1)

        derivatives = [dx_dt, dy_dt]

        # Higher dimensions: attraction toward (0, 0) with the same alpha
        if self.dim > 2:
            residual = x[:, 2:]
            d_residual_dt = self.alpha * residual
            #d_residual_dt = - residual
            derivatives.append(d_residual_dt)

        return torch.cat(derivatives, dim=1)


#Sphere attractor
class LearnableSphereAttractor(LearnableDynamicalSystem):
    """
    Embeds an S^d sphere inside R^D (D > d), with radial attraction toward the sphere
    and optional tangent dynamics + residual attraction.
    """

    def __init__(
        self,
        dim: int,
        sphere_dim: int = 2,
        dt: float = 0.05,
        time_span: Tuple[float, float] = (0, 5),
        noise_std: float = 0.0,
        radius: float = 1.0,
        alpha_init: float = -1.0,
        sigma_init: float = 0.05,
        vf_on_sphere_enabled: bool = False,
        vf_on_sphere_num_terms: int = 5,
    ):
        super().__init__()
        assert dim >= sphere_dim + 1, "Embedding requires dim >= sphere_dim + 1"
        self.dim = dim
        self.sphere_dim = sphere_dim
        self.radius = radius
        self.dt = dt
        self.time_span = time_span
        self.noise_std = noise_std

        # Learnable parameters
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma_init, dtype=torch.float32))

        # Optionally enable the tangent vector field
        self.vf_on_sphere_enabled = vf_on_sphere_enabled
        if self.vf_on_sphere_enabled:
            warnings.warn(
                "vf_on_sphere_enabled=True, but tangent dynamics on the sphere have not yet been implemented. "
                "Currently, only radial attraction is active."
            )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Extract the sphere part and residual part
        x_sphere = x[:, :self.sphere_dim + 1]
        x_residual = x[:, self.sphere_dim + 1:] if self.dim > self.sphere_dim + 1 else None

        # Radial unit vector and deviation from radius
        norm = torch.norm(x_sphere, dim=1, keepdim=True) + 1e-8  # avoid division by zero
        radial_unit = x_sphere / norm
        deviation = norm - self.radius
        dr_dt = self.alpha * deviation
        dx_sphere_radial = dr_dt * radial_unit

        # Placeholder for tangent vector field (currently no implementation)
        dx_sphere_tangent = torch.zeros_like(x_sphere)

        # Radial + Tangent dynamics
        dx_sphere = dx_sphere_radial + dx_sphere_tangent

        # If there is a residual part (dimensions beyond the sphere), apply attraction
        if x_residual is not None:
            dx_residual = self.alpha * x_residual  # attraction towards the origin
            dx_dt = torch.cat([dx_sphere, dx_residual], dim=1)
        else:
            dx_dt = dx_sphere

        return dx_dt

#BCA
class BoundedContinuousAttractor(LearnableDynamicalSystem):
    def __init__(self, dim: int, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5), bounds: float = 1.0):
        """
        A bounded continuous attractor with:
        - Zero flow inside [-1, 1]^D
        - Linear flow to nearest boundary outside

        Args:
            dim (int): Dimensionality D of the system
            dt (float): Time step for numerical integration
            time_span (Tuple[float, float]): Start and end times for trajectory
        """
        super().__init__(dim=dim)
        self.dt = dt
        self.time_span = time_span
        self.bounds = bounds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute flow vector at input point x.

        Args:
            x (torch.Tensor): shape (..., D)

        Returns:
            torch.Tensor: flow vectors, shape (..., D)
        """
        projected = torch.clamp(x, -self.bounds, self.bounds)
        mask = (x < -self.bounds) | (x > self.bounds)
        flow = torch.where(mask, projected - x, torch.zeros_like(x))
        return flow

    def compute_trajectory(self, initial_position: torch.Tensor, time_span: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Simulate trajectory using Euler integration.

        Args:
            initial_position (torch.Tensor): (batch_size, D)
            time_span (Optional[Tuple[float, float]]): overrides default time span

        Returns:
            torch.Tensor: (batch_size, T, D)
        """
        if time_span is None:
            time_span = self.time_span
        t_start, t_end = time_span
        t_values = torch.arange(t_start, t_end, self.dt, device=initial_position.device)
        num_steps = len(t_values)

        batch_size, dim = initial_position.shape
        trajectory = torch.zeros(batch_size, num_steps, dim, device=initial_position.device)
        x = initial_position.clone()

        for i in range(num_steps):
            trajectory[:, i] = x
            dx = self.forward(x)
            x = x + self.dt * dx

        return trajectory

# Composite systems
class LearnableCompositeSystem(LearnableDynamicalSystem):
    """
    A composite dynamical system formed by concatenating multiple sub-systems.
    Each sub-system operates on a distinct subset of the full state space.
    """

    def __init__(
        self,
        systems: List[LearnableDynamicalSystem],
        dims: List[int],  # dimensions of each sub-system
        dt: float = 0.05,
        time_span: Tuple[float, float] = (0, 5),
    ):
        super().__init__()
        assert len(systems) == len(dims), "Each system must have a corresponding dimension."
        self.systems = nn.ModuleList(systems)
        self.dims = dims
        self.dt = dt
        self.time_span = time_span
        self.total_dim = sum(dims)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the composite dynamics by splitting the state and applying each sub-system.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        assert x.shape[1] == self.total_dim, f"Input must have {self.total_dim} dimensions."

        dx_parts = []
        start = 0
        for system, dim in zip(self.systems, self.dims):
            x_part = x[:, start:start + dim]
            dx_part = system(t, x_part)
            dx_parts.append(dx_part)
            start += dim

        return torch.cat(dx_parts, dim=1)



#Analytical classes
class AnalyticDynamicalSystem(nn.Module):
    """
    A superclass for dynamical systems with analytical solutions. This class provides a method
    to compute trajectories directly based on the system's analytical solution.
    """
    def __init__(self, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5)):
        """
        Initialize the system with a given time step and time span.
        
        :param dt: Time step for trajectory generation.
        :param time_span: Tuple (t_start, t_end) specifying the time range for the trajectory.
        """
        super().__init__()
        self.dt = dt
        self.time_span = time_span
    
    def compute_trajectory(self) -> torch.Tensor:
        """
        This method should be implemented by subclasses. It computes the trajectory using the 
        analytical solution of the dynamical system.
        
        :return: A tensor containing the trajectory over the time span.
        """
        raise NotImplementedError("Subclasses must implement this method to compute the trajectory.")


class AnalyticalLinearSystem(AnalyticDynamicalSystem):
    """
    Computes the trajectory of a linear dynamical system defined by \dot{x} = A x,
    where A is a learnable constant matrix initialized to -I.
    """
    def __init__(self, dim: int, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5)):
        """
        Initialize the class with a learnable matrix A initialized to -I.

        :param dim: The dimensionality of the system.
        :param dt: The time step used for discretizing the trajectory.
        :param time_span: Tuple (t_start, t_end) specifying the time range for the trajectory.
        """
        super().__init__(dt=dt, time_span=time_span)
        self.dim = dim
        # Initialize A as a learnable parameter with -I
        A_init = -torch.eye(dim)
        self.A = nn.Parameter(A_init)

    def compute_trajectory(self, initial_position: torch.Tensor, time_span: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Computes the trajectory for the linear system using the analytical solution x(t) = exp(tA)x₀.

        :param initial_position: A tensor of shape (batch_size, dim) representing the initial positions.
        :param time_span: Optional tuple specifying the time range for the trajectory.
        :return: A tensor of shape (batch_size, T, dim) where T is the number of time steps.
        """
        if time_span is None:
            time_span = self.time_span
        batch_size = initial_position.shape[0]

        # Create time vector (shape: T)
        t_start, t_end = time_span
        t_values = torch.arange(t_start, t_end, self.dt).to(initial_position.device)  # shape: (T,)

        # Compute exp(tA) for each t and apply to x₀
        trajectory = torch.zeros(batch_size, len(t_values), self.dim, device=initial_position.device)
        for t_idx, t in enumerate(t_values):
            exp_tA = torch.matrix_exp(self.A * t)
            trajectory[:, t_idx] = initial_position @ exp_tA.T  # shape: (batch_size, dim)

        return trajectory

#TODO: make diagonal (option)
class AnalyticalBoundedLineAttractor(AnalyticDynamicalSystem):
    """
    Piecewise-analytic integration of dx/dt = -x + ReLU(Wx + b),
    detecting activation regime switches and analytically integrating within each regime.
    """

    def __init__(self, W: torch.Tensor, b: torch.Tensor,
                 dt: float = 0.05, time_span: Tuple[float, float] = (0, 5)):
        super().__init__(dt, time_span)
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)
        self.relu = nn.ReLU()

    def compute_trajectory(self, initial_position: torch.Tensor,
                           time_span: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        if time_span is None:
            time_span = self.time_span

        t_start, t_end = time_span
        t_values = torch.arange(t_start, t_end, self.dt).to(initial_position.device)  # (T,)
        T = len(t_values)
        batch_size, dim = initial_position.shape

        trajectory = torch.zeros(batch_size, T, dim, device=initial_position.device)
        x_t = initial_position.clone()

        for t_idx in range(T):
            # Store current state
            trajectory[:, t_idx] = x_t

            # Detect activation pattern
            z = torch.matmul(x_t, self.W.T) + self.b  # (batch_size, dim)
            mask = (z > 0).float()  # (batch_size, dim)

            # Prepare next state for each batch independently
            next_x = torch.zeros_like(x_t)

            for i in range(batch_size):
                m = mask[i]  # (dim,)
                W_eff = self.W * m.unsqueeze(1)  # Keep active units
                b_eff = self.b * m

                A = W_eff - torch.eye(dim, device=x_t.device)
                try:
                    A_exp = torch.matrix_exp(A * self.dt)
                    A_inv = torch.linalg.pinv(A)
                    B_term = (A_exp - torch.eye(dim, device=x_t.device)) @ (A_inv @ b_eff)
                    next_x[i] = (A_exp @ x_t[i]) + B_term
                except RuntimeError:
                    # Fallback to Euler step in rare singular matrix cases
                    next_x[i] = x_t[i] + self.dt * (-x_t[i] + self.relu(torch.matmul(x_t[i], self.W.T) + self.b))

            x_t = next_x

        return trajectory


class AnalyticalLimitCycle(AnalyticDynamicalSystem):
    """
    Computes the trajectory of a limit cycle system defined by r_dot = r(r-1) and theta_dot = v.
    This class uses analytical solutions for r(t) and theta(t) based on the initial condition and velocity.
    """
    def __init__(self, dim: int,  velocity_init: float = 1., alpha_init: float = -1.,  
                 time_span: Tuple[float, float] = (0.0, 5.0), dt: float = 0.05):
        """
        Initialize the class with the given velocity, time span, and time step.
        
        :param velocity_init: Initial velocity (v) in the angular direction (theta_dot = v).
        :param dim: The dimensionality of the system. For a 2D system, dim = 2.
        :param time_span: Tuple (t_start, t_end) specifying the time range for the trajectory.
        :param dt: The time step used for discretizing the trajectory.
        """
        super().__init__(dt, time_span)
        self.time_span = time_span
        self.dt = dt
        if not velocity_init is None:         
            self.velocity = nn.Parameter(torch.tensor(velocity_init, dtype=torch.float32))
        else:
            self.velocity = -1. 
        if not alpha_init is None:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        else:
            self.alpha = -1.        
        self.dim = dim

    def compute_trajectory(self, initial_position: torch.Tensor, time_span: Optional[Tuple[float,float]] = None) -> torch.Tensor:
        """
        Computes the trajectory using the analytical solutions for r(t) and theta(t).

        :param initial_position: A tensor of shape (batch_size, dim) representing the initial positions.
        :return: A tensor of shape (batch_size, T, dim) where T is the number of time steps, and each row is [x(t), y(t), ...].
        """
        # Ensure initial_position is of shape (batch_size, dim)
        if time_span is None:
            time_span = self.time_span
        batch_size = initial_position.shape[0]

        # Compute initial radius (r0) and angle (theta0) for the first 2 dimensions
        x0, y0 = initial_position[:, 0], initial_position[:, 1]  # 2D system for r and theta
        r0 = torch.sqrt(x0**2 + y0**2)  # Initial radius as a scalar for each sample in the batch
        theta0 = torch.atan2(y0, x0)    # Initial angle for each sample in the batch

        # Create time vector (shape: T)
        t_start, t_end = time_span
        t_values = torch.arange(t_start, t_end, self.dt).to(initial_position.device)  # Time steps

        # Expand t_values to (batch_size, T) for broadcasting
        t_values_expanded = t_values.unsqueeze(0).expand(batch_size, -1)

        # Compute the trajectory using the analytical solutions for r(t) and theta(t)
        r_t = (r0.unsqueeze(1) * torch.exp(- self.alpha * t_values.unsqueeze(0))) / \
      (1 + r0.unsqueeze(1) * (torch.exp(- self.alpha * t_values.unsqueeze(0)) - 1))
        theta_t = self.velocity * t_values_expanded + theta0.unsqueeze(1)  # theta(t)

        # Convert polar to Cartesian coordinates for the 2D part
        x_t = r_t * torch.cos(theta_t)
        y_t = r_t * torch.sin(theta_t)

        # Concatenate x_t and y_t to form the 2D trajectory
        trajectory = torch.stack([x_t, y_t], dim=2)  # Shape: (batch_size, T, 2)

        # Handle higher dimensions: attraction toward origin (-x) for residual dimensions
        if self.dim > 2:
            residual = initial_position[:, 2:]  # Get the residual higher dimensions (3D and beyond)
            
            # Dynamics for higher dimensions: attraction to origin (-x)
            residual_t = residual.unsqueeze(1) * torch.exp(self.alpha * t_values_expanded.unsqueeze(2))

            # Concatenate the 2D part with the higher-dimensional residual
            trajectory = torch.cat([trajectory, residual_t], dim=2)  # Shape: (batch_size, T, dim)

        return trajectory

#RA
class AnalyticalRingAttractor(AnalyticDynamicalSystem):
    """
    Computes the trajectory of a limit cycle system defined by r_dot = r(r-1) and theta_dot = v.
    This class uses analytical solutions for r(t) and theta(t) based on the initial condition and velocity.
    """
    def __init__(self,  dim: int, alpha_init: float = -1.,
                 time_span: Tuple[float, float] = (0.0, 5.0), dt: float = 0.05):
        """
        Initialize the class with the given velocity, time span, and time step.
        :param dim: The dimensionality of the system. For a 2D system, dim = 2.
        :param time_span: Tuple (t_start, t_end) specifying the time range for the trajectory.
        :param dt: The time step used for discretizing the trajectory.
        """
        # Initialize parent class with dt and time_span
        super().__init__(dt, time_span)
        self.time_span = time_span
        self.dt = dt
        if not alpha_init is None:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        else:
            self.alpha = -1.
        self.dim = dim

    def compute_trajectory(self, initial_position: torch.Tensor, time_span: Optional[Tuple[float,float]] = None) -> torch.Tensor:
        """
        Computes the trajectory using the analytical solutions for r(t) and theta(t).

        :param initial_position: A tensor of shape (batch_size, dim) representing the initial positions.
        :return: A tensor of shape (batch_size, T, dim) where T is the number of time steps, and each row is [x(t), y(t), ...].
        """
        # Ensure initial_position is of shape (batch_size, dim)
        if time_span is None:
            time_span = self.time_span
        batch_size = initial_position.shape[0]

        # Compute initial radius (r0) and angle (theta0) for the first 2 dimensions
        x0, y0 = initial_position[:, 0], initial_position[:, 1]  # Assuming 2D system for r and theta
        r0 = torch.sqrt(x0**2 + y0**2)  # Initial radius as a scalar for each sample in the batch
        theta0 = torch.atan2(y0, x0)    # Initial angle for each sample in the batch

        # Create time vector (shape: T)
        t_start, t_end = time_span
        t_values = torch.arange(t_start, t_end, self.dt).to(initial_position.device)  # Time steps

        # Expand t_values to (batch_size, T) for broadcasting
        t_values_expanded = t_values.unsqueeze(0).expand(batch_size, -1)

        # Compute the trajectory using the analytical solutions for r(t) and theta(t)
        r_t = (r0.unsqueeze(1) * torch.exp(- self.alpha * t_values.unsqueeze(0))) / \
      (1 + r0.unsqueeze(1) * (torch.exp(- self.alpha * t_values.unsqueeze(0)) - 1))
        theta_t = theta0.unsqueeze(1)  # theta(t)

        # Convert polar to Cartesian coordinates for the 2D part
        x_t = r_t * torch.cos(theta_t)
        y_t = r_t * torch.sin(theta_t)

        # Concatenate x_t and y_t to form the 2D trajectory
        trajectory = torch.stack([x_t, y_t], dim=2)  # Shape: (batch_size, T, 2)

        # Handle higher dimensions: attraction toward origin (-x) for residual dimensions
        if self.dim > 2:
            residual = initial_position[:, 2:]  # Get the residual higher dimensions (3D and beyond)
            
            # Dynamics for higher dimensions: attraction to origin (for negative alpha)
            residual_t = residual.unsqueeze(1) * torch.exp(self.alpha * t_values_expanded.unsqueeze(2))

            # Concatenate the 2D part with the higher-dimensional residual
            trajectory = torch.cat([trajectory, residual_t], dim=2)  # Shape: (batch_size, T, dim)

        return trajectory

#analytical sphere attractor
class AnalyticalSphereAttractor(AnalyticDynamicalSystem):
    """
    Computes the trajectory of an attractor system embedded in S^d inside R^D.
    The system has radial attraction toward the sphere and optional tangent dynamics.
    The dynamics in the residual dimensions are attracted toward the origin.
    """

    def __init__(
        self,
        dim: int,
        sphere_dim: int = 2,  # Default sphere_dim set to 2 for S^2 (3D sphere)
        radius: float = 1.0,
        alpha_init: float = -1.0,
        time_span: Tuple[float, float] = (0.0, 5.0),
        dt: float = 0.05,
    ):
        """
        Initializes the attractor system.

        :param dim: The total dimensionality of the system (R^D).
        :param sphere_dim: The dimension of the embedded sphere (S^d).
        :param radius: The radius of the sphere for attraction.
        :param alpha_init: The strength of radial attraction.
        :param time_span: The time range for the trajectory.
        :param dt: The time step for discretization.
        """
        super().__init__(dt, time_span)
        self.dim = dim
        self.sphere_dim = sphere_dim
        self.radius = radius
        self.alpha = alpha_init
        self.dt = dt

        if self.sphere_dim > 2:
            raise NotImplementedError(
                f"Spherical attractor only implemented for sphere_dim ≤ 2 (got {self.sphere_dim})."
            )

    def compute_trajectory(self, initial_position: torch.Tensor, time_span: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Computes the trajectory using the analytical solutions for the embedded sphere.

        :param initial_position: A tensor of shape (batch_size, dim) representing the initial positions.
        :return: A tensor of shape (batch_size, T, dim) where T is the number of time steps.
        """
        # Ensure initial_position is of shape (batch_size, dim)
        if time_span is None:
            time_span = self.time_span
        batch_size = initial_position.shape[0]

        # Extract sphere components (first `sphere_dim + 1` dimensions)
        x_sphere = initial_position[:, :self.sphere_dim + 1]
        x_residual = initial_position[:, self.sphere_dim + 1:] if self.dim > self.sphere_dim + 1 else None

        # Compute initial radial distance
        norm = torch.norm(x_sphere, dim=1, keepdim=True)
        r0 = norm  # Initial radius (scalar for each sample)
        
        # Create time vector (shape: T)
        t_start, t_end = time_span
        t_values = torch.arange(t_start, t_end, self.dt).to(initial_position.device)

        # Expand t_values to (batch_size, T) for broadcasting
        t_values_expanded = t_values.unsqueeze(0).expand(batch_size, -1)

        # Radial dynamics (attraction to the sphere)
        # Compute radial distance over time
        r_t = (r0.unsqueeze(1) * torch.exp(-self.alpha * t_values.unsqueeze(0))) / \
            (1 + r0.unsqueeze(1) * (torch.exp(-self.alpha * t_values.unsqueeze(0)) - 1))
        r_t = r_t.permute(0, 2, 1)  # Change the shape to (batch_size, T, 1)

        # Initialize the trajectory in spherical coordinates (r_t)
        trajectory_sphere = torch.zeros(batch_size, t_values_expanded.shape[1], self.sphere_dim + 1, device=initial_position.device)

        # For S^1 (2D sphere), calculate azimuthal angle (theta)
        if self.sphere_dim == 1:  # 2D sphere in 2D (circle)
            theta0 = torch.atan2(x_sphere[:, 1], x_sphere[:, 0])  # Azimuthal angle (in 2D)
            # Expand the angles across time
            theta_t = theta0.unsqueeze(1)

            # Update the trajectory with the spherical coordinates
            trajectory_sphere[:, 0, :] = r_t * torch.cos(theta_t)  # x-component
            trajectory_sphere[:, 1, :] = r_t * torch.sin(theta_t)  # y-component

        # For higher spheres (e.g., S^2 in 3D)
        elif self.sphere_dim == 2:  # S^2 (3D sphere)
            r0 = r0.squeeze()
            theta0 = torch.acos(x_sphere[:, 2] / r0)  # Polar angle

            norm_xy = torch.norm(x_sphere[:, :2], dim=1)  
            phi0 = torch.sign(x_sphere[:, 1]) * torch.acos(x_sphere[:, 0] / norm_xy)  

            theta0 = theta0.unsqueeze(1).unsqueeze(2)  # [B] → [B, 1, 1]
            phi0 = phi0.unsqueeze(1).unsqueeze(2)      # [B] → [B, 1, 1]
            # Update the trajectory with the spherical coordinates
            trajectory_sphere[:, :, 0] = (r_t * torch.sin(theta0) * torch.cos(phi0)).squeeze()  # x-component
            trajectory_sphere[:, :, 1] = (r_t * torch.sin(theta0) * torch.sin(phi0)).squeeze()  # y-component
            trajectory_sphere[:, :, 2] = (r_t * torch.cos(theta0)).squeeze()  # z-component


        # Concatenate the residual dynamics (if there are residual dimensions)
        if self.dim > self.sphere_dim + 1:
            residual = initial_position[:, self.sphere_dim + 1:]  # Higher dimensions
            residual_t = residual * torch.exp(self.alpha * t_values_expanded.unsqueeze(2))

            # Concatenate the sphere dynamics with the residual dynamics
            trajectory = torch.cat([trajectory_sphere, residual_t], dim=2)
        else:
            trajectory = trajectory_sphere

        return trajectory


#ABCA

class AnalyticalBoundedContinuousAttractor(AnalyticDynamicalSystem):
    def __init__(self, dim: int, bca_dim: int, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5), bounds: float = 1.0, alpha: float = -1.0):
        """
        A bounded continuous attractor with:
        - Zero flow inside [-1, 1]^bca_dim
        - Linear flow to nearest boundary outside for bca_dim dimensions
        - Exponential decay \(\dot{x} = \alpha x\) for dimensions dim - bca_dim

        Args:
            dim (int): Dimensionality of the system (total space)
            bca_dim (int): Dimensionality of the bounded continuous attractor
            dt (float): Time step for numerical integration
            time_span (Tuple[float, float]): Start and end times for trajectory
            bounds (float): The bounds of the attractor (typically [-1, 1] for each dimension)
            alpha (float): Decay rate for the dimensions outside the BCA
        """
        super().__init__()
        self.dim = dim
        self.bca_dim = bca_dim
        self.dt = dt
        self.time_span = time_span
        self.bounds = bounds
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the flow vector for each point x ∈ ℝ^{..., D}, where the first bca_dim dimensions
        form a bounded continuous attractor.

        Outside the box [-bounds, bounds]^bca_dim, flow pushes linearly toward the closest point on the box boundary.
        """
        flow = torch.zeros_like(x)

        x_bca = x[..., :self.bca_dim]
        x_proj = torch.clamp(x_bca, -self.bounds, self.bounds)
        bca_flow = x_proj - x_bca  # zero inside box, vector toward projection outside

        flow[..., :self.bca_dim] = bca_flow

        # Outside the attractor: exponential decay
        if self.bca_dim < self.dim:
            flow[..., self.bca_dim:] = self.alpha * x[..., self.bca_dim:]

        return flow


    def compute_trajectory(self, initial_position: torch.Tensor, time_span: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Computes the trajectory based on the analytical solution.

        Args:
            initial_position (torch.Tensor): (batch_size, dim)
            time_span (Optional[Tuple[float, float]]): Overrides the default time span

        Returns:
            torch.Tensor: (batch_size, T, dim)
        """
        if time_span is None:
            time_span = self.time_span
        t_start, t_end = time_span
        t_values = torch.arange(t_start, t_end, self.dt, device=initial_position.device)
        num_steps = len(t_values)

        batch_size, dim = initial_position.shape
        trajectory = torch.zeros(batch_size, num_steps, dim, device=initial_position.device)
        x = initial_position.clone()

        for i in range(num_steps):
            trajectory[:, i] = x
            dx = self.forward(x)
            x = x + self.dt * dx

        return trajectory




#### Dynamical system motif construction function
def build_ds_motif(
    ds_motif: Literal["lds", "bla", "ring", "lc", ],
    dim: int,
    time_span: tuple[float, float],
    dt: Optional[float] = None,
    analytic: bool = False,
    canonical: bool = True,
    vf_on_ring_enabled: bool = False,
    alpha_init: Optional[float] = -1.,
    velocity_init: Optional[float] = -1.,
) -> object:
    """
    Constructs a dynamical system motif based on the motif type and analytic/numerical type.
    Args:
        ds_motif: One of "ring", "lc", "lds", "bla".
            1. "ring": Ring attractor system.
            2. "lc": Limit cycle system.
            3. "lds": Linear dynamical system.
            4. "bla": Bounded line attractor system. 
        dim: Dimensionality of the system.
        time_span: Time interval for simulation.
        dt: Time step (required for learnable systems).
        analytic: Whether to use an analytical system.
        canonical: No time parametrization if True.
        vf_on_ring_enabled: Additional option for learnable ring attractor.
        alpha_init, velocity_init: Initial values for learnable parameters.
    Returns:
        Instantiated dynamical system object.
    """
    ds_class_map = {
        'ring': {
            True: AnalyticalRingAttractor,
            False: LearnableNDRingAttractor,
        },
        'lc': {
            True: AnalyticalLimitCycle,
            False: LearnableNDLimitCycle,
        },
        'lds': {
            True: AnalyticalLinearSystem,
            False: LearnableNDLinearSystem,
        },
        'bla': {
            True: AnalyticalBoundedLineAttractor,
            False: LearnableNDBoundedLineAttractor,
        },
    }

    if ds_motif not in ds_class_map:
        raise ValueError(f"Unknown ds_motif '{ds_motif}'. Valid options: {list(ds_class_map.keys())}")
    if analytic not in ds_class_map[ds_motif]:
        raise ValueError(f"No class defined for ds_motif='{ds_motif}' with analytic={analytic}")

    #DSClass = ds_class_map[ds_motif][analytic]
    # # Instantiate with appropriate parameters
    # if analytic:
    #     if canonical:
    #         ds = DSClass(dim=dim, dt=dt, time_span=time_span)
    #     else:
    #         ds = DSClass(dim=dim, dt=dt, time_span=time_span, alpha_init=alpha_init, velocity_init=velocity_init)
    # else:
    #     if canonical:
    #         ds = DSClass(dim=dim, dt=dt, time_span=time_span)
    #     else:
    #         ds = DSClass(dim=dim, dt=dt, time_span=time_span, vf_on_ring_enabled=vf_on_ring_enabled, alpha_init=alpha_init, velocity_init=velocity_init)

    # return ds

    DSClass = ds_class_map[ds_motif][analytic]
    init_params = inspect.signature(DSClass.__init__).parameters

    kwargs = {
        'dim': dim,
        'dt': dt,
        'time_span': time_span,
    }

    # Add only if not canonical
    if not canonical:
        if 'alpha_init' in init_params:
            kwargs['alpha_init'] = alpha_init
        if 'velocity_init' in init_params:
            kwargs['velocity_init'] = velocity_init
        if 'vf_on_ring_enabled' in init_params:
            kwargs['vf_on_ring_enabled'] = vf_on_ring_enabled

    return DSClass(**kwargs)





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
    def __init__(self, sigma: float = 10.0, beta: float = 12., rho: float = 65., dim: int = 3, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5), noise_std: float = 0.0) -> None:
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


#4D systems
class HodgkinHuxleySystem(DynamicalSystem):
    def __init__(self, 
                 I_ext: float = 10.0, 
                 dt: float = 0.01, 
                 time_span: Tuple[float, float] = (0, 50),
                 noise_std: float = 0.0) -> None:
        super().__init__(dim=4, dt=dt, time_span=time_span)
        self.I_ext = I_ext  # external current
        self.noise_std = noise_std
        
        # Parameters
        self.C_m = 1.0  # membrane capacitance, uF/cm^2
        self.g_Na = 120.0  # maximum sodium conductance, mS/cm^2
        self.g_K = 36.0    # maximum potassium conductance, mS/cm^2
        self.g_L = 0.3     # leak conductance, mS/cm^2
        self.E_Na = 50.0   # sodium reversal potential, mV
        self.E_K = -77.0   # potassium reversal potential, mV
        self.E_L = -54.387 # leak reversal potential, mV

    def alpha_m(self, V: torch.Tensor) -> torch.Tensor:
        return 0.1 * (25.0 - V) / (torch.exp((25.0 - V) / 10.0) - 1.0)

    def beta_m(self, V: torch.Tensor) -> torch.Tensor:
        return 4.0 * torch.exp(-V / 18.0)

    def alpha_h(self, V: torch.Tensor) -> torch.Tensor:
        return 0.07 * torch.exp(-V / 20.0)

    def beta_h(self, V: torch.Tensor) -> torch.Tensor:
        return 1.0 / (torch.exp((30.0 - V) / 10.0) + 1.0)

    def alpha_n(self, V: torch.Tensor) -> torch.Tensor:
        return 0.01 * (10.0 - V) / (torch.exp((10.0 - V) / 10.0) - 1.0)

    def beta_n(self, V: torch.Tensor) -> torch.Tensor:
        return 0.125 * torch.exp(-V / 80.0)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        State x: [V, m, h, n]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        V, m, h, n = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        # Ionic currents
        I_Na = self.g_Na * (m ** 3) * h * (V - self.E_Na)
        I_K = self.g_K * (n ** 4) * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        # dV/dt
        dVdt = (self.I_ext - I_Na - I_K - I_L) / self.C_m

        # Gating variables
        dm_dt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dh_dt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dn_dt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n

        dxdt = torch.stack([dVdt, dm_dt, dh_dt, dn_dt], dim=1)

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
    predefined_initial_conditions=None,
    seed: int = None
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

    if seed is not None:
        set_seed(seed)

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
    if time_span is None or len(time_span) == 0:
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


def generate_trajectories_for_training(
    homeo_net, 
    source_system, 
    initial_conditions_target,
    use_transformed_system=False,  
    noise_std=0.0
):
    """
    Generates trajectories for training the homeomorphism network.
    homeo_net: The homeomorphism network to be trained.
    source_system: The source dynamical system (e.g., LimitCycle, VanDerPol).
    initial_conditions_target: Initial conditions for the target system (e.g., LimitCycle, VanDerPol).
    if use_transformed_system use the vector field of the transformed system to generate trajectories.
    """
    initial_conditions_source = homeo_net.inverse(initial_conditions_target)

    if isinstance(source_system, AnalyticDynamicalSystem):
        # If the source system is analytical, use the method from the AnalyticDynamicalSystem class
        trajectories_source = source_system.compute_trajectory(initial_conditions_source)
        transformed_trajectories = [homeo_net(traj.requires_grad_()) for traj in trajectories_source]
        transformed_trajectories = torch.stack(transformed_trajectories)

    elif isinstance(source_system, LearnableDynamicalSystem):
        _, trajectories_source, _ = generate_trajectories(
            system=source_system, 
            predefined_initial_conditions=initial_conditions_source, 
            noise_std=noise_std
        )
        transformed_trajectories = [homeo_net(traj.requires_grad_()) for traj in trajectories_source]
        transformed_trajectories = torch.stack(transformed_trajectories)
        
    else:
        # If the source system is not learnable, use the existing method for generating trajectories
        if use_transformed_system:
            transformed_system = PhiSystemPhiInv(source_system, homeo_net)
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
            transformed_trajectories = [homeo_net(traj.requires_grad_()) for traj in trajectories_source]
            transformed_trajectories = torch.stack(transformed_trajectories)

    return transformed_trajectories



def generate_trajectories_from_initial_conditions(homeo_ds_network, initial_conditions_trg, time_span):
    homeo_net = homeo_ds_network.homeo_network
    source_system = homeo_ds_network.dynamical_system
    with torch.no_grad():
        initial_conditions_src = homeo_net.inverse(initial_conditions_trg)
        if isinstance(source_system, AnalyticDynamicalSystem):
            # If the source system is analytical, use the method from the AnalyticDynamicalSystem class
            trajectories_source = source_system.compute_trajectory(initial_conditions_src, time_span=time_span)
            transformed_trajectories = [homeo_net(traj) for traj in trajectories_source]
        else:
            _, trajectories_source, _ = generate_trajectories_scipy(system=source_system,predefined_initial_conditions=initial_conditions_src,time_span=time_span)
            transformed_trajectories = [homeo_net(traj) for traj in trajectories_source]
    transformed_trajectories = torch.stack(transformed_trajectories).detach().numpy()
    return trajectories_source, transformed_trajectories

def get_ds_vf(model, bounds, num_points=15):
    """
    Get the vector field of the system for a grid.
    """

    x = np.linspace(bounds[0], bounds[1], 15)
    y = np.linspace(bounds[0], bounds[1], 15)
    X, Y = np.meshgrid(x, y)
    XY = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)

    # Compute vector field
    with torch.no_grad():
        dXY = model.forward(t=torch.tensor(0.0), x=XY).numpy()

    U = dXY[:, 0].reshape(X.shape)
    V = dXY[:, 1].reshape(Y.shape)
    return X, Y, U, V