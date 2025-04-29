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
class LearnableLimitCycle(LearnableDynamicalSystem): #encorporated in LearnableNDLimitCycle, special case 2D
    """
    A simple limit cycle system with dynamics defined in polar coordinates.
    The system includes learnable velocity parameters. #alpha?
    """
    def __init__(self, dim: int = 2, dt: float = 0.05, time_span: Tuple[float, float] = (0, 5), noise_std: float = 0.0,
     radius: float = 1.0, velocity_init: float = 1.0): #, alpha_init: float = -1.0):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.time_span = time_span
        self.noise_std = noise_std
        # Initialize learnable parameters for velocity 
        self.velocity = nn.Parameter(torch.tensor(velocity_init, dtype=torch.float32))
        #self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        
        self.radius = 1.0  # Fixed radius for the limit cycle

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension
        
        # Extract the polar coordinates
        x_val, y_val = x[:, 0], x[:, 1]
        r = torch.sqrt(x_val**2 + y_val**2)
        theta = torch.atan2(y_val, x_val)

        # Define the dynamics in polar coordinates
        dtheta_dt = self.velocity  # Constant velocity
        #dr_dt =  self.alpha * r * (self.radius - r)  #  radial dynamics
        dr_dt =  - r * (self.radius - r)
        # Convert the dynamics to Euclidean coordinates
        dx_dt = dr_dt * torch.cos(theta) - r * torch.sin(theta) * dtheta_dt
        dy_dt = dr_dt * torch.sin(theta) + r * torch.cos(theta) * dtheta_dt
        
        return torch.stack([dx_dt, dy_dt], dim=1)
    

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
        #alphainit: float = -1.0,
    ):
        super().__init__()
        assert dim >= 2, "Dimension must be at least 2."
        self.dim = dim
        self.dt = dt
        self.time_span = time_span
        self.noise_std = noise_std
        self.radius = radius  # Fixed radius for the limit cycle

        # Learnable parameters
        self.velocity = nn.Parameter(torch.tensor(velocity_init, dtype=torch.float32))
        #self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension

        # Planar (r, theta) dynamics
        x_val, y_val = x[:, 0], x[:, 1]
        r = torch.sqrt(x_val**2 + y_val**2)
        theta = torch.atan2(y_val, x_val)

        dtheta_dt = self.velocity  # Learnable velocity
        #dr_dt =  - self.alpha * r * (self.radius - r)  # Radial dynamics
        dr_dt =   r * (self.radius - r)  # Radial dynamics


        # Convert polar derivatives back to Cartesian
        dx_dt = dr_dt * torch.cos(theta) - r * torch.sin(theta) * dtheta_dt
        dy_dt = dr_dt * torch.sin(theta) + r * torch.cos(theta) * dtheta_dt

        dx_dt = dx_dt.unsqueeze(1)
        dy_dt = dy_dt.unsqueeze(1)

        derivatives = [dx_dt, dy_dt]

        # Higher dimensions: attraction toward (0, 0) with the same alpha
        if self.dim > 2:
            residual = x[:, 2:]
            #d_residual_dt = self.alpha * residual
            d_residual_dt = - residual
            derivatives.append(d_residual_dt)

        return torch.cat(derivatives, dim=1)



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

    if isinstance(source_system, AnalyticDynamicalSystem):
        # If the source system is analytical, use the method from the AnalyticDynamicalSystem class
        trajectories_source = source_system.compute_trajectory(initial_conditions_source).requires_grad_()

        # Apply the diffeomorphism network to the trajectories generated by the analytical system
        transformed_trajectories = [diffeo_net(traj.requires_grad_()) for traj in trajectories_source]
        transformed_trajectories = torch.stack(transformed_trajectories)

    elif isinstance(source_system, LearnableDynamicalSystem):
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


import torch
import torch.nn as nn
from typing import Tuple

class AnalyticalLimitCycle(AnalyticDynamicalSystem):
    """
    Computes the trajectory of a limit cycle system defined by r_dot = r(r-1) and theta_dot = v.
    This class uses analytical solutions for r(t) and theta(t) based on the initial condition and velocity.
    """
    def __init__(self, velocity_init: float, alpha_init: float,  dim: int, 
                 time_span: Tuple[float, float] = (0.0, 5.0), dt: float = 0.05):
        """
        Initialize the class with the given velocity, time span, and time step.
        
        :param velocity_init: Initial velocity (v) in the angular direction (theta_dot = v).
        :param dim: The dimensionality of the system. For a 2D system, dim = 2.
        :param time_span: Tuple (t_start, t_end) specifying the time range for the trajectory.
        :param dt: The time step used for discretizing the trajectory.
        """
        # Initialize parent class with dt and time_span
        super().__init__(dt, time_span)
        self.time_span = time_span
        self.dt = dt
        # Make the velocity a learnable parameter
        self.velocity = nn.Parameter(torch.tensor(velocity_init, dtype=torch.float32))  # Learnable parameter
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.dim = dim

    def compute_trajectory(self, initial_position: torch.Tensor):
        """
        Computes the trajectory using the analytical solutions for r(t) and theta(t).

        :param initial_position: A tensor of shape (batch_size, dim) representing the initial positions.
        :return: A tensor of shape (batch_size, N, dim) where N is the number of time steps, and each row is [x(t), y(t), ...].
        """
        # Ensure initial_position is of shape (batch_size, dim)
        batch_size = initial_position.shape[0]

        # Compute initial radius (r0) and angle (theta0) for the first 2 dimensions
        x0, y0 = initial_position[:, 0], initial_position[:, 1]  # Assuming 2D system for r and theta
        r0 = torch.sqrt(x0**2 + y0**2)  # Initial radius as a scalar for each sample in the batch
        theta0 = torch.atan2(y0, x0)    # Initial angle for each sample in the batch

        # Create time vector (shape: N)
        t_start, t_end = self.time_span
        t_values = torch.arange(t_start, t_end, self.dt).to(initial_position.device)  # Time steps

        # Expand t_values to (batch_size, N) for broadcasting
        t_values_expanded = t_values.unsqueeze(0).expand(batch_size, -1)

        # Compute the trajectory using the analytical solutions for r(t) and theta(t)
        r_t = (r0.unsqueeze(1) * torch.exp(self.alpha * t_values.unsqueeze(0))) / \
      (1 + r0.unsqueeze(1) * (torch.exp(self.alpha * t_values.unsqueeze(0)) - 1))
        theta_t = self.velocity * t_values_expanded + theta0.unsqueeze(1)  # theta(t)

        # Convert polar to Cartesian coordinates for the 2D part
        x_t = r_t * torch.cos(theta_t)
        y_t = r_t * torch.sin(theta_t)

        # Concatenate x_t and y_t to form the 2D trajectory
        trajectory = torch.stack([x_t, y_t], dim=2)  # Shape: (batch_size, N, 2)

        # Handle higher dimensions: attraction toward origin (-x) for residual dimensions
        if self.dim > 2:
            residual = initial_position[:, 2:]  # Get the residual higher dimensions (3D and beyond)
            
            # Dynamics for higher dimensions: attraction to origin (-x)
            d_residual_dt = -residual  # As you mentioned, dot x = -x
            residual_t = residual.unsqueeze(1) + d_residual_dt.unsqueeze(1) * t_values_expanded.unsqueeze(2)  # Broadcast over time

            # Concatenate the 2D part with the higher-dimensional residual
            trajectory = torch.cat([trajectory, residual_t], dim=2)  # Shape: (batch_size, N, dim)

        return trajectory