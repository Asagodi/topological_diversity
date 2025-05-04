import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
import tqdm
from sklearn.gaussian_process.kernels import RBF, ConstantKernel 
from typing import Optional

def ra_ode(t, x):
    x1, x2 = x
    u = x1 * (1 - np.sqrt(x1**2 + x2**2))
    v = x2 * (1 - np.sqrt(x1**2 + x2**2))
    return np.array([u, v])

# def vector_field_ode_full(t, x, full_grid_u, full_grid_v):
#     x1, x2 = x

#     if x1 < -min_val or x1 > min_val or x2 < -min_val or x2 > min_val:
#         u = x1 * (1 - np.sqrt(x1**2 + x2**2))
#         v = x2 * (1 - np.sqrt(x1**2 + x2**2))
#         return np.array([u, v])
#     else:
#         u = full_grid_u([x2,x1]).item() 
#         v = full_grid_v([x2,x1]).item() 
#         return np.array([u, v])

def vector_field_ode(t, x, grid_u, grid_v, perturb_grid_u, perturb_grid_v, interpol_param):
    x1, x2 = x
    u = x1 * (1 - np.sqrt(x1**2 + x2**2))
    v = x2 * (1 - np.sqrt(x1**2 + x2**2))
    u = (1-interpol_param)*grid_u([x2,x1]).item() + interpol_param*perturb_grid_u([x2,x1]).item() 
    v = (1-interpol_param)*grid_v([x2,x1]).item() + interpol_param*perturb_grid_v([x2,x1]).item() 
    return np.array([u, v])


def simulate_network_ntimes(Nsims, grid_u, grid_v, perturb_grid_u, perturb_grid_v, interpol_param, nonlinearity_ode=vector_field_ode, 
                            y0s=None, y0_dist="uniform", 
                            maxT=10, tsteps=2001, tau=1):
    t = np.linspace(0, maxT, tsteps)
    sols = np.zeros((Nsims, tsteps, 2))
    for ni in range(Nsims):
        if not np.any(y0s):
            if y0_dist=='uniform':
                y0 = np.random.uniform(-1,1,2)
            else:
                y0 = np.random.normal(0,1,2)
        else:
            y0 = y0s[ni]
        sol = solve_ivp(nonlinearity_ode, y0=y0,  t_span=[0,maxT],
                        args=tuple([grid_u, grid_v, perturb_grid_u, perturb_grid_v, interpol_param]),
                        dense_output=True)
        sols[ni,...] = sol.sol(t).T.copy()
    return sols

def create_invariant_manifold(grid_u, grid_v, perturb_grid_u, perturb_grid_v, initial_conditions, interpol_param=1, nonlinearity_ode=vector_field_ode):
    Nsims = initial_conditions.shape[0]

    trajectories = simulate_network_ntimes(Nsims, grid_u, grid_v, perturb_grid_u, perturb_grid_v, interpol_param,
                                           nonlinearity_ode=nonlinearity_ode, 
                                           y0s=initial_conditions, maxT=5, tsteps=2001)
    inv_man = trajectories[:,1000,:]
    inv_man = np.vstack([inv_man, inv_man[0]])
    return inv_man

def create_invariant_manifold_fromring(grid_u, grid_v, perturb_grid_u, perturb_grid_v, interpol_param=1, num_points=100, nonlinearity_ode=vector_field_ode,
                                       radius = 1.):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    initial_conditions = np.array([(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles])
    inv_man = create_invariant_manifold(grid_u, grid_v, perturb_grid_u, perturb_grid_v, initial_conditions=initial_conditions,
                                         interpol_param=interpol_param, nonlinearity_ode=vector_field_ode)
    return inv_man


def create_invman_sequence(perturb_grid_u, perturb_grid_v, n_pert_steps, initial_conditions, num_points,nonlinearity_ode=vector_field_ode):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    ring = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])
    ring = np.vstack([ring, ring[0]])

    inv_mans = np.zeros((n_pert_steps+1, num_points+1, 2))
    inv_mans[0,...] = ring

    for i in tqdm.tqdm(range(1, n_pert_steps+1)):
        try:
            inv_man = create_invariant_manifold(perturb_grid_u, perturb_grid_v, 1./n_pert_steps*i, initial_conditions,nonlinearity_ode=nonlinearity_ode)
            inv_mans[i,...] = inv_man.copy()
        except:
            0
    return inv_mans


def initialize_grids(U, V, U_pert, V_pert, min_val_sim, n_grid):
    grid_u = RegularGridInterpolator((np.linspace(-min_val_sim, min_val_sim, n_grid), np.linspace(-min_val_sim, min_val_sim, n_grid)), U)
    grid_v = RegularGridInterpolator((np.linspace(-min_val_sim, min_val_sim, n_grid), np.linspace(-min_val_sim, min_val_sim, n_grid)), V)

    perturb_grid_u = RegularGridInterpolator((np.linspace(-min_val_sim, min_val_sim, n_grid), np.linspace(-min_val_sim, min_val_sim, n_grid)), U_pert)
    perturb_grid_v = RegularGridInterpolator((np.linspace(-min_val_sim, min_val_sim, n_grid), np.linspace(-min_val_sim, min_val_sim, n_grid)), V_pert)

    full_grid_u = RegularGridInterpolator((np.linspace(-min_val_sim, min_val_sim, n_grid), np.linspace(-min_val_sim, min_val_sim, n_grid)), U + U_pert)
    full_grid_v = RegularGridInterpolator((np.linspace(-min_val_sim, min_val_sim, n_grid), np.linspace(-min_val_sim, min_val_sim, n_grid)), V + V_pert) 

    return grid_u, grid_v, perturb_grid_u, perturb_grid_v, full_grid_u, full_grid_v


def get_all(U, V, random_seed, min_val_sim=3, n_grid=40, norm=0.05, n_pert_steps=15, add_limit_cycle=False, radius=1, num_points=100):
    print("Creating grid and vector field...")
    U_pert, V_pert, perturb_u, perturb_v = get_random_vector_field_from_ringattractor(min_val_sim=min_val_sim, n_grid=n_grid, norm=norm, random_seed=random_seed, add_limit_cycle=add_limit_cycle)
    # Create interpolation functions for the vector field

    print("Creating grids for ODE")
    grid_u, grid_v, perturb_grid_u, perturb_grid_v, full_grid_u, full_grid_v = initialize_grids(U, V, U_pert, V_pert, min_val_sim, n_grid)

    print("Calculating invariant manifolds...")
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    initial_conditions = np.array([(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles])
    inv_mans = create_invman_sequence(perturb_grid_u, perturb_grid_v, n_pert_steps, initial_conditions, num_points)

    return U_pert, V_pert, perturb_u, perturb_v,  grid_u, grid_v, perturb_grid_u, perturb_grid_v, full_grid_u, full_grid_v, inv_mans


def get_random_vector_field_from_ringattractor(min_val_sim = 2, n_grid = 40, norm = 0.05, random_seed = 49, add_limit_cycle=False):
    # Define the grid points
    Y, X = np.mgrid[-min_val_sim:min_val_sim:complex(0, n_grid), -min_val_sim:min_val_sim:complex(0, n_grid)]

    #Ring attractor vector field
    U = X * (1- np.sqrt(X**2 + Y**2))
    V = Y * (1- np.sqrt(X**2 + Y**2))
    #speed = np.sqrt(U*U + V*V)

    #set seed
    np.random.seed(random_seed)
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    #generate random length scale
    length_scale = np.random.uniform(0.1, 1.0)
    #set kernel
    kernel = ConstantKernel(1.0, (1e-4, 1e1)) * RBF(length_scale, (1e-4, 1e1))
    K = kernel(xy)
    #generate random vector field
    perturb_u = np.random.multivariate_normal(np.zeros(xy.shape[0]), K).reshape(X.shape)
    perturb_v = np.random.multivariate_normal(np.zeros(xy.shape[0]), K).reshape(X.shape)
    #scale + set norm

    if not add_limit_cycle:
        magnitude =  np.sqrt(perturb_u**2 + perturb_v**2)
        perturb_u, perturb_v = norm*perturb_u/magnitude, norm*perturb_v/magnitude
        
    else: # add limit cycle
        perturb_u -= Y
        perturb_v += X

        magnitude =  np.sqrt(perturb_u**2 + perturb_v**2)
        perturb_u, perturb_v = norm*perturb_u/magnitude, norm*perturb_v/magnitude

    U_pert = U + perturb_u
    V_pert = V + perturb_v
    return U_pert, V_pert, perturb_u, perturb_v


def vector_field_ode(t, x, grid_u, grid_v, perturb_grid_u, perturb_grid_v, interpol_param):
    x = np.ravel(x)  # Ensure x is a flat array
    if x.shape[0] != 2:
        raise ValueError(f"Expected x to have shape (2,), but got {x.shape}")
    x1, x2 = x[0], x[1]

    u = x1 * (1 - np.sqrt(x1**2 + x2**2))
    v = x2 * (1 - np.sqrt(x1**2 + x2**2))
    u = (1-interpol_param)*grid_u([x2,x1]).item() + interpol_param*perturb_grid_u([x2,x1]).item() 
    v = (1-interpol_param)*grid_v([x2,x1]).item() + interpol_param*perturb_grid_v([x2,x1]).item() 
    return np.array([u, v])




def prepare_initial_conditions(
    mode: str = "around_ring",  # "random", "around_ring", "invariant_manifold"
    num_points: int = 60,
    radius: float = 1.0,
    margin: float = 0.1,
    min_val: float = 1.5,
    invariant_manifold: Optional[np.ndarray] = None,
    seed: int = 42
) -> np.ndarray:
    """
    Prepares initial conditions for simulating target trajectories.

    Args:
        mode: "random", "around_ring", or "invariant_manifold".
        num_points: Number of points to generate.
        radius: Radius for the ring (only in "around_ring").
        margin: Perturbation inward/outward from ring (only in "around_ring").
        min_val: Bound for uniform sampling (only in "random").
        invariant_manifold: Data for initialization if using "invariant_manifold".
        seed: Random seed for reproducibility.

    Returns:
        Initial conditions array of shape (num_points, 2).
    """
    np.random.seed(seed)

    if mode == "random":
        return np.random.uniform(-min_val, min_val, size=(num_points, 2))

    elif mode == "around_ring":
        if num_points % 2 != 0:
            raise ValueError("num_points must be even in 'around_ring' mode.")
        half = num_points // 2
        angles = np.linspace(0, 2 * np.pi, half, endpoint=False)
        inner = [(radius - margin) * np.array([np.cos(θ), np.sin(θ)]) for θ in angles]
        outer = [(radius + margin) * np.array([np.cos(θ), np.sin(θ)]) for θ in angles]
        points = []
        for i in range(half):
            points.append(inner[i])
            points.append(outer[i])
        return np.array(points)

    elif mode == "invariant_manifold":
        if invariant_manifold is None:
            raise ValueError("Must provide invariant_manifold array for this mode.")
        return invariant_manifold

    else:
        raise ValueError(f"Unknown mode: {mode}")

def build_perturbed_ringattractor(perturbation_norm = 0.1, random_seed = 313, min_val_sim=3, n_grid = 40, add_limit_cycle=False,
                                  num_points_invman = 200, maxT = 5, tsteps = 100, number_of_target_trajectories = 100, initial_conditions_mode="around_ring", init_margin=0.1):
    """
    Build and simulate a perturbed ring attractor and approximate its invariant manifold.
    """
    Y, X = np.mgrid[-min_val_sim:min_val_sim:complex(0, n_grid), -min_val_sim:min_val_sim:complex(0, n_grid)]
    U, V = X * (1- np.sqrt(X**2 + Y**2)), Y * (1- np.sqrt(X**2 + Y**2))
    #U_pert, V_pert, perturb_u, perturb_v,  grid_u, grid_v, perturb_grid_u, perturb_grid_v, full_grid_u, full_grid_v, inv_mans = get_all(U, V, random_seed, norm=norm, n_pert_steps=15*2)

    U_pert, V_pert, _, _ = get_random_vector_field_from_ringattractor(min_val_sim=min_val_sim, n_grid=n_grid, norm=perturbation_norm, random_seed=random_seed, add_limit_cycle=add_limit_cycle)
    grid_u, grid_v, perturb_grid_u, perturb_grid_v, full_grid_u, full_grid_v = initialize_grids(U, V, U_pert, V_pert, min_val_sim, n_grid)
    inv_man = create_invariant_manifold_fromring(grid_u, grid_v, perturb_grid_u, perturb_grid_v, num_points=num_points_invman, nonlinearity_ode=vector_field_ode)

    #tsteps = maxT*20
    initial_conditions_pertring = prepare_initial_conditions(mode=initial_conditions_mode, invariant_manifold=inv_man, num_points=number_of_target_trajectories, margin=init_margin)
    number_of_target_trajectories = initial_conditions_pertring.shape[0]
    trajectories_pertring = simulate_network_ntimes(number_of_target_trajectories, grid_u, grid_v, perturb_grid_u, perturb_grid_v, 1, nonlinearity_ode=vector_field_ode, 
                                            y0s=initial_conditions_pertring, maxT=maxT, tsteps=tsteps)

    return X, Y, U_pert, V_pert, grid_u, grid_v, perturb_grid_u, perturb_grid_v, full_grid_u, full_grid_v, inv_man, trajectories_pertring