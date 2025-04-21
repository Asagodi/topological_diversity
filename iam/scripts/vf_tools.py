import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
import tqdm

def ra_ode(t, x):
    x1, x2 = x
    u = x1 * (1 - np.sqrt(x1**2 + x2**2))
    v = x2 * (1 - np.sqrt(x1**2 + x2**2))
    return np.array([u, v])

def vector_field_ode_full(t, x, full_grid_u, full_grid_v, interpol_param):
    x1, x2 = x

    if x1 < -min_val or x1 > min_val or x2 < -min_val or x2 > min_val:
        u = x1 * (1 - np.sqrt(x1**2 + x2**2))
        v = x2 * (1 - np.sqrt(x1**2 + x2**2))
        return np.array([u, v])
    else:
        u = full_grid_u([x2,x1]).item() 
        v = full_grid_v([x2,x1]).item() 
        return np.array([u, v])

def vector_field_ode(t, x, grid_u, grid_v, perturb_grid_u, perturb_grid_v, interpol_param):
    x1 = x[0]
    x2 = x[1]
    u = x1 * (1 - np.sqrt(x1**2 + x2**2))
    v = x2 * (1 - np.sqrt(x1**2 + x2**2))
    u = (1-interpol_param)*grid_u([x2,x1]).item() + interpol_param*perturb_grid_u([x2,x1]).item() 
    v = (1-interpol_param)*grid_v([x2,x1]).item() + interpol_param*perturb_grid_v([x2,x1]).item() 
    return np.array([u, v])



# 
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

def create_invariant_manifold(grid_u, grid_v, perturb_grid_u, perturb_grid_v, interpol_param, initial_conditions, nonlinearity_ode=vector_field_ode):
    Nsims = initial_conditions.shape[0]
    trajectories = simulate_network_ntimes(Nsims, grid_u, grid_v, perturb_grid_u, perturb_grid_v, interpol_param,
                                           nonlinearity_ode=nonlinearity_ode, 
                                           y0s=initial_conditions, maxT=5, tsteps=2001)
    inv_man = trajectories[:,1000,:]
    inv_man = np.vstack([inv_man, inv_man[0]])
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