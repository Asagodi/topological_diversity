import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel 


def get_random_vector_field(min_val_sim = 2, n_grid = 40, norm = 0.05, random_seed = 49):
    # Define the grid points
    Y, X = np.mgrid[-min_val_sim:min_val_sim:complex(0, n_grid), -min_val_sim:min_val_sim:complex(0, n_grid)]

    #set seed
    np.random.seed(random_seed)
    #generate random length scale
    length_scale = np.random.uniform(0.1, 1.0)
    #set kernel
    kernel = ConstantKernel(1.0, (1e-4, 1e1)) * RBF(length_scale, (1e-4, 1e1))
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    K = kernel(xy)
    #generate random vector field
    perturb_u = np.random.multivariate_normal(np.zeros(xy.shape[0]), K).reshape(X.shape)
    perturb_v = np.random.multivariate_normal(np.zeros(xy.shape[0]), K).reshape(X.shape)
    #scale + set norm
    magnitude =  np.sqrt(perturb_u**2 + perturb_v**2)
    perturb_u, perturb_v = norm*perturb_u/magnitude, norm*perturb_v/magnitude
    return perturb_u, perturb_v


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



