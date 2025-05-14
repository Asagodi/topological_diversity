import numpy as np
import torch

def hausdorff_distance(x_inv_man, y_inv_man) -> float:
    """
    Computes the symmetric Hausdorff distance between two point clouds.

    :param x_inv_man: points from the first invariant manifold, shape (B, dim)
    :param y_inv_man: points from the second invariant manifold, shape (B, dim)
    :return: Hausdorff distance (float)
    """
    # Convert to torch tensors if necessary
    if isinstance(x_inv_man, np.ndarray):
        x = torch.from_numpy(x_inv_man).float()
    else:
        x = x_inv_man.float()

    if isinstance(y_inv_man, np.ndarray):
        y = torch.from_numpy(y_inv_man).float()
    else:
        y = y_inv_man.float()

    # Compute pairwise distances
    x = x.unsqueeze(1)  # shape (B, 1, dim)
    y = y.unsqueeze(0)  # shape (1, B, dim)
    dists = torch.norm(x - y, dim=2)  # shape (B, B)

    # Directed Hausdorff distances
    x_to_y = dists.min(dim=1)[0].max()  # max over min distances from x to y
    y_to_x = dists.min(dim=0)[0].max()  # max over min distances from y to x

    return max(x_to_y.item(), y_to_x.item())
