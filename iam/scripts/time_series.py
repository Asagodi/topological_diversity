import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def find_period(X: np.ndarray, threshold: float = 3e-3, min_period: int = 50) -> int:
    x0 = X[-1]
    for t in range(min_period, X.shape[0]-min_period, 1):
        print(t, np.linalg.norm(X[t] - x0))
        if np.linalg.norm(X[t] - x0) < threshold:
            return t + min_period
    raise ValueError("Period not found.")


def find_period_local_min(X: np.ndarray, min_period: int = 10) -> int:
    x0 = X[0]
    distances = np.linalg.norm(X - x0, axis=1)
    
    # Invert distance to find minima using find_peaks
    peaks, _ = find_peaks(-distances)
    
    # Select first peak after min_period
    candidates = peaks[peaks > min_period]
    if len(candidates) == 0:
        raise ValueError("No recurrence detected.")
    
    best_t = candidates[0]
    return best_t



def resample_equal_arc_length(X: np.ndarray, num_points: int) -> np.ndarray:
    # Compute cumulative arc length
    deltas = np.linalg.norm(np.diff(X, axis=0), axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(deltas)])
    
    # Uniform arc length spacing
    uniform_arc = np.linspace(0, arc_lengths[-1], num_points)
    
    # Interpolate each dimension
    interpolators = [interp1d(arc_lengths, X[:, i]) for i in range(X.shape[1])]
    resampled = np.stack([f(uniform_arc) for f in interpolators], axis=1)
    
    return resampled



#centering and scaling
def normalize_scale_pair(trajectories_target_full, training_pairs=False):
    # Compute mean and std for normalization
    mean = trajectories_target_full.mean(dim=(0, 1), keepdim=True)
    std = trajectories_target_full.std(dim=(0, 1), keepdim=True)

    # Apply normalization
    trajectories_target_full = (trajectories_target_full - mean) / std

    if training_pairs:
        # Generate training pairs
        trajectories_target = make_transition_pairs(trajectories_target_full)
    else:
        # Use full trajectories
        trajectories_target = trajectories_target_full.clone()
    
    return trajectories_target_full, trajectories_target, mean, std



import numpy as np

def estimate_poincare_period_radians_np(
    trajectories: np.ndarray,
    eps: float = 1e-2,
    min_skip: int = 5,
) -> np.ndarray:
    """
    Estimate the recurrence period in radians using Poincaré recurrence (NumPy version).

    Args:
        trajectories: Array of shape (B, T, N).
        eps: Distance threshold for recurrence.
        min_skip: Minimum number of steps to skip after t0 to avoid trivial matches.

    Returns:
        periods_radians: Array of shape (B,) with estimated recurrence periods (in radians).
    """
    B, T, N = trajectories.shape
    x0 = trajectories[:, 0, :]  # shape (B, N)
    distances = np.linalg.norm(trajectories - x0[:, None, :], axis=2)  # shape (B, T)

    # Ignore first few steps to avoid trivial matches
    distances[:, :min_skip] = np.inf

    periods_steps = np.full(B, T, dtype=int)  # default to T (no recurrence found)

    for b in range(B):
        match = np.where(distances[b] < eps)[0]
        if len(match) > 0:
            periods_steps[b] = match[0]

    # Convert from steps to radians
    periods = periods_steps.astype(np.float32)
    periods_radians = (periods_steps.astype(np.float32) / (T - 1)) * (2 * np.pi)

    return periods, periods_radians
