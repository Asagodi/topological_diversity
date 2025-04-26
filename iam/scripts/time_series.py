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