import random
import torch
import numpy as np

def set_seed(seed: int):
    """
    Set the seed for reproducibility across all random operations.

    :param seed: The seed value.
    """
    random.seed(seed)  # For Python's built-in random module
    np.random.seed(seed)  # For NumPy random
    torch.manual_seed(seed)  # For PyTorch CPU operations
    torch.cuda.manual_seed(seed)  # For PyTorch GPU operations (if using CUDA)
    torch.cuda.manual_seed_all(seed)  # For all GPUs (if using multiple GPUs)
    torch.backends.cudnn.deterministic = True  # Ensures determinism in CUDA
    torch.backends.cudnn.benchmark = False  # Disables benchmarking in CUDA (might affect performance)
    # If using a non-deterministic layer (e.g., dropout), you can use the following line:
    torch.use_deterministic_algorithms(True)  # For PyTorch 1.8 and later


def format_reg_name(reg: float) -> str:
    if reg == 0:
        return "0"
    elif reg >= 0.1:
        return f"{reg:.1f}"       # e.g., 1.0 → '1.0', 0.1 → '0.1'
    else:
        return f"{reg:.0e}"       # e.g., 0.001 → '1e-3'