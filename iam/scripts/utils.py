import random
import torch
import numpy as np

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior in CNNs
    torch.backends.cudnn.benchmark = False  # May slow down training but ensures reproducibility