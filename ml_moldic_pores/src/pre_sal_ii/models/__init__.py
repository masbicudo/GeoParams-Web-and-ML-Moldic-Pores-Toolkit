import torch
import numpy as np
import random
import os

def set_all_seeds(seed=42):
    random.seed(seed)                  # Python random
    np.random.seed(seed)               # NumPy
    torch.manual_seed(seed)            # CPU
    torch.cuda.manual_seed(seed)       # Current GPU
    torch.cuda.manual_seed_all(seed)   # All GPUs

    # Ensure deterministic behavior (slower but stable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Needed for full reproducibility (CUDA ≥ 10.2)
    torch.use_deterministic_algorithms(True)
    
