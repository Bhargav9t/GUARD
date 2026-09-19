# ==============================================================================
# File: src/utils/seed.py
# Description: Deterministic Random Seed Manager for GUARD.
# Ensures full reproducibility across random number generators in Python,
# NumPy, PyTorch CPU, and PyTorch CUDA.
# ==============================================================================

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """
    Sets the random seed across all libraries to enforce reproducibility.

    Parameters:
        seed (int): The integer seed value to apply across all RNGs. Default is 42.
        deterministic (bool): If True, configures PyTorch and cuDNN to use
                              strictly deterministic algorithms. Note that this
                              may introduce a slight performance penalty on GPU.

    Returns:
        int: The seed that was set.
    """
    # 1. Set Python environment seed for hash-based operations (e.g. dictionary ordering)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Set Python standard library random module seed
    random.seed(seed)

    # 3. Set NumPy random generator seed for data preprocessing / transformations
    np.random.seed(seed)

    # 4. Set PyTorch CPU random seed
    torch.manual_seed(seed)

    # 5. Set PyTorch GPU random seed for current device and all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 6. Configure cuDNN deterministic execution flags
    if deterministic:
        # Ensures that cuDNN selects deterministic convolution algorithms
        torch.backends.cudnn.deterministic = True
        # Disables cuDNN auto-tuner which benchmarks multiple algorithms and picks the fastest,
        # which can result in non-deterministic floating point variations across runs
        torch.backends.cudnn.benchmark = False
    else:
        # If speed is prioritized over strict bitwise reproducibility, benchmark mode can be re-enabled
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return seed
