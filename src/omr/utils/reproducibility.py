"""Reproducibility utilities for deterministic training."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42, fast: bool = False) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed for all generators.
        fast: If True, enable cuDNN benchmark for faster training
              (slightly non-deterministic but ~20-30% faster on CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if fast:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
