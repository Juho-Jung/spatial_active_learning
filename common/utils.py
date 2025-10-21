"""
Common utility functions for segmentation tasks.
"""

import os

import numpy as np
import torch


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id=0):
    """Get the device to use for training/inference."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    return device


def suppress_albumentations_warnings():
    """Suppress albumentations warnings."""
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
