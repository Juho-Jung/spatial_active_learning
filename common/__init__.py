"""
Common modules for segmentation tasks.

This package contains shared components for calcification and calcified nodule segmentation.
"""

from .datasets import SegmentationDataset
from .evaluation import evaluate_model, SegmentationEvaluator
from .losses import combined_loss, combo_loss, dice_loss, focal_loss, segmentation_focal_loss, weighted_bce_loss
from .metrics import calculate_metrics
from .models import create_model, SimpleUNet
from .training import train_model
from .utils import get_device, set_seeds, suppress_albumentations_warnings

__all__ = ['SimpleUNet', 'create_model',
           'dice_loss', 'focal_loss', 'combo_loss', 'combined_loss',
           'weighted_bce_loss', 'segmentation_focal_loss',
           'calculate_metrics',
           'SegmentationDataset',
           'SegmentationEvaluator', 'evaluate_model',
           'train_model',
           'set_seeds', 'get_device', 'suppress_albumentations_warnings']
