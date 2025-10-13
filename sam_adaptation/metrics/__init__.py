#!/usr/bin/env python3
"""
Metrics module for SAM adaptation project.

This module contains evaluation metrics for segmentation tasks including
Dice score, IoU, spatial metrics, and performance coverage calculations.
"""

from .metrics import (
    calculate_metrics,
    divide_image_into_areas,
    calculate_bin_dice_metrics,
    calculate_performance_coverage,
    calculate_spatial_consistency
)

__all__ = [
    'calculate_metrics',
    'divide_image_into_areas',
    'calculate_bin_dice_metrics',
    'calculate_performance_coverage',
    'calculate_spatial_consistency'
]
