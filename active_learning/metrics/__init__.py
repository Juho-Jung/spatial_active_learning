#!/usr/bin/env python3
"""
Metrics module for SPARCL.

This module contains segmentation evaluation metrics (Dice, IoU, spatial
metrics, performance coverage). Detection metrics live in ``metrics.detection``.
"""

from .segmentation import (
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
