#!/usr/bin/env python3
"""
Sample selection module for SAM adaptation project.

This module contains various sample selection strategies for active learning
including random, uncertainty-based, area-based, and diversity-based selection methods.
"""

from .sample_selection import (create_spatial_bins, get_selection_strategy,
                               select_samples_adaptive,
                               select_samples_area_random,
                               select_samples_diversity,
                               select_samples_diversity_uncertainty,
                               select_samples_random,
                               select_samples_uncertainty,
                               select_samples_uncertainty_area)
from .sample_selection_ultra import (
    select_samples_adaptive_improved_ultra,
    select_samples_adaptive_performance_monitoring_ultra,
    select_samples_adaptive_ultra, select_samples_multi_scale_hybrid_ultra)

__all__ = [
    'create_spatial_bins',
    'select_samples_random',
    'select_samples_uncertainty',
    'select_samples_area_random',
    'select_samples_uncertainty_area',
    'select_samples_adaptive',
    'select_samples_diversity',
    'select_samples_diversity_uncertainty',
    'get_selection_strategy'
]
