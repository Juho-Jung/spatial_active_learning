#!/usr/bin/env python3
"""
Sample selection module for SPARCL.

This module contains various sample selection strategies for active learning
including random, uncertainty-based, area-based, and diversity-based selection
methods, plus the SPARCL acquisition and its ablations.
"""

from .sample_selection import (SELECTION_STRATEGIES, create_spatial_bins,
                               get_selection_strategy, select_samples_adaptive,
                               select_samples_area_random,
                               select_samples_diversity,
                               select_samples_diversity_uncertainty,
                               select_samples_random,
                               select_samples_sparcl,
                               select_samples_sparcl_fixed_lambda,
                               select_samples_sparcl_no_coverage,
                               select_samples_sparcl_no_gating,
                               select_samples_uncertainty,
                               select_samples_uncertainty_area)
from .baselines import (select_samples_coreset, select_samples_lunit,
                        select_samples_mcd_alunet, select_samples_TAUDIS,
                        select_samples_usimc)

__all__ = [
    'create_spatial_bins',
    'select_samples_random',
    'select_samples_uncertainty',
    'select_samples_area_random',
    'select_samples_uncertainty_area',
    'select_samples_adaptive',
    'select_samples_diversity',
    'select_samples_diversity_uncertainty',
    'select_samples_coreset',
    'select_samples_TAUDIS',
    'select_samples_usimc',
    'select_samples_mcd_alunet',
    'select_samples_lunit',
    'get_selection_strategy',
    'SELECTION_STRATEGIES'
]
