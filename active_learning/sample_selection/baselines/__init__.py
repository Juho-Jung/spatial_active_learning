#!/usr/bin/env python3
"""
Baseline active-learning selection strategies for SPARCL comparisons.

Each module implements a published AL acquisition strategy used as a baseline
against SPARCL: Coreset (ICLR 2018), Learning Loss / LUNIT (CVPR 2019),
ALUNET (MIDL 2024), and TAUDIS (ICCV 2023).
"""

from .alunet import select_samples_mcd_alunet, select_samples_usimc
from .coreset import select_samples_coreset
from .lunit import select_samples_lunit
from .taudis import select_samples_TAUDIS

__all__ = [
    'select_samples_coreset',
    'select_samples_lunit',
    'select_samples_mcd_alunet',
    'select_samples_usimc',
    'select_samples_TAUDIS',
]
