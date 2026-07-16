#!/usr/bin/env python3
"""
Loss functions module for SPARCL.

This module contains segmentation loss functions (Dice, Focal, and combined
losses). Detection losses live in ``losses.detection``.
"""

from .segmentation import (
    dice_loss,
    focal_loss,
    combo_loss,
    dice_loss_continuous,
    bce_dice_with_soft_targets,
    dice_bce_boundary_loss
)

__all__ = [
    'dice_loss',
    'focal_loss',
    'combo_loss',
    'dice_loss_continuous',
    'bce_dice_with_soft_targets',
    'dice_bce_boundary_loss'
]
