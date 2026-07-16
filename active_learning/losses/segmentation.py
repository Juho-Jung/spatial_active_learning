#!/usr/bin/env python3
"""
Loss functions for SPARCL.
"""

import torch
import torch.nn as nn


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for segmentation."""
    pred = pred.view(-1)
    target = target.view(-1)

    pred = torch.clamp(pred, 0, 1)

    intersection = (pred * target).sum()
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance."""
    pred = pred.view(-1)
    target = target.view(-1)

    pred = torch.clamp(pred, 0, 1)

    bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = alpha * (1 - pt) ** gamma
    focal_loss = focal_weight * bce

    return focal_loss.mean()


def combo_loss(pred, target, dice_weight=0.4, bce_weight=0.3, focal_weight=0.3):
    """Combined loss: Dice + BCE + Focal."""
    dice = dice_loss(pred, target)
    bce = torch.nn.functional.binary_cross_entropy(pred, target)
    focal = focal_loss(pred, target, alpha=0.75, gamma=3.0)

    return dice_weight * dice + bce_weight * bce + focal_weight * focal


def dice_loss_continuous(pred, target, smooth=1e-6):
    """Continuous Dice loss for soft targets."""
    pred = pred.view(-1)
    target = target.view(-1)

    pred = torch.clamp(pred, 0, 1)

    intersection = (pred * target).sum()
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice


def bce_dice_with_soft_targets(pred, target, bce_weight=0.5, dice_weight=0.5):
    """BCE + Dice loss with softened labels."""
    bce = torch.nn.functional.binary_cross_entropy(pred, target)
    dice = dice_loss_continuous(pred, target)

    return bce_weight * bce + dice_weight * dice


def dice_bce_boundary_loss(pred, target, dice_weight=0.4, bce_weight=0.3, boundary_weight=0.3):
    """Dice + BCE + Boundary L1 loss."""
    dice = dice_loss(pred, target)
    bce = torch.nn.functional.binary_cross_entropy(pred, target)

    # Boundary loss (L1 on boundaries)
    pred_boundary = torch.abs(pred - 0.5)
    target_boundary = torch.abs(target - 0.5)
    boundary_loss = torch.nn.functional.l1_loss(pred_boundary, target_boundary)

    return dice_weight * dice + bce_weight * bce + boundary_weight * boundary_loss
