"""
Common loss functions for segmentation tasks.
"""

import torch


def dice_loss(pred, target):
    """Dice loss for segmentation."""
    smooth = 1e-6
    pred = pred.view(-1)
    target = target.view(-1)

    # Clamp predictions to [0,1] range for safety
    pred = torch.clamp(pred, 0, 1)

    intersection = (pred * target).sum()
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance in segmentation."""
    smooth = 1e-6
    pred = pred.view(-1)
    target = target.view(-1)

    # Clamp predictions to [0,1] range for safety
    pred = torch.clamp(pred, 0, 1)

    # BCE loss
    bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')

    # Focal loss
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = alpha * (1 - pt) ** gamma
    focal_loss = focal_weight * bce

    return focal_loss.mean()


def segmentation_focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss specifically for segmentation tasks."""
    smooth = 1e-6
    pred = pred.view(-1)
    target = target.view(-1)

    # Clamp predictions to [0,1] range for safety
    pred = torch.clamp(pred, 0, 1)

    # BCE loss
    bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')

    # Focal loss
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = alpha * (1 - pt) ** gamma
    focal_loss = focal_weight * bce

    return focal_loss.mean()


def combo_loss(pred, target, dice_weight=0.4, bce_weight=0.3, focal_weight=0.3):
    """Combo Loss: Dice + BCE + Focal Loss combination for better segmentation."""
    dice = dice_loss(pred, target)
    bce = torch.nn.functional.binary_cross_entropy(pred, target)
    focal = focal_loss(pred, target, alpha=0.75, gamma=3.0)  # More aggressive focal loss
    return dice_weight * dice + bce_weight * bce + focal_weight * focal


def combined_loss(pred, target, dice_weight=0.2, focal_weight=0.8):
    """Combined Dice and Focal loss with better weighting."""
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target, alpha=0.5, gamma=2.0)  # Higher alpha for more aggressive learning
    return dice_weight * dice + focal_weight * focal


def weighted_bce_loss(pred, target, pos_weight=6.0):
    """Weighted BCE loss to handle class imbalance."""
    # pos_weight = negative_samples / positive_samples ≈ 6.0
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        pred, target, pos_weight=torch.tensor(pos_weight).to(pred.device))
    return bce
