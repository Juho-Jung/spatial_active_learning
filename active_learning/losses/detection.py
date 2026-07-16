#!/usr/bin/env python3
"""
Detection loss functions for object detection tasks.

This module contains loss functions for detection models including
Focal Loss, GIoU Loss, and combined detection losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss_for_detection(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss for object detection.
    
    Args:
        logits: Classification logits [N, num_classes]
        targets: Target class indices [N]
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
    
    Returns:
        Focal loss scalar
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def giou_loss(pred_boxes, target_boxes):
    """
    Generalized IoU (GIoU) Loss for bounding box regression.
    
    Args:
        pred_boxes: Predicted boxes [N, 4] in format [x1, y1, x2, y2]
        target_boxes: Target boxes [N, 4] in format [x1, y1, x2, y2]
    
    Returns:
        GIoU loss scalar
    """
    # Calculate intersection
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union = pred_area + target_area - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-8)
    
    # Calculate enclosing box
    x1_enclosing = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1_enclosing = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2_enclosing = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2_enclosing = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    
    enclosing_area = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing)
    
    # Calculate GIoU
    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-8)
    
    # Return GIoU loss (1 - GIoU)
    return (1 - giou).mean()


def detection_loss(predictions, targets):
    """
    Combined detection loss for training.
    
    This is a wrapper that works with torchvision detection models
    which already compute losses internally during training.
    
    Args:
        predictions: Model output dict with 'loss' keys (from torchvision models)
        targets: Target dicts (not used if predictions already contains losses)
    
    Returns:
        Total loss scalar
    """
    if isinstance(predictions, dict) and 'loss' in predictions:
        # Model already computed losses (torchvision models)
        return predictions['loss']
    else:
        # Fallback: compute loss manually if needed
        # This would require matching predictions to targets
        raise NotImplementedError("Manual detection loss computation not yet implemented")


def smooth_l1_loss_bbox(pred_boxes, target_boxes, beta=1.0):
    """
    Smooth L1 Loss for bounding box regression.
    
    Args:
        pred_boxes: Predicted boxes [N, 4]
        target_boxes: Target boxes [N, 4]
        beta: Threshold for smooth L1
    
    Returns:
        Smooth L1 loss scalar
    """
    diff = torch.abs(pred_boxes - target_boxes)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()
