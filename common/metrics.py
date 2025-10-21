"""
Common metrics calculation functions for segmentation tasks.
"""

import torch


def calculate_metrics(pred, target):
    """Calculate Dice, IoU, lesion-level sensitivity, and image-level AUROC metrics."""

    smooth = 1e-6

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    threshold = 0.5  # Fixed threshold
    pred_binary = (pred > threshold).float()

    # Dice
    intersection = (pred_binary * target).sum()
    dice = (2 * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)

    # IoU
    union = pred_binary.sum() + target.sum() - intersection
    iou = intersection / (union + smooth)

    # Lesion-level sensitivity (True Positive Rate)
    # A lesion is detected if any pixel in the prediction overlaps with the ground truth
    lesion_detected = (pred_binary * target).sum() > 0
    lesion_sensitivity = lesion_detected.float()

    # Image-level AUROC: predict if image contains any positive pixels
    try:
        from sklearn.metrics import roc_auc_score

        # Convert to image-level predictions
        # Image is positive if it contains any positive pixels
        image_pred_scores = pred.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]  # [batch_size]
        image_targets = (target.sum(dim=(1, 2, 3)) > 0).float()  # [batch_size]

        # Calculate AUROC for image-level classification
        if image_targets.sum() > 0 and image_targets.sum() < len(image_targets):
            auroc = roc_auc_score(image_targets.cpu().numpy(), image_pred_scores.detach().cpu().numpy())
        else:
            auroc = 0.5  # Default when no variation

    except Exception as e:
        auroc = 0.5  # Fallback value

    return {'dice': dice.item(),
            'iou': iou.item(),
            'lesion_sensitivity': lesion_sensitivity.item(),
            'auroc': auroc}
