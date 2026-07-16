#!/usr/bin/env python3
"""
Evaluation metrics for SPARCL.
"""

import torch


def calculate_metrics(pred, target):
    """Calculate segmentation metrics."""
    smooth = 1e-6

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    threshold = 0.5
    pred_binary = (pred > threshold).float()

    # Dice
    intersection = (pred_binary * target).sum()
    dice = (2 * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)

    # IoU
    union = pred_binary.sum() + target.sum() - intersection
    iou = intersection / (union + smooth)

    # Lesion-level sensitivity
    lesion_detected = (pred_binary * target).sum() > 0
    lesion_sensitivity = lesion_detected.float()

    # Image-level AUROC
    try:
        from sklearn.metrics import roc_auc_score
        if target_flat.sum() > 0 and target_flat.sum() < len(target_flat):
            auroc = roc_auc_score(target_flat.cpu().numpy(), pred_flat.detach().cpu().numpy())
        else:
            auroc = 0.5
    except:
        auroc = 0.5

    return dice.item(), iou.item(), lesion_sensitivity.item(), auroc


def divide_image_into_areas(image_size=(512, 512), grid_width=2, grid_height=3):
    """Divide chest X-ray image into 3x2 grid (6 areas total).
    Areas are indexed as follows:
    - Top-left: 0, Top-right: 1
    - Middle-left: 2, Middle-right: 3
    - Bottom-left: 4, Bottom-right: 5
    """
    h, w = image_size
    area_h, area_w = h // grid_height, w // grid_width

    areas = []
    for i in range(grid_height):
        for j in range(grid_width):
            y_start = i * area_h
            y_end = (i + 1) * area_h if i < grid_height - 1 else h
            x_start = j * area_w
            x_end = (j + 1) * area_w if j < grid_width - 1 else w
            areas.append((y_start, y_end, x_start, x_end, i * grid_width + j))

    return areas


def calculate_bin_dice_metrics(pred_masks, true_masks, areas, epsilon=1e-8):
    """
    Calculate bin-wise Dice metrics for spatial evaluation.

    Args:
        pred_masks: [B, H, W] predicted masks
        true_masks: [B, H, W] true masks
        areas: List of (y_start, y_end, x_start, x_end, area_idx) tuples defining areas
        epsilon: Small value to avoid division by zero

    Returns:
        dict with bin-wise metrics
    """
    batch_size = pred_masks.shape[0]
    num_areas = len(areas)

    # Calculate bin Dice for each area
    bin_dices = []

    for area_idx, (y_start, y_end, x_start, x_end, _) in enumerate(areas):
        area_pred = pred_masks[:, y_start:y_end, x_start:x_end]
        area_true = true_masks[:, y_start:y_end, x_start:x_end]

        # Calculate Dice for each sample in this area
        intersection = (area_pred * area_true).sum(dim=(1, 2))  # [B]
        union = area_pred.sum(dim=(1, 2)) + area_true.sum(dim=(1, 2))  # [B]

        # Avoid division by zero
        dice = (2.0 * intersection + epsilon) / (union + epsilon)  # [B]
        bin_dices.append(dice)

    # Stack to get [num_areas, batch_size]
    bin_dices = torch.stack(bin_dices, dim=0)  # [num_areas, B]

    # Calculate metrics
    metrics = {}

    # Average bin Dice across all areas and samples
    metrics['avg_bin_dice'] = bin_dices.mean().item()

    # Worst-bin Dice (minimum across areas, averaged across samples)
    worst_bin_dice = bin_dices.min(dim=0)[0]  # [B] - min across areas for each sample
    metrics['worst_bin_dice'] = worst_bin_dice.mean().item()

    # 10th-percentile-bin Dice
    bin_dices_flat = bin_dices.flatten()  # [num_areas * B]
    metrics['p10_bin_dice'] = torch.quantile(bin_dices_flat, 0.1).item()

    # Bin standard deviation
    bin_dice_means = bin_dices.mean(dim=1)  # [num_areas] - mean across samples for each area
    overall_mean = bin_dice_means.mean()  # scalar
    bin_std = torch.sqrt(((bin_dice_means - overall_mean) ** 2).mean())
    metrics['bin_std'] = bin_std.item()

    # Additional metrics for detailed analysis
    metrics['min_bin_dice'] = bin_dices.min().item()
    metrics['max_bin_dice'] = bin_dices.max().item()
    metrics['median_bin_dice'] = torch.median(bin_dices).item()

    return metrics


def calculate_performance_coverage(bin_dices_before, bin_dices_after, areas, debug=False):
    """
    Calculate Performance Coverage metric.

    Performance Coverage measures the spatial breadth of performance improvement:
    C_spatial_perf(t) = 1/K * sum(φ(ΔD_k)) where φ(u) = sqrt(u)
    ΔD_k = max(D_k_post - D_k_pre, 0)

    Args:
        bin_dices_before: [num_areas, batch_size] bin dice scores before training
        bin_dices_after: [num_areas, batch_size] bin dice scores after training
        areas: List of area tuples (for K calculation)
        debug: If True, print debug information

    Returns:
        float: Performance coverage score
    """
    num_areas = len(areas)

    if debug:
        print(f"🔍 Performance Coverage Debug:")
        print(f"   Before shape: {bin_dices_before.shape}")
        print(f"   After shape: {bin_dices_after.shape}")
        print(f"   Before mean: {bin_dices_before.mean().item():.4f}")
        print(f"   After mean: {bin_dices_after.mean().item():.4f}")

    # Calculate improvement for each area: ΔD_k = max(D_k_post - D_k_pre, 0)
    raw_improvement = bin_dices_after - bin_dices_before  # [num_areas, B]
    improvement = torch.clamp(raw_improvement, min=0.0)  # [num_areas, B]

    if debug:
        print(f"   Raw improvement mean: {raw_improvement.mean().item():.4f}")
        print(f"   Clamped improvement mean: {improvement.mean().item():.4f}")
        print(f"   Areas with improvement: {(improvement > 0).sum().item()}/{improvement.numel()}")

    # Apply square root function φ(u) = sqrt(u) to each improvement
    # Use a larger epsilon to avoid very small values after sqrt
    phi_improvement = torch.sqrt(improvement + 1e-6)  # [num_areas, B]

    # Average across samples for each area, then across areas
    area_avg_phi = phi_improvement.mean(dim=1)  # [num_areas]
    performance_coverage = area_avg_phi.mean().item()  # scalar

    # Scale the result to make it more meaningful (multiply by 10 for better visibility)
    performance_coverage = performance_coverage * 10.0

    if debug:
        print(f"   Area-wise phi means: {area_avg_phi.tolist()}")
        print(f"   Final performance coverage: {performance_coverage:.6f}")

    return performance_coverage


def calculate_spatial_consistency(bin_dices, areas):
    """
    Calculate Spatial Consistency metric.

    Measures how consistent the performance is across different spatial areas.
    Lower values indicate more consistent performance across areas.

    Args:
        bin_dices: [num_areas, batch_size] bin dice scores
        areas: List of area tuples

    Returns:
        float: Spatial consistency score (coefficient of variation)
    """
    # Calculate mean dice for each area across samples
    area_means = bin_dices.mean(dim=1)  # [num_areas]

    # Calculate coefficient of variation (std / mean)
    overall_mean = area_means.mean()
    overall_std = torch.sqrt(((area_means - overall_mean) ** 2).mean())

    if overall_mean > 1e-8:
        spatial_consistency = (overall_std / overall_mean).item()
    else:
        spatial_consistency = 0.0

    return spatial_consistency
