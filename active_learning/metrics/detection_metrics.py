#!/usr/bin/env python3
"""
Detection metrics for object detection tasks.

This module contains evaluation metrics for detection models including
mAP (mean Average Precision), AP50, FROC, and other detection-specific metrics.
"""

import numpy as np
import torch


def calculate_bbox_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.

    Args:
        box1: Box [4] in format [x1, y1, x2, y2]
        box2: Box [4] in format [x1, y1, x2, y2]

    Returns:
        IoU scalar
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-8)


def _collect_all_detections(predictions, targets, iou_threshold=0.5):
    """
    Collect all predictions and targets across images for proper mAP calculation.

    Returns:
        all_scores: List of prediction confidence scores
        all_tp: List of True Positive flags (1 if matched, 0 if FP)
        total_gt: Total number of ground truth boxes
        matched_ious: List of IoUs for matched predictions
    """
    all_scores = []
    all_tp = []
    total_gt = 0
    matched_ious = []

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else np.array(pred['boxes'])
        pred_scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else np.array(pred['scores'])
        target_boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else np.array(target['boxes'])

        # Handle empty arrays
        if len(pred_boxes.shape) == 1 and len(pred_boxes) == 0:
            pred_boxes = np.zeros((0, 4))
        if len(target_boxes.shape) == 1 and len(target_boxes) == 0:
            target_boxes = np.zeros((0, 4))

        total_gt += len(target_boxes)

        if len(pred_boxes) == 0:
            continue

        if len(target_boxes) == 0:
            # All predictions are false positives
            all_scores.extend(pred_scores.tolist())
            all_tp.extend([0] * len(pred_scores))
            continue

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(target_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, target_box in enumerate(target_boxes):
                iou_matrix[i, j] = calculate_bbox_iou(pred_box, target_box)

        # Match predictions to targets (greedy matching by score)
        matched_targets = set()
        sorted_indices = np.argsort(pred_scores)[::-1]

        for pred_idx in sorted_indices:
            score = pred_scores[pred_idx]
            all_scores.append(score)

            best_iou = 0.0
            best_target_idx = -1

            for target_idx in range(len(target_boxes)):
                if target_idx in matched_targets:
                    continue
                iou = iou_matrix[pred_idx, target_idx]
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_target_idx = target_idx

            if best_target_idx >= 0:
                matched_targets.add(best_target_idx)
                all_tp.append(1)
                matched_ious.append(best_iou)
            else:
                all_tp.append(0)

    return all_scores, all_tp, total_gt, matched_ious


def calculate_ap_from_pr(precisions, recalls):
    """
    Calculate Average Precision using 101-point interpolation (COCO style).

    Args:
        precisions: List of precision values
        recalls: List of recall values

    Returns:
        AP value
    """
    if len(precisions) == 0 or len(recalls) == 0:
        return 0.0

    # Sort by recall
    sorted_indices = np.argsort(recalls)
    precisions = np.array(precisions)[sorted_indices]
    recalls = np.array(recalls)[sorted_indices]

    # Add sentinel values
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 101-point interpolation (COCO style)
    recall_thresholds = np.linspace(0, 1, 101)
    interpolated_precisions = []

    for t in recall_thresholds:
        # Find the precision at recall >= t
        mask = recalls >= t
        if mask.any():
            interpolated_precisions.append(precisions[mask].max())
        else:
            interpolated_precisions.append(0.0)

    return np.mean(interpolated_precisions)


def calculate_ap_at_iou(predictions, targets, iou_threshold):
    """
    Calculate AP at a specific IoU threshold.

    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        iou_threshold: IoU threshold for matching

    Returns:
        AP value at the specified IoU threshold
    """
    all_scores, all_tp, total_gt, _ = _collect_all_detections(
        predictions, targets, iou_threshold)

    if total_gt == 0:
        return 1.0 if len(all_scores) == 0 else 0.0

    if len(all_scores) == 0:
        return 0.0

    # Sort by score (descending)
    sorted_indices = np.argsort(all_scores)[::-1]
    all_tp = np.array(all_tp)[sorted_indices]

    # Calculate cumulative TP and FP
    cum_tp = np.cumsum(all_tp)
    cum_fp = np.cumsum(1 - all_tp)

    # Calculate precision and recall at each threshold
    precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
    recalls = cum_tp / total_gt

    return calculate_ap_from_pr(precisions, recalls)


def calculate_froc(predictions, targets, num_images, iou_threshold=0.5,
                   fp_rates=[0.125, 0.25, 0.5, 1, 2, 4, 8]):
    """
    Calculate Free-Response ROC (FROC) metrics.

    FROC measures sensitivity at different false positive rates per image,
    which is commonly used in medical imaging for lesion detection.

    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        num_images: Total number of images
        iou_threshold: IoU threshold for matching
        fp_rates: List of false positive rates per image to evaluate

    Returns:
        dict with FROC metrics:
            - froc_score: Average sensitivity across FP rates (standard FROC)
            - sensitivities: Dict mapping FP rate to sensitivity
            - auc_froc: Area under FROC curve
    """
    all_scores, all_tp, total_gt, _ = _collect_all_detections(
        predictions, targets, iou_threshold)

    if total_gt == 0:
        return {
            'froc_score': 1.0 if len(all_scores) == 0 else 0.0,
            'sensitivities': {rate: 1.0 if len(all_scores) == 0 else 0.0 for rate in fp_rates},
            'auc_froc': 1.0 if len(all_scores) == 0 else 0.0
        }

    if len(all_scores) == 0:
        return {
            'froc_score': 0.0,
            'sensitivities': {rate: 0.0 for rate in fp_rates},
            'auc_froc': 0.0
        }

    # Sort by score (descending)
    sorted_indices = np.argsort(all_scores)[::-1]
    all_tp = np.array(all_tp)[sorted_indices]
    all_scores = np.array(all_scores)[sorted_indices]

    # Calculate cumulative TP and FP
    cum_tp = np.cumsum(all_tp)
    cum_fp = np.cumsum(1 - all_tp)

    # Calculate sensitivity and FP per image at each threshold
    sensitivities = cum_tp / total_gt
    fp_per_image = cum_fp / num_images

    # Get sensitivity at specified FP rates
    sens_at_fp = {}
    for fp_rate in fp_rates:
        # Find the maximum sensitivity where FP per image <= fp_rate
        valid_indices = fp_per_image <= fp_rate
        if valid_indices.any():
            sens_at_fp[fp_rate] = sensitivities[valid_indices].max()
        else:
            sens_at_fp[fp_rate] = 0.0

    # FROC score: average sensitivity at standard FP rates
    froc_score = np.mean(list(sens_at_fp.values()))

    # Calculate AUC-FROC (area under FROC curve up to max FP rate)
    max_fp_rate = max(fp_rates)
    valid_mask = fp_per_image <= max_fp_rate
    if valid_mask.any():
        valid_fp = fp_per_image[valid_mask]
        valid_sens = sensitivities[valid_mask]
        # Add starting point
        valid_fp = np.concatenate([[0], valid_fp])
        valid_sens = np.concatenate([[0], valid_sens])
        # Calculate AUC using trapezoidal rule
        auc_froc = np.trapz(valid_sens, valid_fp) / max_fp_rate
    else:
        auc_froc = 0.0

    return {
        'froc_score': froc_score,
        'sensitivities': sens_at_fp,
        'auc_froc': auc_froc
    }


def calculate_detection_metrics(predictions, targets, iou_threshold=0.5):
    """
    Calculate detection metrics: mAP, AP50, FROC, precision, recall, etc.

    Args:
        predictions: List of prediction dicts, each with 'boxes', 'labels', 'scores'
        targets: List of target dicts, each with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching predictions to targets

    Returns:
        dict with metrics: mAP, AP50, FROC, precision, recall, avg_iou, etc.
    """
    num_images = len(predictions)

    # Collect all detections for proper mAP calculation
    all_scores, all_tp, total_gt, matched_ious = _collect_all_detections(
        predictions, targets, iou_threshold)

    # Calculate per-image metrics for simple precision/recall
    all_precisions = []
    all_recalls = []
    all_ious = []
    all_f1_scores = []

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else np.array(pred['boxes'])
        pred_scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else np.array(pred['scores'])
        target_boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else np.array(target['boxes'])

        # Handle empty arrays
        if len(pred_boxes.shape) == 1 and len(pred_boxes) == 0:
            pred_boxes = np.zeros((0, 4))
        if len(target_boxes.shape) == 1 and len(target_boxes) == 0:
            target_boxes = np.zeros((0, 4))

        # Filter predictions by score threshold
        score_threshold = 0.1
        if len(pred_scores) > 0:
            valid_preds = pred_scores >= score_threshold
            pred_boxes = pred_boxes[valid_preds]
            pred_scores = pred_scores[valid_preds]

        # Match predictions to targets
        if len(pred_boxes) == 0:
            if len(target_boxes) > 0:
                precision, recall, avg_iou = 0.0, 0.0, 0.0
            else:
                precision, recall, avg_iou = 1.0, 1.0, 1.0
        elif len(target_boxes) == 0:
            precision, recall, avg_iou = 0.0, 1.0, 0.0
        else:
            # Calculate IoU matrix
            iou_matrix = np.zeros((len(pred_boxes), len(target_boxes)))
            for i, pred_box in enumerate(pred_boxes):
                for j, target_box in enumerate(target_boxes):
                    iou_matrix[i, j] = calculate_bbox_iou(pred_box, target_box)

            # Greedy matching
            matched_targets = set()
            matched_preds = []
            per_image_ious = []

            sorted_indices = np.argsort(pred_scores)[::-1]

            for pred_idx in sorted_indices:
                best_iou = 0.0
                best_target_idx = -1

                for target_idx in range(len(target_boxes)):
                    if target_idx in matched_targets:
                        continue
                    iou = iou_matrix[pred_idx, target_idx]
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_target_idx = target_idx

                if best_target_idx >= 0:
                    matched_targets.add(best_target_idx)
                    matched_preds.append(pred_idx)
                    per_image_ious.append(best_iou)

            tp = len(matched_preds)
            fp = len(pred_boxes) - tp
            fn = len(target_boxes) - tp

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            avg_iou = np.mean(per_image_ious) if per_image_ious else 0.0

        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        all_ious.append(avg_iou)

    # Calculate proper mAP (COCO-style AP@[0.5:0.95])
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps_at_ious = []
    for iou_thresh in iou_thresholds:
        ap = calculate_ap_at_iou(predictions, targets, iou_thresh)
        aps_at_ious.append(ap)
    mAP = np.mean(aps_at_ious)

    # Calculate AP50 (AP at IoU=0.5)
    AP50 = calculate_ap_at_iou(predictions, targets, 0.5)

    # Calculate AP75 (AP at IoU=0.75)
    AP75 = calculate_ap_at_iou(predictions, targets, 0.75)

    # Calculate FROC metrics
    froc_metrics = calculate_froc(predictions, targets, num_images, iou_threshold=0.5)

    # Average metrics across all images
    metrics = {
        'precision': np.mean(all_precisions) if all_precisions else 0.0,
        'recall': np.mean(all_recalls) if all_recalls else 0.0,
        'f1': np.mean(all_f1_scores) if all_f1_scores else 0.0,
        'avg_iou': np.mean(all_ious) if all_ious else 0.0,
        'mAP': mAP,  # COCO-style mAP@[0.5:0.95]
        'AP50': AP50,  # AP at IoU=0.5
        'AP75': AP75,  # AP at IoU=0.75
        'froc_score': froc_metrics['froc_score'],  # Average sensitivity at standard FP rates
        'auc_froc': froc_metrics['auc_froc'],  # Area under FROC curve
        'froc_sensitivities': froc_metrics['sensitivities'],  # Sensitivity at each FP rate
    }

    return metrics


def calculate_detection_metrics_simple(predictions, targets):
    """
    Simplified detection metrics calculation.

    For compatibility with existing framework, returns metrics in similar format
    to segmentation metrics (dice, iou, etc.)

    Args:
        predictions: List of prediction dicts
        targets: List of target dicts

    Returns:
        tuple: (dice_equivalent, iou, sensitivity, auroc)
        Note: These are approximations for detection tasks
    """
    metrics = calculate_detection_metrics(predictions, targets)

    # Map detection metrics to segmentation-like metrics for compatibility
    # Precision ~ Dice equivalent
    dice_equivalent = metrics['precision']

    # Average IoU
    iou = metrics['avg_iou']

    # Recall ~ Sensitivity
    sensitivity = metrics['recall']

    # F1 score as AUROC approximation
    auroc = metrics['f1']

    return dice_equivalent, iou, sensitivity, auroc
