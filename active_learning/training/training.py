#!/usr/bin/env python3
"""
Training utilities for SPARCL.

This module contains training-related functions including model training,
early stopping, and logging utilities.
"""

import logging
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from losses import combo_loss
from losses.detection import detection_loss
from metrics import (calculate_bin_dice_metrics, calculate_metrics,
                     calculate_performance_coverage,
                     calculate_spatial_consistency, divide_image_into_areas)
from metrics.detection import calculate_detection_metrics_simple
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Custom collate function for detection (handles variable number of bboxes)
def detection_collate_fn(batch):
    """Custom collate function for detection tasks."""
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets


def train_model_round(model, train_dataset, val_dataset, device, epochs=50, batch_size=8,
                      round_num=1, output_dir=None, patience=15, min_delta=0.001, prev_bin_dices=None,
                      grid_width=2, grid_height=3, uncertainty_type='base', task_type='segmentation'):
    """
    Train model for one round with early stopping.

    Args:
        model: The model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device: Device to run training on
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        round_num: Current round number (for logging)
        output_dir: Output directory for saving models and logs
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        prev_bin_dices: Previous round's bin dices for performance coverage calculation

    Returns:
        tuple: (best_val_loss, avg_val_dice, avg_val_iou, avg_val_bin_metrics, current_bin_dices, round_model_save_path)
    """

    # Create data loaders
    if task_type == 'detection':
        # For detection, use custom collate function
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, collate_fn=detection_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True, collate_fn=detection_collate_fn)
    else:
        # For segmentation, use default collate
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True, persistent_workers=True)

    # Define areas for spatial evaluation
    areas = divide_image_into_areas(grid_width=grid_width, grid_height=grid_height)

    if uncertainty_type == 'none':
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.decoder.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    # Select criterion based on task type
    if task_type == 'detection':
        criterion = detection_loss
    else:
        criterion = combo_loss

    # Setup logging
    round_logger = _setup_round_logger(output_dir, round_num, train_dataset, val_dataset, epochs, batch_size)

    # Training loop with early stopping
    best_val_loss, best_model_state, best_epoch, final_val_dice, final_val_iou, final_spatial_metrics, current_bin_dices, best_val_metric = _train_with_early_stopping(
        model, train_loader, val_loader, optimizer, scheduler, criterion,
        areas, device, epochs, patience, min_delta, prev_bin_dices, round_logger, task_type
    )

    # Load best model and save checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    round_model_save_path = _save_model_checkpoint(
        model, best_model_state, best_val_loss, best_epoch, round_num, output_dir)

    # Log completion
    _log_round_completion(round_logger, round_num, best_val_loss, best_epoch, epochs, task_type, best_val_metric)

    return best_val_loss, final_val_dice, final_val_iou, final_spatial_metrics, current_bin_dices, round_model_save_path


def _setup_round_logger(output_dir, round_num, train_dataset, val_dataset, epochs, batch_size):
    """Setup round-specific logger."""
    if not output_dir:
        return None

    # Create logs directory
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    round_log_path = os.path.join(logs_dir, f'round{round_num}.log')
    round_logger = logging.getLogger(f'Round_{round_num}')
    round_logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in round_logger.handlers[:]:
        round_logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(round_log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    round_logger.addHandler(file_handler)

    # Log round start
    round_logger.info("=" * 80)
    round_logger.info(f"Round {round_num} Training Started")
    round_logger.info("=" * 80)
    round_logger.info(f"Training samples: {len(train_dataset)}")
    round_logger.info(f"Validation samples: {len(val_dataset)}")
    round_logger.info(f"Epochs: {epochs}")
    round_logger.info(f"Batch size: {batch_size}")
    round_logger.info("-" * 80)

    return round_logger


def _train_with_early_stopping(model, train_loader, val_loader, optimizer, scheduler, criterion,
                               areas, device, epochs, patience, min_delta, prev_bin_dices, round_logger, task_type='segmentation'):
    """Train model with early stopping."""
    best_val_loss = float('inf')
    best_val_metric = 0.0  # For detection: use AP50 (higher is better)
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    all_bin_dices = []

    # Store best metrics (will be updated when best model is found)
    best_val_dice = 0.0
    best_val_iou = 0.0
    best_spatial_metrics = {}

    # Progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training", leave=False)

    for epoch in epoch_pbar:
        # Training phase
        train_loss, train_dice, train_iou = _train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, task_type)

        # Validation phase
        val_loss, val_dice, val_iou, val_bin_metrics, batch_bin_dices = _validate_epoch(
            model, val_loader, criterion, areas, device, task_type
        )

        all_bin_dices.extend(batch_bin_dices)

        # Calculate performance coverage
        performance_coverage = _calculate_performance_coverage(prev_bin_dices, batch_bin_dices, areas)
        val_bin_metrics['performance_coverage'] = performance_coverage

        # Early stopping check - different criteria for detection vs segmentation
        if task_type == 'detection':
            # For detection: use AP50 (higher is better)
            current_metric = val_bin_metrics.get('AP50', 0.0)
            if current_metric > best_val_metric + min_delta:
                best_val_metric = current_metric
                best_val_loss = val_loss  # Also track loss for logging
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                # Store best metrics
                best_val_dice = val_dice
                best_val_iou = val_iou
                best_spatial_metrics = val_bin_metrics.copy()
            else:
                patience_counter += 1
        else:
            # For segmentation: use val_loss (lower is better)
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                # Store best metrics
                best_val_dice = val_dice
                best_val_iou = val_iou
                best_spatial_metrics = val_bin_metrics.copy()
            else:
                patience_counter += 1

        # Log epoch metrics (add detection metrics if available)
        if task_type == 'detection':
            if 'mAP' in val_bin_metrics:
                round_logger.info(
                    f"  Detection Metrics: mAP={val_bin_metrics['mAP']:.4f}, AP50={val_bin_metrics.get('AP50', 0):.4f}, AP75={val_bin_metrics.get('AP75', 0):.4f}")
                round_logger.info(
                    f"  FROC: score={val_bin_metrics.get('froc_score', 0):.4f}, AUC={val_bin_metrics.get('auc_froc', 0):.4f}")

        _log_epoch_metrics(round_logger, epoch, epochs, train_loss, val_loss, train_dice, train_iou, val_dice, val_iou,
                           best_val_loss, best_epoch, patience_counter, patience, scheduler, val_bin_metrics)

        # Update progress bar
        if task_type == 'detection':
            epoch_pbar.set_postfix({
                'TrLoss': f'{train_loss:.4f}',
                'VLoss': f'{val_loss:.4f}',
                'mAP': f'{val_bin_metrics.get("mAP", 0):.4f}',
                'AP50': f'{val_bin_metrics.get("AP50", 0):.4f}',
                'FROC': f'{val_bin_metrics.get("froc_score", 0):.4f}',
                'BestAP50': f'{best_val_metric:.4f}'
            })
        else:
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Train Dice': f'{train_dice:.4f}',
                'Train IoU': f'{train_iou:.4f}',
                'Val Dice': f'{val_dice:.4f}',
                'Val IoU': f'{val_iou:.4f}',
                'Worst Bin': f'{val_bin_metrics["worst_bin_dice"]:.4f}',
                'Perf Cov': f'{val_bin_metrics["performance_coverage"]:.4f}',
                'Best Val Loss': f'{best_val_loss:.4f}'
            })

        # Early stopping
        if patience_counter >= patience:
            if round_logger:
                round_logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                if task_type == 'detection':
                    round_logger.info(f"Best validation AP50: {best_val_metric:.6f} at epoch {best_epoch}")
                else:
                    round_logger.info(f"Best validation Loss: {best_val_loss:.6f} at epoch {best_epoch}")
            if task_type == 'detection':
                print(f"🛑 Early stopping at epoch {epoch + 1} (best AP50: {best_val_metric:.4f} at epoch {best_epoch})")
            else:
                print(
                    f"🛑 Early stopping at epoch {epoch + 1} (best val loss: {best_val_loss:.4f} at epoch {best_epoch})")
            break

    epoch_pbar.close()

    # Use metrics from the best model (not the last epoch)
    final_val_dice = best_val_dice if best_val_dice > 0 else (val_dice if 'val_dice' in locals() else 0.0)
    final_val_iou = best_val_iou if best_val_iou > 0 else (val_iou if 'val_iou' in locals() else 0.0)
    final_spatial_metrics = best_spatial_metrics if best_spatial_metrics else (
        val_bin_metrics if 'val_bin_metrics' in locals() else {})
    final_current_bin_dices = torch.cat(all_bin_dices, dim=1) if all_bin_dices else None

    return best_val_loss, best_model_state, best_epoch, final_val_dice, final_val_iou, final_spatial_metrics, final_current_bin_dices, best_val_metric


def _train_epoch(model, train_loader, optimizer, scheduler, criterion, device, task_type='segmentation'):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    train_dice = 0
    train_iou = 0

    for batch_idx, batch in enumerate(train_loader):
        if task_type == 'detection':
            # Detection: batch is (images, targets) where targets is list of dicts
            images, targets = batch
            # Convert images to list format for detection models
            if isinstance(images, torch.Tensor):
                if images.dim() == 4:
                    images = [img.to(device) for img in images]
                else:
                    images = [images.to(device)]
            else:
                images = [img.to(device) for img in images]

            # Convert targets to list of dicts with proper device
            target_list = []
            for target in targets:
                target_dict = {
                    'boxes': target['boxes'].to(device) if torch.is_tensor(target['boxes']) else target['boxes'],
                    'labels': target['labels'].to(device) if torch.is_tensor(target['labels']) else target['labels']
                }
                target_list.append(target_dict)

            optimizer.zero_grad()
            outputs = model(images, targets=target_list)

            # Detection models return dict with 'loss' key
            if isinstance(outputs, dict):
                loss = outputs['loss']
            else:
                loss = criterion(outputs, target_list)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            # Calculate detection metrics (only every N batches to reduce overhead)
            # Skip metric calculation during training for speed
            # We'll get proper metrics during validation anyway
        else:
            # Segmentation: batch is (images, masks)
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            dice, iou, _, _ = calculate_metrics(outputs, masks)
            train_dice += dice
            train_iou += iou

    # For detection, we don't calculate train metrics during training for speed
    # Train metrics are less important than validation metrics anyway
    if task_type == 'detection':
        avg_dice = 0.0  # Not calculated during training for detection
        avg_iou = 0.0   # Not calculated during training for detection
    else:
        avg_dice = train_dice / len(train_loader)
        avg_iou = train_iou / len(train_loader)

    return train_loss / len(train_loader), avg_dice, avg_iou


def _validate_epoch(model, val_loader, criterion, areas, device, task_type='segmentation'):
    """Validate for one epoch."""
    model.eval()
    val_loss = 0
    val_dice = 0
    val_bin_metrics = {
        'avg_bin_dice': 0, 'worst_bin_dice': 0, 'p10_bin_dice': 0,
        'bin_std': 0, 'min_bin_dice': 0, 'max_bin_dice': 0, 'median_bin_dice': 0,
        'spatial_consistency': 0
    }
    val_batch_count = 0
    all_bin_dices = []
    val_iou = 0

    with torch.no_grad():
        for batch in val_loader:
            if task_type == 'detection':
                # Detection: batch is (images, targets)
                images, targets = batch
                # Convert images to list format
                if isinstance(images, torch.Tensor):
                    if images.dim() == 4:
                        images = [img.to(device) for img in images]
                    else:
                        images = [images.to(device)]
                else:
                    images = [img.to(device) for img in images]

                # Convert targets
                target_list = []
                for target in targets:
                    target_dict = {
                        'boxes': target['boxes'].to(device) if torch.is_tensor(target['boxes']) else target['boxes'],
                        'labels': target['labels'].to(device) if torch.is_tensor(target['labels']) else target['labels']
                    }
                    target_list.append(target_dict)

                # Get predictions (in eval mode, model returns predictions)
                predictions = model(images)

                # Get loss separately by running in training mode
                model.train()
                loss_dict = model(images, targets=target_list)
                loss = loss_dict['loss'] if isinstance(loss_dict, dict) else loss_dict
                model.eval()

                val_loss += loss.item()

                # Calculate full detection metrics including mAP, AP50, FROC
                from metrics.detection import \
                    calculate_detection_metrics
                det_metrics = calculate_detection_metrics(predictions, target_list, iou_threshold=0.5)

                # Use precision as dice equivalent, avg_iou as iou
                dice = det_metrics['precision']
                iou = det_metrics['avg_iou']
                mAP = det_metrics['mAP']
                AP50 = det_metrics['AP50']
                froc_score = det_metrics['froc_score']

                val_dice += dice
                val_iou += iou

                # Initialize detection-specific accumulators if not exist
                if not hasattr(_validate_epoch, 'val_map'):
                    _validate_epoch.val_map = 0.0
                if not hasattr(_validate_epoch, 'val_ap50'):
                    _validate_epoch.val_ap50 = 0.0
                if not hasattr(_validate_epoch, 'val_ap75'):
                    _validate_epoch.val_ap75 = 0.0
                if not hasattr(_validate_epoch, 'val_froc'):
                    _validate_epoch.val_froc = 0.0
                if not hasattr(_validate_epoch, 'val_auc_froc'):
                    _validate_epoch.val_auc_froc = 0.0

                _validate_epoch.val_map += mAP
                _validate_epoch.val_ap50 += AP50
                _validate_epoch.val_ap75 += det_metrics['AP75']
                _validate_epoch.val_froc += froc_score
                _validate_epoch.val_auc_froc += det_metrics['auc_froc']

                # Debug: print first batch metrics occasionally
                if val_batch_count == 0 and len(predictions) > 0:
                    pred = predictions[0]
                    target = target_list[0]
                    num_pred = len(pred['boxes']) if len(pred['boxes']) > 0 else 0
                    num_target = len(target['boxes']) if len(target['boxes']) > 0 else 0
                    if num_pred > 0:
                        avg_score = pred['scores'].mean().item() if torch.is_tensor(
                            pred['scores']) else np.mean(pred['scores'])
                        print(
                            f"  🔍 Detection Debug: pred_boxes={num_pred}, target_boxes={num_target}, avg_score={avg_score:.3f}")
                        print(
                            f"     mAP={mAP:.3f}, AP50={AP50:.3f}, FROC={froc_score:.3f}, precision={dice:.3f}, iou={iou:.3f}")

                # For detection, bin metrics are not directly applicable
                # Use dummy values or skip
                batch_bin_metrics = {
                    'avg_bin_dice': dice, 'worst_bin_dice': dice, 'p10_bin_dice': dice,
                    'bin_std': 0.0, 'min_bin_dice': dice, 'max_bin_dice': dice, 'median_bin_dice': dice,
                    'spatial_consistency': 0.0
                }
                # Dummy bin dices for compatibility
                # Shape should be [num_areas, batch_size] to match segmentation format
                batch_size = len(images)
                num_areas = len(areas)
                # Create [num_areas, batch_size] tensor where each area has the same dice value
                batch_bin_dices = torch.full((num_areas, batch_size), dice, dtype=torch.float32)
                all_bin_dices.append(batch_bin_dices)

                # Update bin metrics
                for key in val_bin_metrics:
                    val_bin_metrics[key] += batch_bin_metrics[key]

                val_batch_count += 1
            else:
                # Segmentation: batch is (images, masks)
                images, masks = batch
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

                dice, iou, _, _ = calculate_metrics(outputs, masks)

                val_loss += loss.item()
                val_dice += dice
                val_iou += iou

                # Calculate bin-wise metrics
                pred_masks = outputs.squeeze(1)  # [B, H, W]
                true_masks = masks.squeeze(1)    # [B, H, W]
                batch_bin_metrics = calculate_bin_dice_metrics(pred_masks, true_masks, areas)

                # Calculate bin dices for this batch
                batch_bin_dices = _calculate_batch_bin_dices(pred_masks, true_masks, areas)
                all_bin_dices.append(batch_bin_dices)

                # Calculate spatial consistency for this batch
                spatial_consistency = calculate_spatial_consistency(batch_bin_dices, areas)
                batch_bin_metrics['spatial_consistency'] = spatial_consistency

            for key in val_bin_metrics:
                val_bin_metrics[key] += batch_bin_metrics[key]
            val_batch_count += 1

    # Average bin metrics
    avg_val_bin_metrics = {key: val / val_batch_count for key, val in val_bin_metrics.items()}

    # Add detection-specific metrics if detection task
    if task_type == 'detection':
        if hasattr(_validate_epoch, 'val_map'):
            avg_val_bin_metrics['mAP'] = _validate_epoch.val_map / val_batch_count
            delattr(_validate_epoch, 'val_map')
        else:
            avg_val_bin_metrics['mAP'] = 0.0

        if hasattr(_validate_epoch, 'val_ap50'):
            avg_val_bin_metrics['AP50'] = _validate_epoch.val_ap50 / val_batch_count
            delattr(_validate_epoch, 'val_ap50')
        else:
            avg_val_bin_metrics['AP50'] = 0.0

        if hasattr(_validate_epoch, 'val_ap75'):
            avg_val_bin_metrics['AP75'] = _validate_epoch.val_ap75 / val_batch_count
            delattr(_validate_epoch, 'val_ap75')
        else:
            avg_val_bin_metrics['AP75'] = 0.0

        if hasattr(_validate_epoch, 'val_froc'):
            avg_val_bin_metrics['froc_score'] = _validate_epoch.val_froc / val_batch_count
            delattr(_validate_epoch, 'val_froc')
        else:
            avg_val_bin_metrics['froc_score'] = 0.0

        if hasattr(_validate_epoch, 'val_auc_froc'):
            avg_val_bin_metrics['auc_froc'] = _validate_epoch.val_auc_froc / val_batch_count
            delattr(_validate_epoch, 'val_auc_froc')
        else:
            avg_val_bin_metrics['auc_froc'] = 0.0
    else:
        avg_val_bin_metrics['mAP'] = 0.0

    return val_loss / len(val_loader), val_dice / len(val_loader), val_iou / len(val_loader), avg_val_bin_metrics, all_bin_dices


def _calculate_batch_bin_dices(pred_masks, true_masks, areas):
    """Calculate bin dices for a batch."""
    batch_bin_dices = []
    for area_idx, (y_start, y_end, x_start, x_end, _) in enumerate(areas):
        area_pred = pred_masks[:, y_start:y_end, x_start:x_end]
        area_true = true_masks[:, y_start:y_end, x_start:x_end]

        intersection = (area_pred * area_true).sum(dim=(1, 2))
        union = area_pred.sum(dim=(1, 2)) + area_true.sum(dim=(1, 2))
        dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
        batch_bin_dices.append(dice)

    return torch.stack(batch_bin_dices, dim=0)  # [num_areas, B]


def _calculate_performance_coverage(prev_bin_dices, current_batch_bin_dices, areas):
    """Calculate performance coverage if previous round data is available."""
    if not current_batch_bin_dices:
        # No current data available
        return 0.0

    # Concatenate all batch bin dices
    current_bin_dices = torch.cat(current_batch_bin_dices, dim=1)  # [num_areas, total_samples]

    if prev_bin_dices is None:
        # For first round, assume previous performance was 0.001 (small baseline)
        num_samples = current_bin_dices.shape[1]
        num_areas = current_bin_dices.shape[0]

        # Create small baseline tensor with same shape as current_bin_dices
        prev_bin_dices = torch.full_like(current_bin_dices, 0.001)

        print(
            f"🔍 First round performance coverage: comparing with 0.001 baseline ({num_areas} areas, {num_samples} samples)")
        return calculate_performance_coverage(prev_bin_dices, current_bin_dices, areas, debug=True)

    # Check if tensor sizes match
    if prev_bin_dices.shape[1] != current_bin_dices.shape[1]:
        print(f"⚠️  Warning: Performance coverage calculation using subset due to size mismatch")
        print(f"   Previous round samples: {prev_bin_dices.shape[1]}")
        print(f"   Current round samples: {current_bin_dices.shape[1]}")

        # Use minimum sample size to calculate coverage
        min_samples = min(prev_bin_dices.shape[1], current_bin_dices.shape[1])
        prev_subset = prev_bin_dices[:, :min_samples]
        current_subset = current_bin_dices[:, :min_samples]

        return calculate_performance_coverage(prev_subset, current_subset, areas, debug=True)

    return calculate_performance_coverage(prev_bin_dices, current_bin_dices, areas, debug=True)


def _log_epoch_metrics(round_logger, epoch, epochs, train_loss, val_loss, train_dice, train_iou, val_dice, val_iou,
                       best_val_loss, best_epoch, patience_counter, patience, scheduler, val_bin_metrics):
    """Log metrics for current epoch."""
    if not round_logger:
        return

    round_logger.info(f"Epoch {epoch + 1}/{epochs}:")
    round_logger.info(f"  Train Loss: {train_loss:.6f}")
    round_logger.info(f"  Val Loss: {val_loss:.6f}")
    round_logger.info(f"  Train Dice: {train_dice:.6f}")
    round_logger.info(f"  Train IoU: {train_iou:.6f}")
    round_logger.info(f"  Val Dice: {val_dice:.6f}")
    round_logger.info(f"  Val IoU: {val_iou:.6f}")
    if 'mAP' in val_bin_metrics and val_bin_metrics['mAP'] > 0:
        round_logger.info(f"  Val mAP: {val_bin_metrics['mAP']:.6f}")
    round_logger.info(f"  Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")
    round_logger.info(f"  Patience: {patience_counter}/{patience}")
    round_logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.8f}")
    round_logger.info("  Spatial Metrics:")
    round_logger.info(f"    Avg Bin Dice: {val_bin_metrics['avg_bin_dice']:.6f}")
    round_logger.info(f"    Worst Bin Dice: {val_bin_metrics['worst_bin_dice']:.6f}")
    round_logger.info(f"    P10 Bin Dice: {val_bin_metrics['p10_bin_dice']:.6f}")
    round_logger.info(f"    Bin Std: {val_bin_metrics['bin_std']:.6f}")
    round_logger.info(
        f"    Min/Max Bin Dice: {val_bin_metrics['min_bin_dice']:.6f}/{val_bin_metrics['max_bin_dice']:.6f}")
    round_logger.info(f"    Median Bin Dice: {val_bin_metrics['median_bin_dice']:.6f}")
    round_logger.info(f"    Spatial Consistency: {val_bin_metrics['spatial_consistency']:.6f}")
    round_logger.info(f"    Performance Coverage: {val_bin_metrics['performance_coverage']:.6f}")
    round_logger.info("-" * 40)


def _save_model_checkpoint(model, best_model_state, best_val_loss, best_epoch, round_num, output_dir):
    """Save model checkpoint."""
    if not output_dir or best_model_state is None:
        return

    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_save_path = os.path.join(checkpoint_dir, f'round{round_num}_best_model.pth')
    torch.save({
        'model_state_dict': best_model_state,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'round_num': round_num
    }, model_save_path)
    print(f"💾 {round_num} model saved: {model_save_path}")

    return model_save_path


# def _calculate_final_metrics(model, val_loader, areas, device, prev_bin_dices):
#     """Calculate final metrics after training."""
#     # This would typically involve running validation again to get final metrics
#     # For now, return placeholder values
#     return {
#         'val_dice': 0.0,
#         'spatial_metrics': {},
#         'current_bin_dices': None
#     }


def _log_round_completion(round_logger, round_num, best_val_loss, best_epoch, total_epochs, task_type='segmentation', best_val_metric=None):
    """Log round completion."""
    if not round_logger:
        return

    round_logger.info("=" * 80)
    round_logger.info(f"Round {round_num} Training Completed")
    if task_type == 'detection' and best_val_metric is not None:
        round_logger.info(f"Best Validation AP50: {best_val_metric:.6f} at epoch {best_epoch}")
    else:
        round_logger.info(f"Best Validation Loss: {best_val_loss:.6f} at epoch {best_epoch}")
    round_logger.info(f"Total epochs trained: {total_epochs}")
    round_logger.info(f"Model saved: checkpoint/round{round_num}_best_model.pth")
    round_logger.info("=" * 80)
