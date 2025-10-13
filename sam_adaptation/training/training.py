#!/usr/bin/env python3
"""
Training utilities for SAM adaptation project.

This module contains training-related functions including model training,
early stopping, and logging utilities.
"""

import os
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import calculate_metrics, divide_image_into_areas, calculate_bin_dice_metrics, calculate_performance_coverage, calculate_spatial_consistency
from losses import combo_loss


def train_model_round(model, train_dataset, val_dataset, device, epochs=50, batch_size=8,
                     round_num=1, output_dir=None, patience=15, min_delta=0.001, prev_bin_dices=None, 
                     grid_width=2, grid_height=3):
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
        tuple: (best_val_loss, avg_val_dice, avg_val_bin_metrics, current_bin_dices)
    """
    # Create data loaders with reduced num_workers to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Define areas for spatial evaluation
    areas = divide_image_into_areas(grid_width=grid_width, grid_height=grid_height)

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.decoder.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    criterion = combo_loss

    # Setup logging
    round_logger = _setup_round_logger(output_dir, round_num, train_dataset, val_dataset, epochs, batch_size)

    # Training loop with early stopping
    best_val_loss, best_model_state, best_epoch, final_val_dice, final_spatial_metrics, current_bin_dices = _train_with_early_stopping(
        model, train_loader, val_loader, optimizer, scheduler, criterion,
        areas, device, epochs, patience, min_delta, prev_bin_dices, round_logger
    )

    # Load best model and save checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    _save_model_checkpoint(model, best_model_state, best_val_loss, best_epoch, round_num, output_dir)

    # Log completion
    _log_round_completion(round_logger, round_num, best_val_loss, best_epoch, epochs)

    return best_val_loss, final_val_dice, final_spatial_metrics, current_bin_dices


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
    round_logger.info("="*80)
    round_logger.info(f"Round {round_num} Training Started")
    round_logger.info("="*80)
    round_logger.info(f"Training samples: {len(train_dataset)}")
    round_logger.info(f"Validation samples: {len(val_dataset)}")
    round_logger.info(f"Epochs: {epochs}")
    round_logger.info(f"Batch size: {batch_size}")
    round_logger.info("-"*80)

    return round_logger


def _train_with_early_stopping(model, train_loader, val_loader, optimizer, scheduler, criterion,
                              areas, device, epochs, patience, min_delta, prev_bin_dices, round_logger):
    """Train model with early stopping."""
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    all_bin_dices = []

    # Progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training", leave=False)

    for epoch in epoch_pbar:
        # Training phase
        train_loss, train_dice = _train_epoch(model, train_loader, optimizer, scheduler, criterion, device)

        # Validation phase
        val_loss, val_dice, val_bin_metrics, batch_bin_dices = _validate_epoch(
            model, val_loader, criterion, areas, device
        )

        all_bin_dices.extend(batch_bin_dices)

        # Calculate performance coverage
        performance_coverage = _calculate_performance_coverage(prev_bin_dices, batch_bin_dices, areas)
        val_bin_metrics['performance_coverage'] = performance_coverage

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Log epoch metrics
        _log_epoch_metrics(round_logger, epoch, epochs, train_loss, val_loss, train_dice, val_dice,
                          best_val_loss, best_epoch, patience_counter, patience, scheduler, val_bin_metrics)

        # Update progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Train Dice': f'{train_dice:.4f}',
            'Val Dice': f'{val_dice:.4f}',
            'Worst Bin': f'{val_bin_metrics["worst_bin_dice"]:.4f}',
            'Perf Cov': f'{val_bin_metrics["performance_coverage"]:.4f}',
            'Best Val Loss': f'{best_val_loss:.4f}'
        })

        # Early stopping
        if patience_counter >= patience:
            if round_logger:
                round_logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                round_logger.info(f"Best validation Loss: {best_val_loss:.6f} at epoch {best_epoch}")
            print(f"🛑 Early stopping at epoch {epoch + 1} (best val loss: {best_val_loss:.4f} at epoch {best_epoch})")
            break

    epoch_pbar.close()

    # Get final metrics from the last epoch
    final_val_dice = val_dice if 'val_dice' in locals() else 0.0
    final_spatial_metrics = val_bin_metrics if 'val_bin_metrics' in locals() else {}
    final_current_bin_dices = torch.cat(all_bin_dices, dim=1) if all_bin_dices else None

    return best_val_loss, best_model_state, best_epoch, final_val_dice, final_spatial_metrics, final_current_bin_dices


def _train_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    train_dice = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        dice, _, _, _ = calculate_metrics(outputs, masks)
        train_dice += dice


    return train_loss / len(train_loader), train_dice / len(train_loader)


def _validate_epoch(model, val_loader, criterion, areas, device):
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

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            dice, _, _, _ = calculate_metrics(outputs, masks)

            val_loss += loss.item()
            val_dice += dice

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

    return val_loss / len(val_loader), val_dice / len(val_loader), avg_val_bin_metrics, all_bin_dices


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

        print(f"🔍 First round performance coverage: comparing with 0.001 baseline ({num_areas} areas, {num_samples} samples)")
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


def _log_epoch_metrics(round_logger, epoch, epochs, train_loss, val_loss, train_dice, val_dice,
                      best_val_loss, best_epoch, patience_counter, patience, scheduler, val_bin_metrics):
    """Log metrics for current epoch."""
    if not round_logger:
        return

    round_logger.info(f"Epoch {epoch + 1}/{epochs}:")
    round_logger.info(f"  Train Loss: {train_loss:.6f}")
    round_logger.info(f"  Val Loss: {val_loss:.6f}")
    round_logger.info(f"  Train Dice: {train_dice:.6f}")
    round_logger.info(f"  Val Dice: {val_dice:.6f}")
    round_logger.info(f"  Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")
    round_logger.info(f"  Patience: {patience_counter}/{patience}")
    round_logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.8f}")
    round_logger.info("  Spatial Metrics:")
    round_logger.info(f"    Avg Bin Dice: {val_bin_metrics['avg_bin_dice']:.6f}")
    round_logger.info(f"    Worst Bin Dice: {val_bin_metrics['worst_bin_dice']:.6f}")
    round_logger.info(f"    P10 Bin Dice: {val_bin_metrics['p10_bin_dice']:.6f}")
    round_logger.info(f"    Bin Std: {val_bin_metrics['bin_std']:.6f}")
    round_logger.info(f"    Min/Max Bin Dice: {val_bin_metrics['min_bin_dice']:.6f}/{val_bin_metrics['max_bin_dice']:.6f}")
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
    print(f"💾 Best model saved: {model_save_path}")


# def _calculate_final_metrics(model, val_loader, areas, device, prev_bin_dices):
#     """Calculate final metrics after training."""
#     # This would typically involve running validation again to get final metrics
#     # For now, return placeholder values
#     return {
#         'val_dice': 0.0,
#         'spatial_metrics': {},
#         'current_bin_dices': None
#     }


def _log_round_completion(round_logger, round_num, best_val_loss, best_epoch, total_epochs):
    """Log round completion."""
    if not round_logger:
        return

    round_logger.info("="*80)
    round_logger.info(f"Round {round_num} Training Completed")
    round_logger.info(f"Best Validation Loss: {best_val_loss:.6f} at epoch {best_epoch}")
    round_logger.info(f"Total epochs trained: {total_epochs}")
    round_logger.info(f"Model saved: checkpoint/round{round_num}_best_model.pth")
    round_logger.info("="*80)
