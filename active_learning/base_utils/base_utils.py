#!/usr/bin/env python3
"""
Utility functions for SAM adaptation project.
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

# Add project root to path
sys.path.append('/opt/pxi')


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(session_dir, rank=0):
    """Setup logger for training."""
    if rank != 0:
        return None  # Only log on rank 0

    # Create logs directory
    logs_dir = os.path.join(session_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Setup logger
    logger = logging.getLogger('SAM_Training')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    log_file = os.path.join(logs_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def setup_ddp():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        return True, rank, world_size, local_rank, device
    else:
        return False, 0, 1, 0, torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cleanup_ddp():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_session_dir(target_lesion, timestamp=None):
    """Create session directory for training."""
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    session_dir = f"/team/team_pxi/workspace/juhojung/spatial_active_learning/sam_{target_lesion}_{timestamp}"
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def plot_validation_curves(results, output_dir, session_name):
    """Plot validation loss and metric curves (Dice for segmentation, mAP for detection)."""
    rounds = list(range(1, len(results) + 1))
    val_losses = [result['val_loss'] for result in results]

    # Detect task type from results
    task_type = results[0].get('task_type', 'segmentation') if results else 'segmentation'
    is_detection = task_type == 'detection'

    if is_detection:
        val_metrics = [result.get('val_mAP', 0) for result in results]
        metric_name = 'mAP'
        metric_label = 'Validation mAP'
    else:
        val_metrics = [result.get('val_dice', 0) for result in results]
        metric_name = 'Dice'
        metric_label = 'Validation Dice Score'

    # Plot validation loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, val_losses, 'r-o', linewidth=2, markersize=8)
    plt.xlabel('Active Learning Round', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title(f'{session_name}: Validation Loss vs Round', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rounds)

    # Add value labels on points
    for i, loss in enumerate(val_losses):
        plt.annotate(f'{loss:.4f}', (rounds[i], loss),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    loss_path = os.path.join(output_dir, 'validation_loss_curve.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot validation metric curve (Dice or mAP)
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, val_metrics, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Active Learning Round', fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.title(f'{session_name}: {metric_label} vs Round', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rounds)
    plt.ylim(0, 1)

    # Add value labels on points
    for i, val in enumerate(val_metrics):
        plt.annotate(f'{val:.3f}', (rounds[i], val),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    metric_path = os.path.join(output_dir, f'validation_{metric_name.lower()}_curve.png')
    plt.savefig(metric_path, dpi=150, bbox_inches='tight')
    plt.close()

    return loss_path, metric_path


def plot_spatial_metrics_curves(results, output_dir, session_name):
    """Plot spatial metrics curves for Active Learning progress."""
    rounds = list(range(1, len(results) + 1))

    # Detect task type from results
    task_type = results[0].get('task_type', 'segmentation') if results else 'segmentation'
    is_detection = task_type == 'detection'

    # Extract spatial metrics
    worst_bin_dices = [result['spatial_metrics'].get('worst_bin_dice', 0) for result in results]
    p10_bin_dices = [result['spatial_metrics'].get('p10_bin_dice', 0) for result in results]
    bin_stds = [result['spatial_metrics'].get('bin_std', 0) for result in results]
    spatial_consistencies = [result['spatial_metrics'].get('spatial_consistency', 0) for result in results]
    performance_coverages = [result['spatial_metrics'].get('performance_coverage', 0) for result in results]

    # Get main metric based on task type
    if is_detection:
        val_metrics = [result.get('val_mAP', 0) for result in results]
        metric_name = 'mAP'
    else:
        val_metrics = [result.get('val_dice', 0) for result in results]
        metric_name = 'Dice'

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{session_name}: Spatial Metrics Progress', fontsize=16, fontweight='bold')

    # Plot 1: Worst Bin Dice
    axes[0, 0].plot(rounds, worst_bin_dices, 'r-o', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Active Learning Round', fontsize=12)
    axes[0, 0].set_ylabel('Worst Bin Dice', fontsize=12)
    axes[0, 0].set_title('Worst Bin Dice vs Round', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(rounds)
    axes[0, 0].set_ylim(0, 1)

    # Add value labels
    for i, val in enumerate(worst_bin_dices):
        axes[0, 0].annotate(f'{val:.3f}', (rounds[i], val),
                            textcoords="offset points", xytext=(0, 10), ha='center')

    # Plot 2: 10th Percentile Bin Dice
    axes[0, 1].plot(rounds, p10_bin_dices, 'g-o', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Active Learning Round', fontsize=12)
    axes[0, 1].set_ylabel('P10 Bin Dice', fontsize=12)
    axes[0, 1].set_title('10th Percentile Bin Dice vs Round', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(rounds)
    axes[0, 1].set_ylim(0, 1)

    # Add value labels
    for i, val in enumerate(p10_bin_dices):
        axes[0, 1].annotate(f'{val:.3f}', (rounds[i], val),
                            textcoords="offset points", xytext=(0, 10), ha='center')

    # Plot 3: Bin Standard Deviation
    axes[0, 2].plot(rounds, bin_stds, 'b-o', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('Active Learning Round', fontsize=12)
    axes[0, 2].set_ylabel('Bin Standard Deviation', fontsize=12)
    axes[0, 2].set_title('Bin Standard Deviation vs Round', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xticks(rounds)

    # Add value labels
    for i, val in enumerate(bin_stds):
        axes[0, 2].annotate(f'{val:.3f}', (rounds[i], val),
                            textcoords="offset points", xytext=(0, 10), ha='center')

    # Plot 4: Spatial Consistency
    axes[1, 0].plot(rounds, spatial_consistencies, 'm-o', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Active Learning Round', fontsize=12)
    axes[1, 0].set_ylabel('Spatial Consistency', fontsize=12)
    axes[1, 0].set_title('Spatial Consistency vs Round', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(rounds)

    # Add value labels
    for i, val in enumerate(spatial_consistencies):
        axes[1, 0].annotate(f'{val:.3f}', (rounds[i], val),
                            textcoords="offset points", xytext=(0, 10), ha='center')

    # Plot 5: Performance Coverage
    axes[1, 1].plot(rounds, performance_coverages, 'c-o', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Active Learning Round', fontsize=12)
    axes[1, 1].set_ylabel('Performance Coverage', fontsize=12)
    axes[1, 1].set_title('Performance Coverage vs Round', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(rounds)

    # Add value labels
    for i, val in enumerate(performance_coverages):
        axes[1, 1].annotate(f'{val:.3f}', (rounds[i], val),
                            textcoords="offset points", xytext=(0, 10), ha='center')

    # Plot 6: Validation Metric (Dice or mAP)
    axes[1, 2].plot(rounds, val_metrics, 'orange', marker='o', linewidth=2, markersize=8)
    axes[1, 2].set_xlabel('Active Learning Round', fontsize=12)
    axes[1, 2].set_ylabel(f'Validation {metric_name}', fontsize=12)
    axes[1, 2].set_title(f'Validation {metric_name} vs Round', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xticks(rounds)
    axes[1, 2].set_ylim(0, 1)

    # Add value labels
    for i, val in enumerate(val_metrics):
        axes[1, 2].annotate(f'{val:.3f}', (rounds[i], val),
                            textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    spatial_metrics_path = os.path.join(output_dir, 'spatial_metrics_curves.png')
    plt.savefig(spatial_metrics_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Create separate combined spatial metrics plot
    _plot_combined_spatial_metrics(results, output_dir, session_name)

    return spatial_metrics_path


def _plot_combined_spatial_metrics(results, output_dir, session_name):
    """Plot combined spatial metrics (normalized) in a separate large plot."""
    rounds = list(range(1, len(results) + 1))

    # Detect task type from results
    task_type = results[0].get('task_type', 'segmentation') if results else 'segmentation'
    is_detection = task_type == 'detection'

    # Extract metrics
    worst_bin_dices = [result['spatial_metrics'].get('worst_bin_dice', 0) for result in results]
    p10_bin_dices = [result['spatial_metrics'].get('p10_bin_dice', 0) for result in results]
    bin_stds = [result['spatial_metrics'].get('bin_std', 0) for result in results]
    spatial_consistencies = [result['spatial_metrics'].get('spatial_consistency', 0) for result in results]
    performance_coverages = [result['spatial_metrics'].get('performance_coverage', 0) for result in results]

    # Get main metric based on task type
    if is_detection:
        val_metrics = [result.get('val_mAP', 0) for result in results]
        metric_name = 'mAP'
    else:
        val_metrics = [result.get('val_dice', 0) for result in results]
        metric_name = 'Dice'

    # Create large figure for combined metrics
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'{session_name}: Combined Spatial Metrics (Normalized)', fontsize=16, fontweight='bold')

    # Normalize all metrics to [0, 1] for comparison
    def normalize_metric(values):
        min_val, max_val = min(values), max(values)
        if max_val - min_val == 0:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    norm_worst = normalize_metric(worst_bin_dices)
    norm_p10 = normalize_metric(p10_bin_dices)
    norm_coverage = normalize_metric(performance_coverages)
    norm_metric = normalize_metric(val_metrics)
    # For consistency and std, lower is better, so invert
    norm_consistency = [1 - v for v in normalize_metric(spatial_consistencies)]
    norm_std = [1 - v for v in normalize_metric(bin_stds)]

    # Plot all normalized metrics
    ax.plot(rounds, norm_worst, 'r-o', linewidth=3, markersize=8, label='Worst Bin Dice')
    ax.plot(rounds, norm_p10, 'g-o', linewidth=3, markersize=8, label='P10 Bin Dice')
    ax.plot(rounds, norm_metric, 'orange', marker='o', linewidth=3, markersize=8, label=f'Validation {metric_name}')
    ax.plot(rounds, norm_coverage, 'c-o', linewidth=3, markersize=8, label='Performance Coverage')
    ax.plot(rounds, norm_consistency, 'm-o', linewidth=3, markersize=8, label='Spatial Consistency (inverted)')
    ax.plot(rounds, norm_std, 'b-o', linewidth=3, markersize=8, label='Bin Std (inverted)')

    ax.set_xlabel('Active Learning Round', fontsize=14)
    ax.set_ylabel('Normalized Score', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(rounds)
    ax.set_ylim(0, 1)

    # Fix legend position to top-left
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    plt.tight_layout()
    combined_metrics_path = os.path.join(output_dir, 'combined_spatial_metrics.png')
    plt.savefig(combined_metrics_path, dpi=150, bbox_inches='tight')
    plt.close()

    return combined_metrics_path


# =============================================================================
# Active Learning Experiment Utilities
# =============================================================================

def setup_experiment(args, default_output_dir='/team/team_pxi/workspace/juhojung/spatial_active_learning/al_results_2026'):
    """
    Setup Active Learning experiment environment and logging.

    Args:
        args: Parsed command line arguments
        default_output_dir: Default output directory

    Returns:
        device, output_dir, logger
    """
    from datetime import datetime

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (f"{args.dataset_source}_{args.target_lesion}_{args.collection}_{args.validate_data_mode}_"
                f"r_{args.round_num}_s_{args.num_samples}_val_{args.num_validation_samples}_b_{args.batch_size}")
    run_name = f"{args.mode}_{args.uncertainty_type}_{args.grid_width}x{args.grid_height}_{timestamp}"
    output_dir = os.path.join(getattr(args, 'output_dir', default_output_dir), exp_name, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(output_dir, rank=0)

    # Log experiment config
    logger.info("=" * 80)
    logger.info("Active Learning Experiment Started")
    logger.info("=" * 80)
    for key, val in vars(args).items():
        logger.info(f"{key}: {val}")
    logger.info(f"Output: {output_dir}")

    return device, output_dir, logger


def log_final_results(logger, results, selected_indices, output_dir, args):
    """
    Log final experiment results.

    Args:
        logger: Logger instance
        results: List of round results
        selected_indices: List of selected sample indices
        output_dir: Output directory path
        args: Experiment arguments
    """
    final = results[-1]
    spatial = final['spatial_metrics']

    # Determine if this is a detection task
    is_detection = 'val_mAP' in final or 'val_AP50' in final

    logger.info("=" * 80)
    logger.info("Active Learning Completed!")
    logger.info("=" * 80)
    logger.info(f"Final Val Loss: {final['val_loss']:.4f}")

    if is_detection:
        logger.info(f"Final Val mAP: {final.get('val_mAP', 0):.4f}")
        logger.info(f"Final Val AP50: {final.get('val_AP50', 0):.4f}")
        logger.info(f"Final Val FROC: {final.get('val_froc_score', 0):.4f}")
    else:
        logger.info(f"Final Val Dice: {final.get('val_dice', 0):.4f}")

    logger.info(f"Final Val IoU: {final.get('val_iou', 0):.4f}")
    logger.info(f"Worst Bin Dice: {spatial['worst_bin_dice']:.4f}")
    logger.info(f"Performance Coverage: {spatial['performance_coverage']:.4f}")
    logger.info(f"Spatial Consistency: {spatial['spatial_consistency']:.4f}")
    logger.info(f"Total samples: {len(selected_indices)}")

    # Print summary to console
    print(f"\n📊 Final Summary ({args.mode}):")
    if is_detection:
        print(
            f"   mAP: {final.get('val_mAP', 0):.4f}, AP50: {final.get('val_AP50', 0):.4f}, FROC: {final.get('val_froc_score', 0):.4f}")
    else:
        print(f"   Val Dice: {final.get('val_dice', 0):.4f}, IoU: {final.get('val_iou', 0):.4f}")
    print(f"   Worst Bin: {spatial['worst_bin_dice']:.4f}, Perf Coverage: {spatial['performance_coverage']:.4f}")
    print(f"   Total samples: {len(selected_indices)}")
    print(f"   Results: {output_dir}")
