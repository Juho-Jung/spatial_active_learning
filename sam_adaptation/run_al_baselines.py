#!/usr/bin/env python3
"""
Active Learning SAM Baselines

Single script for Active Learning experiments with SAM-based lesion segmentation.
Compares different selection strategies: random, uncertainty, area_random, uncertainty_area.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
# Import our modular components
from base_utils import (cleanup_ddp, create_session_dir,
                        plot_spatial_metrics_curves, plot_validation_curves,
                        set_seed, setup_ddp, setup_logger)
from dataset import (SAMLesionDataset, SpatialSplitDataset,
                     create_spatial_validation_split, load_raw_documents)
from losses import combo_loss
from metrics import (calculate_bin_dice_metrics, calculate_metrics,
                     calculate_performance_coverage,
                     calculate_spatial_consistency, divide_image_into_areas)
from models import MCDropoutSAMModel, SAMLesionModel, SegmentationModel
from sample_selection import create_spatial_bins, get_selection_strategy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from training import train_model_round
from uncertainty import get_uncertainty_method

# Suppress albumentations warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Set random seeds for reproducibility
set_seed(42)


def _parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Active Learning Baselines')
    parser.add_argument(
        '--sam_checkpoint', default="/opt/spatial_active_learning/sam_vit_b.pth", help='SAM checkpoint path')
    parser.add_argument('--vit_model', default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h'], help='SAM ViT model size')
    parser.add_argument('--mode', required=True,
                        choices=['random', 'uncertainty', 'area_random', 'uncertainty_area',
                                 'adaptive', 'adaptive_improved',
                                 'adaptive_multi_scale', 'adaptive_performance_monitoring',
                                 'diversity', 'diversity_uncertainty',
                                 # ULTRA AGGRESSIVE versions
                                 'adaptive_ultra', 'adaptive_improved_ultra',
                                 'adaptive_multi_scale_ultra', 'adaptive_performance_monitoring_ultra'],
                        help='Selection strategy')
    parser.add_argument('--uncertainty_type', default='none',
                        choices=['base', 'mc_dropout', 'tta', 'fast_lesionness', 'none'],
                        help='Uncertainty calculation method')
    parser.add_argument('--model_type', default='smp_efficientnet',
                        choices=['smp_efficientnet', 'swinunetr', 'smp'],
                        help='Model type')
    parser.add_argument('--target_lesion', default='calcifiednodule',
                        choices=['calcification', 'calcifiednodule', 'nodule', 'pneumoperitoneum'],
                        help='Target lesion type')
    parser.add_argument('--round_num', type=int, default=10, help='Number of AL rounds')
    parser.add_argument('--num_samples', type=int, default=20, help='Samples per round')
    parser.add_argument('--train_epochs', type=int, default=150, help='Training epochs per round')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument(
        '--output_dir', default='/team/team_pxi/workspace/juhojung/spatial_active_learning/al_results', help='Output directory')
    parser.add_argument('--mc_dropout_T', type=int, default=8, help='MC Dropout samples')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Early stopping minimum delta')
    parser.add_argument('--validate_data_mode', default='spatial_equal_split',
                        choices=['random_split', 'spatial_equal_split', 'spatial_dynamic_split'],
                        help='Validation data split mode: random_split (20/80 ratio), spatial_equal_split (equal per grid), spatial_dynamic_split (proportional per grid)')
    parser.add_argument('--collection', default='both',
                        choices=['validation_collection', 'train_collection', 'sdc_ppm_train-0908', 'both'],
                        help='Data collection to use: validation_collection or train_collection or sdc_ppm_train-0908 or both')
    parser.add_argument('--num_validation_samples', type=int, default=100,
                        help='Number of validation samples to use (default: use all)')
    parser.add_argument('--grid_width', type=int, default=5, help='Spatial grid width (number of columns)')
    parser.add_argument('--grid_height', type=int, default=5, help='Spatial grid height (number of rows)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()


def _setup_experiment(args):
    """Setup experiment environment and logging."""
    # Setup device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir, f"{args.target_lesion}_{args.collection}_{args.validate_data_mode}_r_{args.round_num}_s_{args.num_samples}_val_{args.num_validation_samples}", f"{args.mode}_{args.uncertainty_type}_{args.grid_width}x{args.grid_height}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(output_dir, rank=0)
    _log_experiment_start(logger, args, output_dir)

    return device, output_dir, logger


def _log_experiment_start(logger, args, output_dir):
    """Log experiment start information."""
    logger.info("=" * 80)
    logger.info("Active Learning SAM Baselines Started")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Uncertainty type: {args.uncertainty_type}")
    logger.info(f"Target lesion: {args.target_lesion}")
    logger.info(f"Rounds: {args.round_num}")
    logger.info(f"Samples per round: {args.num_samples}")
    logger.info(f"Training epochs per round: {args.train_epochs}")
    logger.info(f"Early stopping patience: {args.patience}")
    logger.info(f"Early stopping min delta: {args.min_delta}")
    logger.info(f"Validation data mode: {args.validate_data_mode}")
    logger.info(f"Data collection: {args.collection}")
    logger.info(f"Validation samples limit: {args.num_validation_samples if args.num_validation_samples else 'All'}")
    logger.info(f"Output directory: {output_dir}")


def _create_datasets(args):
    """Create training and validation datasets based on configuration."""
    if args.validate_data_mode == 'random_split':
        full_dataset, val_dataset = _create_random_split_datasets(args)
    elif args.validate_data_mode in ['spatial_equal_split', 'spatial_dynamic_split']:
        full_dataset, val_dataset = _create_spatial_split_datasets(args)
    else:
        raise ValueError(f"Unknown validation mode: {args.validate_data_mode}")

    return full_dataset, val_dataset


def _create_spatial_split_datasets(args):
    """Create datasets using spatial validation split."""
    print(f"📊 Loading raw documents for spatial validation split from {args.collection}...")
    raw_documents = load_raw_documents(args.collection, args.target_lesion)

    print("📊 Creating spatial validation split...")
    areas = divide_image_into_areas(grid_width=args.grid_width, grid_height=args.grid_height)
    train_docs, val_docs = create_spatial_validation_split(
        raw_documents,
        args.target_lesion,
        num_samples_per_round=args.num_samples,
        num_rounds=args.round_num,
        areas=areas,
        seed=args.seed,
        num_validation_samples=args.num_validation_samples,
        validation_mode=args.validate_data_mode)

    print("📊 Creating datasets from spatial split...")
    full_dataset = SpatialSplitDataset(train_docs, args.target_lesion)
    val_dataset = SpatialSplitDataset(val_docs, args.target_lesion)

    return full_dataset, val_dataset


def _create_random_split_datasets(args):
    """Create datasets using random validation split with validation sample limitation."""
    print(f"📊 Creating random validation split from {args.collection}...")

    # Load raw documents for efficient processing
    print(f"📊 Loading raw documents for random validation split from {args.collection}...")
    raw_documents = load_raw_documents(args.collection, args.target_lesion)

    # Filter positive documents (with target lesion)
    positive_docs = []
    for doc in raw_documents:
        objects = doc.get('objects', [])
        has_target_lesion = False
        for obj in objects:
            if obj.get('finding_name') == args.target_lesion:
                has_target_lesion = True
                break
        if has_target_lesion:
            positive_docs.append(doc)

    print(f"📊 Total positive documents: {len(positive_docs)}")

    # Shuffle documents
    np.random.seed(args.seed)
    np.random.shuffle(positive_docs)

    # Calculate validation samples
    if args.num_validation_samples is None:
        # Use 20% of data for validation (traditional random split)
        num_validation_samples = max(1, len(positive_docs) // 5)
    else:
        num_validation_samples = min(args.num_validation_samples, len(positive_docs))

    print(f"📊 Validation samples: {num_validation_samples}")

    # Split documents
    val_docs = positive_docs[:num_validation_samples]
    train_docs = positive_docs[num_validation_samples:]

    print(f"📊 Final split: {len(val_docs)} validation, {len(train_docs)} training samples")

    # Create datasets using pre-split documents (efficient loading)
    full_dataset = SpatialSplitDataset(train_docs, args.target_lesion)
    val_dataset = SpatialSplitDataset(val_docs, args.target_lesion)

    return full_dataset, val_dataset


def _create_model_for_uncertainty(args, device, previous_round_model_path=None):
    """Create model for uncertainty calculation."""
    if args.uncertainty_type == 'base':
        print("🔄 Creating SAM model for uncertainty calculation...")
        return SAMLesionModel(args.sam_checkpoint, args.vit_model).to(device)
    elif args.uncertainty_type == 'none':
        print("🔄 Creating segmentation model for uncertainty calculation...")
        model = SegmentationModel(args.model_type, device).to(device)
        # Load previous round checkpoint if available
        if previous_round_model_path and os.path.exists(previous_round_model_path):
            try:
                checkpoint = torch.load(previous_round_model_path, map_location=device)
                # training._save_model_checkpoint saves a dict with 'model_state_dict'
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                model.load_state_dict(state_dict, strict=False)
                print(f"📥 Loaded previous round model weights from: {previous_round_model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load previous round model from {previous_round_model_path}: {e}")
        return model
    else:  # mc_dropout
        print("🔄 Creating MC Dropout SAM model for uncertainty calculation...")
        return MCDropoutSAMModel(args.sam_checkpoint, args.vit_model).to(device)


def _create_model_for_training(args, device):
    """Create model for training."""
    if args.uncertainty_type == 'mc_dropout' and (args.mode == 'uncertainty' or args.mode == 'uncertainty_area'):
        print("🔄 Creating MC Dropout SAM model for training...")
        return MCDropoutSAMModel(args.sam_checkpoint, args.vit_model).to(device)
    elif args.uncertainty_type == 'none':
        print("🔄 Creating segmentation model for training...")
        return SegmentationModel(args.model_type, device).to(device)
    else:
        print("🔄 Creating standard SAM model for training...")
        return SAMLesionModel(args.sam_checkpoint, args.vit_model).to(device)


def _calculate_uncertainties(args, model, full_dataset, pool_indices, selected_indices, device):
    """Calculate uncertainties for sample selection."""
    if args.mode not in ['uncertainty', 'uncertainty_area', 'adaptive', 'adaptive_improved', 'adaptive_multi_scale', 'adaptive_performance_monitoring', 'diversity_uncertainty',
                         'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_multi_scale_ultra', 'adaptive_performance_monitoring_ultra']:
        return None

    uncertainty_func = get_uncertainty_method(args.uncertainty_type)

    if not selected_indices:
        return None  # Will trigger random selection in first round

    # Create dataloader for uncertainty calculation
    temp_dataset = torch.utils.data.Subset(full_dataset, pool_indices)
    temp_loader = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=False)

    # For adaptive modes, we need detailed predictions and uncertainties
    if args.mode in ['adaptive', 'adaptive_improved', 'adaptive_hybrid', 'adaptive_multi_scale', 'adaptive_performance_monitoring',
                     'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_multi_scale_ultra', 'adaptive_performance_monitoring_ultra']:
        if args.uncertainty_type == 'fast_lesionness':
            return uncertainty_func(model, temp_loader, device, return_detailed=True)
        elif args.uncertainty_type == 'mc_dropout':
            return uncertainty_func(model, temp_loader, device, args.mc_dropout_T, return_detailed=True)
        elif args.uncertainty_type == 'none':
            return uncertainty_func(model, temp_loader, device, return_detailed=True)
        else:
            return uncertainty_func(model, temp_loader, device, return_detailed=True)
    elif args.mode in ['diversity', 'diversity_uncertainty']:
        # For diversity modes, we need features
        if args.uncertainty_type == 'mc_dropout':
            return uncertainty_func(model, temp_loader, device, args.mc_dropout_T, return_features=True)
        elif args.uncertainty_type == 'none':
            return uncertainty_func(model, temp_loader, device, return_features=True)
        else:
            return uncertainty_func(model, temp_loader, device, return_features=True)
    else:
        # For other modes, return simple uncertainties
        if args.uncertainty_type == 'mc_dropout':
            return uncertainty_func(model, temp_loader, device, args.mc_dropout_T)
        elif args.uncertainty_type == 'none':
            return uncertainty_func(model, temp_loader, device)
        else:
            return uncertainty_func(model, temp_loader, device)


def _create_spatial_bins_for_dataset(full_dataset, pool_indices, args):
    """Create spatial bin assignments for the dataset."""
    # Get the first sample to determine image dimensions
    if len(pool_indices) > 0:
        sample = full_dataset[pool_indices[0]]
        if isinstance(sample, tuple) and len(sample) >= 2:
            # Assuming sample is (image, mask) tuple
            image = sample[0]
            if hasattr(image, 'shape'):
                height, width = image.shape[-2], image.shape[-1]
            else:
                # Fallback dimensions
                height, width = 256, 256
        else:
            height, width = 256, 256
    else:
        height, width = 256, 256

    return create_spatial_bins(height, width, grid_width=args.grid_width, grid_height=args.grid_height)


def _select_samples(args, pool_indices, uncertainties, selected_indices, areas, round_idx):
    """Select samples using the configured strategy."""
    selection_func = get_selection_strategy(args.mode)

    # Define seeds for first round to ensure fair comparison
    first_round_seed = 42 if round_idx == 0 else None

    # uncertainties are already calculated for pool_indices only, no need to filter again
    filtered_uncertainties = uncertainties

    # Select samples using the strategy function with correct parameters
    if args.mode == 'random':
        return selection_func(pool_indices, args.num_samples, selected_indices, first_round_seed)
    elif args.mode == 'uncertainty':
        return selection_func(pool_indices, filtered_uncertainties, args.num_samples, selected_indices, first_round_seed)
    elif args.mode == 'area_random':
        return selection_func(pool_indices, args.num_samples, selected_indices, areas, first_round_seed)
    elif args.mode == 'uncertainty_area':
        return selection_func(pool_indices, filtered_uncertainties, args.num_samples, selected_indices, areas, first_round_seed)
    elif args.mode == 'diversity':
        features = getattr(args, 'features', None)
        return selection_func(pool_indices, filtered_uncertainties, args.num_samples, selected_indices, areas, features, first_round_seed)
    elif args.mode == 'diversity_uncertainty':
        features = getattr(args, 'features', None)
        return selection_func(pool_indices, filtered_uncertainties, args.num_samples, selected_indices, areas, features, first_round_seed)
    elif args.mode == 'adaptive':
        # For adaptive selection, we need additional parameters
        # These should be passed from the main function or computed here
        probs = getattr(args, 'probs', None)
        lesionness = getattr(args, 'lesionness', None)
        bin_id = getattr(args, 'bin_id', None)
        lambda1 = getattr(args, 'lambda1', 1.0)

        # probs, lesionness, and bin_id are already calculated for pool_indices only

        return selection_func(pool_indices, filtered_uncertainties, args.num_samples, selected_indices,
                              probs, lesionness, bin_id, lambda1, first_round_seed)
    elif args.mode == 'adaptive_improved':
        # For improved adaptive selection, we need additional parameters
        probs = getattr(args, 'probs', None)
        lesionness = getattr(args, 'lesionness', None)
        bin_id = getattr(args, 'bin_id', None)
        lambda1 = getattr(args, 'lambda1', 1.0)

        return selection_func(pool_indices, filtered_uncertainties, args.num_samples, selected_indices,
                              probs, lesionness, bin_id, lambda1, first_round_seed)
    elif args.mode == 'adaptive_hybrid':
        # For hybrid adaptive selection, we need additional parameters
        probs = getattr(args, 'probs', None)
        lesionness = getattr(args, 'lesionness', None)
        bin_id = getattr(args, 'bin_id', None)
        lambda1 = getattr(args, 'lambda1', 1.0)

        return selection_func(pool_indices, filtered_uncertainties, args.num_samples, selected_indices,
                              probs, lesionness, bin_id, lambda1, first_round_seed)
    elif args.mode == 'adaptive_multi_scale':
        # For multi-scale adaptive selection, we need additional parameters
        probs = getattr(args, 'probs', None)
        lesionness = getattr(args, 'lesionness', None)
        bin_id = getattr(args, 'bin_id', None)
        lambda1 = getattr(args, 'lambda1', 1.0)
        grid_scales = [(3, 3), (5, 5), (7, 7)]  # Default grid scales

        return selection_func(pool_indices, filtered_uncertainties, args.num_samples, selected_indices,
                              probs, lesionness, bin_id, lambda1, grid_scales, first_round_seed)
    elif args.mode == 'adaptive_performance_monitoring':
        # For performance monitoring adaptive selection, we need additional parameters
        probs = getattr(args, 'probs', None)
        lesionness = getattr(args, 'lesionness', None)
        bin_id = getattr(args, 'bin_id', None)
        lambda1 = getattr(args, 'lambda1', 1.0)
        performance_history = getattr(args, 'performance_history', None)

        return selection_func(pool_indices, filtered_uncertainties, args.num_samples, selected_indices,
                              probs, lesionness, bin_id, lambda1, performance_history, first_round_seed)
    elif args.mode == 'adaptive_ultra':
        # For ULTRA AGGRESSIVE adaptive selection, we need additional parameters
        probs = getattr(args, 'probs', None)
        lesionness = getattr(args, 'lesionness', None)
        bin_id = getattr(args, 'bin_id', None)
        lambda1 = getattr(args, 'lambda1', 1.0)

        return selection_func(pool_indices, filtered_uncertainties, args.num_samples, selected_indices,
                              probs, lesionness, bin_id, lambda1, first_round_seed)
    elif args.mode == 'adaptive_improved_ultra':
        # For ULTRA AGGRESSIVE improved adaptive selection, we need additional parameters
        probs = getattr(args, 'probs', None)
        lesionness = getattr(args, 'lesionness', None)
        bin_id = getattr(args, 'bin_id', None)
        lambda1 = getattr(args, 'lambda1', 1.0)

        return selection_func(pool_indices, filtered_uncertainties, args.num_samples, selected_indices,
                              probs, lesionness, bin_id, lambda1, first_round_seed)
    elif args.mode == 'adaptive_multi_scale_ultra':
        # For ULTRA AGGRESSIVE multi-scale adaptive selection, we need additional parameters
        probs = getattr(args, 'probs', None)
        lesionness = getattr(args, 'lesionness', None)
        bin_id = getattr(args, 'bin_id', None)
        lambda1 = getattr(args, 'lambda1', 1.0)
        grid_scales = [(3, 3), (5, 5), (7, 7)]  # Default grid scales

        return selection_func(pool_indices, filtered_uncertainties, args.num_samples, selected_indices,
                              probs, lesionness, bin_id, lambda1, grid_scales, first_round_seed)
    elif args.mode == 'adaptive_performance_monitoring_ultra':
        # For ULTRA AGGRESSIVE performance monitoring adaptive selection, we need additional parameters
        probs = getattr(args, 'probs', None)
        lesionness = getattr(args, 'lesionness', None)
        bin_id = getattr(args, 'bin_id', None)
        lambda1 = getattr(args, 'lambda1', 1.0)
        performance_history = getattr(args, 'performance_history', None)

        return selection_func(pool_indices, filtered_uncertainties, args.num_samples, selected_indices,
                              probs, lesionness, bin_id, lambda1, performance_history, first_round_seed)
    else:
        raise ValueError(f"Unknown selection mode: {args.mode}")


def _log_round_results(logger, round_idx, new_indices, selected_indices, val_loss, val_dice, val_bin_metrics):
    """Log results for current round."""
    print(f"✅ Round {round_idx + 1} completed - Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
    print(
        f"   📊 Spatial Metrics - Worst Bin: {val_bin_metrics['worst_bin_dice']:.4f}, Perf Coverage: {val_bin_metrics['performance_coverage']:.4f}, Spatial Consistency: {val_bin_metrics['spatial_consistency']:.4f}")
    logger.info(f"Round {round_idx + 1} completed - Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
    logger.info(
        f"Spatial Metrics - Worst Bin: {val_bin_metrics['worst_bin_dice']:.4f}, Perf Coverage: {val_bin_metrics['performance_coverage']:.4f}, Spatial Consistency: {val_bin_metrics['spatial_consistency']:.4f}")


def _save_results(results, output_dir):
    """Save results to JSON file."""
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)


def _generate_plots(results, output_dir, session_name):
    """Generate and save plots."""
    # Plot validation curves
    loss_path, dice_path = plot_validation_curves(results, output_dir, session_name)
    print(f"📊 Validation loss curve saved: {loss_path}")
    print(f"📊 Validation dice curve saved: {dice_path}")

    # Plot spatial metrics curves
    spatial_metrics_path = plot_spatial_metrics_curves(results, output_dir, session_name)
    print(f"📊 Spatial metrics curves saved: {spatial_metrics_path}")

    # Combined spatial metrics plot is created within plot_spatial_metrics_curves
    combined_metrics_path = os.path.join(output_dir, 'combined_spatial_metrics.png')
    print(f"📊 Combined spatial metrics plot saved: {combined_metrics_path}")


def _log_final_results(logger, results, selected_indices, output_dir):
    """Log final experiment results."""
    final_spatial = results[-1]['spatial_metrics']
    logger.info("=" * 80)
    logger.info("Active Learning Completed!")
    logger.info("=" * 80)
    logger.info(f"Final validation Loss: {results[-1]['val_loss']:.4f}")
    logger.info(f"Final validation Dice: {results[-1]['val_dice']:.4f}")
    logger.info(f"Final spatial metrics:")
    logger.info(f"  Worst Bin Dice: {final_spatial['worst_bin_dice']:.4f}")
    logger.info(f"  P10 Bin Dice: {final_spatial['p10_bin_dice']:.4f}")
    logger.info(f"  Bin Std: {final_spatial['bin_std']:.4f}")
    logger.info(f"  Min/Max Bin Dice: {final_spatial['min_bin_dice']:.4f}/{final_spatial['max_bin_dice']:.4f}")
    logger.info(f"  Spatial Consistency: {final_spatial['spatial_consistency']:.4f}")
    logger.info(f"  Performance Coverage: {final_spatial['performance_coverage']:.4f}")
    logger.info(f"Total samples used: {len(selected_indices)}")
    logger.info(f"Results saved to: {output_dir}")


def _print_final_summary(args, results, selected_indices, output_dir):
    """Print final summary to console."""
    final_spatial = results[-1]['spatial_metrics']
    print(f"\n📊 Final Summary:")
    print(f"   Strategy: {args.mode}")
    print(f"   Uncertainty: {args.uncertainty_type}")
    print(f"   Target lesion: {args.target_lesion}")
    print(f"   Final Val Loss: {results[-1]['val_loss']:.4f}")
    print(f"   Final Val Dice: {results[-1]['val_dice']:.4f}")
    print(f"   📊 Spatial Metrics:")
    print(f"     Worst Bin Dice: {final_spatial['worst_bin_dice']:.4f}")
    print(f"     P10 Bin Dice: {final_spatial['p10_bin_dice']:.4f}")
    print(f"     Bin Std: {final_spatial['bin_std']:.4f}")
    print(f"     Min/Max Bin Dice: {final_spatial['min_bin_dice']:.4f}/{final_spatial['max_bin_dice']:.4f}")
    print(f"     Spatial Consistency: {final_spatial['spatial_consistency']:.4f}")
    print(f"     Performance Coverage: {final_spatial['performance_coverage']:.4f}")
    print(f"   Total samples: {len(selected_indices)}")
    print(f"   Results directory: {output_dir}")


def main():
    """Main function for Active Learning Baselines."""
    args = _parse_arguments()

    # Setup experiment environment
    device, output_dir, logger = _setup_experiment(args)

    # Create datasets
    full_dataset, val_dataset = _create_datasets(args)

    # Store original validation dataset size for logging
    original_val_size = len(val_dataset)

    # Initialize experiment variables
    pool_indices = list(range(len(full_dataset)))
    selected_indices = []
    areas = divide_image_into_areas(grid_width=args.grid_width, grid_height=args.grid_height)
    results = []
    prev_bin_dices = None

    print(f"🎯 Starting Active Learning with {args.mode} strategy...")
    print(f"📊 Total pool size: {len(pool_indices)}")
    print(f"📊 Validation set size: {len(val_dataset)} (original: {original_val_size})")
    if args.num_validation_samples is not None:
        print(f"📊 Validation samples limited to: {args.num_validation_samples}")

    # Active Learning rounds
    previous_round_model_path = None
    for round_idx in range(args.round_num):
        print(f"\n🔄 Round {round_idx + 1}/{args.round_num}")
        logger.info(f"Starting Round {round_idx + 1}/{args.round_num}")

        # Check SAM checkpoint
        if not os.path.exists(args.sam_checkpoint):
            print(f"❌ SAM checkpoint not found: {args.sam_checkpoint}")
            logger.error(f"SAM checkpoint not found: {args.sam_checkpoint}")
            return

        # Calculate uncertainties if needed
        uncertainties = None
        if args.mode in ['uncertainty', 'uncertainty_area', 'adaptive', 'adaptive_improved',
                         'adaptive_multi_scale', 'adaptive_performance_monitoring', 'diversity', 'diversity_uncertainty',
                         'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_multi_scale_ultra', 'adaptive_performance_monitoring_ultra']:
            model = _create_model_for_uncertainty(args, device, previous_round_model_path)
            uncertainty_data = _calculate_uncertainties(
                args, model, full_dataset, pool_indices, selected_indices, device)

            if args.mode in ['adaptive', 'adaptive_improved', 'adaptive_multi_scale', 'adaptive_performance_monitoring',
                             'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_multi_scale_ultra', 'adaptive_performance_monitoring_ultra'] and isinstance(uncertainty_data, dict):
                # Extract uncertainties and prepare additional data for adaptive selection
                uncertainties = uncertainty_data['uncertainties']
                args.probs = uncertainty_data['predictions']  # Custom decoder predictions

                if 'lesionness' in uncertainty_data:
                    args.lesionness = uncertainty_data['lesionness']
                elif 'sam_predictions' in uncertainty_data:
                    args.lesionness = uncertainty_data['sam_predictions']
                else:
                    args.lesionness = uncertainty_data.get('predictions', None)

                args.bin_id = _create_spatial_bins_for_dataset(full_dataset, pool_indices, args)
                args.lambda1 = getattr(args, 'lambda1', 1.0)  # Default lambda1 value
            elif args.mode in ['diversity', 'diversity_uncertainty'] and isinstance(uncertainty_data, dict):
                # Extract uncertainties and features for diversity selection
                uncertainties = uncertainty_data['uncertainties']
                args.features = uncertainty_data['features']  # Feature embeddings for diversity
            else:
                uncertainties = uncertainty_data

        # Select samples
        new_indices = _select_samples(args, pool_indices, uncertainties, selected_indices, areas, round_idx)
        selected_indices.extend(new_indices)

        print(f"📊 Selected {len(new_indices)} new samples (total: {len(selected_indices)})")
        if round_idx == 0:
            print(f"🎲 Using fixed seed 42 for first round to ensure fair comparison")
            logger.info(f"Using fixed seed 42 for first round to ensure fair comparison")
        logger.info(f"Selected {len(new_indices)} new samples (total: {len(selected_indices)})")

        # Create training dataset and train model
        train_dataset = torch.utils.data.Subset(full_dataset, selected_indices)
        model = _create_model_for_training(args, device)

        print("🏋️ Training model...")
        val_loss, val_dice, val_bin_metrics, current_bin_dices, round_model_save_path = train_model_round(
            model, train_dataset, val_dataset, device, args.train_epochs, args.batch_size,
            round_idx + 1, output_dir, args.patience, args.min_delta, prev_bin_dices,
            args.grid_width, args.grid_height, args.uncertainty_type)

        # Save round results
        round_result = {'round': round_idx + 1,
                        'selected_indices': new_indices,
                        'total_selected': len(selected_indices),
                        'val_loss': float(val_loss),
                        'val_dice': float(val_dice),
                        'spatial_metrics': {key: float(val) for key, val in val_bin_metrics.items()}}
        results.append(round_result)

        # Log round results
        _log_round_results(logger, round_idx, new_indices, selected_indices, val_loss, val_dice, val_bin_metrics)

        # Update for next round
        prev_bin_dices = current_bin_dices
        previous_round_model_path = round_model_save_path
        _save_results(results, output_dir)

    # Final results and cleanup
    print(f"\n🎉 Active Learning completed!")
    print(f"📁 Results saved to: {output_dir}")

    # Create session name for plots
    session_name = f"{args.target_lesion}_{args.mode}"

    _generate_plots(results, output_dir, session_name)
    _log_final_results(logger, results, selected_indices, output_dir)
    _print_final_summary(args, results, selected_indices, output_dir)


if __name__ == "__main__":
    main()
