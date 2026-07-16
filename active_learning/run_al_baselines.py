#!/usr/bin/env python3
"""
Active Learning Baselines for Lesion Segmentation

Supports: random, uncertainty, area-based, adaptive, diversity, coreset, and more.
"""

import argparse
import json
import os

import torch
from base_utils import (log_final_results, plot_spatial_metrics_curves,
                        plot_validation_curves, setup_experiment)
from dataset import create_al_datasets
from metrics import divide_image_into_areas
from models import create_model
from models.detection_models import create_detection_model
from sample_selection import create_spatial_bins, get_selection_strategy
from torch.utils.data import DataLoader
from training import train_model_round
from uncertainty import get_uncertainty_method

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


# =============================================================================
# Strategy Groups
# =============================================================================
RANDOM_STRATEGIES = {'random', 'area_random'}
UNCERTAINTY_STRATEGIES = {'uncertainty', 'uncertainty_area', 'mcd_alunet'}
ADAPTIVE_STRATEGIES = {
    'adaptive', 'adaptive_improved', 'adaptive_multi_scale',
    'adaptive_performance_monitoring', 'adaptive_ultra', 'adaptive_improved_ultra',
    'adaptive_multi_scale_ultra', 'adaptive_performance_monitoring_ultra',
    'sparcl', 'sparcl_prefiltering', 'sparcl_det',  # Paper-exact implementation (sparcl_det for detection)
    # SPARCL ablation variants for MICCAI 2026 rebuttal
    'sparcl_no_gating', 'sparcl_fixed_lambda', 'sparcl_no_coverage',
    'sparcl_det_no_gating', 'sparcl_det_fixed_lambda'
}
DIVERSITY_STRATEGIES = {'diversity', 'diversity_uncertainty', 'coreset'}
SPECIAL_STRATEGIES = {'taudis', 'usimc', 'lunit'}

ALL_STRATEGIES = (
    RANDOM_STRATEGIES | UNCERTAINTY_STRATEGIES | ADAPTIVE_STRATEGIES |
    DIVERSITY_STRATEGIES | SPECIAL_STRATEGIES
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Active Learning Baselines')

    # Model & AL settings
    parser.add_argument('--model_type', default='smp_efficientnet',
                        choices=['smp_efficientnet', 'swinunetr', 'smp'])
    parser.add_argument('--mode', default='random', choices=sorted(ALL_STRATEGIES),
                        help='AL strategy (ignored if --full_training is used)')
    parser.add_argument('--uncertainty_type', default='none',
                        choices=['none', 'mc_dropout', 'tta'])

    # Data settings
    parser.add_argument('--dataset_source', default='mdb',
                        choices=['mdb', 'vindr', 'siim', 'chestxdet10'],
                        help='Dataset source: mdb (MongoDB), vindr (VinDr-CXR), siim (SIIM-ACR Pneumothorax), or chestxdet10 (ChestX-Det10)')
    parser.add_argument('--target_lesion', default=['calcifiednodule'], nargs='+',
                        choices=['calcification', 'calcifiednodule', 'nodule', 'pneumoperitoneum',
                                 'cardiomegaly', 'pleural_effusion', 'consolidation', 'pneumothorax',
                                 'aortic_enlargement', 'infiltration', 'lung_opacity',
                                 # ChestX-Det10 specific lesions
                                 'atelectasis', 'effusion', 'emphysema', 'fibrosis', 'fracture', 'mass',
                                 # Special keywords
                                 'concentrated', 'dispersed', 'all'],
                        help='Target lesion type(s). Can specify multiple lesions (e.g., --target_lesion nodule calcification). '
                             'Images containing ANY of the specified lesions will be included. '
                             'Special values: "concentrated" (all concentrated lesions), "dispersed" (all dispersed lesions), "all" (all available lesions)')
    parser.add_argument('--collection', default='both',
                        choices=['validation_collection', 'train_collection', 'sdc_ppm_train-0908', 'both'],
                        help='MongoDB collection name (only used when dataset_source=mdb)')
    parser.add_argument('--vindr_root', default='/team/team_pxi/pxi-dataset/cxr/public/vinbig',
                        help='Root directory for VinDr-CXR dataset (only used when dataset_source=vindr)')
    parser.add_argument('--siim_root',
                        default='/team/team_pxi/pxi-dataset/cxr/public/siim-full/input/input/train',
                        help='Root directory for SIIM dataset (only used when dataset_source=siim)')
    parser.add_argument('--siim_positive_only', action='store_true',
                        help='Only use positive samples (with pneumothorax) in SIIM dataset')
    parser.add_argument('--include_negative', action='store_true',
                        help='Include ALL negative samples (images without target lesion) in the dataset. '
                             'By default, only positive samples are used. Applies to mdb, vindr, chestxdet10.')
    parser.add_argument('--chestxdet10_root',
                        default='/team/team_pxi/pxi-dataset/cxr/public/ChestX-Det10-Dataset',
                        help='Root directory for ChestX-Det10 dataset (only used when dataset_source=chestxdet10)')
    parser.add_argument('--chestxdet10_mask_type', default='detection',
                        choices=['detection', 'rectangular', 'ellipse', 'gaussian'],
                        help='ChestX-Det10 mask type: detection (return bboxes), or convert bbox to mask')
    parser.add_argument('--min_radiologist_agreement', type=int, default=1,
                        help='Minimum number of radiologists agreeing on annotation (for VinDr-CXR)')
    parser.add_argument('--vindr_mask_type', default='detection',
                        choices=['detection', 'rectangular', 'ellipse', 'gaussian'],
                        help='VinDr-CXR mask type: detection (return bboxes), or convert bbox to mask (rectangular/ellipse/gaussian)')
    parser.add_argument('--validate_data_mode', default='spatial_equal_split',
                        choices=['random_split', 'spatial_equal_split', 'spatial_dynamic_split'])
    parser.add_argument('--use_test_split', action='store_true', default=True,
                        help='For VinDr-CXR: use dicom/test for validation (True) or split train data (False)')
    # Training settings
    parser.add_argument('--round_num', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--train_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.001)
    parser.add_argument('--full_training', action='store_true',
                        help='Full dataset training mode (no AL, 80:20 train/val split) for baseline comparison')

    # Spatial & other settings
    parser.add_argument('--grid_width', type=int, default=5)
    parser.add_argument('--grid_height', type=int, default=5)
    parser.add_argument('--num_validation_samples', type=int, default=100)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir',
                        default='/team/team_pxi/workspace/juhojung/spatial_active_learning/al_results_2026')
    parser.add_argument('--mc_dropout_T', type=int, default=8)
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='lambda_max for SPARCL spatial coverage term')

    return parser.parse_args()


def calculate_uncertainties(args, model, full_dataset, pool_indices, selected_indices, device):
    """Calculate uncertainties for sample selection."""
    if args.mode in RANDOM_STRATEGIES or not selected_indices:
        return None

    temp_dataset = torch.utils.data.Subset(full_dataset, pool_indices)

    # Determine task type and select appropriate uncertainty method
    task_type = 'detection' if (
        (hasattr(args, 'dataset_source') and args.dataset_source == 'vindr' and
         hasattr(args, 'vindr_mask_type') and args.vindr_mask_type == 'detection') or
        (hasattr(args, 'dataset_source') and args.dataset_source == 'chestxdet10' and
         getattr(args, 'chestxdet10_mask_type', 'detection') == 'detection')
    ) else 'segmentation'

    # Create DataLoader with appropriate collate function
    if task_type == 'detection':
        # For detection, use custom collate function
        from training import detection_collate_fn
        temp_loader = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=False, collate_fn=detection_collate_fn)
        # Use detection-specific uncertainty method (even for 'none' type)
        uncertainty_type = 'detection'
    else:
        temp_loader = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)
        uncertainty_type = args.uncertainty_type

    uncertainty_func = get_uncertainty_method(uncertainty_type)
    need_detailed = args.mode in ADAPTIVE_STRATEGIES or args.mode == 'taudis' or args.mode in {'sparcl_det', 'sparcl_det_no_gating', 'sparcl_det_fixed_lambda'}
    need_features = args.mode in DIVERSITY_STRATEGIES or args.mode == 'taudis'

    kwargs = {'return_detailed': need_detailed, 'return_features': need_features}
    if uncertainty_type == 'mc_dropout':
        kwargs['T'] = args.mc_dropout_T

    return uncertainty_func(model, temp_loader, device, **kwargs)


def prepare_selection_args(args, model, full_dataset, pool_indices, selected_indices,
                           uncertainty_data, device):
    """Prepare arguments for sample selection."""
    if isinstance(uncertainty_data, dict):
        uncertainties = uncertainty_data.get('uncertainties')
        args.features = uncertainty_data.get('features')
        # For detection: 'probs' is a bbox-based probability map (from calculate_uncertainty_detection).
        # For segmentation: uncertainty functions return 'predictions' (pixel-wise probability maps),
        # which is what SPARCL expects as probs. Fall back to 'predictions' if 'probs' is absent.
        args.probs = uncertainty_data.get('probs')
        if args.probs is None:
            args.probs = uncertainty_data.get('predictions')
        args.lesionness = uncertainty_data.get('lesionness')
        # For detection SPARCL, use predictions directly (bboxes)
        args.predictions = uncertainty_data.get('predictions')  # Detection predictions (bboxes)
    else:
        uncertainties = uncertainty_data
        args.features = args.probs = args.lesionness = args.predictions = None

    # Create spatial bins for adaptive strategies
    args.bin_id = None
    args.areas = None
    if args.mode in ADAPTIVE_STRATEGIES and pool_indices:
        sample = full_dataset[pool_indices[0]]
        if args.mode in {'sparcl_det', 'sparcl_det_no_gating', 'sparcl_det_fixed_lambda'}:
            # For detection, get image size from dataset or use default
            h, w = 512, 512  # Default for detection
            if hasattr(full_dataset, 'target_size'):
                h, w = full_dataset.target_size
            args.bin_id = create_spatial_bins(h, w, grid_width=args.grid_width, grid_height=args.grid_height)
            # Also create areas for easier bbox overlap calculation
            from metrics.metrics import divide_image_into_areas
            args.areas = divide_image_into_areas(image_size=(h, w),
                                                 grid_width=args.grid_width,
                                                 grid_height=args.grid_height)
        else:
            h, w = (sample[0].shape[-2], sample[0].shape[-1]) if isinstance(sample, tuple) else (256, 256)
            args.bin_id = create_spatial_bins(h, w, grid_width=args.grid_width, grid_height=args.grid_height)

    args.lambda1 = getattr(args, 'lambda1', 1.0)

    # For USIMC and LUNIT
    if args.mode in {'usimc', 'lunit'}:
        available = [idx for idx in pool_indices if idx not in selected_indices]
        args.model = model
        args.dataloader = DataLoader(torch.utils.data.Subset(full_dataset, available),
                                     batch_size=args.batch_size, shuffle=False,
                                     num_workers=4, pin_memory=True)
        args.device = device

    return uncertainties


def select_samples(args, pool_indices, uncertainties, selected_indices, areas, round_idx):
    """Select samples using the configured strategy."""
    func = get_selection_strategy(args.mode)
    seed = 42 if round_idx == 0 else None

    if args.mode == 'random':
        return func(pool_indices, args.num_samples, selected_indices, seed)
    if args.mode == 'area_random':
        return func(pool_indices, args.num_samples, selected_indices, areas, seed)
    if args.mode == 'uncertainty':
        return func(pool_indices, uncertainties, args.num_samples, selected_indices, seed)
    if args.mode == 'uncertainty_area':
        return func(pool_indices, uncertainties, args.num_samples, selected_indices, areas, seed)
    if args.mode == 'mcd_alunet':
        return func(pool_indices, uncertainties, args.num_samples, selected_indices, areas, seed)
    if args.mode in DIVERSITY_STRATEGIES:
        return func(pool_indices, uncertainties, args.num_samples, selected_indices,
                    areas, args.features, seed)
    if args.mode == 'taudis':
        return func(pool_indices, uncertainties, args.num_samples, selected_indices,
                    areas, args.features, args.probs, 2.5, 1.5, 0.8, seed)
    if args.mode == 'usimc':
        return func(pool_indices, uncertainties, args.num_samples, selected_indices,
                    areas, args.features, args.model, args.dataloader, args.device, seed)
    if args.mode == 'lunit':
        return func(pool_indices, uncertainties, args.num_samples, selected_indices,
                    args.model, args.dataloader, args.device, seed)
    if args.mode in {'adaptive', 'adaptive_ultra', 'adaptive_improved', 'adaptive_improved_ultra'}:
        return func(pool_indices, uncertainties, args.num_samples, selected_indices,
                    args.probs, args.lesionness, args.bin_id, args.lambda1, seed)
    if args.mode in {'sparcl', 'sparcl_prefiltering'}:
        return func(pool_indices, uncertainties, args.num_samples, selected_indices,
                    args.probs, args.lesionness, args.bin_id, args.lambda1, seed)
    if args.mode in {'sparcl_no_gating', 'sparcl_fixed_lambda', 'sparcl_no_coverage'}:
        return func(pool_indices, uncertainties, args.num_samples, selected_indices,
                    args.probs, args.lesionness, args.bin_id, args.lambda1, seed)
    if args.mode in {'sparcl_det', 'sparcl_det_no_gating', 'sparcl_det_fixed_lambda'}:
        # Detection-specific SPARCL: uses predictions (bboxes) instead of probs/lesionness
        return func(pool_indices, uncertainties, args.num_samples, selected_indices,
                    args.predictions, args.bin_id, args.areas, args.lambda1, seed)
    if args.mode in {'adaptive_multi_scale', 'adaptive_multi_scale_ultra'}:
        return func(pool_indices, uncertainties, args.num_samples, selected_indices,
                    args.probs, args.lesionness, args.bin_id, args.lambda1,
                    [(3, 3), (5, 5), (7, 7)], seed)
    if args.mode in {'adaptive_performance_monitoring', 'adaptive_performance_monitoring_ultra'}:
        return func(pool_indices, uncertainties, args.num_samples, selected_indices,
                    args.probs, args.lesionness, args.bin_id, args.lambda1,
                    getattr(args, 'performance_history', None), seed)

    raise ValueError(f"Unknown mode: {args.mode}")


def save_results(results, output_dir):
    """Save results to JSON."""
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)


def run_full_training(args, device, output_dir, logger):
    """Full dataset training mode (80:20 split, no Active Learning)."""
    from torch.utils.data import random_split

    # Temporarily set high round_num to get ALL data (not limited by AL sampling)
    original_round_num = args.round_num
    original_num_samples = args.num_samples
    args.round_num = 99999  # Very high to include all data
    args.num_samples = 99999

    # Create full dataset (no AL splits)
    full_dataset, _ = create_al_datasets(args)

    # Restore original values
    args.round_num = original_round_num
    args.num_samples = original_num_samples

    # 80:20 train/val split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"📊 Full Training Mode: Total={total_size}, Train={train_size}, Val={val_size}")
    logger.info(f"Full Training Mode: Total={total_size}, Train={train_size}, Val={val_size}")

    # Create model
    model = create_model(args, device)

    # Determine task type
    task_type = 'detection' if (
        (hasattr(args, 'dataset_source') and args.dataset_source == 'vindr' and
         hasattr(args, 'vindr_mask_type') and args.vindr_mask_type == 'detection') or
        (hasattr(args, 'dataset_source') and args.dataset_source == 'chestxdet10' and
         getattr(args, 'chestxdet10_mask_type', 'detection') == 'detection')
    ) else 'segmentation'

    areas = divide_image_into_areas(grid_width=args.grid_width, grid_height=args.grid_height)

    print("🏋️ Training on full dataset...")
    val_loss, val_dice, val_iou, val_bin_metrics, current_bin_dices, checkpoint = train_model_round(
        model, train_dataset, val_dataset, device,
        args.train_epochs, args.batch_size, 1, output_dir,
        args.patience, args.min_delta, None,
        args.grid_width, args.grid_height, args.uncertainty_type, task_type=task_type)

    # Save results
    if task_type == 'detection':
        results = [{
            'mode': 'full_training',
            'task_type': 'detection',
            'total_samples': total_size,
            'train_samples': train_size,
            'val_samples': val_size,
            'val_loss': float(val_loss),
            'val_mAP': float(val_bin_metrics.get('mAP', 0)),
            'val_AP50': float(val_bin_metrics.get('AP50', 0)),
            'val_AP75': float(val_bin_metrics.get('AP75', 0)),
            'val_froc_score': float(val_bin_metrics.get('froc_score', 0)),
            'val_auc_froc': float(val_bin_metrics.get('auc_froc', 0)),
            'val_precision': float(val_dice),
            'val_recall': float(val_bin_metrics.get('recall', 0)),
            'val_iou': float(val_iou),
            'spatial_metrics': {k: float(v) if not isinstance(v, dict) else v for k, v in val_bin_metrics.items()}
        }]
    else:
        results = [{
            'mode': 'full_training',
            'task_type': 'segmentation',
            'total_samples': total_size,
            'train_samples': train_size,
            'val_samples': val_size,
            'val_loss': float(val_loss),
            'val_dice': float(val_dice),
            'val_iou': float(val_iou),
            'spatial_metrics': {k: float(v) for k, v in val_bin_metrics.items()}
        }]

    save_results(results, output_dir)

    print(f"\n🎉 Full Training Completed!")
    if task_type == 'detection':
        mAP = val_bin_metrics.get('mAP', 0)
        AP50 = val_bin_metrics.get('AP50', 0)
        AP75 = val_bin_metrics.get('AP75', 0)
        froc = val_bin_metrics.get('froc_score', 0)
        print(f"✅ Final: mAP={mAP:.4f}, AP50={AP50:.4f}, AP75={AP75:.4f}")
        print(f"   FROC={froc:.4f}, Precision={val_dice:.4f}, IoU={val_iou:.4f}")
        logger.info(
            f"Full Training: Loss={val_loss:.4f}, mAP={mAP:.4f}, AP50={AP50:.4f}, AP75={AP75:.4f}, FROC={froc:.4f}")
    else:
        print(f"✅ Final: Dice={val_dice:.4f}, IoU={val_iou:.4f}, Worst={val_bin_metrics['worst_bin_dice']:.4f}")
        logger.info(f"Full Training: Loss={val_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}")

    return results


def main():
    """Main Active Learning loop."""
    args = parse_arguments()

    # SIIM dataset: automatically set target_lesion to pneumothorax
    if args.dataset_source == 'siim':
        if args.target_lesion != ['pneumothorax']:
            print(f"ℹ️  SIIM dataset only has pneumothorax. Ignoring target_lesion={args.target_lesion}")
        args.target_lesion = ['pneumothorax']

    # Full training mode (no Active Learning)
    if args.full_training:
        args.mode = 'full_training'  # Override mode for directory naming
        device, output_dir, logger = setup_experiment(args)
        run_full_training(args, device, output_dir, logger)
        return

    device, output_dir, logger = setup_experiment(args)

    # Create datasets
    full_dataset, val_dataset = create_al_datasets(args)
    print(f"📊 Pool: {len(full_dataset)}, Validation: {len(val_dataset)}")

    # Initialize
    pool_indices = list(range(len(full_dataset)))
    selected_indices = []
    areas = divide_image_into_areas(grid_width=args.grid_width, grid_height=args.grid_height)
    results = []
    prev_bin_dices = None
    prev_checkpoint = None

    print(f"🎯 Starting {args.mode} strategy...")

    for round_idx in range(args.round_num):
        print(f"\n🔄 Round {round_idx + 1}/{args.round_num}")
        logger.info(f"Round {round_idx + 1}/{args.round_num}")

        # Calculate uncertainties
        model = create_model(args, device, prev_checkpoint)
        uncertainty_data = calculate_uncertainties(
            args, model, full_dataset, pool_indices, selected_indices, device)

        # Prepare and select samples
        uncertainties = prepare_selection_args(
            args, model, full_dataset, pool_indices, selected_indices,
            uncertainty_data if uncertainty_data is not None else {}, device)
        new_indices = select_samples(args, pool_indices, uncertainties, selected_indices, areas, round_idx)
        selected_indices.extend(new_indices)
        print(f"📊 Selected {len(new_indices)} samples (total: {len(selected_indices)})")

        # Train
        train_dataset = torch.utils.data.Subset(full_dataset, selected_indices)
        model = create_model(args, device)

        # Determine task type
        task_type = 'detection' if (
            (hasattr(args, 'dataset_source') and args.dataset_source == 'vindr' and
             hasattr(args, 'vindr_mask_type') and args.vindr_mask_type == 'detection') or
            (hasattr(args, 'dataset_source') and args.dataset_source == 'chestxdet10' and
             getattr(args, 'chestxdet10_mask_type', 'detection') == 'detection')
        ) else 'segmentation'

        print("🏋️ Training...")
        val_loss, val_dice, val_iou, val_bin_metrics, current_bin_dices, checkpoint = train_model_round(
            model, train_dataset, val_dataset, device,
            args.train_epochs, args.batch_size, round_idx + 1, output_dir,
            args.patience, args.min_delta, prev_bin_dices,
            args.grid_width, args.grid_height, args.uncertainty_type, task_type=task_type)

        # Save results
        if task_type == 'detection':
            results.append({
                'round': round_idx + 1,
                'task_type': 'detection',
                'selected_indices': new_indices,
                'total_selected': len(selected_indices),
                'val_loss': float(val_loss),
                'val_mAP': float(val_bin_metrics.get('mAP', 0)),
                'val_AP50': float(val_bin_metrics.get('AP50', 0)),
                'val_AP75': float(val_bin_metrics.get('AP75', 0)),
                'val_froc_score': float(val_bin_metrics.get('froc_score', 0)),
                'val_auc_froc': float(val_bin_metrics.get('auc_froc', 0)),
                'val_precision': float(val_dice),
                'val_iou': float(val_iou),
                'spatial_metrics': {k: float(v) if not isinstance(v, dict) else v for k, v in val_bin_metrics.items()}
            })
        else:
            results.append({
                'round': round_idx + 1,
                'task_type': 'segmentation',
                'selected_indices': new_indices,
                'total_selected': len(selected_indices),
                'val_loss': float(val_loss),
                'val_dice': float(val_dice),
                'val_iou': float(val_iou),
                'spatial_metrics': {k: float(v) for k, v in val_bin_metrics.items()}
            })

        if task_type == 'detection':
            mAP = val_bin_metrics.get('mAP', 0)
            AP50 = val_bin_metrics.get('AP50', 0)
            froc = val_bin_metrics.get('froc_score', 0)
            print(f"✅ Round {round_idx + 1}: mAP={mAP:.4f}, AP50={AP50:.4f}, FROC={froc:.4f}, Prec={val_dice:.4f}")
            logger.info(
                f"Round {round_idx + 1} completed - Val Loss: {val_loss:.4f}, mAP: {mAP:.4f}, AP50: {AP50:.4f}, FROC: {froc:.4f}")
        else:
            worst_bin = val_bin_metrics.get('worst_bin_dice', 0)
            perf_cov = val_bin_metrics.get('performance_coverage', 0)
            spatial_cons = val_bin_metrics.get('spatial_consistency', 0)
            print(f"✅ Round {round_idx + 1}: Dice={val_dice:.4f}, IoU={val_iou:.4f}, Worst={worst_bin:.4f}")
            logger.info(
                f"Round {round_idx + 1} completed - Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")
            logger.info(
                f"Spatial Metrics - Worst Bin: {worst_bin:.4f}, Perf Coverage: {perf_cov:.4f}, Spatial Consistency: {spatial_cons:.4f}")

        prev_bin_dices = current_bin_dices
        prev_checkpoint = checkpoint
        save_results(results, output_dir)

    # Final
    print(f"\n🎉 Completed!")
    # Handle multiple target lesions in session name
    if isinstance(args.target_lesion, list):
        lesion_str = '_'.join(sorted(args.target_lesion))
    else:
        lesion_str = args.target_lesion
    session_name = f"{lesion_str}_{args.mode}"
    plot_validation_curves(results, output_dir, session_name)
    plot_spatial_metrics_curves(results, output_dir, session_name)
    log_final_results(logger, results, selected_indices, output_dir, args)


if __name__ == "__main__":
    main()
