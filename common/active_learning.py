"""
Simplified Active Learning implementation.

This module provides clean, simple Active Learning functionality.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .datasets import SegmentationDataset
from .losses import combo_loss
from .models import create_model
from .sample_selection import get_selection_strategy
from .training import train_model
from .utils import get_device

# Configure logging
logger = logging.getLogger(__name__)


def run_active_learning_simple(finding_name: str = "calcifiednodule",
                               batch_size: int = 16,
                               gpu_id: int = 0,
                               model_type: str = 'smp_efficientnet',
                               train_dataset: str = 'both',
                               use_weighted_sampling: bool = False,
                               num_epochs: int = 100,
                               learning_rate: float = 1e-4,
                               patience: int = 10,
                               num_rounds: int = 5,
                               num_samples_per_round: int = 40,
                               selection_strategy: str = 'uncertainty',
                               uncertainty_type: str = 'base',
                               selection_params: Dict = None) -> str:
    """
    Run Active Learning with multiple rounds of training and sample selection.

    This is a simplified interface that uses the existing common modules.
    """
    config = {
        'finding_name': finding_name,
        'batch_size': batch_size,
        'gpu_id': gpu_id,
        'model_type': model_type,
        'train_dataset': train_dataset,
        'use_weighted_sampling': use_weighted_sampling,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'patience': patience,
        'num_rounds': num_rounds,
        'num_samples_per_round': num_samples_per_round,
        'selection_strategy': selection_strategy,
        'uncertainty_type': uncertainty_type,
        'selection_params': selection_params or {}
    }

    # Create session directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"checkpoints/al_{config['model_type']}_{timestamp}_{config['num_rounds']}r_{config['num_samples_per_round']}s"
    os.makedirs(session_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(session_dir, 'al_training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    logger.info(f"Active Learning session directory: {session_dir}")

    # Load datasets
    logger.info("Loading datasets...")
    full_dataset = SegmentationDataset(
        finding_name=config['finding_name'],
        train_dataset=config['train_dataset'],
        split='train',
        train_data_size=None,
        selection_strategy='random'
    )

    val_dataset = SegmentationDataset(
        finding_name=config['finding_name'],
        train_dataset='validation_collection',
        split='val'
    )

    logger.info(f"Full dataset size: {len(full_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Initialize Active Learning variables
    selected_indices = []
    results = []
    pool_indices = list(range(len(full_dataset)))

    logger.info(f"Active Learning from full dataset: {len(pool_indices)} samples available")

    # Active Learning rounds
    for round_idx in range(config['num_rounds']):
        logger.info(f"Round {round_idx + 1}/{config['num_rounds']}")

        # Select samples (simplified - just use random for now)
        if round_idx == 0:
            # First round: random selection
            np.random.seed(42)
            new_indices = np.random.choice(pool_indices,
                                           min(config['num_samples_per_round'], len(pool_indices)),
                                           replace=False).tolist()
        else:
            # Subsequent rounds: use configured selection strategy
            selection_func = get_selection_strategy(config['selection_strategy'])
            new_indices = selection_func(pool_indices, config['num_samples_per_round'], selected_indices, 42)

        selected_indices.extend(new_indices)
        pool_indices = [i for i in pool_indices if i not in new_indices]

        logger.info(f"Selected {len(new_indices)} new samples (total: {len(selected_indices)})")

        # Train model with selected samples
        train_dataset_subset = Subset(full_dataset, selected_indices)

        best_model_path = train_model(
            finding_name=config['finding_name'],
            train_data_size=None,
            batch_size=config['batch_size'],
            gpu_id=config['gpu_id'],
            model_type=config['model_type'],
            train_dataset=config['train_dataset'],
            use_weighted_sampling=config['use_weighted_sampling'],
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            patience=config['patience'],
            selection_strategy='random',
            selection_params={}
        )

        # Copy best model to session directory
        round_model_path = os.path.join(session_dir, f"round_{round_idx + 1}_best_model.pth")
        torch.save(torch.load(best_model_path, map_location='cpu'), round_model_path)

        # Evaluate model (simplified)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        model = create_model(config['model_type'], get_device(config['gpu_id']))
        model.load_state_dict(torch.load(best_model_path, map_location=get_device(config['gpu_id'])))

        # Calculate validation metrics
        model.eval()
        val_losses = []
        val_dices = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(get_device(config['gpu_id']))
                masks = masks.to(get_device(config['gpu_id']))

                outputs = model(images)
                loss = combo_loss(outputs, masks)
                val_losses.append(loss.item())

                preds = torch.sigmoid(outputs) > 0.5
                dice = (2 * (preds * masks).sum() / (preds.sum() + masks.sum() + 1e-8)).item()
                val_dices.append(dice)

        avg_val_loss = np.mean(val_losses)
        avg_val_dice = np.mean(val_dices)

        # Save round results
        round_result = {
            'round': round_idx + 1,
            'selected_indices': new_indices,
            'total_selected': len(selected_indices),
            'val_loss': float(avg_val_loss),
            'val_dice': float(avg_val_dice),
            'model_path': round_model_path
        }
        results.append(round_result)

        # Save results
        with open(os.path.join(session_dir, 'al_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Round {round_idx + 1} completed - Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

    # Final results
    logger.info("Active Learning completed!")
    logger.info(f"Final samples: {len(selected_indices)}")
    logger.info(f"Final validation Dice: {results[-1]['val_dice']:.4f}")
    logger.info(f"Results saved to: {session_dir}")

    return os.path.join(session_dir, f"round_{len(results)}_best_model.pth")


# Backward compatibility
def run_active_learning_rounds(*args, **kwargs):
    """Backward compatibility wrapper."""
    return run_active_learning_simple(*args, **kwargs)
