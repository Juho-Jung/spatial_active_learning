"""
Calcified Nodule Segmentation Training Script.

Clean interface for training calcified nodule segmentation models with Active Learning support.
Follows the same logic as run_al_baselines.py for validation split and Active Learning.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from common.datasets import SegmentationDataset

# Add segmentation directory to path
sys.path.append('/opt/pxi/projects/calcified_nodule/spatial_active_learning/')

# Import SegmentationDataset

# Configure logging
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training configuration management following run_al_baselines.py logic."""

    def __init__(self):
        self.config = {
            # Basic settings
            'finding_name': 'calcifiednodule',
            'batch_size': 16,
            'gpu_id': 0,
            'model_type': 'smp_efficientnet',
            'train_dataset': 'both',
            'use_weighted_sampling': False,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'patience': 10,

            # Active Learning settings (following run_al_baselines.py)
            'use_active_learning': True,
            'num_rounds': 20,
            'num_samples_per_round': 10,  # Samples per round
            'selection_strategy': 'random',
            'uncertainty_type': 'base',
            'selection_params': {
                'grid_width': 3,
                'grid_height': 2,
                'lambda1': 1.0,
                'seed': 42
            },

            # Validation split settings (following run_al_baselines.py)
            'validate_data_mode': 'random_split',  # 'random_split', 'spatial_equal_split', 'spatial_dynamic_split'
            'num_validation_samples': 50,  # Number of validation samples to reserve
            'collection': 'both',  # 'validation_collection', 'train_collection', 'both'
        }

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self.config.update(updates)
        logger.info(f"Configuration updated: {updates}")


class TrainingManager:
    """Main training manager class following run_al_baselines.py logic."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session_dir = None
        self.full_dataset = None
        self.val_dataset = None

    def setup_environment(self) -> None:
        """Setup training environment."""
        from common import set_seeds, suppress_albumentations_warnings

        suppress_albumentations_warnings()
        set_seeds(42)

        # Setup logging for this session
        self._setup_logging()

        self.logger.info("Environment setup completed")

    def _setup_logging(self) -> None:
        """Setup logging configuration for this session."""
        # Create session directory for logging
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = self.config.get_config()
        self.session_dir = f"checkpoints/{config['model_type']}_{timestamp}_{config['num_rounds']}r_{config['num_samples_per_round']}s"
        os.makedirs(self.session_dir, exist_ok=True)

        # Configure logging to save to session directory
        log_file = os.path.join(self.session_dir, 'training.log')

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add new handlers
        self.logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Logging configured. Session directory: {self.session_dir}")

    def print_config_info(self) -> None:
        """Print configuration information."""
        config = self.config.get_config()

        print("🚀 Starting calcified nodule segmentation training...")
        print("📊 Following run_al_baselines.py logic for validation split and Active Learning")
        print("🎯 Target: Calcified nodule segmentation only")
        print("=" * 50)

        # Basic settings
        print(f"📦 Batch size: {config['batch_size']}")
        print(f"🖥️  GPU ID: {config['gpu_id']}")
        print(f"🤖 Model: {config['model_type']}")
        print(f"📈 Collection: {config['collection']}")
        print(f"⚖️  Use weighted sampling: {config['use_weighted_sampling']}")
        print(f"🔄 Epochs: {config['num_epochs']}")
        print(f"📚 Learning rate: {config['learning_rate']}")
        print(f"⏰ Patience: {config['patience']}")

        # Validation split settings
        print(f"📊 Validation mode: {config['validate_data_mode']}")
        print(f"📊 Validation samples: {config['num_validation_samples']}")

        # Active Learning settings
        if config.get('use_active_learning', False):
            print(f"🎯 Active Learning: {config['num_rounds']} rounds, {config['num_samples_per_round']} samples/round")
            print(f"🎯 Strategy: {config['selection_strategy']}, Uncertainty: {config['uncertainty_type']}")
            print(f"🎯 Selection params: {config['selection_params']}")
        else:
            print(f"🎯 Selection strategy: {config['selection_strategy']}")
            if config['selection_strategy'] != 'random':
                print(f"🎯 Selection params: {config['selection_params']}")

        print("=" * 50)

    def _create_datasets(self) -> Tuple[Any, Any]:
        """Create training and validation datasets following run_al_baselines.py logic."""
        config = self.config.get_config()

        if config['validate_data_mode'] == 'random_split':
            return self._create_random_split_datasets()
        elif config['validate_data_mode'] in ['spatial_equal_split', 'spatial_dynamic_split']:
            return self._create_spatial_split_datasets()
        else:
            raise ValueError(f"Unknown validation mode: {config['validate_data_mode']}")

    def _create_random_split_datasets(self) -> Tuple[Any, Any]:
        """Create datasets using random validation split with SegmentationDataset."""
        config = self.config.get_config()

        print(f"📊 Creating random validation split from {config['collection']}...")

        # Get raw documents from SegmentationDataset
        from common.datasets import SegmentationDataset
        temp_dataset = SegmentationDataset(
            finding_name=config['finding_name'],
            train_dataset=config['collection'],
            split='train'
        )
        raw_documents = temp_dataset.documents

        # Filter positive documents (with target lesion)
        positive_docs = []
        for doc in raw_documents:
            objects = doc.get('objects', [])
            has_target_lesion = False
            for obj in objects:
                if obj.get('finding_name') == config['finding_name']:
                    has_target_lesion = True
                    break
            if has_target_lesion:
                positive_docs.append(doc)

        print(f"📊 Total positive documents: {len(positive_docs)}")

        # Shuffle documents
        import numpy as np
        np.random.seed(config['selection_params']['seed'])
        np.random.shuffle(positive_docs)

        # Calculate validation samples
        num_validation_samples = min(config['num_validation_samples'], len(positive_docs))
        print(f"📊 Validation samples: {num_validation_samples}")

        # Split documents
        val_docs = positive_docs[:num_validation_samples]
        train_docs = positive_docs[num_validation_samples:]

        print(f"📊 Final split: {len(val_docs)} validation, {len(train_docs)} training samples")

        # Create SegmentationDataset for training
        full_dataset = SegmentationDataset(
            finding_name=config['finding_name'],
            train_dataset=config['collection'],
            split='train',
            train_data_size=None,
            selection_strategy='random'
        )

        # Create SegmentationDataset for validation
        val_dataset = SegmentationDataset(
            finding_name=config['finding_name'],
            train_dataset='validation_collection',
            split='val'
        )

        return full_dataset, val_dataset

    def _create_spatial_split_datasets(self) -> Tuple[Any, Any]:
        """Create datasets using spatial validation split with SegmentationDataset."""
        config = self.config.get_config()

        print(f"📊 Loading raw documents for spatial validation split from {config['collection']}...")

        # Get raw documents from SegmentationDataset
        from common.datasets import SegmentationDataset
        temp_dataset = SegmentationDataset(
            finding_name=config['finding_name'],
            train_dataset=config['collection'],
            split='train'
        )
        raw_documents = temp_dataset.documents

        print("📊 Creating spatial validation split...")
        from sam_adaptation.dataset.dataset import (
            create_spatial_validation_split, divide_image_into_areas)

        areas = divide_image_into_areas(grid_width=config['selection_params']['grid_width'],
                                        grid_height=config['selection_params']['grid_height'])
        train_docs, val_docs = create_spatial_validation_split(
            raw_documents,
            config['finding_name'],
            num_samples_per_round=config['num_samples_per_round'],
            num_rounds=config['num_rounds'],
            areas=areas,
            seed=config['selection_params']['seed'],
            num_validation_samples=config['num_validation_samples'],
            validation_mode=config['validate_data_mode']
        )

        print("📊 Creating datasets from spatial split...")

        # Create SegmentationDataset for training
        full_dataset = SegmentationDataset(
            finding_name=config['finding_name'],
            train_dataset=config['collection'],
            split='train',
            train_data_size=None,
            selection_strategy='random'
        )

        # Create SegmentationDataset for validation
        val_dataset = SegmentationDataset(
            finding_name=config['finding_name'],
            train_dataset='validation_collection',
            split='val'
        )

        return full_dataset, val_dataset

    def run_training(self) -> str:
        """Run training based on configuration."""
        config = self.config.get_config()

        if config.get('use_active_learning', False):
            return self._run_active_learning(config)
        else:
            return self._run_standard_training(config)

    def _run_active_learning(self, config: Dict[str, Any]) -> str:
        """Run Active Learning training following run_al_baselines.py logic."""
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Subset

        from sam_adaptation.dataset.dataset import divide_image_into_areas
        from sam_adaptation.metrics import (calculate_bin_dice_metrics,
                                            calculate_performance_coverage,
                                            calculate_spatial_consistency)
        from sam_adaptation.models import MCDropoutSAMModel, SAMLesionModel
        from sam_adaptation.sample_selection import get_selection_strategy
        from sam_adaptation.training import train_model_round
        from sam_adaptation.uncertainty import get_uncertainty_method

        self.logger.info("Starting Active Learning training following run_al_baselines.py logic")

        # Create datasets
        self.full_dataset, self.val_dataset = self._create_datasets()

        # Store original validation dataset size for logging
        original_val_size = len(self.val_dataset)

        # Initialize experiment variables
        pool_indices = list(range(len(self.full_dataset)))
        selected_indices = []
        areas = divide_image_into_areas(grid_width=config['selection_params']['grid_width'],
                                        grid_height=config['selection_params']['grid_height'])
        results = []
        prev_bin_dices = None

        print(f"🎯 Starting Active Learning with {config['selection_strategy']} strategy...")
        print(f"📊 Total pool size: {len(pool_indices)}")
        print(f"📊 Validation set size: {len(self.val_dataset)} (original: {original_val_size})")
        print(f"📊 Validation samples limited to: {config['num_validation_samples']}")

        # Setup device
        device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {device}")

        # Active Learning rounds
        for round_idx in range(config['num_rounds']):
            print(f"\n🔄 Round {round_idx + 1}/{config['num_rounds']}")
            self.logger.info(f"Starting Round {round_idx + 1}/{config['num_rounds']}")

            # Calculate uncertainties if needed
            uncertainties = None
            if config['selection_strategy'] in ['uncertainty', 'uncertainty_area', 'adaptive', 'adaptive_improved', 'diversity', 'diversity_uncertainty']:
                model = self._create_model_for_uncertainty(config, device)
                uncertainty_data = self._calculate_uncertainties(
                    config, model, self.full_dataset, pool_indices, selected_indices, device)

                if config['selection_strategy'] in ['adaptive', 'adaptive_improved'] and isinstance(uncertainty_data, dict):
                    # Extract uncertainties and prepare additional data for adaptive selection
                    uncertainties = uncertainty_data['uncertainties']
                    config['probs'] = uncertainty_data['predictions']
                    config['lesionness'] = uncertainty_data.get(
                        'lesionness', uncertainty_data.get('sam_predictions', None))
                    config['bin_id'] = self._create_spatial_bins_for_dataset(self.full_dataset, pool_indices, config)
                    config['lambda1'] = config['selection_params'].get('lambda1', 1.0)
                elif config['selection_strategy'] in ['diversity', 'diversity_uncertainty'] and isinstance(uncertainty_data, dict):
                    # Extract uncertainties and features for diversity selection
                    uncertainties = uncertainty_data['uncertainties']
                    config['features'] = uncertainty_data['features']
                else:
                    uncertainties = uncertainty_data

            # Select samples
            new_indices = self._select_samples(config, pool_indices, uncertainties, selected_indices, areas, round_idx)
            selected_indices.extend(new_indices)

            print(f"📊 Selected {len(new_indices)} new samples (total: {len(selected_indices)})")
            if round_idx == 0:
                print(f"🎲 Using fixed seed 42 for first round to ensure fair comparison")
                self.logger.info(f"Using fixed seed 42 for first round to ensure fair comparison")
            self.logger.info(f"Selected {len(new_indices)} new samples (total: {len(selected_indices)})")

            # Create training dataset and train model
            train_dataset = Subset(self.full_dataset, selected_indices)
            model = self._create_model_for_training(config, device)

            print("🏋️ Training model...")
            val_loss, val_dice, val_bin_metrics, current_bin_dices = train_model_round(
                model, train_dataset, self.val_dataset, device, config['num_epochs'], config['batch_size'],
                round_idx + 1, self.session_dir, config['patience'], 0.001, prev_bin_dices,
                config['selection_params']['grid_width'], config['selection_params']['grid_height']
            )

            # Save round results
            round_result = {
                'round': round_idx + 1,
                'selected_indices': new_indices,
                'total_selected': len(selected_indices),
                'val_loss': float(val_loss),
                'val_dice': float(val_dice),
                'spatial_metrics': {key: float(val) for key, val in val_bin_metrics.items()}
            }
            results.append(round_result)

            # Log round results
            self._log_round_results(round_idx, new_indices, selected_indices, val_loss, val_dice, val_bin_metrics)

            # Update for next round
            prev_bin_dices = current_bin_dices
            self._save_results(results)

        # Final results
        print(f"\n🎉 Active Learning completed!")
        print(f"📁 Results saved to: {self.session_dir}")

        self._log_final_results(results, selected_indices)
        self._print_final_summary(config, results, selected_indices)

        return os.path.join(self.session_dir, f"round_{len(results)}_best_model.pth")

    def _create_model_for_uncertainty(self, config: Dict[str, Any], device: torch.device):
        """Create model for uncertainty calculation."""
        from sam_adaptation.models import MCDropoutSAMModel, SAMLesionModel

        if config['uncertainty_type'] == 'base':
            print("🔄 Creating SAM model for uncertainty calculation...")
            return SAMLesionModel("/opt/pxi/projects/calcified_nodule/spatial_active_learning/sam_vit_b.pth", "vit_b").to(device)
        else:  # mc_dropout
            print("🔄 Creating MC Dropout SAM model for uncertainty calculation...")
            return MCDropoutSAMModel("/opt/pxi/projects/calcified_nodule/spatial_active_learning/sam_vit_b.pth", "vit_b").to(device)

    def _create_model_for_training(self, config: Dict[str, Any], device: torch.device):
        """Create model for training."""
        from sam_adaptation.models import MCDropoutSAMModel, SAMLesionModel

        if config['uncertainty_type'] == 'mc_dropout' and (config['selection_strategy'] == 'uncertainty' or config['selection_strategy'] == 'uncertainty_area'):
            print("🔄 Creating MC Dropout SAM model for training...")
            return MCDropoutSAMModel("/opt/pxi/projects/calcified_nodule/spatial_active_learning/sam_vit_b.pth", "vit_b").to(device)
        else:
            print("🔄 Creating standard SAM model for training...")
            return SAMLesionModel("/opt/pxi/projects/calcified_nodule/spatial_active_learning/sam_vit_b.pth", "vit_b").to(device)

    def _calculate_uncertainties(self, config: Dict[str, Any], model, full_dataset, pool_indices, selected_indices, device):
        """Calculate uncertainties for sample selection."""
        from torch.utils.data import DataLoader, Subset

        from sam_adaptation.uncertainty import get_uncertainty_method

        if config['selection_strategy'] not in ['uncertainty', 'uncertainty_area', 'adaptive', 'diversity_uncertainty']:
            return None

        uncertainty_func = get_uncertainty_method(config['uncertainty_type'])

        if not selected_indices:
            return None  # Will trigger random selection in first round

        # Create dataloader for uncertainty calculation
        temp_dataset = Subset(full_dataset, pool_indices)
        temp_loader = DataLoader(temp_dataset, batch_size=config['batch_size'], shuffle=False)

        # For adaptive mode, we need detailed predictions and uncertainties
        if config['selection_strategy'] == 'adaptive':
            if config['uncertainty_type'] == 'fast_lesionness':
                return uncertainty_func(model, temp_loader, device, return_detailed=True)
            elif config['uncertainty_type'] == 'mc_dropout':
                return uncertainty_func(model, temp_loader, device, 8, return_detailed=True)
            elif config['uncertainty_type'] == 'none':
                return uncertainty_func(model, temp_loader, device, return_detailed=True)
            else:
                return uncertainty_func(model, temp_loader, device, return_detailed=True)
        elif config['selection_strategy'] in ['diversity', 'diversity_uncertainty']:
            # For diversity modes, we need features
            if config['uncertainty_type'] == 'mc_dropout':
                return uncertainty_func(model, temp_loader, device, 8, return_features=True)
            elif config['uncertainty_type'] == 'none':
                return uncertainty_func(model, temp_loader, device, return_features=True)
            else:
                return uncertainty_func(model, temp_loader, device, return_features=True)
        else:
            # For other modes, return simple uncertainties
            if config['uncertainty_type'] == 'mc_dropout':
                return uncertainty_func(model, temp_loader, device, 8)
            elif config['uncertainty_type'] == 'none':
                return uncertainty_func(model, temp_loader, device)
            else:
                return uncertainty_func(model, temp_loader, device)

    def _create_spatial_bins_for_dataset(self, full_dataset, pool_indices, config):
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

        from sam_adaptation.sample_selection import create_spatial_bins
        return create_spatial_bins(height, width, grid_width=config['selection_params']['grid_width'], grid_height=config['selection_params']['grid_height'])

    def _select_samples(self, config: Dict[str, Any], pool_indices, uncertainties, selected_indices, areas, round_idx):
        """Select samples using the configured strategy."""
        from sam_adaptation.sample_selection import get_selection_strategy

        selection_func = get_selection_strategy(config['selection_strategy'])

        # Define seeds for first round to ensure fair comparison
        first_round_seed = 42 if round_idx == 0 else None

        # uncertainties are already calculated for pool_indices only, no need to filter again
        filtered_uncertainties = uncertainties

        # Select samples using the strategy function with correct parameters
        if config['selection_strategy'] == 'random':
            return selection_func(pool_indices, config['num_samples_per_round'], selected_indices, first_round_seed)
        elif config['selection_strategy'] == 'uncertainty':
            return selection_func(pool_indices, filtered_uncertainties, config['num_samples_per_round'], selected_indices, first_round_seed)
        elif config['selection_strategy'] == 'area_random':
            return selection_func(pool_indices, config['num_samples_per_round'], selected_indices, areas, first_round_seed)
        elif config['selection_strategy'] == 'uncertainty_area':
            return selection_func(pool_indices, filtered_uncertainties, config['num_samples_per_round'], selected_indices, areas, first_round_seed)
        elif config['selection_strategy'] == 'diversity':
            features = getattr(config, 'features', None)
            return selection_func(pool_indices, filtered_uncertainties, config['num_samples_per_round'], selected_indices, areas, features, first_round_seed)
        elif config['selection_strategy'] == 'diversity_uncertainty':
            features = getattr(config, 'features', None)
            return selection_func(pool_indices, filtered_uncertainties, config['num_samples_per_round'], selected_indices, areas, features, first_round_seed)
        elif config['selection_strategy'] == 'adaptive':
            # For adaptive selection, we need additional parameters
            probs = getattr(config, 'probs', None)
            lesionness = getattr(config, 'lesionness', None)
            bin_id = getattr(config, 'bin_id', None)
            lambda1 = getattr(config, 'lambda1', 1.0)

            return selection_func(pool_indices, filtered_uncertainties, config['num_samples_per_round'], selected_indices,
                                  probs, lesionness, bin_id, lambda1, first_round_seed)
        elif config['selection_strategy'] == 'adaptive_improved':
            # For improved adaptive selection, we need additional parameters
            probs = getattr(config, 'probs', None)
            lesionness = getattr(config, 'lesionness', None)
            bin_id = getattr(config, 'bin_id', None)
            lambda1 = getattr(config, 'lambda1', 1.0)

            return selection_func(pool_indices, filtered_uncertainties, config['num_samples_per_round'], selected_indices,
                                  probs, lesionness, bin_id, lambda1, first_round_seed)
        else:
            raise ValueError(f"Unknown selection mode: {config['selection_strategy']}")

    def _log_round_results(self, round_idx, new_indices, selected_indices, val_loss, val_dice, val_bin_metrics):
        """Log results for current round."""
        print(f"✅ Round {round_idx + 1} completed - Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        print(
            f"   📊 Spatial Metrics - Worst Bin: {val_bin_metrics['worst_bin_dice']:.4f}, Perf Coverage: {val_bin_metrics['performance_coverage']:.4f}")
        self.logger.info(f"Round {round_idx + 1} completed - Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        self.logger.info(
            f"Spatial Metrics - Worst Bin: {val_bin_metrics['worst_bin_dice']:.4f}, Perf Coverage: {val_bin_metrics['performance_coverage']:.4f}")

    def _save_results(self, results):
        """Save results to JSON file."""
        import json
        with open(os.path.join(self.session_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    def _log_final_results(self, results, selected_indices):
        """Log final experiment results."""
        final_spatial = results[-1]['spatial_metrics']
        self.logger.info("=" * 80)
        self.logger.info("Active Learning Completed!")
        self.logger.info("=" * 80)
        self.logger.info(f"Final validation Loss: {results[-1]['val_loss']:.4f}")
        self.logger.info(f"Final validation Dice: {results[-1]['val_dice']:.4f}")
        self.logger.info(f"Final spatial metrics:")
        self.logger.info(f"  Worst Bin Dice: {final_spatial['worst_bin_dice']:.4f}")
        self.logger.info(f"  P10 Bin Dice: {final_spatial['p10_bin_dice']:.4f}")
        self.logger.info(f"  Bin Std: {final_spatial['bin_std']:.4f}")
        self.logger.info(f"  Min/Max Bin Dice: {final_spatial['min_bin_dice']:.4f}/{final_spatial['max_bin_dice']:.4f}")
        self.logger.info(f"  Spatial Consistency: {final_spatial['spatial_consistency']:.4f}")
        self.logger.info(f"  Performance Coverage: {final_spatial['performance_coverage']:.4f}")
        self.logger.info(f"Total samples used: {len(selected_indices)}")
        self.logger.info(f"Results saved to: {self.session_dir}")

    def _print_final_summary(self, config, results, selected_indices):
        """Print final summary to console."""
        final_spatial = results[-1]['spatial_metrics']
        print(f"\n📊 Final Summary:")
        print(f"   Strategy: {config['selection_strategy']}")
        print(f"   Uncertainty: {config['uncertainty_type']}")
        print(f"   Target lesion: {config['finding_name']}")
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
        print(f"   Results directory: {self.session_dir}")

    def _run_standard_training(self, config: Dict[str, Any]) -> str:
        """Run standard training."""
        from common import train_model

        self.logger.info("Starting standard training")

        best_model_path = train_model(
            finding_name=config['finding_name'],
            batch_size=config['batch_size'],
            gpu_id=config['gpu_id'],
            model_type=config['model_type'],
            train_dataset=config['train_dataset'],
            use_weighted_sampling=config['use_weighted_sampling'],
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            patience=config['patience'],
            selection_strategy=config['selection_strategy'],
            selection_params=config['selection_params']
        )

        self.logger.info(f"Standard training completed. Best model: {best_model_path}")
        return best_model_path


def main() -> None:
    """Main function for calcified nodule segmentation training following run_al_baselines.py logic."""
    try:
        # Initialize configuration and training manager
        config = TrainingConfig()
        trainer = TrainingManager(config)

        # Setup environment
        trainer.setup_environment()

        # Print configuration
        trainer.print_config_info()

        # Run training
        best_model_path = trainer.run_training()

        # Print results
        print("✅ Training completed!")
        print(f"💾 Best model saved to: {best_model_path}")
        print("🔍 Run evaluation scripts to test the model")
        print("📊 Check the generated metrics JSON file for detailed results")
        print("📁 Results directory: {trainer.session_dir}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
