"""
Common training functions for segmentation tasks.

This module provides a clean, well-organized training interface with proper logging.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .datasets import SegmentationDataset
from .losses import combo_loss
from .metrics import calculate_metrics
from .models import create_model
from .utils import get_device

# Configure logging
logger = logging.getLogger(__name__)


class TrainingSession:
    """Manages a training session with proper logging and metrics tracking."""

    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.device = get_device(config['gpu_id'])
        self.logger = logging.getLogger(__name__)

        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = f"checkpoints/{config['model_type']}_{timestamp}_{config.get('train_data_size', 'full')}"
        os.makedirs(self.session_dir, exist_ok=True)

        # Setup logging for this session
        self._setup_logging()

        # Initialize paths
        self.best_dice_path = os.path.join(self.session_dir, "best_dice_model.pth")
        self.best_dice_json = os.path.join(self.session_dir, "best_dice_metrics.json")
        self.final_model_path = os.path.join(self.session_dir, "final_model.pth")
        self.metrics_path = os.path.join(self.session_dir, "training_metrics.json")
        self.best_loss_path = os.path.join(self.session_dir, "best_loss_model.pth")
        self.best_loss_json = os.path.join(self.session_dir, "best_loss_metrics.json")
        self.best_iou_path = os.path.join(self.session_dir, "best_iou_model.pth")
        self.best_iou_json = os.path.join(self.session_dir, "best_iou_metrics.json")

        self.logger.info(f"Training session directory: {self.session_dir}")

    def _setup_logging(self) -> None:
        """Setup logging configuration for this session."""
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

    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        self.logger.info("Creating data loaders...")

        # Training dataset
        train_dataset = SegmentationDataset(
            finding_name=self.config['finding_name'],
            train_dataset=self.config['train_dataset'],
            split='train',
            train_data_size=self.config.get('train_data_size'),
            selection_strategy=self.config.get('selection_strategy', 'random'),
            selection_params=self.config.get('selection_params', {})
        )

        # Validation dataset
        val_dataset = SegmentationDataset(
            finding_name=self.config['finding_name'],
            train_dataset='validation_collection',
            split='val'
        )

        # Create data loaders
        if self.config.get('use_weighted_sampling', False):
            weights = train_dataset.weights
            sampler = WeightedRandomSampler(weights, len(weights))
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)

        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")

        return train_loader, val_loader

    def create_model_and_optimizer(self):
        """Create model and optimizer."""
        self.logger.info(f"Creating model: {self.config['model_type']}")

        model = create_model(self.config['model_type'], self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])

        return model, optimizer

    def train_epoch(self, model, optimizer, train_loader, epoch):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combo_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                dice, iou, _, _ = calculate_metrics(torch.sigmoid(outputs), masks)
                total_loss += loss.item()
                total_dice += dice
                total_iou += iou
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice:.4f}',
                    'IoU': f'{iou:.4f}'
                })

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches

        return avg_loss, avg_dice, avg_iou

    def validate_epoch(self, model, val_loader):
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = model(images)
                loss = combo_loss(outputs, masks)

                dice, iou, _, _ = calculate_metrics(torch.sigmoid(outputs), masks)
                total_loss += loss.item()
                total_dice += dice
                total_iou += iou
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches

        return avg_loss, avg_dice, avg_iou

    def save_model(self, model, metrics, path, json_path):
        """Save model and metrics."""
        torch.save(model.state_dict(), path)
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def train(self) -> str:
        """Run the complete training process."""
        self.logger.info("Starting training...")
        self.logger.info(f"Model: {self.config['model_type']}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Batch size: {self.config['batch_size']}")
        self.logger.info(f"Epochs: {self.config['num_epochs']}")
        self.logger.info(f"Learning rate: {self.config['learning_rate']}")

        # Create data loaders
        train_loader, val_loader = self.create_data_loaders()

        # Create model and optimizer
        model, optimizer = self.create_model_and_optimizer()

        # Training variables
        best_dice = 0.0
        best_loss = float('inf')
        best_iou = 0.0
        patience_counter = 0
        training_metrics = []

        self.logger.info("Starting training loop...")

        for epoch in range(self.config['num_epochs']):
            # Train epoch
            train_loss, train_dice, train_iou = self.train_epoch(model, optimizer, train_loader, epoch)

            # Validate epoch
            val_loss, val_dice, val_iou = self.validate_epoch(model, val_loader)

            # Log metrics
            self.logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']} - "
                             f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f} - "
                             f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")

            # Save metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_dice': train_dice,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou
            }
            training_metrics.append(epoch_metrics)

            # Save best models
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_model(model, epoch_metrics, self.best_dice_path, self.best_dice_json)
                self.logger.info(f"New best Dice: {best_dice:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model(model, epoch_metrics, self.best_loss_path, self.best_loss_json)
                self.logger.info(f"New best Loss: {best_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if val_iou > best_iou:
                best_iou = val_iou
                self.save_model(model, epoch_metrics, self.best_iou_path, self.best_iou_json)
                self.logger.info(f"New best IoU: {best_iou:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config['patience']:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Save final model and metrics
        torch.save(model.state_dict(), self.final_model_path)
        with open(self.metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)

        self.logger.info("Training completed!")
        self.logger.info(f"Best Dice: {best_dice:.4f}")
        self.logger.info(f"Best Loss: {best_loss:.4f}")
        self.logger.info(f"Best IoU: {best_iou:.4f}")
        self.logger.info(f"Results saved to: {self.session_dir}")

        return self.best_dice_path


def train_model(finding_name: str = "calcifiednodule",
                train_data_size: int = None,
                batch_size: int = 8,
                gpu_id: int = 0,
                model_type: str = 'basic',
                train_dataset: str = 'validation_collection',
                use_weighted_sampling: bool = True,
                num_epochs: int = 100,
                learning_rate: float = 1e-4,
                patience: int = 10,
                selection_strategy: str = 'random',
                selection_params: Dict = None) -> str:
    """
    Train a segmentation model with unified interface.

    This is a simplified interface for backward compatibility.
    """
    config = {
        'finding_name': finding_name,
        'train_data_size': train_data_size,
        'batch_size': batch_size,
        'gpu_id': gpu_id,
        'model_type': model_type,
        'train_dataset': train_dataset,
        'use_weighted_sampling': use_weighted_sampling,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'patience': patience,
        'selection_strategy': selection_strategy,
        'selection_params': selection_params or {}
    }

    session = TrainingSession(config)
    return session.train()
