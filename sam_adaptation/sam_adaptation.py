#!/usr/bin/env python3
"""
SAM Adaptation for Lesion Segmentation

Simple adaptation of SAM for lesion segmentation task.
Uses SAM's image encoder (frozen) + new decoder for lesion detection.
Supports different lesion types: calcification, nodule, etc.
"""

import argparse
import json
import os
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Import our modular components
from base_utils import set_seed, setup_logger, setup_ddp, cleanup_ddp, create_session_dir
from models import SAMLesionModel
from dataset import SAMLesionDataset
from losses import combo_loss
from metrics import calculate_metrics

# Suppress albumentations warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Set random seeds for reproducibility
set_seed(42)


def train_model(sam_checkpoint_path: str, batch_size: int = 4, gpu_id: int = 0,
                train_collection: str = 'validation_collection', epochs: int = 50,
                data_limit: int = None, target_lesion: str = 'calcification',
                patience: int = 20, min_delta: float = 0.001):
    """Train SAM-based lesion segmentation model with early stopping."""

    # Setup DDP
    is_ddp, rank, world_size, local_rank, device = setup_ddp()

    if is_ddp:
        print(f"🚀 DDP Training - Rank: {rank}/{world_size}, Local Rank: {local_rank}, Device: {device}")
    else:
        print(f"🚀 Single GPU Training - Device: {device}")

    print(f"📊 Train collection: {train_collection}")
    print(f"🎯 Target lesion: {target_lesion}")
    print(f"🎯 Epochs: {epochs}")
    print(f"🛑 Early stopping: patience={patience}, min_delta={min_delta}")

    # Create session directory (only on rank 0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = create_session_dir(target_lesion, timestamp)

    if not is_ddp or rank == 0:
        print(f"📁 Session directory: {session_dir}")

        # Setup logger
        logger = setup_logger(session_dir, rank)
        logger.info("="*80)
        logger.info("SAM Lesion Segmentation Training Started")
        logger.info("="*80)
        logger.info(f"Target lesion: {target_lesion}")
        logger.info(f"Train collection: {train_collection}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Early stopping: patience={patience}, min_delta={min_delta}")
        logger.info(f"Session directory: {session_dir}")
        logger.info(f"Device: {device}")
        logger.info(f"DDP: {is_ddp}, World size: {world_size if is_ddp else 1}")
    else:
        logger = None

    # File paths
    final_metrics_path = os.path.join(session_dir, f"final_metrics.json")
    best_model_path = os.path.join(session_dir, f"best_model.pth")

    # Create datasets
    train_dataset = SAMLesionDataset(split='train', train_collection=train_collection, limit=data_limit, target_lesion=target_lesion)
    val_dataset = SAMLesionDataset(split='val', train_collection=train_collection, limit=data_limit, target_lesion=target_lesion)

    # Create data loaders with DDP support
    if is_ddp:
        # DDP: Use DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        # Single GPU: Use WeightedRandomSampler or regular shuffle
        if train_collection in ['train_collection_with_consensus', 'both']:
            train_weights = torch.tensor(train_dataset.weights, dtype=torch.float)
            train_sampler = WeightedRandomSampler(
                weights=train_weights,
                num_samples=len(train_weights),
                replacement=True
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model
    model = SAMLesionModel(sam_checkpoint_path).to(device)

    # Wrap with DDP if using distributed training
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Count parameters (only on rank 0)
    if not is_ddp or rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Model Parameters:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")
        print(f"   Frozen (SAM): {total_params - trainable_params:,}")

        if logger:
            logger.info("Model Parameters:")
            logger.info(f"  Total: {total_params:,}")
            logger.info(f"  Trainable: {trainable_params:,}")
            logger.info(f"  Frozen (SAM): {total_params - trainable_params:,}")

    # Loss and optimizer
    criterion = combo_loss

    # Get model parameters (handle DDP wrapper)
    if is_ddp:
        model_params = model.module.decoder.parameters()
    else:
        model_params = model.decoder.parameters()

    optimizer = optim.AdamW(model_params, lr=1e-4, weight_decay=1e-4)

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )

    # Training loop with early stopping
    best_val_loss = float('inf')
    training_history = []
    early_stopping_counter = 0
    best_epoch = 0

    if logger:
        logger.info("Starting training loop...")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Steps per epoch: {len(train_loader)}")
        logger.info("-"*80)

    for epoch in range(epochs):
        # Training
        model.train()

        # Set epoch for DistributedSampler
        if is_ddp:
            train_sampler.set_epoch(epoch)

        train_loss = 0
        train_dice = 0
        train_iou = 0
        train_lesion_sens = 0
        train_auroc = 0

        # Only show progress bar on rank 0
        if not is_ddp or rank == 0:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        else:
            pbar = train_loader

        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()

            # Gradient clipping
            if is_ddp:
                torch.nn.utils.clip_grad_norm_(model.module.decoder.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Calculate metrics
            dice, iou, lesion_sens, auroc = calculate_metrics(outputs, masks)

            train_loss += loss.item()
            train_dice += dice
            train_iou += iou
            train_lesion_sens += lesion_sens
            train_auroc += auroc

            # Update progress bar (only on rank 0)
            if not is_ddp or rank == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice:.4f}',
                    'IoU': f'{iou:.4f}',
                    'Sens': f'{lesion_sens:.4f}',
                    'AUROC': f'{auroc:.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.6f}'
                })

        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        val_lesion_sens = 0
        val_auroc = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                dice, iou, lesion_sens, auroc = calculate_metrics(outputs, masks)

                val_loss += loss.item()
                val_dice += dice
                val_iou += iou
                val_lesion_sens += lesion_sens
                val_auroc += auroc

        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_dice = train_dice / len(train_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_train_lesion_sens = train_lesion_sens / len(train_loader)
        avg_val_lesion_sens = val_lesion_sens / len(val_loader)
        avg_train_auroc = train_auroc / len(train_loader)
        avg_val_auroc = val_auroc / len(val_loader)

        # Average metrics across all processes in DDP
        if is_ddp:
            # Create tensors for all_reduce
            metrics_tensor = torch.tensor([
                avg_train_loss, avg_val_loss, avg_train_dice, avg_val_dice,
                avg_train_iou, avg_val_iou, avg_train_lesion_sens, avg_val_lesion_sens,
                avg_train_auroc, avg_val_auroc
            ], device=device)

            torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.SUM)
            metrics_tensor /= world_size

            avg_train_loss, avg_val_loss, avg_train_dice, avg_val_dice, \
            avg_train_iou, avg_val_iou, avg_train_lesion_sens, avg_val_lesion_sens, \
            avg_train_auroc, avg_val_auroc = metrics_tensor.cpu().numpy()

        # Save epoch metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': float(avg_train_loss),
            'val_loss': float(avg_val_loss),
            'train_dice': float(avg_train_dice),
            'val_dice': float(avg_val_dice),
            'train_iou': float(avg_train_iou),
            'val_iou': float(avg_val_iou),
            'train_lesion_sensitivity': float(avg_train_lesion_sens),
            'val_lesion_sensitivity': float(avg_val_lesion_sens),
            'train_auroc': float(avg_train_auroc),
            'val_auroc': float(avg_val_auroc),
            'learning_rate': float(scheduler.get_last_lr()[0])
        }
        training_history.append(epoch_metrics)

        # Early stopping logic (only on rank 0)
        if not is_ddp or rank == 0:
            # Check for improvement (val loss 기준)
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                early_stopping_counter = 0

                # Save model state dict (handle DDP wrapper)
                if is_ddp:
                    torch.save(model.module.state_dict(), best_model_path)
                else:
                    torch.save(model.state_dict(), best_model_path)
                print(f'🏆 New best model saved! Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})')
                if logger:
                    logger.info(f"New best model saved! Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
            else:
                early_stopping_counter += 1
                print(f'⏳ No improvement for {early_stopping_counter} epochs (best: {best_val_loss:.4f} at epoch {best_epoch})')
                if logger:
                    logger.info(f"No improvement for {early_stopping_counter} epochs (best: {best_val_loss:.4f} at epoch {best_epoch})")

                # Check for early stopping
                if early_stopping_counter >= patience:
                    print(f'🛑 Early stopping triggered! No improvement for {patience} epochs.')
                    print(f'🏆 Best validation Loss: {best_val_loss:.4f} at epoch {best_epoch}')
                    if logger:
                        logger.warning(f"Early stopping triggered! No improvement for {patience} epochs.")
                        logger.info(f"Best validation Loss: {best_val_loss:.4f} at epoch {best_epoch}")
                    break

            # Save final metrics
            final_metrics = {
                'model_name': 'SAMLesionModel',
                'target_lesion': target_lesion,
                'sam_checkpoint': sam_checkpoint_path,
                'batch_size': batch_size,
                'gpu_id': gpu_id,
                'train_collection': train_collection,
                'epochs': epochs,
                'best_val_loss': float(best_val_loss),
                'best_epoch': best_epoch,
                'current_epoch': epoch + 1,
                'early_stopping_patience': patience,
                'early_stopping_min_delta': min_delta,
                'early_stopping_triggered': early_stopping_counter >= patience,
                'training_history': training_history,
                'timestamp': timestamp,
                'is_ddp': is_ddp,
                'world_size': world_size if is_ddp else 1
            }

            with open(final_metrics_path, 'w') as f:
                json.dump(final_metrics, f, indent=2, default=str)

            print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            print(f'          Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}')
            print(f'          Train IoU: {avg_train_iou:.4f}, Val IoU: {avg_val_iou:.4f}')
            print(f'          Train Lesion Sens: {avg_train_lesion_sens:.4f}, Val Lesion Sens: {avg_val_lesion_sens:.4f}')
            print(f'          Train AUROC: {avg_train_auroc:.4f}, Val AUROC: {avg_val_auroc:.4f}')
            print(f'          Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            print("=" * 80)

            # Log epoch metrics
            if logger:
                logger.info(f"Epoch {epoch + 1}/{epochs}:")
                logger.info(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                logger.info(f"  Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}")
                logger.info(f"  Train IoU: {avg_train_iou:.4f}, Val IoU: {avg_val_iou:.4f}")
                logger.info(f"  Train Lesion Sens: {avg_train_lesion_sens:.4f}, Val Lesion Sens: {avg_val_lesion_sens:.4f}")
                logger.info(f"  Train AUROC: {avg_train_auroc:.4f}, Val AUROC: {avg_val_auroc:.4f}")
                logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
                logger.info("-" * 80)

    # Final output (only on rank 0)
    if not is_ddp or rank == 0:
        print(f'🎉 Training completed!')
        print(f'📁 Session directory: {session_dir}')
        print(f'🏆 Best validation Loss: {best_val_loss:.4f} at epoch {best_epoch}')
        print(f'💾 Best model saved: {best_model_path}')
        if early_stopping_counter >= patience:
            print(f'🛑 Training stopped early due to no improvement for {patience} epochs')
        else:
            print(f'✅ Training completed all {epochs} epochs')

        # Final logging
        if logger:
            logger.info("="*80)
            logger.info("Training Completed!")
            logger.info("="*80)
            logger.info(f"Best validation Loss: {best_val_loss:.4f} at epoch {best_epoch}")
            logger.info(f"Best model saved: {best_model_path}")
            logger.info(f"Session directory: {session_dir}")
            if early_stopping_counter >= patience:
                logger.warning(f"Training stopped early due to no improvement for {patience} epochs")
            else:
                logger.info(f"Training completed all {epochs} epochs")
            logger.info("="*80)

    # Cleanup DDP
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='SAM Adaptation for Lesion Segmentation')
    parser.add_argument('--sam_checkpoint', default="/opt/pxi/projects/calcified_nodule/spatial_active_learning/sam_vit_b.pth",
                       help='SAM checkpoint path')
    parser.add_argument('--vit_model', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM ViT model size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--train_collection', default='validation_collection',
                       choices=['validation_collection', 'train_collection_with_consensus', 'both'],
                       help='Training data collection')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--data_limit', type=int, default=None, help='Limit number of training samples (for few-shot experiments)')
    parser.add_argument('--target_lesion', default='calcification',
                       choices=['calcification', 'nodule', 'pneumonia'],
                       help='Target lesion type to segment')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (epochs)')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum change to qualify as improvement')

    args = parser.parse_args()

    train_model(
        sam_checkpoint_path=args.sam_checkpoint,
        batch_size=args.batch_size,
        gpu_id=args.gpu_id,
        train_collection=args.train_collection,
        epochs=args.epochs,
        data_limit=args.data_limit,
        target_lesion=args.target_lesion,
        patience=args.patience,
        min_delta=args.min_delta
    )


if __name__ == "__main__":
    main()
