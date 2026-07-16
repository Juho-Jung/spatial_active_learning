#!/usr/bin/env python3
"""
SIIM-ACR Pneumothorax Segmentation Dataset for Active Learning Framework.

SIIM-ACR Pneumothorax Segmentation Dataset
- ~10,712 CXR images with pneumothorax segmentation masks
- Binary segmentation (pneumothorax vs background)
- Images and masks in PNG format (1024x1024)

Reference: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation
"""

import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SIIMDataset(Dataset):
    """
    SIIM-ACR Pneumothorax Segmentation Dataset.

    Args:
        data_root: Root directory of SIIM dataset
        split: 'train' or 'test' (only train has masks)
        target_size: Target image size (H, W)
        positive_only: If True, only include images with pneumothorax
        transform: Additional albumentations transforms
    """

    def __init__(
        self,
        data_root: str = '/team/team_pxi/pxi-dataset/cxr/public/siim-full/input/input/train',
        split: str = 'train',
        target_size: tuple = (512, 512),
        positive_only: bool = False,
        resolution: int = 1024,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        self.positive_only = positive_only
        self.resolution = resolution

        # Setup paths
        self.image_dir = self.data_root / 'images' / str(resolution) / 'dicom'
        self.mask_dir = self.data_root / 'images' / str(resolution) / 'mask'

        # Verify directories exist
        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask directory not found: {self.mask_dir}")

        # Load image-mask pairs
        self.samples = self._load_samples()

        # Setup transforms
        self._setup_transforms()

        # Calculate sample weights for weighted sampling
        self._calculate_weights()

        print(f"📊 SIIM Dataset ({split}): {len(self.samples)} samples")
        print(f"   - Positive (with pneumothorax): {sum(1 for s in self.samples if s['has_lesion'])}")
        print(f"   - Negative (without): {sum(1 for s in self.samples if not s['has_lesion'])}")

    def _load_samples(self):
        """Load image-mask pairs from directory."""
        samples = []

        # Get all image files
        image_files = sorted(os.listdir(self.image_dir))

        for img_file in image_files:
            if not img_file.endswith('.png'):
                continue

            image_id = img_file.replace('.png', '')
            mask_file = img_file

            # Check if mask exists
            mask_path = self.mask_dir / mask_file
            if not mask_path.exists():
                continue

            # Quick check if mask has any positive pixels (for filtering and stats)
            mask = np.array(Image.open(mask_path))
            has_lesion = mask.max() > 0

            # Skip negative samples if positive_only is True
            if self.positive_only and not has_lesion:
                continue

            samples.append({
                'image_id': image_id,
                'image_path': str(self.image_dir / img_file),
                'mask_path': str(mask_path),
                'has_lesion': has_lesion,
            })

        return samples

    def _setup_transforms(self):
        """Setup albumentations transforms."""
        self.transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.Normalize(mean=[0.485], std=[0.229]),
        ])

        # Data augmentation for training (optional, can be enabled)
        self.train_transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485], std=[0.229]),
        ])

    def _calculate_weights(self):
        """Calculate sample weights for weighted sampling."""
        positive_count = sum(1 for s in self.samples if s['has_lesion'])
        negative_count = len(self.samples) - positive_count

        if positive_count == 0 or negative_count == 0:
            self.sample_weights = [1.0] * len(self.samples)
        else:
            # Weight positive samples more if they are fewer
            positive_weight = len(self.samples) / (2 * positive_count)
            negative_weight = len(self.samples) / (2 * negative_count)

            self.sample_weights = [
                positive_weight if s['has_lesion'] else negative_weight
                for s in self.samples
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample. Returns (image, mask) tuple for compatibility."""
        sample = self.samples[idx]

        # Load image (grayscale -> RGB for model compatibility)
        image = np.array(Image.open(sample['image_path']).convert('L'))
        # Convert to 3-channel (grayscale -> RGB)
        image = np.stack([image] * 3, axis=-1)

        # Load mask
        mask = np.array(Image.open(sample['mask_path']))

        # Ensure mask is binary (0 or 1)
        mask = (mask > 127).astype(np.float32)

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        # Convert to tensor
        # Image: (H, W, 3) -> (3, H, W)
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))

        # Mask: (H, W) -> (1, H, W)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask.astype(np.float32))

        return image, mask

    def get_image_id(self, idx):
        """Get image ID for a given index."""
        return self.samples[idx]['image_id']

    def get_has_lesion(self, idx):
        """Check if sample has lesion."""
        return self.samples[idx]['has_lesion']

    def get_sample_weights(self):
        """Get sample weights for weighted sampling."""
        return self.sample_weights


# =============================================================================
# Dataset Creation Functions
# =============================================================================

def create_siim_al_datasets(args):
    """
    Create SIIM datasets for Active Learning.

    Args:
        args: Arguments containing:
            - siim_root: Root directory of SIIM dataset
            - num_samples: Samples per round
            - round_num: Number of AL rounds
            - num_validation_samples: Number of validation samples
            - seed: Random seed
            - validate_data_mode: Validation split mode
            - siim_positive_only: If True, only use positive samples

    Returns:
        full_dataset, val_dataset: Training pool and validation datasets
    """
    import random

    siim_root = getattr(args, 'siim_root',
                        '/team/team_pxi/pxi-dataset/cxr/public/siim-full/input/input/train')
    positive_only = getattr(args, 'siim_positive_only', False)
    num_validation_samples = getattr(args, 'num_validation_samples', 100)
    seed = getattr(args, 'seed', 42)

    # Create full dataset
    full_dataset = SIIMDataset(
        data_root=siim_root,
        split='train',
        positive_only=positive_only,
    )

    # Create train/val split
    random.seed(seed)
    np.random.seed(seed)

    total_samples = len(full_dataset.samples)
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)

    # Stratified split: maintain ratio of positive/negative samples
    positive_indices = [i for i in all_indices if full_dataset.samples[i]['has_lesion']]
    negative_indices = [i for i in all_indices if not full_dataset.samples[i]['has_lesion']]

    # Calculate validation size for each class
    positive_ratio = len(positive_indices) / total_samples
    val_positive_count = max(1, int(num_validation_samples * positive_ratio))
    val_negative_count = num_validation_samples - val_positive_count

    # Select validation indices
    val_positive = positive_indices[:val_positive_count]
    val_negative = negative_indices[:val_negative_count]
    val_indices = val_positive + val_negative

    # Remaining indices for training pool
    train_indices = [i for i in all_indices if i not in val_indices]

    # Create subset datasets
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]

    # Create new datasets with split samples
    train_dataset = SIIMSubsetDataset(
        samples=train_samples,
        target_size=full_dataset.target_size,
        transform=full_dataset.transform,
    )

    val_dataset = SIIMSubsetDataset(
        samples=val_samples,
        target_size=full_dataset.target_size,
        transform=full_dataset.transform,
    )

    print(f"\n📊 SIIM Dataset Split:")
    print(f"   Training pool: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   - Val positive: {sum(1 for s in val_samples if s['has_lesion'])}")
    print(f"   - Val negative: {sum(1 for s in val_samples if not s['has_lesion'])}")

    return train_dataset, val_dataset


class SIIMSubsetDataset(Dataset):
    """Subset dataset for SIIM with pre-selected samples."""

    def __init__(self, samples, target_size=(512, 512), transform=None):
        self.samples = samples
        self.target_size = target_size
        self.transform = transform or A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(mean=[0.485], std=[0.229]),
        ])

        # Calculate sample weights
        self._calculate_weights()

    def _calculate_weights(self):
        """Calculate sample weights for weighted sampling."""
        positive_count = sum(1 for s in self.samples if s['has_lesion'])
        negative_count = len(self.samples) - positive_count

        if positive_count == 0 or negative_count == 0:
            self.sample_weights = [1.0] * len(self.samples)
        else:
            positive_weight = len(self.samples) / (2 * positive_count)
            negative_weight = len(self.samples) / (2 * negative_count)

            self.sample_weights = [
                positive_weight if s['has_lesion'] else negative_weight
                for s in self.samples
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample. Returns (image, mask) tuple for compatibility."""
        sample = self.samples[idx]

        # Load image (grayscale -> RGB for model compatibility)
        image = np.array(Image.open(sample['image_path']).convert('L'))
        # Convert to 3-channel (grayscale -> RGB)
        image = np.stack([image] * 3, axis=-1)

        # Load mask
        mask = np.array(Image.open(sample['mask_path']))

        # Ensure mask is binary (0 or 1)
        mask = (mask > 127).astype(np.float32)

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        # Convert to tensor
        # Image: (H, W, 3) -> (3, H, W)
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))

        # Mask: (H, W) -> (1, H, W)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask.astype(np.float32))

        return image, mask

    def get_image_id(self, idx):
        """Get image ID for a given index."""
        return self.samples[idx]['image_id']

    def get_has_lesion(self, idx):
        """Check if sample has lesion."""
        return self.samples[idx]['has_lesion']

    def get_sample_weights(self):
        """Get sample weights for weighted sampling."""
        return self.sample_weights
