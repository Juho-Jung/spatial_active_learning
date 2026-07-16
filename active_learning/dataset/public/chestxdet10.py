#!/usr/bin/env python3
"""
ChestX-Det10 Dataset for Active Learning Framework.

ChestX-Det10: A subset of NIH ChestX-14 with instance-level box annotations
- 3,543 images with 10 categories of thoracic abnormalities
- Annotated by 3 board-certified radiologists
- Bounding box annotations

Categories: Atelectasis, Calcification, Consolidation, Effusion, Emphysema,
            Fibrosis, Fracture, Mass, Nodule, Pneumothorax

Reference: https://arxiv.org/abs/2006.10550v3
"""

import json
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# =============================================================================
# ChestX-Det10 Class Names
# =============================================================================
CHESTXDET10_CLASSES = [
    'Atelectasis', 'Calcification', 'Consolidation', 'Effusion', 'Emphysema',
    'Fibrosis', 'Fracture', 'Mass', 'Nodule', 'Pneumothorax'
]

# Mapping from our target lesion names to ChestX-Det10 class names
LESION_MAPPING = {
    'atelectasis': ['Atelectasis'],
    'calcification': ['Calcification'],
    'consolidation': ['Consolidation'],
    'effusion': ['Effusion'],
    'emphysema': ['Emphysema'],
    'fibrosis': ['Fibrosis'],
    'fracture': ['Fracture'],
    'mass': ['Mass'],
    'nodule': ['Nodule'],
    'pneumothorax': ['Pneumothorax'],
}

# Lesion categories for experimental design
# Concentrated lesions: typically single, well-defined location
CONCENTRATED_LESIONS = {
    'calcification': ['Calcification'],
    'pneumothorax': ['Pneumothorax'],
    'consolidation': ['Consolidation'],
    'effusion': ['Effusion'],
    'atelectasis': ['Atelectasis'],
}

# Dispersed lesions: can be multiple, scattered locations
DISPERSED_LESIONS = {
    'nodule': ['Nodule'],
    'mass': ['Mass'],
    'fracture': ['Fracture'],
    'fibrosis': ['Fibrosis'],
    'emphysema': ['Emphysema'],
}


class ChestXDet10Dataset(Dataset):
    """
    ChestX-Det10 Dataset for lesion detection.

    Args:
        data_root: Root directory of ChestX-Det10 dataset
        split: 'train' or 'test'
        target_lesion: Target lesion type (e.g., 'calcification', 'nodule')
        target_size: Target image size (H, W)
        mask_type: 'detection' (return bboxes) or mask types for segmentation
    """

    def __init__(
        self,
        data_root: str = '/team/team_pxi/pxi-dataset/cxr/public/ChestX-Det10-Dataset',
        split: str = 'train',
        target_lesion: str | list = 'nodule',
        target_size: tuple = (512, 512),
        mask_type: str = 'detection',
        include_negative: bool = False,  # Include images without target lesion
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        self.mask_type = mask_type
        self.include_negative = include_negative

        # Handle single or multiple target lesions
        if isinstance(target_lesion, str):
            target_lesion_list = [target_lesion.lower()]
        else:
            target_lesion_list = [lesion.lower() for lesion in target_lesion]

        # Handle special keywords: "concentrated" and "dispersed"
        expanded_lesions = []
        for lesion in target_lesion_list:
            if lesion == 'concentrated':
                expanded_lesions.extend(CONCENTRATED_LESIONS.keys())
            elif lesion == 'dispersed':
                expanded_lesions.extend(DISPERSED_LESIONS.keys())
            elif lesion == 'all':
                expanded_lesions.extend(LESION_MAPPING.keys())
            else:
                expanded_lesions.append(lesion)

        # Remove duplicates while preserving order
        self.target_lesions = list(dict.fromkeys(expanded_lesions))

        # For backward compatibility
        self.target_lesion = self.target_lesions[0]

        # Get ChestX-Det10 class names for all target lesions
        self.target_classes = []
        for lesion in self.target_lesions:
            if lesion not in LESION_MAPPING:
                raise ValueError(f"Unknown target lesion: {lesion}. "
                                 f"Available: {list(LESION_MAPPING.keys())}")
            self.target_classes.extend(LESION_MAPPING[lesion])

        # Remove duplicates while preserving order
        self.target_classes = list(dict.fromkeys(self.target_classes))

        # Setup paths
        if split == 'train':
            self.image_dir = self.data_root / 'train-old'
            self.annotation_file = self.data_root / 'train.json'
        else:
            self.image_dir = self.data_root / 'test_data'
            self.annotation_file = self.data_root / 'test.json'

        # Load annotations
        self.annotations = self._load_annotations()

        # Filter images with target lesion
        self.image_ids = self._filter_images()

        # Setup transforms
        self._setup_transforms()

        print(f"📊 ChestX-Det10 {split.capitalize()} Dataset:")
        if len(self.target_lesions) > 1:
            print(f"   Target lesions: {', '.join(self.target_lesions)}")
        else:
            print(f"   Target lesion: {self.target_lesion}")
        print(f"   Total images: {len(self.image_ids)}")
        print(f"   Mode: {'Detection (bbox)' if mask_type == 'detection' else f'Segmentation ({mask_type} mask)'}")
        self._print_statistics()

    def _load_annotations(self):
        """Load annotations from JSON file."""
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        with open(self.annotation_file, 'r') as f:
            data = json.load(f)

        # Convert to dict format: {file_name: {'syms': [...], 'boxes': [...]}}
        annotations = {}
        for entry in data:
            file_name = entry['file_name']
            annotations[file_name] = {
                'syms': entry.get('syms', []),
                'boxes': entry.get('boxes', [])
            }

        return annotations

    def _filter_images(self):
        """Filter images that have target lesion annotations."""
        valid_image_ids = []

        for file_name, ann in self.annotations.items():
            # Check if image file exists
            image_path = self.image_dir / file_name
            if not image_path.exists():
                continue

            # Check if image has target lesion
            has_target = any(sym in self.target_classes for sym in ann['syms'])

            if has_target or self.include_negative:
                valid_image_ids.append(file_name)

        return valid_image_ids

    def _setup_transforms(self):
        """Setup data augmentation transforms."""
        if self.mask_type == 'detection':
            # Detection mode with augmentation and bbox support
            self.transform = A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                # Data augmentation for detection
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
                                   border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ], bbox_params=A.BboxParams(
                format='pascal_voc',  # [x_min, y_min, x_max, y_max]
                label_fields=['labels'],
                min_visibility=0.3,  # Keep bbox if at least 30% visible after augmentation
                min_area=100  # Remove very small bboxes
            ))
        else:
            self.transform = A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _print_statistics(self):
        """Print dataset statistics."""
        total_annotations = 0
        class_counts = {cls: 0 for cls in self.target_classes}

        for file_name in self.image_ids:
            ann = self.annotations[file_name]
            for sym, box in zip(ann['syms'], ann['boxes']):
                if sym in self.target_classes:
                    total_annotations += 1
                    class_counts[sym] += 1

        print(f"   Total annotations: {total_annotations}")
        print(f"   Avg annotations per image: {total_annotations / max(1, len(self.image_ids)):.2f}")
        if len(self.target_classes) > 1:
            print(f"   Class distribution:")
            for cls, count in class_counts.items():
                print(f"      {cls}: {count}")
        print("=" * 50)

    def _load_image(self, file_name):
        """Load PNG image and convert to RGB numpy array."""
        image_path = self.image_dir / file_name

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image, image.shape[0], image.shape[1]

        except Exception as e:
            print(f"⚠️  Error loading image {file_name}: {e}")
            return None, None, None

    def _get_target_bboxes(self, file_name):
        """Get bounding boxes for target lesion."""
        ann = self.annotations[file_name]

        bboxes = []
        for sym, box in zip(ann['syms'], ann['boxes']):
            if sym in self.target_classes:
                # box format: [x1, y1, x2, y2]
                bboxes.append({
                    'x_min': box[0],
                    'y_min': box[1],
                    'x_max': box[2],
                    'y_max': box[3],
                    'class': sym
                })

        return bboxes

    def _create_mask_from_bboxes(self, bboxes, orig_h, orig_w, mask_type='rectangular'):
        """Create segmentation mask from bounding boxes."""
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for bbox in bboxes:
            x_min = max(0, int(bbox['x_min']))
            y_min = max(0, int(bbox['y_min']))
            x_max = min(orig_w, int(bbox['x_max']))
            y_max = min(orig_h, int(bbox['y_max']))

            if x_max <= x_min or y_max <= y_min:
                continue

            if mask_type == 'rectangular':
                mask[y_min:y_max, x_min:x_max] = 1
            elif mask_type == 'ellipse':
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                width = x_max - x_min
                height = y_max - y_min

                y, x = np.ogrid[:orig_h, :orig_w]
                a = width / 2.0
                b = height / 2.0
                ellipse_mask = ((x - center_x) / (a + 1e-8))**2 + ((y - center_y) / (b + 1e-8))**2 <= 1.0
                mask[ellipse_mask] = 1
            elif mask_type == 'gaussian':
                temp_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                temp_mask[y_min:y_max, x_min:x_max] = 255
                blurred = cv2.GaussianBlur(temp_mask, (21, 21), 0)
                mask[blurred > 127] = 1
            else:
                mask[y_min:y_max, x_min:x_max] = 1

        return mask

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        file_name = self.image_ids[idx]

        # Load image
        image, orig_h, orig_w = self._load_image(file_name)
        if image is None:
            image = np.zeros((*self.target_size, 3), dtype=np.uint8)
            orig_h, orig_w = self.target_size
            bboxes = []
        else:
            bboxes = self._get_target_bboxes(file_name)

        # Apply transforms
        if self.mask_type == 'detection':
            # Prepare bboxes in pascal_voc format [x_min, y_min, x_max, y_max]
            bbox_list = []
            label_list = []
            for bbox in bboxes:
                # Ensure valid bbox coordinates
                x1 = max(0, min(bbox['x_min'], orig_w - 1))
                y1 = max(0, min(bbox['y_min'], orig_h - 1))
                x2 = max(0, min(bbox['x_max'], orig_w))
                y2 = max(0, min(bbox['y_max'], orig_h))
                if x2 > x1 and y2 > y1:
                    bbox_list.append([x1, y1, x2, y2])
                    label_list.append(1)  # All lesions are class 1

            # Apply transforms with bbox augmentation
            if len(bbox_list) > 0:
                transformed = self.transform(
                    image=image,
                    bboxes=bbox_list,
                    labels=label_list
                )
                image = transformed['image']
                bbox_list = transformed['bboxes']
                label_list = transformed['labels']
            else:
                transformed = self.transform(image=image, bboxes=[], labels=[])
                image = transformed['image']
                bbox_list = []
                label_list = []

            # Convert to tensors
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            if len(bbox_list) > 0:
                boxes = torch.tensor(bbox_list, dtype=torch.float32)
                labels = torch.tensor(label_list, dtype=torch.int64)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            target = {
                'boxes': boxes,
                'labels': labels
            }
            return image, target
        else:
            # Segmentation mode
            mask = self._create_mask_from_bboxes(bboxes, orig_h, orig_w, mask_type=self.mask_type)

            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).float().unsqueeze(0)
            return image, mask

    def get_image_id(self, idx):
        """Get image ID for given index."""
        return self.image_ids[idx]


# =============================================================================
# Dataset Creation Functions
# =============================================================================

def create_chestxdet10_al_datasets(args):
    """
    Create ChestX-Det10 datasets for Active Learning.

    Args:
        args: Arguments containing:
            - chestxdet10_root: Root directory of ChestX-Det10 dataset
            - target_lesion: Target lesion type
            - num_samples: Samples per round
            - round_num: Number of AL rounds
            - seed: Random seed

    Returns:
        full_dataset, val_dataset: Training pool and validation datasets
    """
    mask_type = getattr(args, 'chestxdet10_mask_type', 'detection')

    # Validate and adjust target_lesion for ChestX-Det10
    target_lesion = args.target_lesion
    if isinstance(target_lesion, list):
        target_lesion_list = [l.lower() for l in target_lesion]
    else:
        target_lesion_list = [target_lesion.lower()]

    # Valid lesions for ChestX-Det10
    valid_lesions = set(LESION_MAPPING.keys()) | {'concentrated', 'dispersed', 'all'}

    # Check if any target lesion is invalid
    invalid_lesions = [l for l in target_lesion_list if l not in valid_lesions]
    if invalid_lesions:
        print(f"⚠️  Target lesion(s) {invalid_lesions} not available in ChestX-Det10.")
        print(f"   Available: {list(LESION_MAPPING.keys())}")
        print(f"   Defaulting to 'all' (all 10 lesion classes).")
        target_lesion = 'all'

    # Use 1024x1024 for detection (original image size), 512x512 for segmentation
    target_size = (1024, 1024) if mask_type == 'detection' else (512, 512)

    # Create training dataset
    full_dataset = ChestXDet10Dataset(
        data_root=getattr(args, 'chestxdet10_root',
                          '/team/team_pxi/pxi-dataset/cxr/public/ChestX-Det10-Dataset'),
        split='train',
        target_lesion=target_lesion,
        target_size=target_size,
        mask_type=mask_type,
        include_negative=getattr(args, 'include_negative', False),
    )

    # Create validation dataset from test split
    val_dataset = ChestXDet10Dataset(
        data_root=getattr(args, 'chestxdet10_root',
                          '/team/team_pxi/pxi-dataset/cxr/public/ChestX-Det10-Dataset'),
        split='test',
        target_lesion=target_lesion,
        target_size=target_size,
        mask_type=mask_type,
        include_negative=getattr(args, 'include_negative', False),
    )

    print(f"📊 ChestX-Det10 Split: {len(full_dataset)} training, {len(val_dataset)} validation")

    return full_dataset, val_dataset
