#!/usr/bin/env python3
"""
VinDr-CXR Dataset for Active Learning Framework.

VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations
- 15,000 CXR images with 22 local lesion categories
- 17 experienced radiologists
- Bounding box annotations

Reference: https://vindr.ai/datasets/cxr
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

# =============================================================================
# VinDr-CXR Class Names
# =============================================================================
VINDR_CLASSES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA',
    'ILD', 'Infiltration', 'Lung Opacity', 'Lung cavity', 'Lung cyst',
    'Mediastinal shift', 'Nodule/Mass', 'Pleural effusion', 'Pleural thickening',
    'Pneumothorax', 'Pulmonary fibrosis', 'Rib fracture', 'Other lesion',
    'No finding'
]

# Mapping from our target lesion names to VinDr-CXR class names
LESION_MAPPING = {
    'calcification': ['Calcification'],
    'nodule': ['Nodule/Mass'],
    'cardiomegaly': ['Cardiomegaly'],
    'pleural_effusion': ['Pleural effusion'],
    'consolidation': ['Consolidation'],
    'pneumothorax': ['Pneumothorax'],
    'aortic_enlargement': ['Aortic enlargement'],
    'infiltration': ['Infiltration'],
    'lung_opacity': ['Lung Opacity'],
}

# Lesion categories for experimental design
# Concentrated lesions: typically single, well-defined location (e.g., aorta calcification)
CONCENTRATED_LESIONS = {
    'calcification': ['Calcification'],
    'aortic_enlargement': ['Aortic enlargement'],
    'pneumothorax': ['Pneumothorax'],
    'consolidation': ['Consolidation'],
    'pleural_effusion': ['Pleural effusion'],
    'cardiomegaly': ['Cardiomegaly'],
}

# Dispersed lesions: can be multiple, scattered locations (e.g., nodules)
DISPERSED_LESIONS = {
    'nodule': ['Nodule/Mass'],
    'infiltration': ['Infiltration'],
    'lung_opacity': ['Lung Opacity'],
}


class VinDrCXRDataset(Dataset):
    """
    VinDr-CXR Dataset for lesion segmentation.

    Args:
        data_root: Root directory of VinDr-CXR dataset
        split: 'train' or 'test'
        target_lesion: Target lesion type (e.g., 'calcification', 'nodule')
        target_size: Target image size (H, W)
        min_radiologist_agreement: Minimum number of radiologists agreeing on annotation (consensus)
        transform: Additional albumentations transforms
    """

    def __init__(
        self,
        data_root: str = '/team/team_pxi/pxi-dataset/cxr/public/vinbig',
        split: str = 'train',
        target_lesion: str | list = 'calcification',
        target_size: tuple = (512, 512),
        min_radiologist_agreement: int = 1,
        use_consensus_iou: bool = False,
        consensus_iou_threshold: float = 0.3,
        mask_type: str = 'rectangular',
        include_negative: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split = split
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
                # Add all concentrated lesions
                expanded_lesions.extend(CONCENTRATED_LESIONS.keys())
            elif lesion == 'dispersed':
                # Add all dispersed lesions
                expanded_lesions.extend(DISPERSED_LESIONS.keys())
            else:
                expanded_lesions.append(lesion)

        # Remove duplicates while preserving order
        self.target_lesions = list(dict.fromkeys(expanded_lesions))

        # For backward compatibility, keep target_lesion as first lesion
        self.target_lesion = self.target_lesions[0]

        self.target_size = target_size
        self.min_radiologist_agreement = min_radiologist_agreement
        self.use_consensus_iou = use_consensus_iou
        self.consensus_iou_threshold = consensus_iou_threshold
        # mask_type: 'detection' (return bboxes), 'rectangular', 'ellipse', or 'gaussian'
        self.mask_type = mask_type

        if mask_type == 'detection':
            print("⚠️  Detection mode: VinDr-CXR will return bounding boxes instead of masks.")
            print("    Note: Current framework supports segmentation only. Detection support coming soon.")

        # Get VinDr-CXR class names for all target lesions
        self.target_classes = []
        for lesion in self.target_lesions:
            if lesion not in LESION_MAPPING:
                raise ValueError(f"Unknown target lesion: {lesion}. "
                                 f"Available: {list(LESION_MAPPING.keys())}")
            self.target_classes.extend(LESION_MAPPING[lesion])

        # Remove duplicates while preserving order
        self.target_classes = list(dict.fromkeys(self.target_classes))

        # Setup paths
        self.dicom_dir = self.data_root / 'dicom' / split
        self.annotation_file = self.data_root / 'old' / f'annotations_{split}.csv'

        # Load annotations
        self.annotations = self._load_annotations()

        # Filter images with target lesion
        self.image_ids = self._filter_images()

        # Setup transforms
        self._setup_transforms()

        # Calculate weights for weighted sampling
        self._calculate_weights()

        print(f"📊 VinDr-CXR {split.capitalize()} Dataset:")
        if len(self.target_lesions) > 1:
            print(f"   Target lesions: {', '.join(self.target_lesions)} -> {self.target_classes}")
        else:
            print(f"   Target lesion: {self.target_lesion} -> {self.target_classes}")
        print(f"   Total images: {len(self.image_ids)}")
        print(f"   Min radiologist agreement: {self.min_radiologist_agreement}")
        print(f"   Mode: {'Detection (bbox)' if mask_type == 'detection' else f'Segmentation ({mask_type} mask)'}")
        self._print_statistics()

    def _load_annotations(self):
        """Load annotations from CSV file."""
        annotations = defaultdict(list)

        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        with open(self.annotation_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row['image_id']
                class_name = row['class_name']

                # Parse bounding box if exists
                bbox = None
                if row.get('x_min') and row.get('y_min') and row.get('x_max') and row.get('y_max'):
                    try:
                        bbox = {
                            'x_min': float(row['x_min']),
                            'y_min': float(row['y_min']),
                            'x_max': float(row['x_max']),
                            'y_max': float(row['y_max']),
                        }
                    except (ValueError, KeyError):
                        bbox = None

                # rad_id may not exist in test annotations
                rad_id = row.get('rad_id', 'unknown')

                annotations[image_id].append({
                    'rad_id': rad_id,
                    'class_name': class_name,
                    'bbox': bbox
                })

        return annotations

    def _filter_images(self):
        """Filter images that have target lesion annotations with sufficient consensus."""
        valid_image_ids = []
        negative_image_ids = []

        for image_id, anns in self.annotations.items():
            # Check if DICOM file exists
            dicom_path = self.dicom_dir / f"{image_id}.dicom"
            if not dicom_path.exists():
                continue

            # Count radiologists who annotated target lesion
            target_annotations = [
                ann for ann in anns
                if ann['class_name'] in self.target_classes and ann['bbox'] is not None
            ]

            # Check consensus (number of radiologists agreeing)
            if len(target_annotations) >= self.min_radiologist_agreement:
                valid_image_ids.append(image_id)
            elif self.include_negative:
                # Include images without target lesion (negative samples)
                negative_image_ids.append(image_id)

        if self.include_negative:
            print(f"📊 Including ALL negative samples: {len(valid_image_ids)} positive + {len(negative_image_ids)} negative")
            valid_image_ids.extend(negative_image_ids)

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
            # For segmentation, transform both image and mask
            self.transform = A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _calculate_weights(self):
        """Calculate weights for weighted sampling."""
        # All samples have target lesion, so equal weight
        self.weights = [1.0] * len(self.image_ids)

    def _print_statistics(self):
        """Print dataset statistics."""
        total_annotations = 0
        for image_id in self.image_ids:
            anns = self.annotations[image_id]
            target_anns = [
                ann for ann in anns
                if ann['class_name'] in self.target_classes and ann['bbox'] is not None
            ]
            total_annotations += len(target_anns)

        print(f"   Total annotations: {total_annotations}")
        print(f"   Avg annotations per image: {total_annotations / max(1, len(self.image_ids)):.2f}")
        print("=" * 50)

    def _load_dicom_image(self, image_id):
        """Load DICOM image and convert to RGB numpy array."""
        dicom_path = self.dicom_dir / f"{image_id}.dicom"

        try:
            ds = pydicom.dcmread(str(dicom_path))
            image = ds.pixel_array

            # Apply window/level if available
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                window_center = ds.WindowCenter
                window_width = ds.WindowWidth
                if isinstance(window_center, pydicom.multival.MultiValue):
                    window_center = window_center[0]
                if isinstance(window_width, pydicom.multival.MultiValue):
                    window_width = window_width[0]

                min_val = window_center - window_width / 2
                max_val = window_center + window_width / 2
                image = np.clip(image, min_val, max_val)
                image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                # Normalize to 0-255
                image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)

            # Handle PhotometricInterpretation
            if hasattr(ds, 'PhotometricInterpretation'):
                if ds.PhotometricInterpretation == 'MONOCHROME1':
                    image = 255 - image  # Invert

            # Convert to RGB
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)

            return image, ds.Rows, ds.Columns

        except Exception as e:
            print(f"⚠️  Error loading DICOM {image_id}: {e}")
            return None, None, None

    def _create_mask_from_bboxes(self, bboxes, orig_h, orig_w, mask_type='rectangular'):
        """
        Create segmentation mask from bounding boxes.

        Note: VinDr-CXR provides bounding box annotations, not pixel-level segmentation.
        This function converts bboxes to masks for segmentation training.

        Args:
            bboxes: List of bounding box dictionaries with x_min, y_min, x_max, y_max
            orig_h: Original image height
            orig_w: Original image width
            mask_type: Type of mask conversion
                - 'rectangular': Simple rectangular mask (default, fast but less accurate)
                - 'ellipse': Elliptical mask (more realistic for lesions)
                - 'gaussian': Gaussian-weighted mask (smooth edges)

        Returns:
            Binary mask [H, W] with values in {0, 1}
        """
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for bbox in bboxes:
            x_min = max(0, int(bbox['x_min']))
            y_min = max(0, int(bbox['y_min']))
            x_max = min(orig_w, int(bbox['x_max']))
            y_max = min(orig_h, int(bbox['y_max']))

            if x_max <= x_min or y_max <= y_min:
                continue

            if mask_type == 'rectangular':
                # Simple rectangular mask (fast but less accurate)
                mask[y_min:y_max, x_min:x_max] = 1

            elif mask_type == 'ellipse':
                # Elliptical mask (more realistic for round lesions)
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                width = x_max - x_min
                height = y_max - y_min

                # Create meshgrid
                y, x = np.ogrid[:orig_h, :orig_w]

                # Ellipse equation: ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1
                a = width / 2.0
                b = height / 2.0
                ellipse_mask = ((x - center_x) / (a + 1e-8))**2 + ((y - center_y) / (b + 1e-8))**2 <= 1.0
                mask[ellipse_mask] = 1

            elif mask_type == 'gaussian':
                # Gaussian-weighted mask (smooth edges, more realistic)
                import cv2

                # Create rectangular region first
                temp_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                temp_mask[y_min:y_max, x_min:x_max] = 255

                # Apply Gaussian blur and threshold
                blurred = cv2.GaussianBlur(temp_mask, (21, 21), 0)
                mask[blurred > 127] = 1

            else:
                # Fallback to rectangular
                mask[y_min:y_max, x_min:x_max] = 1

        return mask

    def _get_consensus_bboxes(self, image_id):
        """Get consensus bounding boxes for target lesion."""
        anns = self.annotations[image_id]
        target_anns = [
            ann for ann in anns
            if ann['class_name'] in self.target_classes and ann['bbox'] is not None
        ]

        if self.use_consensus_iou:
            # Advanced: IoU-based consensus
            return self._iou_consensus(target_anns)
        else:
            # Simple: all annotations from different radiologists
            return [ann['bbox'] for ann in target_anns]

    def _iou_consensus(self, annotations):
        """Compute IoU-based consensus bounding boxes."""
        if len(annotations) < 2:
            return [ann['bbox'] for ann in annotations]

        # Group similar bboxes by IoU
        bboxes = [ann['bbox'] for ann in annotations]
        consensus_bboxes = []
        used = set()

        for i, bbox1 in enumerate(bboxes):
            if i in used:
                continue

            similar_bboxes = [bbox1]
            used.add(i)

            for j, bbox2 in enumerate(bboxes):
                if j in used:
                    continue
                if self._compute_iou(bbox1, bbox2) > self.consensus_iou_threshold:
                    similar_bboxes.append(bbox2)
                    used.add(j)

            # Average similar bboxes
            if len(similar_bboxes) >= self.min_radiologist_agreement:
                avg_bbox = {
                    'x_min': np.mean([b['x_min'] for b in similar_bboxes]),
                    'y_min': np.mean([b['y_min'] for b in similar_bboxes]),
                    'x_max': np.mean([b['x_max'] for b in similar_bboxes]),
                    'y_max': np.mean([b['y_max'] for b in similar_bboxes]),
                }
                consensus_bboxes.append(avg_bbox)

        return consensus_bboxes if consensus_bboxes else [ann['bbox'] for ann in annotations]

    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1['x_min'], bbox2['x_min'])
        y1 = max(bbox1['y_min'], bbox2['y_min'])
        x2 = min(bbox1['x_max'], bbox2['x_max'])
        y2 = min(bbox1['y_max'], bbox2['y_max'])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1['x_max'] - bbox1['x_min']) * (bbox1['y_max'] - bbox1['y_min'])
        area2 = (bbox2['x_max'] - bbox2['x_min']) * (bbox2['y_max'] - bbox2['y_min'])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-8)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load DICOM image
        image, orig_h, orig_w = self._load_dicom_image(image_id)
        if image is None:
            # Return zeros if image loading fails
            image = np.zeros((*self.target_size, 3), dtype=np.uint8)
            mask = np.zeros(self.target_size, dtype=np.uint8)
            bboxes = []
        else:
            # Get consensus bounding boxes
            bboxes = self._get_consensus_bboxes(image_id)

            # For detection task: return bboxes directly
            # For segmentation task: convert bboxes to mask
            # Note: VinDr-CXR provides bbox annotations, not pixel-level segmentation
            if self.mask_type == 'detection':
                # Create dummy mask for compatibility (will be ignored in detection mode)
                mask = np.zeros(self.target_size, dtype=np.uint8)
            else:
                # Create mask from bboxes for segmentation
                mask = self._create_mask_from_bboxes(bboxes, orig_h, orig_w, mask_type=self.mask_type)

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

            # For detection models, return dict with 'boxes' and 'labels'
            if len(bbox_list) > 0:
                boxes = torch.tensor(bbox_list, dtype=torch.float32)  # [N, 4]
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
            # For segmentation: transform both image and mask
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

            # Convert to tensors
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).float().unsqueeze(0)
            return image, mask

    def get_image_id(self, idx):
        """Get image ID for given index."""
        return self.image_ids[idx]


# =============================================================================
# Dataset Creation Functions
# =============================================================================

def create_vindr_al_datasets(args):
    """
    Create VinDr-CXR datasets for Active Learning.

    Args:
        args: Arguments containing:
            - vindr_root: Root directory of VinDr-CXR dataset
            - target_lesion: Target lesion type
            - num_samples: Samples per round
            - round_num: Number of AL rounds
            - num_validation_samples: Number of validation samples (ignored if use_test_split=True)
            - seed: Random seed
            - validate_data_mode: Validation split mode ('random_split' or 'spatial_equal_split')
            - grid_width, grid_height: Spatial grid dimensions (for spatial split)
            - use_test_split: If True, use dicom/test for validation instead of splitting train

    Returns:
        full_dataset, val_dataset: Training pool and validation datasets
    """
    # Determine mask type: detection mode for VinDr-CXR (bbox annotations)
    # Note: VinDr-CXR provides bbox annotations, so detection mode is more appropriate
    mask_type = getattr(args, 'vindr_mask_type', 'detection')  # Default to detection for VinDr-CXR
    use_test_split = getattr(args, 'use_test_split', True)  # Default: use test split for validation
    include_negative = getattr(args, 'include_negative', False)

    # Use 1024x1024 for detection (preserve more detail), 512x512 for segmentation
    target_size = (1024, 1024) if mask_type == 'detection' else (512, 512)

    # Create training dataset from dicom/train
    full_dataset = VinDrCXRDataset(
        data_root=getattr(args, 'vindr_root', '/team/team_pxi/pxi-dataset/cxr/public/vinbig'),
        split='train',
        target_lesion=args.target_lesion,
        target_size=target_size,
        min_radiologist_agreement=getattr(args, 'min_radiologist_agreement', 1),
        use_consensus_iou=getattr(args, 'use_consensus_iou', False),
        consensus_iou_threshold=getattr(args, 'consensus_iou_threshold', 0.3),
        mask_type=mask_type,
        include_negative=include_negative,
    )

    if use_test_split:
        # Use dicom/test for validation (proper train/test split)
        print("📊 Using dicom/test for validation (proper train/test split)")
        val_dataset = VinDrCXRDataset(
            data_root=getattr(args, 'vindr_root', '/team/team_pxi/pxi-dataset/cxr/public/vinbig'),
            split='test',
            target_lesion=args.target_lesion,
            target_size=target_size,
            min_radiologist_agreement=getattr(args, 'min_radiologist_agreement', 1),
            use_consensus_iou=getattr(args, 'use_consensus_iou', False),
            consensus_iou_threshold=getattr(args, 'consensus_iou_threshold', 0.3),
            mask_type=mask_type,
            include_negative=include_negative,
        )
        train_dataset = full_dataset  # Use all train data for training pool
        print(
            f"📊 VinDr-CXR Split: {len(train_dataset)} training (dicom/train), {len(val_dataset)} validation (dicom/test)")
    else:
        # Split train into train pool and validation (old behavior)
        total_samples = len(full_dataset)
        num_val = getattr(args, 'num_validation_samples', 100)
        num_val = min(num_val, total_samples // 5)  # At most 20% for validation

        validate_data_mode = getattr(args, 'validate_data_mode', 'random_split')

        if validate_data_mode == 'random_split':
            # Random split with seed
            np.random.seed(args.seed)
            indices = np.random.permutation(total_samples)
            val_indices = indices[:num_val].tolist()
            train_indices = indices[num_val:].tolist()
        else:
            # Spatial split (for VinDr-CXR, we use simple random split as spatial info is not available)
            # Note: VinDr-CXR has bbox annotations, but spatial grid split requires polygon annotations
            # For now, fall back to random split
            print("⚠️ Spatial split not fully supported for VinDr-CXR (requires polygon annotations). Using random split.")
            np.random.seed(args.seed)
            indices = np.random.permutation(total_samples)
            val_indices = indices[:num_val].tolist()
            train_indices = indices[num_val:].tolist()

        # Create subset datasets
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        print(f"📊 VinDr-CXR Split: {len(train_indices)} training, {len(val_indices)} validation (from dicom/train)")

    return train_dataset, val_dataset


def load_vindr_documents(data_root='/team/team_pxi/pxi-dataset/cxr/public/vinbig',
                         split='train', target_lesion='calcification'):
    """
    Load VinDr-CXR data as document-like format for compatibility.

    Returns list of dictionaries compatible with existing framework.
    """
    dataset = VinDrCXRDataset(
        data_root=data_root,
        split=split,
        target_lesion=target_lesion,
        min_radiologist_agreement=1
    )

    documents = []
    for idx in range(len(dataset)):
        image_id = dataset.get_image_id(idx)
        anns = dataset.annotations[image_id]

        # Get target lesion bboxes
        target_anns = [
            ann for ann in anns
            if ann['class_name'] in dataset.target_classes and ann['bbox'] is not None
        ]

        doc = {
            '_id': image_id,
            'image_id': image_id,
            'path_dicom': str(dataset.dicom_dir / f"{image_id}.dicom"),
            'objects': [
                {
                    'finding_name': target_lesion,
                    'bbox': ann['bbox'],
                    'rad_id': ann['rad_id']
                }
                for ann in target_anns
            ]
        }
        documents.append(doc)

    return documents
