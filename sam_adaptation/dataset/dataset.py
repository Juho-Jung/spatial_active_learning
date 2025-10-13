#!/usr/bin/env python3
"""
Dataset classes for SAM adaptation project.
"""

import json
import os
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

# Add project root to path
import sys
sys.path.append('/opt/pxi')

import mdb.document as mdb_d
import mdb.load as mdb_c
import utils.image_io as image_io

# Import metrics functions
from metrics import divide_image_into_areas


class SAMLesionDataset(Dataset):
    """Dataset for SAM-based lesion segmentation."""

    def __init__(self, collection_name="validation_internal", target_size=(512, 512),
                 split='train', train_collection='validation_collection', limit=None, target_lesion='calcification'):
        self.collection_name = collection_name
        self.target_size = target_size
        self.split = split
        self.train_collection = train_collection
        self.limit = limit
        self.target_lesion = target_lesion

        # Load documents (reuse logic from original calcification dataset)
        self.documents = self._load_documents()

        # Create train/val split
        self._create_split()

        # Setup transforms
        self._setup_transforms()

        # Calculate weights for weighted sampling
        self._calculate_weights()

        print(f"📊 {split.capitalize()} Dataset Statistics:")
        print(f"   Total documents: {len(self.documents)}")
        self._print_statistics()

    def _load_documents(self):
        """Load documents based on train_collection setting."""
        if self.train_collection == 'validation_collection':
            return self._load_validation_documents()
        elif self.train_collection == 'train_collection':
            return self._load_train_documents()
        elif self.train_collection == 'sdc_ppm_train-0908':
            return self._load_sdc_ppm_train_documents()
        elif self.train_collection == 'both':
            val_docs = self._load_validation_documents()
            train_docs = self._load_train_documents()
            return val_docs + train_docs
        else:
            raise ValueError(f"Unknown train_collection: {self.train_collection}")

    def _load_validation_documents(self):
        """Load documents from validation_internal collection."""
        collection = mdb_c.get_collection(self.collection_name, db_names=['cxr_new', 'projects', 'public'])
        all_docs = list(collection.find({}))

        # Filter out excluded sources
        excluded_sources = ["amcio_b1_368"]
        documents = []
        for doc in all_docs:
            data_source = doc.get('data_source', '')
            if isinstance(data_source, list):
                data_source = data_source[0] if data_source else ''

            if data_source not in excluded_sources:
                documents.append(doc)

        print(f"📊 Validation Collection: {len(all_docs)} -> {len(documents)} (after filtering)")
        return documents

    def _load_train_documents(self):
        """Load documents from train collection with consensus annotations."""
        collection = mdb_c.get_collection("train", db_names=['cxr_new', 'projects', 'public'])

        # Query for documents with target lesion findings
        query = {
            'labeled_findings': {'$all': [self.target_lesion]},
            'objects.finding_name': {'$in': [self.target_lesion]}
        }

        excluded_sources = ["amcio_b1_368"]
        query['data_source'] = {'$nin': excluded_sources}

        documents = list(collection.find(query))

        print(f"📊 Train Collection: {len(documents)} -> {len(documents)} (after filtering)")
        return documents
    
    def _load_sdc_ppm_train_documents(self):
        """Load documents from train collection with consensus annotations."""
        collection = mdb_c.get_collection("sdc_ppm_train-0908", db_names=['cxr_new', 'projects', 'personal'])

        # Build query based on parameters
        query = {
            'is_pos': {'$in': [1]},
            'is_normal': {'$in': [0, 1]},
        }

        documents = list(collection.find(query))

        print(f"📊 Train Collection: {len(documents)}")
        return documents

    def _create_split(self):
        """Create train/val split maintaining positive/negative ratios."""
        positive_docs = []
        negative_docs = []

        for doc in self.documents:
            objects = doc.get('objects', [])
            has_target_lesion = False
            for obj in objects:
                if obj.get('finding_name') == self.target_lesion:
                    has_target_lesion = True
                    break

            if has_target_lesion:
                positive_docs.append(doc)
            else:
                negative_docs.append(doc)

        # Shuffle both lists
        np.random.seed(42)
        np.random.shuffle(positive_docs)
        np.random.shuffle(negative_docs)

        self.documents = positive_docs

        # Apply limit if specified
        if self.limit is not None and len(self.documents) > self.limit:
            print(f"📊 Limiting dataset to {self.limit} samples (from {len(self.documents)})")
            self.documents = self.documents[:self.limit]

    def _setup_transforms(self):
        """Setup data augmentation transforms."""
        self.transform = A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # if self.split == 'train':
        #     self.transform = A.Compose([
        #         A.Resize(self.target_size[0], self.target_size[1]),
        #         A.HorizontalFlip(p=0.5),
        #         A.VerticalFlip(p=0.3),
        #         A.Rotate(limit=10, p=0.5),
        #         A.ElasticTransform(p=0.3),
        #         A.RandomBrightnessContrast(p=0.3),
        #         A.GaussNoise(p=0.2),
        #         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     ])
        # else:
        #     self.transform = A.Compose([
        #         A.Resize(self.target_size[0], self.target_size[1]),
        #         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     ])


    def _calculate_weights(self):
        """Calculate weights for weighted sampling."""
        self.weights = []
        for doc in self.documents:
            objects = doc.get('objects', [])
            has_target_lesion = False
            for obj in objects:
                if obj.get('finding_name') == self.target_lesion:
                    has_target_lesion = True
                    break

            if has_target_lesion:
                self.weights.append(3.0)  # Positive samples get higher weight
            else:
                self.weights.append(1.0)  # Negative samples

    def _print_statistics(self):
        """Print dataset statistics."""
        positive_samples = 0
        negative_samples = 0

        for doc in self.documents:
            objects = doc.get('objects', [])
            has_target_lesion = False
            for obj in objects:
                if obj.get('finding_name') == self.target_lesion:
                    has_target_lesion = True
                    break

            if has_target_lesion:
                positive_samples += 1
            else:
                negative_samples += 1

        total = positive_samples + negative_samples
        print(f"   Positive samples: {positive_samples}")
        print(f"   Negative samples: {negative_samples}")
        print(f"   Positive ratio: {positive_samples / total * 100:.2f}%")
        print("=" * 50)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc = self.documents[idx]

        # Load image
        image_path = mdb_d.get_valid_image_path(doc['path_image'])
        image = image_io.load_as_vis(image_path=image_path, image_type='path_image')

        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            image = np.array(image)
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)

        # Resize image to target size
        import cv2
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))

        # Create mask
        mask = np.zeros(self.target_size, dtype=np.uint8)
        objects = doc.get('objects', [])

        for obj in objects:
            if obj.get('finding_name') == self.target_lesion:
                polygon = obj.get('polygon')
                if polygon:
                    try:
                        from shapely import from_wkt
                        if isinstance(polygon, str) and polygon.startswith('POLYGON'):
                            poly = from_wkt(polygon)
                            coords = list(poly.exterior.coords)
                            coords = np.array(coords)
                            coords[:, 0] = coords[:, 0] / 100 * self.target_size[1]
                            coords[:, 1] = coords[:, 1] / 100 * self.target_size[0]
                            coords = coords.astype(np.int32)

                            cv2.fillPoly(mask, [coords], 1)
                    except Exception as e:
                        print(f"Error creating mask: {e}")
                        pass

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask





def _get_mask_from_doc(doc, target_lesion, target_size=(512, 512)):
    """Helper function to extract mask from document using the same logic as dataset."""
    try:
        # Create mask using the same logic as SAMLesionDataset.__getitem__
        mask = np.zeros(target_size, dtype=np.uint8)
        objects = doc.get('objects', [])

        for obj in objects:
            if obj.get('finding_name') == target_lesion:
                polygon = obj.get('polygon')
                if polygon:
                    try:
                        from shapely import from_wkt
                        if isinstance(polygon, str) and polygon.startswith('POLYGON'):
                            poly = from_wkt(polygon)
                            coords = list(poly.exterior.coords)
                            coords = np.array(coords)
                            coords[:, 0] = coords[:, 0] / 100 * target_size[1]
                            coords[:, 1] = coords[:, 1] / 100 * target_size[0]
                            coords = coords.astype(np.int32)

                            import cv2
                            cv2.fillPoly(mask, [coords], 1)
                    except Exception as e:
                        print(f"⚠️  Error creating mask from polygon: {e}")
                        continue

        return mask.astype(np.float32)

    except Exception as e:
        print(f"⚠️  Error extracting mask from document: {e}")
        return None


def create_spatial_validation_split(documents, target_lesion, num_samples_per_round, num_rounds, areas=None, seed=42, num_validation_samples=None, grid_width=2, grid_height=3, split_mode='spatial_equal_split'):
    """
    Create validation split ensuring spatial distribution across areas.

    Strategy:
    1. Training: num_samples_per_round * num_rounds samples (for Active Learning)
    2. Validation: num_validation_samples samples distributed across 3x3 spatial bins
       - Start with equal distribution (num_validation_samples / 9 per bin)
       - If some bins don't have enough samples, redistribute based on actual data distribution

    Args:
        documents: List of document dictionaries
        target_lesion: Target lesion type (e.g., 'calcification')
        num_samples_per_round: Number of samples selected per round
        num_rounds: Total number of active learning rounds
        areas: List of area tuples (y_start, y_end, x_start, x_end, area_idx)
        seed: Random seed for reproducibility
        num_validation_samples: Number of validation samples (if None, use remaining samples)

    Returns:
        train_docs, val_docs: Lists of documents for train and validation
    """
    if areas is None:
        areas = divide_image_into_areas(grid_width=grid_width, grid_height=grid_height)

    np.random.seed(seed)

    # Filter positive documents (with target lesion)
    positive_docs = []
    for doc in documents:
        objects = doc.get('objects', [])
        has_target_lesion = False
        for obj in objects:
            if obj.get('finding_name') == target_lesion:
                has_target_lesion = True
                break
        if has_target_lesion:
            positive_docs.append(doc)

    print(f"📊 Total positive documents: {len(positive_docs)}")

    # Calculate training samples
    total_training_samples = num_samples_per_round * num_rounds
    print(f"📊 Total samples for training: {total_training_samples} ({num_samples_per_round} × {num_rounds} rounds)")

    # Determine validation samples
    if num_validation_samples is None:
        num_validation_samples = len(positive_docs) - total_training_samples
    print(f"📊 Target validation samples: {num_validation_samples}")

    if num_validation_samples <= 0:
        print("⚠️  Warning: Not enough data for validation! All data will be used for training.")
        return positive_docs, []

    # Analyze spatial distribution of each document
    doc_area_mapping = [[] for _ in range(len(areas))]  # List of docs for each area

    for doc in positive_docs:
        try:
            mask = _get_mask_from_doc(doc, target_lesion)
            if mask is None:
                continue

            # Find which area has the most coverage for this document
            max_coverage = 0
            best_area_idx = 0

            for i, (y_start, y_end, x_start, x_end, area_idx) in enumerate(areas):
                area_mask = mask[y_start:y_end, x_start:x_end]
                coverage = np.sum(area_mask > 0) / (area_mask.shape[0] * area_mask.shape[1])

                if coverage > max_coverage:
                    max_coverage = coverage
                    best_area_idx = i

            # Assign document to the area with highest coverage
            doc_area_mapping[best_area_idx].append(doc)

        except Exception as e:
            print(f"⚠️  Warning: Could not process document {doc.get('_id', 'unknown')}: {e}")
            continue

    # Print area distribution
    print("📊 Document distribution by spatial areas:")
    for i, docs in enumerate(doc_area_mapping):
        print(f"   Area {i+1}: {len(docs)} documents")

    # Select validation samples based on split mode
    if split_mode == 'random_split':
        print(f"📊 Selecting {num_validation_samples} validation samples randomly from {len(positive_docs)} total samples")
        np.random.shuffle(positive_docs)
        val_docs = positive_docs[:num_validation_samples]
        area_allocations = [0] * len(areas)
        for doc in val_docs:
            for i, area_docs in enumerate(doc_area_mapping):
                if doc in area_docs:
                    area_allocations[i] += 1
                    break
    
    elif split_mode == 'spatial_equal_split':
        print(f"📊 Selecting validation samples with equal distribution across {len(areas)} areas")
        val_docs, area_allocations = _select_equal_split(doc_area_mapping, num_validation_samples, areas)
    
    elif split_mode == 'spatial_dynamic_split':
        print(f"📊 Selecting validation samples with proportional distribution across {len(areas)} areas")
        val_docs, area_allocations = _select_dynamic_split(doc_area_mapping, num_validation_samples, areas)
    
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    # Create training dataset (remaining documents)
    train_docs = [doc for doc in positive_docs if doc not in val_docs]

    # Shuffle the results
    np.random.shuffle(val_docs)
    np.random.shuffle(train_docs)

    print(f"📊 Final split: {len(val_docs)} validation, {len(train_docs)} training samples")
    print(f"📊 Data utilization: {len(val_docs) + len(train_docs)}/{len(positive_docs)} ({100*(len(val_docs) + len(train_docs))/len(positive_docs):.1f}%)")

    # Print final area distribution
    print("📊 Final validation samples per area:")
    for i, count in enumerate(area_allocations):
        total_in_area = len(doc_area_mapping[i])
        if total_in_area > 0:
            percentage = (count / total_in_area) * 100
            print(f"   Area {i+1}: {count} samples ({percentage:.1f}% of {total_in_area} total)")
        else:
            print(f"   Area {i+1}: {count} samples")

    return train_docs, val_docs


def _select_equal_split(doc_area_mapping, num_validation_samples, areas):
    """Select validation samples with equal distribution across areas."""
    target_per_area = num_validation_samples // len(areas)
    remaining_samples = num_validation_samples % len(areas)
    
    print(f"📊 Target samples per area: {target_per_area} (with {remaining_samples} extra)")
    
    val_docs = []
    area_allocations = []
    
    for i, docs in enumerate(doc_area_mapping):
        samples_from_area = target_per_area
        if i < remaining_samples:
            samples_from_area += 1
        
        # Apply 50% rule for small areas (when area has fewer samples than target_per_area)
        if len(docs) < samples_from_area:
            if len(docs) <= target_per_area:  # Small area: use only 50% for validation
                samples_from_area = max(1, len(docs) // 2)
                print(f"⚠️  Area {i+1}: Only {len(docs)} samples available (target: {target_per_area + (1 if i < remaining_samples else 0)}) - using 50% rule: {samples_from_area} samples")
            else:
                samples_from_area = len(docs)
                print(f"⚠️  Area {i+1}: Only {len(docs)} samples available (target: {target_per_area + (1 if i < remaining_samples else 0)}) - using all available")
        
        if samples_from_area > 0:
            np.random.shuffle(docs)
            selected_docs = docs[:samples_from_area]
            val_docs.extend(selected_docs)
            area_allocations.append(samples_from_area)
        else:
            area_allocations.append(0)
    
    # Redistribute remaining samples if needed
    if len(val_docs) < num_validation_samples:
        needed_samples = num_validation_samples - len(val_docs)
        print(f"📊 Need {needed_samples} more samples, redistributing...")
        
        total_docs = sum(len(docs) for docs in doc_area_mapping)
        distribution_ratios = [len(docs) / total_docs for docs in doc_area_mapping]
        
        for i, (docs, ratio) in enumerate(zip(doc_area_mapping, distribution_ratios)):
            additional_samples = int(needed_samples * ratio)
            if additional_samples > 0:
                remaining_docs = [doc for doc in docs if doc not in val_docs]
                if len(remaining_docs) > 0:
                    # For small areas, limit additional samples to maintain 50% rule
                    if len(docs) <= target_per_area:
                        max_additional = max(0, len(docs) // 2 - area_allocations[i])
                        additional_samples = min(additional_samples, max_additional)
                    
                    if additional_samples > 0:
                        np.random.shuffle(remaining_docs)
                        selected_docs = remaining_docs[:min(additional_samples, len(remaining_docs))]
                        val_docs.extend(selected_docs)
                        area_allocations[i] += len(selected_docs)
    
    return val_docs, area_allocations


def _select_dynamic_split(doc_area_mapping, num_validation_samples, areas):
    """Select validation samples with proportional distribution across areas."""
    # Calculate proportional distribution
    total_docs = sum(len(docs) for docs in doc_area_mapping)
    distribution_ratios = [len(docs) / total_docs for docs in doc_area_mapping]
    
    print(f"📊 Proportional distribution ratios:")
    for i, ratio in enumerate(distribution_ratios):
        print(f"   Area {i+1}: {ratio:.3f} ({len(doc_area_mapping[i])} samples)")
    
    val_docs = []
    area_allocations = []
    
    # Calculate target samples per area
    target_samples_per_area = [int(num_validation_samples * ratio) for ratio in distribution_ratios]
    remaining_samples = num_validation_samples - sum(target_samples_per_area)
    
    # Distribute remaining samples to areas with highest ratios
    if remaining_samples > 0:
        sorted_areas = sorted(enumerate(distribution_ratios), key=lambda x: x[1], reverse=True)
        for i in range(remaining_samples):
            area_idx = sorted_areas[i % len(sorted_areas)][0]
            target_samples_per_area[area_idx] += 1
    
    print(f"📊 Target samples per area:")
    for i, target in enumerate(target_samples_per_area):
        print(f"   Area {i+1}: {target} samples")
    
    # Select samples from each area
    for i, (docs, target) in enumerate(zip(doc_area_mapping, target_samples_per_area)):
        # Apply 50% rule for small areas
        if len(docs) < target:
            if len(docs) <= target:  # Small area: use only 50% for validation
                samples_from_area = max(1, len(docs) // 2)
                print(f"⚠️  Area {i+1}: Only {len(docs)} samples available (requested {target}) - using 50% rule: {samples_from_area} samples")
            else:
                samples_from_area = len(docs)
                print(f"⚠️  Area {i+1}: Only {len(docs)} samples available (requested {target}) - using all available")
        else:
            samples_from_area = target
        
        if samples_from_area > 0:
            np.random.shuffle(docs)
            selected_docs = docs[:samples_from_area]
            val_docs.extend(selected_docs)
            area_allocations.append(samples_from_area)
        else:
            area_allocations.append(0)
    
    # Redistribute remaining samples if needed
    if len(val_docs) < num_validation_samples:
        needed_samples = num_validation_samples - len(val_docs)
        print(f"📊 Need {needed_samples} more samples, redistributing...")
        
        for i, (docs, ratio) in enumerate(zip(doc_area_mapping, distribution_ratios)):
            additional_samples = int(needed_samples * ratio)
            if additional_samples > 0:
                remaining_docs = [doc for doc in docs if doc not in val_docs]
                if len(remaining_docs) > 0:
                    # For small areas, limit additional samples to maintain 50% rule
                    # Use the original target for this area
                    original_target = target_samples_per_area[i]
                    if len(docs) <= original_target:
                        max_additional = max(0, len(docs) // 2 - area_allocations[i])
                        additional_samples = min(additional_samples, max_additional)
                    
                    if additional_samples > 0:
                        np.random.shuffle(remaining_docs)
                        selected_docs = remaining_docs[:min(additional_samples, len(remaining_docs))]
                        val_docs.extend(selected_docs)
                        area_allocations[i] += len(selected_docs)
    
    return val_docs, area_allocations


def load_raw_documents(train_collection='validation_collection', target_lesion='calcification'):
    """Load raw documents for spatial analysis."""
    # Access the raw documents before any splitting
    if train_collection == 'validation_collection':
        collection = mdb_c.get_collection("validation_internal", db_names=['cxr_new', 'projects', 'public'])
        all_docs = list(collection.find({}))

        # Filter out excluded sources
        excluded_sources = ["amcio_b1_368"]
        documents = []
        for doc in all_docs:
            data_source = doc.get('data_source', '')
            if isinstance(data_source, list):
                data_source = data_source[0] if data_source else ''

            if data_source not in excluded_sources:
                documents.append(doc)

        print(f"📊 Loaded {len(documents)} raw documents from validation_internal")
        return documents
    else:
        # For other collections, create a temporary dataset to access documents
        temp_dataset = SAMLesionDataset(
            split='train',  # This doesn't matter for raw document loading
            train_collection=train_collection,
            target_lesion=target_lesion
        )
        return temp_dataset._load_documents()


class SpatialSplitDataset(SAMLesionDataset):
    """Custom dataset class that uses pre-split documents."""

    def __init__(self, documents, target_lesion, target_size=(512, 512)):
        # Set required attributes (normally set by parent constructor)
        self.collection_name = "pre_split"  # Dummy value since we're not loading from collection
        self.target_size = target_size
        self.split = 'train'  # Default split
        self.train_collection = 'pre_split'  # Dummy value
        self.limit = None
        self.target_lesion = target_lesion

        # Use pre-loaded documents
        self.documents = documents

        # Skip document loading and splitting since we have pre-split documents
        # Setup transforms and weights
        self._setup_transforms()
        self._calculate_weights()

        # Print statistics
        print(f"📊 Pre-split Dataset Statistics:")
        print(f"   Total documents: {len(self.documents)}")
        self._print_statistics()

    def _load_documents(self):
        return self.documents  # Use pre-loaded documents

    def _create_split(self):
        pass  # Skip splitting, already done
