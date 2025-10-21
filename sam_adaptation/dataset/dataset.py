#!/usr/bin/env python3
"""
Dataset classes for SAM adaptation project.
"""

import json
import os
# Add project root to path
import sys
from pathlib import Path

import albumentations as A
import mdb.document as mdb_d
import mdb.load as mdb_c
import numpy as np
import torch
import utils.image_io as image_io
from torch.utils.data import Dataset

sys.path.append('/opt/pxi')


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
        all_docs = list(collection.find({}).sort('_id', 1))  # Sort by _id for consistent ordering

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
        if self.target_lesion == 'calcification' or self.target_lesion == 'calcifiednodule':
            query = {
                'labeled_findings': {'$all': ['calcification']},
                'objects.finding_name': {'$in': ['calcification']}}
        else:
            query = {
                'labeled_findings': {'$all': [self.target_lesion]},
                'objects.finding_name': {'$in': [self.target_lesion]}}

        excluded_sources = ["amcio_b1_368"]
        query['data_source'] = {'$nin': excluded_sources}

        documents = list(collection.find(query).sort('_id', 1))  # Sort by _id for consistent ordering

        # Load converted MDB data for consensus annotations: 1st round
        converted_data_path_first_round = "/opt/pxi/projects/calcified_nodule/relabeling_update/calcifiednodule/converted_mdb_data_calcifiednodule_1st_round.json"
        if not os.path.exists(converted_data_path_first_round):
            print(
                f"   Warning: converted_mdb_data_calcifiednodule_1st_round.json not found at {converted_data_path_first_round}")
            return []
        with open(converted_data_path_first_round, 'r') as f:
            converted_data_first_round = json.load(f)
        print(f"   Loaded {len(converted_data_first_round)} entries from converted_mdb_data_calcifiednodule_1st_round.json")

        # Load converted MDB data for consensus annotations: 2nd round
        converted_data_path_second_round = "/opt/pxi/projects/calcified_nodule/relabeling_update/calcifiednodule/converted_mdb_data_calcifiednodule_2nd_round.json"
        if not os.path.exists(converted_data_path_second_round):
            print(
                f"   Warning: converted_mdb_data_calcifiednodule_2nd_round.json not found at {converted_data_path_second_round}")
            return []
        with open(converted_data_path_second_round, 'r') as f:
            converted_data_second_round = json.load(f)
        print(f"   Loaded {len(converted_data_second_round)} entries from converted_mdb_data_calcifiednodule_2nd_round.json")

        converted_data = {**converted_data_first_round, **converted_data_second_round}

        # Create mapping from path_dicom stem to doc
        doc_mapping = {}
        for doc in documents:
            if 'path_dicom' not in doc:
                continue
            try:
                # Extract stem from path_dicom (e.g., "0064493-0000345" from "pxi-dataset/cxr/private/internal/210124_nipa/dicom/0064493-0000345.dcm")
                path_dicom = doc['path_dicom']
                stem = Path(path_dicom).stem
                doc_mapping[stem] = doc
            except Exception:
                continue

        print(f"   Created mapping for {len(doc_mapping)} documents with valid path_dicom")

        # Filter documents that exist in converted_data and have valid image paths
        filtered_docs = []
        matched_keys = []
        unmatched_keys = []
        no_path_image = 0
        no_valid_image = 0
        no_consensus = 0
        no_calcifiednodule = 0
        excluded_count = 0

        for json_key in converted_data.keys():
            if json_key in doc_mapping:
                doc = doc_mapping[json_key]
                matched_keys.append(json_key)

                # Check if document has valid image path
                if 'path_image' not in doc:
                    no_path_image += 1
                    continue
                try:
                    image_path = mdb_d.get_valid_image_path(doc['path_image'])
                    if image_path.exists():
                        # Check if this document has consensus annotations
                        if 'consensus' in converted_data[json_key] and converted_data[json_key]['consensus']:
                            consensus_annotations = converted_data[json_key]['consensus']

                            # exclude "finding_name": "Excluded"
                            for annotation in consensus_annotations:
                                for obj in annotation.get('objects', []):
                                    if obj.get('finding_name') == 'Excluded':
                                        excluded_count += 1
                                        continue

                            # Extract consensus objects
                            consensus_objects = []
                            for annotation in consensus_annotations:
                                for obj in annotation.get('objects', []):
                                    if obj.get('finding_name') == 'Calcified Nodule':
                                        # Convert to calcifiednodule format
                                        consensus_objects.append({
                                            'finding_name': 'calcifiednodule',
                                            'polygon': obj.get('polygon'),
                                            'confidence': obj.get('confidence'),
                                            'remark': obj.get('remark', '')
                                        })

                            if consensus_objects:
                                # Create a copy of the document with updated objects (positive sample)
                                updated_doc = doc.copy()
                                updated_doc['objects'] = consensus_objects
                                filtered_docs.append(updated_doc)
                            else:
                                # Create a copy of the document with empty objects (negative sample)
                                updated_doc = doc.copy()
                                updated_doc['objects'] = []
                                filtered_docs.append(updated_doc)
                                no_calcifiednodule += 1

                        else:
                            no_consensus += 1
                    else:
                        no_valid_image += 1
                except Exception:
                    no_valid_image += 1
                    continue
            else:
                unmatched_keys.append(json_key)

        print(f"   Matched keys: {len(matched_keys)}")
        print(f"   Unmatched keys: {len(unmatched_keys)}")
        print(f"   Filtering breakdown:")
        print(f"     - No path_image: {no_path_image}")
        print(f"     - No valid image: {no_valid_image}")
        print(f"     - No consensus: {no_consensus}")
        print(f"     - No calcifiednodule: {no_calcifiednodule}")
        print(f"     - Excluded: {excluded_count}")
        if unmatched_keys:
            print(f"   First 10 unmatched keys: {unmatched_keys[:10]}")

        print(f"   Filtered {len(filtered_docs)} documents with consensus annotations (excluded: {excluded_count})")

        np.random.seed(42)
        np.random.shuffle(filtered_docs)

        print(f"📊 Train Collection: {len(filtered_docs)} (after filtering)")
        return filtered_docs

    def _load_sdc_ppm_train_documents(self):
        """Load documents from train collection with consensus annotations."""
        collection = mdb_c.get_collection("sdc_ppm_train-0908", db_names=['cxr_new', 'projects', 'personal'])

        # Build query based on parameters
        query = {
            'is_pos': {'$in': [1]},
            'is_normal': {'$in': [0, 1]},
        }

        documents = list(collection.find(query).sort('_id', 1))  # Sort by _id for consistent ordering

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


def divide_image_into_areas(image_size=(512, 512), grid_width=2, grid_height=3):
    """Divide chest X-ray image into 3x2 grid (6 areas total).
    Areas are indexed as follows:
    - Top-left: 0, Top-right: 1
    - Middle-left: 2, Middle-right: 3
    - Bottom-left: 4, Bottom-right: 5
    """
    h, w = image_size
    area_h, area_w = h // grid_height, w // grid_width

    areas = []
    for i in range(grid_height):
        for j in range(grid_width):
            y_start = i * area_h
            y_end = (i + 1) * area_h if i < grid_height - 1 else h
            x_start = j * area_w
            x_end = (j + 1) * area_w if j < grid_width - 1 else w
            areas.append((y_start, y_end, x_start, x_end, i * grid_width + j))

    return areas


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


def create_spatial_validation_split(documents, target_lesion, num_samples_per_round, num_rounds, areas=None, seed=42, num_validation_samples=None, validation_mode='spatial_equal_split'):
    """
    Create validation split ensuring spatial distribution across areas.

    Strategy:
    1. Training: num_samples_per_round * num_rounds samples (for Active Learning)
    2. Validation: num_validation_samples samples distributed across spatial areas based on mode:
       - spatial_equal_split: Equal distribution across areas (num_validation_samples / num_areas per area)
       - spatial_dynamic_split: Proportional distribution based on actual data distribution
       - If some areas don't have enough samples, take only 50% to preserve training data

    Args:
        documents: List of document dictionaries
        target_lesion: Target lesion type (e.g., 'calcification')
        num_samples_per_round: Number of samples selected per round
        num_rounds: Total number of active learning rounds
        areas: List of area tuples (y_start, y_end, x_start, x_end, area_idx)
        seed: Random seed for reproducibility
        num_validation_samples: Number of validation samples (if None, use remaining samples)
        validation_mode: Validation split mode ('spatial_equal_split' or 'spatial_dynamic_split')

    Returns:
        train_docs, val_docs: Lists of documents for train and validation
    """
    if areas is None:
        areas = divide_image_into_areas()

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
        print(f"   Area {i + 1}: {len(docs)} documents")

    # Select validation samples based on validation mode
    val_docs = []
    area_allocations = []

    if validation_mode == 'spatial_equal_split':
        # Equal distribution across areas
        target_per_area = num_validation_samples // len(areas)
        remaining_samples = num_validation_samples % len(areas)

        print(f"📊 Target samples per area: {target_per_area} (with {remaining_samples} extra)")

        for i, docs in enumerate(doc_area_mapping):
            # Calculate how many samples to take from this area
            samples_from_area = target_per_area
            if i < remaining_samples:  # Distribute extra samples to first few areas
                samples_from_area += 1

            # Special case: if area has very few samples, take only 50% to preserve training data
            min_samples_for_training = max(1, len(docs) // 2)  # Keep at least 50% for training
            if len(docs) - samples_from_area < min_samples_for_training:
                samples_from_area = max(0, len(docs) - min_samples_for_training)
                print(
                    f"⚠️  Area {i + 1}: Limited to {samples_from_area} samples to preserve training data (requested {target_per_area + (1 if i < remaining_samples else 0)})")
            elif len(docs) < samples_from_area:
                samples_from_area = len(docs)
                print(
                    f"⚠️  Area {i + 1}: Only {len(docs)} samples available (requested {target_per_area + (1 if i < remaining_samples else 0)})")

            # Randomly sample from this area with reproducible seed
            if samples_from_area > 0:
                # Use area-specific seed for reproducible results
                area_seed = seed + i
                np.random.seed(area_seed)
                np.random.shuffle(docs)
                selected_docs = docs[:samples_from_area]
                val_docs.extend(selected_docs)
                area_allocations.append(samples_from_area)
            else:
                area_allocations.append(0)

    elif validation_mode == 'spatial_dynamic_split':
        # Proportional distribution based on actual data distribution
        total_docs = sum(len(docs) for docs in doc_area_mapping)
        distribution_ratios = [len(docs) / total_docs for docs in doc_area_mapping]

        print(f"📊 Data distribution ratios: {[f'{ratio:.3f}' for ratio in distribution_ratios]}")

        for i, (docs, ratio) in enumerate(zip(doc_area_mapping, distribution_ratios)):
            # Calculate target samples for this area based on its data ratio
            target_samples = int(num_validation_samples * ratio)

            # Ensure we don't take more than available
            samples_from_area = min(target_samples, len(docs))

            # Special case: if area has very few samples, take only 50% to preserve training data
            min_samples_for_training = max(1, len(docs) // 2)  # Keep at least 50% for training
            if len(docs) - samples_from_area < min_samples_for_training:
                samples_from_area = max(0, len(docs) - min_samples_for_training)
                print(f"⚠️  Area {i + 1}: Limited to {samples_from_area} samples to preserve training data")

            # Randomly sample from this area with reproducible seed
            if samples_from_area > 0:
                # Use area-specific seed for reproducible results
                area_seed = seed + i
                np.random.seed(area_seed)
                np.random.shuffle(docs)
                selected_docs = docs[:samples_from_area]
                val_docs.extend(selected_docs)
                area_allocations.append(samples_from_area)
            else:
                area_allocations.append(0)

    # Handle redistribution for spatial_equal_split only
    if validation_mode == 'spatial_equal_split' and len(val_docs) < num_validation_samples:
        needed_samples = num_validation_samples - len(val_docs)
        print(f"📊 Need {needed_samples} more samples, redistributing based on data distribution...")

        # Calculate distribution ratios
        total_docs = sum(len(docs) for docs in doc_area_mapping)
        distribution_ratios = [len(docs) / total_docs for docs in doc_area_mapping]

        # Allocate additional samples based on ratios
        for i, (docs, ratio) in enumerate(zip(doc_area_mapping, distribution_ratios)):
            additional_samples = int(needed_samples * ratio)
            if additional_samples > 0:
                # Get remaining docs from this area (not already selected)
                remaining_docs = [doc for doc in docs if doc not in val_docs]
                if len(remaining_docs) > 0:
                    # Use area-specific seed for reproducible results
                    area_seed = seed + i + 1000  # Add offset to avoid collision with main selection
                    np.random.seed(area_seed)
                    np.random.shuffle(remaining_docs)
                    selected_docs = remaining_docs[:min(additional_samples, len(remaining_docs))]
                    val_docs.extend(selected_docs)
                    area_allocations[i] += len(selected_docs)

    # Create training dataset (remaining documents)
    train_docs = [doc for doc in positive_docs if doc not in val_docs]

    # Shuffle the results with reproducible seed
    np.random.seed(seed + 9999)  # Use different seed for final shuffling
    np.random.shuffle(val_docs)
    np.random.shuffle(train_docs)

    print(f"📊 Final split: {len(val_docs)} validation, {len(train_docs)} training samples")
    print(
        f"📊 Data utilization: {len(val_docs) + len(train_docs)}/{len(positive_docs)} ({100 * (len(val_docs) + len(train_docs)) / len(positive_docs):.1f}%)")

    # Print final area distribution
    print("📊 Final validation samples per area:")
    for i, count in enumerate(area_allocations):
        print(f"   Area {i + 1}: {count} samples")

    return train_docs, val_docs


def load_raw_documents(train_collection='validation_collection', target_lesion='calcification'):
    """Load raw documents for spatial analysis."""
    # Access the raw documents before any splitting
    if train_collection == 'validation_collection':
        collection = mdb_c.get_collection("validation_internal", db_names=['cxr_new', 'projects', 'public'])
        all_docs = list(collection.find({}).sort('_id', 1))  # Sort by _id for consistent ordering

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
