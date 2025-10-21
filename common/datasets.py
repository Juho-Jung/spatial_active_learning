"""
Common dataset classes for segmentation tasks.

This module provides clean, well-organized dataset classes with proper logging.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import mdb.document as mdb_d
import mdb.load as mdb_c
import numpy as np
import torch
import utils.image_io as image_io
from torch.utils.data import Dataset

from .sample_selection import select_samples

# Add project root to path
sys.path.append('/opt/pxi')

# Configure logging
logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """Generic dataset for segmentation tasks."""

    def __init__(self,
                 finding_name: str = "calcification",
                 collection_name: str = "validation_internal",
                 target_size: Tuple[int, int] = (512, 512),
                 split: str = 'train',
                 train_dataset: str = 'validation_collection',
                 train_data_size: int = None,
                 selection_strategy: str = 'random',
                 selection_params: Dict = None,
                 num_rounds: int = None,
                 num_samples_per_round: int = None) -> None:
        """
        Initialize the dataset.

        Args:
            finding_name (str): The finding name to filter for ('calcification' or 'calcifiednodule')
            collection_name (str): Collection name to load from
            target_size (tuple): Target image size
            split (str): 'train' or 'val'
            train_dataset (str): Training dataset type
            train_data_size (int): Number of training samples to use
            selection_strategy (str): Sample selection strategy ('random', 'uncertainty', 'area_random', 'uncertainty_area', 'adaptive', 'adaptive_improved', 'diversity', 'diversity_uncertainty')
            selection_params (dict): Parameters for selection strategy
        """
        self.finding_name = finding_name
        self.collection_name = collection_name
        self.target_size = target_size
        self.split = split
        self.train_dataset = train_dataset
        self.train_data_size = train_data_size
        self.selection_strategy = selection_strategy
        self.selection_params = selection_params or {}
        self.num_rounds = num_rounds
        self.num_samples_per_round = num_samples_per_round

        # Load documents based on train_dataset setting
        if self.train_dataset == 'validation_collection':
            self.documents = self._load_validation_collection()
        elif self.train_dataset == 'validation_internal_subgroup':
            self.documents = self._load_validation_internal_subgroup()
        elif self.train_dataset == 'train_collection_with_consensus':
            self.documents = self._load_train_collection_with_consensus()
        elif self.train_dataset == 'train_collection_with_pseudo_label':
            self.documents = self._load_train_collection_with_pseudo_label()
        elif self.train_dataset == 'both':
            self.documents = self._load_both_collections()
        else:
            raise ValueError(f"Unknown train_dataset setting: {self.train_dataset}. "
                             f"Choose from: 'validation_collection', 'train_collection', 'both'")

        # Print dataset statistics
        self._print_dataset_statistics()

        # Filter for specific finding documents only (for training)
        self.finding_docs = [doc for doc in self.documents
                             if any(obj.get('finding_name') == self.finding_name for obj in doc.get('objects', []))]

        print(f"   Documents with {self.finding_name} (for training): {len(self.finding_docs)}")

        # Stratified train/val split to maintain similar positive/negative ratios
        self.positive_docs, self.negative_docs = self._split_positive_negative()

        if self.train_dataset in ['train_collection_with_consensus', 'train_collection_with_pseudo_label', 'both']:
            if self.split == 'train':
                # For Active Learning: use only positive docs (like sam_adaptation)
                # Don't limit here - let Active Learning select samples dynamically
                self.documents = self.positive_docs
                print(f"   Active Learning setup - Using {len(self.documents)} positive samples for selection")
            else:
                # Validation: use all available docs
                self.documents = self.positive_docs + self.negative_docs
        else:
            # For validation collections, use all documents regardless of split
            self.documents = self.positive_docs + self.negative_docs

        # Apply selection strategy if specified and we have enough samples
        if self.split == 'train' and self.selection_strategy != 'random' and len(self.documents) > 0:
            self.documents = self._apply_selection_strategy()

        # Shuffle the final documents
        np.random.shuffle(self.documents)

        # Print split statistics
        self._print_split_statistics()

        # Setup transforms
        self.transform = self._setup_transforms()

        # Calculate weights for weighted sampling
        self.weights = self._calculate_weights()

        print(f"   Weighted sampling ratio - Positive weight: 3.0, Negative weight: 1.0")

    def _load_validation_collection(self) -> List[Dict]:
        """Load only from validation_internal collection."""
        collection = mdb_c.get_collection(self.collection_name, db_names=['cxr_new', 'projects', 'public'])
        all_docs = list(collection.find({}))

        # Filter out excluded sources
        excluded_sources = ["amcio_b1_368"]
        documents = [doc for doc in all_docs
                     if (lambda ds: ds[0] if isinstance(ds, list) and ds else ds)(doc.get('data_source', '')) not in excluded_sources]

        logger.info(f"Dataset Statistics (validation_internal only):")
        logger.info(f"   Total documents: {len(all_docs)}")
        logger.info(f"   After excluding {excluded_sources}: {len(documents)}")

        return documents

    def _load_validation_internal_subgroup(self):
        """Load only from validation_internal collection with specific subgroup."""
        collection = mdb_c.get_collection(self.collection_name, db_names=['cxr_new', 'projects', 'public'])
        all_docs = list(collection.find({}))

        # Filter out excluded sources
        excluded_sources = ["amcio_b1_368"]
        documents = []
        for doc in all_docs:
            data_source = doc.get('data_source', '')
            inhouse_reviews = doc.get('inhouse_reviews', [])
            for review in inhouse_reviews:
                # Use different review criteria based on finding_name
                if self.finding_name == 'calcification' and review == 'AC':
                    if data_source not in excluded_sources:
                        documents.append(doc)
                        break
                elif self.finding_name == 'calcifiednodule' and review == 'fib or caln':
                    if data_source not in excluded_sources:
                        documents.append(doc)
                        break

        logger.info(f"Dataset Statistics (validation_internal_subgroup only):")
        logger.info(f"   Total documents: {len(all_docs)}")
        logger.info(f"   After excluding {excluded_sources}: {len(documents)}")

        return documents

    def _load_train_collection_with_consensus(self):
        """Load only from train collection with consensus annotations."""
        train_docs = self._load_train_collection_documents_with_consensus()
        logger.info(f"Dataset Statistics (train_collection only):")
        logger.info(f"   Loaded {len(train_docs)} documents from train collection")
        return train_docs

    def _load_train_collection_with_pseudo_label(self):
        """Load only from train collection with pseudo label."""
        train_docs = self._load_train_collection_pseudo_label()
        logger.info(f"Dataset Statistics (train_collection only):")
        logger.info(f"   Loaded {len(train_docs)} documents from train collection")
        return train_docs

    def _load_both_collections(self):
        """Load from both validation_internal and train collections."""
        # Load from validation_internal collection first
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

        logger.info(f"Dataset Statistics (validation_internal):")
        logger.info(f"   Total documents: {len(all_docs)}")
        logger.info(f"   After excluding {excluded_sources}: {len(documents)}")

        # Load additional documents from train collection
        train_docs = self._load_train_collection_documents_with_consensus()
        logger.info(f"Additional Train Collection Data:")
        logger.info(f"   Loaded {len(train_docs)} documents from train collection")
        documents.extend(train_docs)
        logger.info(f"   Total documents after merging: {len(documents)}")

        return documents

    def _load_train_collection_documents_with_consensus(self):
        """Load documents from train collection with specific findings."""
        collection = mdb_c.get_collection("train",
                                          db_names=['cxr_new', 'projects', 'public'])

        # Query for documents with specific findings
        query = {'labeled_findings': {'$all': ['calcification']},
                 'objects.finding_name': {'$in': ['calcification']}}

        # Add exclusion filter
        excluded_sources = ["amcio_b1_368"]
        query['data_source'] = {'$nin': excluded_sources}

        docs = list(collection.find(query))
        logger.info(f"   Found {len(docs)} total {self.finding_name} documents in train collection")

        # Load converted MDB data for consensus annotations
        converted_data = self._load_converted_data()
        if not converted_data:
            return []

        # Create mapping from path_dicom stem to doc
        doc_mapping = {}
        for doc in docs:
            if 'path_dicom' not in doc:
                continue
            try:
                path_dicom = doc['path_dicom']
                stem = Path(path_dicom).stem
                doc_mapping[stem] = doc
            except Exception:
                continue

        logger.info(f"   Created mapping for {len(doc_mapping)} documents with valid path_dicom")

        # Filter documents that exist in converted_data and have valid image paths
        filtered_docs = self._filter_documents_with_consensus(doc_mapping, converted_data)

        # Shuffle the filtered documents to ensure proper train/val split
        np.random.seed(42)
        np.random.shuffle(filtered_docs)
        logger.info(f"   Shuffled documents for proper train/val split")

        return filtered_docs

    def _load_converted_data(self):
        """Load converted MDB data for consensus annotations."""
        # Define paths based on finding_name
        if self.finding_name == 'calcification':
            first_round_path = "/opt/pxi/projects/calcified_nodule/labeling_tool_update/converted_mdb_data_1st_round.json"
            second_round_path = "/opt/pxi/projects/calcified_nodule/labeling_tool_update/converted_mdb_data_2nd_round.json"
            third_round_path = "/opt/pxi/projects/calcified_nodule/labeling_tool_update/converted_mdb_data_3rd_round.json"
        elif self.finding_name == 'calcifiednodule':
            first_round_path = "/opt/pxi/projects/calcified_nodule/relabeling_update/calcifiednodule/converted_mdb_data_calcifiednodule_1st_round.json"
            second_round_path = "/opt/pxi/projects/calcified_nodule/relabeling_update/calcifiednodule/converted_mdb_data_calcifiednodule_2nd_round.json"
        else:
            return {}

        # Load first round
        if not os.path.exists(first_round_path):
            print(f"   Warning: {first_round_path} not found")
            return {}

        with open(first_round_path, 'r') as f:
            converted_data_first_round = json.load(f)
        logger.info(f"   Loaded {len(converted_data_first_round)} entries from first round")

        # Load second round
        if not os.path.exists(second_round_path):
            print(f"   Warning: {second_round_path} not found")
            return {}

        with open(second_round_path, 'r') as f:
            converted_data_second_round = json.load(f)
        logger.info(f"   Loaded {len(converted_data_second_round)} entries from second round")

        # Merge the two dictionaries
        return {**converted_data_first_round, **converted_data_second_round}

    def _filter_documents_with_consensus(self, doc_mapping, converted_data):
        """Filter documents that have consensus annotations."""
        filtered_docs = []
        matched_keys = []
        unmatched_keys = []
        no_path_image = 0
        no_valid_image = 0
        no_consensus = 0
        no_finding = 0
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
                                    # Use different finding names based on the target
                                    if self.finding_name == 'calcification' and obj.get('finding_name') == 'Aorta Calcification':
                                        consensus_objects.append({'finding_name': 'calcification',
                                                                  'polygon': obj.get('polygon'),
                                                                  'confidence': obj.get('confidence'),
                                                                  'remark': obj.get('remark', '')
                                                                  })
                                    elif self.finding_name == 'calcifiednodule' and obj.get('finding_name') == 'Calcified Nodule':
                                        consensus_objects.append({'finding_name': 'calcifiednodule',
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
                                no_finding += 1

                        else:
                            no_consensus += 1
                    else:
                        no_valid_image += 1
                except Exception:
                    no_valid_image += 1
                    continue
            else:
                unmatched_keys.append(json_key)

        logger.info(f"   Matched keys: {len(matched_keys)}")
        logger.info(f"   Unmatched keys: {len(unmatched_keys)}")
        logger.info(f"   Filtering breakdown:")
        logger.info(f"     - No path_image: {no_path_image}")
        logger.info(f"     - No valid image: {no_valid_image}")
        logger.info(f"     - No consensus: {no_consensus}")
        logger.info(f"     - No {self.finding_name}: {no_finding}")
        logger.info(f"     - Excluded: {excluded_count}")
        if unmatched_keys:
            logger.info(f"   First 10 unmatched keys: {unmatched_keys[:10]}")

        logger.info(
            f"   Filtered {len(filtered_docs)} documents with consensus annotations (excluded: {excluded_count})")
        return filtered_docs

    def _load_train_collection_pseudo_label(self):
        """Load train collection with pseudo label."""
        def load_pseudo_label(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                ref_data = json.load(f)
            return ref_data["samples"]

        # Define paths based on finding_name
        if self.finding_name == 'calcification':
            json_paths = ["/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p0-0p1.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p1-0p2.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p2-0p3.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p3-0p4.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p4-0p5.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p5-0p6.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p6-0p7.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p7-0p8.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p8-0p9.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p9-1p0.json"]
        else:
            # For calcifiednodule, use the same paths as calcification for now
            json_paths = ["/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p0-0p1.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p1-0p2.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p2-0p3.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p3-0p4.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p4-0p5.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p5-0p6.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p6-0p7.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p7-0p8.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p8-0p9.json",
                          "/opt/pxi/projects/calcified_nodule/segmentation/calcification/pseudo_label/confidence_interval_0p9-1p0.json"]

        # Load all pseudo label data
        ref_data_list = []
        for json_path in json_paths:
            if os.path.exists(json_path):
                ref_data_list.append(load_pseudo_label(json_path))
            else:
                ref_data_list.append([])

        # 1~5 neg, 6~10 pos
        neg_json = ref_data_list[0]
        pos_json = ref_data_list[9]
        undefined_json = sum(ref_data_list[1:9], [])

        filtered_doc = []
        pos_count = 0
        neg_count = 0

        # Process positive samples
        for doc in pos_json:
            consensus_objects = []
            prediction_polygons = doc.get('prediction_polygons', [])
            for polygon in prediction_polygons:
                if polygon:
                    consensus_objects.append({"finding_name": self.finding_name,
                                              "polygon": polygon,
                                              "confidence": 1.0,
                                              "remark": "pseudo_label"})
            if consensus_objects:
                updated_doc = doc.copy()
                updated_doc["objects"] = consensus_objects
                filtered_doc.append(updated_doc)
                pos_count += 1
            else:
                updated_doc = doc.copy()
                updated_doc["objects"] = []
                filtered_doc.append(updated_doc)
                neg_count += 1

        # Process negative samples
        for doc in neg_json:
            updated_doc = doc.copy()
            updated_doc["objects"] = []
            filtered_doc.append(updated_doc)
            neg_count += 1

        logger.info(f"   Filtered positive before adding undefined: {pos_count}")
        logger.info(f"   Filtered negative before adding undefined: {neg_count}")

        # Process undefined samples
        if self.finding_name == 'calcification':
            converted_data_path_third_round = "/opt/pxi/projects/calcified_nodule/relabeling_update/converted_mdb_data_3rd_round.json"
        else:
            converted_data_path_third_round = "/opt/pxi/projects/calcified_nodule/relabeling_update/converted_mdb_data_3rd_round.json"

        if os.path.exists(converted_data_path_third_round):
            with open(converted_data_path_third_round, 'r') as f:
                converted_data_third_round = json.load(f)

            doc_mapping = {}
            for doc in undefined_json:
                try:
                    path_dicom = doc['doc']['path_dicom']
                    stem = Path(path_dicom).stem
                    doc_mapping[stem] = doc['doc']
                except Exception:
                    continue
            logger.info(f"   Undefined documents: {len(doc_mapping)}")

            for json_key in converted_data_third_round.keys():
                consensus_objects = []
                if json_key in doc_mapping:
                    model_prediction = converted_data_third_round[json_key]['consensus'][0].get('objects', [])
                    for obj in model_prediction:
                        if self.finding_name == 'calcification' and obj.get('finding_name') == 'Aorta Calcification':
                            consensus_objects.append({"finding_name": "calcification",
                                                      "polygon": obj.get('polygon'),
                                                      "confidence": 1.0,
                                                      "remark": "pseudo_label"
                                                      })
                        elif self.finding_name == 'calcifiednodule' and obj.get('finding_name') == 'Aorta Calcification':
                            consensus_objects.append({"finding_name": "calcifiednodule",
                                                      "polygon": obj.get('polygon'),
                                                      "confidence": 1.0,
                                                      "remark": "pseudo_label"
                                                      })
                    if consensus_objects:
                        updated_doc = doc_mapping[json_key].copy()
                        updated_doc["objects"] = consensus_objects
                        filtered_doc.append(updated_doc)
                        pos_count += 1
                    else:
                        updated_doc = doc_mapping[json_key].copy()
                        updated_doc["objects"] = []
                        filtered_doc.append(updated_doc)
                        neg_count += 1

        logger.info(f"   Filtered positive after adding undefined: {pos_count}")
        logger.info(f"   Filtered negative after adding undefined: {neg_count}")
        logger.info(f"   Found {len(filtered_doc)} total {self.finding_name} documents in train collection")

        # Shuffle the filtered documents to ensure proper train/val split
        np.random.seed(42)
        np.random.shuffle(filtered_doc)
        logger.info(f"   Shuffled documents for proper train/val split")

        return filtered_doc

    def _print_dataset_statistics(self):
        """Print dataset statistics."""
        positive_samples = 0
        negative_samples = 0
        finding_with_polygon = 0
        finding_without_polygon = 0

        for doc in self.documents:
            objects = doc.get('objects', [])
            has_finding = False
            has_polygon = False

            for obj in objects:
                if obj.get('finding_name') == self.finding_name:
                    has_finding = True
                    if obj.get('polygon'):
                        has_polygon = True
                    break

            if has_finding:
                positive_samples += 1
                if has_polygon:
                    finding_with_polygon += 1
                else:
                    finding_without_polygon += 1
            else:
                negative_samples += 1

        print(f"   Positive samples (with {self.finding_name}): {positive_samples}")
        print(f"     - With polygon: {finding_with_polygon}")
        print(f"     - Without polygon: {finding_without_polygon}")
        print(f"   Negative samples (without {self.finding_name}): {negative_samples}")
        print(f"   Positive ratio: {positive_samples / (positive_samples + negative_samples) * 100:.2f}%")
        print("=" * 50)

    def _split_positive_negative(self) -> Tuple[List[Dict], List[Dict]]:
        """Split documents into positive and negative samples."""
        positive_docs = [doc for doc in self.documents
                         if any(obj.get('finding_name') == self.finding_name for obj in doc.get('objects', []))]
        negative_docs = [doc for doc in self.documents
                         if not any(obj.get('finding_name') == self.finding_name for obj in doc.get('objects', []))]

        # Shuffle both lists
        np.random.seed(42)
        np.random.shuffle(positive_docs)
        np.random.shuffle(negative_docs)

        return positive_docs, negative_docs

    def _apply_split(self, positive_docs: List[Dict], negative_docs: List[Dict], train_data_size: int = None) -> List[Dict]:
        """Apply train/val split."""
        # Split positive and negative samples separately (only for training collections)
        if self.train_dataset in ['train_collection_with_consensus', 'train_collection_with_pseudo_label', 'both']:
            pos_train_size = int(0.8 * len(positive_docs))
            neg_train_size = int(0.8 * len(negative_docs))

            if self.split == 'train':
                print(f"   Pos train size: {pos_train_size}")
                print(f"   Neg train size: {neg_train_size}")
                positive_docs = positive_docs[:pos_train_size]
                negative_docs = negative_docs[:neg_train_size]
                if train_data_size is not None:
                    positive_docs = positive_docs[:train_data_size]
                    negative_docs = negative_docs[:train_data_size]
                return positive_docs + negative_docs
            else:
                print(f"   Pos val size: {len(positive_docs[pos_train_size:])}")
                print(f"   Neg val size: {len(negative_docs[neg_train_size:])}")
                return positive_docs[pos_train_size:] + negative_docs[neg_train_size:]
        else:
            # For validation collections, use all documents regardless of split
            return positive_docs + negative_docs

    def _apply_selection_strategy(self) -> List[Dict]:
        """Apply active learning selection strategy to training samples."""
        if self.selection_strategy == 'random':
            return self.documents

        print(f"🎯 Applying {self.selection_strategy} selection strategy...")

        # Use the new sample selection module
        num_samples = self.train_data_size if self.train_data_size else len(self.documents)

        selected_docs = select_samples(
            documents=self.documents,
            strategy=self.selection_strategy,
            num_samples=num_samples,
            finding_name=self.finding_name,
            **self.selection_params
        )

        return selected_docs

    def _print_split_statistics(self):
        """Print split statistics."""
        split_pos = 0
        split_neg = 0
        for doc in self.documents:
            objects = doc.get('objects', [])
            has_finding = False
            for obj in objects:
                if obj.get('finding_name') == self.finding_name:
                    has_finding = True
                    break

            if has_finding:
                split_pos += 1
            else:
                split_neg += 1

        print(f"   {self.split.capitalize()} split - Positive: {split_pos}, Negative: {split_neg}")
        print(f"   {self.split.capitalize()} split - Positive ratio: {split_pos / (split_pos + split_neg) * 100:.2f}%")
        print("=" * 50)

    def _setup_transforms(self):
        """Setup data transforms."""
        # Enhanced Transforms with medical image specific augmentations
        if self.train_dataset in ['train_collection_with_consensus', 'train_collection_with_pseudo_label', 'both']:
            # Apply transforms only for training collections
            if self.split == 'train':
                return A.Compose([A.Resize(self.target_size[0], self.target_size[1]),
                                  A.HorizontalFlip(p=0.5),
                                  A.VerticalFlip(p=0.3),
                                  A.Rotate(limit=10, p=0.5),
                                  A.GridDistortion(p=0.3),
                                  A.RandomBrightnessContrast(p=0.3),
                                  A.GaussNoise(p=0.2),
                                  A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            else:
                return A.Compose([A.Resize(self.target_size[0], self.target_size[1]),
                                  A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            return A.Compose([A.Resize(self.target_size[0], self.target_size[1]),
                              A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def _calculate_weights(self) -> List[float]:
        """Calculate weights for weighted sampling."""
        return [
            3.0 if any(obj.get('finding_name') == self.finding_name for obj in doc.get('objects', []))
            else 1.0
            for doc in self.documents]

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Resize image to target size first
        import cv2
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))

        # Create mask
        mask = np.zeros(self.target_size, dtype=np.uint8)
        objects = doc.get('objects', [])

        # Check if document has the specific finding
        has_finding = False
        mask_created = False

        for obj in objects:
            if obj.get('finding_name') == self.finding_name:
                has_finding = True
                polygon = obj.get('polygon')
                if polygon:
                    # Simple polygon to mask conversion
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
                            mask_created = True
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
