#!/usr/bin/env python3
"""
Dataset module for SAM adaptation project.

This module contains dataset classes and utilities for loading and processing
medical images and annotations for lesion segmentation.
"""

from .dataset import (LesionDataset, SpatialSplitDataset, _get_mask_from_doc,
                      create_al_datasets, create_spatial_validation_split,
                      divide_image_into_areas, load_raw_documents)
from .vindr_cxr_dataset import (VinDrCXRDataset, create_vindr_al_datasets,
                                 load_vindr_documents, VINDR_CLASSES, LESION_MAPPING)

# Backward compatibility alias
SAMLesionDataset = LesionDataset

__all__ = [
    # Internal datasets
    'LesionDataset',
    'SAMLesionDataset',  # backward compatibility
    'SpatialSplitDataset',
    'divide_image_into_areas',
    'create_spatial_validation_split',
    'load_raw_documents',
    'create_al_datasets',
    '_get_mask_from_doc',
    # VinDr-CXR public dataset
    'VinDrCXRDataset',
    'create_vindr_al_datasets',
    'load_vindr_documents',
    'VINDR_CLASSES',
    'LESION_MAPPING',
]
