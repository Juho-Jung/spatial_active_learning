#!/usr/bin/env python3
"""
Dataset module for SPARCL.

This module contains dataset classes and utilities for loading and processing
medical images and annotations for lesion detection and segmentation. The
in-house (MongoDB) dataset lives here; public benchmark datasets live under
``dataset.public``.
"""

from .dataset import (LesionDataset, SpatialSplitDataset, _get_mask_from_doc,
                      create_al_datasets, create_spatial_validation_split,
                      divide_image_into_areas, load_raw_documents)
from .public.vindr_cxr import (VinDrCXRDataset, create_vindr_al_datasets,
                                load_vindr_documents, VINDR_CLASSES, LESION_MAPPING)

__all__ = [
    # Internal datasets
    'LesionDataset',
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
