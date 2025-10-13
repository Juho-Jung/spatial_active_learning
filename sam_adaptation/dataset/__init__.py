#!/usr/bin/env python3
"""
Dataset module for SAM adaptation project.

This module contains dataset classes and utilities for loading and processing
medical images and annotations for lesion segmentation.
"""

from .dataset import (
    SAMLesionDataset,
    SpatialSplitDataset,
    create_spatial_validation_split,
    load_raw_documents,
    _get_mask_from_doc
)

__all__ = [
    'SAMLesionDataset',
    'SpatialSplitDataset',
    'create_spatial_validation_split',
    'load_raw_documents',
    '_get_mask_from_doc'
]
