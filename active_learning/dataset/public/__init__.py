#!/usr/bin/env python3
"""
Public benchmark datasets for SPARCL.

Wrappers over the public CXR datasets used in the paper. Each dataset is
dual-mode: it can emit bounding boxes (``mask_type='detection'``) or masks
(``mask_type`` in {rectangular, ellipse, gaussian}) so that both the detection
and segmentation tracks share the same data sources.

- ``vindr_cxr``: VinDr-CXR (bbox annotations)
- ``chestxdet10``: ChestX-Det10 (bbox annotations)
- ``siim``: SIIM-ACR Pneumothorax (segmentation masks)
"""

from .chestxdet10 import (CONCENTRATED_LESIONS as CHESTXDET10_CONCENTRATED,
                          DISPERSED_LESIONS as CHESTXDET10_DISPERSED,
                          ChestXDet10Dataset, create_chestxdet10_al_datasets)
from .siim import (SIIMDataset, SIIMSubsetDataset, create_siim_al_datasets)
from .vindr_cxr import (CONCENTRATED_LESIONS, DISPERSED_LESIONS, LESION_MAPPING,
                        VINDR_CLASSES, VinDrCXRDataset,
                        create_vindr_al_datasets, load_vindr_documents)

__all__ = [
    # VinDr-CXR
    'VinDrCXRDataset',
    'create_vindr_al_datasets',
    'load_vindr_documents',
    'VINDR_CLASSES',
    'LESION_MAPPING',
    'CONCENTRATED_LESIONS',
    'DISPERSED_LESIONS',
    # ChestX-Det10
    'ChestXDet10Dataset',
    'create_chestxdet10_al_datasets',
    'CHESTXDET10_CONCENTRATED',
    'CHESTXDET10_DISPERSED',
    # SIIM-ACR Pneumothorax
    'SIIMDataset',
    'SIIMSubsetDataset',
    'create_siim_al_datasets',
]
