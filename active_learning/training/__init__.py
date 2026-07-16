#!/usr/bin/env python3
"""
Training module for SPARCL.

This module contains training utilities including model training functions,
early stopping, and logging utilities for active learning experiments.
"""

from .training import (
    train_model_round,
    _setup_round_logger,
    _train_with_early_stopping,
    _train_epoch,
    _validate_epoch,
    _calculate_batch_bin_dices,
    _calculate_performance_coverage,
    _log_epoch_metrics,
    _save_model_checkpoint,
    _log_round_completion,
    detection_collate_fn,
)

__all__ = [
    'train_model_round',
    '_setup_round_logger',
    '_train_with_early_stopping',
    '_train_epoch',
    '_validate_epoch',
    '_calculate_batch_bin_dices',
    '_calculate_performance_coverage',
    '_log_epoch_metrics',
    '_save_model_checkpoint',
    '_log_round_completion',
    'detection_collate_fn',
]
