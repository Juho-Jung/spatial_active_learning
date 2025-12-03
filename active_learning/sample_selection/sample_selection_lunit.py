#!/usr/bin/env python3
"""
LUNIT (Learning Loss) sample selection strategy.

This module implements the Learning Loss for Active Learning strategy,
which uses a loss prediction module to predict loss values for unlabeled samples
and selects samples with highest predicted loss.

Reference: "Learning Loss for Active Learning" (CVPR 2019)
Paper: https://arxiv.org/abs/1905.03677
"""

import numpy as np
import torch
from torch.utils.data import DataLoader


def select_samples_lunit(pool_indices, uncertainties, num_samples, selected_indices,
                         model, dataloader, device, seed=None):
    """
    Select samples using Learning Loss (LUNIT) strategy.

    The loss prediction module predicts loss values for unlabeled samples,
    and we select samples with highest predicted loss.

    Args:
        pool_indices: List of available pool indices
        uncertainties: Not used (kept for compatibility)
        num_samples: Number of samples to select
        selected_indices: List of already selected indices
        model: Model with loss prediction module (SegmentationModelWithLossPrediction)
        dataloader: DataLoader for unlabeled pool samples
        device: Device to run inference on
        seed: Random seed (not used, kept for compatibility)

    Returns:
        List of selected indices
    """
    if model is None or dataloader is None or device is None:
        print("⚠️ LUNIT: Missing required parameters (model, dataloader, or device). "
              "Falling back to random selection.")
        rng = np.random.RandomState(seed if seed is not None else 42)
        available_indices = [idx for idx in pool_indices if idx not in selected_indices]
        if len(available_indices) < num_samples:
            return available_indices
        return rng.choice(available_indices, size=num_samples, replace=False).tolist()

    # Filter out already selected indices
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]

    if len(available_indices) < num_samples:
        print(f"⚠️ LUNIT: Only {len(available_indices)} available samples, "
              f"requested {num_samples}. Returning all available.")
        return available_indices

    # Check if model has loss prediction module
    if not hasattr(model, 'loss_prediction_module'):
        print("⚠️ LUNIT: Model does not have loss_prediction_module. "
              "Falling back to random selection.")
        rng = np.random.RandomState(seed if seed is not None else 42)
        return rng.choice(available_indices, size=num_samples, replace=False).tolist()

    print(f"🔍 LUNIT: Predicting losses for {len(available_indices)} samples...")

    # Predict losses for all available samples
    model.eval()
    predicted_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Extract images from batch
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images)

            images = images.to(device)

            # Get predicted loss (without ground truth)
            try:
                _, predicted_loss = model(images, return_loss_prediction=True)
                # predicted_loss: [B, 1]
                predicted_losses.extend(predicted_loss.cpu().numpy().flatten())
            except Exception as e:
                print(f"⚠️ LUNIT: Error predicting loss for batch {batch_idx}: {e}")
                # Fallback: assign random losses
                batch_size = images.shape[0]
                predicted_losses.extend(np.random.rand(batch_size).tolist())

    predicted_losses = np.array(predicted_losses)

    # Ensure we have predictions for all samples
    if len(predicted_losses) != len(available_indices):
        print(f"⚠️ LUNIT: Mismatch between predicted losses ({len(predicted_losses)}) "
              f"and available indices ({len(available_indices)}). "
              "Using random selection for missing samples.")
        # Pad with random values if needed
        if len(predicted_losses) < len(available_indices):
            rng = np.random.RandomState(seed if seed is not None else 42)
            padding = rng.rand(len(available_indices) - len(predicted_losses))
            predicted_losses = np.concatenate([predicted_losses, padding])
        else:
            predicted_losses = predicted_losses[:len(available_indices)]

    # Select top-K samples with highest predicted loss
    top_k_indices = np.argsort(predicted_losses)[-num_samples:][::-1]
    selected = [available_indices[i] for i in top_k_indices]

    print(f"✅ LUNIT: Selected {len(selected)} samples with highest predicted loss "
          f"(loss range: {predicted_losses[top_k_indices].min():.4f} - "
          f"{predicted_losses[top_k_indices].max():.4f})")

    return selected
