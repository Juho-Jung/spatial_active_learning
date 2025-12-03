#!/usr/bin/env python3
"""
ALUNET-based sample selection strategies.

This module implements sample selection strategies from ALUNET:
- USIMC: Uncertainty-Aware Submodular Mutual Information with Cosine similarity
- MCD: Monte Carlo Dropout based uncertainty sampling

Reference: https://github.com/Berni1557/ALUNET
Paper: "Active Learning with nnUNet and Sample Selection with Uncertainty-Aware Submodular Mutual Information Measure" (MIDL 2024)
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, Subset

try:
    from submodlib.functions.facilityLocationVariantMutualInformation import \
        FacilityLocationVariantMutualInformationFunction
    SUBMODLIB_AVAILABLE = True
except ImportError:
    SUBMODLIB_AVAILABLE = False
    print("⚠️ submodlib not available. USIMC strategy will fall back to uncertainty selection.")


def _compute_gradients(model, dataloader, device, num_params=10000):
    """
    Compute gradients for samples using the model.
    Optimized version that processes samples individually but reuses model forward pass.

    Args:
        model: PyTorch model
        dataloader: DataLoader for samples
        device: Device to use
        num_params: Maximum number of parameters to use for gradient computation

    Returns:
        gradients: Array of gradients [N, D] where D is gradient dimension
    """
    model.eval()
    gradients = []

    # Get model parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # Pre-compute parameter sampling indices if needed (for reproducibility)
    # We'll determine this after first gradient computation
    param_indices = None

    # Count total samples for progress bar
    total_samples = len(dataloader.dataset)
    processed_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
            if len(batch) > 1:
                targets = batch[1].to(device)
            else:
                targets = None
        else:
            images = batch.to(device)
            targets = None

        batch_size = images.shape[0]

        # Forward pass once for the batch to create targets if needed
        with torch.no_grad():
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                output = outputs[0]
            else:
                output = outputs

            # Create targets from predictions if not provided
            if targets is None:
                if output.shape[1] > 1:  # Multi-class
                    targets = torch.argmax(output, dim=1, keepdim=True)
                else:  # Binary segmentation
                    targets = (torch.sigmoid(output) > 0.5).long()

        # Compute gradients for each sample in the batch individually
        for i in range(batch_size):
            # Get single sample
            single_image = images[i:i + 1]
            single_target = targets[i:i + 1]

            # Forward pass for gradient computation
            outputs = model(single_image)
            if isinstance(outputs, (list, tuple)):
                output = outputs[0]
            else:
                output = outputs

            # Compute loss for single sample
            if output.shape[1] > 1:  # Multi-class
                loss = F.cross_entropy(output, single_target.squeeze(1), reduction='sum')
            else:  # Binary segmentation
                loss = F.binary_cross_entropy_with_logits(
                    output.squeeze(1), single_target.squeeze(1).float(), reduction='sum')

            # Compute gradients
            grad_params = torch.autograd.grad(loss, params, create_graph=False, allow_unused=True)

            # Flatten gradients
            grad_flat = torch.cat([g.flatten() for g in grad_params if g is not None])

            # Limit gradient dimension if needed (only compute indices once)
            if len(grad_flat) > num_params:
                if param_indices is None:
                    # Compute indices once and reuse
                    rng_grad = np.random.RandomState(42)
                    param_indices = rng_grad.choice(len(grad_flat), size=num_params, replace=False)
                grad_flat = grad_flat[param_indices]

            gradients.append(grad_flat.cpu().detach().numpy())
            processed_samples += 1

            # Print progress every 50 samples
            if processed_samples % 50 == 0:
                print(f"   Processed {processed_samples}/{total_samples} samples...")

    return np.array(gradients)


def select_samples_usimc(pool_indices, uncertainties, num_samples, selected_indices, areas,
                         features=None, model=None, dataloader=None, device=None, seed=None):
    """
    USIMC: Uncertainty-Aware Submodular Mutual Information with Cosine similarity.

    This strategy combines uncertainty with diversity using gradient-based submodular mutual information.
    Based on ALUNET's USIMCStrategy.

    Args:
        pool_indices: List of available sample indices
        uncertainties: List of uncertainty values for each sample
        num_samples: Number of samples to select
        selected_indices: List of already selected indices
        areas: List of spatial areas (not used)
        features: Feature embeddings (not used, gradients are computed instead)
        model: PyTorch model for gradient computation (required)
        dataloader: DataLoader for pool samples (required)
        device: Device to use (required)
        seed: Random seed for reproducibility

    Returns:
        List of selected sample indices
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # Use specific seed for reproducibility
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # Check if submodlib is available
    if not SUBMODLIB_AVAILABLE:
        print("⚠️ USIMC: submodlib not available. Falling back to uncertainty selection.")
        if uncertainties is not None:
            from .sample_selection import select_samples_uncertainty
            return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)
        else:
            return rng.choice(available_indices, size=num_samples, replace=False).tolist()

    # Check required parameters
    if model is None or dataloader is None or device is None:
        print("⚠️ USIMC: Missing required parameters (model, dataloader, or device). "
              "Falling back to uncertainty selection.")
        if uncertainties is not None:
            from .sample_selection import select_samples_uncertainty
            return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)
        else:
            return rng.choice(available_indices, size=num_samples, replace=False).tolist()

    # Convert uncertainties to numpy if needed
    if uncertainties is not None:
        if torch.is_tensor(uncertainties):
            uncertainties = uncertainties.cpu().numpy()
        uncertainties = np.array(uncertainties)

        # Filter uncertainties for available indices
        pool_idx_to_pos = {idx: pos for pos, idx in enumerate(pool_indices)}
        available_uncertainties = np.array([uncertainties[pool_idx_to_pos[idx]] for idx in available_indices])

        # Normalize uncertainties to probabilities
        if available_uncertainties.sum() > 0:
            prop = available_uncertainties / available_uncertainties.sum()
        else:
            prop = np.ones(len(available_indices)) / len(available_indices)
    else:
        prop = np.ones(len(available_indices)) / len(available_indices)

    # Optimize: Only compute gradients for top-N uncertain samples (default: top 100)
    max_gradient_samples = 100
    gradient_indices = list(range(len(available_indices)))

    if uncertainties is not None and len(available_uncertainties) > 0:
        # Select top-N uncertain samples for gradient computation
        top_n_indices = np.argsort(available_uncertainties)[-max_gradient_samples:][::-1]
        gradient_indices = top_n_indices.tolist()
        print(f"🔍 USIMC: Computing gradients for top {max_gradient_samples} uncertain samples "
              f"(out of {len(available_indices)} available samples) for speed...")
    else:
        # Random sampling if no uncertainties
        gradient_indices = rng.choice(len(available_indices), size=min(
            max_gradient_samples, len(available_indices)), replace=False).tolist()
        print(f"🔍 USIMC: Computing gradients for {len(gradient_indices)} randomly sampled samples "
              f"(out of {len(available_indices)} available samples) for speed...")

    # Create subset dataloader for selected samples
    subset_dataset = Subset(dataloader.dataset, gradient_indices)
    gradient_dataloader = DataLoader(subset_dataset, batch_size=dataloader.batch_size, shuffle=False)

    try:
        # Use reduced num_params for speed (5000 instead of 10000)
        gradients = _compute_gradients(model, gradient_dataloader, device, num_params=5000)
        print(f"✅ USIMC: Computed gradients with shape {gradients.shape} (for {len(gradient_indices)} samples)")
    except Exception as e:
        print(f"⚠️ USIMC: Failed to compute gradients: {e}. Falling back to uncertainty selection.")
        if uncertainties is not None:
            from .sample_selection import select_samples_uncertainty
            return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)
        else:
            return rng.choice(available_indices, size=num_samples, replace=False).tolist()

    # Estimate number of query samples using knee detection
    # This is simplified - in original ALUNET, they use KneeLocator
    # Note: query samples must be from gradient_indices (samples we computed gradients for)
    num_samples_q = min(num_samples * 5, len(gradient_indices))

    # Sample query set based on uncertainty (only from samples with gradients)
    # Map uncertainty probabilities to gradient_indices
    if uncertainties is not None and len(available_uncertainties) > 0:
        gradient_uncertainties = available_uncertainties[gradient_indices]
        if gradient_uncertainties.sum() > 0:
            gradient_prop = gradient_uncertainties / gradient_uncertainties.sum()
        else:
            gradient_prop = np.ones(len(gradient_indices)) / len(gradient_indices)
    else:
        gradient_prop = np.ones(len(gradient_indices)) / len(gradient_indices)

    # Sample query indices from gradient_indices (local indices 0 to len(gradient_indices)-1)
    query_gradient_indices = rng.choice(len(gradient_indices), size=num_samples_q, replace=True, p=gradient_prop)
    query_gradients = gradients[query_gradient_indices]

    # Use submodular mutual information
    # n is the number of samples with gradients (not all available_indices)
    n = len(gradient_indices)
    num_queries = len(query_gradient_indices)

    # Choose optimizer based on size
    if len(available_indices) < 1000:
        optimizer = 'NaiveGreedy'
    else:
        optimizer = 'StochasticGreedy'

    try:
        # Create submodular function
        obj = FacilityLocationVariantMutualInformationFunction(
            n=n,
            num_queries=num_queries,
            query_sijs=None,
            data=gradients,
            queryData=query_gradients,
            metric='cosine',
            queryDiversityEta=1.0
        )

        # Maximize submodular function
        greedy_list = obj.maximize(
            budget=num_samples,
            optimizer=optimizer,
            stopIfZeroGain=False,
            stopIfNegativeGain=False,
            verbose=False,
            show_progress=False,
            epsilon=0.01
        )

        # Map back from gradient indices to available_indices
        selected_gradient_indices = [x[0] for x in greedy_list]
        # Convert gradient indices to available_indices
        selected_indices_list = [available_indices[gradient_indices[i]] for i in selected_gradient_indices]

        print(f"✅ USIMC: Selected {len(selected_indices_list)} samples using submodular mutual information "
              f"(from {len(gradient_indices)} gradient-computed samples).")

        return selected_indices_list

    except Exception as e:
        print(f"⚠️ USIMC: Failed to run submodular optimization: {e}. Falling back to uncertainty selection.")
        if uncertainties is not None:
            from .sample_selection import select_samples_uncertainty
            return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)
        else:
            return rng.choice(available_indices, size=num_samples, replace=False).tolist()


def select_samples_mcd_alunet(pool_indices, uncertainties, num_samples, selected_indices, areas, seed=None):
    """
    MCD (Monte Carlo Dropout) strategy from ALUNET.

    This is similar to existing MC Dropout uncertainty, but uses ALUNET's specific implementation.
    For now, this is a wrapper around existing uncertainty selection since MC Dropout
    uncertainty is already computed.

    Args:
        pool_indices: List of available sample indices
        uncertainties: List of uncertainty values (from MC Dropout)
        num_samples: Number of samples to select
        selected_indices: List of already selected indices
        areas: List of spatial areas (not used)
        seed: Random seed for reproducibility

    Returns:
        List of selected sample indices
    """
    # This is essentially the same as uncertainty selection with MC Dropout
    from .sample_selection import select_samples_uncertainty
    return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)
