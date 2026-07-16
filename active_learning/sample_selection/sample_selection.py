#!/usr/bin/env python3
"""
Sample selection strategies for active learning.

This module contains various sample selection strategies used in active learning
experiments, including random, uncertainty-based, and area-based selection methods.
"""

import numpy as np
import torch

# Baseline AL strategies: Coreset (ICLR 2018), TAUDIS (ICCV 2023),
# ALUNET (MIDL 2024), LUNIT / Learning Loss (CVPR 2019).
try:
    from .baselines.coreset import select_samples_coreset
except ImportError:
    select_samples_coreset = None

try:
    from .baselines.taudis import select_samples_TAUDIS
except ImportError:
    select_samples_TAUDIS = None

try:
    from .baselines.alunet import (select_samples_mcd_alunet,
                                   select_samples_usimc)
except ImportError:
    select_samples_usimc = None
    select_samples_mcd_alunet = None

try:
    from .baselines.lunit import select_samples_lunit
except ImportError:
    select_samples_lunit = None


def create_spatial_bins(height, width, grid_width=2, grid_height=3):
    """
    Create spatial bin assignments for chest X-ray (3x2 grid = 6 bins).

    Args:
        height: Image height
        width: Image width
        grid_width: Number of columns (default 2 for chest X-ray)
        grid_height: Number of rows (default 3 for chest X-ray)

    Returns:
        bin_id: Array [H, W] with values 0..5 for 6 spatial bins
    """
    bin_id = np.zeros((height, width), dtype=np.int32)

    h_bin_size = height // grid_height
    w_bin_size = width // grid_width

    for i in range(grid_height):
        for j in range(grid_width):
            bin_idx = i * grid_width + j

            # Define bin boundaries
            h_start = i * h_bin_size
            h_end = (i + 1) * h_bin_size if i < grid_height - 1 else height
            w_start = j * w_bin_size
            w_end = (j + 1) * w_bin_size if j < grid_width - 1 else width

            bin_id[h_start:h_end, w_start:w_end] = bin_idx

    return bin_id


def select_samples_random(pool_indices, num_samples, selected_indices, seed=None):
    """Random selection strategy."""
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # Use specific seed for reproducibility with isolated random state
    if seed is not None:
        # Create a new random generator with the specific seed to avoid interference
        rng = np.random.RandomState(seed)
        return rng.choice(available_indices, size=num_samples, replace=False).tolist()
    else:
        return np.random.choice(available_indices, size=num_samples, replace=False).tolist()


def select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed=None):
    """Uncertainty-based selection strategy."""
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round (no uncertainties), use random selection with same seed as random mode
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)

    # Get uncertainties for available indices
    available_uncertainties = [(idx, uncertainties[idx]) for idx in available_indices]
    # Sort by uncertainty (descending)
    available_uncertainties.sort(key=lambda x: x[1], reverse=True)

    return [idx for idx, _ in available_uncertainties[:num_samples]]


def select_samples_area_random(pool_indices, num_samples, selected_indices, areas, seed=None):
    """Area-based random selection strategy."""
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # Use specific seed for reproducibility with isolated random state
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # Group available indices by area
    area_groups = {i: [] for i in range(len(areas))}
    for idx in available_indices:
        # For simplicity, assign indices to areas in round-robin fashion
        area_idx = idx % len(areas)
        area_groups[area_idx].append(idx)

    # Shuffle each area group for randomness
    for area_idx in area_groups:
        rng.shuffle(area_groups[area_idx])

    # Round-robin selection from each area
    selected = []
    area_idx = 0
    consecutive_empty_areas = 0

    while len(selected) < num_samples and len(selected) < len(available_indices):
        if area_groups[area_idx]:
            selected.append(area_groups[area_idx].pop(0))
            consecutive_empty_areas = 0  # Reset counter when we find samples
        else:
            consecutive_empty_areas += 1

        area_idx = (area_idx + 1) % len(areas)

        # If we've checked all areas and found no samples, break to avoid infinite loop
        if consecutive_empty_areas >= len(areas):
            break

    return selected


def select_samples_uncertainty_area(pool_indices, uncertainties, num_samples, selected_indices, areas, seed=None):
    """Uncertainty-based area selection strategy."""
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round (no uncertainties), use area random selection with same seed as area_random mode
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_area_random(pool_indices, num_samples, selected_indices, areas, seed)

    # Group available indices by area with their uncertainties
    area_groups = {i: [] for i in range(len(areas))}
    for idx in available_indices:
        area_idx = idx % len(areas)
        area_groups[area_idx].append((idx, uncertainties[idx]))

    # Sort each area group by uncertainty (descending)
    for area_idx in area_groups:
        area_groups[area_idx].sort(key=lambda x: x[1], reverse=True)

    # Round-robin selection from each area's uncertainty queue
    selected = []
    area_idx = 0
    consecutive_empty_areas = 0

    while len(selected) < num_samples and len(selected) < len(available_indices):
        if area_groups[area_idx]:
            selected.append(area_groups[area_idx].pop(0)[0])  # Take index only
            consecutive_empty_areas = 0  # Reset counter when we find samples
        else:
            consecutive_empty_areas += 1

        area_idx = (area_idx + 1) % len(areas)

        # If we've checked all areas and found no samples, break to avoid infinite loop
        if consecutive_empty_areas >= len(areas):
            break

    return selected


def select_samples_adaptive(pool_indices, uncertainties, num_samples, selected_indices,
                            probs=None, lesionness=None, bin_id=None, lambda1=1.0, seed=None):
    """
    Adaptive selection strategy that adapts to lesion clustering in spatial regions.

    Args:
        pool_indices: List of available sample indices
        uncertainties: List of uncertainty values for each sample (U(x))
        num_samples: Number of samples to select
        selected_indices: List of already selected indices
        probs: Tensor [N, H, W] - pixel prediction probabilities p(i)
        lesionness: Tensor [N, H, W] - lesionness values M(i)
        bin_id: Tensor [H, W] - spatial bin assignments (0..K-1), K=6 for 3x2 grid
        lambda1: Weight for spatial coverage term
        seed: Random seed for reproducibility

    Returns:
        List of selected sample indices
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round (no uncertainties), use random selection
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)

    # If required tensors are not provided, fall back to uncertainty selection
    if probs is None or lesionness is None or bin_id is None:
        print(f"⚠️ Adaptive selection: Missing required data (probs={probs is not None}, "
              f"lesionness={lesionness is not None}, bin_id={bin_id is not None}). "
              f"Falling back to uncertainty selection.")
        return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)

    # Convert to numpy if needed
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    if torch.is_tensor(lesionness):
        lesionness = lesionness.cpu().numpy()
    if torch.is_tensor(bin_id):
        bin_id = bin_id.cpu().numpy()

    # Extract data for available indices only
    N = len(available_indices)

    # Map available_indices to their positions in pool_indices for data extraction
    available_positions = [pool_indices.index(idx) for idx in available_indices]

    # Extract probs and lesionness for available indices only
    available_probs = probs[available_positions]  # [N, H, W]
    available_lesionness = lesionness[available_positions]  # [N, H, W]

    # Verify shapes
    if available_probs.shape[0] != N or available_lesionness.shape[0] != N:
        raise ValueError(f"Mismatch after extraction: available_probs={available_probs.shape[0]}, "
                         f"available_lesionness={available_lesionness.shape[0]}, available_indices={N}")

    # Calculate pixel entropy H(p) = -(p*log(p) + (1-p)*log(1-p))
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    p_safe = np.clip(available_probs, eps, 1 - eps)
    pixel_entropy = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))

    # Calculate weights w(i) = M(i) * H(p(i))
    # If lesionness is all ones (none type), use only pixel entropy
    if np.allclose(available_lesionness, 1.0):
        weights = pixel_entropy
    else:
        # Apply much stronger lesionness weighting for better lesion detection
        # Use exponential weighting to emphasize high lesionness regions
        lesionness_power = 3.0  # Increased from 1.5 to 3.0
        weights = (available_lesionness ** lesionness_power) * pixel_entropy

    # Calculate spatial histogram h_k(x) for each sample
    K = int(np.max(bin_id)) + 1  # Number of bins (should be 6 for 3x2 grid)
    h_histograms = np.zeros((N, K))

    for i, idx in enumerate(available_indices):
        for k in range(K):
            # Sum weights for pixels in bin k
            mask = (bin_id == k)
            h_histograms[i, k] = np.sum(weights[i][mask])

    # Greedy selection with spatial coverage
    selected = []
    H_k = np.zeros(K)  # Cumulative coverage for each bin

    # Define concave function phi(u) = log(1 + u) for better spatial coverage
    def phi(u):
        return np.log(1 + np.maximum(u, 0))  # Ensure non-negative

    for _ in range(min(num_samples, len(available_indices))):
        best_score = -np.inf
        best_idx = None

        for i, idx in enumerate(available_indices):
            if idx in selected:
                continue

            # Calculate incremental spatial gain
            delta_spatial = 0.0
            for k in range(K):
                delta_spatial += phi(H_k[k] + h_histograms[i, k]) - phi(H_k[k])

            # Calculate final score: U(x) + lambda1 * Delta_spatial(x|S)
            # Fix: Use correct uncertainty index mapping
            uncertainty_idx = pool_indices.index(idx)
            uncertainty_score = uncertainties[uncertainty_idx]

            # Optimized hyperparameters for better performance
            # Increase spatial coverage weight more aggressively
            enhanced_lambda1 = lambda1 * 5.0  # 5x spatial coverage weight
            # Add uncertainty scaling for better balance
            scaled_uncertainty = uncertainty_score * 2.0  # 2x uncertainty weight

            score = scaled_uncertainty + enhanced_lambda1 * delta_spatial

            if score > best_score:
                best_score = score
                best_idx = (i, idx)

        if best_idx is not None:
            i, idx = best_idx
            selected.append(idx)
            # Update cumulative coverage
            H_k += h_histograms[i]

    return selected


def select_samples_adaptive_improved(pool_indices, uncertainties, num_samples, selected_indices,
                                     probs=None, lesionness=None, bin_id=None, lambda1=1.0, seed=None):
    """
    Improved adaptive selection strategy with better hyperparameters and fallback mechanisms.

    This version includes:
    - Better uncertainty-spatial balance
    - Adaptive lambda1 based on round progress
    - Minimum sample guarantee
    - Enhanced lesionness weighting
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round (no uncertainties), use random selection
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)

    # If required tensors are not provided, fall back to uncertainty selection
    if probs is None or lesionness is None or bin_id is None:
        print(f"⚠️ Improved adaptive selection: Missing required data. "
              f"Falling back to uncertainty selection.")
        return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)

    # Convert to numpy if needed
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    if torch.is_tensor(lesionness):
        lesionness = lesionness.cpu().numpy()
    if torch.is_tensor(bin_id):
        bin_id = bin_id.cpu().numpy()

    # Extract data for available indices only
    N = len(available_indices)
    available_positions = [pool_indices.index(idx) for idx in available_indices]
    available_probs = probs[available_positions]
    available_lesionness = lesionness[available_positions]

    # Calculate pixel entropy with better numerical stability
    eps = 1e-8
    p_safe = np.clip(available_probs, eps, 1 - eps)
    pixel_entropy = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))

    # Enhanced lesionness weighting with adaptive power
    if np.allclose(available_lesionness, 1.0):
        weights = pixel_entropy
    else:
        # Adaptive lesionness power based on data distribution
        lesionness_std = np.std(available_lesionness)
        lesionness_power = 2.0 + min(lesionness_std * 10, 3.0)  # Adaptive power 2.0-5.0
        weights = (available_lesionness ** lesionness_power) * pixel_entropy

    # Calculate spatial histogram
    K = int(np.max(bin_id)) + 1
    h_histograms = np.zeros((N, K))
    for i, idx in enumerate(available_indices):
        for k in range(K):
            mask = (bin_id == k)
            h_histograms[i, k] = np.sum(weights[i][mask])

    # Adaptive lambda1 based on selection progress
    progress = len(selected_indices) / (len(selected_indices) + num_samples)
    adaptive_lambda1 = lambda1 * (1.0 + progress * 4.0)  # Increase spatial weight over time

    # Greedy selection with improved scoring
    selected = []
    H_k = np.zeros(K)

    def phi(u):
        return np.log(1 + np.maximum(u, 0))

    for _ in range(min(num_samples, len(available_indices))):
        best_score = -np.inf
        best_idx = None

        for i, idx in enumerate(available_indices):
            if idx in selected:
                continue

            # Calculate spatial gain
            delta_spatial = 0.0
            for k in range(K):
                delta_spatial += phi(H_k[k] + h_histograms[i, k]) - phi(H_k[k])

            # Enhanced scoring with better balance
            uncertainty_idx = pool_indices.index(idx)
            uncertainty_score = uncertainties[uncertainty_idx]

            # Adaptive uncertainty scaling
            uncertainty_scale = 1.0 + progress * 2.0  # Increase uncertainty weight over time
            scaled_uncertainty = uncertainty_score * uncertainty_scale

            # Final score with adaptive weights
            score = scaled_uncertainty + adaptive_lambda1 * delta_spatial

            if score > best_score:
                best_score = score
                best_idx = (i, idx)

        if best_idx is not None:
            i, idx = best_idx
            selected.append(idx)
            H_k += h_histograms[i]
        else:
            # Fallback: select remaining samples randomly if no good candidates
            remaining = [idx for idx in available_indices if idx not in selected]
            if remaining:
                selected.extend(remaining[:num_samples - len(selected)])
            break

    return selected


def select_samples_adaptive_hybrid(pool_indices, uncertainties, num_samples, selected_indices,
                                   probs=None, lesionness=None, bin_id=None, lambda1=1.0, seed=None):
    """
    Hybrid adaptive selection strategy combining adaptive_improved and uncertainty advantages.

    This strategy combines:
    1. adaptive_improved's spatial coverage and stability
    2. uncertainty's high performance and consistency
    3. Dynamic uncertainty-spatial balance based on progress
    4. Enhanced spatial coverage guarantee

    Key features:
    - Progressive uncertainty-spatial balance (early: uncertainty-focused, late: spatial-focused)
    - Enhanced spatial coverage guarantee (addresses Worst Bin Dice = 0.0000)
    - Adaptive lesionness weighting based on data distribution
    - Smooth parameter transitions for stability
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round (no uncertainties), use random selection
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)

    # If required tensors are not provided, fall back to uncertainty selection
    if probs is None or lesionness is None or bin_id is None:
        print(f"⚠️ Hybrid adaptive selection: Missing required data. "
              f"Falling back to uncertainty selection.")
        return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)

    # Convert to numpy if needed
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    if torch.is_tensor(lesionness):
        lesionness = lesionness.cpu().numpy()
    if torch.is_tensor(bin_id):
        bin_id = bin_id.cpu().numpy()

    # Extract data for available indices only
    N = len(available_indices)
    available_positions = [pool_indices.index(idx) for idx in available_indices]
    available_probs = probs[available_positions]
    available_lesionness = lesionness[available_positions]

    # Calculate pixel entropy with better numerical stability
    eps = 1e-8
    p_safe = np.clip(available_probs, eps, 1 - eps)
    pixel_entropy = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))

    # Hybrid lesionness weighting (adaptive based on data distribution)
    if np.allclose(available_lesionness, 1.0):
        weights = pixel_entropy
    else:
        lesionness_std = np.std(available_lesionness)
        # Adaptive range: 1.5 → 4.0 (balanced between improved and uncertainty)
        lesionness_power = 1.5 + min(lesionness_std * 10, 2.5)
        weights = (available_lesionness ** lesionness_power) * pixel_entropy

    # Calculate spatial histogram
    K = int(np.max(bin_id)) + 1
    h_histograms = np.zeros((N, K))
    for i, idx in enumerate(available_indices):
        for k in range(K):
            mask = (bin_id == k)
            h_histograms[i, k] = np.sum(weights[i][mask])

    # Hybrid parameter scheduling
    progress = len(selected_indices) / (len(selected_indices) + num_samples)

    # Progressive uncertainty-spatial balance
    # Early rounds: uncertainty-focused (like uncertainty strategy)
    # Late rounds: spatial-focused (like adaptive_improved)
    uncertainty_weight = 2.0 - progress * 1.5  # 2.0 → 0.5 (decreasing)
    spatial_weight = 0.5 + progress * 2.5     # 0.5 → 3.0 (increasing)

    # Adaptive lambda1 with smooth transition
    adaptive_lambda1 = lambda1 * spatial_weight

    # Enhanced uncertainty scaling with progress-based adaptation
    uncertainty_std = np.std(uncertainties)
    uncertainty_scale = uncertainty_weight + min(uncertainty_std * 2, 0.5)

    # Spatial coverage guarantee (stronger than adaptive_improved)
    min_samples_per_bin = max(2, num_samples // (K * 2))  # Stronger coverage requirement

    # Greedy selection with hybrid scoring
    selected = []
    H_k = np.zeros(K)

    def phi(u):
        return np.log(1 + np.maximum(u, 0))

    for _ in range(min(num_samples, len(available_indices))):
        best_score = -np.inf
        best_idx = None

        for i, idx in enumerate(available_indices):
            if idx in selected:
                continue

            # Calculate spatial gain
            delta_spatial = 0.0
            for k in range(K):
                delta_spatial += phi(H_k[k] + h_histograms[i, k]) - phi(H_k[k])

            # Uncertainty score with hybrid scaling
            uncertainty_idx = pool_indices.index(idx)
            uncertainty_score = uncertainties[uncertainty_idx]
            scaled_uncertainty = uncertainty_score * uncertainty_scale

            # Enhanced spatial coverage bonus (stronger than adaptive_improved)
            coverage_bonus = calculate_coverage_bonus(H_k, h_histograms[i], min_samples_per_bin, weight=4.0)

            # Hybrid score: uncertainty + spatial + coverage
            score = scaled_uncertainty + adaptive_lambda1 * delta_spatial + coverage_bonus

            if score > best_score:
                best_score = score
                best_idx = (i, idx)

        if best_idx is not None:
            i, idx = best_idx
            selected.append(idx)
            H_k += h_histograms[i]
        else:
            # Fallback: select remaining samples randomly if no good candidates
            remaining = [idx for idx in available_indices if idx not in selected]
            if remaining:
                selected.extend(remaining[:num_samples - len(selected)])
            break

    return selected


def calculate_diversity_gain(H_k, h_histogram):
    """Calculate diversity gain for spatial regions."""
    # Regions with lower coverage get higher diversity gain
    total_coverage = np.sum(H_k)
    if total_coverage == 0:
        return 1.0

    # Calculate coverage per region
    coverage_per_region = H_k / (total_coverage + 1e-8)

    # Diversity gain is higher for underrepresented regions
    diversity_gain = 0.0
    for k in range(len(h_histogram)):
        if h_histogram[k] > 0:  # This sample contributes to region k
            # Higher gain for underrepresented regions
            diversity_gain += h_histogram[k] * (1.0 - coverage_per_region[k])

    return diversity_gain


def calculate_coverage_bonus(H_k, h_histogram, min_samples_per_bin=1, weight=1.0):
    """Calculate bonus for samples that improve spatial coverage."""
    K = len(H_k)
    bonus = 0.0

    for k in range(K):
        current_coverage = H_k[k]
        if current_coverage < min_samples_per_bin and h_histogram[k] > 0:
            # Higher bonus for underrepresented regions
            coverage_ratio = current_coverage / (min_samples_per_bin + 1e-8)
            bonus += weight * h_histogram[k] * (1.0 - coverage_ratio)

    return bonus


def smooth_schedule(progress, base=1.0, max_mult=2.0):
    """Smooth parameter scheduling to prevent abrupt changes."""
    # S-curve scheduling for smooth transitions
    smooth_progress = 0.5 * (1 + np.tanh(3 * (progress - 0.5)))
    return base * (1 + smooth_progress * (max_mult - 1))


def select_samples_diversity(pool_indices, uncertainties, num_samples, selected_indices, areas, features=None, seed=None):
    """
    Diversity-based selection strategy using feature embedding diversity.

    Args:
        pool_indices: List of available sample indices
        uncertainties: List of uncertainty values for each sample (not used in pure diversity)
        num_samples: Number of samples to select
        selected_indices: List of already selected indices
        areas: List of spatial areas (not used in feature diversity)
        features: Feature embeddings [N, D] for diversity calculation
        seed: Random seed for reproducibility

    Returns:
        List of selected sample indices
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # Use specific seed for reproducibility with isolated random state
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # If no features provided, fall back to random selection
    if features is None:
        return rng.choice(available_indices, size=num_samples, replace=False).tolist()

    # Convert to numpy if needed
    if torch.is_tensor(features):
        features = features.cpu().numpy()

    # Map available_indices to their positions in pool_indices for data extraction
    available_positions = [pool_indices.index(idx) for idx in available_indices]

    # Get features for available indices
    available_features = features[available_positions]  # [N, D]

    # Greedy selection to maximize diversity
    selected = []
    selected_features = []

    # Start with the first sample (randomly chosen)
    first_idx = rng.choice(len(available_indices))
    selected.append(available_indices[first_idx])
    selected_features.append(available_features[first_idx])

    # Remove the selected sample from available
    remaining_indices = [i for i in range(len(available_indices)) if i != first_idx]
    remaining_features = available_features[remaining_indices]

    # Greedily select remaining samples to maximize diversity
    for _ in range(min(num_samples - 1, len(remaining_indices))):
        if len(remaining_indices) == 0:
            break

        # Calculate minimum distance to already selected samples for each remaining sample
        min_distances = []
        for i, feature in enumerate(remaining_features):
            # Calculate distances to all selected features
            distances = [np.linalg.norm(feature - sel_feat) for sel_feat in selected_features]
            min_distances.append(min(distances))

        # Select the sample with maximum minimum distance (most diverse)
        best_idx = np.argmax(min_distances)
        selected_idx = remaining_indices[best_idx]

        selected.append(available_indices[selected_idx])
        selected_features.append(remaining_features[best_idx])

        # Remove selected sample from remaining
        remaining_indices = [i for i in remaining_indices if i != selected_idx]
        remaining_features = np.delete(remaining_features, best_idx, axis=0)

    return selected


def select_samples_diversity_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, areas, features=None, seed=None):
    """
    Diversity-uncertainty combined selection strategy.

    Args:
        pool_indices: List of available sample indices
        uncertainties: List of uncertainty values for each sample
        num_samples: Number of samples to select
        selected_indices: List of already selected indices
        areas: List of spatial areas (not used in feature diversity)
        features: Feature embeddings [N, D] for diversity calculation
        seed: Random seed for reproducibility

    Returns:
        List of selected sample indices
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round (no uncertainties), use diversity selection
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_diversity(pool_indices, uncertainties, num_samples, selected_indices, areas, features, seed)

    # Use specific seed for reproducibility with isolated random state
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # If no features provided, fall back to uncertainty selection
    if features is None:
        return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)

    # Convert to numpy if needed
    if torch.is_tensor(features):
        features = features.cpu().numpy()

    # Map available_indices to their positions in pool_indices for data extraction
    available_positions = [pool_indices.index(idx) for idx in available_indices]

    # Get features and uncertainties for available indices
    available_features = features[available_positions]  # [N, D]
    available_uncertainties = [uncertainties[idx] for idx in available_indices]

    # Greedy selection balancing diversity and uncertainty
    selected = []
    selected_features = []

    # Start with the most uncertain sample
    first_idx = np.argmax(available_uncertainties)
    selected.append(available_indices[first_idx])
    selected_features.append(available_features[first_idx])

    # Remove the selected sample from available
    remaining_indices = [i for i in range(len(available_indices)) if i != first_idx]
    remaining_features = available_features[remaining_indices]
    remaining_uncertainties = [available_uncertainties[i] for i in remaining_indices]

    # Greedily select remaining samples balancing diversity and uncertainty
    for _ in range(min(num_samples - 1, len(remaining_indices))):
        if len(remaining_indices) == 0:
            break

        # Calculate combined score: uncertainty + diversity
        scores = []
        for i, (feature, uncertainty) in enumerate(zip(remaining_features, remaining_uncertainties)):
            # Calculate minimum distance to already selected samples
            distances = [np.linalg.norm(feature - sel_feat) for sel_feat in selected_features]
            min_distance = min(distances) if distances else 0

            # Combined score: uncertainty + diversity (normalized)
            diversity_score = min_distance / (np.linalg.norm(feature) + 1e-8)  # Normalize by feature norm
            combined_score = uncertainty + diversity_score
            scores.append(combined_score)

        # Select the sample with maximum combined score
        best_idx = np.argmax(scores)
        selected_idx = remaining_indices[best_idx]

        selected.append(available_indices[selected_idx])
        selected_features.append(remaining_features[best_idx])

        # Remove selected sample from remaining
        remaining_indices = [i for i in remaining_indices if i != selected_idx]
        remaining_features = np.delete(remaining_features, best_idx, axis=0)
        remaining_uncertainties = [remaining_uncertainties[i]
                                   for i in range(len(remaining_uncertainties)) if i != best_idx]

    return selected


# ============================================================================
# ENHANCED HYBRID STRATEGIES
# ============================================================================

def select_samples_multi_scale_hybrid(pool_indices, uncertainties, num_samples, selected_indices,
                                      probs=None, lesionness=None, bin_id=None, lambda1=1.0,
                                      grid_scales=[(3, 3), (5, 5), (7, 7)], seed=None):
    """
    Multi-scale hybrid selection strategy using multiple grid resolutions.

    This strategy:
    1. Uses multiple grid scales (3x3, 5x5, 7x7) for comprehensive spatial analysis
    2. Combines coarse and fine-grained spatial information
    3. Adapts weights based on data distribution across scales
    4. Provides robust spatial coverage at multiple resolutions
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round, use random selection
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)

    # If required tensors are not provided, fall back to uncertainty selection
    if probs is None or lesionness is None or bin_id is None:
        print(f"⚠️ Multi-scale hybrid selection: Missing required data. "
              f"Falling back to uncertainty selection.")
        return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)

    # Convert to numpy if needed
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    if torch.is_tensor(lesionness):
        lesionness = lesionness.cpu().numpy()
    if torch.is_tensor(bin_id):
        bin_id = bin_id.cpu().numpy()

    # Extract data for available indices
    N = len(available_indices)
    available_positions = [pool_indices.index(idx) for idx in available_indices]
    available_probs = probs[available_positions]
    available_lesionness = lesionness[available_positions]

    # Calculate pixel entropy
    eps = 1e-8
    p_safe = np.clip(available_probs, eps, 1 - eps)
    pixel_entropy = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))

    # Multi-scale spatial analysis
    scale_weights = []
    scale_histograms = []

    for scale_idx, (grid_width, grid_height) in enumerate(grid_scales):
        # Create spatial bins for this scale
        scale_bin_id = create_spatial_bins(
            height=bin_id.shape[0], width=bin_id.shape[1],
            grid_width=grid_width, grid_height=grid_height
        )

        # Calculate weights for this scale
        if np.allclose(available_lesionness, 1.0):
            weights = pixel_entropy
        else:
            lesionness_std = np.std(available_lesionness)
            lesionness_power = 1.5 + min(lesionness_std * 10, 2.5)
            weights = (available_lesionness ** lesionness_power) * pixel_entropy

        # Calculate spatial histogram for this scale
        K = int(np.max(scale_bin_id)) + 1
        h_histograms = np.zeros((N, K))
        for i, idx in enumerate(available_indices):
            for k in range(K):
                mask = (scale_bin_id == k)
                h_histograms[i, k] = np.sum(weights[i][mask])

        # Adaptive weight for this scale based on data distribution
        scale_weight = calculate_scale_weight(h_histograms, grid_width * grid_height)
        scale_weights.append(scale_weight)
        scale_histograms.append(h_histograms)

    # Combine multi-scale information
    combined_histograms = np.zeros_like(scale_histograms[0])
    total_weight = sum(scale_weights)

    for hist, weight in zip(scale_histograms, scale_weights):
        # Normalize and combine
        normalized_weight = weight / total_weight
        combined_histograms += hist * normalized_weight

    # Progressive uncertainty-spatial balance
    progress = len(selected_indices) / (len(selected_indices) + num_samples)
    uncertainty_weight = 2.0 - progress * 1.5
    spatial_weight = 0.5 + progress * 2.5

    # Multi-scale adaptive lambda
    adaptive_lambda1 = lambda1 * spatial_weight * (1 + len(grid_scales) * 0.1)

    # Enhanced uncertainty scaling
    uncertainty_std = np.std(uncertainties)
    uncertainty_scale = uncertainty_weight + min(uncertainty_std * 2, 0.5)

    # Multi-scale spatial coverage guarantee
    K = combined_histograms.shape[1]
    min_samples_per_bin = max(2, num_samples // (K * 2))

    # Greedy selection with multi-scale scoring
    selected = []
    H_k = np.zeros(K)

    def phi(u):
        return np.log(1 + np.maximum(u, 0))

    for _ in range(min(num_samples, len(available_indices))):
        best_score = -np.inf
        best_idx = None

        for i, idx in enumerate(available_indices):
            if idx in selected:
                continue

            # Calculate multi-scale spatial gain
            delta_spatial = 0.0
            for k in range(K):
                delta_spatial += phi(H_k[k] + combined_histograms[i, k]) - phi(H_k[k])

            # Uncertainty score
            uncertainty_idx = pool_indices.index(idx)
            uncertainty_score = uncertainties[uncertainty_idx]
            scaled_uncertainty = uncertainty_score * uncertainty_scale

            # Multi-scale coverage bonus
            coverage_bonus = calculate_multi_scale_coverage_bonus(
                H_k, combined_histograms[i], min_samples_per_bin,
                scale_weights, weight=4.0
            )

            # Multi-scale hybrid score
            score = scaled_uncertainty + adaptive_lambda1 * delta_spatial + coverage_bonus

            if score > best_score:
                best_score = score
                best_idx = (i, idx)

        if best_idx is not None:
            i, idx = best_idx
            selected.append(idx)
            H_k += combined_histograms[i]
        else:
            # Fallback
            remaining = [idx for idx in available_indices if idx not in selected]
            if remaining:
                selected.extend(remaining[:num_samples - len(selected)])
            break

    return selected


def calculate_scale_weight(histograms, num_bins):
    """Calculate adaptive weight for a scale based on data distribution."""
    # Higher weight for scales with more balanced distribution
    bin_counts = np.sum(histograms, axis=0)
    if np.sum(bin_counts) == 0:
        return 0.0

    # Calculate distribution balance (lower std = more balanced)
    distribution_std = np.std(bin_counts)
    max_possible_std = np.std([np.sum(histograms)] * num_bins)

    if max_possible_std == 0:
        return 1.0

    balance_ratio = 1.0 - (distribution_std / max_possible_std)
    return max(0.1, balance_ratio)  # Minimum weight of 0.1


def calculate_multi_scale_coverage_bonus(H_k, h_histogram, min_samples_per_bin, scale_weights, weight=1.0):
    """Calculate coverage bonus considering multiple scales."""
    K = len(H_k)
    bonus = 0.0

    for k in range(K):
        current_coverage = H_k[k]
        if current_coverage < min_samples_per_bin and h_histogram[k] > 0:
            coverage_ratio = current_coverage / (min_samples_per_bin + 1e-8)
            # Weight by scale importance
            scale_bonus = weight * h_histogram[k] * (1.0 - coverage_ratio)
            bonus += scale_bonus

    return bonus


def select_samples_adaptive_performance_monitoring(pool_indices, uncertainties, num_samples, selected_indices,
                                                   probs=None, lesionness=None, bin_id=None, lambda1=1.0,
                                                   performance_history=None, seed=None):
    """
    Adaptive selection with real-time performance monitoring.

    This strategy:
    1. Monitors performance of each spatial area in real-time
    2. Adjusts selection weights based on area performance
    3. Prioritizes underperforming areas
    4. Maintains balance between exploration and exploitation
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round, use random selection
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)

    if probs is None or lesionness is None or bin_id is None:
        print(f"⚠️ Performance monitoring selection: Missing required data. "
              f"Falling back to uncertainty selection.")
        return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)

    # Convert to numpy if needed
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    if torch.is_tensor(lesionness):
        lesionness = lesionness.cpu().numpy()
    if torch.is_tensor(bin_id):
        bin_id = bin_id.cpu().numpy()

    # Extract data for available indices
    N = len(available_indices)
    available_positions = [pool_indices.index(idx) for idx in available_indices]
    available_probs = probs[available_positions]
    available_lesionness = lesionness[available_positions]

    # Calculate pixel entropy
    eps = 1e-8
    p_safe = np.clip(available_probs, eps, 1 - eps)
    pixel_entropy = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))

    # Calculate lesionness weights
    if np.allclose(available_lesionness, 1.0):
        weights = pixel_entropy
    else:
        lesionness_std = np.std(available_lesionness)
        lesionness_power = 1.5 + min(lesionness_std * 10, 2.5)
        weights = (available_lesionness ** lesionness_power) * pixel_entropy

    # Calculate spatial histogram
    K = int(np.max(bin_id)) + 1
    h_histograms = np.zeros((N, K))
    for i, idx in enumerate(available_indices):
        for k in range(K):
            mask = (bin_id == k)
            h_histograms[i, k] = np.sum(weights[i][mask])

    # Performance-based adaptive weighting
    if performance_history is not None:
        area_weights = calculate_performance_based_weights(performance_history, K)
    else:
        area_weights = np.ones(K)  # Equal weights if no history

    # Progressive uncertainty-spatial balance with performance adjustment
    progress = len(selected_indices) / (len(selected_indices) + num_samples)
    uncertainty_weight = 2.0 - progress * 1.5
    spatial_weight = 0.5 + progress * 2.5

    # Performance-adjusted lambda
    performance_factor = np.mean(area_weights) if len(area_weights) > 0 else 1.0
    adaptive_lambda1 = lambda1 * spatial_weight * performance_factor

    # Enhanced uncertainty scaling
    uncertainty_std = np.std(uncertainties)
    uncertainty_scale = uncertainty_weight + min(uncertainty_std * 2, 0.5)

    # Performance-aware spatial coverage
    min_samples_per_bin = max(2, num_samples // (K * 2))

    # Greedy selection with performance monitoring
    selected = []
    H_k = np.zeros(K)

    def phi(u):
        return np.log(1 + np.maximum(u, 0))

    for _ in range(min(num_samples, len(available_indices))):
        best_score = -np.inf
        best_idx = None

        for i, idx in enumerate(available_indices):
            if idx in selected:
                continue

            # Calculate performance-weighted spatial gain
            delta_spatial = 0.0
            for k in range(K):
                area_weight = area_weights[k] if k < len(area_weights) else 1.0
                delta_spatial += area_weight * (phi(H_k[k] + h_histograms[i, k]) - phi(H_k[k]))

            # Uncertainty score
            uncertainty_idx = pool_indices.index(idx)
            uncertainty_score = uncertainties[uncertainty_idx]
            scaled_uncertainty = uncertainty_score * uncertainty_scale

            # Performance-aware coverage bonus
            coverage_bonus = calculate_performance_coverage_bonus(
                H_k, h_histograms[i], min_samples_per_bin, area_weights, weight=4.0
            )

            # Performance-monitored hybrid score
            score = scaled_uncertainty + adaptive_lambda1 * delta_spatial + coverage_bonus

            if score > best_score:
                best_score = score
                best_idx = (i, idx)

        if best_idx is not None:
            i, idx = best_idx
            selected.append(idx)
            H_k += h_histograms[i]
        else:
            # Fallback
            remaining = [idx for idx in available_indices if idx not in selected]
            if remaining:
                selected.extend(remaining[:num_samples - len(selected)])
            break

    return selected


def calculate_performance_based_weights(performance_history, num_areas):
    """Calculate area weights based on performance history."""
    weights = np.ones(num_areas)

    if not performance_history or 'area_performance' not in performance_history:
        return weights

    area_performance = performance_history['area_performance']

    for area_idx in range(num_areas):
        if area_idx in area_performance:
            # Lower performance = higher weight (prioritize underperforming areas)
            performance = area_performance[area_idx]
            if performance < 0.3:  # Severely underperforming
                weights[area_idx] = 3.0
            elif performance < 0.5:  # Underperforming
                weights[area_idx] = 2.0
            elif performance < 0.7:  # Moderate performance
                weights[area_idx] = 1.5
            else:  # Good performance
                weights[area_idx] = 1.0 + (1.0 - performance) * 0.5  # Moderate boost

    return weights


def calculate_performance_coverage_bonus(H_k, h_histogram, min_samples_per_bin, area_weights, weight=1.0):
    """Calculate coverage bonus considering area performance."""
    K = len(H_k)
    bonus = 0.0

    for k in range(K):
        current_coverage = H_k[k]
        if current_coverage < min_samples_per_bin and h_histogram[k] > 0:
            coverage_ratio = current_coverage / (min_samples_per_bin + 1e-8)
            area_weight = area_weights[k] if k < len(area_weights) else 1.0
            bonus += weight * h_histogram[k] * (1.0 - coverage_ratio) * area_weight

    return bonus


# =============================================================================
# SPARCL: Spatial Active Learning with Entropy-Gated Adaptive Weighting
# =============================================================================

def select_samples_sparcl(pool_indices, uncertainties, num_samples, selected_indices,
                          probs=None, lesionness=None, bin_id=None, lambda_max=1.0, seed=None):
    """
    SPARCL: Spatial Active Learning with Entropy-Gated Adaptive Weighting.

    This implementation matches the paper description exactly:

    1. Per-image spatial histogram:
       h_k(x) = Σ_{i∈Ω_k} w(i), where w(i) = M(i) * H_pred(i)

    2. Entropy-gated adaptive weighting:
       Q_k = Σ_{x∈U_t} h_k(x)  (pool-wide mass per bin)
       q_k = Q_k / Σ_j Q_j     (normalized distribution)
       H_pool = -Σ_k q_k * log(q_k)  (pool entropy)
       λ^(t) = λ_max * H_pool / log(K)

    3. Diminishing return function: ψ(z) = √z

    4. Acquisition objective:
       Δ(x|S) = U(x) + λ^(t) * Σ_k [ψ(H_k(S) + h_k(x)) - ψ(H_k(S))]

    Args:
        pool_indices: List of available sample indices
        uncertainties: List of uncertainty values for each sample (U(x))
        num_samples: Number of samples to select
        selected_indices: List of already selected indices
        probs: Tensor [N, H, W] - pixel prediction probabilities p(i)
        lesionness: Tensor [N, H, W] - lesionness values M(i)
        bin_id: Tensor [H, W] - spatial bin assignments (0..K-1)
        lambda_max: Maximum weight for spatial coverage term (λ_max)
        seed: Random seed for reproducibility

    Returns:
        List of selected sample indices
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round (no uncertainties), use random selection
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)

    # If required tensors are not provided, fall back to uncertainty selection
    if probs is None or lesionness is None or bin_id is None:
        print(f"⚠️ SPARCL selection: Missing required data (probs={probs is not None}, "
              f"lesionness={lesionness is not None}, bin_id={bin_id is not None}). "
              f"Falling back to uncertainty selection.")
        return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)

    # Convert to numpy if needed
    # Convert to numpy arrays if needed
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    elif isinstance(probs, list):
        probs = np.array(probs)
    if torch.is_tensor(lesionness):
        lesionness = lesionness.cpu().numpy()
    elif isinstance(lesionness, list):
        lesionness = np.array(lesionness)
    if torch.is_tensor(bin_id):
        bin_id = bin_id.cpu().numpy()
    elif isinstance(bin_id, list):
        bin_id = np.array(bin_id)

    # Ensure probs and lesionness are numpy arrays with correct shape
    if probs is None or len(probs) == 0:
        raise ValueError("probs is required for SPARCL but is None or empty")
    if lesionness is None or len(lesionness) == 0:
        raise ValueError("lesionness is required for SPARCL but is None or empty")

    # Ensure probs and lesionness are 3D arrays [N, H, W]
    if probs.ndim == 2:
        probs = probs[np.newaxis, :, :]  # Add batch dimension
    if lesionness.ndim == 2:
        lesionness = lesionness[np.newaxis, :, :]  # Add batch dimension

    # ==========================================================================
    # Optimized index mapping for O(1) lookup
    # ==========================================================================
    N = len(available_indices)
    pool_idx_to_pos = {idx: pos for pos, idx in enumerate(pool_indices)}
    available_positions = np.array([pool_idx_to_pos[idx] for idx in available_indices])

    available_probs = probs[available_positions]  # [N, H, W]
    available_lesionness = lesionness[available_positions]  # [N, H, W]

    # Pre-compute uncertainty scores for O(1) lookup
    available_uncertainties = np.array([uncertainties[pool_idx_to_pos[idx]] for idx in available_indices])

    # ==========================================================================
    # Step 1: Calculate pixel entropy H_pred(i) = -(p*log(p) + (1-p)*log(1-p))
    # ==========================================================================
    eps = 1e-8
    p_safe = np.clip(available_probs, eps, 1 - eps)
    pixel_entropy = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))

    # ==========================================================================
    # Step 2: Calculate weights w(i) = M(i) * H_pred(i)
    # ==========================================================================
    if np.allclose(available_lesionness, 1.0):
        weights = pixel_entropy
    else:
        weights = available_lesionness * pixel_entropy

    # ==========================================================================
    # Step 3: Vectorized spatial histogram using bincount
    # ==========================================================================
    K = int(np.max(bin_id)) + 1  # Number of bins
    bin_id_flat = bin_id.flatten()
    weights_flat = weights.reshape(N, -1)  # [N, H*W]

    h_histograms = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        h_histograms[i] = np.bincount(bin_id_flat, weights=weights_flat[i], minlength=K)

    # ==========================================================================
    # Step 4: Entropy-gated adaptive weighting
    # Q_k = Σ_{x∈U_t} h_k(x)  (pool-wide mass per bin)
    # q_k = Q_k / Σ_j Q_j     (normalized distribution)
    # H_pool = -Σ_k q_k * log(q_k)
    # λ^(t) = λ_max * H_pool / log(K)
    # ==========================================================================
    Q_k = np.sum(h_histograms, axis=0)  # [K] - pool-wide mass per bin
    Q_total = np.sum(Q_k) + eps
    q_k = Q_k / Q_total  # Normalized distribution

    # Pool entropy: H_pool = -Σ_k q_k * log(q_k)
    q_k_safe = np.clip(q_k, eps, 1.0)
    H_pool = -np.sum(q_k * np.log(q_k_safe))

    # Adaptive lambda: λ^(t) = λ_max * H_pool / log(K)
    log_K = np.log(K) if K > 1 else 1.0
    adaptive_lambda = lambda_max * (H_pool / log_K)

    # ==========================================================================
    # Step 5: Vectorized greedy selection
    # ψ(z) = √z
    # Δ(x|S) = U(x) + λ^(t) * Σ_k [ψ(H_k(S) + h_k(x)) - ψ(H_k(S))]
    # ==========================================================================
    selected = []
    selected_mask = np.zeros(N, dtype=bool)
    H_k = np.zeros(K, dtype=np.float32)  # Cumulative coverage
    sqrt_H_k = np.zeros(K, dtype=np.float32)

    for _ in range(min(num_samples, N)):
        # Vectorized: compute delta_spatial for all candidates at once
        new_H_k = H_k + h_histograms  # [N, K]
        sqrt_new_H_k = np.sqrt(np.maximum(new_H_k, 0))
        delta_spatial = np.sum(sqrt_new_H_k - sqrt_H_k, axis=1)  # [N]

        # Compute scores: U(x) + λ * Δ_spatial(x|S)
        scores = available_uncertainties + adaptive_lambda * delta_spatial

        # Mask out already selected
        scores[selected_mask] = -np.inf

        # Find best candidate
        best_local_idx = np.argmax(scores)
        if scores[best_local_idx] == -np.inf:
            break

        # Add to selected
        selected.append(available_indices[best_local_idx])
        selected_mask[best_local_idx] = True

        # Update cumulative coverage
        H_k += h_histograms[best_local_idx]
        sqrt_H_k = np.sqrt(np.maximum(H_k, 0))

    return selected


def _select_samples_sparcl_with_lambda(pool_indices, uncertainties, num_samples, selected_indices,
                                       probs, lesionness, bin_id, fixed_lambda, seed=None):
    """
    SPARCL ablation: identical to select_samples_sparcl but with a fixed lambda
    instead of the entropy-gated adaptive lambda. Used for component-level ablation
    in the rebuttal (w/o entropy gating; fixed lambda variants).
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)
    if probs is None or lesionness is None or bin_id is None:
        return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)

    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    elif isinstance(probs, list):
        probs = np.array(probs)
    if torch.is_tensor(lesionness):
        lesionness = lesionness.cpu().numpy()
    elif isinstance(lesionness, list):
        lesionness = np.array(lesionness)
    if torch.is_tensor(bin_id):
        bin_id = bin_id.cpu().numpy()
    elif isinstance(bin_id, list):
        bin_id = np.array(bin_id)

    if probs.ndim == 2:
        probs = probs[np.newaxis, :, :]
    if lesionness.ndim == 2:
        lesionness = lesionness[np.newaxis, :, :]

    N = len(available_indices)
    pool_idx_to_pos = {idx: pos for pos, idx in enumerate(pool_indices)}
    available_positions = np.array([pool_idx_to_pos[idx] for idx in available_indices])

    available_probs = probs[available_positions]
    available_lesionness = lesionness[available_positions]
    available_uncertainties = np.array([uncertainties[pool_idx_to_pos[idx]] for idx in available_indices])

    eps = 1e-8
    p_safe = np.clip(available_probs, eps, 1 - eps)
    pixel_entropy = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))
    if np.allclose(available_lesionness, 1.0):
        weights = pixel_entropy
    else:
        weights = available_lesionness * pixel_entropy

    K = int(np.max(bin_id)) + 1
    bin_id_flat = bin_id.flatten()
    weights_flat = weights.reshape(N, -1)

    h_histograms = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        h_histograms[i] = np.bincount(bin_id_flat, weights=weights_flat[i], minlength=K)

    selected = []
    selected_mask = np.zeros(N, dtype=bool)
    H_k = np.zeros(K, dtype=np.float32)
    sqrt_H_k = np.zeros(K, dtype=np.float32)

    for _ in range(min(num_samples, N)):
        new_H_k = H_k + h_histograms
        sqrt_new_H_k = np.sqrt(np.maximum(new_H_k, 0))
        delta_spatial = np.sum(sqrt_new_H_k - sqrt_H_k, axis=1)

        scores = available_uncertainties + fixed_lambda * delta_spatial
        scores[selected_mask] = -np.inf

        best_local_idx = np.argmax(scores)
        if scores[best_local_idx] == -np.inf:
            break

        selected.append(available_indices[best_local_idx])
        selected_mask[best_local_idx] = True
        H_k += h_histograms[best_local_idx]
        sqrt_H_k = np.sqrt(np.maximum(H_k, 0))

    return selected


def select_samples_sparcl_no_gating(pool_indices, uncertainties, num_samples, selected_indices,
                                    probs=None, lesionness=None, bin_id=None, lambda_max=1.0, seed=None):
    """SPARCL ablation: no entropy gating. Uses lambda^(t) = lambda_max constantly."""
    return _select_samples_sparcl_with_lambda(
        pool_indices, uncertainties, num_samples, selected_indices,
        probs, lesionness, bin_id, fixed_lambda=lambda_max, seed=seed)


def select_samples_sparcl_fixed_lambda(pool_indices, uncertainties, num_samples, selected_indices,
                                       probs=None, lesionness=None, bin_id=None, lambda_max=0.5, seed=None):
    """SPARCL ablation: fixed lambda=0.5 (or lambda_max). No adaptive entropy gating."""
    return _select_samples_sparcl_with_lambda(
        pool_indices, uncertainties, num_samples, selected_indices,
        probs, lesionness, bin_id, fixed_lambda=lambda_max, seed=seed)


def select_samples_sparcl_no_coverage(pool_indices, uncertainties, num_samples, selected_indices,
                                      probs=None, lesionness=None, bin_id=None, lambda_max=1.0, seed=None):
    """SPARCL ablation: no spatial coverage term. Equivalent to pure uncertainty selection."""
    return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)


def select_samples_sparcl_prefiltering(pool_indices, uncertainties, num_samples, selected_indices,
                                       probs=None, lesionness=None, bin_id=None, lambda_max=1.0, seed=None,
                                       prefilter_ratio=5):
    """
    SPARCL with Prefiltering: Optimized version for faster execution.

    Two-stage procedure (from paper):
    1. Prefilter top-r candidates by U(x) where r = prefilter_ratio * num_samples
    2. Apply lazy greedy on the reduced set

    This version is much faster than the basic SPARCL by:
    - Processing only top-r candidates instead of all available samples
    - Using bincount for fast histogram calculation
    - Vectorized operations for greedy selection

    Args:
        pool_indices: List of available sample indices
        uncertainties: List of uncertainty values for each sample (U(x))
        num_samples: Number of samples to select
        selected_indices: List of already selected indices
        probs: Tensor [N, H, W] - pixel prediction probabilities p(i)
        lesionness: Tensor [N, H, W] - lesionness values M(i)
        bin_id: Tensor [H, W] - spatial bin assignments (0..K-1)
        lambda_max: Maximum weight for spatial coverage term (λ_max)
        seed: Random seed for reproducibility
        prefilter_ratio: Ratio for prefiltering (r = prefilter_ratio * num_samples)

    Returns:
        List of selected sample indices
    """
    # Use set for O(1) lookup
    selected_set = set(selected_indices)
    available_indices = [idx for idx in pool_indices if idx not in selected_set]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round (no uncertainties), use random selection
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)

    # If required tensors are not provided, fall back to uncertainty selection
    if probs is None or lesionness is None or bin_id is None:
        print(f"⚠️ SPARCL-Prefiltering selection: Missing required data (probs={probs is not None}, "
              f"lesionness={lesionness is not None}, bin_id={bin_id is not None}). "
              f"Falling back to uncertainty selection.")
        return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)

    # Convert to numpy if needed
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    if torch.is_tensor(lesionness):
        lesionness = lesionness.cpu().numpy()
    if torch.is_tensor(bin_id):
        bin_id = bin_id.cpu().numpy()

    # ==========================================================================
    # Stage 1: Prefilter top-r candidates by uncertainty
    # ==========================================================================
    pool_idx_to_pos = {idx: pos for pos, idx in enumerate(pool_indices)}

    # Get uncertainty scores for all available indices
    all_uncertainties = np.array([uncertainties[pool_idx_to_pos[idx]] for idx in available_indices])

    # Prefilter: select top-r by uncertainty
    r = min(prefilter_ratio * num_samples, len(available_indices))
    top_r_local_indices = np.argsort(all_uncertainties)[-r:][::-1]  # Descending order

    # Create filtered subset
    filtered_indices = [available_indices[i] for i in top_r_local_indices]
    filtered_positions = np.array([pool_idx_to_pos[idx] for idx in filtered_indices])

    N = len(filtered_indices)
    filtered_probs = probs[filtered_positions]  # [N, H, W]
    filtered_lesionness = lesionness[filtered_positions]  # [N, H, W]
    filtered_uncertainties = all_uncertainties[top_r_local_indices]

    # ==========================================================================
    # Step 1: Calculate pixel entropy H_pred(i) = -(p*log(p) + (1-p)*log(1-p))
    # ==========================================================================
    eps = 1e-8
    p_safe = np.clip(filtered_probs, eps, 1 - eps)
    pixel_entropy = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))

    # ==========================================================================
    # Step 2: Calculate weights w(i) = M(i) * H_pred(i)
    # ==========================================================================
    if np.allclose(filtered_lesionness, 1.0):
        weights = pixel_entropy
    else:
        weights = filtered_lesionness * pixel_entropy

    # ==========================================================================
    # Step 3: Vectorized spatial histogram calculation using bincount
    # ==========================================================================
    K = int(np.max(bin_id)) + 1  # Number of bins

    # Flatten for efficient bincount-based histogram
    bin_id_flat = bin_id.flatten()
    weights_flat = weights.reshape(N, -1)  # [N, H*W]

    # Use bincount for fast histogram (much faster than masked sum)
    h_histograms = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        h_histograms[i] = np.bincount(bin_id_flat, weights=weights_flat[i], minlength=K)

    # ==========================================================================
    # Step 4: Entropy-gated adaptive weighting
    # ==========================================================================
    Q_k = np.sum(h_histograms, axis=0)  # [K] - pool-wide mass per bin
    Q_total = np.sum(Q_k) + eps
    q_k = Q_k / Q_total  # Normalized distribution

    # Pool entropy: H_pool = -Σ_k q_k * log(q_k)
    q_k_safe = np.clip(q_k, eps, 1.0)
    H_pool = -np.sum(q_k * np.log(q_k_safe))

    # Adaptive lambda: λ^(t) = λ_max * H_pool / log(K)
    log_K = np.log(K) if K > 1 else 1.0
    adaptive_lambda = lambda_max * (H_pool / log_K)

    # ==========================================================================
    # Stage 2: Vectorized lazy greedy selection on reduced set
    # ==========================================================================
    selected = []
    selected_mask = np.zeros(N, dtype=bool)
    H_k = np.zeros(K, dtype=np.float32)  # Cumulative coverage
    sqrt_H_k = np.zeros(K, dtype=np.float32)

    for _ in range(min(num_samples, N)):
        # Vectorized delta_spatial computation
        new_H_k = H_k + h_histograms  # [N, K]
        sqrt_new_H_k = np.sqrt(np.maximum(new_H_k, 0))
        delta_spatial = np.sum(sqrt_new_H_k - sqrt_H_k, axis=1)  # [N]

        # Compute scores: U(x) + λ * Δ_spatial(x|S)
        scores = filtered_uncertainties + adaptive_lambda * delta_spatial

        # Mask out already selected
        scores[selected_mask] = -np.inf

        # Find best candidate
        best_idx = np.argmax(scores)
        if scores[best_idx] == -np.inf:
            break

        # Add to selected
        selected.append(filtered_indices[best_idx])
        selected_mask[best_idx] = True

        # Update cumulative coverage
        H_k += h_histograms[best_idx]
        sqrt_H_k = np.sqrt(np.maximum(H_k, 0))

    return selected


def select_samples_sparcl_det(pool_indices, uncertainties, num_samples, selected_indices,
                              predictions=None, bin_id=None, areas=None, lambda_max=1.0, seed=None,
                              entropy_gate=True, fixed_lambda=None):
    """
    SPARCL for Detection: Spatial Active Learning with Entropy-Gated Adaptive Weighting.

    Detection-specific implementation that uses bbox information instead of pixel-level data.

    Mathematical formulation (same as paper):
    1. Per-image spatial histogram:
       h_k(x) = Σ_{bbox overlapping bin k} w(bbox) * overlap_ratio
       where w(bbox) = score(bbox) * uncertainty(bbox)

    2. Entropy-gated adaptive weighting:
       Q_k = Σ_{x∈U_t} h_k(x)  (pool-wide mass per bin)
       q_k = Q_k / Σ_j Q_j     (normalized distribution)
       H_pool = -Σ_k q_k * log(q_k)  (pool entropy)
       λ^(t) = λ_max * H_pool / log(K)

    3. Diminishing return function: ψ(z) = √z

    4. Acquisition objective:
       Δ(x|S) = U(x) + λ^(t) * Σ_k [ψ(H_k(S) + h_k(x)) - ψ(H_k(S))]

    Args:
        pool_indices: List of available sample indices
        uncertainties: List of uncertainty values for each sample (U(x))
        num_samples: Number of samples to select
        selected_indices: List of already selected indices
        predictions: List of detection predictions, each is a list of dicts with 'boxes', 'scores', 'labels'
        bin_id: Tensor [H, W] - spatial bin assignments (0..K-1)
        areas: List of (y_start, y_end, x_start, x_end, area_idx) tuples (alternative to bin_id)
        lambda_max: Maximum weight for spatial coverage term (λ_max)
        seed: Random seed for reproducibility

    Returns:
        List of selected sample indices
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round (no uncertainties), use random selection
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)

    # If required data is not provided, fall back to uncertainty selection
    if predictions is None or (bin_id is None and areas is None):
        print(f"⚠️ SPARCL-Det selection: Missing required data (predictions={predictions is not None}, "
              f"bin_id={bin_id is not None}, areas={areas is not None}). "
              f"Falling back to uncertainty selection.")
        return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)

    # Convert to numpy if needed
    if torch.is_tensor(bin_id):
        bin_id = bin_id.cpu().numpy()

    # Get spatial bin information
    if bin_id is not None:
        H, W = bin_id.shape
        K = int(np.max(bin_id)) + 1
        # Convert bin_id to areas format for easier bbox overlap calculation
        areas = []
        for k in range(K):
            bin_mask = (bin_id == k)
            y_coords, x_coords = np.where(bin_mask)
            if len(y_coords) > 0:
                y_start, y_end = int(y_coords.min()), int(y_coords.max()) + 1
                x_start, x_end = int(x_coords.min()), int(x_coords.max()) + 1
                areas.append((y_start, y_end, x_start, x_end, k))
    else:
        # Use provided areas
        K = len(areas)
        H, W = 512, 512  # Default, should match dataset

    # ==========================================================================
    # Optimized index mapping for O(1) lookup
    # ==========================================================================
    N = len(available_indices)
    pool_idx_to_pos = {idx: pos for pos, idx in enumerate(pool_indices)}
    available_positions = np.array([pool_idx_to_pos[idx] for idx in available_indices])

    # Pre-compute uncertainty scores for O(1) lookup
    available_uncertainties = np.array([uncertainties[pool_idx_to_pos[idx]] for idx in available_indices])

    # ==========================================================================
    # Step 1: Calculate bbox weights and spatial histogram
    # w(bbox) = score(bbox) * uncertainty(image)
    # h_k(x) = Σ_{bbox overlapping bin k} w(bbox) * overlap_ratio
    # ==========================================================================
    def calculate_bbox_bin_overlap(bbox, bin_area):
        """Calculate overlap ratio between bbox and bin area."""
        y_start, y_end, x_start, x_end, _ = bin_area

        # Bbox format: [x1, y1, x2, y2]
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

        # Calculate intersection
        inter_x1 = max(bbox_x1, x_start)
        inter_y1 = max(bbox_y1, y_start)
        inter_x2 = min(bbox_x2, x_end)
        inter_y2 = min(bbox_y2, y_end)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        # Intersection area
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Bbox area
        bbox_area = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)

        if bbox_area == 0:
            return 0.0

        # Overlap ratio: intersection / bbox_area
        return inter_area / bbox_area

    # Calculate spatial histogram for each image
    h_histograms = np.zeros((N, K), dtype=np.float32)

    # Check predictions length and determine indexing
    num_predictions = len(predictions) if predictions is not None else 0

    # If predictions is too short, use uncertainty-only selection
    if num_predictions < N * 0.5:  # Less than 50% coverage
        print(f"[WARNING] SPARCL-Det: predictions too short ({num_predictions} vs {N} available). "
              f"Falling back to uncertainty selection.")
        return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)

    # Create a mapping from pool index to prediction index
    # predictions come from inference in batch order, matching pool_indices order
    for i, idx in enumerate(available_indices):
        # predictions are indexed sequentially (0 to num_predictions-1)
        # We need to map available_indices to prediction indices
        # Since predictions come from iterating over pool_indices in order,
        # we use the position in pool_indices
        pred_idx = pool_idx_to_pos[idx]

        # If pred_idx is beyond predictions length, skip this sample
        # This can happen if predictions only cover a subset
        if pred_idx >= num_predictions:
            continue

        pred = predictions[pred_idx]

        # Get bboxes and scores
        if isinstance(pred, list) and len(pred) > 0:
            if isinstance(pred[0], dict):
                boxes = pred[0]['boxes']
                scores = pred[0]['scores']
            else:
                boxes = pred[0] if hasattr(pred[0], 'boxes') else []
                scores = pred[0]['scores'] if hasattr(pred[0], 'scores') else []
        elif isinstance(pred, dict):
            boxes = pred.get('boxes', [])
            scores = pred.get('scores', [])
        else:
            boxes = []
            scores = []

        # Convert to numpy if needed
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()

        if len(boxes) == 0:
            # No detections: h_k(x) = 0 for all bins
            continue

        # Get image-level uncertainty
        img_uncertainty = available_uncertainties[i]

        # Calculate weight for each bbox using entropy (like SPARCL)
        # w(bbox) = score * entropy(score) - gives higher weight to uncertain bboxes
        eps_score = 1e-8
        scores_safe = np.clip(scores, eps_score, 1 - eps_score)
        bbox_entropy = -(scores_safe * np.log(scores_safe) + (1 - scores_safe) * np.log(1 - scores_safe))
        bbox_weights = scores * bbox_entropy  # lesionness(score) × entropy(score)

        # Calculate spatial histogram: h_k(x) = Σ_{bbox overlapping bin k} w(bbox) * overlap_ratio
        for bbox, weight in zip(boxes, bbox_weights):
            for bin_idx, bin_area in enumerate(areas):
                overlap_ratio = calculate_bbox_bin_overlap(bbox, bin_area)
                if overlap_ratio > 0:
                    h_histograms[i, bin_idx] += weight * overlap_ratio

    # ==========================================================================
    # Step 2: Entropy-gated adaptive weighting
    # Q_k = Σ_{x∈U_t} h_k(x)  (pool-wide mass per bin)
    # q_k = Q_k / Σ_j Q_j     (normalized distribution)
    # H_pool = -Σ_k q_k * log(q_k)
    # λ^(t) = λ_max * H_pool / log(K)
    # ==========================================================================
    eps = 1e-8
    Q_k = np.sum(h_histograms, axis=0)  # [K] - pool-wide mass per bin
    Q_total = np.sum(Q_k) + eps
    q_k = Q_k / Q_total  # Normalized distribution

    # Pool entropy: H_pool = -Σ_k q_k * log(q_k)
    q_k_safe = np.clip(q_k, eps, 1.0)
    H_pool = -np.sum(q_k * np.log(q_k_safe))

    # Lambda computation (paper-default: entropy-gated; ablations override).
    # - entropy_gate=True (default): λ^(t) = λ_max * H_pool / log(K)
    # - entropy_gate=False: λ^(t) = λ_max  (no gating)
    # - fixed_lambda set: λ^(t) = fixed_lambda  (override entirely)
    log_K = np.log(K) if K > 1 else 1.0
    if fixed_lambda is not None:
        adaptive_lambda = fixed_lambda
    elif entropy_gate:
        adaptive_lambda = lambda_max * (H_pool / log_K)
    else:
        adaptive_lambda = lambda_max

    # ==========================================================================
    # Step 3: Vectorized greedy selection
    # ψ(z) = √z
    # Δ(x|S) = U(x) + λ^(t) * Σ_k [ψ(H_k(S) + h_k(x)) - ψ(H_k(S))]
    # ==========================================================================
    selected = []
    selected_mask = np.zeros(N, dtype=bool)
    H_k = np.zeros(K, dtype=np.float32)  # Cumulative coverage
    sqrt_H_k = np.zeros(K, dtype=np.float32)

    for _ in range(min(num_samples, N)):
        # Vectorized: compute delta_spatial for all candidates at once
        new_H_k = H_k + h_histograms  # [N, K]
        sqrt_new_H_k = np.sqrt(np.maximum(new_H_k, 0))
        delta_spatial = np.sum(sqrt_new_H_k - sqrt_H_k, axis=1)  # [N]

        # Compute scores: U(x) + λ * Δ_spatial(x|S)
        scores = available_uncertainties + adaptive_lambda * delta_spatial

        # Mask out already selected
        scores[selected_mask] = -np.inf

        # Find best candidate
        best_local_idx = np.argmax(scores)
        if scores[best_local_idx] == -np.inf:
            break

        # Add to selected
        selected.append(available_indices[best_local_idx])
        selected_mask[best_local_idx] = True

        # Update cumulative coverage
        H_k += h_histograms[best_local_idx]
        sqrt_H_k = np.sqrt(np.maximum(H_k, 0))

    return selected


def select_samples_sparcl_det_no_gating(pool_indices, uncertainties, num_samples, selected_indices,
                                        predictions=None, bin_id=None, areas=None, lambda_max=1.0, seed=None):
    """SPARCL-Det ablation: no entropy gating. Uses lambda^(t) = lambda_max constantly."""
    return select_samples_sparcl_det(pool_indices, uncertainties, num_samples, selected_indices,
                                     predictions=predictions, bin_id=bin_id, areas=areas,
                                     lambda_max=lambda_max, seed=seed, entropy_gate=False)


def select_samples_sparcl_det_fixed_lambda(pool_indices, uncertainties, num_samples, selected_indices,
                                           predictions=None, bin_id=None, areas=None, lambda_max=0.5, seed=None):
    """SPARCL-Det ablation: fixed lambda (default 0.5). No adaptive gating."""
    return select_samples_sparcl_det(pool_indices, uncertainties, num_samples, selected_indices,
                                     predictions=predictions, bin_id=bin_id, areas=areas,
                                     lambda_max=lambda_max, seed=seed, fixed_lambda=lambda_max)


# Selection strategy mapping
SELECTION_STRATEGIES = {
    'random': select_samples_random,
    'uncertainty': select_samples_uncertainty,
    'area_random': select_samples_area_random,
    'uncertainty_area': select_samples_uncertainty_area,
    'adaptive': select_samples_adaptive,
    'adaptive_improved': select_samples_adaptive_improved,
    'adaptive_multi_scale': select_samples_multi_scale_hybrid,
    'adaptive_performance_monitoring': select_samples_adaptive_performance_monitoring,
    'diversity': select_samples_diversity,
    'diversity_uncertainty': select_samples_diversity_uncertainty,
    'coreset': select_samples_coreset,
    'taudis': select_samples_TAUDIS,
    # ALUNET strategies (MIDL 2024)
    'usimc': select_samples_usimc,
    'mcd_alunet': select_samples_mcd_alunet,
    # LUNIT strategy (Learning Loss, CVPR 2019)
    'lunit': select_samples_lunit,
    # SPARCL: Paper-exact implementation
    'sparcl': select_samples_sparcl,
    'sparcl_prefiltering': select_samples_sparcl_prefiltering,
    'sparcl_det': select_samples_sparcl_det,  # Detection-specific SPARCL
    # SPARCL ablation variants for MICCAI 2026 rebuttal (component-level)
    'sparcl_no_gating': select_samples_sparcl_no_gating,
    'sparcl_fixed_lambda': select_samples_sparcl_fixed_lambda,
    'sparcl_no_coverage': select_samples_sparcl_no_coverage,
    'sparcl_det_no_gating': select_samples_sparcl_det_no_gating,
    'sparcl_det_fixed_lambda': select_samples_sparcl_det_fixed_lambda,
}


def get_selection_strategy(strategy_name):
    """Get selection strategy function by name."""
    if strategy_name not in SELECTION_STRATEGIES:
        raise ValueError(f"Unknown selection strategy: {strategy_name}. "
                         f"Available strategies: {list(SELECTION_STRATEGIES.keys())}")
    return SELECTION_STRATEGIES[strategy_name]
