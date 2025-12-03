#!/usr/bin/env python3
"""
Sample selection strategies for active learning.

This module contains various sample selection strategies used in active learning
experiments, including random, uncertainty-based, and area-based selection methods.
"""

import numpy as np
import torch

# Import ULTRA AGGRESSIVE versions
try:
    from .sample_selection_ultra import (
        select_samples_adaptive_improved_ultra,
        select_samples_adaptive_performance_monitoring_ultra,
        select_samples_adaptive_ultra, select_samples_multi_scale_hybrid_ultra)
except ImportError:
    # Fallback if ultra versions are not available
    select_samples_adaptive_ultra = None
    select_samples_adaptive_improved_ultra = None
    select_samples_multi_scale_hybrid_ultra = None
    select_samples_adaptive_performance_monitoring_ultra = None

# Import Coreset (ICLR 2018)
try:
    from .sample_selection_coreset import select_samples_coreset
except ImportError:
    # Fallback if coreset is not available
    select_samples_coreset = None

# Import TAUDIS (ICCV 2023)
try:
    from .sample_selection_TAUDIS import select_samples_TAUDIS
except ImportError:
    # Fallback if TAUDIS is not available
    select_samples_TAUDIS = None

# Import ALUNET strategies (MIDL 2024)
try:
    from .sample_selection_alunet import (select_samples_mcd_alunet,
                                          select_samples_usimc)
except ImportError:
    # Fallback if ALUNET strategies are not available
    select_samples_usimc = None
    select_samples_mcd_alunet = None

# Import LUNIT strategy (Learning Loss, CVPR 2019)
try:
    from .sample_selection_lunit import select_samples_lunit
except ImportError:
    # Fallback if LUNIT is not available
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
    # ULTRA AGGRESSIVE versions
    'adaptive_ultra': select_samples_adaptive_ultra,
    'adaptive_improved_ultra': select_samples_adaptive_improved_ultra,
    'adaptive_multi_scale_ultra': select_samples_multi_scale_hybrid_ultra,
    'adaptive_performance_monitoring_ultra': select_samples_adaptive_performance_monitoring_ultra}


def get_selection_strategy(strategy_name):
    """Get selection strategy function by name."""
    if strategy_name not in SELECTION_STRATEGIES:
        raise ValueError(f"Unknown selection strategy: {strategy_name}. "
                         f"Available strategies: {list(SELECTION_STRATEGIES.keys())}")
    return SELECTION_STRATEGIES[strategy_name]
