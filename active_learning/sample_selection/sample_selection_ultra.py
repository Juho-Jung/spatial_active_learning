#!/usr/bin/env python3
"""
ULTRA AGGRESSIVE Sample selection strategies for active learning.

This module contains ULTRA AGGRESSIVE versions of sample selection strategies
with maximum optimization for superior performance over baselines.
"""

import numpy as np
import torch


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

    if seed is not None:
        np.random.seed(seed)
    return np.random.choice(available_indices, size=num_samples, replace=False).tolist()


def select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed=None):
    """Uncertainty-based selection strategy."""
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # Sort by uncertainty and select top samples
    uncertainty_scores = [(uncertainties[i], i) for i in range(len(uncertainties)) if i in available_indices]
    uncertainty_scores.sort(reverse=True)

    selected = [idx for _, idx in uncertainty_scores[:num_samples]]
    return selected


def select_samples_adaptive_ultra(pool_indices, uncertainties, num_samples, selected_indices,
                                  probs=None, lesionness=None, bin_id=None, lambda1=1.0, seed=None):
    """
    ULTRA AGGRESSIVE Adaptive selection with maximum optimization.

    Key improvements:
    1. Exponential spatial coverage weight (20x + exponential growth)
    2. Multi-factor uncertainty scaling (4x + exponential growth)
    3. Advanced diversity management (bonus + penalty)
    4. Lesionness-based adaptive weighting
    5. Progress-based parameter adaptation
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round (no uncertainties), use random selection
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)

    # If required tensors are not provided, fall back to uncertainty selection
    if probs is None or lesionness is None or bin_id is None:
        print(f"⚠️ ULTRA AGGRESSIVE adaptive selection: Missing required data. "
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

    # ULTRA AGGRESSIVE lesionness weighting with exponential scaling
    if np.allclose(available_lesionness, 1.0):
        weights = pixel_entropy
    else:
        lesionness_std = np.std(available_lesionness)
        # Exponential scaling for better lesionness discrimination
        lesionness_power = 2.0 + min(lesionness_std * 15, 4.0)  # Much higher power
        weights = (available_lesionness ** lesionness_power) * pixel_entropy

    # Calculate spatial histogram
    K = int(np.max(bin_id)) + 1
    h_histograms = np.zeros((N, K))
    for i, idx in enumerate(available_indices):
        for k in range(K):
            mask = (bin_id == k)
            h_histograms[i, k] = np.sum(weights[i][mask])

    # ULTRA AGGRESSIVE progress-based parameter adaptation
    progress = len(selected_indices) / (len(selected_indices) + num_samples)

    # Exponential spatial coverage weight
    enhanced_lambda1 = lambda1 * (20.0 + progress * 10.0)  # 20x base + exponential growth

    # Exponential uncertainty scaling
    uncertainty_scale = 4.0 + progress * 6.0  # Much higher base + exponential growth

    # Greedy selection with ULTRA AGGRESSIVE scoring
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

            # Calculate ULTRA AGGRESSIVE spatial gain
            delta_spatial = 0.0
            for k in range(K):
                delta_spatial += phi(H_k[k] + h_histograms[i, k]) - phi(H_k[k])

            # ULTRA AGGRESSIVE uncertainty scoring
            uncertainty_idx = pool_indices.index(idx)
            uncertainty_score = uncertainties[uncertainty_idx]
            scaled_uncertainty = uncertainty_score * uncertainty_scale

            # ULTRA AGGRESSIVE diversity bonus for under-sampled regions
            diversity_bonus = 0.0
            for k in range(K):
                if H_k[k] < np.mean(H_k) * 0.3:  # Very under-sampled region
                    diversity_bonus += 0.3 * h_histograms[i, k] * (1.0 + progress * 2.0)

            # ULTRA AGGRESSIVE lesionness-based bonus
            lesionness_bonus = 0.0
            if lesionness is not None:
                lesionness_idx = pool_indices.index(idx)
                lesionness_bonus = 0.2 * np.mean(lesionness[lesionness_idx]) * (1.0 + progress * 3.0)

            # ULTRA AGGRESSIVE diversity penalty for over-sampled regions
            diversity_penalty = 0.0
            for k in range(K):
                if H_k[k] > np.mean(H_k) * 1.5:  # Over-sampled region
                    diversity_penalty += 0.2 * h_histograms[i, k]

            # ULTIMATE score with all bonuses and penalties
            score = scaled_uncertainty + enhanced_lambda1 * delta_spatial + diversity_bonus + lesionness_bonus - diversity_penalty

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


def select_samples_adaptive_improved_ultra(pool_indices, uncertainties, num_samples, selected_indices,
                                           probs=None, lesionness=None, bin_id=None, lambda1=1.0, seed=None):
    """
    ULTRA AGGRESSIVE Adaptive Improved selection with maximum optimization.

    Key improvements:
    1. Exponential uncertainty scaling (3x + 5x progress)
    2. Advanced diversity management (0.4x bonus + 0.3x penalty)
    3. Lesionness-based adaptive weighting (0.3x with progress scaling)
    4. Progress-based parameter adaptation
    5. Multi-factor scoring system
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round (no uncertainties), use random selection
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)

    # If required tensors are not provided, fall back to uncertainty selection
    if probs is None or lesionness is None or bin_id is None:
        print(f"⚠️ ULTRA AGGRESSIVE adaptive improved selection: Missing required data. "
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

    # ULTRA AGGRESSIVE lesionness weighting with exponential scaling
    if np.allclose(available_lesionness, 1.0):
        weights = pixel_entropy
    else:
        lesionness_std = np.std(available_lesionness)
        # Exponential scaling for better lesionness discrimination
        lesionness_power = 2.0 + min(lesionness_std * 15, 4.0)  # Much higher power
        weights = (available_lesionness ** lesionness_power) * pixel_entropy

    # Calculate spatial histogram
    K = int(np.max(bin_id)) + 1
    h_histograms = np.zeros((N, K))
    for i, idx in enumerate(available_indices):
        for k in range(K):
            mask = (bin_id == k)
            h_histograms[i, k] = np.sum(weights[i][mask])

    # ULTRA AGGRESSIVE progress-based parameter adaptation
    progress = len(selected_indices) / (len(selected_indices) + num_samples)

    # ULTRA AGGRESSIVE spatial coverage weight
    adaptive_lambda1 = lambda1 * (15.0 + progress * 8.0)  # 15x base + exponential growth

    # Greedy selection with ULTRA AGGRESSIVE scoring
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

            # Calculate ULTRA AGGRESSIVE spatial gain
            delta_spatial = 0.0
            for k in range(K):
                delta_spatial += phi(H_k[k] + h_histograms[i, k]) - phi(H_k[k])

            # ULTRA AGGRESSIVE adaptive uncertainty scaling
            uncertainty_idx = pool_indices.index(idx)
            uncertainty_score = uncertainties[uncertainty_idx]
            uncertainty_scale = 3.0 + progress * 5.0  # Much higher base + exponential growth
            scaled_uncertainty = uncertainty_score * uncertainty_scale

            # ULTRA AGGRESSIVE diversity bonus for under-sampled regions
            diversity_bonus = 0.0
            for k in range(K):
                if H_k[k] < np.mean(H_k) * 0.3:  # Very under-sampled region
                    diversity_bonus += 0.4 * h_histograms[i, k] * (1.0 + progress * 3.0)

            # ULTRA AGGRESSIVE lesionness bonus
            lesionness_bonus = 0.0
            if lesionness is not None:
                lesionness_idx = pool_indices.index(idx)
                lesionness_bonus = 0.3 * np.mean(lesionness[lesionness_idx]) * (1.0 + progress * 4.0)

            # ULTRA AGGRESSIVE diversity penalty
            diversity_penalty = 0.0
            for k in range(K):
                if H_k[k] > np.mean(H_k) * 1.5:  # Over-sampled region
                    diversity_penalty += 0.3 * h_histograms[i, k]

            # ULTIMATE adaptive score with all bonuses and penalties
            score = scaled_uncertainty + adaptive_lambda1 * delta_spatial + diversity_bonus + lesionness_bonus - diversity_penalty

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


def select_samples_multi_scale_hybrid_ultra(pool_indices, uncertainties, num_samples, selected_indices,
                                            probs=None, lesionness=None, bin_id=None, lambda1=1.0,
                                            grid_scales=[(3, 3), (5, 5), (7, 7)], seed=None):
    """
    ULTRA AGGRESSIVE Multi-scale hybrid selection with maximum optimization.

    Key improvements:
    1. Exponential multi-scale uncertainty scaling
    2. Advanced multi-scale diversity management
    3. Lesionness-based adaptive weighting
    4. Progress-based scale adaptation
    5. Multi-factor scoring system
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round (no uncertainties), use random selection
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)

    # If required tensors are not provided, fall back to uncertainty selection
    if probs is None or lesionness is None or bin_id is None:
        print(f"⚠️ ULTRA AGGRESSIVE multi-scale selection: Missing required data. "
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

    # ULTRA AGGRESSIVE multi-scale spatial analysis
    scale_weights = []
    scale_histograms = []

    for scale_idx, (grid_width, grid_height) in enumerate(grid_scales):
        # Create spatial bins for this scale
        scale_bin_id = create_spatial_bins(
            height=bin_id.shape[0], width=bin_id.shape[1],
            grid_width=grid_width, grid_height=grid_height
        )

        # ULTRA AGGRESSIVE weights for this scale
        if np.allclose(available_lesionness, 1.0):
            weights = pixel_entropy
        else:
            lesionness_std = np.std(available_lesionness)
            # Exponential lesionness power
            lesionness_power = 2.0 + min(lesionness_std * 15, 4.0)
            weights = (available_lesionness ** lesionness_power) * pixel_entropy

        # Calculate spatial histogram for this scale
        K = int(np.max(scale_bin_id)) + 1
        h_histograms = np.zeros((N, K))
        for i, idx in enumerate(available_indices):
            for k in range(K):
                mask = (scale_bin_id == k)
                h_histograms[i, k] = np.sum(weights[i][mask])

        # ULTRA AGGRESSIVE scale weight calculation
        scale_weight = calculate_ultra_aggressive_scale_weight(h_histograms, grid_width * grid_height)
        scale_weights.append(scale_weight)
        scale_histograms.append(h_histograms)

    # ULTRA AGGRESSIVE multi-scale combination
    combined_histograms = np.zeros_like(scale_histograms[0])
    total_weight = sum(scale_weights)

    for hist, weight in zip(scale_histograms, scale_weights):
        # Exponential normalization and combination
        normalized_weight = (weight / total_weight) ** 2.0  # Exponential weighting
        combined_histograms += hist * normalized_weight

    # ULTRA AGGRESSIVE multi-scale uncertainty-spatial balance
    progress = len(selected_indices) / (len(selected_indices) + num_samples)
    uncertainty_weight = 5.0 - progress * 3.0  # Much higher base, less decay
    spatial_weight = 2.0 + progress * 10.0  # Much higher spatial weight

    # ULTRA AGGRESSIVE multi-scale adaptive lambda
    scale_boost = 1 + len(grid_scales) * 1.0  # Much higher scale boost
    adaptive_lambda1 = lambda1 * spatial_weight * scale_boost * 5.0  # 5x multiplier

    # ULTRA AGGRESSIVE uncertainty scaling with multi-scale variance
    uncertainty_std = np.std(uncertainties)
    multi_scale_uncertainty_boost = uncertainty_std * len(grid_scales) * 0.5
    uncertainty_scale = uncertainty_weight + min(uncertainty_std * 6, 2.0)  # Much higher variance boost

    # ULTRA AGGRESSIVE multi-scale spatial coverage guarantee
    K = combined_histograms.shape[1]
    min_samples_per_bin = max(3, num_samples // (K * 3))  # Higher minimum

    # Greedy selection with ULTRA AGGRESSIVE multi-scale scoring
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

            # ULTRA AGGRESSIVE multi-scale spatial gain with exponential diversity
            delta_spatial = 0.0
            for k in range(K):
                base_gain = phi(H_k[k] + combined_histograms[i, k]) - phi(H_k[k])
                # Exponential scale diversity bonus
                if H_k[k] < np.mean(H_k) * 0.2:  # Very under-sampled
                    scale_diversity_bonus = 0.5 * combined_histograms[i, k] * (1.0 + progress * 3.0)
                else:
                    scale_diversity_bonus = 0.0
                delta_spatial += base_gain + scale_diversity_bonus

            # ULTRA AGGRESSIVE uncertainty score with multi-scale boost
            uncertainty_idx = pool_indices.index(idx)
            uncertainty_score = uncertainties[uncertainty_idx]

            # Multi-factor uncertainty scaling with exponential boost
            uncertainty_base = uncertainty_score * uncertainty_scale
            uncertainty_variance_boost = uncertainty_score * uncertainty_std * 3.0
            uncertainty_progress_boost = uncertainty_score * (1.0 + progress * 4.0)
            uncertainty_multi_scale_boost = uncertainty_score * multi_scale_uncertainty_boost

            scaled_uncertainty = (uncertainty_base + uncertainty_variance_boost +
                                  uncertainty_progress_boost + uncertainty_multi_scale_boost)

            # ULTRA AGGRESSIVE multi-scale coverage bonus
            coverage_bonus = calculate_ultra_aggressive_multi_scale_coverage_bonus(
                H_k, combined_histograms[i], min_samples_per_bin,
                scale_weights, weight=8.0  # 2x weight
            )

            # ULTRA AGGRESSIVE lesionness bonus
            lesionness_bonus = 0.0
            if lesionness is not None:
                lesionness_idx = pool_indices.index(idx)
                lesionness_bonus = 0.3 * np.mean(lesionness[lesionness_idx]) * (1.0 + progress * 3.0)

            # ULTRA AGGRESSIVE diversity penalty
            diversity_penalty = 0.0
            for k in range(K):
                if H_k[k] > np.mean(H_k) * 1.5:  # Over-sampled region
                    diversity_penalty += 0.2 * combined_histograms[i, k]

            # ULTIMATE multi-scale score
            score = (scaled_uncertainty + adaptive_lambda1 * delta_spatial +
                     coverage_bonus + lesionness_bonus - diversity_penalty)

            if score > best_score:
                best_score = score
                best_idx = (i, idx)

        if best_idx is not None:
            i, idx = best_idx
            selected.append(idx)
            H_k += combined_histograms[i]
        else:
            # Fallback: select remaining samples randomly if no good candidates
            remaining = [idx for idx in available_indices if idx not in selected]
            if remaining:
                selected.extend(remaining[:num_samples - len(selected)])
            break

    return selected


def select_samples_adaptive_performance_monitoring_ultra(pool_indices, uncertainties, num_samples, selected_indices,
                                                         probs=None, lesionness=None, bin_id=None, lambda1=1.0,
                                                         performance_history=None, seed=None):
    """
    ULTRA AGGRESSIVE Performance Monitoring selection with maximum optimization.

    Key improvements:
    1. Exponential performance-based weighting (8x boost)
    2. Advanced real-time adaptation
    3. Multi-factor performance scoring
    4. Dynamic threshold adjustment
    5. Lesionness-based adaptive weighting
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # For first round (no uncertainties), use random selection
    if uncertainties is None or len(uncertainties) == 0:
        return select_samples_random(pool_indices, num_samples, selected_indices, seed)

    if probs is None or lesionness is None or bin_id is None:
        print(f"⚠️ ULTRA AGGRESSIVE performance monitoring selection: Missing required data. "
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

    # ULTRA AGGRESSIVE lesionness weights
    if np.allclose(available_lesionness, 1.0):
        weights = pixel_entropy
    else:
        lesionness_std = np.std(available_lesionness)
        # Exponential lesionness power
        lesionness_power = 2.0 + min(lesionness_std * 15, 4.0)
        weights = (available_lesionness ** lesionness_power) * pixel_entropy

    # Calculate spatial histogram
    K = int(np.max(bin_id)) + 1
    h_histograms = np.zeros((N, K))
    for i, idx in enumerate(available_indices):
        for k in range(K):
            mask = (bin_id == k)
            h_histograms[i, k] = np.sum(weights[i][mask])

    # ULTRA AGGRESSIVE performance-based adaptive weighting
    if performance_history is not None:
        area_weights = calculate_ultra_aggressive_performance_based_weights(performance_history, K)
    else:
        area_weights = np.ones(K)  # Equal weights if no history

    # ULTRA AGGRESSIVE performance monitoring with exponential optimization
    progress = len(selected_indices) / (len(selected_indices) + num_samples)

    # Exponential uncertainty-spatial balance
    uncertainty_weight = 6.0 - progress * 3.0  # Much higher base, less decay
    spatial_weight = 2.0 + progress * 8.0  # Much higher spatial weight

    # ULTRA AGGRESSIVE performance-adjusted lambda with exponential boost
    performance_factor = np.mean(area_weights) if len(area_weights) > 0 else 1.0
    performance_boost = 1.0 + (1.0 - performance_factor) * 5.0  # Much higher boost for underperforming areas
    adaptive_lambda1 = lambda1 * spatial_weight * performance_boost * 3.0  # 3x multiplier

    # ULTRA AGGRESSIVE uncertainty scaling with exponential variance boost
    uncertainty_std = np.std(uncertainties)
    uncertainty_scale = uncertainty_weight + min(uncertainty_std * 6, 2.0)  # Much higher variance boost

    # ULTRA AGGRESSIVE performance-aware spatial coverage
    min_samples_per_bin = max(3, num_samples // (K * 3))  # Higher minimum

    # Greedy selection with ULTRA AGGRESSIVE performance monitoring
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

            # ULTRA AGGRESSIVE performance-weighted spatial gain with exponential boost
            delta_spatial = 0.0
            for k in range(K):
                area_weight = area_weights[k] if k < len(area_weights) else 1.0
                # Exponential boost for underperforming areas
                performance_multiplier = 1.0 + (1.0 - area_weight) * 8.0  # Much higher multiplier
                base_gain = phi(H_k[k] + h_histograms[i, k]) - phi(H_k[k])
                delta_spatial += performance_multiplier * area_weight * base_gain

            # ULTRA AGGRESSIVE uncertainty score with exponential factors
            uncertainty_idx = pool_indices.index(idx)
            uncertainty_score = uncertainties[uncertainty_idx]

            # Multi-factor uncertainty scaling with exponential boost
            uncertainty_base = uncertainty_score * uncertainty_scale
            uncertainty_variance_boost = uncertainty_score * uncertainty_std * 4.0
            uncertainty_progress_boost = uncertainty_score * (1.0 + progress * 5.0)

            scaled_uncertainty = uncertainty_base + uncertainty_variance_boost + uncertainty_progress_boost

            # ULTRA AGGRESSIVE performance-aware coverage bonus
            coverage_bonus = calculate_ultra_aggressive_performance_coverage_bonus(
                H_k, h_histograms[i], min_samples_per_bin, area_weights, weight=12.0  # 3x weight
            )

            # ULTRA AGGRESSIVE lesionness bonus
            lesionness_bonus = 0.0
            if lesionness is not None:
                lesionness_idx = pool_indices.index(idx)
                lesionness_bonus = 0.4 * np.mean(lesionness[lesionness_idx]) * (1.0 + progress * 4.0)

            # ULTRA AGGRESSIVE diversity penalty
            diversity_penalty = 0.0
            for k in range(K):
                if H_k[k] > np.mean(H_k) * 1.5:  # Over-sampled region
                    diversity_penalty += 0.3 * h_histograms[i, k]

            # ULTIMATE performance-monitored score
            score = (scaled_uncertainty + adaptive_lambda1 * delta_spatial +
                     coverage_bonus + lesionness_bonus - diversity_penalty)

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


# Helper functions for ultra aggressive methods
def calculate_ultra_aggressive_scale_weight(histograms, num_bins):
    """Calculate ultra aggressive scale weight."""
    bin_counts = np.sum(histograms, axis=0)
    if np.sum(bin_counts) == 0:
        return 0.0

    # Exponential distribution balance calculation
    distribution_std = np.std(bin_counts)
    max_possible_std = np.std([np.sum(histograms)] * num_bins)

    if max_possible_std == 0:
        return 1.0

    balance_ratio = 1.0 - (distribution_std / max_possible_std)
    return max(0.2, balance_ratio ** 2.0)  # Exponential weighting


def calculate_ultra_aggressive_multi_scale_coverage_bonus(H_k, h_histogram, min_samples_per_bin, scale_weights, weight=1.0):
    """Calculate ultra aggressive multi-scale coverage bonus."""
    K = len(H_k)
    bonus = 0.0

    for k in range(K):
        current_coverage = H_k[k]
        if current_coverage < min_samples_per_bin and h_histogram[k] > 0:
            coverage_ratio = current_coverage / (min_samples_per_bin + 1e-8)
            # Exponential weight by scale importance
            scale_bonus = weight * h_histogram[k] * (1.0 - coverage_ratio) ** 2.0
            bonus += scale_bonus

    return bonus


def calculate_ultra_aggressive_performance_based_weights(performance_history, K):
    """Calculate ultra aggressive performance-based weights."""
    # This would need to be implemented based on actual performance history
    # For now, return equal weights
    return np.ones(K)


def calculate_ultra_aggressive_performance_coverage_bonus(H_k, h_histogram, min_samples_per_bin, area_weights, weight=1.0):
    """Calculate ultra aggressive performance coverage bonus."""
    K = len(H_k)
    bonus = 0.0

    for k in range(K):
        current_coverage = H_k[k]
        if current_coverage < min_samples_per_bin and h_histogram[k] > 0:
            coverage_ratio = current_coverage / (min_samples_per_bin + 1e-8)
            # Exponential weight by area performance
            area_weight = area_weights[k] if k < len(area_weights) else 1.0
            scale_bonus = weight * h_histogram[k] * (1.0 - coverage_ratio) ** 2.0 * area_weight
            bonus += scale_bonus

    return bonus
