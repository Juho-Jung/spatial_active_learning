#!/usr/bin/env python3
"""
Coreset sample selection strategy (ICLR 2018).

This module contains the Core-Set approach from "Active Learning for Convolutional Neural Networks:
A Core-Set Approach" (Sener & Savarese, ICLR 2018).
"""

import numpy as np
import torch


def select_samples_coreset(pool_indices, uncertainties, num_samples, selected_indices, areas, features=None, seed=None):
    """
    Coreset selection strategy (ICLR 2018) using k-center greedy algorithm.

    This implements the Core-Set approach from "Active Learning for Convolutional Neural Networks:
    A Core-Set Approach" (Sener & Savarese, ICLR 2018).

    The algorithm selects samples that maximize the minimum distance to the labeled set,
    effectively solving a k-center problem in the feature space.

    Args:
        pool_indices: List of available sample indices
        uncertainties: List of uncertainty values for each sample (not used in pure Coreset)
        num_samples: Number of samples to select
        selected_indices: List of already selected indices (labeled set)
        areas: List of spatial areas (not used in Coreset)
        features: Feature embeddings [N, D] for distance calculation
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
        print("⚠️ Coreset selection: No features provided. Falling back to random selection.")
        return rng.choice(available_indices, size=num_samples, replace=False).tolist()

    # Convert to numpy if needed
    if torch.is_tensor(features):
        features = features.cpu().numpy()

    # Map available_indices to their positions in pool_indices for data extraction
    available_positions = [pool_indices.index(idx) for idx in available_indices]

    # Get features for available indices
    available_features = features[available_positions]  # [N, D]

    # Normalize features for better distance computation
    feature_norms = np.linalg.norm(available_features, axis=1, keepdims=True)
    feature_norms = np.where(feature_norms > 0, feature_norms, 1.0)  # Avoid division by zero
    normalized_features = available_features / feature_norms

    # Get features for already selected (labeled) samples
    labeled_features = []
    if selected_indices:
        labeled_positions = [pool_indices.index(idx) for idx in selected_indices if idx in pool_indices]
        if labeled_positions:
            labeled_features = features[labeled_positions]
            labeled_norms = np.linalg.norm(labeled_features, axis=1, keepdims=True)
            labeled_norms = np.where(labeled_norms > 0, labeled_norms, 1.0)
            labeled_features = labeled_features / labeled_norms

    # Coreset k-center greedy selection
    selected = []
    selected_feature_indices = []  # Indices in available_features array

    # Initialize: select first sample
    if selected_indices and len(labeled_features) > 0:
        # Select sample farthest from labeled set
        distances_to_labeled = []
        for i, feat in enumerate(normalized_features):
            # Calculate minimum distance to any labeled sample
            if len(labeled_features) > 0:
                dists = np.linalg.norm(feat - labeled_features, axis=1)
                min_dist = np.min(dists)
            else:
                min_dist = np.linalg.norm(feat)
            distances_to_labeled.append(min_dist)

        first_idx = np.argmax(distances_to_labeled)
    else:
        # First round: select randomly
        first_idx = rng.choice(len(available_indices))

    selected.append(available_indices[first_idx])
    selected_feature_indices.append(first_idx)

    # Greedy selection: in each iteration, select the sample with maximum minimum distance
    # to the current labeled set (including newly selected samples)
    remaining_indices = [i for i in range(len(available_indices)) if i != first_idx]
    current_labeled_features = [normalized_features[first_idx]]

    for _ in range(min(num_samples - 1, len(remaining_indices))):
        if len(remaining_indices) == 0:
            break

        # Calculate minimum distance to labeled set for each remaining sample
        min_distances = []
        for i in remaining_indices:
            feat = normalized_features[i]
            # Distance to already labeled samples (from previous rounds)
            if len(labeled_features) > 0:
                dists_to_old_labeled = np.linalg.norm(feat - labeled_features, axis=1)
                min_dist_to_old = np.min(dists_to_old_labeled) if len(dists_to_old_labeled) > 0 else np.inf
            else:
                min_dist_to_old = np.inf

            # Distance to newly selected samples (in current round)
            if len(current_labeled_features) > 0:
                dists_to_new_labeled = np.linalg.norm(feat - current_labeled_features, axis=1)
                min_dist_to_new = np.min(dists_to_new_labeled)
            else:
                min_dist_to_new = np.inf

            # Minimum distance to entire labeled set
            min_dist = min(min_dist_to_old, min_dist_to_new)
            min_distances.append(min_dist)

        # Select sample with maximum minimum distance (k-center greedy)
        best_remaining_idx = np.argmax(min_distances)
        selected_idx = remaining_indices[best_remaining_idx]

        selected.append(available_indices[selected_idx])
        selected_feature_indices.append(selected_idx)
        current_labeled_features.append(normalized_features[selected_idx])

        # Remove selected sample from remaining
        remaining_indices = [i for i in remaining_indices if i != selected_idx]

    return selected
