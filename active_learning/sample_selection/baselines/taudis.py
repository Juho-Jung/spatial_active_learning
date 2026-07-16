#!/usr/bin/env python3
"""
TAUDIS: Two-step Active learning with Uncertainty and Diversity sampling for Instance Segmentation.

This module implements the TAUDIS algorithm from "Two-Step Active Learning for Instance Segmentation
with Uncertainty and Diversity Sampling" (ICCV 2023).

Reference: https://arxiv.org/pdf/2309.16139
"""

import numpy as np
import torch
from scipy import ndimage


def _extract_instances_from_mask(mask, min_area=10):
    """
    Extract instances from a binary mask using connected components.

    Args:
        mask: Binary mask [H, W] with values in {0, 1}
        min_area: Minimum area for an instance to be considered

    Returns:
        List of instance masks, each as a binary array [H, W]
    """
    # Find connected components
    labeled_mask, num_instances = ndimage.label(mask > 0.5)

    instances = []
    for instance_id in range(1, num_instances + 1):
        instance_mask = (labeled_mask == instance_id).astype(np.float32)
        if instance_mask.sum() >= min_area:
            instances.append(instance_mask)

    return instances


def _extract_instance_features(feature_map, instance_mask):
    """
    Extract feature vector for an instance from a feature map.

    Args:
        feature_map: Feature map [C, H, W] or [H, W, C]
        instance_mask: Binary mask [H, W] for the instance

    Returns:
        Feature vector [C] or [feature_dim]
    """
    # Handle different feature map formats
    if len(feature_map.shape) == 3:
        if feature_map.shape[0] < feature_map.shape[2]:  # [C, H, W]
            C, H, W = feature_map.shape
            feature_map = feature_map.transpose(1, 2, 0)  # [H, W, C]
        # Now feature_map is [H, W, C]
        H, W, C = feature_map.shape
    else:
        raise ValueError(f"Unexpected feature map shape: {feature_map.shape}")

    # Resize instance mask to match feature map if needed
    if instance_mask.shape != (H, W):
        from scipy.ndimage import zoom
        zoom_factors = (H / instance_mask.shape[0], W / instance_mask.shape[1])
        instance_mask = zoom(instance_mask, zoom_factors, order=0)
        instance_mask = (instance_mask > 0.5).astype(np.float32)

    # Extract features: average pooling over instance region
    instance_mask_expanded = instance_mask[..., np.newaxis]  # [H, W, 1]
    masked_features = feature_map * instance_mask_expanded  # [H, W, C]

    # Average pooling
    instance_area = instance_mask.sum()
    if instance_area > 0:
        instance_feature = masked_features.sum(axis=(0, 1)) / instance_area  # [C]
    else:
        instance_feature = np.zeros(C, dtype=np.float32)

    return instance_feature


def _compute_instance_uncertainty(pred_mask, instance_mask, uncertainty_map=None):
    """
    Compute uncertainty score for an instance.

    Args:
        pred_mask: Prediction mask [H, W] with values in [0, 1]
        instance_mask: Binary mask [H, W] for the instance
        uncertainty_map: Optional pre-computed uncertainty map [H, W]

    Returns:
        Uncertainty score (scalar)
    """
    if uncertainty_map is not None:
        # Use pre-computed uncertainty map
        instance_uncertainty = (uncertainty_map * instance_mask).sum() / (instance_mask.sum() + 1e-8)
    else:
        # Compute uncertainty from prediction: segmentation entropy
        # SE = mean(BCE) where BCE = -p*log(p) - (1-p)*log(1-p)
        epsilon = 1e-8
        p_safe = np.clip(pred_mask, epsilon, 1 - epsilon)
        bce = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))

        # Average over instance region
        instance_uncertainty = (bce * instance_mask).sum() / (instance_mask.sum() + 1e-8)

    return float(instance_uncertainty)


def _maximum_k_set_cover(similarity_matrix, k, threshold=0.8):
    """
    Solve maximum k-set cover problem on a bipartite graph (optimized version).

    The problem: Given a bipartite graph defined by similarity matrix S,
    select k rows (subsets) that cover the maximum number of columns (universe elements).

    Args:
        similarity_matrix: Matrix [N, M] where element (i, j) is similarity between
                          row i and column j. Values < threshold are treated as 0.
        k: Number of rows to select
        threshold: Similarity threshold (elements < threshold are set to 0)

    Returns:
        List of selected row indices
    """
    # Set elements below threshold to 0 (in-place for efficiency)
    S = similarity_matrix.copy()
    S[S < threshold] = 0

    N, M = S.shape
    k = min(k, N)  # Can't select more rows than available

    if k == 0:
        return []

    # Optimized greedy algorithm: use boolean arrays and vectorized operations
    selected = []
    covered = np.zeros(M, dtype=bool)
    remaining_rows = set(range(N))  # Track remaining rows efficiently

    for _ in range(k):
        if covered.all():
            # All columns covered, select remaining rows arbitrarily
            if remaining_rows:
                selected.append(remaining_rows.pop())
            break

        # Find row that covers the most uncovered columns (vectorized)
        best_row = -1
        best_coverage = -1

        # Pre-compute uncovered mask once
        uncovered = ~covered

        for i in list(remaining_rows):  # Iterate over copy to allow modification
            # Count uncovered columns covered by this row (vectorized)
            row_covered = (S[i] > 0) & uncovered
            coverage = row_covered.sum()

            if coverage > best_coverage:
                best_coverage = coverage
                best_row = i

        if best_row == -1:
            # No row covers any uncovered columns, select arbitrarily
            if remaining_rows:
                selected.append(remaining_rows.pop())
            break

        selected.append(best_row)
        remaining_rows.discard(best_row)
        # Mark columns covered by this row (vectorized)
        covered = covered | (S[best_row] > 0)

    return selected


def select_samples_TAUDIS(pool_indices, uncertainties, num_samples, selected_indices, areas,
                          features=None, probs=None, alpha=2.5, beta=1.5, sigma=0.8, seed=None):
    """
    TAUDIS: Two-step Active learning with Uncertainty and Diversity sampling.

    Algorithm:
    1. Extract instances from predictions
    2. Compute instance-level uncertainty scores
    3. Select top αB uncertain instances
    4. Construct similarity matrix between uncertain instances and all instances
    5. Use maximum k-set cover to select βB diverse instances
    6. Select B images containing the most selected instances

    Args:
        pool_indices: List of available sample indices
        uncertainties: List of uncertainty values for each sample (image-level)
        num_samples: Number of samples (images) to select (B)
        selected_indices: List of already selected indices
        areas: List of spatial areas (not used in TAUDIS)
        features: Feature embeddings [N, D] for images (used to extract instance features)
        probs: Prediction probabilities [N, H, W] for extracting instances
        alpha: Hyperparameter α (default 2.5, should be > 1)
        beta: Hyperparameter β (default 1.5, should satisfy α > β > 1)
        sigma: Similarity threshold σ (default 0.8, should be in (0, 1))
        seed: Random seed for reproducibility

    Returns:
        List of selected sample indices (images)
    """
    available_indices = [idx for idx in pool_indices if idx not in selected_indices]
    if len(available_indices) < num_samples:
        return available_indices

    # Use specific seed for reproducibility
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # If no features or probs provided, fall back to uncertainty selection
    if features is None or probs is None:
        print("⚠️ TAUDIS selection: Missing features or probs. Falling back to uncertainty selection.")
        from ..sample_selection import select_samples_uncertainty
        return select_samples_uncertainty(pool_indices, uncertainties, num_samples, selected_indices, seed)

    # Convert to numpy if needed
    if torch.is_tensor(features):
        features = features.cpu().numpy()
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()

    # Create efficient index mapping (O(n) instead of O(n*m))
    pool_idx_to_pos = {idx: pos for pos, idx in enumerate(pool_indices)}
    available_positions = [pool_idx_to_pos[idx] for idx in available_indices]

    # Extract all instances from all available images
    all_instances = []  # List of instance dicts
    image_to_instances = {}  # Map image_idx -> list of instance indices in all_instances

    # Pre-compute uncertainty maps for all images (vectorized where possible)
    print(f"🔍 TAUDIS: Extracting instances from {len(available_indices)} images...")
    epsilon = 1e-8

    # Extract instances and their features/uncertainties
    for i, pos in enumerate(available_positions):
        img_idx = available_indices[i]
        pred_mask = probs[pos]  # [H, W]
        img_feature = features[pos]  # [D]

        # Compute uncertainty map (if needed)
        if uncertainties is not None:
            p_safe = np.clip(pred_mask, epsilon, 1 - epsilon)
            uncertainty_map = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))
        else:
            uncertainty_map = np.ones_like(pred_mask) * 0.5

        # Extract instances from prediction mask
        instances = _extract_instances_from_mask(pred_mask, min_area=10)

        if len(instances) == 0:
            continue

        # Extract instance features with spatial information for diversity
        image_instance_indices = []
        H, W = pred_mask.shape
        for instance_mask in instances:
            # Extract instance spatial information
            instance_area = instance_mask.sum()

            # Compute instance center (normalized to [0, 1])
            y_coords, x_coords = np.where(instance_mask > 0.5)
            if len(y_coords) > 0:
                center_y = y_coords.mean() / H  # Normalized to [0, 1]
                center_x = x_coords.mean() / W  # Normalized to [0, 1]
            else:
                center_y, center_x = 0.5, 0.5

            # Compute instance size (normalized)
            instance_size = instance_area / (H * W)  # Normalized to [0, 1]

            # Compute bounding box (normalized)
            if len(y_coords) > 0:
                bbox_y_min = y_coords.min() / H
                bbox_y_max = y_coords.max() / H
                bbox_x_min = x_coords.min() / W
                bbox_x_max = x_coords.max() / W
                bbox_height = bbox_y_max - bbox_y_min
                bbox_width = bbox_x_max - bbox_x_min
            else:
                bbox_y_min = bbox_y_max = bbox_x_min = bbox_x_max = 0.5
                bbox_height = bbox_width = 0.0

            # Combine image-level feature with instance spatial information
            spatial_features = np.array([
                center_y, center_x,  # Instance center
                instance_size,  # Instance size
                bbox_y_min, bbox_y_max, bbox_x_min, bbox_x_max,  # Bounding box
                bbox_height, bbox_width  # Bounding box dimensions
            ], dtype=np.float32)

            # Concatenate image feature with spatial features
            instance_feature = np.concatenate([img_feature, spatial_features])

            # Compute instance uncertainty
            instance_uncertainty = _compute_instance_uncertainty(
                pred_mask, instance_mask, uncertainty_map)

            # Store instance
            instance_idx = len(all_instances)
            all_instances.append({
                'image_idx': img_idx,
                'instance_mask': instance_mask,  # Keep for potential future use
                'feature': instance_feature,
                'uncertainty': instance_uncertainty
            })
            image_instance_indices.append(instance_idx)

        image_to_instances[img_idx] = image_instance_indices

    print(f"✅ TAUDIS: Extracted {len(all_instances)} instances from {len(available_indices)} images.")

    if len(all_instances) == 0:
        print("⚠️ TAUDIS: No instances found. Falling back to random selection.")
        return rng.choice(available_indices, size=num_samples, replace=False).tolist()

    # Step 1: Select top αB uncertain instances
    alpha_B = int(alpha * num_samples)
    # Use numpy array for efficient sorting
    instance_uncertainties = np.array([inst['uncertainty'] for inst in all_instances])
    top_uncertain_indices = np.argsort(instance_uncertainties)[-alpha_B:][::-1]
    top_uncertain_indices = top_uncertain_indices.tolist()

    if len(top_uncertain_indices) == 0:
        print("⚠️ TAUDIS: No uncertain instances found. Falling back to random selection.")
        return rng.choice(available_indices, size=num_samples, replace=False).tolist()

    # Step 2: Construct similarity matrix between top uncertain instances and all instances
    # S[i, j] = cosine similarity between uncertain instance i and all instance j
    print(
        f"🔍 TAUDIS: Processing {len(top_uncertain_indices)} uncertain instances from {len(all_instances)} total instances...")

    # Pre-allocate arrays for efficiency
    TC_features = np.array([all_instances[i]['feature'] for i in top_uncertain_indices])  # [αB, D]
    TF_features = np.array([inst['feature'] for inst in all_instances])  # [N_instances, D]

    # Compute cosine similarity matrix efficiently
    # Normalize features first (vectorized)
    TC_norms = np.linalg.norm(TC_features, axis=1, keepdims=True)
    TC_norms = np.maximum(TC_norms, 1e-10)  # Avoid division by zero
    TC_features_norm = TC_features / TC_norms

    TF_norms = np.linalg.norm(TF_features, axis=1, keepdims=True)
    TF_norms = np.maximum(TF_norms, 1e-10)
    TF_features_norm = TF_features / TF_norms

    # Compute similarity matrix (optimized: use matrix multiplication)
    print(f"🔍 TAUDIS: Computing similarity matrix ({len(TC_features_norm)} x {len(TF_features_norm)})...")
    similarity_matrix = np.dot(TC_features_norm, TF_features_norm.T)  # [αB, N_instances]
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)  # Ensure valid cosine similarity
    print(f"✅ TAUDIS: Similarity matrix computed.")

    # Step 3: Maximum k-set cover to select βB diverse instances
    beta_B = int(beta * num_samples)
    beta_B = min(beta_B, len(top_uncertain_indices))

    print(f"🔍 TAUDIS: Running maximum k-set cover to select {beta_B} diverse instances...")
    selected_instance_indices = _maximum_k_set_cover(similarity_matrix, beta_B, threshold=sigma)
    print(f"✅ TAUDIS: Selected {len(selected_instance_indices)} diverse instances.")

    # Map back to actual instance indices in all_instances
    selected_instances = [top_uncertain_indices[i] for i in selected_instance_indices]

    # Step 4: Select B images containing the most selected instances (majority vote)
    # Use Counter-like approach with dict for efficiency
    image_scores = {}
    for inst_idx in selected_instances:
        img_idx = all_instances[inst_idx]['image_idx']
        image_scores[img_idx] = image_scores.get(img_idx, 0) + 1

    # Sort images by number of selected instances (use heap for top-k if needed, but sorting is fine for small k)
    sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)

    # Select top B images
    selected_images = [img_idx for img_idx, _ in sorted_images[:num_samples]]

    print(f"✅ TAUDIS: Selected {len(selected_images)} images from {len(image_scores)} candidate images.")

    # If we don't have enough images, add remaining randomly
    if len(selected_images) < num_samples:
        remaining = [idx for idx in available_indices if idx not in selected_images]
        needed = num_samples - len(selected_images)
        if remaining:
            additional = rng.choice(remaining, size=min(needed, len(remaining)), replace=False).tolist()
            selected_images.extend(additional)
            print(f"⚠️ TAUDIS: Added {len(additional)} random images to reach {num_samples} samples.")

    return selected_images
