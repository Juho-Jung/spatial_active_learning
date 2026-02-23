#!/usr/bin/env python3
"""
Uncertainty calculation methods for active learning.

This module contains various uncertainty estimation methods used in active learning
experiments, including base uncertainty and Monte Carlo dropout uncertainty.
"""

import numpy as np
import torch


def calculate_uncertainty_base(model, dataloader, device, return_detailed=False, return_features=False):
    """
    Calculate base uncertainty (1 - 2*|pred - 0.5|).

    This method calculates uncertainty based on prediction confidence,
    where predictions closer to 0.5 (the decision boundary) are considered
    more uncertain.

    Args:
        model: The trained model for prediction
        dataloader: DataLoader containing the data to evaluate
        device: Device to run the model on
        return_detailed: If True, return detailed predictions and uncertainties

    Returns:
        np.array or dict: Array of uncertainty scores for each sample, or dict with detailed info
    """
    model.eval()
    uncertainties = []
    all_predictions = []
    all_uncertainties = []
    all_sam_predictions = []
    all_features = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)

            if return_detailed:
                # Get both custom decoder prediction and SAM original prediction
                if return_features:
                    pred_masks, sam_predictions, features = model(
                        images, return_sam_prediction=True, return_features=True)
                    all_features.append(features.cpu().numpy())
                else:
                    pred_masks, sam_predictions = model(images, return_sam_prediction=True)
                pred_masks = pred_masks.squeeze(1).cpu().numpy()
                sam_predictions = sam_predictions.squeeze(1).cpu().numpy()
            elif return_features:
                pred_masks, features = model(images, return_features=True)
                pred_masks = pred_masks.squeeze(1).cpu().numpy()
                all_features.append(features.cpu().numpy())
            else:
                pred_masks = model(images)
                pred_masks = pred_masks.squeeze(1).cpu().numpy()

            # Calculate uncertainty as prediction confidence
            uncertainty = 1.0 - 2.0 * np.abs(pred_masks - 0.5)
            image_uncertainties = np.mean(uncertainty, axis=(1, 2))  # Average over spatial dimensions
            uncertainties.extend(image_uncertainties)

            if return_detailed:
                all_predictions.append(pred_masks)
                all_uncertainties.append(uncertainty)
                all_sam_predictions.append(sam_predictions)

    if return_detailed:
        result = {
            'uncertainties': np.array(uncertainties),
            'predictions': np.concatenate(all_predictions, axis=0) if all_predictions else np.array([]),
            'pixel_uncertainties': np.concatenate(all_uncertainties, axis=0) if all_uncertainties else np.array([]),
            'sam_predictions': np.concatenate(all_sam_predictions, axis=0) if all_sam_predictions else np.array([]),
            # Use SAM predictions as lesionness
            'lesionness': np.concatenate(all_sam_predictions, axis=0) if all_sam_predictions else np.array([])
        }
        if return_features and all_features:
            result['features'] = np.concatenate(all_features, axis=0)
        return result
    elif return_features and all_features:
        return {
            'uncertainties': np.array(uncertainties),
            'features': np.concatenate(all_features, axis=0)
        }
    else:
        return np.array(uncertainties)


def calculate_uncertainty_mc_dropout(model, dataloader, device, T=8, return_detailed=False, return_features=False):
    """
    Calculate Monte Carlo Dropout uncertainty.

    This method uses Monte Carlo sampling with dropout enabled to estimate
    model uncertainty. Higher entropy in predictions indicates higher uncertainty.

    Args:
        model: The trained model with dropout layers
        dataloader: DataLoader containing the data to evaluate
        device: Device to run the model on
        T: Number of Monte Carlo samples (default: 8)
        return_detailed: If True, return detailed predictions and uncertainties
        return_features: If True, return feature embeddings

    Returns:
        np.array or dict: Array of uncertainty scores for each sample, or dict with detailed info
    """
    model.train()  # Enable dropout
    uncertainties = []
    all_predictions = []
    all_uncertainties = []
    all_features = []

    for images, masks in dataloader:
        images = images.to(device)
        batch_uncertainties = []

        # MC Dropout sampling
        pred_samples = []
        for _ in range(T):
            with torch.no_grad():
                if return_features:
                    pred, features = model(images, return_features=True)
                    if _ == 0:  # Only store features from first sample
                        all_features.append(features.cpu().numpy())
                else:
                    pred = model(images)
                pred_samples.append(pred.squeeze(1).cpu().numpy())

        # Calculate mean prediction and entropy
        pred_samples = np.array(pred_samples)  # [T, B, H, W]
        mean_pred = np.mean(pred_samples, axis=0)  # [B, H, W]

        # Calculate entropy: H = -p*log(p) - (1-p)*log(1-p)
        epsilon = 1e-8
        entropy = -mean_pred * np.log(mean_pred + epsilon) - (1 - mean_pred) * np.log(1 - mean_pred + epsilon)

        # Image score: mean of top-10% pixel entropies
        for i in range(len(images)):
            img_entropy = entropy[i].flatten()
            top_10_percent = int(0.1 * len(img_entropy))
            top_entropies = np.sort(img_entropy)[-top_10_percent:]
            image_uncertainty = np.mean(top_entropies)
            batch_uncertainties.append(image_uncertainty)

        uncertainties.extend(batch_uncertainties)

        if return_detailed:
            all_predictions.append(mean_pred)
            all_uncertainties.append(entropy)

    model.eval()  # Disable dropout

    if return_detailed:
        result = {
            'uncertainties': np.array(uncertainties),
            'predictions': np.concatenate(all_predictions, axis=0) if all_predictions else np.array([]),
            'pixel_uncertainties': np.concatenate(all_uncertainties, axis=0) if all_uncertainties else np.array([])
        }
        if return_features and all_features:
            result['features'] = np.concatenate(all_features, axis=0)
        return result
    elif return_features and all_features:
        return {
            'uncertainties': np.array(uncertainties),
            'features': np.concatenate(all_features, axis=0)
        }
    else:
        return np.array(uncertainties)


def calculate_uncertainty_tta(model, dataloader, device, T=4, return_detailed=False):
    """
    Calculate Test Time Augmentation (TTA) uncertainty.

    This method uses TTA to estimate model uncertainty by applying different
    augmentations and measuring prediction variance.

    Args:
        model: The trained model
        dataloader: DataLoader containing the data to evaluate
        device: Device to run the model on
        T: Number of TTA samples (default: 4)
        return_detailed: If True, return detailed predictions and uncertainties

    Returns:
        np.array or dict: Array of uncertainty scores for each sample, or dict with detailed info
    """
    model.eval()
    uncertainties = []
    all_predictions = []
    all_uncertainties = []

    # Define TTA transforms
    def apply_tta_transform(images, transform_type):
        if transform_type == 'original':
            return images
        elif transform_type == 'flip_h':
            return torch.flip(images, dims=[3])
        elif transform_type == 'flip_v':
            return torch.flip(images, dims=[2])
        elif transform_type == 'rotate_90':
            return torch.rot90(images, k=1, dims=[2, 3])
        else:
            return images

    tta_transforms = ['original', 'flip_h', 'flip_v', 'rotate_90'][:T]

    for images, masks in dataloader:
        images = images.to(device)
        batch_uncertainties = []

        # TTA sampling
        pred_samples = []
        for transform_type in tta_transforms:
            with torch.no_grad():
                tta_images = apply_tta_transform(images, transform_type)
                pred = model(tta_images)

                # Apply inverse transform to predictions
                if transform_type == 'flip_h':
                    pred = torch.flip(pred, dims=[3])
                elif transform_type == 'flip_v':
                    pred = torch.flip(pred, dims=[2])
                elif transform_type == 'rotate_90':
                    pred = torch.rot90(pred, k=-1, dims=[2, 3])

                pred_samples.append(pred.squeeze(1).cpu().numpy())

        # Calculate mean prediction and variance
        pred_samples = np.array(pred_samples)  # [T, B, H, W]
        mean_pred = np.mean(pred_samples, axis=0)  # [B, H, W]
        pred_variance = np.var(pred_samples, axis=0)  # [B, H, W]

        # Image score: mean of top-10% pixel variances
        for i in range(len(images)):
            img_variance = pred_variance[i].flatten()
            top_10_percent = int(0.1 * len(img_variance))
            top_variances = np.sort(img_variance)[-top_10_percent:]
            image_uncertainty = np.mean(top_variances)
            batch_uncertainties.append(image_uncertainty)

        uncertainties.extend(batch_uncertainties)

        if return_detailed:
            all_predictions.append(mean_pred)
            all_uncertainties.append(pred_variance)

    if return_detailed:
        return {
            'uncertainties': np.array(uncertainties),
            'predictions': np.concatenate(all_predictions, axis=0) if all_predictions else np.array([]),
            'pixel_uncertainties': np.concatenate(all_uncertainties, axis=0) if all_uncertainties else np.array([])
        }
    else:
        return np.array(uncertainties)


def calculate_uncertainty_fast_lesionness(model, dataloader, device, return_detailed=False):
    """
    Calculate fast lesionness using SAMLesionModel predictions.

    This method uses the SAMLesionModel's own predictions as lesionness,
    avoiding the need for a separate teacher model.

    Args:
        model: The trained SAMLesionModel
        dataloader: DataLoader containing the data to evaluate
        device: Device to run the model on
        return_detailed: If True, return detailed predictions and uncertainties

    Returns:
        np.array or dict: Array of uncertainty scores for each sample, or dict with detailed info
    """
    model.eval()
    uncertainties = []
    all_predictions = []
    all_lesionness = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)

            # Get predictions and lesionness from SAMLesionModel
            if return_detailed:
                pred_masks, sam_predictions = model(images, return_sam_prediction=True)
                pred_masks = pred_masks.squeeze(1).cpu().numpy()
                sam_predictions = sam_predictions.squeeze(1).cpu().numpy()

                # Use SAM predictions as lesionness
                lesionness = sam_predictions

                # Calculate uncertainty as prediction confidence
                uncertainty = 1.0 - 2.0 * np.abs(pred_masks - 0.5)
                image_uncertainties = np.mean(uncertainty, axis=(1, 2))

                all_predictions.append(pred_masks)
                all_lesionness.append(lesionness)
            else:
                pred_masks = model(images)
                pred_masks = pred_masks.squeeze(1).cpu().numpy()

                # Calculate uncertainty as prediction confidence
                uncertainty = 1.0 - 2.0 * np.abs(pred_masks - 0.5)
                image_uncertainties = np.mean(uncertainty, axis=(1, 2))

            uncertainties.extend(image_uncertainties)

    if return_detailed:
        return {
            'uncertainties': np.array(uncertainties),
            'predictions': np.concatenate(all_predictions, axis=0) if all_predictions else np.array([]),
            'lesionness': np.concatenate(all_lesionness, axis=0) if all_lesionness else np.array([]),
            # Same as lesionness for compatibility
            'sam_predictions': np.concatenate(all_lesionness, axis=0) if all_lesionness else np.array([])
        }
    else:
        return np.array(uncertainties)


def calculate_uncertainty_none(model, dataloader, device, return_detailed=False, return_features=False):
    """
    Calculate uncertainty using only entropy (no lesionness multiplication).

    This method calculates pixel-level entropy and uses it directly as uncertainty,
    without multiplying by lesionness values.

    Args:
        dataloader: DataLoader containing the data to evaluate
        device: Device to run the model on
        return_detailed: If True, return detailed predictions and uncertainties

    Returns:
        np.array or dict: Array of uncertainty scores for each sample, or dict with detailed info
    """
    model.eval()
    uncertainties = []
    all_predictions = []
    all_entropies = []
    all_features = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)

            # Get predictions from model
            pred_masks = model(images)
            pred_masks = pred_masks.squeeze(1).cpu().numpy()
            if return_features:
                pred_masks, features = model(images, return_features=True)
                pred_masks = pred_masks.squeeze(1).cpu().numpy()
                all_features.append(features.cpu().numpy())
            else:
                pred_masks = model(images)
                pred_masks = pred_masks.squeeze(1).cpu().numpy()

            # Calculate pixel-level entropy: H = -p*log(p) - (1-p)*log(1-p)
            epsilon = 1e-8
            p_safe = np.clip(pred_masks, epsilon, 1 - epsilon)
            # Replace NaN/Inf to keep logs finite
            p_safe = np.nan_to_num(p_safe, nan=0.5, posinf=1 - epsilon, neginf=epsilon)
            pixel_entropy = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))

            # Image-level uncertainty: mean of top-10% pixel entropies (with guards)
            image_uncertainties = []
            for i in range(len(images)):
                img_entropy = pixel_entropy[i].flatten()
                # ensure at least one element
                top_10_percent = int(0.1 * len(img_entropy))
                top_k = max(1, top_10_percent)
                # sanitize entropy array
                img_entropy = np.nan_to_num(img_entropy, nan=0.0, posinf=0.0, neginf=0.0)
                top_entropies = np.sort(img_entropy)[-top_k:]
                image_uncertainty = float(np.mean(top_entropies))
                if not np.isfinite(image_uncertainty):
                    image_uncertainty = float(np.mean(img_entropy))
                image_uncertainties.append(image_uncertainty)

            uncertainties.extend(image_uncertainties)

            if return_detailed:
                all_predictions.append(pred_masks)
                all_entropies.append(pixel_entropy)

    if return_detailed:
        predictions = np.concatenate(all_predictions, axis=0) if all_predictions else np.array([])
        result = {
            'uncertainties': np.array(uncertainties),
            'predictions': predictions,
            'entropies': np.concatenate(all_entropies, axis=0) if all_entropies else np.array([]),
            # For 'none' type, use uniform lesionness (all ones) so w(i) = M(i)*H(i) = H(i)
            'lesionness': np.ones_like(predictions) if len(predictions) > 0 else np.array([])
        }
        if return_features and all_features:
            result['features'] = np.concatenate(all_features, axis=0)
        return result
    elif return_features and all_features:
        return {
            'uncertainties': np.array(uncertainties),
            'features': np.concatenate(all_features, axis=0)
        }
    else:
        return np.array(uncertainties)


def calculate_uncertainty_detection(model, dataloader, device, return_detailed=False, return_features=False):
    """
    Calculate uncertainty for detection models.

    For detection, uncertainty is based on:
    1. Prediction confidence (scores)
    2. Number of detections
    3. Box localization confidence

    Args:
        model: Detection model
        dataloader: DataLoader with detection data
        device: Device to run model on
        return_detailed: If True, return detailed predictions
        return_features: If True, return features

    Returns:
        np.array or dict: Uncertainty scores or detailed results
    """
    model.eval()
    uncertainties = []
    all_predictions = []
    all_features = []
    image_sizes = []  # Initialize image_sizes for return_detailed mode

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                images, targets = batch
            else:
                images = batch[0]
                targets = None

            # Convert to list format for detection models
            if isinstance(images, torch.Tensor):
                if images.dim() == 4:
                    # Store image sizes before converting
                    if return_detailed:
                        for img in images:
                            _, H, W = img.shape
                            image_sizes.append((H, W))
                    images = [img for img in images]
                else:
                    if return_detailed:
                        _, H, W = images.shape
                        image_sizes.append((H, W))
                    images = [images]
            else:
                # If already a list, get sizes from first image
                if return_detailed and len(images) > 0:
                    if torch.is_tensor(images[0]):
                        _, H, W = images[0].shape
                        image_sizes.extend([(H, W)] * len(images))

            images = [img.to(device) for img in images]

            # Get predictions
            if return_features:
                predictions, features = model(images, return_features=True)
                all_features.append(features.cpu().numpy())
            else:
                predictions = model(images)

            # Calculate uncertainty for each image
            for pred in predictions:
                boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else pred['boxes']
                scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else pred['scores']

                if len(scores) == 0:
                    # No detections: high uncertainty
                    uncertainty = 1.0
                else:
                    # Uncertainty based on:
                    # 1. Average confidence (lower = more uncertain)
                    # 2. Number of detections (too many or too few = uncertain)
                    avg_confidence = np.mean(scores)
                    num_detections = len(scores)

                    # Normalize detection count uncertainty (optimal around 1-3 detections)
                    if num_detections == 0:
                        count_uncertainty = 1.0
                    elif num_detections <= 3:
                        count_uncertainty = 0.2  # Low uncertainty for reasonable count
                    else:
                        count_uncertainty = min(0.5 + (num_detections - 3) * 0.1, 1.0)

                    # Combined uncertainty
                    confidence_uncertainty = 1.0 - avg_confidence
                    uncertainty = 0.6 * confidence_uncertainty + 0.4 * count_uncertainty

                uncertainties.append(uncertainty)

            if return_detailed:
                all_predictions.append(predictions)

    if return_detailed:
        # For detection: create probs and lesionness from bbox predictions
        # This allows SPARCL to work with detection models
        probs_list = []
        lesionness_list = []

        # Use stored image sizes or infer from bboxes
        pred_idx = 0
        for batch_idx, batch_predictions in enumerate(all_predictions):
            for pred in batch_predictions:
                boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else pred['boxes']
                scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else pred['scores']

                # Get image size from stored sizes or infer from bboxes
                if pred_idx < len(image_sizes):
                    H, W = image_sizes[pred_idx]
                elif len(boxes) > 0:
                    # Infer from bboxes
                    max_x = int(np.max(boxes[:, 2])) if len(boxes) > 0 else 512
                    max_y = int(np.max(boxes[:, 3])) if len(boxes) > 0 else 512
                    # Round up to nearest reasonable size
                    H = max(512, ((max_y // 32) + 1) * 32)
                    W = max(512, ((max_x // 32) + 1) * 32)
                else:
                    # Default size
                    H, W = 512, 512

                pred_idx += 1

                # Create probability maps from bbox predictions
                # probs: prediction probability map (bbox areas = scores, elsewhere = 0)
                # lesionness: same as probs for detection (bbox = lesion probability)
                prob_map = np.zeros((H, W), dtype=np.float32)
                lesion_map = np.zeros((H, W), dtype=np.float32)

                # Fill bbox areas with scores
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)

                    if x2 > x1 and y2 > y1:
                        # Use maximum score if boxes overlap
                        prob_map[y1:y2, x1:x2] = np.maximum(prob_map[y1:y2, x1:x2], score)
                        lesion_map[y1:y2, x1:x2] = np.maximum(lesion_map[y1:y2, x1:x2], score)

                probs_list.append(prob_map)
                lesionness_list.append(lesion_map)

        # Flatten all_predictions from nested list to flat list
        flat_predictions = []
        for batch_predictions in all_predictions:
            flat_predictions.extend(batch_predictions)

        result = {
            'uncertainties': np.array(uncertainties),
            'predictions': flat_predictions,  # Flattened detection predictions (one dict per image)
            'probs': np.array(probs_list) if probs_list else np.array([]),  # Bbox-based probability maps
            'lesionness': np.array(lesionness_list) if lesionness_list else np.array([]),  # Bbox-based lesionness maps
        }
        if return_features and all_features:
            result['features'] = np.concatenate(all_features, axis=0)
        return result
    elif return_features and all_features:
        return {
            'uncertainties': np.array(uncertainties),
            'features': np.concatenate(all_features, axis=0)
        }
    else:
        return np.array(uncertainties)


# Uncertainty method mapping
UNCERTAINTY_METHODS = {
    'base': calculate_uncertainty_base,
    'mc_dropout': calculate_uncertainty_mc_dropout,
    'tta': calculate_uncertainty_tta,
    'fast_lesionness': calculate_uncertainty_fast_lesionness,
    'none': calculate_uncertainty_none,
    'detection': calculate_uncertainty_detection,
}


def get_uncertainty_method(method_name):
    """Get uncertainty calculation method by name."""
    if method_name not in UNCERTAINTY_METHODS:
        raise ValueError(f"Unknown uncertainty method: {method_name}. "
                         f"Available methods: {list(UNCERTAINTY_METHODS.keys())}")
    return UNCERTAINTY_METHODS[method_name]
