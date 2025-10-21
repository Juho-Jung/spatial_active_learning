"""
Common evaluation functions for segmentation tasks.

This module provides comprehensive evaluation capabilities for segmentation models
including pixel-level, lesion-level, and image-level metrics.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import SegmentationDataset
from .models import create_model
from .utils import get_device


def _calculate_pixel_level_metrics(pred: torch.Tensor, target: torch.Tensor,
                                   threshold: float = 0.5) -> Dict[str, float]:
    """Calculate comprehensive pixel-level metrics."""
    smooth = 1e-6

    pred_flat = pred.view(-1)
    pred_binary = (pred > threshold).float()

    # Basic metrics
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection

    dice = (2 * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)
    iou = intersection / (union + smooth)

    # Additional metrics
    tp = (pred_binary * target).sum()
    fp = pred_binary.sum() - tp
    fn = target.sum() - tp
    target_flat = target.view(-1)
    tn = len(target_flat) - tp - fp - fn

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    specificity = tn / (tn + fp + smooth)
    accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)
    f1_score = (2 * precision * recall) / (precision + recall + smooth)

    return {'dice': dice.item(),
            'iou': iou.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'specificity': specificity.item(),
            'accuracy': accuracy.item(),
            'f1_score': f1_score.item()
            }


def _calculate_lesion_level_metrics(preds: List[torch.Tensor], targets: List[torch.Tensor],
                                    iou_threshold: float = 0.1) -> Dict[str, float]:
    """Calculate lesion-level metrics."""
    smooth = 1e-6

    iou_based_detection_preds = []
    iou_based_detection_targets = []

    for pred, target in zip(preds, targets):
        pred_flat = pred.view(-1)
        threshold = torch.quantile(pred_flat, 0.5)
        threshold = max(threshold.item(), 0.001)

        pred_binary = (pred > threshold).float()
        has_gt_lesion = target.sum() > 0

        if has_gt_lesion:
            iou_based_detection_targets.append(1)

            intersection = (pred_binary * target).sum()
            union = pred_binary.sum() + target.sum() - intersection
            lesion_iou = intersection / (union + smooth)

            lesion_detected = lesion_iou >= iou_threshold
            iou_based_detection_preds.append(1 if lesion_detected else 0)
        else:
            iou_based_detection_targets.append(0)
            iou_based_detection_preds.append(0)

    if len(iou_based_detection_targets) > 0:
        lesion_auroc = roc_auc_score(iou_based_detection_targets, iou_based_detection_preds)
        lesion_ap = average_precision_score(iou_based_detection_targets, iou_based_detection_preds)
    else:
        lesion_auroc = 0.5
        lesion_ap = 0.0

    return {'lesion_auroc': lesion_auroc,
            'lesion_ap': lesion_ap,
            'lesion_detection_accuracy': np.mean(np.array(iou_based_detection_preds) == np.array(iou_based_detection_targets))
            }


def _calculate_image_level_metrics(preds: List[torch.Tensor], targets: List[torch.Tensor],
                                   threshold: float = 0.5) -> Dict[str, float]:
    """Calculate image-level metrics."""
    image_level_preds = []
    image_level_targets = []

    for pred, target in zip(preds, targets):
        image_pred = pred.max().item()
        image_level_preds.append(image_pred)

        image_target = 1 if target.sum() > 0 else 0
        image_level_targets.append(image_target)

    if len(set(image_level_targets)) > 1:
        image_auroc = roc_auc_score(image_level_targets, image_level_preds)
        image_ap = average_precision_score(image_level_targets, image_level_preds)
    else:
        image_auroc = 0.5
        image_ap = 0.0

    return {'image_auroc': image_auroc,
            'image_ap': image_ap
            }


def _save_evaluation_results(results: Dict, output_dir: str, finding_name: str) -> None:
    """Save evaluation results to JSON file."""
    results_path = os.path.join(output_dir, f"{finding_name}_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)


def _plot_metrics(results: Dict, save_path: Optional[str] = None) -> None:
    """Create visualization plots for evaluation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Pixel-level metrics
    pixel_metrics = results['pixel_level']
    metrics_names = list(pixel_metrics.keys())
    metrics_values = list(pixel_metrics.values())

    axes[0, 0].bar(metrics_names, metrics_values)
    axes[0, 0].set_title('Pixel-level Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Lesion-level metrics
    lesion_metrics = results['lesion_level']
    lesion_names = list(lesion_metrics.keys())
    lesion_values = list(lesion_metrics.values())

    axes[0, 1].bar(lesion_names, lesion_values)
    axes[0, 1].set_title('Lesion-level Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Image-level metrics
    image_metrics = results['image_level']
    image_names = list(image_metrics.keys())
    image_values = list(image_metrics.values())

    axes[1, 0].bar(image_names, image_values)
    axes[1, 0].set_title('Image-level Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Summary
    axes[1, 1].text(0.1, 0.8, f"Total Samples: {results['num_samples']}", fontsize=12)
    axes[1, 1].text(0.1, 0.6, f"Best Dice: {results['pixel_level']['dice']:.4f}", fontsize=12)
    axes[1, 1].text(0.1, 0.4, f"Best IoU: {results['pixel_level']['iou']:.4f}", fontsize=12)
    axes[1, 1].set_title('Summary')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


class SegmentationEvaluator:
    """Unified evaluator for segmentation tasks."""

    def __init__(self, finding_name: str = "calcification", device: Optional[torch.device] = None):
        """
        Initialize the evaluator.

        Args:
            finding_name: The finding name to evaluate ('calcification' or 'calcifiednodule')
            device: Device to use for evaluation
        """
        self.finding_name = finding_name
        self.device = device or get_device()
        self.model: Optional[torch.nn.Module] = None

    def load_model(self, model_path: str, model_type: str = 'smp_efficientnet') -> None:
        """
        Load a trained model.

        Args:
            model_path: Path to the model checkpoint
            model_type: Type of model to load
        """
        print(f"🔍 Loading model: {model_path}")

        self.model = create_model(model_type, self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        print(f"✅ Model loaded successfully!")

    def evaluate_dataset(self, dataset: SegmentationDataset, batch_size: int = 8,
                         save_results: bool = True, output_dir: Optional[str] = None) -> Dict:
        """
        Evaluate the model on a dataset.

        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            save_results: Whether to save results
            output_dir: Directory to save results

        Returns:
            dict: Evaluation results
        """
        print(f"🔍 Evaluating on {len(dataset)} samples...")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Evaluation
        all_preds = []
        all_targets = []

        self.model.eval()
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                preds = self.model(images)
                all_preds.extend(preds.cpu())
                all_targets.extend(masks.cpu())

        # Calculate metrics
        print("📊 Calculating metrics...")

        # Pixel-level metrics
        pixel_metrics = []
        for pred, target in zip(all_preds, all_targets):
            metrics = _calculate_pixel_level_metrics(pred, target)
            pixel_metrics.append(metrics)

        # Average pixel-level metrics
        avg_pixel_metrics = {
            key: np.mean([m[key] for m in pixel_metrics])
            for key in pixel_metrics[0].keys()
        }

        # Lesion-level metrics
        lesion_metrics = _calculate_lesion_level_metrics(all_preds, all_targets)

        # Image-level metrics
        image_metrics = _calculate_image_level_metrics(all_preds, all_targets)

        # Combine all metrics
        results = {'pixel_level': avg_pixel_metrics,
                   'lesion_level': lesion_metrics,
                   'image_level': image_metrics,
                   'num_samples': len(dataset),
                   'timestamp': datetime.now().isoformat()
                   }

        # Save results
        if save_results:
            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"evaluation_results_{self.finding_name}_{timestamp}"

            os.makedirs(output_dir, exist_ok=True)
            _save_evaluation_results(results, output_dir, self.finding_name)
            print(f"📊 Results saved to {output_dir}")

        return results

    def evaluate_validation_collection(self, include_train_collection: bool = True,
                                       batch_size: int = 8) -> Dict:
        """Evaluate on validation collection."""
        print(f"🔍 Evaluating on validation collection...")
        print(f"📊 Include train collection: {include_train_collection}")

        dataset = SegmentationDataset(
            finding_name=self.finding_name,
            train_collection='validation_collection' if not include_train_collection else 'both'
        )

        return self.evaluate_dataset(dataset, batch_size)

    def evaluate_train_collection(self, batch_size: int = 8) -> Dict:
        """Evaluate on train collection."""
        print(f"🔍 Evaluating on train collection...")

        dataset = SegmentationDataset(finding_name=self.finding_name,
                                      train_collection='train_collection_with_consensus'
                                      )

        return self.evaluate_dataset(dataset, batch_size)

    def plot_metrics(self, results: Dict, save_path: Optional[str] = None) -> None:
        """Plot evaluation metrics."""
        _plot_metrics(results, save_path)


def evaluate_model(model_path: str, finding_name: str = "calcification",
                   model_type: str = "smp_efficientnet", include_train_collection: bool = True,
                   batch_size: int = 8, save_results: bool = True) -> Dict:
    """
    Convenience function to evaluate a model.

    Args:
        model_path: Path to the model checkpoint
        finding_name: The finding name to evaluate
        model_type: Type of model to load
        include_train_collection: Whether to include train collection in validation
        batch_size: Batch size for evaluation
        save_results: Whether to save results

    Returns:
        dict: Evaluation results
    """
    evaluator = SegmentationEvaluator(finding_name=finding_name)
    evaluator.load_model(model_path, model_type)

    results = evaluator.evaluate_validation_collection(include_train_collection=include_train_collection,
                                                       batch_size=batch_size
                                                       )

    # Print results
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)

    print(f"\n🎯 Pixel-level Metrics:")
    for key, value in results['pixel_level'].items():
        print(f"   {key}: {value:.4f}")

    print(f"\n🎯 Lesion-level Metrics:")
    for key, value in results['lesion_level'].items():
        print(f"   {key}: {value:.4f}")

    print(f"\n🎯 Image-level Metrics:")
    for key, value in results['image_level'].items():
        print(f"   {key}: {value:.4f}")

    print(f"\n📊 Total samples: {results['num_samples']}")
    print("=" * 50)

    return results
