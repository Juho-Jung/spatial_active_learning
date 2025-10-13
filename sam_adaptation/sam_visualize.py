#!/usr/bin/env python3
"""
SAM Visualization for Lesion Segmentation

Standalone visualization script for trained SAM-based lesion segmentation models.
Loads a trained model and creates visualizations for test samples.
Supports different lesion types: calcification, nodule, etc.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader

# Import our modular components
from utils import set_seed
from models import SAMLesionModel
from dataset import SAMLesionDatasetSimple

# Set random seeds for reproducibility
set_seed(42)


def create_visualization(image, gt_mask, pred_mask, save_path, sample_id, image_path=""):
    """Create visualization with 1x3 subplot layout."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Convert tensors to numpy if needed
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(gt_mask):
        gt_mask = gt_mask.cpu().numpy()
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()

    # Handle different image formats
    if len(image.shape) == 3 and image.shape[0] == 3:
        # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))

    # Denormalize image
    image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)

    # Convert to grayscale
    if len(image.shape) == 3:
        image_gray = np.mean(image, axis=2)
    else:
        image_gray = image

    # Normalize masks to [0, 1]
    if gt_mask.max() > 1.0:
        gt_mask = gt_mask / 255.0
    if pred_mask.max() > 1.0:
        pred_mask = pred_mask / 255.0

    # Ensure masks are binary
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    pred_mask = (pred_mask > 0.5).astype(np.float32)

    # 1. Original image with GT overlay (red)
    axes[0].imshow(image_gray, cmap='gray')
    if gt_mask.sum() > 0:
        gt_overlay = np.zeros((*gt_mask.shape, 4))
        gt_overlay[:, :, 0] = gt_mask  # Red channel
        gt_overlay[:, :, 3] = gt_mask * 0.7  # Alpha channel
        axes[0].imshow(gt_overlay)
    axes[0].set_title(f'Sample {sample_id}: GT (Red)', fontsize=12)
    axes[0].axis('off')

    # 2. Original image with Prediction overlay (blue)
    axes[1].imshow(image_gray, cmap='gray')
    if pred_mask.sum() > 0:
        pred_overlay = np.zeros((*pred_mask.shape, 4))
        pred_overlay[:, :, 2] = pred_mask  # Blue channel
        pred_overlay[:, :, 3] = pred_mask * 0.7  # Alpha channel
        axes[1].imshow(pred_overlay)
    axes[1].set_title(f'Sample {sample_id}: Prediction (Blue)', fontsize=12)
    axes[1].axis('off')

    # 3. Original image with both GT and Prediction overlay
    axes[2].imshow(image_gray, cmap='gray')

    # GT overlay (red)
    if gt_mask.sum() > 0:
        gt_overlay = np.zeros((*gt_mask.shape, 4))
        gt_overlay[:, :, 0] = gt_mask  # Red channel
        gt_overlay[:, :, 3] = gt_mask * 0.5  # Lower alpha for overlap
        axes[2].imshow(gt_overlay)

    # Prediction overlay (blue)
    if pred_mask.sum() > 0:
        pred_overlay = np.zeros((*pred_mask.shape, 4))
        pred_overlay[:, :, 2] = pred_mask  # Blue channel
        pred_overlay[:, :, 3] = pred_mask * 0.5  # Lower alpha for overlap
        axes[2].imshow(pred_overlay)

    axes[2].set_title(f'Sample {sample_id}: GT (Red) + Prediction (Blue)', fontsize=12)
    axes[2].axis('off')

    # Add legend and title
    fig.suptitle(f'Calcification Segmentation Results - Sample {sample_id}', fontsize=14, fontweight='bold')

    # Create custom legend
    legend_elements = [
        patches.Patch(color='red', alpha=0.7, label='Ground Truth'),
        patches.Patch(color='blue', alpha=0.7, label='Model Prediction')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))

    # Add image path info
    if image_path:
        fig.text(0.02, 0.02, f'Image: {os.path.basename(image_path)}', fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_model(model_path, sam_checkpoint_path, device, vit_model='vit_b'):
    """Load trained model."""
    print(f"🔄 Loading model from: {model_path}")

    # Create model
    model = SAMLesionModel(sam_checkpoint_path, vit_model).to(device)

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    print(f"✅ Model loaded successfully!")

    return model


def visualize_samples(model, dataset, device, output_dir, num_samples=20):
    """Create visualizations for test samples."""
    model.eval()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get random samples
    total_samples = len(dataset)
    if num_samples > total_samples:
        num_samples = total_samples
        print(f"⚠️  Requested {num_samples} samples but only {total_samples} available. Using all samples.")

    test_indices = np.random.choice(total_samples, size=num_samples, replace=False)

    print(f"🎨 Creating visualizations for {len(test_indices)} samples...")

    with torch.no_grad():
        for i, idx in enumerate(test_indices):
            # Get sample
            image, gt_mask, image_path = dataset[idx]
            image = image.unsqueeze(0).to(device)  # Add batch dimension

            # Get prediction
            pred_mask = model(image)
            pred_mask = pred_mask.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

            # Create visualization
            save_path = os.path.join(output_dir, f'visualization_{idx:03d}.png')
            create_visualization(image.squeeze(0), gt_mask, pred_mask, save_path, idx, image_path)

            if (i + 1) % 5 == 0:
                print(f"   Processed {i + 1}/{len(test_indices)} samples...")

    print(f"✅ Visualizations saved to: {output_dir}")
    return output_dir


def calculate_uncertainty_batch(model, dataset, device, batch_size=8):
    """Calculate uncertainty scores for both pixels and lesions using batch inference."""
    model.eval()

    # Create data loader for batch processing
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # For pixel-level analysis
    all_pixel_uncertainties = []
    all_pixel_coords = []
    all_pixel_sample_ids = []

    # For lesion-level analysis
    all_lesion_uncertainties = []
    all_lesion_coords = []
    all_lesion_sample_ids = []

    print(f"🔄 Calculating pixel and lesion-level uncertainty for {len(dataset)} samples with batch size {batch_size}...")

    with torch.no_grad():
        for batch_idx, (images, gt_masks, image_paths) in enumerate(dataloader):
            images = images.to(device)
            gt_masks = gt_masks.cpu().numpy()

            # Get model predictions
            pred_masks = model(images)
            pred_masks = pred_masks.squeeze(1).cpu().numpy()  # Remove channel dimension

            # Calculate uncertainty for each sample in batch
            for i in range(len(images)):
                pred_mask = pred_masks[i]
                gt_mask = gt_masks[i]

                # Calculate uncertainty as prediction confidence
                # Higher uncertainty = closer to 0.5 (uncertain), lower uncertainty = closer to 0 or 1 (certain)
                uncertainty = 1.0 - 2.0 * np.abs(pred_mask - 0.5)  # Range: [0, 1]

                # Get GT mask coordinates
                gt_coords = np.where(gt_mask > 0.5)
                if len(gt_coords[0]) > 0:
                    # PIXEL-LEVEL ANALYSIS
                    # Sample coordinates (to avoid too many points)
                    num_points = min(1000, len(gt_coords[0]))  # Limit to 1000 points per sample
                    indices = np.random.choice(len(gt_coords[0]), num_points, replace=False)

                    y_coords = gt_coords[0][indices]
                    x_coords = gt_coords[1][indices]
                    pixel_uncertainties = uncertainty[y_coords, x_coords]

                    all_pixel_uncertainties.extend(pixel_uncertainties)
                    all_pixel_coords.extend(list(zip(x_coords, y_coords)))
                    all_pixel_sample_ids.extend([batch_idx * batch_size + i] * len(pixel_uncertainties))

                    # LESION-LEVEL ANALYSIS
                    # Calculate lesion-level uncertainty (average uncertainty within GT mask)
                    lesion_uncertainty = np.mean(uncertainty[gt_coords])

                    # Calculate lesion center coordinates
                    lesion_center_y = np.mean(gt_coords[0])
                    lesion_center_x = np.mean(gt_coords[1])

                    all_lesion_uncertainties.append(lesion_uncertainty)
                    all_lesion_coords.append([lesion_center_x, lesion_center_y])
                    all_lesion_sample_ids.append(batch_idx * batch_size + i)

            if (batch_idx + 1) % 10 == 0:
                print(f"   Processed {batch_idx + 1}/{len(dataloader)} batches...")

    print(f"✅ Calculated uncertainty for {len(all_pixel_uncertainties)} pixels and {len(all_lesion_uncertainties)} lesions")

    return {
        'pixel': {
            'uncertainties': np.array(all_pixel_uncertainties),
            'coords': np.array(all_pixel_coords),
            'sample_ids': np.array(all_pixel_sample_ids)
        },
        'lesion': {
            'uncertainties': np.array(all_lesion_uncertainties),
            'coords': np.array(all_lesion_coords),
            'sample_ids': np.array(all_lesion_sample_ids)
        }
    }


def create_pixel_uncertainty_scatter(uncertainties, pixel_coords, save_path, title="Pixel-Level Uncertainty Analysis"):
    """Create scatter plot of uncertainty vs pixel coordinates."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Normalize uncertainties to [0, 1] for color mapping
    if len(uncertainties) > 0:
        # Use relative normalization within the data
        min_unc = uncertainties.min()
        max_unc = uncertainties.max()
        if max_unc > min_unc:
            normalized_unc = (uncertainties - min_unc) / (max_unc - min_unc)
        else:
            normalized_unc = np.zeros_like(uncertainties)
    else:
        normalized_unc = uncertainties

    # Create scatter plot with small points for pixels
    scatter = ax.scatter(pixel_coords[:, 0], pixel_coords[:, 1],
                        c=normalized_unc,
                        cmap='RdYlGn_r',  # Red-Yellow-Green reversed (red=high uncertainty, green=low uncertainty)
                        alpha=0.6,
                        s=1,  # Small points for pixels
                        vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Pixel Uncertainty Score (0=Low, 1=High)', fontsize=12)

    # Set labels and title
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'{title}\nRed=High Uncertainty, Green=Low Uncertainty', fontsize=14, fontweight='bold')

    # Invert y-axis to match image coordinates
    ax.invert_yaxis()

    # Add statistics text
    if len(uncertainties) > 0:
        mean_unc = uncertainties.mean()
        std_unc = uncertainties.std()
        min_unc = uncertainties.min()
        max_unc = uncertainties.max()

        stats_text = f'Pixels: {len(uncertainties):,}\n'
        stats_text += f'Mean: {mean_unc:.3f}\n'
        stats_text += f'Std: {std_unc:.3f}\n'
        stats_text += f'Range: [{min_unc:.3f}, {max_unc:.3f}]'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Pixel uncertainty scatter plot saved: {save_path}")
    return save_path


def create_lesion_uncertainty_scatter(uncertainties, lesion_coords, save_path, title="Lesion-Level Uncertainty Analysis"):
    """Create scatter plot of uncertainty vs lesion center coordinates."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Normalize uncertainties to [0, 1] for color mapping
    if len(uncertainties) > 0:
        # Use relative normalization within the data
        min_unc = uncertainties.min()
        max_unc = uncertainties.max()
        if max_unc > min_unc:
            normalized_unc = (uncertainties - min_unc) / (max_unc - min_unc)
        else:
            normalized_unc = np.zeros_like(uncertainties)
    else:
        normalized_unc = uncertainties

    # Create scatter plot with larger points for lesions
    scatter = ax.scatter(lesion_coords[:, 0], lesion_coords[:, 1],
                        c=normalized_unc,
                        cmap='RdYlGn_r',  # Red-Yellow-Green reversed (red=high uncertainty, green=low uncertainty)
                        alpha=0.7,
                        s=50,  # Larger points for lesions
                        vmin=0, vmax=1,
                        edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Lesion Uncertainty Score (0=Low, 1=High)', fontsize=12)

    # Set labels and title
    ax.set_xlabel('X Coordinate (Lesion Center)', fontsize=12)
    ax.set_ylabel('Y Coordinate (Lesion Center)', fontsize=12)
    ax.set_title(f'{title}\nRed=High Uncertainty, Green=Low Uncertainty', fontsize=14, fontweight='bold')

    # Invert y-axis to match image coordinates
    ax.invert_yaxis()

    # Add statistics text
    if len(uncertainties) > 0:
        mean_unc = uncertainties.mean()
        std_unc = uncertainties.std()
        min_unc = uncertainties.min()
        max_unc = uncertainties.max()

        stats_text = f'Lesions: {len(uncertainties):,}\n'
        stats_text += f'Mean: {mean_unc:.3f}\n'
        stats_text += f'Std: {std_unc:.3f}\n'
        stats_text += f'Range: [{min_unc:.3f}, {max_unc:.3f}]'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Lesion uncertainty scatter plot saved: {save_path}")
    return save_path


def analyze_uncertainty(model, dataset, device, output_dir, batch_size=8):
    """Analyze model uncertainty across the entire dataset."""
    print(f"🔍 Starting uncertainty analysis...")

    # Calculate uncertainties for both pixel and lesion levels
    uncertainty_data = calculate_uncertainty_batch(model, dataset, device, batch_size)

    pixel_data = uncertainty_data['pixel']
    lesion_data = uncertainty_data['lesion']

    if len(pixel_data['uncertainties']) == 0 and len(lesion_data['uncertainties']) == 0:
        print("⚠️  No uncertainty data calculated. Check if dataset has GT masks.")
        return None

    # Create pixel-level uncertainty scatter plot
    if len(pixel_data['uncertainties']) > 0:
        pixel_scatter_path = os.path.join(output_dir, 'scatter_pixel_uncertainty.png')
        create_pixel_uncertainty_scatter(
            pixel_data['uncertainties'],
            pixel_data['coords'],
            pixel_scatter_path
        )

    # Create lesion-level uncertainty scatter plot
    if len(lesion_data['uncertainties']) > 0:
        lesion_scatter_path = os.path.join(output_dir, 'scatter_lesion_uncertainty.png')
        create_lesion_uncertainty_scatter(
            lesion_data['uncertainties'],
            lesion_data['coords'],
            lesion_scatter_path
        )

    # Create additional analysis plots (using lesion data for summary statistics)
    if len(lesion_data['uncertainties']) > 0:
        create_uncertainty_histogram(lesion_data['uncertainties'], output_dir)
        create_uncertainty_by_sample(lesion_data['uncertainties'], lesion_data['sample_ids'], output_dir)

    return {
        'pixel_scatter': pixel_scatter_path if len(pixel_data['uncertainties']) > 0 else None,
        'lesion_scatter': lesion_scatter_path if len(lesion_data['uncertainties']) > 0 else None
    }


def create_uncertainty_histogram(uncertainties, output_dir):
    """Create histogram of uncertainty distribution."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.hist(uncertainties, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Uncertainty Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Model Uncertainty Scores', fontsize=14, fontweight='bold')

    # Add statistics
    mean_unc = uncertainties.mean()
    std_unc = uncertainties.std()
    ax.axvline(mean_unc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_unc:.3f}')
    ax.axvline(mean_unc + std_unc, color='orange', linestyle='--', linewidth=1, label=f'+1σ: {mean_unc + std_unc:.3f}')
    ax.axvline(mean_unc - std_unc, color='orange', linestyle='--', linewidth=1, label=f'-1σ: {mean_unc - std_unc:.3f}')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Uncertainty histogram saved")


def create_uncertainty_by_sample(uncertainties, sample_ids, output_dir):
    """Create plot showing uncertainty by sample."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Calculate mean uncertainty per sample
    unique_samples = np.unique(sample_ids)
    sample_uncertainties = []

    for sample_id in unique_samples:
        sample_unc = uncertainties[sample_ids == sample_id]
        sample_uncertainties.append(sample_unc.mean())

    sample_uncertainties = np.array(sample_uncertainties)

    # Create bar plot
    bars = ax.bar(range(len(unique_samples)), sample_uncertainties,
                  color='lightcoral', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Mean Uncertainty Score', fontsize=12)
    ax.set_title('Mean Uncertainty Score by Sample', fontsize=14, fontweight='bold')

    # Add mean line
    overall_mean = sample_uncertainties.mean()
    ax.axhline(overall_mean, color='red', linestyle='--', linewidth=2,
               label=f'Overall Mean: {overall_mean:.3f}')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_by_sample.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Uncertainty by sample plot saved")


def main():
    parser = argparse.ArgumentParser(description='Visualize SAM-based Lesion Segmentation Results')
    parser.add_argument('--model_path', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--sam_checkpoint', default="/opt/pxi/projects/calcified_nodule/spatial_active_learning/sam_vit_b.pth",
                       help='Path to SAM checkpoint')
    parser.add_argument('--vit_model', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM ViT model size')
    parser.add_argument('--output_dir', default='./visualizations', help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--train_collection', default='validation_collection',
                       choices=['validation_collection', 'train_collection_with_consensus', 'both'],
                       help='Dataset collection to use')
    parser.add_argument('--data_limit', type=int, default=None, help='Limit number of samples in dataset')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--uncertainty_analysis', action='store_true', help='Perform uncertainty analysis on entire dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for uncertainty analysis')
    parser.add_argument('--target_lesion', default='calcification',
                       choices=['calcification', 'nodule', 'pneumonia'],
                       help='Target lesion type to segment')

    args = parser.parse_args()

    # Setup device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # Load model
    model = load_model(args.model_path, args.sam_checkpoint, device, args.vit_model)

    # Create dataset
    print(f"📊 Loading dataset...")
    dataset = SAMLesionDatasetSimple(
        train_collection=args.train_collection,
        limit=args.data_limit,
        target_lesion=args.target_lesion
    )

    # Create visualizations
    vis_dir = visualize_samples(model, dataset, device, args.output_dir, args.num_samples)

    # Perform uncertainty analysis if requested
    if args.uncertainty_analysis:
        print(f"\n🔍 Starting uncertainty analysis...")
        uncertainty_results = analyze_uncertainty(model, dataset, device, args.output_dir, args.batch_size)
        if uncertainty_results:
            print(f"✅ Uncertainty analysis completed!")
            if uncertainty_results['pixel_scatter']:
                print(f"📊 Pixel scatter plot: {uncertainty_results['pixel_scatter']}")
            if uncertainty_results['lesion_scatter']:
                print(f"📊 Lesion scatter plot: {uncertainty_results['lesion_scatter']}")
        else:
            print(f"⚠️  Uncertainty analysis failed or no data found.")

    print(f"\n🎉 Visualization completed!")
    print(f"📁 Output directory: {vis_dir}")
    print(f"📊 Generated {args.num_samples} visualizations")
    if args.uncertainty_analysis:
        print(f"📊 Uncertainty analysis: {'✅ Completed' if args.uncertainty_analysis else '❌ Skipped'}")


if __name__ == "__main__":
    main()
