#!/usr/bin/env python3
"""
Visualize detection results: images with bounding boxes.

This script loads VinDr-CXR dataset in detection mode and visualizes
images with their ground truth bounding boxes to verify alignment.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.vindr_cxr_dataset import VinDrCXRDataset


def visualize_detection_samples(dataset, num_samples=5, output_dir='./detection_visualizations'):
    """
    Visualize detection samples with bounding boxes.
    
    Args:
        dataset: VinDrCXRDataset in detection mode
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random indices directly (faster than DataLoader)
    import random
    random.seed(42)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected_indices = indices[:num_samples]
    
    print(f"📊 Selected {len(selected_indices)} samples from {len(dataset)} total images")
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, dataset_idx in enumerate(selected_indices):
        image, target = dataset[dataset_idx]
        
        # Convert image to numpy (if tensor)
        if torch.is_tensor(image):
            img = image.permute(1, 2, 0).cpu().numpy()
        else:
            img = image
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        # Get bounding boxes
        if torch.is_tensor(target['boxes']):
            boxes = target['boxes'].cpu().numpy()  # [N, 4]
        else:
            boxes = target['boxes']
        
        if torch.is_tensor(target['labels']):
            labels = target['labels'].cpu().numpy()  # [N]
        else:
            labels = target['labels']
        
        # Draw image
        ax = axes[idx]
        ax.imshow(img)
        ax.axis('off')
        
        # Color palette for different lesions
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'pink', 'brown']
        
        # Draw bounding boxes - ALL of them
        print(f"  📦 Image {idx+1}: Drawing {len(boxes)} bboxes...")
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Use different colors for different lesions
            color = colors[i % len(colors)]
            
            # Draw rectangle with thicker line for visibility
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=color, linewidth=2.5)
            ax.add_patch(rect)
            
            # Add label with offset to avoid overlap
            label_y_offset = -5 - (i % 3) * 15  # Stagger labels vertically
            ax.text(x1, y1 + label_y_offset, f'L{i+1}', color=color, fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
        
        # Add title
        image_id = dataset.get_image_id(dataset_idx)
        ax.set_title(f'Image ID: {image_id}\nTotal Bboxes: {len(boxes)}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'detection_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    plt.close()
    
    # Also save individual images
    print(f"\n📸 Saving individual images...")
    
    # Use same indices as above
    for idx, dataset_idx in enumerate(selected_indices):
        image, target = dataset[dataset_idx]
        
        # Convert image to numpy
        if torch.is_tensor(image):
            img = image.permute(1, 2, 0).cpu().numpy()
        else:
            img = image
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Get bounding boxes
        if torch.is_tensor(target['boxes']):
            boxes = target['boxes'].cpu().numpy()  # [N, 4]
        else:
            boxes = target['boxes']
        
        if torch.is_tensor(target['labels']):
            labels = target['labels'].cpu().numpy()  # [N]
        else:
            labels = target['labels']
        
        # Color palette for different lesions (BGR format for OpenCV)
        colors_bgr = [
            (0, 0, 255),    # red
            (255, 0, 0),    # blue
            (0, 255, 0),    # green
            (0, 255, 255),  # yellow
            (255, 255, 0),  # cyan
            (255, 0, 255),  # magenta
            (0, 165, 255),  # orange
            (128, 0, 128),  # purple
            (203, 192, 255), # pink
            (42, 42, 165)   # brown
        ]
        
        # Draw ALL bounding boxes with different colors
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Use different colors for different lesions
            color = colors_bgr[i % len(colors_bgr)]
            
            # Draw rectangle with thicker line
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
            
            # Add label with offset to avoid overlap
            label_text = f'L{i+1}'
            label_y = max(y1 - 5 - (i % 3) * 20, 15)  # Stagger labels, but keep above image
            cv2.putText(img_bgr, label_text, (x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Also draw box coordinates for debugging
            coord_text = f'({x1},{y1})-({x2},{y2})'
            cv2.putText(img_bgr, coord_text, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Save image
        image_id = dataset.get_image_id(dataset_idx)
        output_path = os.path.join(output_dir, f'{image_id}_bboxes.jpg')
        cv2.imwrite(output_path, img_bgr)
        print(f"  ✅ {image_id}: {len(boxes)} bboxes -> {output_path}")


def visualize_detection_samples_with_indices(dataset, selected_indices, output_dir='./detection_visualizations'):
    """
    Visualize specific samples by indices (faster, no random selection).
    
    Args:
        dataset: VinDrCXRDataset in detection mode
        selected_indices: List of dataset indices to visualize
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = len(selected_indices)
    print(f"📊 Visualizing {num_samples} samples...")
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, dataset_idx in enumerate(selected_indices):
        image, target = dataset[dataset_idx]
        
        # Convert image to numpy (if tensor)
        if torch.is_tensor(image):
            img = image.permute(1, 2, 0).cpu().numpy()
        else:
            img = image
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        # Get bounding boxes
        if torch.is_tensor(target['boxes']):
            boxes = target['boxes'].cpu().numpy()  # [N, 4]
        else:
            boxes = target['boxes']
        
        if torch.is_tensor(target['labels']):
            labels = target['labels'].cpu().numpy()  # [N]
        else:
            labels = target['labels']
        
        # Draw image
        ax = axes[idx]
        ax.imshow(img)
        ax.axis('off')
        
        # Color palette for different lesions
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'pink', 'brown']
        
        # Draw bounding boxes - ALL of them
        print(f"  📦 Image {idx+1}: Drawing {len(boxes)} bboxes...")
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Use different colors for different lesions
            color = colors[i % len(colors)]
            
            # Draw rectangle with thicker line for visibility
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=color, linewidth=2.5)
            ax.add_patch(rect)
            
            # Add label with offset to avoid overlap
            label_y_offset = -5 - (i % 3) * 15  # Stagger labels vertically
            ax.text(x1, y1 + label_y_offset, f'L{i+1}', color=color, fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
        
        # Add title
        image_id = dataset.get_image_id(dataset_idx)
        ax.set_title(f'Image ID: {image_id}\nTotal Bboxes: {len(boxes)}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'detection_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    plt.close()
    
    # Also save individual images
    print(f"\n📸 Saving individual images...")
    
    for idx, dataset_idx in enumerate(selected_indices):
        image, target = dataset[dataset_idx]
        
        # Convert image to numpy
        if torch.is_tensor(image):
            img = image.permute(1, 2, 0).cpu().numpy()
        else:
            img = image
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Get bounding boxes
        if torch.is_tensor(target['boxes']):
            boxes = target['boxes'].cpu().numpy()  # [N, 4]
        else:
            boxes = target['boxes']
        
        if torch.is_tensor(target['labels']):
            labels = target['labels'].cpu().numpy()  # [N]
        else:
            labels = target['labels']
        
        # Color palette for different lesions (BGR format for OpenCV)
        colors_bgr = [
            (0, 0, 255),    # red
            (255, 0, 0),    # blue
            (0, 255, 0),    # green
            (0, 255, 255),  # yellow
            (255, 255, 0),  # cyan
            (255, 0, 255),  # magenta
            (0, 165, 255),  # orange
            (128, 0, 128),  # purple
            (203, 192, 255), # pink
            (42, 42, 165)   # brown
        ]
        
        # Draw ALL bounding boxes with different colors
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Use different colors for different lesions
            color = colors_bgr[i % len(colors_bgr)]
            
            # Draw rectangle with thicker line
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
            
            # Add label with offset to avoid overlap
            label_text = f'L{i+1}'
            label_y = max(y1 - 5 - (i % 3) * 20, 15)  # Stagger labels, but keep above image
            cv2.putText(img_bgr, label_text, (x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Also draw box coordinates for debugging
            coord_text = f'({x1},{y1})-({x2},{y2})'
            cv2.putText(img_bgr, coord_text, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Save image
        image_id = dataset.get_image_id(dataset_idx)
        output_path = os.path.join(output_dir, f'{image_id}_bboxes.jpg')
        cv2.imwrite(output_path, img_bgr)
        print(f"  ✅ {image_id}: {len(boxes)} bboxes -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize detection samples')
    parser.add_argument('--vindr_root', default='/team/team_pxi/pxi-dataset/cxr/public/vinbig',
                       help='Root directory for VinDr-CXR dataset')
    parser.add_argument('--target_lesion', default='nodule',
                       choices=['calcification', 'nodule', 'cardiomegaly', 'pleural_effusion',
                               'consolidation', 'pneumothorax', 'aortic_enlargement',
                               'infiltration', 'lung_opacity'],
                       help='Target lesion type')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', default='./detection_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--min_radiologist_agreement', type=int, default=1,
                       help='Minimum radiologist agreement')
    parser.add_argument('--min_bboxes', type=int, default=1,
                       help='Minimum number of bboxes per image (to find images with multiple lesions)')
    
    args = parser.parse_args()
    
    print("🔍 Creating VinDr-CXR dataset in detection mode...")
    dataset = VinDrCXRDataset(
        data_root=args.vindr_root,
        split='train',
        target_lesion=args.target_lesion,
        mask_type='detection',
        min_radiologist_agreement=args.min_radiologist_agreement,
    )
    
    print(f"📊 Dataset size: {len(dataset)}")
    
    # If min_bboxes > 1, find images with multiple bboxes
    if args.min_bboxes > 1:
        print(f"🔍 Finding images with at least {args.min_bboxes} bboxes...")
        multi_bbox_indices = []
        for i in range(len(dataset)):
            image, target = dataset[i]
            num_bboxes = len(target['boxes'])
            if num_bboxes >= args.min_bboxes:
                multi_bbox_indices.append(i)
                if len(multi_bbox_indices) >= args.num_samples:
                    break
        
        if len(multi_bbox_indices) < args.num_samples:
            print(f"⚠️  Only found {len(multi_bbox_indices)} images with >= {args.min_bboxes} bboxes")
            print(f"   Using all {len(multi_bbox_indices)} images")
            selected_indices = multi_bbox_indices
        else:
            selected_indices = multi_bbox_indices[:args.num_samples]
        
        print(f"🎨 Visualizing {len(selected_indices)} samples with multiple bboxes...")
        
        # Visualize directly with selected indices
        visualize_detection_samples_with_indices(dataset, selected_indices, args.output_dir)
    else:
        print(f"🎨 Visualizing {args.num_samples} random samples...")
        visualize_detection_samples(dataset, args.num_samples, args.output_dir)
    
    print("\n✅ Visualization complete!")


if __name__ == '__main__':
    main()
