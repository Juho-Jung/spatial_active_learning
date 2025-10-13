#!/usr/bin/env python3
"""
Active Learning Visualization Comparison

Compare spatial distribution of selected samples between two baseline strategies.
Visualizes ground truth center points of selected samples on a 512x512 grid.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import cv2

# Add project root to path
sys.path.append('/opt/pxi/projects/calcified_nodule/spatial_active_learning')
sys.path.append('/opt/pxi/projects/calcified_nodule/spatial_active_learning/sam_adaptation')
sys.path.append('/opt/pxi')

import mdb.document as mdb_d
import mdb.load as mdb_c


def load_raw_documents(train_collection='validation_collection', target_lesion='calcification'):
    """Load raw documents for spatial analysis."""
    # Access the raw documents before any splitting
    if train_collection == 'validation_collection':
        collection = mdb_c.get_collection("validation_internal", db_names=['cxr_new', 'projects', 'public'])
        all_docs = list(collection.find({}))

        # Filter out excluded sources
        excluded_sources = ["amcio_b1_368"]
        documents = []
        for doc in all_docs:
            data_source = doc.get('data_source', '')
            if isinstance(data_source, list):
                data_source = data_source[0] if data_source else ''

            if data_source not in excluded_sources:
                documents.append(doc)

        print(f"📊 Loaded {len(documents)} raw documents from validation_internal")
        return documents
    elif train_collection == 'sdc_ppm_train-0908':
        collection = mdb_c.get_collection("sdc_ppm_train-0908", db_names=['cxr_new', 'projects', 'personal'])
        
        # Build query based on parameters
        query = {
            'is_pos': {'$in': [1]},
            'is_normal': {'$in': [0, 1]},
        }
        
        documents = list(collection.find(query))
        print(f"📊 Loaded {len(documents)} raw documents from sdc_ppm_train-0908")
        return documents
    else:
        print(f"❌ Unknown collection: {train_collection}")
        return []


def _get_mask_from_doc(doc, target_lesion, target_size=(512, 512)):
    """Helper function to extract mask from document using the same logic as dataset."""
    try:
        # Create mask using the same logic as SAMLesionDataset.__getitem__
        mask = np.zeros(target_size, dtype=np.uint8)
        objects = doc.get('objects', [])

        for obj in objects:
            if obj.get('finding_name') == target_lesion:
                polygon = obj.get('polygon')
                if polygon:
                    try:
                        from shapely import from_wkt
                        if isinstance(polygon, str) and polygon.startswith('POLYGON'):
                            poly = from_wkt(polygon)
                            coords = list(poly.exterior.coords)
                            coords = np.array(coords)
                            coords[:, 0] = coords[:, 0] / 100 * target_size[1]
                            coords[:, 1] = coords[:, 1] / 100 * target_size[0]
                            coords = coords.astype(np.int32)

                            cv2.fillPoly(mask, [coords], 1)
                    except Exception as e:
                        print(f"⚠️  Error creating mask from polygon: {e}")
                        continue

        return mask.astype(np.float32)

    except Exception as e:
        print(f"⚠️  Error extracting mask from document: {e}")
        return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Active Learning Visualization Comparison')
    parser.add_argument('--result1_path', required=True,
                       help='Path to first result.json file')
    parser.add_argument('--result2_path', required=True,
                       help='Path to second result.json file')
    parser.add_argument('--output_dir', default='output_comparison/visualization',
                       help='Output directory for visualization plots')
    parser.add_argument('--target_lesion', default='pneumoperitoneum',
                       help='Target lesion type')
    parser.add_argument('--collection', default='sdc_ppm_train-0908',
                       help='Data collection name')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for visualization grid')
    return parser.parse_args()


def extract_model_name(result_path):
    """Extract model name from result path."""
    # Extract the strategy_uncertainty part from path
    path_parts = Path(result_path).parts
    for part in path_parts:
        if '_' in part and any(x in part for x in ['adaptive', 'random', 'uncertainty', 'area_random', 'diversity']):
            # Remove timestamp part (last part after last underscore)
            parts = part.split('_')
            if len(parts) >= 2:
                # Find where timestamp starts (usually 8 digits)
                timestamp_start = None
                for i, p in enumerate(parts):
                    if p.isdigit() and len(p) == 8:
                        timestamp_start = i
                        break
                
                if timestamp_start is not None:
                    return '_'.join(parts[:timestamp_start])
                else:
                    return '_'.join(parts[:-1])  # Remove last part if it looks like timestamp
    return 'unknown'


def load_results(result_path):
    """Load results from JSON file."""
    if not os.path.exists(result_path):
        print(f"❌ Result file not found: {result_path}")
        return None
        
    model_name = extract_model_name(result_path)
    print(f"📊 Loading results for {model_name} from {result_path}")
    
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} rounds for {model_name}")
        return data, model_name
    except Exception as e:
        print(f"❌ Error loading {result_path}: {e}")
        return None, None


def get_gt_center_points(documents, selected_indices, target_lesion, image_size=512):
    """Get ground truth center points for selected samples."""
    center_points = []
    
    for idx in selected_indices:
        if idx >= len(documents):
            print(f"⚠️  Warning: Index {idx} out of range for documents (len={len(documents)})")
            continue
            
        doc = documents[idx]
        
        # Get mask from document
        mask = _get_mask_from_doc(doc, target_lesion, target_size=(image_size, image_size))
        
        if mask is None or mask.sum() == 0:
            print(f"⚠️  Warning: No valid mask found for document {idx}")
            continue
        
        # Find center point of the mask
        coords = np.where(mask > 0.5)
        if len(coords[0]) > 0:
            center_y = np.mean(coords[0])
            center_x = np.mean(coords[1])
            center_points.append((center_x, center_y))
        else:
            print(f"⚠️  Warning: No positive pixels found in mask for document {idx}")
    
    return center_points


def create_comparison_plot(center_points1, center_points2, model1_name, model2_name, 
                          round_num, output_path, image_size=512):
    """Create comparison plot for two sets of center points."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Create 512x512 grid
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match image coordinates
    
    # Plot center points for model 1
    if center_points1:
        x1, y1 = zip(*center_points1)
        ax.scatter(x1, y1, c='red', s=50, alpha=0.7, label=f'{model1_name} ({len(center_points1)} points)', marker='o')
    
    # Plot center points for model 2
    if center_points2:
        x2, y2 = zip(*center_points2)
        ax.scatter(x2, y2, c='blue', s=50, alpha=0.7, label=f'{model2_name} ({len(center_points2)} points)', marker='s')
    
    # Add grid lines for better visualization
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(0, image_size + 1, 64))
    ax.set_yticks(np.arange(0, image_size + 1, 64))
    
    # Add title and labels
    ax.set_title(f'Round {round_num} - Sample Selection Comparison\n{model1_name} vs {model2_name}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate (pixels)')
    ax.set_ylabel('Y Coordinate (pixels)')
    ax.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Round {round_num} comparison plot saved: {output_path}")


def create_individual_plots(center_points1, center_points2, model1_name, model2_name, 
                           round_num, output_dir, image_size=512):
    """Create individual plots for each model."""
    # Model 1 plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    ax1.set_xlim(0, image_size)
    ax1.set_ylim(0, image_size)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    
    if center_points1:
        x1, y1 = zip(*center_points1)
        ax1.scatter(x1, y1, c='red', s=50, alpha=0.7, marker='o')
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(np.arange(0, image_size + 1, 64))
    ax1.set_yticks(np.arange(0, image_size + 1, 64))
    ax1.set_title(f'Round {round_num} - {model1_name}\n({len(center_points1)} samples)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Coordinate (pixels)')
    ax1.set_ylabel('Y Coordinate (pixels)')
    
    output_path1 = os.path.join(output_dir, f'round{round_num}_{model1_name}.png')
    plt.tight_layout()
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model 2 plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    ax2.set_xlim(0, image_size)
    ax2.set_ylim(0, image_size)
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    
    if center_points2:
        x2, y2 = zip(*center_points2)
        ax2.scatter(x2, y2, c='blue', s=50, alpha=0.7, marker='s')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(np.arange(0, image_size + 1, 64))
    ax2.set_yticks(np.arange(0, image_size + 1, 64))
    ax2.set_title(f'Round {round_num} - {model2_name}\n({len(center_points2)} samples)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X Coordinate (pixels)')
    ax2.set_ylabel('Y Coordinate (pixels)')
    
    output_path2 = os.path.join(output_dir, f'round{round_num}_{model2_name}.png')
    plt.tight_layout()
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Individual plots saved: {output_path1}, {output_path2}")


def create_cumulative_plots(all_center_points1, all_center_points2, model1_name, model2_name, 
                           round_num, model1_dir, model2_dir, image_size=512):
    """Create cumulative plots showing all selected samples up to current round."""
    # Model 1 cumulative plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    ax1.set_xlim(0, image_size)
    ax1.set_ylim(0, image_size)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    
    # Plot all previous rounds with same color and shape (red circles)
    for r in range(1, round_num + 1):
        if r in all_center_points1 and all_center_points1[r]:
            x, y = zip(*all_center_points1[r])
            ax1.scatter(x, y, c='red', s=50, alpha=0.7, marker='o')
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(np.arange(0, image_size + 1, 64))
    ax1.set_yticks(np.arange(0, image_size + 1, 64))
    ax1.set_title(f'Cumulative Selection - {model1_name} (Rounds 1-{round_num})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Coordinate (pixels)')
    ax1.set_ylabel('Y Coordinate (pixels)')
    
    output_path1 = os.path.join(model1_dir, f'cumulative_round{round_num}_{model1_name}.png')
    plt.tight_layout()
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model 2 cumulative plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    ax2.set_xlim(0, image_size)
    ax2.set_ylim(0, image_size)
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    
    # Plot all previous rounds with same color and shape (blue squares)
    for r in range(1, round_num + 1):
        if r in all_center_points2 and all_center_points2[r]:
            x, y = zip(*all_center_points2[r])
            ax2.scatter(x, y, c='blue', s=50, alpha=0.7, marker='s')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(np.arange(0, image_size + 1, 64))
    ax2.set_yticks(np.arange(0, image_size + 1, 64))
    ax2.set_title(f'Cumulative Selection - {model2_name} (Rounds 1-{round_num})', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Coordinate (pixels)')
    ax2.set_ylabel('Y Coordinate (pixels)')
    
    output_path2 = os.path.join(model2_dir, f'cumulative_round{round_num}_{model2_name}.png')
    plt.tight_layout()
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Cumulative plots saved: {output_path1}, {output_path2}")


def calculate_spatial_statistics(center_points, model_name):
    """Calculate spatial statistics for center points."""
    if not center_points:
        return {}
    
    x_coords, y_coords = zip(*center_points)
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    stats = {
        'model': model_name,
        'num_samples': len(center_points),
        'x_mean': np.mean(x_coords),
        'y_mean': np.mean(y_coords),
        'x_std': np.std(x_coords),
        'y_std': np.std(y_coords),
        'x_min': np.min(x_coords),
        'x_max': np.max(x_coords),
        'y_min': np.min(y_coords),
        'y_max': np.max(y_coords),
        'spread': np.sqrt(np.var(x_coords) + np.var(y_coords))  # Overall spread
    }
    
    return stats


def print_round_statistics(stats1, stats2, round_num):
    """Print statistics for a round."""
    print(f"\n📊 Round {round_num} Statistics:")
    print(f"   {stats1['model']}: {stats1['num_samples']} samples, "
          f"center=({stats1['x_mean']:.1f}, {stats1['y_mean']:.1f}), "
          f"spread={stats1['spread']:.1f}")
    print(f"   {stats2['model']}: {stats2['num_samples']} samples, "
          f"center=({stats2['x_mean']:.1f}, {stats2['y_mean']:.1f}), "
          f"spread={stats2['spread']:.1f}")


def main():
    """Main function for visualization comparison."""
    args = parse_arguments()
    
    # Create output directory structure
    comparison_name = f"{extract_model_name(args.result1_path)}_{extract_model_name(args.result2_path)}"
    base_output_dir = os.path.join(args.output_dir, comparison_name)
    
    # Create subdirectories
    comparison_dir = os.path.join(base_output_dir, 'comparison')
    model1_dir = os.path.join(base_output_dir, extract_model_name(args.result1_path))
    model2_dir = os.path.join(base_output_dir, extract_model_name(args.result2_path))
    
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(model1_dir, exist_ok=True)
    os.makedirs(model2_dir, exist_ok=True)
    
    print("🚀 Starting Active Learning Visualization Comparison")
    print(f"📁 Output directory: {base_output_dir}")
    print(f"🎯 Target lesion: {args.target_lesion}")
    print(f"📊 Collection: {args.collection}")
    print(f"📁 Subdirectories created:")
    print(f"   - comparison: {comparison_dir}")
    print(f"   - {extract_model_name(args.result1_path)}: {model1_dir}")
    print(f"   - {extract_model_name(args.result2_path)}: {model2_dir}")
    
    # Load results
    result1_data, model1_name = load_results(args.result1_path)
    result2_data, model2_name = load_results(args.result2_path)
    
    if result1_data is None or result2_data is None:
        print("❌ Failed to load one or both result files!")
        return
    
    # Load raw documents
    print(f"📊 Loading raw documents from {args.collection}...")
    raw_documents = load_raw_documents(args.collection, args.target_lesion)
    print(f"✅ Loaded {len(raw_documents)} documents")
    
    # Process each round
    max_rounds = min(len(result1_data), len(result2_data))
    print(f"📊 Processing {max_rounds} rounds...")
    
    # Store all center points for cumulative visualization
    all_center_points1 = {}
    all_center_points2 = {}
    
    for round_idx in range(max_rounds):
        round_num = round_idx + 1
        print(f"\n🔄 Processing Round {round_num}")
        
        # Get selected indices for this round
        selected_indices1 = result1_data[round_idx]['selected_indices']
        selected_indices2 = result2_data[round_idx]['selected_indices']
        
        print(f"   {model1_name}: {len(selected_indices1)} samples")
        print(f"   {model2_name}: {len(selected_indices2)} samples")
        
        # Get center points for each model
        center_points1 = get_gt_center_points(raw_documents, selected_indices1, args.target_lesion, args.image_size)
        center_points2 = get_gt_center_points(raw_documents, selected_indices2, args.target_lesion, args.image_size)
        
        print(f"   {model1_name}: {len(center_points1)} valid center points")
        print(f"   {model2_name}: {len(center_points2)} valid center points")
        
        # Store for cumulative visualization
        all_center_points1[round_num] = center_points1
        all_center_points2[round_num] = center_points2
        
        # Create comparison plot (save to comparison subdirectory)
        comparison_path = os.path.join(comparison_dir, f'round{round_num}_comparison.png')
        create_comparison_plot(center_points1, center_points2, model1_name, model2_name, 
                              round_num, comparison_path, args.image_size)
        
        # Create individual plots (save to respective model subdirectories)
        create_individual_plots(center_points1, center_points2, model1_name, model2_name, 
                                round_num, model1_dir, args.image_size)
        create_individual_plots(center_points1, center_points2, model1_name, model2_name, 
                                round_num, model2_dir, args.image_size)
        
        # Create cumulative plots (save to respective model directories)
        create_cumulative_plots(all_center_points1, all_center_points2, model1_name, model2_name, 
                                round_num, model1_dir, model2_dir, args.image_size)
        
        # Calculate and print statistics
        stats1 = calculate_spatial_statistics(center_points1, model1_name)
        stats2 = calculate_spatial_statistics(center_points2, model2_name)
        print_round_statistics(stats1, stats2, round_num)
    
    print(f"\n🎉 Visualization comparison completed!")
    print(f"📁 Results saved to: {base_output_dir}")
    print(f"📊 Generated plots for {max_rounds} rounds")
    print(f"📁 Directory structure:")
    print(f"   - comparison/: round1_comparison.png, round2_comparison.png, ...")
    print(f"   - {model1_name}/: round1_{model1_name}.png, round2_{model1_name}.png, ...")
    print(f"   - {model1_name}/: cumulative_round1_{model1_name}.png, cumulative_round2_{model1_name}.png, ...")
    print(f"   - {model2_name}/: round1_{model2_name}.png, round2_{model2_name}.png, ...")
    print(f"   - {model2_name}/: cumulative_round1_{model2_name}.png, cumulative_round2_{model2_name}.png, ...")


if __name__ == "__main__":
    main()
