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
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mdb.collection as mdb_c
import mdb.document as mdb_d
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add project root to path
sys.path.append('/opt/pxi/projects/calcified_nodule/spatial_active_learning')
sys.path.append('/opt/pxi/projects/calcified_nodule/spatial_active_learning/sam_adaptation')
sys.path.append('/opt/pxi')


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
    elif train_collection == 'both':
        # For both collection, use the same logic as dataset.py
        # Create a temporary dataset to access documents
        try:
            from sam_adaptation.dataset.dataset import SAMLesionDataset
            temp_dataset = SAMLesionDataset(
                split='train',  # This doesn't matter for raw document loading
                train_collection=train_collection,
                target_lesion=target_lesion
            )
            documents = temp_dataset._load_documents()
            print(f"📊 Loaded {len(documents)} raw documents from both collections")
            return documents
        except Exception as e:
            print(f"⚠️  Error loading from both collections: {e}")
            # Fallback: try to load from validation only
            return load_raw_documents('validation_collection', target_lesion)
    else:
        print(f"❌ Unknown collection: {train_collection}")
        return []


def _get_mask_from_doc(doc, target_lesion, target_size=(512, 512)):
    """Helper function to extract mask from document using the same logic as dataset."""
    try:
        # Create mask using the same logic as SAMLesionDataset.__getitem__
        mask = np.zeros(target_size, dtype=np.uint8)
        objects = doc.get('objects', [])

        # Debug: print document info
        if not objects:
            return None

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
    parser.add_argument('--output_dir', default='/team/team_pxi/workspace/juhojung/spatial_active_learning/output_comparison/visualization',
                        help='Output directory for visualization plots')
    parser.add_argument('--target_lesion', default='calcifiednodule',
                        help='Target lesion type')
    parser.add_argument('--collection', default='both',
                        help='Data collection name')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size for visualization grid')
    parser.add_argument('--timeline_rounds', nargs='+', type=int, default=None,
                        help='Rounds to show in timeline comparison (e.g., --timeline_rounds 1 3 5 or --timeline_rounds 1 2 3 5)')
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
                    full_name = '_'.join(parts[:timestamp_start])
                else:
                    full_name = '_'.join(parts[:-1])  # Remove last part if it looks like timestamp

                # Clean up the name to show only the main strategy
                if 'uncertainty' in full_name:
                    return 'uncertainty'
                elif 'random' in full_name:
                    return 'random'
                elif 'adaptive_improved' in full_name:
                    return 'adaptive_improved'
                elif 'adaptive_multi_scale' in full_name:
                    return 'adaptive_multi_scale'
                elif 'adaptive_performance_monitoring' in full_name:
                    return 'adaptive_performance_monitoring'
                elif 'adaptive' in full_name:
                    return 'adaptive'
                elif 'area_random' in full_name:
                    return 'area_random'
                elif 'diversity' in full_name:
                    return 'diversity'
                else:
                    return full_name
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
    """Create enhanced comparison plot for two sets of center points."""
    # Debug: print center points info
    print(f"🔍 Debug - Round {round_num}: {model1_name} has {len(center_points1) if center_points1 else 0} points, {model2_name} has {len(center_points2) if center_points2 else 0} points")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Left plot: Direct comparison
    ax1.set_xlim(0, image_size)
    ax1.set_ylim(0, image_size)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()

    # Plot center points for model 1
    if center_points1:
        x1, y1 = zip(*center_points1)
        ax1.scatter(x1, y1, c='red', s=50, alpha=0.7, label=f'{model1_name} ({len(center_points1)} points)', marker='o')

    # Plot center points for model 2
    if center_points2:
        x2, y2 = zip(*center_points2)
        ax1.scatter(x2, y2, c='blue', s=50, alpha=0.7,
                    label=f'{model2_name} ({len(center_points2)} points)', marker='s')

    # Add grid lines for better visualization
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(np.arange(0, image_size + 1, 64))
    ax1.set_yticks(np.arange(0, image_size + 1, 64))
    ax1.set_title(f'Round {round_num} - Direct Comparison\n{model1_name} vs {model2_name}',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Coordinate (pixels)')
    ax1.set_ylabel('Y Coordinate (pixels)')
    ax1.legend()

    # Right plot: Spatial coverage analysis
    ax2.set_xlim(0, image_size)
    ax2.set_ylim(0, image_size)
    ax2.set_aspect('equal')
    ax2.invert_yaxis()

    # Create 5x5 grid for coverage analysis
    grid_size = 5
    cell_width = image_size // grid_size
    cell_height = image_size // grid_size

    # Calculate coverage for each grid cell
    coverage1 = np.zeros((grid_size, grid_size))
    coverage2 = np.zeros((grid_size, grid_size))

    if center_points1:
        for x, y in center_points1:
            grid_x = min(int(x // cell_width), grid_size - 1)
            grid_y = min(int(y // cell_height), grid_size - 1)
            coverage1[grid_y, grid_x] += 1

    if center_points2:
        for x, y in center_points2:
            grid_x = min(int(x // cell_width), grid_size - 1)
            grid_y = min(int(y // cell_height), grid_size - 1)
            coverage2[grid_y, grid_x] += 1

    # Plot coverage heatmap
    im1 = ax2.imshow(coverage1, cmap='Reds', alpha=0.7, extent=[0, image_size, image_size, 0])
    im2 = ax2.imshow(coverage2, cmap='Blues', alpha=0.7, extent=[0, image_size, image_size, 0])

    # Add grid lines
    for i in range(grid_size + 1):
        ax2.axhline(y=i * cell_height, color='black', linewidth=1, alpha=0.5)
        ax2.axvline(x=i * cell_width, color='black', linewidth=1, alpha=0.5)

    ax2.set_title(f'Round {round_num} - Spatial Coverage Analysis\nRed: {model1_name}, Blue: {model2_name}',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X Coordinate (pixels)')
    ax2.set_ylabel('Y Coordinate (pixels)')

    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=ax2, fraction=0.046, pad=0.04, label=f'{model1_name} Coverage')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label=f'{model2_name} Coverage')

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Round {round_num} enhanced comparison plot saved: {output_path}")


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


def create_timeline_comparison_plot(all_center_points1, all_center_points2, model1_name, model2_name,
                                    total_rounds, output_path, image_size=512, timeline_rounds=None):
    """Create timeline comparison plot showing specified rounds."""
    # Determine grid size based on number of rounds
    num_rounds = len(timeline_rounds) if timeline_rounds is not None else 3
    fig, axes = plt.subplots(2, num_rounds, figsize=(6 * num_rounds, 14))

    # Determine rounds to show
    if timeline_rounds is not None:
        rounds_to_show = timeline_rounds
        round_labels = [f'Round {r}' for r in timeline_rounds]
    else:
        # Default: 1st, middle, last
        middle_round = (total_rounds + 1) // 2
        rounds_to_show = [1, middle_round, total_rounds]
        round_labels = ['1st Round', f'Middle (R{middle_round})', f'Last (R{total_rounds})']

    # Row 1: Model 1 (result1)
    for col, (round_num, label) in enumerate(zip(rounds_to_show, round_labels)):
        ax = axes[0, col]
        ax.set_xlim(0, image_size)
        ax.set_ylim(0, image_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # Plot cumulative samples up to this round
        all_points = []
        for r in range(1, round_num + 1):
            if r in all_center_points1 and all_center_points1[r]:
                all_points.extend(all_center_points1[r])

        if all_points:
            x, y = zip(*all_points)
            ax.scatter(x, y, c='red', s=30, alpha=0.7, marker='o')

        ax.grid(True, alpha=0.3)
        ax.set_xticks(np.arange(0, image_size + 1, 128))
        ax.set_yticks(np.arange(0, image_size + 1, 128))
        ax.set_title(f'{model1_name}\n{label}\n({len(all_points)} samples)',
                     fontsize=10, fontweight='bold', pad=10)
        ax.set_xlabel('X Coordinate' if col == 0 else '')
        ax.set_ylabel('Y Coordinate' if col == 0 else '')

        # Add sample count annotation
        ax.text(0.02, 0.98, f'Total: {len(all_points)}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                verticalalignment='top')

    # Row 2: Model 2 (result2)
    for col, (round_num, label) in enumerate(zip(rounds_to_show, round_labels)):
        ax = axes[1, col]
        ax.set_xlim(0, image_size)
        ax.set_ylim(0, image_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # Plot cumulative samples up to this round
        all_points = []
        for r in range(1, round_num + 1):
            if r in all_center_points2 and all_center_points2[r]:
                all_points.extend(all_center_points2[r])

        if all_points:
            x, y = zip(*all_points)
            ax.scatter(x, y, c='blue', s=30, alpha=0.7, marker='o')

        ax.grid(True, alpha=0.3)
        ax.set_xticks(np.arange(0, image_size + 1, 128))
        ax.set_yticks(np.arange(0, image_size + 1, 128))
        ax.set_title(f'{model2_name}\n{label}\n({len(all_points)} samples)',
                     fontsize=10, fontweight='bold', pad=10)
        ax.set_xlabel('X Coordinate' if col == 0 else '')
        ax.set_ylabel('Y Coordinate' if col == 0 else '')

        # Add sample count annotation
        ax.text(0.02, 0.98, f'Total: {len(all_points)}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                verticalalignment='top')

    # Add main title (single line to avoid overlap)
    fig.suptitle(f'Active Learning Comparison: {model1_name} vs {model2_name}',
                 fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.25, wspace=0.15)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Timeline comparison plot saved: {output_path}")


def calculate_spatial_statistics(center_points, model_name):
    """Calculate enhanced spatial statistics for center points."""
    if not center_points:
        return {}

    x_coords, y_coords = zip(*center_points)
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    # Basic statistics
    basic_stats = {
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
        'spread': np.sqrt(np.var(x_coords) + np.var(y_coords))
    }

    # Enhanced spatial analysis
    # 1. Grid coverage analysis (5x5 grid)
    grid_size = 5
    image_size = 512
    cell_width = image_size // grid_size
    cell_height = image_size // grid_size

    grid_coverage = np.zeros((grid_size, grid_size))
    for x, y in center_points:
        grid_x = min(int(x // cell_width), grid_size - 1)
        grid_y = min(int(y // cell_height), grid_size - 1)
        grid_coverage[grid_y, grid_x] += 1

    # Coverage metrics
    covered_cells = np.sum(grid_coverage > 0)
    coverage_ratio = covered_cells / (grid_size * grid_size)
    coverage_std = np.std(grid_coverage)
    coverage_balance = 1.0 - (coverage_std / (np.mean(grid_coverage) + 1e-8))

    # 2. Quadrant analysis
    mid_x, mid_y = 256, 256
    quadrants = [0, 0, 0, 0]  # Top-left, Top-right, Bottom-left, Bottom-right

    for x, y in center_points:
        if x < mid_x and y < mid_y:
            quadrants[0] += 1  # Top-left
        elif x >= mid_x and y < mid_y:
            quadrants[1] += 1  # Top-right
        elif x < mid_x and y >= mid_y:
            quadrants[2] += 1  # Bottom-left
        else:
            quadrants[3] += 1  # Bottom-right

    quadrant_std = np.std(quadrants)
    quadrant_balance = 1.0 - (quadrant_std / (np.mean(quadrants) + 1e-8))

    # 3. Spatial clustering analysis
    if len(center_points) > 1:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        points_array = np.array(center_points)
        try:
            n_clusters = min(2, len(center_points))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(points_array)
            silhouette = silhouette_score(points_array, cluster_labels)
        except:
            silhouette = 0.0
    else:
        silhouette = 0.0

    # Combine all statistics
    enhanced_stats = {
        **basic_stats,
        'coverage_ratio': coverage_ratio,
        'covered_cells': covered_cells,
        'coverage_balance': coverage_balance,
        'quadrant_balance': quadrant_balance,
        'silhouette_score': silhouette,
        'grid_coverage': grid_coverage.tolist(),
        'quadrant_distribution': quadrants
    }

    return enhanced_stats


def print_round_statistics(stats1, stats2, round_num):
    """Print enhanced statistics for a round."""
    print(f"\n📊 Round {round_num} Enhanced Statistics:")
    print(f"   {stats1['model']}: {stats1['num_samples']} samples")
    print(f"     - Center: ({stats1['x_mean']:.1f}, {stats1['y_mean']:.1f})")
    print(f"     - Spread: {stats1['spread']:.1f}")
    print(f"     - Coverage: {stats1['coverage_ratio']:.3f} ({stats1['covered_cells']}/25 cells)")
    print(f"     - Balance: {stats1['coverage_balance']:.3f}")
    print(f"     - Quadrant Balance: {stats1['quadrant_balance']:.3f}")
    print(f"     - Clustering Quality: {stats1['silhouette_score']:.3f}")

    print(f"   {stats2['model']}: {stats2['num_samples']} samples")
    print(f"     - Center: ({stats2['x_mean']:.1f}, {stats2['y_mean']:.1f})")
    print(f"     - Spread: {stats2['spread']:.1f}")
    print(f"     - Coverage: {stats2['coverage_ratio']:.3f} ({stats2['covered_cells']}/25 cells)")
    print(f"     - Balance: {stats2['coverage_balance']:.3f}")
    print(f"     - Quadrant Balance: {stats2['quadrant_balance']:.3f}")
    print(f"     - Clustering Quality: {stats2['silhouette_score']:.3f}")

    # Performance comparison
    print(f"\n🏆 Round {round_num} Performance Comparison:")
    if stats1['coverage_ratio'] > stats2['coverage_ratio']:
        print(
            f"   ✅ {stats1['model']} wins in Coverage Ratio ({stats1['coverage_ratio']:.3f} vs {stats2['coverage_ratio']:.3f})")
    else:
        print(
            f"   ✅ {stats2['model']} wins in Coverage Ratio ({stats2['coverage_ratio']:.3f} vs {stats1['coverage_ratio']:.3f})")

    if stats1['coverage_balance'] > stats2['coverage_balance']:
        print(
            f"   ✅ {stats1['model']} wins in Coverage Balance ({stats1['coverage_balance']:.3f} vs {stats2['coverage_balance']:.3f})")
    else:
        print(
            f"   ✅ {stats2['model']} wins in Coverage Balance ({stats2['coverage_balance']:.3f} vs {stats1['coverage_balance']:.3f})")


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

    # Load raw documents using the same method as run_al_baselines.py
    print(f"📊 Loading raw documents from {args.collection}...")
    raw_documents = load_raw_documents(args.collection, args.target_lesion)
    print(f"✅ Loaded {len(raw_documents)} documents")

    # Filter positive documents (same as run_al_baselines.py)
    positive_docs = []
    for doc in raw_documents:
        objects = doc.get('objects', [])
        has_target_lesion = False
        for obj in objects:
            if obj.get('finding_name') == args.target_lesion:
                has_target_lesion = True
                break
        if has_target_lesion:
            positive_docs.append(doc)

    print(f"📊 Filtered to {len(positive_docs)} positive documents with {args.target_lesion}")
    raw_documents = positive_docs

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

    # Create timeline comparison plot (2x3 grid showing specified rounds)
    print(f"\n📊 Creating timeline comparison plot...")
    timeline_path = os.path.join(comparison_dir, 'timeline_comparison.png')
    create_timeline_comparison_plot(all_center_points1, all_center_points2, model1_name, model2_name,
                                    max_rounds, timeline_path, args.image_size, args.timeline_rounds)

    print(f"\n🎉 Visualization comparison completed!")
    print(f"📁 Results saved to: {base_output_dir}")
    print(f"📊 Generated plots for {max_rounds} rounds")
    print(f"📁 Directory structure:")
    print(f"   - comparison/: round1_comparison.png, round2_comparison.png, ..., timeline_comparison.png")
    print(f"   - {model1_name}/: round1_{model1_name}.png, round2_{model1_name}.png, ...")
    print(f"   - {model1_name}/: cumulative_round1_{model1_name}.png, cumulative_round2_{model1_name}.png, ...")
    print(f"   - {model2_name}/: round1_{model2_name}.png, round2_{model2_name}.png, ...")
    print(f"   - {model2_name}/: cumulative_round1_{model2_name}.png, cumulative_round2_{model2_name}.png, ...")
    print(f"\n🎯 Key visualization files:")
    print(f"   - timeline_comparison.png: 2x3 grid showing 1st, middle, last rounds")
    print(f"   - round*_comparison.png: Individual round comparisons")
    print(f"   - cumulative_round*_*.png: Cumulative selection patterns")


if __name__ == "__main__":
    main()
