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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
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

    # Create output directory structure with more descriptive name
    model1_name = extract_model_name(args.result1_path)
    model2_name = extract_model_name(args.result2_path)

    # Extract split type from result path
    split_type = "dynamic" if "dynamic_split" in args.result1_path else "equal"

    # Create descriptive comparison name
    comparison_name = f"{split_type}_{model1_name}_{model2_name}"
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

    # Load documents using the same method as run_al_baselines.py
    print(f"📊 Loading documents from {args.collection}...")

    # Use the same dataset loading logic as run_al_baselines.py
    from active_learning.dataset.dataset import SAMLesionDataset

    # Create dataset to get the same document loading logic
    temp_dataset = SAMLesionDataset(
        split='train',
        train_collection=args.collection,
        target_lesion=args.target_lesion
    )

    # Get all documents (this will be the same as what run_al_baselines.py uses)
    all_documents = temp_dataset._load_documents()
    print(f"✅ Loaded {len(all_documents)} documents")

    # Filter positive documents (same as run_al_baselines.py)
    positive_docs = []
    for doc in all_documents:
        objects = doc.get('objects', [])
        has_target_lesion = False
        for obj in objects:
            if obj.get('finding_name') == args.target_lesion:
                has_target_lesion = True
                break
        if has_target_lesion:
            positive_docs.append(doc)

    print(f"📊 Filtered to {len(positive_docs)} positive documents with {args.target_lesion}")

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
        center_points1 = get_gt_center_points(positive_docs, selected_indices1, args.target_lesion, args.image_size)
        center_points2 = get_gt_center_points(positive_docs, selected_indices2, args.target_lesion, args.image_size)

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

        # Create enhanced analysis plots (new functions)
        print(f"   📊 Creating enhanced analysis plots for round {round_num}...")

        # Selection process analysis
        create_selection_process_analysis(center_points1, center_points2, model1_name, model2_name,
                                          round_num, comparison_dir, args.image_size)

        # Clustering analysis
        create_clustering_analysis(center_points1, center_points2, model1_name, model2_name,
                                   round_num, comparison_dir, args.image_size)

        # Coverage efficiency analysis
        create_coverage_efficiency_analysis(center_points1, center_points2, model1_name, model2_name,
                                            round_num, comparison_dir, args.image_size)

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
    print(f"   - comparison/: selection_process_analysis_round*.png (6-subplot detailed analysis)")
    print(f"   - comparison/: clustering_analysis_round*.png (hierarchical + silhouette analysis)")
    print(f"   - comparison/: coverage_efficiency_analysis_round*.png (Voronoi + efficiency metrics)")
    print(f"   - {model1_name}/: round1_{model1_name}.png, round2_{model1_name}.png, ...")
    print(f"   - {model1_name}/: cumulative_round1_{model1_name}.png, cumulative_round2_{model1_name}.png, ...")
    print(f"   - {model2_name}/: round1_{model2_name}.png, round2_{model2_name}.png, ...")
    print(f"   - {model2_name}/: cumulative_round1_{model2_name}.png, cumulative_round2_{model2_name}.png, ...")
    print(f"\n🎯 Key visualization files:")
    print(f"   - timeline_comparison.png: 2x3 grid showing 1st, middle, last rounds")
    print(f"   - round*_comparison.png: Individual round comparisons")
    print(f"   - cumulative_round*_*.png: Cumulative selection patterns")
    print(f"   - selection_process_analysis_round*.png: 6-subplot detailed analysis")
    print(f"   - clustering_analysis_round*.png: Clustering pattern analysis")
    print(f"   - coverage_efficiency_analysis_round*.png: Coverage efficiency analysis")


def create_selection_process_analysis(center_points1, center_points2, model1_name, model2_name,
                                      round_num, output_dir, image_size=512):
    """Create detailed selection process analysis."""
    if not center_points1 or not center_points2:
        print(f"⚠️  Warning: Missing center points for round {round_num}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Selection Process Analysis - Round {round_num}\n{model1_name} vs {model2_name}',
                 fontsize=16, fontweight='bold')

    # Plot 1: Direct comparison
    ax1 = axes[0, 0]
    ax1.set_xlim(0, image_size)
    ax1.set_ylim(0, image_size)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()

    x1, y1 = zip(*center_points1)
    x2, y2 = zip(*center_points2)
    ax1.scatter(x1, y1, c='red', s=50, alpha=0.7, label=f'{model1_name} ({len(center_points1)})', marker='o')
    ax1.scatter(x2, y2, c='blue', s=50, alpha=0.7, label=f'{model2_name} ({len(center_points2)})', marker='s')

    ax1.grid(True, alpha=0.3)
    ax1.set_title('Direct Comparison')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()

    # Plot 2: Density analysis
    ax2 = axes[0, 1]
    # Create density heatmap for model1
    if len(center_points1) > 1:
        from scipy.stats import gaussian_kde
        points1 = np.array(center_points1)
        kde1 = gaussian_kde(points1.T)

        # Create grid for density calculation
        x_grid = np.linspace(0, image_size, 50)
        y_grid = np.linspace(0, image_size, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        density1 = kde1(positions).reshape(X.shape)

        im1 = ax2.imshow(density1, extent=[0, image_size, image_size, 0],
                         cmap='Reds', alpha=0.7, origin='upper')
        ax2.scatter(x1, y1, c='darkred', s=30, alpha=0.8, marker='o')

    ax2.set_title(f'{model1_name} Density Analysis')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')

    # Plot 3: Density analysis for model2
    ax3 = axes[0, 2]
    if len(center_points2) > 1:
        points2 = np.array(center_points2)
        kde2 = gaussian_kde(points2.T)
        density2 = kde2(positions).reshape(X.shape)

        im2 = ax3.imshow(density2, extent=[0, image_size, image_size, 0],
                         cmap='Blues', alpha=0.7, origin='upper')
        ax3.scatter(x2, y2, c='darkblue', s=30, alpha=0.8, marker='s')

    ax3.set_title(f'{model2_name} Density Analysis')
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')

    # Plot 4: Clustering analysis
    ax4 = axes[1, 0]
    if len(center_points1) > 2:
        # K-means clustering for model1
        n_clusters = min(3, len(center_points1))
        kmeans1 = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels1 = kmeans1.fit_predict(center_points1)

        colors1 = ['red', 'orange', 'pink']
        for i in range(n_clusters):
            cluster_points = [center_points1[j] for j in range(len(center_points1)) if cluster_labels1[j] == i]
            if cluster_points:
                cx, cy = zip(*cluster_points)
                ax4.scatter(cx, cy, c=colors1[i], s=50, alpha=0.7,
                            label=f'{model1_name} Cluster {i + 1}')

        ax4.set_title(f'{model1_name} Clustering Analysis')
        ax4.set_xlabel('X Coordinate')
        ax4.set_ylabel('Y Coordinate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Plot 5: Clustering analysis for model2
    ax5 = axes[1, 1]
    if len(center_points2) > 2:
        n_clusters = min(3, len(center_points2))
        kmeans2 = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels2 = kmeans2.fit_predict(center_points2)

        colors2 = ['blue', 'cyan', 'lightblue']
        for i in range(n_clusters):
            cluster_points = [center_points2[j] for j in range(len(center_points2)) if cluster_labels2[j] == i]
            if cluster_points:
                cx, cy = zip(*cluster_points)
                ax5.scatter(cx, cy, c=colors2[i], s=50, alpha=0.7,
                            label=f'{model2_name} Cluster {i + 1}')

        ax5.set_title(f'{model2_name} Clustering Analysis')
        ax5.set_xlabel('X Coordinate')
        ax5.set_ylabel('Y Coordinate')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # Plot 6: Spatial distribution comparison
    ax6 = axes[1, 2]
    # Calculate spatial statistics

    def calculate_spatial_stats(points):
        if len(points) < 2:
            return {}
        points_array = np.array(points)
        return {
            'mean_x': np.mean(points_array[:, 0]),
            'mean_y': np.mean(points_array[:, 1]),
            'std_x': np.std(points_array[:, 0]),
            'std_y': np.std(points_array[:, 1]),
            'spread': np.sqrt(np.var(points_array[:, 0]) + np.var(points_array[:, 1]))
        }

    stats1 = calculate_spatial_stats(center_points1)
    stats2 = calculate_spatial_stats(center_points2)

    # Create comparison bar chart
    metrics = ['Mean X', 'Mean Y', 'Std X', 'Std Y', 'Spread']
    values1 = [stats1.get('mean_x', 0), stats1.get('mean_y', 0),
               stats1.get('std_x', 0), stats1.get('std_y', 0), stats1.get('spread', 0)]
    values2 = [stats2.get('mean_x', 0), stats2.get('mean_y', 0),
               stats2.get('std_x', 0), stats2.get('std_y', 0), stats2.get('spread', 0)]

    x_pos = np.arange(len(metrics))
    width = 0.35

    ax6.bar(x_pos - width / 2, values1, width, label=model1_name, alpha=0.7, color='red')
    ax6.bar(x_pos + width / 2, values2, width, label=model2_name, alpha=0.7, color='blue')

    ax6.set_title('Spatial Distribution Comparison')
    ax6.set_xlabel('Metrics')
    ax6.set_ylabel('Values')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(metrics, rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'selection_process_analysis_round{round_num}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Selection process analysis saved: {output_path}")


def create_clustering_analysis(center_points1, center_points2, model1_name, model2_name,
                               round_num, output_dir, image_size=512):
    """Create comprehensive clustering analysis."""
    if not center_points1 or not center_points2:
        print(f"⚠️  Warning: Missing center points for round {round_num}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Clustering Analysis - Round {round_num}\n{model1_name} vs {model2_name}',
                 fontsize=16, fontweight='bold')

    # Plot 1: Hierarchical clustering dendrogram for model1
    ax1 = axes[0, 0]
    if len(center_points1) > 2:
        points1 = np.array(center_points1)
        linkage_matrix1 = linkage(points1, method='ward')
        dendrogram(linkage_matrix1, ax=ax1, leaf_rotation=90)
        ax1.set_title(f'{model1_name} Hierarchical Clustering')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Distance')

    # Plot 2: Hierarchical clustering dendrogram for model2
    ax2 = axes[0, 1]
    if len(center_points2) > 2:
        points2 = np.array(center_points2)
        linkage_matrix2 = linkage(points2, method='ward')
        dendrogram(linkage_matrix2, ax=ax2, leaf_rotation=90)
        ax2.set_title(f'{model2_name} Hierarchical Clustering')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Distance')

    # Plot 3: Silhouette analysis
    ax3 = axes[1, 0]
    silhouette_scores1 = []
    silhouette_scores2 = []
    k_range = range(2, min(6, len(center_points1), len(center_points2)))

    for k in k_range:
        if len(center_points1) >= k:
            kmeans1 = KMeans(n_clusters=k, random_state=42)
            cluster_labels1 = kmeans1.fit_predict(center_points1)
            score1 = silhouette_score(center_points1, cluster_labels1)
            silhouette_scores1.append(score1)
        else:
            silhouette_scores1.append(0)

        if len(center_points2) >= k:
            kmeans2 = KMeans(n_clusters=k, random_state=42)
            cluster_labels2 = kmeans2.fit_predict(center_points2)
            score2 = silhouette_score(center_points2, cluster_labels2)
            silhouette_scores2.append(score2)
        else:
            silhouette_scores2.append(0)

    ax3.plot(k_range, silhouette_scores1, marker='o', label=model1_name, linewidth=2)
    ax3.plot(k_range, silhouette_scores2, marker='s', label=model2_name, linewidth=2)
    ax3.set_title('Silhouette Score Analysis')
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Silhouette Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cluster quality comparison
    ax4 = axes[1, 1]
    if len(center_points1) > 2 and len(center_points2) > 2:
        # Calculate optimal number of clusters
        optimal_k1 = k_range[np.argmax(silhouette_scores1)]
        optimal_k2 = k_range[np.argmax(silhouette_scores2)]

        # Final clustering with optimal k
        kmeans1_final = KMeans(n_clusters=optimal_k1, random_state=42)
        kmeans2_final = KMeans(n_clusters=optimal_k2, random_state=42)

        cluster_labels1_final = kmeans1_final.fit_predict(center_points1)
        cluster_labels2_final = kmeans2_final.fit_predict(center_points2)

        # Calculate cluster statistics
        def calculate_cluster_stats(points, labels):
            stats = {}
            for i in range(max(labels) + 1):
                cluster_points = [points[j] for j in range(len(points)) if labels[j] == i]
                if cluster_points:
                    cluster_array = np.array(cluster_points)
                    stats[f'cluster_{i}_size'] = len(cluster_points)
                    stats[f'cluster_{i}_spread'] = np.sqrt(np.var(cluster_array[:, 0]) + np.var(cluster_array[:, 1]))
            return stats

        stats1 = calculate_cluster_stats(center_points1, cluster_labels1_final)
        stats2 = calculate_cluster_stats(center_points2, cluster_labels2_final)

        # Create comparison
        cluster_sizes1 = [stats1.get(f'cluster_{i}_size', 0) for i in range(optimal_k1)]
        cluster_sizes2 = [stats2.get(f'cluster_{i}_size', 0) for i in range(optimal_k2)]

        x_pos1 = np.arange(optimal_k1)
        x_pos2 = np.arange(optimal_k2)

        ax4.bar(x_pos1 - 0.2, cluster_sizes1, 0.4, label=model1_name, alpha=0.7, color='red')
        ax4.bar(x_pos2 + 0.2, cluster_sizes2, 0.4, label=model2_name, alpha=0.7, color='blue')

        ax4.set_title('Optimal Cluster Size Comparison')
        ax4.set_xlabel('Cluster Index')
        ax4.set_ylabel('Cluster Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'clustering_analysis_round{round_num}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Clustering analysis saved: {output_path}")


def create_coverage_efficiency_analysis(center_points1, center_points2, model1_name, model2_name,
                                        round_num, output_dir, image_size=512):
    """Create coverage efficiency analysis."""
    if not center_points1 or not center_points2:
        print(f"⚠️  Warning: Missing center points for round {round_num}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Coverage Efficiency Analysis - Round {round_num}\n{model1_name} vs {model2_name}',
                 fontsize=16, fontweight='bold')

    # Plot 1: Voronoi diagram analysis
    ax1 = axes[0, 0]
    from scipy.spatial import Voronoi, voronoi_plot_2d

    if len(center_points1) > 2:
        points1 = np.array(center_points1)
        vor1 = Voronoi(points1)
        voronoi_plot_2d(vor1, ax=ax1, show_vertices=False, line_colors='red', line_width=2)
        ax1.scatter(points1[:, 0], points1[:, 1], c='red', s=50, alpha=0.8, marker='o')

    ax1.set_xlim(0, image_size)
    ax1.set_ylim(0, image_size)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax1.set_title(f'{model1_name} Voronoi Diagram')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Voronoi diagram for model2
    ax2 = axes[0, 1]
    if len(center_points2) > 2:
        points2 = np.array(center_points2)
        vor2 = Voronoi(points2)
        voronoi_plot_2d(vor2, ax=ax2, show_vertices=False, line_colors='blue', line_width=2)
        ax2.scatter(points2[:, 0], points2[:, 1], c='blue', s=50, alpha=0.8, marker='s')

    ax2.set_xlim(0, image_size)
    ax2.set_ylim(0, image_size)
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    ax2.set_title(f'{model2_name} Voronoi Diagram')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Coverage efficiency metrics
    ax3 = axes[1, 0]

    def calculate_coverage_metrics(points):
        if len(points) < 2:
            return {}
        points_array = np.array(points)

        # Calculate convex hull area
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points_array)
            hull_area = hull.volume if len(points_array[0]) == 2 else hull.area
        except:
            hull_area = 0

        # Calculate bounding box area
        min_x, max_x = np.min(points_array[:, 0]), np.max(points_array[:, 0])
        min_y, max_y = np.min(points_array[:, 1]), np.max(points_array[:, 1])
        bbox_area = (max_x - min_x) * (max_y - min_y)

        # Calculate spread
        spread = np.sqrt(np.var(points_array[:, 0]) + np.var(points_array[:, 1]))

        return {
            'hull_area': hull_area,
            'bbox_area': bbox_area,
            'spread': spread,
            'efficiency': hull_area / bbox_area if bbox_area > 0 else 0
        }

    metrics1 = calculate_coverage_metrics(center_points1)
    metrics2 = calculate_coverage_metrics(center_points2)

    # Create comparison bar chart
    metric_names = ['Hull Area', 'BBox Area', 'Spread', 'Efficiency']
    values1 = [metrics1.get('hull_area', 0), metrics1.get('bbox_area', 0),
               metrics1.get('spread', 0), metrics1.get('efficiency', 0)]
    values2 = [metrics2.get('hull_area', 0), metrics2.get('bbox_area', 0),
               metrics2.get('spread', 0), metrics2.get('efficiency', 0)]

    x_pos = np.arange(len(metric_names))
    width = 0.35

    ax3.bar(x_pos - width / 2, values1, width, label=model1_name, alpha=0.7, color='red')
    ax3.bar(x_pos + width / 2, values2, width, label=model2_name, alpha=0.7, color='blue')

    ax3.set_title('Coverage Efficiency Metrics')
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Values')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metric_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Spatial distribution quality
    ax4 = axes[1, 1]

    def calculate_spatial_quality(points):
        if len(points) < 2:
            return {}
        points_array = np.array(points)

        # Calculate pairwise distances
        distances = pdist(points_array)

        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'uniformity': 1 - (np.std(distances) / np.mean(distances)) if np.mean(distances) > 0 else 0
        }

    quality1 = calculate_spatial_quality(center_points1)
    quality2 = calculate_spatial_quality(center_points2)

    # Create comparison
    quality_names = ['Mean Dist', 'Std Dist', 'Min Dist', 'Max Dist', 'Uniformity']
    q_values1 = [quality1.get('mean_distance', 0), quality1.get('std_distance', 0),
                 quality1.get('min_distance', 0), quality1.get('max_distance', 0),
                 quality1.get('uniformity', 0)]
    q_values2 = [quality2.get('mean_distance', 0), quality2.get('std_distance', 0),
                 quality2.get('min_distance', 0), quality2.get('max_distance', 0),
                 quality2.get('uniformity', 0)]

    x_pos = np.arange(len(quality_names))
    ax4.bar(x_pos - width / 2, q_values1, width, label=model1_name, alpha=0.7, color='red')
    ax4.bar(x_pos + width / 2, q_values2, width, label=model2_name, alpha=0.7, color='blue')

    ax4.set_title('Spatial Distribution Quality')
    ax4.set_xlabel('Quality Metrics')
    ax4.set_ylabel('Values')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(quality_names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'coverage_efficiency_analysis_round{round_num}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Coverage efficiency analysis saved: {output_path}")


if __name__ == "__main__":
    main()
