#!/usr/bin/env python3
"""
Comprehensive Active Learning Performance Analysis and Comparison

Compare performance metrics across different baseline strategies.
Generates comparison plots for spatial metrics, performance coverage, and combined metrics.
Includes comprehensive analysis functionality from result_analysis.py.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
sys.path.append('/opt/pxi/projects/calcified_nodule/spatial_active_learning')
sys.path.append('/opt/pxi/projects/calcified_nodule/spatial_active_learning/sam_adaptation')
sys.path.append('/opt/pxi')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Active Learning Performance Comparison')
    parser.add_argument('--result_paths', nargs='+', required=True,
                        help='List of result.json file paths to compare (max 8)')
    parser.add_argument('--output_dir', default='/team/team_pxi/workspace/juhojung/spatial_active_learning/output_comparison/performance',
                        help='Output directory for comparison plots')
    parser.add_argument('--target_lesion', default='calcifiednodule',
                        help='Target lesion type for plot titles')
    parser.add_argument('--max_models', type=int, default=8,
                        help='Maximum number of models to compare (default: 8)')
    parser.add_argument('--split_type', choices=['spatial_dynamic_split', 'spatial_equal_split'],
                        help='Split type to determine subdirectory (auto-detected if not provided)')
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
                elif 'adaptive_improved_extreme' in full_name:
                    return 'adaptive_improved_extreme'
                elif 'adaptive_multi_scale_extreme' in full_name:
                    return 'adaptive_multi_scale_extreme'
                elif 'adaptive_performance_monitoring_extreme' in full_name:
                    return 'adaptive_performance_monitoring_extreme'
                elif 'adaptive_extreme' in full_name:
                    return 'adaptive_extreme'
                elif 'adaptive_improved_ultra' in full_name:
                    return 'adaptive_improved_ultra'
                elif 'adaptive_multi_scale_ultra' in full_name:
                    return 'adaptive_multi_scale_ultra'
                elif 'adaptive_performance_monitoring_ultra' in full_name:
                    return 'adaptive_performance_monitoring_ultra'
                elif 'adaptive_ultra' in full_name:
                    return 'adaptive_ultra'
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


def detect_split_type(result_paths):
    """Detect split type from result paths."""
    for result_path in result_paths:
        if 'spatial_dynamic_split' in result_path:
            return 'spatial_dynamic_split'
        elif 'spatial_equal_split' in result_path:
            return 'spatial_equal_split'
    return 'unknown'


def load_results(result_paths, max_models=8):
    """Load results from multiple JSON files."""
    results = {}

    if len(result_paths) > max_models:
        print(f"⚠️  Warning: {len(result_paths)} result files provided, but max_models is {max_models}")
        print(f"   Using only the first {max_models} files")
        result_paths = result_paths[:max_models]

    for i, result_path in enumerate(result_paths):
        if not os.path.exists(result_path):
            print(f"⚠️  Warning: Result file not found: {result_path}")
            continue

        model_name = extract_model_name(result_path)
        print(f"📊 Loading results for {model_name} from {result_path}")

        try:
            with open(result_path, 'r') as f:
                data = json.load(f)
            results[model_name] = data
            print(f"✅ Loaded {len(data)} rounds for {model_name}")
        except Exception as e:
            print(f"❌ Error loading {result_path}: {e}")
            continue

    return results


def extract_metrics(results):
    """Extract metrics from results for plotting."""
    metrics_data = defaultdict(lambda: defaultdict(list))

    for model_name, rounds in results.items():
        for round_data in rounds:
            round_num = round_data['round']

            # Basic metrics
            metrics_data[model_name]['round'].append(round_num)
            metrics_data[model_name]['val_loss'].append(round_data['val_loss'])
            metrics_data[model_name]['val_dice'].append(round_data['val_dice'])

            # Spatial metrics
            spatial_metrics = round_data['spatial_metrics']
            metrics_data[model_name]['worst_bin_dice'].append(spatial_metrics['worst_bin_dice'])
            metrics_data[model_name]['p10_bin_dice'].append(spatial_metrics['p10_bin_dice'])
            metrics_data[model_name]['bin_std'].append(spatial_metrics['bin_std'])
            metrics_data[model_name]['spatial_consistency'].append(spatial_metrics['spatial_consistency'])
            metrics_data[model_name]['performance_coverage'].append(spatial_metrics['performance_coverage'])
            metrics_data[model_name]['avg_bin_dice'].append(spatial_metrics['avg_bin_dice'])
            metrics_data[model_name]['min_bin_dice'].append(spatial_metrics['min_bin_dice'])
            metrics_data[model_name]['max_bin_dice'].append(spatial_metrics['max_bin_dice'])

    return metrics_data


def plot_spatial_metrics(metrics_data, output_dir, target_lesion):
    """Plot spatial metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Spatial Metrics Comparison - {target_lesion}', fontsize=16, fontweight='bold')

    # Plot 1: Worst Bin Dice
    ax1 = axes[0, 0]
    for model_name, data in metrics_data.items():
        ax1.plot(data['round'], data['worst_bin_dice'], marker='o', label=model_name, linewidth=2)
    ax1.set_title('Worst Bin Dice')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Worst Bin Dice')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: P10 Bin Dice
    ax2 = axes[0, 1]
    for model_name, data in metrics_data.items():
        ax2.plot(data['round'], data['p10_bin_dice'], marker='s', label=model_name, linewidth=2)
    ax2.set_title('P10 Bin Dice')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('P10 Bin Dice')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Bin Standard Deviation
    ax3 = axes[1, 0]
    for model_name, data in metrics_data.items():
        ax3.plot(data['round'], data['bin_std'], marker='^', label=model_name, linewidth=2)
    ax3.set_title('Bin Standard Deviation')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Bin Std')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Spatial Consistency
    ax4 = axes[1, 1]
    for model_name, data in metrics_data.items():
        ax4.plot(data['round'], data['spatial_consistency'], marker='d', label=model_name, linewidth=2)
    ax4.set_title('Spatial Consistency')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Spatial Consistency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'spatial_metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Spatial metrics plot saved: {output_path}")


def plot_performance_coverage(metrics_data, output_dir, target_lesion):
    """Plot performance coverage comparison."""
    plt.figure(figsize=(10, 6))

    for model_name, data in metrics_data.items():
        plt.plot(data['round'], data['performance_coverage'], marker='o', label=model_name, linewidth=2)

    plt.title(f'Performance Coverage Comparison - {target_lesion}', fontsize=14, fontweight='bold')
    plt.xlabel('Round')
    plt.ylabel('Performance Coverage')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, 'performance_coverage_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Performance coverage plot saved: {output_path}")


def plot_combined_metrics(metrics_data, output_dir, target_lesion):
    """Plot combined metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Combined Metrics Comparison - {target_lesion}', fontsize=16, fontweight='bold')

    # Plot 1: Validation Dice
    ax1 = axes[0, 0]
    for model_name, data in metrics_data.items():
        ax1.plot(data['round'], data['val_dice'], marker='o', label=model_name, linewidth=2)
    ax1.set_title('Validation Dice')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Validation Dice')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    for model_name, data in metrics_data.items():
        ax2.plot(data['round'], data['val_loss'], marker='s', label=model_name, linewidth=2)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Average Bin Dice
    ax3 = axes[1, 0]
    for model_name, data in metrics_data.items():
        ax3.plot(data['round'], data['avg_bin_dice'], marker='^', label=model_name, linewidth=2)
    ax3.set_title('Average Bin Dice')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Average Bin Dice')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Min/Max Bin Dice Range
    ax4 = axes[1, 1]
    for model_name, data in metrics_data.items():
        min_dice = data['min_bin_dice']
        max_dice = data['max_bin_dice']
        ax4.plot(data['round'], min_dice, marker='v', label=f'{model_name} (Min)', linewidth=2, linestyle='--')
        ax4.plot(data['round'], max_dice, marker='^', label=f'{model_name} (Max)', linewidth=2, linestyle='-')
    ax4.set_title('Min/Max Bin Dice Range')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Bin Dice')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'combined_metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Combined metrics plot saved: {output_path}")


def plot_normalized_metrics(metrics_data, output_dir, target_lesion):
    """Plot normalized performance coverage and spatial consistency comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Normalized Spatial Metrics Comparison - {target_lesion}', fontsize=16, fontweight='bold')

    # Plot 1: Normalized Performance Coverage
    ax1 = axes[0]
    for model_name, data in metrics_data.items():
        # Normalize performance coverage to [0, 1] range
        perf_cov = np.array(data['performance_coverage'])
        if len(perf_cov) > 0:
            # Normalize by max value across all models
            all_perf_cov = []
            for other_data in metrics_data.values():
                all_perf_cov.extend(other_data['performance_coverage'])
            max_perf_cov = max(all_perf_cov)
            min_perf_cov = min(all_perf_cov)
            if max_perf_cov > min_perf_cov:
                normalized_perf_cov = (perf_cov - min_perf_cov) / (max_perf_cov - min_perf_cov)
            else:
                normalized_perf_cov = np.ones_like(perf_cov)

            ax1.plot(data['round'], normalized_perf_cov, marker='o', label=model_name, linewidth=2)
    ax1.set_title('Normalized Performance Coverage')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Normalized Performance Coverage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot 2: Normalized Spatial Consistency
    ax2 = axes[1]
    for model_name, data in metrics_data.items():
        # Normalize spatial consistency to [0, 1] range
        spatial_cons = np.array(data['spatial_consistency'])
        if len(spatial_cons) > 0:
            # Normalize by max value across all models
            all_spatial_cons = []
            for other_data in metrics_data.values():
                all_spatial_cons.extend(other_data['spatial_consistency'])
            max_spatial_cons = max(all_spatial_cons)
            min_spatial_cons = min(all_spatial_cons)
            if max_spatial_cons > min_spatial_cons:
                normalized_spatial_cons = (spatial_cons - min_spatial_cons) / (max_spatial_cons - min_spatial_cons)
            else:
                normalized_spatial_cons = np.ones_like(spatial_cons)

            ax2.plot(data['round'], normalized_spatial_cons, marker='s', label=model_name, linewidth=2)
    ax2.set_title('Normalized Spatial Consistency')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Normalized Spatial Consistency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'normalized_spatial_metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Normalized spatial metrics plot saved: {output_path}")


def load_results_from_directory(directory):
    """Load results from a directory (from result_analysis.py)."""
    results = {}
    for method_dir in Path(directory).iterdir():
        if method_dir.is_dir():
            results_file = method_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    method_name = method_dir.name.split('_')[0]  # Extract method name
                    results[method_name] = json.load(f)
    return results


def extract_metrics_from_results(results):
    """Extract key metrics from results (from result_analysis.py)."""
    metrics = {
        'rounds': [],
        'val_dice': [],
        'val_loss': [],
        'avg_bin_dice': [],
        'worst_bin_dice': [],
        'performance_coverage': [],
        'spatial_consistency': [],
        'bin_std': []
    }

    for round_data in results:
        metrics['rounds'].append(round_data['round'])
        metrics['val_dice'].append(round_data['val_dice'])
        metrics['val_loss'].append(round_data['val_loss'])

        spatial = round_data['spatial_metrics']
        metrics['avg_bin_dice'].append(spatial['avg_bin_dice'])
        metrics['worst_bin_dice'].append(spatial['worst_bin_dice'])
        metrics['performance_coverage'].append(spatial['performance_coverage'])
        metrics['spatial_consistency'].append(spatial['spatial_consistency'])
        metrics['bin_std'].append(spatial['bin_std'])

    return metrics


def create_comprehensive_comparison_plots(dynamic_results, equal_results, output_dir, target_lesion):
    """Create comprehensive comparison plots (from result_analysis.py)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{target_lesion} Active Learning Results Comparison', fontsize=16, fontweight='bold')

    # Define colors for different methods
    colors = {
        'uncertainty': '#2E86AB',
        'random': '#A23B72',
        'adaptive': '#F18F01',
        'adaptive_improved': '#F18F01',
        'adaptive_performance_monitoring': '#C73E1D',
        'adaptive_multi_scale': '#7209B7',
        'adaptive_ultra': '#FF6B35',
        'adaptive_improved_ultra': '#FF6B35',
        'adaptive_multi_scale_ultra': '#8B5CF6',
        'adaptive_performance_monitoring_ultra': '#EF4444',
        'adaptive_extreme': '#10B981',
        'adaptive_improved_extreme': '#10B981',
        'adaptive_multi_scale_extreme': '#3B82F6',
        'adaptive_performance_monitoring_extreme': '#F59E0B'
    }

    # Plot 1: Validation Dice Score
    ax1 = axes[0, 0]
    for method, metrics in dynamic_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            ax1.plot(metrics['rounds'], metrics['val_dice'],
                     marker='o', label=f'{method} (dynamic)',
                     color=colors.get(method, '#666666'), linewidth=2)

    for method, metrics in equal_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            ax1.plot(metrics['rounds'], metrics['val_dice'],
                     marker='s', linestyle='--', label=f'{method} (equal)',
                     color=colors.get(method, '#666666'), linewidth=2)

    ax1.set_title('Validation Dice Score', fontweight='bold')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Dice Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Performance Coverage
    ax2 = axes[0, 1]
    for method, metrics in dynamic_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            ax2.plot(metrics['rounds'], metrics['performance_coverage'],
                     marker='o', label=f'{method} (dynamic)',
                     color=colors.get(method, '#666666'), linewidth=2)

    for method, metrics in equal_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            ax2.plot(metrics['rounds'], metrics['performance_coverage'],
                     marker='s', linestyle='--', label=f'{method} (equal)',
                     color=colors.get(method, '#666666'), linewidth=2)

    ax2.set_title('Performance Coverage', fontweight='bold')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Coverage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Spatial Consistency
    ax3 = axes[0, 2]
    for method, metrics in dynamic_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            ax3.plot(metrics['rounds'], metrics['spatial_consistency'],
                     marker='o', label=f'{method} (dynamic)',
                     color=colors.get(method, '#666666'), linewidth=2)

    for method, metrics in equal_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            ax3.plot(metrics['rounds'], metrics['spatial_consistency'],
                     marker='s', linestyle='--', label=f'{method} (equal)',
                     color=colors.get(method, '#666666'), linewidth=2)

    ax3.set_title('Spatial Consistency', fontweight='bold')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Consistency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Average Bin Dice
    ax4 = axes[1, 0]
    for method, metrics in dynamic_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            ax4.plot(metrics['rounds'], metrics['avg_bin_dice'],
                     marker='o', label=f'{method} (dynamic)',
                     color=colors.get(method, '#666666'), linewidth=2)

    for method, metrics in equal_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            ax4.plot(metrics['rounds'], metrics['avg_bin_dice'],
                     marker='s', linestyle='--', label=f'{method} (equal)',
                     color=colors.get(method, '#666666'), linewidth=2)

    ax4.set_title('Average Bin Dice', fontweight='bold')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Dice Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Bin Standard Deviation
    ax5 = axes[1, 1]
    for method, metrics in dynamic_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            ax5.plot(metrics['rounds'], metrics['bin_std'],
                     marker='o', label=f'{method} (dynamic)',
                     color=colors.get(method, '#666666'), linewidth=2)

    for method, metrics in equal_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            ax5.plot(metrics['rounds'], metrics['bin_std'],
                     marker='s', linestyle='--', label=f'{method} (equal)',
                     color=colors.get(method, '#666666'), linewidth=2)

    ax5.set_title('Bin Standard Deviation', fontweight='bold')
    ax5.set_xlabel('Round')
    ax5.set_ylabel('Std Dev')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Final Performance Comparison
    ax6 = axes[1, 2]
    methods = []
    final_dice = []
    colors_list = []

    for method, metrics in dynamic_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            methods.append(f'{method}\n(dynamic)')
            final_dice.append(metrics['val_dice'][-1])
            colors_list.append(colors.get(method, '#666666'))

    for method, metrics in equal_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            methods.append(f'{method}\n(equal)')
            final_dice.append(metrics['val_dice'][-1])
            colors_list.append(colors.get(method, '#666666'))

    bars = ax6.bar(methods, final_dice, color=colors_list, alpha=0.7)
    ax6.set_title('Final Dice Score Comparison', fontweight='bold')
    ax6.set_ylabel('Dice Score')
    ax6.set_ylim(0, 0.6)

    # Add value labels on bars
    for bar, value in zip(bars, final_dice):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'comprehensive_comparison_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Comprehensive comparison plot saved: {output_path}")


def create_advanced_methods_plot(dynamic_results, equal_results, output_dir, target_lesion):
    """Create plot comparing advanced methods (from result_analysis.py)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Advanced Methods Performance Comparison - {target_lesion}', fontsize=16, fontweight='bold')

    colors = {
        'adaptive_performance_monitoring': '#C73E1D',
        'adaptive_multi_scale': '#7209B7',
        'adaptive_performance_monitoring_ultra': '#EF4444',
        'adaptive_multi_scale_ultra': '#8B5CF6',
        'adaptive_performance_monitoring_extreme': '#F59E0B',
        'adaptive_multi_scale_extreme': '#3B82F6',
        'uncertainty': '#2E86AB',
        'random': '#A23B72'
    }

    # Plot 1: Performance Monitoring vs Baseline
    ax1 = axes[0, 0]
    for method, metrics in dynamic_results.items():
        if method in ['adaptive_performance_monitoring', 'adaptive_performance_monitoring_ultra', 'adaptive_performance_monitoring_extreme', 'uncertainty', 'random']:
            ax1.plot(metrics['rounds'], metrics['val_dice'],
                     marker='o', label=method.replace('_', ' ').title(),
                     color=colors.get(method, '#666666'), linewidth=2)

    ax1.set_title('Performance Monitoring vs Baselines (Dynamic)', fontweight='bold')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Dice Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Multi-scale vs Baseline
    ax2 = axes[0, 1]
    for method, metrics in dynamic_results.items():
        if method in ['adaptive_multi_scale', 'adaptive_multi_scale_ultra', 'adaptive_multi_scale_extreme', 'uncertainty', 'random']:
            ax2.plot(metrics['rounds'], metrics['val_dice'],
                     marker='o', label=method.replace('_', ' ').title(),
                     color=colors.get(method, '#666666'), linewidth=2)

    ax2.set_title('Multi-scale vs Baselines (Dynamic)', fontweight='bold')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Performance Coverage Comparison
    ax3 = axes[1, 0]
    for method, metrics in dynamic_results.items():
        if method in ['adaptive_performance_monitoring', 'adaptive_multi_scale', 'adaptive_performance_monitoring_ultra', 'adaptive_multi_scale_ultra', 'adaptive_performance_monitoring_extreme', 'adaptive_multi_scale_extreme', 'uncertainty', 'random']:
            ax3.plot(metrics['rounds'], metrics['performance_coverage'],
                     marker='o', label=method.replace('_', ' ').title(),
                     color=colors.get(method, '#666666'), linewidth=2)

    ax3.set_title('Performance Coverage Comparison', fontweight='bold')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Coverage')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final Performance Bar Chart
    ax4 = axes[1, 1]
    methods = []
    final_dice = []
    colors_list = []

    for method, metrics in dynamic_results.items():
        if method in ['adaptive_performance_monitoring', 'adaptive_multi_scale', 'adaptive_performance_monitoring_ultra', 'adaptive_multi_scale_ultra', 'adaptive_performance_monitoring_extreme', 'adaptive_multi_scale_extreme', 'uncertainty', 'random']:
            methods.append(method.replace('_', ' ').title())
            final_dice.append(metrics['val_dice'][-1])
            colors_list.append(colors.get(method, '#666666'))

    bars = ax4.bar(methods, final_dice, color=colors_list, alpha=0.7)
    ax4.set_title('Final Performance Comparison', fontweight='bold')
    ax4.set_ylabel('Dice Score')
    ax4.set_ylim(0, 0.6)
    ax4.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, final_dice):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'advanced_methods_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Advanced methods plot saved: {output_path}")


def create_spatial_coverage_efficiency_plot(dynamic_results, equal_results, output_dir, target_lesion):
    """Create plot showing spatial coverage efficiency of adaptive methods."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Spatial Coverage Efficiency Analysis - {target_lesion}', fontsize=16, fontweight='bold')

    colors = {
        'uncertainty': '#2E86AB',
        'random': '#A23B72',
        'adaptive': '#F18F01',
        'adaptive_improved': '#F18F01',
        'adaptive_ultra': '#FF6B35',
        'adaptive_improved_ultra': '#FF6B35',
        'adaptive_extreme': '#10B981',
        'adaptive_improved_extreme': '#10B981'
    }

    # Plot 1: Coverage Efficiency (Coverage per Round)
    ax1 = axes[0, 0]
    for method, metrics in dynamic_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            # Calculate coverage efficiency (coverage / round)
            coverage_efficiency = [c / (r + 1) for r, c in zip(metrics['rounds'], metrics['performance_coverage'])]
            ax1.plot(metrics['rounds'], coverage_efficiency,
                     marker='o', label=method.replace('_', ' ').title(),
                     color=colors.get(method, '#666666'), linewidth=2)

    ax1.set_title('Spatial Coverage Efficiency (Coverage/Round)', fontweight='bold')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Coverage Efficiency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coverage Growth Rate
    ax2 = axes[0, 1]
    for method, metrics in dynamic_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            # Calculate coverage growth rate
            coverage = metrics['performance_coverage']
            growth_rate = [0] + [coverage[i] - coverage[i - 1] for i in range(1, len(coverage))]
            ax2.plot(metrics['rounds'], growth_rate,
                     marker='s', label=method.replace('_', ' ').title(),
                     color=colors.get(method, '#666666'), linewidth=2)

    ax2.set_title('Coverage Growth Rate per Round', fontweight='bold')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Coverage Growth Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Spatial Consistency vs Coverage Trade-off
    ax3 = axes[1, 0]
    for method, metrics in dynamic_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            ax3.scatter(metrics['performance_coverage'], metrics['spatial_consistency'],
                        label=method.replace('_', ' ').title(),
                        color=colors.get(method, '#666666'), s=100, alpha=0.7)

    ax3.set_title('Coverage vs Consistency Trade-off', fontweight='bold')
    ax3.set_xlabel('Performance Coverage')
    ax3.set_ylabel('Spatial Consistency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Efficiency Score (Coverage * Consistency)
    ax4 = axes[1, 1]
    for method, metrics in dynamic_results.items():
        if method in ['uncertainty', 'random', 'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra', 'adaptive_extreme', 'adaptive_improved_extreme']:
            # Calculate efficiency score
            efficiency_score = [c * s for c, s in zip(metrics['performance_coverage'], metrics['spatial_consistency'])]
            ax4.plot(metrics['rounds'], efficiency_score,
                     marker='^', label=method.replace('_', ' ').title(),
                     color=colors.get(method, '#666666'), linewidth=2)

    ax4.set_title('Spatial Efficiency Score (Coverage × Consistency)', fontweight='bold')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Efficiency Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'spatial_coverage_efficiency_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Spatial coverage efficiency plot saved: {output_path}")


def create_pareto_frontier_plot(dynamic_results, equal_results, output_dir, target_lesion):
    """Create Pareto frontier plots showing adaptive methods' superior positioning."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Pareto Frontier Analysis: Adaptive Methods Superior Positioning - {target_lesion}', 
                 fontsize=16, fontweight='bold')
    
    # Define colors for different method categories
    colors = {
        'uncertainty': '#2E86AB',      # Blue - Performance-focused
        'random': '#A23B72',           # Purple - Spatial-focused  
        'adaptive': '#F18F01',         # Orange - Balanced
        'adaptive_improved': '#F18F01',
        'adaptive_ultra': '#FF6B35',   # Red-orange - Enhanced
        'adaptive_improved_ultra': '#FF6B35',
        'adaptive_extreme': '#10B981', # Green - Superior
        'adaptive_improved_extreme': '#10B981'
    }
    
    def is_dominated(point1, point2, maximize_both=True):
        """Check if point1 is dominated by point2."""
        if maximize_both:
            # Both objectives should be maximized
            return point2[0] >= point1[0] and point2[1] >= point1[1] and (point2[0] > point1[0] or point2[1] > point1[1])
        else:
            # First objective maximized, second minimized
            return point2[0] >= point1[0] and point2[1] <= point1[1] and (point2[0] > point1[0] or point2[1] < point1[1])
    
    def find_pareto_frontier(points, maximize_both=True):
        """Find Pareto frontier points."""
        frontier = []
        for i, point1 in enumerate(points):
            is_non_dominated = True
            for j, point2 in enumerate(points):
                if i != j and is_dominated(point1, point2, maximize_both):
                    is_non_dominated = False
                    break
            if is_non_dominated:
                frontier.append(point1)
        return frontier
    
    # Plot 1: Dice vs Performance Coverage (PC) - Both maximize
    ax1 = axes[0]
    
    # Collect all points for Pareto analysis
    all_points_pc = []
    point_labels = []
    
    # Plot baseline methods
    if 'uncertainty' in dynamic_results:
        pc_val = dynamic_results['uncertainty']['performance_coverage'][-1]
        dice_val = dynamic_results['uncertainty']['val_dice'][-1]
        ax1.scatter(pc_val, dice_val, s=200, c=colors['uncertainty'], marker='o', 
                   label='Uncertainty (Performance-focused)', alpha=0.8, edgecolors='black', linewidth=2)
        all_points_pc.append((pc_val, dice_val))
        point_labels.append('uncertainty')
    
    if 'random' in dynamic_results:
        pc_val = dynamic_results['random']['performance_coverage'][-1]
        dice_val = dynamic_results['random']['val_dice'][-1]
        ax1.scatter(pc_val, dice_val, s=200, c=colors['random'], marker='s', 
                   label='Random (Spatial-focused)', alpha=0.3, edgecolors='black', linewidth=1)  # Dimmed as dominated
        all_points_pc.append((pc_val, dice_val))
        point_labels.append('random')
    
    # Plot adaptive methods
    adaptive_methods = ['adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra',
                       'adaptive_extreme', 'adaptive_improved_extreme']
    
    for method in adaptive_methods:
        if method in dynamic_results:
            pc_val = dynamic_results[method]['performance_coverage'][-1]
            dice_val = dynamic_results[method]['val_dice'][-1]
            ax1.scatter(pc_val, dice_val, s=150, c=colors.get(method, '#666666'), marker='^', 
                       label=method.replace('_', ' ').title(), alpha=0.8, edgecolors='black', linewidth=1)
            all_points_pc.append((pc_val, dice_val))
            point_labels.append(method)
    
    # Find and draw Pareto frontier
    if len(all_points_pc) > 1:
        frontier_pc = find_pareto_frontier(all_points_pc, maximize_both=True)
        
        if len(frontier_pc) > 1:
            # Sort by PC for proper line drawing
            sorted_frontier = sorted(frontier_pc, key=lambda x: x[0])
            pc_coords = [p[0] for p in sorted_frontier]
            dice_coords = [p[1] for p in sorted_frontier]
            
            ax1.plot(pc_coords, dice_coords, '-', color='red', alpha=0.7, linewidth=3,
                    label='True Pareto Frontier')
    
    ax1.set_title('Dice Score vs Performance Coverage\n(True Pareto Frontier)', fontweight='bold')
    ax1.set_xlabel('Performance Coverage (↑ better)')
    ax1.set_ylabel('Dice Score (↑ better)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add annotation
    ax1.annotate('Adaptive Methods\nSuperior Positioning', 
                xy=(0.5, 0.4), xytext=(0.3, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Plot 2: Dice vs Spatial Consistency (SC) - Dice maximize, SC minimize
    ax2 = axes[1]
    
    # Collect all points for Pareto analysis
    all_points_sc = []
    point_labels_sc = []
    
    # Plot baseline methods
    if 'uncertainty' in dynamic_results:
        sc_val = dynamic_results['uncertainty']['spatial_consistency'][-1]
        dice_val = dynamic_results['uncertainty']['val_dice'][-1]
        ax2.scatter(sc_val, dice_val, s=200, c=colors['uncertainty'], marker='o', 
                   label='Uncertainty (Performance-focused)', alpha=0.8, edgecolors='black', linewidth=2)
        all_points_sc.append((dice_val, sc_val))  # Note: (dice, sc) order for minimize SC
        point_labels_sc.append('uncertainty')
    
    if 'random' in dynamic_results:
        sc_val = dynamic_results['random']['spatial_consistency'][-1]
        dice_val = dynamic_results['random']['val_dice'][-1]
        ax2.scatter(sc_val, dice_val, s=200, c=colors['random'], marker='s', 
                   label='Random (Spatial-focused)', alpha=0.3, edgecolors='black', linewidth=1)  # Dimmed
        all_points_sc.append((dice_val, sc_val))
        point_labels_sc.append('random')
    
    # Plot adaptive methods
    for method in adaptive_methods:
        if method in dynamic_results:
            sc_val = dynamic_results[method]['spatial_consistency'][-1]
            dice_val = dynamic_results[method]['val_dice'][-1]
            ax2.scatter(sc_val, dice_val, s=150, c=colors.get(method, '#666666'), marker='^', 
                       label=method.replace('_', ' ').title(), alpha=0.8, edgecolors='black', linewidth=1)
            all_points_sc.append((dice_val, sc_val))
            point_labels_sc.append(method)
    
    # Find and draw Pareto frontier (maximize dice, minimize SC)
    if len(all_points_sc) > 1:
        frontier_sc = find_pareto_frontier(all_points_sc, maximize_both=False)
        
        if len(frontier_sc) > 1:
            # Sort by SC for proper line drawing
            sorted_frontier = sorted(frontier_sc, key=lambda x: x[1])  # Sort by SC (second element)
            sc_coords = [p[1] for p in sorted_frontier]
            dice_coords = [p[0] for p in sorted_frontier]
            
            ax2.plot(sc_coords, dice_coords, '-', color='red', alpha=0.7, linewidth=3,
                    label='True Pareto Frontier')
    
    ax2.set_title('Dice Score vs Spatial Consistency\n(True Pareto Frontier)', fontweight='bold')
    ax2.set_xlabel('Spatial Consistency (↓ better)')
    ax2.set_ylabel('Dice Score (↑ better)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Add annotation
    ax2.annotate('Adaptive Methods\nSuperior Positioning', 
                xy=(0.5, 0.4), xytext=(0.3, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'pareto_frontier_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Pareto frontier analysis plot saved: {output_path}")


def create_improvement_analysis_plot(dynamic_results, equal_results, output_dir, target_lesion):
    """Create plot showing detailed improvement analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Adaptive Methods: Improvement Analysis - {target_lesion}', fontsize=16, fontweight='bold')

    # Define baseline methods
    baseline_methods = ['uncertainty', 'random']
    adaptive_methods = ['adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra',
                        'adaptive_extreme', 'adaptive_improved_extreme']

    # Plot 1: Improvement over Uncertainty
    ax1 = axes[0, 0]
    if 'uncertainty' in dynamic_results:
        uncertainty_dice = dynamic_results['uncertainty']['val_dice']

        for method in adaptive_methods:
            if method in dynamic_results:
                method_dice = dynamic_results[method]['val_dice']
                improvement = [((m - u) / u * 100) if u > 0 else 0
                               for m, u in zip(method_dice, uncertainty_dice)]
                ax1.plot(dynamic_results[method]['rounds'], improvement,
                         marker='o', label=method.replace('_', ' ').title(), linewidth=2)

    ax1.set_title('Improvement over Uncertainty (%)', fontweight='bold')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Improvement (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Plot 2: Improvement over Random
    ax2 = axes[0, 1]
    if 'random' in dynamic_results:
        random_dice = dynamic_results['random']['val_dice']

        for method in adaptive_methods:
            if method in dynamic_results:
                method_dice = dynamic_results[method]['val_dice']
                improvement = [((m - r) / r * 100) if r > 0 else 0
                               for m, r in zip(method_dice, random_dice)]
                ax2.plot(dynamic_results[method]['rounds'], improvement,
                         marker='s', label=method.replace('_', ' ').title(), linewidth=2)

    ax2.set_title('Improvement over Random (%)', fontweight='bold')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Improvement (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Plot 3: Convergence Speed Analysis
    ax3 = axes[1, 0]
    for method in adaptive_methods:
        if method in dynamic_results:
            method_dice = dynamic_results[method]['val_dice']
            final_perf = method_dice[-1]
            target_perf = 0.9 * final_perf

            # Find convergence round
            convergence_round = next((i for i, val in enumerate(method_dice) if val >= target_perf), len(method_dice))
            ax3.bar(method.replace('_', ' ').title(), convergence_round, alpha=0.7)

    ax3.set_title('Convergence Speed (Rounds to 90% Performance)', fontweight='bold')
    ax3.set_ylabel('Rounds')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final Performance Comparison
    ax4 = axes[1, 1]
    methods = []
    final_dice = []

    for method in baseline_methods + adaptive_methods:
        if method in dynamic_results:
            methods.append(method.replace('_', ' ').title())
            final_dice.append(dynamic_results[method]['val_dice'][-1])

    bars = ax4.bar(methods, final_dice, alpha=0.7)
    ax4.set_title('Final Performance Comparison', fontweight='bold')
    ax4.set_ylabel('Final Dice Score')
    ax4.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, value in zip(bars, final_dice):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'improvement_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Improvement analysis plot saved: {output_path}")


def generate_comprehensive_summary_statistics(dynamic_results, equal_results):
    """Generate comprehensive summary statistics with enhanced improvement calculations."""
    print("=" * 80)
    print("COMPREHENSIVE ACTIVE LEARNING RESULTS ANALYSIS")
    print("=" * 80)

    # Enhanced improvement calculation function
    def calculate_improvement_metrics(adaptive_metrics, baseline_metrics, metric_name):
        """Calculate comprehensive improvement metrics."""
        if len(adaptive_metrics) == 0 or len(baseline_metrics) == 0:
            return {}

        # Final improvement
        final_improvement = ((adaptive_metrics[-1] - baseline_metrics[-1]) /
                             baseline_metrics[-1] * 100) if baseline_metrics[-1] > 0 else 0

        # Average improvement across all rounds
        avg_improvement = np.mean([((a - b) / b * 100) if b > 0 else 0
                                  for a, b in zip(adaptive_metrics, baseline_metrics)])

        # Maximum improvement
        max_improvement = max([((a - b) / b * 100) if b > 0 else 0
                              for a, b in zip(adaptive_metrics, baseline_metrics)])

        # Convergence speed (rounds to reach 90% of final performance)
        final_perf = adaptive_metrics[-1]
        target_perf = 0.9 * final_perf
        convergence_round = next((i for i, val in enumerate(adaptive_metrics)
                                 if val >= target_perf), len(adaptive_metrics))

        return {
            'final_improvement': final_improvement,
            'avg_improvement': avg_improvement,
            'max_improvement': max_improvement,
            'convergence_round': convergence_round,
            'convergence_speed': len(adaptive_metrics) - convergence_round
        }

    # Compare uncertainty vs random for dynamic split
    if 'uncertainty' in dynamic_results and 'random' in dynamic_results:
        unc_dice = dynamic_results['uncertainty']['val_dice']
        rand_dice = dynamic_results['random']['val_dice']

        print(f"\nDYNAMIC SPLIT COMPARISON:")
        print(f"Uncertainty Final Dice: {unc_dice[-1]:.4f}")
        print(f"Random Final Dice: {rand_dice[-1]:.4f}")
        print(f"Improvement: {((unc_dice[-1] - rand_dice[-1]) / rand_dice[-1] * 100):.2f}%")

        # Performance coverage comparison
        unc_coverage = dynamic_results['uncertainty']['performance_coverage']
        rand_coverage = dynamic_results['random']['performance_coverage']
        print(f"Uncertainty Final Coverage: {unc_coverage[-1]:.4f}")
        print(f"Random Final Coverage: {rand_coverage[-1]:.4f}")

    # Compare uncertainty vs random for equal split
    if 'uncertainty' in equal_results and 'random' in equal_results:
        unc_dice_eq = equal_results['uncertainty']['val_dice']
        rand_dice_eq = equal_results['random']['val_dice']

        print(f"\nEQUAL SPLIT COMPARISON:")
        print(f"Uncertainty Final Dice: {unc_dice_eq[-1]:.4f}")
        print(f"Random Final Dice: {rand_dice_eq[-1]:.4f}")
        print(f"Improvement: {((unc_dice_eq[-1] - rand_dice_eq[-1]) / rand_dice_eq[-1] * 100):.2f}%")

    # Enhanced Adaptive methods comparison with detailed metrics
    print(f"\n{'=' * 60}")
    print("ADAPTIVE METHODS: DETAILED IMPROVEMENT ANALYSIS")
    print(f"{'=' * 60}")

    baseline_methods = ['uncertainty', 'random']
    adaptive_methods = [
        'adaptive', 'adaptive_improved', 'adaptive_ultra', 'adaptive_improved_ultra',
        'adaptive_extreme', 'adaptive_improved_extreme',
        'adaptive_performance_monitoring', 'adaptive_multi_scale',
        'adaptive_performance_monitoring_ultra', 'adaptive_multi_scale_ultra',
        'adaptive_performance_monitoring_extreme', 'adaptive_multi_scale_extreme'
    ]

    for method in adaptive_methods:
        if method in dynamic_results:
            method_metrics = dynamic_results[method]
            print(f"\n🎯 {method.upper().replace('_', ' ')}:")
            print(f"   Final Dice: {method_metrics['val_dice'][-1]:.4f}")
            print(f"   Final Coverage: {method_metrics['performance_coverage'][-1]:.4f}")
            print(f"   Final Consistency: {method_metrics['spatial_consistency'][-1]:.4f}")

            # Compare against baselines
            for baseline in baseline_methods:
                if baseline in dynamic_results:
                    baseline_metrics = dynamic_results[baseline]

                    # Dice improvement
                    dice_improvement = calculate_improvement_metrics(
                        method_metrics['val_dice'], baseline_metrics['val_dice'], 'dice')

                    # Coverage improvement
                    coverage_improvement = calculate_improvement_metrics(
                        method_metrics['performance_coverage'], baseline_metrics['performance_coverage'], 'coverage')

                    # Consistency improvement
                    consistency_improvement = calculate_improvement_metrics(
                        method_metrics['spatial_consistency'], baseline_metrics['spatial_consistency'], 'consistency')

                    print(f"   vs {baseline.upper()}:")
                    print(
                        f"     Dice: {dice_improvement['final_improvement']:.2f}% (avg: {dice_improvement['avg_improvement']:.2f}%)")
                    print(
                        f"     Coverage: {coverage_improvement['final_improvement']:.2f}% (avg: {coverage_improvement['avg_improvement']:.2f}%)")
                    print(
                        f"     Consistency: {consistency_improvement['final_improvement']:.2f}% (avg: {consistency_improvement['avg_improvement']:.2f}%)")
                    print(
                        f"     Convergence: {dice_improvement['convergence_round']} rounds (speed: {dice_improvement['convergence_speed']})")


def print_summary(metrics_data):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)

    for model_name, data in metrics_data.items():
        print(f"\n📊 {model_name.upper()}:")
        print(f"   Final Validation Dice: {data['val_dice'][-1]:.4f}")
        print(f"   Final Validation Loss: {data['val_loss'][-1]:.4f}")
        print(f"   Final Worst Bin Dice: {data['worst_bin_dice'][-1]:.4f}")
        print(f"   Final Performance Coverage: {data['performance_coverage'][-1]:.4f}")
        print(f"   Final Spatial Consistency: {data['spatial_consistency'][-1]:.4f}")
        print(f"   Average Bin Dice Improvement: {data['avg_bin_dice'][-1] - data['avg_bin_dice'][0]:.4f}")


def main():
    """Main function for comprehensive performance analysis and comparison."""
    args = parse_arguments()

    # Check if we're doing directory-based analysis or file-based analysis
    if len(args.result_paths) == 1 and os.path.isdir(args.result_paths[0]):
        # Directory-based analysis (from result_analysis.py)
        print("🔍 Directory-based comprehensive analysis mode")

        # Load results from directories
        dynamic_dir = "/opt/spatial_active_learning/result/calcifiednodule_both_spatial_dynamic_split_r_10_s_50_val_100"
        equal_dir = "/opt/spatial_active_learning/result/calcifiednodule_both_spatial_equal_split_r_10_s_50_val_100"

        print("Loading results from directories...")
        dynamic_results = {}
        equal_results = {}

        # Load dynamic split results
        for method_dir in Path(dynamic_dir).iterdir():
            if method_dir.is_dir():
                results_file = method_dir / "results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        method_name = method_dir.name.split('_')[0]
                        dynamic_results[method_name] = extract_metrics_from_results(json.load(f))

        # Load equal split results
        for method_dir in Path(equal_dir).iterdir():
            if method_dir.is_dir():
                results_file = method_dir / "results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        method_name = method_dir.name.split('_')[0]
                        equal_results[method_name] = extract_metrics_from_results(json.load(f))

        # Create output directory
        output_dir = os.path.join(args.output_dir, "comprehensive_analysis")
        os.makedirs(output_dir, exist_ok=True)

        print("🚀 Starting Comprehensive Active Learning Analysis")
        print(f"📁 Output directory: {output_dir}")
        print(f"🎯 Target lesion: {args.target_lesion}")

        # Generate comprehensive summary statistics
        generate_comprehensive_summary_statistics(dynamic_results, equal_results)

        # Create comprehensive comparison plots
        print("\n📊 Generating comprehensive comparison plots...")
        create_comprehensive_comparison_plots(dynamic_results, equal_results, output_dir, args.target_lesion)
        create_advanced_methods_plot(dynamic_results, equal_results, output_dir, args.target_lesion)

        # Create enhanced analysis plots
        print("\n📊 Generating enhanced analysis plots...")
        create_spatial_coverage_efficiency_plot(dynamic_results, equal_results, output_dir, args.target_lesion)
        create_improvement_analysis_plot(dynamic_results, equal_results, output_dir, args.target_lesion)
        create_pareto_frontier_plot(dynamic_results, equal_results, output_dir, args.target_lesion)

        print(f"\n🎉 Comprehensive analysis completed!")
        print(f"📁 Results saved to: {output_dir}")

    else:
        # File-based analysis (original functionality)
        print("📊 File-based performance comparison mode")

        # Load results first to get model names
        results = load_results(args.result_paths, args.max_models)

        if not results:
            print("❌ No valid results found!")
            return

        # Detect split type
        if args.split_type:
            split_type = args.split_type
        else:
            split_type = detect_split_type(args.result_paths)

        if split_type == 'unknown':
            print("⚠️  Warning: Could not detect split type from paths. Using default directory structure.")
            split_type = 'unknown'

        # Create comparison directory based on model names and split type
        model_names = sorted(results.keys())
        comparison_name = "_".join(model_names)

        # Create output directory with split type subdirectory
        if split_type != 'unknown':
            output_dir = os.path.join(args.output_dir, split_type, comparison_name)
        else:
            output_dir = os.path.join(args.output_dir, comparison_name)

        os.makedirs(output_dir, exist_ok=True)

        print("🚀 Starting Active Learning Performance Comparison")
        print(f"📁 Output directory: {output_dir}")
        print(f"🎯 Target lesion: {args.target_lesion}")
        print(f"📊 Split type: {split_type}")
        print(f"📊 Comparing {len(args.result_paths)} result files")
        print(f"📁 Comparison name: {comparison_name}")

        print(f"\n✅ Successfully loaded results for {len(results)} models:")
        for model_name in results.keys():
            print(f"   - {model_name}")

        # Extract metrics
        metrics_data = extract_metrics(results)

        # Generate plots
        print("\n📊 Generating comparison plots...")
        plot_spatial_metrics(metrics_data, output_dir, args.target_lesion)
        plot_performance_coverage(metrics_data, output_dir, args.target_lesion)
        plot_combined_metrics(metrics_data, output_dir, args.target_lesion)
        plot_normalized_metrics(metrics_data, output_dir, args.target_lesion)

        # Print summary
        print_summary(metrics_data)

        print(f"\n🎉 Performance comparison completed!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Split type: {split_type}")


if __name__ == "__main__":
    main()
