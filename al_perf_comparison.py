#!/usr/bin/env python3
"""
Active Learning Performance Comparison

Compare performance metrics across different baseline strategies.
Generates comparison plots for spatial metrics, performance coverage, and combined metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Active Learning Performance Comparison')
    parser.add_argument('--result_paths', nargs='+', required=True,
                       help='List of result.json file paths to compare (max 8)')
    parser.add_argument('--output_dir', default='output_comparison/performance',
                       help='Output directory for comparison plots')
    parser.add_argument('--target_lesion', default='pneumoperitoneum',
                       help='Target lesion type for plot titles')
    parser.add_argument('--max_models', type=int, default=8,
                       help='Maximum number of models to compare (default: 8)')
    return parser.parse_args()


def extract_model_name(result_path):
    """Extract model name from result path."""
    # Extract the strategy_uncertainty part from path
    # e.g., /path/to/adaptive_base_20250919_172458/results.json -> adaptive_base
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


def print_summary(metrics_data):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    
    for model_name, data in metrics_data.items():
        print(f"\n📊 {model_name.upper()}:")
        print(f"   Final Validation Dice: {data['val_dice'][-1]:.4f}")
        print(f"   Final Validation Loss: {data['val_loss'][-1]:.4f}")
        print(f"   Final Worst Bin Dice: {data['worst_bin_dice'][-1]:.4f}")
        print(f"   Final Performance Coverage: {data['performance_coverage'][-1]:.4f}")
        print(f"   Final Spatial Consistency: {data['spatial_consistency'][-1]:.4f}")
        print(f"   Average Bin Dice Improvement: {data['avg_bin_dice'][-1] - data['avg_bin_dice'][0]:.4f}")


def main():
    """Main function for performance comparison."""
    args = parse_arguments()
    
    # Load results first to get model names
    results = load_results(args.result_paths, args.max_models)
    
    if not results:
        print("❌ No valid results found!")
        return
    
    # Create comparison directory based on model names
    model_names = sorted(results.keys())
    comparison_name = "_".join(model_names)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, comparison_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting Active Learning Performance Comparison")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Target lesion: {args.target_lesion}")
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


if __name__ == "__main__":
    main()
