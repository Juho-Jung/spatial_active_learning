#!/usr/bin/env python3
"""
Comprehensive analysis of calcified nodule active learning results
Comparing uncertainty vs random sampling with spatial metrics
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(directory):
    """Load results from a directory"""
    results = {}
    for method_dir in Path(directory).iterdir():
        if method_dir.is_dir():
            results_file = method_dir / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                method_name = method_dir.name.split('_')[0]  # Extract method name
                results[method_name] = json.load(f)
    return results


def extract_metrics(results):
    """Extract key metrics from results"""
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


def create_comparison_plots(dynamic_results, equal_results):
    """Create comprehensive comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Calcified Nodule Active Learning Results Comparison', fontsize=16, fontweight='bold')

    # Define colors for different methods
    colors = {
        'uncertainty': '#2E86AB',
        'random': '#A23B72',
        'adaptive': '#F18F01',
        'adaptive_performance_monitoring': '#C73E1D',
        'adaptive_multi_scale': '#7209B7'
    }

    # Plot 1: Validation Dice Score
    ax1 = axes[0, 0]
    for method, metrics in dynamic_results.items():
        if method in ['uncertainty', 'random']:
            ax1.plot(metrics['rounds'], metrics['val_dice'],
                     marker='o', label=f'{method} (dynamic)',
                     color=colors.get(method, '#666666'), linewidth=2)

    for method, metrics in equal_results.items():
        if method in ['uncertainty', 'random']:
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
        if method in ['uncertainty', 'random']:
            ax2.plot(metrics['rounds'], metrics['performance_coverage'],
                     marker='o', label=f'{method} (dynamic)',
                     color=colors.get(method, '#666666'), linewidth=2)

    for method, metrics in equal_results.items():
        if method in ['uncertainty', 'random']:
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
        if method in ['uncertainty', 'random']:
            ax3.plot(metrics['rounds'], metrics['spatial_consistency'],
                     marker='o', label=f'{method} (dynamic)',
                     color=colors.get(method, '#666666'), linewidth=2)

    for method, metrics in equal_results.items():
        if method in ['uncertainty', 'random']:
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
        if method in ['uncertainty', 'random']:
            ax4.plot(metrics['rounds'], metrics['avg_bin_dice'],
                     marker='o', label=f'{method} (dynamic)',
                     color=colors.get(method, '#666666'), linewidth=2)

    for method, metrics in equal_results.items():
        if method in ['uncertainty', 'random']:
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
        if method in ['uncertainty', 'random']:
            ax5.plot(metrics['rounds'], metrics['bin_std'],
                     marker='o', label=f'{method} (dynamic)',
                     color=colors.get(method, '#666666'), linewidth=2)

    for method, metrics in equal_results.items():
        if method in ['uncertainty', 'random']:
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
        if method in ['uncertainty', 'random']:
            methods.append(f'{method}\n(dynamic)')
            final_dice.append(metrics['val_dice'][-1])
            colors_list.append(colors.get(method, '#666666'))

    for method, metrics in equal_results.items():
        if method in ['uncertainty', 'random']:
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
    return fig


def create_advanced_methods_plot(dynamic_results, equal_results):
    """Create plot comparing advanced methods"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Advanced Methods Performance Comparison', fontsize=16, fontweight='bold')

    colors = {
        'adaptive_performance_monitoring': '#C73E1D',
        'adaptive_multi_scale': '#7209B7',
        'uncertainty': '#2E86AB',
        'random': '#A23B72'
    }

    # Plot 1: Performance Monitoring vs Baseline
    ax1 = axes[0, 0]
    for method, metrics in dynamic_results.items():
        if method in ['adaptive_performance_monitoring', 'uncertainty', 'random']:
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
        if method in ['adaptive_multi_scale', 'uncertainty', 'random']:
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
        if method in ['adaptive_performance_monitoring', 'adaptive_multi_scale', 'uncertainty', 'random']:
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
        if method in ['adaptive_performance_monitoring', 'adaptive_multi_scale', 'uncertainty', 'random']:
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
    return fig


def generate_summary_statistics(dynamic_results, equal_results):
    """Generate summary statistics"""
    print("=" * 80)
    print("CALCIFIED NODULE ACTIVE LEARNING RESULTS ANALYSIS")
    print("=" * 80)

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

    # Advanced methods comparison
    if 'adaptive_performance_monitoring' in dynamic_results:
        perf_mon_dice = dynamic_results['adaptive_performance_monitoring']['val_dice']
        print(f"\nPERFORMANCE MONITORING:")
        print(f"Final Dice: {perf_mon_dice[-1]:.4f}")
        if 'uncertainty' in dynamic_results:
            improvement = ((perf_mon_dice[-1] - unc_dice[-1]) / unc_dice[-1] * 100)
            print(f"vs Uncertainty: {improvement:.2f}%")

    if 'adaptive_multi_scale' in dynamic_results:
        multiscale_dice = dynamic_results['adaptive_multi_scale']['val_dice']
        print(f"\nMULTI-SCALE:")
        print(f"Final Dice: {multiscale_dice[-1]:.4f}")
        if 'uncertainty' in dynamic_results:
            improvement = ((multiscale_dice[-1] - unc_dice[-1]) / unc_dice[-1] * 100)
            print(f"vs Uncertainty: {improvement:.2f}%")


def main():
    """Main analysis function"""
    # Load results
    dynamic_dir = "/opt/spatial_active_learning/result/calcifiednodule_both_spatial_dynamic_split_r_10_s_50_val_100"
    equal_dir = "/opt/spatial_active_learning/result/calcifiednodule_both_spatial_equal_split_r_10_s_50_val_100"

    print("Loading results...")
    dynamic_results = {}
    equal_results = {}

    # Load dynamic split results
    for method_dir in Path(dynamic_dir).iterdir():
        if method_dir.is_dir():
            results_file = method_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    method_name = method_dir.name.split('_')[0]
                    dynamic_results[method_name] = extract_metrics(json.load(f))

    # Load equal split results
    for method_dir in Path(equal_dir).iterdir():
        if method_dir.is_dir():
            results_file = method_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    method_name = method_dir.name.split('_')[0]
                    equal_results[method_name] = extract_metrics(json.load(f))

    # Generate summary statistics
    generate_summary_statistics(dynamic_results, equal_results)

    # Create comparison plots
    print("\nGenerating comparison plots...")
    fig1 = create_comparison_plots(dynamic_results, equal_results)
    fig1.savefig('/opt/spatial_active_learning/result/comparison_analysis.png',
                 dpi=300, bbox_inches='tight')

    fig2 = create_advanced_methods_plot(dynamic_results, equal_results)
    fig2.savefig('/opt/spatial_active_learning/result/advanced_methods_analysis.png',
                 dpi=300, bbox_inches='tight')

    print("Analysis complete! Plots saved to result directory.")

    # Print improvement recommendations
    print("\n" + "=" * 80)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("=" * 80)

    if 'uncertainty' in dynamic_results and 'random' in dynamic_results:
        unc_final = dynamic_results['uncertainty']['val_dice'][-1]
        rand_final = dynamic_results['random']['val_dice'][-1]

        if unc_final > rand_final:
            print("✓ Uncertainty sampling shows improvement over random sampling")
        else:
            print("⚠ Uncertainty sampling underperforms random sampling")

    if 'adaptive_performance_monitoring' in dynamic_results:
        perf_mon_final = dynamic_results['adaptive_performance_monitoring']['val_dice'][-1]
        if 'uncertainty' in dynamic_results:
            if perf_mon_final > dynamic_results['uncertainty']['val_dice'][-1]:
                print("✓ Performance monitoring shows improvement over uncertainty sampling")
            else:
                print("⚠ Performance monitoring needs optimization")

    print("\nKey Recommendations:")
    print("1. Focus on spatial consistency improvement - current values are low")
    print("2. Optimize performance coverage - aim for >0.5 coverage")
    print("3. Consider ensemble uncertainty methods for better sample selection")
    print("4. Implement adaptive learning rate scheduling based on spatial metrics")
    print("5. Add diversity constraints to prevent over-sampling similar regions")


if __name__ == "__main__":
    main()
