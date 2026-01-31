#!/usr/bin/env python3
"""
Run Statistical Analysis on Active Learning Results

Example script to perform statistical significance testing
between different Active Learning strategies.
"""

import os
import sys
from pathlib import Path

from statistical_analysis import run_statistical_analysis


def main():
    """Run statistical analysis on experiment results."""

    # Define paths to results files
    # Update these paths to match your actual experiment results
    result_paths = [
        "/team/team_pxi/workspace/juhojung/spatial_active_learning/al_results/calcifiednodule_both_spatial_equal_split_r_10_s_50_val_100/random_none_5x5_20251020_145954/results.json",
        "/team/team_pxi/workspace/juhojung/spatial_active_learning/al_results/calcifiednodule_both_spatial_equal_split_r_10_s_50_val_100/adaptive_improved_none_5x5_20251020_204137/results.json"
    ]

    # Check if result files exist
    existing_paths = []
    for path in result_paths:
        if os.path.exists(path):
            existing_paths.append(path)
            print(f"✅ Found results: {path}")
        else:
            print(f"❌ Results not found: {path}")

    if len(existing_paths) < 2:
        print("⚠️ Need at least 2 result files for comparison")
        print("Please update the result_paths in this script")
        return

    # Define metrics to analyze
    metrics_to_analyze = [
        'val_dice', 'val_loss',
        'worst_bin_dice', 'p10_bin_dice', 'bin_std',
        'spatial_consistency', 'performance_coverage'
    ]

    # Output directory for analysis results
    output_dir = "./statistical_analysis_results"

    print(f"🔬 Starting statistical analysis...")
    print(f"📊 Analyzing {len(existing_paths)} experiments")
    print(f"📈 Metrics: {', '.join(metrics_to_analyze)}")
    print(f"📁 Output directory: {output_dir}")

    # Run statistical analysis
    try:
        results = run_statistical_analysis(
            result_paths=existing_paths,
            output_dir=output_dir,
            metrics=metrics_to_analyze
        )

        print("\n✅ Statistical analysis completed!")
        print(f"📄 Results saved to: {output_dir}")

        # Print summary of key findings
        print("\n📊 KEY FINDINGS:")
        for metric, comparisons in results.items():
            print(f"\n🔍 {metric.upper()}:")
            for comparison_key, stats in comparisons.items():
                exp1, exp2 = comparison_key.split('_vs_')
                mean_diff = stats['mean_difference']
                t_significant = stats.get('t_significant', False)
                effect_size = stats.get('effect_size', 'unknown')

                print(f"   {exp1} vs {exp2}:")
                print(f"     Mean difference: {mean_diff:.4f}")
                print(f"     t-test significant: {'Yes' if t_significant else 'No'}")
                print(f"     Effect size: {effect_size}")

                if t_significant:
                    direction = "better" if mean_diff > 0 else "worse"
                    print(f"     → {exp1} is statistically {direction} than {exp2}")
                else:
                    print(f"     → No statistically significant difference")

    except Exception as e:
        print(f"❌ Statistical analysis failed: {e}")
        return


if __name__ == "__main__":
    main()
