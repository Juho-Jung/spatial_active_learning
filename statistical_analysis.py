#!/usr/bin/env python3
"""
Statistical Analysis for Active Learning Results

Provides statistical significance testing and effect size calculations
for comparing different Active Learning strategies.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap, mannwhitneyu, ttest_ind


def load_experiment_results(result_paths: List[str]) -> Dict[str, Dict]:
    """
    Load results from multiple experiments for statistical comparison.

    Args:
        result_paths: List of paths to results.json files

    Returns:
        Dictionary with experiment names as keys and results as values
    """
    results = {}

    for path in result_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                # Extract experiment name from path
                exp_name = Path(path).parent.name
                results[exp_name] = data
        else:
            print(f"⚠️ Warning: Results file not found: {path}")

    return results


def extract_metric_series(results: Dict[str, Dict], metric: str) -> Dict[str, List[float]]:
    """
    Extract a specific metric across all rounds for each experiment.

    Args:
        results: Dictionary of experiment results
        metric: Metric name to extract (e.g., 'val_dice', 'worst_bin_dice')

    Returns:
        Dictionary with experiment names and metric values
    """
    metric_series = {}

    for exp_name, exp_results in results.items():
        values = []
        for round_data in exp_results:
            if metric in round_data:
                values.append(round_data[metric])
            elif metric in round_data.get('spatial_metrics', {}):
                values.append(round_data['spatial_metrics'][metric])

        metric_series[exp_name] = values

    return metric_series


def calculate_statistical_significance(data1: List[float], data2: List[float],
                                       alpha: float = 0.05) -> Dict[str, float]:
    """
    Calculate statistical significance between two datasets.

    Args:
        data1: First dataset
        data2: Second dataset
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with statistical test results
    """
    # Convert to numpy arrays
    data1 = np.array(data1)
    data2 = np.array(data2)

    results = {}

    # 1. t-test (parametric)
    try:
        t_stat, t_pvalue = ttest_ind(data1, data2)
        results['t_statistic'] = t_stat
        results['t_pvalue'] = t_pvalue
        results['t_significant'] = t_pvalue < alpha
    except Exception as e:
        print(f"⚠️ t-test failed: {e}")
        results['t_statistic'] = np.nan
        results['t_pvalue'] = np.nan
        results['t_significant'] = False

    # 2. Mann-Whitney U test (non-parametric)
    try:
        u_stat, u_pvalue = mannwhitneyu(data1, data2, alternative='two-sided')
        results['u_statistic'] = u_stat
        results['u_pvalue'] = u_pvalue
        results['u_significant'] = u_pvalue < alpha
    except Exception as e:
        print(f"⚠️ Mann-Whitney U test failed: {e}")
        results['u_statistic'] = np.nan
        results['u_pvalue'] = np.nan
        results['u_significant'] = False

    # 3. Effect size (Cohen's d)
    try:
        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) +
                             (len(data2) - 1) * np.var(data2, ddof=1)) /
                             (len(data1) + len(data2) - 2))
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        results['cohens_d'] = cohens_d
        results['effect_size'] = interpret_effect_size(abs(cohens_d))
    except Exception as e:
        print(f"⚠️ Effect size calculation failed: {e}")
        results['cohens_d'] = np.nan
        results['effect_size'] = 'unknown'

    # 4. Bootstrap confidence intervals
    try:
        def statistic(x, y):
            return np.mean(x) - np.mean(y)

        # Bootstrap the difference
        rng = np.random.default_rng(42)
        bootstrap_result = bootstrap(
            (data1, data2), statistic, n_resamples=1000,
            confidence_level=0.95, random_state=rng
        )

        results['bootstrap_ci_lower'] = bootstrap_result.confidence_interval.low
        results['bootstrap_ci_upper'] = bootstrap_result.confidence_interval.high
        results['bootstrap_ci_contains_zero'] = (
            bootstrap_result.confidence_interval.low <= 0 <= bootstrap_result.confidence_interval.high
        )
    except Exception as e:
        print(f"⚠️ Bootstrap failed: {e}")
        results['bootstrap_ci_lower'] = np.nan
        results['bootstrap_ci_upper'] = np.nan
        results['bootstrap_ci_contains_zero'] = True

    return results


def interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size."""
    if cohens_d < 0.2:
        return 'negligible'
    elif cohens_d < 0.5:
        return 'small'
    elif cohens_d < 0.8:
        return 'medium'
    else:
        return 'large'


def compare_experiments(results: Dict[str, Dict],
                        metrics: List[str] = None,
                        alpha: float = 0.05) -> Dict[str, Dict]:
    """
    Compare multiple experiments statistically.

    Args:
        results: Dictionary of experiment results
        metrics: List of metrics to compare (default: common metrics)
        alpha: Significance level

    Returns:
        Dictionary with comparison results
    """
    if metrics is None:
        metrics = [
            'val_dice', 'val_loss', 'worst_bin_dice', 'p10_bin_dice',
            'bin_std', 'spatial_consistency', 'performance_coverage'
        ]

    comparison_results = {}

    # Get experiment names
    exp_names = list(results.keys())
    if len(exp_names) < 2:
        print("⚠️ Need at least 2 experiments for comparison")
        return comparison_results

    # Compare each metric
    for metric in metrics:
        print(f"📊 Analyzing metric: {metric}")

        # Extract metric data for all experiments
        metric_data = {}
        for exp_name in exp_names:
            values = []
            for round_data in results[exp_name]:
                if metric in round_data:
                    values.append(round_data[metric])
                elif metric in round_data.get('spatial_metrics', {}):
                    values.append(round_data['spatial_metrics'][metric])

            if values:
                metric_data[exp_name] = values
            else:
                print(f"⚠️ No data found for {metric} in {exp_name}")

        if len(metric_data) < 2:
            continue

        # Perform pairwise comparisons
        metric_comparisons = {}
        exp_names_with_data = list(metric_data.keys())

        for i in range(len(exp_names_with_data)):
            for j in range(i + 1, len(exp_names_with_data)):
                exp1, exp2 = exp_names_with_data[i], exp_names_with_data[j]

                comparison_key = f"{exp1}_vs_{exp2}"
                stat_results = calculate_statistical_significance(
                    metric_data[exp1], metric_data[exp2], alpha
                )

                # Add descriptive statistics
                stat_results['exp1_mean'] = np.mean(metric_data[exp1])
                stat_results['exp2_mean'] = np.mean(metric_data[exp2])
                stat_results['exp1_std'] = np.std(metric_data[exp1])
                stat_results['exp2_std'] = np.std(metric_data[exp2])
                stat_results['mean_difference'] = stat_results['exp1_mean'] - stat_results['exp2_mean']

                metric_comparisons[comparison_key] = stat_results

        comparison_results[metric] = metric_comparisons

    return comparison_results


def generate_statistical_report(comparison_results: Dict[str, Dict],
                                output_path: str = None) -> str:
    """
    Generate a comprehensive statistical report.

    Args:
        comparison_results: Results from compare_experiments
        output_path: Path to save the report

    Returns:
        Report as string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("STATISTICAL ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    for metric, comparisons in comparison_results.items():
        report_lines.append(f"📊 METRIC: {metric.upper()}")
        report_lines.append("-" * 50)

        for comparison_key, stats in comparisons.items():
            exp1, exp2 = comparison_key.split('_vs_')

            report_lines.append(f"\n🔍 Comparison: {exp1} vs {exp2}")
            report_lines.append(f"   Mean difference: {stats['mean_difference']:.4f}")
            report_lines.append(f"   {exp1} mean: {stats['exp1_mean']:.4f} ± {stats['exp1_std']:.4f}")
            report_lines.append(f"   {exp2} mean: {stats['exp2_mean']:.4f} ± {stats['exp2_std']:.4f}")

            # t-test results
            if not np.isnan(stats['t_pvalue']):
                report_lines.append(f"   t-test: t={stats['t_statistic']:.4f}, p={stats['t_pvalue']:.4f}")
                report_lines.append(f"   t-test significant: {'Yes' if stats['t_significant'] else 'No'}")

            # Mann-Whitney U test results
            if not np.isnan(stats['u_pvalue']):
                report_lines.append(f"   Mann-Whitney U: U={stats['u_statistic']:.4f}, p={stats['u_pvalue']:.4f}")
                report_lines.append(f"   Mann-Whitney U significant: {'Yes' if stats['u_significant'] else 'No'}")

            # Effect size
            if not np.isnan(stats['cohens_d']):
                report_lines.append(f"   Cohen's d: {stats['cohens_d']:.4f} ({stats['effect_size']} effect)")

            # Bootstrap CI
            if not np.isnan(stats['bootstrap_ci_lower']):
                report_lines.append(
                    f"   Bootstrap 95% CI: [{stats['bootstrap_ci_lower']:.4f}, {stats['bootstrap_ci_upper']:.4f}]")
                report_lines.append(f"   CI contains zero: {'Yes' if stats['bootstrap_ci_contains_zero'] else 'No'}")

            report_lines.append("")

    report_text = "\n".join(report_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"📄 Statistical report saved to: {output_path}")

    return report_text


def run_statistical_analysis(result_paths: List[str],
                             output_dir: str = None,
                             metrics: List[str] = None) -> Dict[str, Dict]:
    """
    Run complete statistical analysis on experiment results.

    Args:
        result_paths: List of paths to results.json files
        output_dir: Directory to save analysis results
        metrics: List of metrics to analyze

    Returns:
        Statistical analysis results
    """
    print("🔬 Starting statistical analysis...")

    # Load results
    results = load_experiment_results(result_paths)
    print(f"📊 Loaded {len(results)} experiments")

    # Compare experiments
    comparison_results = compare_experiments(results, metrics)

    # Generate report
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'statistical_analysis_report.txt')
        generate_statistical_report(comparison_results, report_path)

        # Save detailed results as JSON
        results_path = os.path.join(output_dir, 'statistical_analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        print(f"📄 Detailed results saved to: {results_path}")

    return comparison_results


if __name__ == "__main__":
    # Example usage
    result_paths = [
        "/path/to/random/results.json",
        "/path/to/adaptive_improved/results.json"
    ]

    results = run_statistical_analysis(result_paths, output_dir="./statistical_analysis")
    print("✅ Statistical analysis completed!")
