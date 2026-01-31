#!/usr/bin/env python3
"""
Active Learning Table Comparison Tool

Generate comprehensive tables comparing different methods across rounds.
Shows detailed metrics: Worst-bin Dice, P10-bin Dice, Bin-std, Spatial Consistency, Performance Coverage.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append('/opt/pxi/projects/calcified_nodule/spatial_active_learning')
sys.path.append('/opt/pxi/projects/calcified_nodule/spatial_active_learning/sam_adaptation')
sys.path.append('/opt/pxi')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Active Learning Table Comparison')
    parser.add_argument('--result_paths', nargs='+', required=True,
                        help='List of result.json file paths to compare')
    parser.add_argument('--output_dir', default='/opt/spatial_active_learning/analysis_results/table_comparison',
                        help='Output directory for comparison tables')
    parser.add_argument('--target_lesion', default='calcifiednodule',
                        help='Target lesion type for table titles')
    parser.add_argument('--format', choices=['csv', 'excel', 'html'], default='excel',
                        help='Output format for tables')
    parser.add_argument('--rounds', nargs='+', type=int, default=None,
                        help='Specific rounds to include (default: all rounds)')
    return parser.parse_args()


def extract_model_name(result_path):
    """Extract model name from result path."""
    path_parts = Path(result_path).parts
    for part in path_parts:
        if '_' in part and any(x in part for x in ['adaptive', 'random', 'uncertainty', 'area_random', 'diversity']):
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
                    full_name = '_'.join(parts[:-1])

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


def load_results(result_paths):
    """Load results from multiple JSON files."""
    results = {}

    for result_path in result_paths:
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


def create_round_by_round_table(results, rounds=None):
    """Create round-by-round comparison table."""
    all_rounds = set()
    for method_data in results.values():
        for round_data in method_data:
            all_rounds.add(round_data['round'])

    if rounds is None:
        rounds = sorted(all_rounds)
    else:
        rounds = [r for r in rounds if r in all_rounds]

    # Create comprehensive table
    table_data = []

    for round_num in rounds:
        round_row = {'Round': round_num}

        for method_name, method_data in results.items():
            # Find data for this round
            round_data = None
            for data in method_data:
                if data['round'] == round_num:
                    round_data = data
                    break

            if round_data:
                spatial_metrics = round_data['spatial_metrics']
                round_row[f'{method_name}_worst_bin_dice'] = spatial_metrics['worst_bin_dice']
                round_row[f'{method_name}_p10_bin_dice'] = spatial_metrics['p10_bin_dice']
                round_row[f'{method_name}_bin_std'] = spatial_metrics['bin_std']
                round_row[f'{method_name}_spatial_consistency'] = spatial_metrics['spatial_consistency']
                round_row[f'{method_name}_performance_coverage'] = spatial_metrics['performance_coverage']
                round_row[f'{method_name}_val_dice'] = round_data['val_dice']
                round_row[f'{method_name}_val_loss'] = round_data['val_loss']
            else:
                # Fill with NaN if no data for this round
                for metric in ['worst_bin_dice', 'p10_bin_dice', 'bin_std', 'spatial_consistency', 'performance_coverage', 'val_dice', 'val_loss']:
                    round_row[f'{method_name}_{metric}'] = np.nan

        table_data.append(round_row)

    return pd.DataFrame(table_data)


def create_final_performance_table(results):
    """Create final performance comparison table."""
    table_data = []

    for method_name, method_data in results.items():
        if not method_data:
            continue

        # Get final round data
        final_data = method_data[-1]
        spatial_metrics = final_data['spatial_metrics']

        row = {
            'Method': method_name,
            'Final_Round': final_data['round'],
            'Val_Dice': final_data['val_dice'],
            'Val_Loss': final_data['val_loss'],
            'Worst_Bin_Dice': spatial_metrics['worst_bin_dice'],
            'P10_Bin_Dice': spatial_metrics['p10_bin_dice'],
            'Bin_Std': spatial_metrics['bin_std'],
            'Spatial_Consistency': spatial_metrics['spatial_consistency'],
            'Performance_Coverage': spatial_metrics['performance_coverage'],
            'Avg_Bin_Dice': spatial_metrics['avg_bin_dice'],
            'Min_Bin_Dice': spatial_metrics['min_bin_dice'],
            'Max_Bin_Dice': spatial_metrics['max_bin_dice']
        }

        table_data.append(row)

    return pd.DataFrame(table_data)


def create_improvement_table(results):
    """Create improvement comparison table."""
    # Define baseline methods
    baseline_methods = ['uncertainty', 'random']
    adaptive_methods = [name for name in results.keys() if 'adaptive' in name]

    table_data = []

    for method_name in adaptive_methods:
        if method_name not in results:
            continue

        method_data = results[method_name]
        if not method_data:
            continue

        final_data = method_data[-1]
        spatial_metrics = final_data['spatial_metrics']

        row = {
            'Method': method_name,
            'Final_Val_Dice': final_data['val_dice'],
            'Final_Coverage': spatial_metrics['performance_coverage'],
            'Final_Consistency': spatial_metrics['spatial_consistency']
        }

        # Calculate improvements over baselines
        for baseline in baseline_methods:
            if baseline in results and results[baseline]:
                baseline_final = results[baseline][-1]
                baseline_spatial = baseline_final['spatial_metrics']

                # Calculate improvements
                dice_improvement = ((final_data['val_dice'] - baseline_final['val_dice']) /
                                    baseline_final['val_dice'] * 100) if baseline_final['val_dice'] > 0 else 0
                coverage_improvement = ((spatial_metrics['performance_coverage'] - baseline_spatial['performance_coverage']) /
                                        baseline_spatial['performance_coverage'] * 100) if baseline_spatial['performance_coverage'] > 0 else 0
                consistency_improvement = ((spatial_metrics['spatial_consistency'] - baseline_spatial['spatial_consistency']) /
                                           baseline_spatial['spatial_consistency'] * 100) if baseline_spatial['spatial_consistency'] > 0 else 0

                row[f'vs_{baseline}_dice_improvement'] = dice_improvement
                row[f'vs_{baseline}_coverage_improvement'] = coverage_improvement
                row[f'vs_{baseline}_consistency_improvement'] = consistency_improvement

        table_data.append(row)

    return pd.DataFrame(table_data)


def create_summary_statistics_table(results):
    """Create summary statistics table."""
    table_data = []

    for method_name, method_data in results.items():
        if not method_data:
            continue

        # Calculate statistics across all rounds
        val_dice_values = [round_data['val_dice'] for round_data in method_data]
        coverage_values = [round_data['spatial_metrics']['performance_coverage'] for round_data in method_data]
        consistency_values = [round_data['spatial_metrics']['spatial_consistency'] for round_data in method_data]

        row = {
            'Method': method_name,
            'Total_Rounds': len(method_data),
            'Final_Val_Dice': val_dice_values[-1],
            'Avg_Val_Dice': np.mean(val_dice_values),
            'Max_Val_Dice': np.max(val_dice_values),
            'Min_Val_Dice': np.min(val_dice_values),
            'Final_Coverage': coverage_values[-1],
            'Avg_Coverage': np.mean(coverage_values),
            'Max_Coverage': np.max(coverage_values),
            'Min_Coverage': np.min(coverage_values),
            'Final_Consistency': consistency_values[-1],
            'Avg_Consistency': np.mean(consistency_values),
            'Max_Consistency': np.max(consistency_values),
            'Min_Consistency': np.min(consistency_values)
        }

        # Calculate improvement rates
        if len(val_dice_values) > 1:
            row['Dice_Improvement_Rate'] = (val_dice_values[-1] - val_dice_values[0]) / val_dice_values[0] * 100
        if len(coverage_values) > 1:
            row['Coverage_Improvement_Rate'] = (coverage_values[-1] - coverage_values[0]) / coverage_values[0] * 100
        if len(consistency_values) > 1:
            row['Consistency_Improvement_Rate'] = (
                consistency_values[-1] - consistency_values[0]) / consistency_values[0] * 100

        table_data.append(row)

    return pd.DataFrame(table_data)


def save_tables(tables, output_dir, format_type, target_lesion):
    """Save tables in specified format."""
    os.makedirs(output_dir, exist_ok=True)

    if format_type == 'excel':
        # Save all tables in one Excel file with multiple sheets
        excel_path = os.path.join(output_dir, f'{target_lesion}_comparison_tables.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for table_name, table_df in tables.items():
                table_df.to_excel(writer, sheet_name=table_name, index=False)
        print(f"📊 Excel file saved: {excel_path}")

    elif format_type == 'csv':
        # Save each table as separate CSV file
        for table_name, table_df in tables.items():
            csv_path = os.path.join(output_dir, f'{target_lesion}_{table_name}.csv')
            table_df.to_csv(csv_path, index=False)
            print(f"📊 CSV file saved: {csv_path}")

    elif format_type == 'html':
        # Save as HTML file with all tables
        html_path = os.path.join(output_dir, f'{target_lesion}_comparison_tables.html')
        with open(html_path, 'w') as f:
            f.write(f"<html><head><title>{target_lesion} Active Learning Comparison</title></head><body>")
            f.write(f"<h1>{target_lesion} Active Learning Results Comparison</h1>")

            for table_name, table_df in tables.items():
                f.write(f"<h2>{table_name.replace('_', ' ').title()}</h2>")
                f.write(table_df.to_html(index=False, classes='table table-striped'))
                f.write("<br><br>")

            f.write("</body></html>")
        print(f"📊 HTML file saved: {html_path}")


def main():
    """Main function for table comparison."""
    args = parse_arguments()

    print("🚀 Starting Active Learning Table Comparison")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Target lesion: {args.target_lesion}")
    print(f"📊 Format: {args.format}")

    # Load results
    results = load_results(args.result_paths)

    if not results:
        print("❌ No valid results found!")
        return

    print(f"\n✅ Successfully loaded results for {len(results)} methods:")
    for method_name in results.keys():
        print(f"   - {method_name}")

    # Create tables
    print("\n📊 Generating comparison tables...")

    tables = {}

    # Round-by-round table
    print("   - Creating round-by-round table...")
    tables['round_by_round'] = create_round_by_round_table(results, args.rounds)

    # Final performance table
    print("   - Creating final performance table...")
    tables['final_performance'] = create_final_performance_table(results)

    # Improvement table
    print("   - Creating improvement table...")
    tables['improvement_comparison'] = create_improvement_table(results)

    # Summary statistics table
    print("   - Creating summary statistics table...")
    tables['summary_statistics'] = create_summary_statistics_table(results)

    # Save tables
    print(f"\n💾 Saving tables in {args.format} format...")
    save_tables(tables, args.output_dir, args.format, args.target_lesion)

    print(f"\n🎉 Table comparison completed!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
