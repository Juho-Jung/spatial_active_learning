#!/usr/bin/env python3
"""
Reproducibility Verification Script

Verify that selected_indices are reproducible across different runs.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append('/opt/pxi/projects/calcified_nodule/spatial_active_learning')
sys.path.append('/opt/pxi/projects/calcified_nodule/spatial_active_learning/sam_adaptation')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Verify Reproducibility of Selected Indices')
    parser.add_argument('--result_paths', nargs='+', required=True,
                       help='List of result.json file paths to compare')
    parser.add_argument('--output_file', default='reproducibility_report.txt',
                       help='Output file for reproducibility report')
    return parser.parse_args()


def load_results(result_path):
    """Load results from JSON file."""
    if not os.path.exists(result_path):
        print(f"❌ Result file not found: {result_path}")
        return None
        
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading {result_path}: {e}")
        return None


def extract_selected_indices(results):
    """Extract selected indices from results."""
    selected_indices_by_round = {}
    
    for round_data in results:
        round_num = round_data['round']
        selected_indices = round_data['selected_indices']
        selected_indices_by_round[round_num] = selected_indices
    
    return selected_indices_by_round


def compare_selected_indices(indices1, indices2, model1_name, model2_name):
    """Compare selected indices between two models."""
    comparison_results = {}
    
    # Get common rounds
    common_rounds = set(indices1.keys()) & set(indices2.keys())
    
    for round_num in sorted(common_rounds):
        round1_indices = set(indices1[round_num])
        round2_indices = set(indices2[round_num])
        
        # Calculate overlap
        intersection = round1_indices & round2_indices
        union = round1_indices | round2_indices
        
        if len(union) > 0:
            overlap_ratio = len(intersection) / len(union)
        else:
            overlap_ratio = 1.0
        
        comparison_results[round_num] = {
            'round1_indices': sorted(list(round1_indices)),
            'round2_indices': sorted(list(round2_indices)),
            'intersection': sorted(list(intersection)),
            'union': sorted(list(union)),
            'overlap_ratio': overlap_ratio,
            'round1_only': sorted(list(round1_indices - round2_indices)),
            'round2_only': sorted(list(round2_indices - round1_indices))
        }
    
    return comparison_results


def analyze_reproducibility(comparison_results, model1_name, model2_name):
    """Analyze reproducibility patterns."""
    print(f"\n🔍 Reproducibility Analysis: {model1_name} vs {model2_name}")
    print("="*60)
    
    total_overlap = 0
    total_rounds = 0
    
    for round_num in sorted(comparison_results.keys()):
        result = comparison_results[round_num]
        overlap_ratio = result['overlap_ratio']
        total_overlap += overlap_ratio
        total_rounds += 1
        
        print(f"\n📊 Round {round_num}:")
        print(f"   Overlap ratio: {overlap_ratio:.3f}")
        print(f"   {model1_name} only: {len(result['round1_only'])} samples")
        print(f"   {model2_name} only: {len(result['round2_only'])} samples")
        print(f"   Common: {len(result['intersection'])} samples")
        
        if round_num == 1:
            print(f"   🎯 First round - should be identical for fair comparison")
            if overlap_ratio == 1.0:
                print(f"   ✅ Perfect overlap (reproducible)")
            else:
                print(f"   ⚠️  Not identical - may indicate reproducibility issues")
        else:
            if overlap_ratio > 0.8:
                print(f"   ✅ High overlap (likely reproducible)")
            elif overlap_ratio > 0.5:
                print(f"   ⚠️  Moderate overlap (some reproducibility)")
            else:
                print(f"   ❌ Low overlap (likely not reproducible)")
    
    if total_rounds > 0:
        avg_overlap = total_overlap / total_rounds
        print(f"\n📈 Overall Analysis:")
        print(f"   Average overlap ratio: {avg_overlap:.3f}")
        print(f"   Total rounds compared: {total_rounds}")
        
        if avg_overlap > 0.9:
            print(f"   ✅ High reproducibility")
        elif avg_overlap > 0.7:
            print(f"   ⚠️  Moderate reproducibility")
        else:
            print(f"   ❌ Low reproducibility")


def generate_report(comparison_results, model1_name, model2_name, output_file):
    """Generate detailed reproducibility report."""
    with open(output_file, 'w') as f:
        f.write(f"Reproducibility Report: {model1_name} vs {model2_name}\n")
        f.write("="*60 + "\n\n")
        
        for round_num in sorted(comparison_results.keys()):
            result = comparison_results[round_num]
            f.write(f"Round {round_num}:\n")
            f.write(f"  Overlap ratio: {result['overlap_ratio']:.3f}\n")
            f.write(f"  {model1_name} indices: {result['round1_indices']}\n")
            f.write(f"  {model2_name} indices: {result['round2_indices']}\n")
            f.write(f"  Common indices: {result['intersection']}\n")
            f.write(f"  {model1_name} only: {result['round1_only']}\n")
            f.write(f"  {model2_name} only: {result['round2_only']}\n\n")
    
    print(f"📄 Detailed report saved to: {output_file}")


def main():
    """Main function for reproducibility verification."""
    args = parse_arguments()
    
    print("🔍 Starting Reproducibility Verification")
    print(f"📊 Comparing {len(args.result_paths)} result files")
    
    # Load all results
    all_results = {}
    for result_path in args.result_paths:
        model_name = Path(result_path).parent.name  # Extract model name from path
        results = load_results(result_path)
        if results is not None:
            all_results[model_name] = results
            print(f"✅ Loaded {len(results)} rounds for {model_name}")
        else:
            print(f"❌ Failed to load {result_path}")
    
    if len(all_results) < 2:
        print("❌ Need at least 2 result files to compare")
        return
    
    # Extract selected indices for each model
    indices_by_model = {}
    for model_name, results in all_results.items():
        indices_by_model[model_name] = extract_selected_indices(results)
    
    # Compare all pairs
    model_names = list(indices_by_model.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1_name = model_names[i]
            model2_name = model_names[j]
            
            print(f"\n🔄 Comparing {model1_name} vs {model2_name}")
            
            comparison_results = compare_selected_indices(
                indices_by_model[model1_name],
                indices_by_model[model2_name],
                model1_name,
                model2_name
            )
            
            analyze_reproducibility(comparison_results, model1_name, model2_name)
            
            # Generate detailed report
            report_file = f"reproducibility_{model1_name}_vs_{model2_name}.txt"
            generate_report(comparison_results, model1_name, model2_name, report_file)
    
    print(f"\n🎉 Reproducibility verification completed!")
    print(f"📁 Check individual report files for detailed analysis")


if __name__ == "__main__":
    main()
