#!/bin/bash
"""
Run performance comparison for spatial_equal_split results
"""

echo "🚀 Running Performance Comparison for Spatial Equal Split"
echo "========================================================"

python al_perf_comparison.py \
  --result_paths \
    /opt/spatial_active_learning/result/calcifiednodule_both_spatial_equal_split_r_10_s_50_val_100/adaptive_improved_none_5x5_20251020_204137/results.json \
    /opt/spatial_active_learning/result/calcifiednodule_both_spatial_equal_split_r_10_s_50_val_100/adaptive_multi_scale_none_5x5_20251023_094745/results.json \
    /opt/spatial_active_learning/result/calcifiednodule_both_spatial_equal_split_r_10_s_50_val_100/adaptive_performance_monitoring_none_5x5_20251023_095006/results.json \
    /opt/spatial_active_learning/result/calcifiednodule_both_spatial_equal_split_r_10_s_50_val_100/uncertainty_none_5x5_20251022_084126/results.json \
    /opt/spatial_active_learning/result/calcifiednodule_both_spatial_equal_split_r_10_s_50_val_100/random_none_5x5_20251020_145954/results.json \
  --target_lesion calcifiednodule \
  --split_type spatial_equal_split

echo "✅ Equal Split comparison completed!"
