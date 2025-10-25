#!/bin/bash
"""
Run performance comparison for spatial_dynamic_split results
"""

echo "🚀 Running Performance Comparison for Spatial Dynamic Split"
echo "=========================================================="

python al_perf_comparison.py \
  --result_paths \
    /opt/spatial_active_learning/result/calcifiednodule_both_spatial_dynamic_split_r_10_s_50_val_100/adaptive_improved_none_5x5_20251022_084803/results.json \
    /opt/spatial_active_learning/result/calcifiednodule_both_spatial_dynamic_split_r_10_s_50_val_100/adaptive_multi_scale_none_5x5_20251023_094832/results.json \
    /opt/spatial_active_learning/result/calcifiednodule_both_spatial_dynamic_split_r_10_s_50_val_100/adaptive_performance_monitoring_none_5x5_20251023_094928/results.json \
    /opt/spatial_active_learning/result/calcifiednodule_both_spatial_dynamic_split_r_10_s_50_val_100/uncertainty_none_5x5_20251023_095130/results.json \
    /opt/spatial_active_learning/result/calcifiednodule_both_spatial_dynamic_split_r_10_s_50_val_100/random_none_5x5_20251020_150113/results.json \
  --target_lesion calcifiednodule \
  --split_type spatial_dynamic_split

echo "✅ Dynamic Split comparison completed!"
