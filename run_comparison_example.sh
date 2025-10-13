#!/bin/bash
# Active Learning Comparison Example Script
# This script demonstrates how to use the comparison tools

echo "🚀 Active Learning Comparison Example"
echo "====================================="

# Set paths (modify these paths according to your actual results)
RESULT1_PATH="/team/team_pxi/workspace/juhojung/spatial_active_learning/al_results/pneumoperitoneum_sdc_ppm_train-0908_spatial_split_r_10_s_20_val_500/adaptive_base_20250919_172458/results.json"
RESULT2_PATH="/team/team_pxi/workspace/juhojung/spatial_active_learning/al_results/pneumoperitoneum_sdc_ppm_train-0908_spatial_split_r_10_s_20_val_500/random_base_20250918_205515/results.json"
RESULT3_PATH="/team/team_pxi/workspace/juhojung/spatial_active_learning/al_results/pneumoperitoneum_sdc_ppm_train-0908_spatial_split_r_10_s_20_val_500/uncertainty_base_20250919_174857/results.json"
RESULT4_PATH="/team/team_pxi/workspace/juhojung/spatial_active_learning/al_results/pneumoperitoneum_sdc_ppm_train-0908_spatial_split_r_10_s_20_val_500/diversity_base_20250919_174949/results.json"


# Optional: Add more results for multi-model comparison (up to 8)
# RESULT3_PATH="/path/to/result3.json"
# RESULT4_PATH="/path/to/result4.json"

# Check if result files exist
if [ ! -f "$RESULT1_PATH" ]; then
    echo "❌ Result file 1 not found: $RESULT1_PATH"
    echo "Please update the paths in this script to point to your actual result files."
    exit 1
fi

if [ ! -f "$RESULT2_PATH" ]; then
    echo "❌ Result file 2 not found: $RESULT2_PATH"
    echo "Please update the paths in this script to point to your actual result files."
    exit 1
fi

# Check additional result files if they exist
RESULT_PATHS="$RESULT1_PATH $RESULT2_PATH"
if [ -f "$RESULT3_PATH" ]; then
    echo "✅ Result file 3 found: $RESULT3_PATH"
    RESULT_PATHS="$RESULT_PATHS $RESULT3_PATH"
else
    echo "⚠️  Result file 3 not found: $RESULT3_PATH (skipping)"
fi

if [ -f "$RESULT4_PATH" ]; then
    echo "✅ Result file 4 found: $RESULT4_PATH"
    RESULT_PATHS="$RESULT_PATHS $RESULT4_PATH"
else
    echo "⚠️  Result file 4 not found: $RESULT4_PATH (skipping)"
fi

echo "✅ Result files found:"
echo "   Result 1: $RESULT1_PATH"
echo "   Result 2: $RESULT2_PATH"
if [ -f "$RESULT3_PATH" ]; then
    echo "   Result 3: $RESULT3_PATH"
fi
if [ -f "$RESULT4_PATH" ]; then
    echo "   Result 4: $RESULT4_PATH"
fi

# 1. Performance Comparison
echo ""
echo "📊 Running Performance Comparison..."
python al_perf_comparison.py \
    --result_paths $RESULT_PATHS \
    --output_dir /team/team_pxi/workspace/juhojung/spatial_active_learning/output_comparison/performance \
    --target_lesion pneumoperitoneum \
    --max_models 8

if [ $? -eq 0 ]; then
    echo "✅ Performance comparison completed successfully!"
    echo "📁 Results saved to: /team/team_pxi/workspace/juhojung/spatial_active_learning/output_comparison/performance/"
else
    echo "❌ Performance comparison failed!"
    exit 1
fi

# 2. Visualization Comparison
echo ""
echo "🎨 Running Visualization Comparison..."
python al_visualization_comparison.py \
    --result1_path "$RESULT1_PATH" \
    --result2_path "$RESULT2_PATH" \
    --output_dir /team/team_pxi/workspace/juhojung/spatial_active_learning/output_comparison/visualization \
    --target_lesion pneumoperitoneum \
    --collection sdc_ppm_train-0908

if [ $? -eq 0 ]; then
    echo "✅ Visualization comparison completed successfully!"
    echo "📁 Results saved to: /team/team_pxi/workspace/juhojung/spatial_active_learning/output_comparison/visualization/"
else
    echo "❌ Visualization comparison failed!"
    exit 1
fi

echo ""
echo "🎉 All comparisons completed successfully!"
echo "📁 Check the following directories for results:"
echo "   - /team/team_pxi/workspace/juhojung/spatial_active_learning/output_comparison/performance/ (performance plots including normalized metrics)"
echo "   - /team/team_pxi/workspace/juhojung/spatial_active_learning/output_comparison/visualization/ (spatial distribution plots with organized subdirectories)"
echo ""
echo "📊 New features:"
echo "   - Normalized performance coverage and spatial consistency plots"
echo "   - Support for up to 8 models comparison"
echo "   - Organized visualization with comparison/, adaptive/, random/ subdirectories"
echo "   - Cumulative selection visualization showing all rounds"
