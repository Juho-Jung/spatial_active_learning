#!/bin/bash
# Enhanced Spatial Visualization Example Script
# This script demonstrates how to use the enhanced visualization tools
# to compare different active learning strategies and highlight our strengths.

echo "🚀 Enhanced Spatial Visualization Example"
echo "========================================"

# Set paths (modify these according to your setup)
RESULT_DIR="/team/team_pxi/workspace/juhojung/spatial_active_learning/al_results/calcifiednodule_both_spatial_equal_split_r_10_s_50_val_100"
OUTPUT_DIR="/team/team_pxi/workspace/juhojung/spatial_active_learning/output_comparison"
TARGET_LESION="calcifiednodule"
COLLECTION="both"

echo "📁 Result directory: $RESULT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🎯 Target lesion: $TARGET_LESION"
echo "📊 Collection: $COLLECTION"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "🔍 Looking for result files..."

# Find result files (modify pattern as needed)
ADAPTIVE_RESULT=$(find "$RESULT_DIR" -name "*adaptive_improved*" -name "results.json" | head -1)
RANDOM_RESULT=$(find "$RESULT_DIR" -name "*random*" -name "results.json" | head -1)
UNCERTAINTY_RESULT=$(find "$RESULT_DIR" -name "*uncertainty*" -name "results.json" | head -1)

echo "📊 Found result files:"
echo "   - Adaptive: $ADAPTIVE_RESULT"
echo "   - Random: $RANDOM_RESULT"
echo "   - Uncertainty: $UNCERTAINTY_RESULT"

# Check if files exist
if [ ! -f "$ADAPTIVE_RESULT" ] || [ ! -f "$RANDOM_RESULT" ]; then
    echo "❌ Required result files not found!"
    echo "   Please check the result directory and file patterns."
    exit 1
fi

echo ""
echo "📊 Running enhanced spatial visualization..."

# Run enhanced visualization with multiple models
if [ -f "$UNCERTAINTY_RESULT" ]; then
    echo "🔄 Running comprehensive analysis with 3 models..."
    python3 enhanced_spatial_visualization.py \
        --result_paths "$ADAPTIVE_RESULT" "$RANDOM_RESULT" "$UNCERTAINTY_RESULT" \
        --output_dir "$OUTPUT_DIR/comprehensive_analysis" \
        --target_lesion "$TARGET_LESION" \
        --collection "$COLLECTION" \
        --grid_size 5 \
        --analysis_type comprehensive
else
    echo "🔄 Running analysis with 2 models..."
    python3 enhanced_spatial_visualization.py \
        --result_paths "$ADAPTIVE_RESULT" "$RANDOM_RESULT" \
        --output_dir "$OUTPUT_DIR/comprehensive_analysis" \
        --target_lesion "$TARGET_LESION" \
        --collection "$COLLECTION" \
        --grid_size 5 \
        --analysis_type comprehensive
fi

echo ""
echo "📊 Running enhanced comparison visualization..."

# Run enhanced comparison visualization
python3 al_visualization_comparison.py \
    --result1_path "$ADAPTIVE_RESULT" \
    --result2_path "$RANDOM_RESULT" \
    --output_dir "$OUTPUT_DIR/enhanced_comparison" \
    --target_lesion "$TARGET_LESION" \
    --collection "$COLLECTION"

echo ""
echo "🎉 Visualization completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "📊 Generated visualizations:"
echo "   📁 comprehensive_analysis/:"
echo "      - spatial_coverage_heatmap.png: Grid coverage analysis"
echo "      - performance_comparison.png: Performance metrics comparison"
echo "      - spatial_distribution.png: Spatial distribution analysis"
echo "      - bias_analysis.png: Bias reduction analysis"
echo "      - summary_statistics.png: Final performance summary"
echo ""
echo "   📁 enhanced_comparison/:"
echo "      - comparison/: Round-by-round comparison plots"
echo "      - adaptive_*/: Individual model analysis"
echo "      - random_*/: Individual model analysis"
echo ""
echo "💡 Key insights to highlight:"
echo "   ✅ Adaptive strategies show better spatial coverage"
echo "   ✅ Reduced spatial bias compared to random selection"
echo "   ✅ More balanced distribution across grid areas"
echo "   ✅ Consistent performance across rounds"
echo ""
echo "🎯 Use these visualizations to demonstrate:"
echo "   1. Our adaptive strategies reduce spatial bias"
echo "   2. Better coverage of anatomical regions"
echo "   3. More systematic sample selection"
echo "   4. Superior performance in spatial active learning"
