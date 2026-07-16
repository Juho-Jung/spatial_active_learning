# Spatial Active Learning for Lesion Segmentation

Active learning framework for lesion segmentation experiments.

## How to Run

From the repository root:

```bash
cd active_learning
python run_al_baselines.py --mode <strategy> [options]
```

## Main Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Selection strategy (required) | - |
| `--model_type` | Model architecture | `smp_efficientnet` |
| `--uncertainty_type` | Uncertainty method | `none` |
| `--target_lesion` | Target lesion | `calcifiednodule` |
| `--collection` | Data collection | `both` |
| `--round_num` | Number of AL rounds | `10` |
| `--num_samples` | Samples per round | `20` |
| `--train_epochs` | Number of epochs | `150` |
| `--batch_size` | Batch size | `16` |
| `--gpu_id` | GPU ID | `0` |

## Available Strategies (--mode)

### Basic
- `random` - Random sampling
- `uncertainty` - Uncertainty-based sampling
- `area_random` - Spatial region-based random
- `uncertainty_area` - Spatial region + Uncertainty

### Diversity
- `diversity` - Greedy farthest-point in feature space (diversity among samples selected in the current round)
- `diversity_uncertainty` - Diversity + Uncertainty
- `coreset` - Core-set (k-center) sampling: selects unlabeled samples farthest from the **already labeled set** (Sener & Savarese, ICLR 2018)

### Adaptive
- `adaptive` - Adaptive sampling
- `adaptive_improved` - Improved adaptive
- `adaptive_multi_scale` - Multi-scale adaptive
- `adaptive_performance_monitoring` - Performance-monitoring adaptive

### Literature-based
- `taudis` - TAUDIS
- `usimc` - USIMC
- `mcd_alunet` - MC Dropout variant
- `Learning Loss` - Learning Loss

## Example Commands

```bash
# Random baseline
python run_al_baselines.py --mode random --round_num 10 --num_samples 20

# Uncertainty-based
python run_al_baselines.py --mode uncertainty --uncertainty_type none

# Learning Loss
python run_al_baselines.py --mode Learning Loss --round_num 50 --num_samples 10 --gpu_id 1

# USIMC
python run_al_baselines.py --mode usimc --round_num 50 --num_samples 10 --batch_size 16

# Coreset
python run_al_baselines.py --mode coreset --uncertainty_type none

# Adaptive
python run_al_baselines.py --mode adaptive --uncertainty_type none --grid_width 5 --grid_height 5
```

## Project Structure

```
active_learning/
├── run_al_baselines.py      # Main entry script
├── base_utils/              # Utilities (setup, logging, plotting)
├── dataset/                 # Dataset classes
├── models/                  # Model definitions
├── sample_selection/        # Sample selection strategies
├── training/                # Training routines
├── uncertainty/             # Uncertainty computation
├── metrics/                 # Evaluation metrics
└── losses/                  # Loss functions
```

## More Examples
```bash
python run_al_baselines.py --mode Learning Loss --round_num 50 --num_samples 10 --gpu_id 1
python run_al_baselines.py --mode usimc --round_num 50 --num_samples 10 --batch_size 16
```