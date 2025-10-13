#!/usr/bin/env python3
"""
Cumulative Selection Dynamics Simulation for Active Learning

Simulates cumulative selection dynamics for different active learning strategies:
- Random selection
- Uncertainty-based selection  
- Adaptive selection (ours)

Generates 3x4 subplot visualization showing cumulative selection patterns across rounds.
"""

import os
import warnings

import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(2026)

# Simulation parameters
N_POOL = 800  # Total pool size
BATCH_SIZE = 20  # Samples per round
ROUNDS = [1, 10, 20, 40]  # Rounds to visualize
GRID_SIZE = 3  # 3x3 spatial grid
RESOLUTION = 512  # Grid resolution
GAMMA = 0.02  # Ultra low gamma for maximum focused coverage
LAMBDA_MAX = 5.0  # Ultra high lambda for maximum spatial bias
LAMBDA_FLOOR = 2.0  # Ultra high floor for maximum spatial bias
TAU = 0.005  # Ultra low temperature for maximum focused sampling
SIGMA_DECAY = 0.30  # Ultra large decay radius for maximum focused selection

# Hotspot parameters (3 Gaussian centers in [0,1] space)
HOTSPOTS = [
    {'mu': (0.25, 0.75), 'sigma': 0.06},
    {'mu': (0.75, 0.30), 'sigma': 0.06},
    {'mu': (0.70, 0.80), 'sigma': 0.06}
]

# Policy-specific uncertainty decay rates
ALPHA_RATES = {
    'random': 0.005,  # Extremely low decay for very scattered selection
    'uncertainty': 0.01,  # Very low decay for scattered selection
    'adaptive': 0.8  # Ultra high decay for extremely focused selection
}


def generate_pool_data(n_samples=N_POOL):
    """Generate pool data with 3 hotspots + uniform noise, then scale to 512x512."""
    print("🔄 Generating pool data...")

    # Generate samples from 3 Gaussian hotspots
    hotspot_samples = []
    for hotspot in HOTSPOTS:
        n_hotspot = int(n_samples * 0.8 / 3)  # 80% from hotspots, divided by 3
        samples = np.random.multivariate_normal(
            hotspot['mu'],
            np.eye(2) * hotspot['sigma']**2,
            n_hotspot
        )
        hotspot_samples.append(samples)

    # Add uniform noise (20% of total)
    n_noise = int(n_samples * 0.2)
    noise_samples = np.random.uniform(0, 1, (n_noise, 2))

    # Combine all samples
    all_samples = np.vstack(hotspot_samples + [noise_samples])

    # Clip to [0,1] range
    all_samples = np.clip(all_samples, 0, 1)

    # Scale to 512x512 grid
    all_samples = (all_samples * (RESOLUTION - 1)).astype(int)

    print(f"✅ Generated {len(all_samples)} pool samples")
    return all_samples


def make_field_U0_M(resolution=RESOLUTION):
    """Generate initial uncertainty U0 and lesionness M fields on 512x512 grid."""
    print("🔄 Generating uncertainty and lesionness fields...")

    # Create coordinate grids
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X, Y], axis=-1)

    # Generate U0 field (uncertainty)
    U0 = np.zeros((resolution, resolution))
    for hotspot in HOTSPOTS:
        mu = hotspot['mu']
        sigma = hotspot['sigma']
        # Distance from each grid point to hotspot center
        distances = np.sqrt(np.sum((coords - mu)**2, axis=-1))
        contribution = np.exp(-distances**2 / (2 * sigma**2))
        U0 += contribution

    # Add small Gaussian noise
    noise = np.random.normal(0, 0.05, (resolution, resolution))
    U0 += noise

    # Normalize to [0, 1]
    U0 = (U0 - U0.min()) / (U0.max() - U0.min())

    # Generate M field (lesionness) - slightly different from U0
    M = np.zeros((resolution, resolution))
    # Slightly shifted and scaled hotspots for lesionness
    lesionness_hotspots = [
        {'mu': (0.30, 0.70), 'sigma': 0.07},  # Slightly shifted
        {'mu': (0.70, 0.35), 'sigma': 0.05},  # Different scale
        {'mu': (0.65, 0.75), 'sigma': 0.08}   # Different position
    ]

    for hotspot in lesionness_hotspots:
        mu = hotspot['mu']
        sigma = hotspot['sigma']
        distances = np.sqrt(np.sum((coords - mu)**2, axis=-1))
        contribution = np.exp(-distances**2 / (2 * sigma**2))
        M += contribution

    # Add different noise pattern
    noise = np.random.normal(0, 0.03, (resolution, resolution))
    M += noise

    # Normalize to [0, 1]
    M = (M - M.min()) / (M.max() - M.min())

    print("✅ Generated uncertainty and lesionness fields")
    return U0, M


def bilinear_sample(grid, points):
    """Bilinear interpolation to sample grid values at given points."""
    h, w = grid.shape
    x, y = points[:, 0], points[:, 1]

    # Clamp coordinates to valid range
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    # Get integer coordinates
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    # Get fractional parts
    fx = x - x0
    fy = y - y0

    # Bilinear interpolation
    c00 = grid[y0, x0]
    c01 = grid[y0, x1]
    c10 = grid[y1, x0]
    c11 = grid[y1, x1]

    c0 = c00 * (1 - fx) + c01 * fx
    c1 = c10 * (1 - fx) + c11 * fx
    result = c0 * (1 - fy) + c1 * fy

    return result


def select_batch_random(pool_indices, n_select, selected_indices):
    """Random selection strategy - completely uniform across entire space."""
    available_indices = [i for i in pool_indices if i not in selected_indices]
    if len(available_indices) < n_select:
        return available_indices

    # For truly scattered random selection, add spatial diversity constraint
    selected = []
    remaining_indices = available_indices.copy()

    for _ in range(n_select):
        if not remaining_indices:
            break

        # Random selection with spatial diversity
        if len(selected) > 0:
            # Get coordinates of already selected points
            selected_coords = pool_data[selected]

            # Calculate distances from each remaining point to selected points
            remaining_coords = pool_data[remaining_indices]
            min_distances = []

            for coord in remaining_coords:
                distances = np.sqrt(np.sum((selected_coords - coord)**2, axis=1))
                min_distances.append(np.min(distances))

            # Prefer points that are far from already selected points
            min_distances = np.array(min_distances)
            if np.max(min_distances) > 0:
                # Weight by distance (prefer farther points)
                weights = min_distances / np.max(min_distances)
                weights = weights ** 2  # Square to emphasize distance
                weights = weights / np.sum(weights)

                idx = np.random.choice(len(remaining_indices), p=weights)
            else:
                idx = np.random.choice(len(remaining_indices))
        else:
            # First selection - completely random
            idx = np.random.choice(len(remaining_indices))

        selected.append(remaining_indices.pop(idx))

    return selected


def select_batch_uncertainty(pool_indices, uncertainty_field, n_select, selected_indices):
    """Uncertainty-based selection using probabilistic sampling."""
    available_indices = [i for i in pool_indices if i not in selected_indices]
    if len(available_indices) < n_select:
        return available_indices

    # Get coordinates for available points
    available_coords = pool_data[available_indices]

    # Sample uncertainty values using bilinear interpolation
    uncertainty_values = bilinear_sample(uncertainty_field, available_coords)

    # Create probability distribution (proportional to uncertainty with heavy noise for scattering)
    probs = uncertainty_values.copy()
    # Add heavy noise to make selection very scattered
    noise = np.random.normal(0, 0.8, len(probs))
    probs = probs + noise
    probs = np.clip(probs, 1e-8, 1.0)  # Avoid zero probabilities
    probs = probs / probs.sum()  # Normalize

    # Sample using probability distribution
    selected_idx = np.random.choice(len(available_indices), n_select, replace=False, p=probs)
    return [available_indices[i] for i in selected_idx]


def select_batch_adaptive(pool_indices, uncertainty_field, lesionness_field, n_select, selected_indices):
    """Adaptive selection strategy with spatial coverage."""
    available_indices = [i for i in pool_indices if i not in selected_indices]
    if len(available_indices) < n_select:
        return available_indices

    # Get coordinates for available points
    available_coords = pool_data[available_indices]

    # Sample field values
    uncertainty_values = bilinear_sample(uncertainty_field, available_coords)
    lesionness_values = bilinear_sample(lesionness_field, available_coords)

    # Calculate weights w(u) = U(u) * M(u) with maximum emphasis on high values
    weights = (uncertainty_values * lesionness_values) ** 4.0  # Maximum emphasis on high values

    # Add hotspot concentration: find the most promising hotspot and focus on it
    if len(selected_indices) < 100:  # More rounds: focus on one hotspot
        # Find the hotspot with highest average uncertainty*lesionness
        hotspot_scores = []
        for hotspot in HOTSPOTS:
            # Convert hotspot center to grid coordinates
            center_x = int(hotspot['mu'][0] * RESOLUTION)
            center_y = int(hotspot['mu'][1] * RESOLUTION)

            # Sample in a small region around the hotspot
            region_size = 50
            x_min = max(0, center_x - region_size)
            x_max = min(RESOLUTION, center_x + region_size)
            y_min = max(0, center_y - region_size)
            y_max = min(RESOLUTION, center_y + region_size)

            region_uncertainty = uncertainty_field[y_min:y_max, x_min:x_max]
            region_lesionness = lesionness_field[y_min:y_max, x_min:x_max]
            hotspot_score = np.mean(region_uncertainty * region_lesionness)
            hotspot_scores.append(hotspot_score)

        # Focus on the best hotspot
        best_hotspot_idx = np.argmax(hotspot_scores)
        best_hotspot = HOTSPOTS[best_hotspot_idx]

        # Add distance-based bonus to points near the best hotspot
        for i, coord in enumerate(available_coords):
            dist_to_hotspot = np.sqrt((coord[0] - best_hotspot['mu'][0] * RESOLUTION)**2 +
                                      (coord[1] - best_hotspot['mu'][1] * RESOLUTION)**2)
            # Bonus for being close to the best hotspot
            distance_bonus = np.exp(-dist_to_hotspot / (2 * (15)**2))  # 15 pixel radius for maximum focus
            weights[i] *= (1 + 20.0 * distance_bonus)  # Maximum bonus for hotspot proximity

    # Create 3x3 spatial bins
    bin_assignments = np.zeros(len(available_coords), dtype=int)
    for i, coord in enumerate(available_coords):
        x, y = coord
        bin_x = min(int(x * GRID_SIZE / RESOLUTION), GRID_SIZE - 1)
        bin_y = min(int(y * GRID_SIZE / RESOLUTION), GRID_SIZE - 1)
        bin_id = bin_y * GRID_SIZE + bin_x
        bin_assignments[i] = bin_id

    # Calculate current coverage for each bin
    bin_coverage = {}
    for bin_id in range(GRID_SIZE * GRID_SIZE):
        bin_coverage[bin_id] = 0

        if len(selected_indices) > 0:
            # Get coordinates for selected points
            selected_coords = pool_data[selected_indices]

            # Calculate bin assignments for selected points
            selected_bin_assignments = np.zeros(len(selected_coords), dtype=int)
            for i, coord in enumerate(selected_coords):
                x, y = coord
                bin_x = min(int(x * GRID_SIZE / RESOLUTION), GRID_SIZE - 1)
                bin_y = min(int(y * GRID_SIZE / RESOLUTION), GRID_SIZE - 1)
                bin_id_sel = bin_y * GRID_SIZE + bin_x
                selected_bin_assignments[i] = bin_id_sel

            # Get points in this bin
            bin_mask = selected_bin_assignments == bin_id
            if np.any(bin_mask):
                selected_bin_coords = selected_coords[bin_mask]
                selected_weights = bilinear_sample(lesionness_field, selected_bin_coords) * \
                    bilinear_sample(uncertainty_field, selected_bin_coords)
                bin_coverage[bin_id] = np.sum(selected_weights)

    # Calculate gating parameter λ1
    # Pool distribution for each bin
    pool_bin_weights = {}
    for bin_id in range(GRID_SIZE * GRID_SIZE):
        bin_mask = bin_assignments == bin_id
        if np.any(bin_mask):
            pool_bin_weights[bin_id] = np.sum(weights[bin_mask])
        else:
            pool_bin_weights[bin_id] = 0

    # Calculate entropy
    total_pool_weight = sum(pool_bin_weights.values())
    if total_pool_weight > 0:
        bin_entropy = 0
        for bin_id in range(GRID_SIZE * GRID_SIZE):
            p = pool_bin_weights[bin_id] / total_pool_weight
            if p > 0:
                bin_entropy += -p * np.log(p)
        lambda1 = max(LAMBDA_FLOOR, LAMBDA_MAX * bin_entropy / np.log(GRID_SIZE * GRID_SIZE))
    else:
        lambda1 = LAMBDA_FLOOR

    # Calculate scores for each available point
    scores = []
    for i, idx in enumerate(available_indices):
        bin_id = bin_assignments[i]
        u_uncertainty = uncertainty_values[i]
        u_weight = weights[i]

        # Calculate concave coverage gain
        current_coverage = bin_coverage[bin_id]
        new_coverage = current_coverage + u_weight

        # φ(u) = u^γ
        phi_current = current_coverage ** GAMMA if current_coverage > 0 else 0
        phi_new = new_coverage ** GAMMA

        coverage_gain = phi_new - phi_current

        # Final score
        score = u_uncertainty + lambda1 * coverage_gain
        scores.append(score)

    # Convert to probabilities using softmax
    scores = np.array(scores)
    scores = (scores - scores.max()) / TAU  # Numerical stability
    probs = np.exp(scores)
    probs = probs / probs.sum()

    # Sample using probability distribution
    selected_idx = np.random.choice(len(available_indices), n_select, replace=False, p=probs)
    return [available_indices[i] for i in selected_idx]


def update_uncertainty(uncertainty_field, selected_coords, alpha, sigma_decay=SIGMA_DECAY):
    """Update uncertainty field after selection using radial decay."""
    new_uncertainty = uncertainty_field.copy()

    for coord in selected_coords:
        # Create coordinate grid
        x = np.arange(RESOLUTION)
        y = np.arange(RESOLUTION)
        X, Y = np.meshgrid(x, y)

        # Calculate distances from all grid points to selected point
        distances = np.sqrt((X - coord[0])**2 + (Y - coord[1])**2)

        # Apply radial decay: U(v) *= (1 - α * exp(-||v-u||²/(2σ²)))
        decay_factor = 1 - alpha * np.exp(-distances**2 / (2 * (sigma_decay * RESOLUTION)**2))
        new_uncertainty *= decay_factor

    # Ensure uncertainty stays in [0, 1]
    new_uncertainty = np.clip(new_uncertainty, 0, 1)

    return new_uncertainty


def run_simulation(policy, max_rounds=40):
    """Run complete simulation for a given policy."""
    print(f"🔄 Running simulation for {policy}...")

    # Initialize
    selected_indices = []
    uncertainty_field = U0.copy()
    cumulative_selections = []

    for round_idx in range(max_rounds):
        # Select new samples
        if policy == 'random':
            new_selected = select_batch_random(pool_indices, BATCH_SIZE, selected_indices)
        elif policy == 'uncertainty':
            new_selected = select_batch_uncertainty(pool_indices, uncertainty_field, BATCH_SIZE, selected_indices)
        else:  # adaptive
            new_selected = select_batch_adaptive(pool_indices, uncertainty_field, M, BATCH_SIZE, selected_indices)

        # Add to cumulative selection
        selected_indices.extend(new_selected)
        cumulative_selections.append(selected_indices.copy())

        # Update uncertainty field
        if len(new_selected) > 0:
            selected_coords = pool_data[new_selected]
            uncertainty_field = update_uncertainty(
                uncertainty_field, selected_coords, ALPHA_RATES[policy]
            )

    return cumulative_selections, uncertainty_field


def calculate_movement_metrics(selections_by_round):
    """Calculate movement and clustering metrics."""
    movements = []
    clustering_scores = []

    for round_idx, selected_indices in enumerate(selections_by_round):
        if len(selected_indices) == 0:
            movements.append(0.0)
            clustering_scores.append(0.0)
            continue

        # Get coordinates for selected points
        coords = pool_data[selected_indices]

        # Calculate center of mass
        center = np.mean(coords, axis=0)

        # Calculate average distance from center (clustering measure)
        distances = np.sqrt(np.sum((coords - center)**2, axis=1))
        avg_distance = np.mean(distances)
        clustering_scores.append(avg_distance)

        # Calculate movement from previous round
        if round_idx > 0 and len(selections_by_round[round_idx-1]) > 0:
            prev_coords = pool_data[selections_by_round[round_idx-1]]
            prev_center = np.mean(prev_coords, axis=0)
            movement = np.sqrt(np.sum((center - prev_center)**2))
            movements.append(movement)
        else:
            movements.append(0.0)

    return movements, clustering_scores


def plot_cumulative_selections():
    """Create 3x4 subplot visualization of cumulative selections."""
    print("🎨 Creating cumulative selection visualization...")

    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Cumulative Selection Dynamics (N=40000, batch=50) — random vs uncertainty vs adaptive',
                 fontsize=16, fontweight='bold')

    # Policy names
    policies = ['Random', 'Uncertainty-only', 'Adaptive (ours)']

    # Round names
    round_names = [f'Round {r}' for r in ROUNDS]

    # Set row and column titles
    for i, policy in enumerate(policies):
        axes[i, 0].set_ylabel(policy, fontsize=14, fontweight='bold')

    for j, round_name in enumerate(round_names):
        axes[0, j].set_title(round_name, fontsize=12, fontweight='bold')

    # Run simulation for each policy
    for policy_idx, policy in enumerate(['random', 'uncertainty', 'adaptive']):
        # Run simulation
        cumulative_selections, final_uncertainty = run_simulation(policy)

        # Plot for each round
        for round_idx, round_num in enumerate(ROUNDS):
            ax = axes[policy_idx, round_idx]

            # Get cumulative selections up to this round
            if round_num <= len(cumulative_selections):
                round_selections = cumulative_selections[round_num - 1]
                round_coords = pool_data[round_selections]

                # Plot background uncertainty heatmap (faint)
                ax.imshow(U0, cmap='Greys', alpha=0.18, extent=[0, RESOLUTION, RESOLUTION, 0])

                # Plot cumulative selections with policy-specific styling
                if len(round_coords) > 0:
                    # Policy-specific colors and styles
                    if policy == 'random':
                        color = 'lightblue'
                        size = 6
                        alpha = min(0.4 + (round_num - 1) * 0.01, 0.7)
                        marker = 'o'
                    elif policy == 'uncertainty':
                        color = 'orange'
                        size = 8
                        alpha = min(0.5 + (round_num - 1) * 0.015, 0.8)
                        marker = 's'
                    else:  # adaptive
                        color = 'red'
                        size = 12
                        alpha = min(0.6 + (round_num - 1) * 0.02, 0.9)
                        marker = '*'

                    ax.scatter(round_coords[:, 0], round_coords[:, 1],
                               c=color, s=size, alpha=alpha, marker=marker,
                               edgecolors='black', linewidth=0.3)

            # Set axis properties
            ax.set_xlim(0, RESOLUTION)
            ax.set_ylim(0, RESOLUTION)
            ax.set_aspect('equal')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)

            # Add sample count
            n_samples = len(round_selections) if round_num <= len(cumulative_selections) else 0
            ax.text(0.02, 0.98, f'n={n_samples}', transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Create output directory
    os.makedirs('./figs', exist_ok=True)

    # Save plots
    png_path = './figs/sim_cumulative_3x4.png'
    pdf_path = './figs/sim_cumulative_3x4.pdf'

    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')

    print(f"✅ Visualization saved: {png_path}")
    print(f"✅ Visualization saved: {pdf_path}")

    return fig


def print_summary_statistics():
    """Print summary statistics for all policies."""
    print("\n" + "="*80)
    print("CUMULATIVE SELECTION DYNAMICS SIMULATION SUMMARY")
    print("="*80)

    for policy in ['random', 'uncertainty', 'adaptive']:
        print(f"\n📊 {policy.upper()} POLICY:")

        # Run simulation
        cumulative_selections, final_uncertainty = run_simulation(policy)

        # Calculate metrics for visualization rounds
        vis_rounds = [1, 10, 20, 40]
        movements, clustering = calculate_movement_metrics(cumulative_selections)

        # Print statistics for visualization rounds
        for round_num in vis_rounds:
            if round_num <= len(cumulative_selections):
                round_selections = cumulative_selections[round_num - 1]
                if len(round_selections) > 0:
                    coords = pool_data[round_selections]
                    center = np.mean(coords, axis=0)
                    print(
                        f"  Round {round_num}: center at ({center[0]:.1f}, {center[1]:.1f}), n={len(round_selections)}")

        # Calculate overall movement and clustering
        if len(movements) > 1:
            avg_movement = np.mean(movements[1:])  # Exclude first round
            avg_clustering = np.mean(clustering)
            print(f"  Average round-to-round movement: {avg_movement:.2f}")
            print(f"  Average clustering score: {avg_clustering:.2f}")


def main():
    """Main simulation function."""
    print("🚀 Starting Cumulative Selection Dynamics Simulation")
    print("="*60)

    # Generate data
    global pool_data, pool_indices, U0, M

    pool_data = generate_pool_data()
    pool_indices = list(range(len(pool_data)))
    U0, M = make_field_U0_M()

    print(f"📊 Pool size: {len(pool_data)}")
    print(f"📊 Batch size: {BATCH_SIZE}")
    print(f"📊 Rounds: {ROUNDS}")
    print(f"📊 Grid resolution: {RESOLUTION}x{RESOLUTION}")

    # Create visualization
    plot_cumulative_selections()

    # Print summary statistics
    print_summary_statistics()

    print("\n🎉 Simulation completed successfully!")
    print("📁 Results saved to ./figs/")


if __name__ == "__main__":
    main()
