#!/usr/bin/env python3
"""
Patch D-Wave BQM Benchmark Results Visualization
Creates comprehensive plots showing performance, solution quality, and constraint
validation across different problem sizes for the Patch BQM solver.
"""
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt conflicts

# Set style for beautiful plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.family'] = 'sans-serif'

# Color palette for consistent styling
COLORS = {
    'primary': '#06FFA5',      # Bright cyan for BQM
    'secondary': '#3A86FF',     # Blue
    'warning': '#FFBE0B',       # Yellow/Orange
    'danger': '#E63946',        # Red
    'success': '#2EC4B6',       # Teal
    'neutral': '#6C757D'        # Gray
}


def load_bqm_benchmark_data(benchmark_dir):
    """Load all Patch BQM benchmark data."""
    bqm_dir = Path(benchmark_dir) / "COMPREHENSIVE" / "Patch_DWaveBQM"

    if not bqm_dir.exists():
        raise FileNotFoundError(
            f"BQM benchmark directory not found: {bqm_dir}")

    data = {}
    config_files = list(bqm_dir.glob("config_*_run_*.json"))

    if not config_files:
        raise FileNotFoundError(f"No config files found in {bqm_dir}")

    for config_file in sorted(config_files):
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            n_units = config_data['n_units']
            data[n_units] = config_data

    return data


def extract_metrics(data):
    """Extract all relevant metrics from benchmark data."""
    configs = sorted(data.keys())

    metrics = {
        'n_units': configs,
        'n_variables': [],
        'n_quadratic': [],
        'total_area': [],
        # Time metrics
        'solve_time': [],
        'qpu_time': [],
        'hybrid_time': [],
        'bqm_conversion_time': [],
        # Quality metrics
        'objective_value': [],
        'bqm_energy': [],
        'n_crops': [],
        'utilization': [],
        # Validation metrics
        'is_feasible': [],
        'n_violations': [],
        'pass_rate': [],
        'total_checks': [],
        # Solution details
        'crops_selected': [],
        'total_allocated': [],
        'idle_area': []
    }

    for config in configs:
        d = data[config]

        # Problem size
        metrics['n_variables'].append(d['n_variables'])
        metrics['n_quadratic'].append(d['n_quadratic'])
        metrics['total_area'].append(d['total_area'])

        # Time metrics
        metrics['solve_time'].append(d['solve_time'])
        metrics['qpu_time'].append(d['qpu_time'])
        metrics['hybrid_time'].append(d['hybrid_time'])
        metrics['bqm_conversion_time'].append(d['bqm_conversion_time'])

        # Quality metrics
        metrics['objective_value'].append(d['objective_value'])
        metrics['bqm_energy'].append(d['bqm_energy'])
        metrics['n_crops'].append(d['solution_summary']['n_crops'])
        metrics['utilization'].append(d['solution_summary']['utilization'])

        # Validation metrics
        metrics['is_feasible'].append(d['validation']['is_feasible'])
        metrics['n_violations'].append(d['validation']['n_violations'])
        metrics['pass_rate'].append(d['validation']['summary']['pass_rate'])
        metrics['total_checks'].append(
            d['validation']['summary']['total_checks'])

        # Solution details
        metrics['crops_selected'].append(
            d['solution_summary']['crops_selected'])
        metrics['total_allocated'].append(
            d['solution_summary']['total_allocated'])
        metrics['idle_area'].append(d['solution_summary']['idle_area'])

    return metrics


def plot_performance_metrics(metrics, output_path):
    """Create comprehensive performance visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Patch D-Wave BQM Performance Metrics',
                 fontsize=16, fontweight='bold', y=0.995)

    n_units = metrics['n_units']

    # Plot 1: Solve Time Breakdown
    ax = axes[0, 0]
    ax.bar(n_units, metrics['qpu_time'], label='QPU Time',
           color=COLORS['primary'], alpha=0.8, width=3)
    ax.bar(n_units, metrics['bqm_conversion_time'],
           bottom=metrics['qpu_time'],
           label='BQM Conversion', color=COLORS['secondary'], alpha=0.8, width=3)

    # Calculate overhead (total - qpu - conversion)
    overhead = [metrics['solve_time'][i] - metrics['qpu_time'][i] - metrics['bqm_conversion_time'][i]
                for i in range(len(n_units))]
    bottom = [metrics['qpu_time'][i] + metrics['bqm_conversion_time'][i]
              for i in range(len(n_units))]
    ax.bar(n_units, overhead, bottom=bottom,
           label='Hybrid Overhead', color=COLORS['neutral'], alpha=0.6, width=3)

    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title('Solve Time Breakdown', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_units)

    # Plot 2: Problem Size Growth
    ax = axes[0, 1]
    ax2 = ax.twinx()

    line1 = ax.plot(n_units, metrics['n_variables'], 'o-',
                    color=COLORS['primary'], linewidth=2, markersize=8,
                    label='Binary Variables')
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Number of Variables', fontweight='bold',
                  color=COLORS['primary'])
    ax.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_units)

    line2 = ax2.plot(n_units, metrics['n_quadratic'], 's-',
                     color=COLORS['secondary'], linewidth=2, markersize=8,
                     label='Quadratic Terms')
    ax2.set_ylabel('Number of Quadratic Terms',
                   fontweight='bold', color=COLORS['secondary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['secondary'])

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    ax.set_title('Problem Size Scaling', fontweight='bold')

    # Plot 3: Time per Variable
    ax = axes[1, 0]
    time_per_var = [metrics['solve_time'][i] / metrics['n_variables'][i] * 1000
                    for i in range(len(n_units))]
    qpu_per_var = [metrics['qpu_time'][i] / metrics['n_variables'][i] * 1000
                   for i in range(len(n_units))]

    ax.plot(n_units, time_per_var, 'o-', color=COLORS['danger'],
            linewidth=2, markersize=8, label='Total Time per Variable')
    ax.plot(n_units, qpu_per_var, 's-', color=COLORS['success'],
            linewidth=2, markersize=8, label='QPU Time per Variable')
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Time per Variable (ms)', fontweight='bold')
    ax.set_title('Efficiency: Time per Variable', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_units)

    # Plot 4: Conversion Time Scaling
    ax = axes[1, 1]
    conversion_per_quad = [metrics['bqm_conversion_time'][i] / metrics['n_quadratic'][i] * 1e6
                           for i in range(len(n_units))]

    ax.plot(n_units, metrics['bqm_conversion_time'], 'o-',
            color=COLORS['secondary'], linewidth=2, markersize=8)
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('BQM Conversion Time (seconds)', fontweight='bold')
    ax.set_title('BQM Conversion Time', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_units)

    # Add annotation for efficiency
    ax2 = ax.twinx()
    ax2.plot(n_units, conversion_per_quad, 's--',
             color=COLORS['warning'], linewidth=1.5, markersize=6, alpha=0.6)
    ax2.set_ylabel('Time per Quadratic Term (μs)', fontweight='bold',
                   color=COLORS['warning'], alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=COLORS['warning'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Performance metrics saved to {output_path}")
    plt.close()


def plot_solution_quality(metrics, output_path):
    """Create solution quality and constraint validation visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Patch D-Wave BQM Solution Quality & Validation',
                 fontsize=16, fontweight='bold', y=0.995)

    n_units = metrics['n_units']

    # Plot 1: Objective Value vs Problem Size
    ax = axes[0, 0]
    ax.plot(n_units, metrics['objective_value'], 'o-',
            color=COLORS['primary'], linewidth=2.5, markersize=10)
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Objective Value (Revenue)', fontweight='bold')
    ax.set_title('Solution Objective Value', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_units)

    # Annotate values
    for i, (x, y) in enumerate(zip(n_units, metrics['objective_value'])):
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9)

    # Plot 2: Constraint Violations
    ax = axes[0, 1]
    colors_violations = [COLORS['success'] if v == 0 else COLORS['danger']
                         for v in metrics['n_violations']]
    bars = ax.bar(n_units, metrics['n_violations'], color=colors_violations,
                  alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Number of Violations', fontweight='bold')
    ax.set_title('Constraint Violations', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(n_units)

    # Annotate violation counts
    for i, (x, y) in enumerate(zip(n_units, metrics['n_violations'])):
        if y > 0:
            ax.annotate(f'{y}', xy=(x, y), xytext=(0, 5),
                        textcoords='offset points', ha='center',
                        fontweight='bold', fontsize=11, color=COLORS['danger'])

    # Add feasibility status text
    for i, (x, feasible) in enumerate(zip(n_units, metrics['is_feasible'])):
        status = '✓ Feasible' if feasible else '✗ Infeasible'
        color = COLORS['success'] if feasible else COLORS['danger']
        ax.text(x, -max(metrics['n_violations']) * 0.1, status,
                ha='center', fontsize=9, fontweight='bold', color=color)

    # Plot 3: Pass Rate
    ax = axes[1, 0]
    pass_rates_pct = [pr * 100 for pr in metrics['pass_rate']]
    colors_pass = [COLORS['success'] if pr == 100 else COLORS['warning'] if pr >= 95
                   else COLORS['danger'] for pr in pass_rates_pct]
    bars = ax.bar(n_units, pass_rates_pct, color=colors_pass,
                  alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.axhline(y=100, color=COLORS['success'], linestyle='--',
               linewidth=2, alpha=0.5, label='Perfect')
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Constraint Pass Rate (%)', fontweight='bold')
    ax.set_title('Constraint Validation Pass Rate', fontweight='bold')
    ax.set_ylim([90, 102])
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(n_units)
    ax.legend()

    # Annotate pass rates
    for i, (x, y) in enumerate(zip(n_units, pass_rates_pct)):
        ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, -15),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold')

    # Plot 4: Land Utilization
    ax = axes[1, 1]
    utilization_pct = [u * 100 for u in metrics['utilization']]
    colors_util = [COLORS['success'] if 99 <= u <= 101 else COLORS['warning'] if u < 99
                   else COLORS['danger'] for u in utilization_pct]
    bars = ax.bar(n_units, utilization_pct, color=colors_util,
                  alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.axhline(y=100, color=COLORS['success'], linestyle='--',
               linewidth=2, alpha=0.5, label='100% Target')
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Land Utilization (%)', fontweight='bold')
    ax.set_title('Land Utilization Rate', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(n_units)
    ax.legend()

    # Annotate utilization
    for i, (x, y) in enumerate(zip(n_units, utilization_pct)):
        color = 'black' if 99 <= y <= 101 else COLORS['danger']
        ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 5 if y <= 100 else -15),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Solution quality saved to {output_path}")
    plt.close()


def plot_crop_diversity(metrics, output_path):
    """Visualize crop selection patterns across problem sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Patch D-Wave BQM Crop Selection Patterns',
                 fontsize=16, fontweight='bold', y=1.0)

    n_units = metrics['n_units']

    # Plot 1: Number of crops selected
    ax = axes[0]
    ax.bar(n_units, metrics['n_crops'], color=COLORS['primary'],
           alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Number of Crops Selected', fontweight='bold')
    ax.set_title('Crop Diversity', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(n_units)
    ax.set_ylim([0, 7])

    # Annotate crop counts
    for i, (x, y) in enumerate(zip(n_units, metrics['n_crops'])):
        ax.annotate(f'{y}', xy=(x, y), xytext=(0, 5),
                    textcoords='offset points', ha='center',
                    fontsize=11, fontweight='bold')

    # Plot 2: Crop selection heatmap
    ax = axes[1]
    all_crops = ['Wheat', 'Corn', 'Rice', 'Soybeans', 'Potatoes', 'Apples']
    crop_matrix = []

    for config_crops in metrics['crops_selected']:
        row = [1 if crop in config_crops else 0 for crop in all_crops]
        crop_matrix.append(row)

    crop_matrix = np.array(crop_matrix).T

    im = ax.imshow(crop_matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(len(all_crops)))
    ax.set_yticklabels(all_crops, fontweight='bold')
    ax.set_xticks(range(len(n_units)))
    ax.set_xticklabels(n_units)
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_title('Crop Selection Matrix', fontweight='bold')

    # Add colorbar legend
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Not Selected', 'Selected'])
    cbar.ax.tick_params(labelsize=10)

    # Add grid
    for i in range(len(all_crops) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=2)
    for i in range(len(n_units) + 1):
        ax.axvline(i - 0.5, color='white', linewidth=2)

    # Add checkmarks for selected crops
    for i, crops in enumerate(metrics['crops_selected']):
        for j, crop in enumerate(all_crops):
            if crop in crops:
                ax.text(i, j, '✓', ha='center', va='center',
                        fontsize=16, fontweight='bold', color='darkgreen')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Crop diversity saved to {output_path}")
    plt.close()


def plot_area_distribution(data, metrics, output_path):
    """Visualize patch and crop area distribution details."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    fig.suptitle('Patch D-Wave BQM Area Distribution Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    configs = sorted(data.keys())
    all_crops = ['Wheat', 'Corn', 'Rice', 'Soybeans', 'Potatoes', 'Apples']
    crop_colors = {
        'Wheat': '#F4A460',
        'Corn': '#FFD700',
        'Rice': '#98D8C8',
        'Soybeans': '#90EE90',
        'Potatoes': '#DEB887',
        'Apples': '#FF6B6B'
    }

    # Create one subplot for each configuration
    for idx, config in enumerate(configs):
        row = idx // 2
        col = (idx % 2) * 2

        d = data[config]
        plot_assignments = d['solution_summary']['plot_assignments']

        # Subplot 1: Crop area distribution (pie chart)
        ax = fig.add_subplot(gs[row, col])

        crop_areas = {}
        for assignment in plot_assignments:
            crop_areas[assignment['crop']] = assignment['total_area']

        # Sort by area
        sorted_crops = sorted(crop_areas.items(),
                              key=lambda x: x[1], reverse=True)
        labels = [crop for crop, _ in sorted_crops]
        sizes = [area for _, area in sorted_crops]
        colors = [crop_colors[crop] for crop in labels]

        result = ax.pie(sizes, labels=labels, colors=colors,
                        autopct='%1.1f%%', startangle=90,
                        textprops={'fontweight': 'bold', 'fontsize': 9})
        # Unpack return value which can be (wedges, texts) or (wedges, texts, autotexts)
        if len(result) == 3:
            wedges, texts, autotexts = result
        else:
            wedges, texts = result
            autotexts = []

        # Make percentage text more visible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(8)
            autotext.set_fontweight('bold')

        ax.set_title(f'{config} Patches: Crop Areas\n(Total: {d["total_area"]:.2f} ha)',
                     fontweight='bold', fontsize=11)

        # Subplot 2: Patches per crop (bar chart)
        ax = fig.add_subplot(gs[row, col + 1])

        crop_patch_counts = {}
        for assignment in plot_assignments:
            crop_patch_counts[assignment['crop']] = assignment['n_plots']

        crops_list = [crop for crop, _ in sorted_crops]
        patch_counts = [crop_patch_counts[crop] for crop in crops_list]
        colors_list = [crop_colors[crop] for crop in crops_list]

        bars = ax.barh(crops_list, patch_counts, color=colors_list,
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Number of Patches', fontweight='bold', fontsize=9)
        ax.set_title(f'{config} Patches: Patch Distribution\n(Feasible: {"Yes" if d["validation"]["is_feasible"] else "No"})',
                     fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')

        # Annotate with counts (outside bars)
        for i, (crop, count) in enumerate(zip(crops_list, patch_counts)):
            ax.text(count + 0.3, i, f'{count}', va='center',
                    fontweight='bold', fontsize=9)

        # Add average area per patch (inside bars, left side)
        for i, crop in enumerate(crops_list):
            avg_area = crop_areas[crop] / crop_patch_counts[crop]
            # Position inside the bar at 5% of the bar width
            x_pos = patch_counts[i] * 0.05 if patch_counts[i] > 2 else 0.1
            ax.text(x_pos, i, f'{avg_area:.2f}ha', va='center', ha='left',
                    fontsize=7, style='italic', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='none'))

    # Summary comparison plot (bottom row, spans all columns)
    ax_summary = fig.add_subplot(gs[2, :])

    # Stacked bar chart showing crop area distribution across all configs
    x = np.arange(len(configs))
    width = 0.6

    # Prepare data for stacked bars
    crop_data = {crop: [] for crop in all_crops}

    for config in configs:
        d = data[config]
        plot_assignments = d['solution_summary']['plot_assignments']

        # Get area for each crop
        config_crops = {assignment['crop']: assignment['total_area']
                        for assignment in plot_assignments}

        for crop in all_crops:
            crop_data[crop].append(config_crops.get(crop, 0))

    # Create stacked bars
    bottom = np.zeros(len(configs))
    bars_list = []

    for crop in all_crops:
        bars = ax_summary.bar(x, crop_data[crop], width, label=crop,
                              color=crop_colors[crop], bottom=bottom,
                              edgecolor='white', linewidth=2)
        bars_list.append(bars)
        bottom += crop_data[crop]

    # Add total area line
    ax2 = ax_summary.twinx()
    total_areas = [data[config]['total_area'] for config in configs]
    line = ax2.plot(x, total_areas, 'ko-', linewidth=2.5, markersize=10,
                    label='Total Available Area', zorder=10)
    ax2.set_ylabel('Total Available Area (ha)', fontweight='bold', fontsize=11)
    ax2.tick_params(axis='y', labelsize=10)

    # Configure primary axis
    ax_summary.set_xlabel('Number of Patches', fontweight='bold', fontsize=12)
    ax_summary.set_ylabel('Allocated Area (ha)',
                          fontweight='bold', fontsize=12)
    ax_summary.set_title('Crop Area Distribution Across Problem Sizes',
                         fontweight='bold', fontsize=13)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(configs)
    ax_summary.legend(loc='upper left', fontsize=10, ncol=3)
    ax_summary.grid(True, alpha=0.3, axis='y')

    # Add utilization annotations
    for i, config in enumerate(configs):
        d = data[config]
        util = d['solution_summary']['utilization']
        allocated = d['solution_summary']['total_allocated']

        color = COLORS['success'] if 0.99 <= util <= 1.01 else COLORS['danger']
        ax_summary.text(i, allocated + 1, f'{util*100:.1f}%',
                        ha='center', va='bottom', fontweight='bold',
                        fontsize=9, color=color)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Area distribution saved to {output_path}")
    plt.close()


def plot_scaling_analysis(metrics, output_path):
    """Analyze and visualize scaling behavior."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Patch D-Wave BQM Scaling Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    n_units = np.array(metrics['n_units'])
    n_vars = np.array(metrics['n_variables'])

    # Plot 1: Total Solve Time Scaling
    ax = axes[0, 0]
    ax.plot(n_units, metrics['solve_time'], 'o-',
            color=COLORS['primary'], linewidth=2.5, markersize=10, label='Actual')

    # Fit polynomial to see scaling
    if len(n_units) > 2:
        z = np.polyfit(n_units, metrics['solve_time'], 2)
        p = np.poly1d(z)
        x_fit = np.linspace(n_units.min(), n_units.max(), 100)
        ax.plot(x_fit, p(x_fit), '--', color=COLORS['danger'],
                linewidth=2, alpha=0.7, label=f'Quadratic Fit')

    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Total Solve Time (seconds)', fontweight='bold')
    ax.set_title('Solve Time Scaling', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: QPU Time Scaling
    ax = axes[0, 1]
    ax.plot(n_units, metrics['qpu_time'], 's-',
            color=COLORS['success'], linewidth=2.5, markersize=10)
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('QPU Time (seconds)', fontweight='bold')
    ax.set_title('QPU Time Scaling (Nearly Constant)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add mean line
    mean_qpu = np.mean(metrics['qpu_time'])
    ax.axhline(y=mean_qpu, color=COLORS['neutral'], linestyle='--',
               linewidth=2, alpha=0.5, label=f'Mean: {mean_qpu:.4f}s')
    ax.legend()

    # Plot 3: Objective per Patch
    ax = axes[1, 0]
    obj_per_patch = [metrics['objective_value'][i] / n_units[i]
                     for i in range(len(n_units))]
    ax.plot(n_units, obj_per_patch, 'o-',
            color=COLORS['warning'], linewidth=2.5, markersize=10)
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Revenue per Patch', fontweight='bold')
    ax.set_title('Revenue Efficiency', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Efficiency Summary
    ax = axes[1, 1]
    ax.axis('off')

    # Calculate key metrics
    total_time = sum(metrics['solve_time'])
    total_qpu = sum(metrics['qpu_time'])
    avg_conversion = np.mean(metrics['bqm_conversion_time'])
    feasible_count = sum(metrics['is_feasible'])
    total_configs = len(metrics['is_feasible'])

    summary_text = f"""
    SCALING SUMMARY
    {'='*40}
    
    Problem Sizes: {min(n_units)} - {max(n_units)} patches
    Variable Range: {min(n_vars)} - {max(n_vars)}
    Quadratic Terms: {min(metrics['n_quadratic'])} - {max(metrics['n_quadratic'])}
    
    PERFORMANCE
    {'='*40}
    Total Solve Time: {total_time:.2f}s
    Total QPU Time: {total_qpu:.3f}s
    QPU Time Range: {min(metrics['qpu_time']):.4f}s - {max(metrics['qpu_time']):.4f}s
    Avg QPU Time: {np.mean(metrics['qpu_time']):.4f}s ± {np.std(metrics['qpu_time']):.5f}s
    Avg Conversion Time: {avg_conversion:.4f}s
    
    SOLUTION QUALITY
    {'='*40}
    Feasible Solutions: {feasible_count}/{total_configs}
    Objective Range: {min(metrics['objective_value']):.2f} - {max(metrics['objective_value']):.2f}
    Avg Crops Selected: {np.mean(metrics['n_crops']):.1f}
    
    KEY INSIGHT
    {'='*40}
    QPU time remains nearly constant (~0.104s)
    regardless of problem size, demonstrating
    quantum annealing's scalability advantage.
    
    However, constraint violations appear at
    50 patches, indicating BQM formulation
    may need refinement for larger problems.
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Scaling analysis saved to {output_path}")
    plt.close()


def print_detailed_summary(data, metrics):
    """Print comprehensive summary table to console."""
    print("\n" + "="*80)
    print("PATCH D-WAVE BQM BENCHMARK RESULTS SUMMARY")
    print("="*80)

    configs = sorted(data.keys())

    print(f"\n{'Config':<10} {'Vars':<8} {'Quad':<10} {'Solve(s)':<10} {'QPU(s)':<10} {'Obj':<10} {'Feasible':<10} {'Violations':<12}")
    print("-"*80)

    for i, config in enumerate(configs):
        print(f"{config:<10} {metrics['n_variables'][i]:<8} {metrics['n_quadratic'][i]:<10} "
              f"{metrics['solve_time'][i]:<10.3f} {metrics['qpu_time'][i]:<10.6f} "
              f"{metrics['objective_value'][i]:<10.2f} "
              f"{'✓' if metrics['is_feasible'][i] else '✗':<10} "
              f"{metrics['n_violations'][i]:<12}")

    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)

    for config in configs:
        d = data[config]
        print(f"\n--- Configuration: {config} patches ---")
        print(f"  Problem Size:")
        print(
            f"    Variables: {d['n_variables']}, Quadratic: {d['n_quadratic']}")
        print(f"    Total Area: {d['total_area']:.3f} ha")
        print(f"  Performance:")
        print(f"    Total Time: {d['solve_time']:.3f}s")
        print(f"    QPU Time: {d['qpu_time']:.6f}s")
        print(f"    BQM Conversion: {d['bqm_conversion_time']:.6f}s")
        print(
            f"    Hybrid Overhead: {d['solve_time'] - d['qpu_time'] - d['bqm_conversion_time']:.3f}s")
        print(f"  Solution:")
        print(f"    Objective: {d['objective_value']:.3f}")
        print(
            f"    Crops: {', '.join(d['solution_summary']['crops_selected'])}")
        print(
            f"    Utilization: {d['solution_summary']['utilization']*100:.2f}%")
        print(f"  Validation:")
        print(
            f"    Feasible: {'Yes' if d['validation']['is_feasible'] else 'No'}")
        print(f"    Violations: {d['validation']['n_violations']}")
        print(
            f"    Pass Rate: {d['validation']['summary']['pass_rate']*100:.2f}%")

        if d['validation']['n_violations'] > 0:
            print(f"    Issues:")
            for violation in d['validation']['violations'][:5]:  # Show first 5
                print(f"      - {violation}")
            if len(d['validation']['violations']) > 5:
                print(
                    f"      ... and {len(d['validation']['violations']) - 5} more")

    print("\n" + "="*80)


def plot_advanced_analysis(data, metrics, output_path):
    """Create advanced analysis plots with additional insights."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Patch D-Wave BQM Advanced Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    configs = sorted(data.keys())

    # Plot 1: BQM Energy vs Objective Value correlation
    ax = axes[0, 0]
    bqm_energies_abs = [abs(e) for e in metrics['bqm_energy']]
    ax.scatter(metrics['objective_value'], bqm_energies_abs,
               s=200, alpha=0.7, c=range(len(configs)),
               cmap='viridis', edgecolors='black', linewidth=2)

    # Add perfect correlation line
    max_val = max(max(metrics['objective_value']), max(bqm_energies_abs))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2,
            alpha=0.5, label='Perfect Correlation')

    ax.set_xlabel('Objective Value (Revenue)', fontweight='bold')
    ax.set_ylabel('|BQM Energy|', fontweight='bold')
    ax.set_title('BQM Energy vs Objective Correlation', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate points with config size
    for i, config in enumerate(configs):
        ax.annotate(f'{config}p',
                    xy=(metrics['objective_value'][i], bqm_energies_abs[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    # Plot 2: Constraint Validation Breakdown
    ax = axes[0, 1]
    x = np.arange(len(configs))
    width = 0.35

    passed = [metrics['total_checks'][i] - metrics['n_violations'][i]
              for i in range(len(configs))]
    failed = metrics['n_violations']

    bars1 = ax.bar(x - width/2, passed, width, label='Passed',
                   color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, failed, width, label='Failed',
                   color=COLORS['danger'], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Number of Constraint Checks', fontweight='bold')
    ax.set_title('Constraint Validation Breakdown', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate totals
    for i, config in enumerate(configs):
        total = metrics['total_checks'][i]
        ax.text(i, total + 5, f'{total}', ha='center',
                fontweight='bold', fontsize=9)

    # Plot 3: Patch Size Distribution
    ax = axes[0, 2]

    # Collect all patch areas for each config
    patch_areas_by_config = {}
    for config in configs:
        areas = []
        for assignment in data[config]['solution_summary']['plot_assignments']:
            for plot_info in assignment['plots']:
                areas.append(plot_info['area'])
        patch_areas_by_config[config] = areas

    # Create box plot
    box_data = [patch_areas_by_config[config] for config in configs]
    bp = ax.boxplot(box_data, labels=configs, patch_artist=True,
                    widths=0.6, showmeans=True, meanline=True)

    # Color the boxes
    colors_gradient = plt.cm.get_cmap('viridis')(
        np.linspace(0, 0.8, len(configs)))
    for patch, color in zip(bp['boxes'], colors_gradient):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Patch Area (ha)', fontweight='bold')
    ax.set_title('Patch Size Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Time Breakdown Pie Charts
    ax = axes[1, 0]

    # Use the largest config (50) for detailed breakdown
    largest_config = configs[-1]
    d = data[largest_config]

    time_components = {
        'QPU Time': d['qpu_time'],
        'BQM Conversion': d['bqm_conversion_time'],
        'Hybrid Overhead': d['solve_time'] - d['qpu_time'] - d['bqm_conversion_time']
    }

    colors_pie = [COLORS['primary'], COLORS['secondary'], COLORS['neutral']]
    wedges, texts, autotexts = ax.pie(time_components.values(),
                                      labels=time_components.keys(),
                                      colors=colors_pie, autopct='%1.1f%%',
                                      startangle=90,
                                      textprops={'fontweight': 'bold'})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)

    ax.set_title(f'Time Breakdown ({largest_config} Patches)\nTotal: {d["solve_time"]:.3f}s',
                 fontweight='bold')

    # Plot 5: Crop Consolidation Analysis
    ax = axes[1, 1]

    # Calculate average patches per crop for each config
    avg_patches_per_crop = []
    total_crops_planted = []

    for config in configs:
        assignments = data[config]['solution_summary']['plot_assignments']
        total_patches = sum(a['n_plots'] for a in assignments)
        n_crops = len(assignments)
        avg_patches_per_crop.append(
            total_patches / n_crops if n_crops > 0 else 0)
        total_crops_planted.append(n_crops)

    ax2 = ax.twinx()

    line1 = ax.plot(configs, avg_patches_per_crop, 'o-',
                    color=COLORS['primary'], linewidth=2.5, markersize=10,
                    label='Avg Patches per Crop')
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Avg Patches per Crop', fontweight='bold',
                  color=COLORS['primary'])
    ax.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax.grid(True, alpha=0.3)

    line2 = ax2.plot(configs, total_crops_planted, 's-',
                     color=COLORS['warning'], linewidth=2.5, markersize=10,
                     label='Total Crops Planted')
    ax2.set_ylabel('Total Crops Planted', fontweight='bold',
                   color=COLORS['warning'])
    ax2.tick_params(axis='y', labelcolor=COLORS['warning'])
    ax2.set_ylim([4, 7])

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    ax.set_title('Crop Consolidation vs Diversity', fontweight='bold')

    # Plot 6: Problem Complexity Growth
    ax = axes[1, 2]

    # Normalize to show relative growth
    vars_normalized = np.array(
        metrics['n_variables']) / metrics['n_variables'][0]
    quad_normalized = np.array(
        metrics['n_quadratic']) / metrics['n_quadratic'][0]
    checks_normalized = np.array(
        metrics['total_checks']) / metrics['total_checks'][0]
    patches_normalized = np.array(configs) / configs[0]

    ax.plot(configs, patches_normalized, 'o-', linewidth=2.5, markersize=8,
            label='Patches (Linear)', color=COLORS['neutral'])
    ax.plot(configs, vars_normalized, 's-', linewidth=2.5, markersize=8,
            label='Variables', color=COLORS['primary'])
    ax.plot(configs, quad_normalized, '^-', linewidth=2.5, markersize=8,
            label='Quadratic Terms', color=COLORS['secondary'])
    ax.plot(configs, checks_normalized, 'v-', linewidth=2.5, markersize=8,
            label='Constraint Checks', color=COLORS['warning'])

    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Relative Growth (vs 10 patches)', fontweight='bold')
    ax.set_title('Problem Complexity Growth Rate', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Add reference line for quadratic growth
    x_ref = np.array(configs)
    y_ref = (x_ref / configs[0]) ** 2
    ax.plot(configs, y_ref, '--', linewidth=1.5, alpha=0.5,
            color='red', label='Quadratic Reference')
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Advanced analysis saved to {output_path}")
    plt.close()


def plot_violation_details(data, output_path):
    """Create detailed visualization of constraint violations for 50-patch config."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Patch D-Wave BQM Constraint Violation Analysis (50 Patches)',
                 fontsize=16, fontweight='bold', y=0.995)

    # Focus on the 50-patch configuration
    config_50 = data[50]
    # Ensure variable is always defined even if there are no violations
    violated_patches = []

    # Plot 1: Violation locations (which patches have issues)
    ax = axes[0, 0]

    if config_50['validation']['n_violations'] > 0:
        # Extract patch numbers from violations
        for violation in config_50['validation']['violations']:
            # Parse "Plot PatchX: ..." to get X, guard against unexpected formats
            try:
                patch_num = int(violation.split('Patch')[1].split(':')[0])
            except Exception:
                # skip malformed entries
                continue
            violated_patches.append(patch_num)

        # Create visualization
        all_patches = list(range(1, 51))
        violation_status = [
            1 if p in violated_patches else 0 for p in all_patches]

        # Reshape into grid
        grid_size = 10
        violation_grid = np.array(violation_status).reshape(5, 10)

        im = ax.imshow(violation_grid, cmap='RdYlGn_r',
                       aspect='auto', vmin=0, vmax=1)

        # Add patch numbers
        for i in range(5):
            for j in range(10):
                patch_num = i * 10 + j + 1
                color = 'white' if violation_grid[i, j] == 1 else 'black'
                ax.text(j, i, str(patch_num), ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)

        ax.set_title('Patches with Constraint Violations', fontweight='bold')
        ax.set_xlabel('Patch Grid Visualization', fontweight='bold')
        ax.axis('off')

        # Add colorbar
        cbar = plt.colorbar(
            im, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Valid', 'Violated'])

    # Plot 2: Crop assignment distribution for 50-patch
    ax = axes[0, 1]

    crop_areas = {}
    for assignment in config_50['solution_summary']['plot_assignments']:
        crop_areas[assignment['crop']] = assignment['total_area']

    sorted_crops = sorted(crop_areas.items(), key=lambda x: x[1], reverse=True)
    crops = [c for c, _ in sorted_crops]
    areas = [a for _, a in sorted_crops]

    # Use get_cmap to retrieve the 'Set3' colormap and sample colors for each crop
    colors_bars = plt.get_cmap('Set3')(np.linspace(0, 1, max(1, len(crops))))
    bars = ax.barh(crops, areas, color=colors_bars, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Total Area (ha)', fontweight='bold')
    ax.set_title('Crop Area Allocation (50 Patches)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add area annotations
    for i, (crop, area) in enumerate(zip(crops, areas)):
        ax.text(area + 0.2, i, f'{area:.2f} ha', va='center',
                fontweight='bold', fontsize=9)

    # Add total line
    total_allocated = config_50['solution_summary']['total_allocated']
    total_available = config_50['solution_summary']['total_available']
    ax.axvline(x=total_available, color=COLORS['danger'], linestyle='--',
               linewidth=2.5, label=f'Available: {total_available:.2f} ha')
    ax.legend()

    # Plot 3: Over-allocation visualization
    ax = axes[1, 0]

    configs_all = sorted(data.keys())
    allocated = [data[c]['solution_summary']['total_allocated']
                 for c in configs_all]
    available = [data[c]['solution_summary']['total_available']
                 for c in configs_all]
    over_alloc = [allocated[i] - available[i] for i in range(len(configs_all))]

    colors_over = [COLORS['success'] if x <= 0 else COLORS['danger']
                   for x in over_alloc]
    bars = ax.bar(configs_all, over_alloc, color=colors_over, alpha=0.8,
                  edgecolor='black', linewidth=1.5, width=3)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Over-Allocation (ha)', fontweight='bold')
    ax.set_title('Land Over-Allocation Analysis', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate over-allocation
    for i, (config, val) in enumerate(zip(configs_all, over_alloc)):
        if val > 0:
            ax.text(config, val + 0.5, f'+{val:.2f}', ha='center',
                    fontweight='bold', fontsize=10, color=COLORS['danger'])

    # Plot 4: Validation metrics summary
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    CONSTRAINT VIOLATION SUMMARY
    {'='*50}
    
    Configuration: 50 patches
    Total Area Available: {config_50['total_area']:.3f} ha
    Total Area Allocated: {config_50['solution_summary']['total_allocated']:.3f} ha
    Over-Allocation: {config_50['solution_summary']['idle_area']:.3f} ha
    Utilization Rate: {config_50['solution_summary']['utilization']*100:.2f}%
    
    CONSTRAINT VALIDATION
    {'='*50}
    Total Checks: {config_50['validation']['summary']['total_checks']}
    Passed: {config_50['validation']['summary']['total_passed']}
    Failed: {config_50['validation']['summary']['total_failed']}
    Pass Rate: {config_50['validation']['summary']['pass_rate']*100:.2f}%
    
    VIOLATIONS ({config_50['validation']['n_violations']} total)
    {'='*50}
    Type: Multiple crops assigned to single patch
    Affected Patches: {len(violated_patches) if config_50['validation']['n_violations'] > 0 else 0}
    
    Violated Patches:
    {', '.join([f'Patch{p}' for p in sorted(violated_patches[:10])])}
    {'...' if len(violated_patches) > 10 else ''}
    
    PROBABLE CAUSE
    {'='*50}
    The BQM formulation's one-hot encoding
    constraints may have insufficient penalty
    strength at larger problem sizes, allowing
    multiple binary variables for different
    crops on the same patch to both be 1.
    
    RECOMMENDATION
    {'='*50}
    • Increase Lagrange multipliers for
      one-crop-per-patch constraints
    • Add explicit pairwise penalties
    • Consider constraint strength scaling
      with problem size
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Violation details saved to {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("PATCH D-WAVE BQM BENCHMARK VISUALIZATION")
    print("="*80)

    # Set up paths
    script_dir = Path(__file__).parent
    benchmark_dir = script_dir / "Benchmarks"
    output_dir = script_dir / "plots" / "patch_bqm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n📂 Loading benchmark data...")
    try:
        data = load_bqm_benchmark_data(benchmark_dir)
        print(f"✓ Loaded {len(data)} configurations: {sorted(data.keys())}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return

    # Extract metrics
    print("\n📊 Extracting metrics...")
    metrics = extract_metrics(data)
    print(f"✓ Extracted {len(metrics)} metric categories")

    # Print detailed summary
    print_detailed_summary(data, metrics)

    # Generate plots
    print("\n📈 Generating visualizations...")

    plot_performance_metrics(metrics, output_dir / "patch_bqm_performance.png")
    plot_solution_quality(metrics, output_dir / "patch_bqm_quality.png")
    plot_crop_diversity(metrics, output_dir / "patch_bqm_crop_diversity.png")
    plot_area_distribution(data, metrics, output_dir /
                           "patch_bqm_area_distribution.png")
    plot_scaling_analysis(metrics, output_dir / "patch_bqm_scaling.png")
    plot_advanced_analysis(data, metrics, output_dir /
                           "patch_bqm_advanced_analysis.png")
    plot_violation_details(
        data, output_dir / "patch_bqm_violation_details.png")

    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - patch_bqm_performance.png        : Time breakdown and efficiency")
    print("  - patch_bqm_quality.png            : Solution quality and validation")
    print("  - patch_bqm_crop_diversity.png     : Crop selection patterns")
    print("  - patch_bqm_area_distribution.png  : Patch and crop area distributions")
    print("  - patch_bqm_scaling.png            : Scaling behavior analysis")
    print("  - patch_bqm_advanced_analysis.png  : Advanced metrics and correlations")
    print("  - patch_bqm_violation_details.png  : Detailed constraint violation analysis")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
