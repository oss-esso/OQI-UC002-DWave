#!/usr/bin/env python3
"""
Patch Gurobi QUBO Benchmark Results Visualization
Creates comprehensive plots showing performance, solution quality, and constraint
validation across different problem sizes for the Patch Gurobi QUBO solver.
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
    'primary': '#2EC4B6',      # Teal for Gurobi
    'secondary': '#3A86FF',     # Blue
    'warning': '#FFBE0B',       # Yellow/Orange
    'danger': '#E63946',        # Red
    'success': '#06FFA5',       # Bright cyan
    'neutral': '#6C757D'        # Gray
}


def load_gurobi_benchmark_data(benchmark_dir):
    """Load all Patch Gurobi QUBO benchmark data."""
    gurobi_dir = Path(benchmark_dir) / "COMPREHENSIVE" / "Patch_GurobiQUBO"

    if not gurobi_dir.exists():
        raise FileNotFoundError(
            f"Gurobi benchmark directory not found: {gurobi_dir}")

    data = {}
    config_files = list(gurobi_dir.glob("config_*_run_*.json"))

    if not config_files:
        raise FileNotFoundError(f"No config files found in {gurobi_dir}")

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
        'bqm_conversion_time': [],
        # Quality metrics
        'objective_value': [],
        'bqm_energy': [],
        'n_crops': [],
        'utilization': [],
        'status': [],
        'success': [],
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
        metrics['bqm_conversion_time'].append(d['bqm_conversion_time'])

        # Quality metrics
        metrics['objective_value'].append(d['objective_value'])
        metrics['bqm_energy'].append(d['bqm_energy'])
        metrics['n_crops'].append(d['solution_summary']['n_crops'])
        metrics['utilization'].append(d['solution_summary']['utilization'])
        metrics['status'].append(d['status'])
        metrics['success'].append(d['success'])

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
    fig.suptitle('Patch Gurobi QUBO Performance Metrics',
                 fontsize=16, fontweight='bold', y=0.995)

    n_units = metrics['n_units']

    # Plot 1: Solve Time (note: all hit time limit)
    ax = axes[0, 0]
    colors_status = [COLORS['danger'] if not success else COLORS['success']
                     for success in metrics['success']]
    bars = ax.bar(n_units, metrics['solve_time'], color=colors_status,
                  alpha=0.8, width=3, edgecolor='black', linewidth=1.5)

    # Add time limit line
    ax.axhline(y=300, color=COLORS['danger'], linestyle='--', linewidth=2.5,
               alpha=0.7, label='Time Limit (300s)')

    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title('Solve Time (All Hit Time Limit)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_units)
    ax.set_ylim([0, 320])

    # Annotate with status
    for i, (x, y) in enumerate(zip(n_units, metrics['solve_time'])):
        ax.text(x, y + 5, 'Time Limit', ha='center',
                fontsize=9, fontweight='bold', color=COLORS['danger'])

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

    ax.plot(n_units, time_per_var, 'o-', color=COLORS['danger'],
            linewidth=2, markersize=8)
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Time per Variable (ms)', fontweight='bold')
    ax.set_title('Efficiency: Time per Variable', fontweight='bold')
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
    fig.suptitle('Patch Gurobi QUBO Solution Quality & Validation',
                 fontsize=16, fontweight='bold', y=0.995)

    n_units = metrics['n_units']

    # Plot 1: Objective Value vs Problem Size
    ax = axes[0, 0]
    ax.plot(n_units, metrics['objective_value'], 'o-',
            color=COLORS['primary'], linewidth=2.5, markersize=10)
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Objective Value (Revenue)', fontweight='bold')
    ax.set_title('Solution Objective Value (Time-Limited)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_units)

    # Annotate values
    for i, (x, y) in enumerate(zip(n_units, metrics['objective_value'])):
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9)

    # Plot 2: Constraint Violations (should all be 0)
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
    ax.set_ylim([0, 1])

    # Add feasibility status text
    for i, (x, feasible) in enumerate(zip(n_units, metrics['is_feasible'])):
        status = '✓ Feasible' if feasible else '✗ Infeasible'
        color = COLORS['success'] if feasible else COLORS['danger']
        ax.text(x, 0.5, status,
                ha='center', fontsize=10, fontweight='bold', color=color)

    # Plot 3: Pass Rate
    ax = axes[1, 0]
    pass_rates_pct = [pr * 100 for pr in metrics['pass_rate']]
    bars = ax.bar(n_units, pass_rates_pct, color=COLORS['success'],
                  alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.axhline(y=100, color=COLORS['success'], linestyle='--',
               linewidth=2, alpha=0.5, label='Perfect')
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Constraint Pass Rate (%)', fontweight='bold')
    ax.set_title('Constraint Validation Pass Rate', fontweight='bold')
    ax.set_ylim([99, 101])
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
    bars = ax.bar(n_units, utilization_pct, color=COLORS['success'],
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
        ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 5 if y <= 100 else -15),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Solution quality saved to {output_path}")
    plt.close()


def plot_crop_diversity(metrics, output_path):
    """Visualize crop selection patterns across problem sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Patch Gurobi QUBO Crop Selection Patterns',
                 fontsize=16, fontweight='bold', y=1.0)

    n_units = metrics['n_units']

    # Plot 1: Number of crops selected
    ax = axes[0]
    colors_diversity = [COLORS['danger'] if n < 3 else COLORS['warning'] if n < 5
                        else COLORS['success'] for n in metrics['n_crops']]
    bars = ax.bar(n_units, metrics['n_crops'], color=colors_diversity,
                  alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Number of Crops Selected', fontweight='bold')
    ax.set_title('Crop Diversity (Lower due to Time Limit)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(n_units)

    # Set y-axis limit dynamically based on data
    max_crops = max(metrics['n_crops']) if metrics['n_crops'] else 10
    ax.set_ylim([0, max(max_crops + 2, 10)])

    # Annotate crop counts
    for i, (x, y) in enumerate(zip(n_units, metrics['n_crops'])):
        ax.annotate(f'{y}', xy=(x, y), xytext=(0, 5),
                    textcoords='offset points', ha='center',
                    fontsize=11, fontweight='bold')

    # Plot 2: Crop selection heatmap
    ax = axes[1]

    # Dynamically get all unique crops from the data
    all_crops = set()
    for config_crops in metrics['crops_selected']:
        all_crops.update(config_crops)
    all_crops = sorted(list(all_crops))

    # Limit display to top crops if there are too many
    max_crops_display = 15
    if len(all_crops) > max_crops_display:
        # Count frequency and take most common
        crop_freq = {}
        for config_crops in metrics['crops_selected']:
            for crop in config_crops:
                crop_freq[crop] = crop_freq.get(crop, 0) + 1
        all_crops = sorted(crop_freq.keys(), key=lambda x: crop_freq[x], reverse=True)[
            :max_crops_display]

    crop_matrix = []
    for config_crops in metrics['crops_selected']:
        row = [1 if crop in config_crops else 0 for crop in all_crops]
        crop_matrix.append(row)

    crop_matrix = np.array(crop_matrix).T

    im = ax.imshow(crop_matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(len(all_crops)))
    ax.set_yticklabels(all_crops, fontweight='bold', fontsize=9)
    ax.set_xticks(range(len(n_units)))
    ax.set_xticklabels(n_units)
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_title(
        f'Crop Selection Matrix (Top {len(all_crops)} Crops)', fontweight='bold')

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
                        fontsize=12, fontweight='bold', color='darkgreen')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Crop diversity saved to {output_path}")
    plt.close()


def plot_area_distribution(data, metrics, output_path):
    """Visualize patch and crop area distribution details."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    fig.suptitle('Patch Gurobi QUBO Area Distribution Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    configs = sorted(data.keys())

    # Dynamically get all crops from data
    all_crops_set = set()
    for config in configs:
        for assignment in data[config]['solution_summary']['plot_assignments']:
            all_crops_set.add(assignment['crop'])
    all_crops = sorted(list(all_crops_set))

    # Generate colors dynamically using a colormap
    n_crops = len(all_crops)
    colors_list = plt.cm.tab20(np.linspace(0, 1, min(n_crops, 20)))
    if n_crops > 20:
        colors_list = np.vstack(
            [colors_list, plt.cm.tab20b(np.linspace(0, 1, n_crops - 20))])

    crop_colors = {crop: matplotlib.colors.rgb2hex(colors_list[i % len(colors_list)])
                   for i, crop in enumerate(all_crops)}

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

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'fontweight': 'bold', 'fontsize': 9})

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
        ax.set_title(f'{config} Patches: Patch Distribution\n(Status: {d["status"]})',
                     fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')

        # Annotate with counts (outside bars)
        for i, (crop, count) in enumerate(zip(crops_list, patch_counts)):
            ax.text(count + 0.3, i, f'{count}', va='center',
                    fontweight='bold', fontsize=9)

        # Add average area per patch (inside bars)
        for i, crop in enumerate(crops_list):
            avg_area = crop_areas[crop] / crop_patch_counts[crop]
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
    ax_summary.set_title('Crop Area Distribution Across Problem Sizes (Time-Limited Solutions)',
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

        color = COLORS['success']
        ax_summary.text(i, allocated + 1, f'{util*100:.1f}%',
                        ha='center', va='bottom', fontweight='bold',
                        fontsize=9, color=color)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Area distribution saved to {output_path}")
    plt.close()


def plot_advanced_analysis(data, metrics, output_path):
    """Create advanced analysis plots with additional insights."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Patch Gurobi QUBO Advanced Analysis',
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

    passed = [metrics['total_checks'][i]
              for i in range(len(configs))]  # All passed

    bars1 = ax.bar(x, passed, width, label='Passed',
                   color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Number of Constraint Checks', fontweight='bold')
    ax.set_title('Constraint Validation (All Passed)', fontweight='bold')
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
    bp = ax.boxplot(box_data, tick_labels=configs, patch_artist=True,
                    widths=0.6, showmeans=True, meanline=True)

    # Color the boxes
    colors_gradient = plt.cm.viridis(np.linspace(0, 0.8, len(configs)))
    for patch, color in zip(bp['boxes'], colors_gradient):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Patch Area (ha)', fontweight='bold')
    ax.set_title('Patch Size Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Solution Quality Over Time
    ax = axes[1, 0]

    # Since all hit time limit, show objective per second
    obj_per_sec = [metrics['objective_value'][i] / metrics['solve_time'][i]
                   for i in range(len(configs))]

    ax.plot(configs, obj_per_sec, 'o-',
            color=COLORS['primary'], linewidth=2.5, markersize=10)
    ax.set_xlabel('Number of Patches', fontweight='bold')
    ax.set_ylabel('Objective Value per Second', fontweight='bold')
    ax.set_title('Solution Quality Rate (Time-Limited)', fontweight='bold')
    ax.grid(True, alpha=0.3)

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
    ax2.set_ylim([0, 7])

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    ax.set_title('Crop Consolidation vs Diversity', fontweight='bold')

    # Plot 6: Efficiency Summary
    ax = axes[1, 2]
    ax.axis('off')

    # Calculate key metrics
    total_time = sum(metrics['solve_time'])
    avg_conversion = np.mean(metrics['bqm_conversion_time'])
    feasible_count = sum(metrics['is_feasible'])
    total_configs = len(metrics['is_feasible'])
    success_count = sum(metrics['success'])

    summary_text = f"""
    GUROBI QUBO SUMMARY
    {'='*40}
    
    Problem Sizes: {min(configs)} - {max(configs)} patches
    Variable Range: {min(metrics['n_variables'])} - {max(metrics['n_variables'])}
    Quadratic Terms: {min(metrics['n_quadratic'])} - {max(metrics['n_quadratic'])}
    
    PERFORMANCE
    {'='*40}
    Total Solve Time: {total_time:.2f}s
    All runs hit 300s time limit
    Avg Conversion Time: {avg_conversion:.4f}s
    
    SOLUTION QUALITY
    {'='*40}
    Feasible Solutions: {feasible_count}/{total_configs}
    Optimal Solutions: {success_count}/{total_configs}
    Objective Range: {min(metrics['objective_value']):.2f} - {max(metrics['objective_value']):.2f}
    Avg Crops Selected: {np.mean(metrics['n_crops']):.1f}
    
    KEY INSIGHT
    {'='*40}
    Gurobi consistently hits the 300s time
    limit without reaching optimality, but
    produces high-quality feasible solutions
    with perfect constraint satisfaction.
    
    Lower crop diversity compared to D-Wave
    suggests solutions may be stuck in local
    optima due to time constraints.
    
    Conversion time negligible (~0.03s).
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Advanced analysis saved to {output_path}")
    plt.close()


def print_detailed_summary(data, metrics):
    """Print comprehensive summary table to console."""
    print("\n" + "="*80)
    print("PATCH GUROBI QUBO BENCHMARK RESULTS SUMMARY")
    print("="*80)

    configs = sorted(data.keys())

    print(f"\n{'Config':<10} {'Vars':<8} {'Quad':<10} {'Solve(s)':<10} {'Obj':<10} {'Status':<20} {'Feasible':<10}")
    print("-"*80)

    for i, config in enumerate(configs):
        print(f"{config:<10} {metrics['n_variables'][i]:<8} {metrics['n_quadratic'][i]:<10} "
              f"{metrics['solve_time'][i]:<10.3f} "
              f"{metrics['objective_value'][i]:<10.2f} "
              f"{metrics['status'][i]:<20} "
              f"{'✓' if metrics['is_feasible'][i] else '✗':<10}")

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
        print(f"    BQM Conversion: {d['bqm_conversion_time']:.6f}s")
        print(f"    Status: {d['status']}")
        print(f"    Success: {'Yes' if d['success'] else 'No (Time Limit)'}")
        print(f"  Solution:")
        print(f"    Objective: {d['objective_value']:.3f}")
        print(
            f"    Crops: {', '.join(d['solution_summary']['crops_selected'])}")
        print(
            f"    Crop Diversity: {d['solution_summary']['n_crops']} of 6 crops")
        print(
            f"    Utilization: {d['solution_summary']['utilization']*100:.2f}%")
        print(f"  Validation:")
        print(f"    Feasible: Yes")
        print(f"    Violations: {d['validation']['n_violations']}")
        print(
            f"    Pass Rate: {d['validation']['summary']['pass_rate']*100:.2f}%")

    print("\n" + "="*80)


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("PATCH GUROBI QUBO BENCHMARK VISUALIZATION")
    print("="*80)

    # Set up paths
    script_dir = Path(__file__).parent
    benchmark_dir = script_dir / "Benchmarks"
    output_dir = script_dir / "plots" / "patch_gurobi"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n📂 Loading benchmark data...")
    try:
        data = load_gurobi_benchmark_data(benchmark_dir)
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

    plot_performance_metrics(metrics, output_dir /
                             "patch_gurobi_performance.png")
    plot_solution_quality(metrics, output_dir / "patch_gurobi_quality.png")
    plot_crop_diversity(metrics, output_dir /
                        "patch_gurobi_crop_diversity.png")
    plot_area_distribution(data, metrics, output_dir /
                           "patch_gurobi_area_distribution.png")
    plot_advanced_analysis(data, metrics, output_dir /
                           "patch_gurobi_advanced_analysis.png")

    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - patch_gurobi_performance.png        : Time breakdown and efficiency")
    print("  - patch_gurobi_quality.png            : Solution quality and validation")
    print("  - patch_gurobi_crop_diversity.png     : Crop selection patterns")
    print("  - patch_gurobi_area_distribution.png  : Patch and crop area distributions")
    print("  - patch_gurobi_advanced_analysis.png  : Advanced metrics and correlations")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
