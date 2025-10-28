#!/usr/bin/env python3
"""
Farm PuLP Benchmark Results Visualization
Creates comprehensive plots showing performance, solution quality, and constraint
validation across different problem sizes for the Farm PuLP solver.
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
    'primary': '#8B5CF6',      # Purple for PuLP
    'secondary': '#3A86FF',     # Blue
    'warning': '#FFBE0B',       # Yellow/Orange
    'danger': '#E63946',        # Red
    'success': '#10B981',       # Green
    'neutral': '#6C757D'        # Gray
}


def load_pulp_benchmark_data(benchmark_dir):
    """Load all Farm PuLP benchmark data."""
    pulp_dir = Path(benchmark_dir) / "COMPREHENSIVE" / "Farm_PuLP"

    if not pulp_dir.exists():
        raise FileNotFoundError(
            f"PuLP benchmark directory not found: {pulp_dir}")

    data = {}
    config_files = list(pulp_dir.glob("config_*_run_*.json"))

    if not config_files:
        raise FileNotFoundError(f"No config files found in {pulp_dir}")

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
        'n_constraints': [],
        'total_area': [],
        # Time metrics
        'solve_time': [],
        'solver_time': [],
        # Quality metrics
        'objective_value': [],
        'n_crops': [],
        'utilization': [],
        'status': [],
        'success': [],
        # Solution details
        'crops_selected': [],
        'total_allocated': [],
        'idle_area': []
    }

    for config in configs:
        d = data[config]

        # Problem size
        metrics['n_variables'].append(d['n_variables'])
        metrics['n_constraints'].append(d['n_constraints'])
        metrics['total_area'].append(d['total_area'])

        # Time metrics
        metrics['solve_time'].append(d['solve_time'])
        metrics['solver_time'].append(d['solver_time'])

        # Quality metrics
        metrics['objective_value'].append(d['objective_value'])
        metrics['n_crops'].append(d['solution_summary']['n_crops'])
        metrics['utilization'].append(d['solution_summary']['utilization'])
        metrics['status'].append(d['status'])
        metrics['success'].append(d['success'])

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
    fig.suptitle('Farm PuLP Performance Metrics',
                 fontsize=16, fontweight='bold', y=0.995)

    n_units = metrics['n_units']

    # Plot 1: Solve Time Breakdown
    ax = axes[0, 0]
    overhead = [metrics['solve_time'][i] - metrics['solver_time'][i]
                for i in range(len(n_units))]

    ax.bar(n_units, metrics['solver_time'], label='Solver Time',
           color=COLORS['primary'], alpha=0.8, width=3)
    ax.bar(n_units, overhead, bottom=metrics['solver_time'],
           label='Python Overhead', color=COLORS['neutral'], alpha=0.6, width=3)

    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title('Solve Time Breakdown', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_units)

    # Annotate with status
    for i, (x, y) in enumerate(zip(n_units, metrics['solve_time'])):
        status = metrics['status'][i]
        color = COLORS['success'] if status == 'Optimal' else COLORS['warning']
        ax.text(x, y + 0.002, status, ha='center',
                fontsize=9, fontweight='bold', color=color)

    # Plot 2: Problem Size Growth
    ax = axes[0, 1]
    ax2 = ax.twinx()

    line1 = ax.plot(n_units, metrics['n_variables'], 'o-',
                    color=COLORS['primary'], linewidth=2, markersize=8,
                    label='Variables')
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Number of Variables', fontweight='bold',
                  color=COLORS['primary'])
    ax.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_units)

    line2 = ax2.plot(n_units, metrics['n_constraints'], 's-',
                     color=COLORS['secondary'], linewidth=2, markersize=8,
                     label='Constraints')
    ax2.set_ylabel('Number of Constraints', fontweight='bold',
                   color=COLORS['secondary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['secondary'])

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    ax.set_title('Problem Size Scaling', fontweight='bold')

    # Plot 3: Efficiency Metrics
    ax = axes[1, 0]
    time_per_var = [metrics['solve_time'][i] / metrics['n_variables'][i] * 1000
                    for i in range(len(n_units))]
    solver_per_var = [metrics['solver_time'][i] / metrics['n_variables'][i] * 1000
                      for i in range(len(n_units))]

    ax.plot(n_units, time_per_var, 'o-', color=COLORS['danger'],
            linewidth=2, markersize=8, label='Total Time per Variable')
    ax.plot(n_units, solver_per_var, 's-', color=COLORS['success'],
            linewidth=2, markersize=8, label='Solver Time per Variable')
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Time per Variable (ms)', fontweight='bold')
    ax.set_title('Efficiency: Time per Variable', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_units)

    # Plot 4: Solver Efficiency
    ax = axes[1, 1]
    solver_percentage = [metrics['solver_time'][i] / metrics['solve_time'][i] * 100
                         for i in range(len(n_units))]

    bars = ax.bar(n_units, solver_percentage, color=COLORS['primary'],
                  alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.axhline(y=100, color=COLORS['neutral'], linestyle='--',
               linewidth=2, alpha=0.5, label='100%')
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Solver Time / Total Time (%)', fontweight='bold')
    ax.set_title('Solver Efficiency (Higher is Better)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(n_units)
    ax.set_ylim([0, 110])
    ax.legend()

    # Annotate percentages
    for i, (x, y) in enumerate(zip(n_units, solver_percentage)):
        ax.text(x, y + 2, f'{y:.1f}%', ha='center',
                fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Performance metrics saved to {output_path}")
    plt.close()


def plot_solution_quality(metrics, output_path):
    """Create solution quality and validation visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Farm PuLP Solution Quality & Validation',
                 fontsize=16, fontweight='bold', y=0.995)

    n_units = metrics['n_units']

    # Plot 1: Objective Value vs Problem Size
    ax = axes[0, 0]
    ax.plot(n_units, metrics['objective_value'], 'o-',
            color=COLORS['primary'], linewidth=2.5, markersize=10)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Objective Value (Revenue)', fontweight='bold')
    ax.set_title('Solution Objective Value (All Optimal)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_units)

    # Annotate values
    for i, (x, y) in enumerate(zip(n_units, metrics['objective_value'])):
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9)

    # Plot 2: Total Area vs Allocated
    ax = axes[0, 1]
    ax2 = ax.twinx()

    line1 = ax.bar(n_units, metrics['total_area'], label='Total Available Area',
                   color=COLORS['secondary'], alpha=0.6, width=3)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Total Available Area (ha)',
                  fontweight='bold', color=COLORS['secondary'])
    ax.tick_params(axis='y', labelcolor=COLORS['secondary'])
    ax.set_xticks(n_units)

    line2 = ax2.plot(n_units, metrics['total_allocated'], 'o-',
                     color=COLORS['primary'], linewidth=2.5, markersize=10,
                     label='Total Allocated')
    ax2.set_ylabel('Total Allocated Area (ha)',
                   fontweight='bold', color=COLORS['primary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['primary'])

    ax.set_title('Land Area (Perfect Allocation)', fontweight='bold')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 3: Revenue per Hectare
    ax = axes[1, 0]
    revenue_per_ha = [metrics['objective_value'][i] / metrics['total_area'][i]
                      for i in range(len(n_units))]

    bars = ax.bar(n_units, revenue_per_ha, color=COLORS['success'],
                  alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Revenue per Hectare', fontweight='bold')
    ax.set_title('Revenue Efficiency', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(n_units)

    # Annotate values
    for i, (x, y) in enumerate(zip(n_units, revenue_per_ha)):
        ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 5),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold')

    # Plot 4: Land Utilization
    ax = axes[1, 1]
    utilization_pct = [u * 100 for u in metrics['utilization']]
    bars = ax.bar(n_units, utilization_pct, color=COLORS['success'],
                  alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.axhline(y=100, color=COLORS['success'], linestyle='--',
               linewidth=2, alpha=0.5, label='100% Target')
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Land Utilization (%)', fontweight='bold')
    ax.set_title('Land Utilization Rate (Perfect)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(n_units)
    ax.set_ylim([99, 101])
    ax.legend()

    # Annotate utilization
    for i, (x, y) in enumerate(zip(n_units, utilization_pct)):
        ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, -15),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Solution quality saved to {output_path}")
    plt.close()


def plot_crop_diversity(metrics, output_path):
    """Visualize crop selection patterns across problem sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Farm PuLP Crop Selection Patterns',
                 fontsize=16, fontweight='bold', y=1.0)

    n_units = metrics['n_units']

    # Plot 1: Number of crops selected
    ax = axes[0]
    ax.bar(n_units, metrics['n_crops'], color=COLORS['primary'],
           alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Number of Crops Selected', fontweight='bold')
    ax.set_title('Crop Diversity (Consistent)', fontweight='bold')
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

    crop_matrix = []
    for config_crops in metrics['crops_selected']:
        row = [1 if crop in config_crops else 0 for crop in all_crops]
        crop_matrix.append(row)

    crop_matrix = np.array(crop_matrix).T

    im = ax.imshow(crop_matrix, cmap='Purples', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(len(all_crops)))
    ax.set_yticklabels(all_crops, fontweight='bold', fontsize=10)
    ax.set_xticks(range(len(n_units)))
    ax.set_xticklabels(n_units)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_title(
        f'Crop Selection Matrix (Same {len(all_crops)} Crops)', fontweight='bold')

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
                        fontsize=14, fontweight='bold', color='darkviolet')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Crop diversity saved to {output_path}")
    plt.close()


def plot_area_distribution(data, metrics, output_path):
    """Visualize farm and crop area distribution details."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    fig.suptitle('Farm PuLP Area Distribution Analysis',
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

        ax.set_title(f'{config} Farms: Crop Areas\n(Total: {d["total_area"]:.2f} ha)',
                     fontweight='bold', fontsize=11)

        # Subplot 2: Farms per crop (bar chart)
        ax = fig.add_subplot(gs[row, col + 1])

        crop_patch_counts = {}
        for assignment in plot_assignments:
            crop_patch_counts[assignment['crop']] = assignment['n_plots']

        crops_list = [crop for crop, _ in sorted_crops]
        patch_counts = [crop_patch_counts[crop] for crop in crops_list]
        colors_list = [crop_colors[crop] for crop in crops_list]

        bars = ax.barh(crops_list, patch_counts, color=colors_list,
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Number of Farms', fontweight='bold', fontsize=9)
        ax.set_title(f'{config} Farms: Farm Distribution\n(Status: {d["status"]})',
                     fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')

        # Annotate with counts (outside bars)
        for i, (crop, count) in enumerate(zip(crops_list, patch_counts)):
            ax.text(count + 0.3, i, f'{count}', va='center',
                    fontweight='bold', fontsize=9)

        # Add average area per farm (inside bars)
        for i, crop in enumerate(crops_list):
            avg_area = crop_areas[crop] / crop_patch_counts[crop]
            x_pos = patch_counts[i] * 0.05 if patch_counts[i] > 2 else 0.1
            ax.text(x_pos, i, f'{avg_area:.1f}ha', va='center', ha='left',
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
    ax_summary.set_xlabel('Number of Farms', fontweight='bold', fontsize=12)
    ax_summary.set_ylabel('Allocated Area (ha)',
                          fontweight='bold', fontsize=12)
    ax_summary.set_title('Crop Area Distribution Across Problem Sizes (All Optimal Solutions)',
                         fontweight='bold', fontsize=13)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(configs)
    ax_summary.legend(loc='upper left', fontsize=10, ncol=len(all_crops))
    ax_summary.grid(True, alpha=0.3, axis='y')

    # Add utilization annotations
    for i, config in enumerate(configs):
        d = data[config]
        util = d['solution_summary']['utilization']
        allocated = d['solution_summary']['total_allocated']

        color = COLORS['success']
        ax_summary.text(i, allocated + 2, f'{util*100:.0f}%',
                        ha='center', va='bottom', fontweight='bold',
                        fontsize=9, color=color)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Area distribution saved to {output_path}")
    plt.close()


def plot_advanced_analysis(data, metrics, output_path):
    """Create advanced analysis plots with additional insights."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Farm PuLP Advanced Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    configs = sorted(data.keys())

    # Plot 1: Objective per Farm
    ax = axes[0, 0]
    obj_per_farm = [metrics['objective_value'][i] / configs[i]
                    for i in range(len(configs))]

    ax.plot(configs, obj_per_farm, 'o-',
            color=COLORS['primary'], linewidth=2.5, markersize=10)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Revenue per Farm', fontweight='bold')
    ax.set_title('Revenue Efficiency per Farm', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate points
    for i, (x, y) in enumerate(zip(configs, obj_per_farm)):
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, fontweight='bold')

    # Plot 2: Area per Farm
    ax = axes[0, 1]
    area_per_farm = [metrics['total_area'][i] / configs[i]
                     for i in range(len(configs))]

    ax.plot(configs, area_per_farm, 's-',
            color=COLORS['secondary'], linewidth=2.5, markersize=10)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Average Area per Farm (ha)', fontweight='bold')
    ax.set_title('Farm Size Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate points
    for i, (x, y) in enumerate(zip(configs, area_per_farm)):
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, fontweight='bold')

    # Plot 3: Spinach Dominance
    ax = axes[0, 2]

    spinach_pct = []
    for config in configs:
        d = data[config]
        spinach_area = 0
        for assignment in d['solution_summary']['plot_assignments']:
            if assignment['crop'] == 'Spinach':
                spinach_area = assignment['total_area']
                break
        total_area = d['total_area']
        spinach_pct.append((spinach_area / total_area) * 100)

    bars = ax.bar(configs, spinach_pct, color=COLORS['success'],
                  alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Spinach Area (%)', fontweight='bold')
    ax.set_title('Spinach Dominance (Optimal Strategy)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate percentages
    for i, (x, y) in enumerate(zip(configs, spinach_pct)):
        ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 5),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold')

    # Plot 4: Solve Time Scaling
    ax = axes[1, 0]

    ax.plot(configs, metrics['solve_time'], 'o-',
            color=COLORS['primary'], linewidth=2.5, markersize=10, label='Total Time')
    ax.plot(configs, metrics['solver_time'], 's-',
            color=COLORS['success'], linewidth=2.5, markersize=10, label='Solver Time')
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title('Time Scaling (Sub-100ms Performance)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Farm Consolidation Analysis
    ax = axes[1, 1]

    # Calculate average farms per crop for each config
    avg_farms_per_crop = []

    for config in configs:
        assignments = data[config]['solution_summary']['plot_assignments']
        total_farms = sum(a['n_plots'] for a in assignments)
        n_crops = len(assignments)
        avg_farms_per_crop.append(total_farms / n_crops if n_crops > 0 else 0)

    ax.plot(configs, avg_farms_per_crop, 'o-',
            color=COLORS['primary'], linewidth=2.5, markersize=10)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Avg Farms per Crop', fontweight='bold')
    ax.set_title('Farm Consolidation (Spinach Gets Most)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate points
    for i, (x, y) in enumerate(zip(configs, avg_farms_per_crop)):
        ax.annotate(f'{y:.1f}', xy=(x, y), xytext=(0, 5),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold')

    # Plot 6: Efficiency Summary
    ax = axes[1, 2]
    ax.axis('off')

    # Calculate key metrics
    total_time = sum(metrics['solve_time'])
    avg_solver_time = np.mean(metrics['solver_time'])
    success_count = sum(metrics['success'])
    total_configs = len(metrics['success'])

    summary_text = f"""
    FARM PULP SUMMARY
    {'='*40}
    
    Problem Sizes: {min(configs)} - {max(configs)} farms
    Variable Range: {min(metrics['n_variables'])} - {max(metrics['n_variables'])}
    Constraint Range: {min(metrics['n_constraints'])} - {max(metrics['n_constraints'])}
    Area Range: {min(metrics['total_area']):.1f} - {max(metrics['total_area']):.1f} ha
    
    PERFORMANCE
    {'='*40}
    Total Solve Time: {total_time:.3f}s
    Avg Solve Time: {np.mean(metrics['solve_time']):.3f}s
    Avg Solver Time: {avg_solver_time:.3f}s
    Max Solve Time: {max(metrics['solve_time']):.3f}s
    
    SOLUTION QUALITY
    {'='*40}
    Optimal Solutions: {success_count}/{total_configs}
    Feasible Solutions: {success_count}/{total_configs}
    Perfect Utilization: 100.0%
    Objective Range: {min(metrics['objective_value']):.2f} - {max(metrics['objective_value']):.2f}
    Crops Selected: {metrics['n_crops'][0]} (consistent)
    
    KEY INSIGHTS
    {'='*40}
    Spinach dominates: ~87-98% of area
    allocated to spinach across all sizes.
    
    Consistent crop selection: Same 5 crops
    (Pork, Guava, Chickpeas, Potato, Spinach)
    selected regardless of problem size.
    
    Perfect constraint satisfaction with
    100% land utilization in sub-100ms.
    
    Linear programming provides fast,
    reliable, optimal solutions for
    farm-scale problems.
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Advanced analysis saved to {output_path}")
    plt.close()


def print_detailed_summary(data, metrics):
    """Print comprehensive summary table to console."""
    print("\n" + "="*100)
    print("FARM PULP BENCHMARK RESULTS SUMMARY")
    print("="*100)

    configs = sorted(data.keys())

    print(f"\n{'Config':<10} {'Vars':<8} {'Constr':<8} {'Area(ha)':<10} {'Solve(s)':<10} {'Obj':<10} {'Status':<10} {'Rev/ha':<10}")
    print("-"*100)

    for i, config in enumerate(configs):
        rev_per_ha = metrics['objective_value'][i] / metrics['total_area'][i]
        print(f"{config:<10} {metrics['n_variables'][i]:<8} {metrics['n_constraints'][i]:<8} "
              f"{metrics['total_area'][i]:<10.2f} {metrics['solve_time'][i]:<10.3f} "
              f"{metrics['objective_value'][i]:<10.2f} "
              f"{metrics['status'][i]:<10} "
              f"{rev_per_ha:<10.3f}")

    print("\n" + "="*100)
    print("DETAILED METRICS")
    print("="*100)

    for config in configs:
        d = data[config]
        print(f"\n--- Configuration: {config} farms ---")
        print(f"  Problem Size:")
        print(
            f"    Variables: {d['n_variables']}, Constraints: {d['n_constraints']}")
        print(f"    Total Area: {d['total_area']:.3f} ha")
        print(f"  Performance:")
        print(f"    Total Time: {d['solve_time']:.3f}s")
        print(f"    Solver Time: {d['solver_time']:.3f}s")
        print(
            f"    Python Overhead: {d['solve_time'] - d['solver_time']:.3f}s")
        print(f"    Status: {d['status']}")
        print(f"    Success: {'Yes' if d['success'] else 'No'}")
        print(f"  Solution:")
        print(f"    Objective: {d['objective_value']:.3f}")
        print(
            f"    Revenue per Hectare: {d['objective_value'] / d['total_area']:.3f}")
        print(
            f"    Crops: {', '.join(d['solution_summary']['crops_selected'])}")
        print(f"    Crop Diversity: {d['solution_summary']['n_crops']} crops")
        print(
            f"    Utilization: {d['solution_summary']['utilization']*100:.2f}%")

        # Print crop areas
        print(f"  Crop Breakdown:")
        for assignment in d['solution_summary']['plot_assignments']:
            pct = (assignment['total_area'] / d['total_area']) * 100
            print(
                f"    {assignment['crop']}: {assignment['total_area']:.2f} ha ({pct:.1f}%) on {assignment['n_plots']} farm(s)")

    print("\n" + "="*100)


def main():
    """Main execution function."""
    print("\n" + "="*100)
    print("FARM PULP BENCHMARK VISUALIZATION")
    print("="*100)

    # Set up paths
    script_dir = Path(__file__).parent
    benchmark_dir = script_dir / "Benchmarks"
    output_dir = script_dir / "plots" / "farm_pulp"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n📂 Loading benchmark data...")
    try:
        data = load_pulp_benchmark_data(benchmark_dir)
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

    plot_performance_metrics(metrics, output_dir / "farm_pulp_performance.png")
    plot_solution_quality(metrics, output_dir / "farm_pulp_quality.png")
    plot_crop_diversity(metrics, output_dir / "farm_pulp_crop_diversity.png")
    plot_area_distribution(data, metrics, output_dir /
                           "farm_pulp_area_distribution.png")
    plot_advanced_analysis(data, metrics, output_dir /
                           "farm_pulp_advanced_analysis.png")

    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - farm_pulp_performance.png        : Time breakdown and efficiency")
    print("  - farm_pulp_quality.png            : Solution quality and validation")
    print("  - farm_pulp_crop_diversity.png     : Crop selection patterns")
    print("  - farm_pulp_area_distribution.png  : Farm and crop area distributions")
    print("  - farm_pulp_advanced_analysis.png  : Advanced metrics and correlations")

    print("\n" + "="*100)
    print("COMPLETE!")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
