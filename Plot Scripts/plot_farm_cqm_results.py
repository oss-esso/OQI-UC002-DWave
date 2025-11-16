#!/usr/bin/env python3
"""
Farm D-Wave CQM Benchmark Results Visualization
Creates comprehensive plots showing performance, solution quality, and constraint
validation across different problem sizes for the Farm CQM (Constrained Quadratic Model) solver.
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
    'primary': '#00C9A7',      # Teal/Cyan for CQM
    'secondary': '#3A86FF',     # Blue
    'warning': '#FFBE0B',       # Yellow/Orange
    'danger': '#E63946',        # Red
    'success': '#10B981',       # Green
    'neutral': '#6C757D'        # Gray
}


def load_cqm_benchmark_data(benchmark_dir):
    """Load all Farm D-Wave CQM benchmark data."""
    cqm_dir = Path(benchmark_dir) / "COMPREHENSIVE" / "Farm_DWave"

    if not cqm_dir.exists():
        raise FileNotFoundError(
            f"CQM benchmark directory not found: {cqm_dir}")

    data = {}
    config_files = list(cqm_dir.glob("config_*_run_*.json"))

    if not config_files:
        raise FileNotFoundError(f"No config files found in {cqm_dir}")

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
        'qpu_time': [],
        'hybrid_time': [],
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
        metrics['qpu_time'].append(d['qpu_time'])
        metrics['hybrid_time'].append(d['hybrid_time'])

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
    fig.suptitle('Farm D-Wave CQM Performance Metrics',
                 fontsize=16, fontweight='bold', y=0.995)

    n_units = metrics['n_units']

    # Plot 1: Solve Time Breakdown
    ax = axes[0, 0]
    overhead = [metrics['solve_time'][i] - metrics['qpu_time'][i]
                for i in range(len(n_units))]

    ax.bar(n_units, metrics['qpu_time'], label='QPU Time',
           color=COLORS['primary'], alpha=0.8, width=3)
    ax.bar(n_units, overhead, bottom=metrics['qpu_time'],
           label='Hybrid Overhead', color=COLORS['neutral'], alpha=0.6, width=3)

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
        ax.text(x, y + 0.1, status, ha='center',
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
    qpu_per_var = [metrics['qpu_time'][i] / metrics['n_variables'][i] * 1000
                   for i in range(len(n_units))]

    ax.plot(n_units, time_per_var, 'o-', color=COLORS['danger'],
            linewidth=2, markersize=8, label='Total Time per Variable')
    ax.plot(n_units, qpu_per_var, 's-', color=COLORS['success'],
            linewidth=2, markersize=8, label='QPU Time per Variable')
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Time per Variable (ms)', fontweight='bold')
    ax.set_title('Efficiency: Time per Variable', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_units)

    # Plot 4: QPU Efficiency
    ax = axes[1, 1]
    qpu_percentage = [metrics['qpu_time'][i] / metrics['solve_time'][i] * 100
                      for i in range(len(n_units))]

    bars = ax.bar(n_units, qpu_percentage, color=COLORS['primary'],
                  alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('QPU Time / Total Time (%)', fontweight='bold')
    ax.set_title('QPU Efficiency (Minimal Overhead)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(n_units)
    ax.set_ylim([0, 2])

    # Annotate percentages
    for i, (x, y) in enumerate(zip(n_units, qpu_percentage)):
        ax.text(x, y + 0.05, f'{y:.2f}%', ha='center',
                fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Performance metrics saved to {output_path}")
    plt.close()


def plot_solution_quality(metrics, output_path):
    """Create solution quality and constraint validation visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Farm D-Wave CQM Solution Quality & Validation',
                 fontsize=16, fontweight='bold', y=0.995)

    n_units = metrics['n_units']

    # Plot 1: Objective Value vs Problem Size
    ax = axes[0, 0]
    ax.plot(n_units, metrics['objective_value'], 'o-',
            color=COLORS['primary'], linewidth=2.5, markersize=10)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Objective Value (Revenue)', fontweight='bold')
    ax.set_title('Solution Objective Value', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_units)

    # Annotate values
    for i, (x, y) in enumerate(zip(n_units, metrics['objective_value'])):
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9)

    # Plot 2: Allocated vs Available Area
    ax = axes[0, 1]

    x = np.arange(len(n_units))
    width = 0.35

    bars1 = ax.bar(x - width/2, metrics['total_area'], width, label='Available',
                   color=COLORS['secondary'], alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, metrics['total_allocated'], width, label='Allocated',
                   color=COLORS['warning'], alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Area (ha)', fontweight='bold')
    ax.set_title('Area Allocation', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(n_units)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate values
    for i in range(len(n_units)):
        ax.text(i - width/2, metrics['total_area'][i] + 5, f'{metrics["total_area"][i]:.0f}',
                ha='center', fontweight='bold', fontsize=9)
        ax.text(i + width/2, metrics['total_allocated'][i] + 5, f'{metrics["total_allocated"][i]:.0f}',
                ha='center', fontweight='bold', fontsize=9)

    # Plot 3: Crop Diversity
    ax = axes[1, 0]
    ax.bar(n_units, metrics['n_crops'], color=COLORS['primary'],
           alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Number of Crops Selected', fontweight='bold')
    ax.set_title('Crop Diversity', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(n_units)

    # Annotate crop counts
    for i, (x, y) in enumerate(zip(n_units, metrics['n_crops'])):
        ax.annotate(f'{y}', xy=(x, y), xytext=(0, 5),
                    textcoords='offset points', ha='center',
                    fontsize=11, fontweight='bold')

    # Plot 4: Land Utilization
    ax = axes[1, 1]
    utilization_pct = [u * 100 for u in metrics['utilization']]
    colors_util = [COLORS['success'] if 99 <= u <= 101 else COLORS['warning'] if u < 150
                   else COLORS['danger'] for u in utilization_pct]
    bars = ax.bar(n_units, utilization_pct, color=colors_util,
                  alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.axhline(y=100, color=COLORS['success'], linestyle='--',
               linewidth=2, alpha=0.5, label='100% Target')
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Land Utilization (%)', fontweight='bold')
    ax.set_title('Land Utilization Rate', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(n_units)
    ax.legend()

    # Annotate utilization
    for i, (x, y) in enumerate(zip(n_units, utilization_pct)):
        ax.annotate(f'{y:.0f}%', xy=(x, y), xytext=(0, 5 if y <= 150 else -15),
                    textcoords='offset points', ha='center',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Solution quality saved to {output_path}")
    plt.close()


def plot_crop_diversity(metrics, output_path):
    """Visualize crop selection patterns across problem sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Farm D-Wave CQM Crop Selection Patterns',
                 fontsize=16, fontweight='bold', y=1.0)

    n_units = metrics['n_units']

    # Plot 1: Number of crops selected
    ax = axes[0]
    ax.bar(n_units, metrics['n_crops'], color=COLORS['primary'],
           alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Number of Crops Selected', fontweight='bold')
    ax.set_title('Crop Diversity', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(n_units)

    # Set y-axis limit dynamically based on data
    max_crops = max(metrics['n_crops']) if metrics['n_crops'] else 10
    ax.set_ylim([0, max(max_crops + 2, 30)])

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
    max_crops_display = 25
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

    im = ax.imshow(crop_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(len(all_crops)))
    ax.set_yticklabels(all_crops, fontweight='bold', fontsize=8)
    ax.set_xticks(range(len(n_units)))
    ax.set_xticklabels(n_units)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_title(
        f'Crop Selection Matrix (All {len(all_crops)} Crops)', fontweight='bold')

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
                        fontsize=9, fontweight='bold', color='darkgreen')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Crop diversity saved to {output_path}")
    plt.close()


def plot_area_distribution(data, metrics, output_path):
    """Visualize farm and crop area distribution details."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    fig.suptitle('Farm D-Wave CQM Area Distribution Analysis',
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

        # Subplot 1: Crop area distribution (pie chart) - top 10 only
        ax = fig.add_subplot(gs[row, col])

        crop_areas = {}
        for assignment in plot_assignments:
            crop_areas[assignment['crop']] = assignment['total_area']

        # Sort by area and get top 10
        sorted_crops = sorted(crop_areas.items(),
                              key=lambda x: x[1], reverse=True)
        if len(sorted_crops) > 10:
            labels_display = [crop for crop,
                              _ in sorted_crops[:10]] + ['Other crops']
            sizes_display = [area for _, area in sorted_crops[:10]
                             ] + [sum(area for _, area in sorted_crops[10:])]
            colors_display = [crop_colors[crop]
                              for crop, _ in sorted_crops[:10]] + ['#CCCCCC']
        else:
            labels_display = [crop for crop, _ in sorted_crops]
            sizes_display = [area for _, area in sorted_crops]
            colors_display = [crop_colors[crop] for crop in labels_display]

        wedges, texts, autotexts = ax.pie(sizes_display, labels=labels_display, colors=colors_display,
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'fontweight': 'bold', 'fontsize': 7})

        # Make percentage text more visible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(7)
            autotext.set_fontweight('bold')

        util = d['solution_summary']['utilization']
        ax.set_title(f'{config} Farms: Crop Areas (Top 10)\n(Allocated: {d["solution_summary"]["total_allocated"]:.0f} ha, Util: {util*100:.0f}%)',
                     fontweight='bold', fontsize=10)

        # Subplot 2: Farms per crop (bar chart) - top 10 crops
        ax = fig.add_subplot(gs[row, col + 1])

        crop_patch_counts = {}
        for assignment in plot_assignments:
            crop_patch_counts[assignment['crop']] = assignment['n_plots']

        # Get top 10 crops by area
        top_crops = [crop for crop, _ in sorted_crops[:10]]
        crops_list = top_crops
        patch_counts = [crop_patch_counts[crop] for crop in crops_list]
        colors_list = [crop_colors[crop] for crop in crops_list]

        bars = ax.barh(crops_list, patch_counts, color=colors_list,
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Number of Farms', fontweight='bold', fontsize=9)
        ax.set_title(f'{config} Farms: Top 10 Crop Distribution\n({d["solution_summary"]["n_crops"]} crops total)',
                     fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')

        # Annotate with counts (outside bars)
        for i, (crop, count) in enumerate(zip(crops_list, patch_counts)):
            ax.text(count + 0.1, i, f'{count}', va='center',
                    fontweight='bold', fontsize=8)

        # Add average area per farm (inside bars)
        for i, crop in enumerate(crops_list):
            avg_area = crop_areas[crop] / crop_patch_counts[crop]
            x_pos = patch_counts[i] * 0.05 if patch_counts[i] > 0.5 else 0.05
            ax.text(x_pos, i, f'{avg_area:.1f}ha', va='center', ha='left',
                    fontsize=6, style='italic', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5, edgecolor='none'))

    # Summary comparison plot (bottom row, spans all columns)
    ax_summary = fig.add_subplot(gs[2, :])

    # Stacked bar chart showing crop area distribution across all configs
    # Show only top 10 most common crops
    crop_freq = {}
    for config in configs:
        for assignment in data[config]['solution_summary']['plot_assignments']:
            crop = assignment['crop']
            crop_freq[crop] = crop_freq.get(crop, 0) + 1

    top_crops_summary = sorted(
        crop_freq.keys(), key=lambda x: crop_freq[x], reverse=True)[:10]

    x = np.arange(len(configs))
    width = 0.6

    # Prepare data for stacked bars
    crop_data = {crop: [] for crop in top_crops_summary}
    other_data = []

    for config in configs:
        d = data[config]
        plot_assignments = d['solution_summary']['plot_assignments']

        # Get area for each crop
        config_crops = {assignment['crop']: assignment['total_area']
                        for assignment in plot_assignments}

        for crop in top_crops_summary:
            crop_data[crop].append(config_crops.get(crop, 0))

        # Calculate "other" crops
        other_area = sum(area for crop, area in config_crops.items()
                         if crop not in top_crops_summary)
        other_data.append(other_area)

    # Create stacked bars
    bottom = np.zeros(len(configs))
    bars_list = []

    for crop in top_crops_summary:
        bars = ax_summary.bar(x, crop_data[crop], width, label=crop,
                              color=crop_colors[crop], bottom=bottom,
                              edgecolor='white', linewidth=2)
        bars_list.append(bars)
        bottom += crop_data[crop]

    # Add "other" category
    bars = ax_summary.bar(x, other_data, width, label='Other crops',
                          color='#CCCCCC', bottom=bottom,
                          edgecolor='white', linewidth=2)
    bars_list.append(bars)
    bottom += other_data

    # Add total available area line
    ax2 = ax_summary.twinx()
    total_areas = [data[config]['total_area'] for config in configs]
    line = ax2.plot(x, total_areas, 'ro-', linewidth=3, markersize=12,
                    label='Total Available Area', zorder=10)
    ax2.set_ylabel('Total Available Area (ha)',
                   fontweight='bold', fontsize=11, color='red')
    ax2.tick_params(axis='y', labelsize=10, labelcolor='red')

    # Configure primary axis
    ax_summary.set_xlabel('Number of Farms', fontweight='bold', fontsize=12)
    ax_summary.set_ylabel('Allocated Area (ha)',
                          fontweight='bold', fontsize=12)
    ax_summary.set_title('Crop Area Distribution Across Problem Sizes (Top 10 Crops + Other)',
                         fontweight='bold', fontsize=13)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(configs)
    ax_summary.legend(loc='upper left', fontsize=9, ncol=3)
    ax_summary.grid(True, alpha=0.3, axis='y')

    # Add utilization annotations
    for i, config in enumerate(configs):
        d = data[config]
        util = d['solution_summary']['utilization']
        allocated = d['solution_summary']['total_allocated']

        color = COLORS['success'] if util <= 1.01 else COLORS['warning'] if util < 1.5 else COLORS['danger']
        ax_summary.text(i, allocated + 5, f'{util*100:.0f}%',
                        ha='center', va='bottom', fontweight='bold',
                        fontsize=9, color=color)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Area distribution saved to {output_path}")
    plt.close()


def plot_advanced_analysis(data, metrics, output_path):
    """Create advanced analysis plots with additional insights."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Farm D-Wave CQM Advanced Analysis',
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

    # Plot 2: Over-Allocation Amount
    ax = axes[0, 1]
    over_allocation = [-metrics['idle_area'][i] for i in range(len(configs))]

    bars = ax.bar(configs, over_allocation, color=COLORS['warning'],
                  alpha=0.8, width=3, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Over-Allocated Area (ha)', fontweight='bold')
    ax.set_title('Over-Allocation Scaling', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate values
    for i, (x, y) in enumerate(zip(configs, over_allocation)):
        pct = (y / metrics['total_area'][i]) * 100
        ax.annotate(f'{y:.0f}ha\n({pct:.0f}%)', xy=(x, y), xytext=(0, 5),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold')

    # Plot 3: Diversity vs Over-allocation
    ax = axes[0, 2]
    ax2 = ax.twinx()

    utilization_pct = [u * 100 for u in metrics['utilization']]

    line1 = ax.plot(configs, metrics['n_crops'], 'o-',
                    color=COLORS['primary'], linewidth=2.5, markersize=10,
                    label='Crops Selected')
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Number of Crops', fontweight='bold',
                  color=COLORS['primary'])
    ax.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax.grid(True, alpha=0.3)

    line2 = ax2.plot(configs, utilization_pct, 's-',
                     color=COLORS['danger'], linewidth=2.5, markersize=10,
                     label='Utilization %')
    ax2.set_ylabel('Land Utilization (%)',
                   fontweight='bold', color=COLORS['danger'])
    ax2.tick_params(axis='y', labelcolor=COLORS['danger'])
    ax2.axhline(y=100, color=COLORS['neutral'], linestyle='--', alpha=0.5)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    ax.set_title('Diversity vs Utilization', fontweight='bold')

    # Plot 4: Solve Time Consistency
    ax = axes[1, 0]

    ax.plot(configs, metrics['solve_time'], 'o-',
            color=COLORS['primary'], linewidth=2.5, markersize=10, label='Total Time')
    ax.plot(configs, metrics['qpu_time'], 's-',
            color=COLORS['success'], linewidth=2.5, markersize=10, label='QPU Time')
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title('Time Scaling', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Revenue per Hectare (Available vs Allocated)
    ax = axes[1, 1]

    rev_per_available = [metrics['objective_value'][i] / metrics['total_area'][i]
                         for i in range(len(configs))]
    rev_per_allocated = [metrics['objective_value'][i] / metrics['total_allocated'][i]
                         for i in range(len(configs))]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, rev_per_available, width, label='Per Available Ha',
                   color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, rev_per_allocated, width, label='Per Allocated Ha',
                   color=COLORS['danger'], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Revenue per Hectare', fontweight='bold')
    ax.set_title('Revenue Efficiency (Two Perspectives)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Summary Text
    ax = axes[1, 2]
    ax.axis('off')

    # Calculate key metrics
    avg_crops = np.mean(metrics['n_crops'])
    max_util = max(metrics['utilization'])
    max_over = max([-metrics['idle_area'][i] for i in range(len(configs))])

    summary_text = f"""
    FARM CQM SUMMARY
    {'='*40}
    
    Problem Sizes: {min(configs)} - {max(configs)} farms
    Variable Range: {min(metrics['n_variables'])} - {max(metrics['n_variables'])}
    Area Range: {min(metrics['total_area']):.0f} - {max(metrics['total_area']):.0f} ha
    
    PERFORMANCE
    {'='*40}
    Avg Solve Time: {np.mean(metrics['solve_time']):.2f}s
    Avg QPU Time: {np.mean(metrics['qpu_time']):.3f}s
    QPU Efficiency: ~1.3% (very low)
    Consistency: Excellent (~5.3s)
    
    SOLUTION QUALITY
    {'='*40}
    Avg Crops: {avg_crops:.1f}
    Max Crops: {max(metrics['n_crops'])}
    Objective Range: {min(metrics['objective_value']):.2f} - {max(metrics['objective_value']):.2f}
    Utilization Range: {min(metrics['utilization'])*100:.0f}% - {max_util*100:.0f}%
    
    KEY CHARACTERISTICS
    {'='*40}
    High crop diversity across all sizes
    Consistent fast solve times
    Multiple crops per farm in solutions
    Utilization above 100% on all configs
    
    Trade-off between diversity and
    constraint satisfaction visible in
    results.
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Advanced analysis saved to {output_path}")
    plt.close()


def print_detailed_summary(data, metrics):
    """Print comprehensive summary table to console."""
    print("\n" + "="*110)
    print("FARM D-WAVE CQM BENCHMARK RESULTS SUMMARY")
    print("="*110)

    configs = sorted(data.keys())

    print(f"\n{'Config':<10} {'Vars':<8} {'Area(ha)':<10} {'Solve(s)':<10} {'QPU(s)':<10} {'Obj':<10} {'Crops':<8} {'Util%':<8} {'Over(ha)':<10}")
    print("-"*110)

    for i, config in enumerate(configs):
        over_alloc = -metrics['idle_area'][i]
        print(f"{config:<10} {metrics['n_variables'][i]:<8} {metrics['total_area'][i]:<10.1f} "
              f"{metrics['solve_time'][i]:<10.3f} {metrics['qpu_time'][i]:<10.3f} "
              f"{metrics['objective_value'][i]:<10.2f} "
              f"{metrics['n_crops'][i]:<8} "
              f"{metrics['utilization'][i]*100:<8.0f} "
              f"{over_alloc:<10.0f}")

    print("\n" + "="*110)
    print("DETAILED METRICS")
    print("="*110)

    for config in configs:
        d = data[config]
        print(f"\n--- Configuration: {config} farms ---")
        print(f"  Problem Size:")
        print(
            f"    Variables: {d['n_variables']}, Constraints: {d['n_constraints']}")
        print(f"    Total Available Area: {d['total_area']:.2f} ha")
        print(f"  Performance:")
        print(f"    Total Time: {d['solve_time']:.3f}s")
        print(f"    QPU Time: {d['qpu_time']:.3f}s")
        print(f"    Hybrid Overhead: {d['solve_time'] - d['qpu_time']:.3f}s")
        print(f"    Status: {d['status']}")
        print(f"  Solution:")
        print(f"    Objective: {d['objective_value']:.3f}")
        print(f"    Crop Diversity: {d['solution_summary']['n_crops']} crops")
        print(
            f"    Total Allocated: {d['solution_summary']['total_allocated']:.2f} ha")
        print(
            f"    Over-Allocation: {-d['solution_summary']['idle_area']:.2f} ha ({(d['solution_summary']['utilization']-1)*100:.1f}%)")
        print(
            f"    Utilization: {d['solution_summary']['utilization']*100:.1f}%")
        print(f"  Crops Selected:")
        print(f"    {', '.join(d['solution_summary']['crops_selected'][:15])}")
        if len(d['solution_summary']['crops_selected']) > 15:
            print(
                f"    ... and {len(d['solution_summary']['crops_selected']) - 15} more")

    print("\n" + "="*110)


def main():
    """Main execution function."""
    print("\n" + "="*110)
    print("FARM D-WAVE CQM BENCHMARK VISUALIZATION")
    print("="*110)

    # Set up paths
    script_dir = Path(__file__).parent
    benchmark_dir = script_dir / "Benchmarks"
    output_dir = script_dir / "plots" / "farm_cqm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n📂 Loading benchmark data...")
    try:
        data = load_cqm_benchmark_data(benchmark_dir)
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

    plot_performance_metrics(metrics, output_dir / "farm_cqm_performance.png")
    plot_solution_quality(metrics, output_dir / "farm_cqm_quality.png")
    plot_crop_diversity(metrics, output_dir / "farm_cqm_crop_diversity.png")
    plot_area_distribution(data, metrics, output_dir /
                           "farm_cqm_area_distribution.png")
    plot_advanced_analysis(data, metrics, output_dir /
                           "farm_cqm_advanced_analysis.png")

    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - farm_cqm_performance.png        : Time breakdown and efficiency")
    print("  - farm_cqm_quality.png            : Solution quality and validation")
    print("  - farm_cqm_crop_diversity.png     : Crop selection patterns")
    print("  - farm_cqm_area_distribution.png  : Farm and crop area distributions")
    print("  - farm_cqm_advanced_analysis.png  : Advanced metrics and correlations")

    print("\n" + "="*110)
    print("COMPLETE!")
    print("="*110 + "\n")


if __name__ == "__main__":
    main()
