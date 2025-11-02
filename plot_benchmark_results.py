#!/usr/bin/env python3
"""
Professional Benchmark Visualization System
Creates publication-quality plots with consistent styling, LaTeX integration,
and comprehensive solver performance analysis.

Features:
- Consistent color palettes and marker styles
- LaTeX font rendering for professional typography
- Modular design with custom matplotlib style
- Comprehensive performance and solution analysis
- Cross-solver comparison capabilities
- Pie charts for solution composition analysis
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import json
from typing import Dict, List, Any, Optional, Tuple

# =============================================================================
# CUSTOM STYLE DEFINITION
# =============================================================================

def configure_professional_style():
    """Configure professional matplotlib style with LaTeX integration."""
    
    # LaTeX configuration for publication-quality text
    rcParams.update({
        # Font configuration
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'serif'],
        'font.size': 11,
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'serif',
        
        # Text rendering with LaTeX (disabled for now)
        'text.usetex': True,
        'text.latex.preamble': r'''
            \usepackage{amsmath}
            \usepackage{amsfonts}
            \usepackage{amssymb}
            \usepackage{bm}
            \usepackage{xcolor}
        ''',
        
        # Figure layout
        'figure.figsize': (8, 6),
        'figure.dpi': 300,
        'figure.constrained_layout.use': True,
        
        # Axes configuration
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.grid.which': 'major',
        'axes.grid.axis': 'both',
        'axes.labelpad': 8,
        'axes.titlepad': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        
        # Tick configuration
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        
        # Grid configuration
        'grid.linewidth': 0.8,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        
        # Legend configuration
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'legend.shadow': False,
        
        # Lines configuration
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'lines.markeredgewidth': 1.2,
        
        # Savefig configuration
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.transparent': False,
        
        # Errorbar configuration
        'errorbar.capsize': 3,
    })

# =============================================================================
# COLOR AND STYLE DEFINITIONS
# =============================================================================

class ColorPalette:
    """Professional color palette for consistent visualization."""
    
    # Primary solver colors (distinct and colorblind-friendly)
    SOLVER_COLORS = {
        'Patch_PuLP': '#1f77b4',      # Blue
        'Patch_GurobiQUBO': '#ff7f0e', # Orange
        'Patch_DWave': '#2ca02c',      # Green
        'Patch_DWaveBQM': '#d62728',   # Red
        'Farm_PuLP': '#9467bd',        # Purple
        'Farm_DWave': '#8c564b',       # Brown
    }
    
    # Semantic colors
    SEMANTIC = {
        'success': '#2e8b57',          # Sea Green
        'warning': '#ffa500',          # Orange
        'error': '#dc143c',            # Crimson
        'neutral': '#6c757d',          # Gray
        'info': '#17a2b8',             # Teal
    }
    
    # Qualitative palette for crops/items
    QUALITATIVE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    # Pie chart specific colors
    PIE_COLORS = [
        '#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', 
        '#64b5cd', '#da8bc3', '#8c8c8c', '#b8ca3f', '#5ab6e8'
    ]
    
    @classmethod
    def get_solver_color(cls, solver_name: str) -> str:
        """Get color for specific solver."""
        return cls.SOLVER_COLORS.get(solver_name, cls.SEMANTIC['neutral'])
    
    @classmethod
    def get_marker_style(cls, solver_name: str) -> Dict[str, Any]:
        """Get consistent marker style for solver."""
        marker_map = {
            'Patch_PuLP': 'o',
            'Patch_GurobiQUBO': 's',
            'Patch_DWave': '^',
            'Patch_DWaveBQM': 'D',
            'Farm_PuLP': 'v',
            'Farm_DWave': '<',
        }
        return {
            'marker': marker_map.get(solver_name, 'o'),
            'markersize': 6,
            'markeredgecolor': 'white',
            'markeredgewidth': 1.0,
        }

# =============================================================================
# SOLVER CONFIGURATION
# =============================================================================

SOLVER_CONFIGS = {
    'Patch_PuLP': {
        'dir_name': 'Patch_PuLP',
        'display_name': r'$\text{Patch PuLP}$',
        'problem_type': 'Patch',
        'unit_label': 'Patches',
        'has_qpu_time': False,
        'has_solver_time': True,
        'has_bqm_conversion': False,
        'has_bqm_energy': False,
        'has_constraints': True,
        'has_quadratic': False,
        'has_validation': True,
        'has_status': True,
        'has_success': True,
        'time_limit': None,
    },
    'Patch_GurobiQUBO': {
        'dir_name': 'Patch_GurobiQUBO',
        'display_name': r'$\text{Patch Gurobi QUBO}$',
        'problem_type': 'Patch',
        'unit_label': 'Patches',
        'has_qpu_time': False,
        'has_solver_time': True,
        'has_bqm_conversion': True,
        'has_bqm_energy': True,
        'has_constraints': False,
        'has_quadratic': True,
        'has_validation': True,
        'has_status': True,
        'has_success': True,
        'time_limit': 300,
    },
    'Patch_DWave': {
        'dir_name': 'Patch_DWave',
        'display_name': r'$\text{Patch D-Wave CQM}$',
        'problem_type': 'Patch',
        'unit_label': 'Patches',
        'has_qpu_time': True,
        'has_solver_time': False,
        'has_bqm_conversion': False,
        'has_bqm_energy': False,
        'has_constraints': True,
        'has_quadratic': False,
        'has_validation': True,
        'has_status': True,
        'has_success': True,
        'time_limit': None,
    },
    'Patch_DWaveBQM': {
        'dir_name': 'Patch_DWaveBQM',
        'display_name': r'$\text{Patch D-Wave BQM}$',
        'problem_type': 'Patch',
        'unit_label': 'Patches',
        'has_qpu_time': True,
        'has_solver_time': False,
        'has_bqm_conversion': True,
        'has_bqm_energy': True,
        'has_constraints': False,
        'has_quadratic': True,
        'has_validation': True,
        'has_status': False,
        'has_success': False,
        'time_limit': None,
    },
    'Farm_PuLP': {
        'dir_name': 'Farm_PuLP',
        'display_name': r'$\text{Farm PuLP}$',
        'problem_type': 'Farm',
        'unit_label': 'Farms',
        'has_qpu_time': False,
        'has_solver_time': True,
        'has_bqm_conversion': False,
        'has_bqm_energy': False,
        'has_constraints': True,
        'has_quadratic': False,
        'has_validation': False,
        'has_status': True,
        'has_success': True,
        'time_limit': None,
    },
    'Farm_DWave': {
        'dir_name': 'Farm_DWave',
        'display_name': r'$\text{Farm D-Wave CQM}$',
        'problem_type': 'Farm',
        'unit_label': 'Farms',
        'has_qpu_time': True,
        'has_solver_time': False,
        'has_bqm_conversion': False,
        'has_bqm_energy': False,
        'has_constraints': True,
        'has_quadratic': False,
        'has_validation': False,
        'has_status': True,
        'has_success': True,
        'time_limit': None,
    },
}

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

def load_benchmark_data(benchmark_dir: Path, solver_key: str) -> Dict[int, Dict]:
    """Load benchmark data for a specific solver with error handling."""
    config = SOLVER_CONFIGS[solver_key]
    solver_dir = benchmark_dir / "COMPREHENSIVE" / config['dir_name']

    if not solver_dir.exists():
        raise FileNotFoundError(f"Solver directory not found: {solver_dir}")

    data = {}
    config_files = list(solver_dir.glob("config_*_run_*.json"))

    if not config_files:
        raise FileNotFoundError(f"No configuration files found in {solver_dir}")

    for config_file in sorted(config_files):
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            n_units = config_data['n_units']
            data[n_units] = config_data

    return data

def extract_metrics(data: Dict[int, Dict], solver_key: str) -> Dict[str, List]:
    """Extract comprehensive metrics from benchmark data."""
    config = SOLVER_CONFIGS[solver_key]
    configs = sorted(data.keys())

    metrics = {
        'n_units': configs,
        'n_variables': [],
        'total_area': [],
        'solve_time': [],
        'objective_value': [],
        'n_crops': [],
        'utilization': [],
        'crops_selected': [],
        'total_allocated': [],
        'idle_area': []
    }

    # Add conditional metrics
    conditional_metrics = {
        'has_constraints': 'n_constraints',
        'has_quadratic': 'n_quadratic',
        'has_qpu_time': ['qpu_time', 'hybrid_time'],
        'has_solver_time': 'solver_time',
        'has_bqm_conversion': 'bqm_conversion_time',
        'has_bqm_energy': 'bqm_energy',
        'has_validation': ['is_feasible', 'n_violations', 'pass_rate', 'total_checks'],
        'has_status': 'status',
        'has_success': 'success'
    }

    for metric_key, metric_names in conditional_metrics.items():
        if config[metric_key]:
            if isinstance(metric_names, list):
                for name in metric_names:
                    metrics[name] = []
            else:
                metrics[metric_names] = []

    for conf in configs:
        d = data[conf]

        # Core metrics
        metrics['n_variables'].append(d['n_variables'])
        metrics['total_area'].append(d['total_area'])
        metrics['solve_time'].append(d['solve_time'])
        metrics['objective_value'].append(d['objective_value'])
        
        # Solution summary
        if 'solution_summary' in d:
            summary = d['solution_summary']
            metrics['n_crops'].append(summary['n_crops'])
            metrics['utilization'].append(summary['utilization'])
            metrics['crops_selected'].append(summary['crops_selected'])
            metrics['total_allocated'].append(summary['total_allocated'])
            metrics['idle_area'].append(summary['idle_area'])
        else:
            # Farm problems fallback
            metrics['n_crops'].append(0)
            metrics['utilization'].append(1.0)
            metrics['crops_selected'].append([])
            metrics['total_allocated'].append(d['total_area'])
            metrics['idle_area'].append(0.0)

        # Conditional metrics
        if config['has_constraints']:
            metrics['n_constraints'].append(d['n_constraints'])
        if config['has_quadratic']:
            metrics['n_quadratic'].append(d['n_quadratic'])
        if config['has_qpu_time']:
            metrics['qpu_time'].append(d['qpu_time'])
            metrics['hybrid_time'].append(d.get('hybrid_time', 0))
        if config['has_solver_time']:
            metrics['solver_time'].append(d.get('solver_time', d['solve_time']))
        if config['has_bqm_conversion']:
            metrics['bqm_conversion_time'].append(d.get('bqm_conversion_time', 0))
        if config['has_bqm_energy']:
            metrics['bqm_energy'].append(d.get('bqm_energy', 0))
        if config['has_validation']:
            validation = d['validation']
            metrics['is_feasible'].append(validation['is_feasible'])
            metrics['n_violations'].append(validation['n_violations'])
            metrics['pass_rate'].append(validation['summary']['pass_rate'])
            metrics['total_checks'].append(validation['summary']['total_checks'])
        if config['has_status']:
            metrics['status'].append(d['status'])
        if config['has_success']:
            metrics['success'].append(d['success'])

    return metrics

def extract_solution_composition(data: Dict[int, Dict], solver_key: str) -> Dict[int, Dict[str, float]]:
    """Extract solution composition for pie charts."""
    composition = {}

    for n_units, config_data in data.items():
        crop_areas = {}
        if 'solution_summary' in config_data:
            plot_assignments = config_data['solution_summary']['plot_assignments']
            for assignment in plot_assignments:
                crop = assignment['crop']
                area = assignment.get('area', assignment.get('total_area', 0))
                crop_areas[crop] = crop_areas.get(crop, 0) + area
        elif 'solution' in config_data and 'land_allocations' in config_data['solution']:
            # Fallback for Farm_PuLP (old format)
            land_allocations = config_data['solution']['land_allocations']
            for allocation in land_allocations:
                crop = allocation['crop']
                area = allocation['area']
                crop_areas[crop] = crop_areas.get(crop, 0) + area
        elif 'solution_areas' in config_data:
            # Farm_PuLP format: solution_areas has "Farm#_FoodName": area
            solution_areas = config_data['solution_areas']
            for farm_food, area in solution_areas.items():
                # Extract food name from "Farm#_FoodName"
                if '_' in farm_food:
                    parts = farm_food.split('_', 1)
                    if len(parts) == 2:
                        crop = parts[1]
                        # Include all non-zero allocations
                        if area > 0:
                            crop_areas[crop] = crop_areas.get(crop, 0) + area

        # Calculate percentages
        total_area = sum(crop_areas.values())
        if total_area > 0:
            crop_percentages = {crop: (area / total_area) * 100 for crop, area in crop_areas.items()}
            composition[n_units] = crop_percentages

    return composition

# =============================================================================
# PIE CHART VISUALIZATIONS
# =============================================================================

def plot_solution_composition_pie_charts(all_compositions: Dict[str, Dict], output_path: Path):
    """Create comprehensive pie charts showing solution composition across solvers."""
    solvers = list(all_compositions.keys())
    
    # Find common problem sizes across solvers
    common_sizes = set()
    for solver_compositions in all_compositions.values():
        common_sizes.update(solver_compositions.keys())
    common_sizes = sorted(common_sizes)
    
    if not common_sizes:
        print("⚠ No common problem sizes with solution data for pie charts")
        return
    
    # Limit to 4 most representative sizes for clarity
    display_sizes = common_sizes[:4] if len(common_sizes) > 4 else common_sizes
    
    fig, axes = plt.subplots(len(display_sizes), len(solvers), 
                            figsize=(4 * len(solvers), 4 * len(display_sizes)))
    fig.suptitle(r'\textbf{Solution Composition Analysis}', fontsize=16, y=0.98)
    
    # Ensure axes is 2D array for consistent indexing
    if len(display_sizes) == 1:
        axes = axes.reshape(1, -1)
    if len(solvers) == 1:
        axes = axes.reshape(-1, 1)
    
    # Create unified color mapping for crops across all plots
    all_crops = set()
    for solver_compositions in all_compositions.values():
        for composition in solver_compositions.values():
            all_crops.update(composition.keys())
    all_crops = sorted(list(all_crops))
    
    crop_colors = {crop: ColorPalette.QUALITATIVE[i % len(ColorPalette.QUALITATIVE)] 
                  for i, crop in enumerate(all_crops)}
    
    for col_idx, solver in enumerate(solvers):
        compositions = all_compositions[solver]
        config = SOLVER_CONFIGS[solver]
        
        for row_idx, problem_size in enumerate(display_sizes):
            if problem_size not in compositions:
                axes[row_idx, col_idx].text(0.5, 0.5, 'No Data', 
                                          ha='center', va='center', 
                                          transform=axes[row_idx, col_idx].transAxes)
                axes[row_idx, col_idx].set_title(f'{problem_size} {config["unit_label"]}')
                continue
            
            composition = compositions[problem_size]
            
            # Prepare data for pie chart
            labels = []
            sizes = []
            colors = []
            
            # Sort by percentage (descending) and take top 8, group rest as "Other"
            sorted_items = sorted(composition.items(), key=lambda x: x[1], reverse=True)
            top_items = sorted_items[:8]
            other_percentage = sum(item[1] for item in sorted_items[8:])
            
            for crop, percentage in top_items:
                labels.append(f'{crop}\n({percentage:.1f}\%)')
                sizes.append(percentage)
                colors.append(crop_colors.get(crop, ColorPalette.SEMANTIC['neutral']))
            
            if other_percentage > 0.5:  # Only show "Other" if significant
                labels.append(f'Other\n({other_percentage:.1f}\%)')
                sizes.append(other_percentage)
                colors.append(ColorPalette.SEMANTIC['neutral'])
            
            # Create pie chart
            wedges, texts, autotexts = axes[row_idx, col_idx].pie(
                sizes, labels=labels, colors=colors, autopct='', startangle=90,
                textprops={'fontsize': 8, 'ha': 'center'}
            )
            
            # Improve text appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(7)
            
            # Set title for first row only
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(
                    f'{config["display_name"]}\n{problem_size} {config["unit_label"]}',
                    fontsize=10, pad=20
                )
            else:
                axes[row_idx, col_idx].set_title(
                    f'{problem_size} {config["unit_label"]}',
                    fontsize=10, pad=20
                )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=3000, bbox_inches='tight')
    plt.close()
    print(f"✓ Solution composition pie charts saved to {output_path}")

def plot_individual_solver_pie_charts(data: Dict[int, Dict], solver_key: str, output_path: Path):
    """Create detailed pie chart analysis for individual solver."""
    config = SOLVER_CONFIGS[solver_key]
    color = ColorPalette.get_solver_color(solver_key)
    
    # Extract solution compositions
    compositions = extract_solution_composition(data, solver_key)
    
    if not compositions:
        print(f"⚠ No solution composition data available for {solver_key}")
        return
    
    # Select representative problem sizes (small, medium, large)
    sizes = sorted(compositions.keys())
    if len(sizes) >= 3:
        display_sizes = [sizes[0], sizes[len(sizes)//2], sizes[-1]]
    else:
        display_sizes = sizes
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(rf'\textbf{{{config["display_name"]} Solution Composition}}', 
                 fontsize=16, y=0.95)
    
    # Create unified color mapping
    all_crops = set()
    for composition in compositions.values():
        all_crops.update(composition.keys())
    all_crops = sorted(list(all_crops))
    crop_colors = {crop: ColorPalette.QUALITATIVE[i % len(ColorPalette.QUALITATIVE)] 
                  for i, crop in enumerate(all_crops)}
    
    # Plot 1-3: Pie charts for different problem sizes
    for idx, problem_size in enumerate(display_sizes[:3]):
        ax = axes[idx // 2, idx % 2]
        composition = compositions[problem_size]
        
        # Prepare data
        labels = []
        sizes = []
        colors = []
        
        sorted_items = sorted(composition.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:6]  # Top 6 for clarity
        other_percentage = sum(item[1] for item in sorted_items[6:])
        
        for crop, percentage in top_items:
            labels.append(f'{crop}\n({percentage:.1f}\%)')
            sizes.append(percentage)
            colors.append(crop_colors.get(crop, ColorPalette.SEMANTIC['neutral']))
        
        if other_percentage > 1.0:
            labels.append(f'Other\n({other_percentage:.1f}\%)')
            sizes.append(other_percentage)
            colors.append(ColorPalette.SEMANTIC['neutral'])
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='', startangle=90,
            textprops={'fontsize': 9, 'ha': 'center'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(rf'$\textbf{{{problem_size} {config["unit_label"]}}}$', fontsize=12)
    
    # Plot 4: Crop distribution evolution
    ax = axes[1, 1] if len(display_sizes) == 3 else axes[1, 0]
    
    # Prepare data for stacked area chart
    all_sizes = sorted(compositions.keys())
    top_crops = set()
    
    # Identify top crops across all sizes
    crop_totals = {}
    for composition in compositions.values():
        for crop, percentage in composition.items():
            crop_totals[crop] = crop_totals.get(crop, 0) + percentage
    
    top_crops_list = sorted(crop_totals.items(), key=lambda x: x[1], reverse=True)[:8]
    top_crops = [crop for crop, _ in top_crops_list]
    
    # Create stacked area data
    x = all_sizes
    y_stacks = {crop: [] for crop in top_crops}
    other_stack = []
    
    for size in all_sizes:
        composition = compositions[size]
        remaining = 100.0
        
        for crop in top_crops:
            percentage = composition.get(crop, 0)
            y_stacks[crop].append(percentage)
            remaining -= percentage
        
        other_stack.append(remaining)
    
    # Create stacked area plot
    bottom = np.zeros(len(all_sizes))
    for crop in top_crops:
        ax.fill_between(x, bottom, bottom + y_stacks[crop], 
                       label=crop, alpha=0.7,
                       color=crop_colors.get(crop, ColorPalette.SEMANTIC['neutral']))
        bottom += y_stacks[crop]
    
    if any(other_stack):
        ax.fill_between(x, bottom, bottom + other_stack, 
                       label='Other', alpha=0.7, color=ColorPalette.SEMANTIC['neutral'])
    
    ax.set_xlabel(rf'$\text{{Number of {config["unit_label"]}}}$')
    ax.set_ylabel(r'$\text{Area Distribution (\%)}$')
    ax.set_title(r'$\textbf{Crop Distribution Evolution}$')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=3000, bbox_inches='tight')
    plt.close()
    print(f"✓ Individual solver pie charts saved to {output_path}")

def plot_constraint_satisfaction_pie(all_metrics: Dict[str, Dict], output_path: Path):
    """Create pie charts showing constraint satisfaction rates."""
    validation_solvers = [s for s in all_metrics.keys() 
                         if SOLVER_CONFIGS[s]['has_validation']]
    
    if not validation_solvers:
        print("⚠ No solvers with constraint validation data for pie charts")
        return
    
    n_solvers = len(validation_solvers)
    fig, axes = plt.subplots(1, n_solvers, figsize=(4 * n_solvers, 4))
    fig.suptitle(r'\textbf{Constraint Satisfaction Analysis}', fontsize=16, y=1.0)
    
    if n_solvers == 1:
        axes = [axes]
    
    for idx, solver in enumerate(validation_solvers):
        metrics = all_metrics[solver]
        config = SOLVER_CONFIGS[solver]
        
        # Calculate average pass rate and violation rate with validation
        pass_rates = [pr for pr in metrics['pass_rate'] if pr >= 0 and pr <= 1]
        
        if not pass_rates:
            # Skip this solver if no valid data
            axes[idx].text(0.5, 0.5, 'No valid\nconstraint data', 
                          ha='center', va='center', fontsize=12)
            axes[idx].set_title(f'{config["display_name"]}\nAverage Constraint Satisfaction', 
                               fontsize=12, pad=20)
            continue
        
        avg_pass_rate = np.mean(pass_rates) * 100
        avg_violation_rate = 100 - avg_pass_rate
        
        # Ensure non-negative values
        avg_pass_rate = max(0, min(100, avg_pass_rate))
        avg_violation_rate = max(0, min(100, avg_violation_rate))
        
        # Prepare data for pie chart
        sizes = [avg_pass_rate, avg_violation_rate]
        labels = [f'Passed\n{avg_pass_rate:.1f}%', f'Violations\n{avg_violation_rate:.1f}%']
        colors = [ColorPalette.SEMANTIC['success'], ColorPalette.SEMANTIC['error']]
        explode = (0.1, 0)  # explode the passed slice
        
        wedges, texts, autotexts = axes[idx].pie(
            sizes, explode=explode, labels=labels, colors=colors, 
            autopct='', startangle=90, shadow=True,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        axes[idx].set_title(f'{config["display_name"]}\nAverage Constraint Satisfaction', 
                           fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=3000, bbox_inches='tight')
    plt.close()
    print(f"✓ Constraint satisfaction pie charts saved to {output_path}")

def plot_land_utilization_pie(all_metrics: Dict[str, Dict], output_path: Path):
    """Create pie charts showing land utilization efficiency."""
    solvers = list(all_metrics.keys())
    
    fig, axes = plt.subplots(1, len(solvers), figsize=(4 * len(solvers), 4))
    fig.suptitle(r'\textbf{Land Utilization Efficiency}', fontsize=16, y=1.0)
    
    if len(solvers) == 1:
        axes = [axes]
    
    for idx, solver in enumerate(solvers):
        metrics = all_metrics[solver]
        config = SOLVER_CONFIGS[solver]
        
        # Calculate average utilization with validation
        utilization_values = [u for u in metrics['utilization'] if u >= 0 and u <= 1]
        
        if not utilization_values:
            # Skip this solver if no valid utilization data
            axes[idx].text(0.5, 0.5, 'No valid\nutilization data', 
                          ha='center', va='center', fontsize=12)
            axes[idx].set_title(f'{config["display_name"]}\nLand Utilization', 
                               fontsize=12, pad=20)
            continue
        
        avg_utilization = np.mean(utilization_values) * 100
        idle_percentage = 100 - avg_utilization
        
        # Ensure non-negative values
        avg_utilization = max(0, min(100, avg_utilization))
        idle_percentage = max(0, min(100, idle_percentage))
        
        # Prepare data for pie chart
        sizes = [avg_utilization, idle_percentage]
        labels = [f'Utilized\n{avg_utilization:.1f}%', f'Idle\n{idle_percentage:.1f}%']
        colors = [ColorPalette.SEMANTIC['success'], ColorPalette.SEMANTIC['neutral']]
        explode = (0.1, 0)  # explode the utilized slice
        
        wedges, texts, autotexts = axes[idx].pie(
            sizes, explode=explode, labels=labels, colors=colors, 
            autopct='', startangle=90, shadow=False,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        axes[idx].set_title(f'{config["display_name"]}\nLand Utilization', 
                           fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=3000, bbox_inches='tight')
    plt.close()
    print(f"✓ Land utilization pie charts saved to {output_path}")

# =============================================================================
# PERFORMANCE PLOTTING FUNCTIONS
# =============================================================================

def plot_performance_comparison(all_metrics: Dict[str, Dict], output_path: Path):
    """Create comprehensive performance comparison across solvers."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(r'\textbf{Solver Performance Comparison}', fontsize=16, y=0.98)
    
    solvers = list(all_metrics.keys())
    
    # Plot 1: Solve Time Comparison
    ax = axes[0, 0]
    for solver in solvers:
        metrics = all_metrics[solver]
        color = ColorPalette.get_solver_color(solver)
        marker_style = ColorPalette.get_marker_style(solver)
        
        # Multiply problem size by 27 (number of foods)
        problem_size = [n * 27 for n in metrics['n_units']]
        ax.plot(problem_size, metrics['solve_time'], 
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.set_xlabel(r'$\text{Problem Size (Units} \times \text{Foods)}$')
    ax.set_ylabel(r'$\text{Solve Time (s)}$')
    ax.set_title(r'$\textbf{Solve Time Scaling}$')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Objective Value Comparison
    ax = axes[0, 1]
    for solver in solvers:
        metrics = all_metrics[solver]
        color = ColorPalette.get_solver_color(solver)
        marker_style = ColorPalette.get_marker_style(solver)
        
        # Multiply problem size by 27 (number of foods)
        problem_size = [n * 27 for n in metrics['n_units']]
        ax.plot(problem_size, metrics['objective_value'],
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.set_xlabel(r'$\text{Problem Size (Units} \times \text{Foods)}$')
    ax.set_ylabel(r'$\text{Objective Value}$')
    ax.set_title(r'$\textbf{Solution Quality}$')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Time to Solution Quality
    ax = axes[1, 0]
    for solver in solvers:
        metrics = all_metrics[solver]
        color = ColorPalette.get_solver_color(solver)
        marker_style = ColorPalette.get_marker_style(solver)
        
        # Calculate time to solution quality: solve_time / objective_value
        # Lower is better (less time per unit of objective)
        problem_size = [n * 27 for n in metrics['n_units']]
        time_to_quality = [t / max(obj, 1e-6) for t, obj in 
                          zip(metrics['solve_time'], metrics['objective_value'])]
        ax.plot(problem_size, time_to_quality,
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.set_xlabel(r'$\text{Problem Size (Units} \times \text{Foods)}$')
    ax.set_ylabel(r'$\text{Time per Objective Unit (s)}$')
    ax.set_title(r'$\textbf{Time to Solution Quality}$')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency Comparison
    ax = axes[1, 1]
    for solver in solvers:
        metrics = all_metrics[solver]
        color = ColorPalette.get_solver_color(solver)
        marker_style = ColorPalette.get_marker_style(solver)
        
        problem_size = [n * 27 for n in metrics['n_units']]
        time_per_var = [t/v * 1000 for t, v in zip(metrics['solve_time'], metrics['n_variables'])]
        ax.plot(problem_size, time_per_var,
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.set_xlabel(r'$\text{Problem Size (Units} \times \text{Foods)}$')
    ax.set_ylabel(r'$\text{Time per Variable (ms)}$')
    ax.set_title(r'$\textbf{Solving Efficiency}$')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=3000, bbox_inches='tight')
    plt.close()
    print(f"✓ Performance comparison saved to {output_path}")

def plot_solution_quality_comparison(all_metrics: Dict[str, Dict], output_path: Path):
    """Create solution quality comparison across solvers."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(r'\textbf{Solution Quality Comparison}', fontsize=16, y=0.98)
    
    solvers = list(all_metrics.keys())
    
    # Plot 1: Land Utilization
    ax = axes[0, 0]
    for solver in solvers:
        metrics = all_metrics[solver]
        color = ColorPalette.get_solver_color(solver)
        marker_style = ColorPalette.get_marker_style(solver)
        
        problem_size = [n * 27 for n in metrics['n_units']]
        utilization_pct = [u * 100 for u in metrics['utilization']]
        ax.plot(problem_size, utilization_pct,
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.axhline(y=100, color=ColorPalette.SEMANTIC['neutral'], linestyle='--', alpha=0.7)
    ax.set_xlabel(r'$\text{Problem Size (Units} \times \text{Foods)}$')
    ax.set_ylabel(r'$\text{Land Utilization (\%)}$')
    ax.set_title(r'$\textbf{Resource Utilization}$')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 310)
    #ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Crop Diversity
    ax = axes[0, 1]
    for solver in solvers:
        metrics = all_metrics[solver]
        color = ColorPalette.get_solver_color(solver)
        marker_style = ColorPalette.get_marker_style(solver)
        
        problem_size = [n * 27 for n in metrics['n_units']]
        ax.plot(problem_size, metrics['n_crops'],
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.set_xlabel(r'$\text{Problem Size (Units} \times \text{Foods)}$')
    ax.set_ylabel(r'$\text{Number of Crops}$')
    ax.set_title(r'$\textbf{Crop Diversity}$')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Constraint Satisfaction (if available)
    ax = axes[1, 0]
    validation_solvers = [s for s in solvers if SOLVER_CONFIGS[s]['has_validation']]
    
    if validation_solvers:
        for solver in validation_solvers:
            metrics = all_metrics[solver]
            color = ColorPalette.get_solver_color(solver)
            marker_style = ColorPalette.get_marker_style(solver)
            
            problem_size = [n * 27 for n in metrics['n_units']]
            pass_rates = [p * 100 for p in metrics['pass_rate']]
            ax.plot(problem_size, pass_rates,
                    color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                    **marker_style)
        
        ax.axhline(y=100, color=ColorPalette.SEMANTIC['success'], linestyle='--', alpha=0.7)
        ax.set_xlabel(r'$\text{Problem Size (Units} \times \text{Foods)}$')
        ax.set_ylabel(r'$\text{Constraint Pass Rate (\%)}$')
        ax.set_title(r'$\textbf{Constraint Satisfaction}$')
        ax.legend(fontsize=9)
        ax.set_ylim(0, 110)
        #ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, r'$\text{No Validation Data}$', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(r'$\textbf{Constraint Satisfaction}$')
    
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Solution Value per Unit
    ax = axes[1, 1]
    for solver in solvers:
        metrics = all_metrics[solver]
        color = ColorPalette.get_solver_color(solver)
        marker_style = ColorPalette.get_marker_style(solver)
        
        problem_size = [n * 27 for n in metrics['n_units']]
        obj_per_unit = [obj/n for obj, n in zip(metrics['objective_value'], metrics['n_units'])]
        ax.plot(problem_size, obj_per_unit,
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.set_xlabel(r'$\text{Problem Size (Units} \times \text{Foods)}$')
    ax.set_ylabel(r'$\text{Objective per Unit}$')
    ax.set_title(r'$\textbf{Solution Efficiency}$')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=3000, bbox_inches='tight')
    plt.close()
    print(f"✓ Solution quality comparison saved to {output_path}")

def plot_detailed_solution_analysis(data: Dict[int, Dict], metrics: Dict, 
                                  solver_key: str, output_path: Path):
    """Create detailed solution analysis for individual solver."""
    config = SOLVER_CONFIGS[solver_key]
    color = ColorPalette.get_solver_color(solver_key)
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3)
    fig.suptitle(rf'\textbf{{{config["display_name"]} Detailed Analysis}}', 
                 fontsize=16, y=0.98)
    
    # Plot 1: Area Distribution (Pie chart for largest problem)
    ax1 = fig.add_subplot(gs[0, 0])
    largest_config = max(data.keys())
    largest_data = data[largest_config]
    
    if 'solution_summary' in largest_data:
        plot_assignments = largest_data['solution_summary']['plot_assignments']
        crop_areas = {}
        for assignment in plot_assignments:
            crop = assignment['crop']
            area = assignment.get('area', assignment.get('total_area', 0))
            crop_areas[crop] = crop_areas.get(crop, 0) + area
        
        # Take top 8 crops for clarity
        sorted_crops = sorted(crop_areas.items(), key=lambda x: x[1], reverse=True)[:8]
        labels = [c[0] for c in sorted_crops]
        sizes = [c[1] for c in sorted_crops]
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                          colors=ColorPalette.QUALITATIVE[:len(labels)])
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax1.set_title(rf'$\textbf{{Area Distribution}} \\ \small{{{largest_config} {config["unit_label"]}}}$')
    
    # Plot 2: Time Breakdown
    ax2 = fig.add_subplot(gs[0, 1:])
    n_units = metrics['n_units']
    
    if config['has_qpu_time'] and config['has_bqm_conversion']:
        # D-Wave BQM breakdown
        components = {
            'QPU Time': metrics['qpu_time'],
            'BQM Conversion': metrics['bqm_conversion_time'],
            'Overhead': [metrics['solve_time'][i] - metrics['qpu_time'][i] - metrics['bqm_conversion_time'][i] 
                        for i in range(len(n_units))]
        }
    elif config['has_qpu_time']:
        # D-Wave CQM breakdown
        components = {
            'QPU Time': metrics['qpu_time'],
            'Overhead': [metrics['solve_time'][i] - metrics['qpu_time'][i] 
                        for i in range(len(n_units))]
        }
    elif config['has_solver_time']:
        # Classical solver breakdown
        components = {
            'Solver Time': metrics['solver_time'],
            'Overhead': [metrics['solve_time'][i] - metrics['solver_time'][i] 
                        for i in range(len(n_units))]
        }
    else:
        components = {'Total Time': metrics['solve_time']}
    
    bottom = np.zeros(len(n_units))
    for label, times in components.items():
        ax2.bar(n_units, times, bottom=bottom, label=label, 
               color=color, alpha=0.7, edgecolor='white', linewidth=1)
        bottom += times
    
    ax2.set_xlabel(rf'$\text{{Number of {config["unit_label"]}}}$')
    ax2.set_ylabel(r'$\text{Time (s)}$')
    ax2.set_title(r'$\textbf{Time Breakdown}$')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Performance Metrics
    ax3 = fig.add_subplot(gs[1, 0])
    utilization_pct = [u * 100 for u in metrics['utilization']]
    ax3.plot(n_units, utilization_pct, 'o-', color=color, linewidth=2)
    ax3.axhline(y=100, color=ColorPalette.SEMANTIC['success'], linestyle='--', alpha=0.7)
    ax3.set_xlabel(rf'$\text{{Number of {config["unit_label"]}}}$')
    ax3.set_ylabel(r'$\text{Utilization (\%)}$')
    ax3.set_title(r'$\textbf{Land Utilization}$')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Crop Selection Pattern
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(n_units, metrics['n_crops'], 's-', color=color, linewidth=2)
    ax4.set_xlabel(rf'$\text{{Number of {config["unit_label"]}}}$')
    ax4.set_ylabel(r'$\text{Number of Crops}$')
    ax4.set_title(r'$\textbf{Crop Diversity}$')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Solution Quality Over Time
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(n_units, metrics['objective_value'], '^-', color=color, linewidth=2)
    ax5.set_xlabel(rf'$\text{{Number of {config["unit_label"]}}}$')
    ax5.set_ylabel(r'$\text{Objective Value}$')
    ax5.set_title(r'$\textbf{Solution Quality}$')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary Statistics
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Calculate summary statistics
    summary_stats = [
        rf"$\textbf{{Solver}}: {config['display_name']}$",
        rf"$\textbf{{Problem Type}}: {config['problem_type']}$",
        rf"$\textbf{{Configurations}}: {len(n_units)}$",
        rf"$\textbf{{Max Variables}}: {max(metrics['n_variables']):,}$",
        rf"$\textbf{{Avg Solve Time}}: {np.mean(metrics['solve_time']):.2f}s$",
        rf"$\textbf{{Max Solve Time}}: {max(metrics['solve_time']):.2f}s$",
        rf"$\textbf{{Avg Utilization}}: {np.mean(metrics['utilization'])*100:.1f}\%$",
        rf"$\textbf{{Avg Crops}}: {np.mean(metrics['n_crops']):.1f}$",
    ]
    
    if config['has_validation']:
        feasible_count = sum(metrics['is_feasible'])
        summary_stats.append(rf"$\textbf{{Feasible Solutions}}: {feasible_count}/{len(n_units)}$")
    
    summary_text = '\n'.join(summary_stats)
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=3000, bbox_inches='tight')
    plt.close()
    print(f"✓ Detailed analysis saved to {output_path}")

def plot_constraint_analysis(all_metrics: Dict[str, Dict], output_path: Path):
    """Create constraint satisfaction analysis for solvers with validation."""
    validation_solvers = [s for s in all_metrics.keys() 
                         if SOLVER_CONFIGS[s]['has_validation']]
    
    if not validation_solvers:
        print("⚠ No solvers with constraint validation data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(r'\textbf{Constraint Satisfaction Analysis}', fontsize=16, y=1.0)
    
    # Plot 1: Pass Rate Comparison
    ax = axes[0]
    for solver in validation_solvers:
        metrics = all_metrics[solver]
        color = ColorPalette.get_solver_color(solver)
        marker_style = ColorPalette.get_marker_style(solver)
        
        problem_size = [n * 27 for n in metrics['n_units']]
        pass_rates = [p * 100 for p in metrics['pass_rate']]
        ax.plot(problem_size, pass_rates,
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.axhline(y=100, color=ColorPalette.SEMANTIC['success'], linestyle='--', 
               label='Perfect', alpha=0.7)
    ax.set_xlabel(r'$\text{Problem Size (Units} \times \text{Foods)}$')
    ax.set_ylabel(r'$\text{Constraint Pass Rate (\%)}$')
    ax.set_title(r'$\textbf{Pass Rate Comparison}$')
    ax.legend()
    ax.set_ylim(0, 110)
    #ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Violation Analysis with color-coded points by feasibility
    ax = axes[1]
    
    # Create legend handles manually
    from matplotlib.patches import Patch
    legend_elements = []
    
    for solver in validation_solvers:
        metrics = all_metrics[solver]
        solver_color = ColorPalette.get_solver_color(solver)
        problem_size = [n * 27 for n in metrics['n_units']]
        
        # Plot each point with color based on feasibility
        for x, y, feasible in zip(problem_size, metrics['n_violations'], metrics['is_feasible']):
            if feasible:
                ax.scatter(x, y, color=ColorPalette.SEMANTIC['success'], 
                          s=100, alpha=0.8, edgecolors=solver_color, linewidths=2,
                          marker='o')
            else:
                ax.scatter(x, y, color=ColorPalette.SEMANTIC['error'], 
                          s=100, alpha=0.8, edgecolors=solver_color, linewidths=2,
                          marker='o')
        
        # Add solver to legend
        legend_elements.append(Patch(facecolor=solver_color, 
                                     label=SOLVER_CONFIGS[solver]['display_name']))
    
    # Add feasibility indicators to legend
    legend_elements.append(Patch(facecolor=ColorPalette.SEMANTIC['success'], 
                                 label='Feasible'))
    legend_elements.append(Patch(facecolor=ColorPalette.SEMANTIC['error'], 
                                 label='Infeasible'))
    
    ax.set_xlabel(r'$\text{Problem Size (Units} \times \text{Foods)}$')
    ax.set_ylabel(r'$\text{Number of Violations}$')
    ax.set_title(r'$\textbf{Constraint Violations}$')
    ax.legend(handles=legend_elements, fontsize=9)
    #ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=3000, bbox_inches='tight')
    plt.close()
    print(f"✓ Constraint analysis saved to {output_path}")

def plot_solution_composition_histograms(all_compositions: Dict[str, Dict], output_path: Path):
    """Create histograms showing solution composition across solvers (same layout as pie charts but with crops on x-axis)."""
    solvers = list(all_compositions.keys())
    
    # Find common problem sizes across solvers
    common_sizes = set()
    for solver_compositions in all_compositions.values():
        common_sizes.update(solver_compositions.keys())
    common_sizes = sorted(common_sizes)
    
    if not common_sizes:
        print("⚠ No common problem sizes with solution data for histograms")
        return
    
    # Limit to 4 most representative sizes for clarity
    display_sizes = common_sizes[:4] if len(common_sizes) > 4 else common_sizes
    
    fig, axes = plt.subplots(len(display_sizes), len(solvers), 
                            figsize=(4 * len(solvers), 4 * len(display_sizes)))
    fig.suptitle(r'\textbf{Solution Composition Analysis (Log Scale)}', fontsize=16, y=0.98)
    
    # Ensure axes is 2D array for consistent indexing
    if len(display_sizes) == 1:
        axes = axes.reshape(1, -1)
    if len(solvers) == 1:
        axes = axes.reshape(-1, 1)
    
    # Create unified color mapping for crops across all plots
    all_crops = set()
    for solver_compositions in all_compositions.values():
        for composition in solver_compositions.values():
            all_crops.update(composition.keys())
    all_crops = sorted(list(all_crops))
    
    crop_colors = {crop: ColorPalette.QUALITATIVE[i % len(ColorPalette.QUALITATIVE)] 
                  for i, crop in enumerate(all_crops)}
    
    for col_idx, solver in enumerate(solvers):
        compositions = all_compositions[solver]
        config = SOLVER_CONFIGS[solver]
        
        for row_idx, problem_size in enumerate(display_sizes):
            if problem_size not in compositions:
                axes[row_idx, col_idx].text(0.5, 0.5, 'No Data', 
                                          ha='center', va='center', 
                                          transform=axes[row_idx, col_idx].transAxes)
                axes[row_idx, col_idx].set_title(f'{problem_size} {config["unit_label"]}')
                continue
            
            composition = compositions[problem_size]
            
            # Prepare data for histogram
            crops = []
            percentages = []
            colors = []
            
            # Sort by percentage (descending) and take top 8, group rest as "Other"
            sorted_items = sorted(composition.items(), key=lambda x: x[1], reverse=True)
            top_items = sorted_items[:8]
            other_percentage = sum(item[1] for item in sorted_items[8:])
            
            for crop, percentage in top_items:
                crops.append(crop)
                percentages.append(percentage)
                colors.append(crop_colors.get(crop, ColorPalette.SEMANTIC['neutral']))
            
            if other_percentage > 0.5:  # Only show "Other" if significant
                crops.append('Other')
                percentages.append(other_percentage)
                colors.append(ColorPalette.SEMANTIC['neutral'])
            
            # Create histogram with log scale
            x_pos = np.arange(len(crops))
            bars = axes[row_idx, col_idx].bar(x_pos, percentages, color=colors, 
                                             edgecolor='black', linewidth=1.0)
            
            # Set log scale on y-axis
            axes[row_idx, col_idx].set_yscale('log')
            
            # Set labels and title
            axes[row_idx, col_idx].set_xticks(x_pos)
            axes[row_idx, col_idx].set_xticklabels(crops, rotation=45, ha='right', fontsize=8)
            axes[row_idx, col_idx].set_ylabel(r'$\text{Area (\%)}$', fontsize=9)
            
            # Add grid for better readability
            axes[row_idx, col_idx].grid(True, alpha=0.3, which='both', axis='y')
            
            # Set title for first row only
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(
                    f'{config["display_name"]}\n{problem_size} {config["unit_label"]}',
                    fontsize=10, pad=20
                )
            else:
                axes[row_idx, col_idx].set_title(
                    f'{problem_size} {config["unit_label"]}',
                    fontsize=10, pad=20
                )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=3000, bbox_inches='tight')
    plt.close()
    print(f"✓ Solution composition histograms saved to {output_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function with comprehensive plotting."""
    parser = argparse.ArgumentParser(
        description='Professional benchmark visualization system')
    parser.add_argument('--solvers', nargs='+', 
                       choices=list(SOLVER_CONFIGS.keys()),
                       default=list(SOLVER_CONFIGS.keys()),
                       help='Solvers to analyze (default: all)')
    parser.add_argument('--benchmark-dir', type=str, default='Legacy',
                       help='Benchmark directory (default: Legacy)')
    parser.add_argument('--output-dir', type=str, default='professional_plots',
                       help='Output directory (default: professional_plots)')
    parser.add_argument('--individual', action='store_true',
                       help='Generate individual solver analysis')
    
    args = parser.parse_args()
    
    # Configure professional style
    configure_professional_style()
    
    # Set up paths
    script_dir = Path(__file__).parent
    benchmark_dir = script_dir / args.benchmark_dir
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("PROFESSIONAL BENCHMARK VISUALIZATION SYSTEM")
    print("="*70)
    
    # Load data for all solvers
    print("\n📂 Loading benchmark data...")
    all_data = {}
    all_metrics = {}
    all_compositions = {}
    
    for solver in args.solvers:
        try:
            data = load_benchmark_data(benchmark_dir, solver)
            metrics = extract_metrics(data, solver)
            compositions = extract_solution_composition(data, solver)
            
            all_data[solver] = data
            all_metrics[solver] = metrics
            all_compositions[solver] = compositions
            
            print(f"✓ {SOLVER_CONFIGS[solver]['display_name']}: {len(data)} configurations")
        except FileNotFoundError as e:
            print(f"✗ {solver}: {e}")
            continue
    
    if not all_metrics:
        print("❌ No valid solver data found!")
        return 1
    
    print(f"\n📊 Loaded data for {len(all_metrics)} solvers")
    
    # Generate comparative plots
    print("\n📈 Generating comparative visualizations...")
    
    plot_performance_comparison(all_metrics, output_dir / "performance_comparison.pdf")
    plot_solution_quality_comparison(all_metrics, output_dir / "solution_quality_comparison.pdf")
    plot_constraint_analysis(all_metrics, output_dir / "constraint_analysis.pdf")
    
    # Generate histogram visualizations
    print("\n📊 Generating histogram visualizations...")
    
    if any(all_compositions.values()):
        plot_solution_composition_histograms(all_compositions, output_dir / "solution_composition_histograms.pdf")
    
    # Generate pie chart visualizations
    print("\n🥧 Generating pie chart visualizations...")
    
    if any(all_compositions.values()):
        plot_solution_composition_pie_charts(all_compositions, output_dir / "solution_composition_pies.pdf")
        plot_land_utilization_pie(all_metrics, output_dir / "land_utilization_pies.pdf")
    
    plot_constraint_satisfaction_pie(all_metrics, output_dir / "constraint_satisfaction_pies.pdf")
    
    # Generate individual solver analysis if requested
    if args.individual:
        print("\n📊 Generating individual solver analysis...")
        for solver in all_metrics.keys():
            solver_name = SOLVER_CONFIGS[solver]['dir_name'].lower()
            plot_detailed_solution_analysis(
                all_data[solver], all_metrics[solver], solver,
                output_dir / f"{solver_name}_detailed_analysis.pdf"
            )
            if solver in all_compositions and all_compositions[solver]:
                plot_individual_solver_pie_charts(
                    all_data[solver], solver,
                    output_dir / f"{solver_name}_pie_analysis.pdf"
                )
    
    # Generate summary report
    print("\n📋 Generating summary report...")
    generate_summary_report(all_metrics, output_dir / "benchmark_summary.pdf")
    
    print(f"\n✅ All plots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - performance_comparison.pdf        : Cross-solver performance")
    print("  - solution_quality_comparison.pdf   : Solution quality metrics")
    print("  - constraint_analysis.pdf           : Constraint satisfaction")
    print("  - solution_composition_histograms.pdf : Crop distribution (log scale)")
    print("  - solution_composition_pies.pdf     : Crop distribution pie charts")
    print("  - land_utilization_pies.pdf         : Land utilization efficiency")
    print("  - constraint_satisfaction_pies.pdf  : Constraint satisfaction rates")
    if args.individual:
        print("  - *_detailed_analysis.pdf         : Individual solver details")
        print("  - *_pie_analysis.pdf              : Individual solver composition")
    print("  - benchmark_summary.pdf             : Comprehensive summary")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    
    return 0

def generate_summary_report(all_metrics: Dict[str, Dict], output_path: Path):
    """Generate comprehensive summary report."""
    fig = plt.figure(figsize=(12, 16))
    fig.suptitle(r'\textbf{Benchmark Summary Report}', fontsize=18, y=0.98)
    
    # Create summary table
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Prepare summary data
    summary_data = []
    headers = [r'$\textbf{Solver}$', r'$\textbf{Problems}$', r'$\textbf{Avg Time}$', 
               r'$\textbf{Max Time}$', r'$\textbf{Avg Obj}$', r'$\textbf{Avg Util}$']
    
    for solver, metrics in all_metrics.items():
        config = SOLVER_CONFIGS[solver]
        n_problems = len(metrics['n_units'])
        avg_time = np.mean(metrics['solve_time'])
        max_time = np.max(metrics['solve_time'])
        avg_obj = np.mean(metrics['objective_value'])
        avg_util = np.mean(metrics['utilization']) * 100
        
        summary_data.append([
            config['display_name'],
            f"{n_problems}",
            f"{avg_time:.2f}s",
            f"{max_time:.2f}s",
            f"{avg_obj:.2f}",
            f"{avg_util:.1f}%"
        ])
    
    # Create table
    table = ax.table(cellText=summary_data, colLabels=headers,
                    loc='center', cellLoc='center')
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4C72B0')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternating rows
    for i in range(len(summary_data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i+1, j)].set_facecolor('#f0f0f0')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=3000, bbox_inches='tight')
    plt.close()
    print(f"✓ Summary report saved to {output_path}")

if __name__ == "__main__":
    exit(main())