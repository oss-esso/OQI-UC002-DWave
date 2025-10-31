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
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import json
from typing import Dict, List, Any, Optional

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
            metrics['solver_time'].append(d['solver_time'])
        if config['has_bqm_conversion']:
            metrics['bqm_conversion_time'].append(d['bqm_conversion_time'])
        if config['has_bqm_energy']:
            metrics['bqm_energy'].append(d['bqm_energy'])
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
        
        ax.plot(metrics['n_units'], metrics['solve_time'], 
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.set_xlabel(r'$\text{Problem Size}$')
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
        
        ax.plot(metrics['n_units'], metrics['objective_value'],
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.set_xlabel(r'$\text{Problem Size}$')
    ax.set_ylabel(r'$\text{Objective Value}$')
    ax.set_title(r'$\textbf{Solution Quality}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Problem Size Scaling
    ax = axes[1, 0]
    for solver in solvers:
        metrics = all_metrics[solver]
        color = ColorPalette.get_solver_color(solver)
        marker_style = ColorPalette.get_marker_style(solver)
        
        ax.plot(metrics['n_units'], metrics['n_variables'],
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.set_xlabel(r'$\text{Problem Size}$')
    ax.set_ylabel(r'$\text{Number of Variables}$')
    ax.set_title(r'$\textbf{Problem Complexity}$')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency Comparison
    ax = axes[1, 1]
    for solver in solvers:
        metrics = all_metrics[solver]
        color = ColorPalette.get_solver_color(solver)
        marker_style = ColorPalette.get_marker_style(solver)
        
        time_per_var = [t/v * 1000 for t, v in zip(metrics['solve_time'], metrics['n_variables'])]
        ax.plot(metrics['n_units'], time_per_var,
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.set_xlabel(r'$\text{Problem Size}$')
    ax.set_ylabel(r'$\text{Time per Variable (ms)}$')
    ax.set_title(r'$\textbf{Solving Efficiency}$')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        
        utilization_pct = [u * 100 for u in metrics['utilization']]
        ax.plot(metrics['n_units'], utilization_pct,
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.axhline(y=100, color=ColorPalette.SEMANTIC['neutral'], linestyle='--', alpha=0.7)
    ax.set_xlabel(r'$\text{Problem Size}$')
    ax.set_ylabel(r'$\text{Land Utilization (\%)}$')
    ax.set_title(r'$\textbf{Resource Utilization}$')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Crop Diversity
    ax = axes[0, 1]
    for solver in solvers:
        metrics = all_metrics[solver]
        color = ColorPalette.get_solver_color(solver)
        marker_style = ColorPalette.get_marker_style(solver)
        
        ax.plot(metrics['n_units'], metrics['n_crops'],
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.set_xlabel(r'$\text{Problem Size}$')
    ax.set_ylabel(r'$\text{Number of Crops}$')
    ax.set_title(r'$\textbf{Crop Diversity}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Constraint Satisfaction (if available)
    ax = axes[1, 0]
    validation_solvers = [s for s in solvers if SOLVER_CONFIGS[s]['has_validation']]
    
    if validation_solvers:
        for solver in validation_solvers:
            metrics = all_metrics[solver]
            color = ColorPalette.get_solver_color(solver)
            marker_style = ColorPalette.get_marker_style(solver)
            
            pass_rates = [p * 100 for p in metrics['pass_rate']]
            ax.plot(metrics['n_units'], pass_rates,
                    color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                    **marker_style)
        
        ax.axhline(y=100, color=ColorPalette.SEMANTIC['success'], linestyle='--', alpha=0.7)
        ax.set_xlabel(r'$\text{Problem Size}$')
        ax.set_ylabel(r'$\text{Constraint Pass Rate (\%)}$')
        ax.set_title(r'$\textbf{Constraint Satisfaction}$')
        ax.legend(fontsize=9)
        ax.set_ylim(0, 110)
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
        
        obj_per_unit = [obj/n for obj, n in zip(metrics['objective_value'], metrics['n_units'])]
        ax.plot(metrics['n_units'], obj_per_unit,
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.set_xlabel(r'$\text{Problem Size}$')
    ax.set_ylabel(r'$\text{Objective per Unit}$')
    ax.set_title(r'$\textbf{Solution Efficiency}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        
        pass_rates = [p * 100 for p in metrics['pass_rate']]
        ax.plot(metrics['n_units'], pass_rates,
                color=color, label=SOLVER_CONFIGS[solver]['display_name'],
                **marker_style)
    
    ax.axhline(y=100, color=ColorPalette.SEMANTIC['success'], linestyle='--', 
               label='Perfect', alpha=0.7)
    ax.set_xlabel(r'$\text{Problem Size}$')
    ax.set_ylabel(r'$\text{Constraint Pass Rate (\%)}$')
    ax.set_title(r'$\textbf{Pass Rate Comparison}$')
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Violation Analysis
    ax = axes[1]
    for solver in validation_solvers:
        metrics = all_metrics[solver]
        color = ColorPalette.get_solver_color(solver)
        
        # Color points by feasibility
        for i, (x, y, feasible) in enumerate(zip(metrics['n_units'], 
                                                metrics['n_violations'],
                                                metrics['is_feasible'])):
            point_color = ColorPalette.SEMANTIC['success'] if feasible else ColorPalette.SEMANTIC['error']
            ax.scatter(x, y, color=point_color, s=60, alpha=0.8,
                      label=SOLVER_CONFIGS[solver]['display_name'] if i == 0 else "")
    
    ax.set_xlabel(r'$\text{Problem Size}$')
    ax.set_ylabel(r'$\text{Number of Violations}$')
    ax.set_title(r'$\textbf{Constraint Violations}$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Constraint analysis saved to {output_path}")

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
    
    for solver in args.solvers:
        try:
            data = load_benchmark_data(benchmark_dir, solver)
            metrics = extract_metrics(data, solver)
            all_data[solver] = data
            all_metrics[solver] = metrics
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
    
    # Generate individual solver analysis if requested
    if args.individual:
        print("\n📊 Generating individual solver analysis...")
        for solver in all_metrics.keys():
            solver_name = SOLVER_CONFIGS[solver]['dir_name'].lower()
            plot_detailed_solution_analysis(
                all_data[solver], all_metrics[solver], solver,
                output_dir / f"{solver_name}_detailed_analysis.pdf"
            )
    
    # Generate summary report
    print("\n📋 Generating summary report...")
    generate_summary_report(all_metrics, output_dir / "benchmark_summary.pdf")
    
    print(f"\n✅ All plots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - performance_comparison.pdf       : Cross-solver performance")
    print("  - solution_quality_comparison.pdf  : Solution quality metrics")
    print("  - constraint_analysis.pdf          : Constraint satisfaction")
    if args.individual:
        print("  - *_detailed_analysis.pdf        : Individual solver details")
    print("  - benchmark_summary.pdf            : Comprehensive summary")
    
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
            f"{avg_util:.1f}\%"
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Summary report saved to {output_path}")

if __name__ == "__main__":
    exit(main())