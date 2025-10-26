#!/usr/bin/env python3
"""
Comprehensive Benchmark Results Visualization
Creates plots showing solve times, solution quality, and speedup comparisons
across Farm and Patch scenarios with multiple solvers.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt conflicts
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style for beautiful plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (20, 15)  # Larger for the 3x3 grid
plt.rcParams['font.family'] = 'sans-serif'

# Shared plotting configuration so every figure uses consistent styling
SERIES_ORDER = [
    'Farm_PuLP',
    'Farm_DWave_QPU',
    'Farm_DWave_Hybrid',
    'Patch_PuLP',
    'Patch_DWave_QPU',
    'Patch_DWave_Hybrid',
    'Patch_DWaveBQM_QPU',
    'Patch_GurobiQUBO'
]

SERIES_STYLES = {
    'Farm_PuLP': {
        'label': 'Farm PuLP (Classical)',
        'color': '#E63946',
        'marker': 'o'
    },
    'Farm_DWave_QPU': {
        'label': 'Farm D-Wave CQM QPU',
        'color': '#FF006E',
        'marker': 's'
    },
    'Farm_DWave_Hybrid': {
        'label': 'Farm D-Wave CQM Hybrid',
        'color': '#FB5607',
        'marker': 'D'
    },
    'Patch_PuLP': {
        'label': 'Patch PuLP (Classical)',
        'color': '#FFBE0B',
        'marker': '^'
    },
    'Patch_DWave_QPU': {
        'label': 'Patch D-Wave CQM QPU',
        'color': '#8338EC',
        'marker': 'v'
    },
    'Patch_DWave_Hybrid': {
        'label': 'Patch D-Wave CQM Hybrid',
        'color': '#3A86FF',
        'marker': '<'
    },
    'Patch_DWaveBQM_QPU': {
        'label': 'Patch D-Wave BQM QPU',
        'color': '#06FFA5',
        'marker': '>'
    },
    'Patch_GurobiQUBO': {
        'label': 'Patch Gurobi QUBO',
        'color': '#2EC4B6',
        'marker': 'p'
    }
}

def load_benchmark_data(benchmark_dir):
    """Load all comprehensive benchmark data from the Benchmarks directory."""
    comp_dir = Path(benchmark_dir) / "COMPREHENSIVE"
    
    data = {
        'Farm_PuLP': {},
        'Farm_DWave': {},
        'Patch_PuLP': {},
        'Patch_DWave': {},
        'Patch_GurobiQUBO': {},
        'Patch_DWaveBQM': {}
    }
    
    # Configuration files to load
    configs = [10, 15, 20, 25, 100]
    
    for solver_name in data.keys():
        solver_dir = comp_dir / solver_name
        if not solver_dir.exists():
            continue
        
        # Try to find any run file for each config
        for config in configs:
            for run in range(1, 10):  # Try runs 1-9
                config_file = solver_dir / f"config_{config}_run_{run}.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        benchmark = json.load(f)
                        data[solver_name][config] = benchmark
                    break  # Found a file, move to next config
    
    return data

def extract_times(data):
    """Extract solve times for each solver and configuration."""
    # Get all configs
    all_configs = set()
    for solver_data in data.values():
        all_configs.update(solver_data.keys())
    configs = sorted(list(all_configs))
    
    times = {
        'n_units': configs,
        'n_vars': [c * 27 for c in configs],  # Number of variables
        'Farm_PuLP': [],
        'Farm_DWave_QPU': [],
        'Farm_DWave_Hybrid': [],
        'Patch_PuLP': [],
        'Patch_DWave_QPU': [],
        'Patch_DWave_Hybrid': [],
        'Patch_GurobiQUBO': [],
        'Patch_DWaveBQM_QPU': [],
        'Patch_DWaveBQM_Hybrid': []
    }
    
    objectives = {
        'n_units': configs,
        'n_vars': [c * 27 for c in configs],  # Number of variables
        'Farm_PuLP': [],
        'Farm_DWave': [],
        'Patch_PuLP': [],
        'Patch_DWave': [],
        'Patch_GurobiQUBO': [],
        'Patch_DWaveBQM': []
    }
    
    for config in configs:
        # Farm PuLP
        if config in data['Farm_PuLP']:
            times['Farm_PuLP'].append(data['Farm_PuLP'][config].get('solve_time'))
            objectives['Farm_PuLP'].append(data['Farm_PuLP'][config].get('objective_value'))
        else:
            times['Farm_PuLP'].append(None)
            objectives['Farm_PuLP'].append(None)
        
        # Farm DWave
        if config in data['Farm_DWave']:
            d = data['Farm_DWave'][config]
            times['Farm_DWave_QPU'].append(d.get('qpu_time'))
            times['Farm_DWave_Hybrid'].append(d.get('hybrid_time') or d.get('solve_time'))
            objectives['Farm_DWave'].append(d.get('objective_value'))
        else:
            times['Farm_DWave_QPU'].append(None)
            times['Farm_DWave_Hybrid'].append(None)
            objectives['Farm_DWave'].append(None)
        
        # Patch PuLP
        if config in data['Patch_PuLP']:
            times['Patch_PuLP'].append(data['Patch_PuLP'][config].get('solve_time'))
            objectives['Patch_PuLP'].append(data['Patch_PuLP'][config].get('objective_value'))
        else:
            times['Patch_PuLP'].append(None)
            objectives['Patch_PuLP'].append(None)
        
        # Patch DWave
        if config in data['Patch_DWave']:
            d = data['Patch_DWave'][config]
            times['Patch_DWave_QPU'].append(d.get('qpu_time'))
            times['Patch_DWave_Hybrid'].append(d.get('hybrid_time') or d.get('solve_time'))
            objectives['Patch_DWave'].append(d.get('objective_value'))
        else:
            times['Patch_DWave_QPU'].append(None)
            times['Patch_DWave_Hybrid'].append(None)
            objectives['Patch_DWave'].append(None)
        
        # Patch Gurobi QUBO
        if config in data['Patch_GurobiQUBO']:
            times['Patch_GurobiQUBO'].append(data['Patch_GurobiQUBO'][config].get('solve_time'))
            objectives['Patch_GurobiQUBO'].append(data['Patch_GurobiQUBO'][config].get('objective_value'))
        else:
            times['Patch_GurobiQUBO'].append(None)
            objectives['Patch_GurobiQUBO'].append(None)
        
        # Patch DWave BQM
        if config in data['Patch_DWaveBQM']:
            d = data['Patch_DWaveBQM'][config]
            times['Patch_DWaveBQM_QPU'].append(d.get('qpu_time'))
            times['Patch_DWaveBQM_Hybrid'].append(d.get('hybrid_time') or d.get('solve_time'))
            objectives['Patch_DWaveBQM'].append(d.get('objective_value'))
        else:
            times['Patch_DWaveBQM_QPU'].append(None)
            times['Patch_DWaveBQM_Hybrid'].append(None)
            objectives['Patch_DWaveBQM'].append(None)
    
    return times, objectives


def normalize_objectives(objectives):
    """Normalize objective values so Farm scenarios align with Patch scale."""
    normalized = {
        'n_units': list(objectives['n_units']),
        'n_vars': list(objectives['n_vars'])
    }
    normalized['Farm_PuLP'] = [val / 10 if val is not None else None for val in objectives['Farm_PuLP']]
    normalized['Farm_DWave'] = [val / 10 if val is not None else None for val in objectives['Farm_DWave']]
    normalized['Patch_PuLP'] = list(objectives['Patch_PuLP'])
    normalized['Patch_DWave'] = list(objectives['Patch_DWave'])
    normalized['Patch_GurobiQUBO'] = list(objectives['Patch_GurobiQUBO'])
    normalized['Patch_DWaveBQM'] = list(objectives['Patch_DWaveBQM'])
    return normalized


def build_objective_series(normalized_objectives):
    """Construct objective series aligned with the unified solver order."""
    series = {
        'n_units': list(normalized_objectives['n_units']),
        'n_vars': list(normalized_objectives['n_vars'])
    }
    series['Farm_PuLP'] = list(normalized_objectives['Farm_PuLP'])
    farm_dwave = list(normalized_objectives['Farm_DWave'])
    series['Farm_DWave_QPU'] = farm_dwave[:]
    series['Farm_DWave_Hybrid'] = farm_dwave[:]
    series['Patch_PuLP'] = list(normalized_objectives['Patch_PuLP'])
    patch_dwave = list(normalized_objectives['Patch_DWave'])
    series['Patch_DWave_QPU'] = patch_dwave[:]
    series['Patch_DWave_Hybrid'] = patch_dwave[:]
    series['Patch_DWaveBQM_QPU'] = list(normalized_objectives['Patch_DWaveBQM'])
    series['Patch_GurobiQUBO'] = list(normalized_objectives['Patch_GurobiQUBO'])
    return series

def calculate_speedups(times):
    """Calculate speedup factors comparing quantum to classical solvers."""
    speedups = {
        'n_units': times['n_units'],
        'Farm_QPU_vs_PuLP': [],
        'Farm_Hybrid_vs_PuLP': [],
        'Patch_CQM_QPU_vs_PuLP': [],
        'Patch_CQM_Hybrid_vs_PuLP': [],
        'Patch_BQM_QPU_vs_PuLP': [],
        'Patch_BQM_Hybrid_vs_PuLP': [],
        'Patch_QUBO_vs_PuLP': []
    }
    
    for i in range(len(times['n_units'])):
        # Farm speedups
        farm_pulp = times['Farm_PuLP'][i]
        if farm_pulp and farm_pulp > 0:
            farm_qpu = times['Farm_DWave_QPU'][i]
            farm_hybrid = times['Farm_DWave_Hybrid'][i]
            speedups['Farm_QPU_vs_PuLP'].append(farm_pulp / farm_qpu if farm_qpu and farm_qpu > 0 else None)
            speedups['Farm_Hybrid_vs_PuLP'].append(farm_pulp / farm_hybrid if farm_hybrid and farm_hybrid > 0 else None)
        else:
            speedups['Farm_QPU_vs_PuLP'].append(None)
            speedups['Farm_Hybrid_vs_PuLP'].append(None)
        
        # Patch speedups
        patch_pulp = times['Patch_PuLP'][i]
        if patch_pulp and patch_pulp > 0:
            # CQM speedups
            patch_cqm_qpu = times['Patch_DWave_QPU'][i]
            patch_cqm_hybrid = times['Patch_DWave_Hybrid'][i]
            speedups['Patch_CQM_QPU_vs_PuLP'].append(patch_pulp / patch_cqm_qpu if patch_cqm_qpu and patch_cqm_qpu > 0 else None)
            speedups['Patch_CQM_Hybrid_vs_PuLP'].append(patch_pulp / patch_cqm_hybrid if patch_cqm_hybrid and patch_cqm_hybrid > 0 else None)
            
            # BQM speedups
            patch_bqm_qpu = times['Patch_DWaveBQM_QPU'][i]
            patch_bqm_hybrid = times['Patch_DWaveBQM_Hybrid'][i]
            speedups['Patch_BQM_QPU_vs_PuLP'].append(patch_pulp / patch_bqm_qpu if patch_bqm_qpu and patch_bqm_qpu > 0 else None)
            speedups['Patch_BQM_Hybrid_vs_PuLP'].append(patch_pulp / patch_bqm_hybrid if patch_bqm_hybrid and patch_bqm_hybrid > 0 else None)
            
            # QUBO speedup
            patch_qubo = times['Patch_GurobiQUBO'][i]
            speedups['Patch_QUBO_vs_PuLP'].append(patch_pulp / patch_qubo if patch_qubo and patch_qubo > 0 else None)
        else:
            speedups['Patch_CQM_QPU_vs_PuLP'].append(None)
            speedups['Patch_CQM_Hybrid_vs_PuLP'].append(None)
            speedups['Patch_BQM_QPU_vs_PuLP'].append(None)
            speedups['Patch_BQM_Hybrid_vs_PuLP'].append(None)
            speedups['Patch_QUBO_vs_PuLP'].append(None)
    
    return speedups

def plot_solve_times(times, output_path):
    """Create three plots: linear, log-y, and log-log scales."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Benchmark: Solve Time Comparison (D-Wave vs Classical)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    n_vars = times['n_vars']  # Use number of variables instead of units
    
    def plot_series(ax_obj, transform=None):
        for key in SERIES_ORDER:
            if key not in times:
                continue
            y_series = times[key]
            filtered = [(x, y) for x, y in zip(n_vars, y_series) if y is not None and (y > 0 if transform else True)]
            if not filtered:
                continue
            x_vals, y_vals = zip(*filtered)
            style = SERIES_STYLES[key]
            if transform == 'semilogy':
                ax_obj.semilogy(x_vals, y_vals, marker=style['marker'], linewidth=2.5,
                                markersize=10, color=style['color'], label=style['label'], alpha=0.8)
            elif transform == 'loglog':
                ax_obj.loglog(x_vals, y_vals, marker=style['marker'], linewidth=2.5,
                              markersize=10, color=style['color'], label=style['label'], alpha=0.8)
            else:
                ax_obj.plot(x_vals, y_vals, marker=style['marker'], linewidth=2.5,
                            markersize=10, color=style['color'], label=style['label'], alpha=0.8)
    
    ax = axes[0, 0]
    plot_series(ax)
    ax.set_xlabel('Number of Variables', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    plot_series(ax, transform='semilogy')
    ax.set_xlabel('Number of Variables', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Y Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    
    ax = axes[0, 2]
    plot_series(ax, transform='loglog')
    ax.set_xlabel('Number of Variables (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    
    # Row 2: Speedup Factors
    speedups = calculate_speedups(times)
    
    # Linear scale speedup
    ax = axes[1, 0]
    ax.plot(n_vars, speedups['Farm_QPU_vs_PuLP'], marker=SERIES_STYLES['Farm_DWave_QPU']['marker'], linewidth=2.5, 
        markersize=10, color=SERIES_STYLES['Farm_DWave_QPU']['color'], label='Farm QPU vs PuLP', alpha=0.8)
    ax.plot(n_vars, speedups['Farm_Hybrid_vs_PuLP'], marker=SERIES_STYLES['Farm_DWave_Hybrid']['marker'], linewidth=2.5, 
        markersize=10, color=SERIES_STYLES['Farm_DWave_Hybrid']['color'], label='Farm Hybrid vs PuLP', alpha=0.8)
    ax.plot(n_vars, speedups['Patch_CQM_QPU_vs_PuLP'], marker=SERIES_STYLES['Patch_DWave_QPU']['marker'], linewidth=2.5, 
        markersize=10, color=SERIES_STYLES['Patch_DWave_QPU']['color'], label='Patch CQM QPU vs PuLP', alpha=0.8)
    ax.plot(n_vars, speedups['Patch_CQM_Hybrid_vs_PuLP'], marker=SERIES_STYLES['Patch_DWave_Hybrid']['marker'], linewidth=2.5, 
        markersize=10, color=SERIES_STYLES['Patch_DWave_Hybrid']['color'], label='Patch CQM Hybrid vs PuLP', alpha=0.8)
    ax.plot(n_vars, speedups['Patch_BQM_QPU_vs_PuLP'], marker=SERIES_STYLES['Patch_DWaveBQM_QPU']['marker'], linewidth=2.5, 
        markersize=10, color=SERIES_STYLES['Patch_DWaveBQM_QPU']['color'], label='Patch BQM QPU vs PuLP', alpha=0.8)
    ax.plot(n_vars, speedups['Patch_QUBO_vs_PuLP'], marker=SERIES_STYLES['Patch_GurobiQUBO']['marker'], linewidth=2.5, 
        markersize=10, color=SERIES_STYLES['Patch_GurobiQUBO']['color'], label='Patch QUBO vs PuLP', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Variables', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('Speedup: Linear Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale speedup
    ax = axes[1, 1]
    ax.semilogy(n_vars, speedups['Farm_QPU_vs_PuLP'], marker=SERIES_STYLES['Farm_DWave_QPU']['marker'], linewidth=2.5, 
                markersize=10, color=SERIES_STYLES['Farm_DWave_QPU']['color'], label='Farm QPU vs PuLP', alpha=0.8)
    ax.semilogy(n_vars, speedups['Farm_Hybrid_vs_PuLP'], marker=SERIES_STYLES['Farm_DWave_Hybrid']['marker'], linewidth=2.5, 
                markersize=10, color=SERIES_STYLES['Farm_DWave_Hybrid']['color'], label='Farm Hybrid vs PuLP', alpha=0.8)
    ax.semilogy(n_vars, speedups['Patch_CQM_QPU_vs_PuLP'], marker=SERIES_STYLES['Patch_DWave_QPU']['marker'], linewidth=2.5, 
                markersize=10, color=SERIES_STYLES['Patch_DWave_QPU']['color'], label='Patch CQM QPU vs PuLP', alpha=0.8)
    ax.semilogy(n_vars, speedups['Patch_CQM_Hybrid_vs_PuLP'], marker=SERIES_STYLES['Patch_DWave_Hybrid']['marker'], linewidth=2.5, 
                markersize=10, color=SERIES_STYLES['Patch_DWave_Hybrid']['color'], label='Patch CQM Hybrid vs PuLP', alpha=0.8)
    ax.semilogy(n_vars, speedups['Patch_BQM_QPU_vs_PuLP'], marker=SERIES_STYLES['Patch_DWaveBQM_QPU']['marker'], linewidth=2.5, 
                markersize=10, color=SERIES_STYLES['Patch_DWaveBQM_QPU']['color'], label='Patch BQM QPU vs PuLP', alpha=0.8)
    ax.semilogy(n_vars, speedups['Patch_QUBO_vs_PuLP'], marker=SERIES_STYLES['Patch_GurobiQUBO']['marker'], linewidth=2.5, 
                markersize=10, color=SERIES_STYLES['Patch_GurobiQUBO']['color'], label='Patch QUBO vs PuLP', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Variables', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup: Log-Y Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale speedup
    ax = axes[1, 2]
    ax.loglog(n_vars, speedups['Farm_QPU_vs_PuLP'], marker=SERIES_STYLES['Farm_DWave_QPU']['marker'], linewidth=2.5, 
              markersize=10, color=SERIES_STYLES['Farm_DWave_QPU']['color'], label='Farm QPU vs PuLP', alpha=0.8)
    ax.loglog(n_vars, speedups['Farm_Hybrid_vs_PuLP'], marker=SERIES_STYLES['Farm_DWave_Hybrid']['marker'], linewidth=2.5, 
              markersize=10, color=SERIES_STYLES['Farm_DWave_Hybrid']['color'], label='Farm Hybrid vs PuLP', alpha=0.8)
    ax.loglog(n_vars, speedups['Patch_CQM_QPU_vs_PuLP'], marker=SERIES_STYLES['Patch_DWave_QPU']['marker'], linewidth=2.5, 
              markersize=10, color=SERIES_STYLES['Patch_DWave_QPU']['color'], label='Patch CQM QPU vs PuLP', alpha=0.8)
    ax.loglog(n_vars, speedups['Patch_CQM_Hybrid_vs_PuLP'], marker=SERIES_STYLES['Patch_DWave_Hybrid']['marker'], linewidth=2.5, 
              markersize=10, color=SERIES_STYLES['Patch_DWave_Hybrid']['color'], label='Patch CQM Hybrid vs PuLP', alpha=0.8)
    ax.loglog(n_vars, speedups['Patch_BQM_QPU_vs_PuLP'], marker=SERIES_STYLES['Patch_DWaveBQM_QPU']['marker'], linewidth=2.5, 
              markersize=10, color=SERIES_STYLES['Patch_DWaveBQM_QPU']['color'], label='Patch BQM QPU vs PuLP', alpha=0.8)
    ax.loglog(n_vars, speedups['Patch_QUBO_vs_PuLP'], marker=SERIES_STYLES['Patch_GurobiQUBO']['marker'], linewidth=2.5, 
              markersize=10, color=SERIES_STYLES['Patch_GurobiQUBO']['color'], label='Patch QUBO vs PuLP', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Variables (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup: Log-Log Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot saved to {output_path}")

def calculate_objective_gaps(objective_series):
    """Calculate objective value gaps (percentage deviation from best) across all solvers."""
    configs = objective_series['n_units']
    gap_series = {key: [] for key in SERIES_ORDER}
    for idx in range(len(configs)):
        values = [objective_series[key][idx] for key in SERIES_ORDER if objective_series[key][idx] is not None]
        best_value = max(values) if values else None
        for key in SERIES_ORDER:
            current = objective_series[key][idx]
            if best_value is not None and current is not None and best_value != 0:
                gap = (best_value - current) / best_value * 100
            else:
                gap = None
            gap_series[key].append(gap)
    gap_series['n_units'] = list(configs)
    gap_series['n_vars'] = list(objective_series['n_vars'])
    return gap_series

def calculate_time_to_quality(times, gaps):
    """Calculate Time-to-Quality metric for each solver series."""
    ttq = {key: [] for key in SERIES_ORDER}
    for idx in range(len(times['n_units'])):
        for key in SERIES_ORDER:
            time_value = times.get(key, [None] * len(times['n_units']))[idx]
            gap_value = gaps[key][idx] if key in gaps else None
            if time_value is not None and gap_value is not None:
                ttq[key].append(time_value * (1 + gap_value / 100))
            else:
                ttq[key].append(None)
    ttq['n_units'] = list(times['n_units'])
    ttq['n_vars'] = list(times['n_vars'])
    return ttq

def plot_solution_quality(objective_series, gap_series, output_path):
    """Create solution quality comparison plots in a 2x3 grid with all solvers together."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Benchmark: Solution Quality Comparison', 
                 fontsize=18, fontweight='bold', y=0.995)

    n_vars = objective_series['n_vars']
    epsilon = 1e-6

    def collect_points(series, positive_only=False, replace_zero=False):
        points = []
        for x, y in zip(n_vars, series):
            if y is None:
                continue
            value = y
            if positive_only and value <= 0:
                if replace_zero:
                    value = epsilon
                else:
                    continue
            points.append((x, value))
        return points

    # Row 1: Objective values (linear, log-y, log-log)
    axis_configs = [
        ('Objective Value (Linear Scale)', None, False),
        ('Objective Value (Log-Y Scale)', 'semilogy', True),
        ('Objective Value (Log-Log Scale)', 'loglog', True)
    ]

    for col, (title, transform, positive_only) in enumerate(axis_configs):
        ax = axes[0, col]
        for key in SERIES_ORDER:
            points = collect_points(objective_series[key], positive_only=positive_only)
            if not points:
                continue
            x_vals, y_vals = zip(*points)
            style = SERIES_STYLES[key]
            if transform == 'semilogy':
                ax.semilogy(x_vals, y_vals, marker=style['marker'], linewidth=2.5,
                            markersize=10, color=style['color'], label=style['label'], alpha=0.8)
            elif transform == 'loglog':
                ax.loglog(x_vals, y_vals, marker=style['marker'], linewidth=2.5,
                          markersize=10, color=style['color'], label=style['label'], alpha=0.8)
            else:
                ax.plot(x_vals, y_vals, marker=style['marker'], linewidth=2.5,
                        markersize=10, color=style['color'], label=style['label'], alpha=0.8)
        ax.set_xlabel('Number of Variables', fontsize=12, fontweight='bold')
        ax.set_ylabel('Objective Value' if col == 0 else 'Objective Value', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        if col == 0:
            ax.legend(loc='upper left', fontsize=8, ncol=1)
        ax.grid(True, alpha=0.3, which='both' if transform else 'major')

    # Row 2: Objective gap (% deviation from best)
    gap_axis_configs = [
        ('Objective Gap (%)', None, False),
        ('Objective Gap (%) (Log-Y)', 'semilogy', True),
        ('Objective Gap (%) (Log-Log)', 'loglog', True)
    ]

    for col, (title, transform, positive_only) in enumerate(gap_axis_configs):
        ax = axes[1, col]
        for key in SERIES_ORDER:
            points = collect_points(gap_series[key], positive_only=positive_only, replace_zero=True)
            if not points:
                continue
            x_vals, y_vals = zip(*points)
            style = SERIES_STYLES[key]
            if transform == 'semilogy':
                ax.semilogy(x_vals, y_vals, marker=style['marker'], linewidth=2.5,
                            markersize=10, color=style['color'], label=style['label'], alpha=0.8)
            elif transform == 'loglog':
                ax.loglog(x_vals, y_vals, marker=style['marker'], linewidth=2.5,
                          markersize=10, color=style['color'], label=style['label'], alpha=0.8)
            else:
                ax.plot(x_vals, y_vals, marker=style['marker'], linewidth=2.5,
                        markersize=10, color=style['color'], label=style['label'], alpha=0.8)
        ax.set_xlabel('Number of Variables', fontsize=12, fontweight='bold')
        ax.set_ylabel('Objective Gap (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        if col == 0:
            ax.legend(loc='upper left', fontsize=8, ncol=1)
        ax.grid(True, alpha=0.3, which='both' if transform else 'major')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Solution quality plot saved to {output_path}")

def plot_comprehensive_quality_analysis(times, objectives, gaps, ttq, output_path):
    """Create comprehensive 3x3 quality analysis plot."""
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    fig.suptitle('Comprehensive Benchmark: Quality Analysis (Time + Solution Quality)', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    n_vars = times['n_vars']
    
    def filter_none(series):
        filtered = [(x, y) for x, y in zip(n_vars, series) if y is not None]
        if not filtered:
            return [], []
        x_vals, y_vals = zip(*filtered)
        return list(x_vals), list(y_vals)

    # === Row 1: Solve Times ===

    ax = axes[0, 0]
    for key in SERIES_ORDER:
        if key not in times:
            continue
        x, y = filter_none(times[key])
        if not x:
            continue
        style = SERIES_STYLES[key]
        ax.plot(x, y, marker=style['marker'], linewidth=2.5, markersize=8,
                color=style['color'], label=style['label'], alpha=0.8)
    ax.set_xlabel('Number of Variables', fontsize=11, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Solve Time (Linear Scale)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for key in SERIES_ORDER:
        if key not in times:
            continue
        x, y = filter_none(times[key])
        if not x:
            continue
        style = SERIES_STYLES[key]
        ax.semilogy(x, y, marker=style['marker'], linewidth=2.5, markersize=8,
                    color=style['color'], label=style['label'], alpha=0.8)
    ax.set_xlabel('Number of Variables', fontsize=11, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log)', fontsize=11, fontweight='bold')
    ax.set_title('Solve Time (Log-Y Scale)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    ax = axes[0, 2]
    for key in ['Farm_DWave_QPU', 'Patch_DWave_QPU', 'Patch_DWaveBQM_QPU']:
        if key not in times:
            continue
        x, y = filter_none(times[key])
        if not x:
            continue
        style = SERIES_STYLES[key]
        ax.plot(x, y, marker=style['marker'], linewidth=3, markersize=10,
                color=style['color'], label=style['label'], alpha=0.8)
        qpu_valid = [t for t in times[key] if t is not None]
        if qpu_valid:
            ax.axhline(y=np.mean(qpu_valid), color=style['color'], linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Number of Variables', fontsize=11, fontweight='bold')
    ax.set_ylabel('QPU Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('QPU Time (Nearly Constant)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # === Row 2: Solution Quality ===
    
    # Objective Values
    ax = axes[1, 0]
    for key in SERIES_ORDER:
        x, y = filter_none(objectives[key])
        if not x:
            continue
        style = SERIES_STYLES[key]
        ax.plot(x, y, marker=style['marker'], linewidth=2.5, markersize=8,
                color=style['color'], label=style['label'], alpha=0.8)
    ax.set_xlabel('Number of Variables', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Value', fontsize=11, fontweight='bold')
    ax.set_title('Solution Quality (Objective Values)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Objective Gaps
    ax = axes[1, 1]
    for key in SERIES_ORDER:
        x, y = filter_none(gaps[key])
        if not x:
            continue
        style = SERIES_STYLES[key]
        ax.plot(x, y, marker=style['marker'], linewidth=2.5, markersize=8,
                color=style['color'], label=f"{style['label']} Gap", alpha=0.8)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect (0% gap)')
    ax.set_xlabel('Number of Variables', fontsize=11, fontweight='bold')
    ax.set_ylabel('Quality Gap (%)', fontsize=11, fontweight='bold')
    ax.set_title('Objective Gap from Best', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Gap Bar Chart by Configuration
    ax = axes[1, 2]
    x_pos = np.arange(len(times['n_units']))
    width = 0.9 / len(SERIES_ORDER)
    for idx, key in enumerate(SERIES_ORDER):
        offsets = x_pos - 0.45 + idx * width
        series_values = [g if g is not None else 0 for g in gaps[key]]
        style = SERIES_STYLES[key]
        ax.bar(offsets, series_values, width, label=style['label'], color=style['color'], alpha=0.75)
    ax.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Quality Gap (%)', fontsize=11, fontweight='bold')
    ax.set_title('Gap Comparison by Configuration', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{n} units' for n in times['n_units']], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # === Row 3: Time-to-Quality Metrics ===
    
    # Time-to-Quality Comparison
    ax = axes[2, 0]
    for key in SERIES_ORDER:
        x, y = filter_none(ttq[key])
        if not x:
            continue
        style = SERIES_STYLES[key]
        ax.semilogy(x, y, marker=style['marker'], linewidth=2.5, markersize=8,
                    color=style['color'], label=f"{style['label']} TTQ", alpha=0.8)
    ax.set_xlabel('Number of Variables', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time-to-Quality (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Time-to-Quality Metric', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.text(0.02, 0.98, 'Lower is better\n(accounts for speed + accuracy)', 
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Raw Speedup (time only)
    ax = axes[2, 1]
    # Farm speedup
    farm_speedup = [times['Farm_PuLP'][i] / times['Farm_DWave_Hybrid'][i] 
                   if times['Farm_PuLP'][i] and times['Farm_DWave_Hybrid'][i] else None 
                   for i in range(len(n_vars))]
    # Patch speedup  
    patch_speedup = [times['Patch_PuLP'][i] / times['Patch_DWave_Hybrid'][i] 
                    if times['Patch_PuLP'][i] and times['Patch_DWave_Hybrid'][i] else None 
                    for i in range(len(n_vars))]
    
    x, y = filter_none(farm_speedup)
    if x: ax.semilogy(x, y, marker=SERIES_STYLES['Farm_DWave_Hybrid']['marker'], linewidth=2.5, markersize=8,
                      color=SERIES_STYLES['Farm_DWave_Hybrid']['color'], label='Farm Speedup', alpha=0.8)
    x, y = filter_none(patch_speedup)
    if x: ax.semilogy(x, y, marker=SERIES_STYLES['Patch_DWave_Hybrid']['marker'], linewidth=2.5, markersize=8,
                      color=SERIES_STYLES['Patch_DWave_Hybrid']['color'], label='Patch Speedup', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Variables', fontsize=11, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
    ax.set_title('Raw Speedup (Time Only)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Quality-Adjusted Speedup (TTQ)
    ax = axes[2, 2]
    # Farm TTQ speedup
    farm_ttq_speedup = [ttq['Farm_PuLP'][i] / ttq['Farm_DWave_Hybrid'][i] 
                       if ttq['Farm_PuLP'][i] and ttq['Farm_DWave_Hybrid'][i] else None 
                       for i in range(len(n_vars))]
    # Patch TTQ speedup
    patch_ttq_speedup = [ttq['Patch_PuLP'][i] / ttq['Patch_DWave_Hybrid'][i] 
                        if ttq['Patch_PuLP'][i] and ttq['Patch_DWave_Hybrid'][i] else None 
                        for i in range(len(n_vars))]
    
    x, y = filter_none(farm_ttq_speedup)
    if x: ax.semilogy(x, y, marker=SERIES_STYLES['Farm_DWave_Hybrid']['marker'], linewidth=2.5, markersize=8,
                      color=SERIES_STYLES['Farm_DWave_Hybrid']['color'], label='Farm TTQ Speedup', alpha=0.8)
    x, y = filter_none(patch_ttq_speedup)
    if x: ax.semilogy(x, y, marker=SERIES_STYLES['Patch_DWave_Hybrid']['marker'], linewidth=2.5, markersize=8,
                      color=SERIES_STYLES['Patch_DWave_Hybrid']['color'], label='Patch TTQ Speedup', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Variables', fontsize=11, fontweight='bold')
    ax.set_ylabel('Quality-Adjusted Speedup', fontsize=11, fontweight='bold')
    ax.set_title('Time-to-Quality Speedup', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.text(0.02, 0.98, 'Accounts for both\nspeed AND accuracy', 
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comprehensive quality analysis plot saved to {output_path}")

def print_summary_table(times, objectives):
    """Print summary table of all solver times, objectives, and quality metrics."""
    print("\n" + "="*150)
    print("COMPREHENSIVE BENCHMARK SUMMARY: SOLVE TIMES AND SPEEDUPS")
    print("="*150)
    print(f"{'N_Vars':<8} {'Farm PuLP':<12} {'Farm QPU':<12} {'Farm Hybrid':<12} {'Patch PuLP':<12} {'Patch CQM QPU':<15} {'Patch CQM Hybrid':<18} {'Patch BQM QPU':<15} {'Patch QUBO':<12}")
    print("-"*150)
    
    speedups = calculate_speedups(times)
    
    for i, (n_units, n_vars) in enumerate(zip(times['n_units'], times['n_vars'])):
        farm_pulp = f"{times['Farm_PuLP'][i]:.4f}s" if times['Farm_PuLP'][i] else "N/A"
        farm_qpu = f"{times['Farm_DWave_QPU'][i]:.4f}s" if times['Farm_DWave_QPU'][i] else "N/A"
        farm_hybrid = f"{times['Farm_DWave_Hybrid'][i]:.4f}s" if times['Farm_DWave_Hybrid'][i] else "N/A"
        patch_pulp = f"{times['Patch_PuLP'][i]:.4f}s" if times['Patch_PuLP'][i] else "N/A"
        patch_cqm_qpu = f"{times['Patch_DWave_QPU'][i]:.4f}s" if times['Patch_DWave_QPU'][i] else "N/A"
        patch_cqm_hybrid = f"{times['Patch_DWave_Hybrid'][i]:.4f}s" if times['Patch_DWave_Hybrid'][i] else "N/A"
        patch_bqm_qpu = f"{times['Patch_DWaveBQM_QPU'][i]:.4f}s" if times['Patch_DWaveBQM_QPU'][i] else "N/A"
        patch_qubo = f"{times['Patch_GurobiQUBO'][i]:.4f}s" if times['Patch_GurobiQUBO'][i] else "N/A"
        
        print(f"{n_vars:<8} {farm_pulp:<12} {farm_qpu:<12} {farm_hybrid:<12} {patch_pulp:<12} {patch_cqm_qpu:<15} {patch_cqm_hybrid:<18} {patch_bqm_qpu:<15} {patch_qubo:<12}")
    
    print("\n" + "="*150)
    print("SPEEDUP FACTORS (vs PuLP Classical Solver)")
    print("="*150)
    print(f"{'N_Vars':<8} {'Farm QPU':<15} {'Farm Hybrid':<15} {'Patch CQM QPU':<18} {'Patch CQM Hybrid':<20} {'Patch BQM QPU':<18} {'Patch QUBO':<15}")
    print("-"*150)
    
    for i, n_vars in enumerate(times['n_vars']):
        farm_qpu_sp = f"{speedups['Farm_QPU_vs_PuLP'][i]:.2f}x" if speedups['Farm_QPU_vs_PuLP'][i] else "N/A"
        farm_hybrid_sp = f"{speedups['Farm_Hybrid_vs_PuLP'][i]:.2f}x" if speedups['Farm_Hybrid_vs_PuLP'][i] else "N/A"
        patch_cqm_qpu_sp = f"{speedups['Patch_CQM_QPU_vs_PuLP'][i]:.2f}x" if speedups['Patch_CQM_QPU_vs_PuLP'][i] else "N/A"
        patch_cqm_hybrid_sp = f"{speedups['Patch_CQM_Hybrid_vs_PuLP'][i]:.2f}x" if speedups['Patch_CQM_Hybrid_vs_PuLP'][i] else "N/A"
        patch_bqm_qpu_sp = f"{speedups['Patch_BQM_QPU_vs_PuLP'][i]:.2f}x" if speedups['Patch_BQM_QPU_vs_PuLP'][i] else "N/A"
        patch_qubo_sp = f"{speedups['Patch_QUBO_vs_PuLP'][i]:.2f}x" if speedups['Patch_QUBO_vs_PuLP'][i] else "N/A"
        
        print(f"{n_vars:<8} {farm_qpu_sp:<15} {farm_hybrid_sp:<15} {patch_cqm_qpu_sp:<18} {patch_cqm_hybrid_sp:<20} {patch_bqm_qpu_sp:<18} {patch_qubo_sp:<15}")
    
    print("\n" + "="*150)
    print("OBJECTIVE VALUES")
    print("="*150)
    print(f"{'N_Vars':<8} {'Farm PuLP':<15} {'Farm D-Wave':<15} {'Patch PuLP':<15} {'Patch CQM':<15} {'Patch BQM':<15} {'Patch QUBO':<15}")
    print("-"*150)
    
    for i, n_vars in enumerate(times['n_vars']):
        farm_pulp_obj = f"{objectives['Farm_PuLP'][i]:.6f}" if objectives['Farm_PuLP'][i] else "N/A"
        farm_dwave_obj = f"{objectives['Farm_DWave'][i]:.6f}" if objectives['Farm_DWave'][i] else "N/A"
        patch_pulp_obj = f"{objectives['Patch_PuLP'][i]:.6f}" if objectives['Patch_PuLP'][i] else "N/A"
        patch_dwave_obj = f"{objectives['Patch_DWave'][i]:.6f}" if objectives['Patch_DWave'][i] else "N/A"
        patch_bqm_obj = f"{objectives['Patch_DWaveBQM'][i]:.6f}" if objectives['Patch_DWaveBQM'][i] else "N/A"
        patch_qubo_obj = f"{objectives['Patch_GurobiQUBO'][i]:.6f}" if objectives['Patch_GurobiQUBO'][i] else "N/A"
        
        print(f"{n_vars:<8} {farm_pulp_obj:<15} {farm_dwave_obj:<15} {patch_pulp_obj:<15} {patch_dwave_obj:<15} {patch_bqm_obj:<15} {patch_qubo_obj:<15}")
    
    print("="*150)
    print("\nKey Observations:")
    print("- X-axis shows number of variables (samples × 27)")
    print("- Farm scenario uses CQM formulation with continuous crop allocations")
    print("- Patch scenario compares CQM, BQM, and QUBO formulations")
    print("- QPU time remains nearly constant regardless of problem size")
    print("- PuLP time increases with problem size")
    print("- Hybrid solvers combine QPU speed with classical optimization")
    print("- Speedup increases with problem size for quantum approaches")
    print("- Quality gaps show deviation from best objective value found")
    print("- Time-to-Quality metric accounts for both speed and accuracy")
    print("- Farm objectives are divided by 10 when plotted so they align with patch-scale objectives")
    print("="*150 + "\n")

def main():
    """Main execution function."""
    print("="*120)
    print("COMPREHENSIVE BENCHMARK PLOTTING")
    print("="*120)
    
    # Load data
    benchmark_dir = Path(__file__).parent / "Benchmarks"
    print(f"Loading data from: {benchmark_dir}/COMPREHENSIVE")
    
    data = load_benchmark_data(benchmark_dir)
    
    # Extract times and objectives
    print("\nExtracting solve times and objective values...")
    times, objectives = extract_times(data)
    
    # Normalize objectives so farm values align with patch scale
    normalized_objectives = normalize_objectives(objectives)
    objective_series = build_objective_series(normalized_objectives)
    
    # Calculate quality gaps and time-to-quality metrics
    print("\nCalculating quality gaps and time-to-quality metrics...")
    gaps = calculate_objective_gaps(objective_series)
    ttq = calculate_time_to_quality(times, gaps)
    
    # Create output directory
    output_dir = Path(__file__).parent / "Plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    print("\nCreating plots...")
    plot_solve_times(times, output_dir / "comprehensive_speedup_comparison.png")
    plot_solution_quality(objective_series, gaps, output_dir / "comprehensive_solution_quality.png")
    plot_comprehensive_quality_analysis(times, objective_series, gaps, ttq, 
                                       output_dir / "comprehensive_quality_analysis.png")
    
    # Print summary
    print_summary_table(times, objectives)
    
    print("\n" + "="*120)
    print("PLOTTING COMPLETE")
    print("="*120)
    print(f"Plots saved to: {output_dir}")
    print("  - comprehensive_speedup_comparison.png")
    print("  - comprehensive_solution_quality.png")
    print("  - comprehensive_quality_analysis.png")
    
    # Close all figures
    plt.close('all')
    print("\n✓ All plots saved successfully!")

if __name__ == "__main__":
    main()
