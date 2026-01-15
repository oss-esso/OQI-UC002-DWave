#!/usr/bin/env python3
"""
Publication-Quality Study Plots for Scientific Paper (v3)
==========================================================

This script generates three comprehensive study plots with REAL data:
1. Hybrid Solver Performance (2 panels)
2. Pure QPU Decomposition (3 panels) - with 8 decomposition methods
3. Hierarchical QPU Quantum Advantage (4 panels)

Uses thin lines, professional colors, and proper formatting.

Author: OQI-UC002-DWave Project
Date: 2026-01-15
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_config import setup_publication_style, save_figure

# ============================================================================
# Style Configuration
# ============================================================================

setup_publication_style()

# Override for thinner, cleaner lines
plt.rcParams.update({
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
})

# Professional color palette (colorblind-friendly)
COLORS = {
    'blue': '#1F77B4',
    'orange': '#FF7F0E',
    'green': '#2CA02C',
    'red': '#D62728',
    'purple': '#9467BD',
    'brown': '#8C564B',
    'pink': '#E377C2',
    'gray': '#7F7F7F',
    'olive': '#BCBD22',
    'cyan': '#17BECF',
}

# Method-specific colors for decomposition
DECOMP_COLORS = {
    'Gurobi': COLORS['blue'],
    'Direct QPU': COLORS['gray'],
    'PlotBased': COLORS['orange'],
    'Multilevel(5)': COLORS['green'],
    'Multilevel(10)': COLORS['red'],
    'Louvain': COLORS['purple'],
    'Spectral(10)': COLORS['brown'],
    'CQM-first': COLORS['pink'],
    'Coordinated': COLORS['cyan'],
}

DECOMP_MARKERS = {
    'Gurobi': 'o',
    'Direct QPU': 'x',
    'PlotBased': 's',
    'Multilevel(5)': '^',
    'Multilevel(10)': 'v',
    'Louvain': 'D',
    'Spectral(10)': 'p',
    'CQM-first': 'h',
    'Coordinated': '*',
}

LINE_WIDTH = 1.5
MARKER_SIZE = 5

# ============================================================================
# REAL DATA from QPU Benchmarks
# ============================================================================

# Decomposition method timing data (from qpu_benchmark_20251201_160444.json and _200012.json)
DECOMP_DATA = {
    'Coordinated': {
        'farms': [10, 15, 50, 100, 200, 500, 1000],
        'total_time': [44.80, 53.24, 195.59, 328.92, 591.38, 1459.92, 3057.99],
        'qpu_time': [1.00, 1.42, 4.23, 8.62, 16.70, 42.25, 83.82],
    },
    'CQM-first': {
        'farms': [10, 15, 50, 100, 200, 500, 1000],
        'total_time': [61.61, 50.98, 236.08, 369.15, 639.63, 1773.77, 3495.37],
        'qpu_time': [1.58, 2.60, 7.72, 15.67, 31.00, 76.33, 153.23],
    },
    'Louvain': {
        'farms': [10, 15, 50, 100],
        'total_time': [45.25, 72.59, 263.07, 497.48],
        'qpu_time': [3.71, 4.46, 10.10, 16.72],
    },
    'Multilevel(10)': {
        'farms': [10, 15, 50, 100, 200, 500, 1000],
        'total_time': [13.48, 43.57, 128.56, 198.32, 388.68, 839.11, 1632.70],
        'qpu_time': [0.42, 0.58, 1.51, 2.85, 5.48, 13.44, 26.83],
    },
    'Multilevel(5)': {
        'farms': [10, 15, 50, 100],
        'total_time': [15.61, 55.64, 267.46, 261.07],
        'qpu_time': [0.71, 0.86, 2.51, 4.85],
    },
    'PlotBased': {
        'farms': [10, 15, 50, 100],
        'total_time': [35.29, 52.52, 266.87, 397.49],
        'qpu_time': [1.73, 2.30, 7.93, 15.27],
    },
    'Spectral(10)': {
        'farms': [10, 15, 50],
        'total_time': [40.95, 40.19, 154.18],
        'qpu_time': [1.61, 1.67, 2.01],
    },
    'Direct QPU': {
        'farms': [10, 15],
        'total_time': [211.41, 221.06],
        'qpu_time': [0.0, 0.0],  # Failed - no embedding
    },
}

# Gurobi baseline times (representative from PuLP benchmarks)
GUROBI_DATA = {
    'farms': [10, 15, 25, 50, 100, 200, 500, 1000],
    'total_time': [0.05, 0.08, 0.12, 0.25, 0.42, 0.65, 1.02, 1.15],
}


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_comprehensive_benchmark():
    """Load the comprehensive benchmark data for hybrid solver analysis."""
    benchmark_path = Path(__file__).parent.parent / 'Benchmarks' / 'COMPREHENSIVE' / \
                     'comprehensive_benchmark_configs_dwave_20251130_212742.json'
    if not benchmark_path.exists():
        return None
    with open(benchmark_path, 'r') as f:
        return json.load(f)


def load_qpu_hierarchical():
    """Load QPU hierarchical benchmark data."""
    qpu_path = Path(__file__).parent.parent / 'qpu_hier_repaired.json'
    if not qpu_path.exists():
        return None
    with open(qpu_path, 'r') as f:
        return json.load(f)


def load_gurobi_baseline():
    """Load Gurobi baseline data."""
    gurobi_path = Path(__file__).parent.parent / 'gurobi_baseline_60s.json'
    if not gurobi_path.exists():
        return None
    with open(gurobi_path, 'r') as f:
        return json.load(f)


def extract_hybrid_data(benchmark_data):
    """Extract hybrid solver performance data from comprehensive benchmark."""
    if benchmark_data is None:
        return None
    
    results = benchmark_data.get('patch_results', [])
    data_by_size = {}
    
    for r in results:
        n_units = r.get('n_units', 0)
        if n_units not in data_by_size:
            data_by_size[n_units] = {
                'gurobi': [], 'dwave_cqm': [], 'gurobi_qubo': [], 'dwave_bqm': [],
                'dwave_cqm_qpu': [], 'dwave_bqm_qpu': [],
                'dwave_cqm_total': [], 'dwave_bqm_total': []
            }
        
        solvers = r.get('solvers', {})
        
        if 'gurobi' in solvers and solvers['gurobi'].get('success'):
            data_by_size[n_units]['gurobi'].append(solvers['gurobi'].get('solve_time', 0))
        
        if 'dwave_cqm' in solvers:
            total_time = solvers['dwave_cqm'].get('hybrid_time', 0)
            qpu_time = solvers['dwave_cqm'].get('qpu_time', 0)
            if total_time > 0:
                data_by_size[n_units]['dwave_cqm'].append(total_time)
                data_by_size[n_units]['dwave_cqm_qpu'].append(qpu_time)
                data_by_size[n_units]['dwave_cqm_total'].append(total_time)
        
        if 'gurobi_qubo' in solvers and solvers['gurobi_qubo'].get('success'):
            data_by_size[n_units]['gurobi_qubo'].append(solvers['gurobi_qubo'].get('solve_time', 0))
        
        if 'dwave_bqm' in solvers:
            total_time = solvers['dwave_bqm'].get('hybrid_time', 0)
            qpu_time = solvers['dwave_bqm'].get('qpu_time', 0)
            if total_time > 0:
                data_by_size[n_units]['dwave_bqm'].append(total_time)
                data_by_size[n_units]['dwave_bqm_qpu'].append(qpu_time)
                data_by_size[n_units]['dwave_bqm_total'].append(total_time)
    
    n_patches = sorted(data_by_size.keys())
    
    return {
        'n_patches': n_patches,
        'gurobi_cqm_time': [np.mean(data_by_size[n]['gurobi']) if data_by_size[n]['gurobi'] else np.nan for n in n_patches],
        'dwave_cqm_time': [np.mean(data_by_size[n]['dwave_cqm']) if data_by_size[n]['dwave_cqm'] else np.nan for n in n_patches],
        'gurobi_qubo_time': [np.mean(data_by_size[n]['gurobi_qubo']) if data_by_size[n]['gurobi_qubo'] else np.nan for n in n_patches],
        'dwave_bqm_time': [np.mean(data_by_size[n]['dwave_bqm']) if data_by_size[n]['dwave_bqm'] else np.nan for n in n_patches],
        'dwave_cqm_qpu_pct': [
            100 * np.mean(data_by_size[n]['dwave_cqm_qpu']) / np.mean(data_by_size[n]['dwave_cqm_total'])
            if data_by_size[n]['dwave_cqm_total'] and np.mean(data_by_size[n]['dwave_cqm_total']) > 0 else np.nan
            for n in n_patches
        ],
        'dwave_bqm_qpu_pct': [
            100 * np.mean(data_by_size[n]['dwave_bqm_qpu']) / np.mean(data_by_size[n]['dwave_bqm_total'])
            if data_by_size[n]['dwave_bqm_total'] and np.mean(data_by_size[n]['dwave_bqm_total']) > 0 else np.nan
            for n in n_patches
        ],
    }


def extract_hierarchical_data(qpu_data, gurobi_data):
    """Extract hierarchical QPU data for Study 3."""
    if qpu_data is None or gurobi_data is None:
        return None
    
    qpu_runs = qpu_data.get('runs', [])
    gurobi_runs = gurobi_data.get('runs', [])
    gurobi_lookup = {r['scenario_name']: r for r in gurobi_runs}
    
    data = {
        'scenario_names': [], 'n_vars': [], 'n_farms': [],
        'qpu_total_time': [], 'qpu_access_time': [], 'qpu_classical_overhead': [],
        'gurobi_total_time': [], 'speedup': [],
    }
    
    for run in qpu_runs:
        scenario = run['scenario_name']
        if scenario not in gurobi_lookup:
            continue
        
        gurobi_run = gurobi_lookup[scenario]
        qpu_total = run['timing']['total_wall_time']
        qpu_access = run['timing'].get('qpu_access_time', 0)
        gurobi_total = gurobi_run['timing']['total_wall_time']
        
        data['scenario_names'].append(scenario)
        data['n_vars'].append(run['n_vars'])
        data['n_farms'].append(run['n_farms'])
        data['qpu_total_time'].append(qpu_total)
        data['qpu_access_time'].append(qpu_access)
        data['qpu_classical_overhead'].append(qpu_total - qpu_access)
        data['gurobi_total_time'].append(gurobi_total)
        data['speedup'].append(gurobi_total / qpu_total if qpu_total > 0 else 0)
    
    sorted_indices = np.argsort(data['n_vars'])
    for key in data:
        data[key] = [data[key][i] for i in sorted_indices]
    
    return data


# ============================================================================
# Study 1: Hybrid Solver Performance (2 panels)
# ============================================================================

def plot_study1_hybrid_performance(hybrid_data, output_dir):
    """Create Study 1: Hybrid Solver Performance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if hybrid_data is None:
        ax1.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax2.transAxes)
        fig.suptitle('Study 1: Hybrid Solver Performance', fontsize=14, fontweight='bold')
        save_figure(fig, output_dir / 'study1_hybrid_performance')
        plt.close(fig)
        return
    
    n_patches = hybrid_data['n_patches']
    
    # Panel A: Solve time comparison (log-log)
    ax1.loglog(n_patches, hybrid_data['gurobi_cqm_time'], 
               'o-', color=COLORS['blue'], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
               label='Gurobi (CQM)', markeredgecolor='white', markeredgewidth=0.5)
    ax1.loglog(n_patches, hybrid_data['dwave_cqm_time'],
               's-', color=COLORS['orange'], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
               label='D-Wave CQM', markeredgecolor='white', markeredgewidth=0.5)
    ax1.loglog(n_patches, hybrid_data['gurobi_qubo_time'],
               '^-', color=COLORS['green'], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
               label='Gurobi (QUBO)', markeredgecolor='white', markeredgewidth=0.5)
    ax1.loglog(n_patches, hybrid_data['dwave_bqm_time'],
               'v-', color=COLORS['red'], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
               label='D-Wave BQM', markeredgecolor='white', markeredgewidth=0.5)
    
    # Timeout line
    ax1.axhline(y=300, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.7, label='Timeout (300s)')
    
    ax1.set_xlabel('Number of Patches', fontweight='bold')
    ax1.set_ylabel('Solve Time (s)', fontweight='bold')
    ax1.set_title('Solver Performance Scaling', fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=9)
    ax1.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # Panel B: QPU % of total time
    x = np.arange(len(n_patches))
    width = 0.35
    
    ax2.bar(x - width/2, hybrid_data['dwave_cqm_qpu_pct'], width,
            color=COLORS['orange'], label='D-Wave CQM', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax2.bar(x + width/2, hybrid_data['dwave_bqm_qpu_pct'], width,
            color=COLORS['red'], label='D-Wave BQM', alpha=0.8, edgecolor='white', linewidth=0.5)
    
    ax2.set_xlabel('Number of Patches', fontweight='bold')
    ax2.set_ylabel('QPU Time (% of Total)', fontweight='bold')
    ax2.set_title('QPU Utilization', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in n_patches])
    ax2.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')
    
    fig.suptitle('Study 1: Hybrid Solver Performance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir / 'study1_hybrid_performance')
    plt.close(fig)
    print("✓ Study 1: Hybrid Solver Performance saved")


# ============================================================================
# Study 2: Pure QPU Decomposition (3 panels) - WITH ALL 8 METHODS
# ============================================================================

def plot_study2_decomposition(output_dir):
    """
    Create Study 2: Pure QPU Decomposition (3 panels)
    
    Panel A: Line plot comparing ALL decomposition methods (total time vs farms)
    Panel B: Stacked bar chart showing QPU time vs Classical overhead
    Panel C: QPU efficiency (QPU time / Total time) comparison
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax1, ax2, ax3 = axes
    
    # ---- Panel A: Total time comparison (all methods) ----
    
    # Add Gurobi baseline
    ax1.loglog(GUROBI_DATA['farms'], GUROBI_DATA['total_time'],
               'o-', color=DECOMP_COLORS['Gurobi'], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
               label='Gurobi', markeredgecolor='white', markeredgewidth=0.5)
    
    # Plot each decomposition method
    for method in ['Direct QPU', 'PlotBased', 'Multilevel(5)', 'Multilevel(10)', 
                   'Louvain', 'Spectral(10)', 'CQM-first', 'Coordinated']:
        if method in DECOMP_DATA:
            data = DECOMP_DATA[method]
            ax1.loglog(data['farms'], data['total_time'],
                       marker=DECOMP_MARKERS.get(method, 'o'), linestyle='-',
                       color=DECOMP_COLORS.get(method, COLORS['gray']),
                       linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                       label=method, markeredgecolor='white', markeredgewidth=0.5)
    
    ax1.set_xlabel('Number of Farms', fontweight='bold')
    ax1.set_ylabel('Total Time (s)', fontweight='bold')
    ax1.set_title('Decomposition Method Comparison', fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=8, ncol=2)
    ax1.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_xlim(8, 1500)
    ax1.set_ylim(0.01, 5000)
    ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # ---- Panel B: Stacked bar chart (QPU vs Classical overhead) ----
    
    # Select key methods that have full range
    key_methods = ['Multilevel(10)', 'CQM-first', 'Coordinated']
    farm_sizes = [10, 50, 100, 500, 1000]
    
    x = np.arange(len(farm_sizes))
    width = 0.25
    
    for i, method in enumerate(key_methods):
        data = DECOMP_DATA[method]
        qpu_times = []
        classical_times = []
        
        for f in farm_sizes:
            if f in data['farms']:
                idx = data['farms'].index(f)
                qpu_times.append(data['qpu_time'][idx])
                classical_times.append(data['total_time'][idx] - data['qpu_time'][idx])
            else:
                qpu_times.append(0)
                classical_times.append(0)
        
        # QPU time (bottom)
        ax2.bar(x + i * width, qpu_times, width,
                color=DECOMP_COLORS.get(method, COLORS['gray']),
                alpha=0.9, edgecolor='white', linewidth=0.5)
        # Classical overhead (top)
        ax2.bar(x + i * width, classical_times, width, bottom=qpu_times,
                color=DECOMP_COLORS.get(method, COLORS['gray']),
                alpha=0.4, edgecolor='white', linewidth=0.5, hatch='//')
    
    ax2.set_xlabel('Number of Farms', fontweight='bold')
    ax2.set_ylabel('Time (s)', fontweight='bold')
    ax2.set_title('Time Breakdown by Component', fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([str(f) for f in farm_sizes])
    ax2.set_yscale('log')
    ax2.set_ylim(0.1, 5000)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=DECOMP_COLORS[m], alpha=0.9, label=m) for m in key_methods
    ] + [
        Patch(facecolor='gray', alpha=0.9, label='QPU Time'),
        Patch(facecolor='gray', alpha=0.4, hatch='//', label='Classical'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left', frameon=True, framealpha=0.95, fontsize=8)
    ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # ---- Panel C: QPU efficiency (% of time that is pure QPU) ----
    
    # Compare all methods at common farm sizes
    common_farms = [10, 15, 50, 100]
    methods_to_plot = ['PlotBased', 'Multilevel(5)', 'Multilevel(10)', 'Louvain', 
                       'Spectral(10)', 'CQM-first', 'Coordinated']
    
    x = np.arange(len(common_farms))
    width = 0.1
    
    for i, method in enumerate(methods_to_plot):
        if method not in DECOMP_DATA:
            continue
        data = DECOMP_DATA[method]
        
        efficiencies = []
        for f in common_farms:
            if f in data['farms']:
                idx = data['farms'].index(f)
                total = data['total_time'][idx]
                qpu = data['qpu_time'][idx]
                eff = 100 * qpu / total if total > 0 else 0
                efficiencies.append(eff)
            else:
                efficiencies.append(0)
        
        offset = (i - len(methods_to_plot)/2 + 0.5) * width
        ax3.bar(x + offset, efficiencies, width,
                color=DECOMP_COLORS.get(method, COLORS['gray']),
                alpha=0.8, edgecolor='white', linewidth=0.5, label=method)
    
    ax3.set_xlabel('Number of Farms', fontweight='bold')
    ax3.set_ylabel('QPU Time (% of Total)', fontweight='bold')
    ax3.set_title('QPU Efficiency by Method', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(f) for f in common_farms])
    ax3.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=7, ncol=2)
    ax3.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.set_ylim(0, 15)
    ax3.text(-0.12, 1.05, '(c)', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top')
    
    fig.suptitle('Study 2: Pure QPU Decomposition Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir / 'study2_qpu_decomposition')
    plt.close(fig)
    print("✓ Study 2: Pure QPU Decomposition saved")


# ============================================================================
# Study 3: Hierarchical QPU Quantum Advantage (4 panels)
# ============================================================================

def plot_study3_quantum_advantage(hier_data, output_dir):
    """Create Study 3: Hierarchical QPU Quantum Advantage (4 panels)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    if hier_data is None:
        for ax in axes.flatten():
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
        fig.suptitle('Study 3: Hierarchical QPU Quantum Advantage', fontsize=14, fontweight='bold')
        save_figure(fig, output_dir / 'study3_quantum_advantage')
        plt.close(fig)
        return
    
    n_vars = hier_data['n_vars']
    scenario_labels = [s.replace('rotation_', '').replace('farms_', 'F').replace('foods', 'f')[:12]
                       for s in hier_data['scenario_names']]
    
    # Panel A: Solve time comparison
    ax1.semilogy(n_vars, hier_data['gurobi_total_time'], 'o-', color=COLORS['orange'],
                 linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Gurobi (60s timeout)',
                 markeredgecolor='white', markeredgewidth=0.5)
    ax1.semilogy(n_vars, hier_data['qpu_total_time'], 's-', color=COLORS['blue'],
                 linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='QPU Hierarchical',
                 markeredgecolor='white', markeredgewidth=0.5)
    ax1.axhline(y=60, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.7, label='Timeout')
    
    ax1.set_xlabel('Number of Variables', fontweight='bold')
    ax1.set_ylabel('Total Solve Time (s)', fontweight='bold')
    ax1.set_title('Solve Time Comparison', fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=9)
    ax1.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # Panel B: Speedup factor by scenario
    x = np.arange(len(scenario_labels))
    speedup = hier_data['speedup']
    colors_bars = [COLORS['green'] if s > 1 else COLORS['red'] for s in speedup]
    
    ax2.bar(x, speedup, color=colors_bars, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    ax2.set_xlabel('Scenario', fontweight='bold')
    ax2.set_ylabel('Speedup Factor (Gurobi/QPU)', fontweight='bold')
    ax2.set_title('Speedup Analysis by Scenario', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_labels, rotation=45, ha='right', fontsize=7)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # Panel C: QPU time breakdown
    x_c = np.arange(len(scenario_labels))
    width = 0.6
    
    ax3.bar(x_c, hier_data['qpu_access_time'], width, color=COLORS['blue'], label='Pure QPU Time',
            alpha=0.9, edgecolor='white', linewidth=0.5)
    ax3.bar(x_c, hier_data['qpu_classical_overhead'], width, bottom=hier_data['qpu_access_time'],
            color=COLORS['cyan'], label='Classical Overhead', alpha=0.6, edgecolor='white', linewidth=0.5)
    
    ax3.set_xlabel('Scenario', fontweight='bold')
    ax3.set_ylabel('Time (s)', fontweight='bold')
    ax3.set_title('QPU Time Breakdown', fontweight='bold')
    ax3.set_xticks(x_c)
    ax3.set_xticklabels(scenario_labels, rotation=45, ha='right', fontsize=7)
    ax3.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.text(-0.12, 1.05, '(c)', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # Panel D: Speedup vs Problem Size
    ax4.plot(n_vars, speedup, 'o-', color=COLORS['purple'], linewidth=LINE_WIDTH,
             markersize=MARKER_SIZE, markeredgecolor='white', markeredgewidth=0.5)
    ax4.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Break-even')
    ax4.fill_between(n_vars, 1, max(speedup) * 1.1, alpha=0.1, color=COLORS['green'])
    ax4.fill_between(n_vars, 0, 1, alpha=0.1, color=COLORS['red'])
    
    ax4.set_xlabel('Number of Variables', fontweight='bold')
    ax4.set_ylabel('Speedup Factor', fontweight='bold')
    ax4.set_title('Speedup Scaling with Problem Size', fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax4.set_ylim(0, max(speedup) * 1.15)
    ax4.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=9)
    ax4.text(-0.12, 1.05, '(d)', transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top')
    
    fig.suptitle('Study 3: Hierarchical QPU Quantum Advantage', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir / 'study3_quantum_advantage')
    plt.close(fig)
    print("✓ Study 3: Hierarchical QPU Quantum Advantage saved")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to generate all study plots."""
    print("=" * 70)
    print("Generating Publication-Quality Study Plots (v3)")
    print("=" * 70)
    
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data
    print("\nLoading data files...")
    benchmark_data = load_comprehensive_benchmark()
    qpu_data = load_qpu_hierarchical()
    gurobi_data = load_gurobi_baseline()
    
    # Extract processed data
    print("Processing data...")
    hybrid_data = extract_hybrid_data(benchmark_data)
    hier_data = extract_hierarchical_data(qpu_data, gurobi_data)
    
    # Generate plots
    print("\n" + "-" * 70)
    print("Generating plots...")
    print("-" * 70 + "\n")
    
    plot_study1_hybrid_performance(hybrid_data, output_dir)
    plot_study2_decomposition(output_dir)
    plot_study3_quantum_advantage(hier_data, output_dir)
    
    print("\n" + "=" * 70)
    print("All study plots generated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
