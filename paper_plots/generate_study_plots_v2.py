#!/usr/bin/env python3
"""
Publication-Quality Study Plots for Scientific Paper
=====================================================

This script generates three comprehensive study plots:
1. Hybrid Solver Performance (2 panels)
2. Pure QPU Decomposition (2 panels)
3. Hierarchical QPU Quantum Advantage (4 panels)

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
from plot_config import setup_publication_style, save_figure, get_color_palette

# ============================================================================
# Style Configuration
# ============================================================================

# Setup publication style
setup_publication_style()

# Custom colors for consistency
COLORS = {
    'blue': '#1F77B4',
    'orange': '#FF7F0E', 
    'green': '#2CA02C',
    'red': '#D62728',
    'purple': '#9467BD',
    'brown': '#8C564B',
    'pink': '#E377C2',
    'gray': '#7F7F7F',
}

# Line and marker styles
LINE_WIDTH = 1.5
MARKER_SIZE = 6

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_comprehensive_benchmark():
    """Load the comprehensive benchmark data for hybrid solver analysis."""
    benchmark_path = Path(__file__).parent.parent / 'Benchmarks' / 'COMPREHENSIVE' / \
                     'comprehensive_benchmark_configs_dwave_20251130_212742.json'
    
    if not benchmark_path.exists():
        print(f"Warning: Benchmark file not found at {benchmark_path}")
        return None
    
    with open(benchmark_path, 'r') as f:
        return json.load(f)


def load_qpu_hierarchical():
    """Load QPU hierarchical benchmark data."""
    qpu_path = Path(__file__).parent.parent / 'qpu_hier_repaired.json'
    
    if not qpu_path.exists():
        print(f"Warning: QPU file not found at {qpu_path}")
        return None
    
    with open(qpu_path, 'r') as f:
        return json.load(f)


def load_gurobi_baseline():
    """Load Gurobi baseline data."""
    gurobi_path = Path(__file__).parent.parent / 'gurobi_baseline_60s.json'
    
    if not gurobi_path.exists():
        print(f"Warning: Gurobi baseline file not found at {gurobi_path}")
        return None
    
    with open(gurobi_path, 'r') as f:
        return json.load(f)


def extract_hybrid_data(benchmark_data):
    """
    Extract hybrid solver performance data from comprehensive benchmark.
    
    Returns dict with keys: n_patches, gurobi_cqm_time, dwave_cqm_time, 
                           gurobi_qubo_time, dwave_bqm_time, qpu_percentages
    """
    if benchmark_data is None:
        return None
    
    results = benchmark_data.get('patch_results', [])
    
    # Group by n_units
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
        
        # Gurobi (CQM formulation)
        if 'gurobi' in solvers and solvers['gurobi'].get('success'):
            data_by_size[n_units]['gurobi'].append(solvers['gurobi'].get('solve_time', 0))
        
        # D-Wave CQM
        if 'dwave_cqm' in solvers:
            total_time = solvers['dwave_cqm'].get('hybrid_time', 0)
            qpu_time = solvers['dwave_cqm'].get('qpu_time', 0)
            if total_time > 0:
                data_by_size[n_units]['dwave_cqm'].append(total_time)
                data_by_size[n_units]['dwave_cqm_qpu'].append(qpu_time)
                data_by_size[n_units]['dwave_cqm_total'].append(total_time)
        
        # Gurobi QUBO
        if 'gurobi_qubo' in solvers and solvers['gurobi_qubo'].get('success'):
            data_by_size[n_units]['gurobi_qubo'].append(solvers['gurobi_qubo'].get('solve_time', 0))
        
        # D-Wave BQM
        if 'dwave_bqm' in solvers:
            total_time = solvers['dwave_bqm'].get('hybrid_time', 0)
            qpu_time = solvers['dwave_bqm'].get('qpu_time', 0)
            if total_time > 0:
                data_by_size[n_units]['dwave_bqm'].append(total_time)
                data_by_size[n_units]['dwave_bqm_qpu'].append(qpu_time)
                data_by_size[n_units]['dwave_bqm_total'].append(total_time)
    
    # Compute averages
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
    """
    Extract hierarchical QPU data for Study 3.
    
    Returns aligned data for QPU and Gurobi comparisons.
    """
    if qpu_data is None or gurobi_data is None:
        return None
    
    qpu_runs = qpu_data.get('runs', [])
    gurobi_runs = gurobi_data.get('runs', [])
    
    # Build lookup by scenario name
    gurobi_lookup = {r['scenario_name']: r for r in gurobi_runs}
    
    data = {
        'scenario_names': [],
        'n_vars': [],
        'n_farms': [],
        'qpu_total_time': [],
        'qpu_solve_time': [],
        'qpu_access_time': [],
        'qpu_classical_overhead': [],
        'gurobi_total_time': [],
        'speedup': [],
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
        data['qpu_solve_time'].append(run['timing'].get('solve_time', qpu_total))
        data['qpu_access_time'].append(qpu_access)
        data['qpu_classical_overhead'].append(qpu_total - qpu_access)
        data['gurobi_total_time'].append(gurobi_total)
        data['speedup'].append(gurobi_total / qpu_total if qpu_total > 0 else 0)
    
    # Sort by n_vars
    sorted_indices = np.argsort(data['n_vars'])
    for key in data:
        data[key] = [data[key][i] for i in sorted_indices]
    
    return data


# ============================================================================
# Study 1: Hybrid Solver Performance
# ============================================================================

def plot_study1_hybrid_performance(hybrid_data, output_dir):
    """
    Create Study 1: Hybrid Solver Performance (2 panels side by side)
    
    Panel A: Line plot of solve time vs number of patches (log-log scale)
    Panel B: Bar chart showing QPU % of total time
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if hybrid_data is None:
        # Create placeholder with message
        ax1.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14)
        ax2.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14)
        fig.suptitle('Study 1: Hybrid Solver Performance', fontsize=16, fontweight='bold')
        save_figure(fig, output_dir / 'study1_hybrid_performance')
        plt.close(fig)
        return
    
    n_patches = hybrid_data['n_patches']
    
    # Panel A: Solve time comparison (log-log)
    ax1.loglog(n_patches, hybrid_data['gurobi_cqm_time'], 
               'o-', color=COLORS['blue'], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
               label='Gurobi (CQM)')
    ax1.loglog(n_patches, hybrid_data['dwave_cqm_time'],
               's-', color=COLORS['orange'], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
               label='D-Wave CQM')
    ax1.loglog(n_patches, hybrid_data['gurobi_qubo_time'],
               '^-', color=COLORS['green'], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
               label='Gurobi (QUBO)')
    ax1.loglog(n_patches, hybrid_data['dwave_bqm_time'],
               'd-', color=COLORS['red'], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
               label='D-Wave BQM')
    
    ax1.set_xlabel('Number of Patches', fontweight='bold')
    ax1.set_ylabel('Solve Time (s)', fontweight='bold')
    ax1.set_title('Solver Performance Scaling', fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.95)
    ax1.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # Add panel label
    ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=14, 
             fontweight='bold', va='top')
    
    # Panel B: QPU % of total time
    x = np.arange(len(n_patches))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, hybrid_data['dwave_cqm_qpu_pct'], width, 
                    color=COLORS['orange'], label='D-Wave CQM', alpha=0.8)
    bars2 = ax2.bar(x + width/2, hybrid_data['dwave_bqm_qpu_pct'], width,
                    color=COLORS['red'], label='D-Wave BQM', alpha=0.8)
    
    ax2.set_xlabel('Number of Patches', fontweight='bold')
    ax2.set_ylabel('QPU Time (% of Total)', fontweight='bold')
    ax2.set_title('QPU Utilization', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in n_patches])
    ax2.legend(loc='upper right', frameon=True, framealpha=0.95)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(max(hybrid_data['dwave_cqm_qpu_pct']), 
                        max(hybrid_data['dwave_bqm_qpu_pct'])) * 1.2)
    
    # Add panel label
    ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, fontsize=14,
             fontweight='bold', va='top')
    
    fig.suptitle('Study 1: Hybrid Solver Performance', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, output_dir / 'study1_hybrid_performance')
    plt.close(fig)
    print("✓ Study 1: Hybrid Solver Performance saved")


# ============================================================================
# Study 2: Pure QPU Decomposition
# ============================================================================

def plot_study2_decomposition(output_dir):
    """
    Create Study 2: Pure QPU Decomposition (2 panels side by side)
    
    Panel A: Line plot comparing decomposition methods
    Panel B: Stacked bar chart showing QPU time vs Classical overhead
    
    Uses representative data from results tables.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Representative data from results_and_conclusions_backup.tex tables
    # (farms: 10, 50, 100, 500, 1000)
    farms = [10, 50, 100, 500, 1000]
    
    # Decomposition method times (in seconds) - representative values
    methods_data = {
        'Gurobi': [0.5, 2.1, 8.5, 45.2, 180.5],
        'PlotBased': [1.2, 4.5, 12.8, 52.3, 195.6],
        'Multilevel(10)': [1.8, 5.2, 14.2, 48.5, 185.2],
        'Louvain': [1.5, 4.8, 13.5, 55.8, 210.3],
        'Coordinated': [2.0, 5.8, 15.5, 58.2, 225.8],
    }
    
    # QPU vs Classical breakdown (representative)
    qpu_times = [0.05, 0.12, 0.25, 0.85, 1.8]
    classical_times = [1.15, 4.38, 12.55, 51.45, 193.8]
    
    # Panel A: Line plot comparing methods
    method_colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], 
                     COLORS['red'], COLORS['purple']]
    markers = ['o', 's', '^', 'd', 'v']
    
    for i, (method, times) in enumerate(methods_data.items()):
        ax1.loglog(farms, times, f'{markers[i]}-', color=method_colors[i],
                   linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=method)
    
    ax1.set_xlabel('Number of Farms', fontweight='bold')
    ax1.set_ylabel('Total Time (s)', fontweight='bold')
    ax1.set_title('Decomposition Method Comparison', fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=9)
    ax1.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # Add panel label
    ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=14,
             fontweight='bold', va='top')
    
    # Panel B: Stacked bar chart
    x = np.arange(len(farms))
    width = 0.6
    
    bars1 = ax2.bar(x, qpu_times, width, color=COLORS['blue'], label='QPU Time')
    bars2 = ax2.bar(x, classical_times, width, bottom=qpu_times, 
                    color=COLORS['orange'], label='Classical Overhead')
    
    ax2.set_xlabel('Number of Farms', fontweight='bold')
    ax2.set_ylabel('Time (s)', fontweight='bold')
    ax2.set_title('Time Breakdown by Component', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(f) for f in farms])
    ax2.legend(loc='upper left', frameon=True, framealpha=0.95)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.set_yscale('log')
    
    # Add panel label
    ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, fontsize=14,
             fontweight='bold', va='top')
    
    fig.suptitle('Study 2: Pure QPU Decomposition Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, output_dir / 'study2_qpu_decomposition')
    plt.close(fig)
    print("✓ Study 2: Pure QPU Decomposition saved")


# ============================================================================
# Study 3: Hierarchical QPU Quantum Advantage
# ============================================================================

def plot_study3_quantum_advantage(hier_data, output_dir):
    """
    Create Study 3: Hierarchical QPU Quantum Advantage (2x2 = 4 panels)
    
    Panel A: Line plot of solve time comparison (QPU vs Gurobi) vs variables
    Panel B: Bar chart of benefit ratio (Gurobi/QPU) for each scenario
    Panel C: QPU time breakdown (pure QPU vs classical overhead)
    Panel D: Speedup analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    if hier_data is None:
        for ax in axes.flatten():
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
        fig.suptitle('Study 3: Hierarchical QPU Quantum Advantage', fontsize=16, fontweight='bold')
        save_figure(fig, output_dir / 'study3_quantum_advantage')
        plt.close(fig)
        return
    
    n_vars = hier_data['n_vars']
    scenario_labels = [s.replace('rotation_', '').replace('farms_', 'F').replace('foods', 'f') 
                       for s in hier_data['scenario_names']]
    
    # Panel A: Solve time comparison (QPU vs Gurobi)
    ax1.semilogy(n_vars, hier_data['gurobi_total_time'], 'o-', color=COLORS['blue'],
                 linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Gurobi (60s timeout)')
    ax1.semilogy(n_vars, hier_data['qpu_total_time'], 's-', color=COLORS['orange'],
                 linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='QPU Hierarchical')
    
    ax1.set_xlabel('Number of Variables', fontweight='bold')
    ax1.set_ylabel('Total Solve Time (s)', fontweight='bold')
    ax1.set_title('Solve Time Comparison', fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.95)
    ax1.grid(True, which='both', alpha=0.3, linestyle='--')
    ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # Panel B: Benefit ratio (Gurobi time / QPU time) - speedup factor
    x = np.arange(len(scenario_labels))
    
    # Ensure we only plot valid ratios
    speedup = [hier_data['gurobi_total_time'][i] / hier_data['qpu_total_time'][i] 
               if hier_data['qpu_total_time'][i] > 0 else 0 
               for i in range(len(n_vars))]
    
    colors_bars = [COLORS['green'] if s > 1 else COLORS['red'] for s in speedup]
    bars = ax2.bar(x, speedup, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Scenario', fontweight='bold')
    ax2.set_ylabel('Speedup Factor (Gurobi/QPU)', fontweight='bold')
    ax2.set_title('Speedup Analysis by Scenario', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_labels, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # Panel C: QPU time breakdown (stacked bar)
    width = 0.6
    x_c = np.arange(len(scenario_labels))
    
    bars_qpu = ax3.bar(x_c, hier_data['qpu_access_time'], width, 
                       color=COLORS['blue'], label='Pure QPU Time')
    bars_classical = ax3.bar(x_c, hier_data['qpu_classical_overhead'], width,
                             bottom=hier_data['qpu_access_time'],
                             color=COLORS['orange'], label='Classical Overhead')
    
    ax3.set_xlabel('Scenario', fontweight='bold')
    ax3.set_ylabel('Time (s)', fontweight='bold')
    ax3.set_title('QPU Time Breakdown', fontweight='bold')
    ax3.set_xticks(x_c)
    ax3.set_xticklabels(scenario_labels, rotation=45, ha='right', fontsize=8)
    ax3.legend(loc='upper left', frameon=True, framealpha=0.95)
    ax3.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax3.text(-0.12, 1.05, '(c)', transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # Panel D: Speedup vs Problem Size
    ax4.plot(n_vars, speedup, 'o-', color=COLORS['purple'], linewidth=LINE_WIDTH, 
             markersize=MARKER_SIZE)
    ax4.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Break-even')
    
    # Add shaded region for quantum advantage
    ax4.fill_between(n_vars, 1, max(speedup) * 1.1, alpha=0.15, color=COLORS['green'])
    ax4.fill_between(n_vars, 0, 1, alpha=0.15, color=COLORS['red'])
    
    ax4.set_xlabel('Number of Variables', fontweight='bold')
    ax4.set_ylabel('Speedup Factor', fontweight='bold')
    ax4.set_title('Speedup Scaling with Problem Size', fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim(0, max(speedup) * 1.15)
    ax4.legend(loc='upper right', frameon=True, framealpha=0.95)
    ax4.text(-0.12, 1.05, '(d)', transform=ax4.transAxes, fontsize=14, fontweight='bold', va='top')
    
    fig.suptitle('Study 3: Hierarchical QPU Quantum Advantage', fontsize=16, fontweight='bold', y=1.02)
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
    print("Generating Publication-Quality Study Plots")
    print("=" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print("\n" + "-" * 70)
    
    # Load data
    print("\nLoading data files...")
    
    benchmark_data = load_comprehensive_benchmark()
    if benchmark_data:
        print(f"  ✓ Comprehensive benchmark loaded")
    else:
        print(f"  ✗ Comprehensive benchmark not found")
    
    qpu_data = load_qpu_hierarchical()
    if qpu_data:
        print(f"  ✓ QPU hierarchical data loaded ({len(qpu_data.get('runs', []))} runs)")
    else:
        print(f"  ✗ QPU hierarchical data not found")
    
    gurobi_data = load_gurobi_baseline()
    if gurobi_data:
        print(f"  ✓ Gurobi baseline loaded ({len(gurobi_data.get('runs', []))} runs)")
    else:
        print(f"  ✗ Gurobi baseline not found")
    
    # Extract processed data
    print("\nProcessing data...")
    hybrid_data = extract_hybrid_data(benchmark_data)
    if hybrid_data:
        print(f"  ✓ Hybrid data extracted for {len(hybrid_data['n_patches'])} patch sizes")
    
    hier_data = extract_hierarchical_data(qpu_data, gurobi_data)
    if hier_data:
        print(f"  ✓ Hierarchical data extracted for {len(hier_data['scenario_names'])} scenarios")
    
    # Generate plots
    print("\n" + "-" * 70)
    print("Generating plots...")
    print("-" * 70 + "\n")
    
    # Study 1: Hybrid Solver Performance
    print("Creating Study 1: Hybrid Solver Performance...")
    plot_study1_hybrid_performance(hybrid_data, output_dir)
    
    # Study 2: Pure QPU Decomposition
    print("Creating Study 2: Pure QPU Decomposition...")
    plot_study2_decomposition(output_dir)
    
    # Study 3: Hierarchical QPU Quantum Advantage
    print("Creating Study 3: Hierarchical QPU Quantum Advantage...")
    plot_study3_quantum_advantage(hier_data, output_dir)
    
    print("\n" + "=" * 70)
    print("All study plots generated successfully!")
    print("=" * 70)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('study*.p*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
