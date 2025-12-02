#!/usr/bin/env python3
"""
QPU Benchmark Results Plotter

Creates professional plots comparing:
1. All QPU decomposition methods (scales 10-100)
2. Large-scale QPU methods (scales 200-1000) 
3. Comparison with classical and hybrid solvers

Follows the style of existing professional plots.

Author: OQI-UC002-DWave Project
Date: 2025-12-02
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import seaborn as sns
from collections import defaultdict

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.dpi'] = 150

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "Benchmarks" / "COMPREHENSIVE"
QPU_RESULTS_DIR = PROJECT_ROOT / "@todo" / "qpu_benchmark_results"
OUTPUT_DIR = PROJECT_ROOT / "professional_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Color scheme - matching existing plots
COLORS = {
    # Classical solvers
    'PuLP': '#E63946',
    'Gurobi': '#E63946',
    'GurobiQUBO': '#D62828',
    
    # Hybrid solvers
    'DWave_Hybrid': '#118AB2',
    'DWave_CQM': '#118AB2',
    'DWave_BQM': '#073B4C',
    
    # Pure QPU methods
    'PlotBased_QPU': '#06FFA5',
    'Multilevel(5)_QPU': '#2EC4B6',
    'Multilevel(10)_QPU': '#20A39E',
    'Louvain_QPU': '#3DDC97',
    'Spectral(10)_QPU': '#5CDB95',
    'cqm_first_PlotBased': '#8338EC',
    'coordinated': '#FF6B6B',
    'direct_qpu': '#FFD93D',
}

MARKERS = {
    'PuLP': 'o',
    'Gurobi': 'o',
    'GurobiQUBO': 's',
    'DWave_Hybrid': 'D',
    'DWave_CQM': 'D',
    'DWave_BQM': 'p',
    'PlotBased_QPU': '^',
    'Multilevel(5)_QPU': 'v',
    'Multilevel(10)_QPU': '<',
    'Louvain_QPU': '>',
    'Spectral(10)_QPU': 'h',
    'cqm_first_PlotBased': 'P',
    'coordinated': 'X',
    'direct_qpu': '*',
}


def load_qpu_benchmark_data():
    """Load QPU benchmark results from JSON files."""
    data = {
        'small_scale': None,  # 10, 15, 50, 100
        'large_scale': None,  # 200, 500, 1000
    }
    
    # Find the benchmark files
    small_scale_file = QPU_RESULTS_DIR / "qpu_benchmark_20251201_160444.json"
    large_scale_file = QPU_RESULTS_DIR / "qpu_benchmark_20251201_200012.json"
    
    if small_scale_file.exists():
        with open(small_scale_file, 'r') as f:
            data['small_scale'] = json.load(f)
        print(f"  Loaded small scale data: {small_scale_file.name}")
    
    if large_scale_file.exists():
        with open(large_scale_file, 'r') as f:
            data['large_scale'] = json.load(f)
        print(f"  Loaded large scale data: {large_scale_file.name}")
    
    return data


def load_classical_hybrid_data():
    """Load classical and hybrid solver benchmark data."""
    data = {
        'PuLP': {},
        'DWave_Hybrid': {},
        'DWave_BQM': {},
        'GurobiQUBO': {},
    }
    
    configs = [10, 15, 25, 50, 100, 200, 500, 1000]
    
    # Load PuLP data
    pulp_dir = BENCHMARK_DIR / "Patch_PuLP"
    for config in configs:
        config_file = pulp_dir / f"config_{config}_run_1.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                data['PuLP'][config] = json.load(f)
    
    # Load D-Wave Hybrid (CQM) data
    dwave_dir = BENCHMARK_DIR / "Patch_DWave"
    for config in configs:
        config_file = dwave_dir / f"config_{config}_run_1.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                data['DWave_Hybrid'][config] = json.load(f)
    
    # Load D-Wave BQM data
    bqm_dir = BENCHMARK_DIR / "Patch_DWaveBQM"
    for config in configs:
        config_file = bqm_dir / f"config_{config}_run_1.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                data['DWave_BQM'][config] = json.load(f)
    
    # Load Gurobi QUBO data
    gurobi_dir = BENCHMARK_DIR / "Patch_GurobiQUBO"
    for config in configs:
        config_file = gurobi_dir / f"config_{config}_run_1.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                data['GurobiQUBO'][config] = json.load(f)
    
    print(f"  Loaded PuLP data for configs: {list(data['PuLP'].keys())}")
    print(f"  Loaded DWave_Hybrid data for configs: {list(data['DWave_Hybrid'].keys())}")
    print(f"  Loaded DWave_BQM data for configs: {list(data['DWave_BQM'].keys())}")
    print(f"  Loaded GurobiQUBO data for configs: {list(data['GurobiQUBO'].keys())}")
    
    return data


def extract_qpu_metrics(qpu_data, scale_type='small_scale'):
    """Extract metrics from QPU benchmark data."""
    if qpu_data[scale_type] is None:
        return None
    
    results = qpu_data[scale_type]['results']
    metrics = defaultdict(lambda: {'n_farms': [], 'objective': [], 'gap': [], 
                                    'wall_time': [], 'qpu_time': [], 'violations': []})
    
    for scale_result in results:
        n_farms = scale_result['n_farms']
        
        # Ground truth (Gurobi)
        if 'ground_truth' in scale_result:
            gt = scale_result['ground_truth']
            metrics['Gurobi']['n_farms'].append(n_farms)
            metrics['Gurobi']['objective'].append(gt.get('objective', 0))
            metrics['Gurobi']['gap'].append(0)
            metrics['Gurobi']['wall_time'].append(gt.get('solve_time', 0))
            metrics['Gurobi']['qpu_time'].append(0)
            metrics['Gurobi']['violations'].append(gt.get('violations', 0))
        
        gt_obj = scale_result.get('ground_truth', {}).get('objective', 0)
        
        # QPU methods
        for method_name, method_result in scale_result.get('method_results', {}).items():
            if not method_result.get('success', False):
                continue
            
            # Clean up method name for display
            display_name = method_name.replace('decomposition_', '').replace('_QPU', '_QPU')
            if display_name.startswith('cqm_first_'):
                display_name = display_name
            
            obj = method_result.get('objective', 0)
            gap = ((gt_obj - obj) / gt_obj * 100) if gt_obj > 0 and obj > 0 else 0
            
            timings = method_result.get('timings', {})
            wall_time = method_result.get('wall_time', method_result.get('total_time', 0))
            qpu_time = timings.get('qpu_access_total', 0)
            violations = method_result.get('violations', 0)
            
            metrics[display_name]['n_farms'].append(n_farms)
            metrics[display_name]['objective'].append(obj)
            metrics[display_name]['gap'].append(gap)
            metrics[display_name]['wall_time'].append(wall_time)
            metrics[display_name]['qpu_time'].append(qpu_time)
            metrics[display_name]['violations'].append(violations)
    
    return dict(metrics)


def extract_classical_metrics(classical_data):
    """Extract metrics from classical/hybrid benchmark data."""
    metrics = defaultdict(lambda: {'n_farms': [], 'objective': [], 'solve_time': [], 
                                    'qpu_time': [], 'hybrid_time': []})
    
    for solver, data in classical_data.items():
        for config, result in data.items():
            metrics[solver]['n_farms'].append(config)
            metrics[solver]['objective'].append(result.get('objective_value', 0))
            metrics[solver]['solve_time'].append(result.get('solve_time', 0))
            metrics[solver]['qpu_time'].append(result.get('qpu_time', 0))
            metrics[solver]['hybrid_time'].append(result.get('hybrid_time', 0))
    
    # Sort by n_farms
    for solver in metrics:
        sorted_indices = np.argsort(metrics[solver]['n_farms'])
        for key in metrics[solver]:
            metrics[solver][key] = [metrics[solver][key][i] for i in sorted_indices]
    
    return dict(metrics)


def plot_small_scale_comparison(qpu_metrics, classical_metrics, output_path):
    """
    Plot 1: Small scale (10-100 farms) comparison of all QPU methods
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('QPU Decomposition Methods Benchmark (10-100 Farms)\nPure Quantum Annealing vs Classical Solvers', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ===== Plot 1: Solution Quality (Objective) =====
    ax = axes[0, 0]
    
    # Plot PuLP as baseline
    if 'PuLP' in classical_metrics:
        farms = [f for f in classical_metrics['PuLP']['n_farms'] if f <= 100]
        objs = [classical_metrics['PuLP']['objective'][i] for i, f in enumerate(classical_metrics['PuLP']['n_farms']) if f <= 100]
        ax.plot(farms, objs, 'o-', linewidth=2.5, markersize=10, 
                color=COLORS['PuLP'], label='PuLP (Optimal)', alpha=0.9)
    
    # Plot QPU methods
    qpu_methods = ['PlotBased_QPU', 'Multilevel(5)_QPU', 'Multilevel(10)_QPU', 
                   'Louvain_QPU', 'cqm_first_PlotBased', 'coordinated']
    for method in qpu_methods:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            ax.plot(qpu_metrics[method]['n_farms'], qpu_metrics[method]['objective'],
                    marker=MARKERS.get(method, 'o'), linestyle='-', linewidth=2, markersize=8,
                    color=COLORS.get(method, '#888888'), label=method, alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Solution Quality', fontsize=14)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # ===== Plot 2: Optimality Gap =====
    ax = axes[0, 1]
    
    for method in qpu_methods:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            ax.plot(qpu_metrics[method]['n_farms'], qpu_metrics[method]['gap'],
                    marker=MARKERS.get(method, 'o'), linestyle='-', linewidth=2, markersize=8,
                    color=COLORS.get(method, '#888888'), label=method, alpha=0.8)
    
    ax.axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Optimal')
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Gap from Optimal (%)', fontsize=12)
    ax.set_title('Optimality Gap', fontsize=14)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # ===== Plot 3: Wall Time (Log Scale) =====
    ax = axes[1, 0]
    
    # Classical solvers
    if 'PuLP' in classical_metrics:
        farms = [f for f in classical_metrics['PuLP']['n_farms'] if f <= 100]
        times = [classical_metrics['PuLP']['solve_time'][i] for i, f in enumerate(classical_metrics['PuLP']['n_farms']) if f <= 100]
        ax.semilogy(farms, times, 'o-', linewidth=2.5, markersize=10, 
                    color=COLORS['PuLP'], label='PuLP', alpha=0.9)
    
    if 'DWave_Hybrid' in classical_metrics:
        farms = [f for f in classical_metrics['DWave_Hybrid']['n_farms'] if f <= 100]
        times = [classical_metrics['DWave_Hybrid']['hybrid_time'][i] for i, f in enumerate(classical_metrics['DWave_Hybrid']['n_farms']) if f <= 100]
        if times and all(t > 0 for t in times):
            ax.semilogy(farms, times, 'D-', linewidth=2.5, markersize=10, 
                        color=COLORS['DWave_Hybrid'], label='D-Wave Hybrid CQM', alpha=0.9)
    
    # QPU methods
    for method in qpu_methods:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            times = qpu_metrics[method]['wall_time']
            if all(t > 0 for t in times):
                ax.semilogy(qpu_metrics[method]['n_farms'], times,
                            marker=MARKERS.get(method, 'o'), linestyle='-', linewidth=2, markersize=8,
                            color=COLORS.get(method, '#888888'), label=method, alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Wall Time (seconds, log)', fontsize=12)
    ax.set_title('Total Execution Time', fontsize=14)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    
    # ===== Plot 4: Pure QPU Time =====
    ax = axes[1, 1]
    
    for method in qpu_methods:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            qpu_times = qpu_metrics[method]['qpu_time']
            if any(t > 0 for t in qpu_times):
                ax.plot(qpu_metrics[method]['n_farms'], qpu_times,
                        marker=MARKERS.get(method, 'o'), linestyle='-', linewidth=2, markersize=8,
                        color=COLORS.get(method, '#888888'), label=method, alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('QPU Access Time (seconds)', fontsize=12)
    ax.set_title('Pure QPU Time', fontsize=14)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_large_scale_comparison(qpu_metrics, classical_metrics, output_path):
    """
    Plot 2: Large scale (200-1000 farms) comparison with restricted methods
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Large-Scale QPU Benchmark (200-1000 Farms)\nScalable Decomposition Methods vs Classical Solvers', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    large_scale_methods = ['Multilevel(10)_QPU', 'cqm_first_PlotBased', 'coordinated']
    
    # ===== Plot 1: Solution Quality =====
    ax = axes[0, 0]
    
    # PuLP baseline
    if 'PuLP' in classical_metrics:
        farms = [f for f in classical_metrics['PuLP']['n_farms'] if f >= 200]
        objs = [classical_metrics['PuLP']['objective'][i] for i, f in enumerate(classical_metrics['PuLP']['n_farms']) if f >= 200]
        if farms and objs:
            ax.plot(farms, objs, 'o-', linewidth=2.5, markersize=12, 
                    color=COLORS['PuLP'], label='PuLP (Optimal)', alpha=0.9)
    
    # QPU methods
    for method in large_scale_methods:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            farms = [f for f in qpu_metrics[method]['n_farms'] if f >= 200]
            objs = [qpu_metrics[method]['objective'][i] for i, f in enumerate(qpu_metrics[method]['n_farms']) if f >= 200]
            if farms:
                ax.plot(farms, objs, marker=MARKERS.get(method, 'o'), linestyle='-', 
                        linewidth=2.5, markersize=10, color=COLORS.get(method, '#888888'), 
                        label=method, alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Solution Quality at Scale', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ===== Plot 2: Optimality Gap =====
    ax = axes[0, 1]
    
    for method in large_scale_methods:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            farms = [f for f in qpu_metrics[method]['n_farms'] if f >= 200]
            gaps = [qpu_metrics[method]['gap'][i] for i, f in enumerate(qpu_metrics[method]['n_farms']) if f >= 200]
            if farms:
                ax.plot(farms, gaps, marker=MARKERS.get(method, 'o'), linestyle='-', 
                        linewidth=2.5, markersize=10, color=COLORS.get(method, '#888888'), 
                        label=method, alpha=0.8)
    
    ax.axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Optimal')
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Gap from Optimal (%)', fontsize=12)
    ax.set_title('Optimality Gap at Scale', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ===== Plot 3: Wall Time Comparison =====
    ax = axes[1, 0]
    
    # Classical
    if 'PuLP' in classical_metrics:
        farms = [f for f in classical_metrics['PuLP']['n_farms'] if f >= 200]
        times = [classical_metrics['PuLP']['solve_time'][i] for i, f in enumerate(classical_metrics['PuLP']['n_farms']) if f >= 200]
        if farms and times:
            ax.semilogy(farms, times, 'o-', linewidth=2.5, markersize=12, 
                        color=COLORS['PuLP'], label='PuLP', alpha=0.9)
    
    # D-Wave Hybrid
    if 'DWave_Hybrid' in classical_metrics:
        farms = [f for f in classical_metrics['DWave_Hybrid']['n_farms'] if f >= 200]
        times = [classical_metrics['DWave_Hybrid']['hybrid_time'][i] for i, f in enumerate(classical_metrics['DWave_Hybrid']['n_farms']) if f >= 200]
        if farms and times and all(t > 0 for t in times):
            ax.semilogy(farms, times, 'D-', linewidth=2.5, markersize=12, 
                        color=COLORS['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.9)
    
    # D-Wave BQM
    if 'DWave_BQM' in classical_metrics:
        farms = [f for f in classical_metrics['DWave_BQM']['n_farms'] if f >= 200]
        times = [classical_metrics['DWave_BQM']['hybrid_time'][i] for i, f in enumerate(classical_metrics['DWave_BQM']['n_farms']) if f >= 200]
        if farms and times and all(t > 0 for t in times):
            ax.semilogy(farms, times, 'p-', linewidth=2.5, markersize=12, 
                        color=COLORS['DWave_BQM'], label='D-Wave BQM Hybrid', alpha=0.9)
    
    # QPU methods
    for method in large_scale_methods:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            farms = [f for f in qpu_metrics[method]['n_farms'] if f >= 200]
            times = [qpu_metrics[method]['wall_time'][i] for i, f in enumerate(qpu_metrics[method]['n_farms']) if f >= 200]
            if farms and times and all(t > 0 for t in times):
                ax.semilogy(farms, times, marker=MARKERS.get(method, 'o'), linestyle='-', 
                            linewidth=2.5, markersize=10, color=COLORS.get(method, '#888888'), 
                            label=method, alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Wall Time (seconds, log)', fontsize=12)
    ax.set_title('Execution Time Comparison', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # ===== Plot 4: QPU Time vs Violations =====
    ax = axes[1, 1]
    
    # Bar chart: violations by method at each scale
    scales = [200, 500, 1000]
    x = np.arange(len(scales))
    width = 0.25
    
    for i, method in enumerate(large_scale_methods):
        if method in qpu_metrics:
            violations = []
            for scale in scales:
                if scale in qpu_metrics[method]['n_farms']:
                    idx = qpu_metrics[method]['n_farms'].index(scale)
                    violations.append(qpu_metrics[method]['violations'][idx])
                else:
                    violations.append(0)
            ax.bar(x + i*width, violations, width, label=method, 
                   color=COLORS.get(method, '#888888'), alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Constraint Violations', fontsize=12)
    ax.set_title('Feasibility: Constraint Violations', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(scales)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_comprehensive_solver_comparison(qpu_metrics_small, qpu_metrics_large, classical_metrics, output_path):
    """
    Plot 3: Comprehensive comparison across all scales and solver types
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Solver Comparison: Classical vs Hybrid vs Pure QPU\nBinary Crop Allocation Problem', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Combine metrics
    all_qpu = {}
    if qpu_metrics_small:
        for method, data in qpu_metrics_small.items():
            all_qpu[method] = data.copy()
    if qpu_metrics_large:
        for method, data in qpu_metrics_large.items():
            if method in all_qpu:
                for key in data:
                    all_qpu[method][key].extend(data[key])
            else:
                all_qpu[method] = data.copy()
    
    # ===== Row 1: Time Comparisons =====
    
    # Plot 1: Classical vs All (Log scale)
    ax = axes[0, 0]
    
    # PuLP
    if 'PuLP' in classical_metrics:
        ax.semilogy(classical_metrics['PuLP']['n_farms'], classical_metrics['PuLP']['solve_time'],
                    'o-', linewidth=3, markersize=10, color=COLORS['PuLP'], label='PuLP (Classical)', alpha=0.9)
    
    # D-Wave Hybrid
    if 'DWave_Hybrid' in classical_metrics:
        times = [t for t in classical_metrics['DWave_Hybrid']['hybrid_time'] if t > 0]
        farms = [classical_metrics['DWave_Hybrid']['n_farms'][i] for i, t in enumerate(classical_metrics['DWave_Hybrid']['hybrid_time']) if t > 0]
        if times:
            ax.semilogy(farms, times, 'D-', linewidth=3, markersize=10, 
                        color=COLORS['DWave_Hybrid'], label='D-Wave Hybrid CQM', alpha=0.9)
    
    # Best QPU method (cqm_first_PlotBased)
    if 'cqm_first_PlotBased' in all_qpu:
        ax.semilogy(all_qpu['cqm_first_PlotBased']['n_farms'], all_qpu['cqm_first_PlotBased']['wall_time'],
                    'P-', linewidth=3, markersize=10, color=COLORS['cqm_first_PlotBased'], 
                    label='CQM-First PlotBased (QPU)', alpha=0.9)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Solve Time (seconds, log)', fontsize=12)
    ax.set_title('Time: Classical vs Hybrid vs QPU', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: QPU Time Only
    ax = axes[0, 1]
    
    qpu_methods = ['PlotBased_QPU', 'Multilevel(10)_QPU', 'cqm_first_PlotBased', 'coordinated']
    for method in qpu_methods:
        if method in all_qpu and len(all_qpu[method]['n_farms']) > 0:
            qpu_times = [t for t in all_qpu[method]['qpu_time'] if t > 0]
            farms = [all_qpu[method]['n_farms'][i] for i, t in enumerate(all_qpu[method]['qpu_time']) if t > 0]
            if qpu_times:
                ax.plot(farms, qpu_times, marker=MARKERS.get(method, 'o'), linestyle='-', 
                        linewidth=2.5, markersize=9, color=COLORS.get(method, '#888888'), 
                        label=method, alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Pure QPU Time (seconds)', fontsize=12)
    ax.set_title('Pure Quantum Time', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Embedding vs QPU breakdown
    ax = axes[0, 2]
    
    # For cqm_first_PlotBased, show breakdown
    if 'cqm_first_PlotBased' in all_qpu:
        farms = all_qpu['cqm_first_PlotBased']['n_farms']
        wall = all_qpu['cqm_first_PlotBased']['wall_time']
        qpu = all_qpu['cqm_first_PlotBased']['qpu_time']
        embed = [w - q for w, q in zip(wall, qpu)]
        
        ax.bar(farms, embed, label='Embedding + Classical', color='#FFB703', alpha=0.8)
        ax.bar(farms, qpu, bottom=embed, label='QPU Access', color=COLORS['cqm_first_PlotBased'], alpha=0.8)
        
        ax.set_xlabel('Number of Farms', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Time Breakdown: CQM-First PlotBased', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    # ===== Row 2: Quality Comparisons =====
    
    # Plot 4: Objective Value
    ax = axes[1, 0]
    
    if 'PuLP' in classical_metrics:
        ax.plot(classical_metrics['PuLP']['n_farms'], classical_metrics['PuLP']['objective'],
                'o-', linewidth=3, markersize=10, color=COLORS['PuLP'], label='PuLP (Optimal)', alpha=0.9)
    
    for method in ['cqm_first_PlotBased', 'coordinated', 'Multilevel(10)_QPU']:
        if method in all_qpu and len(all_qpu[method]['n_farms']) > 0:
            ax.plot(all_qpu[method]['n_farms'], all_qpu[method]['objective'],
                    marker=MARKERS.get(method, 'o'), linestyle='-', linewidth=2.5, markersize=9,
                    color=COLORS.get(method, '#888888'), label=method, alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Solution Quality Comparison', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Gap from Optimal
    ax = axes[1, 1]
    
    for method in ['cqm_first_PlotBased', 'coordinated', 'Multilevel(10)_QPU', 'PlotBased_QPU', 'Louvain_QPU']:
        if method in all_qpu and len(all_qpu[method]['n_farms']) > 0:
            ax.plot(all_qpu[method]['n_farms'], all_qpu[method]['gap'],
                    marker=MARKERS.get(method, 'o'), linestyle='-', linewidth=2, markersize=8,
                    color=COLORS.get(method, '#888888'), label=method, alpha=0.8)
    
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Optimal')
    ax.axhline(y=10, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='10% Gap')
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Gap from Optimal (%)', fontsize=12)
    ax.set_title('Optimality Gap', fontsize=14)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Feasibility (Violations)
    ax = axes[1, 2]
    
    methods_for_viol = ['PlotBased_QPU', 'Multilevel(10)_QPU', 'cqm_first_PlotBased', 'coordinated', 'Louvain_QPU']
    all_scales = sorted(set([f for m in methods_for_viol if m in all_qpu for f in all_qpu[m]['n_farms']]))
    
    x = np.arange(len(all_scales))
    width = 0.15
    
    for i, method in enumerate(methods_for_viol):
        if method in all_qpu:
            violations = []
            for scale in all_scales:
                if scale in all_qpu[method]['n_farms']:
                    idx = all_qpu[method]['n_farms'].index(scale)
                    violations.append(all_qpu[method]['violations'][idx])
                else:
                    violations.append(np.nan)
            ax.bar(x + i*width, violations, width, label=method, 
                   color=COLORS.get(method, '#888888'), alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Constraint Violations', fontsize=12)
    ax.set_title('Feasibility: Violations by Method', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(all_scales)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_summary_table(qpu_metrics_small, qpu_metrics_large, classical_metrics, output_path):
    """
    Create a summary table figure
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Prepare table data
    headers = ['Scale', 'Method', 'Objective', 'Gap (%)', 'Wall Time (s)', 'QPU Time (s)', 'Violations', 'Status']
    
    rows = []
    
    # Combine all QPU data
    all_qpu = {}
    if qpu_metrics_small:
        for method, data in qpu_metrics_small.items():
            all_qpu[method] = {k: list(v) for k, v in data.items()}
    if qpu_metrics_large:
        for method, data in qpu_metrics_large.items():
            if method in all_qpu:
                for key in data:
                    all_qpu[method][key].extend(data[key])
            else:
                all_qpu[method] = {k: list(v) for k, v in data.items()}
    
    scales = [10, 15, 50, 100, 200, 500, 1000]
    methods_order = ['Gurobi', 'PlotBased_QPU', 'Multilevel(10)_QPU', 'Louvain_QPU', 
                     'cqm_first_PlotBased', 'coordinated']
    
    for scale in scales:
        for method in methods_order:
            if method in all_qpu and scale in all_qpu[method]['n_farms']:
                idx = all_qpu[method]['n_farms'].index(scale)
                obj = all_qpu[method]['objective'][idx]
                gap = all_qpu[method]['gap'][idx]
                wall = all_qpu[method]['wall_time'][idx]
                qpu = all_qpu[method]['qpu_time'][idx]
                viol = all_qpu[method]['violations'][idx]
                status = '✓ Feas' if viol == 0 else f'⚠ {viol}v'
                
                rows.append([scale, method, f'{obj:.4f}', f'{gap:.1f}', f'{wall:.2f}', 
                            f'{qpu:.3f}' if qpu > 0 else 'N/A', viol, status])
    
    # Create table
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for i, key in enumerate(headers):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    ax.set_title('QPU Benchmark Summary Table', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("="*80)
    print("QPU Benchmark Results Plotter")
    print("="*80)
    print()
    
    print("[1/4] Loading QPU benchmark data...")
    qpu_data = load_qpu_benchmark_data()
    
    print("\n[2/4] Loading classical/hybrid benchmark data...")
    classical_data = load_classical_hybrid_data()
    
    print("\n[3/4] Extracting metrics...")
    qpu_metrics_small = extract_qpu_metrics(qpu_data, 'small_scale')
    qpu_metrics_large = extract_qpu_metrics(qpu_data, 'large_scale')
    classical_metrics = extract_classical_metrics(classical_data)
    
    print("\n[4/4] Creating plots...")
    
    # Plot 1: Small scale comparison
    if qpu_metrics_small:
        print("  Creating small-scale comparison plot...")
        plot_small_scale_comparison(
            qpu_metrics_small, classical_metrics,
            OUTPUT_DIR / "qpu_benchmark_small_scale.png"
        )
    
    # Plot 2: Large scale comparison
    if qpu_metrics_large:
        print("  Creating large-scale comparison plot...")
        plot_large_scale_comparison(
            qpu_metrics_large, classical_metrics,
            OUTPUT_DIR / "qpu_benchmark_large_scale.png"
        )
    
    # Plot 3: Comprehensive comparison
    print("  Creating comprehensive solver comparison...")
    plot_comprehensive_solver_comparison(
        qpu_metrics_small, qpu_metrics_large, classical_metrics,
        OUTPUT_DIR / "qpu_benchmark_comprehensive.png"
    )
    
    # Plot 4: Summary table
    print("  Creating summary table...")
    plot_summary_table(
        qpu_metrics_small, qpu_metrics_large, classical_metrics,
        OUTPUT_DIR / "qpu_benchmark_summary_table.png"
    )
    
    print("\n" + "="*80)
    print("✅ All plots generated successfully!")
    print(f"   Output directory: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
