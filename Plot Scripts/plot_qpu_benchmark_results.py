#!/usr/bin/env python3
"""
QPU Benchmark Results Plotter

Creates professional plots comparing:
1. All QPU decomposition methods (scales 10-100) with classical/hybrid baselines
2. Large-scale QPU methods (scales 200-1000) with classical/hybrid baselines
3. Comprehensive comparison across all scales and solver types
4. Time-to-Solution Quality Analysis

Time definitions:
- PuLP: solve_time (classical solver time)
- DWave Hybrid CQM: hybrid_time (total hybrid solver time)
- DWave BQM Hybrid: hybrid_time (total hybrid solver time)  
- QPU methods: qpu_time only (embedding is overhead, not counted)

Author: OQI-UC002-DWave Project
Date: 2025-12-03
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import unified plot configuration
from plot_config import (
    setup_publication_style, QUALITATIVE_COLORS, METHOD_COLORS,
    save_figure, get_method_color
)

# Apply publication style
setup_publication_style()

# Paths
BENCHMARK_DIR = PROJECT_ROOT / "Benchmarks" / "COMPREHENSIVE"
QPU_RESULTS_DIR = PROJECT_ROOT / "@todo" / "qpu_benchmark_results"
OUTPUT_DIR = PROJECT_ROOT / "professional_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Color scheme - use unified colors from plot_config
COLORS = METHOD_COLORS.copy()
# Add any additional color mappings if needed
COLORS.update({
    'GurobiQUBO': '#D62828',
    'DWave_Hybrid': '#118AB2',
    'DWave_CQM': '#118AB2',
    'DWave_BQM': '#073B4C',
})
    'Louvain_QPU': '#264653',         # Dark slate
    'Spectral(10)_QPU': '#9B5DE5',    # Purple
    'cqm_first_PlotBased': '#8338EC', # Violet
    'coordinated': '#FF6B6B',         # Coral red
    'direct_qpu': '#FFD93D',          # Yellow
    
    # New HybridGrid methods - green shades
    'HybridGrid(5,9)_QPU': '#06D6A0',   # Bright green
    'HybridGrid(10,9)_QPU': '#1B9AAA',  # Teal-blue
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
    'HybridGrid(5,9)_QPU': '*',
    'HybridGrid(10,9)_QPU': 'X',
    'coordinated': 'X',
    'direct_qpu': '*',
}

# Define method groups
SMALL_SCALE_QPU_METHODS = [
    'PlotBased_QPU', 'Multilevel(5)_QPU', 'Multilevel(10)_QPU',
    'Louvain_QPU', 'Spectral(10)_QPU', 'cqm_first_PlotBased',
    'HybridGrid(5,9)_QPU', 'HybridGrid(10,9)_QPU'
]
LARGE_SCALE_QPU_METHODS = ['Multilevel(10)_QPU', 'cqm_first_PlotBased', 'coordinated',
                           'HybridGrid(5,9)_QPU', 'HybridGrid(10,9)_QPU']
BASELINE_METHODS = ['PuLP', 'DWave_Hybrid', 'DWave_BQM', 'HybridGrid(5,9)_QPU', 'HybridGrid(10,9)_QPU']


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
    
    # Load HybridGrid QPU results from qpu_benchmark_results
    data['HybridGrid(5,9)_QPU'] = {}
    data['HybridGrid(10,9)_QPU'] = {}
    
    for f in sorted(QPU_RESULTS_DIR.glob('qpu_benchmark_*.json')):
        with open(f) as fp:
            qpu_data = json.load(fp)
        
        for r in qpu_data.get('results', []):
            scale = r.get('n_farms')
            gt = r.get('ground_truth', {})
            
            for method, result in r.get('method_results', {}).items():
                if 'HybridGrid(5,9)' in method and result.get('success'):
                    if scale not in data['HybridGrid(5,9)_QPU']:
                        data['HybridGrid(5,9)_QPU'][scale] = {
                            'objective_value': result.get('objective', 0),
                            'violations': result.get('violations', 0),
                            'n_violations': result.get('violations', 0),
                            'qpu_time': result.get('timings', {}).get('qpu_access_total', 0),
                            'hybrid_time': result.get('timings', {}).get('qpu_access_total', 0),
                            'solve_time': result.get('timings', {}).get('qpu_access_total', 0),
                            'n_units': scale,
                            'n_variables': scale * 27 + 27,
                            'gt_objective': gt.get('objective', 0),
                        }
                elif 'HybridGrid(10,9)' in method and result.get('success'):
                    if scale not in data['HybridGrid(10,9)_QPU']:
                        data['HybridGrid(10,9)_QPU'][scale] = {
                            'objective_value': result.get('objective', 0),
                            'violations': result.get('violations', 0),
                            'n_violations': result.get('violations', 0),
                            'qpu_time': result.get('timings', {}).get('qpu_access_total', 0),
                            'hybrid_time': result.get('timings', {}).get('qpu_access_total', 0),
                            'solve_time': result.get('timings', {}).get('qpu_access_total', 0),
                            'n_units': scale,
                            'n_variables': scale * 27 + 27,
                            'gt_objective': gt.get('objective', 0),
                        }
    
    print(f"  Loaded PuLP data for configs: {list(data['PuLP'].keys())}")
    print(f"  Loaded DWave_Hybrid data for configs: {list(data['DWave_Hybrid'].keys())}")
    print(f"  Loaded DWave_BQM data for configs: {list(data['DWave_BQM'].keys())}")
    print(f"  Loaded GurobiQUBO data for configs: {list(data['GurobiQUBO'].keys())}")
    print(f"  Loaded HybridGrid(5,9)_QPU data for configs: {sorted(data['HybridGrid(5,9)_QPU'].keys())}")
    print(f"  Loaded HybridGrid(10,9)_QPU data for configs: {sorted(data['HybridGrid(10,9)_QPU'].keys())}")
    
    return data


def extract_qpu_metrics(qpu_data, scale_type='small_scale'):
    """Extract metrics from QPU benchmark data."""
    if qpu_data[scale_type] is None:
        return None
    
    results = qpu_data[scale_type]['results']
    metrics = defaultdict(lambda: {'n_farms': [], 'objective': [], 'gap': [], 
                                    'wall_time': [], 'qpu_time': [], 'violations': [],
                                    'n_constraints': [], 'normalized_quality': [],
                                    'n_variables': [], 'efficiency': []})
    
    for scale_result in results:
        n_farms = scale_result['n_farms']
        n_constraints = scale_result.get('metadata', {}).get('n_constraints', 100)  # Default fallback
        
        # Ground truth (Gurobi)
        if 'ground_truth' in scale_result:
            gt = scale_result['ground_truth']
            gt_viol = gt.get('violations', 0)
            gt_obj = gt.get('objective', 0)
            # Normalized quality: objective * (1 - 0.99 * violations/constraints)
            gt_penalty = max(0.01, 1 - 0.99 * gt_viol / n_constraints) if n_constraints > 0 else 1
            gt_norm_qual = gt_obj * gt_penalty
            metrics['Gurobi']['n_farms'].append(n_farms)
            metrics['Gurobi']['objective'].append(gt_obj)
            metrics['Gurobi']['gap'].append(0)
            metrics['Gurobi']['wall_time'].append(gt.get('solve_time', 0))
            metrics['Gurobi']['qpu_time'].append(0)
            metrics['Gurobi']['violations'].append(gt_viol)
            metrics['Gurobi']['n_constraints'].append(n_constraints)
            metrics['Gurobi']['normalized_quality'].append(gt_norm_qual)
            
            # Variables: use consistent formula (n_farms * 27 + 27) for fair comparison
            n_vars = n_farms * 27 + 27  # Y variables + U variables
            metrics['Gurobi']['n_variables'].append(n_vars)
            gt_time = gt.get('solve_time', 1)
            gt_eff = (1.0 / gt_time) * gt_norm_qual if gt_time > 0 else 0
            metrics['Gurobi']['efficiency'].append(gt_eff)
        
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
            # Use QPU time only (embedding is overhead)
            qpu_time = timings.get('qpu_access_total', 0)
            violations = method_result.get('violations', 0)
            
            # Normalized quality: objective * (1 - 0.99 * violations/constraints)
            penalty = max(0.01, 1 - 0.99 * violations / n_constraints) if n_constraints > 0 else 1
            norm_qual = obj * penalty
            
            metrics[display_name]['n_farms'].append(n_farms)
            metrics[display_name]['objective'].append(obj)
            metrics[display_name]['gap'].append(gap)
            metrics[display_name]['wall_time'].append(wall_time)
            metrics[display_name]['qpu_time'].append(qpu_time)
            metrics[display_name]['violations'].append(violations)
            metrics[display_name]['n_constraints'].append(n_constraints)
            metrics[display_name]['normalized_quality'].append(norm_qual)
            
            # Variables: use consistent formula (n_farms * 27 + 27) for fair comparison
            # Don't use QUBO-expanded n_variables which varies by formulation
            n_vars = n_farms * 27 + 27  # Y variables + U variables
            metrics[display_name]['n_variables'].append(n_vars)
            
            # NEW EFFICIENCY METRIC: 1/(qpu_time × effective_gap)
            # effective_gap = gap + 5% per violation (penalize violations)
            # Higher is better: rewards BOTH low time AND low gap
            effective_gap = max(0.1, gap + violations * 5.0)  # Min 0.1 to avoid div by 0
            if qpu_time > 0:
                efficiency = 1.0 / (qpu_time * effective_gap)
            else:
                efficiency = 0
            metrics[display_name]['efficiency'].append(efficiency)
    
    return dict(metrics)


def extract_classical_metrics(classical_data):
    """
    Extract metrics from classical/hybrid benchmark data.
    
    Time definitions:
    - PuLP: solve_time (classical solver time)
    - DWave_Hybrid: hybrid_time (total hybrid solver time)
    - DWave_BQM: hybrid_time (total hybrid solver time)
    
    Normalized quality: objective * (1 - violations/constraints)
    
    NOTE: We manually count violations by checking how many plots have >1 crop assigned,
    because the validation data in some files is incorrect (especially BQM).
    """
    metrics = defaultdict(lambda: {'n_farms': [], 'objective': [], 'solve_time': [], 
                                    'qpu_time': [], 'hybrid_time': [], 'violations': [],
                                    'n_constraints': [], 'normalized_quality': [], 'gap': [],
                                    'n_variables': [], 'efficiency': []})
    
    for solver, data in classical_data.items():
        for config, result in data.items():
            obj = result.get('objective_value', 0)
            n_units = result.get('n_units', config)  # Number of plots
            
            # For HybridGrid QPU results, use violations directly from the result
            if 'HybridGrid' in solver:
                n_violations = result.get('violations', result.get('n_violations', 0))
            else:
                # Manually count violations by checking solution
                # A violation = plot with more than 1 crop assigned
                solution = result.get('solution_plantations', {})
                crops_per_plot = {}
                
                for var, val in solution.items():
                    # Check if crop is assigned (value is 1 or "1")
                    if val == 1 or val == 1.0 or val == "1":
                        # Parse variable name - handles both "Y_Patch1_Apple" and "Patch1_Apple" formats
                        var_clean = var.replace('Y_', '')
                        parts = var_clean.split('_', 1)
                        if len(parts) == 2:
                            plot = parts[0]
                            crops_per_plot[plot] = crops_per_plot.get(plot, 0) + 1
                
                # Count violations: plots with more than 1 crop
                n_violations = sum(1 for count in crops_per_plot.values() if count > 1)
            
            # Use n_units as the constraint count (one "at most 1 crop" constraint per plot)
            n_constraints = n_units
            
            # Normalized quality: objective * (1 - 0.99 * violations/constraints)
            # Linear penalty: 1.0 at v=0, 0.01 at v=c (never exactly 0 for log scale)
            penalty_factor = max(0.01, 1 - 0.99 * n_violations / n_constraints) if n_constraints > 0 else 1
            norm_qual = obj * penalty_factor
            
            metrics[solver]['n_farms'].append(config)
            metrics[solver]['objective'].append(obj)
            # For PuLP, use solve_time; for DWave methods, use hybrid_time
            if solver == 'PuLP':
                metrics[solver]['solve_time'].append(result.get('solve_time', 0))
            else:
                # DWave_Hybrid and DWave_BQM use hybrid_time
                metrics[solver]['solve_time'].append(result.get('hybrid_time', result.get('solve_time', 0)))
            metrics[solver]['qpu_time'].append(result.get('qpu_time', 0))
            metrics[solver]['hybrid_time'].append(result.get('hybrid_time', 0))
            metrics[solver]['violations'].append(n_violations)
            metrics[solver]['n_constraints'].append(n_constraints)
            metrics[solver]['normalized_quality'].append(norm_qual)
            metrics[solver]['gap'].append(0)  # Will be calculated later if needed
            
            # Variables: use consistent formula (config * 27 + 27) for fair comparison
            # Don't use QUBO-expanded n_variables which varies by formulation (esp. BQM)
            n_variables = config * 27 + 27  # Y variables + U variables
            metrics[solver]['n_variables'].append(n_variables)
            
            # Efficiency metric: (1/solve_time) * obj * exp(-viol/constr)
            # Higher is better: fast solve + high quality + few violations
            solve_time = result.get('hybrid_time', result.get('solve_time', 1))
            if solve_time > 0:
                efficiency = (1.0 / solve_time) * norm_qual
            else:
                efficiency = 0
            metrics[solver]['efficiency'].append(efficiency)
    
    # Sort by n_farms
    for solver in metrics:
        sorted_indices = np.argsort(metrics[solver]['n_farms'])
        for key in metrics[solver]:
            metrics[solver][key] = [metrics[solver][key][i] for i in sorted_indices]
    
    return dict(metrics)


def plot_small_scale_comparison(qpu_metrics, classical_metrics, output_path):
    """
    Plot 1: Small scale (10-100 farms) comparison of ALL methods
    Includes: 6 QPU decomposition methods + PuLP + DWave Hybrid + DWave BQM
    
    Time shown:
    - PuLP: solve_time (classical)
    - DWave Hybrid/BQM: hybrid_time
    - QPU methods: qpu_time only (embedding is overhead)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('QPU Decomposition Methods Benchmark (10-100 Farms)\n6 QPU Methods + Classical/Hybrid Baselines', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ===== Plot 1: Solution Quality (Objective) =====
    ax = axes[0, 0]
    
    # Baseline methods
    for method in BASELINE_METHODS:
        if method in classical_metrics:
            farms = [f for f in classical_metrics[method]['n_farms'] if f <= 100]
            objs = [classical_metrics[method]['objective'][i] for i, f in enumerate(classical_metrics[method]['n_farms']) if f <= 100]
            if farms and objs:
                label = {'PuLP': 'PuLP (Optimal)', 'DWave_Hybrid': 'DWave Hybrid CQM', 'DWave_BQM': 'DWave BQM Hybrid'}
                ax.plot(farms, objs, marker=MARKERS.get(method, 'o'), linestyle='-', linewidth=2.5, markersize=10, 
                        color=COLORS.get(method, '#888888'), label=label.get(method, method), alpha=0.9)
    
    # All 6 QPU decomposition methods
    for method in SMALL_SCALE_QPU_METHODS:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            ax.plot(qpu_metrics[method]['n_farms'], qpu_metrics[method]['objective'],
                    marker=MARKERS.get(method, 'o'), linestyle='-', linewidth=2, markersize=8,
                    color=COLORS.get(method, '#888888'), label=method, alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Solution Quality', fontsize=14)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # ===== Plot 2: Optimality Gap =====
    ax = axes[0, 1]
    
    for method in SMALL_SCALE_QPU_METHODS:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            ax.plot(qpu_metrics[method]['n_farms'], qpu_metrics[method]['gap'],
                    marker=MARKERS.get(method, 'o'), linestyle='-', linewidth=2, markersize=8,
                    color=COLORS.get(method, '#888888'), label=method, alpha=0.8)
    
    ax.axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Optimal')
    ax.axhline(y=10, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='10% Gap')
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Gap from Optimal (%)', fontsize=12)
    ax.set_title('Optimality Gap (vs Gurobi)', fontsize=14)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # ===== Plot 3: Solve Time (Log Scale) =====
    # PuLP: solve_time, DWave: hybrid_time, QPU: qpu_time
    ax = axes[1, 0]
    
    # Classical solvers - use solve_time for PuLP
    if 'PuLP' in classical_metrics:
        farms = [f for f in classical_metrics['PuLP']['n_farms'] if f <= 100]
        times = [classical_metrics['PuLP']['solve_time'][i] for i, f in enumerate(classical_metrics['PuLP']['n_farms']) if f <= 100]
        if farms and times and all(t > 0 for t in times):
            ax.semilogy(farms, times, 'o-', linewidth=2.5, markersize=10, 
                        color=COLORS['PuLP'], label='PuLP (solve_time)', alpha=0.9)
    
    # DWave Hybrid - use hybrid_time
    if 'DWave_Hybrid' in classical_metrics:
        farms = [f for f in classical_metrics['DWave_Hybrid']['n_farms'] if f <= 100]
        times = [classical_metrics['DWave_Hybrid']['hybrid_time'][i] for i, f in enumerate(classical_metrics['DWave_Hybrid']['n_farms']) if f <= 100]
        if farms and times and all(t > 0 for t in times):
            ax.semilogy(farms, times, 'D-', linewidth=2.5, markersize=10, 
                        color=COLORS['DWave_Hybrid'], label='DWave Hybrid (hybrid_time)', alpha=0.9)
    
    # DWave BQM - use hybrid_time
    if 'DWave_BQM' in classical_metrics:
        farms = [f for f in classical_metrics['DWave_BQM']['n_farms'] if f <= 100]
        times = [classical_metrics['DWave_BQM']['hybrid_time'][i] for i, f in enumerate(classical_metrics['DWave_BQM']['n_farms']) if f <= 100]
        if farms and times and all(t > 0 for t in times):
            ax.semilogy(farms, times, 'p-', linewidth=2.5, markersize=10, 
                        color=COLORS['DWave_BQM'], label='DWave BQM (hybrid_time)', alpha=0.9)
    
    # QPU methods - use qpu_time only (embedding is overhead)
    for method in SMALL_SCALE_QPU_METHODS:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            qpu_times = qpu_metrics[method]['qpu_time']
            if any(t > 0 for t in qpu_times):
                ax.semilogy(qpu_metrics[method]['n_farms'], qpu_times,
                            marker=MARKERS.get(method, 'o'), linestyle='-', linewidth=2, markersize=8,
                            color=COLORS.get(method, '#888888'), label=f'{method} (qpu_time)', alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Time (seconds, log)', fontsize=12)
    ax.set_title('Solve Time Comparison\n(PuLP: solve_time, DWave: hybrid_time, QPU: qpu_time)', fontsize=12)
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    
    # ===== Plot 4: Efficiency vs Problem Size =====
    # QPU Efficiency = 1/(qpu_time × effective_gap) where effective_gap = gap + 5% per violation
    # Classical Efficiency = obj/time (since gap=0 for optimal)
    # X-axis: number of variables, Y-axis: efficiency
    ax = axes[1, 1]
    
    # Baseline methods
    for method in BASELINE_METHODS:
        if method in classical_metrics:
            indices = [i for i, f in enumerate(classical_metrics[method]['n_farms']) if f <= 100]
            
            n_vars = [classical_metrics[method]['n_variables'][i] for i in indices]
            efficiency = [classical_metrics[method]['efficiency'][i] for i in indices]
            
            if n_vars and efficiency:
                # Sort by n_variables
                sorted_data = sorted(zip(n_vars, efficiency))
                n_vars_s, eff_s = zip(*sorted_data)
                
                label = {'PuLP': 'PuLP', 'DWave_Hybrid': 'DWave Hybrid', 'DWave_BQM': 'DWave BQM'}
                ax.plot(n_vars_s, eff_s, marker=MARKERS.get(method, 'o'), linestyle='-',
                        linewidth=2.5, markersize=10, color=COLORS.get(method, '#888888'), 
                        label=label.get(method, method), alpha=0.9)
    
    # QPU methods
    for method in SMALL_SCALE_QPU_METHODS:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            n_vars = qpu_metrics[method]['n_variables']
            efficiency = qpu_metrics[method]['efficiency']
            
            # Filter valid data points
            valid_data = [(nv, e) for nv, e in zip(n_vars, efficiency) if e > 0]
            if valid_data:
                # Sort by n_variables
                valid_data.sort(key=lambda x: x[0])
                n_vars_s, eff_s = zip(*valid_data)
                
                ax.plot(n_vars_s, eff_s, marker=MARKERS.get(method, 'o'), linestyle='-',
                        linewidth=2, markersize=8, color=COLORS.get(method, '#888888'), 
                        label=method, alpha=0.8)
    
    ax.set_xlabel('Number of Variables', fontsize=12)
    ax.set_ylabel('Efficiency\n(QPU: 1/(t×gap), Classical: obj/t)', fontsize=11)
    ax.set_title('Solution Efficiency vs Problem Size\n(higher is better)', fontsize=12)
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_large_scale_comparison(qpu_metrics, classical_metrics, output_path):
    """
    Plot 2: Large scale (200-1000 farms) comparison with 3 scalable QPU methods
    Includes: 3 QPU methods + PuLP + DWave Hybrid + DWave BQM
    
    Time shown:
    - PuLP: solve_time (classical)
    - DWave Hybrid/BQM: hybrid_time
    - QPU methods: qpu_time only (embedding is overhead)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Large-Scale QPU Benchmark (200-1000 Farms)\n3 Scalable QPU Methods + Classical/Hybrid Baselines', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ===== Plot 1: Solution Quality =====
    ax = axes[0, 0]
    
    # Baseline methods
    for method in BASELINE_METHODS:
        if method in classical_metrics:
            farms = [f for f in classical_metrics[method]['n_farms'] if f >= 200]
            objs = [classical_metrics[method]['objective'][i] for i, f in enumerate(classical_metrics[method]['n_farms']) if f >= 200]
            if farms and objs:
                label = {'PuLP': 'PuLP (Optimal)', 'DWave_Hybrid': 'DWave Hybrid CQM', 'DWave_BQM': 'DWave BQM Hybrid'}
                ax.plot(farms, objs, marker=MARKERS.get(method, 'o'), linestyle='-', 
                        linewidth=2.5, markersize=12, color=COLORS.get(method, '#888888'), 
                        label=label.get(method, method), alpha=0.9)
    
    # Large scale QPU methods (3 methods)
    for method in LARGE_SCALE_QPU_METHODS:
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
    ax.set_yscale('log')
    
    # ===== Plot 2: Optimality Gap (includes hybrid methods) =====
    ax = axes[0, 1]
    
    # Hybrid methods gap (DWave_Hybrid and DWave_BQM)
    # Calculate gap vs PuLP (optimal)
    if 'PuLP' in classical_metrics:
        pulp_farms = classical_metrics['PuLP']['n_farms']
        pulp_objs = classical_metrics['PuLP']['objective']
        pulp_dict = dict(zip(pulp_farms, pulp_objs))
        
        for method in ['DWave_Hybrid', 'DWave_BQM']:
            if method in classical_metrics:
                farms = [f for f in classical_metrics[method]['n_farms'] if f >= 200]
                indices = [i for i, f in enumerate(classical_metrics[method]['n_farms']) if f >= 200]
                objs = [classical_metrics[method]['objective'][i] for i in indices]
                
                # Calculate gap vs PuLP
                gaps = []
                valid_farms = []
                for f, o in zip(farms, objs):
                    if f in pulp_dict and pulp_dict[f] > 0:
                        gap = ((pulp_dict[f] - o) / pulp_dict[f] * 100)
                        gaps.append(gap)
                        valid_farms.append(f)
                
                if valid_farms and gaps:
                    label = {'DWave_Hybrid': 'DWave Hybrid CQM', 'DWave_BQM': 'DWave BQM Hybrid'}
                    ax.plot(valid_farms, gaps, marker=MARKERS.get(method, 'o'), linestyle='-', 
                            linewidth=2.5, markersize=10, color=COLORS.get(method, '#888888'), 
                            label=label.get(method, method), alpha=0.9)
    
    # QPU methods
    for method in LARGE_SCALE_QPU_METHODS:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            farms = [f for f in qpu_metrics[method]['n_farms'] if f >= 200]
            gaps = [qpu_metrics[method]['gap'][i] for i, f in enumerate(qpu_metrics[method]['n_farms']) if f >= 200]
            if farms:
                ax.plot(farms, gaps, marker=MARKERS.get(method, 'o'), linestyle='-', 
                        linewidth=2.5, markersize=10, color=COLORS.get(method, '#888888'), 
                        label=method, alpha=0.8)
    
    ax.axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Optimal')
    ax.axhline(y=10, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='10% Gap')
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Gap from Optimal (%)', fontsize=12)
    ax.set_title('Optimality Gap at Scale\n(includes Hybrid methods)', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ===== Plot 3: Solve Time Comparison =====
    # PuLP: solve_time, DWave: hybrid_time, QPU: qpu_time
    ax = axes[1, 0]
    
    # PuLP - use solve_time
    if 'PuLP' in classical_metrics:
        farms = [f for f in classical_metrics['PuLP']['n_farms'] if f >= 200]
        times = [classical_metrics['PuLP']['solve_time'][i] for i, f in enumerate(classical_metrics['PuLP']['n_farms']) if f >= 200]
        if farms and times and all(t > 0 for t in times):
            ax.semilogy(farms, times, 'o-', linewidth=2.5, markersize=12, 
                        color=COLORS['PuLP'], label='PuLP (solve_time)', alpha=0.9)
    
    # D-Wave Hybrid - use hybrid_time
    if 'DWave_Hybrid' in classical_metrics:
        farms = [f for f in classical_metrics['DWave_Hybrid']['n_farms'] if f >= 200]
        times = [classical_metrics['DWave_Hybrid']['hybrid_time'][i] for i, f in enumerate(classical_metrics['DWave_Hybrid']['n_farms']) if f >= 200]
        if farms and times and all(t > 0 for t in times):
            ax.semilogy(farms, times, 'D-', linewidth=2.5, markersize=12, 
                        color=COLORS['DWave_Hybrid'], label='DWave Hybrid (hybrid_time)', alpha=0.9)
    
    # D-Wave BQM - use hybrid_time
    if 'DWave_BQM' in classical_metrics:
        farms = [f for f in classical_metrics['DWave_BQM']['n_farms'] if f >= 200]
        times = [classical_metrics['DWave_BQM']['hybrid_time'][i] for i, f in enumerate(classical_metrics['DWave_BQM']['n_farms']) if f >= 200]
        if farms and times and all(t > 0 for t in times):
            ax.semilogy(farms, times, 'p-', linewidth=2.5, markersize=12, 
                        color=COLORS['DWave_BQM'], label='DWave BQM (hybrid_time)', alpha=0.9)
    
    # QPU methods - use qpu_time only
    for method in LARGE_SCALE_QPU_METHODS:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            farms = [f for f in qpu_metrics[method]['n_farms'] if f >= 200]
            times = [qpu_metrics[method]['qpu_time'][i] for i, f in enumerate(qpu_metrics[method]['n_farms']) if f >= 200]
            if farms and times and all(t > 0 for t in times):
                ax.semilogy(farms, times, marker=MARKERS.get(method, 'o'), linestyle='-', 
                            linewidth=2.5, markersize=10, color=COLORS.get(method, '#888888'), 
                            label=f'{method} (qpu_time)', alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Time (seconds, log)', fontsize=12)
    ax.set_title('Solve Time Comparison\n(PuLP: solve_time, DWave: hybrid_time, QPU: qpu_time)', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # ===== Plot 4: Efficiency vs Problem Size =====
    # Efficiency = (1/solve_time) * obj * (1 - viol/constr)
    ax = axes[1, 1]
    
    # Baseline methods
    for method in BASELINE_METHODS:
        if method in classical_metrics:
            indices = [i for i, f in enumerate(classical_metrics[method]['n_farms']) if f >= 200]
            
            n_vars = [classical_metrics[method]['n_variables'][i] for i in indices]
            efficiency = [classical_metrics[method]['efficiency'][i] for i in indices]
            
            if n_vars and efficiency:
                # Sort by n_variables
                sorted_data = sorted(zip(n_vars, efficiency))
                n_vars_s, eff_s = zip(*sorted_data)
                
                label = {'PuLP': 'PuLP', 'DWave_Hybrid': 'DWave Hybrid', 'DWave_BQM': 'DWave BQM'}
                ax.plot(n_vars_s, eff_s, marker=MARKERS.get(method, 'o'), linestyle='-',
                        linewidth=2.5, markersize=10, color=COLORS.get(method, '#888888'), 
                        label=label.get(method, method), alpha=0.9)
    
    # QPU methods
    for method in LARGE_SCALE_QPU_METHODS:
        if method in qpu_metrics and len(qpu_metrics[method]['n_farms']) > 0:
            indices = [i for i, f in enumerate(qpu_metrics[method]['n_farms']) if f >= 200]
            
            n_vars = [qpu_metrics[method]['n_variables'][i] for i in indices]
            efficiency = [qpu_metrics[method]['efficiency'][i] for i in indices]
            
            # Filter valid data points
            valid_data = [(nv, e) for nv, e in zip(n_vars, efficiency) if e > 0]
            if valid_data:
                # Sort by n_variables
                valid_data.sort(key=lambda x: x[0])
                n_vars_s, eff_s = zip(*valid_data)
                
                ax.plot(n_vars_s, eff_s, marker=MARKERS.get(method, 'o'), linestyle='-',
                        linewidth=2.5, markersize=10, color=COLORS.get(method, '#888888'), 
                        label=method, alpha=0.8)
    
    ax.set_xlabel('Number of Variables', fontsize=12)
    ax.set_ylabel('Efficiency\n(QPU: 1/(t×gap), Classical: obj/t)', fontsize=11)
    ax.set_title('Solution Efficiency vs Problem Size\n(higher is better)', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_comprehensive_solver_comparison(qpu_metrics_small, qpu_metrics_large, classical_metrics, output_path):
    """
    Plot 3: Comprehensive comparison across all scales and solver types
    Includes all methods with proper time definitions.
    
    Time shown:
    - PuLP: solve_time (classical)
    - DWave Hybrid/BQM: hybrid_time
    - QPU methods: qpu_time only (embedding is overhead)
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Solver Comparison: Classical vs Hybrid vs Pure QPU\nBinary Crop Allocation Problem', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Combine QPU metrics from small and large scale
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
    
    # Define all QPU methods to show
    all_qpu_methods = list(set(SMALL_SCALE_QPU_METHODS + LARGE_SCALE_QPU_METHODS))
    
    # ===== Row 1: Time Comparisons =====
    
    # Plot 1: Time Comparison (solve_time for PuLP, hybrid_time for DWave, qpu_time for QPU)
    ax = axes[0, 0]
    
    # PuLP - use solve_time
    if 'PuLP' in classical_metrics:
        times = [t for t in classical_metrics['PuLP']['solve_time'] if t > 0]
        farms = [classical_metrics['PuLP']['n_farms'][i] for i, t in enumerate(classical_metrics['PuLP']['solve_time']) if t > 0]
        if times:
            ax.semilogy(farms, times, 'o-', linewidth=3, markersize=10, 
                        color=COLORS['PuLP'], label='PuLP (solve_time)', alpha=0.9)
    
    # D-Wave Hybrid - use hybrid_time
    if 'DWave_Hybrid' in classical_metrics:
        times = [t for t in classical_metrics['DWave_Hybrid']['hybrid_time'] if t > 0]
        farms = [classical_metrics['DWave_Hybrid']['n_farms'][i] for i, t in enumerate(classical_metrics['DWave_Hybrid']['hybrid_time']) if t > 0]
        if times:
            ax.semilogy(farms, times, 'D-', linewidth=3, markersize=10, 
                        color=COLORS['DWave_Hybrid'], label='DWave Hybrid (hybrid_time)', alpha=0.9)
    
    # D-Wave BQM - use hybrid_time
    if 'DWave_BQM' in classical_metrics:
        times = [t for t in classical_metrics['DWave_BQM']['hybrid_time'] if t > 0]
        farms = [classical_metrics['DWave_BQM']['n_farms'][i] for i, t in enumerate(classical_metrics['DWave_BQM']['hybrid_time']) if t > 0]
        if times:
            ax.semilogy(farms, times, 'p-', linewidth=3, markersize=10, 
                        color=COLORS['DWave_BQM'], label='DWave BQM (hybrid_time)', alpha=0.9)
    
    # QPU methods - use qpu_time only
    for method in all_qpu_methods:
        if method in all_qpu and len(all_qpu[method]['n_farms']) > 0:
            qpu_times = [t for t in all_qpu[method]['qpu_time'] if t > 0]
            farms = [all_qpu[method]['n_farms'][i] for i, t in enumerate(all_qpu[method]['qpu_time']) if t > 0]
            if qpu_times:
                ax.semilogy(farms, qpu_times, marker=MARKERS.get(method, 'o'), linestyle='-', 
                            linewidth=2, markersize=8, color=COLORS.get(method, '#888888'), 
                            label=f'{method} (qpu_time)', alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Time (seconds, log)', fontsize=12)
    ax.set_title('Solve Time Comparison\n(PuLP: solve_time, DWave: hybrid_time, QPU: qpu_time)', fontsize=12)
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Efficiency vs Problem Size
    # Efficiency = (1/solve_time) * obj * (1 - viol/constr)
    ax = axes[0, 1]
    
    # Baseline methods
    for method in BASELINE_METHODS:
        if method in classical_metrics:
            n_vars = classical_metrics[method]['n_variables']
            efficiency = classical_metrics[method]['efficiency']
            
            # Filter valid data and sort by n_variables
            valid_data = [(nv, e) for nv, e in zip(n_vars, efficiency) if e > 0]
            if valid_data:
                valid_data.sort(key=lambda x: x[0])
                nv_valid, eff_valid = zip(*valid_data)
                
                label = {'PuLP': 'PuLP', 'DWave_Hybrid': 'DWave Hybrid', 'DWave_BQM': 'DWave BQM'}
                ax.plot(nv_valid, eff_valid, marker=MARKERS.get(method, 'o'), linestyle='-',
                        linewidth=2.5, markersize=10, color=COLORS.get(method, '#888888'), 
                        label=label.get(method, method), alpha=0.9)
    
    # QPU methods
    for method in all_qpu_methods:
        if method in all_qpu and len(all_qpu[method]['n_farms']) > 0:
            n_vars = all_qpu[method]['n_variables']
            efficiency = all_qpu[method]['efficiency']
            
            # Filter valid data and sort by n_variables
            valid_data = [(nv, e) for nv, e in zip(n_vars, efficiency) if e > 0]
            if valid_data:
                valid_data.sort(key=lambda x: x[0])
                nv_valid, eff_valid = zip(*valid_data)
                
                ax.plot(nv_valid, eff_valid, marker=MARKERS.get(method, 'o'), linestyle='-',
                        linewidth=2, markersize=8, color=COLORS.get(method, '#888888'), 
                        label=method, alpha=0.8)
    
    ax.set_xlabel('Number of Variables', fontsize=12)
    ax.set_ylabel('Efficiency\n(QPU: 1/(t×gap), Classical: obj/t)', fontsize=11)
    ax.set_title('Solution Efficiency vs Problem Size\n(higher is better)', fontsize=12)
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Efficiency by Scale (colored by scale)
    ax = axes[0, 2]
    
    # Create line plots for each method
    scale_colors = {10: '#1f77b4', 15: '#ff7f0e', 50: '#2ca02c', 100: '#d62728', 
                    200: '#9467bd', 500: '#8c564b', 1000: '#e377c2'}
    
    # For QPU methods, plot n_variables vs efficiency with lines
    for method in all_qpu_methods:
        if method in all_qpu and len(all_qpu[method]['n_farms']) > 0:
            n_vars = all_qpu[method]['n_variables']
            efficiency = all_qpu[method]['efficiency']
            farms = all_qpu[method]['n_farms']
            
            # Filter valid data and sort by n_variables
            valid_data = [(nv, e, f) for nv, e, f in zip(n_vars, efficiency, farms) if e > 0]
            if valid_data:
                valid_data.sort(key=lambda x: x[0])
                nv_valid, eff_valid, f_valid = zip(*valid_data)
                
                # Plot line
                ax.plot(nv_valid, eff_valid, linestyle='-', linewidth=1.5, 
                        color=COLORS.get(method, '#888888'), alpha=0.5)
                
                # Plot markers colored by scale
                for nv, eff, f in zip(nv_valid, eff_valid, f_valid):
                    ax.scatter(nv, eff, s=80, marker=MARKERS.get(method, 'o'),
                              color=scale_colors.get(f, '#888888'), alpha=0.8)
    
    # Add legend for scales
    for scale, color in scale_colors.items():
        ax.scatter([], [], s=80, marker='o', color=color, label=f'{scale} farms', alpha=0.7)
    
    ax.set_xlabel('Number of Variables', fontsize=12)
    ax.set_ylabel('Efficiency\n(1/(qpu_time × gap))', fontsize=11)
    ax.set_title('QPU Efficiency by Scale\n(higher is better)', fontsize=12)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # ===== Row 2: Quality Comparisons =====
    
    # Plot 4: Objective Value - All methods
    ax = axes[1, 0]
    
    # Baseline methods
    for method in BASELINE_METHODS:
        if method in classical_metrics:
            label = {'PuLP': 'PuLP (Optimal)', 'DWave_Hybrid': 'DWave Hybrid CQM', 'DWave_BQM': 'DWave BQM Hybrid'}
            ax.plot(classical_metrics[method]['n_farms'], classical_metrics[method]['objective'],
                    marker=MARKERS.get(method, 'o'), linestyle='-', linewidth=2.5, markersize=10, 
                    color=COLORS.get(method, '#888888'), label=label.get(method, method), alpha=0.9)
    
    # QPU methods
    for method in all_qpu_methods:
        if method in all_qpu and len(all_qpu[method]['n_farms']) > 0:
            ax.plot(all_qpu[method]['n_farms'], all_qpu[method]['objective'],
                    marker=MARKERS.get(method, 'o'), linestyle='-', linewidth=2, markersize=8,
                    color=COLORS.get(method, '#888888'), label=method, alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Solution Quality Comparison', fontsize=14)
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 5: Gap from Optimal - All QPU methods
    ax = axes[1, 1]
    
    for method in all_qpu_methods:
        if method in all_qpu and len(all_qpu[method]['n_farms']) > 0:
            ax.plot(all_qpu[method]['n_farms'], all_qpu[method]['gap'],
                    marker=MARKERS.get(method, 'o'), linestyle='-', linewidth=2, markersize=8,
                    color=COLORS.get(method, '#888888'), label=method, alpha=0.8)
    
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Optimal')
    ax.axhline(y=10, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='10% Gap')
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Gap from Optimal (%)', fontsize=12)
    ax.set_title('Optimality Gap - All QPU Methods', fontsize=14)
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Feasibility (Violations) - All methods
    ax = axes[1, 2]
    
    all_scales = sorted(set([f for m in all_qpu_methods if m in all_qpu for f in all_qpu[m]['n_farms']]))
    
    x = np.arange(len(all_scales))
    n_methods = len([m for m in all_qpu_methods if m in all_qpu])
    width = 0.8 / max(n_methods, 1)
    
    for i, method in enumerate(all_qpu_methods):
        if method in all_qpu:
            violations = []
            for scale in all_scales:
                if scale in all_qpu[method]['n_farms']:
                    idx = all_qpu[method]['n_farms'].index(scale)
                    violations.append(all_qpu[method]['violations'][idx])
                else:
                    violations.append(np.nan)
            ax.bar(x + i*width - (n_methods-1)*width/2, violations, width, label=method, 
                   color=COLORS.get(method, '#888888'), alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=12)
    ax.set_ylabel('Constraint Violations', fontsize=12)
    ax.set_title('Feasibility: Violations by Method', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(all_scales)
    ax.legend(loc='best', fontsize=7, ncol=2)
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
                status = 'Feasible' if viol == 0 else f'{viol} viol.'
                
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
    
    # Add HybridGrid QPU metrics to the qpu_metrics dictionaries
    # These are stored in classical_metrics but need to be in qpu_metrics for QPU plots
    for hybrid_method in ['HybridGrid(5,9)_QPU', 'HybridGrid(10,9)_QPU']:
        if hybrid_method in classical_metrics:
            cm = classical_metrics[hybrid_method]
            # Add to small scale (farms <= 100)
            if qpu_metrics_small is not None:
                qpu_metrics_small[hybrid_method] = {
                    'n_farms': [f for f in cm['n_farms'] if f <= 100],
                    'objective': [cm['objective'][i] for i, f in enumerate(cm['n_farms']) if f <= 100],
                    'gap': [0] * len([f for f in cm['n_farms'] if f <= 100]),  # Gap will be calculated
                    'wall_time': [cm['solve_time'][i] for i, f in enumerate(cm['n_farms']) if f <= 100],
                    'qpu_time': [cm['solve_time'][i] for i, f in enumerate(cm['n_farms']) if f <= 100],
                    'violations': [cm['violations'][i] for i, f in enumerate(cm['n_farms']) if f <= 100],
                    'n_constraints': [cm['n_constraints'][i] for i, f in enumerate(cm['n_farms']) if f <= 100],
                    'normalized_quality': [cm['normalized_quality'][i] for i, f in enumerate(cm['n_farms']) if f <= 100],
                    'n_variables': [cm['n_variables'][i] for i, f in enumerate(cm['n_farms']) if f <= 100],
                    'efficiency': [cm['efficiency'][i] for i, f in enumerate(cm['n_farms']) if f <= 100],
                }
            # Add to large scale (farms >= 200)
            if qpu_metrics_large is not None:
                qpu_metrics_large[hybrid_method] = {
                    'n_farms': [f for f in cm['n_farms'] if f >= 200],
                    'objective': [cm['objective'][i] for i, f in enumerate(cm['n_farms']) if f >= 200],
                    'gap': [0] * len([f for f in cm['n_farms'] if f >= 200]),
                    'wall_time': [cm['solve_time'][i] for i, f in enumerate(cm['n_farms']) if f >= 200],
                    'qpu_time': [cm['solve_time'][i] for i, f in enumerate(cm['n_farms']) if f >= 200],
                    'violations': [cm['violations'][i] for i, f in enumerate(cm['n_farms']) if f >= 200],
                    'n_constraints': [cm['n_constraints'][i] for i, f in enumerate(cm['n_farms']) if f >= 200],
                    'normalized_quality': [cm['normalized_quality'][i] for i, f in enumerate(cm['n_farms']) if f >= 200],
                    'n_variables': [cm['n_variables'][i] for i, f in enumerate(cm['n_farms']) if f >= 200],
                    'efficiency': [cm['efficiency'][i] for i, f in enumerate(cm['n_farms']) if f >= 200],
                }
    
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
