#!/usr/bin/env python3
"""
Comprehensive Study Plots for OQI Phase 3 Report
=================================================

This script generates all plots for the three research studies:

Study 1: Hybrid Solver Benchmarking (CQM/BQM vs Gurobi) - Formulation A (27 crops)
Study 2: 8 QPU Decomposition Methods vs Gurobi - Formulation A (27 crops)  
Study 3: Quantum Advantage Demo - Formulation B (6 families, rotation)

Data Sources:
- Study 1: Benchmarks/COMPREHENSIVE/comprehensive_benchmark_configs_dwave_*.json
- Study 2: @todo/qpu_benchmark_results/qpu_benchmark_20251201_*.json
- Study 3: qpu_hier_repaired.json + @todo/gurobi_timeout_verification/gurobi_timeout_300s_20251222_*.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

DATA_DIR = Path(__file__).parent
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Color schemes
COLORS = {
    'gurobi': '#2ecc71',      # Green
    'cqm': '#3498db',         # Blue  
    'bqm': '#e74c3c',         # Red
    'qpu': '#9b59b6',         # Purple
    'hybrid': '#f39c12',      # Orange
    'timeout': '#95a5a6',     # Gray
}

METHOD_COLORS = {
    'direct_qpu': '#e74c3c',
    'decomposition_PlotBased_QPU': '#3498db',
    'decomposition_Multilevel(5)_QPU': '#2ecc71',
    'decomposition_Multilevel(10)_QPU': '#9b59b6',
    'decomposition_Louvain_QPU': '#f39c12',
    'decomposition_Spectral(10)_QPU': '#1abc9c',
    'cqm_first_PlotBased': '#e67e22',
    'coordinated': '#34495e',
}

METHOD_LABELS = {
    'direct_qpu': 'Direct QPU',
    'decomposition_PlotBased_QPU': 'PlotBased',
    'decomposition_Multilevel(5)_QPU': 'Multilevel(5)',
    'decomposition_Multilevel(10)_QPU': 'Multilevel(10)',
    'decomposition_Louvain_QPU': 'Louvain',
    'decomposition_Spectral(10)_QPU': 'Spectral',
    'cqm_first_PlotBased': 'CQM-First',
    'coordinated': 'Coordinated',
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_json(filepath: Path) -> dict | list | None:
    """Load JSON file safely."""
    if not filepath.exists():
        print(f"  ⚠️ File not found: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️ Error loading {filepath}: {e}")
        return None


def load_study1_data() -> dict:
    """Load Study 1: Comprehensive benchmark data (CQM/BQM vs Gurobi)."""
    print("\n📊 Loading Study 1 data...")
    
    # Primary file with multi-scale results
    benchmark_dir = DATA_DIR / "Benchmarks" / "COMPREHENSIVE"
    
    # Find the best comprehensive file
    candidates = list(benchmark_dir.glob("comprehensive_benchmark_configs_dwave_*.json"))
    if not candidates:
        print("  ⚠️ No comprehensive benchmark files found!")
        return {}
    
    # Use the multi-scale one (configs usually has multiple scales)
    best_file = max(candidates, key=lambda p: p.stat().st_size)
    print(f"  Using: {best_file.name}")
    
    data = load_json(best_file)
    if not data:
        return {}
    
    # Extract patch results
    results = {
        'gurobi': [],
        'dwave_cqm': [],
        'dwave_bqm': [],
    }
    
    patch_results = data.get('patch_results', [])
    print(f"  Found {len(patch_results)} patch result entries")
    
    for entry in patch_results:
        n_units = entry.get('n_units', 0)
        n_vars = entry.get('n_variables', 0)
        solvers = entry.get('solvers', {})
        
        # Gurobi
        if 'gurobi' in solvers:
            g = solvers['gurobi']
            results['gurobi'].append({
                'scale': n_units,
                'n_vars': n_vars,
                'objective': g.get('objective_value', 0),
                'time': g.get('solve_time', 0),
                'status': g.get('status', 'unknown'),
                'feasible': g.get('status') == 'Optimal',
            })
        
        # D-Wave CQM
        if 'dwave_cqm' in solvers:
            c = solvers['dwave_cqm']
            results['dwave_cqm'].append({
                'scale': n_units,
                'n_vars': n_vars,
                'objective': c.get('objective_value', 0),
                'time': c.get('hybrid_time', c.get('solve_time', 0)),
                'qpu_time': c.get('qpu_time', 0),
                'status': c.get('status', 'unknown'),
                'feasible': str(c.get('is_feasible', 'False')).lower() == 'true',
            })
        
        # D-Wave BQM
        if 'dwave_bqm' in solvers:
            b = solvers['dwave_bqm']
            results['dwave_bqm'].append({
                'scale': n_units,
                'n_vars': n_vars,
                'objective': b.get('objective_value', 0),
                'time': b.get('solve_time', b.get('hybrid_time', 0)),
                'qpu_time': b.get('qpu_time', 0),
                'status': b.get('status', 'unknown'),
                'feasible': b.get('success', False),
            })
    
    for method, data_list in results.items():
        print(f"  {method}: {len(data_list)} results")
    
    return results


def load_study2_data() -> dict:
    """Load Study 2: 8 Decomposition methods benchmark data."""
    print("\n📊 Loading Study 2 data...")
    
    benchmark_dir = DATA_DIR / "@todo" / "qpu_benchmark_results"
    
    # Files with all 8 methods
    target_files = [
        benchmark_dir / "qpu_benchmark_20251201_142434.json",  # 25 farms
        benchmark_dir / "qpu_benchmark_20251201_160444.json",  # 10, 15, 50, 100 farms
    ]
    
    all_results = defaultdict(list)
    ground_truth = {}
    
    for filepath in target_files:
        if not filepath.exists():
            print(f"  ⚠️ Missing: {filepath.name}")
            continue
            
        data = load_json(filepath)
        if not data:
            continue
            
        print(f"  Loading: {filepath.name}")
        results = data.get('results', [])
        
        for entry in results:
            n_farms = entry.get('n_farms', 0)
            metadata = entry.get('metadata', {})
            n_foods = metadata.get('n_foods', 27)
            n_vars = metadata.get('n_variables', 0)
            
            # Store ground truth
            gt = entry.get('ground_truth', {})
            if gt:
                ground_truth[n_farms] = {
                    'objective': gt.get('objective', 0),
                    'time': gt.get('solve_time', 0),
                }
            
            # Extract method results
            method_results = entry.get('method_results', {})
            for method_name, result in method_results.items():
                timings = result.get('timings', {})
                
                all_results[method_name].append({
                    'n_farms': n_farms,
                    'n_foods': n_foods,
                    'n_vars': n_vars,
                    'objective': result.get('objective', 0),
                    'total_time': timings.get('total', 0),
                    'qpu_time': timings.get('qpu_access_total', 0),
                    'embedding_time': timings.get('embedding_total', 0),
                    'feasible': result.get('feasible', False),
                    'violations': result.get('violations', 0),
                    'n_partitions': result.get('n_partitions', 1),
                })
    
    print(f"  Methods found: {list(all_results.keys())}")
    for method, data_list in all_results.items():
        scales = sorted(set(r['n_farms'] for r in data_list))
        print(f"    {METHOD_LABELS.get(method, method)}: scales {scales}")
    
    return {'methods': dict(all_results), 'ground_truth': ground_truth}


def load_study3_data() -> dict:
    """Load Study 3: Quantum Advantage data (QPU vs Gurobi timeout)."""
    print("\n📊 Loading Study 3 data...")
    
    # QPU hierarchical results
    qpu_file = DATA_DIR / "qpu_hier_repaired.json"
    qpu_data = load_json(qpu_file)
    
    # Try multiple Gurobi sources in order of preference
    gurobi_data = None
    
    # 1. Try gurobi_timeout_verification folder (300s tests)
    gurobi_dir = DATA_DIR / "@todo" / "gurobi_timeout_verification"
    gurobi_files = sorted(gurobi_dir.glob("gurobi_timeout_test_*.json"), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
    
    for gfile in gurobi_files[:3]:  # Try top 3 most recent
        test_data = load_json(gfile)
        if test_data and isinstance(test_data, list) and len(test_data) > 0:
            # Check if it has rotation scenarios
            scenarios = [e.get('metadata', e).get('scenario', '') for e in test_data]
            if any('rotation' in s for s in scenarios):
                print(f"  Using Gurobi file: {gfile.name}")
                gurobi_data = test_data
                break
    
    # 2. Fallback to gurobi_baseline_60s.json
    if not gurobi_data:
        gurobi_file = DATA_DIR / "gurobi_baseline_60s.json"
        if gurobi_file.exists():
            print(f"  Using fallback: gurobi_baseline_60s.json")
            baseline_data = load_json(gurobi_file)
            if baseline_data:
                # Convert schema v1.0 format to array format
                gurobi_data = []
                for run in baseline_data.get('runs', []):
                    timing = run.get('timing', {})
                    gurobi_data.append({
                        'metadata': {
                            'scenario': run.get('scenario_name', ''),
                            'n_farms': run.get('n_farms', 0),
                            'n_foods': run.get('n_foods', 0),
                            'n_periods': run.get('n_periods', 3),
                        },
                        'result': {
                            'scenario': run.get('scenario_name', ''),
                            'n_vars': run.get('n_vars', 0),
                            'status': run.get('status', ''),
                            'objective_value': run.get('objective_miqp', 0),
                            'solve_time': timing.get('total_wall_time', 0),
                            'mip_gap': run.get('mip_gap', 0),
                            'hit_timeout': 'timeout' in run.get('status', '').lower(),
                        }
                    })
    
    results = {
        'qpu': [],
        'gurobi': [],
    }
    
    # Process QPU data
    if qpu_data:
        for run in qpu_data.get('runs', []):
            timing = run.get('timing', {})
            violations = run.get('constraint_violations', {})
            
            results['qpu'].append({
                'scenario': run.get('scenario_name', ''),
                'n_farms': run.get('n_farms', 0),
                'n_foods': run.get('n_foods', 0),
                'n_vars': run.get('n_vars', 0),
                'objective': run.get('objective_miqp', 0),
                'benefit': -run.get('objective_miqp', 0),  # SIGN CORRECTION!
                'total_time': timing.get('total_wall_time', 0),
                'qpu_time': timing.get('qpu_access_time', 0),
                'feasible': run.get('feasible', False),
                'violations': violations.get('total_violations', 0),
                'formulation': '6-family' if run.get('n_foods', 0) <= 6 else '27-food',
            })
        print(f"  QPU: {len(results['qpu'])} runs")
    
    # Process Gurobi data
    if gurobi_data:
        for entry in gurobi_data:
            meta = entry.get('metadata', entry)
            result = entry.get('result', entry)
            
            scenario = meta.get('scenario', result.get('scenario', ''))
            results['gurobi'].append({
                'scenario': scenario,
                'n_farms': meta.get('n_farms', 0),
                'n_foods': meta.get('n_foods', 0),
                'n_vars': result.get('n_vars', 0),
                'objective': result.get('objective_value', 0),
                'benefit': result.get('objective_value', 0),  # Already positive
                'total_time': result.get('solve_time', 0),
                'timeout': result.get('hit_timeout', False) or 'timeout' in str(result.get('status', '')).lower(),
                'mip_gap': result.get('mip_gap', 0),
                'formulation': '6-family' if meta.get('n_foods', 0) <= 6 else '27-food',
            })
        print(f"  Gurobi: {len(results['gurobi'])} runs")
    
    return results


# ============================================================================
# STUDY 1 PLOTS: Hybrid Solver Benchmarking
# ============================================================================

def plot_study1(data: dict) -> None:
    """Generate Study 1 plots: CQM/BQM vs Gurobi."""
    if not data:
        print("  ⚠️ No Study 1 data to plot")
        return
    
    print("\n🎨 Generating Study 1 plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Study 1: Hybrid Solver Benchmarking (Formulation A - 27 Crops)', fontsize=14, fontweight='bold')
    
    # Aggregate by scale
    def aggregate_by_scale(results: list) -> dict:
        by_scale = defaultdict(list)
        for r in results:
            by_scale[r['scale']].append(r)
        return {s: {
            'objective': np.mean([r['objective'] for r in rs]),
            'objective_std': np.std([r['objective'] for r in rs]),
            'time': np.mean([r['time'] for r in rs]),
            'time_std': np.std([r['time'] for r in rs]),
            'feasible_rate': np.mean([r['feasible'] for r in rs]),
            'count': len(rs),
        } for s, rs in by_scale.items()}
    
    gurobi_agg = aggregate_by_scale(data.get('gurobi', []))
    cqm_agg = aggregate_by_scale(data.get('dwave_cqm', []))
    bqm_agg = aggregate_by_scale(data.get('dwave_bqm', []))
    
    scales = sorted(set(gurobi_agg.keys()) | set(cqm_agg.keys()) | set(bqm_agg.keys()))
    
    if not scales:
        print("  ⚠️ No scales found in data")
        return
    
    x = np.arange(len(scales))
    width = 0.25
    
    # Plot 1: Solve Time Comparison
    ax = axes[0, 0]
    gurobi_times = [gurobi_agg.get(s, {}).get('time', 0) for s in scales]
    cqm_times = [cqm_agg.get(s, {}).get('time', 0) for s in scales]
    bqm_times = [bqm_agg.get(s, {}).get('time', 0) for s in scales]
    
    ax.bar(x - width, gurobi_times, width, label='Gurobi', color=COLORS['gurobi'], alpha=0.8)
    ax.bar(x, cqm_times, width, label='D-Wave CQM', color=COLORS['cqm'], alpha=0.8)
    ax.bar(x + width, bqm_times, width, label='D-Wave BQM', color=COLORS['bqm'], alpha=0.8)
    
    ax.set_xlabel('Number of Patches')
    ax.set_ylabel('Solve Time (s)')
    ax.set_title('(a) Solve Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Objective Value Comparison
    ax = axes[0, 1]
    gurobi_obj = [gurobi_agg.get(s, {}).get('objective', 0) for s in scales]
    cqm_obj = [cqm_agg.get(s, {}).get('objective', 0) for s in scales]
    bqm_obj = [bqm_agg.get(s, {}).get('objective', 0) for s in scales]
    
    ax.plot(scales, gurobi_obj, 'o-', label='Gurobi', color=COLORS['gurobi'], linewidth=2, markersize=8)
    ax.plot(scales, cqm_obj, 's--', label='D-Wave CQM', color=COLORS['cqm'], linewidth=2, markersize=8)
    ax.plot(scales, bqm_obj, '^:', label='D-Wave BQM', color=COLORS['bqm'], linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Patches')
    ax.set_ylabel('Objective Value')
    ax.set_title('(b) Solution Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Speedup Ratio
    ax = axes[1, 0]
    cqm_speedup = [gurobi_agg.get(s, {}).get('time', 1) / max(cqm_agg.get(s, {}).get('time', 1), 0.001) for s in scales]
    bqm_speedup = [gurobi_agg.get(s, {}).get('time', 1) / max(bqm_agg.get(s, {}).get('time', 1), 0.001) for s in scales]
    
    ax.bar(x - width/2, cqm_speedup, width, label='Gurobi / CQM', color=COLORS['cqm'], alpha=0.8)
    ax.bar(x + width/2, bqm_speedup, width, label='Gurobi / BQM', color=COLORS['bqm'], alpha=0.8)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Break-even')
    
    ax.set_xlabel('Number of Patches')
    ax.set_ylabel('Speedup Ratio (Gurobi time / Hybrid time)')
    ax.set_title('(c) Speedup Analysis (>1 = Gurobi faster)')
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Feasibility Rate
    ax = axes[1, 1]
    gurobi_feas = [gurobi_agg.get(s, {}).get('feasible_rate', 0) * 100 for s in scales]
    cqm_feas = [cqm_agg.get(s, {}).get('feasible_rate', 0) * 100 for s in scales]
    bqm_feas = [bqm_agg.get(s, {}).get('feasible_rate', 0) * 100 for s in scales]
    
    ax.bar(x - width, gurobi_feas, width, label='Gurobi', color=COLORS['gurobi'], alpha=0.8)
    ax.bar(x, cqm_feas, width, label='D-Wave CQM', color=COLORS['cqm'], alpha=0.8)
    ax.bar(x + width, bqm_feas, width, label='D-Wave BQM', color=COLORS['bqm'], alpha=0.8)
    
    ax.set_xlabel('Number of Patches')
    ax.set_ylabel('Feasibility Rate (%)')
    ax.set_title('(d) Solution Feasibility')
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'study1_hybrid_benchmark.png')
    plt.close()
    print(f"  ✓ Saved: study1_hybrid_benchmark.png")


# ============================================================================
# STUDY 2 PLOTS: 8 Decomposition Methods
# ============================================================================

def plot_study2(data: dict) -> None:
    """Generate Study 2 plots: 8 Decomposition Methods."""
    if not data or 'methods' not in data:
        print("  ⚠️ No Study 2 data to plot")
        return
    
    print("\n🎨 Generating Study 2 plots...")
    
    methods_data = data['methods']
    ground_truth = data.get('ground_truth', {})
    
    # Get all scales
    all_scales = set()
    for method_results in methods_data.values():
        for r in method_results:
            all_scales.add(r['n_farms'])
    scales = sorted(all_scales)
    
    if not scales:
        print("  ⚠️ No scales found")
        return
    
    # Filter to methods in METHOD_LABELS (the 8 main methods)
    target_methods = [m for m in methods_data.keys() if m in METHOD_LABELS]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Study 2: 8 QPU Decomposition Methods (Formulation A - 27 Crops)', fontsize=14, fontweight='bold')
    
    # Plot 1: Total Time by Method
    ax = axes[0, 0]
    x = np.arange(len(scales))
    width = 0.8 / len(target_methods)
    
    for i, method in enumerate(target_methods):
        method_results = methods_data.get(method, [])
        times = []
        for scale in scales:
            scale_results = [r for r in method_results if r['n_farms'] == scale]
            if scale_results:
                times.append(np.mean([r['total_time'] for r in scale_results]))
            else:
                times.append(0)
        
        offset = (i - len(target_methods)/2 + 0.5) * width
        ax.bar(x + offset, times, width, 
               label=METHOD_LABELS.get(method, method),
               color=METHOD_COLORS.get(method, f'C{i}'),
               alpha=0.8)
    
    # Add Gurobi ground truth
    gurobi_times = [ground_truth.get(s, {}).get('time', 0) for s in scales]
    ax.plot(x, gurobi_times, 'k*-', markersize=12, linewidth=2, label='Gurobi (baseline)')
    
    ax.set_xlabel('Number of Farms')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('(a) Total Solve Time')
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: QPU Time Only
    ax = axes[0, 1]
    for i, method in enumerate(target_methods):
        method_results = methods_data.get(method, [])
        qpu_times = []
        for scale in scales:
            scale_results = [r for r in method_results if r['n_farms'] == scale]
            if scale_results:
                qpu_times.append(np.mean([r['qpu_time'] for r in scale_results]))
            else:
                qpu_times.append(0)
        
        ax.plot(scales, qpu_times, 'o-', 
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, f'C{i}'),
                linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of Farms')
    ax.set_ylabel('Pure QPU Time (s)')
    ax.set_title('(b) Pure QPU Access Time')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Objective Gap vs Gurobi
    ax = axes[0, 2]
    for i, method in enumerate(target_methods):
        method_results = methods_data.get(method, [])
        gaps = []
        for scale in scales:
            scale_results = [r for r in method_results if r['n_farms'] == scale]
            gurobi_obj = ground_truth.get(scale, {}).get('objective', 1)
            if scale_results and gurobi_obj:
                method_obj = np.mean([r['objective'] for r in scale_results])
                gap = abs(method_obj - gurobi_obj) / abs(gurobi_obj) * 100
                gaps.append(gap)
            else:
                gaps.append(np.nan)
        
        ax.plot(scales, gaps, 'o-',
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, f'C{i}'),
                linewidth=2, markersize=6)
    
    ax.axhline(y=20, color='red', linestyle='--', linewidth=1, alpha=0.7, label='20% threshold')
    ax.set_xlabel('Number of Farms')
    ax.set_ylabel('Gap vs Gurobi (%)')
    ax.set_title('(c) Optimality Gap')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Time Decomposition (Stacked)
    ax = axes[1, 0]
    
    # Use one representative method for stacked bar
    method = 'decomposition_PlotBased_QPU'
    if method in methods_data:
        method_results = methods_data[method]
        qpu_times = []
        embedding_times = []
        other_times = []
        
        for scale in scales:
            scale_results = [r for r in method_results if r['n_farms'] == scale]
            if scale_results:
                avg_qpu = np.mean([r['qpu_time'] for r in scale_results])
                avg_embed = np.mean([r['embedding_time'] for r in scale_results])
                avg_total = np.mean([r['total_time'] for r in scale_results])
                qpu_times.append(avg_qpu)
                embedding_times.append(avg_embed)
                other_times.append(max(0, avg_total - avg_qpu - avg_embed))
            else:
                qpu_times.append(0)
                embedding_times.append(0)
                other_times.append(0)
        
        ax.bar(scales, qpu_times, label='QPU Access', color='#9b59b6')
        ax.bar(scales, embedding_times, bottom=qpu_times, label='Embedding', color='#3498db')
        ax.bar(scales, other_times, bottom=np.array(qpu_times) + np.array(embedding_times), 
               label='Other', color='#95a5a6')
    
    ax.set_xlabel('Number of Farms')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'(d) Time Decomposition ({METHOD_LABELS.get(method, method)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Scaling Exponents
    ax = axes[1, 1]
    
    for i, method in enumerate(target_methods[:4]):  # Limit to 4 for clarity
        method_results = methods_data.get(method, [])
        if not method_results:
            continue
            
        x_data = []
        y_data = []
        for r in method_results:
            if r['total_time'] > 0 and r['n_farms'] > 0:
                x_data.append(r['n_farms'])
                y_data.append(r['total_time'])
        
        if len(x_data) >= 2:
            ax.loglog(x_data, y_data, 'o',
                      label=METHOD_LABELS.get(method, method),
                      color=METHOD_COLORS.get(method, f'C{i}'),
                      markersize=8, alpha=0.7)
            
            # Fit power law
            try:
                log_x = np.log(x_data)
                log_y = np.log(y_data)
                slope, intercept = np.polyfit(log_x, log_y, 1)
                x_fit = np.linspace(min(x_data), max(x_data), 50)
                y_fit = np.exp(intercept) * x_fit ** slope
                ax.loglog(x_fit, y_fit, '--', color=METHOD_COLORS.get(method, f'C{i}'), alpha=0.5)
            except:
                pass
    
    ax.set_xlabel('Number of Farms (log)')
    ax.set_ylabel('Total Time (s, log)')
    ax.set_title('(e) Scaling Behavior')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Feasibility Rate
    ax = axes[1, 2]
    
    for i, method in enumerate(target_methods):
        method_results = methods_data.get(method, [])
        feas_rates = []
        for scale in scales:
            scale_results = [r for r in method_results if r['n_farms'] == scale]
            if scale_results:
                feas_rates.append(np.mean([r['feasible'] for r in scale_results]) * 100)
            else:
                feas_rates.append(0)
        
        ax.plot(scales, feas_rates, 'o-',
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, f'C{i}'),
                linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of Farms')
    ax.set_ylabel('Feasibility Rate (%)')
    ax.set_title('(f) Solution Feasibility')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'study2_decomposition_methods.png')
    plt.close()
    print(f"  ✓ Saved: study2_decomposition_methods.png")


# ============================================================================
# STUDY 3 PLOTS: Quantum Advantage
# ============================================================================

def plot_study3(data: dict) -> None:
    """Generate Study 3 plots: Quantum Advantage (sign-corrected!)."""
    if not data:
        print("  ⚠️ No Study 3 data to plot")
        return
    
    print("\n🎨 Generating Study 3 plots...")
    
    qpu_results = data.get('qpu', [])
    gurobi_results = data.get('gurobi', [])
    
    if not qpu_results or not gurobi_results:
        print("  ⚠️ Missing QPU or Gurobi results")
        return
    
    # Match scenarios
    qpu_by_scenario = {r['scenario']: r for r in qpu_results}
    gurobi_by_scenario = {r['scenario']: r for r in gurobi_results}
    
    common_scenarios = sorted(set(qpu_by_scenario.keys()) & set(gurobi_by_scenario.keys()),
                              key=lambda s: qpu_by_scenario[s]['n_vars'])
    
    if not common_scenarios:
        print("  ⚠️ No matching scenarios found")
        return
    
    print(f"  Found {len(common_scenarios)} matching scenarios")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Study 3: Quantum Advantage (Formulation B - 6 Families, Rotation)\n'
                 '⚠️ Sign-Corrected: QPU benefit = -1 × objective', fontsize=14, fontweight='bold')
    
    # Prepare data
    scenarios = common_scenarios
    qpu_benefits = [qpu_by_scenario[s]['benefit'] for s in scenarios]
    gurobi_benefits = [gurobi_by_scenario[s]['benefit'] for s in scenarios]
    qpu_times = [qpu_by_scenario[s]['total_time'] for s in scenarios]
    gurobi_times = [gurobi_by_scenario[s]['total_time'] for s in scenarios]
    n_vars = [qpu_by_scenario[s]['n_vars'] for s in scenarios]
    violations = [qpu_by_scenario[s]['violations'] for s in scenarios]
    
    # Plot 1: Benefit Comparison (SIGN CORRECTED)
    ax = axes[0, 0]
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax.bar(x - width/2, gurobi_benefits, width, label='Gurobi (300s timeout)', color=COLORS['gurobi'], alpha=0.8)
    ax.bar(x + width/2, qpu_benefits, width, label='QPU Hierarchical', color=COLORS['qpu'], alpha=0.8)
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Benefit Achieved')
    ax.set_title('(a) Benefit Comparison (Higher = Better)')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('rotation_', '').replace('farms_', 'f_') for s in scenarios], 
                       rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Advantage Ratio
    ax = axes[0, 1]
    advantage_ratios = [qpu_benefits[i] / max(gurobi_benefits[i], 0.001) for i in range(len(scenarios))]
    
    colors = ['#2ecc71' if r > 1 else '#e74c3c' for r in advantage_ratios]
    ax.bar(x, advantage_ratios, color=colors, alpha=0.8)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Break-even')
    ax.axhline(y=3.8, color='blue', linestyle=':', linewidth=1.5, label='Avg advantage (3.8×)')
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Advantage Ratio (QPU / Gurobi)')
    ax.set_title('(b) Quantum Advantage Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('rotation_', '').replace('farms_', 'f_') for s in scenarios], 
                       rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Time Comparison
    ax = axes[0, 2]
    
    ax.bar(x - width/2, gurobi_times, width, label='Gurobi', color=COLORS['gurobi'], alpha=0.8)
    ax.bar(x + width/2, qpu_times, width, label='QPU', color=COLORS['qpu'], alpha=0.8)
    
    # Mark timeouts
    for i, s in enumerate(scenarios):
        if gurobi_by_scenario[s].get('timeout', False):
            ax.annotate('T', (i - width/2, gurobi_times[i] + 5), ha='center', fontsize=8, color='red')
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Solve Time (s)')
    ax.set_title('(c) Time Comparison (T = timeout)')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('rotation_', '').replace('farms_', 'f_') for s in scenarios], 
                       rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Speedup vs Problem Size
    ax = axes[1, 0]
    speedups = [gurobi_times[i] / max(qpu_times[i], 0.001) for i in range(len(scenarios))]
    
    ax.scatter(n_vars, speedups, c=COLORS['qpu'], s=100, alpha=0.7, edgecolors='black')
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Break-even')
    
    # Fit trend line
    if len(n_vars) > 2:
        z = np.polyfit(n_vars, speedups, 1)
        p = np.poly1d(z)
        ax.plot(sorted(n_vars), p(sorted(n_vars)), 'r--', alpha=0.5, label='Trend')
    
    ax.set_xlabel('Number of Variables')
    ax.set_ylabel('Speedup (Gurobi time / QPU time)')
    ax.set_title('(d) Speedup vs Problem Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Violations vs Benefit
    ax = axes[1, 1]
    
    ax.scatter(violations, qpu_benefits, c=COLORS['qpu'], s=100, alpha=0.7, 
               edgecolors='black', label='QPU benefit')
    
    # Add trend line
    if len([v for v in violations if v > 0]) > 1:
        mask = np.array(violations) > 0
        if sum(mask) > 1:
            z = np.polyfit(np.array(violations)[mask], np.array(qpu_benefits)[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(violations), max(violations), 50)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label='Trend')
    
    ax.set_xlabel('Number of Violations')
    ax.set_ylabel('QPU Benefit Achieved')
    ax.set_title('(e) Violations vs Benefit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary Statistics
    ax = axes[1, 2]
    
    avg_advantage = np.mean(advantage_ratios)
    pct_advantage = sum(1 for r in advantage_ratios if r > 1) / len(advantage_ratios) * 100
    avg_qpu_time = np.mean(qpu_times)
    avg_gurobi_time = np.mean(gurobi_times)
    
    metrics = ['Avg Advantage\nRatio', 'Scenarios with\nAdvantage (%)', 
               'Avg QPU\nTime (s)', 'Avg Gurobi\nTime (s)']
    values = [avg_advantage, pct_advantage, avg_qpu_time, avg_gurobi_time]
    colors = [COLORS['qpu'], COLORS['qpu'], COLORS['qpu'], COLORS['gurobi']]
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Value')
    ax.set_title('(f) Summary Statistics')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'study3_quantum_advantage.png')
    plt.close()
    print(f"  ✓ Saved: study3_quantum_advantage.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate all study plots."""
    print("=" * 70)
    print("  OQI PHASE 3 REPORT - COMPREHENSIVE PLOTTING")
    print("=" * 70)
    
    # Load all data
    study1_data = load_study1_data()
    study2_data = load_study2_data()
    study3_data = load_study3_data()
    
    # Generate plots
    plot_study1(study1_data)
    plot_study2(study2_data)
    plot_study3(study3_data)
    
    print("\n" + "=" * 70)
    print("  ✅ ALL PLOTS GENERATED")
    print(f"  📁 Output directory: {PLOTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
