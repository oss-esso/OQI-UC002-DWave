#!/usr/bin/env python3
"""
Plot ALL runtime traces from ALL data files.
One line per solver per file.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'Data'
PLOTS_DIR = SCRIPT_DIR.parent / 'Plots'
PLOTS_DIR.mkdir(exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 9

# Color cycle for many lines
COLORS = plt.cm.tab20(np.linspace(0, 1, 20)).tolist() + plt.cm.tab20b(np.linspace(0, 1, 20)).tolist()

def extract_all_traces(json_file: Path) -> List[Dict]:
    """Extract ALL runtime traces from a JSON file."""
    traces = []
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except:
        return traces
    
    filename = json_file.name
    
    # Type 1: Statistical comparison format
    if isinstance(data, dict) and 'results_by_size' in data:
        for size_str, size_data in data['results_by_size'].items():
            n_vars = size_data.get('n_variables', 0)
            
            for method_name, method_data in size_data.get('methods', {}).items():
                runs = method_data.get('runs', [])
                for run in runs:
                    if run.get('success') and run.get('wall_time', 0) > 0:
                        traces.append({
                            'file': filename,
                            'method': method_name,
                            'n_vars': n_vars,
                            'runtime': run['wall_time'],
                            'trace_name': f"{filename.replace('.json', '')}_{method_name}"
                        })
    
    # Type 2: Hierarchical format
    elif isinstance(data, dict) and any(k in ['25', '50', '100'] for k in data.keys()):
        for size_str, size_data in data.items():
            if not size_str.isdigit():
                continue
            
            n_vars = size_data.get('data_info', {}).get('n_variables_aggregated', 0)
            
            # Gurobi runs
            for run in size_data.get('gurobi', []):
                if run.get('success') and run.get('solve_time', 0) > 0:
                    traces.append({
                        'file': filename,
                        'method': 'gurobi',
                        'n_vars': n_vars,
                        'runtime': run['solve_time'],
                        'trace_name': f"{filename.replace('.json', '')}_gurobi"
                    })
            
            # Hierarchical runs
            for run in size_data.get('hierarchical_qpu', []):
                if run.get('success') and run.get('solve_time', 0) > 0:
                    traces.append({
                        'file': filename,
                        'method': 'hierarchical_qpu',
                        'n_vars': n_vars,
                        'runtime': run['solve_time'],
                        'trace_name': f"{filename.replace('.json', '')}_hierarchical_qpu"
                    })
    
    # Type 3: Roadmap/benchmark with results list
    elif isinstance(data, dict) and 'results' in data and isinstance(data['results'], list):
        for result in data['results']:
            n_vars = result.get('n_variables', 0)
            method = result.get('method', 'unknown')
            runtime = result.get('solve_time', result.get('total_time', result.get('wall_time', 0)))
            
            if n_vars > 0 and runtime > 0 and result.get('success', True):
                traces.append({
                    'file': filename,
                    'method': method,
                    'n_vars': n_vars,
                    'runtime': runtime,
                    'trace_name': f"{filename.replace('.json', '')}_{method}"
                })
    
    # Type 4: Scaling test (list with formulation field)
    elif isinstance(data, list) and len(data) > 0 and 'formulation' in data[0]:
        for result in data:
            n_vars = result.get('n_vars', 0)
            
            # Gurobi
            gt = result.get('gurobi_time', 0)
            if n_vars > 0 and gt > 0:
                traces.append({
                    'file': filename,
                    'method': 'gurobi',
                    'n_vars': n_vars,
                    'runtime': gt,
                    'trace_name': f"{filename.replace('.json', '')}_gurobi"
                })
            
            # Quantum
            qt = result.get('quantum_time', 0)
            if n_vars > 0 and qt > 0:
                traces.append({
                    'file': filename,
                    'method': 'quantum',
                    'n_vars': n_vars,
                    'runtime': qt,
                    'trace_name': f"{filename.replace('.json', '')}_quantum"
                })
    
    # Type 5: Significant scenarios (list with scenario field)
    elif isinstance(data, list) and len(data) > 0 and 'scenario' in data[0]:
        for result in data:
            n_vars = result.get('n_vars', 0)
            
            # Gurobi
            gt = result.get('gurobi_runtime', 0)
            if n_vars > 0 and gt > 0:
                traces.append({
                    'file': filename,
                    'method': 'gurobi',
                    'n_vars': n_vars,
                    'runtime': gt,
                    'trace_name': f"{filename.replace('.json', '')}_gurobi"
                })
            
            # QPU
            qt = result.get('qpu_runtime', 0)
            qpu_method = result.get('qpu_method', 'qpu')
            if n_vars > 0 and qt > 0 and result.get('qpu_status') != 'error':
                traces.append({
                    'file': filename,
                    'method': qpu_method,
                    'n_vars': n_vars,
                    'runtime': qt,
                    'trace_name': f"{filename.replace('.json', '')}_{qpu_method}"
                })
    
    # Type 6: Single benchmark (method + timings)
    elif isinstance(data, dict) and 'method' in data and 'timings' in data:
        runtime = data.get('timings', {}).get('total', 0)
        method = data.get('method', 'unknown')
        
        # Estimate n_vars from solution keys if present
        n_vars = 0
        if 'family_solution' in data:
            n_vars = len(data['family_solution'])
        
        if n_vars > 0 and runtime > 0:
            traces.append({
                'file': filename,
                'method': method,
                'n_vars': n_vars,
                'runtime': runtime,
                'trace_name': f"{filename.replace('.json', '')}_{method}"
            })
    
    return traces


def main():
    print("="*80)
    print("RUNTIME LINEPLOT: ALL SOLVERS FROM ALL FILES")
    print("="*80)
    print()
    
    # Get all JSON files
    json_files = list(DATA_DIR.glob('*.json'))
    json_files.extend(DATA_DIR.glob('significant/*.json'))
    
    print(f"Found {len(json_files)} JSON files")
    print()
    
    # Extract all traces
    all_traces = []
    file_method_counts = {}
    
    for json_file in sorted(json_files):
        traces = extract_all_traces(json_file)
        
        if traces:
            all_traces.extend(traces)
            
            # Count methods per file
            methods_in_file = set(t['method'] for t in traces)
            file_method_counts[json_file.name] = {
                'methods': methods_in_file,
                'count': len(methods_in_file),
                'total_points': len(traces)
            }
            
            print(f"✓ {json_file.name}")
            print(f"    Methods: {', '.join(sorted(methods_in_file))}")
            print(f"    Data points: {len(traces)}")
        else:
            print(f"✗ {json_file.name} (no data extracted)")
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files with data: {len(file_method_counts)}/{len(json_files)}")
    print(f"Total traces (file+method combinations): {len(set(t['trace_name'] for t in all_traces))}")
    print(f"Total data points: {len(all_traces)}")
    print()
    
    # Count unique methods across all files
    all_methods = set(t['method'] for t in all_traces)
    print(f"Unique methods across all files: {len(all_methods)}")
    print(f"  {', '.join(sorted(all_methods))}")
    print()
    
    if not all_traces:
        print("No data to plot!")
        return
    
    # Group by trace_name
    traces_grouped = {}
    for trace in all_traces:
        name = trace['trace_name']
        if name not in traces_grouped:
            traces_grouped[name] = []
        traces_grouped[name].append((trace['n_vars'], trace['runtime']))
    
    print(f"Plotting {len(traces_grouped)} unique traces...")
    print()
    
    # Filter outliers and clean data
    print("Cleaning data...")
    traces_cleaned = {}
    outliers_removed = 0
    
    for trace_name, points in traces_grouped.items():
        if len(points) < 2:
            traces_cleaned[trace_name] = points
            continue
        
        # Sort by n_vars
        points_sorted = sorted(points, key=lambda x: x[0])
        
        # Remove obvious outliers: runtime jumps > 100x or < 0.01x from previous
        cleaned = [points_sorted[0]]
        for i in range(1, len(points_sorted)):
            prev_runtime = cleaned[-1][1]
            curr_runtime = points_sorted[i][1]
            
            # Check if this point is reasonable
            if prev_runtime > 0:
                ratio = curr_runtime / prev_runtime
                # Allow 100x increase or 0.01x decrease max
                if 0.01 <= ratio <= 100:
                    cleaned.append(points_sorted[i])
                else:
                    outliers_removed += 1
            else:
                cleaned.append(points_sorted[i])
        
        if cleaned:
            traces_cleaned[trace_name] = cleaned
    
    print(f"  Removed {outliers_removed} outliers")
    print()
    
    # Create plot with better styling
    fig, ax = plt.subplots(figsize=(20, 11))
    
    # Categorize traces by configuration (based on source file analysis)
    config_a_traces = []  # statistical_comparison (5-25 farms, 6 families, 300s timeout)
    config_b_traces = []  # hierarchical (25-100 farms, 27->6, 300s timeout)
    config_c_traces = []  # qpu_benchmark/roadmap (various, 200s timeout)
    config_d_traces = []  # significant_scenarios (5-100 farms, 100s timeout)
    
    gurobi_traces = []
    quantum_traces = []
    
    for trace_name in sorted(traces_cleaned.keys()):
        # Categorize by config
        if 'statistical_comparison' in trace_name:
            config_a_traces.append(trace_name)
        elif 'hierarchical' in trace_name and ('results' in trace_name or '25' in trace_name or '50' in trace_name or '100' in trace_name):
            config_b_traces.append(trace_name)
        elif 'roadmap' in trace_name or 'hybrid_test' in trace_name:
            config_c_traces.append(trace_name)
        elif 'benchmark_results' in trace_name or 'significant' in trace_name:
            config_d_traces.append(trace_name)
        elif 'scaling_test' in trace_name:
            config_c_traces.append(trace_name)
        elif any(x in trace_name for x in ['benchmark_50', 'benchmark_100', 'rotation_small']):
            config_b_traces.append(trace_name)
        
        # Categorize by solver type
        if 'gurobi' in trace_name.lower() or 'ground_truth' in trace_name.lower():
            gurobi_traces.append(trace_name)
        elif any(x in trace_name.lower() for x in ['quantum', 'qpu', 'clique', 'spatial', 'hierarchical']):
            quantum_traces.append(trace_name)
    
    # Group for coloring
    config_groups = {
        'A: Statistical (5-25f, 6fam, 300s)': config_a_traces,
        'B: Hierarchical (25-100f, 27→6, 300s)': config_b_traces,
        'C: QPU Bench (various, 200s)': config_c_traces,
        'D: Significant (5-100f, 100s)': config_d_traces,
    }
    
    # Define color schemes for each configuration
    colors_a = plt.cm.Blues(np.linspace(0.3, 0.9, max(len(config_a_traces), 1)))
    colors_b = plt.cm.Reds(np.linspace(0.3, 0.9, max(len(config_b_traces), 1)))
    colors_c = plt.cm.Greens(np.linspace(0.3, 0.9, max(len(config_c_traces), 1)))
    colors_d = plt.cm.Purples(np.linspace(0.3, 0.9, max(len(config_d_traces), 1)))
    
    # Plot each configuration group
    for idx, trace_name in enumerate(config_a_traces):
        points = traces_cleaned[trace_name]
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        marker = 'o' if trace_name in gurobi_traces else '^'
        linewidth = 2.5 if trace_name in gurobi_traces else 1.5
        ax.plot(x_vals, y_vals, marker=marker, linewidth=linewidth, alpha=0.8,
                color=colors_a[idx], markersize=7, label=trace_name)
    
    for idx, trace_name in enumerate(config_b_traces):
        points = traces_cleaned[trace_name]
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        marker = 's' if trace_name in gurobi_traces else 'D'
        linewidth = 2.5 if trace_name in gurobi_traces else 1.5
        ax.plot(x_vals, y_vals, marker=marker, linewidth=linewidth, alpha=0.8,
                color=colors_b[idx], markersize=6, label=trace_name)
    
    for idx, trace_name in enumerate(config_c_traces):
        points = traces_cleaned[trace_name]
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        marker = 'v' if trace_name in gurobi_traces else '<'
        linewidth = 2 if trace_name in gurobi_traces else 1.3
        ax.plot(x_vals, y_vals, marker=marker, linewidth=linewidth, alpha=0.7,
                color=colors_c[idx], markersize=5, label=trace_name)
    
    for idx, trace_name in enumerate(config_d_traces):
        points = traces_cleaned[trace_name]
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        marker = 'p' if trace_name in gurobi_traces else 'h'
        linewidth = 2 if trace_name in gurobi_traces else 1.3
        ax.plot(x_vals, y_vals, marker=marker, linewidth=linewidth, alpha=0.7,
                color=colors_d[idx], markersize=6, label=trace_name)
    
    # Add timeout reference lines
    ax.axhline(y=300, color='blue', linestyle='--', linewidth=2, alpha=0.4, 
               label='Config A,B timeout (300s)')
    ax.axhline(y=200, color='green', linestyle='--', linewidth=2, alpha=0.4,
               label='Config C timeout (200s)')
    ax.axhline(y=100, color='purple', linestyle='--', linewidth=2, alpha=0.4,
               label='Config D timeout (100s)')
    
    ax.set_xlabel('Number of Variables', fontsize=14, fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontsize=14, fontweight='bold')
    
    title_text = 'Runtime Comparison: All Solvers from All Files (Configurations Highlighted)\n'
    title_text += f'Blue=Config A (5-25f, 6fam, 300s) | Red=Config B (25-100f, 27→6, 300s) | '
    title_text += f'Green=Config C (various, 200s) | Purple=Config D (5-100f, 100s)\n'
    title_text += f'({len(traces_cleaned)} traces, {sum(len(p) for p in traces_cleaned.values())} points, {outliers_removed} outliers removed)'
    
    ax.set_title(title_text, fontsize=13, fontweight='bold')
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.grid(True, alpha=0.15, which='minor', linestyle=':')
    
    # Set reasonable axis limits
    all_x = [p[0] for points in traces_cleaned.values() for p in points]
    all_y = [p[1] for points in traces_cleaned.values() for p in points]
    
    ax.set_xlim(min(all_x) * 0.8, max(all_x) * 1.2)
    ax.set_ylim(min(all_y) * 0.5, max(all_y) * 2)
    
    # Add configuration boxes as text annotations
    ax.text(0.02, 0.98, 'Configuration Legend:\n' +
            '━ Config A: Statistical (5-25 farms, 6 families, 300s)\n' +
            '━ Config B: Hierarchical (25-100 farms, 27→6, 300s)\n' +
            '━ Config C: QPU Benchmark (various sizes, 200s)\n' +
            '━ Config D: Significant Scenarios (5-100 farms, 100s)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Legend - only show first few entries to avoid clutter
    handles, labels = ax.get_legend_handles_labels()
    # Keep timeout lines + first 20 traces
    timeout_handles = handles[-3:]
    timeout_labels = labels[-3:]
    trace_handles = handles[:-3][:20]
    trace_labels = labels[:-3][:20]
    
    legend = ax.legend(timeout_handles + trace_handles, 
                      timeout_labels + trace_labels,
                      loc='lower right', fontsize=6, ncol=2,
                      framealpha=0.9, title='Key Traces & Timeouts', title_fontsize=7)
    legend.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    
    # Save
    output_file = PLOTS_DIR / 'all_runtime_traces_lineplot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.savefig(PLOTS_DIR / 'all_runtime_traces_lineplot.pdf', bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / 'all_runtime_traces_lineplot.pdf'}")
    
    plt.close()
    
    print()
    print("="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
