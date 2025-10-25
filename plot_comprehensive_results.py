#!/usr/bin/env python3
"""
Comprehensive Benchmark Results Visualization

This script creates comprehensive visualizations for the benchmark results comparing
8 different solver configurations across farm and patch scenarios:

Farm Scenario:
1. Farm + PuLP/Gurobi
2. Farm + DWave CQM

Patch Scenario:
3. Patch + PuLP
4. Patch + DWave CQM
5. Patch + DWave BQM
6. Patch + Gurobi QUBO
7. Patch + Simulated Annealing (if available)
8. Patch + Hybrid comparison

Generates multiple plots:
- Solve times vs problem size
- Solution quality vs problem size
- Speedup comparisons (quantum vs classical)
- Success rates and solver reliability
- Performance distribution analysis
"""

import json
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Set up beautiful plotting style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (20, 15)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Color scheme for different solvers
SOLVER_COLORS = {
    # Farm solvers
    'farm_gurobi': '#2E86AB',           # Blue
    'farm_dwave_cqm': '#F18F01',        # Orange
    
    # Patch solvers
    'patch_gurobi': '#06A77D',          # Green
    'patch_dwave_cqm': '#F18F01',       # Orange
    'patch_gurobi_qubo': '#A4243B',     # Dark Red
    'patch_dwave_bqm': '#D00000',       # Red
}

SOLVER_MARKERS = {
    'farm_gurobi': 'o',
    'farm_dwave_cqm': 'D',
    'patch_gurobi': 's',
    'patch_dwave_cqm': 'D',
    'patch_gurobi_qubo': 'v',
    'patch_dwave_bqm': '^',
}

SOLVER_NAMES = {
    'farm_gurobi': 'Farm + Gurobi',
    'farm_dwave_cqm': 'Farm + D-Wave CQM',
    'patch_gurobi': 'Patch + Gurobi',
    'patch_dwave_cqm': 'Patch + D-Wave CQM',
    'patch_gurobi_qubo': 'Patch + Gurobi QUBO',
    'patch_dwave_bqm': 'Patch + D-Wave BQM',
}

def load_benchmark_results(json_file: str) -> Dict:
    """
    Load comprehensive benchmark results from JSON file.
    
    Args:
        json_file: Path to the JSON results file
        
    Returns:
        Parsed benchmark results dictionary
    """
    print(f"📖 Loading benchmark results from {json_file}...")
    
    try:
        with open(json_file, 'r') as f:
            results = json.load(f)
        
        print(f"   ✓ Loaded results from {results['metadata']['timestamp']}")
        print(f"   📊 Farm samples: {len(results['farm_results'])}")
        print(f"   📊 Patch samples: {len(results['patch_results'])}")
        print(f"   ⏱️ Total runtime: {results['metadata']['total_runtime']:.1f}s")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Failed to load results: {e}")
        raise

def extract_solver_data(results: Dict) -> Dict[str, Dict]:
    """
    Extract and organize solver performance data.
    
    Args:
        results: Comprehensive benchmark results
        
    Returns:
        Dictionary organized by solver configuration
    """
    print(f"🔧 Extracting solver performance data...")
    
    solver_data = {}
    
    # Process farm results
    for farm_result in results['farm_results']:
        n_units = farm_result['n_units']
        total_area = farm_result['total_area']
        problem_size = n_units * farm_result['n_foods']
        
        for solver_name, solver_result in farm_result['solvers'].items():
            key = f"farm_{solver_name}"
            
            if key not in solver_data:
                solver_data[key] = {
                    'problem_sizes': [],
                    'n_units': [],
                    'total_areas': [],
                    'solve_times': [],
                    'objective_values': [],
                    'qpu_times': [],
                    'hybrid_times': [],
                    'success_rates': [],
                    'statuses': []
                }
            
            solver_data[key]['problem_sizes'].append(problem_size)
            solver_data[key]['n_units'].append(n_units)
            solver_data[key]['total_areas'].append(total_area)
            solver_data[key]['solve_times'].append(solver_result.get('solve_time', None))
            solver_data[key]['objective_values'].append(solver_result.get('objective_value', None))
            solver_data[key]['qpu_times'].append(solver_result.get('qpu_time', None))
            solver_data[key]['hybrid_times'].append(solver_result.get('hybrid_time', None))
            solver_data[key]['success_rates'].append(1 if solver_result.get('success', False) else 0)
            solver_data[key]['statuses'].append(solver_result.get('status', 'Unknown'))
    
    # Process patch results
    for patch_result in results['patch_results']:
        n_units = patch_result['n_units']
        total_area = patch_result['total_area']
        problem_size = n_units * patch_result['n_foods']
        
        for solver_name, solver_result in patch_result['solvers'].items():
            key = f"patch_{solver_name}"
            
            if key not in solver_data:
                solver_data[key] = {
                    'problem_sizes': [],
                    'n_units': [],
                    'total_areas': [],
                    'solve_times': [],
                    'objective_values': [],
                    'qpu_times': [],
                    'hybrid_times': [],
                    'success_rates': [],
                    'statuses': []
                }
            
            solver_data[key]['problem_sizes'].append(problem_size)
            solver_data[key]['n_units'].append(n_units)
            solver_data[key]['total_areas'].append(total_area)
            solver_data[key]['solve_times'].append(solver_result.get('solve_time', None))
            solver_data[key]['objective_values'].append(solver_result.get('objective_value', None))
            solver_data[key]['qpu_times'].append(solver_result.get('qpu_time', None))
            solver_data[key]['hybrid_times'].append(solver_result.get('hybrid_time', None))
            solver_data[key]['success_rates'].append(1 if solver_result.get('success', False) else 0)
            solver_data[key]['statuses'].append(solver_result.get('status', 'Unknown'))
    
    # Convert lists to numpy arrays and filter out None values
    for solver_key, data in solver_data.items():
        # Only keep valid runs where solve_time exists and solver was successful
        valid_indices = [i for i, (solve_time, success) in enumerate(zip(data['solve_times'], data['success_rates'])) 
                        if solve_time is not None and success == 1]
        
        if valid_indices:
            for metric in data:
                if metric in ['solve_times', 'objective_values', 'qpu_times', 'hybrid_times']:
                    # Filter out None values and keep only valid indices
                    filtered_values = []
                    for i in valid_indices:
                        if i < len(data[metric]) and data[metric][i] is not None:
                            filtered_values.append(data[metric][i])
                    data[metric] = filtered_values
                else:
                    data[metric] = [data[metric][i] for i in valid_indices if i < len(data[metric])]
        else:
            # No valid data, clear all lists
            for metric in data:
                data[metric] = []
    
    print(f"   ✓ Extracted data for {len(solver_data)} solver configurations")
    for solver_key, data in solver_data.items():
        if data['solve_times']:
            print(f"     {solver_key}: {len(data['solve_times'])} successful runs")
    
    return solver_data

def plot_solve_times(solver_data: Dict[str, Dict], output_dir: str):
    """Create solve time comparison plots."""
    print(f"📊 Creating solve time plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Plot 1: All solvers - Linear scale
    ax1.set_title('Solver Performance Comparison - Linear Scale', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Problem Size (Units × Foods)', fontsize=14)
    ax1.set_ylabel('Solve Time (seconds)', fontsize=14)
    
    for solver_key, data in solver_data.items():
        if data['solve_times'] and solver_key in SOLVER_COLORS:
            ax1.scatter(data['problem_sizes'], data['solve_times'], 
                       color=SOLVER_COLORS[solver_key], marker=SOLVER_MARKERS[solver_key],
                       s=100, alpha=0.7, label=SOLVER_NAMES[solver_key])
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: All solvers - Log scale
    ax2.set_title('Solver Performance Comparison - Log Scale', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Problem Size (Units × Foods)', fontsize=14)
    ax2.set_ylabel('Solve Time (seconds)', fontsize=14)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    
    for solver_key, data in solver_data.items():
        if data['solve_times'] and solver_key in SOLVER_COLORS:
            ax2.scatter(data['problem_sizes'], data['solve_times'], 
                       color=SOLVER_COLORS[solver_key], marker=SOLVER_MARKERS[solver_key],
                       s=100, alpha=0.7, label=SOLVER_NAMES[solver_key])
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Farm vs Patch scenarios
    ax3.set_title('Farm vs Patch Scenarios', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel('Total Area (hectares)', fontsize=14)
    ax3.set_ylabel('Solve Time (seconds)', fontsize=14)
    ax3.set_yscale('log')
    
    for solver_key, data in solver_data.items():
        if data['solve_times'] and solver_key in SOLVER_COLORS:
            scenario = 'farm' if solver_key.startswith('farm_') else 'patch'
            marker_style = 'o' if scenario == 'farm' else 's'
            ax3.scatter(data['total_areas'], data['solve_times'], 
                       color=SOLVER_COLORS[solver_key], marker=marker_style,
                       s=100, alpha=0.7, label=SOLVER_NAMES[solver_key])
    
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: QPU vs Hybrid times (for D-Wave solvers)
    ax4.set_title('D-Wave QPU vs Hybrid Times', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xlabel('Problem Size (Units × Foods)', fontsize=14)
    ax4.set_ylabel('Time (seconds)', fontsize=14)
    ax4.set_yscale('log')
    
    for solver_key, data in solver_data.items():
        if 'dwave' in solver_key and data['qpu_times']:
            qpu_times = [t for t in data['qpu_times'] if t is not None and t > 0]
            hybrid_times = [t for t in data['hybrid_times'] if t is not None and t > 0]
            problem_sizes = data['problem_sizes'][:len(qpu_times)]
            
            if qpu_times:
                ax4.scatter(problem_sizes, qpu_times, 
                           color=SOLVER_COLORS[solver_key], marker='D',
                           s=80, alpha=0.8, label=f'{SOLVER_NAMES[solver_key]} (QPU)')
            if hybrid_times:
                ax4.scatter(problem_sizes, hybrid_times, 
                           color=SOLVER_COLORS[solver_key], marker='o',
                           s=60, alpha=0.6, label=f'{SOLVER_NAMES[solver_key]} (Hybrid)')
    
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'comprehensive_solve_times.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Solve time plots saved to {output_path}")

def plot_solution_quality(solver_data: Dict[str, Dict], output_dir: str):
    """Create solution quality comparison plots."""
    print(f"📊 Creating solution quality plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Plot 1: Objective values vs problem size
    ax1.set_title('Solution Quality Comparison', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Problem Size (Units × Foods)', fontsize=14)
    ax1.set_ylabel('Objective Value', fontsize=14)
    
    for solver_key, data in solver_data.items():
        if data['objective_values'] and solver_key in SOLVER_COLORS:
            obj_values = [obj for obj in data['objective_values'] if obj is not None]
            if obj_values:
                ax1.scatter(data['problem_sizes'][:len(obj_values)], obj_values, 
                           color=SOLVER_COLORS[solver_key], marker=SOLVER_MARKERS[solver_key],
                           s=100, alpha=0.7, label=SOLVER_NAMES[solver_key])
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Success rates
    ax2.set_title('Solver Success Rates', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Solver Configuration', fontsize=14)
    ax2.set_ylabel('Success Rate', fontsize=14)
    
    solver_names = []
    success_rates = []
    colors = []
    
    for solver_key, data in solver_data.items():
        if solver_key in SOLVER_NAMES and data['success_rates']:
            solver_names.append(SOLVER_NAMES[solver_key])
            success_rates.append(np.mean(data['success_rates']))
            colors.append(SOLVER_COLORS[solver_key])
    
    bars = ax2.bar(solver_names, success_rates, color=colors, alpha=0.7)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Success Rate')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Solve time distribution
    ax3.set_title('Solve Time Distribution', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel('Solve Time (seconds)', fontsize=14)
    ax3.set_ylabel('Density', fontsize=14)
    ax3.set_xscale('log')
    
    for solver_key, data in solver_data.items():
        if data['solve_times'] and len(data['solve_times']) > 2 and solver_key in SOLVER_COLORS:
            solve_times = [t for t in data['solve_times'] if t is not None and t > 0]
            if solve_times:
                ax3.hist(solve_times, bins=20, alpha=0.6, 
                        color=SOLVER_COLORS[solver_key], label=SOLVER_NAMES[solver_key])
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Farm vs Patch objective comparison
    ax4.set_title('Farm vs Patch Solution Quality', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xlabel('Total Area (hectares)', fontsize=14)
    ax4.set_ylabel('Objective Value', fontsize=14)
    
    for solver_key, data in solver_data.items():
        if data['objective_values'] and solver_key in SOLVER_COLORS:
            obj_values = [obj for obj in data['objective_values'] if obj is not None]
            if obj_values:
                scenario = 'farm' if solver_key.startswith('farm_') else 'patch'
                marker_style = 'o' if scenario == 'farm' else 's'
                ax4.scatter(data['total_areas'][:len(obj_values)], obj_values, 
                           color=SOLVER_COLORS[solver_key], marker=marker_style,
                           s=100, alpha=0.7, label=SOLVER_NAMES[solver_key])
    
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'comprehensive_solution_quality.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Solution quality plots saved to {output_path}")

def plot_speedup_analysis(solver_data: Dict[str, Dict], output_dir: str):
    """Create speedup comparison plots."""
    print(f"📊 Creating speedup analysis plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Find baseline solvers (Gurobi) for speedup calculation
    farm_baseline = solver_data.get('farm_gurobi', {})
    patch_baseline = solver_data.get('patch_gurobi', {})
    
    # Plot 1: Quantum vs Classical speedup (Farm scenario)
    ax1.set_title('Farm Scenario: D-Wave vs Gurobi Speedup', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Problem Size (Units × Foods)', fontsize=14)
    ax1.set_ylabel('Speedup Factor', fontsize=14)
    
    if farm_baseline.get('solve_times') and solver_data.get('farm_dwave_cqm', {}).get('solve_times'):
        baseline_times = farm_baseline['solve_times']
        dwave_times = solver_data['farm_dwave_cqm']['solve_times']
        problem_sizes = farm_baseline['problem_sizes']
        
        speedups = [bt/dt if dt is not None and dt > 0 else 0 for bt, dt in zip(baseline_times, dwave_times)]
        ax1.plot(problem_sizes, speedups, 'o-', color='#F18F01', linewidth=3, markersize=8,
                label='D-Wave CQM vs Gurobi')
        ax1.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No speedup')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Multiple quantum solver speedups (Patch scenario)
    ax2.set_title('Patch Scenario: Quantum Solvers vs Gurobi Speedup', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Problem Size (Units × Foods)', fontsize=14)
    ax2.set_ylabel('Speedup Factor', fontsize=14)
    
    if patch_baseline.get('solve_times'):
        baseline_times = patch_baseline['solve_times']
        problem_sizes = patch_baseline['problem_sizes']
        
        for solver_key in ['patch_dwave_cqm', 'patch_dwave_bqm', 'patch_gurobi_qubo']:
            if solver_data.get(solver_key, {}).get('solve_times'):
                solver_times = solver_data[solver_key]['solve_times']
                min_len = min(len(baseline_times), len(solver_times))
                speedups = [baseline_times[i]/solver_times[i] if solver_times[i] is not None and solver_times[i] > 0 else 0 
                           for i in range(min_len)]
                ax2.plot(problem_sizes[:min_len], speedups, 'o-', 
                        color=SOLVER_COLORS[solver_key], linewidth=3, markersize=8,
                        label=f'{SOLVER_NAMES[solver_key]} vs Gurobi')
        
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No speedup')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: QPU efficiency (QPU time vs total time)
    ax3.set_title('D-Wave QPU Efficiency', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel('Total Solve Time (seconds)', fontsize=14)
    ax3.set_ylabel('QPU Time (seconds)', fontsize=14)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    for solver_key, data in solver_data.items():
        if 'dwave' in solver_key and data.get('qpu_times') and data.get('solve_times'):
            qpu_times = [t for t in data['qpu_times'] if t is not None and t > 0]
            solve_times = data['solve_times'][:len(qpu_times)]
            
            if qpu_times and solve_times:
                ax3.scatter(solve_times, qpu_times, 
                           color=SOLVER_COLORS[solver_key], marker='D',
                           s=100, alpha=0.7, label=SOLVER_NAMES[solver_key])
    
    # Add diagonal line for reference
    if ax3.get_xlim()[0] > 0 and ax3.get_ylim()[0] > 0:
        ax3.plot([ax3.get_xlim()[0], ax3.get_xlim()[1]], 
                [ax3.get_ylim()[0], ax3.get_ylim()[1]], 
                'k--', alpha=0.5, label='QPU = Total')
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Scaling comparison
    ax4.set_title('Solver Scaling Comparison', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xlabel('Problem Size (Units × Foods)', fontsize=14)
    ax4.set_ylabel('Solve Time (seconds)', fontsize=14)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    # Fit scaling trends for each solver
    for solver_key, data in solver_data.items():
        if data['solve_times'] and len(data['solve_times']) > 3 and solver_key in SOLVER_COLORS:
            problem_sizes = np.array(data['problem_sizes'])
            solve_times = np.array(data['solve_times'])
            
            # Fit polynomial in log space
            log_sizes = np.log(problem_sizes)
            log_times = np.log(solve_times)
            
            try:
                coeffs = np.polyfit(log_sizes, log_times, 1)
                scaling_factor = coeffs[0]
                
                # Plot trend line
                x_trend = np.linspace(problem_sizes.min(), problem_sizes.max(), 100)
                y_trend = np.exp(coeffs[1]) * (x_trend ** scaling_factor)
                
                ax4.plot(x_trend, y_trend, '--', color=SOLVER_COLORS[solver_key], alpha=0.7)
                ax4.scatter(problem_sizes, solve_times, 
                           color=SOLVER_COLORS[solver_key], marker=SOLVER_MARKERS[solver_key],
                           s=100, alpha=0.8, label=f'{SOLVER_NAMES[solver_key]} (O(n^{scaling_factor:.1f}))')
            except:
                ax4.scatter(problem_sizes, solve_times, 
                           color=SOLVER_COLORS[solver_key], marker=SOLVER_MARKERS[solver_key],
                           s=100, alpha=0.8, label=SOLVER_NAMES[solver_key])
    
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'comprehensive_speedup_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Speedup analysis plots saved to {output_path}")

def create_summary_report(results: Dict, solver_data: Dict[str, Dict], output_dir: str):
    """Create a text summary report."""
    print(f"📄 Creating summary report...")
    
    report_path = Path(output_dir) / 'comprehensive_benchmark_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE BENCHMARK RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Metadata
        metadata = results['metadata']
        f.write(f"Benchmark Date: {metadata['timestamp']}\n")
        f.write(f"Total Samples: {metadata['n_samples']}\n")
        f.write(f"Total Runtime: {metadata['total_runtime']:.1f} seconds\n")
        f.write(f"D-Wave Enabled: {metadata['dwave_enabled']}\n\n")
        
        # Summary statistics
        f.write("SOLVER PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n\n")
        
        for solver_key, data in solver_data.items():
            if data['solve_times'] and solver_key in SOLVER_NAMES:
                solve_times = [t for t in data['solve_times'] if t > 0]
                success_rate = np.mean(data['success_rates'])
                
                f.write(f"{SOLVER_NAMES[solver_key]}:\n")
                f.write(f"  Successful runs: {len(solve_times)}\n")
                f.write(f"  Success rate: {success_rate:.2%}\n")
                
                if solve_times:
                    f.write(f"  Average solve time: {np.mean(solve_times):.3f}s\n")
                    f.write(f"  Min solve time: {np.min(solve_times):.3f}s\n")
                    f.write(f"  Max solve time: {np.max(solve_times):.3f}s\n")
                    
                    if data['objective_values']:
                        obj_values = [obj for obj in data['objective_values'] if obj is not None]
                        if obj_values:
                            f.write(f"  Average objective: {np.mean(obj_values):.4f}\n")
                
                f.write("\n")
        
        # Speedup analysis
        f.write("SPEEDUP ANALYSIS\n")
        f.write("-" * 20 + "\n\n")
        
        # Farm scenario speedup
        farm_baseline = solver_data.get('farm_gurobi', {})
        farm_dwave = solver_data.get('farm_dwave_cqm', {})
        
        if farm_baseline.get('solve_times') and farm_dwave.get('solve_times'):
            baseline_avg = np.mean(farm_baseline['solve_times'])
            dwave_avg = np.mean(farm_dwave['solve_times'])
            speedup = baseline_avg / dwave_avg if dwave_avg > 0 else 0
            f.write(f"Farm Scenario - D-Wave vs Gurobi:\n")
            f.write(f"  Average speedup: {speedup:.2f}x\n")
            f.write(f"  Gurobi avg: {baseline_avg:.3f}s\n")
            f.write(f"  D-Wave CQM avg: {dwave_avg:.3f}s\n\n")
        
        # Patch scenario speedups
        patch_baseline = solver_data.get('patch_gurobi', {})
        if patch_baseline.get('solve_times'):
            baseline_avg = np.mean(patch_baseline['solve_times'])
            f.write(f"Patch Scenario Speedups vs Gurobi:\n")
            
            for solver_key in ['patch_dwave_cqm', 'patch_dwave_bqm', 'patch_gurobi_qubo']:
                if solver_data.get(solver_key, {}).get('solve_times'):
                    solver_avg = np.mean(solver_data[solver_key]['solve_times'])
                    speedup = baseline_avg / solver_avg if solver_avg > 0 else 0
                    f.write(f"  {SOLVER_NAMES[solver_key]}: {speedup:.2f}x\n")
            f.write("\n")
        
        # QPU utilization
        f.write("D-WAVE QPU UTILIZATION\n")
        f.write("-" * 25 + "\n\n")
        
        for solver_key, data in solver_data.items():
            if 'dwave' in solver_key and data.get('qpu_times'):
                qpu_times = [t for t in data['qpu_times'] if t is not None and t > 0]
                solve_times = data['solve_times'][:len(qpu_times)]
                
                if qpu_times and solve_times:
                    qpu_avg = np.mean(qpu_times)
                    total_avg = np.mean(solve_times)
                    efficiency = qpu_avg / total_avg if total_avg > 0 else 0
                    
                    f.write(f"{SOLVER_NAMES[solver_key]}:\n")
                    f.write(f"  Average QPU time: {qpu_avg:.4f}s\n")
                    f.write(f"  Average total time: {total_avg:.3f}s\n")
                    f.write(f"  QPU efficiency: {efficiency:.1%}\n\n")
    
    print(f"   ✓ Summary report saved to {report_path}")

def main():
    """Main plotting function."""
    parser = argparse.ArgumentParser(
        description='Create comprehensive visualizations for benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_comprehensive_results.py results.json
  python plot_comprehensive_results.py results.json --output plots/
        '''
    )
    
    parser.add_argument('json_file', type=str,
                       help='JSON file containing comprehensive benchmark results')
    parser.add_argument('--output', type=str, default='.',
                       help='Output directory for plots (default: current directory)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.json_file).exists():
        print(f"❌ Error: JSON file not found: {args.json_file}")
        return False
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE BENCHMARK VISUALIZATION")
    print(f"{'='*80}")
    print(f"Input file: {args.json_file}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load and process data
        results = load_benchmark_results(args.json_file)
        solver_data = extract_solver_data(results)
        
        if not solver_data:
            print(f"❌ No solver data found in results")
            return False
        
        # Create plots
        plot_solve_times(solver_data, output_dir)
        plot_solution_quality(solver_data, output_dir)
        plot_speedup_analysis(solver_data, output_dir)
        
        # Create summary report
        create_summary_report(results, solver_data, output_dir)
        
        print(f"\n{'='*80}")
        print(f"✅ VISUALIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Generated files:")
        print(f"  📊 comprehensive_solve_times.png")
        print(f"  📊 comprehensive_solution_quality.png")
        print(f"  📊 comprehensive_speedup_analysis.png")
        print(f"  📄 comprehensive_benchmark_report.txt")
        print(f"\nAll files saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)