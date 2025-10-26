#!/usr/bin/env python3
"""
Comprehensive Benchmark Visualization: Farm and Patch Scenarios
Creates beautiful plots showing solve times and speedup comparisons
in linear, log-y, and log-log scales.
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
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.family'] = 'sans-serif'

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
    
    # Configuration files to load (using run_1 for consistency)
    configs = [10, 15, 20, 25]
    
    for solver_name in data.keys():
        solver_dir = comp_dir / solver_name
        if not solver_dir.exists():
            continue
        for config in configs:
            config_file = solver_dir / f"config_{config}_run_1.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    benchmark = json.load(f)
                    data[solver_name][config] = benchmark
    
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
    
    for config in configs:
        # Farm PuLP
        if config in data['Farm_PuLP']:
            times['Farm_PuLP'].append(data['Farm_PuLP'][config].get('solve_time'))
        else:
            times['Farm_PuLP'].append(None)
        
        # Farm DWave
        if config in data['Farm_DWave']:
            d = data['Farm_DWave'][config]
            times['Farm_DWave_QPU'].append(d.get('qpu_time'))
            times['Farm_DWave_Hybrid'].append(d.get('hybrid_time') or d.get('solve_time'))
        else:
            times['Farm_DWave_QPU'].append(None)
            times['Farm_DWave_Hybrid'].append(None)
        
        # Patch PuLP
        if config in data['Patch_PuLP']:
            times['Patch_PuLP'].append(data['Patch_PuLP'][config].get('solve_time'))
        else:
            times['Patch_PuLP'].append(None)
        
        # Patch DWave
        if config in data['Patch_DWave']:
            d = data['Patch_DWave'][config]
            times['Patch_DWave_QPU'].append(d.get('qpu_time'))
            times['Patch_DWave_Hybrid'].append(d.get('hybrid_time') or d.get('solve_time'))
        else:
            times['Patch_DWave_QPU'].append(None)
            times['Patch_DWave_Hybrid'].append(None)
        
        # Patch Gurobi QUBO
        if config in data['Patch_GurobiQUBO']:
            times['Patch_GurobiQUBO'].append(data['Patch_GurobiQUBO'][config].get('solve_time'))
        else:
            times['Patch_GurobiQUBO'].append(None)
        
        # Patch DWave BQM
        if config in data['Patch_DWaveBQM']:
            d = data['Patch_DWaveBQM'][config]
            times['Patch_DWaveBQM_QPU'].append(d.get('qpu_time'))
            times['Patch_DWaveBQM_Hybrid'].append(d.get('hybrid_time') or d.get('solve_time'))
        else:
            times['Patch_DWaveBQM_QPU'].append(None)
            times['Patch_DWaveBQM_Hybrid'].append(None)
    
    return times

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
    fig.suptitle('Comprehensive Benchmark: Solve Time Comparison', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    n_units = times['n_units']
    
    # Colors for each solver type
    colors = {
        'Farm_PuLP': '#E63946',          # Red
        'Farm_DWave': '#118AB2',         # Blue
        'Patch_PuLP': '#E63946',         # Red
        'Patch_DWave_CQM': '#118AB2',    # Blue
        'Patch_DWave_BQM': '#06FFA5',    # Cyan
        'Patch_QUBO': '#A4243B'          # Dark Red
    }
    
    markers = {
        'Farm_PuLP': 'o',
        'Farm_DWave': 'D',
        'Patch_PuLP': 's',
        'Patch_DWave_CQM': '^',
        'Patch_DWave_BQM': 'v',
        'Patch_QUBO': 'p'
    }
    
    # Row 1: Solve Times
    # Linear scale
    ax = axes[0, 0]
    ax.plot(n_units, times['Farm_PuLP'], marker=markers['Farm_PuLP'], linewidth=2.5, 
            markersize=10, color=colors['Farm_PuLP'], label='Farm PuLP', alpha=0.8)
    ax.plot(n_units, times['Farm_DWave_Hybrid'], marker=markers['Farm_DWave'], linewidth=2.5, 
            markersize=10, color=colors['Farm_DWave'], label='Farm D-Wave CQM', alpha=0.8)
    ax.plot(n_units, times['Patch_PuLP'], marker=markers['Patch_PuLP'], linewidth=2.5, 
            markersize=10, color=colors['Patch_PuLP'], label='Patch PuLP', alpha=0.8)
    ax.plot(n_units, times['Patch_DWave_Hybrid'], marker=markers['Patch_DWave_CQM'], linewidth=2.5, 
            markersize=10, color=colors['Patch_DWave_CQM'], label='Patch D-Wave CQM', alpha=0.8)
    ax.plot(n_units, times['Patch_DWaveBQM_Hybrid'], marker=markers['Patch_DWave_BQM'], linewidth=2.5, 
            markersize=10, color=colors['Patch_DWave_BQM'], label='Patch D-Wave BQM', alpha=0.8)
    ax.plot(n_units, times['Patch_GurobiQUBO'], marker=markers['Patch_QUBO'], linewidth=2.5, 
            markersize=10, color=colors['Patch_QUBO'], label='Patch Gurobi QUBO', alpha=0.8)
    ax.set_xlabel('Number of Units', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale
    ax = axes[0, 1]
    ax.semilogy(n_units, times['Farm_PuLP'], marker=markers['Farm_PuLP'], linewidth=2.5, 
                markersize=10, color=colors['Farm_PuLP'], label='Farm PuLP', alpha=0.8)
    ax.semilogy(n_units, times['Farm_DWave_Hybrid'], marker=markers['Farm_DWave'], linewidth=2.5, 
                markersize=10, color=colors['Farm_DWave'], label='Farm D-Wave CQM', alpha=0.8)
    ax.semilogy(n_units, times['Patch_PuLP'], marker=markers['Patch_PuLP'], linewidth=2.5, 
                markersize=10, color=colors['Patch_PuLP'], label='Patch PuLP', alpha=0.8)
    ax.semilogy(n_units, times['Patch_DWave_Hybrid'], marker=markers['Patch_DWave_CQM'], linewidth=2.5, 
                markersize=10, color=colors['Patch_DWave_CQM'], label='Patch D-Wave CQM', alpha=0.8)
    ax.semilogy(n_units, times['Patch_DWaveBQM_Hybrid'], marker=markers['Patch_DWave_BQM'], linewidth=2.5, 
                markersize=10, color=colors['Patch_DWave_BQM'], label='Patch D-Wave BQM', alpha=0.8)
    ax.semilogy(n_units, times['Patch_GurobiQUBO'], marker=markers['Patch_QUBO'], linewidth=2.5, 
                markersize=10, color=colors['Patch_QUBO'], label='Patch Gurobi QUBO', alpha=0.8)
    ax.set_xlabel('Number of Units', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Y Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale
    ax = axes[0, 2]
    ax.loglog(n_units, times['Farm_PuLP'], marker=markers['Farm_PuLP'], linewidth=2.5, 
              markersize=10, color=colors['Farm_PuLP'], label='Farm PuLP', alpha=0.8)
    ax.loglog(n_units, times['Farm_DWave_Hybrid'], marker=markers['Farm_DWave'], linewidth=2.5, 
              markersize=10, color=colors['Farm_DWave'], label='Farm D-Wave CQM', alpha=0.8)
    ax.loglog(n_units, times['Patch_PuLP'], marker=markers['Patch_PuLP'], linewidth=2.5, 
              markersize=10, color=colors['Patch_PuLP'], label='Patch PuLP', alpha=0.8)
    ax.loglog(n_units, times['Patch_DWave_Hybrid'], marker=markers['Patch_DWave_CQM'], linewidth=2.5, 
              markersize=10, color=colors['Patch_DWave_CQM'], label='Patch D-Wave CQM', alpha=0.8)
    ax.loglog(n_units, times['Patch_DWaveBQM_Hybrid'], marker=markers['Patch_DWave_BQM'], linewidth=2.5, 
              markersize=10, color=colors['Patch_DWave_BQM'], label='Patch D-Wave BQM', alpha=0.8)
    ax.loglog(n_units, times['Patch_GurobiQUBO'], marker=markers['Patch_QUBO'], linewidth=2.5, 
              markersize=10, color=colors['Patch_QUBO'], label='Patch Gurobi QUBO', alpha=0.8)
    ax.set_xlabel('Number of Units (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Row 2: Speedup Factors
    speedups = calculate_speedups(times)
    
    # Linear scale speedup
    ax = axes[1, 0]
    ax.plot(n_units, speedups['Farm_Hybrid_vs_PuLP'], marker='D', linewidth=2.5, 
            markersize=10, color=colors['Farm_DWave'], label='Farm D-Wave vs PuLP', alpha=0.8)
    ax.plot(n_units, speedups['Patch_CQM_Hybrid_vs_PuLP'], marker='^', linewidth=2.5, 
            markersize=10, color=colors['Patch_DWave_CQM'], label='Patch CQM vs PuLP', alpha=0.8)
    ax.plot(n_units, speedups['Patch_BQM_Hybrid_vs_PuLP'], marker='v', linewidth=2.5, 
            markersize=10, color=colors['Patch_DWave_BQM'], label='Patch BQM vs PuLP', alpha=0.8)
    ax.plot(n_units, speedups['Patch_QUBO_vs_PuLP'], marker='p', linewidth=2.5, 
            markersize=10, color=colors['Patch_QUBO'], label='Patch QUBO vs PuLP', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Units', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('Speedup: Linear Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale speedup
    ax = axes[1, 1]
    ax.semilogy(n_units, speedups['Farm_Hybrid_vs_PuLP'], marker='D', linewidth=2.5, 
                markersize=10, color=colors['Farm_DWave'], label='Farm D-Wave vs PuLP', alpha=0.8)
    ax.semilogy(n_units, speedups['Patch_CQM_Hybrid_vs_PuLP'], marker='^', linewidth=2.5, 
                markersize=10, color=colors['Patch_DWave_CQM'], label='Patch CQM vs PuLP', alpha=0.8)
    ax.semilogy(n_units, speedups['Patch_BQM_Hybrid_vs_PuLP'], marker='v', linewidth=2.5, 
                markersize=10, color=colors['Patch_DWave_BQM'], label='Patch BQM vs PuLP', alpha=0.8)
    ax.semilogy(n_units, speedups['Patch_QUBO_vs_PuLP'], marker='p', linewidth=2.5, 
                markersize=10, color=colors['Patch_QUBO'], label='Patch QUBO vs PuLP', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Units', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup: Log-Y Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale speedup
    ax = axes[1, 2]
    ax.loglog(n_units, speedups['Farm_Hybrid_vs_PuLP'], marker='D', linewidth=2.5, 
              markersize=10, color=colors['Farm_DWave'], label='Farm D-Wave vs PuLP', alpha=0.8)
    ax.loglog(n_units, speedups['Patch_CQM_Hybrid_vs_PuLP'], marker='^', linewidth=2.5, 
              markersize=10, color=colors['Patch_DWave_CQM'], label='Patch CQM vs PuLP', alpha=0.8)
    ax.loglog(n_units, speedups['Patch_BQM_Hybrid_vs_PuLP'], marker='v', linewidth=2.5, 
              markersize=10, color=colors['Patch_DWave_BQM'], label='Patch BQM vs PuLP', alpha=0.8)
    ax.loglog(n_units, speedups['Patch_QUBO_vs_PuLP'], marker='p', linewidth=2.5, 
              markersize=10, color=colors['Patch_QUBO'], label='Patch QUBO vs PuLP', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Units (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup: Log-Log Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot saved to {output_path}")

def print_summary_table(times):
    """Print summary table of all solver times."""
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK SUMMARY TABLE")
    print("="*80)
    print(f"{'Config':<10} {'Farm PuLP':<12} {'Farm DWave':<12} {'Patch PuLP':<12} {'Patch DWave':<12} {'Patch BQM':<12} {'Patch QUBO':<12}")
    print("-"*92)
    
    for i, n in enumerate(times['n_units']):
        farm_pulp = f"{times['Farm_PuLP'][i]:.4f}s" if times['Farm_PuLP'][i] else "N/A"
        farm_dwave = f"{times['Farm_DWave_Hybrid'][i]:.4f}s" if times['Farm_DWave_Hybrid'][i] else "N/A"
        patch_pulp = f"{times['Patch_PuLP'][i]:.4f}s" if times['Patch_PuLP'][i] else "N/A"
        patch_dwave = f"{times['Patch_DWave_Hybrid'][i]:.4f}s" if times['Patch_DWave_Hybrid'][i] else "N/A"
        patch_bqm = f"{times['Patch_DWaveBQM_Hybrid'][i]:.4f}s" if times['Patch_DWaveBQM_Hybrid'][i] else "N/A"
        patch_qubo = f"{times['Patch_GurobiQUBO'][i]:.4f}s" if times['Patch_GurobiQUBO'][i] else "N/A"
        
        print(f"{n:<10} {farm_pulp:<12} {farm_dwave:<12} {patch_pulp:<12} {patch_dwave:<12} {patch_bqm:<12} {patch_qubo:<12}")

def main():
    """Main execution function."""
    # Determine benchmark directory
    if len(sys.argv) > 1:
        benchmark_dir = sys.argv[1]
    else:
        benchmark_dir = "Benchmarks"
    
    print("="*80)
    print("COMPREHENSIVE BENCHMARK PLOTTING")
    print("="*80)
    print(f"Loading data from: {benchmark_dir}/COMPREHENSIVE")
    
    # Load data
    data = load_benchmark_data(benchmark_dir)
    
    # Extract times
    print("\nExtracting solve times...")
    times = extract_times(data)
    
    # Create output directory
    output_dir = Path("Plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    print("\nCreating plots...")
    output_path = output_dir / "comprehensive_speedup_comparison.png"
    plot_solve_times(times, output_path)
    
    # Print summary
    print_summary_table(times)
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
