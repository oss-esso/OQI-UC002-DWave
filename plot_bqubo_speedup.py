"""
BQUBO Benchmark Visualization: D-Wave vs Classical Solvers
Creates beautiful plots showing solve times and speedup comparisons
in linear, log-y, and log-log scales.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for beautiful plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.family'] = 'sans-serif'

def load_benchmark_data(benchmark_dir):
    """Load all BQUBO benchmark data from the Benchmarks directory."""
    bqubo_dir = Path(benchmark_dir) / "BQUBO"
    
    data = {
        'DWave': {},
        'PuLP': {},
        'GurobiQUBO': {},
        'CQM': {}
    }
    
    # Configuration files to load (using run_1 for consistency)
    configs = [5, 19, 72, 279, 1096]
    
    for solver in data.keys():
        solver_dir = bqubo_dir / solver
        for config in configs:
            config_file = solver_dir / f"config_{config}_run_1.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    benchmark = json.load(f)
                    data[solver][config] = benchmark
    
    return data

def extract_times(data):
    """Extract solve times for each solver and configuration."""
    configs = sorted([k for k in data['DWave'].keys()] + [k for k in data['PuLP'].keys()] + [k for k in data['GurobiQUBO'].keys()])
    configs = sorted(list(set(configs)))  # Remove duplicates
    
    times = {
        'n_farms': configs,
        'PuLP': [],
        'GurobiQUBO': [],
        'CQM': [],
        'DWave_QPU': [],
        'DWave_Total': []
    }
    
    for config in configs:
        # PuLP times
        if config in data['PuLP']:
            pulp_time = data['PuLP'][config].get('result', {}).get('solve_time', None)
        else:
            pulp_time = None
        times['PuLP'].append(pulp_time)
        
        # GurobiQUBO times
        if config in data['GurobiQUBO']:
            gurobi_time = data['GurobiQUBO'][config].get('result', {}).get('solve_time', None)
        else:
            gurobi_time = None
        times['GurobiQUBO'].append(gurobi_time)
        
        # CQM times
        if config in data['CQM']:
            cqm_time = data['CQM'][config].get('result', {}).get('cqm_time', None)
        else:
            cqm_time = None
        times['CQM'].append(cqm_time)
        
        # DWave times
        if config in data['DWave']:
            dwave = data['DWave'][config].get('result', {})
            qpu_time = dwave.get('qpu_time', None)
            hybrid_time = dwave.get('hybrid_time', None)
        else:
            qpu_time = None
            hybrid_time = None
        
        times['DWave_QPU'].append(qpu_time)
        times['DWave_Total'].append(hybrid_time)
    
    return times

def calculate_speedups(times):
    """Calculate speedup factors comparing D-Wave to classical solvers."""
    speedups = {
        'n_farms': times['n_farms'],
        'QPU_vs_PuLP': [],
        'QPU_vs_GurobiQUBO': [],
        'QPU_vs_CQM': [],
        'Total_vs_PuLP': [],
        'Total_vs_GurobiQUBO': [],
        'Total_vs_CQM': []
    }
    
    for i in range(len(times['n_farms'])):
        # QPU speedups
        if times['PuLP'][i] and times['DWave_QPU'][i]:
            speedups['QPU_vs_PuLP'].append(times['PuLP'][i] / times['DWave_QPU'][i])
        else:
            speedups['QPU_vs_PuLP'].append(None)
        
        if times['GurobiQUBO'][i] and times['DWave_QPU'][i]:
            speedups['QPU_vs_GurobiQUBO'].append(times['GurobiQUBO'][i] / times['DWave_QPU'][i])
        else:
            speedups['QPU_vs_GurobiQUBO'].append(None)
        
        if times['CQM'][i] and times['DWave_QPU'][i]:
            speedups['QPU_vs_CQM'].append(times['CQM'][i] / times['DWave_QPU'][i])
        else:
            speedups['QPU_vs_CQM'].append(None)
        
        # Total speedups
        if times['PuLP'][i] and times['DWave_Total'][i]:
            speedups['Total_vs_PuLP'].append(times['PuLP'][i] / times['DWave_Total'][i])
        else:
            speedups['Total_vs_PuLP'].append(None)
        
        if times['GurobiQUBO'][i] and times['DWave_Total'][i]:
            speedups['Total_vs_GurobiQUBO'].append(times['GurobiQUBO'][i] / times['DWave_Total'][i])
        else:
            speedups['Total_vs_GurobiQUBO'].append(None)
        
        if times['CQM'][i] and times['DWave_Total'][i]:
            speedups['Total_vs_CQM'].append(times['CQM'][i] / times['DWave_Total'][i])
        else:
            speedups['Total_vs_CQM'].append(None)
    
    return speedups

def plot_solve_times(times, output_path):
    """Create three plots: linear, log-y, and log-log scales."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('BQUBO Benchmark: Solve Time Comparison (D-Wave vs Classical Solvers)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    n_farms = times['n_farms']
    
    # Colors for each solver
    colors = {
        'PuLP': '#E63946',      # Red
        'GurobiQUBO': '#9D4EDD', # Purple
        'CQM': '#F77F00',       # Orange
        'DWave_QPU': '#06FFA5',  # Green
        'DWave_Total': '#118AB2' # Blue
    }
    
    markers = {
        'PuLP': 'o',
        'GurobiQUBO': 'x',
        'CQM': 's',
        'DWave_QPU': '^',
        'DWave_Total': 'D'
    }
    
    # Row 1: Solve Times
    # Linear scale
    ax = axes[0, 0]
    ax.plot(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='PuLP (Classical)', alpha=0.8)
    ax.plot(n_farms, times['GurobiQUBO'], marker=markers['GurobiQUBO'], linewidth=2.5, 
            markersize=10, color=colors['GurobiQUBO'], label='Gurobi QUBO (BQM)', alpha=0.8)
    ax.plot(n_farms, times['CQM'], marker=markers['CQM'], linewidth=2.5, 
            markersize=10, color=colors['CQM'], label='CQM (D-Wave Classical)', alpha=0.8)
    ax.plot(n_farms, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
            markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU Time', alpha=0.8)
    ax.plot(n_farms, times['DWave_Total'], marker=markers['DWave_Total'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Total'], label='D-Wave Total Time', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale
    ax = axes[0, 1]
    ax.semilogy(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
                markersize=10, color=colors['PuLP'], label='PuLP (Classical)', alpha=0.8)
    ax.semilogy(n_farms, times['GurobiQUBO'], marker=markers['GurobiQUBO'], linewidth=2.5, 
                markersize=10, color=colors['GurobiQUBO'], label='Gurobi QUBO (BQM)', alpha=0.8)
    ax.semilogy(n_farms, times['CQM'], marker=markers['CQM'], linewidth=2.5, 
                markersize=10, color=colors['CQM'], label='CQM (D-Wave Classical)', alpha=0.8)
    ax.semilogy(n_farms, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
                markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU Time', alpha=0.8)
    ax.semilogy(n_farms, times['DWave_Total'], marker=markers['DWave_Total'], linewidth=2.5, 
                markersize=10, color=colors['DWave_Total'], label='D-Wave Total Time', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Y Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale
    ax = axes[0, 2]
    ax.loglog(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
              markersize=10, color=colors['PuLP'], label='PuLP (Classical)', alpha=0.8)
    ax.loglog(n_farms, times['GurobiQUBO'], marker=markers['GurobiQUBO'], linewidth=2.5, 
              markersize=10, color=colors['GurobiQUBO'], label='Gurobi QUBO (BQM)', alpha=0.8)
    ax.loglog(n_farms, times['CQM'], marker=markers['CQM'], linewidth=2.5, 
              markersize=10, color=colors['CQM'], label='CQM (D-Wave Classical)', alpha=0.8)
    ax.loglog(n_farms, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
              markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU Time', alpha=0.8)
    ax.loglog(n_farms, times['DWave_Total'], marker=markers['DWave_Total'], linewidth=2.5, 
              markersize=10, color=colors['DWave_Total'], label='D-Wave Total Time', alpha=0.8)
    ax.set_xlabel('Number of Farms (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Row 2: Speedup Factors
    speedups = calculate_speedups(times)
    
    # Linear scale speedup
    ax = axes[1, 0]
    ax.plot(n_farms, speedups['QPU_vs_PuLP'], marker='^', linewidth=2.5, 
            markersize=10, color='#06FFA5', label='QPU vs PuLP', alpha=0.8)
    ax.plot(n_farms, speedups['QPU_vs_GurobiQUBO'], marker='x', linewidth=2.5, 
            markersize=10, color='#9D4EDD', label='QPU vs Gurobi QUBO', alpha=0.8)
    ax.plot(n_farms, speedups['QPU_vs_CQM'], marker='^', linewidth=2.5, 
            markersize=10, color='#06D89E', label='QPU vs CQM', alpha=0.8, linestyle='--')
    ax.plot(n_farms, speedups['Total_vs_PuLP'], marker='D', linewidth=2.5, 
            markersize=10, color='#118AB2', label='Total vs PuLP', alpha=0.8)
    ax.plot(n_farms, speedups['Total_vs_GurobiQUBO'], marker='X', linewidth=2.5, 
            markersize=10, color='#7209B7', label='Total vs Gurobi QUBO', alpha=0.8)
    ax.plot(n_farms, speedups['Total_vs_CQM'], marker='D', linewidth=2.5, 
            markersize=10, color='#073B4C', label='Total vs CQM', alpha=0.8, linestyle='--')
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('Speedup: Linear Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale speedup
    ax = axes[1, 1]
    ax.semilogy(n_farms, speedups['QPU_vs_PuLP'], marker='^', linewidth=2.5, 
                markersize=10, color='#06FFA5', label='QPU vs PuLP', alpha=0.8)
    ax.semilogy(n_farms, speedups['QPU_vs_GurobiQUBO'], marker='x', linewidth=2.5, 
                markersize=10, color='#9D4EDD', label='QPU vs Gurobi QUBO', alpha=0.8)
    ax.semilogy(n_farms, speedups['QPU_vs_CQM'], marker='^', linewidth=2.5, 
                markersize=10, color='#06D89E', label='QPU vs CQM', alpha=0.8, linestyle='--')
    ax.semilogy(n_farms, speedups['Total_vs_PuLP'], marker='D', linewidth=2.5, 
                markersize=10, color='#118AB2', label='Total vs PuLP', alpha=0.8)
    ax.semilogy(n_farms, speedups['Total_vs_GurobiQUBO'], marker='X', linewidth=2.5, 
                markersize=10, color='#7209B7', label='Total vs Gurobi QUBO', alpha=0.8)
    ax.semilogy(n_farms, speedups['Total_vs_CQM'], marker='D', linewidth=2.5, 
                markersize=10, color='#073B4C', label='Total vs CQM', alpha=0.8, linestyle='--')
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup: Log-Y Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale speedup
    ax = axes[1, 2]
    ax.loglog(n_farms, speedups['QPU_vs_PuLP'], marker='^', linewidth=2.5, 
              markersize=10, color='#06FFA5', label='QPU vs PuLP', alpha=0.8)
    ax.loglog(n_farms, speedups['QPU_vs_GurobiQUBO'], marker='x', linewidth=2.5, 
              markersize=10, color='#9D4EDD', label='QPU vs Gurobi QUBO', alpha=0.8)
    ax.loglog(n_farms, speedups['QPU_vs_CQM'], marker='^', linewidth=2.5, 
              markersize=10, color='#06D89E', label='QPU vs CQM', alpha=0.8, linestyle='--')
    ax.loglog(n_farms, speedups['Total_vs_PuLP'], marker='D', linewidth=2.5, 
              markersize=10, color='#118AB2', label='Total vs PuLP', alpha=0.8)
    ax.loglog(n_farms, speedups['Total_vs_GurobiQUBO'], marker='X', linewidth=2.5, 
              markersize=10, color='#7209B7', label='Total vs Gurobi QUBO', alpha=0.8)
    ax.loglog(n_farms, speedups['Total_vs_CQM'], marker='D', linewidth=2.5, 
              markersize=10, color='#073B4C', label='Total vs CQM', alpha=0.8, linestyle='--')
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Farms (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup: Log-Log Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    return fig

def print_summary_table(times):
    """Print a summary table of solve times and speedups."""
    print("\n" + "="*130)
    print("BQUBO BENCHMARK SUMMARY: SOLVE TIMES AND SPEEDUPS")
    print("="*130)
    print(f"{'N_Farms':<10} {'PuLP (s)':<12} {'Gurobi (s)':<12} {'CQM (s)':<12} {'QPU (s)':<12} {'Total (s)':<12} {'QPU/PuLP':<12} {'Total/PuLP':<12}")
    print("-"*130)
    
    speedups = calculate_speedups(times)
    
    for i, n in enumerate(times['n_farms']):
        pulp_time = f"{times['PuLP'][i]:.4f}" if times['PuLP'][i] else "N/A"
        gurobi_time = f"{times['GurobiQUBO'][i]:.4f}" if times['GurobiQUBO'][i] else "N/A"
        cqm_time = f"{times['CQM'][i]:.4f}" if times['CQM'][i] else "N/A"
        qpu_time = f"{times['DWave_QPU'][i]:.4f}" if times['DWave_QPU'][i] else "N/A"
        total_time = f"{times['DWave_Total'][i]:.4f}" if times['DWave_Total'][i] else "N/A"
        qpu_speedup = f"{speedups['QPU_vs_PuLP'][i]:.2f}x" if speedups['QPU_vs_PuLP'][i] else "N/A"
        total_speedup = f"{speedups['Total_vs_PuLP'][i]:.2f}x" if speedups['Total_vs_PuLP'][i] else "N/A"
        
        print(f"{n:<10} {pulp_time:<12} {gurobi_time:<12} {cqm_time:<12} {qpu_time:<12} {total_time:<12} {qpu_speedup:<12} {total_speedup:<12}")
    
    print("="*130)
    print("\nKey Observations:")
    print("- QPU time remains nearly constant regardless of problem size")
    print("- Gurobi QUBO solver struggles with larger BQM problems (time limits)")
    print("- Classical solvers (PuLP, CQM) show varying solve times with problem size")
    print("- D-Wave Total Time includes QPU time plus overhead (embedding, compilation, etc.)")
    print("- Speedup increases significantly with problem size for QPU approach")
    print("="*130 + "\n")

def main():
    # Load data
    benchmark_dir = Path(__file__).parent / "Benchmarks"
    data = load_benchmark_data(benchmark_dir)
    
    # Extract times
    times = extract_times(data)
    
    # Print summary
    print_summary_table(times)
    
    # Create plots
    output_path = Path(__file__).parent / "Plots" / "bqubo_speedup_comparison.png"
    output_path.parent.mkdir(exist_ok=True)
    plot_solve_times(times, output_path)
    
    # Also create individual high-resolution plot for linear scale
    fig_linear = plt.figure(figsize=(12, 8))
    ax = fig_linear.add_subplot(111)
    
    n_farms = times['n_farms']
    colors = {
        'PuLP': '#E63946',
        'CQM': '#F77F00',
        'DWave_QPU': '#06FFA5',
        'DWave_Total': '#118AB2'
    }
    
    ax.plot(n_farms, times['PuLP'], 'o-', linewidth=3, markersize=12, 
            color=colors['PuLP'], label='PuLP (Classical)', alpha=0.8)
    ax.plot(n_farms, times['CQM'], 's-', linewidth=3, markersize=12, 
            color=colors['CQM'], label='CQM (D-Wave Classical)', alpha=0.8)
    ax.plot(n_farms, times['DWave_QPU'], '^-', linewidth=3, markersize=12, 
            color=colors['DWave_QPU'], label='D-Wave QPU Time', alpha=0.8)
    ax.plot(n_farms, times['DWave_Total'], 'D-', linewidth=3, markersize=12, 
            color=colors['DWave_Total'], label='D-Wave Total Time', alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=14, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('BQUBO Benchmark: Solve Time Comparison (Linear Scale)', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_linear = Path(__file__).parent / "Plots" / "bqubo_solve_times_linear.png"
    plt.savefig(output_linear, dpi=300, bbox_inches='tight')
    print(f"Linear scale plot saved to: {output_linear}")
    
    plt.show()

if __name__ == "__main__":
    main()
