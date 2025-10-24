"""
LQ Benchmark Visualization: D-Wave vs Classical Solvers
Creates beautiful plots showing solve times and speedup comparisons
in linear, log-y, and log-log scales.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt conflicts
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
    """Load all LQ benchmark data from the Benchmarks directory."""
    lq_dir = Path(benchmark_dir) / "LQ"
    
    data = {
        'DWave': {},
        'PuLP': {},
        'CQM': {},
        'Pyomo': {}
    }
    
    # Configuration files to load (using run_1 for consistency)
    configs = [5, 19, 72, 279]
    
    for solver in data.keys():
        solver_dir = lq_dir / solver
        if not solver_dir.exists():
            continue
        for config in configs:
            config_file = solver_dir / f"config_{config}_run_1.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    benchmark = json.load(f)
                    data[solver][config] = benchmark
    
    return data

def extract_times(data):
    """Extract solve times for each solver and configuration."""
    configs = sorted([k for k in data['DWave'].keys()])
    
    times = {
        'n_farms': configs,
        'PuLP': [],
        'Pyomo': [],
        'CQM': [],
        'DWave_QPU': [],
        'DWave_Hybrid': []
    }
    
    for config in configs:
        # PuLP times
        if config in data['PuLP']:
            times['PuLP'].append(data['PuLP'][config]['result']['solve_time'])
        else:
            times['PuLP'].append(None)
        
        # Pyomo times
        if config in data['Pyomo']:
            times['Pyomo'].append(data['Pyomo'][config]['result']['solve_time'])
        else:
            times['Pyomo'].append(None)
        
        # CQM times
        if config in data['CQM']:
            times['CQM'].append(data['CQM'][config]['result']['cqm_time'])
        else:
            times['CQM'].append(None)
        
        # D-Wave times
        if config in data['DWave']:
            times['DWave_QPU'].append(data['DWave'][config]['result']['qpu_time'])
            times['DWave_Hybrid'].append(data['DWave'][config]['result']['dwave_time'])
        else:
            times['DWave_QPU'].append(None)
            times['DWave_Hybrid'].append(None)
    
    return times

def calculate_speedups(times):
    """Calculate speedup factors comparing D-Wave to classical solvers."""
    speedups = {
        'n_farms': times['n_farms'],
        'QPU_vs_PuLP': [],
        'QPU_vs_Pyomo': [],
        'QPU_vs_CQM': [],
        'Hybrid_vs_PuLP': [],
        'Hybrid_vs_Pyomo': [],
        'Hybrid_vs_CQM': []
    }
    
    for i in range(len(times['n_farms'])):
        # QPU vs PuLP
        if times['PuLP'][i] and times['DWave_QPU'][i]:
            speedups['QPU_vs_PuLP'].append(times['PuLP'][i] / times['DWave_QPU'][i])
        else:
            speedups['QPU_vs_PuLP'].append(None)
        
        # QPU vs Pyomo
        if times['Pyomo'][i] and times['DWave_QPU'][i]:
            speedups['QPU_vs_Pyomo'].append(times['Pyomo'][i] / times['DWave_QPU'][i])
        else:
            speedups['QPU_vs_Pyomo'].append(None)
        
        # QPU vs CQM
        if times['CQM'][i] and times['DWave_QPU'][i]:
            speedups['QPU_vs_CQM'].append(times['CQM'][i] / times['DWave_QPU'][i])
        else:
            speedups['QPU_vs_CQM'].append(None)
        
        # Hybrid vs PuLP
        if times['PuLP'][i] and times['DWave_Hybrid'][i]:
            speedups['Hybrid_vs_PuLP'].append(times['PuLP'][i] / times['DWave_Hybrid'][i])
        else:
            speedups['Hybrid_vs_PuLP'].append(None)
        
        # Hybrid vs Pyomo
        if times['Pyomo'][i] and times['DWave_Hybrid'][i]:
            speedups['Hybrid_vs_Pyomo'].append(times['Pyomo'][i] / times['DWave_Hybrid'][i])
        else:
            speedups['Hybrid_vs_Pyomo'].append(None)
        
        # Hybrid vs CQM
        if times['CQM'][i] and times['DWave_Hybrid'][i]:
            speedups['Hybrid_vs_CQM'].append(times['CQM'][i] / times['DWave_Hybrid'][i])
        else:
            speedups['Hybrid_vs_CQM'].append(None)
    
    return speedups

def plot_solve_times(times, output_path):
    """Create three plots: linear, log-y, and log-log scales."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('LQ Benchmark: Solve Time Comparison (D-Wave vs Classical Solvers)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    n_farms = times['n_farms']
    
    # Colors for each solver
    colors = {
        'PuLP': '#E63946',      # Red
        'Pyomo': '#F77F00',     # Orange
        'CQM': '#FFB703',       # Yellow-Orange
        'DWave_QPU': '#06FFA5',  # Cyan
        'DWave_Hybrid': '#118AB2' # Blue
    }
    
    markers = {
        'PuLP': 'o',
        'Pyomo': 's',
        'CQM': 'D',
        'DWave_QPU': '^',
        'DWave_Hybrid': 'v'
    }
    
    # Row 1: Solve Times
    # Linear scale
    ax = axes[0, 0]
    ax.plot(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='PuLP (Classical)', alpha=0.8)
    ax.plot(n_farms, times['Pyomo'], marker=markers['Pyomo'], linewidth=2.5, 
            markersize=10, color=colors['Pyomo'], label='Pyomo (Classical)', alpha=0.8)
    if any(times['CQM']):
        ax.plot(n_farms, times['CQM'], marker=markers['CQM'], linewidth=2.5, 
                markersize=10, color=colors['CQM'], label='CQM (D-Wave Classical)', alpha=0.8)
    ax.plot(n_farms, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
            markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU Time', alpha=0.8)
    ax.plot(n_farms, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid Time', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale
    ax = axes[0, 1]
    ax.semilogy(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
                markersize=10, color=colors['PuLP'], label='PuLP (Classical)', alpha=0.8)
    ax.semilogy(n_farms, times['Pyomo'], marker=markers['Pyomo'], linewidth=2.5, 
                markersize=10, color=colors['Pyomo'], label='Pyomo (Classical)', alpha=0.8)
    if any(times['CQM']):
        ax.semilogy(n_farms, times['CQM'], marker=markers['CQM'], linewidth=2.5, 
                    markersize=10, color=colors['CQM'], label='CQM (D-Wave Classical)', alpha=0.8)
    ax.semilogy(n_farms, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
                markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU Time', alpha=0.8)
    ax.semilogy(n_farms, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
                markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid Time', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Y Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale
    ax = axes[0, 2]
    ax.loglog(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
              markersize=10, color=colors['PuLP'], label='PuLP (Classical)', alpha=0.8)
    ax.loglog(n_farms, times['Pyomo'], marker=markers['Pyomo'], linewidth=2.5, 
              markersize=10, color=colors['Pyomo'], label='Pyomo (Classical)', alpha=0.8)
    if any(times['CQM']):
        ax.loglog(n_farms, times['CQM'], marker=markers['CQM'], linewidth=2.5, 
                  markersize=10, color=colors['CQM'], label='CQM (D-Wave Classical)', alpha=0.8)
    ax.loglog(n_farms, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
              markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU Time', alpha=0.8)
    ax.loglog(n_farms, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
              markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid Time', alpha=0.8)
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
    ax.plot(n_farms, speedups['QPU_vs_Pyomo'], marker='^', linewidth=2.5, 
            markersize=10, color='#06D89E', label='QPU vs Pyomo', alpha=0.8, linestyle='--')
    ax.plot(n_farms, speedups['Hybrid_vs_PuLP'], marker='v', linewidth=2.5, 
            markersize=10, color='#118AB2', label='Hybrid vs PuLP', alpha=0.8)
    ax.plot(n_farms, speedups['Hybrid_vs_Pyomo'], marker='v', linewidth=2.5, 
            markersize=10, color='#073B4C', label='Hybrid vs Pyomo', alpha=0.8, linestyle='--')
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
    ax.semilogy(n_farms, speedups['QPU_vs_Pyomo'], marker='^', linewidth=2.5, 
                markersize=10, color='#06D89E', label='QPU vs Pyomo', alpha=0.8, linestyle='--')
    ax.semilogy(n_farms, speedups['Hybrid_vs_PuLP'], marker='v', linewidth=2.5, 
                markersize=10, color='#118AB2', label='Hybrid vs PuLP', alpha=0.8)
    ax.semilogy(n_farms, speedups['Hybrid_vs_Pyomo'], marker='v', linewidth=2.5, 
                markersize=10, color='#073B4C', label='Hybrid vs Pyomo', alpha=0.8, linestyle='--')
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
    ax.loglog(n_farms, speedups['QPU_vs_Pyomo'], marker='^', linewidth=2.5, 
              markersize=10, color='#06D89E', label='QPU vs Pyomo', alpha=0.8, linestyle='--')
    ax.loglog(n_farms, speedups['Hybrid_vs_PuLP'], marker='v', linewidth=2.5, 
              markersize=10, color='#118AB2', label='Hybrid vs PuLP', alpha=0.8)
    ax.loglog(n_farms, speedups['Hybrid_vs_Pyomo'], marker='v', linewidth=2.5, 
              markersize=10, color='#073B4C', label='Hybrid vs Pyomo', alpha=0.8, linestyle='--')
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
    print("\n" + "="*120)
    print("LQ BENCHMARK SUMMARY: SOLVE TIMES AND SPEEDUPS")
    print("="*120)
    print(f"{'N_Farms':<10} {'PuLP (s)':<12} {'Pyomo (s)':<12} {'CQM (s)':<12} {'QPU (s)':<12} {'Hybrid (s)':<12} {'QPU vs PuLP':<15} {'Hybrid vs PuLP':<15}")
    print("-"*120)
    
    speedups = calculate_speedups(times)
    
    for i, n in enumerate(times['n_farms']):
        pulp_time = f"{times['PuLP'][i]:.4f}" if times['PuLP'][i] else "N/A"
        pyomo_time = f"{times['Pyomo'][i]:.4f}" if times['Pyomo'][i] else "N/A"
        cqm_time = f"{times['CQM'][i]:.4f}" if times['CQM'][i] else "N/A"
        qpu_time = f"{times['DWave_QPU'][i]:.4f}" if times['DWave_QPU'][i] else "N/A"
        hybrid_time = f"{times['DWave_Hybrid'][i]:.4f}" if times['DWave_Hybrid'][i] else "N/A"
        
        qpu_speedup = f"{speedups['QPU_vs_PuLP'][i]:.2f}x" if speedups['QPU_vs_PuLP'][i] else "N/A"
        hybrid_speedup = f"{speedups['Hybrid_vs_PuLP'][i]:.2f}x" if speedups['Hybrid_vs_PuLP'][i] else "N/A"
        
        print(f"{n:<10} {pulp_time:<12} {pyomo_time:<12} {cqm_time:<12} {qpu_time:<12} {hybrid_time:<12} {qpu_speedup:<15} {hybrid_speedup:<15}")
    
    print("="*120)
    print("\nKey Observations:")
    print("- LQ formulation includes LINEAR objective + QUADRATIC synergy terms")
    print("- QPU time remains nearly constant regardless of problem size")
    print("- Classical solvers show varying solve times with problem size")
    print("- D-Wave Hybrid Time includes QPU time plus overhead (embedding, compilation, etc.)")
    print("- Speedup increases significantly with problem size for QPU approach")
    print("\n⚠️  IMPORTANT - SOLUTION QUALITY WARNING:")
    print("- D-Wave shows SIGNIFICANT QUALITY GAPS (up to 32% worse at 279 farms)")
    print("- This means D-Wave is FAST but finds SUBOPTIMAL solutions")
    print("- For quality-critical applications, use classical solvers (PuLP/Pyomo)")
    print("- Run 'python plot_lq_quality_speedup.py' for detailed quality analysis")
    print("="*120 + "\n")

def main():
    # Load data
    benchmark_dir = Path(__file__).parent / "Benchmarks"
    data = load_benchmark_data(benchmark_dir)
    
    # Extract times
    times = extract_times(data)
    
    # Print summary
    print_summary_table(times)
    
    # Create plots
    output_path = Path(__file__).parent / "Plots" / "lq_speedup_comparison.png"
    output_path.parent.mkdir(exist_ok=True)
    plot_solve_times(times, output_path)
    
    # Also create individual high-resolution plot for linear scale
    fig_linear = plt.figure(figsize=(12, 8))
    ax = fig_linear.add_subplot(111)
    
    n_farms = times['n_farms']
    colors = {
        'PuLP': '#E63946',
        'Pyomo': '#F77F00',
        'CQM': '#FFB703',
        'DWave_QPU': '#06FFA5',
        'DWave_Hybrid': '#118AB2'
    }
    
    ax.plot(n_farms, times['PuLP'], 'o-', linewidth=3, markersize=12, 
            color=colors['PuLP'], label='PuLP (Classical)', alpha=0.8)
    ax.plot(n_farms, times['Pyomo'], 's-', linewidth=3, markersize=12, 
            color=colors['Pyomo'], label='Pyomo (Classical)', alpha=0.8)
    if any(times['CQM']):
        ax.plot(n_farms, times['CQM'], 'D-', linewidth=3, markersize=12, 
                color=colors['CQM'], label='CQM (D-Wave Classical)', alpha=0.8)
    ax.plot(n_farms, times['DWave_QPU'], '^-', linewidth=3, markersize=12, 
            color=colors['DWave_QPU'], label='D-Wave QPU Time', alpha=0.8)
    ax.plot(n_farms, times['DWave_Hybrid'], 'v-', linewidth=3, markersize=12, 
            color=colors['DWave_Hybrid'], label='D-Wave Hybrid Time', alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=14, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('LQ Benchmark: Solve Time Comparison (Linear Scale)', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_linear = Path(__file__).parent / "Plots" / "lq_solve_times_linear.png"
    plt.savefig(output_linear, dpi=300, bbox_inches='tight')
    print(f"Linear scale plot saved to: {output_linear}")
    
    # Close figures to avoid display issues
    plt.close('all')
    print("\n✓ All plots saved successfully!")

if __name__ == "__main__":
    main()
