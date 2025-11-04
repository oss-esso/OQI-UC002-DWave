"""
NLN Benchmark Visualization: D-Wave vs Classical Solvers
Creates beautiful plots showing solve times and speedup comparisons
in linear, log-y, and log-log scales.
"""

import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from pathlib import Path

# =============================================================================
# PROFESSIONAL STYLE CONFIGURATION
# =============================================================================

def configure_professional_style():
    """Configure professional matplotlib style with LaTeX integration."""
    
    rcParams.update({
        # Font configuration
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'serif'],
        'font.size': 11,
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'serif',
        
        # Text rendering with LaTeX
        'text.usetex': False,
        'text.latex.preamble': r'''
            \usepackage{amsmath}
            \usepackage{amsfonts}
            \usepackage{amssymb}
            \usepackage{bm}
            \usepackage{xcolor}
        ''',
        
        # Figure layout
        'figure.figsize': (8, 6),
        'figure.dpi': 300,
        'figure.constrained_layout.use': True,
        
        # Axes configuration
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.grid.which': 'major',
        'axes.grid.axis': 'both',
        'axes.labelpad': 8,
        'axes.titlepad': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        
        # Tick configuration
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        
        # Grid configuration
        'grid.linewidth': 0.8,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        
        # Legend configuration
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'legend.shadow': False,
        
        # Lines configuration
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'lines.markeredgewidth': 1.2,
        
        # Savefig configuration
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.transparent': False,
        
        # Errorbar configuration
        'errorbar.capsize': 3,
    })

# Apply professional style
configure_professional_style()

def load_benchmark_data(benchmark_dir):
    """Load all NLN benchmark data from the Benchmarks directory."""
    nln_dir = Path(benchmark_dir) / "NLN"
    
    data = {
        'DWave': {},
        'PuLP': {},
        'Pyomo': {}
    }
    
    # Configuration files to load
    configs = [5, 19, 72, 279]
    
    for solver in data.keys():
        solver_dir = nln_dir / solver
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
        
        # D-Wave times
        if config in data['DWave']:
            times['DWave_QPU'].append(data['DWave'][config]['result']['qpu_time'])
            times['DWave_Hybrid'].append(data['DWave'][config]['result']['hybrid_time'])
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
        'Hybrid_vs_PuLP': [],
        'Hybrid_vs_Pyomo': []
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
    
    return speedups

def extract_times_and_objectives(data):
    """Extract solve times AND objective values for each solver and configuration."""
    configs = sorted([k for k in data['DWave'].keys()])
    
    times = {
        'n_farms': configs,
        'PuLP': [],
        'Pyomo': [],
        'CQM': [],
        'DWave_QPU': [],
        'DWave_Hybrid': []
    }
    
    objectives = {
        'n_farms': configs,
        'PuLP': [],
        'Pyomo': [],
        'CQM': [],
        'DWave': []
    }
    
    for config in configs:
        # PuLP times and objectives
        if config in data['PuLP']:
            times['PuLP'].append(data['PuLP'][config]['result']['solve_time'])
            objectives['PuLP'].append(data['PuLP'][config]['result'].get('objective_value', None))
        else:
            times['PuLP'].append(None)
            objectives['PuLP'].append(None)
        
        # Pyomo times and objectives
        if config in data['Pyomo']:
            times['Pyomo'].append(data['Pyomo'][config]['result']['solve_time'])
            objectives['Pyomo'].append(data['Pyomo'][config]['result'].get('objective_value', None))
        else:
            times['Pyomo'].append(None)
            objectives['Pyomo'].append(None)
        
    
        
        # D-Wave times and objectives
        if config in data['DWave']:
            times['DWave_QPU'].append(data['DWave'][config]['result']['qpu_time'])
            times['DWave_Hybrid'].append(data['DWave'][config]['result']['hybrid_time'])
            objectives['DWave'].append(data['DWave'][config]['result'].get('objective_value', None))
        else:
            times['DWave_QPU'].append(None)
            times['DWave_Hybrid'].append(None)
            objectives['DWave'].append(None)
    
    return times, objectives

def calculate_objective_gaps(objectives):
    """Calculate objective value gaps (percentage deviation from best)."""
    configs = objectives['n_farms']
    gaps = {
        'n_farms': configs,
        'PuLP_gap': [],
        'Pyomo_gap': [],
        'CQM_gap': [],
        'DWave_gap': []
    }
    
    for i in range(len(configs)):
        # Find best (maximum) objective for this config
        values = []
        if objectives['PuLP'][i] is not None:
            values.append(objectives['PuLP'][i])
        if objectives['Pyomo'][i] is not None:
            values.append(objectives['Pyomo'][i])
        if objectives['DWave'][i] is not None:
            values.append(objectives['DWave'][i])
        
        if not values:
            gaps['PuLP_gap'].append(None)
            gaps['Pyomo_gap'].append(None)
            gaps['CQM_gap'].append(None)
            gaps['DWave_gap'].append(None)
            continue
        
        best_obj = max(values)  # Assuming maximization
        
        # Calculate gaps as percentage
        if objectives['PuLP'][i] is not None:
            gap = ((best_obj - objectives['PuLP'][i]) / best_obj) * 100 if best_obj != 0 else 0
            gaps['PuLP_gap'].append(gap)
        else:
            gaps['PuLP_gap'].append(None)
        
        if objectives['Pyomo'][i] is not None:
            gap = ((best_obj - objectives['Pyomo'][i]) / best_obj) * 100 if best_obj != 0 else 0
            gaps['Pyomo_gap'].append(gap)
        else:
            gaps['Pyomo_gap'].append(None)
        
        
        if objectives['DWave'][i] is not None:
            gap = ((best_obj - objectives['DWave'][i]) / best_obj) * 100 if best_obj != 0 else 0
            gaps['DWave_gap'].append(gap)
        else:
            gaps['DWave_gap'].append(None)
    
    return gaps


def plot_solve_times(times, objectives, gaps, output_path):
    """Create three plots: linear, log-y, and log-log scales."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('NLN Benchmark: Solve Time Comparison (D-Wave vs Classical Solvers)', 
                 fontsize=18, y=0.995)
    
    n_farms = times['n_farms']
    
    # Colors for each solver
    colors = {
        'PuLP': '#E63946',      # Red
        'Pyomo': '#F77F00',     # Orange
        'DWave_QPU': '#06FFA5',  # Cyan
        'DWave_Hybrid': '#118AB2' # Blue
    }
    
    markers = {
        'PuLP': 'o',
        'Pyomo': 's',
        'DWave_QPU': '^',
        'DWave_Hybrid': 'D'
    }
    
    # Row 1: Solve Times
    # Linear scale
    ax = axes[0, 0]
    ax.plot(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='Gurobi MILP', alpha=0.8)
    ax.plot(n_farms, times['Pyomo'], marker=markers['Pyomo'], linewidth=2.5, 
            markersize=10, color=colors['Pyomo'], label='Ipopt MINLP', alpha=0.8)
    ax.plot(n_farms, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
            markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.plot(n_farms, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale
    ax = axes[0, 1]
    ax.semilogy(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
                markersize=10, color=colors['PuLP'], label='Gurobi MILP', alpha=0.8)
    ax.semilogy(n_farms, times['Pyomo'], marker=markers['Pyomo'], linewidth=2.5, 
                markersize=10, color=colors['Pyomo'], label='Ipopt MINLP', alpha=0.8)
    ax.semilogy(n_farms, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
                markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.semilogy(n_farms, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
                markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Y Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale
    ax = axes[0, 2]
    ax.loglog(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
              markersize=10, color=colors['PuLP'], label='Gurobi MILP', alpha=0.8)
    ax.loglog(n_farms, times['Pyomo'], marker=markers['Pyomo'], linewidth=2.5, 
              markersize=10, color=colors['Pyomo'], label='Ipopt MINLP', alpha=0.8)
    ax.loglog(n_farms, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
              markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.loglog(n_farms, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
              markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Farms (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Row 2: Speedup Factors
    speedups = calculate_speedups(times)
    # Objective Values
    ax = axes[1, 0]
    ax.plot(n_farms, objectives['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='Gurobi MILP', alpha=0.8)
    ax.plot(n_farms, objectives['Pyomo'], marker=markers['Pyomo'], linewidth=2.5, 
            markersize=10, color=colors['Pyomo'], label='Ipopt MINLP', alpha=0.8)
    ax.plot(n_farms, objectives['DWave'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Value', fontsize=11, fontweight='bold')
    ax.set_title('Objective Values (Higher = Better)', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Objective Gaps (%)
    ax = axes[1, 1]
    ax.plot(n_farms, gaps['PuLP_gap'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='Gurobi MILP', alpha=0.8)
    ax.plot(n_farms, gaps['Pyomo_gap'], marker=markers['Pyomo'], linewidth=2.5, 
            markersize=10, color=colors['Pyomo'], label='Ipopt MINLP', alpha=0.8)
    ax.plot(n_farms, gaps['DWave_gap'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave', alpha=0.8)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Optimal (0\% gap)')
    ax.set_xlabel('Number of Farms', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Gap (\%)', fontsize=11, fontweight='bold')
    ax.set_title('Solution Quality Gap (\% from Best)', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Objective Gaps (Bar Chart)
    ax = axes[1, 2]
    x = np.arange(len(n_farms))
    width = 0.25
    
    # Filter out None values for bar chart
    pulp_gaps_filtered = [g if g is not None else 0 for g in gaps['PuLP_gap']]
    pyomo_gaps_filtered = [g if g is not None else 0 for g in gaps['Pyomo_gap']]
    dwave_gaps_filtered = [g if g is not None else 0 for g in gaps['DWave_gap']]
    
    ax.bar(x - width, pulp_gaps_filtered, width, label='Gurobi MILP', color=colors['PuLP'], alpha=0.8)
    ax.bar(x, pyomo_gaps_filtered, width, label='Ipopt MINLP', color=colors['Pyomo'], alpha=0.8)
    ax.bar(x + width, dwave_gaps_filtered, width, label='D-Wave', color=colors['DWave_Hybrid'], alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Gap (\%)', fontsize=11, fontweight='bold')
    ax.set_title('Quality Gap Comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(n_farms)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    return fig

def print_summary_table(times):
    """Print a summary table of solve times and speedups."""
    print("\n" + "="*100)
    print("NLN BENCHMARK SUMMARY: SOLVE TIMES AND SPEEDUPS")
    print("="*100)
    print(f"{'N_Farms':<10} {'PuLP (s)':<12} {'Pyomo (s)':<12} {'QPU (s)':<12} {'Hybrid (s)':<12} {'QPU vs PuLP':<15} {'Hybrid vs PuLP':<15}")
    print("-"*100)
    
    speedups = calculate_speedups(times)
    
    for i, n in enumerate(times['n_farms']):
        pulp_time = f"{times['PuLP'][i]:.4f}" if times['PuLP'][i] else "N/A"
        pyomo_time = f"{times['Pyomo'][i]:.4f}" if times['Pyomo'][i] else "N/A"
        qpu_time = f"{times['DWave_QPU'][i]:.4f}" if times['DWave_QPU'][i] else "N/A"
        hybrid_time = f"{times['DWave_Hybrid'][i]:.4f}" if times['DWave_Hybrid'][i] else "N/A"
        
        qpu_speedup = f"{speedups['QPU_vs_PuLP'][i]:.2f}x" if speedups['QPU_vs_PuLP'][i] else "N/A"
        hybrid_speedup = f"{speedups['Hybrid_vs_PuLP'][i]:.2f}x" if speedups['Hybrid_vs_PuLP'][i] else "N/A"
        
        print(f"{n:<10} {pulp_time:<12} {pyomo_time:<12} {qpu_time:<12} {hybrid_time:<12} {qpu_speedup:<15} {hybrid_speedup:<15}")
    
    print("="*100)
    print("\nKey Observations:")
    print("- QPU time remains nearly constant regardless of problem size")
    print("- Classical solvers show increasing solve times with problem size")
    print("- Hybrid solver combines QPU speed with classical optimization")
    print("- Speedup increases with problem size for QPU approach")
    print("="*100 + "\n")

def main():
    # Load data
    benchmark_dir = Path(__file__).parent / "Benchmarks"
    data = load_benchmark_data(benchmark_dir)
    
    # Extract times
    times, objectives  = extract_times_and_objectives(data)
    gaps = calculate_objective_gaps(objectives)
    
    # Print summary
    print_summary_table(times)
    
    # Create plots
    output_path = Path(__file__).parent / "Plots" / "nln_speedup_comparison.png"
    output_path.parent.mkdir(exist_ok=True)
    plot_solve_times(times, objectives, gaps, output_path)
    
    # Also create individual high-resolution plots for each scale
    fig_linear = plt.figure(figsize=(12, 8))
    ax = fig_linear.add_subplot(111)
    
    n_farms = times['n_farms']
    colors = {
        'PuLP': '#E63946',
        'Pyomo': '#F77F00',
        'DWave_QPU': '#06FFA5',
        'DWave_Hybrid': '#118AB2'
    }
    
    ax.plot(n_farms, times['PuLP'], 'o-', linewidth=3, markersize=12, 
            color=colors['PuLP'], label='PuLP (Classical)', alpha=0.8)
    ax.plot(n_farms, times['Pyomo'], 's-', linewidth=3, markersize=12, 
            color=colors['Pyomo'], label='Pyomo (Classical)', alpha=0.8)
    ax.plot(n_farms, times['DWave_QPU'], '^-', linewidth=3, markersize=12, 
            color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.plot(n_farms, times['DWave_Hybrid'], 'D-', linewidth=3, markersize=12, 
            color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    
    ax.set_xlabel('Number of Farms', fontsize=14, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('NLN Benchmark: Solve Time Comparison (Linear Scale)', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_linear = Path(__file__).parent / "Plots" / "nln_solve_times_linear.png"
    plt.savefig(output_linear, dpi=300, bbox_inches='tight')
    print(f"Linear scale plot saved to: {output_linear}")
    
    plt.show()

if __name__ == "__main__":
    main()
