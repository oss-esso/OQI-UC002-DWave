"""
BQUBO Benchmark Visualization: D-Wave vs Classical Solvers
Creates beautiful plots showing solve times and speedup comparisons
in linear, log-y, and log-log scales.
"""

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    """Load all BQUBO benchmark data from the Benchmarks directory."""
    bqubo_dir = Path(benchmark_dir) / "BQUBO"
    
    data = {
        'DWave': {},
        'PuLP': {},
        'GurobiQUBO': {},
        'CQM': {}
    }
    
    # Configuration files to load (using run_1 for consistency)
    configs = [5, 19, 72, 279]
    
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
        'DWave_Hybrid': []
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
        
        
        # DWave times
        if config in data['DWave']:
            dwave = data['DWave'][config].get('result', {})
            qpu_time = dwave.get('qpu_time', None)
            hybrid_time = dwave.get('hybrid_time', None)
        else:
            qpu_time = None
            hybrid_time = None
        
        times['DWave_QPU'].append(qpu_time)
        times['DWave_Hybrid'].append(hybrid_time)
    
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
        
        
        # Total speedups
        if times['PuLP'][i] and times['DWave_Hybrid'][i]:
            speedups['Total_vs_PuLP'].append(times['PuLP'][i] / times['DWave_Hybrid'][i])
        else:
            speedups['Total_vs_PuLP'].append(None)
        
        if times['GurobiQUBO'][i] and times['DWave_Hybrid'][i]:
            speedups['Total_vs_GurobiQUBO'].append(times['GurobiQUBO'][i] / times['DWave_Hybrid'][i])
        else:
            speedups['Total_vs_GurobiQUBO'].append(None)
        
        if times['CQM'][i] and times['DWave_Hybrid'][i]:
            speedups['Total_vs_CQM'].append(times['CQM'][i] / times['DWave_Hybrid'][i])
        else:
            speedups['Total_vs_CQM'].append(None)
    
    return speedups

def extract_times_and_objectives(data):
    """Extract solve times AND objective values for each solver and configuration."""
    configs = sorted([k for k in data['DWave'].keys()])
    
    times = {
        'n_farms': configs,
        'PuLP': [],
        'GurobiQUBO': [],
        'DWave_QPU': [],
        'DWave_Hybrid': []
    }
    
    objectives = {
        'n_farms': configs,
        'PuLP': [],
        'GurobiQUBO': [],
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
        
        # GurobiQUBO times
        if config in data['GurobiQUBO']:
            gurobi_time = data['GurobiQUBO'][config].get('result', {}).get('solve_time', None)
            gurobi_obj = data['GurobiQUBO'][config].get('result', {}).get('objective_value', None)
        else:
            gurobi_time = None
            gurobi_obj = None
        times['GurobiQUBO'].append(gurobi_time)
        objectives['GurobiQUBO'].append(gurobi_obj)
        
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
        'DWave_gap': [],
        'GurobiQUBO_gap': []
    }
    
    for i in range(len(configs)):
        # Find best (maximum) objective for this config
        values = []
        if objectives['PuLP'][i] is not None:
            values.append(objectives['PuLP'][i])
        if objectives['DWave'][i] is not None:
            values.append(objectives['DWave'][i])
        if objectives['GurobiQUBO'][i] is not None:
            values.append(objectives['GurobiQUBO'][i])
        
        if not values:
            gaps['PuLP_gap'].append(None)
            gaps['DWave_gap'].append(None)
            gaps['GurobiQUBO_gap'].append(None)
            continue
        
        best_obj = max(values)  # Assuming maximization
        
        # Calculate gaps as percentage
        if objectives['PuLP'][i] is not None:
            gap = ((best_obj - objectives['PuLP'][i]) / best_obj) * 100 if best_obj != 0 else 0
            gaps['PuLP_gap'].append(gap)
        else:
            gaps['PuLP_gap'].append(None)
        
        if objectives['GurobiQUBO'][i] is not None:
            gap = ((best_obj - objectives['GurobiQUBO'][i]) / best_obj) * 100 if best_obj != 0 else 0
            gaps['GurobiQUBO_gap'].append(gap)
        else:
            gaps['GurobiQUBO_gap'].append(None)
        
        if objectives['DWave'][i] is not None:
            gap = ((best_obj - objectives['DWave'][i]) / best_obj) * 100 if best_obj != 0 else 0
            gaps['DWave_gap'].append(gap)
        else:
            gaps['DWave_gap'].append(None)
    
    return gaps

def calculate_time_to_quality(times, gaps):
    """
    Calculate "Time-to-Quality" metric: time * (1 + gap/100)
    This penalizes solutions that are fast but inaccurate.
    """
    configs = times['n_farms']
    ttq = {
        'n_farms': configs,
        'PuLP_ttq': [],
        'CQM_ttq': [],
        'DWave_Hybrid_ttq': []
    }
    
    for i in range(len(configs)):
        # PuLP TTQ
        if times['PuLP'][i] is not None and gaps['PuLP_gap'][i] is not None:
            ttq['PuLP_ttq'].append(times['PuLP'][i] * (1 + gaps['PuLP_gap'][i] / 100))
        else:
            ttq['PuLP_ttq'].append(None)
        
        
        # DWave Hybrid TTQ
        if times['DWave_Hybrid'][i] is not None and gaps['DWave_gap'][i] is not None:
            ttq['DWave_Hybrid_ttq'].append(times['DWave_Hybrid'][i] * (1 + gaps['DWave_gap'][i] / 100))
        else:
            ttq['DWave_Hybrid_ttq'].append(None)
    
    return ttq


def plot_solve_times(times, objectives, gaps, output_path):
    """Create three plots: linear, log-y, and log-log scales."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('BQUBO Benchmark: Solve Time Comparison (D-Wave vs Classical Solvers)', 
                 fontsize=18, y=0.995)
    
    n_farms = 27 * np.array(times['n_farms'])
    
    # Colors for each solver
    colors = {
        'PuLP': '#E63946',      # Red
        'GurobiQUBO': '#9D4EDD', # Purple
        'DWave_QPU': '#06FFA5',  # Green
        'DWave_Hybrid': '#118AB2' # Blue
    }
    
    markers = {
        'PuLP': 'o',
        'GurobiQUBO': 'x',
        'DWave_QPU': '^',
        'DWave_Hybrid': 'D'
    }
    
    # Row 1: Solve Times
    # Linear scale
    ax = axes[0, 0]
    ax.plot(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='Gurobi MILP', alpha=0.8)
    ax.plot(n_farms, times['GurobiQUBO'], marker=markers['GurobiQUBO'], linewidth=2.5, 
            markersize=10, color=colors['GurobiQUBO'], label='Gurobi QUBO', alpha=0.8)
    ax.plot(n_farms, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
            markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.plot(n_farms, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale
    ax = axes[0, 1]
    ax.semilogy(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
                markersize=10, color=colors['PuLP'], label='Gurobi MILP', alpha=0.8)
    ax.semilogy(n_farms, times['GurobiQUBO'], marker=markers['GurobiQUBO'], linewidth=2.5, 
                markersize=10, color=colors['GurobiQUBO'], label='Gurobi QUBO', alpha=0.8)
    ax.semilogy(n_farms, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
                markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.semilogy(n_farms, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
                markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Y Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale
    ax = axes[0, 2]
    ax.loglog(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
              markersize=10, color=colors['PuLP'], label='Gurobi MILP', alpha=0.8)
    ax.loglog(n_farms, times['GurobiQUBO'], marker=markers['GurobiQUBO'], linewidth=2.5, 
              markersize=10, color=colors['GurobiQUBO'], label='Gurobi QUBO', alpha=0.8)
    ax.loglog(n_farms, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
              markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.loglog(n_farms, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
              markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Problem Size (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # === Row 2: Solution Quality ===
    
    # Objective Values
    ax = axes[1, 0]
    ax.plot(n_farms, objectives['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='Gurobi MILP', alpha=0.8)
    ax.plot(n_farms, objectives['DWave'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave', alpha=0.8)
    ax.plot(n_farms, objectives['GurobiQUBO'], marker=markers['GurobiQUBO'], linewidth=2.5, 
            markersize=10, color=colors['GurobiQUBO'], label='Gurobi QUBO', alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Value', fontsize=11, fontweight='bold')
    ax.set_title('Objective Values (Higher = Better)', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Objective Gaps (%)
    ax = axes[1, 1]
    ax.plot(n_farms, gaps['PuLP_gap'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='Gurobi MILP', alpha=0.8)
    ax.plot(n_farms, gaps['DWave_gap'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave', alpha=0.8)
    ax.plot(n_farms, gaps['GurobiQUBO_gap'], marker=markers['GurobiQUBO'], linewidth=2.5, 
            markersize=10, color=colors['GurobiQUBO'], label='Gurobi QUBO', alpha=0.8)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Optimal (0\% gap)')
    ax.set_xlabel('Problem Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Gap (\%)', fontsize=11, fontweight='bold')
    ax.set_title('Solution Quality Gap (\% from Best)', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Objective Gaps (Bar Chart)
    ax = axes[1, 2]
    x = np.arange(len(n_farms))
    width = 0.3
    ax.bar(x - width, gaps['PuLP_gap'], width, label='Gurobi MILP', color=colors['PuLP'], alpha=0.8)
    ax.bar(x + width, gaps['DWave_gap'], width, label='D-Wave', color=colors['DWave_Hybrid'], alpha=0.8)
    ax.bar(x + 3*width, gaps['GurobiQUBO_gap'], width, label='Gurobi QUBO', color=colors['GurobiQUBO'], alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=11, fontweight='bold')
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
    print("\n" + "="*130)
    print("BQUBO BENCHMARK SUMMARY: SOLVE TIMES AND SPEEDUPS")
    print("="*130)
    print(f"{'N_Farms':<10} {'PuLP (s)':<12} {'Gurobi (s)':<12} {'CQM (s)':<12} {'QPU (s)':<12} {'Total (s)':<12} {'QPU/PuLP':<12} {'Total/PuLP':<12}")
    print("-"*130)
    
    #speedups = calculate_speedups(times)
    
    #for i, n in enumerate(times['n_farms']):
    #    pulp_time = f"{times['PuLP'][i]:.4f}" if times['PuLP'][i] else "N/A"
    #    gurobi_time = f"{times['GurobiQUBO'][i]:.4f}" if times['GurobiQUBO'][i] else "N/A"
    #    #cqm_time = f"{times['CQM'][i]:.4f}" if times['CQM'][i] else "N/A"
    #    qpu_time = f"{times['DWave_QPU'][i]:.4f}" if times['DWave_QPU'][i] else "N/A"
    #    total_time = f"{times['DWave_Hybrid'][i]:.4f}" if times['DWave_Hybrid'][i] else "N/A"
    #    qpu_speedup = f"{speedups['QPU_vs_PuLP'][i]:.2f}x" if speedups['QPU_vs_PuLP'][i] else "N/A"
    #    total_speedup = f"{speedups['Total_vs_PuLP'][i]:.2f}x" if speedups['Total_vs_PuLP'][i] else "N/A"
    #    
    #    print(f"{n:<10} {pulp_time:<12} {gurobi_time:<12} {qpu_time:<12} {total_time:<12} {qpu_speedup:<12} {total_speedup:<12}")
    #
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
    times, objectives = extract_times_and_objectives(data)
    gaps = calculate_objective_gaps(objectives)
    # Print summary
    print_summary_table(times)
    
    # Create plots
    output_path = Path(__file__).parent / "Plots" / "bqubo_speedup_comparison.png"
    output_path.parent.mkdir(exist_ok=True)
    plot_solve_times(times, objectives, gaps, output_path)
    
    # Also create individual high-resolution plot for linear scale
    fig_linear = plt.figure(figsize=(12, 8))
    ax = fig_linear.add_subplot(111)
    
    n_farms = times['n_farms']
    colors = {
        'PuLP': '#E63946',
        'CQM': '#F77F00',
        'DWave_QPU': '#06FFA5',
        'DWave_Hybrid': '#118AB2'
    }
    
    ax.plot(n_farms, times['PuLP'], 'o-', linewidth=3, markersize=12, 
            color=colors['PuLP'], label='Gurobi MILP', alpha=0.8)
    ax.plot(n_farms, times['DWave_QPU'], '^-', linewidth=3, markersize=12, 
            color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.plot(n_farms, times['DWave_Hybrid'], 'D-', linewidth=3, markersize=12, 
            color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    
    ax.set_xlabel('Problem Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('BQUBO Benchmark: Solve Time Comparison (Linear Scale)', 
                 fontsize=16)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_linear = Path(__file__).parent / "Plots" / "bqubo_solve_times_linear.png"
    plt.savefig(output_linear, dpi=300, bbox_inches='tight')
    print(f"Linear scale plot saved to: {output_linear}")
    
    plt.show()

if __name__ == "__main__":
    main()
