"""
NLN Benchmark Visualization with Solution Quality Analysis
Shows solve times, speedup, AND solution quality (objective value gaps).
Introduces "Time-to-Quality" metric that accounts for both speed and accuracy.
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
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (20, 15)
plt.rcParams['font.family'] = 'sans-serif'

def load_benchmark_data(benchmark_dir):
    """Load all NLN benchmark data from the Benchmarks directory."""
    nln_dir = Path(benchmark_dir) / "NLN"
    
    data = {
        'DWave': {},
        'PuLP': {},
        'Pyomo': {},
        'CQM': {}
    }
    
    # Configuration files to load (using run_1 for consistency)
    configs = [5, 19, 72, 279]
    
    for solver in data.keys():
        solver_dir = nln_dir / solver
        if not solver_dir.exists():
            continue
        for config in configs:
            config_file = solver_dir / f"config_{config}_run_1.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    benchmark = json.load(f)
                    data[solver][config] = benchmark
    
    return data

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
        
        # CQM times and objectives
        if config in data['CQM']:
            times['CQM'].append(data['CQM'][config]['result']['cqm_time'])
            objectives['CQM'].append(data['CQM'][config]['result'].get('objective_value', None))
        else:
            times['CQM'].append(None)
            objectives['CQM'].append(None)
        
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
        if objectives['CQM'][i] is not None:
            values.append(objectives['CQM'][i])
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
        
        if objectives['CQM'][i] is not None:
            gap = ((best_obj - objectives['CQM'][i]) / best_obj) * 100 if best_obj != 0 else 0
            gaps['CQM_gap'].append(gap)
        else:
            gaps['CQM_gap'].append(None)
        
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
        'Pyomo_ttq': [],
        'DWave_Hybrid_ttq': []
    }
    
    for i in range(len(configs)):
        # PuLP TTQ
        if times['PuLP'][i] is not None and gaps['PuLP_gap'][i] is not None:
            ttq['PuLP_ttq'].append(times['PuLP'][i] * (1 + gaps['PuLP_gap'][i] / 100))
        else:
            ttq['PuLP_ttq'].append(None)
        
        # Pyomo TTQ
        if times['Pyomo'][i] is not None and gaps['Pyomo_gap'][i] is not None:
            ttq['Pyomo_ttq'].append(times['Pyomo'][i] * (1 + gaps['Pyomo_gap'][i] / 100))
        else:
            ttq['Pyomo_ttq'].append(None)
        
        # DWave Hybrid TTQ
        if times['DWave_Hybrid'][i] is not None and gaps['DWave_gap'][i] is not None:
            ttq['DWave_Hybrid_ttq'].append(times['DWave_Hybrid'][i] * (1 + gaps['DWave_gap'][i] / 100))
        else:
            ttq['DWave_Hybrid_ttq'].append(None)
    
    return ttq

def plot_comprehensive_analysis(times, objectives, gaps, ttq, output_path):
    """Create comprehensive 3x3 plot with times, objectives, and quality-adjusted metrics."""
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    fig.suptitle('NLN Benchmark: Comprehensive Analysis (Time + Solution Quality)', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    n_farms = times['n_farms']
    
    # Colors for each solver
    colors = {
        'PuLP': '#E63946',
        'Pyomo': '#F77F00',
        'CQM': '#FFB703',
        'DWave_QPU': '#06FFA5',
        'DWave_Hybrid': '#118AB2'
    }
    
    markers = {
        'PuLP': 'o',
        'Pyomo': 's',
        'CQM': 'D',
        'DWave_QPU': '^',
        'DWave_Hybrid': 'v'
    }
    
    # === Row 1: Solve Times ===
    
    # Linear scale
    ax = axes[0, 0]
    ax.plot(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='PuLP', alpha=0.8)
    ax.plot(n_farms, times['Pyomo'], marker=markers['Pyomo'], linewidth=2.5, 
            markersize=10, color=colors['Pyomo'], label='Pyomo', alpha=0.8)
    ax.plot(n_farms, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=11, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Solve Time (Linear Scale)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale
    ax = axes[0, 1]
    ax.semilogy(n_farms, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
                markersize=10, color=colors['PuLP'], label='PuLP', alpha=0.8)
    ax.semilogy(n_farms, times['Pyomo'], marker=markers['Pyomo'], linewidth=2.5, 
                markersize=10, color=colors['Pyomo'], label='Pyomo', alpha=0.8)
    ax.semilogy(n_farms, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
                markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=11, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log)', fontsize=11, fontweight='bold')
    ax.set_title('Solve Time (Log-Y Scale)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # QPU Time Focus
    ax = axes[0, 2]
    ax.plot(n_farms, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=3, 
            markersize=12, color=colors['DWave_QPU'], label='D-Wave QPU Time', alpha=0.8)
    ax.axhline(y=np.mean([t for t in times['DWave_QPU'] if t is not None]), 
               color=colors['DWave_QPU'], linestyle='--', linewidth=2, alpha=0.5, 
               label=f'Mean: {np.mean([t for t in times["DWave_QPU"] if t is not None]):.4f}s')
    ax.set_xlabel('Number of Farms', fontsize=11, fontweight='bold')
    ax.set_ylabel('QPU Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('QPU Time (Nearly Constant)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # === Row 2: Solution Quality ===
    
    # Objective Values
    ax = axes[1, 0]
    ax.plot(n_farms, objectives['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='PuLP', alpha=0.8)
    ax.plot(n_farms, objectives['Pyomo'], marker=markers['Pyomo'], linewidth=2.5, 
            markersize=10, color=colors['Pyomo'], label='Pyomo', alpha=0.8)
    ax.plot(n_farms, objectives['DWave'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Value', fontsize=11, fontweight='bold')
    ax.set_title('Objective Values (Higher = Better)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Objective Gaps (%)
    ax = axes[1, 1]
    ax.plot(n_farms, gaps['PuLP_gap'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='PuLP', alpha=0.8)
    ax.plot(n_farms, gaps['Pyomo_gap'], marker=markers['Pyomo'], linewidth=2.5, 
            markersize=10, color=colors['Pyomo'], label='Pyomo', alpha=0.8)
    ax.plot(n_farms, gaps['DWave_gap'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave', alpha=0.8)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Optimal (0% gap)')
    ax.set_xlabel('Number of Farms', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Gap (%)', fontsize=11, fontweight='bold')
    ax.set_title('Solution Quality Gap (% from Best)', fontsize=13, fontweight='bold')
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
    
    ax.bar(x - width, pulp_gaps_filtered, width, label='PuLP', color=colors['PuLP'], alpha=0.8)
    ax.bar(x, pyomo_gaps_filtered, width, label='Pyomo', color=colors['Pyomo'], alpha=0.8)
    ax.bar(x + width, dwave_gaps_filtered, width, label='D-Wave', color=colors['DWave_Hybrid'], alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Gap (%)', fontsize=11, fontweight='bold')
    ax.set_title('Quality Gap Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(n_farms)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # === Row 3: Time-to-Quality Metric ===
    
    # Time-to-Quality (Linear)
    ax = axes[2, 0]
    ax.plot(n_farms, ttq['PuLP_ttq'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='PuLP', alpha=0.8)
    ax.plot(n_farms, ttq['Pyomo_ttq'], marker=markers['Pyomo'], linewidth=2.5, 
            markersize=10, color=colors['Pyomo'], label='Pyomo', alpha=0.8)
    ax.plot(n_farms, ttq['DWave_Hybrid_ttq'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time-to-Quality (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Time-to-Quality = Time × (1 + Gap/100)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Time-to-Quality (Log)
    ax = axes[2, 1]
    ax.semilogy(n_farms, ttq['PuLP_ttq'], marker=markers['PuLP'], linewidth=2.5, 
                markersize=10, color=colors['PuLP'], label='PuLP', alpha=0.8)
    ax.semilogy(n_farms, ttq['Pyomo_ttq'], marker=markers['Pyomo'], linewidth=2.5, 
                markersize=10, color=colors['Pyomo'], label='Pyomo', alpha=0.8)
    ax.semilogy(n_farms, ttq['DWave_Hybrid_ttq'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
                markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time-to-Quality (seconds, log)', fontsize=11, fontweight='bold')
    ax.set_title('Time-to-Quality (Log Scale)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Quality-Adjusted Speedup
    ax = axes[2, 2]
    speedup_pulp = [ttq['PuLP_ttq'][i] / ttq['DWave_Hybrid_ttq'][i] 
                    if ttq['PuLP_ttq'][i] and ttq['DWave_Hybrid_ttq'][i] else None 
                    for i in range(len(n_farms))]
    speedup_pyomo = [ttq['Pyomo_ttq'][i] / ttq['DWave_Hybrid_ttq'][i] 
                     if ttq['Pyomo_ttq'][i] and ttq['DWave_Hybrid_ttq'][i] else None 
                     for i in range(len(n_farms))]
    
    ax.plot(n_farms, speedup_pulp, marker='o', linewidth=3, 
            markersize=12, color=colors['PuLP'], label='PuLP / D-Wave', alpha=0.8)
    ax.plot(n_farms, speedup_pyomo, marker='s', linewidth=3, 
            markersize=12, color=colors['Pyomo'], label='Pyomo / D-Wave', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2.5, label='Break-even (1x)')
    ax.set_xlabel('Number of Farms', fontsize=11, fontweight='bold')
    ax.set_ylabel('Quality-Adjusted Speedup', fontsize=11, fontweight='bold')
    ax.set_title('Speedup (Accounting for Quality)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive analysis plot saved to: {output_path}")
    
    return fig

def print_comprehensive_summary(times, objectives, gaps, ttq):
    """Print a comprehensive summary table with all metrics."""
    print("\n" + "="*160)
    print("NLN COMPREHENSIVE BENCHMARK SUMMARY: TIME + SOLUTION QUALITY")
    print("="*160)
    print(f"{'N_Farms':<10} {'PuLP (s)':<12} {'Pyomo (s)':<12} {'DWave (s)':<12} {'PuLP Obj':<12} {'Pyomo Obj':<12} {'DWave Obj':<12} {'PuLP Gap%':<12} {'Pyomo Gap%':<12} {'DWave Gap%':<12}")
    print("-"*160)
    
    for i, n in enumerate(times['n_farms']):
        pulp_time = f"{times['PuLP'][i]:.4f}" if times['PuLP'][i] else "N/A"
        pyomo_time = f"{times['Pyomo'][i]:.4f}" if times['Pyomo'][i] else "N/A"
        dwave_time = f"{times['DWave_Hybrid'][i]:.4f}" if times['DWave_Hybrid'][i] else "N/A"
        
        pulp_obj = f"{objectives['PuLP'][i]:.4f}" if objectives['PuLP'][i] else "N/A"
        pyomo_obj = f"{objectives['Pyomo'][i]:.4f}" if objectives['Pyomo'][i] else "N/A"
        dwave_obj = f"{objectives['DWave'][i]:.4f}" if objectives['DWave'][i] else "N/A"
        
        pulp_gap = f"{gaps['PuLP_gap'][i]:.2f}" if gaps['PuLP_gap'][i] is not None else "N/A"
        pyomo_gap = f"{gaps['Pyomo_gap'][i]:.2f}" if gaps['Pyomo_gap'][i] is not None else "N/A"
        dwave_gap = f"{gaps['DWave_gap'][i]:.2f}" if gaps['DWave_gap'][i] is not None else "N/A"
        
        print(f"{n:<10} {pulp_time:<12} {pyomo_time:<12} {dwave_time:<12} {pulp_obj:<12} {pyomo_obj:<12} {dwave_obj:<12} {pulp_gap:<12} {pyomo_gap:<12} {dwave_gap:<12}")
    
    print("="*160)
    print("\nKEY OBSERVATIONS:")
    print("- NLN formulation: Nonlinear objective with piecewise linearization")
    print("- D-Wave shows QUALITY GAPS: Solutions are suboptimal compared to classical solvers")
    print("- This is the most complex formulation, making quality gaps expected")
    print("- Time-to-Quality metric accounts for both speed AND solution accuracy")
    print("- For critical applications, quality gap may outweigh speed advantages")
    print("\nRECOMMENDATIONS:")
    print("- Small problems: Use PuLP (fast enough, guarantees optimality)")
    print("- Large problems: Consider D-Wave IF approximate solutions are acceptable")
    print("- Quality-critical applications: Stick with classical solvers (PuLP/Pyomo)")
    print("- Future work: Tune D-Wave parameters to improve solution quality")
    print("="*160 + "\n")

def main():
    # Load data
    benchmark_dir = Path(__file__).parent / "Benchmarks"
    data = load_benchmark_data(benchmark_dir)
    
    # Extract times and objectives
    times, objectives = extract_times_and_objectives(data)
    
    # Calculate quality metrics
    gaps = calculate_objective_gaps(objectives)
    ttq = calculate_time_to_quality(times, gaps)
    
    # Print summary
    print_comprehensive_summary(times, objectives, gaps, ttq)
    
    # Create comprehensive plot
    output_path = Path(__file__).parent / "Plots" / "nln_comprehensive_quality_analysis.png"
    output_path.parent.mkdir(exist_ok=True)
    plot_comprehensive_analysis(times, objectives, gaps, ttq, output_path)
    
    print("\n✓ Comprehensive quality analysis complete!")
    print(f"  Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
