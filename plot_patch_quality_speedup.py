"""
PATCH Benchmark Visualization with Solution Quality Analysis
Shows solve times, speedup, AND solution quality (objective value gaps).
Introduces "Time-to-Quality" metric that accounts for both speed and accuracy.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for beautiful plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (20, 15)
plt.rcParams['font.family'] = 'sans-serif'

def load_benchmark_data(benchmark_dir):
    """Load all PATCH benchmark data from the Benchmarks directory."""
    patch_dir = Path(benchmark_dir) / "PATCH"
    
    data = {
        'DWave': {},
        'PuLP': {},
        'CQM': {}
    }
    
    # Configuration files to load (using run_1 for consistency)
    configs = [5, 10, 15, 25]
    
    for solver in data.keys():
        solver_dir = patch_dir / solver
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
        'n_patches': configs,
        'PuLP': [],
        'CQM': [],
        'DWave_QPU': [],
        'DWave_Hybrid': []
    }
    
    objectives = {
        'n_patches': configs,
        'PuLP': [],
        'CQM': [],
        'DWave': []
    }
    
    for config in configs:
        # PuLP times and objectives
        if config in data['PuLP']:
            times['PuLP'].append(data['PuLP'][config]['result']['solve_time'])
            objectives['PuLP'].append(data['PuLP'][config]['result']['objective_value'])
        else:
            times['PuLP'].append(None)
            objectives['PuLP'].append(None)
        
        # CQM times (for reference, though it's the same model)
        if config in data['CQM']:
            times['CQM'].append(data['CQM'][config]['result']['cqm_time'])
            # CQM doesn't solve, it's just model creation
            objectives['CQM'].append(None)
        else:
            times['CQM'].append(None)
            objectives['CQM'].append(None)
        
        # D-Wave times and objectives
        if config in data['DWave']:
            times['DWave_QPU'].append(data['DWave'][config]['result']['qpu_time'])
            times['DWave_Hybrid'].append(data['DWave'][config]['result']['hybrid_time'])
            objectives['DWave'].append(data['DWave'][config]['result']['objective_value'])
        else:
            times['DWave_QPU'].append(None)
            times['DWave_Hybrid'].append(None)
            objectives['DWave'].append(None)
    
    return times, objectives

def calculate_objective_gaps(objectives):
    """Calculate objective value gaps (percentage deviation from best)."""
    configs = objectives['n_patches']
    gaps = {
        'n_patches': configs,
        'PuLP_gap': [],
        'DWave_gap': []
    }
    
    for i in range(len(configs)):
        # Get all valid objectives for this config
        valid_objs = []
        if objectives['PuLP'][i] is not None:
            valid_objs.append(('PuLP', objectives['PuLP'][i]))
        if objectives['DWave'][i] is not None:
            valid_objs.append(('DWave', objectives['DWave'][i]))
        
        if not valid_objs:
            gaps['PuLP_gap'].append(None)
            gaps['DWave_gap'].append(None)
            continue
        
        # Find best (maximum for maximization problem)
        best_obj = max(obj for _, obj in valid_objs)
        
        # Calculate gaps (percentage deviation from best)
        # Gap = (best - current) / best * 100
        pulp_gap = ((best_obj - objectives['PuLP'][i]) / best_obj * 100) if objectives['PuLP'][i] is not None else None
        dwave_gap = ((best_obj - objectives['DWave'][i]) / best_obj * 100) if objectives['DWave'][i] is not None else None
        
        gaps['PuLP_gap'].append(pulp_gap)
        gaps['DWave_gap'].append(dwave_gap)
    
    return gaps

def calculate_time_to_quality(times, gaps):
    """
    Calculate "Time-to-Quality" metric: time * (1 + gap/100)
    This penalizes solutions that are fast but inaccurate.
    """
    configs = times['n_patches']
    ttq = {
        'n_patches': configs,
        'PuLP_ttq': [],
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

def plot_comprehensive_analysis(times, objectives, gaps, ttq, output_path):
    """Create comprehensive 3x3 plot with times, objectives, and quality-adjusted metrics."""
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    fig.suptitle('PATCH Benchmark: Comprehensive Analysis (Time + Solution Quality)', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    n_patches = times['n_patches']
    
    # Colors for each solver
    colors = {
        'PuLP': '#E63946',
        'CQM': '#FFB703',
        'DWave_QPU': '#06FFA5',
        'DWave_Hybrid': '#118AB2'
    }
    
    markers = {
        'PuLP': 'o',
        'CQM': 'D',
        'DWave_QPU': '^',
        'DWave_Hybrid': 'v'
    }
    
    # === Row 1: Solve Times ===
    
    # Linear scale
    ax = axes[0, 0]
    ax.plot(n_patches, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='PuLP', alpha=0.8)
    ax.plot(n_patches, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Patches', fontsize=11, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Solve Time (Linear Scale)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale
    ax = axes[0, 1]
    ax.semilogy(n_patches, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
                markersize=10, color=colors['PuLP'], label='PuLP', alpha=0.8)
    ax.semilogy(n_patches, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
                markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Patches', fontsize=11, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log)', fontsize=11, fontweight='bold')
    ax.set_title('Solve Time (Log-Y Scale)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # QPU Time Focus
    ax = axes[0, 2]
    ax.plot(n_patches, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=3, 
            markersize=12, color=colors['DWave_QPU'], label='D-Wave QPU Time', alpha=0.8)
    qpu_valid = [t for t in times['DWave_QPU'] if t is not None]
    if qpu_valid:
        ax.axhline(y=np.mean(qpu_valid), 
                   color=colors['DWave_QPU'], linestyle='--', linewidth=2, alpha=0.5, 
                   label=f'Mean: {np.mean(qpu_valid):.4f}s')
    ax.set_xlabel('Number of Patches', fontsize=11, fontweight='bold')
    ax.set_ylabel('QPU Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('QPU Time (Nearly Constant)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # === Row 2: Solution Quality ===
    
    # Objective Values
    ax = axes[1, 0]
    ax.plot(n_patches, objectives['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='PuLP', alpha=0.8)
    ax.plot(n_patches, objectives['DWave'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Patches', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Value', fontsize=11, fontweight='bold')
    ax.set_title('Solution Quality (Objective Values)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Objective Gaps (% deviation from best)
    ax = axes[1, 1]
    ax.plot(n_patches, gaps['PuLP_gap'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='PuLP Gap', alpha=0.8)
    ax.plot(n_patches, gaps['DWave_gap'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Gap', alpha=0.8)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect (0% gap)')
    ax.set_xlabel('Number of Patches', fontsize=11, fontweight='bold')
    ax.set_ylabel('Quality Gap (%)', fontsize=11, fontweight='bold')
    ax.set_title('Objective Gap from Best', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Gap Bar Chart
    ax = axes[1, 2]
    x_pos = np.arange(len(n_patches))
    width = 0.35
    pulp_gaps = [g if g is not None else 0 for g in gaps['PuLP_gap']]
    dwave_gaps = [g if g is not None else 0 for g in gaps['DWave_gap']]
    ax.bar(x_pos - width/2, pulp_gaps, width, label='PuLP', color=colors['PuLP'], alpha=0.8)
    ax.bar(x_pos + width/2, dwave_gaps, width, label='D-Wave', color=colors['DWave_Hybrid'], alpha=0.8)
    ax.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Quality Gap (%)', fontsize=11, fontweight='bold')
    ax.set_title('Gap Comparison by Configuration', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{n} patches' for n in n_patches], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # === Row 3: Time-to-Quality Metrics ===
    
    # Time-to-Quality Comparison
    ax = axes[2, 0]
    ax.plot(n_patches, ttq['PuLP_ttq'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='PuLP TTQ', alpha=0.8)
    ax.plot(n_patches, ttq['DWave_Hybrid_ttq'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave TTQ', alpha=0.8)
    ax.set_xlabel('Number of Patches', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time-to-Quality (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Time-to-Quality Metric', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, 'Lower is better\n(accounts for speed + accuracy)', 
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Speedup (raw time)
    ax = axes[2, 1]
    speedup_hybrid = [times['PuLP'][i] / times['DWave_Hybrid'][i] 
                      if times['PuLP'][i] and times['DWave_Hybrid'][i] else None 
                      for i in range(len(n_patches))]
    ax.plot(n_patches, speedup_hybrid, marker='D', linewidth=2.5, 
            markersize=10, color='#118AB2', label='Hybrid vs PuLP', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Patches', fontsize=11, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
    ax.set_title('Raw Speedup (Time Only)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Quality-Adjusted Speedup (TTQ)
    ax = axes[2, 2]
    speedup_ttq = [ttq['PuLP_ttq'][i] / ttq['DWave_Hybrid_ttq'][i] 
                   if ttq['PuLP_ttq'][i] and ttq['DWave_Hybrid_ttq'][i] else None 
                   for i in range(len(n_patches))]
    ax.plot(n_patches, speedup_ttq, marker='D', linewidth=2.5, 
            markersize=10, color='#06A77D', label='TTQ Speedup', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Patches', fontsize=11, fontweight='bold')
    ax.set_ylabel('Quality-Adjusted Speedup', fontsize=11, fontweight='bold')
    ax.set_title('Time-to-Quality Speedup', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, 'Accounts for both\nspeed AND accuracy', 
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive analysis plot saved to: {output_path}")
    
    return fig

def print_comprehensive_summary(times, objectives, gaps, ttq):
    """Print comprehensive summary including quality metrics."""
    print("\n" + "="*120)
    print("PATCH BENCHMARK: COMPREHENSIVE SUMMARY (Time + Quality)")
    print("="*120)
    print(f"{'N_Patches':<12} {'PuLP Time':<12} {'DWave Time':<12} {'PuLP Obj':<12} {'DWave Obj':<12} {'PuLP Gap%':<12} {'DWave Gap%':<12} {'Time Speedup':<14} {'TTQ Speedup':<14}")
    print("-"*120)
    
    for i, n in enumerate(times['n_patches']):
        pulp_time = f"{times['PuLP'][i]:.4f}" if times['PuLP'][i] else "N/A"
        dwave_time = f"{times['DWave_Hybrid'][i]:.4f}" if times['DWave_Hybrid'][i] else "N/A"
        
        pulp_obj = f"{objectives['PuLP'][i]:.6f}" if objectives['PuLP'][i] else "N/A"
        dwave_obj = f"{objectives['DWave'][i]:.6f}" if objectives['DWave'][i] else "N/A"
        
        pulp_gap = f"{gaps['PuLP_gap'][i]:.2f}%" if gaps['PuLP_gap'][i] is not None else "N/A"
        dwave_gap = f"{gaps['DWave_gap'][i]:.2f}%" if gaps['DWave_gap'][i] is not None else "N/A"
        
        time_speedup = f"{times['PuLP'][i] / times['DWave_Hybrid'][i]:.2f}x" if times['PuLP'][i] and times['DWave_Hybrid'][i] else "N/A"
        ttq_speedup = f"{ttq['PuLP_ttq'][i] / ttq['DWave_Hybrid_ttq'][i]:.2f}x" if ttq['PuLP_ttq'][i] and ttq['DWave_Hybrid_ttq'][i] else "N/A"
        
        print(f"{n:<12} {pulp_time:<12} {dwave_time:<12} {pulp_obj:<12} {dwave_obj:<12} {pulp_gap:<12} {dwave_gap:<12} {time_speedup:<14} {ttq_speedup:<14}")
    
    print("="*120)
    print("\nKey Insights:")
    print("- BQM_PATCH formulation: Binary plot-crop assignments with implicit idle")
    print("- Time-to-Quality (TTQ) accounts for both speed AND solution accuracy")
    print("- Lower quality gap = better solution quality")
    print("- TTQ Speedup shows true performance advantage accounting for quality")
    print("="*120 + "\n")

def main():
    # Load data
    benchmark_dir = Path(__file__).parent / "Benchmarks"
    data = load_benchmark_data(benchmark_dir)
    
    # Extract times and objectives
    times, objectives = extract_times_and_objectives(data)
    
    # Calculate quality gaps
    gaps = calculate_objective_gaps(objectives)
    
    # Calculate time-to-quality
    ttq = calculate_time_to_quality(times, gaps)
    
    # Print comprehensive summary
    print_comprehensive_summary(times, objectives, gaps, ttq)
    
    # Create comprehensive plot
    output_path = Path(__file__).parent / "Plots" / "patch_quality_speedup_analysis.png"
    output_path.parent.mkdir(exist_ok=True)
    plot_comprehensive_analysis(times, objectives, gaps, ttq, output_path)
    
    plt.show()

if __name__ == "__main__":
    main()
