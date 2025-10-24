"""
PATCH Benchmark Visualization: D-Wave vs Classical Solver (PuLP)
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
    """Load all PATCH benchmark data from the Benchmarks directory."""
    patch_dir = Path(benchmark_dir) / "PATCH"
    
    data = {
        'DWave': {},
        'PuLP': {}
    }
    
    # Configuration files to load
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

def extract_times(data):
    """Extract solve times for each solver and configuration."""
    configs = sorted([k for k in data['DWave'].keys()])
    
    times = {
        'n_patches': configs,
        'PuLP': [],
        'DWave_QPU': [],
        'DWave_Hybrid': []
    }
    
    for config in configs:
        # PuLP times
        if config in data['PuLP']:
            times['PuLP'].append(data['PuLP'][config]['result']['solve_time'])
        else:
            times['PuLP'].append(None)
        
        # D-Wave times
        if config in data['DWave']:
            times['DWave_QPU'].append(data['DWave'][config]['result']['qpu_time'])
            times['DWave_Hybrid'].append(data['DWave'][config]['result']['hybrid_time'])
        else:
            times['DWave_QPU'].append(None)
            times['DWave_Hybrid'].append(None)
    
    return times

def calculate_speedups(times):
    """Calculate speedup factors comparing D-Wave to PuLP."""
    speedups = {
        'n_patches': times['n_patches'],
        'QPU_vs_PuLP': [],
        'Hybrid_vs_PuLP': []
    }
    
    for i in range(len(times['n_patches'])):
        # QPU vs PuLP
        if times['PuLP'][i] and times['DWave_QPU'][i]:
            speedups['QPU_vs_PuLP'].append(times['PuLP'][i] / times['DWave_QPU'][i])
        else:
            speedups['QPU_vs_PuLP'].append(None)
        
        # Hybrid vs PuLP
        if times['PuLP'][i] and times['DWave_Hybrid'][i]:
            speedups['Hybrid_vs_PuLP'].append(times['PuLP'][i] / times['DWave_Hybrid'][i])
        else:
            speedups['Hybrid_vs_PuLP'].append(None)
    
    return speedups

def plot_solve_times(times, output_path):
    """Create three plots: linear, log-y, and log-log scales."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('PATCH Benchmark: Solve Time Comparison (D-Wave vs PuLP)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    n_patches = times['n_patches']
    
    # Colors for each solver
    colors = {
        'PuLP': '#E63946',      # Red
        'DWave_QPU': '#06FFA5',  # Cyan
        'DWave_Hybrid': '#118AB2' # Blue
    }
    
    markers = {
        'PuLP': 'o',
        'DWave_QPU': '^',
        'DWave_Hybrid': 'D'
    }
    
    # Row 1: Solve Times
    # Linear scale
    ax = axes[0, 0]
    ax.plot(n_patches, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
            markersize=10, color=colors['PuLP'], label='PuLP (Classical)', alpha=0.8)
    ax.plot(n_patches, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
            markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.plot(n_patches, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
            markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Patches', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale
    ax = axes[0, 1]
    ax.semilogy(n_patches, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
                markersize=10, color=colors['PuLP'], label='PuLP (Classical)', alpha=0.8)
    ax.semilogy(n_patches, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
                markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.semilogy(n_patches, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
                markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Patches', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Y Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale
    ax = axes[0, 2]
    ax.loglog(n_patches, times['PuLP'], marker=markers['PuLP'], linewidth=2.5, 
              markersize=10, color=colors['PuLP'], label='PuLP (Classical)', alpha=0.8)
    ax.loglog(n_patches, times['DWave_QPU'], marker=markers['DWave_QPU'], linewidth=2.5, 
              markersize=10, color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.loglog(n_patches, times['DWave_Hybrid'], marker=markers['DWave_Hybrid'], linewidth=2.5, 
              markersize=10, color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Patches (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Row 2: Speedup Factors
    speedups = calculate_speedups(times)
    
    # Linear scale speedup
    ax = axes[1, 0]
    ax.plot(n_patches, speedups['QPU_vs_PuLP'], marker='^', linewidth=2.5, 
            markersize=10, color='#06FFA5', label='QPU vs PuLP', alpha=0.8)
    ax.plot(n_patches, speedups['Hybrid_vs_PuLP'], marker='D', linewidth=2.5, 
            markersize=10, color='#118AB2', label='Hybrid vs PuLP', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Patches', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('Speedup: Linear Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale speedup
    ax = axes[1, 1]
    ax.semilogy(n_patches, speedups['QPU_vs_PuLP'], marker='^', linewidth=2.5, 
                markersize=10, color='#06FFA5', label='QPU vs PuLP', alpha=0.8)
    ax.semilogy(n_patches, speedups['Hybrid_vs_PuLP'], marker='D', linewidth=2.5, 
                markersize=10, color='#118AB2', label='Hybrid vs PuLP', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Patches', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup: Log-Y Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale speedup
    ax = axes[1, 2]
    ax.loglog(n_patches, speedups['QPU_vs_PuLP'], marker='^', linewidth=2.5, 
              markersize=10, color='#06FFA5', label='QPU vs PuLP', alpha=0.8)
    ax.loglog(n_patches, speedups['Hybrid_vs_PuLP'], marker='D', linewidth=2.5, 
              markersize=10, color='#118AB2', label='Hybrid vs PuLP', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2, label='Break-even (1x)')
    ax.set_xlabel('Number of Patches (log scale)', fontsize=12, fontweight='bold')
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
    print("\n" + "="*90)
    print("PATCH BENCHMARK SUMMARY: SOLVE TIMES AND SPEEDUPS")
    print("="*90)
    print(f"{'N_Patches':<12} {'PuLP (s)':<12} {'QPU (s)':<12} {'Hybrid (s)':<12} {'QPU vs PuLP':<15} {'Hybrid vs PuLP':<15}")
    print("-"*90)
    
    speedups = calculate_speedups(times)
    
    for i, n in enumerate(times['n_patches']):
        pulp_time = f"{times['PuLP'][i]:.4f}" if times['PuLP'][i] else "N/A"
        qpu_time = f"{times['DWave_QPU'][i]:.4f}" if times['DWave_QPU'][i] else "N/A"
        hybrid_time = f"{times['DWave_Hybrid'][i]:.4f}" if times['DWave_Hybrid'][i] else "N/A"
        
        qpu_speedup = f"{speedups['QPU_vs_PuLP'][i]:.2f}x" if speedups['QPU_vs_PuLP'][i] else "N/A"
        hybrid_speedup = f"{speedups['Hybrid_vs_PuLP'][i]:.2f}x" if speedups['Hybrid_vs_PuLP'][i] else "N/A"
        
        print(f"{n:<12} {pulp_time:<12} {qpu_time:<12} {hybrid_time:<12} {qpu_speedup:<15} {hybrid_speedup:<15}")
    
    print("="*90)
    print("\nKey Observations:")
    print("- BQM_PATCH formulation: plot-crop assignment with implicit idle area")
    print("- QPU time remains nearly constant regardless of problem size")
    print("- PuLP time increases with problem size (patches × crops)")
    print("- Hybrid solver combines QPU speed with classical optimization")
    print("- Speedup increases with problem size for QPU approach")
    print("="*90 + "\n")

def main():
    # Load data
    benchmark_dir = Path(__file__).parent / "Benchmarks"
    data = load_benchmark_data(benchmark_dir)
    
    # Extract times
    times = extract_times(data)
    
    # Print summary
    print_summary_table(times)
    
    # Create plots
    output_path = Path(__file__).parent / "Plots" / "patch_speedup_comparison.png"
    output_path.parent.mkdir(exist_ok=True)
    plot_solve_times(times, output_path)
    
    # Also create individual high-resolution plots for each scale
    fig_linear = plt.figure(figsize=(12, 8))
    ax = fig_linear.add_subplot(111)
    
    n_patches = times['n_patches']
    colors = {
        'PuLP': '#E63946',
        'DWave_QPU': '#06FFA5',
        'DWave_Hybrid': '#118AB2'
    }
    
    ax.plot(n_patches, times['PuLP'], 'o-', linewidth=3, markersize=12, 
            color=colors['PuLP'], label='PuLP (Classical)', alpha=0.8)
    ax.plot(n_patches, times['DWave_QPU'], '^-', linewidth=3, markersize=12, 
            color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.plot(n_patches, times['DWave_Hybrid'], 'D-', linewidth=3, markersize=12, 
            color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    
    ax.set_xlabel('Number of Patches', fontsize=14, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('PATCH Benchmark: Solve Time Comparison (Linear Scale)', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_linear = Path(__file__).parent / "Plots" / "patch_solve_times_linear.png"
    plt.savefig(output_linear, dpi=300, bbox_inches='tight')
    print(f"Linear scale plot saved to: {output_linear}")
    
    plt.show()

if __name__ == "__main__":
    main()
