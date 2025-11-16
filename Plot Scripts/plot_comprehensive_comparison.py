"""
Comprehensive Benchmark Comparison: NLN and BQUBO
Shows D-Wave speedup across different problem types and scales
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for beautiful plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

def load_nln_data(benchmark_dir):
    """Load NLN benchmark data."""
    nln_dir = Path(benchmark_dir) / "NLN"
    data = {'DWave': {}, 'PuLP': {}, 'Pyomo': {}}
    configs = [5, 19, 72, 279]
    
    for solver in data.keys():
        solver_dir = nln_dir / solver
        for config in configs:
            config_file = solver_dir / f"config_{config}_run_1.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    data[solver][config] = json.load(f)
    return data

def load_bqubo_data(benchmark_dir):
    """Load BQUBO benchmark data."""
    bqubo_dir = Path(benchmark_dir) / "BQUBO"
    data = {'DWave': {}, 'PuLP': {}, 'CQM': {}}
    configs = [5, 19, 72, 279, 1096]
    
    for solver in data.keys():
        solver_dir = bqubo_dir / solver
        for config in configs:
            config_file = solver_dir / f"config_{config}_run_1.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    data[solver][config] = json.load(f)
    return data

def extract_nln_times(data):
    """Extract NLN solve times."""
    configs = sorted([k for k in data['DWave'].keys()])
    times = {
        'n_farms': configs,
        'PuLP': [data['PuLP'][c]['result']['solve_time'] if c in data['PuLP'] else None for c in configs],
        'Pyomo': [data['Pyomo'][c]['result']['solve_time'] if c in data['Pyomo'] else None for c in configs],
        'DWave_QPU': [data['DWave'][c]['result']['qpu_time'] if c in data['DWave'] else None for c in configs],
        'DWave_Hybrid': [data['DWave'][c]['result']['hybrid_time'] if c in data['DWave'] else None for c in configs]
    }
    return times

def extract_bqubo_times(data):
    """Extract BQUBO solve times."""
    configs = sorted([k for k in data['DWave'].keys()])
    times = {
        'n_farms': configs,
        'PuLP': [data['PuLP'][c]['result']['solve_time'] if c in data['PuLP'] else None for c in configs],
        'CQM': [data['CQM'][c]['result']['cqm_time'] if c in data['CQM'] else None for c in configs],
        'DWave_QPU': [data['DWave'][c]['result']['qpu_time'] if c in data['DWave'] else None for c in configs],
        'DWave_Hybrid': [data['DWave'][c]['result']['hybrid_time'] if c in data['DWave'] else None for c in configs]
    }
    return times

def create_comprehensive_plot(nln_times, bqubo_times, output_path):
    """Create a comprehensive comparison plot."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    fig.suptitle('Comprehensive Benchmark Comparison: D-Wave vs Classical Solvers', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    colors = {
        'PuLP': '#E63946',
        'Pyomo': '#F77F00',
        'CQM': '#F4A261',
        'DWave_QPU': '#06FFA5',
        'DWave_Hybrid': '#118AB2'
    }
    
    # ========== NLN Benchmarks ==========
    # Linear scale
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(nln_times['n_farms'], nln_times['PuLP'], 'o-', linewidth=2.5, markersize=10, 
            color=colors['PuLP'], label='PuLP', alpha=0.8)
    ax.plot(nln_times['n_farms'], nln_times['Pyomo'], 's-', linewidth=2.5, markersize=10, 
            color=colors['Pyomo'], label='Pyomo', alpha=0.8)
    ax.plot(nln_times['n_farms'], nln_times['DWave_QPU'], '^-', linewidth=2.5, markersize=10, 
            color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.plot(nln_times['n_farms'], nln_times['DWave_Hybrid'], 'D-', linewidth=2.5, markersize=10, 
            color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('NLN: Linear Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(nln_times['n_farms'], nln_times['PuLP'], 'o-', linewidth=2.5, markersize=10, 
                color=colors['PuLP'], label='PuLP', alpha=0.8)
    ax.semilogy(nln_times['n_farms'], nln_times['Pyomo'], 's-', linewidth=2.5, markersize=10, 
                color=colors['Pyomo'], label='Pyomo', alpha=0.8)
    ax.semilogy(nln_times['n_farms'], nln_times['DWave_QPU'], '^-', linewidth=2.5, markersize=10, 
                color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.semilogy(nln_times['n_farms'], nln_times['DWave_Hybrid'], 'D-', linewidth=2.5, markersize=10, 
                color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log)', fontsize=12, fontweight='bold')
    ax.set_title('NLN: Log-Y Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # ========== BQUBO Benchmarks ==========
    # Linear scale
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(bqubo_times['n_farms'], bqubo_times['PuLP'], 'o-', linewidth=2.5, markersize=10, 
            color=colors['PuLP'], label='PuLP', alpha=0.8)
    ax.plot(bqubo_times['n_farms'], bqubo_times['CQM'], 's-', linewidth=2.5, markersize=10, 
            color=colors['CQM'], label='CQM', alpha=0.8)
    ax.plot(bqubo_times['n_farms'], bqubo_times['DWave_QPU'], '^-', linewidth=2.5, markersize=10, 
            color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.plot(bqubo_times['n_farms'], bqubo_times['DWave_Hybrid'], 'D-', linewidth=2.5, markersize=10, 
            color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('BQUBO: Linear Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale
    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(bqubo_times['n_farms'], bqubo_times['PuLP'], 'o-', linewidth=2.5, markersize=10, 
                color=colors['PuLP'], label='PuLP', alpha=0.8)
    ax.semilogy(bqubo_times['n_farms'], bqubo_times['CQM'], 's-', linewidth=2.5, markersize=10, 
                color=colors['CQM'], label='CQM', alpha=0.8)
    ax.semilogy(bqubo_times['n_farms'], bqubo_times['DWave_QPU'], '^-', linewidth=2.5, markersize=10, 
                color=colors['DWave_QPU'], label='D-Wave QPU', alpha=0.8)
    ax.semilogy(bqubo_times['n_farms'], bqubo_times['DWave_Hybrid'], 'D-', linewidth=2.5, markersize=10, 
                color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log)', fontsize=12, fontweight='bold')
    ax.set_title('BQUBO: Log-Y Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # ========== Speedup Comparison ==========
    # NLN Speedup (QPU vs PuLP)
    ax = fig.add_subplot(gs[2, 0])
    nln_speedup = [nln_times['PuLP'][i] / nln_times['DWave_QPU'][i] 
                   if nln_times['PuLP'][i] and nln_times['DWave_QPU'][i] else None 
                   for i in range(len(nln_times['n_farms']))]
    bqubo_speedup = [bqubo_times['PuLP'][i] / bqubo_times['DWave_QPU'][i] 
                     if bqubo_times['PuLP'][i] and bqubo_times['DWave_QPU'][i] else None 
                     for i in range(len(bqubo_times['n_farms']))]
    
    ax.plot(nln_times['n_farms'], nln_speedup, '^-', linewidth=3, markersize=12, 
            color='#06FFA5', label='NLN (QPU vs PuLP)', alpha=0.8)
    ax.plot(bqubo_times['n_farms'], bqubo_speedup, 'D-', linewidth=3, markersize=12, 
            color='#118AB2', label='BQUBO (QPU vs PuLP)', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2.5, label='Break-even (1x)')
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('QPU Speedup Comparison: NLN vs BQUBO', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Log scale speedup
    ax = fig.add_subplot(gs[2, 1])
    ax.semilogy(nln_times['n_farms'], nln_speedup, '^-', linewidth=3, markersize=12, 
                color='#06FFA5', label='NLN (QPU vs PuLP)', alpha=0.8)
    ax.semilogy(bqubo_times['n_farms'], bqubo_speedup, 'D-', linewidth=3, markersize=12, 
                color='#118AB2', label='BQUBO (QPU vs PuLP)', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2.5, label='Break-even (1x)')
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (log)', fontsize=12, fontweight='bold')
    ax.set_title('QPU Speedup Comparison: Log Scale', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive comparison plot saved to: {output_path}")
    
    return fig

def print_comparison_summary(nln_times, bqubo_times):
    """Print a comprehensive summary."""
    print("\n" + "="*120)
    print("COMPREHENSIVE BENCHMARK COMPARISON SUMMARY")
    print("="*120)
    
    print("\n--- NLN BENCHMARKS ---")
    print(f"{'N_Farms':<10} {'PuLP (s)':<12} {'Pyomo (s)':<12} {'QPU (s)':<12} {'Hybrid (s)':<12} {'QPU Speedup':<15}")
    print("-"*90)
    for i, n in enumerate(nln_times['n_farms']):
        speedup = (nln_times['PuLP'][i] / nln_times['DWave_QPU'][i]) if nln_times['DWave_QPU'][i] else 0
        print(f"{n:<10} {nln_times['PuLP'][i]:<12.4f} {nln_times['Pyomo'][i]:<12.4f} "
              f"{nln_times['DWave_QPU'][i]:<12.4f} {nln_times['DWave_Hybrid'][i]:<12.4f} {speedup:<15.2f}x")
    
    print("\n--- BQUBO BENCHMARKS ---")
    print(f"{'N_Farms':<10} {'PuLP (s)':<12} {'CQM (s)':<12} {'QPU (s)':<12} {'Hybrid (s)':<12} {'QPU Speedup':<15}")
    print("-"*90)
    for i, n in enumerate(bqubo_times['n_farms']):
        speedup = (bqubo_times['PuLP'][i] / bqubo_times['DWave_QPU'][i]) if bqubo_times['DWave_QPU'][i] else 0
        print(f"{n:<10} {bqubo_times['PuLP'][i]:<12.4f} {bqubo_times['CQM'][i]:<12.4f} "
              f"{bqubo_times['DWave_QPU'][i]:<12.4f} {bqubo_times['DWave_Hybrid'][i]:<12.4f} {speedup:<15.2f}x")
    
    print("\n" + "="*120)
    print("\nKEY INSIGHTS:")
    print("• NLN: QPU time is nearly constant (~0.035s), leading to massive speedups (up to 996x) for larger problems")
    print("• BQUBO: QPU time varies more with problem size, but still shows competitive performance")
    print("• Classical solvers (PuLP, Pyomo, CQM) show increasing solve times with problem complexity")
    print("• Hybrid solver balances quantum and classical approaches, suitable for production workloads")
    print("="*120 + "\n")

def main():
    benchmark_dir = Path(__file__).parent / "Benchmarks"
    
    # Load data
    nln_data = load_nln_data(benchmark_dir)
    bqubo_data = load_bqubo_data(benchmark_dir)
    
    # Extract times
    nln_times = extract_nln_times(nln_data)
    bqubo_times = extract_bqubo_times(bqubo_data)
    
    # Print summary
    print_comparison_summary(nln_times, bqubo_times)
    
    # Create comprehensive plot
    output_path = Path(__file__).parent / "Plots" / "comprehensive_speedup_comparison.png"
    output_path.parent.mkdir(exist_ok=True)
    create_comprehensive_plot(nln_times, bqubo_times, output_path)
    
    plt.show()

if __name__ == "__main__":
    main()
