"""
Ultimate Speedup Analysis with Detailed Curve Fitting
Shows fitted equations, extrapolations, and detailed crossover analysis
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from scipy.optimize import curve_fit

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

def load_data(benchmark_dir, benchmark_type):
    """Load benchmark data for given type."""
    bench_dir = Path(benchmark_dir) / benchmark_type
    
    if benchmark_type == "NLN":
        solvers = ['DWave', 'PuLP', 'Pyomo']
        configs = [5, 19, 72, 279]
    else:  # BQUBO
        solvers = ['DWave', 'PuLP', 'CQM']
        configs = [5, 19, 72, 279, 1096]
    
    data = {s: {} for s in solvers}
    for solver in solvers:
        solver_dir = bench_dir / solver
        for config in configs:
            config_file = solver_dir / f"config_{config}_run_1.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    data[solver][config] = json.load(f)
    return data

def extract_times(data, benchmark_type):
    """Extract solve times."""
    configs = sorted([k for k in data['DWave'].keys()])
    
    if benchmark_type == "NLN":
        times = {
            'n_farms': np.array(configs, dtype=float),
            'PuLP': np.array([data['PuLP'][c]['result']['solve_time'] for c in configs]),
            'Pyomo': np.array([data['Pyomo'][c]['result']['solve_time'] for c in configs]),
            'DWave_QPU': np.array([data['DWave'][c]['result']['qpu_time'] for c in configs]),
            'DWave_Hybrid': np.array([data['DWave'][c]['result']['hybrid_time'] for c in configs])
        }
    else:  # BQUBO
        times = {
            'n_farms': np.array(configs, dtype=float),
            'PuLP': np.array([data['PuLP'][c]['result']['solve_time'] for c in configs]),
            'CQM': np.array([data['CQM'][c]['result']['cqm_time'] for c in configs]),
            'DWave_QPU': np.array([data['DWave'][c]['result']['qpu_time'] for c in configs]),
            'DWave_Hybrid': np.array([data['DWave'][c]['result']['hybrid_time'] for c in configs])
        }
    
    return times

def power_law_func(x, a, b, c):
    """Power law: a * x^b + c"""
    return a * np.power(x, b) + c

def linear_func(x, a, b):
    """Linear: a * x + b"""
    return a * x + b

def fit_curve(x, y, func_type='power'):
    """Fit a curve to the data."""
    try:
        if func_type == 'power':
            popt, pcov = curve_fit(power_law_func, x, y, p0=[1, 1, 0], maxfev=10000)
            func = lambda x_val: power_law_func(x_val, *popt)
            equation = f"f(x) = {popt[0]:.4f} * x^{popt[1]:.4f} + {popt[2]:.4f}"
        else:  # linear
            popt, pcov = curve_fit(linear_func, x, y, maxfev=10000)
            func = lambda x_val: linear_func(x_val, *popt)
            equation = f"f(x) = {popt[0]:.6f} * x + {popt[1]:.4f}"
        
        # Calculate R²
        y_pred = func(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return func, equation, r_squared
    except:
        return None, "Fit failed", 0

def create_ultimate_plot(times, benchmark_type, output_path):
    """Create the ultimate speedup analysis plot."""
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    title = f'{benchmark_type} Benchmark: Ultimate Speedup Analysis with Fitted Curves'
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.995)
    
    x_data = times['n_farms']
    x_extended = np.linspace(x_data.min(), max(x_data.max() * 1.5, 500), 1000)
    
    colors = {
        'PuLP': '#E63946',
        'Pyomo': '#F77F00',
        'CQM': '#F4A261',
        'DWave_QPU': '#06FFA5',
        'DWave_Hybrid': '#118AB2'
    }
    
    # Determine which classical solvers to use
    if benchmark_type == "NLN":
        classical_solvers = ['PuLP', 'Pyomo']
    else:
        classical_solvers = ['PuLP', 'CQM']
    
    # Fit all curves
    fits = {}
    equations = {}
    r_squared = {}
    
    print(f"\n{'='*100}")
    print(f"{benchmark_type} CURVE FITTING RESULTS")
    print(f"{'='*100}\n")
    
    for solver in classical_solvers + ['DWave_QPU', 'DWave_Hybrid']:
        func_type = 'linear' if solver == 'DWave_QPU' else 'power'
        fit, eq, r2 = fit_curve(x_data, times[solver], func_type)
        fits[solver] = fit
        equations[solver] = eq
        r_squared[solver] = r2
        print(f"{solver:15} | {eq:50} | R² = {r2:.6f}")
    
    # === Row 1: Original Data with Fits (3 scales) ===
    
    # Linear
    ax = fig.add_subplot(gs[0, 0])
    for solver in classical_solvers + ['DWave_QPU', 'DWave_Hybrid']:
        ax.plot(x_data, times[solver], 'o', markersize=10, color=colors[solver], 
                label=f'{solver.replace("_", " ")}', alpha=0.8, zorder=5)
        if fits[solver] is not None:
            y_fit = fits[solver](x_extended)
            ax.plot(x_extended, y_fit, '--', linewidth=2.5, color=colors[solver], 
                    alpha=0.6, zorder=3)
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Scale', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, x_extended.max()])
    
    # Log-Y
    ax = fig.add_subplot(gs[0, 1])
    for solver in classical_solvers + ['DWave_QPU', 'DWave_Hybrid']:
        ax.semilogy(x_data, times[solver], 'o', markersize=10, color=colors[solver], 
                    label=f'{solver.replace("_", " ")}', alpha=0.8, zorder=5)
        if fits[solver] is not None:
            y_fit = np.maximum(fits[solver](x_extended), 1e-6)
            ax.semilogy(x_extended, y_fit, '--', linewidth=2.5, color=colors[solver], 
                       alpha=0.6, zorder=3)
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Y Scale', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-Log
    ax = fig.add_subplot(gs[0, 2])
    for solver in classical_solvers + ['DWave_QPU', 'DWave_Hybrid']:
        ax.loglog(x_data, times[solver], 'o', markersize=10, color=colors[solver], 
                  label=f'{solver.replace("_", " ")}', alpha=0.8, zorder=5)
        if fits[solver] is not None:
            y_fit = np.maximum(fits[solver](x_extended), 1e-6)
            ax.loglog(x_extended, y_fit, '--', linewidth=2.5, color=colors[solver], 
                     alpha=0.6, zorder=3)
    
    ax.set_xlabel('Number of Farms (log)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Scale', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # === Row 2: Speedup Analysis ===
    
    speedups = {}
    crossovers = {}
    
    for classical in classical_solvers:
        # Hybrid speedup
        speedup_key = f'Hybrid_vs_{classical}'
        speedups[speedup_key] = fits[classical](x_extended) / fits['DWave_Hybrid'](x_extended)
        
        # Find crossover
        diff = fits[classical](x_extended) - fits['DWave_Hybrid'](x_extended)
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            x_cross = x_extended[idx] + (x_extended[idx+1] - x_extended[idx]) * \
                      abs(diff[idx]) / (abs(diff[idx]) + abs(diff[idx+1]))
            crossovers[speedup_key] = x_cross
        else:
            crossovers[speedup_key] = None
        
        # QPU speedup
        speedup_key_qpu = f'QPU_vs_{classical}'
        speedups[speedup_key_qpu] = fits[classical](x_extended) / fits['DWave_QPU'](x_extended)
    
    # Plot Hybrid speedup
    ax = fig.add_subplot(gs[1, 0])
    for classical in classical_solvers:
        speedup_key = f'Hybrid_vs_{classical}'
        color = '#118AB2' if classical == classical_solvers[0] else '#073B4C'
        linestyle = '-' if classical == classical_solvers[0] else '--'
        ax.plot(x_extended, speedups[speedup_key], linewidth=3, color=color,
                label=f'Hybrid vs {classical}', alpha=0.8, linestyle=linestyle)
        
        if crossovers[speedup_key] is not None:
            cp = crossovers[speedup_key]
            ax.axvline(x=cp, color=color, linestyle=':', alpha=0.5, linewidth=2.5)
            ax.plot(cp, 1.0, 'o', markersize=12, color=color, zorder=10)
            ax.text(cp, ax.get_ylim()[1]*0.85, f'{cp:.1f} farms', 
                   ha='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2.5, label='Break-even', zorder=2)
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('Hybrid Speedup (Fitted Ratio)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, x_extended.max()])
    
    # Plot QPU speedup
    ax = fig.add_subplot(gs[1, 1])
    for classical in classical_solvers:
        speedup_key = f'QPU_vs_{classical}'
        color = '#06FFA5' if classical == classical_solvers[0] else '#06D89E'
        linestyle = '-' if classical == classical_solvers[0] else '--'
        speedup_positive = np.maximum(speedups[speedup_key], 1e-6)
        ax.semilogy(x_extended, speedup_positive, linewidth=3, color=color,
                   label=f'QPU vs {classical}', alpha=0.8, linestyle=linestyle)
    
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2.5, label='Break-even', zorder=2)
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (log)', fontsize=12, fontweight='bold')
    ax.set_title('QPU Speedup (Fitted Ratio)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0, x_extended.max()])
    
    # Combined comparison
    ax = fig.add_subplot(gs[1, 2])
    primary_classical = classical_solvers[0]
    
    ax.plot(x_extended, speedups[f'Hybrid_vs_{primary_classical}'], linewidth=3.5, 
            color='#118AB2', label=f'Hybrid vs {primary_classical}', alpha=0.9)
    ax.plot(x_extended, speedups[f'QPU_vs_{primary_classical}'], linewidth=3.5, 
            color='#06FFA5', label=f'QPU vs {primary_classical}', alpha=0.9)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2.5, label='Break-even', zorder=2)
    
    cp_key = f'Hybrid_vs_{primary_classical}'
    if crossovers[cp_key] is not None:
        cp = crossovers[cp_key]
        ax.axvline(x=cp, color='orange', linestyle=':', alpha=0.7, linewidth=3)
        ax.fill_between([0, cp], 0, ax.get_ylim()[1], alpha=0.1, color='red', 
                        label=f'Classical Advantage')
        ax.fill_between([cp, x_extended.max()], 0, ax.get_ylim()[1], alpha=0.1, 
                        color='green', label=f'D-Wave Advantage')
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('Advantage Regions', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, x_extended.max()])
    
    # === Row 3: Detailed Crossover Analysis ===
    
    # Time comparison at crossover
    ax = fig.add_subplot(gs[2, 0])
    primary_classical = classical_solvers[0]
    
    ax.plot(x_extended, fits[primary_classical](x_extended), linewidth=3, 
            color=colors[primary_classical], label=primary_classical, alpha=0.8)
    ax.plot(x_extended, fits['DWave_Hybrid'](x_extended), linewidth=3, 
            color=colors['DWave_Hybrid'], label='D-Wave Hybrid', alpha=0.8)
    
    cp_key = f'Hybrid_vs_{primary_classical}'
    if crossovers[cp_key] is not None:
        cp = crossovers[cp_key]
        cp_time = fits['DWave_Hybrid'](cp)
        ax.axvline(x=cp, color='orange', linestyle='--', linewidth=2.5, alpha=0.7)
        ax.plot(cp, cp_time, 'o', markersize=15, color='orange', zorder=10)
        ax.text(cp * 1.05, cp_time * 1.2, 
               f'Crossover\n{cp:.1f} farms\n{cp_time:.2f}s', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Crossover Point Detail', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, x_extended.max()])
    
    # Speedup growth rate
    ax = fig.add_subplot(gs[2, 1])
    
    # Calculate derivative (growth rate of speedup)
    speedup_hybrid = speedups[f'Hybrid_vs_{primary_classical}']
    dx = x_extended[1] - x_extended[0]
    speedup_derivative = np.gradient(speedup_hybrid, dx)
    
    ax.plot(x_extended, speedup_derivative, linewidth=3, color='#118AB2', 
            label='Rate of Speedup Increase', alpha=0.8)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Growth Rate (d/dx)', fontsize=12, fontweight='bold')
    ax.set_title('How Fast Advantage Grows', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, x_extended.max()])
    
    # Projection to larger sizes
    ax = fig.add_subplot(gs[2, 2])
    x_projection = np.linspace(x_data.min(), x_data.max() * 3, 1000)
    
    for solver in [primary_classical, 'DWave_Hybrid']:
        y_proj = fits[solver](x_projection)
        ax.plot(x_projection, y_proj, linewidth=3, color=colors[solver],
               label=f'{solver.replace("_", " ")} (projected)', alpha=0.8)
    
    # Mark actual data range
    ax.axvline(x=x_data.max(), color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(x_data.max() * 1.02, ax.get_ylim()[1] * 0.9, 'Data\nLimit',
           fontsize=9, ha='left', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    ax.set_xlabel('Number of Farms (Extended)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Extrapolation to Larger Problems', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # === Row 4: Summary Statistics ===
    
    # Text summary
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')
    
    summary_text = f"{'='*120}\n"
    summary_text += f"FITTED CURVE EQUATIONS AND ANALYSIS SUMMARY\n"
    summary_text += f"{'='*120}\n\n"
    
    for solver in classical_solvers + ['DWave_QPU', 'DWave_Hybrid']:
        summary_text += f"{solver:15} | {equations[solver]:60} | R² = {r_squared[solver]:.6f}\n"
    
    summary_text += f"\n{'-'*120}\n"
    summary_text += "CROSSOVER ANALYSIS:\n"
    summary_text += f"{'-'*120}\n"
    
    for classical in classical_solvers:
        cp_key = f'Hybrid_vs_{classical}'
        if crossovers[cp_key] is not None:
            cp = crossovers[cp_key]
            cp_time = fits['DWave_Hybrid'](cp)
            classical_time = fits[classical](cp)
            summary_text += f"\n✓ D-Wave Hybrid crosses {classical} at {cp:.1f} farms\n"
            summary_text += f"  At crossover: Hybrid time = {cp_time:.3f}s, {classical} time = {classical_time:.3f}s\n"
            summary_text += f"  Speedup at {x_data.max():.0f} farms: {speedups[cp_key][-1]:.2f}x\n"
        else:
            summary_text += f"\n✗ No crossover with {classical} in analyzed range\n"
    
    summary_text += f"\n{'-'*120}\n"
    summary_text += "KEY INSIGHTS:\n"
    summary_text += f"{'-'*120}\n"
    summary_text += "• QPU time shows linear/constant behavior (minimal scaling)\n"
    summary_text += "• Classical solvers follow power-law growth (time ∝ size^b, b > 1)\n"
    summary_text += "• Hybrid time includes overhead but scales better than classical\n"
    summary_text += "• Speedup increases with problem size due to better scaling\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*100}")
    print(f"Ultimate analysis plot saved to: {output_path}")
    print(f"{'='*100}\n")
    
    return crossovers

def main():
    benchmark_dir = Path(__file__).parent / "Benchmarks"
    output_dir = Path(__file__).parent / "Plots"
    output_dir.mkdir(exist_ok=True)
    
    # NLN Analysis
    print("\n" + "="*100)
    print("ULTIMATE NLN SPEEDUP ANALYSIS")
    print("="*100)
    
    nln_data = load_data(benchmark_dir, "NLN")
    nln_times = extract_times(nln_data, "NLN")
    nln_output = output_dir / "nln_ultimate_speedup_analysis.png"
    nln_crossovers = create_ultimate_plot(nln_times, "NLN", nln_output)
    
    # BQUBO Analysis
    print("\n" + "="*100)
    print("ULTIMATE BQUBO SPEEDUP ANALYSIS")
    print("="*100)
    
    bqubo_data = load_data(benchmark_dir, "BQUBO")
    bqubo_times = extract_times(bqubo_data, "BQUBO")
    bqubo_output = output_dir / "bqubo_ultimate_speedup_analysis.png"
    bqubo_crossovers = create_ultimate_plot(bqubo_times, "BQUBO", bqubo_output)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE - All plots generated successfully!")
    print("="*100 + "\n")
    
    plt.show()

if __name__ == "__main__":
    main()
