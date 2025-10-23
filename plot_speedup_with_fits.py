"""
Speedup Analysis with Power Law Fits
Shows classical vs D-Wave timing data and extrapolated curves to identify crossover points
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# Set matplotlib style for professional plots
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8

def load_nln_data(benchmark_dir):
    """Load NLN benchmark data."""
    nln_dir = Path(benchmark_dir) / "NLN"
    data = {'DWave': {}, 'PuLP': {}, 'Pyomo': {}}
    configs = [5, 19, 72, 279]
    
    for solver in data.keys():
        for config in configs:
            file_path = nln_dir / solver / f"config_{config}_run_1.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data[solver][config] = json.load(f)
    return data

def load_bqubo_data(benchmark_dir):
    """Load BQUBO benchmark data."""
    bqubo_dir = Path(benchmark_dir) / "BQUBO"
    data = {'DWave': {}, 'PuLP': {}, 'CQM': {}}
    configs = [5, 19, 72, 279, 1096]
    
    for solver in data.keys():
        for config in configs:
            file_path = bqubo_dir / solver / f"config_{config}_run_1.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data[solver][config] = json.load(f)
    return data

def extract_nln_times(data):
    """Extract NLN solve times."""
    configs = sorted([k for k in data['DWave'].keys()])
    times = {
        'n_farms': np.array(configs, dtype=float),
        'PuLP': np.array([data['PuLP'][c]['result']['solve_time'] for c in configs]),
        'Pyomo': np.array([data['Pyomo'][c]['result']['solve_time'] for c in configs]),
        'DWave_QPU': np.array([data['DWave'][c]['result']['qpu_time'] for c in configs]),
        'DWave_Hybrid': np.array([data['DWave'][c]['result']['hybrid_time'] for c in configs])
    }
    return times

def extract_bqubo_times(data):
    """Extract BQUBO solve times."""
    configs = sorted([k for k in data['DWave'].keys()])
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

def fit_power_law(x, y):
    """Fit power law to data."""
    try:
        popt, pcov = curve_fit(power_law_func, x, y, 
                               p0=[y[0], 1.5, 0], 
                               maxfev=10000,
                               bounds=([0, 0, -np.inf], [np.inf, 5, np.inf]))
        # Calculate R-squared
        residuals = y - power_law_func(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return lambda x_val: power_law_func(x_val, *popt), popt, r_squared
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, None, None

def find_crossover(x_range, fit_classical, fit_dwave):
    """Find where D-Wave becomes faster than classical."""
    if fit_classical is None or fit_dwave is None:
        return None
    
    classical_vals = fit_classical(x_range)
    dwave_vals = fit_dwave(x_range)
    
    # Find where D-Wave becomes faster (classical > dwave)
    speedup_mask = classical_vals > dwave_vals
    if np.any(speedup_mask):
        crossover_idx = np.where(speedup_mask)[0][0]
        return x_range[crossover_idx]
    return None

def create_comparison_plot(times, benchmark_type, output_path):
    """Create comprehensive comparison plot with power law fits."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25, 
                          left=0.08, right=0.98, top=0.94, bottom=0.06)
    
    fig.suptitle(f'{benchmark_type} Benchmark: Solve Time & Speedup Analysis (Power Law Fits)', 
                 fontsize=16, fontweight='bold')
    
    x_data = times['n_farms']
    # Extend x-axis for extrapolation (up to 3x the max or 2000, whichever is smaller)
    x_max_extended = min(x_data.max() * 3, 2000)
    x_extended = np.linspace(x_data.min(), x_max_extended, 1000)
    
    colors = {
        'PuLP': '#1f77b4',      # Professional blue
        'Pyomo': '#ff7f0e',     # Orange
        'CQM': '#2ca02c',       # Green
        'DWave_QPU': '#d62728', # Red
        'DWave_Hybrid': '#9467bd' # Purple
    }
    
    markers = {
        'PuLP': 'o',
        'Pyomo': 's',
        'CQM': '^',
        'DWave_QPU': 'D',
        'DWave_Hybrid': 'v'
    }
    
    # Determine which classical solvers we have
    if benchmark_type == "NLN":
        classical_solvers = ['PuLP', 'Pyomo']
    else:
        classical_solvers = ['PuLP', 'CQM']
    
    # Store fits
    fits = {}
    params = {}
    r_squared = {}
    
    print(f"\n{'='*100}")
    print(f"{benchmark_type} POWER LAW FITTING RESULTS")
    print(f"{'='*100}\n")
    
    # Fit all curves
    for solver in classical_solvers + ['DWave_QPU', 'DWave_Hybrid']:
        y_data = times[solver]
        fits[solver], params[solver], r_squared[solver] = fit_power_law(x_data, y_data)
        
        print(f"{solver}:")
        if params[solver] is not None:
            a, b, c = params[solver]
            print(f"  Equation: {a:.6e} * x^{b:.4f} + {c:.6e}")
            print(f"  R² = {r_squared[solver]:.6f}")
        print()
    
    # ========== ROW 1: Time Data with Fits (Linear & Log-Y) ==========
    
    # Linear scale
    ax = fig.add_subplot(gs[0, 0])
    for solver in classical_solvers + ['DWave_QPU', 'DWave_Hybrid']:
        # Plot original data
        ax.plot(x_data, times[solver], markers[solver], markersize=9, 
                color=colors[solver], label=f'{solver.replace("_", " ")} (data)',
                markeredgecolor='white', markeredgewidth=1.5, zorder=3)
        # Plot fit
        if fits[solver] is not None:
            ax.plot(x_extended, fits[solver](x_extended), '--', linewidth=2, 
                   color=colors[solver], alpha=0.7, zorder=2)
    
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontweight='bold')
    ax.set_title('Solve Time Comparison - Linear Scale', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='black')
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_xlim([0, x_max_extended])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Log-Y scale
    ax = fig.add_subplot(gs[0, 1])
    for solver in classical_solvers + ['DWave_QPU', 'DWave_Hybrid']:
        ax.semilogy(x_data, times[solver], markers[solver], markersize=9, 
                    color=colors[solver], label=f'{solver.replace("_", " ")} (data)',
                    markeredgecolor='white', markeredgewidth=1.5, zorder=3)
        if fits[solver] is not None:
            y_fit = fits[solver](x_extended)
            y_fit = np.maximum(y_fit, 1e-6)
            ax.semilogy(x_extended, y_fit, '--', linewidth=2, 
                       color=colors[solver], alpha=0.7, zorder=2)
    
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontweight='bold')
    ax.set_title('Solve Time Comparison - Log-Y Scale', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='black')
    ax.grid(True, linestyle='-', alpha=0.3, which='both')
    ax.set_xlim([0, x_max_extended])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ========== ROW 2: QPU Speedup Analysis ==========
    
    print(f"\n{'='*100}")
    print(f"QPU SPEEDUP CROSSOVER POINTS")
    print(f"{'='*100}\n")
    
    # QPU vs Classical - Linear scale
    ax = fig.add_subplot(gs[1, 0])
    for classical in classical_solvers:
        if fits[classical] is not None and fits['DWave_QPU'] is not None:
            speedup = fits[classical](x_extended) / fits['DWave_QPU'](x_extended)
            speedup = np.maximum(speedup, 0)
            ax.plot(x_extended, speedup, '-', linewidth=2.5, 
                   color=colors[classical], label=f'QPU vs {classical}')
            
            crossover = find_crossover(x_extended, fits[classical], fits['DWave_QPU'])
            if crossover:
                print(f"QPU vs {classical}: Crossover at {crossover:.1f} farms")
                ax.axvline(crossover, color=colors[classical], linestyle=':', 
                          linewidth=2, alpha=0.6)
                ax.text(crossover, ax.get_ylim()[1]*0.85, f'{crossover:.0f}', 
                       ha='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor=colors[classical], linewidth=1.5))
    
    ax.axhline(1, color='black', linestyle='--', linewidth=1.5, label='No Speedup', alpha=0.5)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontweight='bold')
    ax.set_title('QPU Speedup vs Classical - Linear Scale', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='black')
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_xlim([0, x_max_extended])
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # QPU vs Classical - Log scale
    ax = fig.add_subplot(gs[1, 1])
    for classical in classical_solvers:
        if fits[classical] is not None and fits['DWave_QPU'] is not None:
            speedup = fits[classical](x_extended) / fits['DWave_QPU'](x_extended)
            speedup = np.maximum(speedup, 1e-3)
            ax.semilogy(x_extended, speedup, '-', linewidth=2.5, 
                       color=colors[classical], label=f'QPU vs {classical}')
            
            crossover = find_crossover(x_extended, fits[classical], fits['DWave_QPU'])
            if crossover:
                ax.axvline(crossover, color=colors[classical], linestyle=':', 
                          linewidth=2, alpha=0.6)
    
    ax.axhline(1, color='black', linestyle='--', linewidth=1.5, label='No Speedup', alpha=0.5)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Speedup Factor (log scale)', fontweight='bold')
    ax.set_title('QPU Speedup vs Classical - Log-Y Scale', fontweight='bold')
    ax.legend(loc='best', framealpha=0.95, edgecolor='black')
    ax.grid(True, linestyle='-', alpha=0.3, which='both')
    ax.set_xlim([0, x_max_extended])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ========== ROW 3: Hybrid Speedup Analysis ==========
    
    print(f"\n{'='*100}")
    print(f"HYBRID SPEEDUP CROSSOVER POINTS")
    print(f"{'='*100}\n")
    
    # Hybrid vs Classical - Linear scale
    ax = fig.add_subplot(gs[2, 0])
    for classical in classical_solvers:
        if fits[classical] is not None and fits['DWave_Hybrid'] is not None:
            speedup = fits[classical](x_extended) / fits['DWave_Hybrid'](x_extended)
            speedup = np.maximum(speedup, 0)
            ax.plot(x_extended, speedup, '-', linewidth=2.5, 
                   color=colors[classical], label=f'Hybrid vs {classical}')
            
            crossover = find_crossover(x_extended, fits[classical], fits['DWave_Hybrid'])
            if crossover:
                print(f"Hybrid vs {classical}: Crossover at {crossover:.1f} farms")
                ax.axvline(crossover, color=colors[classical], linestyle=':', 
                          linewidth=2, alpha=0.6)
                ax.text(crossover, ax.get_ylim()[1]*0.85, f'{crossover:.0f}', 
                       ha='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor=colors[classical], linewidth=1.5))
            else:
                print(f"Hybrid vs {classical}: No crossover found (within {x_max_extended:.0f} farms)")
    
    ax.axhline(1, color='black', linestyle='--', linewidth=1.5, label='No Speedup', alpha=0.5)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontweight='bold')
    ax.set_title('Hybrid Speedup vs Classical - Linear Scale', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='black')
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_xlim([0, x_max_extended])
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Hybrid vs Classical - Log scale
    ax = fig.add_subplot(gs[2, 1])
    for classical in classical_solvers:
        if fits[classical] is not None and fits['DWave_Hybrid'] is not None:
            speedup = fits[classical](x_extended) / fits['DWave_Hybrid'](x_extended)
            speedup = np.maximum(speedup, 1e-3)
            ax.semilogy(x_extended, speedup, '-', linewidth=2.5, 
                       color=colors[classical], label=f'Hybrid vs {classical}')
            
            crossover = find_crossover(x_extended, fits[classical], fits['DWave_Hybrid'])
            if crossover:
                ax.axvline(crossover, color=colors[classical], linestyle=':', 
                          linewidth=2, alpha=0.6)
    
    ax.axhline(1, color='black', linestyle='--', linewidth=1.5, label='No Speedup', alpha=0.5)
    ax.set_xlabel('Number of Farms', fontweight='bold')
    ax.set_ylabel('Speedup Factor (log scale)', fontweight='bold')
    ax.set_title('Hybrid Speedup vs Classical - Log-Y Scale', fontweight='bold')
    ax.legend(loc='best', framealpha=0.95, edgecolor='black')
    ax.grid(True, linestyle='-', alpha=0.3, which='both')
    ax.set_xlim([0, x_max_extended])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_path}")
    plt.close()

def main():
    benchmark_dir = Path("Benchmarks")
    output_dir = Path("Plots")
    output_dir.mkdir(exist_ok=True)
    
    # Process NLN benchmarks
    print("\n" + "="*100)
    print("LOADING NLN BENCHMARKS")
    print("="*100)
    nln_data = load_nln_data(benchmark_dir)
    nln_times = extract_nln_times(nln_data)
    create_comparison_plot(nln_times, "NLN", output_dir / "nln_speedup_with_fits.png")
    
    # Process BQUBO benchmarks
    print("\n" + "="*100)
    print("LOADING BQUBO BENCHMARKS")
    print("="*100)
    bqubo_data = load_bqubo_data(benchmark_dir)
    bqubo_times = extract_bqubo_times(bqubo_data)
    create_comparison_plot(bqubo_times, "BQUBO", output_dir / "bqubo_speedup_with_fits.png")
    
    print("\n" + "="*100)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*100)

if __name__ == "__main__":
    main()
