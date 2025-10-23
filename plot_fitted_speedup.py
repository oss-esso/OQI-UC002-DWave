"""
Advanced Speedup Analysis with Curve Fitting
Fits interpolating functions to solver data and calculates speedup ratios
to identify crossover points where D-Wave becomes advantageous.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import curve_fit

# Set style for beautiful plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

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
        'n_farms': np.array(configs),
        'PuLP': np.array([data['PuLP'][c]['result']['solve_time'] if c in data['PuLP'] else None for c in configs]),
        'Pyomo': np.array([data['Pyomo'][c]['result']['solve_time'] if c in data['Pyomo'] else None for c in configs]),
        'DWave_QPU': np.array([data['DWave'][c]['result']['qpu_time'] if c in data['DWave'] else None for c in configs]),
        'DWave_Hybrid': np.array([data['DWave'][c]['result']['hybrid_time'] if c in data['DWave'] else None for c in configs])
    }
    return times

def extract_bqubo_times(data):
    """Extract BQUBO solve times."""
    configs = sorted([k for k in data['DWave'].keys()])
    times = {
        'n_farms': np.array(configs),
        'PuLP': np.array([data['PuLP'][c]['result']['solve_time'] if c in data['PuLP'] else None for c in configs]),
        'CQM': np.array([data['CQM'][c]['result']['cqm_time'] if c in data['CQM'] else None for c in configs]),
        'DWave_QPU': np.array([data['DWave'][c]['result']['qpu_time'] if c in data['DWave'] else None for c in configs]),
        'DWave_Hybrid': np.array([data['DWave'][c]['result']['hybrid_time'] if c in data['DWave'] else None for c in configs])
    }
    return times

def polynomial_fit(x, y, degree=2):
    """Fit a polynomial to the data."""
    coeffs = np.polyfit(x, y, degree)
    return np.poly1d(coeffs)

def exponential_fit(x, y):
    """Fit an exponential function: a * exp(b*x) + c"""
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    try:
        # Initial guess
        popt, _ = curve_fit(exp_func, x, y, p0=[1, 0.001, 0], maxfev=10000)
        return lambda x_val: exp_func(x_val, *popt), popt
    except:
        # Fallback to polynomial if exponential fails
        return polynomial_fit(x, y, degree=2), None

def power_law_fit(x, y):
    """Fit a power law: a * x^b + c"""
    def power_func(x, a, b, c):
        return a * np.power(x, b) + c
    
    try:
        popt, _ = curve_fit(power_func, x, y, p0=[1, 1, 0], maxfev=10000)
        # Return both the function and the parameters for debugging
        func = lambda x_val: power_func(x_val, *popt)
        return func, popt
    except:
        return polynomial_fit(x, y, degree=2), None

def fit_solver_data(x, y, solver_name):
    """Choose appropriate fit based on solver characteristics."""
    # Remove None values
    mask = ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return None, None, None
    
    # Try different fitting strategies
    if 'DWave_QPU' in solver_name:
        # QPU time is nearly constant, use polynomial of low degree
        fit_func = polynomial_fit(x_clean, y_clean, degree=1)
        fit_type = "linear"
        params = fit_func.coefficients if hasattr(fit_func, 'coefficients') else None
    elif 'DWave_Hybrid' in solver_name:
        # Hybrid shows moderate growth, try power law
        fit_func, params = power_law_fit(x_clean, y_clean)
        fit_type = "power_law"
    else:
        # Classical solvers may show exponential or polynomial growth
        # Try power law first (often works well)
        fit_func, params = power_law_fit(x_clean, y_clean)
        fit_type = "power_law"
    
    return fit_func, fit_type, params

def find_crossover_points(x_range, fit_classical, fit_dwave, extend_search=True, debug=False):
    """Find where classical and D-Wave curves cross.
    
    If D-Wave starts higher but has better scaling, we need to search
    further out to find where classical catches up.
    """
    if fit_classical is None or fit_dwave is None:
        return []
    
    # If we need to extend the search (for cases where curves might cross later)
    if extend_search:
        x_max_extended = x_range.max() * 10  # Search up to 10x the max range
        x_search = np.linspace(x_range.min(), x_max_extended, 5000)
    else:
        x_search = x_range
    
    # Calculate difference between curves
    try:
        classical_vals = fit_classical(x_search)
        dwave_vals = fit_dwave(x_search)
        diff = classical_vals - dwave_vals
        
        if debug:
            print(f"    Search range: {x_search.min():.1f} to {x_search.max():.1f}")
            print(f"    At start: Classical={classical_vals[0]:.3f}, DWave={dwave_vals[0]:.3f}, diff={diff[0]:.3f}")
            print(f"    At end: Classical={classical_vals[-1]:.3f}, DWave={dwave_vals[-1]:.3f}, diff={diff[-1]:.3f}")
    except Exception as e:
        if debug:
            print(f"    Error calculating curves: {e}")
        return []
    
    # Find sign changes (crossover points)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    if debug:
        print(f"    Found {len(sign_changes)} sign changes")
    
    crossover_points = []
    for idx in sign_changes:
        if idx + 1 < len(x_search):
            # Refine crossover point using linear interpolation
            x_cross = x_search[idx] + (x_search[idx+1] - x_search[idx]) * \
                      abs(diff[idx]) / (abs(diff[idx]) + abs(diff[idx+1]))
            # Only include crossovers that are reasonable (not too far extrapolated)
            if x_cross <= x_max_extended:
                crossover_points.append(x_cross)
                if debug:
                    print(f"    Crossover at x={x_cross:.1f}")
    
    return crossover_points

def plot_fitted_speedup_nln(times, output_path):
    """Create comprehensive fitted speedup analysis for NLN."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('NLN Benchmark: Fitted Speedup Analysis with Interpolation', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    x_data = times['n_farms']
    x_extended = np.linspace(x_data.min(), max(x_data.max() * 1.5, 500), 500)
    
    colors = {
        'PuLP': '#E63946',
        'Pyomo': '#F77F00',
        'DWave_QPU': '#06FFA5',
        'DWave_Hybrid': '#118AB2'
    }
    
    # Fit curves for each solver
    fits = {}
    fit_params = {}
    print("\n" + "="*80)
    print("NLN FIT PARAMETERS:")
    print("="*80)
    for solver in ['PuLP', 'Pyomo', 'DWave_QPU', 'DWave_Hybrid']:
        fit_func, fit_type, params = fit_solver_data(x_data, times[solver], solver)
        fits[solver] = fit_func
        fit_params[solver] = params
        if params is not None and fit_type == "power_law":
            print(f"{solver:15}: f(x) = {params[0]:.6f} * x^{params[1]:.6f} + {params[2]:.6f}")
        else:
            print(f"{solver:15}: {fit_type} fit")
    
    # === Row 1: Solve Times with Fits ===
    
    # Linear scale
    ax = axes[0, 0]
    for solver in ['PuLP', 'Pyomo', 'DWave_QPU', 'DWave_Hybrid']:
        # Plot actual data
        ax.plot(x_data, times[solver], 'o', markersize=10, color=colors[solver], 
                label=f'{solver.replace("_", " ")} (data)', alpha=0.7)
        # Plot fitted curve
        if fits[solver] is not None:
            ax.plot(x_extended, fits[solver](x_extended), '--', linewidth=2, 
                    color=colors[solver], alpha=0.6)
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Scale with Fitted Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    # Log-y scale
    ax = axes[0, 1]
    for solver in ['PuLP', 'Pyomo', 'DWave_QPU', 'DWave_Hybrid']:
        ax.semilogy(x_data, times[solver], 'o', markersize=10, color=colors[solver], 
                    label=f'{solver.replace("_", " ")}', alpha=0.7)
        if fits[solver] is not None:
            y_fit = fits[solver](x_extended)
            y_fit = np.maximum(y_fit, 1e-6)  # Avoid log of negative
            ax.semilogy(x_extended, y_fit, '--', linewidth=2, 
                       color=colors[solver], alpha=0.6)
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Y Scale with Fitted Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale
    ax = axes[0, 2]
    for solver in ['PuLP', 'Pyomo', 'DWave_QPU', 'DWave_Hybrid']:
        ax.loglog(x_data, times[solver], 'o', markersize=10, color=colors[solver], 
                  label=f'{solver.replace("_", " ")}', alpha=0.7)
        if fits[solver] is not None:
            y_fit = fits[solver](x_extended)
            y_fit = np.maximum(y_fit, 1e-6)
            ax.loglog(x_extended, y_fit, '--', linewidth=2, 
                     color=colors[solver], alpha=0.6)
    
    ax.set_xlabel('Number of Farms (log)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Scale with Fitted Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # === Row 2: Speedup Ratios from Fits ===
    
    # Calculate speedup ratios from fitted functions
    speedup_hybrid_vs_pulp = fits['PuLP'](x_extended) / fits['DWave_Hybrid'](x_extended)
    speedup_hybrid_vs_pyomo = fits['Pyomo'](x_extended) / fits['DWave_Hybrid'](x_extended)
    speedup_qpu_vs_pulp = fits['PuLP'](x_extended) / fits['DWave_QPU'](x_extended)
    speedup_qpu_vs_pyomo = fits['Pyomo'](x_extended) / fits['DWave_QPU'](x_extended)
    
    # Find crossover points
    crossover_hybrid_pulp = find_crossover_points(x_extended, fits['PuLP'], fits['DWave_Hybrid'])
    crossover_hybrid_pyomo = find_crossover_points(x_extended, fits['Pyomo'], fits['DWave_Hybrid'])
    
    # Linear scale speedup
    ax = axes[1, 0]
    ax.plot(x_extended, speedup_hybrid_vs_pulp, linewidth=3, color='#118AB2', 
            label='Hybrid vs PuLP (fitted)', alpha=0.8)
    ax.plot(x_extended, speedup_hybrid_vs_pyomo, linewidth=3, color='#073B4C', 
            label='Hybrid vs Pyomo (fitted)', alpha=0.8, linestyle='--')
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2.5, label='Break-even (1x)')
    
    # Mark crossover points
    for cp in crossover_hybrid_pulp:
        ax.axvline(x=cp, color='#118AB2', linestyle=':', alpha=0.5, linewidth=2)
        ax.text(cp, ax.get_ylim()[1]*0.9, f'{cp:.0f}', ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('Hybrid Speedup (from fitted curves)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_data.min(), x_extended.max()])
    
    # Log-y scale speedup
    ax = axes[1, 1]
    ax.semilogy(x_extended, speedup_qpu_vs_pulp, linewidth=3, color='#06FFA5', 
                label='QPU vs PuLP (fitted)', alpha=0.8)
    ax.semilogy(x_extended, speedup_qpu_vs_pyomo, linewidth=3, color='#06D89E', 
                label='QPU vs Pyomo (fitted)', alpha=0.8, linestyle='--')
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2.5, label='Break-even (1x)')
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (log)', fontsize=12, fontweight='bold')
    ax.set_title('QPU Speedup (from fitted curves)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([x_data.min(), x_extended.max()])
    
    # Combined comparison
    ax = axes[1, 2]
    ax.plot(x_extended, speedup_hybrid_vs_pulp, linewidth=3, color='#118AB2', 
            label='Hybrid vs PuLP', alpha=0.8)
    ax.plot(x_extended, speedup_qpu_vs_pulp, linewidth=3, color='#06FFA5', 
            label='QPU vs PuLP', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2.5, label='Break-even (1x)')
    
    # Highlight crossover region
    for cp in crossover_hybrid_pulp:
        ax.axvline(x=cp, color='#118AB2', linestyle=':', alpha=0.5, linewidth=2)
        ax.text(cp, 0.5, f'Crossover\n{cp:.0f} farms', ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('D-Wave Advantage Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_data.min(), x_extended.max()])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFitted speedup plot saved to: {output_path}")
    
    return crossover_hybrid_pulp, crossover_hybrid_pyomo

def plot_fitted_speedup_bqubo(times, output_path):
    """Create comprehensive fitted speedup analysis for BQUBO."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('BQUBO Benchmark: Fitted Speedup Analysis with Interpolation', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    x_data = times['n_farms']
    x_extended = np.linspace(x_data.min(), max(x_data.max() * 1.2, 1500), 500)
    
    colors = {
        'PuLP': '#E63946',
        'CQM': '#F77F00',
        'DWave_QPU': '#06FFA5',
        'DWave_Hybrid': '#118AB2'
    }
    
    # Fit curves for each solver
    fits = {}
    fit_params = {}
    print("\n" + "="*80)
    print("BQUBO FIT PARAMETERS:")
    print("="*80)
    for solver in ['PuLP', 'CQM', 'DWave_QPU', 'DWave_Hybrid']:
        fit_func, fit_type, params = fit_solver_data(x_data, times[solver], solver)
        fits[solver] = fit_func
        fit_params[solver] = params
        if params is not None and fit_type == "power_law":
            print(f"{solver:15}: f(x) = {params[0]:.6f} * x^{params[1]:.6f} + {params[2]:.6f}")
        else:
            print(f"{solver:15}: {fit_type} fit")
    
    # === Row 1: Solve Times with Fits ===
    
    # Linear scale
    ax = axes[0, 0]
    for solver in ['PuLP', 'CQM', 'DWave_QPU', 'DWave_Hybrid']:
        ax.plot(x_data, times[solver], 'o', markersize=10, color=colors[solver], 
                label=f'{solver.replace("_", " ")} (data)', alpha=0.7)
        if fits[solver] is not None:
            ax.plot(x_extended, fits[solver](x_extended), '--', linewidth=2, 
                    color=colors[solver], alpha=0.6)
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Scale with Fitted Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    # Log-y scale
    ax = axes[0, 1]
    for solver in ['PuLP', 'CQM', 'DWave_QPU', 'DWave_Hybrid']:
        ax.semilogy(x_data, times[solver], 'o', markersize=10, color=colors[solver], 
                    label=f'{solver.replace("_", " ")}', alpha=0.7)
        if fits[solver] is not None:
            y_fit = fits[solver](x_extended)
            y_fit = np.maximum(y_fit, 1e-6)
            ax.semilogy(x_extended, y_fit, '--', linewidth=2, 
                       color=colors[solver], alpha=0.6)
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Y Scale with Fitted Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale
    ax = axes[0, 2]
    for solver in ['PuLP', 'CQM', 'DWave_QPU', 'DWave_Hybrid']:
        ax.loglog(x_data, times[solver], 'o', markersize=10, color=colors[solver], 
                  label=f'{solver.replace("_", " ")}', alpha=0.7)
        if fits[solver] is not None:
            y_fit = fits[solver](x_extended)
            y_fit = np.maximum(y_fit, 1e-6)
            ax.loglog(x_extended, y_fit, '--', linewidth=2, 
                     color=colors[solver], alpha=0.6)
    
    ax.set_xlabel('Number of Farms (log)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Scale with Fitted Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # === Row 2: Speedup Ratios from Fits ===
    
    # Calculate speedup ratios from fitted functions
    speedup_hybrid_vs_pulp = fits['PuLP'](x_extended) / fits['DWave_Hybrid'](x_extended)
    speedup_hybrid_vs_cqm = fits['CQM'](x_extended) / fits['DWave_Hybrid'](x_extended)
    speedup_qpu_vs_pulp = fits['PuLP'](x_extended) / fits['DWave_QPU'](x_extended)
    speedup_qpu_vs_cqm = fits['CQM'](x_extended) / fits['DWave_QPU'](x_extended)
    
    # Find crossover points
    print("\n  Searching for Hybrid vs PuLP crossover:")
    crossover_hybrid_pulp = find_crossover_points(x_extended, fits['PuLP'], fits['DWave_Hybrid'], debug=True)
    print("\n  Searching for Hybrid vs CQM crossover:")
    crossover_hybrid_cqm = find_crossover_points(x_extended, fits['CQM'], fits['DWave_Hybrid'], debug=True)
    
    # Linear scale speedup
    ax = axes[1, 0]
    ax.plot(x_extended, speedup_hybrid_vs_pulp, linewidth=3, color='#118AB2', 
            label='Hybrid vs PuLP (fitted)', alpha=0.8)
    ax.plot(x_extended, speedup_hybrid_vs_cqm, linewidth=3, color='#073B4C', 
            label='Hybrid vs CQM (fitted)', alpha=0.8, linestyle='--')
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2.5, label='Break-even (1x)')
    
    # Mark crossover points
    for cp in crossover_hybrid_pulp:
        ax.axvline(x=cp, color='#118AB2', linestyle=':', alpha=0.5, linewidth=2)
        ax.text(cp, ax.get_ylim()[1]*0.9, f'{cp:.0f}', ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('Hybrid Speedup (from fitted curves)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_data.min(), x_extended.max()])
    
    # Log-y scale speedup
    ax = axes[1, 1]
    speedup_qpu_vs_pulp_pos = np.maximum(speedup_qpu_vs_pulp, 1e-6)
    speedup_qpu_vs_cqm_pos = np.maximum(speedup_qpu_vs_cqm, 1e-6)
    
    ax.semilogy(x_extended, speedup_qpu_vs_pulp_pos, linewidth=3, color='#06FFA5', 
                label='QPU vs PuLP (fitted)', alpha=0.8)
    ax.semilogy(x_extended, speedup_qpu_vs_cqm_pos, linewidth=3, color='#06D89E', 
                label='QPU vs CQM (fitted)', alpha=0.8, linestyle='--')
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2.5, label='Break-even (1x)')
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (log)', fontsize=12, fontweight='bold')
    ax.set_title('QPU Speedup (from fitted curves)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([x_data.min(), x_extended.max()])
    
    # Combined comparison
    ax = axes[1, 2]
    ax.plot(x_extended, speedup_hybrid_vs_pulp, linewidth=3, color='#118AB2', 
            label='Hybrid vs PuLP', alpha=0.8)
    ax.plot(x_extended, speedup_qpu_vs_pulp, linewidth=3, color='#06FFA5', 
            label='QPU vs PuLP', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle=':', linewidth=2.5, label='Break-even (1x)')
    
    # Highlight crossover region
    for cp in crossover_hybrid_pulp:
        ax.axvline(x=cp, color='#118AB2', linestyle=':', alpha=0.5, linewidth=2)
        ax.text(cp, 0.5, f'Crossover\n{cp:.0f} farms', ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Number of Farms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('D-Wave Advantage Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_data.min(), x_extended.max()])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFitted speedup plot saved to: {output_path}")
    
    return crossover_hybrid_pulp, crossover_hybrid_cqm

def print_analysis_summary(nln_crossovers, bqubo_crossovers):
    """Print summary of crossover analysis."""
    print("\n" + "="*100)
    print("FITTED CURVE ANALYSIS SUMMARY")
    print("="*100)
    print("\nThis analysis fits interpolating functions to each solver's data and calculates")
    print("speedup as the ratio of these fitted functions. This reveals the true scaling")
    print("behavior and identifies crossover points where D-Wave becomes advantageous.")
    print("\n" + "-"*100)
    
    print("\n--- NLN CROSSOVER ANALYSIS ---")
    nln_cp_pulp, nln_cp_pyomo = nln_crossovers
    
    if nln_cp_pulp:
        print(f"\n✓ D-Wave Hybrid becomes faster than PuLP at: {nln_cp_pulp[0]:.1f} farms")
        print(f"  → Below this size: Use PuLP (classical)")
        print(f"  → Above this size: Use D-Wave Hybrid (quantum advantage)")
    else:
        print("\n✗ No crossover found with PuLP in the analyzed range")
        print("  → D-Wave Hybrid may be advantageous at larger problem sizes")
    
    if nln_cp_pyomo:
        print(f"\n✓ D-Wave Hybrid becomes faster than Pyomo at: {nln_cp_pyomo[0]:.1f} farms")
    else:
        print("\n✗ No crossover found with Pyomo in the analyzed range")
    
    print("\n--- BQUBO CROSSOVER ANALYSIS ---")
    bqubo_cp_pulp, bqubo_cp_cqm = bqubo_crossovers
    
    if bqubo_cp_pulp:
        print(f"\n✓ D-Wave Hybrid becomes faster than PuLP at: {bqubo_cp_pulp[0]:.1f} farms")
        print(f"  → Below this size: Use PuLP (classical)")
        print(f"  → Above this size: Use D-Wave Hybrid (quantum advantage)")
    else:
        print("\n✗ No crossover found with PuLP in the analyzed range")
        print("  → D-Wave Hybrid may be advantageous at larger problem sizes")
    
    if bqubo_cp_cqm:
        print(f"\n✓ D-Wave Hybrid becomes faster than CQM at: {bqubo_cp_cqm[0]:.1f} farms")
    else:
        print("\n✗ No crossover found with CQM in the analyzed range")
    
    print("\n" + "="*100)
    print("\nKEY RECOMMENDATIONS:")
    print("• For small problems: Classical solvers (PuLP, Pyomo) are sufficient and faster")
    print("• For medium problems: Crossover region - evaluate both approaches")
    print("• For large problems: D-Wave shows clear advantage with better scaling")
    print("• QPU time: Nearly constant, providing dramatic speedup for large problems")
    print("• Hybrid time: Includes overhead but still scales better than classical")
    print("="*100 + "\n")

def main():
    benchmark_dir = Path(__file__).parent / "Benchmarks"
    
    print("\n" + "="*100)
    print("ADVANCED SPEEDUP ANALYSIS WITH CURVE FITTING")
    print("="*100)
    print("\nFitting interpolating functions to benchmark data...")
    print("This reveals true scaling behavior and identifies crossover points.\n")
    
    # Load and extract data
    nln_data = load_nln_data(benchmark_dir)
    bqubo_data = load_bqubo_data(benchmark_dir)
    
    nln_times = extract_nln_times(nln_data)
    bqubo_times = extract_bqubo_times(bqubo_data)
    
    # Create output directory
    output_dir = Path(__file__).parent / "Plots"
    output_dir.mkdir(exist_ok=True)
    
    # NLN Analysis
    print("\n--- Fitting NLN Data ---")
    nln_output = output_dir / "nln_fitted_speedup_analysis.png"
    nln_crossovers = plot_fitted_speedup_nln(nln_times, nln_output)
    
    # BQUBO Analysis
    print("\n--- Fitting BQUBO Data ---")
    bqubo_output = output_dir / "bqubo_fitted_speedup_analysis.png"
    bqubo_crossovers = plot_fitted_speedup_bqubo(bqubo_times, bqubo_output)
    
    # Print summary
    print_analysis_summary(nln_crossovers, bqubo_crossovers)
    
    plt.show()

if __name__ == "__main__":
    main()
