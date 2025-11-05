"""
Unified Benchmark Visualization: Cross-Scenario Comparison
Creates 3 plots comparing all scenarios (BQUBO, LQ, NLN) for each solver type:
1. Gurobi MILP across all scenarios
2. Pyomo/Ipopt across all scenarios  
3. D-Wave (QPU + Hybrid) across all scenarios
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt conflicts
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from pathlib import Path

# =============================================================================
# PROFESSIONAL STYLE CONFIGURATION
# =============================================================================

def configure_professional_style():
    """Configure professional matplotlib style."""
    
    rcParams.update({
        # Font configuration
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'serif'],
        'font.size': 11,
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'serif',
        
        # Text rendering
        'text.usetex': False,
        
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
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
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

def load_all_benchmark_data(benchmark_dir):
    """Load benchmark data from all three scenarios: BQUBO, LQ, NLN."""
    
    configs = [5, 19, 72, 279]
    
    # Initialize data structure
    all_data = {
        'BQUBO': {'PuLP': {}, 'Pyomo': {}, 'DWave': {}},
        'LQ': {'PuLP': {}, 'Pyomo': {}, 'DWave': {}},
        'NLN': {'PuLP': {}, 'Pyomo': {}, 'DWave': {}}
    }
    
    # Load BQUBO data
    bqubo_dir = Path(benchmark_dir) / "BQUBO"
    for solver in ['PuLP', 'DWave']:
        solver_dir = bqubo_dir / solver
        if solver_dir.exists():
            for config in configs:
                config_file = solver_dir / f"config_{config}_run_1.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        all_data['BQUBO'][solver][config] = json.load(f)
    
    # Load LQ data
    lq_dir = Path(benchmark_dir) / "LQ"
    for solver in ['PuLP', 'Pyomo', 'DWave']:
        solver_dir = lq_dir / solver
        if solver_dir.exists():
            for config in configs:
                config_file = solver_dir / f"config_{config}_run_1.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        all_data['LQ'][solver][config] = json.load(f)
    
    # Load NLN data
    nln_dir = Path(benchmark_dir) / "NLN"
    for solver in ['PuLP', 'Pyomo', 'DWave']:
        solver_dir = nln_dir / solver
        if solver_dir.exists():
            for config in configs:
                config_file = solver_dir / f"config_{config}_run_1.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        all_data['NLN'][solver][config] = json.load(f)
    
    return all_data

def extract_times_by_scenario(all_data):
    """Extract solve times AND objectives organized by scenario."""
    
    configs = [5, 19, 72, 279]
    n_farms = np.array(configs) * 27
    
    times = {
        'n_farms': n_farms,
        'BQUBO': {'PuLP': [], 'Pyomo': [], 'DWave_QPU': [], 'DWave_Hybrid': []},
        'LQ': {'PuLP': [], 'Pyomo': [], 'DWave_QPU': [], 'DWave_Hybrid': []},
        'NLN': {'PuLP': [], 'Pyomo': [], 'DWave_QPU': [], 'DWave_Hybrid': []}
    }
    
    objectives = {
        'n_farms': n_farms,
        'BQUBO': {'PuLP': [], 'Pyomo': [], 'DWave': []},
        'LQ': {'PuLP': [], 'Pyomo': [], 'DWave': []},
        'NLN': {'PuLP': [], 'Pyomo': [], 'DWave': []}
    }
    
    for config in configs:
        # BQUBO times and objectives
        if config in all_data['BQUBO']['PuLP']:
            times['BQUBO']['PuLP'].append(all_data['BQUBO']['PuLP'][config]['result']['solve_time'])
            objectives['BQUBO']['PuLP'].append(all_data['BQUBO']['PuLP'][config]['result'].get('objective_value', None))
        else:
            times['BQUBO']['PuLP'].append(None)
            objectives['BQUBO']['PuLP'].append(None)
        
        times['BQUBO']['Pyomo'].append(None)  # BQUBO doesn't have Pyomo
        objectives['BQUBO']['Pyomo'].append(None)
        
        if config in all_data['BQUBO']['DWave']:
            times['BQUBO']['DWave_QPU'].append(all_data['BQUBO']['DWave'][config]['result']['qpu_time'])
            times['BQUBO']['DWave_Hybrid'].append(all_data['BQUBO']['DWave'][config]['result']['hybrid_time'])
            objectives['BQUBO']['DWave'].append(all_data['BQUBO']['DWave'][config]['result'].get('objective_value', None))
        else:
            times['BQUBO']['DWave_QPU'].append(None)
            times['BQUBO']['DWave_Hybrid'].append(None)
            objectives['BQUBO']['DWave'].append(None)
        
        # LQ times and objectives
        if config in all_data['LQ']['PuLP']:
            times['LQ']['PuLP'].append(all_data['LQ']['PuLP'][config]['result']['solve_time'])
            objectives['LQ']['PuLP'].append(all_data['LQ']['PuLP'][config]['result'].get('normalized_objective', None))
        else:
            times['LQ']['PuLP'].append(None)
            objectives['LQ']['PuLP'].append(None)
        
        if config in all_data['LQ']['Pyomo']:
            times['LQ']['Pyomo'].append(all_data['LQ']['Pyomo'][config]['result']['solve_time'])
            objectives['LQ']['Pyomo'].append(all_data['LQ']['Pyomo'][config]['result'].get('normalized_objective', None))
        else:
            times['LQ']['Pyomo'].append(None)
            objectives['LQ']['Pyomo'].append(None)
        
        if config in all_data['LQ']['DWave']:
            times['LQ']['DWave_QPU'].append(all_data['LQ']['DWave'][config]['result']['qpu_time'])
            times['LQ']['DWave_Hybrid'].append(all_data['LQ']['DWave'][config]['result']['hybrid_time'])
            objectives['LQ']['DWave'].append(all_data['LQ']['DWave'][config]['result'].get('normalized_objective', None))
        else:
            times['LQ']['DWave_QPU'].append(None)
            times['LQ']['DWave_Hybrid'].append(None)
            objectives['LQ']['DWave'].append(None)
        
        # NLN times and objectives
        if config in all_data['NLN']['PuLP']:
            times['NLN']['PuLP'].append(all_data['NLN']['PuLP'][config]['result']['solve_time'])
            objectives['NLN']['PuLP'].append(all_data['NLN']['PuLP'][config]['result'].get('objective_value', None))
        else:
            times['NLN']['PuLP'].append(None)
            objectives['NLN']['PuLP'].append(None)
        
        if config in all_data['NLN']['Pyomo']:
            times['NLN']['Pyomo'].append(all_data['NLN']['Pyomo'][config]['result']['solve_time'])
            objectives['NLN']['Pyomo'].append(all_data['NLN']['Pyomo'][config]['result'].get('objective_value', None))
        else:
            times['NLN']['Pyomo'].append(None)
            objectives['NLN']['Pyomo'].append(None)
        
        if config in all_data['NLN']['DWave']:
            times['NLN']['DWave_QPU'].append(all_data['NLN']['DWave'][config]['result']['qpu_time'])
            times['NLN']['DWave_Hybrid'].append(all_data['NLN']['DWave'][config]['result']['hybrid_time'])
            objectives['NLN']['DWave'].append(all_data['NLN']['DWave'][config]['result'].get('objective_value', None))
        else:
            times['NLN']['DWave_QPU'].append(None)
            times['NLN']['DWave_Hybrid'].append(None)
            objectives['NLN']['DWave'].append(None)
    
    return times, objectives

def calculate_objective_gaps_by_scenario(objectives):
    """Calculate objective value gaps for each scenario."""
    n_farms = objectives['n_farms']
    
    gaps = {
        'n_farms': n_farms,
        'BQUBO': {'PuLP_gap': [], 'Pyomo_gap': [], 'DWave_gap': []},
        'LQ': {'PuLP_gap': [], 'Pyomo_gap': [], 'DWave_gap': []},
        'NLN': {'PuLP_gap': [], 'Pyomo_gap': [], 'DWave_gap': []}
    }
    
    for scenario in ['BQUBO', 'LQ', 'NLN']:
        for i in range(len(n_farms)):
            # Find best (maximum) objective for this config in this scenario
            values = []
            if objectives[scenario]['PuLP'][i] is not None:
                values.append(objectives[scenario]['PuLP'][i])
            if objectives[scenario]['Pyomo'][i] is not None:
                values.append(objectives[scenario]['Pyomo'][i])
            if objectives[scenario]['DWave'][i] is not None:
                values.append(objectives[scenario]['DWave'][i])
            
            if not values:
                gaps[scenario]['PuLP_gap'].append(None)
                gaps[scenario]['Pyomo_gap'].append(None)
                gaps[scenario]['DWave_gap'].append(None)
                continue
            
            best_obj = max(values)  # Assuming maximization
            
            # Calculate gaps as percentage
            if objectives[scenario]['PuLP'][i] is not None:
                gap = ((best_obj - objectives[scenario]['PuLP'][i]) / best_obj) * 100 if best_obj != 0 else 0
                gaps[scenario]['PuLP_gap'].append(gap)
            else:
                gaps[scenario]['PuLP_gap'].append(None)
            
            if objectives[scenario]['Pyomo'][i] is not None:
                gap = ((best_obj - objectives[scenario]['Pyomo'][i]) / best_obj) * 100 if best_obj != 0 else 0
                gaps[scenario]['Pyomo_gap'].append(gap)
            else:
                gaps[scenario]['Pyomo_gap'].append(None)
            
            if objectives[scenario]['DWave'][i] is not None:
                gap = ((best_obj - objectives[scenario]['DWave'][i]) / best_obj) * 100 if best_obj != 0 else 0
                gaps[scenario]['DWave_gap'].append(gap)
            else:
                gaps[scenario]['DWave_gap'].append(None)
    
    return gaps

def plot_cross_scenario_comparison(times, objectives, gaps, output_dir):
    """Create 3 plots with 2x3 layout: one per solver type, comparing all scenarios."""
    
    n_farms = times['n_farms']
    
    # Scenario colors
    scenario_colors = {
        'BQUBO': '#E63946',   # Red
        'LQ': '#F77F00',      # Orange
        'NLN': '#06A77D'      # Green
    }
    
    scenario_markers = {
        'BQUBO': 'o',
        'LQ': 's',
        'NLN': '^'
    }
    
    # =============================================================================
    # PLOT 1: Gurobi MILP (PuLP) across all scenarios - 2x3 layout
    # =============================================================================
    fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig1.suptitle('Gurobi MILP Solver: Cross-Scenario Performance Comparison', 
                  fontsize=18, y=0.995, fontweight='bold')
    
    # Row 1: Solve Times
    # Linear scale
    ax = axes[0, 0]
    for scenario in ['BQUBO', 'LQ', 'NLN']:
        ax.plot(n_farms, times[scenario]['PuLP'], 
                marker=scenario_markers[scenario], linewidth=2.5, markersize=10,
                color=scenario_colors[scenario], label=scenario, alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale
    ax = axes[0, 1]
    for scenario in ['BQUBO', 'LQ', 'NLN']:
        ax.semilogy(n_farms, times[scenario]['PuLP'], 
                    marker=scenario_markers[scenario], linewidth=2.5, markersize=10,
                    color=scenario_colors[scenario], label=scenario, alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Y Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale
    ax = axes[0, 2]
    for scenario in ['BQUBO', 'LQ', 'NLN']:
        ax.loglog(n_farms, times[scenario]['PuLP'], 
                  marker=scenario_markers[scenario], linewidth=2.5, markersize=10,
                  color=scenario_colors[scenario], label=scenario, alpha=0.8)
    ax.set_xlabel('Problem Size (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # === Row 2: Solution Quality ===
    # Objective Values
    ax = axes[1, 0]
    for scenario in ['BQUBO', 'LQ', 'NLN']:
        ax.plot(n_farms, objectives[scenario]['PuLP'], 
                marker=scenario_markers[scenario], linewidth=2.5, markersize=10,
                color=scenario_colors[scenario], label=scenario, alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Value', fontsize=11, fontweight='bold')
    ax.set_title('Objective Values (Higher = Better)', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Objective Gaps (%)
    ax = axes[1, 1]
    for scenario in ['BQUBO', 'LQ', 'NLN']:
        ax.plot(n_farms, gaps[scenario]['PuLP_gap'], 
                marker=scenario_markers[scenario], linewidth=2.5, markersize=10,
                color=scenario_colors[scenario], label=scenario, alpha=0.8)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Optimal (0% gap)')
    ax.set_xlabel('Problem Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Gap (%)', fontsize=11, fontweight='bold')
    ax.set_title('Solution Quality Gap (% from Best)', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Objective Gaps (Bar Chart)
    ax = axes[1, 2]
    x = np.arange(len(n_farms))
    width = 0.25
    for i, scenario in enumerate(['BQUBO', 'LQ', 'NLN']):
        gaps_array = [g if g is not None else 0 for g in gaps[scenario]['PuLP_gap']]
        ax.bar(x + i*width - width, gaps_array, width, 
               label=scenario, color=scenario_colors[scenario], alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Gap (%)', fontsize=11, fontweight='bold')
    ax.set_title('Quality Gap Comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([int(n) for n in n_farms])
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path1 = output_dir / "gurobi_milp_cross_scenario.png"
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"✓ Gurobi MILP comparison saved to: {output_path1}")
    plt.close()
    
    # =============================================================================
    # PLOT 2: Pyomo/Ipopt (MINLP) across LQ and NLN scenarios - 2x3 layout
    # =============================================================================
    fig2, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig2.suptitle('Ipopt MINLP Solver: Cross-Scenario Performance Comparison', 
                  fontsize=18, y=0.995, fontweight='bold')
    
    # Row 1: Solve Times
    # Linear scale
    ax = axes[0, 0]
    for scenario in ['LQ', 'NLN']:
        ax.plot(n_farms, times[scenario]['Pyomo'], 
                marker=scenario_markers[scenario], linewidth=2.5, markersize=10,
                color=scenario_colors[scenario], label=scenario, alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, 'Note: BQUBO not applicable to MINLP',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Log-y scale
    ax = axes[0, 1]
    for scenario in ['LQ', 'NLN']:
        ax.semilogy(n_farms, times[scenario]['Pyomo'], 
                    marker=scenario_markers[scenario], linewidth=2.5, markersize=10,
                    color=scenario_colors[scenario], label=scenario, alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Y Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale
    ax = axes[0, 2]
    for scenario in ['LQ', 'NLN']:
        ax.loglog(n_farms, times[scenario]['Pyomo'], 
                  marker=scenario_markers[scenario], linewidth=2.5, markersize=10,
                  color=scenario_colors[scenario], label=scenario, alpha=0.8)
    ax.set_xlabel('Problem Size (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # === Row 2: Solution Quality ===
    # Objective Values
    ax = axes[1, 0]
    for scenario in ['LQ', 'NLN']:
        ax.plot(n_farms, objectives[scenario]['Pyomo'], 
                marker=scenario_markers[scenario], linewidth=2.5, markersize=10,
                color=scenario_colors[scenario], label=scenario, alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Value', fontsize=11, fontweight='bold')
    ax.set_title('Objective Values (Higher = Better)', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, 'Note: BQUBO not applicable',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Objective Gaps (%)
    ax = axes[1, 1]
    for scenario in ['LQ', 'NLN']:
        ax.plot(n_farms, gaps[scenario]['Pyomo_gap'], 
                marker=scenario_markers[scenario], linewidth=2.5, markersize=10,
                color=scenario_colors[scenario], label=scenario, alpha=0.8)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Optimal (0% gap)')
    ax.set_xlabel('Problem Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Gap (%)', fontsize=11, fontweight='bold')
    ax.set_title('Solution Quality Gap (% from Best)', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Objective Gaps (Bar Chart)
    ax = axes[1, 2]
    x = np.arange(len(n_farms))
    width = 0.35
    for i, scenario in enumerate(['LQ', 'NLN']):
        gaps_array = [g if g is not None else 0 for g in gaps[scenario]['Pyomo_gap']]
        ax.bar(x + i*width - width/2, gaps_array, width, 
               label=scenario, color=scenario_colors[scenario], alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Gap (%)', fontsize=11, fontweight='bold')
    ax.set_title('Quality Gap Comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([int(n) for n in n_farms])
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path2 = output_dir / "ipopt_minlp_cross_scenario.png"
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Ipopt MINLP comparison saved to: {output_path2}")
    plt.close()
    
    # =============================================================================
    # PLOT 3: D-Wave (QPU + Hybrid) across all scenarios - 2x3 layout with 6 traces
    # =============================================================================
    fig3, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig3.suptitle('D-Wave Hybrid Solvers: Cross-Scenario Performance Comparison', 
                  fontsize=18, y=0.995, fontweight='bold')
    
    # Create linestyles to differentiate QPU vs Hybrid
    linestyles = {'QPU': '-', 'Hybrid': '--'}
    
    # Row 1: Solve Times (QPU + Hybrid on same plots)
    # Linear scale
    ax = axes[0, 0]
    for scenario in ['BQUBO', 'LQ', 'NLN']:
        ax.plot(n_farms, times[scenario]['DWave_QPU'], 
                marker=scenario_markers[scenario], linestyle=linestyles['QPU'],
                linewidth=2.5, markersize=10,
                color=scenario_colors[scenario], label=f'{scenario} QPU', alpha=0.8)
        ax.plot(n_farms, times[scenario]['DWave_Hybrid'], 
                marker=scenario_markers[scenario], linestyle=linestyles['Hybrid'],
                linewidth=2.5, markersize=10,
                color=scenario_colors[scenario], label=f'{scenario} Hybrid', alpha=0.6)
    ax.set_xlabel('Problem Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Log-y scale
    ax = axes[0, 1]
    for scenario in ['BQUBO', 'LQ', 'NLN']:
        ax.semilogy(n_farms, times[scenario]['DWave_QPU'], 
                    marker=scenario_markers[scenario], linestyle=linestyles['QPU'],
                    linewidth=2.5, markersize=10,
                    color=scenario_colors[scenario], label=f'{scenario} QPU', alpha=0.8)
        ax.semilogy(n_farms, times[scenario]['DWave_Hybrid'], 
                    marker=scenario_markers[scenario], linestyle=linestyles['Hybrid'],
                    linewidth=2.5, markersize=10,
                    color=scenario_colors[scenario], label=f'{scenario} Hybrid', alpha=0.6)
    ax.set_xlabel('Problem Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Y Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    
    # Log-log scale
    ax = axes[0, 2]
    for scenario in ['BQUBO', 'LQ', 'NLN']:
        ax.loglog(n_farms, times[scenario]['DWave_QPU'], 
                  marker=scenario_markers[scenario], linestyle=linestyles['QPU'],
                  linewidth=2.5, markersize=10,
                  color=scenario_colors[scenario], label=f'{scenario} QPU', alpha=0.8)
        ax.loglog(n_farms, times[scenario]['DWave_Hybrid'], 
                  marker=scenario_markers[scenario], linestyle=linestyles['Hybrid'],
                  linewidth=2.5, markersize=10,
                  color=scenario_colors[scenario], label=f'{scenario} Hybrid', alpha=0.6)
    ax.set_xlabel('Problem Size (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Scale', fontsize=14)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    
    # === Row 2: Solution Quality ===
    # Objective Values
    ax = axes[1, 0]
    for scenario in ['BQUBO', 'LQ', 'NLN']:
        ax.plot(n_farms, objectives[scenario]['DWave'], 
                marker=scenario_markers[scenario], linewidth=2.5, markersize=10,
                color=scenario_colors[scenario], label=scenario, alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Value', fontsize=11, fontweight='bold')
    ax.set_title('Objective Values (Higher = Better)', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Objective Gaps (%)
    ax = axes[1, 1]
    for scenario in ['BQUBO', 'LQ', 'NLN']:
        ax.plot(n_farms, gaps[scenario]['DWave_gap'], 
                marker=scenario_markers[scenario], linewidth=2.5, markersize=10,
                color=scenario_colors[scenario], label=scenario, alpha=0.8)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Optimal (0% gap)')
    ax.set_xlabel('Problem Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Gap (%)', fontsize=11, fontweight='bold')
    ax.set_title('Solution Quality Gap (% from Best)', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Objective Gaps (Bar Chart)
    ax = axes[1, 2]
    x = np.arange(len(n_farms))
    width = 0.25
    for i, scenario in enumerate(['BQUBO', 'LQ', 'NLN']):
        gaps_array = [g if g is not None else 0 for g in gaps[scenario]['DWave_gap']]
        ax.bar(x + i*width - width, gaps_array, width, 
               label=scenario, color=scenario_colors[scenario], alpha=0.8)
    ax.set_xlabel('Problem Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Objective Gap (%)', fontsize=11, fontweight='bold')
    ax.set_title('Quality Gap Comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([int(n) for n in n_farms])
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path3 = output_dir / "dwave_cross_scenario.png"
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"✓ D-Wave comparison saved to: {output_path3}")
    plt.close()

def print_summary_table(times):
    """Print comprehensive summary table."""
    
    print("\n" + "="*150)
    print("CROSS-SCENARIO BENCHMARK SUMMARY: SOLVE TIMES COMPARISON")
    print("="*150)
    print(f"{'Problem':<10} {'Scenario':<10} {'Gurobi (s)':<15} {'Ipopt (s)':<15} {'D-Wave QPU (s)':<18} {'D-Wave Hybrid (s)':<20}")
    print("-"*150)
    
    n_farms = times['n_farms']
    
    for i, n in enumerate(n_farms):
        for j, scenario in enumerate(['BQUBO', 'LQ', 'NLN']):
            problem_label = f"{int(n)}" if j == 0 else ""
            
            pulp_time = f"{times[scenario]['PuLP'][i]:.4f}" if times[scenario]['PuLP'][i] else "N/A"
            pyomo_time = f"{times[scenario]['Pyomo'][i]:.4f}" if times[scenario]['Pyomo'][i] else "N/A"
            qpu_time = f"{times[scenario]['DWave_QPU'][i]:.6f}" if times[scenario]['DWave_QPU'][i] else "N/A"
            hybrid_time = f"{times[scenario]['DWave_Hybrid'][i]:.4f}" if times[scenario]['DWave_Hybrid'][i] else "N/A"
            
            print(f"{problem_label:<10} {scenario:<10} {pulp_time:<15} {pyomo_time:<15} {qpu_time:<18} {hybrid_time:<20}")
        
        if i < len(n_farms) - 1:
            print("-"*150)
    
    print("="*150)
    print("\nKey Insights:")
    print("• BQUBO: Pure binary quadratic formulation - most suitable for QPU")
    print("• LQ: Linear objective + Quadratic synergy terms - hybrid approach effective")
    print("• NLN: Fully nonlinear formulation - challenges for all solvers")
    print("\n• Gurobi MILP: Consistent performance across BQUBO/LQ/NLN")
    print("• Ipopt MINLP: Better for LQ than NLN (nonlinearity complexity)")
    print("• D-Wave QPU: Nearly constant time regardless of problem size or formulation")
    print("• D-Wave Hybrid: Balances QPU speed with classical optimization overhead")
    print("="*150 + "\n")

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("CROSS-SCENARIO BENCHMARK ANALYSIS")
    print("Comparing BQUBO, LQ, and NLN formulations across all solvers")
    print("="*80 + "\n")
    
    # Load all benchmark data
    benchmark_dir = Path(__file__).parent / "Benchmarks"
    print("Loading benchmark data from all scenarios...")
    all_data = load_all_benchmark_data(benchmark_dir)
    
    # Extract times
    print("Extracting solve times...")
    times, objectives = extract_times_by_scenario(all_data)
    
    # Calculate quality metrics
    print("Calculating objective gaps...")
    gaps = calculate_objective_gaps_by_scenario(objectives)
    
    # Print summary table
    print_summary_table(times)
    
    # Create plots
    output_dir = Path(__file__).parent / "Plots"
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating cross-scenario comparison plots...")
    plot_cross_scenario_comparison(times, objectives, gaps, output_dir)
    
    print("\n" + "="*80)
    print("✓ All cross-scenario comparison plots generated successfully!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
