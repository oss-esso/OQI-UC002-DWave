"""
Scalability Benchmark Script
Tests different combinations of farms and food groups to analyze solver performance
"""
import os
import sys
import json
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from Utils.farm_sampler import generate_farms
from src.scenarios import load_food_data
from solver_runner_NLD import (
    create_cqm, 
    solve_with_pulp, 
    solve_with_pyomo, 
    solve_with_dwave,
    solve_with_dwave_charnes_cooper
)
from Utils.benchmark_cache import BenchmarkCache, serialize_cqm
import pulp as pl

# Benchmark configurations
# Format: number of farms to test with full_family scenario
# 6 points logarithmically scaled from 5 to 1535 farms
# Reduced from 40 points for faster testing with multiple runs
BENCHMARK_CONFIGS = [
    5, 19, 72, 279, 1096, 1535
]

# Number of runs per configuration for statistical analysis
NUM_RUNS = 5

def load_full_family_with_n_farms(n_farms, seed=42):
    """
    Load full_family scenario with specified number of farms.
    Uses the same logic as the scaling analysis.
    """
    import pandas as pd
    
    # Generate farms
    L = generate_farms(n_farms=n_farms, seed=seed)
    farms = list(L.keys())
    
    # Load food data from Excel or use fallback
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, "Inputs", "Combined_Food_Data.xlsx")
    
    if not os.path.exists(excel_path):
        print("Excel file not found, using fallback foods")
        # Fallback: use simple food data
        foods = {
            'Wheat': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.3, 
                      'affordability': 0.8, 'sustainability': 0.7},
            'Corn': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.4, 
                     'affordability': 0.9, 'sustainability': 0.6},
            'Rice': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.6, 
                     'affordability': 0.7, 'sustainability': 0.5},
            'Soybeans': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.2, 
                         'affordability': 0.6, 'sustainability': 0.8},
            'Potatoes': {'nutritional_value': 0.5, 'nutrient_density': 0.4, 'environmental_impact': 0.3, 
                         'affordability': 0.9, 'sustainability': 0.7},
            'Apples': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.2, 
                       'affordability': 0.5, 'sustainability': 0.8},
            'Tomatoes': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.2, 
                         'affordability': 0.7, 'sustainability': 0.9},
            'Carrots': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.2, 
                        'affordability': 0.8, 'sustainability': 0.8},
            'Lentils': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.2, 
                        'affordability': 0.7, 'sustainability': 0.8},
            'Spinach': {'nutritional_value': 0.8, 'nutrient_density': 0.9, 'environmental_impact': 0.1, 
                        'affordability': 0.6, 'sustainability': 0.9},
        }
        food_groups = {
            'Grains': ['Wheat', 'Corn', 'Rice'],
            'Legumes': ['Soybeans', 'Lentils'],
            'Vegetables': ['Potatoes', 'Tomatoes', 'Carrots', 'Spinach'],
            'Fruits': ['Apples'],
        }
    else:
        # Load from Excel - USE ALL FOODS, not just 2 per group
        df = pd.read_excel(excel_path)
        
        # Use ALL foods from the dataset
        foods_list = df['Food_Name'].tolist()
        
        filt = df[df['Food_Name'].isin(foods_list)][['Food_Name', 'food_group',
                                                       'nutritional_value', 'nutrient_density',
                                                       'environmental_impact', 'affordability',
                                                       'sustainability']].copy()
        filt.rename(columns={'Food_Name': 'Food_Name', 'food_group': 'Food_Group'}, inplace=True)
        
        objectives = ['nutritional_value', 'nutrient_density', 'environmental_impact', 'affordability', 'sustainability']
        for obj in objectives:
            filt[obj] = filt[obj].fillna(0.5).clip(0, 1)
        
        # Build foods dict
        foods = {}
        for _, row in filt.iterrows():
            fname = row['Food_Name']
            foods[fname] = {
                'nutritional_value': float(row['nutritional_value']),
                'nutrient_density': float(row['nutrient_density']),
                'environmental_impact': float(row['environmental_impact']),
                'affordability': float(row['affordability']),
                'sustainability': float(row['sustainability'])
            }
        
        # Build food groups
        food_groups = {}
        for _, row in filt.iterrows():
            g = row['Food_Group']
            fname = row['Food_Name']
            if g not in food_groups:
                food_groups[g] = []
            food_groups[g].append(fname)
    
    # Set minimum planting areas based on smallest farm and number of food groups
    smallest_farm = min(L.values())
    n_food_groups = len(food_groups)
    # Each farm must plant at least 1 crop from each food group
    # Reserve some margin for safety
    min_area_per_crop = (smallest_farm / n_food_groups) * 0.9  # 90% safety margin
    
    min_areas = {food: min_area_per_crop for food in foods.keys()}
    
    # Build config
    parameters = {
        'land_availability': L,
        'minimum_planting_area': min_areas,
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': {
            g: {'min_foods': 1, 'max_foods': len(lst)}  # At least 1 food per group
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }
    
    config = {'parameters': parameters}
    
    return farms, foods, food_groups, config

def run_benchmark(n_farms, run_number=1, total_runs=1):
    """
    Run a single benchmark test with full_family scenario.
    Returns timing results and problem size metrics for all three solvers.
    
    Args:
        n_farms: Number of farms to test
        run_number: Current run number (for display)
        total_runs: Total number of runs (for display)
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK: full_family scenario with {n_farms} Farms (Run {run_number}/{total_runs})")
    print(f"{'='*80}")
    
    try:
        # Load full_family scenario with specified number of farms
        farms, foods, food_groups, config = load_full_family_with_n_farms(n_farms, seed=42 + run_number)
        
        n_foods = len(foods)
        n_vars = 2 * n_farms * n_foods  # Binary + continuous
        n_constraints = n_farms + 2*n_farms*n_foods + 2*len(food_groups)*n_farms
        problem_size = 5 * n_foods * n_farms  # Custom metric
        
        print(f"  Foods: {n_foods}")
        print(f"  Variables: {n_vars}")
        print(f"  Constraints: {n_constraints}")
        print(f"  Problem Size: {problem_size}")
        
        # Solve with PuLP (Dinkelbach)
        print(f"\n  Solving with PuLP (Dinkelbach)...")
        pulp_start = time.time()
        pulp_model, pulp_results = solve_with_pulp(farms, foods, food_groups, config)
        pulp_time = time.time() - pulp_start
        
        print(f"    Status: {pulp_results['status']}")
        print(f"    Iterations: {pulp_results.get('iterations', 'N/A')}")
        print(f"    Time: {pulp_time:.3f}s")
        
        # Solve with Pyomo (direct MINLP)
        pyomo_time = None
        pyomo_status = 'Not Run'
        pyomo_objective = None
        
        print(f"\n  Solving with Pyomo (MINLP)...")
        try:
            pyomo_start = time.time()
            pyomo_model, pyomo_results = solve_with_pyomo(farms, foods, food_groups, config)
            pyomo_time = time.time() - pyomo_start
            pyomo_status = pyomo_results.get('status', 'Failed')
            pyomo_objective = pyomo_results.get('objective_value')
            
            print(f"    Status: {pyomo_status}")
            if pyomo_status == 'Optimal':
                print(f"    Solver: {pyomo_results.get('solver', 'N/A')}")
                print(f"    Objective: {pyomo_objective:.6f}")
            print(f"    Time: {pyomo_time:.3f}s")
        except Exception as e:
            print(f"    ERROR: {e}")
            pyomo_time = 0
        
        # Solve with DWave (always attempt, diagnose failures)
        dwave_time = None
        qpu_time = None
        hybrid_time = None
        dwave_feasible = False
        
        print(f"\n  Solving with DWave...")
        
        
        try:
            # Create CQM
            print(f"    Creating CQM model...")
            cqm, A, Y, constraint_metadata = create_cqm(farms, foods, food_groups, config)
            print(f"    ✅ CQM created: {len(cqm.variables)} vars, {len(cqm.constraints)} constraints")
            
            # Solve
            #token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
            token = None
            print(f"    Submitting to D-Wave Leap...")
            sampleset, dwave_time = solve_with_dwave(cqm, token)
            
            # Extract timing info
            if hasattr(sampleset, 'info') and isinstance(sampleset.info, dict):
                qpu_time = sampleset.info.get('qpu_access_time', 0) / 1000000.0  # microseconds to seconds
                hybrid_time = sampleset.info.get('run_time', 0) / 1000000.0  # microseconds to seconds
            
            feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
            dwave_feasible = len(feasible_sampleset) > 0
            
            print(f"    ✅ Time: {dwave_time:.3f}s")
            print(f"    QPU Time: {qpu_time:.6f}s" if qpu_time else "    QPU Time: N/A")
            print(f"    Hybrid Time: {hybrid_time:.3f}s" if hybrid_time else "    Hybrid Time: N/A")
            print(f"    Feasible: {dwave_feasible}")
            print(f"    Total samples: {len(sampleset)}")
            print(f"    Feasible samples: {len(feasible_sampleset)}")
        except Exception as e:
            print(f"\n    ❌ DWAVE FAILED")
            print(f"    Error: {e}")
            print(f"    Error type: {type(e).__name__}")
            
        
        result = {
            'n_farms': n_farms,
            'n_foods': n_foods,
            'n_vars': n_vars,
            'n_constraints': n_constraints,
            'problem_size': n_farms * n_foods,  # Simplified: farms * foods
            'pulp_time': pulp_time,
            'pulp_status': pulp_results['status'],
            'pulp_objective': pulp_results.get('objective_value'),
            'pulp_iterations': pulp_results.get('iterations'),
            'pyomo_time': pyomo_time,
            'pyomo_status': pyomo_status,
            'pyomo_objective': pyomo_objective,
            'dwave_time': dwave_time,
            'qpu_time': qpu_time,
            'hybrid_time': hybrid_time,
            'dwave_feasible': dwave_feasible
        }
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(results, output_file='scalability_benchmark_nld.png'):
    """
    Create beautiful plots for presentation with error bars.
    Results should be aggregated statistics with mean and std.
    """
    # Filter valid results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid results to plot!")
        return
    
    # Extract data
    problem_sizes = [r['problem_size'] for r in valid_results]
    
    # PuLP times with error bars
    pulp_times = [r['pulp_time_mean'] for r in valid_results if r['pulp_time_mean'] is not None]
    pulp_errors = [r['pulp_time_std'] for r in valid_results if r['pulp_time_mean'] is not None]
    pulp_problem_sizes = [r['problem_size'] for r in valid_results if r['pulp_time_mean'] is not None]
    
    # Pyomo times with error bars
    pyomo_times = [r['pyomo_time_mean'] for r in valid_results if r['pyomo_time_mean'] is not None]
    pyomo_errors = [r['pyomo_time_std'] for r in valid_results if r['pyomo_time_mean'] is not None]
    pyomo_problem_sizes = [r['problem_size'] for r in valid_results if r['pyomo_time_mean'] is not None]
    
    # DWave times (only where available)
    dwave_times = [r['dwave_time_mean'] for r in valid_results if r['dwave_time_mean'] is not None]
    dwave_errors = [r['dwave_time_std'] for r in valid_results if r['dwave_time_mean'] is not None]
    dwave_problem_sizes = [r['problem_size'] for r in valid_results if r['dwave_time_mean'] is not None]
    
    # Create figure with professional styling
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: All times on log scale with error bars
    ax1.errorbar(pulp_problem_sizes, pulp_times, yerr=pulp_errors, marker='o', linestyle='-',
                linewidth=2.5, markersize=8, capsize=5, capthick=2,
                label='PuLP (Dinkelbach)', color='#2E86AB', alpha=0.9)
    
    if pyomo_times:
        ax1.errorbar(pyomo_problem_sizes, pyomo_times, yerr=pyomo_errors, marker='s', linestyle='-',
                    linewidth=2.5, markersize=8, capsize=5, capthick=2,
                    label='Pyomo (Ipopt MINLP)', color='#6A4C93', alpha=0.9)
    
    if dwave_times:
        ax1.errorbar(dwave_problem_sizes, dwave_times, yerr=dwave_errors, marker='^', linestyle='-',
                    linewidth=2.5, markersize=8, capsize=5, capthick=2,
                    label='D-Wave (Charnes-Cooper)', color='#A23B72', alpha=0.9)
    
    ax1.set_xlabel('Problem Size (Farms × Foods)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Solve Time (seconds)', fontsize=14, fontweight='bold')
    ax1.set_title('MINLP Fractional Programming: Solver Performance', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Add annotations for key points
    if len(pulp_problem_sizes) > 0:
        # Annotate largest problem
        max_idx = pulp_problem_sizes.index(max(pulp_problem_sizes))
        ax1.annotate(f'{pulp_problem_sizes[max_idx]} vars\n{pulp_times[max_idx]:.2f}s',
                    xy=(pulp_problem_sizes[max_idx], pulp_times[max_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Speedup ratio (Classical vs Quantum)
    if dwave_times and len(dwave_times) > 0:
        speedup_ratios = [pulp_times[pulp_problem_sizes.index(ps)] / dt 
                         for ps, dt in zip(dwave_problem_sizes, dwave_times)]
        
        ax2.plot(dwave_problem_sizes, speedup_ratios, 'o-', linewidth=2.5, markersize=8,
                color='#6A4C93', alpha=0.9)
        ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Parity')
        
        ax2.set_xlabel('Problem Size (Farms × Foods)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Speedup Ratio (PuLP / D-Wave)', fontsize=14, fontweight='bold')
        ax2.set_title('Classical vs Quantum Speedup', fontsize=16, fontweight='bold', pad=20)
        ax2.legend(loc='best', fontsize=11, framealpha=0.95)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xscale('log')
        
        # Add annotation for quantum advantage threshold
        ax2.fill_between(dwave_problem_sizes, 0, 1, alpha=0.2, color='red', label='Quantum Advantage Zone')
        ax2.text(min(dwave_problem_sizes), 0.5, 'Quantum\nFaster', fontsize=10, ha='left', va='center',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        ax2.text(min(dwave_problem_sizes), 1.5, 'Classical\nFaster', fontsize=10, ha='left', va='center',
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    else:
        ax2.text(0.5, 0.5, 'No Quantum Data Available\n(Problems too large)', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")
    
    # Also create a summary table plot
    create_summary_table(valid_results, 'scalability_table_nld.png')

def create_summary_table(results, output_file='scalability_table_nld.png'):
    """
    Create a beautiful summary table.
    """
    fig, ax = plt.subplots(figsize=(14, len(results) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Farms', 'Foods', 'n', 'Vars', 'Constraints', 
               'PuLP\nTime (s)', 'D-Wave\nTime (s)', 'QPU\nTime (s)', 'Hybrid\nTime (s)', 'Winner']
    
    table_data = []
    for r in results:
        # Determine winner
        if r['dwave_time'] is not None and r['pulp_time'] is not None:
            if r['dwave_time'] < r['pulp_time']:
                winner = '🏆 D-Wave'
            elif r['pulp_time'] < r['dwave_time']:
                winner = '🏆 PuLP'
            else:
                winner = 'Tie'
        else:
            winner = 'N/A'
        
        row = [
            r['n_farms'],
            r['n_foods'],
            r['problem_size'],
            r['n_vars'],
            r['n_constraints'],
            f"{r['pulp_time']:.3f}",
            f"{r['dwave_time']:.3f}" if r['dwave_time'] else 'N/A',
            f"{r['qpu_time']:.6f}" if r['qpu_time'] else 'N/A',
            f"{r['hybrid_time']:.3f}" if r['hybrid_time'] else 'N/A',
            winner
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F4F8')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Scalability Benchmark Results', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Table saved to: {output_file}")

def main():
    """
    Run all benchmarks with multiple runs and calculate statistics.
    """
    print("="*80)
    print("MINLP FRACTIONAL SCALABILITY BENCHMARK")
    print("="*80)
    print(f"Configurations: {len(BENCHMARK_CONFIGS)} points")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(f"Total benchmarks: {len(BENCHMARK_CONFIGS) * NUM_RUNS}")
    print("="*80)
    
    all_results = []
    aggregated_results = []
    
    for n_farms in BENCHMARK_CONFIGS:
        print(f"\n" + "="*80)
        print(f"TESTING CONFIGURATION: {n_farms} Farms")
        print("="*80)
        
        config_results = []
        
        # Run multiple times for this configuration
        for run_num in range(1, NUM_RUNS + 1):
            result = run_benchmark(n_farms, run_number=run_num, total_runs=NUM_RUNS)
            if result:
                config_results.append(result)
                all_results.append(result)
        
        # Calculate statistics for this configuration
        if config_results:
            pulp_times = [r['pulp_time'] for r in config_results if r['pulp_time'] is not None]
            pulp_iterations = [r['pulp_iterations'] for r in config_results if r.get('pulp_iterations') is not None]
            pyomo_times = [r['pyomo_time'] for r in config_results if r['pyomo_time'] is not None]
            dwave_times = [r['dwave_time'] for r in config_results if r['dwave_time'] is not None]
            
            # Calculate objective value differences if both solvers succeeded
            objective_diffs = []
            for r in config_results:
                if r.get('pulp_objective') is not None and r.get('pyomo_objective') is not None:
                    diff = abs(r['pulp_objective'] - r['pyomo_objective']) / abs(r['pyomo_objective']) * 100
                    objective_diffs.append(diff)
            
            aggregated = {
                'n_farms': n_farms,
                'n_foods': config_results[0]['n_foods'],
                'problem_size': config_results[0]['problem_size'],
                'n_vars': config_results[0]['n_vars'],
                'n_constraints': config_results[0]['n_constraints'],
                
                # PuLP (Dinkelbach) stats
                'pulp_time_mean': float(np.mean(pulp_times)) if pulp_times else None,
                'pulp_time_std': float(np.std(pulp_times)) if pulp_times else None,
                'pulp_time_min': float(np.min(pulp_times)) if pulp_times else None,
                'pulp_time_max': float(np.max(pulp_times)) if pulp_times else None,
                'pulp_iterations_mean': float(np.mean(pulp_iterations)) if pulp_iterations else None,
                'pulp_iterations_std': float(np.std(pulp_iterations)) if pulp_iterations else None,
                
                # Pyomo (MINLP) stats
                'pyomo_time_mean': float(np.mean(pyomo_times)) if pyomo_times else None,
                'pyomo_time_std': float(np.std(pyomo_times)) if pyomo_times else None,
                'pyomo_time_min': float(np.min(pyomo_times)) if pyomo_times else None,
                'pyomo_time_max': float(np.max(pyomo_times)) if pyomo_times else None,
                
                # D-Wave stats
                'dwave_time_mean': float(np.mean(dwave_times)) if dwave_times else None,
                'dwave_time_std': float(np.std(dwave_times)) if dwave_times else None,
                'dwave_time_min': float(np.min(dwave_times)) if dwave_times else None,
                'dwave_time_max': float(np.max(dwave_times)) if dwave_times else None,
                
                # Objective comparison
                'objective_diff_mean': float(np.mean(objective_diffs)) if objective_diffs else None,
                'objective_diff_std': float(np.std(objective_diffs)) if objective_diffs else None,
                
                'num_runs': len(config_results)
            }
            
            aggregated_results.append(aggregated)
            
            # Print statistics
            print(f"\n  Statistics for {n_farms} farms ({len(config_results)} runs):")
            if aggregated['pulp_time_mean']:
                print(f"    PuLP (Dinkelbach): {aggregated['pulp_time_mean']:.3f}s ± {aggregated['pulp_time_std']:.3f}s")
                if aggregated['pulp_iterations_mean']:
                    print(f"      Iterations:      {aggregated['pulp_iterations_mean']:.1f} ± {aggregated['pulp_iterations_std']:.1f}")
            if aggregated['pyomo_time_mean']:
                print(f"    Pyomo (MINLP):     {aggregated['pyomo_time_mean']:.3f}s ± {aggregated['pyomo_time_std']:.3f}s")
            if aggregated['dwave_time_mean']:
                print(f"    D-Wave (CQM):      {aggregated['dwave_time_mean']:.3f}s ± {aggregated['dwave_time_std']:.3f}s")
            if aggregated['objective_diff_mean'] is not None:
                print(f"    Obj Difference:    {aggregated['objective_diff_mean']:.2f}% ± {aggregated['objective_diff_std']:.2f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all individual runs
    all_results_file = f'benchmark_nld_all_runs_{timestamp}.json'
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save aggregated statistics
    aggregated_file = f'benchmark_nld_aggregated_{timestamp}.json'
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"All runs saved to: {all_results_file}")
    print(f"Aggregated stats saved to: {aggregated_file}")
    
    # Create plots
    print(f"\nGenerating plots...")
    plot_results(aggregated_results, f'scalability_benchmark_nld_{timestamp}.png')
    
    print(f"\n🎉 All done! Ready for your presentation!")

if __name__ == "__main__":
    main()
