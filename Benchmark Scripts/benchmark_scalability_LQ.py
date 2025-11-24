"""
Scalability Benchmark Script for Linear-Quadratic (LQ) Solver
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.farm_sampler import generate_farms
from src.scenarios import load_food_data
from solver_runner_LQ import create_cqm, solve_with_pulp, solve_with_pyomo, solve_with_dwave, SYNERGY_OPTIMIZER_TYPE
from Utils.benchmark_cache import BenchmarkCache, serialize_cqm
import pulp as pl

# Helper function to clean solution data for JSON serialization
def clean_solution_for_json(solution):
    """Convert solution dictionary to JSON-serializable format."""
    if not solution:
        return {}
    
    cleaned = {}
    for key, value in solution.items():
        # Convert tuple keys to strings
        if isinstance(key, tuple):
            key_str = '_'.join(str(k) for k in key)
        else:
            key_str = str(key)
        
        # Convert values to basic types
        if hasattr(value, 'item'):  # numpy types
            cleaned[key_str] = value.item()
        elif isinstance(value, (int, float, str, bool, type(None))):
            cleaned[key_str] = value
        else:
            cleaned[key_str] = str(value)
    
    return cleaned

# Benchmark configurations
# Format: number of farms to test with full_family scenario
# 6 points logarithmically scaled from 5 to 1535 farms
# Reduced from 30 points for faster testing with multiple runs
BENCHMARK_CONFIGS = [
    10,
    15,
    #25,
    50,
    279,
    1096
]

# Number of runs per configuration for statistical analysis
NUM_RUNS = 1

# Global cache for synergy matrix (same for all farm counts)
_SYNERGY_MATRIX_CACHE = None
_FOODS_CACHE = None
_FOOD_GROUPS_CACHE = None

# Global cache for CQM models (reuse across runs with same farm count)
_CQM_CACHE = {}  # Key: (n_farms, fixed_total_land)

def load_full_family_with_n_farms(n_farms, seed=42, fixed_total_land=None, use_cached_synergy=True):
    """
    Load full_family scenario with specified number of farms.
    Uses the same logic as the scaling analysis but with synergy matrix.
    
    Args:
        n_farms: Number of farms to generate
        seed: Random seed for reproducibility
        fixed_total_land: If provided, scale farms to this total land area (in ha).
                         If None, use generated farm sizes as-is.
    """
    import pandas as pd
    
    # Generate farms
    L = generate_farms(n_farms=n_farms, seed=seed)
    
    # Scale to fixed total land if specified
    if fixed_total_land is not None:
        total_generated = sum(L.values())
        scale_factor = fixed_total_land / total_generated if total_generated > 0 else 0
        L = {farm: area * scale_factor for farm, area in L.items()}
    
    farms = list(L.keys())
    
    # Load food data from Excel or use fallback
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    excel_path = os.path.join(project_root, "Inputs", "Combined_Food_Data.xlsx")
    
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
    
    # --- Generate synergy matrix (NEW for LQ solver) ---
    # Use cached synergy matrix if available (it's the same for all farm counts!)
    global _SYNERGY_MATRIX_CACHE, _FOODS_CACHE, _FOOD_GROUPS_CACHE
    
    if use_cached_synergy and _SYNERGY_MATRIX_CACHE is not None:
        synergy_matrix = _SYNERGY_MATRIX_CACHE
        # Verify foods match (they should)
        if set(foods.keys()) != set(_FOODS_CACHE.keys()):
            print("⚠️  Foods changed, regenerating synergy matrix")
            _SYNERGY_MATRIX_CACHE = None
    
    if not use_cached_synergy or _SYNERGY_MATRIX_CACHE is None:
        synergy_matrix = {}
        default_boost = 0.1  # A default boost value for pairs in the same group

        for group_name, crops_in_group in food_groups.items():
            for i in range(len(crops_in_group)):
                for j in range(i + 1, len(crops_in_group)):
                    crop1 = crops_in_group[i]
                    crop2 = crops_in_group[j]

                    if crop1 not in synergy_matrix:
                        synergy_matrix[crop1] = {}
                    if crop2 not in synergy_matrix:
                        synergy_matrix[crop2] = {}

                    # Add symmetric entries for the pair
                    synergy_matrix[crop1][crop2] = default_boost
                    synergy_matrix[crop2][crop1] = default_boost
        
        # Cache for future use
        _SYNERGY_MATRIX_CACHE = synergy_matrix
        _FOODS_CACHE = foods
        _FOOD_GROUPS_CACHE = food_groups
    # --- End synergy matrix generation ---
    
    # Set minimum planting areas based on smallest farm and number of food groups
    smallest_farm = min(L.values())
    n_food_groups = len(food_groups)
    # Each farm must plant at least 1 crop from each food group
    # Reserve some margin for safety
    min_area_per_crop = 0.001 
    
    min_areas = {food: min_area_per_crop for food in foods.keys()}
    
    # Build config with synergy matrix
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
            'sustainability': 0.15,
            'synergy_bonus': 0.1  # NEW weight for synergy bonus
        },
        'synergy_matrix': synergy_matrix  # NEW parameter
    }
    
    config = {'parameters': parameters}
    
    return farms, foods, food_groups, config

def run_benchmark(n_farms, run_number=1, total_runs=1, cache=None, save_to_cache=True, fixed_total_land=None):
    """
    Run a single benchmark test with full_family scenario using LQ solver.
    Returns timing results and problem size metrics for all three solvers.
    
    Args:
        n_farms: Number of farms to test
        run_number: Current run number (for display)
        total_runs: Total number of runs (for display)
        cache: BenchmarkCache instance for saving results
        save_to_cache: Whether to save results to cache (default: True)
        fixed_total_land: If provided, scale farms to this total land area (in ha)
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK: full_family scenario with {n_farms} Farms (Run {run_number}/{total_runs})")
    if fixed_total_land:
        print(f"Fixed Total Land: {fixed_total_land} ha")
    print(f"Synergy Optimizer: {SYNERGY_OPTIMIZER_TYPE}")
    print(f"{'='*80}")
    
    try:
        # Load full_family scenario with specified number of farms
        farms, foods, food_groups, config = load_full_family_with_n_farms(n_farms, seed=42 + run_number, fixed_total_land=fixed_total_land)
        
        n_foods = len(foods)
        # For LQ solver, we don't have lambda variables (simpler than NLN)
        n_vars_base = 2 * n_farms * n_foods  # Binary + continuous (A and Y)
        
        # Count synergy pairs (for PuLP linearization, we add Z variables)
        synergy_matrix = config['parameters'].get('synergy_matrix', {})
        n_synergy_pairs = 0
        for crop1, pairs in synergy_matrix.items():
            n_synergy_pairs += len(pairs)
        n_synergy_pairs = n_synergy_pairs // 2  # Each pair counted twice
        
        n_z_vars_pulp = n_farms * n_synergy_pairs  # Z variables for PuLP linearization
        n_vars_pulp = n_vars_base + n_z_vars_pulp
        n_vars_quadratic = n_vars_base  # For CQM and Pyomo (native quadratic)
        
        n_constraints_base = n_farms + 2*n_farms*n_foods + 2*len(food_groups)*n_farms
        n_linearization_constraints = n_z_vars_pulp * 3  # 3 constraints per Z variable (McCormick)
        n_constraints_pulp = n_constraints_base + n_linearization_constraints
        n_constraints_quadratic = n_constraints_base
        
        problem_size = n_farms * n_foods  # n = farms × foods
        
        # Verify total land area
        total_land_area = sum(config['parameters']['land_availability'].values())
        
        print(f"  Foods: {n_foods}")
        print(f"  Total Land Area: {total_land_area:.2f} ha")
        print(f"  Synergy Pairs: {n_synergy_pairs}")
        print(f"  Base Variables (A+Y): {n_vars_base}")
        print(f"  PuLP Variables (A+Y+Z): {n_vars_pulp}")
        print(f"  CQM/Pyomo Variables: {n_vars_quadratic}")
        print(f"  PuLP Constraints: {n_constraints_pulp}")
        print(f"  CQM/Pyomo Constraints: {n_constraints_quadratic}")
        print(f"  Problem Size (n): {problem_size}")
        
        # Create CQM (needed for DWave)
        # Check if we can reuse a cached CQM for this configuration
        cache_key = (n_farms, fixed_total_land, run_number) if run_number == 1 else None
        use_cached_cqm = cache_key is not None and cache_key in _CQM_CACHE
        
        if use_cached_cqm:
            print(f"\n  Reusing cached CQM model from run 1...")
            cqm_start = time.time()
            cqm, A, Y, constraint_metadata = _CQM_CACHE[cache_key]
            cqm_time = time.time() - cqm_start
            print(f"    ✅ CQM retrieved: {len(cqm.variables)} vars, {len(cqm.constraints)} constraints ({cqm_time:.5f}s - cache hit!)")
        else:
            print(f"\n  Creating CQM model... (run {run_number})")
            cqm_start = time.time()
            cqm, A, Y, constraint_metadata = create_cqm(
                farms, foods, food_groups, config
            )
            cqm_time = time.time() - cqm_start
            print(f"    ✅ CQM created: {len(cqm.variables)} vars, {len(cqm.constraints)} constraints ({cqm_time:.2f}s)")
            
            # Cache the CQM for run 1 to reuse in subsequent runs
            if run_number == 1 and cache_key is not None:
                _CQM_CACHE[cache_key] = (cqm, A, Y, constraint_metadata)
                print(f"    📦 CQM cached for reuse in runs 2-{total_runs}")
        
        # Save CQM to cache if requested
        if save_to_cache and cache:
            cqm_data = serialize_cqm(cqm)
            cqm_result = {
                'cqm_time': cqm_time,
                'num_variables': len(cqm.variables),
                'num_constraints': len(cqm.constraints),
                'n_foods': n_foods,
                'problem_size': problem_size,
                'n_vars_quadratic': n_vars_quadratic,
                'n_constraints_quadratic': n_constraints_quadratic,
                'n_synergy_pairs': n_synergy_pairs
            }
            cache.save_result('LQ', 'CQM', n_farms, run_number, cqm_result, cqm_data=cqm_data)
        
        # Solve with PuLP (linearized quadratic)
        print(f"\n  Solving with PuLP (Linearized Quadratic)...")
        pulp_model, pulp_results = solve_with_pulp(farms, foods, food_groups, config)
        pulp_time = pulp_results['solve_time']  # Use internal timing (solver only, no Python overhead)
        
        print(f"    Status: {pulp_results['status']}")
        print(f"    Objective: {pulp_results.get('objective_value', 'N/A')}")
        print(f"    Solve Time (solver only): {pulp_time:.3f}s")
        
        # Save PuLP results to cache in comprehensive benchmark format
        if save_to_cache and cache:
            pulp_cache_result = {
                'status': pulp_results['status'],
                'objective_value': pulp_results.get('objective_value'),
                'solve_time': pulp_time,
                'solver_time': pulp_time,
                'success': pulp_results['status'] == 'Optimal',
                'sample_id': run_number - 1,
                'n_units': n_farms,
                'total_area': pulp_results.get('total_area'),
                'n_foods': n_foods,
                'n_variables': n_vars_quadratic,
                'n_constraints': n_constraints_quadratic,
                'solution_areas': pulp_results.get('areas', {}),
                'solution_selections': pulp_results.get('selections', {}),
                'total_covered_area': pulp_results.get('total_area', 0),
                'solution_summary': pulp_results.get('solution_summary', {}),
                'validation': pulp_results.get('validation', {})
            }
            cache.save_result('LQ', 'PuLP', n_farms, run_number, pulp_cache_result)
        
        # Solve with Pyomo (native quadratic)
        print(f"\n  Solving with Pyomo (Native Quadratic)...")
        pyomo_model, pyomo_results = solve_with_pyomo(farms, foods, food_groups, config)
        pyomo_time = pyomo_results.get('solve_time', None)  # Use internal timing (solver only)
        
        if pyomo_results.get('error'):
            print(f"    Status: {pyomo_results['status']}")
            print(f"    Error: {pyomo_results.get('error')}")
            pyomo_time = None
            pyomo_objective = None
        else:
            print(f"    Status: {pyomo_results['status']}")
            print(f"    Objective: {pyomo_results.get('objective_value', 'N/A')}")
            print(f"    Solve Time (solver only): {pyomo_time:.3f}s" if pyomo_time else "    Solve Time: N/A")
            pyomo_objective = pyomo_results.get('objective_value')
        
        # Save Pyomo results to cache in comprehensive benchmark format
        if save_to_cache and cache:
            pyomo_cache_result = {
                'status': pyomo_results.get('status', 'Error'),
                'objective_value': pyomo_objective,
                'solve_time': pyomo_time,
                'solver_time': pyomo_time,
                'success': pyomo_objective is not None and pyomo_results.get('status', '').startswith('optimal'),
                'sample_id': run_number - 1,
                'n_units': n_farms,
                'total_area': pyomo_results.get('total_area', 0),
                'n_foods': n_foods,
                'n_variables': n_vars_quadratic,
                'n_constraints': n_constraints_quadratic,
                'solver': pyomo_results.get('solver'),
                'solution_areas': pyomo_results.get('areas', {}),
                'solution_selections': pyomo_results.get('selections', {}),
                'total_covered_area': pyomo_results.get('total_area', 0),
                'solution_summary': pyomo_results.get('solution_summary', {}),
                'validation': pyomo_results.get('validation', {}),
                'error': pyomo_results.get('error')
            }
            cache.save_result('LQ', 'Pyomo', n_farms, run_number, pyomo_cache_result)
        
        # Solve with DWave
        print(f"\n  Solving with DWave...")
        #token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
        token = None

        dwave_time = None
        qpu_time = None
        hybrid_time = None
        dwave_feasible = False
        dwave_objective = None
        dwave_total_area = None
        dwave_normalized_objective = None
        sampleset = None
        feasible_sampleset = None

        if not token:
            print(f"    SKIPPED: DWAVE_API_TOKEN not found.")
        else:
            try:
                sampleset, dwave_time = solve_with_dwave(cqm, token)

                feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
                dwave_feasible = len(feasible_sampleset) > 0

                if dwave_feasible:
                    best = feasible_sampleset.first
                    dwave_objective = -best.energy
                    
                    # Calculate total area from the solution
                    # CQM variables use "A_{farm}_{crop}" prefix (NOT "Area_")
                    dwave_total_area = 0.0
                    for var_name, var_value in best.sample.items():
                        if var_name.startswith('A_') and var_value > 1e-6:
                            dwave_total_area += var_value
                    
                    # Calculate normalized objective
                    dwave_normalized_objective = dwave_objective / dwave_total_area if dwave_total_area > 1e-6 else 0.0
                    
                    # Extract timing info correctly - DWave returns times in seconds already
                    qpu_time = sampleset.info.get('qpu_access_time', 0)  / 1e6
                    hybrid_time = sampleset.info.get('charge_time', 0)   / 1e6
                    run_time = sampleset.info.get('run_time', 0) / 1e6

                    print(f"    Status: {len(feasible_sampleset)} feasible solutions")
                    print(f"    Objective: {dwave_objective:.6f}")
                    print(f"    Total Area: {dwave_total_area:.2f}")
                    print(f"    Normalized Objective: {dwave_normalized_objective:.6f}")
                    print(f"    Total Time: {dwave_time:.3f}s")
                    if hybrid_time and hybrid_time > 0:
                        print(f"    Charge Time: {hybrid_time:.3f}s")
                    if qpu_time and qpu_time > 0:
                        print(f"    QPU Access Time: {qpu_time:.4f}s")
                    if run_time and run_time > 0:
                        print(f"    Run Time: {run_time:.3f}s")
                else:
                    print("    Status: No feasible solutions found")

            except Exception as e:
                print(f"    ERROR: DWave solving failed: {str(e)}")
        
        # Save DWave results to cache in comprehensive benchmark format (matching Farm scenario extraction)
        if save_to_cache and cache:
            # Extract solution variables and run validation if feasible solution found
            dwave_solution_areas = {}
            dwave_solution_selections = {}
            dwave_land_data = {}
            dwave_validation = {}
            
            if dwave_feasible and sampleset:
                from solver_runner_LQ import validate_solution_constraints, extract_solution_summary
                
                # Extract best solution
                best = feasible_sampleset.first
                cqm_sample = dict(best.sample)
                
                # Calculate total covered area (sum of all A_ variables) - matching comprehensive_benchmark.py
                total_covered_area = sum(
                    v for k, v in cqm_sample.items() 
                    if k.startswith('A_') and isinstance(v, (int, float)) and v > 0
                )
                
                # Build solution dictionary and extract areas/selections
                # CQM variables are named "A_{farm}_{crop}" and "Y_{farm}_{crop}" (NOT "Area_")
                solution = {}
                for var_name, var_value in cqm_sample.items():
                    if var_name.startswith('A_'):
                        # A_ variables are already in correct format
                        solution[var_name] = var_value
                        # Store in solution_areas dict (full variable name with A_ prefix)
                        dwave_solution_areas[var_name] = var_value
                    elif var_name.startswith('Y_'):
                        # Y variables already have correct prefix
                        solution[var_name] = var_value
                        # Store in solution_selections dict (full variable name with Y_ prefix)
                        dwave_solution_selections[var_name] = var_value
                
                # Calculate land usage per farm - use EXACT land_data from config
                land_availability = config['parameters']['land_availability']
                for farm in farms:
                    farm_total = sum(cqm_sample.get(f"A_{farm}_{crop}", 0) for crop in foods)
                    dwave_land_data[farm] = farm_total
                
                # Run validation using exact same approach as comprehensive_benchmark.py Farm scenario
                dwave_validation = validate_solution_constraints(
                    solution, farms, foods, food_groups, land_availability, config
                )
                
                # Extract solution summary
                dwave_solution_summary = extract_solution_summary(solution, farms, foods, land_availability)
            else:
                dwave_solution_summary = {}
                total_covered_area = 0
            
            dwave_cache_result = {
                'status': 'Optimal' if dwave_feasible else 'Infeasible',
                'objective_value': dwave_objective if dwave_feasible else None,
                'solve_time': dwave_time,
                'qpu_time': qpu_time,
                'hybrid_time': hybrid_time,
                'is_feasible': dwave_feasible,
                'success': dwave_feasible,
                'sample_id': run_number - 1,
                'n_units': n_farms,
                'total_area': sum(config['parameters']['land_availability'].values()) if config else 0,
                'n_foods': n_foods,
                'n_variables': n_vars_quadratic,
                'n_constraints': n_constraints_quadratic,
                'total_covered_area': total_covered_area,
                'num_samples': len(feasible_sampleset) if dwave_feasible and feasible_sampleset else 0,
                'solution_areas': dwave_solution_areas,  # Full variable names with A_ prefix
                'land_data': dwave_land_data,  # Actual land usage per farm
                'validation': dwave_validation
            }
            cache.save_result('LQ', 'DWave', n_farms, run_number, dwave_cache_result)
        
        # Calculate difference between PuLP and Pyomo (should be exact)
        pulp_error = None
        if pyomo_objective is not None and pulp_results.get('objective_value') is not None:
            pulp_error = abs(pulp_results['objective_value'] - pyomo_objective) / abs(pyomo_objective) * 100
            print(f"\n  Solution Comparison:")
            print(f"    PuLP vs Pyomo: {pulp_error:.4f}% (should be ~0% - exact)")
        
        # Build result in comprehensive_benchmark.py format
        result = {
            'sample_id': run_number - 1,  # 0-indexed
            'scenario_type': 'lq_farm',
            'n_units': n_farms,
            'total_area': sum(config['parameters']['land_availability'].values()),
            'n_foods': n_foods,
            'n_variables': n_vars_quadratic,  # Use quadratic count (CQM/Pyomo)
            'n_constraints': n_constraints_quadratic,
            'cqm_time': cqm_time,
            'solvers': {}
        }
        
        # Add PuLP results under 'gurobi' key (for consistency with comprehensive_benchmark)
        result['solvers']['gurobi'] = {
            'status': pulp_results['status'],
            'objective_value': pulp_results.get('objective_value'),
            'solve_time': pulp_time,
            'solver_time': pulp_time,  # PuLP uses same for both
            'success': pulp_results['status'] == 'Optimal',
            'sample_id': run_number - 1,
            'n_units': n_farms,
            'total_area': pulp_results.get('total_area', 0),
            'n_foods': n_foods,
            'n_variables': n_vars_quadratic,
            'n_constraints': n_constraints_quadratic,
            'solution_areas': pulp_results.get('areas', {}),
            'solution_selections': pulp_results.get('selections', {}),
            'total_covered_area': pulp_results.get('total_area', 0),
            'solution_summary': pulp_results.get('solution_summary', {}),
            'validation': pulp_results.get('validation', {})
        }
        
        # Add Pyomo results under 'pyomo_native' key
        if pyomo_objective is not None or pyomo_results.get('error'):
            result['solvers']['pyomo_native'] = {
                'status': pyomo_results.get('status', 'Error'),
                'objective_value': pyomo_objective,
                'solve_time': pyomo_time,
                'solver_time': pyomo_time,
                'success': pyomo_objective is not None and pyomo_results.get('status', '').startswith('optimal'),
                'sample_id': run_number - 1,
                'n_units': n_farms,
                'total_area': pyomo_results.get('total_area', 0),
                'n_foods': n_foods,
                'n_variables': n_vars_quadratic,
                'n_constraints': n_constraints_quadratic,
                'solver': pyomo_results.get('solver'),
                'solution_areas': pyomo_results.get('areas', {}),
                'solution_selections': pyomo_results.get('selections', {}),
                'total_covered_area': pyomo_results.get('total_area', 0),
                'solution_summary': pyomo_results.get('solution_summary', {}),
                'validation': pyomo_results.get('validation', {}),
                'error': pyomo_results.get('error')
            }
        
        # Add DWave results under 'dwave_cqm' key
        if dwave_time is not None:
            result['solvers']['dwave_cqm'] = {
                'status': 'Optimal' if dwave_feasible else 'No Solutions',
                'objective_value': dwave_objective,
                'solve_time': dwave_time,
                'qpu_time': qpu_time,
                'hybrid_time': hybrid_time,
                'is_feasible': dwave_feasible,
                'success': dwave_feasible,
                'sample_id': run_number - 1,
                'n_units': n_farms,
                'total_area': sum(config['parameters']['land_availability'].values()),
                'n_foods': n_foods,
                'n_variables': n_vars_quadratic,
                'n_constraints': n_constraints_quadratic,
                'total_covered_area': dwave_total_area if dwave_total_area else 0,
                'num_feasible': 1 if dwave_feasible else 0
            }
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(results, output_file='scalability_benchmark_lq.png'):
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
    
    # PuLP times
    pulp_times = [r['pulp_time_mean'] for r in valid_results]
    pulp_errors = [r['pulp_time_std'] for r in valid_results]
    
    # Pyomo times
    pyomo_times = [r['pyomo_time_mean'] for r in valid_results if r['pyomo_time_mean'] is not None]
    pyomo_errors = [r['pyomo_time_std'] for r in valid_results if r['pyomo_time_mean'] is not None]
    pyomo_problem_sizes = [r['problem_size'] for r in valid_results if r['pyomo_time_mean'] is not None]
    
    # Solution difference (should be near zero)
    solution_diffs = [r['pulp_diff_mean'] for r in valid_results if r['pulp_diff_mean'] is not None]
    solution_diffs_std = [r['pulp_diff_std'] for r in valid_results if r['pulp_diff_mean'] is not None]
    diff_problem_sizes = [r['problem_size'] for r in valid_results if r['pulp_diff_mean'] is not None]
    
    # Create figure with professional styling
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Solve times with error bars
    ax1.errorbar(problem_sizes, pulp_times, yerr=pulp_errors, marker='o', linestyle='-', 
                linewidth=2.5, markersize=8, capsize=5, capthick=2,
                label='PuLP (Linearized)', color='#2E86AB', alpha=0.9)
    
    if pyomo_times:
        ax1.errorbar(pyomo_problem_sizes, pyomo_times, yerr=pyomo_errors, marker='s', linestyle='-',
                    linewidth=2.5, markersize=8, capsize=5, capthick=2,
                    label='Pyomo (Native Quadratic)', color='#A23B72', alpha=0.9)
    
    ax1.set_xlabel('Problem Size (n = Farms × Foods)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Solve Time (seconds)', fontsize=14, fontweight='bold')
    ax1.set_title('Linear-Quadratic Solver Performance (Linear + Synergy Bonus)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Add annotations for key points
    if len(problem_sizes) > 0:
        # Annotate largest problem
        max_idx = problem_sizes.index(max(problem_sizes))
        ax1.annotate(f'n={problem_sizes[max_idx]}\nPuLP: {pulp_times[max_idx]:.2f}s',
                    xy=(problem_sizes[max_idx], pulp_times[max_idx]),
                    xytext=(-60, -30), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Solution accuracy (should be near zero)
    if solution_diffs:
        ax2.errorbar(diff_problem_sizes, solution_diffs, yerr=solution_diffs_std,
                    marker='o', linestyle='-', linewidth=2.5, markersize=8,
                    capsize=5, capthick=2, color='#06A77D', alpha=0.9,
                    label='PuLP vs Pyomo')
        
        ax2.set_xlabel('Problem Size (n = Farms × Foods)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Solution Difference (%)', fontsize=14, fontweight='bold')
        ax2.set_title('Linearization Accuracy (McCormick Relaxation)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xscale('log')
        
        # Add horizontal line at y=0 (perfect match)
        ax2.axhline(y=0.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Exact Match')
        ax2.axhline(y=0.01, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='0.01% Diff')
        ax2.legend(loc='best', fontsize=11, framealpha=0.95)
        
        # Add average difference annotation
        avg_diff = np.mean(solution_diffs)
        ax2.text(0.98, 0.98, f'Avg: {avg_diff:.4f}%\n± {np.mean(solution_diffs_std):.4f}%', 
                transform=ax2.transAxes, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    else:
        ax2.text(0.5, 0.5, 'No Pyomo Data\nfor Comparison', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")
    
    # Also create a summary table plot in the same folder as the main plot
    table_file = output_file.replace('scalability_benchmark_lq', 'scalability_table_lq')
    create_summary_table(valid_results, table_file)

def create_summary_table(results, output_file='scalability_table_lq.png'):
    """
    Create a beautiful summary table for LQ benchmark.
    """
    fig, ax = plt.subplots(figsize=(18, len(results) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Farms', 'Foods', 'n', 'Synergy\nPairs', 'PuLP\nVars', 'Quad\nVars', 
               'PuLP\nTime (s)', 'Pyomo\nTime (s)', 'Diff\n(%)', 'Runs', 'Winner']
    
    table_data = []
    for r in results:
        # Determine winner (faster solver)
        if r.get('pyomo_time_mean') is not None and r.get('pulp_time_mean') is not None:
            if r['pulp_time_mean'] < r['pyomo_time_mean']:
                winner = '🏆 PuLP'
            elif r['pyomo_time_mean'] < r['pulp_time_mean']:
                winner = '🏆 Pyomo'
            else:
                winner = 'Tie'
        else:
            winner = 'PuLP'
        
        row = [
            r['n_farms'],
            r['n_foods'],
            r['problem_size'],
            r.get('n_synergy_pairs', 'N/A'),
            r.get('n_vars_pulp', 'N/A'),
            r.get('n_vars_quadratic', 'N/A'),
            f"{r['pulp_time_mean']:.3f} ± {r['pulp_time_std']:.3f}",
            f"{r['pyomo_time_mean']:.3f} ± {r['pyomo_time_std']:.3f}" if r.get('pyomo_time_mean') else 'N/A',
            f"{r['pulp_diff_mean']:.4f}" if r.get('pulp_diff_mean') else 'N/A',
            r['num_runs'],
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
    
    plt.title('Linear-Quadratic Scalability Benchmark Results', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Table saved to: {output_file}")

def main():
    """
    Run all benchmarks with multiple runs and calculate statistics.
    Uses intelligent caching to avoid redundant runs.
    All benchmarks use fixed 100 ha total land area for comparability.
    """
    # Initialize cache
    cache = BenchmarkCache()
    
    print("="*80)
    print("LINEAR-QUADRATIC SCALABILITY BENCHMARK")
    print("="*80)
    print(f"Configurations: {len(BENCHMARK_CONFIGS)} points")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(f"Total benchmarks: {len(BENCHMARK_CONFIGS) * NUM_RUNS}")
    print(f"Objective: Linear area + Quadratic synergy bonus")
    print(f"Fixed Total Land: 100 ha (for all configurations)")
    print("="*80)
    
    # Check cache status
    print("\n" + "="*80)
    print("CHECKING CACHE STATUS")
    print("="*80)
    print("⚠️  NOTE: Any existing cached results used unbounded farms.")
    print("    This benchmark now uses fixed 100 ha total land.")
    print("    Cache may need to be cleared for accurate comparisons.")
    cache.print_cache_status('LQ', BENCHMARK_CONFIGS, NUM_RUNS)
    
    all_results = []
    aggregated_results = []
    
    # Clear CQM cache at start of benchmark to ensure fresh models
    global _CQM_CACHE
    _CQM_CACHE = {}
    print(f"\n📦 CQM caching enabled: Run 1 CQM will be cached and reused for runs 2-{NUM_RUNS}")
    print(f"   This saves ~{NUM_RUNS-1}x CQM build time per configuration!")
    
    for n_farms in BENCHMARK_CONFIGS:
        print(f"\n" + "="*80)
        print(f"TESTING CONFIGURATION: {n_farms} Farms (Fixed 100 ha)")
        print("="*80)
        
        # Check which runs are needed
        runs_needed = cache.get_runs_needed('LQ', n_farms, NUM_RUNS)
        
        # NOTE: Skip loading existing cached results since they were generated 
        # with unbounded farms. We need fresh runs with fixed 100 ha total land.
        print(f"\n  Skipping cached results (incompatible with fixed 100 ha approach)")
        config_results = []
        
        # Add cached results to all_results list (none in this case)
        all_results.extend(config_results)
        
        # Force all runs to be executed with fixed land
        all_missing_runs = list(range(1, NUM_RUNS + 1))
        
        if all_missing_runs:
            print(f"  Running {len(all_missing_runs)} benchmarks with fixed 100 ha total land")
            
            # Run the missing benchmarks with fixed 100 ha total land
            for run_num in all_missing_runs:
                result = run_benchmark(n_farms, run_number=run_num, total_runs=NUM_RUNS, 
                                     cache=cache, save_to_cache=True, fixed_total_land=100.0)
                if result:
                    config_results.append(result)
                    all_results.append(result)
        else:
            print(f"  ✅ All {NUM_RUNS} runs already completed!")
        
        # Calculate statistics for this configuration
        if config_results:
            # Extract times from the new nested format
            pulp_times = [r['solvers']['gurobi']['solve_time'] for r in config_results 
                         if 'gurobi' in r['solvers'] and r['solvers']['gurobi'].get('solve_time') is not None]
            pyomo_times = [r['solvers']['pyomo_native']['solve_time'] for r in config_results 
                          if 'pyomo_native' in r['solvers'] and r['solvers']['pyomo_native'].get('solve_time') is not None]
            cqm_times = [r['cqm_time'] for r in config_results if r.get('cqm_time') is not None]
            
            # Calculate solution differences for validation
            pulp_diffs = []
            for r in config_results:
                if 'gurobi' in r['solvers'] and 'pyomo_native' in r['solvers']:
                    gurobi_obj = r['solvers']['gurobi'].get('objective_value')
                    pyomo_obj = r['solvers']['pyomo_native'].get('objective_value')
                    if gurobi_obj is not None and pyomo_obj is not None and pyomo_obj != 0:
                        diff = abs(gurobi_obj - pyomo_obj) / abs(pyomo_obj) * 100
                        pulp_diffs.append(diff)
            
            aggregated = {
                'n_farms': n_farms,
                'n_foods': config_results[0]['n_foods'],
                'problem_size': n_farms * config_results[0]['n_foods'],
                'n_variables': config_results[0]['n_variables'],
                'n_constraints': config_results[0]['n_constraints'],
                
                # CQM creation stats
                'cqm_time_mean': float(np.mean(cqm_times)) if cqm_times else None,
                'cqm_time_std': float(np.std(cqm_times)) if cqm_times else None,
                
                # PuLP stats
                'pulp_time_mean': float(np.mean(pulp_times)) if pulp_times else None,
                'pulp_time_std': float(np.std(pulp_times)) if pulp_times else None,
                'pulp_time_min': float(np.min(pulp_times)) if pulp_times else None,
                'pulp_time_max': float(np.max(pulp_times)) if pulp_times else None,
                
                # Pyomo stats
                'pyomo_time_mean': float(np.mean(pyomo_times)) if pyomo_times else None,
                'pyomo_time_std': float(np.std(pyomo_times)) if pyomo_times else None,
                'pyomo_time_min': float(np.min(pyomo_times)) if pyomo_times else None,
                'pyomo_time_max': float(np.max(pyomo_times)) if pyomo_times else None,
                
                # Solution difference stats (should be near zero)
                'pulp_diff_mean': float(np.mean(pulp_diffs)) if pulp_diffs else None,
                'pulp_diff_std': float(np.std(pulp_diffs)) if pulp_diffs else None,
                
                'num_runs': len(config_results),
            }
            
            aggregated_results.append(aggregated)
            
            # Print statistics
            print(f"\n  Statistics for {n_farms} farms ({len(config_results)} runs):")
            print(f"    CQM Creation: {aggregated['cqm_time_mean']:.3f}s ± {aggregated['cqm_time_std']:.3f}s")
            print(f"    PuLP:         {aggregated['pulp_time_mean']:.3f}s ± {aggregated['pulp_time_std']:.3f}s")
            if aggregated['pyomo_time_mean']:
                print(f"    Pyomo:        {aggregated['pyomo_time_mean']:.3f}s ± {aggregated['pyomo_time_std']:.3f}s")
            if aggregated['pulp_diff_mean']:
                print(f"    Solution Diff: {aggregated['pulp_diff_mean']:.4f}% ± {aggregated['pulp_diff_std']:.4f}%")
    
    # Results are saved to individual solver directories (PuLP, Pyomo, DWave, CQM)
    # No aggregated files are created
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to individual solver directories in Benchmarks/LQ/")
    
    # Print cache performance summary
    if NUM_RUNS > 1:
        print(f"\n📊 CACHE PERFORMANCE:")
        print(f"   Synergy matrix: Generated once, reused for all configs")
        print(f"   CQM models: Created once per config, reused for {NUM_RUNS-1} additional runs")
        total_saved = sum(r.get('cqm_time_mean', 0) * (NUM_RUNS - 1) for r in aggregated_results if r.get('cqm_time_mean'))
        print(f"   Estimated time saved: {total_saved:.2f}s across all configs")
    
    # Create plots in Plots folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    plots_dir = os.path.join(project_root, "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nGenerating plots...")
    plot_results(aggregated_results, os.path.join(plots_dir, f'scalability_benchmark_lq_{timestamp}.png'))
    
    print(f"\n🎉 All done! Ready for your presentation!")

if __name__ == "__main__":
    main()
