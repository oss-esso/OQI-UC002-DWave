#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for Quantum/Classical Optimization Comparison

This script runs a comprehensive comparison between:
- Farm Scenario: Full-scale farm optimization (Gurobi, D-Wave CQM)
- Patch Scenario: Smaller patch optimization (Gurobi, D-Wave CQM, Gurobi QUBO, D-Wave BQM)

Total solver configurations tested: 6
1. Farm + Gurobi
2. Farm + D-Wave CQM
3. Patch + Gurobi
4. Patch + D-Wave CQM  
5. Patch + Gurobi QUBO
6. Patch + D-Wave BQM

Plot lines: 9 total (3 D-Wave × 2 timing modes + 3 Gurobi × 1 timing mode)

Results are saved to a comprehensive JSON file for analysis and plotting.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import farm and patch generators
from Utils.farm_sampler import generate_farms as generate_farms_large
from Utils.patch_sampler import generate_farms as generate_patches_small
from src.scenarios import load_food_data

# Import solvers from unified solver_runner_BINARY (handles both formulations)
import solver_runner_BINARY as solver_runner

from dimod import cqm_to_bqm

# Benchmark configurations
# Format: number of units (farms or patches) to test
BENCHMARK_CONFIGS = [
    10,
    15,
    #20,
    25,
    #50,
    #100
]

# Number of runs per configuration for statistical analysis
NUM_RUNS = 1

# Gurobi QUBO timeout in seconds
#GUROBI_QUBO_TIMEOUT = 0  # 300 seconds (5 minutes) to match TimeLimit in solver

def generate_sample_data(config_values: List[int], seed_offset: int = 0, fixed_total_land: float = 100.0) -> List[Tuple[Dict, Dict]]:
    """
    Generate paired samples of continuous (farm) and binary (patch) scenarios.

    Uses solver_runner_BINARY.py's land generation approach:
    - Uses a FIXED total land area (default 100 ha) for all scenarios
    - For each config value N:
      * Farm scenario: N farms with uneven_distribution (realistic farm sizes)
      * Patch scenario: N patches with even_grid (equal-sized plots)
    - This ensures direct comparability across all refinement levels

    Args:
        config_values: List of configuration values (number of units to generate).
        seed_offset: Offset for random seed to ensure variety.
        fixed_total_land: Fixed total land area in hectares (default: 100.0 ha)

    Returns:
        A list of tuples, where each tuple is `(farm_sample, patch_sample)`.
    """
    print(f"\n{'='*80}")
    print(f"GENERATING PAIRED SAMPLES FOR CONFIGS: {config_values}")
    print(f"Fixed Total Land: {fixed_total_land:.2f} ha (same for all scenarios)")
    print(f"{'='*80}")

    paired_samples = []

    for i, n_units in enumerate(config_values):
        print(f"\n--- Generating pair for {n_units} units (total {fixed_total_land:.2f} ha) ---")
        
        # 1. Generate FARM scenario (uneven_distribution) - Continuous formulation
        seed = 42 + seed_offset + i * 100
        farms_unscaled = generate_farms_large(n_farms=n_units, seed=seed)
        farms_total = sum(farms_unscaled.values())
        farm_scale_factor = fixed_total_land / farms_total if farms_total > 0 else 0
        farms_scaled = {k: v * farm_scale_factor for k, v in farms_unscaled.items()}
        
        farm_sample = {
            'sample_id': i,
            'type': 'farm',
            'land_method': 'uneven_distribution',
            'data': farms_scaled,
            'total_area': sum(farms_scaled.values()),
            'n_units': n_units,
            'seed': seed
        }
        
        # Calculate farm statistics
        areas = list(farms_scaled.values())
        print(f"  ✓ Farm (continuous): {farm_sample['n_units']} farms")
        print(f"     Area: min={min(areas):.2f} ha, max={max(areas):.2f} ha, avg={sum(areas)/len(areas):.2f} ha")
        print(f"     Total: {sum(farms_scaled.values()):.2f} ha")

        # 2. Generate PATCH scenario (even_grid) - Binary formulation
        patch_seed = seed + 50
        patches_unscaled = generate_patches_small(n_farms=n_units, seed=patch_seed)
        patches_total = sum(patches_unscaled.values())
        patch_scale_factor = fixed_total_land / patches_total if patches_total > 0 else 0
        patches_scaled = {k: v * patch_scale_factor for k, v in patches_unscaled.items()}

        patch_sample = {
            'sample_id': i,
            'type': 'patch',
            'land_method': 'even_grid',
            'data': patches_scaled,
            'total_area': sum(patches_scaled.values()),
            'n_units': n_units,
            'seed': patch_seed
        }
        
        # Calculate patch statistics
        patch_area = fixed_total_land / n_units
        print(f"  ✓ Patch (binary): {patch_sample['n_units']} plots")
        print(f"     Area per plot: {patch_area:.3f} ha (equal grid)")
        print(f"     Total: {sum(patches_scaled.values()):.2f} ha")

        paired_samples.append((farm_sample, patch_sample))

    print(f"\nGenerated {len(paired_samples)} paired samples (all with {fixed_total_land:.2f} ha total land).")
    return paired_samples

def create_food_config(land_data: Dict[str, float], scenario_type: str = 'comprehensive') -> Tuple[Dict, Dict, Dict]:
    """
    Create food and configuration data compatible with solvers.
    
    Args:
        land_data: Dictionary of land availability
        scenario_type: Type of scenario to create
        
    Returns:
        Tuple of (foods, food_groups, config)
    """
    # Load food data from scenarios module - use full_family for complete food set
    try:
        food_list, foods, food_groups, _ = load_food_data('full_family')
    except Exception as e:
        print(f"    Warning: Food data loading failed ({e}), using fallback")
        #foods, food_groups = create_fallback_foods()
    
    # Create configuration matching original solver_runner.py formulation
    config = {
        'parameters': {
            'land_availability': land_data,
            'minimum_planting_area': {food: 0.0001 for food in foods},  # No minimum area constraint
            # Add food group constraints: at least 1 food from each group
            'food_group_constraints': {
                group: {'min_foods': 1, 'max_foods': len(food_list)}
                for group, food_list in food_groups.items()
            },
            'weights': {
                'nutritional_value': 0.25,
                'nutrient_density': 0.2,
                'environmental_impact': 0.25,
                'affordability': 0.15,
                'sustainability': 0.15
            },
            'idle_penalty_lambda': 0.0 #set to zero to avoid too big weights
        }
    }
    
    return foods, food_groups, config

#def create_fallback_foods():
#    """Create fallback food data if Excel is not available."""
#    foods = {
#        'Wheat': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.6, 'affordability': 0.9, 'sustainability': 0.7},
#        'Corn': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.5, 'affordability': 0.8, 'sustainability': 0.6},
#        'Rice': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.7, 'affordability': 0.7, 'sustainability': 0.8},
#        'Soybeans': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.4, 'affordability': 0.6, 'sustainability': 0.9},
#        'Potatoes': {'nutritional_value': 0.5, 'nutrient_density': 0.4, 'environmental_impact': 0.8, 'affordability': 0.9, 'sustainability': 0.6}
#    }
#    food_groups = {
#        'grains': ['Wheat', 'Corn', 'Rice'],
#        'proteins': ['Soybeans'],
#        'vegetables': ['Potatoes']
#    }
#    return foods, food_groups

def check_cached_results(scenario_type: str, solver_name: str, config_id: int, run_id: int = 1) -> Optional[Dict]:
    """
    Check if a cached result exists for the given configuration.
    
    Args:
        scenario_type: 'farm' or 'patch'
        solver_name: Name of solver (gurobi, dwave_cqm, gurobi_qubo, dwave_bqm)
        config_id: Configuration ID (number of units)
        run_id: Run number for this configuration
        
    Returns:
        Cached result dictionary if found, None otherwise
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Map solver names to subdirectory names
    solver_dir_map = {
        'farm_gurobi': 'Farm_PuLP',
        'farm_dwave_cqm': 'Farm_DWave',
        'patch_gurobi': 'Patch_PuLP', 
        'patch_dwave_cqm': 'Patch_DWave',
        'patch_gurobi_qubo': 'Patch_GurobiQUBO',
        'patch_dwave_bqm': 'Patch_DWaveBQM'
    }
    
    solver_key = f"{scenario_type}_{solver_name}"
    if solver_key not in solver_dir_map:
        return None
    
    solver_dir = solver_dir_map[solver_key]
    result_dir = os.path.join(script_dir, "Benchmarks", "COMPREHENSIVE", solver_dir)
    
    # Check if result file exists
    filename = f"config_{config_id}_run_{run_id}.json"
    filepath = os.path.join(result_dir, filename)
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: Failed to load cached result {filepath}: {e}")
            return None
    
    return None

def save_solver_result(result: Dict, scenario_type: str, solver_name: str, config_id: int, run_id: int = 1):
    """
    Save individual solver result to appropriate subdirectory.
    
    Args:
        result: Solver result dictionary
        scenario_type: 'farm' or 'patch'
        solver_name: Name of solver (gurobi, dwave_cqm, gurobi_qubo, dwave_bqm)
        config_id: Configuration ID (number of units)
        run_id: Run number for this configuration
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Map solver names to subdirectory names matching other benchmarks
    solver_dir_map = {
        'farm_gurobi': 'Farm_PuLP',
        'farm_dwave_cqm': 'Farm_DWave',
        'patch_gurobi': 'Patch_PuLP', 
        'patch_dwave_cqm': 'Patch_DWave',
        'patch_gurobi_qubo': 'Patch_GurobiQUBO',
        'patch_dwave_bqm': 'Patch_DWaveBQM'
    }
    
    solver_key = f"{scenario_type}_{solver_name}"
    if solver_key not in solver_dir_map:
        print(f"  Warning: Unknown solver key {solver_key}, skipping save")
        return
    
    solver_dir = solver_dir_map[solver_key]
    output_dir = os.path.join(script_dir, "Benchmarks", "COMPREHENSIVE", solver_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename matching other benchmark format
    filename = f"config_{config_id}_run_{run_id}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save result
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2, default=str)

def run_farm_scenario(sample_data: Dict, dwave_token: Optional[str] = None) -> Dict:
    """
    Run Farm Scenario: Continuous formulation (uneven_distribution) with mixed-integer variables.
    
    Uses solver_runner_BINARY.py's create_cqm_farm() and solve_with_pulp_farm().
    
    Solvers:
    - Gurobi (PuLP): MINLP solver for continuous areas with binary selection
    - D-Wave CQM: Quantum-classical hybrid for constrained quadratic models
    
    Args:
        sample_data: Farm sample data with uneven_distribution land method
        dwave_token: D-Wave API token (optional)
        
    Returns:
        Dictionary with results for both solvers
    """
    print(f"\n  🌾 FARM SCENARIO (Continuous) - Sample {sample_data['sample_id']}")
    print(f"     {sample_data['n_units']} farms, {sample_data['total_area']:.1f} ha")
    print(f"     Method: {sample_data['land_method']}")
    
    # Create problem setup
    land_data = sample_data['data']
    farms_list = list(land_data.keys())
    foods, food_groups, config = create_food_config(land_data, 'farm')
    
    # Create CQM using FARM formulation (continuous + binary)
    cqm_start = time.time()
    cqm, A, Y, constraint_metadata = solver_runner.create_cqm_farm(farms_list, foods, food_groups, config)
    cqm_time = time.time() - cqm_start
    
    results = {
        'sample_id': sample_data['sample_id'],
        'scenario_type': 'farm',
        'land_method': sample_data['land_method'],
        'n_units': sample_data['n_units'],
        'total_area': sample_data['total_area'],
        'n_foods': len(foods),
        'n_variables': len(cqm.variables),
        'n_constraints': len(cqm.constraints),
        'cqm_time': cqm_time,
        'solvers': {}
    }
    
    # 1. Gurobi Solver (MINLP with continuous and binary variables)
    print(f"     Running Gurobi (MINLP)...")
    
    # Check for cached result first (config_id = n_units, run_id = 1)
    cached = check_cached_results('farm', 'gurobi', sample_data['n_units'], run_id=1)
    if cached:
        print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
        results['solvers']['gurobi'] = cached
    else:
        try:
            pulp_start = time.time()
            pulp_model, pulp_results = solver_runner.solve_with_pulp_farm(farms_list, foods, food_groups, config)
            pulp_time = time.time() - pulp_start
            
            gurobi_result = {
                'status': pulp_results['status'],
                'objective_value': pulp_results.get('objective_value'),
                'solve_time': pulp_time,
                'solver_time': pulp_results.get('solve_time', pulp_time),
                'success': pulp_results['status'] == 'Optimal',
                'sample_id': sample_data['sample_id'],
                'n_units': sample_data['n_units'],
                'total_area': sample_data['total_area'],
                'n_foods': len(foods),
                'n_variables': len(cqm.variables),
                'n_constraints': len(cqm.constraints)
            }
            
            # Extract solution data for farm scenario
            if pulp_results['status'] == 'Optimal':
                gurobi_result['solution_areas'] = pulp_results.get('areas', {})
                gurobi_result['solution_selections'] = pulp_results.get('selections', {})
                # Calculate total covered area (sum of all areas across farms and foods)
                total_covered_area = sum(gurobi_result['solution_areas'].values())
                gurobi_result['total_covered_area'] = total_covered_area
            
            results['solvers']['gurobi'] = gurobi_result
            
            # Save individual result file (config_id = n_units, run_id = 1)
            save_solver_result(gurobi_result, 'farm', 'gurobi', sample_data['n_units'], run_id=1)
            
            print(f"       ✓ Gurobi: {pulp_results['status']} in {pulp_time:.3f}s")
            
        except Exception as e:
            print(f"       ❌ Gurobi failed: {e}")
            results['solvers']['gurobi'] = {'status': 'Error', 'error': str(e), 'success': False}
    
    # 2. DWave CQM Solver (if token available)
    if dwave_token:
        print(f"     Running DWave CQM...")
        
        # Check for cached result first (config_id = n_units, run_id = 1)
        cached = check_cached_results('farm', 'dwave_cqm', sample_data['n_units'], run_id=1)
        if cached:
            print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
            results['solvers']['dwave_cqm'] = cached
        else:
            try:

                sampleset, solve_time, qpu_time = solver_runner.solve_with_dwave_cqm(cqm, dwave_token)

                # Extract best solution
                if len(sampleset) > 0:
                    best = sampleset.first
                    is_feasible = best.is_feasible
                    cqm_sample = dict(best.sample)
                    
                    # Calculate total covered area (sum of all areas in solution for farm scenario)
                    total_covered_area = sum(v for v in cqm_sample.values() if isinstance(v, (int, float)) and v > 0)
                    
                    dwave_result = {
                        'status': 'Optimal' if is_feasible else 'Infeasible',
                        'objective_value': -best.energy / total_covered_area if is_feasible else None,
                        'qpu_time': qpu_time,
                        'solve_time': solve_time,
                        'is_feasible': is_feasible,
                        'num_samples': len(sampleset),
                        'success': is_feasible,
                        'sample_id': sample_data['sample_id'],
                        'n_units': sample_data['n_units'],
                        'total_area': sample_data['total_area'],
                        'n_foods': len(foods),
                        'n_variables': len(cqm.variables),
                        'n_constraints': len(cqm.constraints),
                        'total_covered_area': total_covered_area,
                        'solution_areas': cqm_sample
                    }
                    
                    print(f"       ✓ DWave CQM: {'Feasible' if is_feasible else 'Infeasible'} in {solve_time:.3f}s")
                else:
                    dwave_result = {
                        'status': 'No Solutions',
                        'success': False,
                        'solve_time': solve_time
                    }
                    print(f"       ❌ DWave CQM: No solutions returned")
                
                results['solvers']['dwave_cqm'] = dwave_result
                
                # Save individual result file (config_id = n_units, run_id = 1)
                save_solver_result(dwave_result, 'farm', 'dwave_cqm', sample_data['n_units'], run_id=1)
                
            except Exception as e:
                print(f"       ❌ DWave CQM failed: {e}")
                results['solvers']['dwave_cqm'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     DWave CQM: SKIPPED (no token)")
        results['solvers']['dwave_cqm'] = {'status': 'Skipped', 'success': False}
    
    return results

def run_binary_scenario(sample_data: Dict, dwave_token: Optional[str] = None) -> Dict:
    """
    Run Patch Scenario: Binary formulation (even_grid) with pure binary variables.
    
    Uses solver_runner_BINARY.py's create_cqm_plots() and solve_with_pulp_plots().
    
    Solvers:
    - Gurobi (PuLP): BIP solver for binary plot assignments
    - D-Wave CQM: Quantum-classical hybrid for constrained quadratic models
    - Gurobi QUBO: Native QUBO solver after CQM→BQM conversion
    - D-Wave BQM: Quantum annealer with higher QPU utilization
    
    Args:
        sample_data: Patch sample data with even_grid land method
        dwave_token: D-Wave API token (optional)
        
    Returns:
        Dictionary with results for all solvers
    """
    print(f"\n  📊 PATCH SCENARIO (Binary) - Sample {sample_data['sample_id']}")
    print(f"     {sample_data['n_units']} plots, {sample_data['total_area']:.1f} ha")
    print(f"     Method: {sample_data['land_method']}")
    
    # Create problem setup
    land_data = sample_data['data']
    plots_list = list(land_data.keys())
    foods, food_groups, config = create_food_config(land_data, 'patch')
    
    # Create CQM using BINARY formulation (pure binary)
    cqm_start = time.time()
    cqm, Y, constraint_metadata = solver_runner.create_cqm_plots(plots_list, foods, food_groups, config)
    cqm_time = time.time() - cqm_start
    
    results = {
        'sample_id': sample_data['sample_id'],
        'scenario_type': 'patch',
        'land_method': sample_data['land_method'],
        'n_units': sample_data['n_units'],
        'total_area': sample_data['total_area'],
        'n_foods': len(foods),
        'n_variables': len(cqm.variables),
        'n_constraints': len(cqm.constraints),
        'cqm_time': cqm_time,
        'solvers': {}
    }
    
    # 1. Gurobi Solver (BIP with only binary variables)
    print(f"     Running Gurobi (BIP)...")
    
    # Check for cached result first
    cached = check_cached_results('patch', 'gurobi', sample_data['n_units'], run_id=1)
    if cached:
        print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
        results['solvers']['gurobi'] = cached
    else:
        try:
            pulp_start = time.time()
            pulp_model, pulp_results = solver_runner.solve_with_pulp_plots(plots_list, foods, food_groups, config)
            pulp_time = time.time() - pulp_start
            
            gurobi_result = {
                'status': pulp_results['status'],
                'objective_value': pulp_results.get('objective_value'),
                'solve_time': pulp_time,
                'solver_time': pulp_results.get('solve_time', pulp_time),
                'success': pulp_results['status'] == 'Optimal',
                'sample_id': sample_data['sample_id'],
                'n_units': sample_data['n_units'],
                'total_area': sample_data['total_area'],
                'n_foods': len(foods),
                'n_variables': len(cqm.variables),
                'n_constraints': len(cqm.constraints)
            }
            
            # Extract solution data (binary plantations)
            if pulp_results['status'] == 'Optimal':
                gurobi_result['solution_plantations'] = pulp_results.get('plantations', {})
                # Calculate total covered area (number of plots selected * area per plot)
                num_plots_selected = sum(1 for v in gurobi_result['solution_plantations'].values() if v > 0.5)
                total_covered_area = num_plots_selected * (sample_data['total_area'] / sample_data['n_units'])
                gurobi_result['total_covered_area'] = total_covered_area
            
            results['solvers']['gurobi'] = gurobi_result
            
            # Save individual result file
            save_solver_result(gurobi_result, 'patch', 'gurobi', sample_data['n_units'], run_id=1)
            
            print(f"       ✓ Gurobi: {pulp_results['status']} in {pulp_time:.3f}s")
            
        except Exception as e:
            print(f"       ❌ Gurobi failed: {e}")
            results['solvers']['gurobi'] = {'status': 'Error', 'error': str(e), 'success': False}
    
    # 2. DWave CQM Solver (if token available)
    if dwave_token:
        print(f"     Running DWave CQM...")
        
        cached = check_cached_results('patch', 'dwave_cqm', sample_data['n_units'], run_id=1)
        if cached:
            print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
            results['solvers']['dwave_cqm'] = cached
        else:
            try:
                sampleset, hybrid_time, qpu_time = solver_runner.solve_with_dwave_cqm(cqm, dwave_token)
                
                # Extract best solution
                if len(sampleset) > 0:
                    best = sampleset.first
                    is_feasible = best.is_feasible
                    cqm_sample = dict(best.sample)
                    
                    # Calculate total covered area (number of plots selected * area per plot)
                    num_plots_selected = sum(1 for v in cqm_sample.values() if v > 0.5)
                    total_covered_area = num_plots_selected * (sample_data['total_area'] / sample_data['n_units'])
                    
                    # Normalize objective by total area to match PuLP formulation
                    normalized_objective = -best.energy / sample_data['total_area'] if is_feasible else None
                    
                    # Validate constraints
                    validation_result = solver_runner.validate_solution_constraints(
                        cqm_sample, plots_list, foods, food_groups, land_data, config
                    )
                    
                    dwave_result = {
                        'status': 'Feasible' if is_feasible else 'Infeasible',
                        'objective_value': normalized_objective,
                        'hybrid_time': hybrid_time,
                        'qpu_time': qpu_time,
                        'is_feasible': is_feasible,
                        'num_samples': len(sampleset),
                        'success': is_feasible,
                        'sample_id': sample_data['sample_id'],
                        'n_units': sample_data['n_units'],
                        'total_area': sample_data['total_area'],
                        'n_foods': len(foods),
                        'n_variables': len(cqm.variables),
                        'n_constraints': len(cqm.constraints),
                        'total_covered_area': total_covered_area,
                        'solution_plantations': cqm_sample,
                        'validation': validation_result
                    }
                    
                    # Print result with validation info
                    if validation_result['n_violations'] > 0:
                        print(f"       ⚠️  DWave CQM: {validation_result['n_violations']} constraint violations in {hybrid_time:.3f}s")
                    else:
                        print(f"       ✅ DWave CQM: Feasible in {hybrid_time:.3f}s (obj: {normalized_objective:.6f})")
                else:
                    dwave_result = {
                        'status': 'No Solutions',
                        'success': False,
                        'solve_time': hybrid_time if hybrid_time else 0
                    }
                    print(f"       ❌ DWave CQM: No solutions returned")
                
                results['solvers']['dwave_cqm'] = dwave_result
                save_solver_result(dwave_result, 'patch', 'dwave_cqm', sample_data['n_units'], run_id=1)
                
            except Exception as e:
                print(f"       ❌ DWave CQM failed: {e}")
                results['solvers']['dwave_cqm'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     DWave CQM: SKIPPED (no token)")
        results['solvers']['dwave_cqm'] = {'status': 'Skipped', 'success': False}
    
    # 3. Convert CQM to BQM for additional solvers
    bqm = None
    invert = None
    bqm_conversion_time = None
    
    print(f"     Converting CQM to BQM...")
    try:
        bqm_start = time.time()
        lagrange_multiplier = 100000.0  # Very large multiplier to strongly enforce constraints
        print(f"       Using Lagrange multiplier: {lagrange_multiplier}")
        
        bqm, invert = cqm_to_bqm(cqm, lagrange_multiplier=lagrange_multiplier)
        bqm_conversion_time = time.time() - bqm_start
        print(f"       ✓ BQM: {len(bqm.variables)} vars, {len(bqm.quadratic)} interactions ({bqm_conversion_time:.3f}s)")
    except Exception as e:
        print(f"       ❌ BQM conversion failed: {e}")
    
    # 4. DWave BQM Solver (if BQM available and token)
    if bqm is not None and dwave_token:
        print(f"     Running DWave BQM...")
        
        cached = check_cached_results('patch', 'dwave_bqm', sample_data['n_units'], run_id=1)
        if cached:
            print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
            results['solvers']['dwave_bqm'] = cached
        else:
            try:
                print(f"       Submitting to DWave Leap HybridBQM solver...")
                
                sampleset_bqm, hybrid_time, qpu_time, bqm_conv_time, invert_fn = solver_runner.solve_with_dwave_bqm(
                    cqm,  # Pass CQM, function converts internally
                    dwave_token
                )
                
                # Extract best solution
                if len(sampleset_bqm) > 0:
                    best_sample = sampleset_bqm.first
                    bqm_energy = best_sample.energy
                    
                    # Invert BQM solution back to CQM
                    cqm_sample_raw = invert_fn(best_sample.sample)
                    
                    # Convert string values to integers (DWave returns strings)
                    cqm_sample = {k: int(v) if isinstance(v, str) else v for k, v in cqm_sample_raw.items()}
                    
                    # Calculate total covered area (number of plots selected * area per plot)
                    num_plots_selected = sum(1 for v in cqm_sample.values() if v > 0.5)
                    total_covered_area = num_plots_selected * (sample_data['total_area'] / sample_data['n_units'])
                    
                    # Normalize objective by total area to match PuLP formulation (POSITIVE sign)
                    normalized_objective = -bqm_energy / sample_data['total_area']
                    
                    # Validate constraints
                    validation_result = solver_runner.validate_solution_constraints(
                        cqm_sample, plots_list, foods, food_groups, land_data, config
                    )
                    
                    dwave_bqm_result = {
                        'status': 'Optimal',
                        'objective_value': normalized_objective,
                        'bqm_energy': bqm_energy,
                        'solve_time': hybrid_time,
                        'hybrid_time': hybrid_time,
                        'qpu_time': qpu_time,
                        'bqm_conversion_time': bqm_conv_time,
                        'success': True,
                        'sample_id': sample_data['sample_id'],
                        'n_units': sample_data['n_units'],
                        'total_area': sample_data['total_area'],
                        'n_foods': len(foods),
                        'n_variables': len(bqm.variables),
                        'bqm_interactions': len(bqm.quadratic),
                        'total_covered_area': total_covered_area,
                        'solution_plantations': cqm_sample,
                        'validation': validation_result
                    }
                    
                    results['solvers']['dwave_bqm'] = dwave_bqm_result
                    save_solver_result(dwave_bqm_result, 'patch', 'dwave_bqm', sample_data['n_units'], run_id=1)
                    
                    # Print result with validation info
                    if validation_result['n_violations'] > 0:
                        print(f"       ⚠️  DWave BQM: {validation_result['n_violations']} constraint violations in {hybrid_time:.3f}s (QPU: {qpu_time:.3f}s)")
                    else:
                        print(f"       ✅ DWave BQM: Optimal in {hybrid_time:.3f}s (QPU: {qpu_time:.3f}s, obj: {normalized_objective:.6f})")
                else:
                    dwave_bqm_result = {
                        'status': 'No Solutions',
                        'success': False,
                        'solve_time': hybrid_time if hybrid_time else 0
                    }
                    print(f"       ❌ DWave BQM: No solutions returned")
                    results['solvers']['dwave_bqm'] = dwave_bqm_result
                
            except Exception as e:
                print(f"       ❌ DWave BQM failed: {e}")
                results['solvers']['dwave_bqm'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     DWave BQM: SKIPPED (no BQM or token)")
        results['solvers']['dwave_bqm'] = {'status': 'Skipped', 'success': False}
    
    # 5. Gurobi QUBO Solver (if BQM available)
    if bqm is not None:
        print(f"     Running Gurobi QUBO...")
        
        cached = check_cached_results('patch', 'gurobi_qubo', sample_data['n_units'], run_id=1)
        if cached:
            print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
            results['solvers']['gurobi_qubo'] = cached
        else:
            try:
                # Use the solve_with_gurobi_qubo function from solver_runner_BINARY
                qubo_start = time.time()
                qubo_result = solver_runner.solve_with_gurobi_qubo(
                    bqm, 
                    farms=plots_list, 
                    foods=foods, 
                    food_groups=food_groups,
                    land_availability=land_data,
                    weights=config['parameters']['weights'],
                    idle_penalty=config['parameters'].get('idle_penalty_lambda', 0.0),
                    config=config,
                    time_limit=100  # 100 second time limit
                )
                qubo_time = time.time() - qubo_start
                
                # Extract solution from result - solver returns 'solution' dict
                bqm_solution = qubo_result.get('solution', {})
                # Convert BQM solution back to CQM variable names for plantations
                cqm_sample = {}
                for var_name, value in bqm_solution.items():
                    # BQM variables are the original Y variables from CQM
                    cqm_sample[var_name] = value
                
                cqm_objective = qubo_result.get('objective_value')
                
                # Calculate total covered area (number of plots selected * area per plot)
                num_plots_selected = sum(1 for v in cqm_sample.values() if v > 0.5)
                total_covered_area = num_plots_selected * (sample_data['total_area'] / sample_data['n_units'])
                
                # Validate constraints
                validation = solver_runner.validate_solution_constraints(
                    cqm_sample, plots_list, foods, food_groups, land_data, config
                )
                
                # Format objective value for printing
                obj_str = f"{cqm_objective:.6f}" if cqm_objective is not None else "N/A"
                violations_str = f", {validation['n_violations']} violations" if validation else ""
                print(f"       ✓ Gurobi QUBO: {qubo_result['status']} in {qubo_time:.3f}s (obj: {obj_str}{violations_str})")
                
                gurobi_qubo_result = {
                    'status': qubo_result['status'],
                    'objective_value': cqm_objective,
                    'bqm_energy': qubo_result.get('bqm_energy'),
                    'solve_time': qubo_time,
                    'bqm_conversion_time': bqm_conversion_time,
                    'success': qubo_result['status'] == 'Optimal',
                    'sample_id': sample_data['sample_id'],
                    'n_units': sample_data['n_units'],
                    'total_area': sample_data['total_area'],
                    'n_foods': len(foods),
                    'n_variables': len(bqm.variables),
                    'bqm_interactions': qubo_result.get('bqm_interactions', 0),
                    'total_covered_area': total_covered_area,
                    'solution_plantations': cqm_sample,
                    'validation': validation
                }
                
                results['solvers']['gurobi_qubo'] = gurobi_qubo_result
                save_solver_result(gurobi_qubo_result, 'patch', 'gurobi_qubo', sample_data['n_units'], run_id=1)
                
            except Exception as e:
                print(f"       ❌ Gurobi QUBO failed: {e}")
                import traceback
                traceback.print_exc()
                results['solvers']['gurobi_qubo'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     Gurobi QUBO: SKIPPED (no BQM)")
        results['solvers']['gurobi_qubo'] = {'status': 'Skipped', 'success': False}
    
    return results

def run_comprehensive_benchmark(config_values: List[int], dwave_token: Optional[str] = None) -> Dict:
    """
    Run the comprehensive benchmark for the given configuration values.
    
    Args:
        config_values: List of configuration values (number of units to test)
        dwave_token: D-Wave API token (optional)
        
    Returns:
        Complete benchmark results dictionary
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE BENCHMARK - CONFIGS: {config_values}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Generate PAIRED sample data
    paired_samples = generate_sample_data(config_values)
    
    farm_results = []
    patch_results = []

    # Iterate over the paired samples
    for farm_sample, patch_sample in paired_samples:
        print(f"\n{'='*80}")
        print(f"PROCESSING SCENARIO FOR {farm_sample['n_units']} UNITS (Sample ID: {farm_sample['sample_id']})")
        print(f"{'='*80}")

        # Run Farm Scenario (Continuous)
        try:
            farm_result = run_farm_scenario(farm_sample, dwave_token)
            farm_results.append(farm_result)
        except Exception as e:
            print(f"  ❌ Farm sample {farm_sample['sample_id']} failed: {e}")

        # Run Binary Scenario (Discretized)
        try:
            patch_result = run_binary_scenario(patch_sample, dwave_token)
            patch_results.append(patch_result)
        except Exception as e:
            print(f"  ❌ Binary sample {patch_sample['sample_id']} failed: {e}")
    
    total_time = time.time() - start_time
    
    # Compile comprehensive results
    benchmark_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'config_values': config_values,
            'total_runtime': total_time,
            'dwave_enabled': dwave_token is not None,
            'scenarios': ['farm', 'patch'],
            'solvers': {
                'farm': ['gurobi', 'dwave_cqm'],
                'patch': ['gurobi', 'dwave_cqm', 'gurobi_qubo', 'dwave_bqm']
            }
        },
        'farm_results': farm_results,
        'patch_results': patch_results,
        'summary': {
            'farm_samples_completed': len(farm_results),
            'patch_samples_completed': len(patch_results),
            'total_solver_runs': sum(len(r['solvers']) for r in farm_results + patch_results)
        }
    }
    
    return benchmark_results

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive benchmark comparing quantum and classical solvers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python comprehensive_benchmark.py 5                    # 5 samples, no D-Wave
  python comprehensive_benchmark.py --configs            # Use BENCHMARK_CONFIGS, no D-Wave
  python comprehensive_benchmark.py 10 --dwave           # 10 samples with D-Wave
  python comprehensive_benchmark.py --configs --dwave    # Use configs with D-Wave
  python comprehensive_benchmark.py 3 --output my_results.json
        '''
    )
    
    parser.add_argument('n_samples', type=int, nargs='?', default=None,
                       help='Number of samples to generate for each scenario (overrides --configs)')
    parser.add_argument('--configs', action='store_true',
                       help='Use predefined BENCHMARK_CONFIGS instead of n_samples')
    parser.add_argument('--dwave', action='store_true',
                       help='Enable D-Wave solvers (requires DWAVE_API_TOKEN)')
    parser.add_argument('--output', type=str, 
                       default=None,
                       help='Output JSON filename (default: auto-generated)')
    parser.add_argument('--token', type=str,
                       help='D-Wave API token (overrides environment variable)')
    
    args = parser.parse_args()
    
    # Default behavior: if no arguments provided, use --configs and --dwave
    if len(sys.argv) == 1:
        print("No arguments provided. Using default: --configs --dwave")
        args.configs = True
        args.dwave = True
    
    # Determine sample configurations
    if args.configs:
        # Use predefined configurations
        sample_configs = BENCHMARK_CONFIGS
        print(f"Using predefined BENCHMARK_CONFIGS: {sample_configs}")
    elif args.n_samples is not None:
        # Use single n_samples value
        sample_configs = [args.n_samples]
    else:
        print("Error: Must specify either n_samples or --configs")
        parser.print_help()
        sys.exit(1)
    
    # Validate inputs
    for n in sample_configs:
        if n < 1:
            print(f"Error: n_samples ({n}) must be at least 1")
            sys.exit(1)
    
    if max(sample_configs) > 50:
        print(f"Warning: Large number of samples ({max(sample_configs)}) may take significant time and D-Wave budget")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Set up D-Wave token
    dwave_token = None
    if args.dwave:
        if args.token:
            dwave_token = args.token
        else:
            dwave_token = os.getenv('DWAVE_API_TOKEN', '45FS-7b81782896495d7c6a061bda257a9d9b03b082cd')
        
        if not dwave_token:
            print("Warning: D-Wave enabled but no token found. Set DWAVE_API_TOKEN environment variable or use --token")
            print("Continuing with classical solvers only...")
        else:
            print(f"✓ D-Wave token configured (length: {len(dwave_token)})")
    
    # Generate output filename and ensure Benchmarks directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmarks_dir = os.path.join(script_dir, "Benchmarks", "COMPREHENSIVE")
    os.makedirs(benchmarks_dir, exist_ok=True)
    
    if args.output:
        # If user provides relative path, put it in Benchmarks/COMPREHENSIVE
        if not os.path.isabs(args.output):
            output_filename = os.path.join(benchmarks_dir, args.output)
        else:
            output_filename = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dwave_suffix = "_dwave" if dwave_token else "_classical"
        if args.configs:
            config_str = "configs"
        else:
            config_str = f"{sample_configs[0]}samples"
        output_filename = os.path.join(benchmarks_dir, f"comprehensive_benchmark_{config_str}{dwave_suffix}_{timestamp}.json")
    
    print(f"\nRunning comprehensive benchmark:")
    print(f"  Configurations: {sample_configs}")
    print(f"  D-Wave: {'Enabled' if dwave_token else 'Disabled'}")
    print(f"  Output: {output_filename}")
    
    # Run benchmark for all configurations
    try:
        print(f"\n{'='*80}")
        print(f"RUNNING BENCHMARK WITH CONFIGS: {sample_configs}")
        print(f"{'='*80}")
        
        results = run_comprehensive_benchmark(sample_configs, dwave_token)
        
        final_results = results
        
        # Save results
        print(f"\n{'='*80}")
        print("SAVING RESULTS")
        print(f"{'='*80}")
        
        with open(output_filename, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {output_filename}")
        
        # Print summary
        print(f"   Total runtime: {final_results['metadata']['total_runtime']:.1f} seconds")
        print(f"   Farm samples: {final_results['summary']['farm_samples_completed']}")
        print(f"   Patch samples: {final_results['summary']['patch_samples_completed']}")
        print(f"   Total solver runs: {final_results['summary']['total_solver_runs']}")
        
        print(f"\nNext steps:")
        print(f"  1. Run plotting: python plot_comprehensive_results.py {output_filename}")
        print(f"  2. Analyze results in the JSON file")
        
    except KeyboardInterrupt:
        print(f"\n\n❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()