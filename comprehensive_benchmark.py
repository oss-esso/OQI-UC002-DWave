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
from farm_sampler import generate_farms as generate_farms_large
from patch_sampler import generate_farms as generate_patches_small
from src.scenarios import load_food_data

# Import solvers
from solver_runner_PATCH import (
    create_cqm, solve_with_pulp, solve_with_dwave, solve_with_dwave_cqm,
    solve_with_simulated_annealing, solve_with_gurobi_qubo,
    calculate_original_objective, extract_solution_summary,
    validate_solution_constraints
)
from dimod import cqm_to_bqm

# Benchmark configurations
# Format: number of units (farms or patches) to test
BENCHMARK_CONFIGS = [
    #10,
    #15,
    #20,
    #25,
    50,
    #100
]

# Number of runs per configuration for statistical analysis
NUM_RUNS = 1

# Gurobi QUBO timeout in seconds
GUROBI_QUBO_TIMEOUT = 30  # 300 seconds (5 minutes) to match TimeLimit in solver

def generate_sample_data(config_values: List[int], seed_offset: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate samples for each configuration value of farms and patches in parallel.
    
    Args:
        config_values: List of configuration values (number of units to generate)
        seed_offset: Offset for random seed to ensure variety
        
    Returns:
        Tuple of (farms_list, patches_list) with areas
    """
    print(f"\n{'='*80}")
    print(f"GENERATING SAMPLES FOR CONFIGS: {config_values}")
    print(f"{'='*80}")
    
    farms_list = []
    patches_list = []
    
    def generate_farm_sample(sample_idx, n_farms):
        """Generate a single farm sample with specified number of farms."""
        seed = 42 + seed_offset + sample_idx * 100
        farms = generate_farms_large(n_farms=n_farms, seed=seed)
        total_area = sum(farms.values())
        return {
            'sample_id': sample_idx,
            'type': 'farm',
            'data': farms,
            'total_area': total_area,
            'n_units': len(farms),
            'seed': seed
        }
    
    def generate_patch_sample(sample_idx, n_patches):
        """Generate a single patch sample with specified number of patches."""
        seed = 42 + seed_offset + sample_idx * 100 + 50
        patches = generate_patches_small(n_farms=n_patches, seed=seed)
        total_area = sum(patches.values())
        return {
            'sample_id': sample_idx,
            'type': 'patch', 
            'data': patches,
            'total_area': total_area,
            'n_units': len(patches),
            'seed': seed
        }
    
    # Generate samples in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit farm generation tasks for each config value
        farm_futures = [executor.submit(generate_farm_sample, i, n_units) for i, n_units in enumerate(config_values)]
        patch_futures = [executor.submit(generate_patch_sample, i, n_units) for i, n_units in enumerate(config_values)]
        
        # Collect farm results
        print(f"  Generating {len(config_values)} farm samples...")
        for future in as_completed(farm_futures):
            try:
                result = future.result()
                farms_list.append(result)
                print(f"    ✓ Farm sample {result['sample_id']}: {result['n_units']} farms, {result['total_area']:.1f} ha")
            except Exception as e:
                print(f"    ❌ Farm sample failed: {e}")
        
        # Collect patch results  
        print(f"  Generating {len(config_values)} patch samples...")
        for future in as_completed(patch_futures):
            try:
                result = future.result()
                patches_list.append(result)
                print(f"    ✓ Patch sample {result['sample_id']}: {result['n_units']} patches, {result['total_area']:.1f} ha")
            except Exception as e:
                print(f"    ❌ Patch sample failed: {e}")
    
    # Sort by sample_id for consistency
    farms_list.sort(key=lambda x: x['sample_id'])
    patches_list.sort(key=lambda x: x['sample_id'])
    
    print(f"\n  ✅ Generated {len(farms_list)} farm samples and {len(patches_list)} patch samples")
    return farms_list, patches_list

def create_food_config(land_data: Dict[str, float], scenario_type: str = 'comprehensive') -> Tuple[Dict, Dict, Dict]:
    """
    Create food and configuration data compatible with solvers.
    
    Args:
        land_data: Dictionary of land availability
        scenario_type: Type of scenario to create
        
    Returns:
        Tuple of (foods, food_groups, config)
    """
    # Load food data from scenarios module
    try:
        food_list, foods, food_groups, _ = load_food_data('simple')
    except Exception as e:
        print(f"    Warning: Food data loading failed ({e}), using fallback")
        foods, food_groups = create_fallback_foods()
    
    # Create configuration matching original solver_runner.py formulation
    config = {
        'parameters': {
            'land_availability': land_data,
            'minimum_planting_area': {food: 0.0 for food in foods},  # Min area per crop
            # NO max_percentage_per_crop - this was the bug!
            'food_group_constraints': {},  # Can add min/max crops per food group
            'weights': {
                'nutritional_value': 0.25,
                'nutrient_density': 0.2,
                'environmental_impact': 0.25,
                'affordability': 0.15,
                'sustainability': 0.15
            },
            'idle_penalty_lambda': 0.1
        }
    }
    
    return foods, food_groups, config

def create_fallback_foods():
    """Create fallback food data if Excel is not available."""
    foods = {
        'Wheat': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.6, 'affordability': 0.9, 'sustainability': 0.7},
        'Corn': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.5, 'affordability': 0.8, 'sustainability': 0.6},
        'Rice': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.7, 'affordability': 0.7, 'sustainability': 0.8},
        'Soybeans': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.4, 'affordability': 0.6, 'sustainability': 0.9},
        'Potatoes': {'nutritional_value': 0.5, 'nutrient_density': 0.4, 'environmental_impact': 0.8, 'affordability': 0.9, 'sustainability': 0.6}
    }
    food_groups = {
        'grains': ['Wheat', 'Corn', 'Rice'],
        'proteins': ['Soybeans'],
        'vegetables': ['Potatoes']
    }
    return foods, food_groups

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
    Run Farm Scenario: DWaveCQM and PuLP/Gurobi solvers.
    
    Args:
        sample_data: Farm sample data
        dwave_token: D-Wave API token (optional)
        
    Returns:
        Dictionary with results for both solvers
    """
    print(f"\n  🏢 FARM SCENARIO - Sample {sample_data['sample_id']}")
    print(f"     {sample_data['n_units']} farms, {sample_data['total_area']:.1f} ha")
    
    # Create problem setup
    land_data = sample_data['data']
    foods, food_groups, config = create_food_config(land_data, 'farm')
    
    # Create CQM
    cqm_start = time.time()
    cqm, (X, Y), constraint_metadata = create_cqm(land_data, foods, food_groups, config)
    cqm_time = time.time() - cqm_start
    
    results = {
        'sample_id': sample_data['sample_id'],
        'scenario_type': 'farm',
        'n_units': sample_data['n_units'],
        'total_area': sample_data['total_area'],
        'n_foods': len(foods),
        'n_variables': len(cqm.variables),
        'n_constraints': len(cqm.constraints),
        'cqm_time': cqm_time,
        'solvers': {}
    }
    
    # 1. Gurobi Solver
    print(f"     Running Gurobi...")
    
    # Check for cached result first (config_id = n_units, run_id = 1)
    cached = check_cached_results('farm', 'gurobi', sample_data['n_units'], run_id=1)
    if cached:
        print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
        results['solvers']['gurobi'] = cached
    else:
        try:
            pulp_start = time.time()
            pulp_model, pulp_results = solve_with_pulp(land_data, foods, food_groups, config)
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
                # Use NATIVE CQM solver - no BQM conversion!
                sampleset, hybrid_time, qpu_time = solve_with_dwave_cqm(cqm, dwave_token)
                
                success = len(sampleset) > 0
                objective_value = None
                if success:
                    best = sampleset.first
                    # For farm scenario, we can calculate objective from CQM sample
                    objective_value = -best.energy  # Assuming energy is negative of objective
                
                dwave_result = {
                    'status': 'Optimal' if success else 'No solution',
                    'objective_value': objective_value,
                    'solve_time': hybrid_time if hybrid_time is not None else 0,
                    'qpu_time': qpu_time if qpu_time is not None else 0,
                    'hybrid_time': hybrid_time if hybrid_time is not None else 0,
                    'success': success,
                    'sample_id': sample_data['sample_id'],
                    'n_units': sample_data['n_units'],
                    'total_area': sample_data['total_area'],
                    'n_foods': len(foods),
                    'n_variables': len(cqm.variables),
                    'n_constraints': len(cqm.constraints)
                }
                
                results['solvers']['dwave_cqm'] = dwave_result
                
                # Save individual result file (config_id = n_units, run_id = 1)
                save_solver_result(dwave_result, 'farm', 'dwave_cqm', sample_data['n_units'], run_id=1)
                
                print(f"       ✓ DWave CQM: {'Optimal' if success else 'No solution'} in {hybrid_time:.3f}s")
                
            except Exception as e:
                print(f"       ❌ DWave CQM failed: {e}")
                results['solvers']['dwave_cqm'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     DWave CQM: SKIPPED (no token)")
        results['solvers']['dwave_cqm'] = {'status': 'Skipped', 'success': False}
    
    return results

def run_patch_scenario(sample_data: Dict, dwave_token: Optional[str] = None) -> Dict:
    """
    Run Patch Scenario: PuLP, DWaveCQM, DWaveBQM, and Gurobi QUBO solvers.
    
    Args:
        sample_data: Patch sample data  
        dwave_token: D-Wave API token (optional)
        
    Returns:
        Dictionary with results for all solvers
    """
    print(f"\n  🏞️  PATCH SCENARIO - Sample {sample_data['sample_id']}")
    print(f"     {sample_data['n_units']} patches, {sample_data['total_area']:.1f} ha")
    
    # Create problem setup
    land_data = sample_data['data']
    foods, food_groups, config = create_food_config(land_data, 'patch')
    
    # Create CQM
    cqm_start = time.time()
    cqm, (X, Y), constraint_metadata = create_cqm(land_data, foods, food_groups, config)
    cqm_time = time.time() - cqm_start
    
    results = {
        'sample_id': sample_data['sample_id'],
        'scenario_type': 'patch',
        'n_units': sample_data['n_units'],
        'total_area': sample_data['total_area'],
        'n_foods': len(foods),
        'n_variables': len(cqm.variables),
        'n_constraints': len(cqm.constraints),
        'cqm_time': cqm_time,
        'solvers': {}
    }
    
    # 1. Gurobi Solver
    print(f"     Running Gurobi...")
    
    # Check for cached result first (config_id = n_units, run_id = 1)
    cached = check_cached_results('patch', 'gurobi', sample_data['n_units'], run_id=1)
    if cached:
        print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
        results['solvers']['gurobi'] = cached
    else:
        try:
            pulp_start = time.time()
            pulp_model, pulp_results = solve_with_pulp(land_data, foods, food_groups, config)
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
            
            results['solvers']['gurobi'] = gurobi_result
            
            # Save individual result file (config_id = n_units, run_id = 1)
            save_solver_result(gurobi_result, 'patch', 'gurobi', sample_data['n_units'], run_id=1)
            
            print(f"       ✓ Gurobi: {pulp_results['status']} in {pulp_time:.3f}s")
            
        except Exception as e:
            print(f"       ❌ Gurobi failed: {e}")
            results['solvers']['gurobi'] = {'status': 'Error', 'error': str(e), 'success': False}
    
    # 2. DWave CQM Solver (if token available)
    if dwave_token:
        print(f"     Running DWave CQM...")
        
        # Check for cached result first (config_id = n_units, run_id = 1)
        cached = check_cached_results('patch', 'dwave_cqm', sample_data['n_units'], run_id=1)
        if cached:
            print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
            results['solvers']['dwave_cqm'] = cached
        else:
            try:
                # Use NATIVE CQM solver - no BQM conversion!
                sampleset_cqm, hybrid_time_cqm, qpu_time_cqm = solve_with_dwave_cqm(cqm, dwave_token)
                
                success = len(sampleset_cqm) > 0
                objective_value = None
                if success:
                    best = sampleset_cqm.first
                    objective_value = -best.energy
                
                dwave_result = {
                    'status': 'Optimal' if success else 'No solution',
                    'objective_value': objective_value,
                    'solve_time': hybrid_time_cqm if hybrid_time_cqm is not None else 0,
                    'qpu_time': qpu_time_cqm if qpu_time_cqm is not None else 0,
                    'hybrid_time': hybrid_time_cqm if hybrid_time_cqm is not None else 0,
                    'success': success,
                    'sample_id': sample_data['sample_id'],
                    'n_units': sample_data['n_units'],
                    'total_area': sample_data['total_area'],
                    'n_foods': len(foods),
                    'n_variables': len(cqm.variables),
                    'n_constraints': len(cqm.constraints)
                }
                
                results['solvers']['dwave_cqm'] = dwave_result
                
                # Save individual result file (config_id = n_units, run_id = 1)
                save_solver_result(dwave_result, 'patch', 'dwave_cqm', sample_data['n_units'], run_id=1)
                
                print(f"       ✓ DWave CQM: {'Optimal' if success else 'No solution'} in {hybrid_time_cqm:.3f}s")
                
            except Exception as e:
                print(f"       ❌ DWave CQM failed: {e}")
                results['solvers']['dwave_cqm'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     DWave CQM: SKIPPED (no token)")
        results['solvers']['dwave_cqm'] = {'status': 'Skipped', 'success': False}
    
    # Convert CQM to BQM for BQM-based solvers
    bqm = None
    invert = None
    bqm_conversion_time = None
    
    # Always convert to BQM to allow Gurobi QUBO to run
    print(f"     Converting CQM to BQM...")
    try:
        bqm_start = time.time()
        
        # Manually set a strong Lagrange multiplier to enforce constraints.
        # The automatic multiplier was found to be too weak for this complex formulation,
        # leading to constraint violations where multiple crops were assigned to the same plot.
        # A value of 25 is chosen to be significantly larger than the objective coefficients,
        # ensuring that violating a constraint is always more "expensive" than respecting it.
        
        lagrange_multiplier = 150.0
        print(f"       Using manual Lagrange multiplier: {lagrange_multiplier}")
        
        bqm, invert = cqm_to_bqm(cqm, lagrange_multiplier=lagrange_multiplier)
        bqm_conversion_time = time.time() - bqm_start
        print(f"       ✓ BQM: {len(bqm.variables)} vars, {len(bqm.quadratic)} interactions ({bqm_conversion_time:.3f}s)")
    except Exception as e:
        print(f"       ❌ BQM conversion failed: {e}")
    
    # 3. DWave BQM Solver (if BQM available)
    if bqm is not None and dwave_token:
        print(f"     Running DWave BQM...")
        
        # Check for cached result first (config_id = n_units, run_id = 1)
        cached = check_cached_results('patch', 'dwave_bqm', sample_data['n_units'], run_id=1)
        if cached:
            print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
            results['solvers']['dwave_bqm'] = cached
        else:
            try:
                from dwave.system import LeapHybridBQMSampler
                sampler = LeapHybridBQMSampler(token=dwave_token)
                
                dwave_bqm_start = time.time()
                sampleset_bqm = sampler.sample(bqm, label="Comprehensive Benchmark - BQM")
                dwave_bqm_time = time.time() - dwave_bqm_start
                
                success = len(sampleset_bqm) > 0
                bqm_energy = None
                original_objective = None
                solution_summary = None
                validation = None
                qpu_time_bqm = None
                
                if success:
                    best = sampleset_bqm.first
                    bqm_energy = best.energy  # This is BQM energy, not original objective
                    
                    # Extract solution and calculate original objective
                    solution = dict(best.sample)
                    original_objective = calculate_original_objective(
                        solution,
                        farms=list(land_data.keys()),
                        foods=foods,
                        land_availability=land_data,
                        weights=config['parameters']['weights'],
                        idle_penalty=config['parameters'].get('idle_penalty_lambda', 0.1)
                    )
                    
                    # Extract solution summary
                    solution_summary = extract_solution_summary(solution, list(land_data.keys()), foods, land_data)
                    
                    # Validate constraints
                    validation = validate_solution_constraints(
                        solution, list(land_data.keys()), foods, food_groups, land_data, config
                    )
                    
                    timing_info = sampleset_bqm.info.get('timing', {})
                    
                    # Hybrid solve time (total time including QPU)
                    hybrid_time_bqm = (timing_info.get('run_time') or 
                                      sampleset_bqm.info.get('run_time') or
                                      timing_info.get('charge_time') or
                                      sampleset_bqm.info.get('charge_time'))
                    
                    if hybrid_time_bqm is not None:
                        hybrid_time_bqm = hybrid_time_bqm / 1e6  # Convert from microseconds to seconds
                    else:
                        hybrid_time_bqm = dwave_bqm_time  # Fallback to wall clock time
                    
                    # QPU access time
                    qpu_time_bqm = (timing_info.get('qpu_access_time') or
                                   sampleset_bqm.info.get('qpu_access_time'))
                    
                    if qpu_time_bqm is not None:
                        qpu_time_bqm = qpu_time_bqm / 1e6  # Convert to seconds
                
                dwave_bqm_result = {
                    'status': 'Optimal' if success else 'No solution',
                    'objective_value': original_objective,  # Reconstructed CQM objective
                    'solve_time': hybrid_time_bqm if hybrid_time_bqm is not None else dwave_bqm_time,
                    'qpu_time': qpu_time_bqm,
                    'hybrid_time': hybrid_time_bqm if hybrid_time_bqm is not None else dwave_bqm_time,
                    'bqm_conversion_time': bqm_conversion_time,
                    'bqm_energy': bqm_energy,
                    'success': success,
                    'sample_id': sample_data['sample_id'],
                    'n_units': sample_data['n_units'],
                    'total_area': sample_data['total_area'],
                    'n_foods': len(foods),
                    'n_variables': len(bqm.variables),
                    'n_quadratic': len(bqm.quadratic),
                    'note': 'objective_value is reconstructed from BQM solution; comparable to CQM'
                }
                
                # Add solution summary if available
                if solution_summary is not None:
                    dwave_bqm_result['solution_summary'] = solution_summary
                
                # Add validation if available
                if validation is not None:
                    dwave_bqm_result['validation'] = {
                        'is_feasible': validation['is_feasible'],
                        'n_violations': validation['n_violations'],
                        'violations': validation['violations'],
                        'summary': validation['summary']
                    }
                
                results['solvers']['dwave_bqm'] = dwave_bqm_result
                
                # Save individual result file (config_id = n_units, run_id = 1)
                save_solver_result(dwave_bqm_result, 'patch', 'dwave_bqm', sample_data['n_units'], run_id=1)
                
                print(f"       ✓ DWave BQM: {'Optimal' if success else 'No solution'} in {dwave_bqm_time:.3f}s")
                
            except Exception as e:
                print(f"       ❌ DWave BQM failed: {e}")
                results['solvers']['dwave_bqm'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     DWave BQM: SKIPPED (no BQM available)")
        results['solvers']['dwave_bqm'] = {'status': 'Skipped', 'success': False}
    
    # 4. Gurobi QUBO Solver (if BQM available)
    if bqm is not None:
        print(f"     Running Gurobi QUBO...")
        
        # Check for cached result first (config_id = n_units, run_id = 1)
        cached = check_cached_results('patch', 'gurobi_qubo', sample_data['n_units'], run_id=1)
        if cached:
            print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
            results['solvers']['gurobi_qubo'] = cached
        else:
            try:
                # Pass parameters to calculate original objective and validate
                gurobi_qubo_result_raw = solve_with_gurobi_qubo(
                    bqm,
                    farms=list(land_data.keys()),
                    foods=foods,
                    food_groups=food_groups,
                    land_availability=land_data,
                    weights=config['parameters']['weights'],
                    idle_penalty=config['parameters'].get('idle_penalty_lambda', 0.1),
                    config=config
                )
                
                gurobi_qubo_result = {
                    'status': gurobi_qubo_result_raw['status'],
                    'objective_value': gurobi_qubo_result_raw['objective_value'],  # Reconstructed CQM objective
                    'solve_time': gurobi_qubo_result_raw['solve_time'],
                    'bqm_energy': gurobi_qubo_result_raw['bqm_energy'],
                    'bqm_conversion_time': bqm_conversion_time,
                    'success': gurobi_qubo_result_raw['status'] == 'Optimal',
                    'sample_id': sample_data['sample_id'],
                    'n_units': sample_data['n_units'],
                    'total_area': sample_data['total_area'],
                    'n_foods': len(foods),
                    'n_variables': len(bqm.variables),
                    'n_quadratic': len(bqm.quadratic),
                    'note': 'objective_value is reconstructed from BQM solution; comparable to CQM'
                }
                
                # Add solution summary if available
                if 'solution_summary' in gurobi_qubo_result_raw:
                    gurobi_qubo_result['solution_summary'] = gurobi_qubo_result_raw['solution_summary']
                
                # Add validation if available
                if 'validation' in gurobi_qubo_result_raw:
                    gurobi_qubo_result['validation'] = {
                        'is_feasible': gurobi_qubo_result_raw['validation']['is_feasible'],
                        'n_violations': gurobi_qubo_result_raw['validation']['n_violations'],
                        'violations': gurobi_qubo_result_raw['validation']['violations'],
                        'summary': gurobi_qubo_result_raw['validation']['summary']
                    }
                
                results['solvers']['gurobi_qubo'] = gurobi_qubo_result
                
                # Save individual result file (config_id = n_units, run_id = 1)
                save_solver_result(gurobi_qubo_result, 'patch', 'gurobi_qubo', sample_data['n_units'], run_id=1)
                
                print(f"       ✓ Gurobi QUBO: {gurobi_qubo_result_raw['status']} in {gurobi_qubo_result_raw['solve_time']:.3f}s")
                
            except Exception as e:
                print(f"       ❌ Gurobi QUBO failed: {e}")
                results['solvers']['gurobi_qubo'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     Gurobi QUBO: SKIPPED (no BQM available)")
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
    
    # Generate sample data
    farms_list, patches_list = generate_sample_data(config_values)
    
    # Run Farm Scenarios
    print(f"\n{'='*80}")
    print(f"FARM SCENARIOS ({len(farms_list)} samples)")
    print(f"{'='*80}")
    
    farm_results = []
    for farm_sample in farms_list:
        try:
            result = run_farm_scenario(farm_sample, dwave_token)
            farm_results.append(result)
        except Exception as e:
            print(f"  ❌ Farm sample {farm_sample['sample_id']} failed: {e}")
    
    # Run Patch Scenarios  
    print(f"\n{'='*80}")
    print(f"PATCH SCENARIOS ({len(patches_list)} samples)")
    print(f"{'='*80}")
    
    patch_results = []
    for patch_sample in patches_list:
        try:
            result = run_patch_scenario(patch_sample, dwave_token)
            patch_results.append(result)
        except Exception as e:
            print(f"  ❌ Patch sample {patch_sample['sample_id']} failed: {e}")
    
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
            dwave_token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
        
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