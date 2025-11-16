#!/usr/bin/env python3
"""
3-Period Crop Rotation Benchmark Script

This script benchmarks the 3-period crop rotation optimization problem
with different numbers of plots, testing multiple solvers:

ROTATION Scenario (3-period binary):
- Gurobi (PuLP): BIP solver for 3-period binary plot assignments  
- Pyomo: Native quadratic solver (no McCormick linearization)
- D-Wave CQM: Quantum-classical hybrid for constrained quadratic models
- Gurobi QUBO: Native QUBO solver after CQM→BQM conversion
- D-Wave BQM: Quantum annealer with higher QPU utilization

Solvers tested: Farm PuLP, Farm Pyomo, Patch PuLP, Patch Pyomo, D-Wave CQM, Gurobi QUBO, D-Wave BQM

The implementation follows the binary formulation from crop_rotation.tex:
- Time periods: t ∈ {1, 2, 3}
- Variables: Y_{p,c,t} for plot p, crop c, period t
- Objective: Linear crop values + quadratic rotation synergy
- Constraints: Per-period plot assignments and food group constraints
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import patch generator (plots/patches for rotation model)
from Utils.patch_sampler import generate_farms as generate_patches
from src.scenarios import load_food_data

# Import rotation solver
import solver_runner_ROTATION as solver_runner

from dimod import cqm_to_bqm

# Benchmark configurations - number of plots to test
BENCHMARK_CONFIGS = [
    10,
    15,
    #20,
    25,
    #50,
    #100
]

# Number of runs per configuration
NUM_RUNS = 1

# Default gamma (rotation synergy weight)
DEFAULT_GAMMA = 0.5


def generate_rotation_samples(config_values: List[int], seed_offset: int = 0, fixed_total_land: float = 100.0) -> List[Dict]:
    """
    Generate rotation scenario samples.
    
    Uses fixed total land area for all scenarios.
    
    Args:
        config_values: List of configuration values (number of plots to generate)
        seed_offset: Offset for random seed
        fixed_total_land: Fixed total land area in hectares
        
    Returns:
        List of scenario samples
    """
    print(f"\n{'='*80}")
    print(f"GENERATING ROTATION SAMPLES FOR CONFIGS: {config_values}")
    print(f"Fixed Total Land: {fixed_total_land:.2f} ha (same for all scenarios)")
    print(f"{'='*80}")
    
    samples = []
    
    for i, n_plots in enumerate(config_values):
        print(f"\n--- Generating rotation scenario for {n_plots} plots ({fixed_total_land:.2f} ha) ---")
        
        seed = 42 + seed_offset + i * 100
        patches_unscaled = generate_patches(n_farms=n_plots, seed=seed)
        patches_total = sum(patches_unscaled.values())
        scale_factor = fixed_total_land / patches_total if patches_total > 0 else 0
        patches_scaled = {k: v * scale_factor for k, v in patches_unscaled.items()}
        
        sample = {
            'sample_id': i,
            'type': 'rotation_3period',
            'land_method': 'even_grid',
            'data': patches_scaled,
            'total_area': sum(patches_scaled.values()),
            'n_plots': n_plots,
            'plot_area': fixed_total_land / n_plots,
            'seed': seed
        }
        
        print(f"  ✓ Rotation (3-period): {sample['n_plots']} plots")
        print(f"     Area per plot: {sample['plot_area']:.3f} ha (equal grid)")
        print(f"     Total: {sample['total_area']:.2f} ha")
        
        samples.append(sample)
    
    print(f"\nGenerated {len(samples)} rotation samples (all with {fixed_total_land:.2f} ha total land).")
    return samples


def create_rotation_config(land_data: Dict[str, float]) -> Tuple[Dict, Dict, Dict]:
    """
    Create food and configuration data for rotation model.
    
    Args:
        land_data: Dictionary of land availability per plot
        
    Returns:
        Tuple of (foods, food_groups, config)
    """
    # Load food data
    try:
        food_list, foods, food_groups, _ = load_food_data('full_family')
    except Exception as e:
        print(f"    Warning: Food data loading failed ({e})")
        raise
    
    # Create configuration
    config = {
        'parameters': {
            'land_availability': land_data,
            'minimum_planting_area': {food: 0.0001 for food in foods},
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
            'idle_penalty_lambda': 0.0
        }
    }
    
    return foods, food_groups, config


def check_cached_results(solver_name: str, config_id: int, run_id: int = 1) -> Optional[Dict]:
    """
    Check if a cached result exists for the given configuration.
    
    Args:
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
        'farm_pyomo': 'Farm_Pyomo',
        'farm_dwave_cqm': 'Farm_DWave',
        'patch_gurobi': 'Patch_PuLP',
        'patch_pyomo': 'Patch_Pyomo',
        'patch_dwave_cqm': 'Patch_DWave',
        'patch_gurobi_qubo': 'Patch_GurobiQUBO',
        'patch_dwave_bqm': 'Patch_DWaveBQM'
    }
    
    if solver_name not in solver_dir_map:
        return None
    
    solver_dir = solver_dir_map[solver_name]
    result_dir = os.path.join(script_dir, "Benchmarks", "ROTATION", solver_dir)
    
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


def save_solver_result(result: Dict, solver_name: str, config_id: int, run_id: int = 1):
    """
    Save individual solver result to appropriate subdirectory.
    
    Args:
        result: Solver result dictionary
        solver_name: Name of solver (gurobi, dwave_cqm, gurobi_qubo, dwave_bqm)
        config_id: Configuration ID (number of units)
        run_id: Run number for this configuration
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Map solver names to subdirectory names
    solver_dir_map = {
        'farm_gurobi': 'Farm_PuLP',
        'farm_pyomo': 'Farm_Pyomo',
        'farm_dwave_cqm': 'Farm_DWave',
        'patch_gurobi': 'Patch_PuLP',
        'patch_pyomo': 'Patch_Pyomo',
        'patch_dwave_cqm': 'Patch_DWave',
        'patch_gurobi_qubo': 'Patch_GurobiQUBO',
        'patch_dwave_bqm': 'Patch_DWaveBQM'
    }
    
    if solver_name not in solver_dir_map:
        print(f"  Warning: Unknown solver name {solver_name}, skipping save")
        return
    
    solver_dir = solver_dir_map[solver_name]
    output_dir = os.path.join(script_dir, "Benchmarks", "ROTATION", solver_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename matching other benchmark format
    filename = f"config_{config_id}_run_{run_id}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save result
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2, default=str)


def run_rotation_scenario(sample_data: Dict, dwave_token: Optional[str] = None, gamma: float = DEFAULT_GAMMA) -> Dict:
    """
    Run ROTATION Scenario: 3-period binary formulation with 4 solvers.
    
    Solvers:
    - Gurobi (PuLP): BIP solver for 3-period binary plot assignments
    - D-Wave CQM: Quantum-classical hybrid for constrained quadratic models
    - Gurobi QUBO: Native QUBO solver after CQM→BQM conversion
    - D-Wave BQM: Quantum annealer with higher QPU utilization
    
    Args:
        sample_data: Rotation sample data
        dwave_token: D-Wave API token (optional)
        gamma: Rotation synergy weight coefficient
        
    Returns:
        Dictionary with results for all solvers
    """
    print(f"\n  🔄 ROTATION SCENARIO (3-period) - Sample {sample_data['sample_id']}")
    print(f"     {sample_data['n_plots']} plots, {sample_data['total_area']:.1f} ha")
    print(f"     Rotation synergy weight (gamma): {gamma}")
    
    # Create problem setup
    land_data = sample_data['data']
    plots_list = list(land_data.keys())
    foods, food_groups, config = create_rotation_config(land_data)
    
    # 0. Create 3-period rotation CQM (farm formulation) and run farm solvers
    # NOTE: Farm PuLP (MILP) is slower than patch PuLP (BIP) because:
    #   - Uses continuous area variables A_{f,c,t} + binary indicators Y_{f,c,t} (MILP)
    #   - Requires McCormick relaxation to linearize quadratic rotation terms
    #   - Creates 11,000-27,500 auxiliary binary variables Z for linearization
    #   - Pure binary formulation (patch) is much faster for classical solvers
    #   - Farm formulation is designed for D-Wave CQM which handles continuous variables natively
    print(f"     Creating 3-period rotation CQM (farm formulation)...")
    farm_cqm_start = time.time()
    try:
        farm_cqm, farm_vars, farm_constraint_metadata = solver_runner.create_cqm_farm_rotation_3period(
            plots_list, foods, food_groups, config, gamma=gamma
        )
        farm_cqm_time = time.time() - farm_cqm_start
    except Exception as e:
        print(f"       ❌ Failed to build farm CQM: {e}")
        farm_cqm = None
        farm_cqm_time = 0.0

    # Run Farm PuLP (linearized McCormick) solver
    if farm_cqm is not None:
        print(f"     Running Farm PuLP (BIP/MILP)...")
        try:
            pulp_farm_start = time.time()
            pulp_farm_model, pulp_farm_results = solver_runner.solve_with_pulp_farm_rotation(
                plots_list, foods, food_groups, config, gamma=gamma
            )
            pulp_farm_time = time.time() - pulp_farm_start
            farm_result = {
                'status': pulp_farm_results.get('status'),
                'objective_value': pulp_farm_results.get('objective_value'),
                'solve_time': pulp_farm_time,
                'details': pulp_farm_results
            }
            save_solver_result(farm_result, 'farm_gurobi', sample_data['n_plots'], run_id=1)
            print(f"       ✓ Farm PuLP: {farm_result['status']} in {pulp_farm_time:.3f}s")
        except Exception as e:
            print(f"       ❌ Farm PuLP failed: {e}")
            farm_result = {'status': 'Error', 'error': str(e), 'success': False}

        # Run Farm Pyomo (native quadratic, no linearization)
        print(f"     Running Farm Pyomo...")
        try:
            pyomo_farm_start = time.time()
            pyomo_farm_model, pyomo_farm_results = solver_runner.solve_with_pyomo_farm_rotation(
                plots_list, foods, food_groups, config, gamma=gamma
            )
            pyomo_farm_time = time.time() - pyomo_farm_start
            farm_pyomo_result = {
                'status': pyomo_farm_results.get('status'),
                'objective_value': pyomo_farm_results.get('objective_value'),
                'solve_time': pyomo_farm_time,
                'details': pyomo_farm_results
            }
            save_solver_result(farm_pyomo_result, 'farm_pyomo', sample_data['n_plots'], run_id=1)
            print(f"       ✓ Farm Pyomo: {farm_pyomo_result['status']} in {pyomo_farm_time:.3f}s")
        except Exception as e:
            print(f"       ❌ Farm Pyomo failed: {e}")
            farm_pyomo_result = {'status': 'Error', 'error': str(e), 'success': False}

        # Run Farm D-Wave CQM (if token provided)
        if dwave_token:
            print(f"     Running Farm DWave CQM...")
            try:
                farm_sampleset, farm_hybrid_time, farm_qpu_time = solver_runner.solve_with_dwave_cqm(farm_cqm, dwave_token)
                farm_dwave_result = {
                    'status': 'OK' if len(farm_sampleset) > 0 else 'NoSamples',
                    'sampleset': farm_sampleset,
                    'hybrid_time': farm_hybrid_time,
                    'qpu_time': farm_qpu_time
                }
                save_solver_result(farm_dwave_result, 'farm_dwave_cqm', sample_data['n_plots'], run_id=1)
                print(f"       ✓ Farm DWave CQM returned {len(farm_sampleset)} samples")
            except Exception as e:
                print(f"       ❌ Farm DWave failed: {e}")
                farm_dwave_result = {'status': 'Error', 'error': str(e), 'success': False}
        else:
            farm_dwave_result = {'status': 'Skipped', 'success': False}
    else:
        farm_result = {'status': 'Skipped', 'success': False}
        farm_dwave_result = {'status': 'Skipped', 'success': False}


    # Create 3-period rotation CQM (plots/binary formulation)
    print(f"     Creating 3-period rotation CQM (plots/binary formulation)...")
    cqm_start = time.time()
    cqm, Y, constraint_metadata = solver_runner.create_cqm_plots_rotation_3period(
        plots_list, foods, food_groups, config, gamma=gamma
    )
    cqm_time = time.time() - cqm_start
    
    results = {
        'sample_id': sample_data['sample_id'],
        'scenario_type': 'rotation_3period',
        'n_plots': sample_data['n_plots'],
        'total_area': sample_data['total_area'],
        'n_foods': len(foods),
        'n_periods': 3,
        'n_variables': len(cqm.variables),
        'n_constraints': len(cqm.constraints),
        'gamma': gamma,
        'cqm_time': cqm_time,
        'solvers': {
            'farm_pulp': farm_result,
            'farm_dwave': farm_dwave_result
        }
    }
    
    print(f"       Variables: {results['n_variables']}")
    print(f"       Constraints: {results['n_constraints']}")
    print(f"       CQM build time: {cqm_time:.3f}s")
    
    # 1. Gurobi Solver (BIP for 3-period rotation)
    print(f"     Running Gurobi (BIP)...")
    
    cached = check_cached_results('patch_gurobi', sample_data['n_plots'], run_id=1)
    if cached:
        print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
        results['solvers']['plot_pulp'] = cached
    else:
        try:
            pulp_start = time.time()
            pulp_model, pulp_results = solver_runner.solve_with_pulp_plots_rotation(
                plots_list, foods, food_groups, config, gamma=gamma
            )
            pulp_time = time.time() - pulp_start
            
            gurobi_result = {
                'status': pulp_results['status'],
                'objective_value': pulp_results.get('objective_value'),
                'solve_time': pulp_time,
                'solver_time': pulp_results.get('solve_time', pulp_time),
                'success': pulp_results['status'] == 'Optimal',
                'sample_id': sample_data['sample_id'],
                'n_plots': sample_data['n_plots'],
                'total_area': sample_data['total_area'],
                'n_foods': len(foods),
                'n_variables': len(cqm.variables),
                'n_constraints': len(cqm.constraints)
            }
            
            # Extract solution data
            if pulp_results['status'] == 'Optimal':
                gurobi_result['solution'] = pulp_results.get('solution', {})
            
            results['solvers']['plot_pulp'] = gurobi_result
            save_solver_result(gurobi_result, 'patch_gurobi', sample_data['n_plots'], run_id=1)
            
            print(f"       ✓ Gurobi: {pulp_results['status']} in {pulp_time:.3f}s")
            
        except Exception as e:
            print(f"       ❌ Gurobi failed: {e}")
            results['solvers']['gurobi'] = {'status': 'Error', 'error': str(e), 'success': False}
    
    # 1b. Pyomo Solver (native quadratic, no linearization)
    print(f"     Running Pyomo (BIP/MIQP)...")
    
    cached = check_cached_results('patch_pyomo', sample_data['n_plots'], run_id=1)
    if cached:
        print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
        results['solvers']['plot_pyomo'] = cached
    else:
        try:
            pyomo_start = time.time()
            pyomo_model, pyomo_results = solver_runner.solve_with_pyomo_plots_rotation(
                plots_list, foods, food_groups, config, gamma=gamma
            )
            pyomo_time = time.time() - pyomo_start
            
            pyomo_result = {
                'status': pyomo_results['status'],
                'objective_value': pyomo_results.get('objective_value'),
                'solve_time': pyomo_time,
                'solver_time': pyomo_results.get('solve_time', pyomo_time),
                'success': pyomo_results['status'] == 'Optimal',
                'sample_id': sample_data['sample_id'],
                'n_plots': sample_data['n_plots'],
                'total_area': sample_data['total_area'],
                'n_foods': len(foods),
                'n_variables': len(cqm.variables),
                'n_constraints': len(cqm.constraints)
            }
            
            # Extract solution data
            if pyomo_results['status'] == 'Optimal':
                pyomo_result['solution'] = pyomo_results.get('solution', {})
            
            results['solvers']['plot_pyomo'] = pyomo_result
            save_solver_result(pyomo_result, 'patch_pyomo', sample_data['n_plots'], run_id=1)
            
            print(f"       ✓ Pyomo: {pyomo_results['status']} in {pyomo_time:.3f}s")
            
        except Exception as e:
            print(f"       ❌ Pyomo failed: {e}")
            results['solvers']['pyomo'] = {'status': 'Error', 'error': str(e), 'success': False}
    
    # 2. DWave CQM Solver
    if dwave_token:
        print(f"     Running DWave CQM...")
        
        cached = check_cached_results('patch_dwave_cqm', sample_data['n_plots'], run_id=1)
        if cached:
            print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
            results['solvers']['dwave_cqm'] = cached
        else:
            try:
                sampleset, hybrid_time, qpu_time = solver_runner.solve_with_dwave_cqm(cqm, dwave_token)
                
                if len(sampleset) > 0:
                    best = sampleset.first
                    is_feasible = best.is_feasible
                    cqm_sample = dict(best.sample)
                    
                    # Calculate rotation objective
                    rotation_obj_components = solver_runner.calculate_rotation_objective(
                        cqm_sample, plots_list, foods, land_data,
                        config['parameters']['weights'], gamma=gamma
                    )
                    
                    # Extract total objective
                    rotation_objective = rotation_obj_components['total']
                    
                    # Normalize by total area
                    normalized_objective = rotation_objective / sample_data['total_area'] if is_feasible else None
                    
                    dwave_result = {
                        'status': 'Feasible' if is_feasible else 'Infeasible',
                        'objective_value': normalized_objective,
                        'rotation_objective': rotation_objective,
                        'hybrid_time': hybrid_time,
                        'qpu_time': qpu_time,
                        'is_feasible': is_feasible,
                        'num_samples': len(sampleset),
                        'success': is_feasible,
                        'sample_id': sample_data['sample_id'],
                        'n_plots': sample_data['n_plots'],
                        'total_area': sample_data['total_area'],
                        'n_foods': len(foods),
                        'n_variables': len(cqm.variables),
                        'n_constraints': len(cqm.constraints)
                    }
                    
                    obj_str = f"{normalized_objective:.6f}" if normalized_objective else "N/A"
                    print(f"       ✓ DWave CQM: {'Feasible' if is_feasible else 'Infeasible'} in {hybrid_time:.3f}s (obj: {obj_str})")
                else:
                    dwave_result = {
                        'status': 'No Solutions',
                        'success': False,
                        'solve_time': hybrid_time
                    }
                    print(f"       ❌ DWave CQM: No solutions returned")
                
                results['solvers']['plot_dwave_cqm'] = dwave_result
                save_solver_result(dwave_result, 'patch_dwave_cqm', sample_data['n_plots'], run_id=1)
                
            except Exception as e:
                print(f"       ❌ DWave CQM failed: {e}")
                results['solvers']['plot_dwave_cqm'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     DWave CQM: SKIPPED (no token)")
    results['solvers']['plot_dwave_cqm'] = {'status': 'Skipped', 'success': False}
    
    # 3. Convert CQM to BQM for additional solvers
    bqm = None
    invert = None
    bqm_conversion_time = None
    
    print(f"     Converting CQM to BQM...")
    try:
        bqm_start = time.time()
        lagrange_multiplier = 100000.0
        print(f"       Using Lagrange multiplier: {lagrange_multiplier}")
        
        bqm, invert = cqm_to_bqm(cqm, lagrange_multiplier=lagrange_multiplier)
        bqm_conversion_time = time.time() - bqm_start
        print(f"       ✓ BQM: {len(bqm.variables)} vars, {len(bqm.quadratic)} interactions ({bqm_conversion_time:.3f}s)")
    except Exception as e:
        print(f"       ❌ BQM conversion failed: {e}")
    
    # 4. DWave BQM Solver
    if bqm is not None and dwave_token:
        print(f"     Running DWave BQM...")
        
        cached = check_cached_results('patch_dwave_bqm', sample_data['n_plots'], run_id=1)
        if cached:
            print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
            results['solvers']['dwave_bqm'] = cached
        else:
            try:
                sampleset_bqm, hybrid_time, qpu_time, bqm_conv_time, invert_fn = solver_runner.solve_with_dwave_bqm(
                    cqm, dwave_token
                )
                
                if len(sampleset_bqm) > 0:
                    best_sample = sampleset_bqm.first
                    bqm_energy = best_sample.energy
                    
                    # Invert BQM solution back to CQM
                    cqm_sample_raw = invert_fn(best_sample.sample)
                    cqm_sample = {k: int(v) if isinstance(v, str) else v for k, v in cqm_sample_raw.items()}
                    
                    # Calculate rotation objective
                    rotation_obj_components = solver_runner.calculate_rotation_objective(
                        cqm_sample, plots_list, foods, land_data,
                        config['parameters']['weights'], gamma=gamma
                    )
                    
                    rotation_objective = rotation_obj_components['total']
                    normalized_objective = rotation_objective / sample_data['total_area']
                    
                    dwave_bqm_result = {
                        'status': 'Optimal',
                        'objective_value': normalized_objective,
                        'rotation_objective': rotation_objective,
                        'bqm_energy': bqm_energy,
                        'solve_time': hybrid_time,
                        'hybrid_time': hybrid_time,
                        'qpu_time': qpu_time,
                        'bqm_conversion_time': bqm_conv_time,
                        'success': True,
                        'sample_id': sample_data['sample_id'],
                        'n_plots': sample_data['n_plots'],
                        'total_area': sample_data['total_area'],
                        'n_foods': len(foods),
                        'n_variables': len(bqm.variables),
                        'bqm_interactions': len(bqm.quadratic)
                    }
                    
                    results['solvers']['plot_dwave_bqm'] = dwave_bqm_result
                    save_solver_result(dwave_bqm_result, 'patch_dwave_bqm', sample_data['n_plots'], run_id=1)
                    
                    print(f"       ✓ DWave BQM: Optimal in {hybrid_time:.3f}s (QPU: {qpu_time:.3f}s, obj: {normalized_objective:.6f})")
                else:
                    dwave_bqm_result = {
                        'status': 'No Solutions',
                        'success': False,
                        'solve_time': hybrid_time if hybrid_time else 0
                    }
                    print(f"       ❌ DWave BQM: No solutions returned")
                    results['solvers']['plot_dwave_bqm'] = dwave_bqm_result
                
            except Exception as e:
                print(f"       ❌ DWave BQM failed: {e}")
                results['solvers']['plot_dwave_bqm'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     DWave BQM: SKIPPED (no BQM or token)")
    results['solvers']['plot_dwave_bqm'] = {'status': 'Skipped', 'success': False}
    
    # 5. Gurobi QUBO Solver
    if bqm is not None:
        print(f"     Running Gurobi QUBO...")
        
        cached = check_cached_results('patch_gurobi_qubo', sample_data['n_plots'], run_id=1)
        if cached:
            print(f"       ✓ Using cached result (objective: {cached.get('objective_value', 'N/A')})")
            results['solvers']['plot_qubo'] = cached
        else:
            try:
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
                    time_limit=100
                )
                qubo_time = time.time() - qubo_start
                
                bqm_solution = qubo_result.get('solution', {})
                cqm_sample = bqm_solution
                
                # Calculate rotation objective
                rotation_obj_components = solver_runner.calculate_rotation_objective(
                    cqm_sample, plots_list, foods, land_data,
                    config['parameters']['weights'], gamma=gamma
                )
                
                rotation_objective = rotation_obj_components['total']
                normalized_objective = rotation_objective / sample_data['total_area'] if rotation_objective else None
                
                # Extract solution summary
                solution_summary = solver_runner.extract_rotation_solution_summary(
                    cqm_sample, plots_list, foods, land_data
                )
                
                # Validate constraints
                validation = solver_runner.validate_solution_constraints(
                    cqm_sample, plots_list, foods, food_groups, land_data, config
                )
                
                obj_str = f"{rotation_objective:.6f}" if rotation_objective is not None else "N/A"
                violations_str = f", {validation['n_violations']} violations" if validation else ""
                print(f"       ✓ Gurobi QUBO: {qubo_result['status']} in {qubo_time:.3f}s (rotation_obj: {obj_str}{violations_str})")
                
                gurobi_qubo_result = {
                    'status': qubo_result['status'],
                    'objective_value': normalized_objective,
                    'rotation_objective': rotation_objective,
                    'objective_components': rotation_obj_components,
                    'bqm_energy': qubo_result.get('bqm_energy'),
                    'solve_time': qubo_time,
                    'bqm_conversion_time': bqm_conversion_time,
                    'success': qubo_result['status'] == 'Optimal',
                    'sample_id': sample_data['sample_id'],
                    'n_plots': sample_data['n_plots'],
                    'total_area': sample_data['total_area'],
                    'n_foods': len(foods),
                    'n_variables': len(bqm.variables),
                    'bqm_interactions': qubo_result.get('bqm_interactions', 0),
                    'solution_summary': solution_summary,
                    'constraint_validation': validation,
                    'solution': cqm_sample if len(str(cqm_sample)) < 50000 else "Solution too large to save"
                }
                
                results['solvers']['plot_qubo'] = gurobi_qubo_result
                save_solver_result(gurobi_qubo_result, 'patch_gurobi_qubo', sample_data['n_plots'], run_id=1)
                
            except Exception as e:
                print(f"       ❌ Gurobi QUBO failed: {e}")
                import traceback
                traceback.print_exc()
                results['solvers']['plot_qubo'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     Gurobi QUBO: SKIPPED (no BQM)")
    results['solvers']['plot_qubo'] = {'status': 'Skipped', 'success': False}
    
    return results


def run_rotation_benchmark(config_values: List[int], dwave_token: Optional[str] = None, 
                          gamma: float = DEFAULT_GAMMA, fixed_total_land: float = 100.0) -> Dict:
    """
    Run rotation benchmarks for multiple configurations.
    
    Args:
        config_values: List of plot counts to test
        dwave_token: D-Wave API token
        gamma: Rotation synergy weight
        fixed_total_land: Total land area in hectares
        
    Returns:
        Complete benchmark results dictionary
    """
    print(f"\n{'='*80}")
    print(f"3-PERIOD CROP ROTATION BENCHMARK")
    print(f"{'='*80}")
    print(f"Configurations: {config_values}")
    print(f"Rotation weight (gamma): {gamma}")
    print(f"Total land: {fixed_total_land} ha")
    print(f"D-Wave: {'Yes' if dwave_token else 'No'}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Generate rotation samples
    samples = generate_rotation_samples(config_values, fixed_total_land=fixed_total_land)
    
    rotation_results = []
    
    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] Running rotation scenario for {sample['n_plots']} plots...")
        
        try:
            result = run_rotation_scenario(sample, dwave_token, gamma)
            rotation_results.append(result)
        except Exception as e:
            print(f"  ❌ Scenario failed: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    
    # Aggregate results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    benchmark_results = {
        'metadata': {
            'timestamp': timestamp,
            'benchmark_type': 'rotation_3period',
            'configs': config_values,
            'gamma': gamma,
            'fixed_total_land': fixed_total_land,
            'n_runs_per_config': NUM_RUNS,
            'total_time': total_time
        },
        'results': rotation_results
    }
    
    # Save aggregated results
    output_file = f"benchmark_rotation_3period_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return benchmark_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='3-Period Crop Rotation Benchmark')
    parser.add_argument('--configs', nargs='+', type=int, default=BENCHMARK_CONFIGS,
                       help='List of plot counts to test')
    parser.add_argument('--dwave-token', type=str, default=None,
                       help='D-Wave API token')
    parser.add_argument('--no-dwave', action='store_true',
                       help='Skip D-Wave solvers')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA,
                       help='Rotation synergy weight coefficient')
    parser.add_argument('--total-land', type=float, default=100.0,
                       help='Total land area in hectares')
    
    args = parser.parse_args()
    
    # Get D-Wave token from environment if not provided
    if args.no_dwave:
        dwave_token = None
    else:
        dwave_token = args.dwave_token or os.getenv('DWAVE_API_TOKEN')
    
    if not dwave_token and not args.no_dwave:
        print("⚠️  Warning: No D-Wave token provided. D-Wave solvers will be skipped.")
        print("   Use --dwave-token TOKEN or set DWAVE_API_TOKEN environment variable.")
        print("   Or use --no-dwave to explicitly skip D-Wave solvers.\n")
    
    # Run benchmark
    run_rotation_benchmark(
        config_values=args.configs,
        dwave_token=dwave_token,
        gamma=args.gamma,
        fixed_total_land=args.total_land
    )


if __name__ == "__main__":
    main()
