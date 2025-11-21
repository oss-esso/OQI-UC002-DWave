"""
Utility functions for Custom Hybrid Workflow Benchmark

Modular design for easy testing and maintenance.
"""

import json
import time
from typing import Dict, Optional

from src.scenarios import load_food_data
from Utils.farm_sampler import generate_farms
from Utils.patch_sampler import generate_grid

from solver_runner_CUSTOM_HYBRID import (
    create_cqm_farm,
    create_cqm_plots,
    solve_with_pulp_farm,
    solve_with_pulp_plots,
    solve_with_custom_hybrid_workflow
)


def generate_farm_data(n_units: int, total_land: float = 100.0, seed: int = 42) -> Dict:
    """Generate farm scenario data (continuous formulation)."""
    farms_unscaled = generate_farms(n_farms=n_units, seed=seed)
    total = sum(farms_unscaled.values())
    scale = total_land / total
    farms_scaled = {k: v * scale for k, v in farms_unscaled.items()}
    
    return {
        'type': 'farm',
        'land_data': farms_scaled,
        'n_units': n_units,
        'total_area': sum(farms_scaled.values())
    }


def generate_patch_data(n_units: int, total_land: float = 100.0, seed: int = 42) -> Dict:
    """Generate patch scenario data (binary formulation)."""
    patches = generate_grid(n_farms=n_units, area=total_land, seed=seed)
    
    return {
        'type': 'patch',
        'land_data': patches,
        'n_units': n_units,
        'total_area': sum(patches.values())
    }


def create_config(land_data: Dict, scenario_type: str = 'full_family'):
    """Create configuration for solver (uses full_family scenario with all 27 foods)."""
    _, foods, food_groups, base_config = load_food_data(scenario_type)
    
    # Convert max_percentage_per_crop to maximum_planting_area (absolute values)
    max_percentage = base_config['parameters'].get('max_percentage_per_crop', {})
    total_land = sum(land_data.values())
    maximum_planting_area = {crop: max_pct * total_land for crop, max_pct in max_percentage.items()}
    
    config = {
        'parameters': {
            'land_availability': land_data,
            'minimum_planting_area': base_config['parameters'].get('minimum_planting_area', {}),
            'maximum_planting_area': maximum_planting_area,
            'food_group_constraints': base_config['parameters'].get('food_group_constraints', {}),
            'weights': base_config['parameters'].get('weights', {}),
        }
    }
    
    return foods, food_groups, config


def run_farm_solver(farms_list, foods, food_groups, config, cqm, solver_name: str, dwave_token: Optional[str] = None) -> Dict:
    """Run a specific solver on farm scenario."""
    print(f"    [{solver_name}]", end=" ", flush=True)
    
    try:
        if solver_name == 'gurobi':
            start = time.time()
            _, result = solve_with_pulp_farm(farms_list, foods, food_groups, config)
            solve_time = time.time() - start
            
            return {
                'solver': solver_name,
                'status': result['status'],
                'objective_value': result.get('objective_value'),
                'solve_time': solve_time,
                'success': result['status'] == 'Optimal'
            }
            
        elif solver_name == 'custom_hybrid' and dwave_token:
            result = solve_with_custom_hybrid_workflow(cqm, dwave_token)
            
            return {
                'solver': solver_name,
                'status': result['status'],
                'objective_value': result.get('objective_value'),
                'solve_time': result['solve_time'],
                'qpu_access_time': result.get('qpu_access_time'),
                'iterations': result.get('iterations'),
                'success': result['status'] in ['Optimal', 'Converged']
            }
        else:
            return {'solver': solver_name, 'status': 'Skipped', 'success': False}
            
    except Exception as e:
        print(f"❌")
        return {'solver': solver_name, 'status': 'Failed', 'error': str(e), 'success': False}
    
    print(f"✓")
    return result


def run_patch_solver(patches_list, foods, food_groups, config, cqm, solver_name: str, dwave_token: Optional[str] = None) -> Dict:
    """Run a specific solver on patch scenario."""
    print(f"    [{solver_name}]", end=" ", flush=True)
    
    try:
        if solver_name == 'gurobi':
            start = time.time()
            _, result = solve_with_pulp_plots(patches_list, foods, food_groups, config)
            solve_time = time.time() - start
            
            return {
                'solver': solver_name,
                'status': result['status'],
                'objective_value': result.get('objective_value'),
                'solve_time': solve_time,
                'success': result['status'] == 'Optimal'
            }
            
        elif solver_name == 'custom_hybrid' and dwave_token:
            result = solve_with_custom_hybrid_workflow(cqm, dwave_token)
            
            return {
                'solver': solver_name,
                'status': result['status'],
                'objective_value': result.get('objective_value'),
                'solve_time': result['solve_time'],
                'qpu_access_time': result.get('qpu_access_time'),
                'iterations': result.get('iterations'),
                'success': result['status'] in ['Optimal', 'Converged']
            }
        else:
            return {'solver': solver_name, 'status': 'Skipped', 'success': False}
            
    except Exception as e:
        print(f"❌")
        return {'solver': solver_name, 'status': 'Failed', 'error': str(e), 'success': False}
    
    print(f"✓")
    return result


def run_single_benchmark(n_units: int, dwave_token: Optional[str] = None, total_land: float = 100.0) -> Dict:
    """
    Run complete benchmark for a single configuration.
    
    Args:
        n_units: Number of units (farms or patches)
        dwave_token: D-Wave API token (optional)
        total_land: Total land area in hectares
        
    Returns:
        Dictionary with all benchmark results
    """
    results = {
        'n_units': n_units,
        'total_land': total_land,
        'scenarios': {}
    }
    
    # Farm scenario
    print(f"\n[FARM SCENARIO: {n_units} farms]")
    farm_data = generate_farm_data(n_units, total_land)
    farms_list = list(farm_data['land_data'].keys())
    foods, food_groups, config = create_config(farm_data['land_data'], 'full_family')
    
    print("  Creating CQM...", end=" ", flush=True)
    cqm_farm, _, _, _ = create_cqm_farm(farms_list, foods, food_groups, config)
    print(f"✓ ({len(cqm_farm.variables)} vars, {len(cqm_farm.constraints)} constraints)")
    
    print("  Running solvers:")
    farm_results = {}
    farm_results['gurobi'] = run_farm_solver(farms_list, foods, food_groups, config, cqm_farm, 'gurobi')
    
    if dwave_token:
        farm_results['custom_hybrid'] = run_farm_solver(farms_list, foods, food_groups, config, cqm_farm, 'custom_hybrid', dwave_token)
    
    results['scenarios']['farm'] = {
        'n_units': n_units,
        'n_variables': len(cqm_farm.variables),
        'n_constraints': len(cqm_farm.constraints),
        'solvers': farm_results
    }
    
    # Patch scenario
    print(f"\n[PATCH SCENARIO: {n_units} patches]")
    patch_data = generate_patch_data(n_units, total_land)
    patches_list = list(patch_data['land_data'].keys())
    foods, food_groups, config = create_config(patch_data['land_data'], 'full_family')
    
    print("  Creating CQM...", end=" ", flush=True)
    cqm_patch, _, _ = create_cqm_plots(patches_list, foods, food_groups, config)
    print(f"✓ ({len(cqm_patch.variables)} vars, {len(cqm_patch.constraints)} constraints)")
    
    print("  Running solvers:")
    patch_results = {}
    patch_results['gurobi'] = run_patch_solver(patches_list, foods, food_groups, config, cqm_patch, 'gurobi')
    
    if dwave_token:
        patch_results['custom_hybrid'] = run_patch_solver(patches_list, foods, food_groups, config, cqm_patch, 'custom_hybrid', dwave_token)
    
    results['scenarios']['patch'] = {
        'n_units': n_units,
        'n_variables': len(cqm_patch.variables),
        'n_constraints': len(cqm_patch.constraints),
        'solvers': patch_results
    }
    
    return results


def save_results(results: Dict, filepath: str):
    """Save results to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def print_summary(results: Dict):
    """Print benchmark summary."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    for scenario_name, scenario_data in results['scenarios'].items():
        print(f"\n{scenario_name.upper()} Scenario:")
        print(f"  Units: {scenario_data['n_units']}")
        print(f"  Variables: {scenario_data['n_variables']}")
        print(f"  Constraints: {scenario_data['n_constraints']}")
        print(f"  Solvers:")
        
        for solver_name, solver_result in scenario_data['solvers'].items():
            status = solver_result.get('status', 'Unknown')
            obj = solver_result.get('objective_value', 'N/A')
            time_val = solver_result.get('solve_time', 'N/A')
            
            print(f"    {solver_name}: {status} | Obj: {obj} | Time: {time_val}s")
            
            if solver_name == 'custom_hybrid' and 'iterations' in solver_result:
                print(f"              Iterations: {solver_result['iterations']}")
                print(f"              QPU Time: {solver_result.get('qpu_access_time', 'N/A')}s")
