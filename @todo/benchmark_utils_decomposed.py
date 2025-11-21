"""
Utility functions for Decomposed QPU Benchmark

Modular design implementing strategic problem decomposition:
- Farm scenarios: HYBRID DECOMPOSITION (Gurobi + QPU)
- Patch scenarios: Quantum-only (low-level QPU)
"""

import json
import time
from typing import Dict, Optional

from src.scenarios import load_food_data
from Utils.farm_sampler import generate_farms
from Utils.patch_sampler import generate_grid
from dimod import cqm_to_bqm

from solver_runner_DECOMPOSED import (
    create_cqm_farm,
    create_cqm_plots,
    solve_with_pulp_farm,
    solve_with_pulp_plots,
    solve_with_decomposed_qpu,
    solve_farm_with_hybrid_decomposition
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


def run_farm_hybrid(farms_list, foods, food_groups, config, cqm, dwave_token: Optional[str] = None, **qpu_kwargs) -> Dict:
    """Run farm scenario with HYBRID DECOMPOSITION (Gurobi + QPU)."""
    print(f"    [hybrid_decomposition]", end=" ", flush=True)
    
    try:
        result = solve_farm_with_hybrid_decomposition(
            farms_list, foods, food_groups, config, dwave_token, **qpu_kwargs
        )
        
        print(f"✓")
        
        return {
            'solver': 'hybrid_decomposition',
            'status': result['status'],
            'objective_value': result.get('objective_value'),
            'solve_time': result['solve_time'],
            'gurobi_time': result.get('gurobi_time'),
            'qpu_time': result.get('qpu_time'),
            'qpu_access_time': result.get('qpu_access_time'),
            'relaxation_objective': result.get('relaxation_objective'),
            'final_objective': result.get('final_objective'),
            'success': result['status'] == 'Optimal',
            'solver_type': 'hybrid_decomposition_gurobi_qpu'
        }
        
    except Exception as e:
        print(f"❌")
        return {'solver': 'hybrid_decomposition', 'status': 'Failed', 'error': str(e), 'success': False}


def run_patch_quantum(patches_list, foods, food_groups, config, cqm, dwave_token: Optional[str] = None, **qpu_kwargs) -> Dict:
    """Run patch scenario with quantum solver (or SimulatedAnnealing fallback)."""
    print(f"    [decomposed_qpu]", end=" ", flush=True)
    
    try:
        # Convert CQM to BQM
        print(f"converting...", end=" ", flush=True)
        bqm, invert = cqm_to_bqm(cqm)
        
        # Solve with low-level QPU
        result = solve_with_decomposed_qpu(bqm, dwave_token, **qpu_kwargs)
        
        # Invert solution back to CQM space
        cqm_sample = invert(result['solution'])
        
        print(f"✓")
        
        return {
            'solver': 'decomposed_qpu',
            'status': result['status'],
            'objective_value': result.get('objective_value'),
            'solve_time': result['solve_time'],
            'qpu_access_time': result.get('qpu_access_time'),
            'qpu_programming_time': result.get('qpu_programming_time'),
            'qpu_sampling_time': result.get('qpu_sampling_time'),
            'num_reads': result.get('num_reads'),
            'num_occurrences': result.get('num_occurrences'),
            'success': result['status'] == 'Optimal',
            'solver_type': 'quantum_annealing',
            'qpu_config': result.get('qpu_config', {})
        }
        
    except Exception as e:
        print(f"❌")
        return {'solver': 'decomposed_qpu', 'status': 'Failed', 'error': str(e), 'success': False}


def run_single_benchmark(n_units: int, dwave_token: Optional[str] = None, total_land: float = 100.0, **qpu_kwargs) -> Dict:
    """
    Run complete benchmark for a single configuration.
    
    Strategic Decomposition:
    - Farm scenario: HYBRID DECOMPOSITION (Gurobi continuous + QPU binary)
    - Patch scenario: Quantum-only (DWaveSampler QPU)
    
    Args:
        n_units: Number of units (farms or patches)
        dwave_token: D-Wave API token (optional)
        total_land: Total land area in hectares
        **qpu_kwargs: Additional QPU parameters (num_reads, annealing_time, etc.)
        
    Returns:
        Dictionary with all benchmark results
    """
    results = {
        'n_units': n_units,
        'total_land': total_land,
        'scenarios': {}
    }
    
    # Farm scenario - HYBRID DECOMPOSITION
    print(f"\n[FARM SCENARIO: {n_units} farms - HYBRID DECOMPOSITION (Gurobi + QPU)]")
    farm_data = generate_farm_data(n_units, total_land)
    farms_list = list(farm_data['land_data'].keys())
    foods, food_groups, config = create_config(farm_data['land_data'], 'full_family')
    
    print("  Creating CQM...", end=" ", flush=True)
    cqm_farm, _, _, _ = create_cqm_farm(farms_list, foods, food_groups, config)
    print(f"✓ ({len(cqm_farm.variables)} vars, {len(cqm_farm.constraints)} constraints)")
    
    print("  Running solvers:")
    farm_results = {}
    farm_results['hybrid_decomposition'] = run_farm_hybrid(
        farms_list, foods, food_groups, config, cqm_farm, dwave_token, **qpu_kwargs
    )
    
    results['scenarios']['farm'] = {
        'n_units': n_units,
        'n_variables': len(cqm_farm.variables),
        'n_constraints': len(cqm_farm.constraints),
        'strategy': 'hybrid_decomposition',
        'solvers': farm_results
    }
    
    # Patch scenario - QUANTUM ONLY
    print(f"\n[PATCH SCENARIO: {n_units} patches - QUANTUM OPTIMIZATION]")
    patch_data = generate_patch_data(n_units, total_land)
    patches_list = list(patch_data['land_data'].keys())
    foods, food_groups, config = create_config(patch_data['land_data'], 'full_family')
    
    print("  Creating CQM...", end=" ", flush=True)
    cqm_patch, _, _ = create_cqm_plots(patches_list, foods, food_groups, config)
    print(f"✓ ({len(cqm_patch.variables)} vars, {len(cqm_patch.constraints)} constraints)")
    
    print("  Running solvers:")
    patch_results = {}
    patch_results['decomposed_qpu'] = run_patch_quantum(
        patches_list, foods, food_groups, config, cqm_patch, dwave_token, **qpu_kwargs
    )
    
    results['scenarios']['patch'] = {
        'n_units': n_units,
        'n_variables': len(cqm_patch.variables),
        'n_constraints': len(cqm_patch.constraints),
        'strategy': 'quantum_only',
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
    print("BENCHMARK SUMMARY - Strategic Problem Decomposition")
    print("="*80)
    
    for scenario_name, scenario_data in results['scenarios'].items():
        print(f"\n{scenario_name.upper()} Scenario ({scenario_data['strategy']}):")
        print(f"  Units: {scenario_data['n_units']}")
        print(f"  Variables: {scenario_data['n_variables']}")
        print(f"  Constraints: {scenario_data['n_constraints']}")
        print(f"  Solvers:")
        
        for solver_name, solver_result in scenario_data['solvers'].items():
            status = solver_result.get('status', 'Unknown')
            obj = solver_result.get('objective_value', 'N/A')
            time_val = solver_result.get('solve_time', 'N/A')
            solver_type = solver_result.get('solver_type', 'unknown')
            
            print(f"    {solver_name} ({solver_type}):")
            print(f"      Status: {status} | Obj: {obj} | Time: {time_val}s")
            
            if solver_name == 'decomposed_qpu':
                qpu_time = solver_result.get('qpu_access_time', 'N/A')
                num_reads = solver_result.get('num_reads', 'N/A')
                print(f"      QPU Access Time: {qpu_time}s")
                print(f"      Num Reads: {num_reads}")
