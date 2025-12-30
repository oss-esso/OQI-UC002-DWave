#!/usr/bin/env python3
"""
Debug: Replicate exact benchmark conditions for hierarchical solver.

Test if we can reproduce the negative objectives from the benchmark.
Uses SA instead of QPU.
"""

import os
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("BENCHMARK REPLICATION DEBUG")
print("="*80)

# ============================================================================
# Load EXACT data as benchmark does
# ============================================================================

from src.scenarios import load_food_data
from data_loader_utils import load_food_data_as_dict

# Replicate benchmark's load_scenario_data function
def load_scenario_data(scenario):
    """Replicate benchmark's data loading exactly."""
    n_farms = scenario['n_farms']
    n_foods = scenario['n_foods']
    
    # Load base scenario
    if n_foods == 27:
        base_scenario = 'rotation_250farms_27foods'
    elif n_foods == 6:
        if n_farms <= 5:
            base_scenario = 'rotation_micro_25'
        elif n_farms <= 10:
            base_scenario = 'rotation_small_50'
        else:
            base_scenario = 'rotation_medium_100'
    else:
        base_scenario = scenario.get('name', 'rotation_medium_100')
    
    print(f"Loading base scenario: {base_scenario}")
    data = load_food_data_as_dict(base_scenario)
    
    # Adjust farm count
    if len(data['farm_names']) > n_farms:
        print(f"  Trimming from {len(data['farm_names'])} to {n_farms} farms")
        data['farm_names'] = data['farm_names'][:n_farms]
        data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
        data['total_area'] = sum(data['land_availability'].values())
    elif len(data['farm_names']) < n_farms:
        print(f"  Expanding from {len(data['farm_names'])} to {n_farms} farms (duplicating)")
        original_farms = data['farm_names'].copy()
        while len(data['farm_names']) < n_farms:
            idx = len(data['farm_names']) - len(original_farms)
            farm = original_farms[idx % len(original_farms)]
            new_farm = f"{farm}_dup{idx}"
            data['farm_names'].append(new_farm)
            data['land_availability'][new_farm] = data['land_availability'][farm]
        data['total_area'] = sum(data['land_availability'].values())
    
    return data


# Test scenarios (hierarchical ones that showed negative objectives)
SCENARIOS = [
    {
        'name': 'rotation_250farms_27foods',
        'n_farms': 25,
        'n_foods': 27,
        'n_periods': 3,
    },
]

# ============================================================================
# Run test
# ============================================================================

from hierarchical_quantum_solver import solve_hierarchical

QPU_CONFIG = {
    'num_reads': 100,
    'farms_per_cluster': 5,
    'num_iterations': 3,
}

for scenario in SCENARIOS:
    print(f"\n{'='*70}")
    print(f"Testing: {scenario['name']}")
    print(f"{'='*70}")
    
    data = load_scenario_data(scenario)
    print(f"Data loaded: {len(data['farm_names'])} farms × {len(data['food_names'])} foods")
    print(f"Total area: {data['total_area']:.2f}")
    
    # Run with SA (not QPU) to save resources
    hier_config = {
        'farms_per_cluster': QPU_CONFIG.get('farms_per_cluster', 5),
        'num_reads': QPU_CONFIG['num_reads'],
        'num_iterations': QPU_CONFIG.get('num_iterations', 3),
    }
    
    print(f"\nRunning hierarchical solver (SA mode)...")
    print(f"Config: {hier_config}")
    
    result = solve_hierarchical(
        data=data,
        config=hier_config,
        use_qpu=False,  # USE SA, not QPU!
        verbose=True
    )
    
    print(f"\n{'='*70}")
    print("RESULT ANALYSIS")
    print(f"{'='*70}")
    print(f"Objective reported: {result.get('objective')}")
    print(f"Objective before post-processing: {result.get('objective_before_postprocessing')}")
    print(f"Violations: {result.get('violations')}")
    print(f"Success: {result.get('success')}")
    
    # Check if negative
    obj = result.get('objective', 0)
    if obj < 0:
        print(f"\n⚠️  NEGATIVE OBJECTIVE DETECTED: {obj}")
        print("This matches the benchmark bug!")
    else:
        print(f"\n✓ Objective is POSITIVE: {obj}")
        print("This is DIFFERENT from the benchmark results!")
        print("The benchmark ran with use_qpu=True, which might behave differently.")
    
    # Check solution format
    fam_sol = result.get('family_solution', {})
    if fam_sol:
        print(f"\nFamily solution has {len(fam_sol)} entries")
        sample_key = list(fam_sol.keys())[0] if fam_sol else None
        print(f"Sample key: {sample_key}")
    
    # Check the 'solution' key that benchmark expects
    sol = result.get('solution')
    print(f"\n'solution' key value: {sol}")
    print("(Benchmark tries to validate this - if None, gets 999 violations)")
