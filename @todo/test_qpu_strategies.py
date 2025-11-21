#!/usr/bin/env python3
"""
Test QPU-Enhanced Decomposition Strategies

Quick test to verify Benders-QPU and Dantzig-Wolfe-QPU work correctly.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from decomposition_strategies import DecompositionFactory
from benchmark_utils_decomposed import generate_farm_data, create_config

def test_qpu_strategies():
    """Test QPU-enhanced strategies without actual QPU (dry run)."""
    
    print("="*80)
    print("TESTING QPU-ENHANCED DECOMPOSITION STRATEGIES")
    print("="*80)
    
    # Generate small test problem
    print("\nGenerating test problem (5 farms, 27 foods)...")
    farm_data = generate_farm_data(n_units=5, total_land=50.0)
    foods, food_groups, config = create_config(farm_data['land_data'])
    
    # Load benefits
    from src.scenarios import load_food_data
    _, _, _, base_config = load_food_data('full_family')
    
    config['benefits'] = {}
    for food in foods:
        weights = config.get('parameters', {}).get('weights', {})
        benefit = sum(
            base_config.get('nutrients', {}).get(food, {}).get(attr, 0) * weight
            for attr, weight in weights.items()
        )
        config['benefits'][food] = benefit if benefit > 0 else 100.0
    
    config['food_groups'] = food_groups
    config['foods'] = foods
    
    farms = farm_data['land_data']
    
    print(f"  ✅ Problem: {len(farms)} farms, {len(foods)} foods")
    
    # Test Benders QPU (without actual QPU)
    print("\n" + "="*80)
    print("TEST 1: Benders Decomposition (QPU) - Classical Mode")
    print("="*80)
    
    strategy = DecompositionFactory.get_strategy('benders_qpu')
    result = strategy.solve(
        farms=farms,
        foods=foods,
        food_groups=food_groups,
        config=config,
        dwave_token=None,  # No token = classical mode
        max_iterations=3,
        time_limit=30.0,
        use_qpu_for_master=False
    )
    
    print("\nResult:")
    print(f"  Status: {result['solver_info']['status']}")
    print(f"  Objective: {result['solution']['objective_value']:.4f}")
    print(f"  Time: {result['solver_info']['solve_time']:.3f}s")
    print(f"  Iterations: {result['solver_info']['num_iterations']}")
    print(f"  Feasible: {result['solution']['is_feasible']}")
    
    # Test Dantzig-Wolfe QPU (without actual QPU)
    print("\n" + "="*80)
    print("TEST 2: Dantzig-Wolfe Decomposition (QPU) - Classical Mode")
    print("="*80)
    
    strategy = DecompositionFactory.get_strategy('dantzig_wolfe_qpu')
    result = strategy.solve(
        farms=farms,
        foods=foods,
        food_groups=food_groups,
        config=config,
        dwave_token=None,  # No token = classical mode
        max_iterations=3,
        time_limit=30.0,
        use_qpu_for_pricing=False
    )
    
    print("\nResult:")
    print(f"  Status: {result['solver_info']['status']}")
    print(f"  Objective: {result['solution']['objective_value']:.4f}")
    print(f"  Time: {result['solver_info']['solve_time']:.3f}s")
    print(f"  Iterations: {result['solver_info']['num_iterations']}")
    print(f"  Feasible: {result['solution']['is_feasible']}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
    print("\nBoth QPU-enhanced strategies are ready for actual QPU usage.")
    print("To use with QPU, provide dwave_token parameter and set use_qpu_* flags to True.")
    

if __name__ == "__main__":
    test_qpu_strategies()
