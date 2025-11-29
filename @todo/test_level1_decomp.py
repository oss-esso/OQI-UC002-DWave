#!/usr/bin/env python3
"""
Test Level 1 Decomposition Strategies (Benders, ADMM, DW)

These are for Farm scenario (continuous A + binary Y variables)
Tests without D-Wave token - uses SA fallback or classical solvers.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=" * 70)
print("TESTING LEVEL 1 DECOMPOSITION STRATEGIES")
print("=" * 70)

# Test with small farm scenario
N_FARMS = 5

# Setup test data
print("\n1. Setting up test data...")
try:
    from src.scenarios import load_food_data
    from Utils.farm_sampler import generate_farms
    
    farms_unscaled = generate_farms(n_farms=N_FARMS, seed=42)
    # Scale to total 100 ha
    total = sum(farms_unscaled.values())
    farms = {k: v * 100.0 / total for k, v in farms_unscaled.items()}
    print(f"  ✓ Generated {len(farms)} farms (total: {sum(farms.values()):.1f} ha)")
    
    _, foods_data, food_groups, config = load_food_data('full_family')
    foods = list(foods_data.keys())
    print(f"  ✓ Loaded {len(foods)} foods, {len(food_groups)} food groups")
    
    # Create config with benefits
    config['benefits'] = {food: sum([
        config['parameters']['weights'].get('nutritional_value', 0.25) * foods_data[food].get('nutritional_value', 0),
        config['parameters']['weights'].get('nutrient_density', 0.2) * foods_data[food].get('nutrient_density', 0),
        -config['parameters']['weights'].get('environmental_impact', 0.25) * foods_data[food].get('environmental_impact', 0),
        config['parameters']['weights'].get('affordability', 0.15) * foods_data[food].get('affordability', 0),
        config['parameters']['weights'].get('sustainability', 0.15) * foods_data[food].get('sustainability', 0)
    ]) for food in foods}
    print(f"  ✓ Created benefits dict with {len(config['benefits'])} entries")
    
except Exception as e:
    print(f"  ✗ Failed to setup: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Benders (classical)
print("\n2. Testing Benders Decomposition (classical)...")
try:
    from decomposition_benders import solve_with_benders
    
    result = solve_with_benders(
        farms=farms,
        foods=foods,
        food_groups=food_groups,
        config=config,
        max_iterations=10,
        time_limit=60.0
    )
    
    print(f"  ✓ Converged: {result.get('result', {}).get('converged', result.get('converged', False))}")
    print(f"  ✓ Iterations: {len(result.get('decomposition_specific', {}).get('iterations_detail', []))}")
    print(f"  ✓ Objective: {result.get('result', {}).get('objective_value', 'N/A')}")
    print(f"  ✓ Solve time: {result.get('result', {}).get('solve_time', 0):.2f}s")
    
    # Check solution quality
    sol_areas = result.get('result', {}).get('solution_areas', {})
    if sol_areas:
        total_area = sum(sol_areas.values())
        land_usage = total_area / sum(farms.values()) * 100
        print(f"  ✓ Land usage: {land_usage:.1f}%")
        
except ImportError as e:
    print(f"  ✗ Import error: {e}")
except Exception as e:
    print(f"  ✗ Benders failed: {e}")
    import traceback
    traceback.print_exc()

# Test ADMM (classical)
print("\n3. Testing ADMM Decomposition (classical)...")
try:
    from decomposition_admm import solve_with_admm
    
    result = solve_with_admm(
        farms=farms,
        foods=foods,
        food_groups=food_groups,
        config=config,
        max_iterations=10,
        rho=1.0,
        time_limit=60.0
    )
    
    print(f"  ✓ Converged: {result.get('result', {}).get('converged', False)}")
    print(f"  ✓ Iterations: {len(result.get('decomposition_specific', {}).get('iterations_detail', []))}")
    print(f"  ✓ Objective: {result.get('result', {}).get('objective_value', 'N/A')}")
    print(f"  ✓ Solve time: {result.get('result', {}).get('solve_time', 0):.2f}s")
    
except ImportError as e:
    print(f"  ✗ Import error: {e}")
except Exception as e:
    print(f"  ✗ ADMM failed: {e}")
    import traceback
    traceback.print_exc()

# Test Dantzig-Wolfe (classical)
print("\n4. Testing Dantzig-Wolfe Decomposition (classical)...")
try:
    from decomposition_dantzig_wolfe import solve_with_dantzig_wolfe
    
    result = solve_with_dantzig_wolfe(
        farms=farms,
        foods=foods,
        food_groups=food_groups,
        config=config,
        max_iterations=20,
        time_limit=60.0
    )
    
    print(f"  ✓ Converged: {result.get('result', {}).get('converged', False)}")
    print(f"  ✓ Iterations: {len(result.get('decomposition_specific', {}).get('iterations_detail', []))}")
    print(f"  ✓ Objective: {result.get('result', {}).get('objective_value', 'N/A')}")
    print(f"  ✓ Solve time: {result.get('result', {}).get('solve_time', 0):.2f}s")
    
except ImportError as e:
    print(f"  ✗ Import error: {e}")
except Exception as e:
    print(f"  ✗ Dantzig-Wolfe failed: {e})")
    import traceback
    traceback.print_exc()

# Test Benders QPU (should fallback to SA)
print("\n5. Testing Benders QPU (should fallback to SA)...")
try:
    from decomposition_benders_qpu import solve_with_benders_qpu
    
    result = solve_with_benders_qpu(
        farms=farms,
        foods=foods,
        food_groups=food_groups,
        config=config,
        dwave_token=None,  # Force SA fallback
        max_iterations=5,
        time_limit=60.0,
        use_qpu_for_master=True
    )
    
    print(f"  ✓ Converged: {result.get('result', {}).get('converged', False)}")
    print(f"  ✓ Iterations: {len(result.get('decomposition_specific', {}).get('iterations_detail', []))}")
    print(f"  ✓ Objective: {result.get('result', {}).get('objective_value', 'N/A')}")
    print(f"  ✓ Solve time: {result.get('result', {}).get('solve_time', 0):.2f}s")
    
except ImportError as e:
    print(f"  ✗ Import error: {e}")
except Exception as e:
    print(f"  ✗ Benders QPU failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)