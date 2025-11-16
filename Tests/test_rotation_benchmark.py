#!/usr/bin/env python3
"""
Test script for 3-period crop rotation benchmark.

This script runs a minimal test with 3 plots and 3 crops to verify
that the rotation benchmark implementation works correctly with all 4 solvers.
"""

import os
import sys

# Add project root to path
# Add project root and Benchmark Scripts to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

from rotation_benchmark import (
    generate_rotation_samples,
    create_rotation_config,
    run_rotation_scenario
)


def test_rotation_scenario():
    """Test basic rotation scenario generation and CQM creation."""
    print("=" * 80)
    print("TEST: 3-Period Crop Rotation Benchmark (4 Solvers)")
    print("=" * 80)
    
    # Test 1: Generate scenario
    print("\n[Test 1] Generating rotation scenario with 3 plots...")
    try:
        samples = generate_rotation_samples([3], seed_offset=0, fixed_total_land=10.0)
        scenario = samples[0]
        print(f"  ✓ Generated scenario: {scenario['n_plots']} plots, {scenario['total_area']:.1f} ha")
        print(f"    Plot area: {scenario['plot_area']:.2f} ha each")
        assert scenario['n_plots'] == 3
        assert scenario['total_area'] == 10.0
        assert len(scenario['data']) == 3
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    # Test 2: Create config
    print("\n[Test 2] Creating rotation configuration...")
    try:
        foods, food_groups, config = create_rotation_config(scenario['data'])
        print(f"  ✓ Created config: {len(foods)} foods, {len(food_groups)} food groups")
        print(f"    Sample foods: {list(foods.keys())[:5]}")
        assert len(foods) > 0
        assert len(food_groups) > 0
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    # Test 3: Run scenario (without D-Wave - just CQM creation)
    print("\n[Test 3] Running rotation scenario (CQM creation + BQM conversion)...")
    try:
        result = run_rotation_scenario(scenario, dwave_token=None, gamma=0.1)
        print(f"  ✓ Created CQM successfully")
        print(f"    Variables: {result['n_variables']}")
        print(f"    Constraints: {result['n_constraints']}")
        print(f"    CQM build time: {result['cqm_time']:.3f}s")
        print(f"    Periods: {result['n_periods']}")
        print(f"    Gamma: {result['gamma']}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Verify variable count
    print("\n[Test 4] Verifying 3-period variable count...")
    try:
        expected_vars = scenario['n_plots'] * len(foods) * 3  # plots × crops × periods
        actual_vars = result['n_variables']
        print(f"  Expected: {expected_vars} variables (3 plots × {len(foods)} crops × 3 periods)")
        print(f"  Actual: {actual_vars} variables")
        assert actual_vars == expected_vars, f"Variable count mismatch: {actual_vars} != {expected_vars}"
        print(f"  ✓ Variable count verified: {actual_vars}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    # Test 5: Verify all solver slots exist
    print("\n[Test 5] Verifying all 4 solver slots...")
    try:
        expected_solvers = ['gurobi', 'dwave_cqm', 'dwave_bqm', 'gurobi_qubo']
        actual_solvers = list(result['solvers'].keys())
        print(f"  Expected solvers: {expected_solvers}")
        print(f"  Actual solvers: {actual_solvers}")
        for solver in expected_solvers:
            assert solver in actual_solvers, f"Solver {solver} not found"
            status = result['solvers'][solver].get('status', 'Unknown')
            print(f"    - {solver}: {status}")
        print(f"  ✓ All 4 solver slots present")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    print("\nRotation benchmark is ready with 4 solvers:")
    print("  1. Gurobi (PuLP) - BIP for 3-period rotation")
    print("  2. D-Wave CQM - Quantum-classical hybrid")
    print("  3. Gurobi QUBO - Native QUBO solver")
    print("  4. D-Wave BQM - Quantum annealer")
    print("\n")
    
    return True


if __name__ == "__main__":
    success = test_rotation_scenario()
    sys.exit(0 if success else 1)
