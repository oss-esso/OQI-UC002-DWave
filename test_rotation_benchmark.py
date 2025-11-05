#!/usr/bin/env python3
"""
Test script for 3-period crop rotation benchmark.

This script runs a minimal test with 3 plots and 3 crops to verify
that the rotation benchmark implementation works correctly.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from rotation_benchmark import (
    generate_rotation_scenario,
    create_rotation_config,
    run_rotation_scenario
)


def test_rotation_scenario():
    """Test basic rotation scenario generation and CQM creation."""
    print("=" * 80)
    print("TEST: 3-Period Crop Rotation Benchmark")
    print("=" * 80)
    
    # Test 1: Generate scenario
    print("\n[Test 1] Generating rotation scenario with 3 plots...")
    try:
        scenario = generate_rotation_scenario(n_plots=3, seed=42, fixed_total_land=10.0)
        print(f"  ✓ Generated scenario: {scenario['n_plots']} plots, {scenario['total_area']:.1f} ha")
        print(f"    Plot area: {scenario['plot_area']:.2f} ha each")
        assert scenario['n_plots'] == 3
        assert scenario['total_area'] == 10.0
        assert len(scenario['land_availability']) == 3
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    # Test 2: Create config
    print("\n[Test 2] Creating rotation configuration...")
    try:
        foods, food_groups, config = create_rotation_config(scenario['land_availability'])
        print(f"  ✓ Created config: {len(foods)} foods, {len(food_groups)} food groups")
        print(f"    Sample foods: {list(foods.keys())[:5]}")
        assert len(foods) > 0
        assert len(food_groups) > 0
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    # Test 3: Run scenario (without D-Wave - just CQM creation)
    print("\n[Test 3] Running rotation scenario (CQM creation only)...")
    try:
        result = run_rotation_scenario(scenario, dwave_token=None, gamma=0.1)
        print(f"  ✓ Created CQM successfully")
        print(f"    Variables: {result['n_variables']}")
        print(f"    Constraints: {result['n_constraints']}")
        print(f"    CQM build time: {result['cqm_time']:.3f}s")
        
        # Verify expected structure
        assert result['scenario_type'] == 'rotation_3period'
        assert result['n_plots'] == 3
        assert result['n_periods'] == 3
        assert result['n_variables'] > 0
        assert result['n_constraints'] > 0
        
        # For 3 plots, n crops, 3 periods: expect 3 * n_crops * 3 variables
        expected_vars = 3 * len(foods) * 3
        assert result['n_variables'] == expected_vars, \
            f"Expected {expected_vars} variables, got {result['n_variables']}"
        
        print(f"  ✓ Variable count verified: {result['n_variables']} (3 plots × {len(foods)} crops × 3 periods)")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_rotation_scenario()
    sys.exit(0 if success else 1)
