#!/usr/bin/env python3
"""
Comprehensive Test Suite for Hierarchical Quantum Solver

Tests all components without QPU access (uses SimulatedAnnealing).
Validates the complete 3-level optimization pipeline on multiple problem sizes.

Author: OQI-UC002-DWave Project
Date: 2025-12-12
"""

import sys
import os
import time
from pathlib import Path

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print("="*80)
print("HIERARCHICAL QUANTUM SOLVER - COMPREHENSIVE TEST SUITE")
print("="*80)
print("\n⚠️  NO QPU ACCESS - Using SimulatedAnnealing for all tests")
print("=" *80)

# Import modules
from food_grouping import (
    aggregate_foods_to_families,
    validate_family_data,
    print_aggregation_summary,
)

from hierarchical_quantum_solver import (
    solve_hierarchical,
    DEFAULT_CONFIG,
)

from src.scenarios import load_food_data

# ============================================================================
# TEST 1: Food Grouping Module
# ============================================================================

def test_food_grouping():
    """Test food aggregation from 27 foods to 6 families."""
    print("\n" + "-"*80)
    print("TEST 1: Food Grouping (27 foods → 6 families)")
    print("-"*80)
    
    try:
        # Load a large scenario
        print("Loading rotation_250farms_27foods...")
        farms, foods, food_groups, config = load_food_data('rotation_250farms_27foods')
        
        params = config.get('parameters', {})
        weights = params.get('weights', {})
        land_availability = params.get('land_availability', {})
        farm_names = list(land_availability.keys())
        total_area = sum(land_availability.values())
        
        food_names = list(foods.keys())
        food_benefits = {}
        for food in food_names:
            food_data = foods.get(food, {})
            benefit = sum(food_data.get(attr, 0.5) * w for attr, w in weights.items())
            food_benefits[food] = benefit
        
        data = {
            'foods': foods,
            'food_names': food_names,
            'food_groups': food_groups,
            'food_benefits': food_benefits,
            'weights': weights,
            'land_availability': land_availability,
            'farm_names': farm_names,
            'total_area': total_area,
            'n_farms': len(farm_names),
            'n_foods': len(food_names),
            'config': config,
        }
        
        print(f"  Original: {len(food_names)} foods × {len(farm_names)} farms")
        
        # Aggregate
        family_data = aggregate_foods_to_families(data)
        
        # Validate
        valid = validate_family_data(family_data)
        
        if valid:
            print_aggregation_summary(data, family_data)
            print("\n✅ TEST 1 PASSED: Food grouping successful")
            return True
        else:
            print("\n❌ TEST 1 FAILED: Validation failed")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 2: Small-Scale Hierarchical Solve (rotation_small_50)
# ============================================================================

def test_small_scale():
    """Test hierarchical solver on small problem (already family-level)."""
    print("\n" + "-"*80)
    print("TEST 2: Small-Scale Solver (10 farms × 6 families × 3 periods)")
    print("-"*80)
    
    try:
        print("Loading rotation_small_50...")
        farms, foods, food_groups, config = load_food_data('rotation_small_50')
        
        params = config.get('parameters', {})
        weights = params.get('weights', {})
        land_availability = params.get('land_availability', {})
        farm_names = list(land_availability.keys())
        total_area = sum(land_availability.values())
        
        food_names = list(foods.keys())
        food_benefits = {}
        for food in food_names:
            food_data = foods.get(food, {})
            benefit = sum(food_data.get(attr, 0.5) * w for attr, w in weights.items())
            food_benefits[food] = benefit
        
        data = {
            'foods': foods,
            'food_names': food_names,
            'food_groups': food_groups,
            'food_benefits': food_benefits,
            'weights': weights,
            'land_availability': land_availability,
            'farm_names': farm_names,
            'total_area': total_area,
            'n_farms': len(farm_names),
            'n_foods': len(food_names),
            'config': config,
        }
        
        # Configure solver (fast settings for testing)
        solver_config = DEFAULT_CONFIG.copy()
        solver_config['farms_per_cluster'] = 5
        solver_config['num_iterations'] = 2
        solver_config['num_reads'] = 20  # Fast SA
        
        print(f"  Problem: {len(farm_names)} farms × {len(food_names)} families × 3 periods")
        print(f"  Config: {solver_config['farms_per_cluster']} farms/cluster, {solver_config['num_iterations']} iterations")
        
        # Solve
        start_time = time.time()
        result = solve_hierarchical(data, solver_config, use_qpu=False, verbose=False)
        total_time = time.time() - start_time
        
        # Check results
        if result['success'] and result['violations'] == 0:
            print(f"\n  ✅ Solved in {total_time:.1f}s")
            print(f"  Objective: {result['objective']:.4f}")
            print(f"  Violations: {result['violations']}")
            print(f"  Unique crops: {result['diversity_stats']['total_unique_crops']}")
            print("\n✅ TEST 2 PASSED")
            return True
        else:
            print(f"\n❌ TEST 2 FAILED: Violations = {result['violations']}")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 3: Medium-Scale with Food Aggregation (rotation_250farms_27foods)
# ============================================================================

def test_medium_scale_aggregation():
    """Test full pipeline: 250 farms × 27 foods → aggregate → decompose → solve."""
    print("\n" + "-"*80)
    print("TEST 3: Medium-Scale with Aggregation (250 farms × 27 foods → 6 families)")
    print("-"*80)
    
    try:
        print("Loading rotation_250farms_27foods...")
        farms, foods, food_groups, config = load_food_data('rotation_250farms_27foods')
        
        params = config.get('parameters', {})
        weights = params.get('weights', {})
        land_availability = params.get('land_availability', {})
        farm_names = list(land_availability.keys())[:50]  # Test on subset (50 farms)
        land_availability = {f: land_availability[f] for f in farm_names}
        total_area = sum(land_availability.values())
        
        food_names = list(foods.keys())
        food_benefits = {}
        for food in food_names:
            food_data = foods.get(food, {})
            benefit = sum(food_data.get(attr, 0.5) * w for attr, w in weights.items())
            food_benefits[food] = benefit
        
        data = {
            'foods': foods,
            'food_names': food_names,
            'food_groups': food_groups,
            'food_benefits': food_benefits,
            'weights': weights,
            'land_availability': land_availability,
            'farm_names': farm_names,
            'total_area': total_area,
            'n_farms': len(farm_names),
            'n_foods': len(food_names),
            'config': config,
        }
        
        print(f"  Problem: {len(farm_names)} farms × {len(food_names)} foods × 3 periods")
        print(f"  Total variables BEFORE aggregation: {len(farm_names) * len(food_names) * 3:,}")
        
        # Configure solver
        solver_config = DEFAULT_CONFIG.copy()
        solver_config['farms_per_cluster'] = 10
        solver_config['num_iterations'] = 2
        solver_config['num_reads'] = 20  # Fast SA
        
        # Solve (will auto-aggregate)
        start_time = time.time()
        result = solve_hierarchical(data, solver_config, use_qpu=False, verbose=True)
        total_time = time.time() - start_time
        
        # Check results
        if result['success']:
            print(f"\n  ✅ Solved in {total_time:.1f}s")
            print(f"  Food aggregation: {result['levels']['food_aggregation']}")
            print(f"  Decomposition: {result['levels']['decomposition']['n_clusters']} clusters")
            print(f"  Objective: {result['objective']:.4f}")
            print(f"  Violations: {result['violations']}")
            print(f"  Unique crops (post-processing): {result['diversity_stats']['total_unique_crops']}/{result['diversity_stats']['max_possible_crops']}")
            print(f"  Shannon diversity: {result['diversity_stats']['shannon_diversity']:.3f}")
            print("\n✅ TEST 3 PASSED")
            return True
        else:
            print(f"\n❌ TEST 3 FAILED")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests."""
    print("\nStarting test suite...\n")
    
    results = {}
    
    # Test 1: Food grouping
    results['food_grouping'] = test_food_grouping()
    
    # Test 2: Small-scale solver
    results['small_scale'] = test_small_scale()
    
    # Test 3: Medium-scale with aggregation
    results['medium_scale'] = test_medium_scale_aggregation()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:30s}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Ready for QPU run.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Review before QPU run.")
    
    print("="*80)
    
    return passed_tests == total_tests


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
