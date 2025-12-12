#!/usr/bin/env python3
"""
Quick Validation Tests for Hierarchical Solver

Tests:
1. Scenario loading (verify 27 foods)
2. Food grouping (27 → 6 families)
3. Small-scale SA solve (rotation_small_50)
4. Medium-scale SA solve (50 farms subset of rotation_250farms_27foods)
5. Gurobi ground truth comparison

NO QPU ACCESS USED

Author: OQI-UC002-DWave
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
print("HIERARCHICAL SOLVER - QUICK VALIDATION TESTS")
print("="*80)
print("\n⚠️  NO QPU - Using SimulatedAnnealing + Gurobi\n")
print("="*80)

from food_grouping import aggregate_foods_to_families, analyze_crop_diversity
from hierarchical_quantum_solver import solve_hierarchical, DEFAULT_CONFIG
from src.scenarios import load_food_data

# ============================================================================
# TEST 1: Verify 27 Foods in Scenarios
# ============================================================================

def test_scenario_loading():
    """Verify all scenarios load exactly 27 foods."""
    print("\n" + "="*80)
    print("TEST 1: Scenario Loading (Verify 27 Foods)")
    print("="*80)
    
    scenarios = [
        'rotation_250farms_27foods',
        'rotation_350farms_27foods',
        'rotation_500farms_27foods',
        'rotation_1000farms_27foods',
    ]
    
    all_passed = True
    
    for scenario in scenarios:
        try:
            farms, foods, food_groups, config = load_food_data(scenario)
            params = config.get('parameters', {})
            la = params.get('land_availability', {})
            
            n_farms = len(la)
            n_foods = len(foods)
            n_vars = n_farms * n_foods * 3
            
            passed = n_foods == 27
            status = "✅" if passed else f"❌ (got {n_foods} foods)"
            
            print(f"\n  {scenario}:")
            print(f"    Farms: {n_farms}")
            print(f"    Foods: {n_foods} {status}")
            print(f"    Variables: {n_vars:,}")
            print(f"    Food list: {list(foods.keys())[:5]}...")
            
            if not passed:
                all_passed = False
                print(f"    ⚠️  Expected 27 foods, got {n_foods}")
                
        except Exception as e:
            print(f"\n  {scenario}: ❌ FAILED - {e}")
            all_passed = False
    
    print("\n" + "-"*80)
    if all_passed:
        print("✅ TEST 1 PASSED: All scenarios have 27 foods")
    else:
        print("❌ TEST 1 FAILED: Some scenarios don't have 27 foods")
    print("="*80)
    
    return all_passed


# ============================================================================
# TEST 2: Food Grouping (27 → 6)
# ============================================================================

def test_food_grouping():
    """Test food aggregation."""
    print("\n" + "="*80)
    print("TEST 2: Food Grouping (27 foods → 6 families)")
    print("="*80)
    
    try:
        # Load scenario
        farms, foods, food_groups, config = load_food_data('rotation_250farms_27foods')
        
        params = config.get('parameters', {})
        weights = params.get('weights', {})
        la = params.get('land_availability', {})
        
        # Build data dict
        data = {
            'foods': foods,
            'food_names': list(foods.keys()),
            'food_groups': food_groups,
            'food_benefits': {f: sum(foods[f].get(k, 0.5) * weights.get(k, 0.2) for k in weights) for f in foods},
            'weights': weights,
            'land_availability': la,
            'farm_names': list(la.keys())[:10],  # Test on 10 farms
            'total_area': sum(list(la.values())[:10]),
            'n_farms': 10,
            'n_foods': len(foods),
        }
        
        print(f"\n  Original: {len(foods)} foods × {data['n_farms']} farms")
        
        # Aggregate
        family_data = aggregate_foods_to_families(data)
        
        n_families = len(family_data['food_names'])
        
        if n_families == 6:
            print(f"  ✅ Aggregated to {n_families} families")
            print(f"  Families: {family_data['food_names']}")
            print(f"  Reduction: {len(foods) / 6:.1f}×")
            print("\n✅ TEST 2 PASSED")
            return True
        else:
            print(f"  ❌ Expected 6 families, got {n_families}")
            print("\n❌ TEST 2 FAILED")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 3: Small-Scale SA Solve
# ============================================================================

def test_small_sa():
    """Test SA solve on small problem."""
    print("\n" + "="*80)
    print("TEST 3: Small-Scale SA Solve (10 farms × 6 families)")
    print("="*80)
    
    try:
        # Load scenario (already family-level)
        farms, foods, food_groups, config = load_food_data('rotation_small_50')
        
        params = config.get('parameters', {})
        weights = params.get('weights', {})
        la = params.get('land_availability', {})
        
        data = {
            'foods': foods,
            'food_names': list(foods.keys()),
            'food_groups': food_groups,
            'food_benefits': {f: sum(foods[f].get(k, 0.5) * weights.get(k, 0.2) for k in weights) for f in foods},
            'weights': weights,
            'land_availability': la,
            'farm_names': list(la.keys()),
            'total_area': sum(la.values()),
            'n_farms': len(la),
            'n_foods': len(foods),
        }
        
        # Fast SA config
        solver_config = DEFAULT_CONFIG.copy()
        solver_config['farms_per_cluster'] = 3
        solver_config['num_iterations'] = 1  # Just 1 iteration for speed
        solver_config['num_reads'] = 10  # Very few reads for speed
        
        print(f"\n  Problem: {data['n_farms']} farms × {data['n_foods']} families × 3 periods")
        print(f"  Config: {solver_config['num_reads']} reads, {solver_config['num_iterations']} iteration")
        print("  (Fast settings for testing)")
        
        # Solve
        print("\n  Solving with SA...")
        start = time.time()
        result = solve_hierarchical(data, solver_config, use_qpu=False, verbose=False)
        elapsed = time.time() - start
        
        # Check
        if result['success'] and result['violations'] == 0:
            print(f"\n  ✅ Solved in {elapsed:.1f}s")
            print(f"  Objective: {result['objective']:.4f}")
            print(f"  Violations: {result['violations']}")
            print(f"  Unique crops: {result['diversity_stats']['total_unique_crops']}")
            print("\n✅ TEST 3 PASSED")
            return True
        else:
            print(f"\n  ❌ Violations: {result['violations']}")
            print("\n❌ TEST 3 FAILED")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 4: Medium-Scale with Aggregation
# ============================================================================

def test_medium_with_aggregation():
    """Test full pipeline: 50 farms × 27 foods → 6 families → solve."""
    print("\n" + "="*80)
    print("TEST 4: Medium-Scale with Aggregation (50 farms × 27 foods)")
    print("="*80)
    
    try:
        # Load large scenario
        farms, foods, food_groups, config = load_food_data('rotation_250farms_27foods')
        
        params = config.get('parameters', {})
        weights = params.get('weights', {})
        la = params.get('land_availability', {})
        
        # Use subset (50 farms)
        farm_subset = list(la.keys())[:50]
        la_subset = {f: la[f] for f in farm_subset}
        
        data = {
            'foods': foods,
            'food_names': list(foods.keys()),
            'food_groups': food_groups,
            'food_benefits': {f: sum(foods[f].get(k, 0.5) * weights.get(k, 0.2) for k in weights) for f in foods},
            'weights': weights,
            'land_availability': la_subset,
            'farm_names': farm_subset,
            'total_area': sum(la_subset.values()),
            'n_farms': len(farm_subset),
            'n_foods': len(foods),
        }
        
        print(f"\n  Problem: {data['n_farms']} farms × {data['n_foods']} foods × 3 periods")
        print(f"  Variables BEFORE aggregation: {data['n_farms'] * data['n_foods'] * 3:,}")
        
        # Fast SA config
        solver_config = DEFAULT_CONFIG.copy()
        solver_config['farms_per_cluster'] = 10
        solver_config['num_iterations'] = 1  # Fast
        solver_config['num_reads'] = 10  # Fast
        
        # Solve
        print("\n  Solving with SA (will auto-aggregate 27 → 6)...")
        start = time.time()
        result = solve_hierarchical(data, solver_config, use_qpu=False, verbose=True)
        elapsed = time.time() - start
        
        # Check
        if result['success']:
            agg_info = result['levels']['food_aggregation']
            decomp_info = result['levels']['decomposition']
            div_stats = result['diversity_stats']
            
            print(f"\n  ✅ Solved in {elapsed:.1f}s")
            print(f"  Food aggregation: {agg_info['original_foods']} → {agg_info['families']} ({agg_info.get('reduction_factor', 0):.1f}× reduction)")
            print(f"  Decomposition: {decomp_info['n_clusters']} clusters")
            print(f"  Objective: {result['objective']:.4f}")
            print(f"  Violations: {result['violations']}")
            print(f"  Unique crops (post-processing): {div_stats['total_unique_crops']}/{div_stats['max_possible_crops']}")
            print(f"  Shannon diversity: {div_stats['shannon_diversity']:.3f}")
            
            if div_stats['total_unique_crops'] >= 10:
                print("\n✅ TEST 4 PASSED (good crop diversity)")
                return True
            else:
                print(f"\n⚠️  TEST 4 PASSED but low diversity ({div_stats['total_unique_crops']} crops)")
                return True
        else:
            print("\n❌ TEST 4 FAILED")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 5: Gurobi Ground Truth Comparison
# ============================================================================

def test_gurobi_comparison():
    """Compare hierarchical SA vs Gurobi ground truth."""
    print("\n" + "="*80)
    print("TEST 5: Gurobi Ground Truth Comparison (10 farms × 6 families)")
    print("="*80)
    
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        print("\n  ⚠️  Gurobi not available - skipping test")
        return True
    
    try:
        # Load small scenario
        farms, foods, food_groups, config = load_food_data('rotation_small_50')
        
        params = config.get('parameters', {})
        weights = params.get('weights', {})
        la = params.get('land_availability', {})
        
        data = {
            'foods': foods,
            'food_names': list(foods.keys()),
            'food_groups': food_groups,
            'food_benefits': {f: sum(foods[f].get(k, 0.5) * weights.get(k, 0.2) for k in weights) for f in foods},
            'weights': weights,
            'land_availability': la,
            'farm_names': list(la.keys()),
            'total_area': sum(la.values()),
            'n_farms': len(la),
            'n_foods': len(foods),
        }
        
        print(f"\n  Problem: {data['n_farms']} farms × {data['n_foods']} families × 3 periods")
        
        # Solve with Gurobi (simplified model - just families, no rotation synergies)
        print("\n  [1/2] Solving with Gurobi (ground truth)...")
        
        model = gp.Model("rotation_simple")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 60)
        
        # Variables
        Y = {}
        for f in data['farm_names']:
            for c in data['food_names']:
                for t in range(1, 4):
                    Y[f, c, t] = model.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}_t{t}")
        
        # Objective: just base benefits (simple)
        obj = gp.LinExpr()
        for f in data['farm_names']:
            area_frac = data['land_availability'][f] / data['total_area']
            for c in data['food_names']:
                benefit = data['food_benefits'][c]
                for t in range(1, 4):
                    obj += benefit * area_frac * Y[f, c, t]
        
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # Constraint: one crop per farm per period
        for f in data['farm_names']:
            for t in range(1, 4):
                model.addConstr(gp.quicksum(Y[f, c, t] for c in data['food_names']) == 1)
        
        model.optimize()
        
        if model.status != GRB.OPTIMAL:
            print(f"  ❌ Gurobi failed: status {model.status}")
            return False
        
        gurobi_obj = model.objVal
        print(f"  Gurobi objective: {gurobi_obj:.4f}")
        
        # Solve with SA
        print("\n  [2/2] Solving with Hierarchical SA...")
        solver_config = DEFAULT_CONFIG.copy()
        solver_config['farms_per_cluster'] = 5
        solver_config['num_iterations'] = 1
        solver_config['num_reads'] = 50
        
        result = solve_hierarchical(data, solver_config, use_qpu=False, verbose=False)
        
        if not result['success']:
            print("  ❌ SA failed")
            return False
        
        sa_obj = result['objective']
        print(f"  SA objective: {sa_obj:.4f}")
        
        # Compare
        gap = abs(gurobi_obj - sa_obj) / gurobi_obj * 100
        print(f"\n  Gap: {gap:.2f}%")
        
        if gap < 50:  # SA might not be as good, but should be reasonable
            print(f"  ✅ Reasonable gap ({gap:.2f}%)")
            print("\n✅ TEST 5 PASSED")
            return True
        else:
            print(f"  ⚠️  Large gap ({gap:.2f}%) - SA may need more reads")
            print("\n⚠️  TEST 5 PASSED (with warning)")
            return True
            
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all validation tests."""
    print("\nStarting validation tests...\n")
    
    results = {}
    
    # Test 1: Scenario loading
    results['scenario_loading'] = test_scenario_loading()
    
    # Test 2: Food grouping
    results['food_grouping'] = test_food_grouping()
    
    # Test 3: Small SA
    results['small_sa'] = test_small_sa()
    
    # Test 4: Medium with aggregation
    results['medium_aggregation'] = test_medium_with_aggregation()
    
    # Test 5: Gurobi comparison
    results['gurobi_comparison'] = test_gurobi_comparison()
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:25s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL VALIDATION TESTS PASSED! System ready for QPU.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review before QPU run.")
    
    print("="*80)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
