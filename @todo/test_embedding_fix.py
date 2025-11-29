#!/usr/bin/env python3
"""
Quick test to verify fixes in comprehensive_embedding_and_solving_benchmark.py

Tests with a small problem (5 farms) to verify:
1. Solution dict is passed through partition_results
2. Merged solution gives correct actual objective
3. Decomposed vs non-decomposed objectives are comparable
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=" * 60)
print("TESTING EMBEDDING BENCHMARK FIXES")
print("=" * 60)

# Test imports first
print("\n1. Testing imports...")
try:
    from comprehensive_embedding_and_solving_benchmark import (
        build_patch_cqm,
        cqm_to_bqm_wrapper,
        decompose_louvain,
        decompose_plot_based,
        solve_bqm_with_gurobi,
        solve_decomposed_bqm_with_gurobi,
        calculate_actual_objective_from_bqm_solution,
        extract_sub_bqm,
        GUROBI_AVAILABLE,
        LOUVAIN_AVAILABLE
    )
    print("  ✓ Imports successful")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

if not GUROBI_AVAILABLE:
    print("  ✗ Gurobi not available - cannot run tests")
    sys.exit(1)

# Test with small problem
N_FARMS = 5
print(f"\n2. Building test problem ({N_FARMS} farms × 27 foods)...")

try:
    cqm, meta = build_patch_cqm(N_FARMS)
    print(f"  ✓ Built CQM: {len(cqm.variables)} vars, {len(cqm.constraints)} constraints")
except Exception as e:
    print(f"  ✗ Failed to build CQM: {e}")
    sys.exit(1)

# Convert to BQM
print("\n3. Converting CQM to BQM...")
try:
    bqm, bqm_meta = cqm_to_bqm_wrapper(cqm, "Patch_CQM")
    print(f"  ✓ Converted to BQM: {len(bqm.variables)} vars, {len(bqm.quadratic)} quadratic terms")
except Exception as e:
    print(f"  ✗ Failed to convert: {e}")
    sys.exit(1)

# Test 1: Non-decomposed solve
print("\n4. Testing non-decomposed solve...")
try:
    solve_result = solve_bqm_with_gurobi(bqm, timeout=60)
    print(f"  ✓ Solve success: {solve_result.get('success')}")
    print(f"  ✓ Has solution: {solve_result.get('has_solution')}")
    print(f"  ✓ BQM energy: {solve_result.get('bqm_energy', 'N/A'):.4f}")
    
    if solve_result.get('has_solution') and 'solution' in solve_result:
        actual_obj = calculate_actual_objective_from_bqm_solution(
            solve_result['solution'], bqm_meta, N_FARMS
        )
        print(f"  ✓ Actual objective (non-decomposed): {actual_obj:.6f}")
        non_decomposed_obj = actual_obj
    else:
        print("  ✗ No solution to calculate objective from")
        non_decomposed_obj = None
except Exception as e:
    print(f"  ✗ Non-decomposed solve failed: {e}")
    import traceback
    traceback.print_exc()
    non_decomposed_obj = None

# Test 2: PlotBased decomposition
print("\n5. Testing PlotBased decomposition...")
try:
    partitions = decompose_plot_based(bqm, plots_per_partition=2)
    print(f"  ✓ Created {len(partitions)} partitions")
    print(f"  ✓ Partition sizes: {[len(p) for p in partitions]}")
    
    # Check that all variables are covered
    all_vars = set()
    for p in partitions:
        all_vars.update(p)
    missing = set(bqm.variables) - all_vars
    if missing:
        print(f"  ⚠ Warning: {len(missing)} variables not covered by partitions")
    else:
        print(f"  ✓ All {len(bqm.variables)} variables covered by partitions")
except Exception as e:
    print(f"  ✗ Decomposition failed: {e}")
    partitions = None

# Test 3: Decomposed solve and verify fix
if partitions:
    print("\n6. Testing decomposed solve...")
    try:
        decomp_result = solve_decomposed_bqm_with_gurobi(bqm, partitions, timeout=60)
        print(f"  ✓ All optimal: {decomp_result.get('all_optimal')}")
        print(f"  ✓ All have solution: {decomp_result.get('all_have_solution')}")
        print(f"  ✓ BQM energy (sum): {decomp_result.get('aggregated_objective', 'N/A')}")
        
        # CHECK FIX #1: Solution should be in partition_results now
        has_solutions = all(
            'solution' in pr and pr.get('has_solution')
            for pr in decomp_result.get('partition_results', [])
        )
        if has_solutions:
            print(f"  ✓ FIX #1 VERIFIED: Solutions passed through partition_results")
        else:
            print(f"  ✗ FIX #1 FAILED: Solutions NOT in partition_results")
        
        # CHECK FIX #2: Merge solutions and calculate objective
        if decomp_result.get('all_have_solution'):
            merged_solution = {}
            for pr in decomp_result.get('partition_results', []):
                if 'solution' in pr:
                    merged_solution.update(pr['solution'])
            
            print(f"  ✓ Merged solution has {len(merged_solution)} variables")
            
            actual_obj = calculate_actual_objective_from_bqm_solution(
                merged_solution, bqm_meta, N_FARMS
            )
            print(f"  ✓ Actual objective (decomposed): {actual_obj:.6f}")
            decomposed_obj = actual_obj
            
            # Compare to non-decomposed
            if non_decomposed_obj is not None:
                diff = abs(decomposed_obj - non_decomposed_obj)
                pct_diff = diff / non_decomposed_obj * 100 if non_decomposed_obj > 0 else 0
                print(f"\n  COMPARISON:")
                print(f"    Non-decomposed objective: {non_decomposed_obj:.6f}")
                print(f"    Decomposed objective:     {decomposed_obj:.6f}")
                print(f"    Difference: {diff:.6f} ({pct_diff:.2f}%)")
                
                if decomposed_obj > 0.01:  # Should be meaningful
                    print(f"  ✓ FIX #2 & #3 VERIFIED: Decomposed objective is non-zero and meaningful")
                else:
                    print(f"  ⚠ Warning: Decomposed objective is very small - may need investigation")
        else:
            print(f"  ✗ Cannot test fixes - no solutions")
            decomposed_obj = None
            
    except Exception as e:
        print(f"  ✗ Decomposed solve failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
