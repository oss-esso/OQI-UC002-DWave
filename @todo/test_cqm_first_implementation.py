#!/usr/bin/env python3
"""
Test script for CQM-First Decomposition implementation.

This tests that partitioning the CQM first (before BQM conversion)
preserves constraints and achieves 0 violations.

Key findings:
- BQM-First: Converts CQM→BQM first, then partitions. Cutting penalty edges
  breaks constraint enforcement, causing 25-30 violations.
- CQM-First: Partitions CQM first, then converts each partition to BQM.
  Constraints are preserved, achieving 0 violations with PlotBased partition.

NOTE: PlotBased partitioning (1 farm per partition) works best because it
allows precise tracking of MaxPlots constraints across all farms.
"""

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')

# Import from qpu_benchmark
from qpu_benchmark import (
    load_problem_data, 
    build_binary_cqm, 
    solve_ground_truth,
    solve_decomposition_sa,
    solve_cqm_first_decomposition_sa,
    count_violations,
    extract_solution,
    LOG
)

def main():
    print("="*80)
    print("CQM-FIRST DECOMPOSITION TEST")
    print("Comparing: BQM-First (standard) vs CQM-First (constraint-preserving)")
    print("="*80)
    
    # Test with 25 farms (manageable size)
    n_farms = 25
    print(f"\nProblem size: {n_farms} farms")
    
    # Load data and build CQM
    print("\n[1] Loading data and building CQM...")
    data = load_problem_data(n_farms)
    cqm, metadata = build_binary_cqm(data)
    print(f"    CQM: {metadata['n_variables']} vars, {metadata['n_constraints']} constraints")
    
    # Ground truth
    print("\n[2] Solving ground truth (Gurobi)...")
    gt_result = solve_ground_truth(data)
    print(f"    Objective: {gt_result['objective']:.6f}")
    print(f"    Violations: {gt_result['violations']}")
    
    results = {}
    
    # Test BQM-First decomposition (standard approach)
    methods_to_test = ['PlotBased']  # Focus on PlotBased - best for constraint preservation
    
    print("\n" + "="*80)
    print("BQM-FIRST DECOMPOSITION (Standard - May Cut Constraint Edges)")
    print("="*80)
    
    for method in methods_to_test:
        print(f"\n[3a] Testing BQM-First {method}...")
        bqm_result = solve_decomposition_sa(cqm, data, method=method, verbose=False)
        gap = ((gt_result['objective'] - bqm_result['objective']) / gt_result['objective'] * 100)
        print(f"    Objective: {bqm_result['objective']:.6f} (gap: {gap:.1f}%)")
        print(f"    Violations: {bqm_result['violations']} {'✗ INFEASIBLE' if bqm_result['violations'] > 0 else '✓ FEASIBLE'}")
        results[f'BQM_First_{method}'] = bqm_result
    
    print("\n" + "="*80)
    print("CQM-FIRST DECOMPOSITION (Constraint-Preserving)")
    print("="*80)
    
    for method in methods_to_test:
        print(f"\n[3b] Testing CQM-First {method}...")
        cqm_result = solve_cqm_first_decomposition_sa(cqm, data, method=method, verbose=False)
        gap = ((gt_result['objective'] - cqm_result['objective']) / gt_result['objective'] * 100)
        print(f"    Objective: {cqm_result['objective']:.6f} (gap: {gap:.1f}%)")
        print(f"    Violations: {cqm_result['violations']} {'✗ INFEASIBLE' if cqm_result['violations'] > 0 else '✓ FEASIBLE'}")
        results[f'CQM_First_{method}'] = cqm_result
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Method':<30} {'Objective':>10} {'Gap%':>8} {'Violations':>12} {'Status':<12}")
    print("-"*80)
    print(f"{'Ground Truth (Gurobi)':<30} {gt_result['objective']:>10.6f} {'0.0':>8} {gt_result['violations']:>12} {'✓ OPTIMAL':<12}")
    
    for name, r in results.items():
        gap = ((gt_result['objective'] - r['objective']) / gt_result['objective'] * 100) if gt_result['objective'] > 0 else 0
        status = '✓ FEASIBLE' if r['violations'] == 0 else '✗ INFEASIBLE'
        print(f"{name:<30} {r['objective']:>10.6f} {gap:>8.1f} {r['violations']:>12} {status:<12}")
    
    # Check key hypothesis: CQM-First PlotBased should have 0 violations
    cqm_first_plotbased_violations = results.get('CQM_First_PlotBased', {}).get('violations', -1)
    bqm_first_violations = sum(r['violations'] for name, r in results.items() if 'BQM_First' in name)
    
    print("\n" + "="*80)
    print("HYPOTHESIS TEST")
    print("="*80)
    print(f"\nBQM-First PlotBased violations: {results.get('BQM_First_PlotBased', {}).get('violations', 'N/A')}")
    print(f"CQM-First PlotBased violations: {cqm_first_plotbased_violations}")
    
    if cqm_first_plotbased_violations == 0 and bqm_first_violations > 0:
        print("\n✅ HYPOTHESIS CONFIRMED: CQM-First PlotBased decomposition preserves constraints!")
        print("   BQM-First cuts penalty edges, causing violations.")
        print("   CQM-First partitions before BQM conversion, preserving constraints.")
    elif cqm_first_plotbased_violations == 0 and bqm_first_violations == 0:
        print("\n⚠️ Both methods achieved 0 violations - may need larger problem to show difference")
    else:
        print(f"\n⚠️ Unexpected: CQM-First PlotBased had {cqm_first_plotbased_violations} violations - investigate!")
    
    return 0 if cqm_first_plotbased_violations == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
