#!/usr/bin/env python3
"""Test that the CQM formulation gives objective close to PuLP reference (0.387)"""

import sys
import time
sys.path.insert(0, 'D:/Projects/OQI-UC002-DWave')

from dimod import BinaryQuadraticModel
import numpy as np

# Import from the benchmark script
from comprehensive_embedding_and_solving_benchmark import (
    build_patch_cqm,
    solve_cqm_with_gurobi,
    load_real_data,
    generate_land_data
)

def test_cqm_objective():
    """Test CQM gives objective matching PuLP reference"""
    
    print("=" * 70)
    print("TESTING CQM OBJECTIVE vs PuLP REFERENCE")
    print("=" * 70)
    print()
    print("PuLP Reference for 25 farms: 0.3876 (maximization, normalized by 100ha)")
    print()
    
    n_farms = 25
    
    # Build and solve CQM with Gurobi
    print("[1/1] Building and solving CQM...")
    cqm, meta = build_patch_cqm(n_farms)
    print(f"  Variables: {meta.get('variables', len(cqm.variables))}")
    print(f"  Constraints: {meta.get('constraints', len(cqm.constraints))}")
    print(f"  Type: {meta.get('type', 'unknown')}")
    
    cqm_result = solve_cqm_with_gurobi(cqm)
    cqm_obj = cqm_result.get('objective', None)
    # CQM minimizes negative, so negate to get actual maximized value
    cqm_actual = -cqm_obj if cqm_obj else None
    print(f"  CQM objective (raw): {cqm_obj}")
    print(f"  CQM objective (actual): {cqm_actual}")
    print()
    
    # Results comparison
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"  PuLP Reference:     0.3876")
    print(f"  CQM (Gurobi):       {cqm_actual:.4f}" if cqm_actual else "  CQM (Gurobi): FAILED")
    print()
    
    # Check if CQM is close to reference
    if cqm_actual:
        error = abs(cqm_actual - 0.3876) / 0.3876 * 100
        print(f"  CQM error vs PuLP: {error:.2f}%")
        if error < 5:
            print("  ✅ CQM objective matches PuLP reference!")
        elif error < 15:
            print("  ⚠️  CQM objective is within 15% (acceptable for different constraint sets)")
        else:
            print("  ❌ CQM objective differs significantly from reference")

if __name__ == "__main__":
    test_cqm_objective()
