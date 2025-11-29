#!/usr/bin/env python3
"""Quick test to verify CQM and BQM objectives match PuLP reference"""

import sys
sys.path.insert(0, 'D:/Projects/OQI-UC002-DWave')
sys.path.insert(0, 'D:/Projects/OQI-UC002-DWave/@todo')

from comprehensive_embedding_and_solving_benchmark import (
    load_real_data, generate_land_data, create_real_config,
    solve_cqm_with_gurobi, solve_bqm_with_gurobi,
    calculate_actual_objective_from_bqm_solution,
    N_FOODS
)
from dimod import cqm_to_bqm

# Import create_cqm_plots
sys.path.insert(0, 'D:/Projects/OQI-UC002-DWave/Benchmark Scripts')
from solver_runner_BINARY import create_cqm_plots

def test_objectives():
    n_farms = 25
    
    print("="*70)
    print("OBJECTIVE VERIFICATION TEST")
    print("="*70)
    print(f"\nPuLP Reference for 25 farms: ~0.33-0.38")
    print()
    
    # Build CQM
    print("Building CQM...")
    foods, food_groups, base_config, weights = load_real_data()
    land_data = generate_land_data(n_farms, total_land=100.0)
    config = create_real_config(land_data, base_config)
    
    farms = list(land_data.keys())
    food_names = list(foods.keys())[:N_FOODS]
    
    cqm, Y, constraint_metadata = create_cqm_plots(farms, foods, food_groups, config)
    
    print(f"  CQM: {len(cqm.variables)} vars, {len(cqm.constraints)} constraints")
    
    # Solve CQM with Gurobi
    print("\n[1/2] Solving CQM with Gurobi...")
    cqm_result = solve_cqm_with_gurobi(cqm, timeout=300)
    cqm_raw = cqm_result.get("objective", None)
    cqm_obj = -cqm_raw if cqm_raw else None  # CQM minimizes negative
    
    print(f"  CQM objective (raw): {cqm_raw}")
    print(f"  CQM objective (actual): {cqm_obj}")
    
    # Convert to BQM and solve
    print("\n[2/2] Converting to BQM and solving with Gurobi...")
    bqm, _ = cqm_to_bqm(cqm, lagrange_multiplier=10.0)
    print(f"  BQM: {len(bqm.variables)} vars, {len(bqm.quadratic)} quadratic terms")
    
    meta = {
        "unit_names": farms,
        "food_names": food_names,
        "weights": weights,
        "total_land": 100.0
    }
    
    bqm_result = solve_bqm_with_gurobi(bqm, timeout=300)
    bqm_energy = bqm_result.get("bqm_energy", None)
    
    # Calculate actual objective from BQM solution
    actual_obj = None
    if bqm_result.get("has_solution") and "solution" in bqm_result:
        actual_obj = calculate_actual_objective_from_bqm_solution(
            bqm_result["solution"], meta, n_farms
        )
    
    print(f"  BQM energy (with penalties): {bqm_energy}")
    print(f"  BQM actual objective: {actual_obj}")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  PuLP Reference:  ~0.33-0.38")
    print(f"  CQM (Gurobi):    {cqm_obj:.4f}" if cqm_obj else "  CQM (Gurobi):    FAILED")
    print(f"  BQM (Gurobi):    {actual_obj:.4f}" if actual_obj else "  BQM (Gurobi):    FAILED/TIMEOUT")
    
    if cqm_obj:
        status = "✅" if 0.25 <= cqm_obj <= 0.50 else "❌"
        print(f"\n{status} CQM objective in expected range [0.25, 0.50]")
    
    if actual_obj:
        match = "✅" if abs(actual_obj - cqm_obj) < 0.1 else "❌"
        print(f"{match} BQM objective matches CQM (diff: {abs(actual_obj - cqm_obj):.4f})")

if __name__ == "__main__":
    test_objectives()
