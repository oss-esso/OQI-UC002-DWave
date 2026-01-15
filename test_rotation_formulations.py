#!/usr/bin/env python3
"""
Test script to compare three rotation formulation variants in Gurobi:
1. Hard constraint only (no R[c,c] diagonal penalty in objective)
2. Rotation term only (no hard constraint, rely on R[c,c] soft penalty)
3. Both (current implementation)

This validates whether the hard constraint and soft penalty produce different results.
"""

import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from unified_benchmark.scenarios import load_scenario, build_spatial_neighbors
from unified_benchmark.miqp_scorer import MIQP_PARAMS, compute_miqp_objective, check_constraints


def build_rotation_matrix_variant(food_names: list, variant: str, seed: int = 42) -> np.ndarray:
    """
    Build rotation matrix with different diagonal (monoculture) treatments.
    
    variant:
        - "standard": R[c,c] = -1.2 (monoculture penalty in objective)
        - "zero_diagonal": R[c,c] = 0 (no monoculture penalty in objective)
    """
    rng = np.random.default_rng(seed)
    n = len(food_names)
    R = np.zeros((n, n))
    
    negative_strength = MIQP_PARAMS.get("negative_strength", -0.8)
    frustration_ratio = MIQP_PARAMS.get("frustration_ratio", 0.7)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                if variant == "zero_diagonal":
                    R[i, j] = 0.0  # No monoculture penalty in objective
                else:  # "standard"
                    R[i, j] = negative_strength * 1.5  # = -1.2
            elif rng.random() < frustration_ratio:
                R[i, j] = rng.uniform(negative_strength * 1.2, negative_strength * 0.3)
            else:
                R[i, j] = rng.uniform(0.02, 0.20)
    
    return R


def solve_with_gurobi(
    scenario_name: str,
    use_hard_constraint: bool,
    use_rotation_diagonal: bool,
    time_limit: float = 60.0,
    seed: int = 42
) -> dict:
    """
    Solve the rotation problem with Gurobi using specified formulation variant.
    
    Args:
        scenario_name: Name of scenario to load
        use_hard_constraint: If True, add Y[f,c,t] + Y[f,c,t+1] <= 1 constraint
        use_rotation_diagonal: If True, include R[c,c] diagonal in objective
        time_limit: Gurobi time limit in seconds
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with solution details and metrics
    """
    # Load scenario
    data = load_scenario(scenario_name)
    farm_names = data["farm_names"]
    food_names = data["food_names"]
    land_availability = data["land_availability"]
    food_benefits = data["food_benefits"]
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    n_periods = MIQP_PARAMS.get("n_periods", 3)
    
    # Build matrices
    R_variant = "standard" if use_rotation_diagonal else "zero_diagonal"
    R = build_rotation_matrix_variant(food_names, R_variant, seed=seed)
    neighbor_edges = build_spatial_neighbors(farm_names, k_neighbors=MIQP_PARAMS.get("k_neighbors", 4))
    
    # Parameters
    rotation_gamma = MIQP_PARAMS.get("rotation_gamma", 0.2)
    spatial_gamma = MIQP_PARAMS.get("spatial_gamma", 0.1)
    one_hot_penalty = MIQP_PARAMS.get("one_hot_penalty", 3.0)
    diversity_bonus = MIQP_PARAMS.get("diversity_bonus", 0.15)
    
    total_area = sum(land_availability.values())
    
    # Create model
    model = gp.Model("rotation_test")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Seed", seed)
    
    # Decision variables: Y[farm, crop, period]
    Y = {}
    for f_idx in range(n_farms):
        for c_idx in range(n_foods):
            for t in range(1, n_periods + 1):
                Y[(f_idx, c_idx, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{f_idx}_{c_idx}_{t}")
    
    # Auxiliary variables for diversity
    U = {}  # U[farm, crop] = 1 if crop used on farm in any period
    for f_idx in range(n_farms):
        for c_idx in range(n_foods):
            U[(f_idx, c_idx)] = model.addVar(vtype=GRB.BINARY, name=f"U_{f_idx}_{c_idx}")
    
    model.update()
    
    # Build objective
    obj = gp.QuadExpr()
    
    # Part 1: Base benefit (linear)
    for f_idx, farm in enumerate(farm_names):
        area_frac = land_availability[farm] / total_area
        for c_idx, crop in enumerate(food_names):
            benefit = food_benefits[crop]
            for t in range(1, n_periods + 1):
                obj += benefit * area_frac * Y[(f_idx, c_idx, t)]
    
    # Part 2: Temporal synergy (quadratic) - uses R matrix
    for f_idx, farm in enumerate(farm_names):
        area_frac = land_availability[farm] / total_area
        for t in range(2, n_periods + 1):
            for c1_idx in range(n_foods):
                for c2_idx in range(n_foods):
                    synergy = R[c1_idx, c2_idx]
                    if abs(synergy) > 1e-8:
                        obj += rotation_gamma * area_frac * synergy * Y[(f_idx, c1_idx, t-1)] * Y[(f_idx, c2_idx, t)]
    
    # Part 3: Spatial synergy (quadratic)
    for (f1_idx, f2_idx) in neighbor_edges:
        for t in range(1, n_periods + 1):
            for c1_idx in range(n_foods):
                for c2_idx in range(n_foods):
                    synergy = R[c1_idx, c2_idx]
                    if abs(synergy) > 1e-8:
                        obj += spatial_gamma * synergy * 0.3 / total_area * Y[(f1_idx, c1_idx, t)] * Y[(f2_idx, c2_idx, t)]
    
    # Part 4: One-hot penalty (quadratic soft constraint)
    for f_idx in range(n_farms):
        for t in range(1, n_periods + 1):
            sum_y = gp.LinExpr()
            for c_idx in range(n_foods):
                sum_y += Y[(f_idx, c_idx, t)]
            # Penalty: lambda_oh * (sum_y - 1)^2
            obj -= one_hot_penalty * (sum_y * sum_y - 2 * sum_y + 1)
    
    # Part 5: Diversity bonus
    for f_idx in range(n_farms):
        for c_idx in range(n_foods):
            obj += diversity_bonus * U[(f_idx, c_idx)]
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints
    
    # Constraint 1: Max 2 crops per farm-period (soft upper bound)
    for f_idx in range(n_farms):
        for t in range(1, n_periods + 1):
            model.addConstr(
                gp.quicksum(Y[(f_idx, c_idx, t)] for c_idx in range(n_foods)) <= 2,
                name=f"max_crops_{f_idx}_{t}"
            )
    
    # Constraint 2: Min 1 crop per farm-period (soft lower bound)
    for f_idx in range(n_farms):
        for t in range(1, n_periods + 1):
            model.addConstr(
                gp.quicksum(Y[(f_idx, c_idx, t)] for c_idx in range(n_foods)) >= 1,
                name=f"min_crops_{f_idx}_{t}"
            )
    
    # Constraint 3: HARD ROTATION CONSTRAINT (conditional)
    if use_hard_constraint:
        for f_idx in range(n_farms):
            for c_idx in range(n_foods):
                for t in range(1, n_periods):
                    model.addConstr(
                        Y[(f_idx, c_idx, t)] + Y[(f_idx, c_idx, t + 1)] <= 1,
                        name=f"rotation_{f_idx}_{c_idx}_{t}"
                    )
    
    # Constraint 4: Link U to Y (U[f,c] = 1 if any Y[f,c,t] = 1)
    for f_idx in range(n_farms):
        for c_idx in range(n_foods):
            for t in range(1, n_periods + 1):
                model.addConstr(U[(f_idx, c_idx)] >= Y[(f_idx, c_idx, t)], name=f"link_U_{f_idx}_{c_idx}_{t}")
    
    # Solve
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time
    
    # Extract results
    result = {
        "scenario": scenario_name,
        "use_hard_constraint": use_hard_constraint,
        "use_rotation_diagonal": use_rotation_diagonal,
        "variant_name": get_variant_name(use_hard_constraint, use_rotation_diagonal),
        "solve_time": solve_time,
        "status": model.Status,
        "status_name": get_status_name(model.Status),
    }
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        # Extract solution
        solution = {}
        for f_idx in range(n_farms):
            for c_idx in range(n_foods):
                for t in range(1, n_periods + 1):
                    if Y[(f_idx, c_idx, t)].X > 0.5:
                        solution[(f_idx, c_idx, t)] = 1
        
        result["gurobi_objective"] = model.ObjVal
        result["solution_size"] = len(solution)
        
        # Count rotation violations in solution
        rotation_violations = 0
        for f_idx in range(n_farms):
            for c_idx in range(n_foods):
                for t in range(1, n_periods):
                    if solution.get((f_idx, c_idx, t), 0) == 1 and solution.get((f_idx, c_idx, t+1), 0) == 1:
                        rotation_violations += 1
        
        result["rotation_violations"] = rotation_violations
        
        # Compute MIQP objective with STANDARD R matrix (for fair comparison)
        R_standard = build_rotation_matrix_variant(food_names, "standard", seed=seed)
        miqp_obj = compute_miqp_objective(
            solution, data, R=R_standard, neighbor_edges=neighbor_edges, params=MIQP_PARAMS
        )
        result["miqp_objective_standard"] = miqp_obj
        
        # Also compute with zero-diagonal R for comparison
        R_zero = build_rotation_matrix_variant(food_names, "zero_diagonal", seed=seed)
        miqp_obj_zero = compute_miqp_objective(
            solution, data, R=R_zero, neighbor_edges=neighbor_edges, params=MIQP_PARAMS
        )
        result["miqp_objective_zero_diag"] = miqp_obj_zero
        
        # Store solution for later analysis
        result["solution"] = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in solution.items()}
        
    else:
        result["gurobi_objective"] = None
        result["solution_size"] = 0
        result["rotation_violations"] = None
        result["miqp_objective_standard"] = None
        result["miqp_objective_zero_diag"] = None
        result["solution"] = {}
    
    return result


def get_variant_name(use_hard_constraint: bool, use_rotation_diagonal: bool) -> str:
    """Get human-readable variant name."""
    if use_hard_constraint and use_rotation_diagonal:
        return "BOTH (hard constraint + R diagonal)"
    elif use_hard_constraint and not use_rotation_diagonal:
        return "HARD CONSTRAINT ONLY (no R diagonal)"
    elif not use_hard_constraint and use_rotation_diagonal:
        return "ROTATION TERM ONLY (no hard constraint)"
    else:
        return "NEITHER (no rotation enforcement)"


def get_status_name(status: int) -> str:
    """Convert Gurobi status code to name."""
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return status_map.get(status, f"UNKNOWN({status})")


def run_comparison_test(scenarios: list, time_limit: float = 60.0, seed: int = 42) -> dict:
    """
    Run comparison across all three formulation variants for given scenarios.
    """
    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "time_limit": time_limit,
            "seed": seed,
            "miqp_params": MIQP_PARAMS,
        },
        "scenarios": {}
    }
    
    variants = [
        (True, True, "both"),           # Hard constraint + R diagonal
        (True, False, "hard_only"),     # Hard constraint only
        (False, True, "soft_only"),     # R diagonal only (no hard constraint)
    ]
    
    for scenario_name in scenarios:
        print(f"\n{'='*60}")
        print(f"Testing scenario: {scenario_name}")
        print(f"{'='*60}")
        
        scenario_results = {}
        
        for use_hard, use_diag, variant_key in variants:
            variant_name = get_variant_name(use_hard, use_diag)
            print(f"\n  Variant: {variant_name}")
            print(f"  {'-'*50}")
            
            result = solve_with_gurobi(
                scenario_name=scenario_name,
                use_hard_constraint=use_hard,
                use_rotation_diagonal=use_diag,
                time_limit=time_limit,
                seed=seed
            )
            
            print(f"    Status: {result['status_name']}")
            print(f"    Solve time: {result['solve_time']:.2f}s")
            
            if result['gurobi_objective'] is not None:
                print(f"    Gurobi objective: {result['gurobi_objective']:.6f}")
                print(f"    MIQP objective (std R): {result['miqp_objective_standard']:.6f}")
                print(f"    MIQP objective (zero diag): {result['miqp_objective_zero_diag']:.6f}")
                print(f"    Rotation violations: {result['rotation_violations']}")
                print(f"    Solution size: {result['solution_size']} assignments")
            
            scenario_results[variant_key] = result
        
        # Compute differences
        if all(scenario_results[v]["gurobi_objective"] is not None for v in ["both", "hard_only", "soft_only"]):
            print(f"\n  COMPARISON:")
            print(f"  {'-'*50}")
            
            both = scenario_results["both"]
            hard = scenario_results["hard_only"]
            soft = scenario_results["soft_only"]
            
            # Compare objectives
            print(f"    MIQP Objective (standard R matrix):")
            print(f"      BOTH:      {both['miqp_objective_standard']:.6f}")
            print(f"      HARD ONLY: {hard['miqp_objective_standard']:.6f}")
            print(f"      SOFT ONLY: {soft['miqp_objective_standard']:.6f}")
            
            diff_hard_both = hard['miqp_objective_standard'] - both['miqp_objective_standard']
            diff_soft_both = soft['miqp_objective_standard'] - both['miqp_objective_standard']
            
            print(f"\n    Differences from BOTH:")
            print(f"      HARD ONLY - BOTH: {diff_hard_both:+.6f}")
            print(f"      SOFT ONLY - BOTH: {diff_soft_both:+.6f}")
            
            print(f"\n    Rotation violations:")
            print(f"      BOTH:      {both['rotation_violations']}")
            print(f"      HARD ONLY: {hard['rotation_violations']}")
            print(f"      SOFT ONLY: {soft['rotation_violations']}")
            
            scenario_results["comparison"] = {
                "diff_hard_vs_both": diff_hard_both,
                "diff_soft_vs_both": diff_soft_both,
                "hard_violations": hard['rotation_violations'],
                "soft_violations": soft['rotation_violations'],
                "both_violations": both['rotation_violations'],
            }
        
        results["scenarios"][scenario_name] = scenario_results
    
    return results


def main():
    """Main entry point."""
    # Test scenarios (from small to larger)
    scenarios = [
    "rotation_micro_25",
    "rotation_small_50",
    "rotation_15farms_6foods",
    "rotation_medium_100",
    "rotation_25farms_6foods",
    "rotation_large_200",
    "rotation_50farms_6foods",
    "rotation_75farms_6foods",
    "rotation_100farms_6foods",
    "rotation_25farms_27foods",
    "rotation_150farms_6foods",
    "rotation_50farms_27foods",
    "rotation_75farms_27foods",
    "rotation_100farms_27foods",
    "rotation_150farms_27foods",
    "rotation_200farms_27foods",
    "rotation_250farms_27foods",
    "rotation_350farms_27foods",
    "rotation_500farms_27foods",
    "rotation_1000farms_27foods",
]
    
    print("="*60)
    print("ROTATION FORMULATION COMPARISON TEST")
    print("="*60)
    print("\nThis test compares three formulation variants:")
    print("  1. BOTH: Hard constraint + R[c,c] diagonal penalty")
    print("  2. HARD ONLY: Hard constraint, R[c,c] = 0")
    print("  3. SOFT ONLY: No hard constraint, R[c,c] = -1.2")
    print("\nAll solutions are re-scored with the STANDARD R matrix")
    print("to enable fair objective comparison.")
    print("="*60)
    
    # Run tests
    results = run_comparison_test(
        scenarios=scenarios,
        time_limit=60.0,
        seed=42
    )
    
    # Save results
    output_file = Path("rotation_formulation_comparison_results.json")
    with open(output_file, "w") as f:
        # Convert solution dicts for JSON serialization
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for scenario_name, scenario_results in results["scenarios"].items():
        print(f"\n{scenario_name}:")
        
        if "comparison" in scenario_results:
            comp = scenario_results["comparison"]
            print(f"  Objective difference (HARD vs BOTH): {comp['diff_hard_vs_both']:+.6f}")
            print(f"  Objective difference (SOFT vs BOTH): {comp['diff_soft_vs_both']:+.6f}")
            print(f"  Rotation violations - BOTH: {comp['both_violations']}, HARD: {comp['hard_violations']}, SOFT: {comp['soft_violations']}")
            
            if comp['soft_violations'] > 0:
                print(f"  ⚠️  SOFT ONLY has rotation violations!")
            if abs(comp['diff_hard_vs_both']) < 1e-6 and abs(comp['diff_soft_vs_both']) < 1e-6:
                print(f"  ✓ All variants produce same objective (formulations equivalent)")
            elif abs(comp['diff_hard_vs_both']) < 1e-6:
                print(f"  ✓ HARD ONLY equivalent to BOTH")
            else:
                print(f"  ⚠️  Formulations produce different results!")
        else:
            print(f"  Could not compare (some variants failed)")


if __name__ == "__main__":
    main()
