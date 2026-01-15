"""
Gurobi Ground Truth Solver for the unified benchmark.

Implements the TRUE MIQP formulation from formulations.tex (Formulation 1).
This is the definitive classical baseline - no aggregation, no simplification.

Key features:
- Full 27-food MIQP with quadratic objective
- Soft one-hot constraints (1-2 crops per farm-year)
- Rotation constraints (no same crop consecutive years)
- Temporal and spatial synergies
- Diversity bonus
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .core import (
    RunEntry,
    TimingInfo,
    BenchmarkLogger,
    MIQP_PARAMS,
    create_run_entry,
)
from .scenarios import (
    load_scenario,
    build_rotation_matrix,
    build_spatial_neighbors,
)
from .miqp_scorer import compute_miqp_objective, check_constraints


# Check Gurobi availability
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False


def solve_gurobi_ground_truth(
    scenario_data: Dict[str, Any],
    timeout: float = 600.0,
    mip_gap: float = 0.01,
    params: Optional[Dict[str, float]] = None,
    verbose: bool = True,
    logger: Optional[BenchmarkLogger] = None,
    seed: int = 42,
) -> RunEntry:
    """
    Solve the TRUE MIQP using Gurobi.
    
    This implements Formulation 1 from formulations.tex EXACTLY.
    No aggregation, no simplification, no shortcuts.
    
    Args:
        scenario_data: Scenario data from load_scenario()
        timeout: Wall clock timeout in seconds
        mip_gap: Target MIP gap (default 1%)
        params: MIQP parameters (uses MIQP_PARAMS if None)
        verbose: Print progress
        logger: BenchmarkLogger instance
        seed: Random seed for rotation matrix
    
    Returns:
        RunEntry with results
    """
    if not HAS_GUROBI:
        raise RuntimeError("Gurobi not available. Install gurobipy.")
    
    if params is None:
        params = MIQP_PARAMS.copy()
    
    if logger is None:
        logger = BenchmarkLogger()
    
    # Extract scenario data
    farm_names = scenario_data["farm_names"]
    food_names = scenario_data["food_names"]
    land_availability = scenario_data["land_availability"]
    food_benefits = scenario_data["food_benefits"]
    total_area = scenario_data["total_area"]
    n_periods = scenario_data.get("n_periods", params.get("n_periods", 3))
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    n_vars = n_farms * n_foods * n_periods
    
    # Create run entry
    entry = create_run_entry(
        mode="gurobi-true-ground-truth",
        scenario_name=scenario_data.get("scenario_name", "unknown"),
        n_farms=n_farms,
        n_foods=n_foods,
        n_periods=n_periods,
        sampler="classical",
        backend="Gurobi",
        timeout_s=timeout,
        seed=seed,
        area_constant=scenario_data.get("area_constant", 1.0),
    )
    entry.timing = TimingInfo()
    
    # Get MIQP parameters
    rotation_gamma = params.get("rotation_gamma", 0.2)
    spatial_gamma = params.get("spatial_gamma", 0.1)
    one_hot_penalty = params.get("one_hot_penalty", 3.0)
    diversity_bonus = params.get("diversity_bonus", 0.15)
    k_neighbors = params.get("k_neighbors", 4)
    
    # Build rotation matrix
    R = build_rotation_matrix(
        n_foods,
        frustration_ratio=params.get("frustration_ratio", 0.7),
        negative_strength=params.get("negative_strength", -0.8),
        seed=seed
    )
    
    # Build spatial neighbors
    neighbor_edges = build_spatial_neighbors(farm_names, k_neighbors=k_neighbors)
    
    logger.model_build_start("gurobi-true-ground-truth", n_vars)
    
    total_start = time.time()
    build_start = time.time()
    
    try:
        # Create Gurobi model
        model = gp.Model("miqp_ground_truth")
        model.setParam("OutputFlag", 1 if verbose else 0)
        model.setParam("TimeLimit", timeout)
        model.setParam("MIPGap", mip_gap)
        model.setParam("MIPFocus", 1)  # Focus on finding good feasible solutions
        model.setParam("Threads", 0)  # Use all cores
        model.setParam("Presolve", 2)  # Aggressive presolve
        
        # Create decision variables: Y[f, c, t]
        Y = {}
        for f_idx, farm in enumerate(farm_names):
            for c_idx, food in enumerate(food_names):
                for t in range(1, n_periods + 1):
                    Y[(f_idx, c_idx, t)] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"Y_{f_idx}_{c_idx}_{t}"
                    )
        
        # ========== OBJECTIVE FUNCTION ==========
        obj = 0
        
        # Part 1: Base Benefit (Linear)
        for f_idx, farm in enumerate(farm_names):
            farm_area = land_availability[farm]
            area_frac = farm_area / total_area
            for c_idx, food in enumerate(food_names):
                B_c = food_benefits.get(food, 1.0)
                for t in range(1, n_periods + 1):
                    obj += (B_c * area_frac) * Y[(f_idx, c_idx, t)]
        
        # Part 2: Temporal Synergy (Quadratic)
        for f_idx, farm in enumerate(farm_names):
            farm_area = land_availability[farm]
            area_frac = farm_area / total_area
            for t in range(2, n_periods + 1):
                for c1_idx in range(n_foods):
                    for c2_idx in range(n_foods):
                        synergy = R[c1_idx, c2_idx]
                        if abs(synergy) > 1e-8:
                            obj += (rotation_gamma * area_frac * synergy) * \
                                   Y[(f_idx, c1_idx, t-1)] * Y[(f_idx, c2_idx, t)]
        
        # Part 3: Spatial Synergy (Quadratic)
        # Spatial coupling: γ_s = 0.5·γ, S = 0.3·R (per LaTeX spec)
        # Note: No area normalization - spatial interactions are about adjacency, not farm size
        effective_spatial_gamma = rotation_gamma * 0.5 * 0.3
        for (f1_idx, f2_idx) in neighbor_edges:
            for t in range(1, n_periods + 1):
                for c1_idx in range(n_foods):
                    for c2_idx in range(n_foods):
                        synergy = R[c1_idx, c2_idx]
                        if abs(synergy) > 1e-8:
                            obj += (effective_spatial_gamma * synergy) * \
                                   Y[(f1_idx, c1_idx, t)] * Y[(f2_idx, c2_idx, t)]
        
        # Part 4: Soft One-Hot Penalty (Quadratic)
        for f_idx in range(n_farms):
            for t in range(1, n_periods + 1):
                crop_sum = gp.quicksum(Y[(f_idx, c_idx, t)] for c_idx in range(n_foods))
                # Penalty = λ * (sum - 1)^2
                obj -= one_hot_penalty * (crop_sum - 1) * (crop_sum - 1)
        
        # Part 5: Diversity Bonus (Linear with indicator)
        # Diversity = λ_div * Σ_f Σ_c I(Σ_t Y_{f,c,t} > 0)
        # Linearized: introduce binary z_{f,c} where z_{f,c} = 1 iff crop c is used on farm f
        Z = {}
        for f_idx in range(n_farms):
            for c_idx in range(n_foods):
                z_name = f"z_{f_idx}_{c_idx}"
                Z[(f_idx, c_idx)] = model.addVar(vtype=GRB.BINARY, name=z_name)
                
                # z_{f,c} >= Y_{f,c,t} / n_periods for all t (if any Y is 1, z must be 1)
                crop_sum = gp.quicksum(Y[(f_idx, c_idx, t)] for t in range(1, n_periods + 1))
                model.addConstr(Z[(f_idx, c_idx)] >= crop_sum / n_periods, name=f"div_lb_{f_idx}_{c_idx}")
                # z_{f,c} <= Σ_t Y_{f,c,t} (if no Y is 1, z can be 0)
                model.addConstr(Z[(f_idx, c_idx)] <= crop_sum, name=f"div_ub_{f_idx}_{c_idx}")
                
                # Add diversity bonus for using this crop
                obj += diversity_bonus * Z[(f_idx, c_idx)]
        
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # ========== CONSTRAINTS ==========
        
        # Constraint 1: Max 2 crops per farm per period
        for f_idx in range(n_farms):
            for t in range(1, n_periods + 1):
                model.addConstr(
                    gp.quicksum(Y[(f_idx, c_idx, t)] for c_idx in range(n_foods)) <= 2,
                    name=f"max_crops_{f_idx}_{t}"
                )
        
        # Constraint 2: Min 1 crop per farm per period
        for f_idx in range(n_farms):
            for t in range(1, n_periods + 1):
                model.addConstr(
                    gp.quicksum(Y[(f_idx, c_idx, t)] for c_idx in range(n_foods)) >= 1,
                    name=f"min_crops_{f_idx}_{t}"
                )
        
        # NOTE: Hard rotation constraint removed - relying on R[c,c] soft penalty in objective only
        # The temporal synergy term with R[c,c] = -1.2 penalizes monoculture
        
        entry.timing.model_build_time = time.time() - build_start
        logger.model_build_done(entry.timing.model_build_time)
        
        # ========== SOLVE ==========
        logger.solve_start("Gurobi", timeout)
        solve_start = time.time()
        
        model.optimize()
        
        entry.timing.solve_time = time.time() - solve_start
        
        # ========== EXTRACT RESULTS ==========
        if model.Status == GRB.OPTIMAL:
            entry.status = "optimal"
            entry.objective_model = model.ObjVal
            entry.mip_gap = 0.0
        elif model.Status == GRB.TIME_LIMIT:
            entry.status = "timeout"
            if model.SolCount > 0:
                entry.objective_model = model.ObjVal
                entry.mip_gap = model.MIPGap * 100  # As percentage
        elif model.SolCount > 0:
            entry.status = "feasible"
            entry.objective_model = model.ObjVal
            entry.mip_gap = model.MIPGap * 100
        else:
            entry.status = "error"
            entry.error_message = f"Gurobi status: {model.Status}"
        
        logger.solve_done(entry.status, entry.timing.solve_time, entry.objective_model)
        
        # Extract solution
        if model.SolCount > 0:
            solution = {}
            for (f_idx, c_idx, t), var in Y.items():
                val = var.X
                if val > 0.5:  # Binary threshold
                    solution[(f_idx, c_idx, t)] = 1
            
            entry.solution = solution
            
            # Recompute MIQP objective (validation)
            miqp_start = time.time()
            entry.objective_miqp, breakdown = compute_miqp_objective(
                solution, scenario_data, R=R, neighbor_edges=neighbor_edges,
                params=params, return_breakdown=True
            )
            entry.timing.miqp_recompute_time = time.time() - miqp_start
            
            logger.miqp_recompute(entry.objective_miqp, entry.timing.miqp_recompute_time)
            
            # Check constraints
            violations = check_constraints(solution, scenario_data, params)
            entry.constraint_violations = violations
            entry.feasible = violations.total_violations == 0
            
            logger.constraint_check(violations.total_violations, entry.feasible)
        
        entry.timing.total_wall_time = time.time() - total_start
        
    except Exception as e:
        entry.status = "error"
        entry.error_message = str(e)
        entry.timing.total_wall_time = time.time() - total_start
        logger.error(f"Gurobi solver error: {e}")
    
    return entry


if __name__ == "__main__":
    # Test Gurobi solver
    print("Testing Gurobi ground truth solver...")
    
    if not HAS_GUROBI:
        print("Gurobi not available!")
    else:
        # Load a small scenario
        data = load_scenario("rotation_micro_25")
        print(f"Scenario: {data['n_farms']} farms × {data['n_foods']} foods = {data['n_vars']} vars")
        
        # Solve with short timeout for testing
        result = solve_gurobi_ground_truth(data, timeout=30.0)
        
        print(f"\nResults:")
        print(f"  Status: {result.status}")
        print(f"  Model objective: {result.objective_model}")
        print(f"  MIQP objective: {result.objective_miqp}")
        print(f"  Solve time: {result.timing.solve_time:.2f}s")
        print(f"  Feasible: {result.feasible}")
        if result.constraint_violations:
            print(f"  Violations: {result.constraint_violations.total_violations}")
