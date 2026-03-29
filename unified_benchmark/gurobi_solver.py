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
import scipy.sparse
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
        model.setParam("MIPFocus", 1)
        model.setParam("Threads", 0)
        model.setParam("Presolve", 2)

        # ── Variable layout ──────────────────────────────────────────────────
        # Flat binary vector x = [Y_flat, Z_flat]
        # Y[f, c, t] → x[f*n_foods*n_periods + c*n_periods + t]   (t: 0-based)
        # Z[f, c]    → x[n_Y + f*n_foods + c]
        n_Y = n_farms * n_foods * n_periods
        n_Z = n_farms * n_foods
        n_x = n_Y + n_Z
        x = model.addMVar(n_x, vtype=GRB.BINARY, name="x")

        fa = np.arange(n_farms)
        ca = np.arange(n_foods)
        ta = np.arange(n_periods)

        # Y_idx[f, c, t]: integer index into x  (shape n_farms × n_foods × n_periods)
        Y_idx = (fa[:, None, None] * n_foods * n_periods
                 + ca[None, :, None] * n_periods
                 + ta[None, None, :])
        # Z_idx[f, c]: integer index into x  (shape n_farms × n_foods)
        Z_idx = n_Y + fa[:, None] * n_foods + ca[None, :]

        # ── Pre-compute coefficient arrays ───────────────────────────────────
        area_arr = np.array([land_availability[farm] / total_area for farm in farm_names])
        B_arr    = np.array([food_benefits.get(food, 1.0) for food in food_names])

        # ── Linear objective vector c_vec ────────────────────────────────────
        c_vec = np.zeros(n_x)
        # Benefit: B_c * area_f per Y[f,c,t]
        benefit_3d = area_arr[:, None, None] * B_arr[None, :, None] * np.ones((1, 1, n_periods))
        c_vec[Y_idx.ravel()] += benefit_3d.ravel()
        # One-hot linear contribution +penalty per Y  (from expanding -(sum-1)^2)
        c_vec[:n_Y] += one_hot_penalty
        # Diversity bonus per Z[f,c]
        c_vec[Z_idx.ravel()] += diversity_bonus

        # ── Quadratic matrix Q (sparse, symmetric) ───────────────────────────
        # setMObjective convention: max 0.5·x'Qx + c'x
        # For term a·xi·xj (i≠j): Q[i,j]=Q[j,i]=a → contribution = a·xi·xj ✓
        q_r: list = []
        q_c: list = []
        q_v: list = []

        # Pre-filter R to non-zero (c1, c2) pairs
        nz_c1, nz_c2 = np.where(np.abs(R) > 1e-8)
        nz_R = R[nz_c1, nz_c2]                                 # (n_nz,)

        # Temporal synergy: γ_rot · area[f] · R[c1,c2] · Y[f,c1,t] · Y[f,c2,t+1]
        for t in range(n_periods - 1):
            r_t = fa[:, None] * n_foods * n_periods + nz_c1[None, :] * n_periods + t
            c_t = fa[:, None] * n_foods * n_periods + nz_c2[None, :] * n_periods + (t + 1)
            v_t = rotation_gamma * area_arr[:, None] * nz_R[None, :]  # (n_farms, n_nz)
            q_r += [r_t.ravel(), c_t.ravel()]
            q_c += [c_t.ravel(), r_t.ravel()]
            q_v += [v_t.ravel(), v_t.ravel()]

        # Spatial synergy: γ_spat · 0.3/A_tot · R[c1,c2] · Y[f1,c1,t] · Y[f2,c2,t]
        if len(neighbor_edges) > 0:
            ea   = np.array(neighbor_edges, dtype=np.intp)
            f1e, f2e = ea[:, 0], ea[:, 1]
            sp_base  = spatial_gamma * 0.3 / total_area
            for t in range(n_periods):
                r_s = f1e[:, None] * n_foods * n_periods + nz_c1[None, :] * n_periods + t
                c_s = f2e[:, None] * n_foods * n_periods + nz_c2[None, :] * n_periods + t
                v_s = np.broadcast_to(sp_base * nz_R[None, :], r_s.shape).copy()
                q_r += [r_s.ravel(), c_s.ravel()]
                q_c += [c_s.ravel(), r_s.ravel()]
                q_v += [v_s.ravel(), v_s.ravel()]

        # One-hot quadratic penalty: -2·penalty · Y[f,c1,t]·Y[f,c2,t]  (c1 < c2)
        oh_c1, oh_c2 = np.triu_indices(n_foods, k=1)
        for t in range(n_periods):
            r_oh = fa[:, None] * n_foods * n_periods + oh_c1[None, :] * n_periods + t
            c_oh = fa[:, None] * n_foods * n_periods + oh_c2[None, :] * n_periods + t
            v_oh = np.full(r_oh.size, -2.0 * one_hot_penalty)
            q_r += [r_oh.ravel(), c_oh.ravel()]
            q_c += [c_oh.ravel(), r_oh.ravel()]
            q_v += [v_oh, v_oh]

        Q_mat = scipy.sparse.csr_matrix(
            (np.concatenate(q_v), (np.concatenate(q_r), np.concatenate(q_c))),
            shape=(n_x, n_x),
        )
        model.setMObjective(Q_mat, c_vec, 0.0, sense=GRB.MAXIMIZE)

        # ── Constraints (sparse matrix form) ────────────────────────────────
        # Crop-sum selector: row f*n_periods+t → all Y[f,*,t] columns
        fg, tg, cg = np.meshgrid(fa, ta, ca, indexing='ij')  # (n_farms, n_periods, n_foods)
        aft_r = (fg * n_periods + tg).ravel()
        aft_c = Y_idx[fg, cg, tg].ravel()
        n_ft  = n_farms * n_periods
        A_yt  = scipy.sparse.csr_matrix(
            (np.ones(len(aft_c)), (aft_r, aft_c)), shape=(n_ft, n_x)
        )
        model.addMConstr(A_yt, x, '<', 2.0 * np.ones(n_ft))  # max 2 crops/farm/period
        model.addMConstr(A_yt, x, '>', np.ones(n_ft))         # min 1 crop/farm/period

        # Diversity: Z[f,c] ≥ (1/T)·Σ_t Y[f,c,t]  and  Z[f,c] ≤ Σ_t Y[f,c,t]
        fg2, cg2, tg2 = np.meshgrid(fa, ca, ta, indexing='ij')  # (n_farms, n_foods, n_periods)
        dy_r  = (fg2 * n_foods + cg2).ravel()
        dy_c  = Y_idx[fg2, cg2, tg2].ravel()
        dz_r  = (fa[:, None] * n_foods + ca[None, :]).ravel()
        dz_c  = Z_idx.ravel()
        n_fc  = n_farms * n_foods
        ny_d  = len(dy_c)
        A_dlb = scipy.sparse.csr_matrix(
            (np.concatenate([-np.ones(ny_d) / n_periods, np.ones(n_fc)]),
             (np.concatenate([dy_r, dz_r]), np.concatenate([dy_c, dz_c]))),
            shape=(n_fc, n_x),
        )
        model.addMConstr(A_dlb, x, '>', np.zeros(n_fc))  # Z - (1/T)·ΣY ≥ 0
        A_dub = scipy.sparse.csr_matrix(
            (np.concatenate([np.ones(ny_d), -np.ones(n_fc)]),
             (np.concatenate([dy_r, dz_r]), np.concatenate([dy_c, dz_c]))),
            shape=(n_fc, n_x),
        )
        model.addMConstr(A_dub, x, '>', np.zeros(n_fc))  # ΣY - Z ≥ 0

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
            x_vals = x.X                            # numpy array (n_x,)
            Y_vals = x_vals[Y_idx]                  # (n_farms, n_foods, n_periods)
            f_s, c_s, t_s = np.where(Y_vals > 0.5)
            solution = {(int(f), int(c), int(t) + 1): 1
                        for f, c, t in zip(f_s, c_s, t_s)}
            
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
