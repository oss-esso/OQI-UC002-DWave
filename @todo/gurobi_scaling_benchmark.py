"""
Gurobi Scaling Benchmark: Binary Patch Scenario

Tests Gurobi performance on the binary patch formulation with:
1. Optimality proof (MIPGap=0, no time limit)
2. Without optimality proof (MIPGap=1%, time limit)

Scales from small problems to ~1 million variables using log sweep
(2 points per decade).

The binary formulation:
- Variables: n_patches * n_foods + n_foods (Y + U variables)
- Y[p,c] = 1 if crop c is assigned to patch p
- U[c] = 1 if crop c is used on any patch

Uses gurobipy directly for performance (with fallback to PuLP+CBC for testing).
"""

import os
import sys
import json
import math
import time
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# Try gurobipy first (much faster than PuLP for large models)
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
    print("Using Gurobi (gurobipy) - high performance mode")
except ImportError:
    HAS_GUROBI = False
    import pulp as pl
    print("Gurobi not available, falling back to PuLP (CBC solver)")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

N_FOODS = 27  # Fixed number of foods (from full_family scenario)

# Log sweep: 2 points per decade from 100 to 1,000,000 variables
# Variables = n_patches * n_foods + n_foods = n_patches * 27 + 27
# n_patches = (variables - 27) / 27
# For 1M variables: n_patches = (1,000,000 - 27) / 27 ≈ 37,036 patches

# Points: 100, 316, 1000, 3162, 10000, 31623, 100000, 316228, 1000000
LOG_SWEEP_VARIABLES = [
    100,      # ~3 patches
    316,      # ~11 patches
    1_000,    # ~36 patches
    3_162,    # ~116 patches
    10_000,   # ~370 patches
    31_623,   # ~1,170 patches
    100_000,  # ~3,700 patches
    316_228,  # ~11,700 patches
    1_000_000 # ~37,000 patches
]

# Gurobi settings (matching comprehensive_benchmark.py)
GUROBI_TIMEOUT_PROOF = 900       # 10 minutes for optimality proof
GUROBI_TIMEOUT_NO_PROOF = 300    # 100 seconds for quick solve (matches comprehensive)
MIP_GAP_PROOF = 0.0              # Prove optimality
MIP_GAP_NO_PROOF = 0.01          # 1% gap acceptable


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    n_patches: int
    n_foods: int
    n_variables: int
    n_constraints: int
    mode: str  # 'with_proof' or 'without_proof'
    
    # Timing
    model_build_time: float
    solve_time: float
    total_time: float
    
    # Solution quality
    status: str  # 'optimal', 'feasible', 'timeout', 'infeasible', 'error'
    objective_value: Optional[float]
    mip_gap: Optional[float]
    bound: Optional[float]
    
    # Gurobi stats
    node_count: Optional[int]
    iteration_count: Optional[int]
    
    # Metadata
    timestamp: str
    error_message: Optional[str] = None


def generate_synthetic_food_data(n_foods: int, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Generate synthetic food data for benchmarking.
    
    Returns dict mapping food name to properties:
    - nutritional_value
    - nutrient_density
    - environmental_impact
    - affordability
    - sustainability
    """
    np.random.seed(seed)
    
    foods = {}
    for i in range(n_foods):
        foods[f"crop_{i:03d}"] = {
            'nutritional_value': np.random.uniform(0.3, 1.0),
            'nutrient_density': np.random.uniform(0.2, 0.9),
            'environmental_impact': np.random.uniform(0.1, 0.8),
            'affordability': np.random.uniform(0.3, 1.0),
            'sustainability': np.random.uniform(0.2, 0.9),
        }
    return foods


def generate_patches(n_patches: int, total_area: float = 100.0) -> Dict[str, float]:
    """
    Generate even-grid patches (all equal area).
    
    Returns dict mapping patch name to area (hectares).
    """
    patch_area = total_area / n_patches
    return {f"patch_{i:05d}": patch_area for i in range(n_patches)}


def variables_to_patches(target_vars: int, n_foods: int = N_FOODS) -> int:
    """
    Convert target variable count to number of patches.
    
    Variables = n_patches * n_foods + n_foods
    n_patches = (variables - n_foods) / n_foods
    """
    n_patches = (target_vars - n_foods) / n_foods
    return max(1, int(round(n_patches)))


def build_and_solve_gurobi(
    n_patches: int,
    n_foods: int,
    with_optimality_proof: bool,
    seed: int = 42,
    verbose: bool = True
) -> BenchmarkResult:
    """
    Build and solve the binary patch formulation with Gurobi.
    
    Args:
        n_patches: Number of land patches
        n_foods: Number of crops
        with_optimality_proof: If True, use MIPGap=0 and longer timeout
        seed: Random seed for data generation
        verbose: Print progress
    
    Returns:
        BenchmarkResult with timing and solution info
    """
    mode = "with_proof" if with_optimality_proof else "without_proof"
    timeout = GUROBI_TIMEOUT_PROOF if with_optimality_proof else GUROBI_TIMEOUT_NO_PROOF
    mip_gap = MIP_GAP_PROOF if with_optimality_proof else MIP_GAP_NO_PROOF
    
    timestamp = datetime.now().isoformat()
    total_start = time.time()
    
    # Generate synthetic data
    logger.info(f"Generating data: {n_patches} patches × {n_foods} foods")
    foods = generate_synthetic_food_data(n_foods, seed)
    food_names = list(foods.keys())
    
    total_area = 100.0  # Fixed total area
    patch_area = total_area / n_patches  # Even grid
    
    # Precompute benefits per food (vectorized)
    weights = {
        'nutritional_value': 0.25,
        'nutrient_density': 0.20,
        'environmental_impact': 0.15,
        'affordability': 0.20,
        'sustainability': 0.20,
    }
    
    benefits = np.zeros(n_foods)
    for c_idx, c in enumerate(food_names):
        food = foods[c]
        benefits[c_idx] = (
            weights['nutritional_value'] * food['nutritional_value'] +
            weights['nutrient_density'] * food['nutrient_density'] -
            weights['environmental_impact'] * food['environmental_impact'] +
            weights['affordability'] * food['affordability'] +
            weights['sustainability'] * food['sustainability']
        )
    
    # Normalize by total area (patch_area / total_area = 1/n_patches)
    normalized_benefits = benefits * (patch_area / total_area)
    
    # Build model
    logger.info(f"Building model (mode={mode})")
    build_start = time.time()
    
    if HAS_GUROBI:
        return _solve_with_gurobipy(
            n_patches, n_foods, food_names, normalized_benefits,
            with_optimality_proof, timeout, mip_gap, mode, timestamp,
            build_start, total_start, verbose
        )
    else:
        return _solve_with_pulp_cbc(
            n_patches, n_foods, food_names, normalized_benefits,
            with_optimality_proof, timeout, mip_gap, mode, timestamp,
            build_start, total_start, verbose
        )


def _solve_with_gurobipy(
    n_patches: int,
    n_foods: int,
    food_names: List[str],
    normalized_benefits: np.ndarray,
    with_optimality_proof: bool,
    timeout: float,
    mip_gap: float,
    mode: str,
    timestamp: str,
    build_start: float,
    total_start: float,
    verbose: bool
) -> BenchmarkResult:
    """Solve using gurobipy directly (high performance)."""
    try:
        model = gp.Model("Binary_Patch_Benchmark")
        model.setParam("OutputFlag", 1 if verbose else 0)
        model.setParam("TimeLimit", timeout)
        model.setParam("MIPGap", mip_gap)
        model.setParam("Threads", 0)  # Use all cores
        model.setParam("Presolve", 2)  # Aggressive presolve
        
        if with_optimality_proof:
            model.setParam("MIPFocus", 2)  # Focus on proving optimality
        else:
            model.setParam("MIPFocus", 1)  # Focus on finding good solutions
            model.setParam("Heuristics", 0.1)
        
        # Create variables efficiently using addVars
        Y = model.addVars(n_patches, n_foods, vtype=GRB.BINARY, name="Y")
        U = model.addVars(n_foods, vtype=GRB.BINARY, name="U")
        
        # Objective: maximize area-weighted benefit (vectorized)
        obj = gp.quicksum(
            normalized_benefits[c_idx] * Y[p_idx, c_idx]
            for p_idx in range(n_patches)
            for c_idx in range(n_foods)
        )
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # Constraint 1: At most one crop per patch
        for p_idx in range(n_patches):
            model.addConstr(
                gp.quicksum(Y[p_idx, c_idx] for c_idx in range(n_foods)) <= 1,
                name=f"MaxAssign_{p_idx}"
            )
        
        # Constraint 2: U-Y linking
        for c_idx in range(n_foods):
            # U[c] >= Y[p,c] for all p (if any Y=1, U must be 1)
            for p_idx in range(n_patches):
                model.addConstr(Y[p_idx, c_idx] <= U[c_idx], name=f"ULink_{p_idx}_{c_idx}")
            # U[c] <= sum(Y[p,c]) (if U=1, at least one Y must be 1)
            model.addConstr(
                U[c_idx] <= gp.quicksum(Y[p_idx, c_idx] for p_idx in range(n_patches)),
                name=f"UBound_{c_idx}"
            )
        
        # Constraint 3: Minimum 5 unique foods
        model.addConstr(
            gp.quicksum(U[c_idx] for c_idx in range(n_foods)) >= 5,
            name="MinDiversity"
        )
        
        n_variables = n_patches * n_foods + n_foods
        n_constraints = n_patches + n_patches * n_foods + n_foods + 1
        
        build_time = time.time() - build_start
        logger.info(f"Model built: {n_variables:,} vars, {n_constraints:,} constraints, {build_time:.2f}s")
        
        # Solve
        logger.info(f"Solving (timeout={timeout}s, MIPGap={mip_gap*100:.1f}%)")
        solve_start = time.time()
        model.optimize()
        solve_time = time.time() - solve_start
        
        # Extract results
        status_map = {
            GRB.OPTIMAL: "optimal",
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "infeasible",
            GRB.UNBOUNDED: "unbounded",
            GRB.TIME_LIMIT: "timeout",
            GRB.ITERATION_LIMIT: "limit",
            GRB.NODE_LIMIT: "limit",
        }
        status = status_map.get(model.Status, "unknown")
        
        objective_value = None
        mip_gap_actual = None
        bound = None
        node_count = None
        iteration_count = None
        
        if model.SolCount > 0:
            objective_value = model.ObjVal
            mip_gap_actual = model.MIPGap * 100  # As percentage
            bound = model.ObjBound
            node_count = int(model.NodeCount)
            iteration_count = int(model.IterCount)
        
        total_time = time.time() - total_start
        
        obj_str = f"{objective_value:.4f}" if objective_value is not None else "N/A"
        gap_str = f"{mip_gap_actual:.2f}%" if mip_gap_actual is not None else "N/A"
        logger.info(f"Solved: status={status}, obj={obj_str}, gap={gap_str}, time={solve_time:.2f}s")
        
        return BenchmarkResult(
            n_patches=n_patches,
            n_foods=n_foods,
            n_variables=n_variables,
            n_constraints=n_constraints,
            mode=mode,
            model_build_time=build_time,
            solve_time=solve_time,
            total_time=total_time,
            status=status,
            objective_value=objective_value,
            mip_gap=mip_gap_actual,
            bound=bound,
            node_count=node_count,
            iteration_count=iteration_count,
            timestamp=timestamp,
        )
        
    except Exception as e:
        import traceback
        total_time = time.time() - total_start
        error_detail = traceback.format_exc()
        logger.error(f"Gurobi error: {e}")
        logger.error(error_detail)
        return BenchmarkResult(
            n_patches=n_patches,
            n_foods=n_foods,
            n_variables=n_patches * n_foods + n_foods,
            n_constraints=0,
            mode=mode,
            model_build_time=0,
            solve_time=0,
            total_time=total_time,
            status="error",
            objective_value=None,
            mip_gap=None,
            bound=None,
            node_count=None,
            iteration_count=None,
            timestamp=timestamp,
            error_message=str(e),
        )


def _solve_with_pulp_cbc(
    n_patches: int,
    n_foods: int,
    food_names: List[str],
    normalized_benefits: np.ndarray,
    with_optimality_proof: bool,
    timeout: float,
    mip_gap: float,
    mode: str,
    timestamp: str,
    build_start: float,
    total_start: float,
    verbose: bool
) -> BenchmarkResult:
    """Solve using PuLP with CBC solver (fallback)."""
    try:
        model = pl.LpProblem("Binary_Patch_Benchmark", pl.LpMaximize)
        
        # Create variables using index tuples (more efficient)
        Y = pl.LpVariable.dicts("Y", 
            ((p, c) for p in range(n_patches) for c in range(n_foods)),
            cat='Binary')
        U = pl.LpVariable.dicts("U", range(n_foods), cat='Binary')
        
        # Objective
        objective = pl.lpSum(
            normalized_benefits[c_idx] * Y[(p_idx, c_idx)]
            for p_idx in range(n_patches)
            for c_idx in range(n_foods)
        )
        model += objective, "Objective"
        
        # Constraint 1: At most one crop per patch
        for p_idx in range(n_patches):
            model += pl.lpSum(Y[(p_idx, c_idx)] for c_idx in range(n_foods)) <= 1
        
        # Constraint 2: U-Y linking
        for c_idx in range(n_foods):
            for p_idx in range(n_patches):
                model += Y[(p_idx, c_idx)] <= U[c_idx]
            model += U[c_idx] <= pl.lpSum(Y[(p_idx, c_idx)] for p_idx in range(n_patches))
        
        # Constraint 3: Minimum diversity
        model += pl.lpSum(U[c_idx] for c_idx in range(n_foods)) >= 5
        
        n_variables = n_patches * n_foods + n_foods
        n_constraints = n_patches + n_patches * n_foods + n_foods + 1
        
        build_time = time.time() - build_start
        logger.info(f"Model built: {n_variables:,} vars, {n_constraints:,} constraints, {build_time:.2f}s")
        
        # Solve with CBC
        logger.info(f"Solving with CBC (timeout={timeout}s, gap={mip_gap*100:.1f}%)")
        solve_start = time.time()
        
        solver = pl.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=timeout, gapRel=mip_gap)
        model.solve(solver)
        solve_time = time.time() - solve_start
        
        status_map = {
            pl.LpStatusOptimal: "optimal",
            pl.LpStatusNotSolved: "not_solved",
            pl.LpStatusInfeasible: "infeasible",
            pl.LpStatusUnbounded: "unbounded",
            pl.LpStatusUndefined: "undefined",
        }
        status = status_map.get(model.status, "unknown")
        
        if status == "optimal" and solve_time >= timeout - 1:
            status = "timeout"
        
        objective_value = pl.value(model.objective) if model.status == pl.LpStatusOptimal else None
        
        total_time = time.time() - total_start
        
        logger.info(f"Solved: status={status}, obj={objective_value}, time={solve_time:.2f}s")
        
        return BenchmarkResult(
            n_patches=n_patches,
            n_foods=n_foods,
            n_variables=n_variables,
            n_constraints=n_constraints,
            mode=mode,
            model_build_time=build_time,
            solve_time=solve_time,
            total_time=total_time,
            status=status,
            objective_value=objective_value,
            mip_gap=None,
            bound=None,
            node_count=None,
            iteration_count=None,
            timestamp=timestamp,
        )
        
    except Exception as e:
        import traceback
        total_time = time.time() - total_start
        error_detail = traceback.format_exc()
        logger.error(f"PuLP error: {e}")
        logger.error(error_detail)
        return BenchmarkResult(
            n_patches=n_patches,
            n_foods=n_foods,
            n_variables=n_patches * n_foods + n_foods,
            n_constraints=0,
            mode=mode,
            model_build_time=0,
            solve_time=0,
            total_time=total_time,
            status="error",
            objective_value=None,
            mip_gap=None,
            bound=None,
            node_count=None,
            iteration_count=None,
            timestamp=timestamp,
            error_message=str(e),
        )


def run_scaling_benchmark(
    target_variables: List[int] = LOG_SWEEP_VARIABLES,
    seed: int = 42,
    output_dir: str = None,
    continue_until_limit: bool = False,
    max_variables: int = 1_000_000,
) -> List[BenchmarkResult]:
    """
    Run the full scaling benchmark.
    
    Tests each scale point with and without optimality proof.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    results = []
    
    logger.info("=" * 70)
    logger.info("GUROBI SCALING BENCHMARK: Binary Patch Formulation")
    logger.info(f"Target scales: {len(target_variables)} points")
    logger.info(f"Range: {min(target_variables):,} to {max(target_variables):,} variables")
    logger.info("=" * 70)
    
    for i, target_vars in enumerate(target_variables):
        n_patches = variables_to_patches(target_vars, N_FOODS)
        actual_vars = n_patches * N_FOODS + N_FOODS
        
        logger.info("")
        logger.info(f"[{i+1}/{len(target_variables)}] Scale: {actual_vars:,} variables ({n_patches} patches)")
        logger.info("-" * 50)
        
        # Test 1: Without optimality proof (faster)
        logger.info("Mode: WITHOUT optimality proof")
        result_no_proof = build_and_solve_gurobi(
            n_patches=n_patches,
            n_foods=N_FOODS,
            with_optimality_proof=False,
            seed=seed,
            verbose=True,
        )
        results.append(result_no_proof)
        
        # Test 2: With optimality proof (slower)
        # Skip for very large problems to save time
        if actual_vars <= 100_000_000:
            logger.info("Mode: WITH optimality proof")
            result_proof = build_and_solve_gurobi(
                n_patches=n_patches,
                n_foods=N_FOODS,
                with_optimality_proof=True,
                seed=seed,
                verbose=True,
            )
            results.append(result_proof)
        else:
            logger.info("Skipping optimality proof mode for large problem")
        
        # Save intermediate results
        save_results(results, output_dir)

    # Optionally continue increasing problem size (doubling) until a solve limit is hit
    if continue_until_limit:
        logger.info("Continuing to increase variables until a solver limit is hit...")

        # Start from the last target size
        current_vars = target_variables[-1]
        # Define statuses that indicate a solver limit/problem
        limit_statuses = {"timeout", "error", "unknown"}

        while current_vars < max_variables:
            next_vars = min(current_vars * 2, max_variables)
            if next_vars == current_vars:
                break

            n_patches = variables_to_patches(next_vars, N_FOODS)
            actual_vars = n_patches * N_FOODS + N_FOODS

            logger.info("")
            logger.info(f"[continued] Scale: {actual_vars:,} variables ({n_patches} patches)")
            logger.info("-" * 50)

            # Without proof
            logger.info("Mode: WITHOUT optimality proof")
            res_no = build_and_solve_gurobi(
                n_patches=n_patches,
                n_foods=N_FOODS,
                with_optimality_proof=False,
                seed=seed,
                verbose=True,
            )
            results.append(res_no)
            save_results(results, output_dir)

            # With proof
            logger.info("Mode: WITH optimality proof")
            try:
                res_with = build_and_solve_gurobi(
                    n_patches=n_patches,
                    n_foods=N_FOODS,
                    with_optimality_proof=True,
                    seed=seed,
                    verbose=True,
                )
                results.append(res_with)
                save_results(results, output_dir)
            except Exception as e:
                logger.error(f"Error running with-proof at size {actual_vars}: {e}")
                results.append(BenchmarkResult(
                    n_patches=n_patches,
                    n_foods=N_FOODS,
                    n_variables=actual_vars,
                    n_constraints=0,
                    mode="with_proof",
                    model_build_time=0,
                    solve_time=0,
                    total_time=0,
                    status="error",
                    objective_value=None,
                    mip_gap=None,
                    bound=None,
                    node_count=None,
                    iteration_count=None,
                    timestamp=datetime.now().isoformat(),
                    error_message=str(e),
                ))
                save_results(results, output_dir)

            # If either run hit a solver limit, stop
            if res_no.status in limit_statuses:
                logger.info(f"Stopping: WITHOUT mode hit limit (status={res_no.status}) at {actual_vars} variables")
                break
            if 'res_with' in locals() and res_with.status in limit_statuses:
                logger.info(f"Stopping: WITH mode hit limit (status={res_with.status}) at {actual_vars} variables")
                break

            current_vars = next_vars
    
    return results


def save_results(results: List[BenchmarkResult], output_dir: str):
    """Save results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gurobi_scaling_benchmark_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    data = {
        "benchmark": "gurobi_scaling_binary_patch",
        "timestamp": datetime.now().isoformat(),
        "n_foods": N_FOODS,
        "config": {
            "timeout_proof": GUROBI_TIMEOUT_PROOF,
            "timeout_no_proof": GUROBI_TIMEOUT_NO_PROOF,
            "mip_gap_proof": MIP_GAP_PROOF,
            "mip_gap_no_proof": MIP_GAP_NO_PROOF,
        },
        "results": [asdict(r) for r in results],
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to: {filepath}")


def print_summary(results: List[BenchmarkResult]):
    """Print summary table."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Variables':>12} | {'Patches':>8} | {'Mode':>15} | {'Status':>10} | {'Solve (s)':>10} | {'Objective':>12}")
    print("-" * 100)
    
    for r in results:
        obj_str = f"{r.objective_value:.4f}" if r.objective_value else "N/A"
        print(f"{r.n_variables:>12,} | {r.n_patches:>8,} | {r.mode:>15} | {r.status:>10} | {r.solve_time:>10.2f} | {obj_str:>12}")
    
    print("=" * 100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Gurobi scaling benchmark")
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--until-limit', action='store_true', help='Continue doubling variables until solver limit is hit')
    parser.add_argument('--max-vars', type=int, default=1_000_000, help='Maximum number of variables when using --until-limit')
    parser.add_argument('--no-gurobi', action='store_true', help='Force fallback to PuLP/CBC even if gurobipy is available')

    args = parser.parse_args()

    if args.no_gurobi:
        HAS_GUROBI = False

    results = run_scaling_benchmark(
        target_variables=LOG_SWEEP_VARIABLES,
        seed=args.seed,
        output_dir=args.output_dir,
        continue_until_limit=args.until_limit,
        max_variables=args.max_vars,
    )

    print_summary(results)
