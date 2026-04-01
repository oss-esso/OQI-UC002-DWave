#!/usr/bin/env python3
"""
Targeted QUBO rerun: re-solve Gurobi QUBO for all 8 Study 1 scales
with a higher timeout (1200s) and update the comprehensive results JSON.
"""

import json
import os
import sys
import time
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))  # for solver_runner_BINARY
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from Utils.patch_sampler import generate_grid
from src.scenarios import load_food_data

# Reuse the comprehensive benchmark's config builder
from comprehensive_benchmark import create_food_config, BENCHMARK_CONFIGS

import solver_runner_BINARY as solver_runner
from dimod import cqm_to_bqm


TIMEOUT = 300  # seconds
LAGRANGE = 100000.0


def run_qubo_for_scale(n_units: int, timeout: float = TIMEOUT) -> dict:
    """Build CQM → BQM → run Gurobi QUBO for one scale."""
    land = generate_grid(n_units, area=100.0, seed=42)
    plots_list = list(land.keys())
    foods, food_groups, config = create_food_config(land, "patch")

    # Build CQM
    cqm, Y, _ = solver_runner.create_cqm_plots(plots_list, foods, food_groups, config)

    # Convert to BQM
    bqm, invert = cqm_to_bqm(cqm, lagrange_multiplier=LAGRANGE)
    print(f"  BQM: {len(bqm.variables)} vars, {len(bqm.quadratic)} interactions")

    # Solve
    result = solver_runner.solve_with_gurobi_qubo(
        bqm,
        farms=plots_list,
        foods=foods,
        food_groups=food_groups,
        land_availability=land,
        weights=config["parameters"]["weights"],
        idle_penalty=config["parameters"].get("idle_penalty_lambda", 0.0),
        config=config,
        time_limit=timeout,
    )

    obj = result.get("objective_value")
    status = result.get("status", "?")
    solve_time = result.get("solve_time", 0)
    print(f"  n={n_units:>5}  obj={'%.6f' % obj if obj is not None else 'None':>12}  "
          f"time={solve_time:.2f}s  status={status}")
    return {
        "status": status,
        "objective_value": obj,
        "bqm_energy": result.get("bqm_energy"),
        "solve_time": solve_time,
        "success": status == "Optimal",
        "n_units": n_units,
        "n_variables": len(bqm.variables),
        "bqm_interactions": result.get("bqm_interactions", 0),
    }


def main():
    print(f"Rerunning Gurobi QUBO with timeout={TIMEOUT}s")
    print(f"Scales: {BENCHMARK_CONFIGS}\n")

    new_qubo = {}
    for n in BENCHMARK_CONFIGS:
        print(f"[{n}]")
        new_qubo[n] = run_qubo_for_scale(n, timeout=TIMEOUT)

    # Update the comprehensive JSON
    json_path = PROJECT_ROOT / "data" / "comprehensive_benchmark_configs_dwave_20251130_212742.json"
    with open(json_path) as f:
        data = json.load(f)

    for entry in data["patch_results"]:
        n = entry["n_units"]
        if n in new_qubo:
            qr = new_qubo[n]
            entry["solvers"]["gurobi_qubo"] = {
                "status": qr["status"],
                "objective_value": qr["objective_value"],
                "bqm_energy": qr["bqm_energy"],
                "solve_time": qr["solve_time"],
                "success": qr["success"],
                "n_units": n,
                "n_variables": qr["n_variables"],
                "bqm_interactions": qr["bqm_interactions"],
            }

    # Save updated JSON (overwrite in-place)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nUpdated {json_path.name}")

    # Summary
    print(f"\n{'n':>6}  {'obj':>12}  {'time':>10}  status")
    print("-" * 45)
    for n in BENCHMARK_CONFIGS:
        r = new_qubo[n]
        obj_s = "%.6f" % r["objective_value"] if r["objective_value"] is not None else "None"
        print(f"{n:>6}  {obj_s:>12}  {r['solve_time']:>10.2f}  {r['status']}")


if __name__ == "__main__":
    main()
