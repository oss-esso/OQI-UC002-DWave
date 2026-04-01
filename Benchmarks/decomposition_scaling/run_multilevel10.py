#!/usr/bin/env python3
"""
Targeted runner: benchmark Gurobi_decomposed with Multilevel(10) on Variant A
across all FARM_SIZES and merge results into solver_comparison_results_hard.json
(and the mode-agnostic fallback).

Usage:
    conda run -n oqi_project python Benchmarks/decomposition_scaling/run_multilevel10.py --timeout 200
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

# Re-use everything from the comparison benchmark
from benchmark_solvers_comparison import (
    FARM_SIZES,
    FOOD_NAMES_27,
    _partition_multilevel_A,
    solve_gurobi_decomposed_A,
)
from Utils.patch_sampler import generate_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
LOG = logging.getLogger(__name__)


def _ser(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def run(timeout: float = 200.0):
    food_names = FOOD_NAMES_27[:27]
    partitioner = lambda fn, cn: _partition_multilevel_A(fn, cn, 10)
    new_results = []

    for n_farms in FARM_SIZES:
        land = generate_grid(n_farms, area=100.0, seed=42)
        farm_names = list(land.keys())
        n_vars_a = n_farms * len(food_names) + len(food_names)

        LOG.info(f"n_farms={n_farms}  [A] Gurobi decomposed (Multilevel(10))")
        r = solve_gurobi_decomposed_A(farm_names, food_names, land, partitioner, timeout=timeout)
        r.update(
            solver="Gurobi_decomposed",
            variant="A",
            decomposition="Multilevel(10)",
            n_farms=n_farms,
            n_vars=n_vars_a,
        )
        new_results.append(r)
        LOG.info(
            f"  -> obj={r.get('objective'):.4f}  time={r.get('wall_time'):.3f}s"
            if r.get("objective") is not None
            else f"  -> NONE  time={r.get('wall_time'):.3f}s"
        )

    return new_results


def merge_and_save(new_results, dest_paths):
    KEY = ("solver", "variant", "decomposition", "n_farms")

    for dest in dest_paths:
        if dest.exists():
            with open(dest) as f:
                existing = json.load(f)
        else:
            existing = []

        # Remove any old Multilevel(10) entries so we don't duplicate
        existing = [
            r for r in existing
            if not (r.get("variant") == "A" and r.get("decomposition") == "Multilevel(10)")
        ]
        merged = existing + new_results

        with open(dest, "w") as f:
            json.dump(merged, f, indent=2, default=_ser)
        LOG.info(f"Merged {len(new_results)} new entries into {dest}")


if __name__ == "__main__":
    import argparse

    _p = argparse.ArgumentParser()
    _p.add_argument("--timeout", type=float, default=200.0)
    _args = _p.parse_args()

    out_dir = Path(__file__).parent
    results = run(timeout=_args.timeout)

    dest_files = [
        out_dir / "solver_comparison_results_hard.json",
        out_dir / "solver_comparison_results.json",
    ]
    merge_and_save(results, dest_files)

    print("\nDone. Results for Multilevel(10):")
    print(f"{'n_farms':>8}  {'obj':>12}  {'time(s)':>10}  {'status'}")
    print("-" * 50)
    for r in results:
        obj_str = f"{r['objective']:.6f}" if r.get("objective") is not None else "TIMEOUT"
        print(f"{r['n_farms']:>8}  {obj_str:>12}  {r['wall_time']:>10.3f}  {r.get('status', '?')}")
