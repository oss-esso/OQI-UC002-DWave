"""Extract timing data from Study 2 QPU files + Gurobi decomposed data."""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).parent
QPU_DIR = HERE / "@todo" / "qpu_benchmark_results"
DECOMP_DIR = HERE / "Benchmarks" / "decomposition_scaling"

def safe_float(val):
    if val is None: return None
    try: return float(val)
    except: return None

# Check what keys a method result actually has
fname = "qpu_benchmark_20251203_121526.json"
fpath = QPU_DIR / fname
if fpath.exists():
    raw = json.loads(fpath.read_text())
    print("=== Inspecting QPU result structure ===")
    for result in raw.get("results", [])[:2]:
        n = result["n_farms"]
        print(f"\nn={n}")
        if "ground_truth" in result:
            gt = result["ground_truth"]
            print(f"  GT keys: {list(gt.keys())}")
            if "timings" in gt:
                print(f"    timings: {gt['timings']}")
        for mkey, mdata in list(result.get("method_results", {}).items())[:3]:
            print(f"\n  Method: {mkey}")
            print(f"    Top-level keys: {list(mdata.keys())}")
            if "timings" in mdata:
                print(f"    timings: {mdata['timings']}")
            if "solution" in mdata:
                sol = mdata["solution"]
                print(f"    solution keys: {list(sol.keys()) if isinstance(sol, dict) else type(sol)}")
            # Check for time in top-level
            for k in ["wall_time", "total_time", "solve_time", "time"]:
                if k in mdata:
                    print(f"    {k}: {mdata[k]}")

# ── Gurobi decomposed ─────────────────────────────────────────────────────────
for fname in ["solver_comparison_results_hard.json"]:
    fpath = DECOMP_DIR / fname
    if fpath.exists():
        data = json.loads(fpath.read_text())
        print(f"\n\n=== Gurobi Decomposed: {fname} ===")
        # Group by decomposition
        from collections import defaultdict
        decomp_data = defaultdict(dict)
        gurobi_full = {}
        for r in data:
            if r.get("variant") != "A":
                continue
            n = r["n_farms"]
            d = r.get("decomposition", "none")
            obj = safe_float(r.get("objective"))
            wt = safe_float(r.get("wall_time"))
            viols = int(r.get("violations", 0))
            healed = safe_float(r.get("healed_objective"))
            if d == "none":
                gurobi_full[n] = {"obj": obj, "wt": wt}
            else:
                decomp_data[d][n] = {"obj": obj, "wt": wt, "viols": viols, "healed": healed}

        print("\nGurobi Full:")
        for n in sorted(gurobi_full):
            d = gurobi_full[n]
            print(f"  n={n:5d}  obj={d['obj']:.6f}  time={d['wt']:.4f}s")

        for dname in sorted(decomp_data):
            print(f"\nDecomp: {dname}")
            for n in sorted(decomp_data[dname]):
                d = decomp_data[dname][n]
                print(f"  n={n:5d}  obj={d['obj']}  time={d['wt']:.4f}s  viols={d['viols']}  healed={d['healed']}")

# ── Generated tables ──────────────────────────────────────────────────────────
tpath = HERE / "Benchmarks" / "decomposition_scaling" / "tables" / "report_tables.tex"
if tpath.exists():
    print(f"\n\n=== Generated LaTeX Tables ===")
    print(tpath.read_text()[:5000])
else:
    print(f"\nTables not found at {tpath}")
