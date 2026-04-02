"""Extract plot data for IEEE caption writing."""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
QPU_DIR = HERE / "@todo" / "qpu_benchmark_results"

def safe_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

# ── Study 1 ────────────────────────────────────────────────────────────────
print("=" * 80)
print("STUDY 1: Hybrid Solver Benchmark")
print("=" * 80)

path = DATA_DIR / "comprehensive_benchmark_configs_dwave_20251130_212742.json"
data = json.loads(path.read_text())
entries = sorted(data["patch_results"], key=lambda e: e["n_units"])

# Also load solver comparison for Gurobi canonical times
decomp_dir = HERE / "Benchmarks" / "decomposition_scaling"
sc_data = json.loads((decomp_dir / "solver_comparison_results_hard.json").read_text())
gurobi_full = {}
for r in sc_data:
    if r.get("variant") == "A" and r.get("decomposition", "none") == "none":
        gurobi_full[r["n_farms"]] = r

print("\nGurobi (from solver_comparison Variant-A):")
for n in sorted(gurobi_full):
    r = gurobi_full[n]
    print(f"  n={n}: wall_time={r.get('wall_time'):.4f}s, obj={r.get('objective'):.6f}")

print("\nD-Wave CQM (Hybrid):")
for e in entries:
    cqm = e["solvers"].get("dwave_cqm", {})
    n = e["n_units"]
    ht = safe_float(cqm.get("hybrid_time"))
    obj = safe_float(cqm.get("objective_value"))
    viols = (cqm.get("validation") or {}).get("n_violations", 0)
    feas = (cqm.get("validation") or {}).get("is_feasible")
    print(f"  n={n}: hybrid_time={ht}, obj={obj}, viols={viols}, feasible={feas}")

print("\nD-Wave BQM (Hybrid):")
for e in entries:
    bqm = e["solvers"].get("dwave_bqm", {})
    n = e["n_units"]
    ht = safe_float(bqm.get("hybrid_time") or bqm.get("solve_time"))
    obj = safe_float(bqm.get("objective_value"))
    viols = (bqm.get("validation") or {}).get("n_violations", 0)
    feas = (bqm.get("validation") or {}).get("is_feasible")
    print(f"  n={n}: hybrid_time={ht}, obj={obj}, viols={viols}, feasible={feas}")

print("\nGurobi QUBO:")
for e in entries:
    gq = e["solvers"].get("gurobi_qubo", {})
    n = e["n_units"]
    ok = gq.get("success", False) and gq.get("status", "").lower() != "error"
    if ok:
        print(f"  n={n}: time={gq.get('solve_time')}, obj={gq.get('objective_value')}, status={gq.get('status')}")
    else:
        print(f"  n={n}: FAILED status={gq.get('status')}, success={gq.get('success')}")

# ── Study 1: Absolute gaps ──
print("\nStudy 1 Absolute Gaps |solver - Gurobi|:")
for e in entries:
    n = e["n_units"]
    gt_obj = gurobi_full.get(n, {}).get("objective")
    if gt_obj is None:
        continue
    cqm_obj = safe_float(e["solvers"].get("dwave_cqm", {}).get("objective_value"))
    bqm_obj = safe_float(e["solvers"].get("dwave_bqm", {}).get("objective_value"))
    gq = e["solvers"].get("gurobi_qubo", {})
    gq_ok = gq.get("success", False) and gq.get("status", "").lower() != "error"
    gq_obj = safe_float(gq.get("objective_value")) if gq_ok else None
    cqm_gap = f"{abs(cqm_obj - gt_obj):.6f}" if cqm_obj else "N/A"
    bqm_gap = f"{abs(bqm_obj - gt_obj):.6f}" if bqm_obj else "N/A"
    qubo_gap = f"{abs(gq_obj - gt_obj):.6f}" if gq_obj else "N/A"
    print(f"  n={n}: GT={gt_obj:.6f}, CQM_gap={cqm_gap}, BQM_gap={bqm_gap}, QUBO_gap={qubo_gap}")

# ── Study 2 ────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STUDY 2: QPU Decomposition Benchmark")
print("=" * 80)

QPU_FILES = [
    ("qpu_benchmark_20251201_160444.json", [10, 15, 50, 100]),
    ("qpu_benchmark_20251201_142434.json", [25]),
    ("qpu_benchmark_20251201_200012.json", [200, 500, 1000]),
    ("qpu_benchmark_20251203_121526.json", [10, 15, 50, 200, 500, 1000]),
    ("qpu_benchmark_20251203_110358.json", [25]),
    ("qpu_benchmark_20251203_111656.json", [100]),
]

ALL_METHODS = [
    "decomposition_PlotBased_QPU", "decomposition_Multilevel(5)_QPU",
    "decomposition_Multilevel(10)_QPU", "decomposition_Louvain_QPU",
    "decomposition_Spectral(10)_QPU", "cqm_first_PlotBased",
    "coordinated", "decomposition_HybridGrid(5,9)_QPU",
]
FULL_SPAN = [
    "decomposition_Multilevel(10)_QPU", "cqm_first_PlotBased",
    "coordinated", "decomposition_HybridGrid(5,9)_QPU",
]
METHOD_DISPLAY = {
    "decomposition_PlotBased_QPU": "PlotBased",
    "decomposition_Multilevel(5)_QPU": "Multilevel(5)",
    "decomposition_Multilevel(10)_QPU": "Multilevel(10)",
    "decomposition_Louvain_QPU": "Louvain",
    "decomposition_Spectral(10)_QPU": "Spectral(10)",
    "cqm_first_PlotBased": "CQM-First",
    "coordinated": "Coordinated",
    "decomposition_HybridGrid(5,9)_QPU": "HybridGrid(5,9)",
}

method_data = {}
qpu_gt = {}
for fname, _ in QPU_FILES:
    fpath = QPU_DIR / fname
    if not fpath.exists():
        continue
    d = json.loads(fpath.read_text())
    for result in d.get("results", []):
        n = result["n_farms"]
        if n not in qpu_gt and "ground_truth" in result:
            qpu_gt[n] = result["ground_truth"]
        for mkey, mdata in result.get("method_results", {}).items():
            if mkey in ALL_METHODS:
                method_data[(n, mkey)] = mdata

print("\nGround truth (from QPU benchmark):")
for n in sorted(qpu_gt):
    gt = qpu_gt[n]
    print(f"  n={n}: obj={gt.get('objective')}, time={gt.get('solve_time') or gt.get('wall_time')}")

print("\nFull-span QPU methods (Comprehensive plot data):")
for mkey in FULL_SPAN:
    disp = METHOD_DISPLAY[mkey]
    print(f"\n  {disp}:")
    for n in sorted({nn for (nn, mk) in method_data if mk == mkey}):
        entry = method_data.get((n, mkey))
        if entry is None:
            continue
        obj = safe_float(entry.get("objective"))
        wt = safe_float(entry.get("wall_time"))
        viols = entry.get("violations", 0)
        gt_obj = safe_float(qpu_gt.get(n, {}).get("objective"))
        gap = abs(obj - gt_obj) if (obj is not None and gt_obj is not None) else None
        print(f"    n={n}: obj={obj}, wall_time={wt:.1f}s, viols={viols}, gap={gap}")

# Gurobi decomposed
print("\nGurobi Decomposed (Variant-A):")
gurobi_decomp = {}
for r in sc_data:
    if r.get("variant") == "A" and r.get("decomposition", "none") != "none":
        dn = r["decomposition"]
        n = r["n_farms"]
        gurobi_decomp.setdefault(dn, {})[n] = r

for dn in sorted(gurobi_decomp):
    print(f"\n  {dn}:")
    for n in sorted(gurobi_decomp[dn]):
        r = gurobi_decomp[dn][n]
        obj = r.get("objective")
        wt = r.get("wall_time")
        healed = r.get("healed_objective")
        viols = r.get("violations", 0)
        print(f"    n={n}: obj={obj}, wall_time={wt}, healed={healed}, viols={viols}")
