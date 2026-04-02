"""Extract objective data for Study 1 (BQM/CQM vs Gurobi) and remaining Study 2 data."""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).parent

# Study 1 data
s1 = HERE / "Benchmarks" / "hybrid_benchmark_results.json"
if s1.exists():
    data = json.loads(s1.read_text())
    print("=== Study 1: Objectives & Gap ===")
    for r in data:
        n = r["n_patches"]
        gt = r.get("gurobi_objective")
        cqm = r.get("cqm_objective")
        bqm = r.get("bqm_objective")
        qubo = r.get("qubo_gurobi_objective")
        cqm_v = r.get("cqm_violations", 0)
        bqm_v = r.get("bqm_violations", 0)
        cqm_healed = r.get("cqm_healed_objective")
        bqm_healed = r.get("bqm_healed_objective")
        gap_cqm = abs(cqm - gt) if cqm and gt else None
        gap_bqm = abs(bqm - gt) if bqm and gt else None
        gap_qubo = abs(qubo - gt) if qubo and gt else None
        print(f"  n={n:5d}  GT={gt:.6f}  CQM={cqm:.6f}(v={cqm_v},healed={cqm_healed})  BQM={bqm:.6f}(v={bqm_v},healed={bqm_healed})  QUBO={qubo:.6f}")
        print(f"          |gap_CQM|={gap_cqm:.6f}  |gap_BQM|={gap_bqm:.6f}  |gap_QUBO|={gap_qubo:.6f}")
else:
    print(f"Study 1 file not found at {s1}")

# Study 2 QPU: objectives and gaps
print("\n\n=== Study 2: QPU objectives and gaps vs GT ===")
QPU_DIR = HERE / "@todo" / "qpu_benchmark_results"
QPU_FILES = [
    ("qpu_benchmark_20251201_160444.json", [10, 15, 50, 100]),
    ("qpu_benchmark_20251201_142434.json", [25]),
    ("qpu_benchmark_20251201_200012.json", [200, 500, 1000]),
    ("qpu_benchmark_20251203_121526.json", [10, 15, 50, 200, 500, 1000]),
    ("qpu_benchmark_20251203_110358.json", [25]),
    ("qpu_benchmark_20251203_111656.json", [100]),
]

FULL_SPAN_METHODS = [
    "decomposition_Multilevel(10)_QPU",
    "cqm_first_PlotBased",
    "coordinated",
    "decomposition_HybridGrid(5,9)_QPU",
]

# Collect best per (method, n)
from collections import defaultdict
best = defaultdict(dict)  # method -> n -> {obj, gt, viols, healed}

for fname, scales in QPU_FILES:
    fpath = QPU_DIR / fname
    if not fpath.exists():
        continue
    raw = json.loads(fpath.read_text())
    for result in raw.get("results", []):
        n = result["n_farms"]
        if n not in scales:
            continue
        gt_obj = result.get("ground_truth", {}).get("objective")
        for mkey in FULL_SPAN_METHODS:
            mdata = result.get("method_results", {}).get(mkey)
            if not mdata:
                continue
            obj = mdata.get("objective")
            viols = mdata.get("violations", 0)
            healed = mdata.get("healed_objective")
            if n not in best[mkey] or (obj is not None and obj > (best[mkey][n].get("obj") or 0)):
                best[mkey][n] = {"obj": obj, "gt": gt_obj, "viols": viols, "healed": healed}

for mkey in FULL_SPAN_METHODS:
    print(f"\nMethod: {mkey}")
    for n in sorted(best[mkey]):
        d = best[mkey][n]
        gap = abs(d["obj"] - d["gt"]) if d["obj"] and d["gt"] else None
        print(f"  n={n:5d}  GT={d['gt']:.6f}  obj={d['obj']:.6f}  |gap|={gap:.6f}  viols={d['viols']}  healed={d['healed']}")

# Violation details for Study 2
print("\n\n=== Study 2: Violation details for HybridGrid ===")
for fname, scales in QPU_FILES:
    fpath = QPU_DIR / fname
    if not fpath.exists():
        continue
    raw = json.loads(fpath.read_text())
    for result in raw.get("results", []):
        n = result["n_farms"]
        mdata = result.get("method_results", {}).get("decomposition_HybridGrid(5,9)_QPU")
        if mdata and mdata.get("violations", 0) > 0:
            print(f"  n={n}  viols={mdata['violations']}  details={mdata.get('violation_details', {})}")
