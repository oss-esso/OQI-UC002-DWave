"""Extract key data points from benchmark files for IEEE_v2.tex verification."""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).parent

# Study 1 data
s1_path = HERE / "data" / "benchmark_results.json"
if s1_path.exists():
    s1 = json.loads(s1_path.read_text())
    print("=== STUDY 1: Hybrid Solver Benchmark ===")
    for e in s1:
        n = e["n_units"]
        g = e["solvers"].get("gurobi", {})
        cqm = e["solvers"].get("dwave_cqm", {})
        bqm = e["solvers"].get("dwave_bqm", {})
        gq = e["solvers"].get("gurobi_qubo", {})
        print(f"\nn={n}")
        print(f"  Gurobi:  time={g.get('solve_time'):.4f}s  obj={g.get('objective_value')}")
        print(f"  CQM:     time={cqm.get('hybrid_time','?')}s  obj={cqm.get('objective_value','?')}")
        bqm_t = bqm.get('hybrid_time') or bqm.get('solve_time')
        print(f"  BQM:     time={bqm_t}s  obj={bqm.get('objective_value','?')}")
        gq_t = gq.get('solve_time', '?')
        gq_obj = gq.get('objective_value', '?')
        print(f"  GurobiQ: time={gq_t}s  obj={gq_obj}  status={gq.get('status','?')}")
else:
    print(f"Study 1 file not found: {s1_path}")

# Study 2 QPU data
qpu_dir = HERE / "@todo" / "qpu_benchmark_results"
if qpu_dir.exists():
    print("\n\n=== STUDY 2: QPU Benchmark Files ===")
    for f in sorted(qpu_dir.glob("*.json")):
        data = json.loads(f.read_text())
        if isinstance(data, list):
            scales = sorted(set(e.get("n_farms") or e.get("n_units", 0) for e in data))
            print(f"  {f.name}: {len(data)} entries, scales={scales}")

# Solver comparison (Study 2 Gurobi decomposed)
sc_path = HERE / "Benchmarks" / "decomposition_scaling" / "solver_comparison_results_hard.json"
if sc_path.exists():
    sc = json.loads(sc_path.read_text())
    print("\n\n=== Gurobi Decomposed (Variant A) ===")
    for r in sc:
        if r.get("variant") == "A" and r.get("decomposition", "none") != "none":
            print(f"  n={r['n_farms']:5d}  decomp={r['decomposition']:20s}  obj={r.get('objective','-'):>10}  viols={r.get('violations',0)}  healed={r.get('healed_objective','-')}")

# Study 2 QPU: key metrics for comprehensive figure
print("\n\n=== Study 2: Full-span QPU methods at n=1000 ===")
from collections import defaultdict

qpu_files = [
    "qpu_benchmark_20251201_160444.json",
    "qpu_benchmark_20251201_142434.json",
    "qpu_benchmark_20251201_200012.json",
    "qpu_benchmark_20251203_121526.json",
    "qpu_benchmark_20251203_110358.json",
    "qpu_benchmark_20251203_111656.json",
]
method_data = {}
for fname in qpu_files:
    fpath = qpu_dir / fname
    if not fpath.exists():
        continue
    entries = json.loads(fpath.read_text())
    for e in entries:
        n = e.get("n_farms") or e.get("n_units", 0)
        mkey = e.get("method", "unknown")
        method_data[(n, mkey)] = e

full_span = [
    "decomposition_Multilevel(10)_QPU",
    "cqm_first_PlotBased",
    "coordinated",
    "decomposition_HybridGrid(5,9)_QPU",
]
for mkey in full_span:
    entry = method_data.get((1000, mkey))
    if entry:
        obj = entry.get("objective")
        viols = entry.get("violations", 0)
        timings = entry.get("timings", {})
        total = timings.get("total_time", "?")
        qpu_t = timings.get("qpu_time", "?")
        embed = timings.get("embedding_time", "?")
        print(f"  {mkey:40s}  obj={obj}  viols={viols}  total={total}s  qpu={qpu_t}s  embed={embed}s")
    else:
        print(f"  {mkey:40s}  NOT FOUND at n=1000")

# Ground truth
gt = method_data.get((1000, "ground_truth"))
if gt:
    print(f"\n  Ground truth n=1000: obj={gt.get('objective')}  time={gt.get('timings',{}).get('total_time','?')}s")

# Print HybridGrid at all scales
print("\n=== HybridGrid(5,9) across all scales ===")
for n in [10, 15, 25, 50, 100, 200, 500, 1000]:
    e = method_data.get((n, "decomposition_HybridGrid(5,9)_QPU"))
    if e:
        obj = e.get("objective")
        viols = e.get("violations", 0)
        t = e.get("timings", {})
        print(f"  n={n:5d}  obj={obj}  viols={viols}  total={t.get('total_time','?')}s  qpu={t.get('qpu_time','?')}s  embed={t.get('embedding_time','?')}s")
