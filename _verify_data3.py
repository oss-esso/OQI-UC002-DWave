"""Extract Study 1 + Study 2 data for IEEE_v2.tex verification."""
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

# ── Study 1 ──────────────────────────────────────────────────────────────────
s1_path = DATA_DIR / "comprehensive_benchmark_configs_dwave_20251130_212742.json"
if s1_path.exists():
    raw = json.loads(s1_path.read_text())
    entries = sorted(raw["patch_results"], key=lambda e: e["n_units"])
    print("=== STUDY 1: Hybrid Solver Benchmark ===")
    for e in entries:
        n = e["n_units"]
        s = e["solvers"]
        g = s.get("gurobi", {})
        cqm = s.get("dwave_cqm", {})
        bqm = s.get("dwave_bqm", {})
        gq = s.get("gurobi_qubo", {})
        print(f"\nn={n}")
        print(f"  Gurobi:    time={g.get('solve_time','?')}s  obj={g.get('objective_value','?')}")
        cqm_t = safe_float(cqm.get('hybrid_time')) or safe_float(cqm.get('solve_time'))
        cqm_obj = cqm.get('objective_value')
        cqm_viols = (cqm.get('validation') or {}).get('n_violations', 0)
        cqm_qpu = safe_float(cqm.get('qpu_access_time'))
        print(f"  CQM:       time={cqm_t}s  obj={cqm_obj}  viols={cqm_viols}  qpu_ms={cqm_qpu}")
        bqm_t = safe_float(bqm.get('hybrid_time')) or safe_float(bqm.get('solve_time'))
        bqm_obj = bqm.get('objective_value')
        bqm_viols = (bqm.get('validation') or {}).get('n_violations', 0)
        print(f"  BQM:       time={bqm_t}s  obj={bqm_obj}  viols={bqm_viols}")
        gq_t = safe_float(gq.get('solve_time'))
        gq_obj = gq.get('objective_value')
        print(f"  GurobiQ:   time={gq_t}s  obj={gq_obj}  status={gq.get('status','?')}")
else:
    print(f"Study 1 not found: {s1_path}")

# ── Study 2 ──────────────────────────────────────────────────────────────────
QPU_FILES = [
    "qpu_benchmark_20251201_160444.json",
    "qpu_benchmark_20251201_142434.json",
    "qpu_benchmark_20251201_200012.json",
    "qpu_benchmark_20251203_121526.json",
    "qpu_benchmark_20251203_110358.json",
    "qpu_benchmark_20251203_111656.json",
]

method_data = {}
gt_data = {}

for fname in QPU_FILES:
    fpath = QPU_DIR / fname
    if not fpath.exists():
        continue
    raw = json.loads(fpath.read_text())
    for result in raw.get("results", []):
        n = result["n_farms"]
        if "ground_truth" in result:
            gt_data[n] = result["ground_truth"]
        for mkey, mdata in result.get("method_results", {}).items():
            method_data[(n, mkey)] = mdata

print("\n\n=== STUDY 2: Ground Truth ===")
for n in sorted(gt_data):
    gt = gt_data[n]
    obj = gt.get("objective")
    t = (gt.get("timings") or {}).get("total_time", "?")
    print(f"  n={n:5d}  obj={obj}  time={t}s")

full_span = [
    "decomposition_Multilevel(10)_QPU",
    "cqm_first_PlotBased",
    "coordinated",
    "decomposition_HybridGrid(5,9)_QPU",
]

print("\n=== Study 2: Full-span QPU methods ===")
for mkey in full_span:
    print(f"\n--- {mkey} ---")
    for n in [10, 15, 25, 50, 100, 200, 500, 1000]:
        e = method_data.get((n, mkey))
        if e:
            obj = safe_float(e.get("objective"))
            viols = int((e.get("validation") or {}).get("n_violations", 0))
            t = e.get("timings") or {}
            gt_obj = safe_float(gt_data.get(n, {}).get("objective"))
            gap = abs(obj - gt_obj) if obj is not None and gt_obj is not None else "?"
            qpu_pct = 0
            tot = safe_float(t.get("total_time"))
            qpu_t = safe_float(t.get("qpu_time"))
            embed = safe_float(t.get("embedding_time"))
            if tot and qpu_t:
                qpu_pct = 100.0 * qpu_t / tot
            if tot and embed:
                emb_pct = 100.0 * embed / tot
            else:
                emb_pct = 0
            print(f"  n={n:5d}  obj={obj}  viols={viols}  gap={gap}  total={tot}s  qpu={qpu_t}s({qpu_pct:.1f}%)  embed={embed}s({emb_pct:.1f}%)")
        else:
            print(f"  n={n:5d}  NOT FOUND")

# Also print small-scale methods
small_scale = [
    "decomposition_PlotBased_QPU",
    "decomposition_Multilevel(5)_QPU",
    "decomposition_Louvain_QPU",
    "decomposition_Spectral(10)_QPU",
]
print("\n=== Study 2: Small-scale methods ===")
for mkey in small_scale:
    print(f"\n--- {mkey} ---")
    for n in [10, 15, 25, 50, 100]:
        e = method_data.get((n, mkey))
        if e:
            obj = safe_float(e.get("objective"))
            viols = int((e.get("validation") or {}).get("n_violations", 0))
            t = e.get("timings") or {}
            gt_obj = safe_float(gt_data.get(n, {}).get("objective"))
            gap = abs(obj - gt_obj) if obj is not None and gt_obj is not None else "?"
            tot = safe_float(t.get("total_time"))
            print(f"  n={n:5d}  obj={obj}  viols={viols}  gap={gap}  total={tot}s")
        else:
            print(f"  n={n:5d}  NOT FOUND")
