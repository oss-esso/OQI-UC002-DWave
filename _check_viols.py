"""Quick check of violation counts in Study 1 and Study 2 data."""
import json
from pathlib import Path

# Study 1
data = json.loads(Path("data/comprehensive_benchmark_configs_dwave_20251130_212742.json").read_text())
entries = sorted(data["patch_results"], key=lambda x: x["n_units"])
print("=== Study 1 violations ===")
for e in entries:
    n = e["n_units"]
    s = e["solvers"]
    cqm_v = (s.get("dwave_cqm") or {}).get("validation", {}).get("n_violations", 0)
    bqm_v = (s.get("dwave_bqm") or {}).get("validation", {}).get("n_violations", 0)
    gq_v = (s.get("gurobi_qubo") or {}).get("validation", {}).get("n_violations", 0)
    print(f"  n={n:5d}  CQM={cqm_v:4d}  BQM={bqm_v:4d}  GQUBO={gq_v:4d}")

# Study 2 QPU
print("\n=== Study 2 QPU violations ===")
qpu_dir = Path("@todo/qpu_benchmark_results")
QPU_FILES = [
    "qpu_benchmark_20251201_160444.json",
    "qpu_benchmark_20251201_142434.json",
    "qpu_benchmark_20251201_200012.json",
    "qpu_benchmark_20251203_121526.json",
    "qpu_benchmark_20251203_110358.json",
    "qpu_benchmark_20251203_111656.json",
]
METHODS = [
    "decomposition_Multilevel(10)_QPU",
    "cqm_first_PlotBased",
    "coordinated",
    "decomposition_HybridGrid(5,9)_QPU",
]
seen = {}
for fname in QPU_FILES:
    fpath = qpu_dir / fname
    if not fpath.exists():
        continue
    d = json.loads(fpath.read_text())
    for r in d.get("results", []):
        n = r["n_farms"]
        for mkey, mdata in r.get("method_results", {}).items():
            if mkey in METHODS:
                v = mdata.get("violations", 0)
                seen[(n, mkey)] = int(v)

for (n, m), v in sorted(seen.items()):
    if v > 0:
        short = m.replace("decomposition_", "").replace("_QPU", "")
        print(f"  n={n:5d}  method={short:30s}  viols={v}")
