"""Extract Study 2 QPU data for verification."""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).parent
QPU_DIR = HERE / "@todo" / "qpu_benchmark_results"

QPU_FILES = [
    "qpu_benchmark_20251201_160444.json",
    "qpu_benchmark_20251201_142434.json",
    "qpu_benchmark_20251201_200012.json",
    "qpu_benchmark_20251203_121526.json",
    "qpu_benchmark_20251203_110358.json",
    "qpu_benchmark_20251203_111656.json",
]

method_data: dict[tuple[int, str], dict] = {}
gt_data: dict[int, dict] = {}

for fname in QPU_FILES:
    fpath = QPU_DIR / fname
    if not fpath.exists():
        print(f"SKIP: {fname}")
        continue
    raw = json.loads(fpath.read_text())
    results = raw.get("results", {})
    if isinstance(results, list):
        # Flat list of entries
        for entry in results:
            if not isinstance(entry, dict):
                continue
            n = entry.get("n_farms") or entry.get("n_units", 0)
            mkey = entry.get("method", "unknown")
            if mkey == "ground_truth":
                gt_data[n] = entry
            else:
                method_data[(n, mkey)] = entry
    elif isinstance(results, dict):
        for scale_key, scale_results in results.items():
            n = int(scale_key)
            if isinstance(scale_results, dict):
                for mkey, mdata in scale_results.items():
                    if not isinstance(mdata, dict):
                        continue
                    if mkey == "ground_truth":
                        gt_data[n] = mdata
                    else:
                        method_data[(n, mkey)] = mdata

print("=== Ground truth ===")
for n in sorted(gt_data.keys()):
    gt = gt_data[n]
    obj = gt.get("objective")
    t = gt.get("timings", {}).get("total_time", "?")
    print(f"  n={n:5d}  obj={obj}  time={t}s")

full_span = [
    "decomposition_Multilevel(10)_QPU",
    "cqm_first_PlotBased",
    "coordinated",
    "decomposition_HybridGrid(5,9)_QPU",
]

print("\n=== Full-span methods at key scales ===")
for mkey in full_span:
    print(f"\n--- {mkey} ---")
    for n in [10, 15, 25, 50, 100, 200, 500, 1000]:
        e = method_data.get((n, mkey))
        if e:
            obj = e.get("objective")
            viols = e.get("violations", 0)
            t = e.get("timings", {})
            gt_obj = gt_data.get(n, {}).get("objective", 0)
            gap = abs(obj - gt_obj) if obj is not None and gt_obj else "?"
            print(f"  n={n:5d}  obj={obj:.4f}  viols={viols}  gap={gap:.4f}  total={t.get('total_time','?')}s  qpu={t.get('qpu_time','?')}s  embed={t.get('embedding_time','?')}s")
        else:
            print(f"  n={n:5d}  NOT FOUND")

# Also check Study 1 data location
print("\n\n=== Looking for Study 1 data ===")
for p in HERE.rglob("benchmark_results*.json"):
    if "@todo" not in str(p) and "decomposition" not in str(p):
        print(f"  Found: {p}")

# Check the actual Study 1 path used in generate_report_visuals.py
s1_path = HERE / "data" / "study1_patch_benchmark.json"
if s1_path.exists():
    print(f"\n  Study 1 at: {s1_path}")
else:
    print(f"  Not at: {s1_path}")
    # Search likely locations
    for candidate in [
        HERE / "data",
        HERE / "@todo",
        HERE / "Benchmarks",
    ]:
        if candidate.exists():
            for f in candidate.rglob("*.json"):
                if "study1" in f.name.lower() or "patch" in f.name.lower() or "benchmark_results" in f.name.lower():
                    print(f"  Candidate: {f}")
