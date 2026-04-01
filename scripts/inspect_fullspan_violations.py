"""Inspect violation_details for FULL_SPAN_METHODS specifically."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
QPU_DIR = ROOT / "@todo" / "qpu_benchmark_results"

FULL_SPAN = [
    "decomposition_Multilevel(10)_QPU",
    "cqm_first_PlotBased",
    "coordinated",
    "decomposition_HybridGrid(5,9)_QPU",
]

FULL_SPAN_FILES = [
    "qpu_benchmark_20251201_160444.json",
    "qpu_benchmark_20251201_142434.json",
    "qpu_benchmark_20251201_200012.json",
    "qpu_benchmark_20251203_121526.json",
    "qpu_benchmark_20251203_110358.json",
    "qpu_benchmark_20251203_111656.json",
]

print("=== FULL_SPAN_METHODS violation details ===\n")
for fname in FULL_SPAN_FILES:
    fpath = QPU_DIR / fname
    if not fpath.exists():
        print(f"MISSING: {fname}")
        continue
    d = json.loads(fpath.read_text())
    for r in d.get("results", []):
        n = r["n_farms"]
        for mkey in FULL_SPAN:
            mval = r.get("method_results", {}).get(mkey)
            if mval is None:
                continue
            viols = mval.get("violations", 0) or 0
            vd = mval.get("violation_details")
            obj = mval.get("objective")
            print(f"{fname} | n={n} | {mkey}")
            print(f"  violations={viols}, objective={obj}")
            if viols > 0:
                if vd:
                    print(f"  violation_details={json.dumps(vd, indent=4)}")
                else:
                    print(f"  violation_details=null")
                    # Show partition results sample
                    pr = mval.get("partition_results", [])
                    if pr:
                        print(f"  n_partitions={len(pr)}")
                        for i, p in enumerate(pr[:1]):
                            if isinstance(p, dict):
                                print(f"   partition[0] keys: {list(p.keys())}")
                                # Show first few variables/assignments
                                vars_ = p.get("variables", {})
                                if vars_:
                                    sample = dict(list(vars_.items())[:5])
                                    print(f"   partition[0] vars sample: {sample}")
            print()
