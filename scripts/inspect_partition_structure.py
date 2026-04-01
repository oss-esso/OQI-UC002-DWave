"""Deep inspect partition_results for Coordinated violating cases."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
QPU_DIR = ROOT / "@todo" / "qpu_benchmark_results"

# file with 23 coordinated violations at n=1000
target_files = [
    ("qpu_benchmark_20251201_200012.json", 1000, "coordinated"),
    ("qpu_benchmark_20251201_160444.json", 10, "coordinated"),
]

for fname, target_n, target_m in target_files:
    fpath = QPU_DIR / fname
    if not fpath.exists():
        print(f"MISSING: {fname}")
        continue
    d = json.loads(fpath.read_text())
    for r in d.get("results", []):
        n = r["n_farms"]
        if n != target_n:
            continue
        mval = r.get("method_results", {}).get(target_m)
        if mval is None or not mval.get("violations"):
            continue
        print(f"\n{fname} | n={n} | {target_m} | violations={mval['violations']}")
        print(f"  objective={mval['objective']}")
        print(f"  Result keys: {list(mval.keys())}")
        
        # Check assignments dict if available
        ass = mval.get("assignments")
        if ass:
            print(f"  assignments sample: {dict(list(ass.items())[:5])}")
        
        # Check partition_results
        pr = mval.get("partition_results", [])
        print(f"  n_partitions={len(pr)}")
        if pr:
            p0 = pr[0]
            if isinstance(p0, dict):
                print(f"  partition[0] keys: {list(p0.keys())}")
                # variables
                vars_ = p0.get("variables")
                if isinstance(vars_, dict):
                    print(f"  partition[0].variables type=dict, sample: {dict(list(vars_.items())[:5])}")
                elif isinstance(vars_, list):
                    print(f"  partition[0].variables type=list, sample: {vars_[:5]}")
                # partition
                part = p0.get("partition")
                if isinstance(part, list):
                    print(f"  partition[0].partition type=list, sample: {part[:5]}")
                elif isinstance(part, dict):
                    print(f"  partition[0].partition type=dict, sample: {dict(list(part.items())[:5])}")
                # best_sample
                bs = p0.get("best_sample")
                if bs:
                    print(f"  partition[0].best_sample sample: {dict(list(bs.items())[:5]) if isinstance(bs, dict) else bs[:5]}")
                # result
                res = p0.get("result")
                if res:
                    print(f"  partition[0].result keys: {list(res.keys()) if isinstance(res, dict) else type(res)}")
        break
