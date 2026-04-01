"""Inspect QPU benchmark data for violation_details structure."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
QPU_DIR = ROOT / "@todo" / "qpu_benchmark_results"

# First: show all top-level keys across files/methods
print("=== QPU benchmark data structure ===")
for fpath in sorted(QPU_DIR.glob("*.json")):
    d = json.loads(fpath.read_text())
    for r in d.get("results", []):
        n = r["n_farms"]
        for mkey, mval in r.get("method_results", {}).items():
            if not isinstance(mval, dict):
                continue
            viols = mval.get("violations", 0) or 0
            vd = mval.get("violation_details")
            if viols > 0:
                print(f"\n  {fpath.name} | n={n} | {mkey} | violations={viols}")
                print(f"  violation_details = {json.dumps(vd, indent=4)}")
                print(f"  objective = {mval.get('objective')}")
                # Also show assignment/partition info summary
                pr = mval.get("partition_results", [])
                if pr:
                    print(f"  n_partitions = {len(pr)}")
                    for i, p in enumerate(pr[:2]):
                        print(f"   partition[{i}] keys: {list(p.keys()) if isinstance(p, dict) else type(p)}")

print("\n=== Done ===")
