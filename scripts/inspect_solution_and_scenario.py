"""Inspect 'solution' field for Coordinated and check scenario data."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
QPU_DIR = ROOT / "@todo" / "qpu_benchmark_results"

# Check Coordinated solution structure (n=10 for readability)
fname = "qpu_benchmark_20251201_160444.json"
fpath = QPU_DIR / fname
d = json.loads(fpath.read_text())
for r in d.get("results", []):
    n = r["n_farms"]
    if n != 10:
        continue
    mval = r.get("method_results", {}).get("coordinated")
    if mval is None:
        continue
    
    print(f"=== Coordinated n=10 ===")
    print(f"violations={mval['violations']}")
    print(f"objective={mval['objective']}")
    
    sol = mval.get("solution")
    if sol is None:
        print("solution=None")
    elif isinstance(sol, dict):
        print(f"solution type=dict, len={len(sol)}")
        sample = dict(list(sol.items())[:10])
        print(f"solution sample: {sample}")
    elif isinstance(sol, list):
        print(f"solution type=list, len={len(sol)}")
        print(f"solution sample: {sol[:5]}")
    
    # Check ground truth for comparison
    gt = r.get("ground_truth", {})
    print(f"\nGround truth keys: {list(gt.keys())}")
    gt_sol = gt.get("solution")
    if isinstance(gt_sol, dict):
        print(f"GT solution sample: {dict(list(gt_sol.items())[:5])}")
    
    # Check metadata for scenario info
    meta = r.get("metadata", {})
    print(f"\nMetadata keys: {list(meta.keys())}")
    print(f"Metadata: {json.dumps(meta, indent=2)}")
    break

print("\n\n=== Scenario data path check ===")
# Find scenario/data files
for p in [
    ROOT / "data",
    ROOT / "@todo" / "qpu_benchmark_results" / "scenarios",
    ROOT / "rotation_data",
    ROOT / "Inputs",
]:
    if p.exists():
        print(f"  {p} exists, contents: {[x.name for x in p.iterdir()][:10]}")
