"""Inspect solver comparison data for Gurobi decomposed violations."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DECOMP_DIR = ROOT / "Benchmarks" / "decomposition_scaling"

for fname in ("solver_comparison_results_hard.json", "solver_comparison_results.json"):
    fpath = DECOMP_DIR / fname
    if not fpath.exists():
        continue
    data = json.loads(fpath.read_text())
    print(f"=== {fname} ===")
    # Show first entry to understand structure
    if isinstance(data, list):
        print(f"Type: list, len={len(data)}")
        r = data[0]
        print(f"Entry keys: {list(r.keys())}")
        print(f"First entry: {json.dumps(r, indent=2)[:2000]}")
    elif isinstance(data, dict):
        print(f"Type: dict, keys={list(data.keys())}")
    break

print("\n\n=== Checking decomposed entries ===")
for fname in ("solver_comparison_results_hard.json", "solver_comparison_results.json"):
    fpath = DECOMP_DIR / fname
    if not fpath.exists():
        continue
    data = json.loads(fpath.read_text())
    if not isinstance(data, list):
        continue
    # Find decomposed Gurobi entries
    for r in data[:5]:
        decomp = r.get("decomposition", "none")
        if decomp != "none":
            print(f"\n  n={r.get('n_farms')} decomp={decomp} variant={r.get('variant')}")
            print(f"  keys: {list(r.keys())}")
            vd = r.get("violation_details")
            v = r.get("violations", 0)
            print(f"  violations={v}, violation_details={json.dumps(vd, indent=2) if vd else None}")
    break
