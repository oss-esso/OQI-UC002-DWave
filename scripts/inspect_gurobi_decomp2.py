"""Inspect Gurobi decomposed healed_objective and violation structure."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DECOMP_DIR = ROOT / "Benchmarks" / "decomposition_scaling"
fpath = DECOMP_DIR / "solver_comparison_results_hard.json"
data = json.loads(fpath.read_text())

print("=== All decomposed entries at n=10 ===")
for r in data:
    if r.get("n_farms") == 10 and r.get("decomposition", "none") != "none":
        print(f"\n  decomp={r['decomposition']} | variant={r.get('variant')}")
        print(f"  all keys: {list(r.keys())}")
        obj_ = r.get('objective')
        obj_str = f"{obj_:.4f}" if obj_ is not None else "None"
        print(f"  objective={obj_str}")
        print(f"  healed_objective={r.get('healed_objective')}")
        print(f"  violations={r.get('violations')}")
        vc = r.get("violation_counts")
        print(f"  violation_counts={json.dumps(vc, indent=4) if vc else None}")

print("\n\n=== All Variant-A decomposed entries ===")
for r in data:
    if r.get("variant") == "A" and r.get("decomposition", "none") != "none":
        viols = r.get("violations")
        ho = r.get("healed_objective")
        obj = r.get("objective")
        n = r["n_farms"]
        d = r["decomposition"]
        print(f"  n={n:4d} | {d:20s} | obj={obj:.4f} | healed={ho} | viols={viols}")
