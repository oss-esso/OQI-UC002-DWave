"""Inspect Gurobi decomposed healed_objective and violation_counts in detail."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DECOMP_DIR = ROOT / "Benchmarks" / "decomposition_scaling"
fpath = DECOMP_DIR / "solver_comparison_results_hard.json"
data = json.loads(fpath.read_text())

print("=== Gurobi Decomposed: all variants and violations ===")
by_decomp: dict = {}
for r in data:
    if r.get("decomposition", "none") == "none":
        continue
    d = r["decomposition"]
    n = r["n_farms"]
    key = (d, n)
    by_decomp[key] = r

for key in sorted(by_decomp):
    d, n = key
    r = by_decomp[key]
    print(f"\n{d} | n={n} | variant={r.get('variant')}")
    print(f"  objective={r['objective']:.4f}")
    print(f"  healed_objective={r.get('healed_objective')}")
    print(f"  violations={r['violations']}")
    print(f"  violation_counts={r.get('violation_counts')}")

print("\n\n=== All decompositions at n=10 ===")
for r in data:
    if r.get("n_farms") == 10 and r.get("decomposition", "none") != "none":
        print(f"\n  {r['decomposition']} | variant={r.get('variant')}")
        vc = r.get("violation_counts", {})
        print(f"  violations={r['violations']}, violation_counts={json.dumps(vc, indent=4)}")
        print(f"  healed_objective={r.get('healed_objective')}")
