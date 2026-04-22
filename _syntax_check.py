import json
with open("Benchmarks/decomposition_scaling/solver_comparison_results_hard.json") as f:
    data = json.load(f)

b_full = [r for r in data if r.get("variant") == "B" and r.get("decomposition") == "none"]
print("--- Variant B Gurobi_full rows ---")
for r in sorted(b_full, key=lambda x: x["n_farms"]):
    print(f"  n={r['n_farms']:5d}  obj={str(r.get('objective','—'))[:12]:12s}"
          f"  healed={str(r.get('healed_objective','—'))[:12]:12s}"
          f"  violations={str(r.get('violations','—')):5s}"
          f"  one_hot_viols={str(r.get('one_hot_violations','—')):5s}"
          f"  v%={str(r.get('violation_rate_pct','—'))[:6]:6s}"
          f"  feasible={r.get('feasible','—')}"
          f"  status={r.get('status','?')}")
