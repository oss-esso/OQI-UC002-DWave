"""Extract Study 1 data for IEEE paper tables."""
import json

with open("data/comprehensive_benchmark_configs_dwave_20251130_212742.json") as f:
    data = json.load(f)

print("=" * 80)
print("STUDY 1 DATA EXTRACTION")
print("=" * 80)

for entry in data["patch_results"]:
    n = entry["n_units"]
    nv = entry["n_variables"]
    solvers = entry["solvers"]
    print(f"\n--- n={n} (vars={nv}) ---")
    for sname, s in solvers.items():
        t = s.get("solve_time", 0)
        obj = s.get("objective_value", None)
        qpu = s.get("qpu_access_time", None)
        status = s.get("status", "?")
        nv_count = s.get("n_violations", "?")
        feasible = s.get("is_feasible", "?")
        print(f"  {sname}: time={t:.4f}s  obj={obj}  qpu={qpu}  status={status}  nv={nv_count}  feasible={feasible}")
        # Check for healed objective
        healed = s.get("healed_objective", None)
        if healed is not None:
            print(f"    healed_obj={healed}  inflation={obj - healed if obj else '?'}")
