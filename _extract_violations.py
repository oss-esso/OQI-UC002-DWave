"""Extract complete violation data by constraint type for IEEE tables."""
import json

with open("data/comprehensive_benchmark_configs_dwave_20251130_212742.json") as f:
    data = json.load(f)

print("COMPLETE STUDY 1 DATA")
print("=" * 120)
header = f"{'n':>5} | {'Solver':>12} | {'Time(s)':>8} | {'QPU(ms)':>8} | {'Obj':>8} | {'1-hot viol':>10} | {'FG viol':>7} | {'Feasible':>8} | {'CovArea':>8} | {'TotArea':>8}"
print(header)
print("-" * 120)

for entry in data["patch_results"]:
    n = entry["n_units"]
    for sname in ["gurobi", "dwave_cqm", "dwave_bqm", "gurobi_qubo"]:
        s = entry["solvers"].get(sname, {})
        t = s.get("hybrid_time") or s.get("solve_time", 0)
        qpu = s.get("qpu_time", 0) * 1000  # to ms
        obj = s.get("objective_value", 0)
        val = s.get("validation", {})
        cc = val.get("constraint_checks", {})

        # One-hot violations
        onehot = cc.get("at_most_one_per_plot", {}).get("failed", 0)
        # Food group violations
        fg = cc.get("food_group_constraints", {}).get("failed", 0)

        feasible = val.get("is_feasible", s.get("is_feasible", "?"))
        cov_area = s.get("total_covered_area", 0)
        tot_area = s.get("total_area", entry.get("total_area", 0))

        print(f"{n:>5} | {sname:>12} | {t:>8.3f} | {qpu:>8.1f} | {obj:>8.3f} | {onehot:>10} | {fg:>7} | {str(feasible):>8} | {cov_area:>8.1f} | {tot_area:>8.1f}")
    print()

# Now compute inflation from covered_area (how many total "assigned hectares" vs actual hectares)
print("\nINFLATION ANALYSIS (CoveredArea / TotalArea ratio => one-hot violation severity)")
print("=" * 80)
for entry in data["patch_results"]:
    n = entry["n_units"]
    for sname in ["dwave_cqm", "dwave_bqm"]:
        s = entry["solvers"][sname]
        cov = s.get("total_covered_area", 0)
        tot = s.get("total_area", entry.get("total_area", 0))
        ratio = cov / tot if tot > 0 else 0
        obj = s.get("objective_value", 0)
        gobj = entry["solvers"]["gurobi"]["objective_value"]
        val = s.get("validation", {})
        cc = val.get("constraint_checks", {})
        onehot = cc.get("at_most_one_per_plot", {}).get("failed", 0)
        fg = cc.get("food_group_constraints", {}).get("failed", 0)
        print(f"n={n:>4} {sname:>10}: cov/tot={ratio:.2f}  obj={obj:.3f}  gurobi_obj={gobj:.3f}  inflation={obj-gobj:.3f}  1hot={onehot}  fg={fg}")
