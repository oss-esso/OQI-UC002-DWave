import json
from pathlib import Path

HERE = Path(__file__).parent.parent

# Check solver_comparison decomposed violation details
data = json.loads((HERE / "Benchmarks/decomposition_scaling/solver_comparison_results_hard.json").read_text())
rows = [r for r in data if r.get("variant") == "A" and r.get("decomposition", "none") != "none" and int(r.get("violations", 0)) > 0]
print(f"Decomposed rows with violations: {len(rows)}")
for r in rows[:4]:
    print(f"  n={r['n_farms']} decomp={r['decomposition']} viols={r['violations']} keys={list(r.keys())}")
    vd = r.get("violation_details") or {}
    print(f"  violation_details keys: {list(vd.keys())}")
    hobj = r.get("healed_objective")
    print(f"  healed_objective: {hobj}")
    sol = r.get("solution") or {}
    print(f"  solution keys: {list(sol.keys())[:6]}")

print()

# Check Study 1 solver data for violations
study1_dir = HERE / "Benchmarks" / "comprehensive_benchmark"
cands = list(study1_dir.glob("comprehensive_benchmark_configs_dwave_*.json"))
if cands:
    s1 = json.loads(cands[-1].read_text())
    entries = s1.get("results") or s1 if isinstance(s1, list) else []
    if not entries and isinstance(s1, dict):
        entries = s1.get("entries") or []
    print(f"Study 1 file: {cands[-1].name}, entries type: {type(s1)}")
    if isinstance(s1, dict):
        print(f"  top-level keys: {list(s1.keys())[:8]}")
    # Try to find one entry with CQM violation
    for e in (entries if isinstance(entries, list) else []):
        s = e.get("solvers") or {}
        cqm = s.get("dwave_cqm") or {}
        cqm_val = cqm.get("validation") or {}
        if int(cqm_val.get("total_violations", 0)) > 0:
            print(f"  CQM violation entry n={e.get('n_units')} keys={list(cqm.keys())}")
            print(f"    cqm objective_value={cqm.get('objective_value')}")
            print(f"    cqm healed_objective={cqm.get('healed_objective')}")
            print(f"    validation: {cqm_val}")
            break
