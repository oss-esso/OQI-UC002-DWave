"""Check healed vs raw gap for Study 1."""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from generate_report_visuals import (
    load_study1, load_benefit_matrix, compute_healed_obj_study1,
    _load_solver_comparison, load_gurobi_full_a, safe_float,
)

entries = load_study1()
bm = load_benefit_matrix()
healed = compute_healed_obj_study1(entries, bm)
sc = _load_solver_comparison()
gurobi_a = load_gurobi_full_a(sc)

print("\n=== Study 1: raw vs healed gap ===")
for e in entries:
    n = e["n_units"]
    gt = gurobi_a.get(n, {}).get("objective")
    if gt is None:
        continue
    for sk, label in (("dwave_cqm", "cqm"), ("dwave_bqm", "bqm")):
        raw = safe_float(e["solvers"].get(sk, {}).get("objective_value"))
        heal = healed.get(label, {}).get(n)
        if raw is None:
            continue
        raw_gap = abs(raw - gt)
        heal_gap = abs(heal - gt) if heal is not None else None
        worse = "WORSE" if heal_gap is not None and heal_gap > raw_gap else ""
        h_str = f"{heal:.4f}" if heal is not None else "N/A"
        hg_str = f"{heal_gap:.4f}" if heal_gap is not None else "N/A"
        print(f"  n={n:5d} {label:3s}  GT={gt:.4f}  raw={raw:.4f}  healed={h_str:>8s}  raw_gap={raw_gap:.4f}  heal_gap={hg_str:>8s}  {worse}")
