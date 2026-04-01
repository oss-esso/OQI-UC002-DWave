import json
from pathlib import Path
HERE = Path(__file__).parent.parent
f = HERE / "data/comprehensive_benchmark_configs_dwave_20251130_212742.json"
data = json.loads(f.read_text())
entries = sorted(data["patch_results"], key=lambda e: e["n_units"])
# Show solution_plantations for n=10 bqm (has 9 violations)
e = entries[0]
bqm = e["solvers"]["dwave_bqm"]
sp = bqm.get("solution_plantations")
print(f"n={e['n_units']} solution_plantations type={type(sp)}")
if isinstance(sp, dict):
    for k, v in list(sp.items())[:4]:
        print(f"  {k!r}: {v}")
elif isinstance(sp, list):
    for item in sp[:4]:
        print(f"  {item}")
print(f"total_area={bqm.get('total_area')} n_units={bqm.get('n_units')}")
