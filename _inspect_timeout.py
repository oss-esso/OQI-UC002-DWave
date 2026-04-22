import json
from pathlib import Path

d = Path("@todo/gurobi_timeout_verification")
files = sorted(d.glob("gurobi_timeout_test_*.json"))
latest = files[-1]
print("Using:", latest.name)
with open(latest) as f:
    data = json.load(f)
runs = data if isinstance(data, list) else data.get("runs", [])
print("Entries:", len(runs))
for r in sorted(runs, key=lambda x: (x.get("metadata", {}).get("n_foods", 0),
                                      x.get("metadata", {}).get("n_farms", 0))):
    meta = r.get("metadata", {})
    res = r.get("result", {})
    nf = meta.get("n_farms", 0)
    nfoods = meta.get("n_foods", 0)
    nv = res.get("n_vars", 0)
    obj = res.get("objective_value")
    mg = res.get("mip_gap")
    t = res.get("solve_time", 0)
    print("  n_farms=%3d  n_foods=%3d  n_vars=%6d  obj=%10.4f  mip_gap=%8.2f%%  time=%8.2fs" % (
        nf, nfoods, nv, obj or 0, mg or 0, t))
