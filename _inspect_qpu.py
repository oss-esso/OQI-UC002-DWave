import json
with open("qpu_hier_repaired.json") as f:
    d = json.load(f)
runs = d.get("runs", d) if isinstance(d, dict) else d
for r in sorted(runs, key=lambda x: (x.get("n_foods", 0), x.get("n_farms", 0))):
    timing = r.get("timing", {})
    viols = r.get("constraint_violations", {})
    wt = timing.get("total_wall_time", 0)
    tv = viols.get("total_violations", 0)
    print("  n_farms=%3d  n_foods=%3d  n_vars=%6d  obj_miqp=%10.4f  wall=%8.2fs  viols=%d  status=%s" % (
        r.get("n_farms", 0), r.get("n_foods", 0), r.get("n_vars", 0),
        r.get("objective_miqp", 0) or 0, wt, tv, r.get("status", "?")))
