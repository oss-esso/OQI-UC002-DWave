import json
from pathlib import Path

p = Path(r'D:\Projects\OQI-UC002-DWave\@todo\qpu_benchmark_results\roadmap_phase1_20251211_101235.json')
with open(p, 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f'Phase: {d["phase"]}')
print(f'Tests: {len(d["results"])}')
for i, r in enumerate(d['results'][:15]):
    print(f'{i+1}. {r["method"]}: {r["n_farms"]}farms, {r["n_variables"]}vars, solve={r["solve_time"]:.2f}s, obj={r.get("objective", "N/A")}')
