import json
with open(r'@todo/gurobi_timeout_verification/gurobi_timeout_test_20251224_103144.json') as f:
    d = json.load(f)

for entry in d[:3]:  # First 3 results
    r = entry['result']
    info = entry['benchmark_info']
    print(f"Ref: scenario={r['scenario']}, n_farms={info['n_farms']}, n_foods={info['n_foods']}")
    print(f"     status={r['status']}, obj={r['objective_value']:.4f}, time={r['solve_time']:.2f}s, gap={r['mip_gap']*100:.1f}%")
    print()
