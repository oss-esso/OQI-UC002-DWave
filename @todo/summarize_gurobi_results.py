"""
Quick script to summarize Gurobi scaling benchmark results
"""
import json
import sys

# Read the latest results file
filepath = "D:/Projects/OQI-UC002-DWave/@todo/gurobi_scaling_benchmark_20260120_185821.json"

with open(filepath) as f:
    data = json.load(f)

results = data.get('results', [])

print("\n" + "="*100)
print("GUROBI SCALING BENCHMARK RESULTS (Binary Patch Formulation)")
print("="*100)
print(f"Timestamp: {data['timestamp']}")
print(f"Settings: Timeout={data['config']['timeout_no_proof']}s (no proof), {data['config']['timeout_proof']}s (proof)")
print(f"          MIPGap={data['config']['mip_gap_no_proof']} (no proof), {data['config']['mip_gap_proof']} (proof)")
print("="*100)
print(f"{'Variables':>10} | {'Patches':>7} | {'Mode':>15} | {'Status':>12} | {'Solve (s)':>10} | {'Build (s)':>10} | {'Objective':>12}")
print("-"*100)

for r in results:
    mode = r['mode'].replace('_', ' ').title()
    obj_str = f"{r['objective_value']:.6f}" if r['objective_value'] is not None else "N/A"
    print(f"{r['n_variables']:>10,} | {r['n_patches']:>7,} | {mode:>15} | {r['status']:>12} | {r['solve_time']:>10.2f} | {r['model_build_time']:>10.2f} | {obj_str:>12}")

print("="*100)

# Summary statistics
optimal_results = [r for r in results if r['status'] == 'optimal']
if optimal_results:
    print(f"\nSuccessful solves: {len(optimal_results)}/{len(results)}")
    max_vars = max(r['n_variables'] for r in optimal_results)
    print(f"Largest problem solved optimally: {max_vars:,} variables")
    
    # Compare with/without proof for same problem size
    sizes = set(r['n_variables'] for r in optimal_results)
    print(f"\n{'Variables':>10} | {'Without Proof (s)':>18} | {'With Proof (s)':>15} | {'Speedup':>8}")
    print("-"*60)
    for size in sorted(sizes):
        no_proof = [r for r in optimal_results if r['n_variables'] == size and r['mode'] == 'without_proof']
        with_proof = [r for r in optimal_results if r['n_variables'] == size and r['mode'] == 'with_proof']
        if no_proof and with_proof:
            speedup = with_proof[0]['solve_time'] / no_proof[0]['solve_time']
            print(f"{size:>10,} | {no_proof[0]['solve_time']:>18.3f} | {with_proof[0]['solve_time']:>15.3f} | {speedup:>8.2f}x")
