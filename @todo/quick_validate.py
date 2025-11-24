#!/usr/bin/env python3
"""
Quick Solution Validator for Decomposition Results

Compares objectives and checks for constraint violations.
"""

import json
import sys

def main():
    benchmark_file = sys.argv[1] if len(sys.argv) > 1 else "../Benchmarks/ALL_STRATEGIES/all_strategies_config_25_20251124_140925.json"
    pulp_file = "../Benchmarks/COMPREHENSIVE/Farm_PuLP/config_25_run_1.json"
    
    with open(benchmark_file, 'r') as f:
        benchmark = json.load(f)
    
    with open(pulp_file, 'r') as f:
        pulp_ref = json.load(f)
    
    total_area = pulp_ref['total_area']
    pulp_obj_norm = pulp_ref['objective_value']
    
    print(f"\n{'='*100}")
    print(f"DECOMPOSITION STRATEGY VALIDATION")
    print(f"{'='*100}")
    print(f"PuLP Optimal (normalized): {pulp_obj_norm:.6f}")
    print(f"Total Area: {total_area:.2f}")
    print(f"{'='*100}\n")
    
    print(f"{'Strategy':<25} {'Raw Obj':<12} {'Normalized':<12} {'vs PuLP':<10} {'Violations':<12} {'Status':<30}")
    print("─" * 110)
    
    results_sorted = []
    
    for strategy, result in benchmark['results'].items():
        # New standardized format - direct access
        obj_raw = result.get('objective_value', 0.0)
        is_feasible = result.get('success', False)
        
        # Get violations
        violations = 0
        if 'validation' in result and 'n_violations' in result['validation']:
            violations = result['validation']['n_violations']
        
        # Normalize
        obj_norm = obj_raw / total_area if total_area > 0 else 0
        ratio = obj_norm / pulp_obj_norm if pulp_obj_norm > 0 else 0
        ratio_pct = (ratio - 1.0) * 100
        
        # Status
        if obj_raw == 0:
            status = "❌ No solution"
        elif obj_norm > pulp_obj_norm * 1.01:
            status = "⚠️  INFEASIBLE (exceeds optimum)"
        elif violations > 0:
            status = f"⚠️  INFEASIBLE ({violations} violations)"
        elif obj_norm < pulp_obj_norm * 0.95:
            status = "⚠️  SUBOPTIMAL"
        elif abs(obj_norm - pulp_obj_norm) < pulp_obj_norm * 0.01:
            status = "✅ OPTIMAL"
        else:
            status = "✅ Near optimal"
        
        results_sorted.append({
            'strategy': strategy,
            'obj_raw': obj_raw,
            'obj_norm': obj_norm,
            'ratio_pct': ratio_pct,
            'violations': violations,
            'status': status
        })
    
    # Sort by normalized objective (descending)
    results_sorted.sort(key=lambda x: x['obj_norm'], reverse=True)
    
    for r in results_sorted:
        print(f"{r['strategy']:<25} {r['obj_raw']:<12.6f} {r['obj_norm']:<12.6f} {r['ratio_pct']:>9.2f}% {r['violations']:<12} {r['status']:<30}")
    
    print("─" * 110)
    
    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    
    optimal_count = sum(1 for r in results_sorted if '✅ OPTIMAL' in r['status'])
    infeasible_count = sum(1 for r in results_sorted if 'INFEASIBLE' in r['status'])
    suboptimal_count = sum(1 for r in results_sorted if 'SUBOPTIMAL' in r['status'])
    failed_count = sum(1 for r in results_sorted if 'No solution' in r['status'])
    
    print(f"Total strategies: {len(results_sorted)}")
    print(f"Optimal: {optimal_count}")
    print(f"Near optimal: {sum(1 for r in results_sorted if 'Near optimal' in r['status'])}")
    print(f"Suboptimal: {suboptimal_count}")
    print(f"Infeasible: {infeasible_count}")
    print(f"Failed: {failed_count}")
    
    best = max(results_sorted, key=lambda x: x['obj_norm'] if x['obj_norm'] <= pulp_obj_norm * 1.01 else 0)
    print(f"\nBest feasible solution: {best['strategy']} ({best['obj_norm']:.6f}, {best['ratio_pct']:+.2f}% vs PuLP)")
    
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()
