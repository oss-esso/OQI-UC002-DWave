#!/usr/bin/env python3
"""
Ultra-fast SA validation - minimal config for quick verification.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("ULTRA-FAST SA VALIDATION")
print("="*80)

from data_loader_utils import load_food_data_as_dict
from hierarchical_quantum_solver import solve_hierarchical

# Minimal test scenarios
SCENARIOS = [
    {'name': '10 farms, 6 foods', 'n_farms': 10, 'n_foods': 6},
    {'name': '15 farms, 27 foods', 'n_farms': 15, 'n_foods': 27},
]

results = []

for scenario in SCENARIOS:
    print(f"\nTesting: {scenario['name']}...")
    
    # Load data
    if scenario['n_foods'] == 6:
        data = load_food_data_as_dict('rotation_medium_100')
    else:
        data = load_food_data_as_dict('rotation_250farms_27foods')
    
    data['farm_names'] = data['farm_names'][:scenario['n_farms']]
    data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
    data['total_area'] = sum(data['land_availability'].values())
    
    # Ultra-minimal config
    config = {
        'farms_per_cluster': 5,
        'num_reads': 10,  # Very few
        'num_iterations': 1,  # Single pass
    }
    
    result = solve_hierarchical(
        data=data,
        config=config,
        use_qpu=False,
        verbose=False
    )
    
    obj = result.get('objective', 0)
    viol = result.get('violations', -1)
    has_sol = 'solution' in result
    
    status = 'PASS' if (obj > 0 and has_sol) else 'FAIL'
    results.append({'name': scenario['name'], 'obj': obj, 'viol': viol, 'status': status})
    
    print(f"  Objective: {obj:.4f}, Violations: {viol}, Status: {status}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
all_pass = all(r['status'] == 'PASS' for r in results)
for r in results:
    emoji = "✓" if r['status'] == 'PASS' else "✗"
    print(f"  {emoji} {r['name']}: obj={r['obj']:.4f}, viol={r['viol']}")

if all_pass:
    print("\n✓ ALL TESTS PASSED!")
else:
    print("\n✗ SOME TESTS FAILED")
print("="*80)
