#!/usr/bin/env python3
"""
Verification script for rotation formulation fix.
Checks code structure without requiring D-Wave libraries.
"""

import re

print("="*80)
print("ROTATION FORMULATION FIX - CODE VERIFICATION")
print("="*80)

# Read the modified file
with open('qpu_benchmark.py', 'r') as f:
    code = f.read()

checks = []

# Check 1: build_rotation_cqm exists
if 'def build_rotation_cqm(data: Dict, n_periods: int = 3)' in code:
    checks.append(("✓", "build_rotation_cqm() function defined"))
else:
    checks.append(("✗", "build_rotation_cqm() function NOT found"))

# Check 2: solve_ground_truth_rotation exists  
if 'def solve_ground_truth_rotation(data: Dict, timeout: int = 120)' in code:
    checks.append(("✓", "solve_ground_truth_rotation() function defined"))
else:
    checks.append(("✗", "solve_ground_truth_rotation() function NOT found"))

# Check 3: Rotation detection in solve_ground_truth
if "is_rotation = scenario_name.startswith('rotation_')" in code:
    checks.append(("✓", "Rotation scenario detection added to solve_ground_truth()"))
else:
    checks.append(("✗", "Rotation detection NOT found in solve_ground_truth()"))

# Check 4: Rotation detection in CQM building
if 'is_rotation = use_scenarios and scenario_name.startswith' in code:
    checks.append(("✓", "Rotation detection added to CQM building logic"))
else:
    checks.append(("✗", "Rotation detection NOT found in CQM building"))

# Check 5: build_rotation_cqm creates Y[f,c,t] variables
if 'Y[(f, c, t)] = Binary(f"Y_{f}_{c}_t{t}")' in code:
    checks.append(("✓", "3-period Y[f,c,t] variables implemented"))
else:
    checks.append(("✗", "3-period variables NOT found"))

# Check 6: Rotation synergy matrix creation
if 'R = np.zeros((n_families, n_families))' in code and 'frustration_ratio' in code:
    checks.append(("✓", "Rotation synergy matrix R with frustration implemented"))
else:
    checks.append(("✗", "Rotation matrix NOT found"))

# Check 7: Spatial neighbor graph
if 'neighbor_edges' in code and 'k_neighbors' in code:
    checks.append(("✓", "Spatial neighbor graph implemented"))
else:
    checks.append(("✗", "Spatial neighbor graph NOT found"))

# Check 8: Quadratic rotation synergies in objective
if 'Y[(f, c1, t-1)] * Y[(f, c2, t)]' in code:
    checks.append(("✓", "Temporal quadratic coupling Y[t-1]*Y[t] implemented"))
else:
    checks.append(("✗", "Temporal coupling NOT found"))

# Check 9: Spatial quadratic terms
if 'Y[(f1, c1, t)] * Y[(f2, c2, t)]' in code:
    checks.append(("✓", "Spatial quadratic coupling Y[f1]*Y[f2] implemented"))
else:
    checks.append(("✗", "Spatial coupling NOT found"))

# Check 10: Soft one-hot penalty
if 'one_hot_penalty' in code and 'crop_count * crop_count' in code:
    checks.append(("✓", "Soft one-hot quadratic penalty implemented"))
else:
    checks.append(("✗", "Soft one-hot penalty NOT found"))

# Check 11: Diversity bonus
if 'diversity_bonus' in code and 'crop_used' in code:
    checks.append(("✓", "Diversity bonus implemented"))
else:
    checks.append(("✗", "Diversity bonus NOT found"))

# Check 12: Config passed through load_problem_data_from_scenario
if "'config': config_loaded" in code:
    checks.append(("✓", "Config passed through from scenario loader"))
else:
    checks.append(("✗", "Config NOT passed through"))

# Check 13: Gurobi rotation formulation matches benchmark_rotation_gurobi.py
rotation_gurobi_elements = [
    'rotation_gamma',
    'R[c1_idx, c2_idx]',
    'spatial_gamma',
    'one_hot_penalty',
    'diversity_bonus'
]
missing = [e for e in rotation_gurobi_elements if e not in code]
if not missing:
    checks.append(("✓", "Gurobi rotation formulation complete (all elements present)"))
else:
    checks.append(("⚠", f"Rotation formulation may be incomplete (missing: {missing})"))

# Print results
print("\nCode Structure Verification:")
print("-" * 80)
for symbol, message in checks:
    print(f"{symbol} {message}")

# Summary
passed = sum(1 for s, _ in checks if s == "✓")
failed = sum(1 for s, _ in checks if s == "✗")
warnings = sum(1 for s, _ in checks if s == "⚠")

print("\n" + "="*80)
print(f"SUMMARY: {passed} passed, {failed} failed, {warnings} warnings (out of {len(checks)} checks)")
print("="*80)

if failed == 0:
    print("\n✓ All critical checks PASSED!")
    print("  The rotation formulation appears to be correctly implemented.")
    print("  Ready for testing with actual QPU/Gurobi environment.")
else:
    print(f"\n✗ {failed} checks FAILED - implementation incomplete!")
    print("  Review the failed checks above and fix before deployment.")

print()
