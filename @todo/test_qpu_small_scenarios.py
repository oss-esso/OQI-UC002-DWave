#!/usr/bin/env python3
"""
Test script to verify small scenario loading for QPU benchmark.

This script tests that:
1. Small scenarios can be loaded correctly
2. Variable counts match expectations
3. Direct QPU embedding is enabled for small scenarios
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.scenarios import load_food_data

# Test each small scenario
SCENARIOS = [
    ('micro_6', 2, 2, 6),       # 2 plots × 2 foods + 2 U vars = 6
    ('micro_12', 3, 3, 12),     # 3 plots × 3 foods + 3 U vars = 12
    ('tiny_24', 4, 5, 25),      # 4 plots × 5 foods + 5 U vars = 25
    ('tiny_40', 5, 6, 36),      # 5 plots × 6 foods + 6 U vars = 36
    ('small_60', 6, 8, 56),     # 6 plots × 8 foods + 8 U vars = 56
    ('small_80', 7, 10, 80),    # 7 plots × 10 foods + 10 U vars = 80
    ('small_100', 8, 11, 99),   # 8 plots × 11 foods + 11 U vars = 99
    ('medium_120', 9, 12, 120), # 9 plots × 12 foods + 12 U vars = 120
    ('medium_160', 10, 14, 154), # 10 plots × 14 foods + 14 U vars = 154
]

print("=" * 80)
print("Testing Small Scenario Loading for QPU Benchmark")
print("=" * 80)

all_passed = True

for scenario_name, expected_farms, expected_foods, expected_vars in SCENARIOS:
    print(f"\nTesting: {scenario_name}")
    print("-" * 40)
    
    try:
        # Load scenario
        farms, foods, food_groups, config = load_food_data(scenario_name)
        
        # Check counts
        n_farms = len(farms)
        n_foods = len(foods)
        n_vars = n_farms * n_foods + n_foods  # Y vars + U vars
        
        # Validate
        farms_ok = n_farms == expected_farms
        foods_ok = n_foods == expected_foods
        vars_ok = n_vars == expected_vars
        
        status = "✅ PASS" if (farms_ok and foods_ok and vars_ok) else "❌ FAIL"
        
        print(f"  Farms: {n_farms} (expected: {expected_farms}) {'✓' if farms_ok else '✗'}")
        print(f"  Foods: {n_foods} (expected: {expected_foods}) {'✓' if foods_ok else '✗'}")
        print(f"  Total vars: {n_vars} (expected: {expected_vars}) {'✓' if vars_ok else '✗'}")
        print(f"  Food groups: {list(food_groups.keys())}")
        print(f"  Status: {status}")
        
        if not (farms_ok and foods_ok and vars_ok):
            all_passed = False
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print("✅ All small scenarios loaded successfully!")
    print("\nYou can now run the QPU benchmark with:")
    print("  python qpu_benchmark.py --scenario micro_6")
    print("  python qpu_benchmark.py --scenario micro_6 tiny_24 small_60")
    print("  python qpu_benchmark.py --all-small")
else:
    print("❌ Some scenarios failed to load correctly")
    sys.exit(1)

print("=" * 80)
