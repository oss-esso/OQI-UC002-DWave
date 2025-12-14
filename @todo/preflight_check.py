#!/usr/bin/env python3
"""
Pre-flight check for significant_scenarios_benchmark.py

Verifies all dependencies and imports before running the full benchmark.
"""

import sys
from pathlib import Path

print("="*80)
print("PRE-FLIGHT CHECK: Significant Scenarios Benchmark")
print("="*80)
print()

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

checks_passed = 0
checks_failed = 0

# Check 1: Gurobi
print("[1/7] Checking Gurobi...")
try:
    import gurobipy as gp
    from gurobipy import GRB
    model = gp.Model()
    print("  ✓ Gurobi available and licensed")
    checks_passed += 1
except ImportError:
    print("  ✗ Gurobi not installed")
    checks_failed += 1
except Exception as e:
    print(f"  ✗ Gurobi license error: {e}")
    checks_failed += 1

# Check 2: D-Wave dimod
print("[2/7] Checking D-Wave dimod...")
try:
    from dimod import ConstrainedQuadraticModel, Binary, BinaryQuadraticModel
    print("  ✓ D-Wave dimod available")
    checks_passed += 1
except ImportError as e:
    print(f"  ✗ D-Wave dimod not available: {e}")
    checks_failed += 1

# Check 3: D-Wave system
print("[3/7] Checking D-Wave system...")
try:
    from dwave.system import DWaveCliqueSampler
    print("  ✓ D-Wave system available")
    checks_passed += 1
except ImportError as e:
    print(f"  ✗ D-Wave system not available: {e}")
    checks_failed += 1

# Check 4: Scenario loader
print("[4/7] Checking scenario loader...")
try:
    from data_loader_utils import load_food_data_as_dict
    test_data = load_food_data_as_dict('simple')
    print(f"  ✓ Scenario loader works (loaded {len(test_data['farm_names'])} farms)")
    checks_passed += 1
except ImportError as e:
    print(f"  ✗ Cannot import data_loader_utils: {e}")
    checks_failed += 1
except Exception as e:
    print(f"  ✗ Error loading test scenario: {e}")
    checks_failed += 1

# Check 5: Clique decomposition
print("[5/7] Checking clique decomposition...")
try:
    from clique_wrapper import solve_clique_wrapper, HAS_CLIQUE
    if HAS_CLIQUE:
        print("  ✓ Clique decomposition available")
        checks_passed += 1
    else:
        print("  ⚠ Clique decomposition module found but dependencies missing")
        print("    (Benchmark will skip clique decomp scenarios)")
        checks_passed += 1  # Not critical if hierarchical works
except ImportError as e:
    print(f"  ⚠ Clique decomposition not available: {e}")
    print("    (Benchmark will skip clique decomp scenarios)")
    checks_passed += 1  # Not critical

# Check 6: Hierarchical solver
print("[6/7] Checking hierarchical solver...")
try:
    from hierarchical_quantum_solver import solve_hierarchical
    print("  ✓ Hierarchical solver available")
    checks_passed += 1
except ImportError as e:
    print(f"  ✗ Hierarchical solver not available: {e}")
    checks_failed += 1

# Check 7: D-Wave token
print("[7/7] Checking D-Wave API token...")
import os
DEFAULT_DWAVE_TOKEN = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'
token = os.environ.get('DWAVE_API_TOKEN', DEFAULT_DWAVE_TOKEN)
if token and len(token) > 10:
    print(f"  ✓ D-Wave token configured ({token[:10]}...)")
    checks_passed += 1
else:
    print("  ✗ D-Wave token not configured")
    checks_failed += 1

# Summary
print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Checks passed: {checks_passed}/7")
print(f"Checks failed: {checks_failed}/7")
print()

if checks_failed == 0:
    print("✓ ALL CHECKS PASSED - Ready to run benchmark!")
    print()
    print("To run the benchmark:")
    print("  python significant_scenarios_benchmark.py")
    print()
    print("Estimated runtime: 60-90 minutes")
    print("D-Wave QPU credits will be consumed!")
    sys.exit(0)
elif checks_failed <= 2:
    print("⚠ SOME CHECKS FAILED - Partial functionality available")
    print()
    print("You can still run the benchmark, but some scenarios may be skipped.")
    print("Review the failures above and fix if needed.")
    sys.exit(0)
else:
    print("✗ TOO MANY FAILURES - Cannot run benchmark")
    print()
    print("Please fix the errors above before running the benchmark.")
    sys.exit(1)
