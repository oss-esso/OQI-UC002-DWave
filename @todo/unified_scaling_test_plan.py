#!/usr/bin/env python3
"""
Unified Scaling Test: 27-Food Hybrid Formulation Across All Sizes

Uses hybrid formulation for fair comparison:
- All problems: 27 food variables
- Rotation synergies: 6-family structure (via hybrid matrix)
- Auto-detection: Choose decomposition based on size
- Problem sizes: 5, 10, 15, 20, 25, 30, 40, 50, 75, 100 farms

This fills the missing data points and removes the formulation confound!

Author: OQI-UC002-DWave
Date: 2025-12-12
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from hybrid_formulation import (
    build_hybrid_rotation_matrix,
    get_food_family_mapping,
    detect_decomposition_strategy,
    recommend_parameters,
)

print("="*80)
print("UNIFIED SCALING TEST: Hybrid 27-Food Formulation")
print("="*80)
print()
print("Strategy: Keep 27 foods as variables, use 6-family synergy structure")
print("Goal: Fair comparison across all problem sizes (5-100 farms)")
print()

# Test configuration
TEST_SIZES = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
N_FOODS = 27
N_PERIODS = 3

print("Test plan:")
print("-"*80)
for size in TEST_SIZES:
    n_vars = size * N_FOODS * N_PERIODS
    strategy = detect_decomposition_strategy(size, N_FOODS, N_PERIODS)
    print(f"  {size:3d} farms: {n_vars:5d} vars → {strategy['method']}")

print()
print("="*80)
print("KEY ADVANTAGES:")
print("="*80)
print("""
1. ✅ Consistent formulation: All tests use 27 foods
2. ✅ No aggregation confound: Compare apples-to-apples
3. ✅ Hybrid synergies: Full expressiveness + tractable structure
4. ✅ Auto-decomposition: Optimal strategy per size
5. ✅ Fills gaps: Missing data points (25-40 farms)
6. ✅ Fair baseline: Gurobi sees same problem structure

Expected results:
- Gap should be consistent (~15-20%) across all sizes
- No sudden jump at 25 farms (formulation is consistent!)
- Speedup should scale predictably with problem size
- QPU time should scale linearly with variables
""")

print("="*80)
print("IMPLEMENTATION STEPS:")
print("="*80)
print("""
To run this unified test:

1. Create load_hybrid_data() function:
   - Load 27-food scenario
   - Build hybrid rotation matrix (27×27 from 6×6 template)
   - Keep all 27 foods as variables
   
2. Modify solver to accept hybrid matrix:
   - Use R[food_i, food_j] for synergies (not family-level)
   - Auto-detect decomposition strategy
   - Apply spatial decomposition if needed
   
3. Run test across all sizes:
   - 5-20 farms: Compare to existing statistical test
   - 25-100 farms: New data with consistent formulation
   - Gurobi baseline: Same problem structure for all

4. Analyze results:
   - Gap should be consistent (no 135% jump!)
   - Clear scaling laws visible
   - Fair quantum vs classical comparison
""")

print()
print("Would you like me to implement the full unified test? (Y/n)")
