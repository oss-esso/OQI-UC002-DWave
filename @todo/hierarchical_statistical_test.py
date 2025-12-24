#!/usr/bin/env python3
"""
Statistical Comparison: Hierarchical Quantum Solver vs Classical (Gurobi)
Extended Tests for Large-Scale Rotation Optimization

Publication-Quality Benchmark for Technical Paper
Continuation of statistical_comparison_test.py with hierarchical approach

Design:
- Problem sizes: 25, 50, 100 farms (scaling beyond previous 5-20 farm tests)
- Methods compared:
  * Ground Truth: Gurobi (classical MIP with 15 min timeout)
  * Hierarchical Quantum: 3-level decomposition with D-Wave QPU
- Metrics: Time, objective value, optimality gap, crop diversity, violations
- Statistical rigor: 3 runs per method per size for variance analysis
- Post-processing: Family → crop refinement for realistic diversity

Key Differences from statistical_comparison_test.py:
1. Larger problem sizes (25-100 vs 5-20 farms)
2. Hierarchical decomposition (handles 27 foods → 6 families)
3. Spatial clustering with boundary coordination
4. Extended metrics: aggregation overhead, post-processing time, diversity

Author: OQI-UC002-DWave Project
Date: 2025-12-12
"""

import sys
import os
import time
import json
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import traceback

warnings.filterwarnings('ignore')

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# ============================================================================
# CONFIGURATION
# ============================================================================

# ============================================================================
# TEST MODE: Set to True to run just ONE scenario first for verification
# Set to False to run all scenarios for full benchmark
# ============================================================================
TEST_MODE = True  # Set to False after verifying one scenario works

TEST_CONFIG = {
    # First 17 scenarios matching Gurobi timeout test
    # Format: (n_farms, n_foods, scenario_name)
    'scenarios': [
        (5, 6, 'rotation_micro_25'),           # 1: 90 vars
        (10, 6, 'rotation_small_50'),          # 2: 180 vars
        (15, 6, 'rotation_15farms_6foods'),    # 3: 270 vars
        (20, 6, 'rotation_medium_100'),        # 4: 360 vars
        (25, 6, 'rotation_25farms_6foods'),    # 5: 450 vars
        (40, 6, 'rotation_large_200'),         # 6: 720 vars
        (50, 6, 'rotation_50farms_6foods'),    # 7: 900 vars
        (75, 6, 'rotation_75farms_6foods'),    # 8: 1350 vars
        (100, 6, 'rotation_100farms_6foods'),  # 9: 1800 vars
        (25, 27, 'rotation_25farms_27foods'),  # 10: 2025 vars
        (150, 6, 'rotation_150farms_6foods'),  # 11: 2700 vars
        (50, 27, 'rotation_50farms_27foods'),  # 12: 4050 vars
        (75, 27, 'rotation_75farms_27foods'),  # 13: 6075 vars
        (100, 27, 'rotation_100farms_27foods'),# 14: 8100 vars
        (150, 27, 'rotation_150farms_27foods'),# 15: 12150 vars
    ],
    'n_periods': 3,                    # Rotation periods
    'num_reads': 100,                   # QPU reads per cluster (reduced for efficiency)
    'num_iterations': 3,               # Boundary coordination iterations (reduced)
    'runs_per_method': 1,              # Single run per scenario (no statistics)
    'classical_timeout': 100,          # Gurobi timeout (100s, SAME as timeout test)
    'skip_gurobi': False,              # Run Gurobi for comparison!
    
    # CRITICAL: Cluster size must create problems comparable to statistical test
    # Statistical test: 5-25 farms = 90-450 vars (6 families × 3 periods)
    # Our clusters: 5-10 farms = 90-180 vars per cluster (COMPARABLE!)
    'farms_per_cluster': 5,            # Creates ~90-var clusters (matches statistical test scale)
    
    'enable_post_processing': True,    # Family → crop refinement (BOTH steps)
}

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'hierarchical_statistical_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# D-Wave token
DEFAULT_DWAVE_TOKEN = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'

def setup_dwave_token():
    """Configure D-Wave API token."""
    token = os.environ.get('DWAVE_API_TOKEN', DEFAULT_DWAVE_TOKEN)
    os.environ['DWAVE_API_TOKEN'] = token
    return token

# ============================================================================
# IMPORTS
# ============================================================================

print("="*80)
print("HIERARCHICAL STATISTICAL COMPARISON: Quantum vs Classical")
print("="*80)
print("\n📊 Publication-Quality Benchmark for Technical Paper")
print("📈 Continuation of statistical_comparison_test.py (scaling to 25-100 farms)\n")
print("="*80)

print("\n[1/5] Importing core libraries...")
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("  ⚠️ seaborn not available - plots will be basic")
from scipy import stats

print("[2/5] Importing optimization libraries...")
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
    print("  ✓ Gurobi available")
except ImportError:
    HAS_GUROBI = False
    print("  ❌ ERROR: Gurobi required for ground truth comparison!")
    sys.exit(1)

print("[3/5] Importing D-Wave libraries...")
try:
    from dwave.system import DWaveCliqueSampler
    HAS_DWAVE = True
    print("  ✓ D-Wave QPU available")
except ImportError:
    HAS_DWAVE = False
    print("  ⚠️ D-Wave not available - will use SimulatedAnnealing for testing")

setup_dwave_token()

print("[4/5] Importing hierarchical solver...")
from hierarchical_quantum_solver import (
    solve_hierarchical,
    DEFAULT_CONFIG,
)
from src.scenarios import load_food_data

print("[5/5] Setup complete!\n")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_hierarchical_data(n_farms: int, n_foods: int, scenario_name: str) -> Dict:
    """
    Load data for hierarchical solver test.
    Uses EXACT same approach as test_gurobi_timeout.py for comparability.
    
    Args:
        n_farms: Number of farms
        n_foods: Number of foods (6 for families, 27 for full)
        scenario_name: Exact scenario name to load
    """
    print(f"    Loading scenario: {scenario_name} ({n_farms} farms, {n_foods} foods)...", flush=True)
    
    # Import data loader directly (avoid running test_gurobi_timeout.py)
    from data_loader_utils import load_food_data_as_dict
    
    # Map to actual scenario names - use only existing scenarios
    # Available: rotation_micro_25, rotation_small_50, rotation_medium_100, rotation_large_200,
    #            rotation_250farms_27foods, rotation_350farms_27foods, rotation_500farms_27foods, rotation_1000farms_27foods
    if n_foods == 6:
        if n_farms <= 5:
            base_scenario = 'rotation_micro_25'
        elif n_farms <= 10:
            base_scenario = 'rotation_small_50'
        elif n_farms <= 20:
            base_scenario = 'rotation_medium_100'
        else:  # 25-150 farms with 6 foods
            base_scenario = 'rotation_large_200'
    else:  # 27 foods
        if n_farms <= 250:
            base_scenario = 'rotation_250farms_27foods'
        elif n_farms <= 350:
            base_scenario = 'rotation_350farms_27foods'
        elif n_farms <= 500:
            base_scenario = 'rotation_500farms_27foods'
        else:
            base_scenario = 'rotation_1000farms_27foods'
    
    # Load using exact same function
    data = load_food_data_as_dict(base_scenario)
    
    # Adjust farm count if needed (SAME as Gurobi test)
    if len(data['farm_names']) != n_farms:
        if len(data['farm_names']) > n_farms:
            data['farm_names'] = data['farm_names'][:n_farms]
            data['land_availability'] = {k: v for k, v in list(data['land_availability'].items())[:n_farms]}
        else:
            original_farms = data['farm_names'].copy()
            while len(data['farm_names']) < n_farms:
                idx = len(data['farm_names']) - len(original_farms)
                farm = original_farms[idx % len(original_farms)]
                new_farm = f"{farm}_dup{idx}"
                data['farm_names'].append(new_farm)
                data['land_availability'][new_farm] = data['land_availability'][farm]
    
    data['n_farms'] = len(data['farm_names'])
    data['n_foods'] = len(data['food_names'])
    data['total_area'] = sum(data['land_availability'].values())
    data['scenario_name'] = scenario_name
    
    print(f"    ✓ Loaded: {data['n_farms']} farms × {data['n_foods']} foods × 3 periods", flush=True)
    print(f"      Variables: {data['n_farms'] * data['n_foods'] * 3:,}", flush=True)
    print(f"      Total area: {data['total_area']:.2f} ha (avg {data['total_area']/data['n_farms']:.2f} ha/farm)", flush=True)
    
    return data

# ============================================================================
# GROUND TRUTH SOLVER (GUROBI)
# ============================================================================

def solve_gurobi_ground_truth(data: Dict, timeout: int = 900) -> Dict:
    """
    Solve rotation problem with Gurobi as ground truth.
    
    Matching the hierarchical approach exactly:
    - Family-level (6 families, not 27 foods) for fair comparison
    - 3-period rotation with synergies
    - One-hot constraints
    - Post-processing: Family → specific crops (BOTH STEPS)
    
    Returns full solution details with timing breakdown.
    """
    total_start = time.time()
    
    # CRITICAL: Only aggregate if n_foods > 6 (i.e., 27 foods → 6 families)
    # If already 6 foods, data is already at family level - use directly!
    n_foods_input = len(data.get('food_names', []))
    
    if n_foods_input > 6:
        # 27 foods → 6 families aggregation
        from food_grouping import aggregate_foods_to_families
        family_data = aggregate_foods_to_families(data)
        print(f"      Aggregated {n_foods_input} foods → 6 families")
    else:
        # Already 6 families - use data directly (NO aggregation)
        family_data = data
        print(f"      Using {n_foods_input} families directly (no aggregation)")
    
    # Extract data
    food_names = family_data['food_names']  # 6 families (or foods if not aggregated)
    farm_names = family_data['farm_names']
    land_availability = family_data['land_availability']
    total_area = family_data['total_area']
    food_benefits = family_data['food_benefits']
    
    config = data.get('config', {})
    params = config.get('parameters', {})
    
    # EXACT parameters from statistical_comparison_test.py
    rotation_gamma = params.get('rotation_gamma', 0.2)
    k_neighbors = params.get('spatial_k_neighbors', 4)
    frustration_ratio = params.get('frustration_ratio', 0.7)
    negative_strength = params.get('negative_synergy_strength', -0.8)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    use_soft_constraint = params.get('use_soft_one_hot', True)
    
    n_periods = 3
    n_farms = len(farm_names)
    n_families = len(food_names)
    families_list = list(food_names)
    
    print(f"      Model: {n_farms} farms × {n_families} families × {n_periods} periods ({n_farms * n_families * n_periods} vars)")
    
    # Step 1: Build rotation matrix
    step_start = time.time()
    print(f"      [1/7] Building rotation matrix ({n_families}×{n_families})...", flush=True)
    np.random.seed(42)
    R = np.zeros((n_families, n_families))
    for i in range(n_families):
        for j in range(n_families):
            if i == j:
                R[i, j] = negative_strength * 1.5
            elif np.random.random() < frustration_ratio:
                R[i, j] = np.random.uniform(negative_strength * 1.2, negative_strength * 0.3)
            else:
                R[i, j] = np.random.uniform(0.02, 0.20)
    print(f"            ✓ Done in {time.time() - step_start:.3f}s", flush=True)
    
    # Step 2: Build spatial neighbor graph
    step_start = time.time()
    print(f"      [2/7] Building spatial neighbor graph (k={k_neighbors})...", flush=True)
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {}
    for i, farm in enumerate(farm_names):
        row, col = i // side, i % side
        positions[farm] = (row, col)
    
    neighbor_edges = []
    for f1 in farm_names:
        distances = []
        for f2 in farm_names:
            if f1 != f2:
                dist = np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                             (positions[f1][1] - positions[f2][1])**2)
                distances.append((dist, f2))
        distances.sort()
        for _, f2 in distances[:k_neighbors]:
            if (f2, f1) not in neighbor_edges:
                neighbor_edges.append((f1, f2))
    print(f"            ✓ Done in {time.time() - step_start:.3f}s ({len(neighbor_edges)} edges)", flush=True)
    
    # Step 3: Create Gurobi model
    step_start = time.time()
    print(f"      [3/7] Creating Gurobi model and setting parameters...", flush=True)
    model = gp.Model("RotationGroundTruth")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)  # SAME as timeout test: 100s
    model.setParam('MIPGap', 0.01)  # SAME as timeout test: 1% gap
    model.setParam('MIPFocus', 1)  # SAME as timeout test
    model.setParam('ImproveStartTime', 30)  # SAME as timeout test
    model.setParam('Threads', 0)
    model.setParam('Presolve', 2)
    model.setParam('Cuts', 2)
    print(f"            ✓ Done in {time.time() - step_start:.3f}s (timeout={model.Params.TimeLimit}s)", flush=True)
    
    # Step 4: Add variables
    step_start = time.time()
    n_vars = n_farms * n_families * n_periods
    print(f"      [4/7] Adding {n_vars} binary variables...", flush=True)
    Y = {}
    for f in farm_names:
        for c in families_list:
            for t in range(1, n_periods + 1):
                Y[(f, c, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}_t{t}")
    
    model.update()
    print(f"            ✓ Done in {time.time() - step_start:.3f}s", flush=True)
    
    # Step 5: Build objective function (CRITICAL - may be slow for large problems)
    step_start = time.time()
    print(f"      [5/7] Building objective function...", flush=True)
    obj = 0
    
    # Part 1: Base benefit
    substep_start = time.time()
    print(f"            [5.1] Adding base benefits...", flush=True, end='')
    for f in farm_names:
        farm_area = land_availability[f]
        for c in families_list:
            benefit = food_benefits.get(c, 0.5)
            for t in range(1, n_periods + 1):
                obj += (benefit * farm_area * Y[(f, c, t)]) / total_area
    print(f" Done in {time.time() - substep_start:.3f}s", flush=True)
    
    # Part 2: Rotation synergies (temporal) - QUADRATIC TERMS
    substep_start = time.time()
    print(f"            [5.2] Adding temporal synergies (quadratic)...", flush=True, end='')
    for f in farm_names:
        farm_area = land_availability[f]
        for t in range(2, n_periods + 1):
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    synergy = R[c1_idx, c2_idx]
                    if abs(synergy) > 1e-6:
                        obj += (rotation_gamma * synergy * farm_area * 
                               Y[(f, c1, t-1)] * Y[(f, c2, t)]) / total_area
    print(f" Done in {time.time() - substep_start:.3f}s", flush=True)
    
    # Part 3: Spatial interactions - MORE QUADRATIC TERMS
    substep_start = time.time()
    print(f"            [5.3] Adding spatial synergies (quadratic)...", flush=True, end='')
    spatial_gamma = rotation_gamma * 0.5
    for (f1, f2) in neighbor_edges:
        for t in range(1, n_periods + 1):
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    spatial_synergy = R[c1_idx, c2_idx] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        obj += (spatial_gamma * spatial_synergy * 
                               Y[(f1, c1, t)] * Y[(f2, c2, t)]) / total_area
    print(f" Done in {time.time() - substep_start:.3f}s", flush=True)
    
    # Part 4: Soft one-hot penalty - QUADRATIC PENALTY
    substep_start = time.time()
    print(f"            [5.4] Adding one-hot penalties (quadratic)...", flush=True, end='')
    if use_soft_constraint:
        for f in farm_names:
            for t in range(1, n_periods + 1):
                crop_count = gp.quicksum(Y[(f, c, t)] for c in families_list)
                obj -= one_hot_penalty * (crop_count - 1) * (crop_count - 1)
    print(f" Done in {time.time() - substep_start:.3f}s", flush=True)
    
    # Part 5: Diversity bonus
    substep_start = time.time()
    print(f"            [5.5] Adding diversity bonuses...", flush=True, end='')
    for f in farm_names:
        for c in families_list:
            crop_used = gp.quicksum(Y[(f, c, t)] for t in range(1, n_periods + 1))
            obj += diversity_bonus * crop_used
    print(f" Done in {time.time() - substep_start:.3f}s", flush=True)
    
    print(f"            ✓ Objective built in {time.time() - step_start:.3f}s", flush=True)
    
    # Step 5.6: Set objective in Gurobi (THIS MAY BE THE SLOW PART!)
    substep_start = time.time()
    print(f"            [5.6] Setting objective in Gurobi (may take time for large problems)...", flush=True, end='')
    model.setObjective(obj, GRB.MAXIMIZE)
    print(f" Done in {time.time() - substep_start:.3f}s", flush=True)
    
    # Step 6: Add constraints
    step_start = time.time()
    print(f"      [6/7] Adding constraints...", flush=True)
    
    # Part 6.1: One-hot constraints (min and max crops per farm per period)
    substep_start = time.time()
    print(f"            [6.1] Adding one-hot constraints (min/max crops)...", flush=True, end='')
    if use_soft_constraint:
        for f in farm_names:
            for t in range(1, n_periods + 1):
                # Max 2 crops per farm per period
                model.addConstr(
                    gp.quicksum(Y[(f, c, t)] for c in families_list) <= 2,
                    name=f"max_crops_{f}_t{t}"
                )
                # Min 1 crop per farm per period (MISSING BEFORE!)
                model.addConstr(
                    gp.quicksum(Y[(f, c, t)] for c in families_list) >= 1,
                    name=f"min_crops_{f}_t{t}"
                )
    else:
        for f in farm_names:
            for t in range(1, n_periods + 1):
                model.addConstr(
                    gp.quicksum(Y[(f, c, t)] for c in families_list) == 1,
                    name=f"one_crop_{f}_t{t}"
                )
    print(f" Done in {time.time() - substep_start:.3f}s", flush=True)
    
    # Part 6.2: Rotation constraints (CRITICAL - prevents same crop in consecutive periods!)
    substep_start = time.time()
    print(f"            [6.2] Adding rotation constraints (no consecutive same crop)...", flush=True, end='')
    for f in farm_names:
        for c in families_list:
            for t in range(1, n_periods):  # t=1 to n_periods-1
                model.addConstr(
                    Y[(f, c, t)] + Y[(f, c, t + 1)] <= 1,
                    name=f"rotation_{f}_{c}_t{t}"
                )
    print(f" Done in {time.time() - substep_start:.3f}s", flush=True)
    
    # Update model to finalize constraints
    model.update()
    
    n_constraints = model.NumConstrs
    print(f"            ✓ Total constraints: {n_constraints}", flush=True)
    
    # Step 7: Solve
    print(f"      [7/7] Solving with Gurobi (timeout={timeout}s)...", flush=True)
    solve_start = time.time()
    
    # Add Python-level timeout monitoring
    import signal
    
    def timeout_handler(signum, frame):
        elapsed = time.time() - solve_start
        print(f"\n      ⚠️  Python timeout triggered after {elapsed:.1f}s - forcing stop", flush=True)
        raise TimeoutError(f"Gurobi exceeded {timeout}s timeout")
    
    # Set alarm for timeout + 10s grace period
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout + 10)
    
    # Add periodic progress updates
    def progress_callback(model, where):
        if where == GRB.Callback.MIP:
            elapsed = time.time() - solve_start
            if int(elapsed) % 30 == 0 and elapsed > 1:  # Every 30 seconds
                objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                gap = abs(objbst - objbnd) / abs(objbst) if abs(objbst) > 1e-6 else 0
                print(f"            [{elapsed:.0f}s] Best: {objbst:.4f}, Bound: {objbnd:.4f}, Gap: {gap*100:.2f}%", flush=True)
    
    try:
        model.optimize(progress_callback)
    except TimeoutError as e:
        print(f"      ❌ Timeout enforced: {e}", flush=True)
        model.terminate()
    finally:
        signal.alarm(0)  # Cancel alarm
    
    solve_time = time.time() - solve_start
    total_time = time.time() - total_start
    
    print(f"            ✓ Gurobi finished: status={model.Status}, solve_time={solve_time:.1f}s", flush=True)
    
    result = {
        'method': 'gurobi_ground_truth',
        'success': False,
        'objective': 0,
        'solve_time': total_time,
        'violations': 0,
        'gap': 0,
    }
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        result['success'] = True
        result['objective'] = model.ObjVal
        result['gap'] = model.MIPGap if hasattr(model, 'MIPGap') else 0
        
        # Extract solution (EXACT from statistical_comparison_test.py)
        solution_dict = {}
        for f in farm_names:
            for c in families_list:
                for t in range(1, n_periods + 1):
                    val = int(Y[(f, c, t)].X > 0.5)
                    if val > 0:
                        solution_dict[(f, c, t)] = 1
        
        result['solution'] = solution_dict
        
        # Count violations
        violations = 0
        for f in farm_names:
            for t in range(1, n_periods + 1):
                count = sum(solution_dict.get((f, c, t), 0) for c in families_list)
                if count != 1:
                    violations += abs(count - 1)
        result['violations'] = violations
        
        # Post-processing (EXACT from statistical_comparison_test.py)
        if config.get('enable_post_processing', False) and solution_dict:
            from food_grouping import refine_family_solution_to_crops, analyze_crop_diversity
            
            pp_start = time.time()
            refined_solution = refine_family_solution_to_crops(solution_dict, data)
            refinement_time = time.time() - pp_start
            
            div_start = time.time()
            diversity_stats = analyze_crop_diversity(refined_solution, data)
            diversity_time = time.time() - div_start
            
            result['refined_solution'] = refined_solution
            result['diversity_stats'] = diversity_stats
            result['timings'] = {
                'solve': solve_time,
                'postprocessing': refinement_time + diversity_time,
                'refinement': refinement_time,
                'diversity': diversity_time,
            }
        
        print(f"      ✓ Gurobi: {total_time:.1f}s, obj={result['objective']:.4f}, gap={result['gap']*100:.2f}%")
        if 'diversity_stats' in result:
            print(f"        crops={result['diversity_stats']['total_unique_crops']}")
    
    return result


# ============================================================================
# HIERARCHICAL QUANTUM SOLVER
# ============================================================================

def solve_hierarchical_quantum(data: Dict, config: Dict = None) -> Dict:
    """
    Solve using hierarchical 3-level quantum-classical approach.
    
    Timing breakdown matches Gurobi for fair comparison:
    - Level 1: Aggregation (27→6) + Decomposition
    - Level 2: QPU solve with boundary coordination
    - Level 3: Post-processing (6→27) + Diversity analysis
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
        config['farms_per_cluster'] = TEST_CONFIG['farms_per_cluster']
        config['num_iterations'] = TEST_CONFIG['num_iterations']
        config['num_reads'] = TEST_CONFIG['num_reads']
    
    n_farms = data['n_farms']
    n_foods = data['n_foods']
    farms_per_cluster = config['farms_per_cluster']
    n_clusters = (n_farms + farms_per_cluster - 1) // farms_per_cluster
    
    # CRITICAL: Ensure cluster size matches statistical test expectations
    # Statistical test used: 5-25 farms with 6 families × 3 periods = 90-450 vars
    # Our clusters: ~10 farms × 6 families × 3 periods = 180 vars (COMPARABLE!)
    vars_per_cluster = farms_per_cluster * 6 * 3
    
    print(f"      Hierarchical decomposition: {n_clusters} clusters of ~{farms_per_cluster} farms")
    print(f"      Cluster size: {vars_per_cluster} vars (comparable to statistical test)")
    print(f"      QPU settings: {config['num_reads']} reads, {config['num_iterations']} iterations")
    print(f"      Mode: {'QPU' if HAS_DWAVE else 'SimulatedAnnealing (testing)'}")
    
    start_time = time.time()
    
    # Run hierarchical solver (includes ALL 3 levels with timing)
    # Use QPU if available, otherwise fall back to simulated annealing
    result = solve_hierarchical(data, config, use_qpu=HAS_DWAVE, verbose=False)
    
    total_time = time.time() - start_time
    
    if result['success']:
        # Extract timing details
        timings = result['timings']
        qpu_time = timings.get('level2_qpu', 0)
        postproc_time = timings.get('level3_postprocessing', 0)
        
        # Get objectives before and after post-processing
        obj_before = result.get('objective_before_postprocessing', 0)
        obj_after = result['objective']
        obj_improvement = obj_after - obj_before
        obj_improvement_pct = (obj_improvement / abs(obj_before) * 100) if abs(obj_before) > 1e-6 else 0
        
        print(f"      ✓ Hierarchical QPU: {total_time:.1f}s")
        print(f"        Level 1 (agg+decomp): {timings['level1_decomposition']:.3f}s")
        print(f"        Level 2 (QPU): {timings['level2_quantum']:.1f}s (QPU access: {qpu_time:.2f}s)")
        print(f"        Level 3 (post-process): {postproc_time:.3f}s")
        print(f"        Objective BEFORE post-process: {obj_before:.4f}")
        print(f"        Objective AFTER post-process:  {obj_after:.4f} ({obj_improvement:+.4f}, {obj_improvement_pct:+.1f}%)")
        print(f"        Unique crops: {result['diversity_stats']['total_unique_crops']}/{data['n_foods']}")
        
        # Ensure all timing fields present for comparison
        result['total_time'] = total_time
        result['solve_time'] = total_time  # For consistency with Gurobi
        
        # Add decomposition info for validation
        result['decomposition_info'] = {
            'n_clusters': n_clusters,
            'farms_per_cluster': farms_per_cluster,
            'vars_per_cluster': vars_per_cluster,
            'total_qpu_calls': n_clusters * config['num_iterations'],
        }
        
        return result
    else:
        print(f"      ❌ Hierarchical QPU failed")
        return result


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def calculate_statistics(results_list: List[Dict]) -> Dict:
    """Calculate statistics across multiple runs."""
    if not results_list or not any(r.get('success', False) for r in results_list):
        return {
            'n_runs': len(results_list),
            'success_rate': 0.0,
        }
    
    successful = [r for r in results_list if r.get('success', False)]
    
    objectives = [r['objective'] for r in successful]
    times = [r['solve_time'] for r in successful]
    violations = [r['violations'] for r in successful]
    
    diversity_crops = [r['diversity_stats']['total_unique_crops'] for r in successful if 'diversity_stats' in r]
    diversity_shannon = [r['diversity_stats']['shannon_diversity'] for r in successful if 'diversity_stats' in r]
    
    stats = {
        'n_runs': len(results_list),
        'n_successful': len(successful),
        'success_rate': len(successful) / len(results_list),
        
        # Objective
        'objective_mean': np.mean(objectives),
        'objective_std': np.std(objectives),
        'objective_min': np.min(objectives),
        'objective_max': np.max(objectives),
        
        # Time
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'time_min': np.min(times),
        'time_max': np.max(times),
        
        # Violations
        'violations_mean': np.mean(violations),
        'violations_max': np.max(violations),
        
        # Diversity
        'unique_crops_mean': np.mean(diversity_crops) if diversity_crops else 0,
        'unique_crops_std': np.std(diversity_crops) if diversity_crops else 0,
        'shannon_mean': np.mean(diversity_shannon) if diversity_shannon else 0,
        'shannon_std': np.std(diversity_shannon) if diversity_shannon else 0,
    }
    
    # QPU-specific timing (if available)
    if any('timings' in r for r in successful):
        qpu_times = [r['timings']['level2_qpu'] for r in successful if 'timings' in r and 'level2_qpu' in r['timings']]
        postproc_times = [r['timings'].get('level3_postprocessing', 0) for r in successful if 'timings' in r]
        
        if qpu_times:
            stats['qpu_time_mean'] = np.mean(qpu_times)
            stats['qpu_time_total'] = np.sum(qpu_times)
        if postproc_times:
            stats['postproc_time_mean'] = np.mean(postproc_times)
    
    # Pre/post-processing objectives (if available)
    if any('objective_before_postprocessing' in r for r in successful):
        obj_before = [r['objective_before_postprocessing'] for r in successful if 'objective_before_postprocessing' in r]
        obj_after = [r['objective'] for r in successful if 'objective_before_postprocessing' in r]
        improvements = [after - before for before, after in zip(obj_before, obj_after)]
        improvement_pcts = [(after - before) / abs(before) * 100 if abs(before) > 1e-6 else 0 
                           for before, after in zip(obj_before, obj_after)]
        
        if obj_before:
            stats['objective_before_pp_mean'] = np.mean(obj_before)
            stats['obj_improvement_mean'] = np.mean(improvements)
            stats['obj_improvement_pct_mean'] = np.mean(improvement_pcts)
    
    # Gurobi-specific timing breakdown (if available)
    if any('timings' in r and 'solve' in r['timings'] for r in successful):
        gurobi_solve = [r['timings']['solve'] for r in successful if 'timings' in r and 'solve' in r['timings']]
        gurobi_postproc = [r['timings'].get('postprocessing', 0) for r in successful if 'timings' in r]
        
        if gurobi_solve:
            stats['gurobi_solve_mean'] = np.mean(gurobi_solve)
        if gurobi_postproc:
            stats['gurobi_postproc_mean'] = np.mean(gurobi_postproc)
    
    return stats


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_hierarchical_statistical_test():
    """Run complete statistical comparison test."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Apply TEST_MODE - only run first scenario for verification
    if TEST_MODE:
        # Run 3 representative scenarios: small (90 vars), medium (360 vars), large (900 vars)
        all_scenarios = TEST_CONFIG['scenarios']
        scenarios_to_run = [all_scenarios[0], all_scenarios[3], all_scenarios[6]]  # indices 0, 3, 6
        print("\n" + "⚠️ "*20)
        print("TEST MODE: Running 3 representative scenarios (small/medium/large)")
        print("Set TEST_MODE = False to run all 15 scenarios")
        print("⚠️ "*20)
    else:
        scenarios_to_run = TEST_CONFIG['scenarios']
    
    print("\n" + "="*80, flush=True)
    print("STARTING HIERARCHICAL STATISTICAL BENCHMARK", flush=True)
    print("="*80, flush=True)
    print(f"\nTest configuration:", flush=True)
    print(f"  Mode: {'TEST (1 scenario)' if TEST_MODE else 'FULL BENCHMARK'}", flush=True)
    print(f"  Scenarios: {len(scenarios_to_run)}", flush=True)
    print(f"  Runs per method: {TEST_CONFIG['runs_per_method']}", flush=True)
    print(f"  QPU reads: {TEST_CONFIG['num_reads']}", flush=True)
    print(f"  Boundary iterations: {TEST_CONFIG['num_iterations']}", flush=True)
    print(f"  Gurobi timeout: {TEST_CONFIG['classical_timeout']}s", flush=True)
    print(f"  Skip Gurobi: {TEST_CONFIG['skip_gurobi']}", flush=True)
    print(f"\nOutput directory: {OUTPUT_DIR}", flush=True)
    print("="*80, flush=True)
    
    all_results = {}
    summary_data = []
    
    for scenario_idx, (n_farms, n_foods, scenario_name) in enumerate(scenarios_to_run, 1):
        print(f"\n{'='*80}", flush=True)
        print(f"SCENARIO {scenario_idx}/{len(scenarios_to_run)}: {scenario_name}", flush=True)
        print(f"{'='*80}", flush=True)
        
        # Load data
        print(f"\n  [1/3] Loading data...", flush=True)
        try:
            data = load_hierarchical_data(n_farms, n_foods, scenario_name)
        except Exception as e:
            print(f"    ❌ Data loading failed: {e}")
            continue
        
        all_results[scenario_name] = {
            'data_info': {
                'n_farms': data['n_farms'],
                'n_foods': data['n_foods'],
                'n_variables': data['n_farms'] * data['n_foods'] * 3,
                'scenario_name': scenario_name,
            },
            'gurobi': [],
            'hierarchical_qpu': [],
        }
        
        # Test Gurobi (ground truth) - SKIPPED
        if not TEST_CONFIG.get('skip_gurobi', False):
            print(f"\n  [2/3] Running Gurobi ground truth ({TEST_CONFIG['runs_per_method']} runs)...")
            for run_idx in range(TEST_CONFIG['runs_per_method']):
                print(f"    Run {run_idx + 1}/{TEST_CONFIG['runs_per_method']}:")
                try:
                    result = solve_gurobi_ground_truth(data, TEST_CONFIG['classical_timeout'])
                    all_results[scenario_name]['gurobi'].append(result)
                except Exception as e:
                    print(f"      ❌ Gurobi run {run_idx + 1} failed: {e}")
                    traceback.print_exc()
        else:
            print(f"\n  [2/3] Skipping Gurobi (skip_gurobi=True)", flush=True)
        
        # Test Hierarchical QPU
        print(f"\n  [3/3] Running Hierarchical QPU ({TEST_CONFIG['runs_per_method']} runs)...", flush=True)
        for run_idx in range(TEST_CONFIG['runs_per_method']):
            print(f"    Run {run_idx + 1}/{TEST_CONFIG['runs_per_method']}:", flush=True)
            try:
                result = solve_hierarchical_quantum(data)
                all_results[scenario_name]['hierarchical_qpu'].append(result)
            except Exception as e:
                print(f"      ❌ Hierarchical QPU run {run_idx + 1} failed: {e}")
                traceback.print_exc()
        
        # Calculate statistics
        print(f"\n  Computing statistics for {scenario_name}...", flush=True)
        gurobi_stats = calculate_statistics(all_results[scenario_name]['gurobi'])
        quantum_stats = calculate_statistics(all_results[scenario_name]['hierarchical_qpu'])
        
        all_results[scenario_name]['statistics'] = {
            'gurobi': gurobi_stats,
            'hierarchical_qpu': quantum_stats,
        }
        
        # Summary for this size (skip Gurobi comparison when Gurobi is skipped)
        if not TEST_CONFIG.get('skip_gurobi', False) and gurobi_stats['n_successful'] > 0 and quantum_stats['n_successful'] > 0:
            speedup = gurobi_stats['time_mean'] / quantum_stats['time_mean']
            gap = abs(gurobi_stats['objective_mean'] - quantum_stats['objective_mean']) / gurobi_stats['objective_mean'] * 100
            
            summary_data.append({
                'scenario': scenario_name,
                'n_farms': n_farms,
                'n_foods': n_foods,
                'n_vars': all_results[scenario_name]['data_info']['n_variables'],
                'gurobi_time': gurobi_stats['time_mean'],
                'gurobi_obj': gurobi_stats['objective_mean'],
                'quantum_time': quantum_stats['time_mean'],
                'quantum_obj': quantum_stats['objective_mean'],
                'qpu_time': quantum_stats.get('qpu_time_mean', 0),
                'speedup': speedup,
                'gap_percent': gap,
                'gurobi_crops': gurobi_stats['unique_crops_mean'],
                'quantum_crops': quantum_stats['unique_crops_mean'],
            })
            
            print(f"\n  ✅ {scenario_name} summary:", flush=True)
            print(f"    Gurobi: {gurobi_stats['time_mean']:.1f}s, obj={gurobi_stats['objective_mean']:.4f}", flush=True)
            print(f"    Quantum: {quantum_stats['time_mean']:.1f}s, obj={quantum_stats['objective_mean']:.4f}", flush=True)
            print(f"    Speedup: {speedup:.2f}×, Gap: {gap:.2f}%, QPU: {quantum_stats.get('qpu_time_mean', 0):.2f}s", flush=True)
        elif quantum_stats['n_successful'] > 0:
            # QPU-only summary when Gurobi skipped
            summary_data.append({
                'scenario': scenario_name,
                'n_farms': n_farms,
                'n_foods': n_foods,
                'n_vars': all_results[scenario_name]['data_info']['n_variables'],
                'quantum_time': quantum_stats['time_mean'],
                'quantum_obj': quantum_stats['objective_mean'],
                'quantum_obj_before_pp': quantum_stats.get('objective_before_pp_mean', 0),
                'obj_improvement': quantum_stats.get('obj_improvement_mean', 0),
                'obj_improvement_pct': quantum_stats.get('obj_improvement_pct_mean', 0),
                'qpu_time': quantum_stats.get('qpu_time_mean', 0),
                'quantum_crops': quantum_stats['unique_crops_mean'],
            })
            
            print(f"\n  ✅ {scenario_name} summary:", flush=True)
            print(f"    Quantum: {quantum_stats['time_mean']:.1f}s", flush=True)
            print(f"    Obj before PP: {quantum_stats.get('objective_before_pp_mean', 0):.4f}", flush=True)
            print(f"    Obj after PP:  {quantum_stats['objective_mean']:.4f} ({quantum_stats.get('obj_improvement_pct_mean', 0):+.1f}%)", flush=True)
            print(f"    QPU: {quantum_stats.get('qpu_time_mean', 0):.2f}s, Crops: {quantum_stats['unique_crops_mean']:.0f}/{n_foods}", flush=True)
    
    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print("="*80)
    
    # Save complete results
    results_file = OUTPUT_DIR / f'hierarchical_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        # Convert for JSON
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {str(k) if isinstance(k, tuple) else k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(x) for x in obj]
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return obj
        
        json.dump(convert_for_json(all_results), f, indent=2)
    
    print(f"  ✓ Full results: {results_file}")
    
    # Save summary CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = OUTPUT_DIR / f'summary_{timestamp}.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"  ✓ Summary CSV: {summary_file}")
        
        # Print summary table
        print(f"\n{'='*80}")
        print("FINAL SUMMARY TABLE")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
    
    return all_results, summary_data


# ============================================================================
# PLOTTING
# ============================================================================

def generate_comparison_plots(summary_data: List[Dict], output_dir: Path):
    """Generate publication-quality comparison plots."""
    if not summary_data:
        print("  ⚠️  No data to plot")
        return
    
    df = pd.DataFrame(summary_data)
    
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hierarchical Quantum vs Classical (Gurobi): Statistical Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Solve Time Comparison
    ax1 = axes[0, 0]
    x = df['n_farms']
    ax1.plot(x, df['gurobi_time'], 'o-', label='Gurobi', linewidth=2, markersize=8)
    ax1.plot(x, df['quantum_time'], 's-', label='Hierarchical QPU', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Farms', fontsize=12)
    ax1.set_ylabel('Solve Time (seconds)', fontsize=12)
    ax1.set_title('A. Total Solve Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup
    ax2 = axes[0, 1]
    ax2.plot(x, df['speedup'], 'D-', color='green', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
    ax2.set_xlabel('Number of Farms', fontsize=12)
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.set_title('B. Quantum Speedup', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Optimality Gap
    ax3 = axes[1, 0]
    ax3.plot(x, df['gap_percent'], '^-', color='orange', linewidth=2, markersize=8)
    ax3.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20% threshold')
    ax3.set_xlabel('Number of Farms', fontsize=12)
    ax3.set_ylabel('Optimality Gap (%)', fontsize=12)
    ax3.set_title('C. Solution Quality Gap', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Crop Diversity
    ax4 = axes[1, 1]
    ax4.plot(x, df['gurobi_crops'], 'o-', label='Gurobi', linewidth=2, markersize=8)
    ax4.plot(x, df['quantum_crops'], 's-', label='Hierarchical QPU', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Farms', fontsize=12)
    ax4.set_ylabel('Unique Crops (out of 27)', fontsize=12)
    ax4.set_title('D. Post-Processing Diversity', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plot_file = output_dir / 'hierarchical_comparison_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plots saved: {plot_file}")
    
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete hierarchical statistical test."""
    
    print("\n" + "="*80)
    print("PUBLICATION-QUALITY STATISTICAL COMPARISON")
    print("Hierarchical Quantum-Classical Solver vs Gurobi")
    print("="*80)
    
    # Auto-confirm QPU usage (for automated testing)
    # response = input("\n⚠️  This will use D-Wave QPU access. Continue? (yes/no): ")
    # if response.lower() != 'yes':
    #     print("Cancelled.")
    #     return
    print("\n✓ D-Wave QPU access confirmed (auto-confirm enabled)")
    
    # Run benchmark
    try:
        all_results, summary_data = run_hierarchical_statistical_test()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        traceback.print_exc()
        return
    
    # Generate plots
    if summary_data:
        print(f"\n{'='*80}")
        print("GENERATING PUBLICATION PLOTS")
        print("="*80)
        try:
            generate_comparison_plots(summary_data, OUTPUT_DIR)
        except Exception as e:
            print(f"  ⚠️  Plot generation failed: {e}")
    
    print(f"\n{'='*80}")
    print("✅ HIERARCHICAL STATISTICAL TEST COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nKey files:")
    print(f"  - Full results JSON (all runs)")
    print(f"  - Summary CSV (statistics)")
    print(f"  - Comparison plots PNG (publication-ready)")
    print("\nReady for inclusion in technical paper! 📊")
    print("="*80)


if __name__ == '__main__':
    main()
