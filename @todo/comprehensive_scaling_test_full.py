#!/usr/bin/env python3
"""
Comprehensive Scaling Test: Full Solution Details with REAL D-Wave QPU
Tests ALL scenarios from test_gurobi_timeout.py (up to 200 farms)
and stores FULL solution details for analysis.

This test:
1. Runs Gurobi with timeout (classical baseline)
2. Runs REAL D-Wave QPU using hierarchical decomposition with DWaveCliqueSampler
3. Stores FULL solution details (variable assignments, areas, violations)

The quantum solver uses:
- Spatial decomposition (clusters of farms)
- BinaryQuadraticModel per cluster
- DWaveCliqueSampler for QPU solving
- Boundary coordination between clusters
- 6 food families (not 27 crops) for QPU efficiency

Author: OQI-UC002-DWave
Date: 2025-12-24
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader_utils import load_food_data_as_dict

print("="*80)
print("COMPREHENSIVE SCALING TEST: Full Solution Details")
print("="*80)
print()

# Configuration
N_PERIODS = 3
GUROBI_TIMEOUT = 100  # Same as test_gurobi_timeout.py

# ALL SCENARIOS from test_gurobi_timeout.py (up to 200 farms = 16,200 vars)
# Skip scenarios we already have from previous tests
SCENARIOS = [
    # Already solved in previous tests (skip these):
    # {'name': 'rotation_micro_25', 'n_farms': 5, 'n_foods': 6, 'n_vars': 90},
    # {'name': 'rotation_small_50', 'n_farms': 10, 'n_foods': 6, 'n_vars': 180},
    # {'name': 'rotation_medium_100', 'n_farms': 20, 'n_foods': 6, 'n_vars': 360},
    # {'name': 'rotation_50farms_6foods', 'n_farms': 50, 'n_foods': 6, 'n_vars': 900},
    
    # New scenarios to test:
    {'name': 'rotation_15farms_6foods', 'n_farms': 15, 'n_foods': 6, 'n_vars': 270},
    {'name': 'rotation_25farms_6foods', 'n_farms': 25, 'n_foods': 6, 'n_vars': 450},
    {'name': 'rotation_large_200', 'n_farms': 40, 'n_foods': 6, 'n_vars': 720},
    {'name': 'rotation_75farms_6foods', 'n_farms': 75, 'n_foods': 6, 'n_vars': 1350},
    {'name': 'rotation_100farms_6foods', 'n_farms': 100, 'n_foods': 6, 'n_vars': 1800},
    {'name': 'rotation_25farms_27foods', 'n_farms': 25, 'n_foods': 27, 'n_vars': 2025},
    {'name': 'rotation_150farms_6foods', 'n_farms': 150, 'n_foods': 6, 'n_vars': 2700},
    {'name': 'rotation_50farms_27foods', 'n_farms': 50, 'n_foods': 27, 'n_vars': 4050},
    {'name': 'rotation_75farms_27foods', 'n_farms': 75, 'n_foods': 27, 'n_vars': 6075},
    {'name': 'rotation_100farms_27foods', 'n_farms': 100, 'n_foods': 27, 'n_vars': 8100},
    {'name': 'rotation_150farms_27foods', 'n_farms': 150, 'n_foods': 27, 'n_vars': 12150},
    {'name': 'rotation_200farms_27foods', 'n_farms': 200, 'n_foods': 27, 'n_vars': 16200},
]

OUTPUT_DIR = Path(__file__).parent / 'comprehensive_full_results'
OUTPUT_DIR.mkdir(exist_ok=True)

# D-Wave setup
DEFAULT_DWAVE_TOKEN = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'
os.environ['DWAVE_API_TOKEN'] = os.environ.get('DWAVE_API_TOKEN', DEFAULT_DWAVE_TOKEN)

print(f"Testing {len(SCENARIOS)} scenarios with full solution extraction")
print(f"Gurobi timeout: {GUROBI_TIMEOUT}s")
print()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_scenario_data(scenario: Dict) -> Dict:
    """Load data for scenario - matches test_gurobi_timeout.py logic."""
    n_farms = scenario['n_farms']
    n_foods = scenario['n_foods']
    
    # Map scenario parameters to actual scenario names
    if n_foods == 6:
        if n_farms <= 5:
            scenario_name = 'rotation_micro_25'
        elif n_farms <= 10:
            scenario_name = 'rotation_small_50'
        elif n_farms <= 20:
            scenario_name = 'rotation_medium_100'
        elif n_farms <= 40:
            scenario_name = 'rotation_large_200'
        else:
            scenario_name = 'rotation_large_200'
    else:  # 27 foods
        if n_farms <= 250:
            scenario_name = 'rotation_250farms_27foods'
        elif n_farms <= 350:
            scenario_name = 'rotation_350farms_27foods'
        elif n_farms <= 500:
            scenario_name = 'rotation_500farms_27foods'
        else:
            scenario_name = 'rotation_1000farms_27foods'
    
    data = load_food_data_as_dict(scenario_name)
    
    # Adjust farm count
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
    
    # Adjust food count
    if len(data['food_names']) > n_foods:
        data['food_names'] = data['food_names'][:n_foods]
        data['food_benefits'] = {k: v for k, v in list(data['food_benefits'].items())[:n_foods]}
    
    data['total_area'] = sum(data['land_availability'].values())
    
    return data

# ============================================================================
# GUROBI SOLVER WITH FULL SOLUTION EXTRACTION
# ============================================================================

def solve_gurobi_full(data: Dict, scenario: Dict, timeout: int = 100) -> Dict:
    """
    Solve with Gurobi and extract FULL solution details.
    Matches test_gurobi_timeout.py formulation EXACTLY.
    """
    import gurobipy as gp
    from gurobipy import GRB
    
    n_farms = scenario['n_farms']
    n_foods = scenario['n_foods']
    n_periods = N_PERIODS
    
    farm_names = data['farm_names']
    food_names = data['food_names'][:n_foods]
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    
    # CRITICAL: Use same problem complexity parameters as test_gurobi_timeout.py
    rotation_gamma = 0.2
    one_hot_penalty = 3.0
    diversity_bonus = 0.15
    k_neighbors = 4
    frustration_ratio = 0.7
    negative_strength = -0.8
    
    # Build rotation synergy matrix (MIQP)
    rng = np.random.RandomState(42)
    R = np.zeros((n_foods, n_foods))
    for i in range(n_foods):
        for j in range(n_foods):
            if i == j:
                R[i, j] = negative_strength * 1.5
            elif rng.random() < frustration_ratio:
                R[i, j] = rng.uniform(negative_strength * 1.2, negative_strength * 0.3)
            else:
                R[i, j] = rng.uniform(0.02, 0.20)
    
    # Create spatial neighbor graph
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {}
    for i, farm in enumerate(farm_names):
        row, col = i // side, i % side
        positions[farm] = (row, col)
    
    neighbor_edges = []
    for f1_idx, f1 in enumerate(farm_names):
        distances = []
        for f2_idx, f2 in enumerate(farm_names):
            if f1 != f2:
                dist = np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                             (positions[f1][1] - positions[f2][1])**2)
                distances.append((dist, f2_idx, f2))
        distances.sort()
        for _, f2_idx, f2 in distances[:k_neighbors]:
            if (f1_idx, f2_idx) not in neighbor_edges and (f2_idx, f1_idx) not in neighbor_edges:
                neighbor_edges.append((f1_idx, f2_idx))
    
    start_time = time.time()
    
    model = gp.Model("comprehensive_test")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    model.setParam('MIPGap', 0.01)
    model.setParam('MIPFocus', 1)
    model.setParam('ImproveStartTime', 30)
    model.setParam('Threads', 0)
    model.setParam('Presolve', 2)
    model.setParam('Cuts', 2)
    
    # Variables
    Y = {}
    for i, farm in enumerate(farm_names):
        for j, food in enumerate(food_names):
            for t in range(1, n_periods + 1):
                Y[(i, j, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_f{i}_c{j}_t{t}")
    
    # Objective
    obj = 0
    
    # Part 1: Base benefit
    for i, farm in enumerate(farm_names):
        farm_area = land_availability[farm]
        for j, food in enumerate(food_names):
            benefit = food_benefits.get(food, 0.5)
            for t in range(1, n_periods + 1):
                obj += (benefit * farm_area * Y[(i, j, t)]) / total_area
    
    # Part 2: Rotation synergies (QUADRATIC)
    for i in range(n_farms):
        farm_area = land_availability[farm_names[i]]
        for t in range(2, n_periods + 1):
            for j1 in range(n_foods):
                for j2 in range(n_foods):
                    synergy = R[j1, j2]
                    if abs(synergy) > 1e-6:
                        obj += (rotation_gamma * synergy * farm_area * 
                               Y[(i, j1, t-1)] * Y[(i, j2, t)]) / total_area
    
    # Part 3: Spatial interactions (QUADRATIC)
    spatial_gamma = rotation_gamma * 0.5
    for (f1_idx, f2_idx) in neighbor_edges:
        for t in range(1, n_periods + 1):
            for j1 in range(n_foods):
                for j2 in range(n_foods):
                    spatial_synergy = R[j1, j2] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        obj += (spatial_gamma * spatial_synergy * 
                               Y[(f1_idx, j1, t)] * Y[(f2_idx, j2, t)]) / total_area
    
    # Part 4: Soft one-hot penalty (QUADRATIC)
    for i in range(n_farms):
        for t in range(1, n_periods + 1):
            crop_count = gp.quicksum(Y[(i, j, t)] for j in range(n_foods))
            obj -= one_hot_penalty * (crop_count - 1) * (crop_count - 1)
    
    # Part 5: Diversity bonus
    for i in range(n_farms):
        for j in range(n_foods):
            crop_used = gp.quicksum(Y[(i, j, t)] for t in range(1, n_periods + 1))
            obj += diversity_bonus * crop_used
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints
    for i in range(n_farms):
        for t in range(1, n_periods + 1):
            model.addConstr(gp.quicksum(Y[(i, j, t)] for j in range(n_foods)) <= 2)
            model.addConstr(gp.quicksum(Y[(i, j, t)] for j in range(n_foods)) >= 1)
    
    for i in range(n_farms):
        for j in range(n_foods):
            for t in range(1, n_periods):
                model.addConstr(Y[(i, j, t)] + Y[(i, j, t + 1)] <= 1)
    
    model.optimize()
    solve_time = time.time() - start_time
    
    result = {
        'method': 'gurobi',
        'success': False,
        'objective': 0,
        'solve_time': solve_time,
        'mip_gap': None,
        'status': 'unknown',
        'violations': {'max_crops': 0, 'rotation': 0, 'total': 0},
        'solution_binary': {},
        'solution_areas': {},
        'n_assigned': 0,
        'crop_distribution': {},
    }
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        result['success'] = True
        result['objective'] = model.ObjVal
        result['mip_gap'] = model.MIPGap * 100
        result['status'] = 'optimal' if model.Status == GRB.OPTIMAL else 'timeout'
        
        # Extract FULL solution
        crop_counts = {food: 0 for food in food_names}
        violations_max_crops = 0
        violations_rotation = 0
        
        for i, farm in enumerate(farm_names):
            farm_area = land_availability[farm]
            for t in range(1, n_periods + 1):
                crops_in_period = 0
                for j, food in enumerate(food_names):
                    val = Y[(i, j, t)].X
                    if val > 0.5:
                        key = f"{farm}_{food}_t{t}"
                        result['solution_binary'][key] = 1
                        result['solution_areas'][key] = farm_area
                        crop_counts[food] += 1
                        crops_in_period += 1
                        
                        # Check rotation constraint
                        if t > 1:
                            prev_val = Y[(i, j, t-1)].X
                            if prev_val > 0.5:
                                violations_rotation += 1
                
                if crops_in_period > 2:
                    violations_max_crops += (crops_in_period - 2)
        
        result['n_assigned'] = len(result['solution_binary'])
        result['crop_distribution'] = crop_counts
        result['violations'] = {
            'max_crops': violations_max_crops,
            'rotation': violations_rotation,
            'total': violations_max_crops + violations_rotation
        }
    
    return result

# ============================================================================
# QUANTUM SOLVER (REAL QPU - Hierarchical Decomposition with DWaveCliqueSampler)
# ============================================================================

# Import the hierarchical solver
from hierarchical_quantum_solver import solve_hierarchical, DEFAULT_CONFIG

def solve_quantum_qpu(data: Dict, scenario: Dict) -> Dict:
    """
    Solve using REAL D-Wave QPU with hierarchical decomposition.
    Uses DWaveCliqueSampler with spatial decomposition and boundary coordination.
    
    This is the REAL quantum solver, not simulation!
    """
    n_farms = scenario['n_farms']
    n_foods = scenario['n_foods']
    n_vars = scenario['n_vars']
    
    farm_names = data['farm_names']
    food_names = data['food_names'][:n_foods]
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    
    # Configure hierarchical solver
    config = DEFAULT_CONFIG.copy()
    config['n_periods'] = N_PERIODS
    config['num_reads'] = 100  # QPU reads per cluster
    config['num_iterations'] = 3  # Boundary coordination iterations
    config['farms_per_cluster'] = min(10, max(5, n_farms // 5))  # Adaptive cluster size
    
    try:
        # Call the hierarchical quantum solver with USE_QPU=True
        result = solve_hierarchical(
            data=data,
            config=config,
            use_qpu=True,  # USE REAL QPU!
            verbose=False
        )
        
        # Extract results
        family_solution = result.get('family_solution', {})
        timings = result.get('timings', {})
        
        # Get timing info - detailed breakdown
        total_time = timings.get('total', 0)
        # qpu_access_time: Total time QPU was accessed (includes programming, sampling, readout, network)
        qpu_access_time = result.get('qpu_time', timings.get('level2_qpu', 0))
        # qpu_sampling_time: Actual quantum annealing time (the "real" QPU compute)
        qpu_sampling_time = result.get('qpu_sampling_time', 0)
        # qpu_programming_time: Time to program the QPU
        qpu_programming_time = result.get('qpu_programming_time', 0)
        
        # Convert family solution to crop distribution
        crop_counts = {}
        for food in food_names:
            crop_counts[food] = 0
        
        solution_binary = {}
        n_assigned = 0
        
        for key, val in family_solution.items():
            if val == 1:
                farm, family, period = key
                # Map to variable name
                var_name = f"{farm}_{family}_t{period}"
                solution_binary[var_name] = 1
                n_assigned += 1
                if family in crop_counts:
                    crop_counts[family] += 1
        
        # Get violations
        violations = result.get('violations', 0)
        if isinstance(violations, dict):
            total_violations = violations.get('total', 0)
        else:
            total_violations = violations
        
        # Get objective
        obj = result.get('objective', result.get('family_objective', 0))
        
        return {
            'method': 'quantum_qpu',
            'success': True,
            'objective': obj,
            'solve_time': total_time,
            'qpu_access_time': qpu_access_time,  # Total QPU access (network + programming + sampling + readout)
            'qpu_sampling_time': qpu_sampling_time,  # Actual quantum annealing time (~ms)
            'qpu_programming_time': qpu_programming_time,  # QPU programming time
            'violations': {'max_crops': 0, 'rotation': 0, 'total': total_violations},
            'solution_binary': solution_binary,
            'solution_areas': {},
            'n_assigned': n_assigned,
            'crop_distribution': crop_counts,
            'hierarchical_result': result,  # Full result for analysis
        }
        
    except Exception as e:
        print(f"    ⚠️ QPU Error: {e}")
        print(f"    Falling back to SimulatedAnnealing...")
        
        # Fallback to SA if QPU fails
        result = solve_hierarchical(
            data=data,
            config=config,
            use_qpu=False,  # SimulatedAnnealing fallback
            verbose=False
        )
        
        family_solution = result.get('family_solution', {})
        timings = result.get('timings', {})
        total_time = timings.get('total', 0)
        
        crop_counts = {food: 0 for food in food_names}
        solution_binary = {}
        n_assigned = 0
        
        for key, val in family_solution.items():
            if val == 1:
                farm, family, period = key
                var_name = f"{farm}_{family}_t{period}"
                solution_binary[var_name] = 1
                n_assigned += 1
                if family in crop_counts:
                    crop_counts[family] += 1
        
        violations = result.get('violations', 0)
        obj = result.get('objective', result.get('family_objective', 0))
        
        return {
            'method': 'quantum_sa_fallback',
            'success': True,
            'objective': obj,
            'solve_time': total_time,
            'qpu_access_time': 0,
            'qpu_sampling_time': 0,
            'qpu_programming_time': 0,
            'violations': {'max_crops': 0, 'rotation': 0, 'total': violations if isinstance(violations, int) else 0},
            'solution_binary': solution_binary,
            'solution_areas': {},
            'n_assigned': n_assigned,
            'crop_distribution': crop_counts,
        }

# ============================================================================
# RUN TESTS
# ============================================================================

print("="*80)
print("RUNNING COMPREHENSIVE TESTS")
print("="*80)

all_results = []
all_solutions = []

for idx, scenario in enumerate(SCENARIOS, 1):
    print(f"\n{'='*80}")
    print(f"SCENARIO {idx}/{len(SCENARIOS)}: {scenario['name']}")
    print(f"  Size: {scenario['n_farms']} farms × {scenario['n_foods']} foods = {scenario['n_vars']} vars")
    print(f"{'='*80}")
    
    # Load data
    data = load_scenario_data(scenario)
    
    # Gurobi
    print(f"  Running Gurobi ({GUROBI_TIMEOUT}s timeout)...")
    gurobi_result = solve_gurobi_full(data, scenario, GUROBI_TIMEOUT)
    print(f"    Status: {gurobi_result['status']}, Obj: {gurobi_result['objective']:.4f}, "
          f"Time: {gurobi_result['solve_time']:.1f}s, Violations: {gurobi_result['violations']['total']}")
    
    # Quantum (REAL QPU with hierarchical decomposition!)
    print(f"  Running Quantum (REAL QPU - DWaveCliqueSampler)...")
    quantum_result = solve_quantum_qpu(data, scenario)
    qpu_access = quantum_result.get('qpu_access_time', 0)
    qpu_sampling = quantum_result.get('qpu_sampling_time', 0)
    print(f"    Obj: {quantum_result['objective']:.4f}, Total: {quantum_result['solve_time']:.1f}s, "
          f"QPU access: {qpu_access:.3f}s, QPU sampling: {qpu_sampling*1000:.2f}ms, "
          f"Violations: {quantum_result['violations']['total']}")
    
    # Calculate gap and speedup
    if gurobi_result['objective'] != 0:
        gap = (gurobi_result['objective'] - quantum_result['objective']) / abs(gurobi_result['objective']) * 100
    else:
        gap = 0
    
    speedup = gurobi_result['solve_time'] / quantum_result['solve_time'] if quantum_result['solve_time'] > 0 else 0
    
    # Summary record
    all_results.append({
        'scenario': scenario['name'],
        'n_farms': scenario['n_farms'],
        'n_foods': scenario['n_foods'],
        'n_vars': scenario['n_vars'],
        'gurobi_obj': gurobi_result['objective'],
        'gurobi_time': gurobi_result['solve_time'],
        'gurobi_gap': gurobi_result['mip_gap'],
        'gurobi_status': gurobi_result['status'],
        'gurobi_violations': gurobi_result['violations']['total'],
        'gurobi_n_assigned': gurobi_result['n_assigned'],
        'quantum_obj': quantum_result['objective'],
        'quantum_time': quantum_result['solve_time'],
        'qpu_access_time': quantum_result.get('qpu_access_time', 0),
        'qpu_sampling_time': quantum_result.get('qpu_sampling_time', 0),
        'qpu_programming_time': quantum_result.get('qpu_programming_time', 0),
        'quantum_violations': quantum_result['violations']['total'],
        'quantum_n_assigned': quantum_result['n_assigned'],
        'gap': gap,
        'speedup': speedup,
    })
    
    # Full solution record (for detailed analysis)
    all_solutions.append({
        'scenario': scenario['name'],
        'n_vars': scenario['n_vars'],
        'gurobi': {
            'objective': gurobi_result['objective'],
            'solve_time': gurobi_result['solve_time'],
            'status': gurobi_result['status'],
            'violations': gurobi_result['violations'],
            'crop_distribution': gurobi_result['crop_distribution'],
            'n_assigned': gurobi_result['n_assigned'],
            # Store just counts, not full solution (too large)
        },
        'quantum': {
            'objective': quantum_result['objective'],
            'solve_time': quantum_result['solve_time'],
            'qpu_access_time': quantum_result.get('qpu_access_time', 0),
            'qpu_sampling_time': quantum_result.get('qpu_sampling_time', 0),
            'qpu_programming_time': quantum_result.get('qpu_programming_time', 0),
            'violations': quantum_result['violations'],
            'crop_distribution': quantum_result['crop_distribution'],
            'n_assigned': quantum_result['n_assigned'],
        }
    })
    
    print(f"  Gap: {gap:.1f}%, Speedup: {speedup:.1f}×")

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Summary CSV
df = pd.DataFrame(all_results)
csv_file = OUTPUT_DIR / f'comprehensive_results_{timestamp}.csv'
df.to_csv(csv_file, index=False)
print(f"\n✓ Summary CSV saved to: {csv_file}")

# Full JSON with solution details
json_file = OUTPUT_DIR / f'comprehensive_solutions_{timestamp}.json'
with open(json_file, 'w') as f:
    json.dump(all_solutions, f, indent=2)
print(f"✓ Full solutions JSON saved to: {json_file}")

# Print summary table
print("\n" + "="*120)
print("SUMMARY TABLE")
print("="*120)
print(f"{'Scenario':<28} {'Vars':>6} {'G-Obj':>8} {'G-Time':>7} {'G-Viol':>6} {'Q-Obj':>8} {'Q-Time':>7} {'QPU-Acc':>8} {'QPU-Samp':>9} {'Q-Viol':>6} {'Gap%':>6}")
print("-"*120)
for r in all_results:
    qpu_samp_ms = r.get('qpu_sampling_time', 0) * 1000  # Convert to ms
    print(f"{r['scenario']:<28} {r['n_vars']:>6} {r['gurobi_obj']:>8.2f} {r['gurobi_time']:>7.1f} {r['gurobi_violations']:>6} "
          f"{r['quantum_obj']:>8.2f} {r['quantum_time']:>7.1f} {r.get('qpu_access_time', 0):>7.3f}s {qpu_samp_ms:>7.2f}ms {r['quantum_violations']:>6} {r['gap']:>6.1f}")
print("-"*120)

print("\n✓ Comprehensive test complete!")
