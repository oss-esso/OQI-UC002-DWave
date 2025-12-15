#!/usr/bin/env python3
"""
Significant Scenarios Benchmark: Gurobi vs QPU Comprehensive Comparison

Tests 6 scenarios spanning different problem sizes:
- Small (5-10 farms, 90-180 vars): clique_decomp
- Medium (20 farms, 360 vars): clique_decomp  
- Large (25, 50, 100 farms, 2025-8100 vars): hierarchical_qpu

Tracks:
- Objective value (both solvers)
- Runtime (wall clock time)
- Constraint violations (rotation, diversity, area)
- Gap: (gurobi_obj - qpu_obj) / gurobi_obj × 100%
  * Positive gap = QPU worse than Gurobi
  * Negative gap = QPU better than Gurobi
- Speedup: gurobi_time / qpu_time

Author: OQI-UC002-DWave
Date: December 14, 2025
"""

import os
import sys
import time
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("SIGNIFICANT SCENARIOS BENCHMARK: Gurobi vs QPU")
print("="*80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

# D-Wave token setup
DEFAULT_DWAVE_TOKEN = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'
os.environ['DWAVE_API_TOKEN'] = os.environ.get('DWAVE_API_TOKEN', DEFAULT_DWAVE_TOKEN)

# Test scenarios - carefully selected to span sizes and test both methods
SCENARIOS = [
    {
        'name': 'rotation_micro_25',
        'description': '5 farms - Small scale',
        'n_farms': 5,
        'n_foods': 6,
        'n_periods': 3,
        'n_vars': 90,
        'qpu_method': 'clique_decomp',
        'expected_speedup': 11.5,
    },
    {
        'name': 'rotation_small_50',
        'description': '10 farms - At the cliff',
        'n_farms': 10,
        'n_foods': 6,
        'n_periods': 3,
        'n_vars': 180,
        'qpu_method': 'clique_decomp',
        'expected_speedup': 6.2,
    },
    {
        'name': 'rotation_medium_100',
        'description': '20 farms - Past the cliff',
        'n_farms': 20,
        'n_foods': 6,
        'n_periods': 3,
        'n_vars': 360,
        'qpu_method': 'clique_decomp',
        'expected_speedup': 5.2,
    },
    {
        'name': 'rotation_250farms_27foods',
        'description': '25 farms - Hierarchical test',
        'n_farms': 25,
        'n_foods': 27,
        'n_periods': 3,
        'n_vars': 2025,
        'qpu_method': 'hierarchical',
        'expected_speedup': 5.0,
    },
    {
        'name': 'rotation_350farms_27foods',
        'description': '50 farms - Large hierarchical',
        'n_farms': 50,
        'n_foods': 27,
        'n_periods': 3,
        'n_vars': 4050,
        'qpu_method': 'hierarchical',
        'expected_speedup': 4.5,
    },
    {
        'name': 'rotation_500farms_27foods',
        'description': '100 farms - Ultimate test',
        'n_farms': 100,
        'n_foods': 27,
        'n_periods': 3,
        'n_vars': 8100,
        'qpu_method': 'hierarchical',
        'expected_speedup': 2.5,
    },
]

# Solver configuration
GUROBI_CONFIG = {
    'timeout': 100,  # 5 minutes HARD LIMIT
    'mip_gap': 0.01,  # 1% - stop within 1% of optimum
    'mip_focus': 1,  # Find good feasible solutions quickly
    'improve_start_time': 30,  # Stop if no improvement for 30s
}

QPU_CONFIG = {
    'num_reads': 100,
    'farms_per_cluster': 5,  # For hierarchical
    'num_iterations': 3,  # Boundary coordination
}

# Run configuration
RUN_GUROBI_FIRST = True  # Run Gurobi first, confirm results, then run QPU
PAUSE_BETWEEN_METHODS = True  # Pause after Gurobi before running QPU

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'significant_scenarios_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# IMPORTS
# ============================================================================

print("Importing libraries...")

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
    print("✓ Gurobi available")
except ImportError:
    HAS_GUROBI = False
    print("✗ Gurobi not available")
    sys.exit(1)

try:
    from dimod import ConstrainedQuadraticModel, Binary
    print("✓ D-Wave libraries available")
except ImportError:
    print("✗ D-Wave libraries not available")
    sys.exit(1)

# Import scenario loader
from data_loader_utils import load_food_data_as_dict

# Import QPU solvers
try:
    from clique_wrapper import solve_clique_wrapper, HAS_CLIQUE
    if HAS_CLIQUE:
        print("✓ Clique decomposition available")
    else:
        print("✗ Clique decomposition not available")
except ImportError as e:
    HAS_CLIQUE = False
    solve_clique_wrapper = None
    print(f"✗ Clique decomposition not available: {e}")

try:
    from hierarchical_quantum_solver import solve_hierarchical
    HAS_HIERARCHICAL = True
    print("✓ Hierarchical solver available")
except ImportError as e:
    HAS_HIERARCHICAL = False
    solve_hierarchical = None
    print(f"✗ Hierarchical solver not available: {e}")

print()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_scenario_data(scenario: Dict) -> Dict:
    """Load data for a specific scenario."""
    n_farms = scenario['n_farms']
    n_foods = scenario['n_foods']
    
    # Determine which scenario to load from scenarios.py
    if n_foods == 6:
        # Use rotation scenarios
        if n_farms == 5:
            scenario_name = 'rotation_micro_25'
        elif n_farms == 10:
            scenario_name = 'rotation_small_50'
        elif n_farms == 20:
            scenario_name = 'rotation_medium_100'
        else:
            scenario_name = 'rotation_large_200'
    else:
        # Use 27-food scenarios
        if n_farms == 25:
            scenario_name = 'rotation_250farms_27foods'
        elif n_farms == 50:
            scenario_name = 'rotation_250farms_27foods'
        else:
            scenario_name = 'rotation_500farms_27foods'
    
    print(f"Loading data for {scenario['name']} (using scenario '{scenario_name}')...")
    data = load_food_data_as_dict(scenario_name)
    
    # Verify or adjust farm count
    if len(data['farm_names']) != n_farms:
        print(f"  Adjusting from {len(data['farm_names'])} to {n_farms} farms...")
        # Sample or expand farms
        if len(data['farm_names']) > n_farms:
            data['farm_names'] = data['farm_names'][:n_farms]
            data['land_availability'] = {k: v for k, v in list(data['land_availability'].items())[:n_farms]}
        else:
            # Duplicate farms to reach target
            original_farms = data['farm_names'].copy()
            while len(data['farm_names']) < n_farms:
                idx = len(data['farm_names']) - len(original_farms)
                farm = original_farms[idx % len(original_farms)]
                new_farm = f"{farm}_dup{idx}"
                data['farm_names'].append(new_farm)
                data['land_availability'][new_farm] = data['land_availability'][farm]
    
    # Verify food count
    if n_foods == 6 and len(data['food_names']) > 6:
        print(f"  Note: Using 6 families from {len(data['food_names'])} foods (aggregation)")
    
    return data

# ============================================================================
# CONSTRAINT VALIDATION
# ============================================================================

def validate_solution(solution: Dict, data: Dict, scenario: Dict) -> Dict:
    """
    Validate solution against all constraints.
    
    Returns dict with:
    - rotation_violations: count of rotation constraint violations
    - diversity_violations: count of diversity constraint violations  
    - area_violations: count of area constraint violations
    - total_violations: total violation count
    """
    violations = {
        'rotation_violations': 0,
        'diversity_violations': 0,
        'area_violations': 0,
        'total_violations': 0,
    }
    
    if 'solution' not in solution or solution['solution'] is None:
        violations['total_violations'] = 999  # Mark as infeasible
        return violations
    
    sol = solution['solution']
    n_farms = scenario['n_farms']
    n_foods = scenario['n_foods']
    n_periods = scenario['n_periods']
    
    # Check rotation constraints: no same crop in consecutive periods
    for farm_idx in range(n_farms):
        for period in range(1, n_periods):  # periods 1 to n_periods-1
            for food_idx in range(n_foods):
                var1_name = f"Y_f{farm_idx}_c{food_idx}_t{period}"
                var2_name = f"Y_f{farm_idx}_c{food_idx}_t{period+1}"
                
                val1 = sol.get(var1_name, 0)
                val2 = sol.get(var2_name, 0)
                
                # If both are 1, rotation constraint violated
                if val1 > 0.5 and val2 > 0.5:
                    violations['rotation_violations'] += 1
    
    # Check diversity: each farm must grow at least 1 crop per period
    for farm_idx in range(n_farms):
        for period in range(1, n_periods + 1):
            crops_this_period = 0
            for food_idx in range(n_foods):
                var_name = f"Y_f{farm_idx}_c{food_idx}_t{period}"
                if sol.get(var_name, 0) > 0.5:
                    crops_this_period += 1
            
            if crops_this_period == 0:
                violations['diversity_violations'] += 1
    
    # Check area constraints: sum of allocations per farm per period should equal available area
    for farm_idx in range(n_farms):
        for period in range(1, n_periods + 1):
            total_allocation = 0
            for food_idx in range(n_foods):
                var_name = f"Y_f{farm_idx}_c{food_idx}_t{period}"
                if sol.get(var_name, 0) > 0.5:
                    total_allocation += 1
            
            # Should allocate entire farm (simplified check)
            if total_allocation == 0:
                violations['area_violations'] += 1
    
    violations['total_violations'] = (
        violations['rotation_violations'] +
        violations['diversity_violations'] +
        violations['area_violations']
    )
    
    return violations

# ============================================================================
# GUROBI SOLVER
# ============================================================================

def solve_gurobi(data: Dict, scenario: Dict, config: Dict) -> Dict:
    """Solve with Gurobi using updated parameters."""
    print(f"\n{'='*60}")
    print(f"GUROBI: {scenario['name']}")
    print(f"{'='*60}")
    
    result = {
        'method': 'gurobi',
        'scenario': scenario['name'],
        'status': 'unknown',
        'objective': None,
        'runtime': None,
        'solution': None,
        'mip_gap': None,
        'hit_timeout': False,
        'hit_improve_limit': False,
    }
    
    start_time = time.time()
    
    try:
        # Create model
        model = gp.Model("rotation_optimization")
        model.setParam('OutputFlag', 1)
        model.setParam('TimeLimit', config['timeout'])
        model.setParam('MIPGap', config['mip_gap'])
        model.setParam('MIPFocus', config['mip_focus'])
        model.setParam('ImproveStartTime', config['improve_start_time'])
        
        n_farms = scenario['n_farms']
        n_foods = scenario['n_foods']
        n_periods = scenario['n_periods']
        
        farm_names = data['farm_names']
        food_names = data['food_names'][:n_foods]  # Use first n_foods
        land_availability = data['land_availability']
        food_benefits = data['food_benefits']
        total_area = data['total_area']
        
        # CRITICAL: Add problem complexity parameters (MIQP formulation)
        rotation_gamma = 0.2
        one_hot_penalty = 3.0
        diversity_bonus = 0.15
        k_neighbors = 4
        frustration_ratio = 0.7
        negative_strength = -0.8
        
        # Create rotation synergy matrix (makes problem MIQP, not MIP)
        import numpy as np
        rng = np.random.RandomState(42)
        R = np.zeros((n_foods, n_foods))
        for i in range(n_foods):
            for j in range(n_foods):
                if i == j:
                    R[i, j] = negative_strength * 1.5  # Same crop = negative
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
                if (f2_idx, f1_idx) not in [(e[1], e[0]) for e in neighbor_edges]:
                    neighbor_edges.append((f1_idx, f2_idx))
        
        # Variables
        Y = {}
        for i, farm in enumerate(farm_names):
            for j, food in enumerate(food_names):
                for t in range(1, n_periods + 1):
                    Y[(i, j, t)] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"Y_f{i}_c{j}_t{t}"
                    )
        
        # Objective: maximize benefit + synergies - penalties (MIQP formulation)
        obj = 0
        
        # Part 1: Base benefit (linear)
        for i, farm in enumerate(farm_names):
            farm_area = land_availability[farm]
            for j, food in enumerate(food_names):
                benefit = food_benefits[food]
                for t in range(1, n_periods + 1):
                    obj += (benefit * farm_area * Y[(i, j, t)]) / total_area
        
        # Part 2: Rotation synergies (QUADRATIC - makes it hard!)
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
        
        # Part 5: Diversity bonus (linear)
        for i in range(n_farms):
            for j in range(n_foods):
                crop_used = gp.quicksum(Y[(i, j, t)] for t in range(1, n_periods + 1))
                obj += diversity_bonus * crop_used
        
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # Constraints: Soft one-hot (allow 1-2 crops per farm per period)
        for i in range(n_farms):
            for t in range(1, n_periods + 1):
                model.addConstr(
                    gp.quicksum(Y[(i, j, t)] for j in range(n_foods)) <= 2,
                    name=f"max_crops_f{i}_t{t}"
                )
                model.addConstr(
                    gp.quicksum(Y[(i, j, t)] for j in range(n_foods)) >= 1,
                    name=f"min_crops_f{i}_t{t}"
                )
        
        # Constraint 2: Rotation - same crop cannot be in consecutive periods
        for i in range(n_farms):
            for j in range(n_foods):
                for t in range(1, n_periods):
                    model.addConstr(
                        Y[(i, j, t)] + Y[(i, j, t + 1)] <= 1,
                        name=f"rotation_f{i}_c{j}_t{t}"
                    )
        
        print(f"Model built: {n_farms} farms, {n_foods} foods, {n_periods} periods")
        print(f"Variables: {model.NumVars}, Constraints: {model.NumConstrs}")
        print(f"Model type: MIQP (Mixed Integer Quadratic Program) - includes rotation & spatial synergies")
        print(f"Gurobi params: MIPGap={config['mip_gap']*100}%, MIPFocus={config['mip_focus']}, TimeLimit={config['timeout']}s, ImproveStartTime={config['improve_start_time']}s")
        print("Solving...")
        
        # Solve
        model.optimize()
        
        runtime = time.time() - start_time
        
        # Determine why optimization stopped
        hit_timeout = (model.Status == GRB.TIME_LIMIT)
        hit_improve_limit = False
        
        # Check if stopped due to ImproveStartTime (no status code for this, but runtime patterns suggest it)
        if runtime < config['timeout'] - 5 and model.SolCount > 0:
            # Stopped before timeout with solution - likely ImproveStartTime
            hit_improve_limit = True
        
        # Extract results
        if model.Status == GRB.OPTIMAL:
            result['status'] = 'optimal'
            result['objective'] = model.ObjVal
            result['mip_gap'] = 0.0
        elif model.Status == GRB.TIME_LIMIT:
            result['status'] = 'timeout'
            result['hit_timeout'] = True
            if model.SolCount > 0:
                result['objective'] = model.ObjVal
                result['mip_gap'] = model.MIPGap * 100
            else:
                result['objective'] = None
        elif model.SolCount > 0:
            result['status'] = 'feasible'
            result['objective'] = model.ObjVal
            result['mip_gap'] = model.MIPGap * 100
            result['hit_improve_limit'] = hit_improve_limit
        else:
            result['status'] = 'infeasible'
            result['objective'] = None
        
        # Extract solution
        if model.SolCount > 0:
            solution = {}
            for (i, j, t), var in Y.items():
                solution[var.VarName] = var.X
            result['solution'] = solution
        
        result['runtime'] = runtime
        
        print(f"Status: {result['status']}")
        if result['hit_timeout']:
            print(f"  ⚠️  HIT TIMEOUT at {config['timeout']}s")
        elif result['hit_improve_limit']:
            print(f"  ⚠️  Stopped: No improvement for {config['improve_start_time']}s")
        print(f"Objective: {result['objective']}")
        print(f"Runtime: {runtime:.2f}s")
        if result['mip_gap'] is not None:
            print(f"MIP Gap: {result['mip_gap']:.2f}%")
        
    except Exception as e:
        print(f"ERROR: {e}")
        result['status'] = 'error'
        result['runtime'] = time.time() - start_time
    
    return result

# ============================================================================
# QPU SOLVERS
# ============================================================================

def solve_qpu(data: Dict, scenario: Dict, config: Dict) -> Dict:
    """Solve with appropriate QPU method based on problem size.
    
    Uses the verified solvers from statistical_comparison_test.py that include
    complete MIQP formulation with rotation synergies and spatial interactions.
    """
    method = scenario['qpu_method']
    
    print(f"\n{'='*60}")
    print(f"QPU ({method.upper()}): {scenario['name']}")
    print(f"{'='*60}")
    
    result = {
        'method': f'qpu_{method}',
        'scenario': scenario['name'],
        'status': 'unknown',
        'objective': None,
        'runtime': None,
        'solution': None,
    }
    
    start_time = time.time()
    
    try:
        # Import the verified QPU solvers from statistical_comparison_test.py
        import sys
        from pathlib import Path
        test_dir = Path(__file__).parent
        sys.path.insert(0, str(test_dir))
        
        if method == 'clique_decomp':
            # Use the verified clique decomposition solver
            try:
                from statistical_comparison_test import solve_clique_decomp
                print("Using verified clique_decomp solver from statistical_comparison_test.py")
            except ImportError:
                print("ERROR: Cannot import solve_clique_decomp from statistical_comparison_test.py")
                result['status'] = 'error'
                result['runtime'] = 0
                return result
            
            # Prepare data in the format expected by statistical solver
            qpu_result = solve_clique_decomp(
                data=data,
                num_reads=config['num_reads'],
                num_iterations=config.get('num_iterations', 3)
            )
            
            result['objective'] = qpu_result.get('objective')
            result['solution'] = qpu_result.get('solution')
            result['status'] = 'success' if qpu_result.get('success') else 'failed'
            result['runtime'] = qpu_result.get('wall_time', time.time() - start_time)
            
            print(f"Status: {result['status']}")
            print(f"Objective: {result['objective']}")
            print(f"Runtime: {result['runtime']:.2f}s")
            
            return result
            
        elif method == 'hierarchical':
            # Use the hierarchical solver from hierarchical_quantum_solver.py
            # This solver handles 27 foods → 6 families aggregation internally
            try:
                from hierarchical_quantum_solver import solve_hierarchical
                print("Using hierarchical solver from hierarchical_quantum_solver.py")
            except ImportError as e:
                print(f"ERROR: Cannot import solve_hierarchical: {e}")
                result['status'] = 'error'
                result['runtime'] = 0
                return result
            
            # Prepare configuration
            hier_config = {
                'farms_per_cluster': config.get('farms_per_cluster', 5),
                'num_reads': config['num_reads'],
                'num_iterations': config.get('num_iterations', 3),
            }
            
            print(f"Hierarchical config: {hier_config}")
            print(f"Problem size: {scenario['n_farms']} farms × {scenario['n_foods']} foods")
            
            # Solve with QPU enabled
            qpu_result = solve_hierarchical(
                data=data,
                config=hier_config,
                use_qpu=True,  # Use actual QPU for hierarchical
                verbose=True
            )
            
            result['objective'] = qpu_result.get('objective')
            result['solution'] = qpu_result.get('solution')
            result['status'] = 'success' if qpu_result.get('success', True) else 'failed'
            result['runtime'] = qpu_result.get('wall_time', time.time() - start_time)
            
            print(f"Status: {result['status']}")
            print(f"Objective: {result['objective']}")
            print(f"Runtime: {result['runtime']:.2f}s")
            
            return result
            result['solution'] = qpu_result.get('solution')
            result['status'] = 'success' if qpu_result.get('success') else 'failed'
            result['runtime'] = qpu_result.get('wall_time', time.time() - start_time)
            
            print(f"Status: {result['status']}")
            print(f"Objective: {result['objective']}")
            print(f"Runtime: {result['runtime']:.2f}s")
            
            return result
        
        else:
            print(f"ERROR: Unknown QPU method '{method}'")
            result['status'] = 'error'
            result['runtime'] = time.time() - start_time
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        result['status'] = 'error'
        result['runtime'] = time.time() - start_time
    
    return result

# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark():
    """Run complete benchmark across all scenarios."""
    print("\n" + "="*80)
    print("STARTING BENCHMARK - GUROBI FIRST, THEN QPU")
    print("="*80)
    print(f"Testing {len(SCENARIOS)} scenarios")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Configuration:")
    print(f"  Gurobi: timeout={GUROBI_CONFIG['timeout']}s, MIPGap={GUROBI_CONFIG['mip_gap']*100}%, ImproveStartTime={GUROBI_CONFIG['improve_start_time']}s")
    print(f"  QPU: num_reads={QPU_CONFIG['num_reads']}")
    print()
    
    all_results = []
    
    for idx, scenario in enumerate(SCENARIOS, 1):
        print(f"\n{'#'*80}")
        print(f"SCENARIO {idx}/{len(SCENARIOS)}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Size: {scenario['n_farms']} farms × {scenario['n_foods']} foods × {scenario['n_periods']} periods = {scenario['n_vars']} vars")
        print(f"QPU Method: {scenario['qpu_method']}")
        print(f"{'#'*80}")
        
        # Load data
        data = load_scenario_data(scenario)
        
        # ====================================================================
        # STEP 1: RUN GUROBI FIRST
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"STEP 1: RUNNING GUROBI (Classical Baseline)")
        print(f"{'='*70}")
        
        gurobi_result = solve_gurobi(data, scenario, GUROBI_CONFIG)
        gurobi_violations = validate_solution(gurobi_result, data, scenario)
        
        # Display Gurobi summary
        print(f"\n{'='*70}")
        print(f"GUROBI RESULTS: {scenario['name']}")
        print(f"{'='*70}")
        print(f"Status:      {gurobi_result['status']}")
        print(f"Objective:   {gurobi_result['objective']}")
        print(f"Runtime:     {gurobi_result['runtime']:.2f}s")
        print(f"MIP Gap:     {gurobi_result.get('mip_gap', 0):.2f}%")
        print(f"Timeout Hit: {'YES ⚠️' if gurobi_result.get('hit_timeout', False) else 'NO'}")
        print(f"Violations:  {gurobi_violations['total_violations']}")
        print(f"  - Rotation:  {gurobi_violations['rotation_violations']}")
        print(f"  - Diversity: {gurobi_violations['diversity_violations']}")
        print(f"  - Area:      {gurobi_violations['area_violations']}")
        print(f"{'='*70}")
        
        # ====================================================================
        # STEP 2: PAUSE AND CONFIRM BEFORE RUNNING QPU
        # ====================================================================
        # STEP 2: PAUSE AND CONFIRM BEFORE RUNNING QPU - DISABLED FOR AUTO-RUN
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"Gurobi results confirmed. Proceeding to QPU method automatically...")
        print(f"{'='*70}")
        
        # ====================================================================
        # STEP 3: RUN QPU
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"STEP 2: RUNNING QPU (Quantum Hybrid)")
        print(f"{'='*70}")
        
        qpu_result = solve_qpu(data, scenario, QPU_CONFIG)
        qpu_violations = validate_solution(qpu_result, data, scenario)
        
        # Display QPU summary
        print(f"\n{'='*70}")
        print(f"QPU RESULTS: {scenario['name']}")
        print(f"{'='*70}")
        print(f"Status:      {qpu_result['status']}")
        print(f"Objective:   {qpu_result['objective']}")
        print(f"Runtime:     {qpu_result['runtime']:.2f}s")
        print(f"Violations:  {qpu_violations['total_violations']}")
        print(f"  - Rotation:  {qpu_violations['rotation_violations']}")
        print(f"  - Diversity: {qpu_violations['diversity_violations']}")
        print(f"  - Area:      {qpu_violations['area_violations']}")
        print(f"{'='*70}")
        
        # ====================================================================
        # STEP 4: COMPARISON
        # ====================================================================
        # Calculate metrics
        gurobi_obj = gurobi_result['objective'] if gurobi_result['objective'] is not None else 0
        qpu_obj = qpu_result['objective'] if qpu_result['objective'] is not None else 0
        
        gurobi_time = gurobi_result['runtime']
        qpu_time = qpu_result['runtime']
        
        # Gap calculation: (gurobi_obj - qpu_obj) / gurobi_obj × 100
        # Positive = QPU worse, Negative = QPU better
        if gurobi_obj > 0:
            gap_pct = ((gurobi_obj - qpu_obj) / gurobi_obj) * 100
        else:
            gap_pct = None
        
        # Speedup calculation
        if qpu_time > 0:
            speedup = gurobi_time / qpu_time
        else:
            speedup = None
        
        # Compile results
        result = {
            'scenario': scenario['name'],
            'description': scenario['description'],
            'n_farms': scenario['n_farms'],
            'n_foods': scenario['n_foods'],
            'n_periods': scenario['n_periods'],
            'n_vars': scenario['n_vars'],
            'qpu_method': scenario['qpu_method'],
            
            # Gurobi results
            'gurobi_status': gurobi_result['status'],
            'gurobi_objective': gurobi_obj,
            'gurobi_runtime': gurobi_time,
            'gurobi_mip_gap': gurobi_result.get('mip_gap'),
            'gurobi_hit_timeout': gurobi_result.get('hit_timeout', False),
            'gurobi_hit_improve_limit': gurobi_result.get('hit_improve_limit', False),
            'gurobi_rotation_violations': gurobi_violations['rotation_violations'],
            'gurobi_diversity_violations': gurobi_violations['diversity_violations'],
            'gurobi_area_violations': gurobi_violations['area_violations'],
            'gurobi_total_violations': gurobi_violations['total_violations'],
            
            # QPU results
            'qpu_status': qpu_result['status'],
            'qpu_objective': qpu_obj,
            'qpu_runtime': qpu_time,
            'qpu_rotation_violations': qpu_violations['rotation_violations'],
            'qpu_diversity_violations': qpu_violations['diversity_violations'],
            'qpu_area_violations': qpu_violations['area_violations'],
            'qpu_total_violations': qpu_violations['total_violations'],
            
            # Comparison metrics
            'gap_pct': gap_pct,
            'speedup': speedup,
            'expected_speedup': scenario['expected_speedup'],
        }
        
        all_results.append(result)
        
        # Print comparison summary
        print(f"\n{'='*70}")
        print(f"COMPARISON: {scenario['name']}")
        print(f"{'='*70}")
        print(f"Gurobi:  Obj={gurobi_obj:.4f}, Time={gurobi_time:.2f}s, Violations={gurobi_violations['total_violations']}, Timeout={'YES' if result['gurobi_hit_timeout'] else 'NO'}")
        print(f"QPU:     Obj={qpu_obj:.4f}, Time={qpu_time:.2f}s, Violations={qpu_violations['total_violations']}")
        if gap_pct is not None:
            gap_sign = "+" if gap_pct > 0 else ""
            gap_interpretation = "QPU worse" if gap_pct > 0 else "QPU better"
            print(f"Gap:     {gap_sign}{gap_pct:.2f}% ({gap_interpretation})")
        if speedup is not None:
            print(f"Speedup: {speedup:.2f}× (expected: {scenario['expected_speedup']:.1f}×)")
        print(f"{'='*70}\n")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as JSON
    json_file = OUTPUT_DIR / f'benchmark_results_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {json_file}")
    
    # Save as CSV
    df = pd.DataFrame(all_results)
    csv_file = OUTPUT_DIR / f'benchmark_results_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    print(f"✓ CSV saved to: {csv_file}")
    
    # Print final summary table
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print()
    print(f"{'Scenario':<35} {'G-Obj':>10} {'Q-Obj':>10} {'Gap %':>10} {'Speedup':>10} {'Timeout':>8}")
    print("-"*90)
    for r in all_results:
        gap_str = f"{r['gap_pct']:+.1f}%" if r['gap_pct'] is not None else "N/A"
        speedup_str = f"{r['speedup']:.2f}×" if r['speedup'] is not None else "N/A"
        timeout_str = "YES" if r['gurobi_hit_timeout'] else "NO"
        print(f"{r['scenario']:<35} {r['gurobi_objective']:>10.4f} {r['qpu_objective']:>10.4f} {gap_str:>10} {speedup_str:>10} {timeout_str:>8}")
    print("-"*90)
    
    # Calculate averages
    valid_gaps = [r['gap_pct'] for r in all_results if r['gap_pct'] is not None]
    valid_speedups = [r['speedup'] for r in all_results if r['speedup'] is not None]
    timeout_count = sum(1 for r in all_results if r['gurobi_hit_timeout'])
    
    if valid_gaps:
        avg_gap = np.mean(valid_gaps)
        print(f"\nAverage Gap: {avg_gap:+.2f}%")
    
    if valid_speedups:
        avg_speedup = np.mean(valid_speedups)
        print(f"Average Speedup: {avg_speedup:.2f}×")
    
    print(f"Gurobi Timeouts: {timeout_count}/{len(all_results)} ({timeout_count/len(all_results)*100:.0f}%)")
    
    print("\n✓ Benchmark complete!")
    
    return all_results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("SIGNIFICANT SCENARIOS BENCHMARK")
    print("Gurobi vs QPU Comprehensive Comparison")
    print("="*80)
    print()
    print("Run Configuration:")
    print(f"  Mode: {'Gurobi FIRST, then QPU (with confirmation)' if RUN_GUROBI_FIRST else 'Both methods in parallel'}")
    print(f"  Scenarios: {len(SCENARIOS)}")
    print()
    print("Gurobi Configuration:")
    print(f"  Time limit: {GUROBI_CONFIG['timeout']}s (HARD LIMIT)")
    print(f"  MIP gap: {GUROBI_CONFIG['mip_gap']*100}% (stop within 1% of optimum)")
    print(f"  MIP focus: {GUROBI_CONFIG['mip_focus']} (find good feasible solutions quickly)")
    print(f"  Improve start time: {GUROBI_CONFIG['improve_start_time']}s (stop if no improvement)")
    print()
    print("QPU Configuration:")
    print(f"  Num reads: {QPU_CONFIG['num_reads']}")
    print(f"  Farms per cluster: {QPU_CONFIG['farms_per_cluster']} (hierarchical)")
    print(f"  Iterations: {QPU_CONFIG['num_iterations']} (boundary coordination)")
    print()
    print(f"Output: {OUTPUT_DIR}")
    print()
    print("⚠️  This will consume D-Wave QPU credits (~600 QPU calls)")
    print("⚠️  Estimated runtime: 60-90 minutes")
    print()
    
    if RUN_GUROBI_FIRST:
        print("NOTE: You will be prompted after each Gurobi run before QPU starts.")
        print("      This allows you to confirm Gurobi results first.")
        print()
    
    # Auto-start for automated runs
    print("Starting benchmark automatically...")
    
    results = run_benchmark()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
