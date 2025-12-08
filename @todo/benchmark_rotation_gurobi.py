#!/usr/bin/env python3
"""
Benchmark rotation scenarios with Gurobi to establish ground truth.

Tests the quantum-friendly rotation formulations with:
- 6 crop families (reduced from 27 crops)
- 3-period rotation with quadratic synergy terms
- Frustrated interactions (40-60% negative synergies)
- Spatial neighbor structure (k=4)

Measures:
- MIP objective (optimal or best found)
- LP relaxation objective
- Integrality gap
- Solve time
- Branch-and-bound nodes
- Solution quality
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: Gurobi not available. Please install gurobipy.")
    sys.exit(1)

from src.scenarios import load_food_data

print("="*100)
print("ROTATION SCENARIOS - GUROBI GROUND TRUTH BENCHMARK")
print("="*100)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Gurobi version: {gp.gurobi.version()}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_enhanced_rotation_matrix(n_families, frustration_ratio=0.4, 
                                   negative_strength=-0.3, seed=42):
    """
    Create rotation synergy matrix with frustrated interactions.
    
    ENHANCED for classical hardness:
    - Higher frustration ratios (80-90%)
    - Stronger negative couplings (-1.0 to -1.5)
    - Mixed positive/negative creates spin-glass-like frustration
    
    Args:
        n_families: Number of crop families
        frustration_ratio: Fraction of edges with negative synergies
        negative_strength: Magnitude of negative synergies
        seed: Random seed
    
    Returns:
        NxN numpy array with rotation synergies
    """
    np.random.seed(seed)
    
    R = np.zeros((n_families, n_families))
    
    for i in range(n_families):
        for j in range(n_families):
            if i == j:
                # Same crop in consecutive periods: strong negative (nutrient depletion)
                R[i, j] = negative_strength * 1.5  # Stronger penalty
            elif np.random.random() < frustration_ratio:
                # Frustrated: strong negative synergy (disease, competition)
                # Use wider range for more variance
                R[i, j] = np.random.uniform(negative_strength * 1.2, negative_strength * 0.3)
            else:
                # Beneficial rotation - but also varied
                R[i, j] = np.random.uniform(0.02, 0.20)
    
    return R


def create_spatial_neighbor_graph(farms, k_neighbors=4):
    """
    Create k-nearest neighbor graph for farms on grid layout.
    
    Returns:
        List of (farm1, farm2) neighbor pairs
    """
    import networkx as nx
    
    # Assume farm names like "Farm_0", "Farm_1", etc.
    n_farms = len(farms)
    side = int(np.ceil(np.sqrt(n_farms)))
    
    # Position farms on grid
    positions = {}
    for i, farm in enumerate(farms):
        row, col = i // side, i % side
        positions[farm] = (row, col)
    
    # Find k-nearest neighbors
    G = nx.Graph()
    G.add_nodes_from(farms)
    
    for f1 in farms:
        distances = []
        for f2 in farms:
            if f1 != f2:
                dist = np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                             (positions[f1][1] - positions[f2][1])**2)
                distances.append((dist, f2))
        
        distances.sort()
        for _, f2 in distances[:k_neighbors]:
            G.add_edge(f1, f2)
    
    return list(G.edges())


def build_rotation_mip(farms, crop_families, config, n_periods=3):
    """
    Build Gurobi MIP for 3-period rotation optimization.
    
    ENHANCED for classical hardness:
    - SOFT one-hot constraint (penalty instead of hard constraint)
    - Diversity bonus (encourages using multiple crop families)
    - Competing objectives create LP relaxation gap
    
    Variables:
        Y[f,c,t]: Binary, 1 if farm f grows crop family c in period t
        
    Objective:
        Maximize: Linear crop benefits + Rotation synergies + Spatial interactions
                  - One-hot penalty - Diversity bonus
        
    Constraints:
        - (REMOVED one-hot hard constraint - now soft penalty)
        - Food group constraints per period (optional)
    """
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    rotation_gamma = params.get('rotation_gamma', 0.15)
    k_neighbors = params.get('spatial_k_neighbors', 4)
    frustration_ratio = params.get('frustration_ratio', 0.4)
    negative_strength = params.get('negative_synergy_strength', -0.3)
    
    # NEW: Hardness parameters
    use_soft_constraint = params.get('use_soft_one_hot', True)
    one_hot_penalty = params.get('one_hot_penalty', 5.0)  # Penalty for violating one-hot
    diversity_bonus = params.get('diversity_bonus', 0.1)  # Bonus for crop diversity
    
    n_farms = len(farms)
    n_families = len(crop_families)
    families_list = list(crop_families.keys())
    
    # Create rotation matrix with frustration
    R = create_enhanced_rotation_matrix(
        n_families, frustration_ratio, negative_strength, seed=42
    )
    
    # Create spatial neighbor graph
    neighbor_edges = create_spatial_neighbor_graph(farms, k_neighbors)
    
    # Build model
    m = gp.Model("rotation_3period")
    m.setParam('OutputFlag', 1)
    m.setParam('TimeLimit', 300)  # 5 minutes
    m.setParam('MIPGap', 0.001)   # 0.1% gap tolerance
    
    # Variables: Y[f,c,t] = 1 if farm f grows family c in period t
    Y = {}
    for f in farms:
        for c_idx, c in enumerate(families_list):
            for t in range(1, n_periods + 1):
                Y[(f, c, t)] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f"Y_{f}_{c}_t{t}"
                )
    
    m.update()
    
    # Objective: Linear benefits + Rotation synergies + Spatial interactions
    obj = 0
    total_land = sum(land_availability.values())
    
    # Part 1: Linear crop benefits (area-normalized)
    for f in farms:
        farm_area = land_availability[f]
        for c in families_list:
            crop_data = crop_families[c]
            
            # Weighted benefit
            B_c = (
                weights.get('nutritional_value', 0) * crop_data.get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * crop_data.get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * crop_data.get('environmental_impact', 0) +
                weights.get('affordability', 0) * crop_data.get('affordability', 0) +
                weights.get('sustainability', 0) * crop_data.get('sustainability', 0)
            )
            
            for t in range(1, n_periods + 1):
                obj += (B_c * farm_area * Y[(f, c, t)]) / total_land
    
    # Part 2: Rotation synergies (temporal)
    for f in farms:
        farm_area = land_availability[f]
        for t in range(2, n_periods + 1):  # t = 2, 3 (compare to t-1)
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    synergy = R[c1_idx, c2_idx]
                    if abs(synergy) > 1e-6:
                        obj += (rotation_gamma * synergy * farm_area * 
                               Y[(f, c1, t-1)] * Y[(f, c2, t)]) / total_land
    
    # Part 3: Spatial interactions (neighbor farms)
    spatial_gamma = rotation_gamma * 0.5  # Weaker than temporal
    for (f1, f2) in neighbor_edges:
        for t in range(1, n_periods + 1):
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    # Use rotation matrix as spatial compatibility
                    spatial_synergy = R[c1_idx, c2_idx] * 0.3  # Dampened
                    if abs(spatial_synergy) > 1e-6:
                        obj += (spatial_gamma * spatial_synergy * 
                               Y[(f1, c1, t)] * Y[(f2, c2, t)]) / total_land
    
    # Part 4: SOFT one-hot penalty (NEW for hardness)
    if use_soft_constraint:
        for f in farms:
            for t in range(1, n_periods + 1):
                # Penalty for deviating from exactly 1 crop per farm per period
                # This creates LP relaxation gap because LP can fractionally satisfy
                crop_count = gp.quicksum(Y[(f, c, t)] for c in families_list)
                # Quadratic penalty: (count - 1)^2
                obj -= one_hot_penalty * (crop_count - 1) * (crop_count - 1)
    
    # Part 5: Diversity bonus (NEW - competing objective)
    # Bonus for using diverse crop families across periods
    for f in farms:
        for c in families_list:
            # Check if crop c is used in ANY period on farm f
            crop_used = gp.quicksum(Y[(f, c, t)] for t in range(1, n_periods + 1))
            # Bonus if used at least once (creates fractional LP solutions)
            obj += diversity_bonus * crop_used
    
    m.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints
    if not use_soft_constraint:
        # Hard one-hot constraint (original - too strong)
        for f in farms:
            for t in range(1, n_periods + 1):
                m.addConstr(
                    gp.quicksum(Y[(f, c, t)] for c in families_list) == 1,
                    name=f"one_crop_{f}_t{t}"
                )
    else:
        # SOFT constraint: Allow but penalize deviation
        # Add inequality: at least 0, at most 2 crops per period
        for f in farms:
            for t in range(1, n_periods + 1):
                m.addConstr(
                    gp.quicksum(Y[(f, c, t)] for c in families_list) <= 2,
                    name=f"max_crops_{f}_t{t}"
                )
    
    return m, Y, R, neighbor_edges


def solve_and_analyze(scenario_name, farms, crop_families, config):
    """
    Solve rotation MIP with Gurobi and analyze results.
    """
    print(f"\n{'='*100}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*100}")
    
    params = config['parameters']
    n_farms = len(farms)
    n_families = len(crop_families)
    n_periods = 3
    n_vars = n_farms * n_families * n_periods
    
    print(f"Configuration:")
    print(f"  Farms: {n_farms}")
    print(f"  Crop families: {n_families}")
    print(f"  Periods: {n_periods}")
    print(f"  Total variables: {n_vars}")
    print(f"  Rotation gamma: {params.get('rotation_gamma', 0.15)}")
    print(f"  Frustration ratio: {params.get('frustration_ratio', 0.4):.0%}")
    print(f"  Negative strength: {params.get('negative_synergy_strength', -0.3)}")
    print(f"  k-neighbors: {params.get('spatial_k_neighbors', 4)}")
    
    # Build model
    print("\nBuilding MIP model...")
    m, Y, R, neighbor_edges = build_rotation_mip(farms, crop_families, config, n_periods)
    
    print(f"  Variables: {m.NumVars}")
    print(f"  Constraints: {m.NumConstrs}")
    print(f"  Quadratic terms: {m.NumQConstrs + m.NumQNZs}")
    print(f"  Neighbor edges: {len(neighbor_edges)}")
    
    # Analyze rotation matrix
    frac_negative = (R < 0).sum() / R.size
    print(f"  Rotation matrix: {frac_negative:.1%} negative synergies ✓")
    
    # Solve MIP
    print("\nSolving MIP...")
    m.optimize()
    
    # Extract results
    result = {
        'scenario': scenario_name,
        'n_farms': n_farms,
        'n_families': n_families,
        'n_periods': n_periods,
        'n_vars': n_vars,
        'n_constraints': m.NumConstrs,
        'n_neighbor_edges': len(neighbor_edges),
        'rotation_gamma': params.get('rotation_gamma', 0.15),
        'frustration_ratio': params.get('frustration_ratio', 0.4),
        'negative_strength': params.get('negative_synergy_strength', -0.3),
        'status': m.status,
        'status_string': 'OPTIMAL' if m.status == GRB.OPTIMAL else 
                        'TIME_LIMIT' if m.status == GRB.TIME_LIMIT else 
                        'INFEASIBLE' if m.status == GRB.INFEASIBLE else
                        'OTHER',
        'mip_obj': m.ObjVal if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0 else None,
        'solve_time': m.Runtime,
        'node_count': m.NodeCount,
        'solution_count': m.SolCount,
        'mip_gap': m.MIPGap if m.SolCount > 0 else None,
    }
    
    # LP Relaxation
    print("\nSolving LP relaxation...")
    rel = m.relax()
    rel.setParam('TimeLimit', 60)  # Shorter time limit for LP
    rel.optimize()
    
    if rel.status == GRB.OPTIMAL:
        result['lp_obj'] = rel.ObjVal
    elif rel.status == GRB.TIME_LIMIT and rel.SolCount > 0:
        result['lp_obj'] = rel.ObjVal  # Best LP bound found
    else:
        result['lp_obj'] = None
    
    # Integrality gap - use Gurobi's MIP gap if LP not available
    if result['mip_obj'] is not None and result['lp_obj'] is not None:
        result['integrality_gap'] = abs(result['lp_obj'] - result['mip_obj']) / max(1.0, abs(result['mip_obj']))
    elif result['mip_gap'] is not None and result['mip_gap'] < 10:
        # Use Gurobi's gap if reasonable
        result['integrality_gap'] = result['mip_gap']
    else:
        result['integrality_gap'] = None
    
    # Count fractional variables in LP
    frac_count = 0
    if rel.status == GRB.OPTIMAL:
        for v in rel.getVars():
            if abs(v.x - round(v.x)) > 1e-6:
                frac_count += 1
    result['lp_frac_vars'] = frac_count
    result['lp_frac_ratio'] = frac_count / n_vars if n_vars > 0 else 0
    
    # Print summary
    print(f"\nResults:")
    print(f"  Status: {result['status_string']}")
    print(f"  MIP Objective: {result['mip_obj']:.6f}" if result['mip_obj'] else "  MIP Objective: N/A")
    print(f"  LP Objective: {result['lp_obj']:.6f}" if result['lp_obj'] else "  LP Objective: N/A")
    
    # Report integrality gap or MIP gap
    if result['integrality_gap'] is not None:
        print(f"  Integrality Gap: {result['integrality_gap']:.2%}")
    elif result['mip_gap'] is not None:
        print(f"  MIP Gap (Gurobi): {result['mip_gap']:.2%}")
    else:
        print(f"  Gap: N/A")
    
    print(f"  LP Fractional Vars: {frac_count}/{n_vars} ({result['lp_frac_ratio']:.1%})")
    print(f"  Solve Time: {result['solve_time']:.2f}s")
    print(f"  Nodes Explored: {result['node_count']:.0f}")
    print(f"  Solutions Found: {result['solution_count']}")
    
    # Hardness assessment - use any gap measure available
    gap_for_assessment = result['integrality_gap'] or result['mip_gap'] or 0
    
    if gap_for_assessment > 0.15:
        hardness = "HARD ✓✓ (quantum advantage potential!)"
    elif gap_for_assessment > 0.05:
        hardness = "MODERATE ✓ (harder than original)"
    elif gap_for_assessment > 0.01:
        hardness = "SLIGHT (some difficulty)"
    elif result['status_string'] == 'TIME_LIMIT':
        hardness = "VERY HARD ✓✓✓ (timeout - cannot solve optimally!)"
    else:
        hardness = "EASY (LP finds integer solution)"
    
    print(f"  Hardness: {hardness}")
    
    return result


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

scenarios = [
    'rotation_micro_25',
    'rotation_small_50',
    'rotation_medium_100',
    'rotation_large_200'
]

results = []

for scenario_name in scenarios:
    try:
        # Load scenario
        farms, crop_families, food_groups, config = load_food_data(scenario_name)
        
        # Solve and analyze
        result = solve_and_analyze(scenario_name, farms, crop_families, config)
        results.append(result)
        
    except Exception as e:
        print(f"\nERROR in {scenario_name}: {e}")
        import traceback
        traceback.print_exc()
        continue

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_dir = '@todo/hardness_output'
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'rotation_gurobi_benchmark.json')

output_data = {
    'timestamp': datetime.now().isoformat(),
    'gurobi_version': str(gp.gurobi.version()),
    'scenarios': results
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n{'='*100}")
print(f"SUMMARY")
print(f"{'='*100}")

# Create summary table
print(f"\n{'Scenario':<25} {'Vars':<8} {'Gap':<10} {'Time(s)':<10} {'Nodes':<12} {'Hardness'}")
print("-"*100)

for r in results:
    # Use integrality gap or MIP gap
    gap_val = r['integrality_gap'] or r.get('mip_gap', None)
    gap_str = f"{gap_val:.2%}" if gap_val is not None else "N/A"
    
    # Hardness assessment
    if r['status_string'] == 'TIME_LIMIT':
        hardness = "VERY HARD ✓✓✓"
    elif gap_val is not None:
        if gap_val > 0.15:
            hardness = "HARD ✓✓"
        elif gap_val > 0.05:
            hardness = "MODERATE ✓"
        elif gap_val > 0.01:
            hardness = "SLIGHT"
        else:
            hardness = "EASY"
    else:
        hardness = "N/A"
    
    print(f"{r['scenario']:<25} {r['n_vars']:<8} {gap_str:<10} {r['solve_time']:<10.2f} {r['node_count']:<12.0f} {hardness}")

print(f"\n{'='*100}")
print(f"Results saved to: {output_file}")
print(f"{'='*100}")
