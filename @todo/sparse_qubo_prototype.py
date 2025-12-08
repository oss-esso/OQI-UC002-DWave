#!/usr/bin/env python3
"""
Prototype: Quantum-Friendly Crop Optimization Reformulation

Implements IDEA 3 (MWIS) + IDEA 1 (Spatial Synergies) with controlled sparsity.

Key design principles:
1. Native QUBO structure (no constraint penalties needed)
2. Sparse graph (bounded degree via spatial locality)
3. Frustrated interactions (competing synergies/conflicts)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import networkx as nx
from collections import defaultdict
import json
import os

# D-Wave imports
from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, cqm_to_bqm

# Project imports
from src.scenarios import load_food_data
from Utils import patch_sampler

print("="*100)
print("QUANTUM-FRIENDLY REFORMULATION: SPARSE QUBO WITH SPATIAL SYNERGIES")
print("="*100)

# ============================================================================
# STEP 1: Build spatial adjacency graph for farms
# ============================================================================

def build_farm_adjacency(n_farms, k_neighbors=4, seed=42):
    """
    Build sparse farm adjacency graph.
    Each farm connects to at most k nearest neighbors.
    """
    np.random.seed(seed)
    
    # Place farms on a grid
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {}
    farm_names = []
    
    for i in range(n_farms):
        row, col = i // side, i % side
        farm_name = f"Plot{i+1}"
        positions[farm_name] = (row + np.random.uniform(-0.1, 0.1), 
                                col + np.random.uniform(-0.1, 0.1))
        farm_names.append(farm_name)
    
    # Build k-nearest neighbor graph
    G = nx.Graph()
    G.add_nodes_from(farm_names)
    
    for f1 in farm_names:
        distances = []
        for f2 in farm_names:
            if f1 != f2:
                d = np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                           (positions[f1][1] - positions[f2][1])**2)
                distances.append((d, f2))
        
        distances.sort()
        for _, f2 in distances[:k_neighbors]:
            G.add_edge(f1, f2)
    
    return G, farm_names, positions

# ============================================================================
# STEP 2: Build crop synergy/conflict matrix
# ============================================================================

def build_crop_interactions(foods, food_groups):
    """
    Build crop synergy and conflict matrices.
    
    Synergy: crops from different groups benefit from proximity
    Conflict: crops from same group compete for resources
    """
    food_names = list(foods.keys())
    n_foods = len(food_names)
    
    # Map foods to groups
    food_to_group = {}
    for group, members in food_groups.items():
        for food in members:
            food_to_group[food] = group
    
    synergy = np.zeros((n_foods, n_foods))
    conflict = np.zeros((n_foods, n_foods))
    
    for i, f1 in enumerate(food_names):
        for j, f2 in enumerate(food_names):
            if i >= j:
                continue
            
            g1 = food_to_group.get(f1, 'unknown')
            g2 = food_to_group.get(f2, 'unknown')
            
            if g1 != g2:
                # Different groups: synergy (companion planting)
                synergy[i, j] = synergy[j, i] = np.random.uniform(0.05, 0.15)
            else:
                # Same group: conflict (compete for nutrients)
                conflict[i, j] = conflict[j, i] = np.random.uniform(0.1, 0.2)
    
    return synergy, conflict, food_names

# ============================================================================
# STEP 3: Build Native QUBO (no CQM conversion!)
# ============================================================================

def build_sparse_qubo(n_farms, foods, food_groups, food_benefits, 
                      k_neighbors=4, same_farm_penalty=10.0, seed=42):
    """
    Build sparse QUBO directly (no CQM→BQM conversion).
    
    Variables: x[f,c] = 1 if farm f grows crop c
    
    Objective (QUBO - minimize):
      - Σ benefit[f,c] * x[f,c]                           (linear: maximize benefits)
      + penalty * Σ x[f,c1] * x[f,c2]  for c1≠c2          (same-farm conflict)
      - Σ synergy[c1,c2] * x[f1,c1] * x[f2,c2]            (neighbor synergies)
      + Σ conflict[c1,c2] * x[f1,c1] * x[f2,c2]           (neighbor conflicts)
    """
    
    # Build farm adjacency
    farm_graph, farm_names, _ = build_farm_adjacency(n_farms, k_neighbors, seed)
    
    # Build crop interactions
    synergy, conflict, food_names = build_crop_interactions(foods, food_groups)
    n_foods = len(food_names)
    
    # Create BQM directly
    bqm = BinaryQuadraticModel('BINARY')
    
    # Variable naming
    def var_name(farm, crop):
        return f"x_{farm}_{crop}"
    
    # 1. Linear terms: -benefit (we minimize, so negate)
    for farm in farm_names:
        for crop in food_names:
            benefit = food_benefits.get(crop, 0.1)
            bqm.add_variable(var_name(farm, crop), -benefit)
    
    # 2. Same-farm conflict: penalty for selecting multiple crops on same farm
    for farm in farm_names:
        for i, c1 in enumerate(food_names):
            for j, c2 in enumerate(food_names):
                if i < j:
                    bqm.add_interaction(var_name(farm, c1), var_name(farm, c2), 
                                       same_farm_penalty)
    
    # 3. Neighbor interactions (SPARSE - only adjacent farms!)
    for f1, f2 in farm_graph.edges():
        for i, c1 in enumerate(food_names):
            for j, c2 in enumerate(food_names):
                # Synergy (negative = good, we minimize)
                syn = synergy[i, j]
                # Conflict (positive = bad)
                conf = conflict[i, j]
                
                interaction = conf - syn  # Net interaction
                
                if abs(interaction) > 1e-6:
                    bqm.add_interaction(var_name(f1, c1), var_name(f2, c2), interaction)
    
    return bqm, farm_graph, farm_names, food_names

# ============================================================================
# MAIN: Compare OLD vs NEW formulation
# ============================================================================

if __name__ == "__main__":
    print("\n[1/4] Loading food data...")
    _, foods, food_groups, config_loaded = load_food_data('full_family')

    weights = config_loaded.get('parameters', {}).get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15
    })

    food_names = list(foods.keys())
    n_foods = len(food_names)

    food_benefits = {}
    for food in food_names:
        benefit = (
            weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) +
            weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) -
            weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) +
            weights.get('affordability', 0) * foods[food].get('affordability', 0) +
            weights.get('sustainability', 0) * foods[food].get('sustainability', 0)
        )
        food_benefits[food] = benefit

    print(f"   Foods: {n_foods}")
    print(f"   Food groups: {len(food_groups)}")

    print("\n[2/4] Comparing OLD (CQM→BQM) vs NEW (Sparse QUBO)...")
    print()
    print(f"{'Scale':<8} {'OLD Vars':<10} {'OLD Quad':<14} {'OLD MaxDeg':<12} | {'NEW Vars':<10} {'NEW Quad':<14} {'NEW MaxDeg':<12} | {'Deg Impr':<10} {'Quad Impr'}")
    print("-"*120)

    scales = [10, 15, 25, 50, 100, 200, 500]
    results = []

    for n_plots in scales:
        # ===== OLD FORMULATION (CQM → BQM) =====
        land_availability = patch_sampler.generate_grid(n_plots, area=100.0, seed=42)
        farms = list(land_availability.keys())
        total_area = sum(land_availability.values())
        
        fg_constraints = {
            group: {'min_foods': 1, 'max_foods': len(foods_in_group)}
            for group, foods_in_group in food_groups.items()
        }
        
        cqm = ConstrainedQuadraticModel()
        
        Y = {}
        for farm in farms:
            for food in food_names:
                Y[(farm, food)] = Binary(f'Y_{farm}_{food}')
        
        U = {}
        for food in food_names:
            U[food] = Binary(f'U_{food}')
        
        objective = sum(
            food_benefits[food] * land_availability[farm] * Y[(farm, food)]
            for farm in farms for food in food_names
        ) / total_area
        cqm.set_objective(-objective)
        
        for farm in farms:
            cqm.add_constraint(sum(Y[(farm, food)] for food in food_names) <= 1, 
                              label=f'Max_Assignment_{farm}')
        
        for food in food_names:
            for farm in farms:
                cqm.add_constraint(Y[(farm, food)] - U[food] <= 0, 
                                  label=f'U_Link_{farm}_{food}')
            cqm.add_constraint(U[food] - sum(Y[(farm, food)] for farm in farms) <= 0,
                              label=f'U_Bound_{food}')
        
        for group_name, limits in fg_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            if foods_in_group:
                group_sum = sum(U[f] for f in foods_in_group if f in U)
                group_label = group_name.replace(' ', '_').replace(',', '').replace('-', '_')
                min_foods = limits.get('min_foods', 0)
                if min_foods > 0:
                    cqm.add_constraint(group_sum >= min_foods, 
                                      label=f'MinFoodGroup_{group_label}')
        
        old_bqm, _ = cqm_to_bqm(cqm, lagrange_multiplier=10.0)
        
        old_vars = len(old_bqm.variables)
        old_quad = len(old_bqm.quadratic)
        
        # Compute degree
        old_degree = defaultdict(int)
        for (v1, v2) in old_bqm.quadratic.keys():
            old_degree[v1] += 1
            old_degree[v2] += 1
        old_max_deg = max(old_degree.values()) if old_degree else 0
        
        # ===== NEW FORMULATION (Sparse QUBO) =====
        new_bqm, farm_graph, _, _ = build_sparse_qubo(
            n_plots, foods, food_groups, food_benefits, 
            k_neighbors=4, same_farm_penalty=10.0, seed=42
        )
        
        new_vars = len(new_bqm.variables)
        new_quad = len(new_bqm.quadratic)
        
        new_degree = defaultdict(int)
        for (v1, v2) in new_bqm.quadratic.keys():
            new_degree[v1] += 1
            new_degree[v2] += 1
        new_max_deg = max(new_degree.values()) if new_degree else 0
        
        # Improvement
        deg_improvement = old_max_deg / new_max_deg if new_max_deg > 0 else float('inf')
        quad_improvement = old_quad / new_quad if new_quad > 0 else float('inf')
        
        print(f"{n_plots:<8} {old_vars:<10} {old_quad:<14,} {old_max_deg:<12} | {new_vars:<10} {new_quad:<14,} {new_max_deg:<12} | {deg_improvement:<10.1f}x {quad_improvement:.1f}x")
        
        results.append({
            'n_plots': n_plots,
            'old': {'vars': old_vars, 'quad': old_quad, 'max_deg': old_max_deg},
            'new': {'vars': new_vars, 'quad': new_quad, 'max_deg': new_max_deg},
            'deg_improvement': deg_improvement,
            'quad_improvement': quad_improvement
        })

    print()
    print("="*100)
    print("SUMMARY")
    print("="*100)
    print()
    print("OLD FORMULATION (CQM → BQM):")
    print(f"  - Max degree grows linearly: {results[0]['old']['max_deg']} → {results[-1]['old']['max_deg']}")
    print(f"  - Quadratic terms explode: {results[0]['old']['quad']:,} → {results[-1]['old']['quad']:,}")
    print(f"  - NOT embeddable on Pegasus (max degree ~15) for n_plots > 15")
    print()
    print("NEW FORMULATION (Sparse QUBO with k=4 neighbors):")
    print(f"  - Max degree BOUNDED: {results[0]['new']['max_deg']} → {results[-1]['new']['max_deg']}")
    print(f"  - Quadratic terms scale linearly: {results[0]['new']['quad']:,} → {results[-1]['new']['quad']:,}")
    print(f"  - POTENTIALLY embeddable on Pegasus for much larger problems!")
    print()
    print(f"DEGREE IMPROVEMENT: {results[0]['deg_improvement']:.0f}x → {results[-1]['deg_improvement']:.0f}x")
    print(f"QUADRATIC IMPROVEMENT: {results[0]['quad_improvement']:.0f}x → {results[-1]['quad_improvement']:.0f}x")

    # Save results
    os.makedirs('@todo/hardness_output', exist_ok=True)
    out_path = '@todo/hardness_output/sparse_qubo_comparison.json'
    with open(out_path, 'w') as f:
        json.dump({
            'description': 'Comparison of OLD (CQM→BQM) vs NEW (Sparse QUBO) formulations',
            'k_neighbors': 4,
            'results': results
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")
