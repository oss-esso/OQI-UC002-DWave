#!/usr/bin/env python3
"""
Hybrid Decomposition Benchmark: Binary Subproblem Strategies

Compares decomposition approaches for the farm allocation MINLP:
1. Pure CQM (binary-only, all at once)
2. Benders Decomposition (Master: Y binary, Sub: A continuous)
   - At each iteration, test different binary subproblem decompositions
3. Dantzig-Wolfe Decomposition (Column generation)
   - At each iteration, test different pricing subproblem decompositions

For each approach, we test binary decomposition strategies:
- None (solve full binary problem)
- PlotBased (partition by farm)
- Spectral (graph clustering)
- Louvain (community detection)
- Multilevel (hierarchical grouping)
- Cutset (fine-grained)

This benchmark answers: Does decomposing the BINARY part of Benders/DW help?
"""

import os
import sys
import time
import json
import numpy as np
import networkx as nx
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Callable

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("HYBRID DECOMPOSITION BENCHMARK")
print("Binary Subproblem Strategies for Benders & Dantzig-Wolfe")
print("=" * 80)

# Imports
print("\n[1/5] Importing libraries...")
import_start = time.time()

import gurobipy as gp
from gurobipy import GRB

# sklearn for spectral
try:
    from sklearn.cluster import SpectralClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Louvain
try:
    from networkx.algorithms.community import louvain_communities
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

# Real data
from src.scenarios import load_food_data
from Utils import patch_sampler

print(f"  [OK] Imports done in {time.time() - import_start:.2f}s")

# ============================================================================
# CONFIGURATION
# ============================================================================

N_FARMS = 25
N_FOODS = 27
MAX_BENDERS_ITERATIONS = 20
MAX_DW_ITERATIONS = 20
TIME_LIMIT = 300  # 5 minutes total per method
SOLVE_TIMEOUT = 30  # per subproblem

OUTPUT_DIR = Path(__file__).parent / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_problem_data(n_farms):
    """Load food data and create farm configuration."""
    print(f"\n[2/5] Loading problem data for {n_farms} farms...")
    
    _, foods, food_groups, config_loaded = load_food_data('full_family')
    weights = config_loaded.get('parameters', {}).get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15
    })
    
    # Land availability (continuous areas per farm)
    land_availability = patch_sampler.generate_grid(n_farms, area=100.0, seed=42)
    farm_names = list(land_availability.keys())
    total_area = sum(land_availability.values())
    
    # Food benefits
    food_benefits = {}
    for food in foods:
        benefit = (
            weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) +
            weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) -
            weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) +
            weights.get('affordability', 0) * foods[food].get('affordability', 0) +
            weights.get('sustainability', 0) * foods[food].get('sustainability', 0)
        )
        food_benefits[food] = benefit
    
    # Group mappings
    group_name_mapping = {
        'Animal-source foods': 'Proteins',
        'Pulses, nuts, and seeds': 'Legumes',
        'Starchy staples': 'Staples',
        'Fruits': 'Fruits',
        'Vegetables': 'Vegetables'
    }
    reverse_mapping = {v: k for k, v in group_name_mapping.items()}
    
    # Food group constraints with diversity
    food_group_constraints = {
        'Proteins': {'min': 2, 'max': 5},
        'Fruits': {'min': 2, 'max': 5},
        'Legumes': {'min': 2, 'max': 5},
        'Staples': {'min': 2, 'max': 5},
        'Vegetables': {'min': 2, 'max': 5}
    }
    
    # Max plots per crop (diversity)
    max_plots_per_crop = 5
    
    # Minimum planting area (continuous)
    min_planting_area = {food: 0.1 for food in foods}  # 0.1 ha minimum
    
    # Maximum planting area
    max_planting_area = {food: total_area * 0.3 for food in foods}  # 30% max per crop
    
    print(f"  Farms: {len(farm_names)}")
    print(f"  Foods: {len(foods)}")
    print(f"  Total area: {total_area:.1f} ha")
    print(f"  Food group constraints: min=2, max=5 per group")
    print(f"  Max plots per crop: {max_plots_per_crop}")
    
    return {
        'foods': foods,
        'food_names': list(foods.keys()),
        'food_groups': food_groups,
        'food_benefits': food_benefits,
        'weights': weights,
        'land_availability': land_availability,
        'farm_names': farm_names,
        'total_area': total_area,
        'food_group_constraints': food_group_constraints,
        'max_plots_per_crop': max_plots_per_crop,
        'min_planting_area': min_planting_area,
        'max_planting_area': max_planting_area,
        'group_name_mapping': group_name_mapping,
        'reverse_mapping': reverse_mapping
    }


# ============================================================================
# BINARY PARTITION METHODS
# ============================================================================

def build_y_variable_graph(data):
    """Build graph for Y variable partitioning."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    food_groups = data['food_groups']
    
    G = nx.Graph()
    
    # Add Y variables as nodes
    for farm in farm_names:
        for food in food_names:
            G.add_node(f"Y_{farm}_{food}")
    
    # Edges: one_per_farm connects all Y[farm,*]
    for farm in farm_names:
        y_vars = [f"Y_{farm}_{food}" for food in food_names]
        for i, v1 in enumerate(y_vars):
            for v2 in y_vars[i+1:]:
                G.add_edge(v1, v2, weight=1)
    
    # Edges: food group constraints connect Y[*,food] for foods in same group
    for group, group_foods in food_groups.items():
        for farm in farm_names:
            for i, f1 in enumerate(group_foods):
                for f2 in group_foods[i+1:]:
                    v1 = f"Y_{farm}_{f1}"
                    v2 = f"Y_{farm}_{f2}"
                    if G.has_node(v1) and G.has_node(v2):
                        if G.has_edge(v1, v2):
                            G[v1][v2]['weight'] += 1
                        else:
                            G.add_edge(v1, v2, weight=1)
    
    return G


def partition_none(data):
    """No partitioning - solve all Y together."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    all_vars = {f"Y_{farm}_{food}" for farm in farm_names for food in food_names}
    return [all_vars], "None"


def partition_plot_based(data):
    """One partition per farm."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    partitions = []
    for farm in farm_names:
        partitions.append({f"Y_{farm}_{food}" for food in food_names})
    return partitions, "PlotBased"


def partition_spectral(data, n_clusters=4):
    """Spectral clustering."""
    if not HAS_SKLEARN:
        return None, "sklearn not available"
    
    G = build_y_variable_graph(data)
    nodes = list(G.nodes())
    adj = nx.to_numpy_array(G, nodelist=nodes)
    
    try:
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                               random_state=42, n_init=10)
        labels = sc.fit_predict(adj + np.eye(len(nodes)) * 0.01)
        
        partitions = defaultdict(set)
        for i, node in enumerate(nodes):
            partitions[labels[i]].add(node)
        return list(partitions.values()), f"Spectral({n_clusters})"
    except:
        return None, "Spectral failed"


def partition_louvain(data):
    """Louvain community detection."""
    if not HAS_LOUVAIN:
        return None, "Louvain not available"
    
    G = build_y_variable_graph(data)
    try:
        communities = louvain_communities(G, seed=42)
        return [set(c) for c in communities], "Louvain"
    except:
        return None, "Louvain failed"


def partition_multilevel(data, group_size=5):
    """Group farms together."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    partitions = []
    for i in range(0, len(farm_names), group_size):
        group_farms = farm_names[i:i+group_size]
        part = set()
        for farm in group_farms:
            for food in food_names:
                part.add(f"Y_{farm}_{food}")
        partitions.append(part)
    return partitions, f"Multilevel({group_size})"


def partition_cutset(data, farms_per_cut=2):
    """Fine-grained partitioning."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    partitions = []
    for i in range(0, len(farm_names), farms_per_cut):
        cut_farms = farm_names[i:i+farms_per_cut]
        part = set()
        for farm in cut_farms:
            for food in food_names:
                part.add(f"Y_{farm}_{food}")
        partitions.append(part)
    return partitions, f"Cutset({farms_per_cut})"


# All partition methods
PARTITION_METHODS = [
    ("None", partition_none),
    ("PlotBased", partition_plot_based),
    ("Spectral(4)", lambda d: partition_spectral(d, 4)),
    ("Louvain", partition_louvain),
    ("Multilevel(5)", lambda d: partition_multilevel(d, 5)),
    ("Cutset(2)", lambda d: partition_cutset(d, 2)),
]


# ============================================================================
# GROUND TRUTH SOLVER (Full MINLP)
# ============================================================================

def solve_ground_truth_minlp(data, timeout=60):
    """Solve the full MINLP with Gurobi (ground truth)."""
    food_names = data['food_names']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
    max_plots_per_crop = data['max_plots_per_crop']
    min_planting_area = data['min_planting_area']
    max_planting_area = data['max_planting_area']
    reverse_mapping = data['reverse_mapping']
    total_area = data['total_area']
    
    model = gp.Model("GroundTruth_MINLP")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    # Binary variables: Y[farm, food] = 1 if food planted on farm
    Y = {}
    for farm in farm_names:
        for food in food_names:
            Y[(farm, food)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{farm}_{food}")
    
    # Continuous variables: A[farm, food] = area allocated
    A = {}
    for farm in farm_names:
        for food in food_names:
            A[(farm, food)] = model.addVar(lb=0.0, ub=land_availability[farm], 
                                           name=f"A_{farm}_{food}")
    
    # Unique food selection U[food] for food group constraints
    U = {}
    for food in food_names:
        U[food] = model.addVar(vtype=GRB.BINARY, name=f"U_{food}")
    
    # Objective: maximize area-weighted benefit
    obj = gp.quicksum(food_benefits[food] * A[(farm, food)] 
                      for farm in farm_names for food in food_names) / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraint 1: Area capacity per farm
    for farm in farm_names:
        model.addConstr(gp.quicksum(A[(farm, food)] for food in food_names) <= land_availability[farm])
    
    # Constraint 2: A > 0 only if Y = 1, and min/max area
    for farm in farm_names:
        for food in food_names:
            min_area = min_planting_area.get(food, 0.1)
            max_area = min(max_planting_area.get(food, land_availability[farm]), land_availability[farm])
            model.addConstr(A[(farm, food)] <= max_area * Y[(farm, food)])
            model.addConstr(A[(farm, food)] >= min_area * Y[(farm, food)])
    
    # Constraint 3: U-Y linking
    for food in food_names:
        for farm in farm_names:
            model.addConstr(U[food] >= Y[(farm, food)])
    
    # Constraint 4: Food group constraints on U
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        if foods_in_group:
            group_sum = gp.quicksum(U[f] for f in foods_in_group if f in U)
            if limits.get('min', 0) > 0:
                model.addConstr(group_sum >= limits['min'])
            if 'max' in limits:
                model.addConstr(group_sum <= limits['max'])
    
    # Constraint 5: Max plots per crop
    if max_plots_per_crop:
        for food in food_names:
            model.addConstr(gp.quicksum(Y[(farm, food)] for farm in farm_names) <= max_plots_per_crop)
    
    start = time.time()
    model.optimize()
    solve_time = time.time() - start
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        solution_Y = {(f, c): int(Y[(f, c)].X) for f in farm_names for c in food_names}
        solution_A = {(f, c): A[(f, c)].X for f in farm_names for c in food_names}
        solution_U = {c: int(U[c].X) for c in food_names}
        
        return {
            'objective': model.ObjVal,
            'solve_time': solve_time,
            'Y': solution_Y,
            'A': solution_A,
            'U': solution_U,
            'success': True
        }
    
    return {'objective': 0, 'solve_time': solve_time, 'success': False}


# ============================================================================
# BENDERS DECOMPOSITION WITH BINARY PARTITIONING
# ============================================================================

def solve_benders_master_partitioned(data, Y_fixed, partition, fixed_vars, timeout=30):
    """Solve master Y problem for a partition with some variables fixed."""
    food_names = data['food_names']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
    max_plots_per_crop = data['max_plots_per_crop']
    reverse_mapping = data['reverse_mapping']
    total_area = data['total_area']
    
    model = gp.Model("Benders_Master_Partition")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    # Variables for this partition only
    Y = {}
    for var_name in partition:
        parts = var_name.split("_", 2)
        farm, food = parts[1], parts[2]
        Y[(farm, food)] = model.addVar(vtype=GRB.BINARY, name=var_name)
    
    # Helper to get variable value
    def get_y(farm, food):
        if (farm, food) in Y:
            return Y[(farm, food)]
        var_name = f"Y_{farm}_{food}"
        if var_name in fixed_vars:
            return fixed_vars[var_name]
        return 0
    
    # Objective: simple proxy based on food benefits
    obj = 0
    for var_name in partition:
        parts = var_name.split("_", 2)
        farm, food = parts[1], parts[2]
        obj += food_benefits[food] * land_availability[farm] * Y[(farm, food)]
    obj = obj / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints that involve this partition's variables
    # Food group constraints (if U variables in partition - not for Y only)
    # Max plots per crop
    if max_plots_per_crop:
        for food in food_names:
            y_vars = []
            fixed_count = 0
            for farm in farm_names:
                var_name = f"Y_{farm}_{food}"
                if var_name in partition:
                    y_vars.append(Y[(farm, food)])
                elif var_name in fixed_vars:
                    fixed_count += fixed_vars[var_name]
            if y_vars:
                model.addConstr(gp.quicksum(y_vars) + fixed_count <= max_plots_per_crop)
    
    model.optimize()
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        solution = {var_name: int(Y[(var_name.split("_", 2)[1], var_name.split("_", 2)[2])].X) 
                    for var_name in partition}
        return {'success': True, 'solution': solution, 'objective': model.ObjVal}
    
    return {'success': False, 'solution': {var_name: 0 for var_name in partition}}


def solve_benders_subproblem(data, Y_fixed, timeout=30):
    """Solve Benders subproblem: optimize A given fixed Y."""
    food_names = data['food_names']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    min_planting_area = data['min_planting_area']
    max_planting_area = data['max_planting_area']
    total_area = data['total_area']
    
    model = gp.Model("Benders_Subproblem")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    # Continuous A variables
    A = {}
    for farm in farm_names:
        for food in food_names:
            A[(farm, food)] = model.addVar(lb=0.0, name=f"A_{farm}_{food}")
    
    # Objective
    obj = gp.quicksum(food_benefits[food] * A[(farm, food)] 
                      for farm in farm_names for food in food_names) / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Capacity constraint
    for farm in farm_names:
        model.addConstr(gp.quicksum(A[(farm, food)] for food in food_names) <= land_availability[farm])
    
    # Linking: A > 0 only if Y = 1
    for farm in farm_names:
        for food in food_names:
            y_val = Y_fixed.get((farm, food), 0)
            min_area = min_planting_area.get(food, 0.1)
            max_area = min(max_planting_area.get(food, land_availability[farm]), land_availability[farm])
            model.addConstr(A[(farm, food)] <= max_area * y_val)
            model.addConstr(A[(farm, food)] >= min_area * y_val)
    
    model.optimize()
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        solution = {(f, c): A[(f, c)].X for f in farm_names for c in food_names}
        return {'success': True, 'A': solution, 'objective': model.ObjVal}
    
    return {'success': False, 'A': {}, 'objective': 0}


def solve_benders_with_partitioning(data, partition_fn, max_iterations=20, timeout=300):
    """
    Benders decomposition with partitioned master problem.
    
    Master: Y variables (binary) - solved with partition decomposition
    Subproblem: A variables (continuous) given Y*
    """
    start_time = time.time()
    
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    # Get partitions for Y variables
    partitions, partition_name = partition_fn(data)
    if partitions is None:
        return {'success': False, 'error': partition_name}
    
    print(f"    Benders with {partition_name}: {len(partitions)} partitions")
    
    # Initialize Y
    Y_current = {(f, c): 0 for f in farm_names for c in food_names}
    
    best_objective = 0
    best_Y = Y_current.copy()
    best_A = {}
    
    iterations = []
    
    for iter_num in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        iter_start = time.time()
        
        # Solve master (Y) by partitions
        Y_solutions = {}
        master_time = 0
        
        for partition in partitions:
            part_start = time.time()
            result = solve_benders_master_partitioned(data, Y_current, partition, Y_solutions, timeout=SOLVE_TIMEOUT)
            master_time += time.time() - part_start
            
            if result['success']:
                for var_name, val in result['solution'].items():
                    parts = var_name.split("_", 2)
                    Y_solutions[var_name] = val
                    Y_current[(parts[1], parts[2])] = val
        
        # Solve subproblem (A) given Y*
        sub_start = time.time()
        sub_result = solve_benders_subproblem(data, Y_current, timeout=SOLVE_TIMEOUT)
        sub_time = time.time() - sub_start
        
        if sub_result['success']:
            obj = sub_result['objective']
            if obj > best_objective:
                best_objective = obj
                best_Y = Y_current.copy()
                best_A = sub_result['A'].copy()
        
        iterations.append({
            'iteration': iter_num + 1,
            'objective': sub_result.get('objective', 0),
            'master_time': master_time,
            'sub_time': sub_time,
            'total_time': time.time() - iter_start
        })
        
        # Simple convergence check
        if iter_num > 0 and abs(iterations[-1]['objective'] - iterations[-2]['objective']) < 1e-6:
            break
    
    total_time = time.time() - start_time
    
    return {
        'success': True,
        'objective': best_objective,
        'Y': best_Y,
        'A': best_A,
        'iterations': iterations,
        'total_time': total_time,
        'partition_method': partition_name,
        'n_partitions': len(partitions),
        'n_iterations': len(iterations)
    }


# ============================================================================
# DANTZIG-WOLFE WITH BINARY PARTITIONING
# ============================================================================

def solve_dw_pricing_partitioned(data, duals, partition, fixed_vars, timeout=30):
    """Solve pricing subproblem for a partition."""
    food_names = data['food_names']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    min_planting_area = data['min_planting_area']
    max_planting_area = data['max_planting_area']
    max_plots_per_crop = data['max_plots_per_crop']
    total_area = data['total_area']
    
    model = gp.Model("DW_Pricing_Partition")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    # Variables for this partition
    Y = {}
    A = {}
    
    for var_name in partition:
        parts = var_name.split("_", 2)
        farm, food = parts[1], parts[2]
        Y[(farm, food)] = model.addVar(vtype=GRB.BINARY, name=var_name)
        A[(farm, food)] = model.addVar(lb=0.0, name=f"A_{farm}_{food}")
    
    # Reduced cost objective
    obj = 0
    for var_name in partition:
        parts = var_name.split("_", 2)
        farm, food = parts[1], parts[2]
        # Benefit minus dual price
        dual_price = duals.get(farm, 0)
        obj += (food_benefits[food] - dual_price) * A[(farm, food)] / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Capacity constraint (for farms in this partition)
    farms_in_partition = set()
    for var_name in partition:
        parts = var_name.split("_", 2)
        farms_in_partition.add(parts[1])
    
    for farm in farms_in_partition:
        farm_vars = [(f, c) for (f, c) in Y.keys() if f == farm]
        if farm_vars:
            model.addConstr(gp.quicksum(A[(f, c)] for (f, c) in farm_vars) <= land_availability[farm])
    
    # Linking constraints
    for (farm, food), y_var in Y.items():
        min_area = min_planting_area.get(food, 0.1)
        max_area = min(max_planting_area.get(food, land_availability[farm]), land_availability[farm])
        model.addConstr(A[(farm, food)] <= max_area * y_var)
        model.addConstr(A[(farm, food)] >= min_area * y_var)
    
    # Max plots per crop
    if max_plots_per_crop:
        for food in food_names:
            y_vars = [Y[(f, c)] for (f, c) in Y.keys() if c == food]
            fixed_count = sum(fixed_vars.get(f"Y_{f}_{food}", 0) for f in farm_names 
                            if f"Y_{f}_{food}" not in partition)
            if y_vars:
                model.addConstr(gp.quicksum(y_vars) + fixed_count <= max_plots_per_crop)
    
    model.optimize()
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        y_sol = {(f, c): int(Y[(f, c)].X) for (f, c) in Y.keys()}
        a_sol = {(f, c): A[(f, c)].X for (f, c) in A.keys()}
        return {'success': True, 'Y': y_sol, 'A': a_sol, 'reduced_cost': model.ObjVal}
    
    return {'success': False, 'reduced_cost': 0}


def solve_dw_with_partitioning(data, partition_fn, max_iterations=20, timeout=300):
    """
    Dantzig-Wolfe with partitioned pricing subproblems.
    """
    start_time = time.time()
    
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    # Get partitions
    partitions, partition_name = partition_fn(data)
    if partitions is None:
        return {'success': False, 'error': partition_name}
    
    print(f"    D-W with {partition_name}: {len(partitions)} partitions")
    
    # Initialize column pool with simple patterns
    columns = []
    for farm in farm_names:
        # Simple pattern: best food for this farm
        best_food = max(food_names, key=lambda f: data['food_benefits'][f])
        col = {
            'Y': {(farm, best_food): 1},
            'A': {(farm, best_food): data['min_planting_area'].get(best_food, 0.1)},
            'objective': data['food_benefits'][best_food] * data['min_planting_area'].get(best_food, 0.1) / data['total_area']
        }
        columns.append(col)
    
    best_objective = 0
    best_Y = {}
    best_A = {}
    
    iterations = []
    
    # Initialize duals
    duals = {farm: 0 for farm in farm_names}
    
    for iter_num in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        iter_start = time.time()
        
        # Solve pricing subproblems by partition
        new_columns = []
        pricing_time = 0
        total_reduced_cost = 0
        
        fixed_vars = {}  # Track fixed variables across partitions
        
        for partition in partitions:
            part_start = time.time()
            result = solve_dw_pricing_partitioned(data, duals, partition, fixed_vars, timeout=SOLVE_TIMEOUT)
            pricing_time += time.time() - part_start
            
            if result['success']:
                total_reduced_cost += result['reduced_cost']
                if result['reduced_cost'] > 1e-6:
                    new_col = {
                        'Y': result['Y'],
                        'A': result['A'],
                        'objective': sum(data['food_benefits'][c] * a for (f, c), a in result['A'].items()) / data['total_area']
                    }
                    new_columns.append(new_col)
                
                # Update fixed vars for next partition
                for (f, c), val in result['Y'].items():
                    fixed_vars[f"Y_{f}_{c}"] = val
        
        # Add new columns
        columns.extend(new_columns)
        
        # Simple RMP: select best column combination
        if columns:
            best_col = max(columns, key=lambda c: c['objective'])
            if best_col['objective'] > best_objective:
                best_objective = best_col['objective']
                best_Y = best_col['Y']
                best_A = best_col['A']
        
        iterations.append({
            'iteration': iter_num + 1,
            'reduced_cost': total_reduced_cost,
            'n_new_columns': len(new_columns),
            'total_columns': len(columns),
            'pricing_time': pricing_time,
            'total_time': time.time() - iter_start
        })
        
        # Convergence check
        if total_reduced_cost < 1e-6:
            break
    
    total_time = time.time() - start_time
    
    return {
        'success': True,
        'objective': best_objective,
        'Y': best_Y,
        'A': best_A,
        'iterations': iterations,
        'total_time': total_time,
        'partition_method': partition_name,
        'n_partitions': len(partitions),
        'n_iterations': len(iterations)
    }


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print(f"HYBRID DECOMPOSITION BENCHMARK: {N_FARMS} farms × {N_FOODS} foods")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data = load_problem_data(N_FARMS)
    
    # Solve ground truth
    print("\n[3/5] Solving ground truth MINLP...")
    gt_result = solve_ground_truth_minlp(data, timeout=60)
    print(f"  Ground Truth Objective: {gt_result['objective']:.6f}")
    print(f"  Solve time: {gt_result['solve_time']:.3f}s")
    
    # Results storage
    results = {
        'problem': {
            'n_farms': N_FARMS,
            'n_foods': N_FOODS,
            'total_area': data['total_area']
        },
        'ground_truth': {
            'objective': gt_result['objective'],
            'solve_time': gt_result['solve_time']
        },
        'benders': {},
        'dantzig_wolfe': {}
    }
    
    # Test Benders with different binary partitions
    print("\n[4/5] Testing Benders Decomposition with binary partitioning...")
    print("-" * 60)
    
    for partition_name, partition_fn in PARTITION_METHODS:
        print(f"\n  Testing Benders + {partition_name}...")
        
        try:
            result = solve_benders_with_partitioning(data, partition_fn, 
                                                      max_iterations=MAX_BENDERS_ITERATIONS,
                                                      timeout=TIME_LIMIT)
            
            if result['success']:
                gap = (gt_result['objective'] - result['objective']) / gt_result['objective'] * 100
                print(f"    Objective: {result['objective']:.6f} (gap: {gap:+.1f}%)")
                print(f"    Iterations: {result['n_iterations']}, Time: {result['total_time']:.2f}s")
                
                results['benders'][partition_name] = {
                    'objective': result['objective'],
                    'gap_percent': gap,
                    'n_iterations': result['n_iterations'],
                    'n_partitions': result['n_partitions'],
                    'total_time': result['total_time'],
                    'success': True
                }
            else:
                print(f"    FAILED: {result.get('error', 'Unknown')}")
                results['benders'][partition_name] = {'success': False}
        except Exception as e:
            print(f"    ERROR: {e}")
            results['benders'][partition_name] = {'success': False, 'error': str(e)}
    
    # Test Dantzig-Wolfe with different binary partitions
    print("\n[5/5] Testing Dantzig-Wolfe Decomposition with binary partitioning...")
    print("-" * 60)
    
    for partition_name, partition_fn in PARTITION_METHODS:
        print(f"\n  Testing D-W + {partition_name}...")
        
        try:
            result = solve_dw_with_partitioning(data, partition_fn,
                                                 max_iterations=MAX_DW_ITERATIONS,
                                                 timeout=TIME_LIMIT)
            
            if result['success']:
                gap = (gt_result['objective'] - result['objective']) / gt_result['objective'] * 100 if gt_result['objective'] > 0 else 0
                print(f"    Objective: {result['objective']:.6f} (gap: {gap:+.1f}%)")
                print(f"    Iterations: {result['n_iterations']}, Time: {result['total_time']:.2f}s")
                
                results['dantzig_wolfe'][partition_name] = {
                    'objective': result['objective'],
                    'gap_percent': gap,
                    'n_iterations': result['n_iterations'],
                    'n_partitions': result['n_partitions'],
                    'total_time': result['total_time'],
                    'success': True
                }
            else:
                print(f"    FAILED: {result.get('error', 'Unknown')}")
                results['dantzig_wolfe'][partition_name] = {'success': False}
        except Exception as e:
            print(f"    ERROR: {e}")
            results['dantzig_wolfe'][partition_name] = {'success': False, 'error': str(e)}
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = OUTPUT_DIR / f"hybrid_decomp_benchmark_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY: HYBRID DECOMPOSITION BENCHMARK")
    print("=" * 100)
    print(f"Problem: {N_FARMS} farms × {N_FOODS} foods, Total area: {data['total_area']:.1f} ha")
    print(f"Ground Truth: obj={gt_result['objective']:.6f}, time={gt_result['solve_time']:.3f}s")
    print()
    
    print(f"{'Method':<30} {'Parts':>6} {'Iters':>6} {'Time':>10} {'Objective':>12} {'Gap':>10}")
    print("-" * 100)
    
    print("BENDERS DECOMPOSITION:")
    for method, data_r in results['benders'].items():
        if data_r.get('success', False):
            print(f"  {method:<28} {data_r['n_partitions']:>6} {data_r['n_iterations']:>6} {data_r['total_time']:>9.2f}s {data_r['objective']:>12.6f} {data_r['gap_percent']:>+9.1f}%")
        else:
            print(f"  {method:<28} {'FAILED':>40}")
    
    print("\nDANTZIG-WOLFE DECOMPOSITION:")
    for method, data_r in results['dantzig_wolfe'].items():
        if data_r.get('success', False):
            print(f"  {method:<28} {data_r['n_partitions']:>6} {data_r['n_iterations']:>6} {data_r['total_time']:>9.2f}s {data_r['objective']:>12.6f} {data_r['gap_percent']:>+9.1f}%")
        else:
            print(f"  {method:<28} {'FAILED':>40}")
    
    print("=" * 100)
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
