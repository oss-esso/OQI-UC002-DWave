#!/usr/bin/env python3
"""
Comprehensive Formulation Benchmark

Compares three approaches for the farm allocation problem:

1. CQM (Binary-only, Plot-based formulation)
   - Y[patch, food] binary variables only
   - Each patch gets exactly one food
   - All decomposition strategies

2. Benders Decomposition (Farm formulation with continuous)
   - Master: Y[farm, food] binary
   - Subproblem: A[farm, food] continuous areas
   - All decomposition strategies for binary subproblem

3. Dantzig-Wolfe Decomposition (Farm formulation with continuous)
   - Column generation with pricing subproblems
   - All decomposition strategies for pricing

Metrics for each:
- Solve time
- Embed time (for CQM methods)
- Objective value
- Gap from ground truth
- Constraint violations
"""

import os
import sys
import time
import json
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("COMPREHENSIVE FORMULATION BENCHMARK")
print("CQM (Binary) vs Benders (Continuous) vs Dantzig-Wolfe (Continuous)")
print("=" * 100)

# Imports
print("\n[1/6] Importing libraries...")
import_start = time.time()

import gurobipy as gp
from gurobipy import GRB
from dimod import ConstrainedQuadraticModel, Binary, cqm_to_bqm

# Embedding
try:
    from minorminer import find_embedding
    from dwave.system import DWaveSampler
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False
    print("  Warning: Embedding not available")

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

# Pegasus graph
try:
    from dwave.system import DWaveSampler
    HAS_DWAVE = True
except ImportError:
    HAS_DWAVE = False

try:
    from dwave.embedding.pegasus import pegasus_graph
    HAS_PEGASUS = True
except ImportError:
    try:
        import dwave_networkx as dnx
        pegasus_graph = dnx.pegasus_graph
        HAS_PEGASUS = True
    except ImportError:
        HAS_PEGASUS = False

# Real data
from src.scenarios import load_food_data
from Utils import patch_sampler

print(f"  [OK] Imports done in {time.time() - import_start:.2f}s")

# ============================================================================
# CONFIGURATION
# ============================================================================

N_FARMS = 25
N_FOODS = 27
MAX_ITERATIONS = 15  # For Benders/DW
EMBED_TIMEOUT = 60  # seconds per partition
SOLVE_TIMEOUT = 30  # seconds per solve

OUTPUT_DIR = Path(__file__).parent / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_problem_data(n_farms):
    """Load food data and create configuration."""
    print(f"\n[2/6] Loading problem data for {n_farms} farms...")
    
    _, foods, food_groups, config_loaded = load_food_data('full_family')
    weights = config_loaded.get('parameters', {}).get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15
    })
    
    # Land availability
    land_availability = patch_sampler.generate_grid(n_farms, area=100.0, seed=42)
    farm_names = list(land_availability.keys())
    total_area = sum(land_availability.values())
    
    # Food benefits
    food_names = list(foods.keys())
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
    
    # Group mappings
    group_name_mapping = {
        'Animal-source foods': 'Proteins',
        'Pulses, nuts, and seeds': 'Legumes',
        'Starchy staples': 'Staples',
        'Fruits': 'Fruits',
        'Vegetables': 'Vegetables'
    }
    reverse_mapping = {v: k for k, v in group_name_mapping.items()}
    
    # Constraints
    food_group_constraints = {
        'Proteins': {'min': 2, 'max': 5},
        'Fruits': {'min': 2, 'max': 5},
        'Legumes': {'min': 2, 'max': 5},
        'Staples': {'min': 2, 'max': 5},
        'Vegetables': {'min': 2, 'max': 5}
    }
    
    max_plots_per_crop = 5
    min_planting_area = {food: 0.1 for food in food_names}
    max_planting_area = {food: total_area * 0.3 for food in food_names}
    
    print(f"  Farms: {len(farm_names)}, Foods: {len(food_names)}")
    print(f"  Total area: {total_area:.1f} ha")
    print(f"  Constraints: min=2 per group, max=5 per crop")
    
    return {
        'foods': foods,
        'food_names': food_names,
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
# GROUND TRUTH SOLVER
# ============================================================================

def solve_ground_truth(data, timeout=60):
    """Solve full MINLP with Gurobi (ground truth for continuous formulation)."""
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
    
    model = gp.Model("GroundTruth")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    # Binary Y[farm, food]
    Y = {}
    for farm in farm_names:
        for food in food_names:
            Y[(farm, food)] = model.addVar(vtype=GRB.BINARY)
    
    # Continuous A[farm, food]
    A = {}
    for farm in farm_names:
        for food in food_names:
            A[(farm, food)] = model.addVar(lb=0.0, ub=land_availability[farm])
    
    # Unique U[food]
    U = {}
    for food in food_names:
        U[food] = model.addVar(vtype=GRB.BINARY)
    
    # Objective
    obj = gp.quicksum(food_benefits[food] * A[(farm, food)] 
                      for farm in farm_names for food in food_names) / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Capacity
    for farm in farm_names:
        model.addConstr(gp.quicksum(A[(farm, food)] for food in food_names) <= land_availability[farm])
    
    # Linking A-Y
    for farm in farm_names:
        for food in food_names:
            min_area = min_planting_area.get(food, 0.1)
            max_area = min(max_planting_area.get(food, land_availability[farm]), land_availability[farm])
            model.addConstr(A[(farm, food)] <= max_area * Y[(farm, food)])
            model.addConstr(A[(farm, food)] >= min_area * Y[(farm, food)])
    
    # U-Y linking
    for food in food_names:
        for farm in farm_names:
            model.addConstr(U[food] >= Y[(farm, food)])
    
    # Food group constraints
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        if foods_in_group:
            group_sum = gp.quicksum(U[f] for f in foods_in_group if f in U)
            if limits.get('min', 0) > 0:
                model.addConstr(group_sum >= limits['min'])
            if 'max' in limits:
                model.addConstr(group_sum <= limits['max'])
    
    # Max plots per crop
    for food in food_names:
        model.addConstr(gp.quicksum(Y[(farm, food)] for farm in farm_names) <= max_plots_per_crop)
    
    start = time.time()
    model.optimize()
    solve_time = time.time() - start
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        return {
            'objective': model.ObjVal,
            'solve_time': solve_time,
            'success': True
        }
    return {'objective': 0, 'solve_time': solve_time, 'success': False}


def solve_ground_truth_binary(data, timeout=60):
    """Solve binary-only formulation (ground truth for CQM)."""
    food_names = data['food_names']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
    max_plots_per_crop = data['max_plots_per_crop']
    reverse_mapping = data['reverse_mapping']
    total_area = data['total_area']
    
    model = gp.Model("GroundTruth_Binary")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    # Binary Y[farm, food]
    Y = {}
    for farm in farm_names:
        for food in food_names:
            Y[(farm, food)] = model.addVar(vtype=GRB.BINARY)
    
    # Unique U[food]
    U = {}
    for food in food_names:
        U[food] = model.addVar(vtype=GRB.BINARY)
    
    # Objective: area-weighted benefit (assume full patch area if selected)
    obj = gp.quicksum(food_benefits[food] * land_availability[farm] * Y[(farm, food)] 
                      for farm in farm_names for food in food_names) / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # At most one food per farm/patch
    for farm in farm_names:
        model.addConstr(gp.quicksum(Y[(farm, food)] for food in food_names) <= 1)
    
    # U-Y linking
    for food in food_names:
        for farm in farm_names:
            model.addConstr(U[food] >= Y[(farm, food)])
    
    # Food group constraints
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        if foods_in_group:
            group_sum = gp.quicksum(U[f] for f in foods_in_group if f in U)
            if limits.get('min', 0) > 0:
                model.addConstr(group_sum >= limits['min'])
            if 'max' in limits:
                model.addConstr(group_sum <= limits['max'])
    
    # Max plots per crop
    for food in food_names:
        model.addConstr(gp.quicksum(Y[(farm, food)] for farm in farm_names) <= max_plots_per_crop)
    
    start = time.time()
    model.optimize()
    solve_time = time.time() - start
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        sol_Y = {(f, c): int(Y[(f, c)].X) for f in farm_names for c in food_names}
        sol_U = {c: int(U[c].X) for c in food_names}
        return {
            'objective': model.ObjVal,
            'solve_time': solve_time,
            'Y': sol_Y,
            'U': sol_U,
            'success': True
        }
    return {'objective': 0, 'solve_time': solve_time, 'success': False}


# ============================================================================
# PARTITIONING METHODS
# ============================================================================

def build_variable_graph(data, include_u=True):
    """Build graph for partitioning."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    food_groups = data['food_groups']
    
    G = nx.Graph()
    
    for farm in farm_names:
        for food in food_names:
            G.add_node(f"Y_{farm}_{food}")
    
    if include_u:
        for food in food_names:
            G.add_node(f"U_{food}")
    
    # Edges: one_per_farm
    for farm in farm_names:
        y_vars = [f"Y_{farm}_{food}" for food in food_names]
        for i, v1 in enumerate(y_vars):
            for v2 in y_vars[i+1:]:
                G.add_edge(v1, v2, weight=1)
    
    # Edges: U-Y linking
    if include_u:
        for food in food_names:
            u_var = f"U_{food}"
            for farm in farm_names:
                y_var = f"Y_{farm}_{food}"
                G.add_edge(u_var, y_var, weight=1)
    
    return G


def partition_none(data):
    """No partitioning."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    all_vars = {f"Y_{farm}_{food}" for farm in farm_names for food in food_names}
    all_vars.update({f"U_{food}" for food in food_names})
    return [all_vars], "None"


def partition_plot_based(data):
    """One partition per farm + U partition."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    partitions = []
    for farm in farm_names:
        partitions.append({f"Y_{farm}_{food}" for food in food_names})
    partitions.append({f"U_{food}" for food in food_names})
    return partitions, "PlotBased"


def partition_spectral(data, n_clusters=4):
    """Spectral clustering."""
    if not HAS_SKLEARN:
        return None, "sklearn not available"
    
    G = build_variable_graph(data)
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
    
    G = build_variable_graph(data)
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
    partitions.append({f"U_{food}" for food in food_names})
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
    partitions.append({f"U_{food}" for food in food_names})
    return partitions, f"Cutset({farms_per_cut})"


def partition_spatial_grid(data, grid_size=5):
    """Spatial grid decomposition."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    patches_per_partition = max(1, len(farm_names) // grid_size)
    partitions = []
    
    for i in range(0, len(farm_names), patches_per_partition):
        grid_farms = farm_names[i:i+patches_per_partition]
        part = set()
        for farm in grid_farms:
            for food in food_names:
                part.add(f"Y_{farm}_{food}")
        partitions.append(part)
    partitions.append({f"U_{food}" for food in food_names})
    return partitions, f"SpatialGrid({grid_size})"


# All partition methods
PARTITION_METHODS = [
    ("None", partition_none),
    ("PlotBased", partition_plot_based),
    ("Spectral(4)", lambda d: partition_spectral(d, 4)),
    ("Louvain", partition_louvain),
    ("Multilevel(5)", lambda d: partition_multilevel(d, 5)),
    ("Cutset(2)", lambda d: partition_cutset(d, 2)),
    ("SpatialGrid(5)", lambda d: partition_spatial_grid(d, 5)),
]


# ============================================================================
# CQM BINARY FORMULATION
# ============================================================================

def build_cqm(data):
    """Build CQM for binary-only formulation."""
    food_names = data['food_names']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
    max_plots_per_crop = data['max_plots_per_crop']
    reverse_mapping = data['reverse_mapping']
    total_area = data['total_area']
    
    cqm = ConstrainedQuadraticModel()
    
    Y = {}
    for farm in farm_names:
        for food in food_names:
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")
    
    U = {}
    for food in food_names:
        U[food] = Binary(f"U_{food}")
    
    # Objective (negated for minimization)
    obj = sum(food_benefits[food] * land_availability[farm] * Y[(farm, food)]
              for farm in farm_names for food in food_names) / total_area
    cqm.set_objective(-obj)
    
    # One per farm
    for farm in farm_names:
        cqm.add_constraint(sum(Y[(farm, food)] for food in food_names) <= 1,
                          label=f"one_per_farm_{farm}")
    
    # U-Y linking
    for food in food_names:
        for farm in farm_names:
            cqm.add_constraint(U[food] - Y[(farm, food)] >= 0,
                              label=f"link_{food}_{farm}")
    
    # Food group constraints
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        if foods_in_group:
            group_sum = sum(U[f] for f in foods_in_group if f in U)
            if limits.get('min', 0) > 0:
                cqm.add_constraint(group_sum >= limits['min'], label=f"group_min_{constraint_group}")
            if 'max' in limits:
                cqm.add_constraint(group_sum <= limits['max'], label=f"group_max_{constraint_group}")
    
    # Max plots per crop
    for food in food_names:
        cqm.add_constraint(sum(Y[(farm, food)] for farm in farm_names) <= max_plots_per_crop,
                          label=f"max_plots_{food}")
    
    return cqm, Y, U


def get_pegasus_graph():
    """Get Pegasus topology."""
    if HAS_DWAVE:
        try:
            sampler = DWaveSampler()
            return sampler.to_networkx_graph()
        except:
            pass
    if HAS_PEGASUS:
        return pegasus_graph(16)
    # Fallback: create a simple large graph for testing
    import networkx as nx
    G = nx.complete_graph(5000)
    return G


def study_embedding(bqm, target_graph, timeout=60):
    """Try to embed BQM."""
    if not HAS_EMBEDDING:
        return {'success': False, 'time': 0, 'physical_qubits': 0, 'max_chain': 0}
    
    source_graph = nx.Graph()
    source_graph.add_nodes_from(bqm.variables)
    source_graph.add_edges_from(bqm.quadratic.keys())
    
    start = time.time()
    try:
        embedding = find_embedding(source_graph, target_graph, timeout=timeout, random_seed=42)
        embed_time = time.time() - start
        
        if embedding:
            total_qubits = sum(len(chain) for chain in embedding.values())
            max_chain = max(len(chain) for chain in embedding.values())
            return {
                'success': True,
                'time': embed_time,
                'physical_qubits': total_qubits,
                'max_chain': max_chain
            }
    except:
        pass
    
    return {'success': False, 'time': time.time() - start, 'physical_qubits': 0, 'max_chain': 0}


def solve_cqm_partition(partition, data, fixed_vars, timeout=30):
    """Solve a CQM partition with Gurobi."""
    food_names = data['food_names']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
    max_plots_per_crop = data['max_plots_per_crop']
    reverse_mapping = data['reverse_mapping']
    total_area = data['total_area']
    
    model = gp.Model("CQM_Partition")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    gp_vars = {}
    for var_name in partition:
        gp_vars[var_name] = model.addVar(vtype=GRB.BINARY, name=var_name)
    
    def get_var(var_name):
        if var_name in gp_vars:
            return gp_vars[var_name]
        elif var_name in fixed_vars:
            return fixed_vars[var_name]
        return None
    
    # Objective
    obj = 0
    for farm in farm_names:
        for food in food_names:
            var_name = f"Y_{farm}_{food}"
            if var_name in partition:
                obj += food_benefits[food] * land_availability[farm] * gp_vars[var_name]
    obj = obj / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # One per farm
    for farm in farm_names:
        y_vars = []
        fixed_count = 0
        skip = False
        for food in food_names:
            var_name = f"Y_{farm}_{food}"
            val = get_var(var_name)
            if val is None:
                skip = True
                break
            elif isinstance(val, (int, float)):
                fixed_count += val
            else:
                y_vars.append(val)
        if not skip and y_vars:
            model.addConstr(gp.quicksum(y_vars) + fixed_count <= 1)
    
    # U-Y linking
    for food in food_names:
        u_name = f"U_{food}"
        u_var = get_var(u_name)
        if u_var is None:
            continue
        for farm in farm_names:
            y_name = f"Y_{farm}_{food}"
            y_var = get_var(y_name)
            if y_var is None:
                continue
            if isinstance(u_var, (int, float)) and isinstance(y_var, (int, float)):
                pass
            elif isinstance(u_var, (int, float)):
                model.addConstr(y_var <= u_var)
            elif isinstance(y_var, (int, float)):
                if y_var == 1:
                    model.addConstr(u_var >= 1)
            else:
                model.addConstr(u_var >= y_var)
    
    # Food group constraints
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        group_vars = []
        fixed_sum = 0
        skip = False
        for food in foods_in_group:
            var_name = f"U_{food}"
            val = get_var(var_name)
            if val is None:
                skip = True
                break
            elif isinstance(val, (int, float)):
                fixed_sum += val
            else:
                group_vars.append(val)
        if not skip:
            if group_vars:
                group_sum = gp.quicksum(group_vars) + fixed_sum
            else:
                group_sum = fixed_sum
            if limits.get('min', 0) > 0:
                model.addConstr(group_sum >= limits['min'])
            if 'max' in limits:
                model.addConstr(group_sum <= limits['max'])
    
    # Max plots per crop
    for food in food_names:
        y_vars = []
        fixed_count = 0
        for farm in farm_names:
            var_name = f"Y_{farm}_{food}"
            if var_name in partition:
                y_vars.append(gp_vars[var_name])
            elif var_name in fixed_vars:
                fixed_count += fixed_vars[var_name]
        if y_vars:
            model.addConstr(gp.quicksum(y_vars) + fixed_count <= max_plots_per_crop)
    
    model.optimize()
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        solution = {var_name: int(gp_vars[var_name].X) for var_name in partition}
        return {'success': True, 'solution': solution}
    return {'success': False, 'solution': {var_name: 0 for var_name in partition}}


def solve_cqm_with_decomposition(data, partition_fn, target_graph):
    """Solve CQM with given decomposition strategy."""
    partitions, partition_name = partition_fn(data)
    if partitions is None:
        return {'success': False, 'error': partition_name}
    
    start_time = time.time()
    
    # Build CQM and convert to BQM for embedding study
    cqm, _, _ = build_cqm(data)
    bqm, _ = cqm_to_bqm(cqm, lagrange_multiplier=10.0)
    
    # Embedding analysis
    embed_results = []
    total_embed_time = 0
    total_physical_qubits = 0
    all_embeddable = True
    
    for partition in partitions:
        if len(partition) == 0:
            continue
        
        # Create sub-BQM for this partition
        sub_bqm_linear = {v: bqm.get_linear(v) for v in partition if v in bqm.variables}
        sub_bqm_quadratic = {(u, v): bqm.get_quadratic(u, v) 
                           for u in partition for v in partition 
                           if u < v and (u, v) in bqm.quadratic}
        
        from dimod import BinaryQuadraticModel
        sub_bqm = BinaryQuadraticModel(sub_bqm_linear, sub_bqm_quadratic, 0.0, 'BINARY')
        
        embed_result = study_embedding(sub_bqm, target_graph, timeout=EMBED_TIMEOUT)
        embed_results.append(embed_result)
        total_embed_time += embed_result['time']
        if embed_result['success']:
            total_physical_qubits += embed_result['physical_qubits']
        else:
            all_embeddable = False
    
    n_embeddable = sum(1 for r in embed_results if r['success'])
    
    # Solve with Gurobi
    all_solutions = {}
    solve_times = []
    
    for partition in partitions:
        if len(partition) == 0:
            continue
        part_start = time.time()
        result = solve_cqm_partition(partition, data, all_solutions, timeout=SOLVE_TIMEOUT)
        solve_times.append(time.time() - part_start)
        if result['success']:
            all_solutions.update(result['solution'])
        else:
            for var_name in partition:
                all_solutions[var_name] = 0
    
    total_solve_time = sum(solve_times)
    
    # Calculate objective
    food_names = data['food_names']
    farm_names = data['farm_names']
    food_benefits = data['food_benefits']
    land_availability = data['land_availability']
    total_area = data['total_area']
    
    objective = 0
    for farm in farm_names:
        for food in food_names:
            var_name = f"Y_{farm}_{food}"
            if all_solutions.get(var_name, 0) == 1:
                objective += food_benefits[food] * land_availability[farm]
    objective /= total_area
    
    # Check violations
    violations = check_violations(all_solutions, data)
    
    total_time = time.time() - start_time
    
    return {
        'success': True,
        'partition_method': partition_name,
        'n_partitions': len(partitions),
        'objective': objective,
        'solve_time': total_solve_time,
        'embed_time': total_embed_time,
        'total_time': total_time,
        'n_embeddable': n_embeddable,
        'total_partitions': len([p for p in partitions if len(p) > 0]),
        'physical_qubits': total_physical_qubits if all_embeddable else None,
        'violations': len(violations)
    }


def check_violations(solution, data):
    """Check constraint violations."""
    food_names = data['food_names']
    food_groups = data['food_groups']
    farm_names = data['farm_names']
    food_group_constraints = data['food_group_constraints']
    max_plots_per_crop = data['max_plots_per_crop']
    reverse_mapping = data['reverse_mapping']
    
    violations = []
    
    # One per farm
    for farm in farm_names:
        count = sum(1 for food in food_names if solution.get(f"Y_{farm}_{food}", 0) == 1)
        if count > 1:
            violations.append(f"Farm {farm}: {count} foods")
    
    # Food group constraints
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        selected = set()
        for food in foods_in_group:
            if solution.get(f"U_{food}", 0) == 1:
                selected.add(food)
            else:
                for farm in farm_names:
                    if solution.get(f"Y_{farm}_{food}", 0) == 1:
                        selected.add(food)
                        break
        count = len(selected)
        if count < limits.get('min', 0):
            violations.append(f"Group {constraint_group}: {count} < min {limits['min']}")
        if count > limits.get('max', 999):
            violations.append(f"Group {constraint_group}: {count} > max {limits['max']}")
    
    # Max plots per crop
    for food in food_names:
        count = sum(1 for farm in farm_names if solution.get(f"Y_{farm}_{food}", 0) == 1)
        if count > max_plots_per_crop:
            violations.append(f"Food {food}: {count} > max {max_plots_per_crop}")
    
    return violations


# ============================================================================
# BENDERS DECOMPOSITION
# ============================================================================

def solve_benders_subproblem(data, Y_fixed, timeout=30):
    """Solve continuous subproblem given Y."""
    food_names = data['food_names']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    min_planting_area = data['min_planting_area']
    max_planting_area = data['max_planting_area']
    total_area = data['total_area']
    
    model = gp.Model("Benders_Sub")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    A = {}
    for farm in farm_names:
        for food in food_names:
            A[(farm, food)] = model.addVar(lb=0.0)
    
    obj = gp.quicksum(food_benefits[food] * A[(farm, food)] 
                      for farm in farm_names for food in food_names) / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    for farm in farm_names:
        model.addConstr(gp.quicksum(A[(farm, food)] for food in food_names) <= land_availability[farm])
    
    for farm in farm_names:
        for food in food_names:
            y_val = Y_fixed.get((farm, food), 0)
            min_area = min_planting_area.get(food, 0.1)
            max_area = min(max_planting_area.get(food, land_availability[farm]), land_availability[farm])
            model.addConstr(A[(farm, food)] <= max_area * y_val)
            model.addConstr(A[(farm, food)] >= min_area * y_val)
    
    model.optimize()
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        return {'success': True, 'objective': model.ObjVal}
    return {'success': False, 'objective': 0}


def solve_benders_with_decomposition(data, partition_fn, max_iterations=15):
    """Solve using Benders with binary decomposition."""
    partitions, partition_name = partition_fn(data)
    if partitions is None:
        return {'success': False, 'error': partition_name}
    
    start_time = time.time()
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    # Initialize Y
    Y_current = {(f, c): 0 for f in farm_names for c in food_names}
    best_objective = 0
    best_Y = Y_current.copy()
    
    iterations_data = []
    total_solve_time = 0
    
    for iter_num in range(max_iterations):
        iter_start = time.time()
        
        # Solve Y partitions sequentially
        Y_solutions = {}
        master_time = 0
        
        for partition in partitions:
            if len(partition) == 0:
                continue
            part_start = time.time()
            result = solve_cqm_partition(partition, data, Y_solutions, timeout=SOLVE_TIMEOUT)
            master_time += time.time() - part_start
            
            if result['success']:
                for var_name, val in result['solution'].items():
                    Y_solutions[var_name] = val
                    if var_name.startswith("Y_"):
                        parts = var_name.split("_", 2)
                        Y_current[(parts[1], parts[2])] = val
        
        # Solve continuous subproblem
        sub_start = time.time()
        sub_result = solve_benders_subproblem(data, Y_current, timeout=SOLVE_TIMEOUT)
        sub_time = time.time() - sub_start
        
        total_solve_time += master_time + sub_time
        
        if sub_result['success'] and sub_result['objective'] > best_objective:
            best_objective = sub_result['objective']
            best_Y = Y_current.copy()
        
        iterations_data.append({
            'iteration': iter_num + 1,
            'objective': sub_result.get('objective', 0),
            'time': time.time() - iter_start
        })
        
        # Convergence check
        if iter_num > 0 and abs(iterations_data[-1]['objective'] - iterations_data[-2]['objective']) < 1e-6:
            break
    
    # Build solution dict for violation check
    solution = {}
    for (f, c), val in best_Y.items():
        solution[f"Y_{f}_{c}"] = val
    # Set U based on Y
    for food in food_names:
        u_val = 0
        for farm in farm_names:
            if best_Y.get((farm, food), 0) == 1:
                u_val = 1
                break
        solution[f"U_{food}"] = u_val
    
    violations = check_violations(solution, data)
    
    return {
        'success': True,
        'partition_method': partition_name,
        'n_partitions': len(partitions),
        'objective': best_objective,
        'solve_time': total_solve_time,
        'embed_time': 0,  # Benders doesn't embed
        'total_time': time.time() - start_time,
        'n_iterations': len(iterations_data),
        'violations': len(violations)
    }


# ============================================================================
# DANTZIG-WOLFE DECOMPOSITION
# ============================================================================

def solve_dw_pricing(data, duals, partition, fixed_vars, timeout=30):
    """Solve D-W pricing subproblem."""
    food_names = data['food_names']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    min_planting_area = data['min_planting_area']
    max_planting_area = data['max_planting_area']
    max_plots_per_crop = data['max_plots_per_crop']
    total_area = data['total_area']
    
    model = gp.Model("DW_Pricing")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    Y = {}
    A = {}
    
    farms_in_partition = set()
    for var_name in partition:
        if var_name.startswith("Y_"):
            parts = var_name.split("_", 2)
            farm, food = parts[1], parts[2]
            farms_in_partition.add(farm)
            Y[(farm, food)] = model.addVar(vtype=GRB.BINARY)
            A[(farm, food)] = model.addVar(lb=0.0)
    
    # Reduced cost objective
    obj = 0
    for (farm, food), y_var in Y.items():
        dual_price = duals.get(farm, 0)
        obj += (food_benefits[food] - dual_price) * A[(farm, food)] / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Capacity
    for farm in farms_in_partition:
        farm_vars = [(f, c) for (f, c) in A.keys() if f == farm]
        if farm_vars:
            model.addConstr(gp.quicksum(A[key] for key in farm_vars) <= land_availability[farm])
    
    # Linking
    for (farm, food), y_var in Y.items():
        min_area = min_planting_area.get(food, 0.1)
        max_area = min(max_planting_area.get(food, land_availability[farm]), land_availability[farm])
        model.addConstr(A[(farm, food)] <= max_area * y_var)
        model.addConstr(A[(farm, food)] >= min_area * y_var)
    
    # Max plots per crop
    for food in food_names:
        y_vars = [Y[(f, c)] for (f, c) in Y.keys() if c == food]
        fixed_count = sum(fixed_vars.get(f"Y_{f}_{food}", 0) for f in farm_names 
                        if f"Y_{f}_{food}" not in partition and f"Y_{f}_{food}" in fixed_vars)
        if y_vars:
            model.addConstr(gp.quicksum(y_vars) + fixed_count <= max_plots_per_crop)
    
    model.optimize()
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        y_sol = {key: int(Y[key].X) for key in Y.keys()}
        a_sol = {key: A[key].X for key in A.keys()}
        col_obj = sum(food_benefits[c] * a for (f, c), a in a_sol.items()) / total_area
        return {'success': True, 'Y': y_sol, 'A': a_sol, 'objective': col_obj, 'reduced_cost': model.ObjVal}
    return {'success': False, 'reduced_cost': 0}


def solve_dw_with_decomposition(data, partition_fn, max_iterations=15):
    """Solve using Dantzig-Wolfe with binary decomposition."""
    partitions, partition_name = partition_fn(data)
    if partitions is None:
        return {'success': False, 'error': partition_name}
    
    start_time = time.time()
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    # Column pool
    columns = []
    # Initialize with simple columns
    for farm in farm_names:
        best_food = max(food_names, key=lambda f: data['food_benefits'][f])
        min_area = data['min_planting_area'].get(best_food, 0.1)
        col = {
            'Y': {(farm, best_food): 1},
            'A': {(farm, best_food): min_area},
            'objective': data['food_benefits'][best_food] * min_area / data['total_area']
        }
        columns.append(col)
    
    best_objective = 0
    best_Y = {}
    duals = {farm: 0 for farm in farm_names}
    
    iterations_data = []
    total_solve_time = 0
    
    for iter_num in range(max_iterations):
        iter_start = time.time()
        
        # Solve pricing by partition
        fixed_vars = {}
        new_columns = []
        pricing_time = 0
        
        for partition in partitions:
            if len(partition) == 0:
                continue
            # Filter to Y variables only
            y_partition = {v for v in partition if v.startswith("Y_")}
            if not y_partition:
                continue
                
            part_start = time.time()
            result = solve_dw_pricing(data, duals, y_partition, fixed_vars, timeout=SOLVE_TIMEOUT)
            pricing_time += time.time() - part_start
            
            if result['success'] and result['reduced_cost'] > 1e-6:
                new_columns.append({
                    'Y': result['Y'],
                    'A': result['A'],
                    'objective': result['objective']
                })
                # Update fixed vars
                for key, val in result['Y'].items():
                    fixed_vars[f"Y_{key[0]}_{key[1]}"] = val
        
        total_solve_time += pricing_time
        columns.extend(new_columns)
        
        # Find best column
        if columns:
            best_col = max(columns, key=lambda c: c['objective'])
            if best_col['objective'] > best_objective:
                best_objective = best_col['objective']
                best_Y = best_col['Y']
        
        iterations_data.append({
            'iteration': iter_num + 1,
            'n_new_columns': len(new_columns),
            'objective': best_objective,
            'time': time.time() - iter_start
        })
        
        # Convergence
        if len(new_columns) == 0:
            break
    
    # Build solution for violation check
    solution = {}
    for key, val in best_Y.items():
        solution[f"Y_{key[0]}_{key[1]}"] = val
    for food in food_names:
        u_val = 0
        for farm in farm_names:
            if best_Y.get((farm, food), 0) == 1:
                u_val = 1
                break
        solution[f"U_{food}"] = u_val
    
    violations = check_violations(solution, data)
    
    return {
        'success': True,
        'partition_method': partition_name,
        'n_partitions': len(partitions),
        'objective': best_objective,
        'solve_time': total_solve_time,
        'embed_time': 0,
        'total_time': time.time() - start_time,
        'n_iterations': len(iterations_data),
        'violations': len(violations)
    }


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def main():
    print("\n" + "=" * 100)
    print(f"COMPREHENSIVE FORMULATION BENCHMARK: {N_FARMS} farms × {N_FOODS} foods")
    print("=" * 100)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data = load_problem_data(N_FARMS)
    
    # Solve ground truths
    print("\n[3/6] Solving ground truth...")
    gt_continuous = solve_ground_truth(data, timeout=60)
    gt_binary = solve_ground_truth_binary(data, timeout=60)
    print(f"  Ground Truth (Continuous MINLP): {gt_continuous['objective']:.6f}")
    print(f"  Ground Truth (Binary-only):      {gt_binary['objective']:.6f}")
    
    # Get Pegasus graph
    print("\n[4/6] Getting Pegasus graph for embedding...")
    target_graph = get_pegasus_graph()
    print(f"  Pegasus: {target_graph.number_of_nodes()} qubits, {target_graph.number_of_edges()} couplers")
    
    # Results storage
    results = {
        'cqm': [],
        'benders': [],
        'dantzig_wolfe': []
    }
    
    # Test CQM with all decompositions
    print("\n[5/6] Testing CQM (Binary-only) with all decompositions...")
    print("-" * 80)
    
    for method_name, method_fn in PARTITION_METHODS:
        print(f"\n  CQM + {method_name}...")
        try:
            result = solve_cqm_with_decomposition(data, method_fn, target_graph)
            if result['success']:
                gap = (gt_binary['objective'] - result['objective']) / gt_binary['objective'] * 100 if gt_binary['objective'] > 0 else 0
                print(f"    Obj: {result['objective']:.6f} (gap: {gap:+.1f}%), Solve: {result['solve_time']:.3f}s, Embed: {result['embed_time']:.2f}s, Viol: {result['violations']}")
                result['gap'] = gap
                result['method'] = method_name
                results['cqm'].append(result)
            else:
                print(f"    FAILED: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # Test Benders with all decompositions
    print("\n  Testing Benders (Continuous) with all decompositions...")
    print("-" * 80)
    
    for method_name, method_fn in PARTITION_METHODS:
        print(f"\n  Benders + {method_name}...")
        try:
            result = solve_benders_with_decomposition(data, method_fn, max_iterations=MAX_ITERATIONS)
            if result['success']:
                gap = (gt_continuous['objective'] - result['objective']) / gt_continuous['objective'] * 100 if gt_continuous['objective'] > 0 else 0
                print(f"    Obj: {result['objective']:.6f} (gap: {gap:+.1f}%), Solve: {result['solve_time']:.3f}s, Iters: {result['n_iterations']}, Viol: {result['violations']}")
                result['gap'] = gap
                result['method'] = method_name
                results['benders'].append(result)
            else:
                print(f"    FAILED: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # Test Dantzig-Wolfe with all decompositions
    print("\n  Testing Dantzig-Wolfe (Continuous) with all decompositions...")
    print("-" * 80)
    
    for method_name, method_fn in PARTITION_METHODS:
        print(f"\n  D-W + {method_name}...")
        try:
            result = solve_dw_with_decomposition(data, method_fn, max_iterations=MAX_ITERATIONS)
            if result['success']:
                gap = (gt_continuous['objective'] - result['objective']) / gt_continuous['objective'] * 100 if gt_continuous['objective'] > 0 else 0
                print(f"    Obj: {result['objective']:.6f} (gap: {gap:+.1f}%), Solve: {result['solve_time']:.3f}s, Iters: {result['n_iterations']}, Viol: {result['violations']}")
                result['gap'] = gap
                result['method'] = method_name
                results['dantzig_wolfe'].append(result)
            else:
                print(f"    FAILED: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = OUTPUT_DIR / f"comprehensive_formulation_benchmark_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'ground_truth_continuous': gt_continuous['objective'],
            'ground_truth_binary': gt_binary['objective'],
            'results': results
        }, f, indent=2, default=str)
    print(f"\n\nResults saved to: {results_file}")
    
    # Print summary table
    print("\n" + "=" * 120)
    print("SUMMARY: COMPREHENSIVE FORMULATION BENCHMARK")
    print("=" * 120)
    print(f"Problem: {N_FARMS} farms × {N_FOODS} foods")
    print(f"Ground Truth (Continuous MINLP): {gt_continuous['objective']:.6f}")
    print(f"Ground Truth (Binary-only CQM):  {gt_binary['objective']:.6f}")
    print()
    
    print(f"{'Formulation':<12} {'Decomposition':<15} {'Parts':>6} {'Solve Time':>12} {'Embed Time':>12} {'Objective':>12} {'Gap':>10} {'Viol':>6}")
    print("-" * 120)
    
    # CQM results
    print("CQM (Binary):")
    for r in results['cqm']:
        embed_str = f"{r['embed_time']:.2f}s" if r['embed_time'] > 0 else "N/A"
        viol_str = f"✅ {r['violations']}" if r['violations'] == 0 else f"❌ {r['violations']}"
        print(f"  {'CQM':<10} {r['method']:<15} {r['n_partitions']:>6} {r['solve_time']:>11.3f}s {embed_str:>12} {r['objective']:>12.6f} {r['gap']:>+9.1f}% {viol_str:>6}")
    
    # Benders results
    print("\nBenders (Continuous):")
    for r in results['benders']:
        viol_str = f"✅ {r['violations']}" if r['violations'] == 0 else f"❌ {r['violations']}"
        print(f"  {'Benders':<10} {r['method']:<15} {r['n_partitions']:>6} {r['solve_time']:>11.3f}s {'N/A':>12} {r['objective']:>12.6f} {r['gap']:>+9.1f}% {viol_str:>6}")
    
    # D-W results
    print("\nDantzig-Wolfe (Continuous):")
    for r in results['dantzig_wolfe']:
        viol_str = f"✅ {r['violations']}" if r['violations'] == 0 else f"❌ {r['violations']}"
        print(f"  {'D-W':<10} {r['method']:<15} {r['n_partitions']:>6} {r['solve_time']:>11.3f}s {'N/A':>12} {r['objective']:>12.6f} {r['gap']:>+9.1f}% {viol_str:>6}")
    
    print("=" * 120)
    
    # Best results
    print("\n📊 KEY INSIGHTS:")
    print("-" * 80)
    
    if results['cqm']:
        best_cqm = min(results['cqm'], key=lambda x: x['gap'])
        print(f"  🏆 Best CQM: {best_cqm['method']} (gap: {best_cqm['gap']:+.1f}%, violations: {best_cqm['violations']})")
    
    if results['benders']:
        best_benders = min(results['benders'], key=lambda x: x['gap'])
        print(f"  🏆 Best Benders: {best_benders['method']} (gap: {best_benders['gap']:+.1f}%, violations: {best_benders['violations']})")
    
    if results['dantzig_wolfe']:
        best_dw = min(results['dantzig_wolfe'], key=lambda x: x['gap'])
        print(f"  🏆 Best D-W: {best_dw['method']} (gap: {best_dw['gap']:+.1f}%, violations: {best_dw['violations']})")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
