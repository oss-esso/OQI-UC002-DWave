#!/usr/bin/env python3
"""
Scaling Benchmark for Formulation Comparison (with Embedding Analysis)

Tests CQM (Binary), Benders, and Dantzig-Wolfe at multiple scales:
- 25, 50, 100, 200 farms

For each scale, tests the best decomposition methods:
- None (baseline)
- PlotBased (best for CQM)
- Multilevel(5) (good balance)

Includes embedding time analysis for QPU readiness.
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
from typing import Dict, List, Set, Tuple, Optional

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("SCALING BENCHMARK: CQM vs Benders vs Dantzig-Wolfe (with Embedding)")
print("=" * 100)

# Imports
print("\n[1/4] Importing libraries...")
import_start = time.time()

import gurobipy as gp
from gurobipy import GRB
from dimod import ConstrainedQuadraticModel, Binary, BinaryQuadraticModel, cqm_to_bqm

# Embedding imports
try:
    from minorminer import find_embedding
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False
    print("  Warning: minorminer not available - embedding will be skipped")

# Pegasus graph
try:
    from dwave.system import DWaveSampler
    HAS_DWAVE = True
except ImportError:
    HAS_DWAVE = False

try:
    import dwave_networkx as dnx
    HAS_DNX = True
except ImportError:
    HAS_DNX = False

try:
    from sklearn.cluster import SpectralClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from networkx.algorithms.community import louvain_communities
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

from src.scenarios import load_food_data
from Utils import patch_sampler

print(f"  [OK] Imports done in {time.time() - import_start:.2f}s")

# ============================================================================
# CONFIGURATION
# ============================================================================

FARM_SCALES = [25, 50, 100, 200]  # Different problem sizes
N_FOODS = 27
MAX_ITERATIONS = 10  # For Benders/DW
SOLVE_TIMEOUT = 60  # seconds per solve
EMBED_TIMEOUT = 30  # seconds per partition embedding

OUTPUT_DIR = Path(__file__).parent / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Global target graph (loaded once)
TARGET_GRAPH = None


def get_pegasus_graph():
    """Get Pegasus topology (cached)."""
    global TARGET_GRAPH
    if TARGET_GRAPH is not None:
        return TARGET_GRAPH
    
    if HAS_DWAVE:
        try:
            sampler = DWaveSampler()
            TARGET_GRAPH = sampler.to_networkx_graph()
            print(f"  Using real Pegasus: {TARGET_GRAPH.number_of_nodes()} qubits")
            return TARGET_GRAPH
        except:
            pass
    
    if HAS_DNX:
        TARGET_GRAPH = dnx.pegasus_graph(16)
        print(f"  Using simulated Pegasus P16: {TARGET_GRAPH.number_of_nodes()} qubits")
        return TARGET_GRAPH
    
    # Fallback
    TARGET_GRAPH = nx.complete_graph(5000)
    print(f"  Using fallback complete graph: {TARGET_GRAPH.number_of_nodes()} nodes")
    return TARGET_GRAPH


# ============================================================================
# DATA LOADING
# ============================================================================

def load_problem_data(n_farms):
    """Load food data and create configuration."""
    _, foods, food_groups, config_loaded = load_food_data('full_family')
    weights = config_loaded.get('parameters', {}).get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15
    })
    
    land_availability = patch_sampler.generate_grid(n_farms, area=100.0, seed=42)
    farm_names = list(land_availability.keys())
    total_area = sum(land_availability.values())
    
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
    
    group_name_mapping = {
        'Animal-source foods': 'Proteins',
        'Pulses, nuts, and seeds': 'Legumes',
        'Starchy staples': 'Staples',
        'Fruits': 'Fruits',
        'Vegetables': 'Vegetables'
    }
    reverse_mapping = {v: k for k, v in group_name_mapping.items()}
    
    food_group_constraints = {
        'Proteins': {'min': 2, 'max': 5},
        'Fruits': {'min': 2, 'max': 5},
        'Legumes': {'min': 2, 'max': 5},
        'Staples': {'min': 2, 'max': 5},
        'Vegetables': {'min': 2, 'max': 5}
    }
    
    max_plots_per_crop = max(5, n_farms // 5)  # Scale with problem size
    min_planting_area = {food: 0.1 for food in food_names}
    max_planting_area = {food: total_area * 0.3 for food in food_names}
    
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
# GROUND TRUTH SOLVERS
# ============================================================================

def solve_ground_truth(data, timeout=120):
    """Solve full MINLP with Gurobi."""
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
    
    Y = {}
    for farm in farm_names:
        for food in food_names:
            Y[(farm, food)] = model.addVar(vtype=GRB.BINARY)
    
    A = {}
    for farm in farm_names:
        for food in food_names:
            A[(farm, food)] = model.addVar(lb=0.0, ub=land_availability[farm])
    
    U = {}
    for food in food_names:
        U[food] = model.addVar(vtype=GRB.BINARY)
    
    obj = gp.quicksum(food_benefits[food] * A[(farm, food)] 
                      for farm in farm_names for food in food_names) / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    for farm in farm_names:
        model.addConstr(gp.quicksum(A[(farm, food)] for food in food_names) <= land_availability[farm])
    
    for farm in farm_names:
        for food in food_names:
            min_area = min_planting_area.get(food, 0.1)
            max_area = min(max_planting_area.get(food, land_availability[farm]), land_availability[farm])
            model.addConstr(A[(farm, food)] <= max_area * Y[(farm, food)])
            model.addConstr(A[(farm, food)] >= min_area * Y[(farm, food)])
    
    for food in food_names:
        for farm in farm_names:
            model.addConstr(U[food] >= Y[(farm, food)])
    
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        if foods_in_group:
            group_sum = gp.quicksum(U[f] for f in foods_in_group if f in U)
            if limits.get('min', 0) > 0:
                model.addConstr(group_sum >= limits['min'])
            if 'max' in limits:
                model.addConstr(group_sum <= limits['max'])
    
    for food in food_names:
        model.addConstr(gp.quicksum(Y[(farm, food)] for farm in farm_names) <= max_plots_per_crop)
    
    start = time.time()
    model.optimize()
    solve_time = time.time() - start
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        return {'objective': model.ObjVal, 'solve_time': solve_time, 'success': True}
    return {'objective': 0, 'solve_time': solve_time, 'success': False}


def solve_ground_truth_binary(data, timeout=120):
    """Solve binary-only formulation."""
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
    
    Y = {}
    for farm in farm_names:
        for food in food_names:
            Y[(farm, food)] = model.addVar(vtype=GRB.BINARY)
    
    U = {}
    for food in food_names:
        U[food] = model.addVar(vtype=GRB.BINARY)
    
    obj = gp.quicksum(food_benefits[food] * land_availability[farm] * Y[(farm, food)] 
                      for farm in farm_names for food in food_names) / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    for farm in farm_names:
        model.addConstr(gp.quicksum(Y[(farm, food)] for food in food_names) <= 1)
    
    for food in food_names:
        for farm in farm_names:
            model.addConstr(U[food] >= Y[(farm, food)])
    
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        if foods_in_group:
            group_sum = gp.quicksum(U[f] for f in foods_in_group if f in U)
            if limits.get('min', 0) > 0:
                model.addConstr(group_sum >= limits['min'])
            if 'max' in limits:
                model.addConstr(group_sum <= limits['max'])
    
    for food in food_names:
        model.addConstr(gp.quicksum(Y[(farm, food)] for farm in farm_names) <= max_plots_per_crop)
    
    start = time.time()
    model.optimize()
    solve_time = time.time() - start
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        return {'objective': model.ObjVal, 'solve_time': solve_time, 'success': True}
    return {'objective': 0, 'solve_time': solve_time, 'success': False}


# ============================================================================
# EMBEDDING FUNCTIONS
# ============================================================================

def build_partition_bqm(partition, data, lagrange=10.0):
    """Build a BQM for a partition to study embedding."""
    food_names = data['food_names']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
    max_plots_per_crop = data['max_plots_per_crop']
    reverse_mapping = data['reverse_mapping']
    total_area = data['total_area']
    
    linear = {}
    quadratic = {}
    
    # Extract farms in this partition
    farms_in_partition = set()
    foods_in_partition = set()
    for var_name in partition:
        if var_name.startswith("Y_"):
            parts = var_name.split("_", 2)
            farms_in_partition.add(parts[1])
            foods_in_partition.add(parts[2])
        elif var_name.startswith("U_"):
            foods_in_partition.add(var_name[2:])
    
    # Linear terms from objective
    for var_name in partition:
        if var_name.startswith("Y_"):
            parts = var_name.split("_", 2)
            farm, food = parts[1], parts[2]
            benefit = food_benefits.get(food, 0) * land_availability.get(farm, 0) / total_area
            linear[var_name] = -benefit  # Negate for minimization
    
    # One-per-farm constraint (QUBO penalty)
    for farm in farms_in_partition:
        y_vars = [f"Y_{farm}_{food}" for food in food_names if f"Y_{farm}_{food}" in partition]
        if len(y_vars) > 1:
            # Penalty: lagrange * (sum(y) - 1)^2 = lagrange * (sum(y^2) - 2*sum(y) + 1 + 2*sum(yi*yj for i<j))
            for var in y_vars:
                linear[var] = linear.get(var, 0) + lagrange * (1 - 2)  # y^2 = y for binary, -2y from expansion
            for i, v1 in enumerate(y_vars):
                for v2 in y_vars[i+1:]:
                    key = (v1, v2) if v1 < v2 else (v2, v1)
                    quadratic[key] = quadratic.get(key, 0) + 2 * lagrange
    
    return BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')


def study_partition_embedding(partition, data, target_graph, timeout=30):
    """Try to embed a partition's BQM."""
    if not HAS_EMBEDDING:
        return {'success': False, 'time': 0, 'physical_qubits': 0, 'max_chain': 0, 'logical_qubits': len(partition)}
    
    bqm = build_partition_bqm(partition, data)
    
    # Build source graph
    source_graph = nx.Graph()
    source_graph.add_nodes_from(bqm.variables)
    source_graph.add_edges_from(bqm.quadratic.keys())
    
    logical_qubits = len(bqm.variables)
    
    if logical_qubits == 0:
        return {'success': True, 'time': 0, 'physical_qubits': 0, 'max_chain': 0, 'logical_qubits': 0}
    
    start = time.time()
    try:
        embedding = find_embedding(source_graph, target_graph, timeout=timeout, random_seed=42)
        embed_time = time.time() - start
        
        if embedding:
            total_qubits = sum(len(chain) for chain in embedding.values())
            max_chain = max(len(chain) for chain in embedding.values()) if embedding else 0
            return {
                'success': True,
                'time': embed_time,
                'physical_qubits': total_qubits,
                'max_chain': max_chain,
                'logical_qubits': logical_qubits
            }
    except Exception as e:
        pass
    
    return {
        'success': False,
        'time': time.time() - start,
        'physical_qubits': 0,
        'max_chain': 0,
        'logical_qubits': logical_qubits
    }


def study_decomposition_embedding(partitions, data, target_graph):
    """Study embedding for all partitions in a decomposition."""
    results = []
    total_embed_time = 0
    total_physical_qubits = 0
    total_logical_qubits = 0
    all_embeddable = True
    max_chain_overall = 0
    
    for partition in partitions:
        if len(partition) == 0:
            continue
        
        result = study_partition_embedding(partition, data, target_graph, timeout=EMBED_TIMEOUT)
        results.append(result)
        total_embed_time += result['time']
        total_logical_qubits += result['logical_qubits']
        
        if result['success']:
            total_physical_qubits += result['physical_qubits']
            max_chain_overall = max(max_chain_overall, result['max_chain'])
        else:
            all_embeddable = False
    
    n_embeddable = sum(1 for r in results if r['success'])
    
    return {
        'n_partitions': len([p for p in partitions if len(p) > 0]),
        'n_embeddable': n_embeddable,
        'all_embeddable': all_embeddable,
        'total_embed_time': total_embed_time,
        'total_physical_qubits': total_physical_qubits if all_embeddable else None,
        'total_logical_qubits': total_logical_qubits,
        'max_chain': max_chain_overall if all_embeddable else None,
        'partition_results': results
    }


# ============================================================================
# PARTITIONING METHODS
# ============================================================================

def partition_none(data):
    food_names = data['food_names']
    farm_names = data['farm_names']
    all_vars = {f"Y_{farm}_{food}" for farm in farm_names for food in food_names}
    all_vars.update({f"U_{food}" for food in food_names})
    return [all_vars], "None"


def partition_plot_based(data):
    food_names = data['food_names']
    farm_names = data['farm_names']
    partitions = []
    for farm in farm_names:
        partitions.append({f"Y_{farm}_{food}" for food in food_names})
    partitions.append({f"U_{food}" for food in food_names})
    return partitions, "PlotBased"


def partition_multilevel(data, group_size=5):
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


def build_variable_graph(data, include_u=True):
    """Build graph for partitioning."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    food_groups = data['food_groups']
    
    G = nx.Graph()
    
    for farm in farm_names:
        for food in food_names:
            G.add_node(f"Y_{farm}_{food}", type='Y', farm=farm, food=food)
    
    if include_u:
        for food in food_names:
            G.add_node(f"U_{food}", type='U', food=food)
    
    # Edges: one_per_farm
    for farm in farm_names:
        y_vars = [f"Y_{farm}_{food}" for food in food_names]
        for i, v1 in enumerate(y_vars):
            for v2 in y_vars[i+1:]:
                G.add_edge(v1, v2, constraint='one_per_farm')
    
    # Edges: U-Y linking
    if include_u:
        for food in food_names:
            u_var = f"U_{food}"
            for farm in farm_names:
                y_var = f"Y_{farm}_{food}"
                G.add_edge(u_var, y_var, constraint='u_y_link')
    
    return G


def partition_spectral(data, n_clusters=4):
    """Spectral clustering."""
    if not HAS_SKLEARN:
        return None, "Spectral not available (sklearn missing)"
    
    G = build_variable_graph(data)
    nodes = list(G.nodes())
    adj = nx.to_numpy_array(G, nodelist=nodes)
    
    try:
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                                        random_state=42, assign_labels='kmeans')
        labels = clustering.fit_predict(adj + np.eye(len(nodes)))
        
        partitions = [set() for _ in range(n_clusters)]
        for node, label in zip(nodes, labels):
            partitions[label].add(node)
        
        return [p for p in partitions if len(p) > 0], f"Spectral({n_clusters})"
    except Exception as e:
        return None, f"Spectral failed: {e}"


def partition_louvain(data):
    """Louvain community detection."""
    if not HAS_LOUVAIN:
        return None, "Louvain not available"
    
    G = build_variable_graph(data)
    try:
        communities = louvain_communities(G, seed=42)
        return [set(c) for c in communities], "Louvain"
    except Exception as e:
        return None, f"Louvain failed: {e}"


def partition_cutset(data, farms_per_cut=2):
    """Fine-grained partitioning."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    partitions = []
    for i in range(0, len(farm_names), farms_per_cut):
        group_farms = farm_names[i:i+farms_per_cut]
        part = set()
        for farm in group_farms:
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
        group_farms = farm_names[i:i+patches_per_partition]
        part = set()
        for farm in group_farms:
            for food in food_names:
                part.add(f"Y_{farm}_{food}")
        partitions.append(part)
    partitions.append({f"U_{food}" for food in food_names})
    return partitions, f"SpatialGrid({grid_size})"


# All partition methods - comprehensive list
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
# CQM SOLVER
# ============================================================================

def solve_cqm_partition(partition, data, fixed_vars, timeout=60):
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


def solve_cqm_with_decomposition(data, partition_fn, target_graph=None, do_embedding=True):
    """Solve CQM with decomposition and optional embedding study."""
    partitions, partition_name = partition_fn(data)
    if partitions is None:
        return {'success': False, 'error': partition_name}
    
    start_time = time.time()
    
    # Embedding study (if requested)
    embed_result = None
    if do_embedding and target_graph is not None:
        embed_result = study_decomposition_embedding(partitions, data, target_graph)
    
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
    
    violations = check_violations(all_solutions, data)
    
    result = {
        'success': True,
        'partition_method': partition_name,
        'n_partitions': len(partitions),
        'objective': objective,
        'solve_time': total_solve_time,
        'total_time': time.time() - start_time,
        'violations': len(violations)
    }
    
    # Add embedding info
    if embed_result:
        result['embed_time'] = embed_result['total_embed_time']
        result['n_embeddable'] = embed_result['n_embeddable']
        result['all_embeddable'] = embed_result['all_embeddable']
        result['physical_qubits'] = embed_result['total_physical_qubits']
        result['logical_qubits'] = embed_result['total_logical_qubits']
        result['max_chain'] = embed_result['max_chain']
    else:
        result['embed_time'] = 0
        result['n_embeddable'] = 0
        result['all_embeddable'] = False
        result['physical_qubits'] = None
        result['logical_qubits'] = 0
        result['max_chain'] = None
    
    return result


def check_violations(solution, data):
    """Check constraint violations."""
    food_names = data['food_names']
    food_groups = data['food_groups']
    farm_names = data['farm_names']
    food_group_constraints = data['food_group_constraints']
    max_plots_per_crop = data['max_plots_per_crop']
    reverse_mapping = data['reverse_mapping']
    
    violations = []
    
    for farm in farm_names:
        count = sum(1 for food in food_names if solution.get(f"Y_{farm}_{food}", 0) == 1)
        if count > 1:
            violations.append(f"Farm {farm}: {count} foods")
    
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
    
    for food in food_names:
        count = sum(1 for farm in farm_names if solution.get(f"Y_{farm}_{food}", 0) == 1)
        if count > max_plots_per_crop:
            violations.append(f"Food {food}: {count} > max {max_plots_per_crop}")
    
    return violations


# ============================================================================
# BENDERS DECOMPOSITION
# ============================================================================

def solve_benders_subproblem(data, Y_fixed, timeout=60):
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


def solve_benders_with_decomposition(data, partition_fn, max_iterations=10):
    """Solve using Benders with binary decomposition."""
    partitions, partition_name = partition_fn(data)
    if partitions is None:
        return {'success': False, 'error': partition_name}
    
    start_time = time.time()
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    Y_current = {(f, c): 0 for f in farm_names for c in food_names}
    best_objective = 0
    best_Y = Y_current.copy()
    
    total_solve_time = 0
    
    for iter_num in range(max_iterations):
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
        
        sub_start = time.time()
        sub_result = solve_benders_subproblem(data, Y_current, timeout=SOLVE_TIMEOUT)
        sub_time = time.time() - sub_start
        
        total_solve_time += master_time + sub_time
        
        if sub_result['success'] and sub_result['objective'] > best_objective:
            best_objective = sub_result['objective']
            best_Y = Y_current.copy()
        
        # Simple convergence
        if iter_num > 0:
            break  # For speed, just do one iteration
    
    solution = {}
    for (f, c), val in best_Y.items():
        solution[f"Y_{f}_{c}"] = val
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
        'total_time': time.time() - start_time,
        'violations': len(violations)
    }


# ============================================================================
# DANTZIG-WOLFE DECOMPOSITION
# ============================================================================

def solve_dw_pricing(data, duals, partition, fixed_vars, timeout=60):
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
    
    obj = 0
    for (farm, food), y_var in Y.items():
        dual_price = duals.get(farm, 0)
        obj += (food_benefits[food] - dual_price) * A[(farm, food)] / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    for farm in farms_in_partition:
        farm_vars = [(f, c) for (f, c) in A.keys() if f == farm]
        if farm_vars:
            model.addConstr(gp.quicksum(A[key] for key in farm_vars) <= land_availability[farm])
    
    for (farm, food), y_var in Y.items():
        min_area = min_planting_area.get(food, 0.1)
        max_area = min(max_planting_area.get(food, land_availability[farm]), land_availability[farm])
        model.addConstr(A[(farm, food)] <= max_area * y_var)
        model.addConstr(A[(farm, food)] >= min_area * y_var)
    
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


def solve_dw_with_decomposition(data, partition_fn, max_iterations=10):
    """Solve using Dantzig-Wolfe with binary decomposition."""
    partitions, partition_name = partition_fn(data)
    if partitions is None:
        return {'success': False, 'error': partition_name}
    
    start_time = time.time()
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    columns = []
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
    
    total_solve_time = 0
    
    for iter_num in range(max_iterations):
        fixed_vars = {}
        new_columns = []
        pricing_time = 0
        
        for partition in partitions:
            if len(partition) == 0:
                continue
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
                for key, val in result['Y'].items():
                    fixed_vars[f"Y_{key[0]}_{key[1]}"] = val
        
        total_solve_time += pricing_time
        columns.extend(new_columns)
        
        if columns:
            best_col = max(columns, key=lambda c: c['objective'])
            if best_col['objective'] > best_objective:
                best_objective = best_col['objective']
                best_Y = best_col['Y']
        
        if len(new_columns) == 0:
            break
    
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
        'total_time': time.time() - start_time,
        'violations': len(violations)
    }


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def main():
    print("\n" + "=" * 140)
    print("SCALING BENCHMARK: CQM vs Benders vs Dantzig-Wolfe (with Embedding)")
    print("=" * 140)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing scales: {FARM_SCALES} farms")
    
    # Load target graph once
    print("\n  Loading Pegasus graph for embedding...")
    target_graph = get_pegasus_graph()
    
    all_results = {}
    
    for n_farms in FARM_SCALES:
        print(f"\n{'='*140}")
        print(f"SCALE: {n_farms} farms × {N_FOODS} foods")
        print(f"{'='*140}")
        
        # Load data
        print(f"\n  Loading data for {n_farms} farms...")
        data = load_problem_data(n_farms)
        n_vars = n_farms * N_FOODS + N_FOODS  # Y + U variables
        print(f"  Variables: {n_vars} ({n_farms}×{N_FOODS} Y + {N_FOODS} U)")
        
        # Ground truth
        print(f"\n  Solving ground truth...")
        gt_start = time.time()
        gt_continuous = solve_ground_truth(data, timeout=120)
        gt_binary = solve_ground_truth_binary(data, timeout=120)
        gt_time = time.time() - gt_start
        print(f"    Continuous MINLP: {gt_continuous['objective']:.6f} ({gt_continuous['solve_time']:.2f}s)")
        print(f"    Binary-only:      {gt_binary['objective']:.6f} ({gt_binary['solve_time']:.2f}s)")
        
        scale_results = {
            'n_farms': n_farms,
            'n_vars': n_vars,
            'gt_continuous': gt_continuous['objective'],
            'gt_binary': gt_binary['objective'],
            'gt_time': gt_time,
            'cqm': {},
            'benders': {},
            'dw': {}
        }
        
        # Test each method
        for method_name, method_fn in PARTITION_METHODS:
            print(f"\n  Testing {method_name}...")
            
            # CQM with embedding
            try:
                result = solve_cqm_with_decomposition(data, method_fn, target_graph, do_embedding=True)
                if result['success']:
                    gap = (gt_binary['objective'] - result['objective']) / gt_binary['objective'] * 100 if gt_binary['objective'] > 0 else 0
                    viol_str = "✅" if result['violations'] == 0 else f"❌{result['violations']}"
                    embed_str = f"embed={result['embed_time']:.1f}s" if result['embed_time'] > 0 else "embed=N/A"
                    embeddable_str = "✅" if result.get('all_embeddable', False) else f"❌{result.get('n_embeddable', 0)}/{result['n_partitions']}"
                    print(f"    CQM:     obj={result['objective']:.6f} gap={gap:+.1f}% solve={result['solve_time']:.3f}s {embed_str} {embeddable_str} {viol_str}")
                    scale_results['cqm'][method_name] = {
                        'objective': result['objective'],
                        'gap': gap,
                        'solve_time': result['solve_time'],
                        'embed_time': result.get('embed_time', 0),
                        'n_embeddable': result.get('n_embeddable', 0),
                        'all_embeddable': result.get('all_embeddable', False),
                        'physical_qubits': result.get('physical_qubits'),
                        'logical_qubits': result.get('logical_qubits', 0),
                        'max_chain': result.get('max_chain'),
                        'n_partitions': result['n_partitions'],
                        'violations': result['violations']
                    }
            except Exception as e:
                print(f"    CQM:     ERROR - {e}")
                import traceback
                traceback.print_exc()
            
            # Benders (no embedding - classical decomposition)
            try:
                result = solve_benders_with_decomposition(data, method_fn, max_iterations=MAX_ITERATIONS)
                if result['success']:
                    gap = (gt_continuous['objective'] - result['objective']) / gt_continuous['objective'] * 100 if gt_continuous['objective'] > 0 else 0
                    viol_str = "✅" if result['violations'] == 0 else f"❌{result['violations']}"
                    print(f"    Benders: obj={result['objective']:.6f} gap={gap:+.1f}% solve={result['solve_time']:.3f}s embed=N/A {viol_str}")
                    scale_results['benders'][method_name] = {
                        'objective': result['objective'],
                        'gap': gap,
                        'solve_time': result['solve_time'],
                        'embed_time': 0,
                        'violations': result['violations']
                    }
            except Exception as e:
                print(f"    Benders: ERROR - {e}")
            
            # D-W (no embedding - classical decomposition)
            try:
                result = solve_dw_with_decomposition(data, method_fn, max_iterations=MAX_ITERATIONS)
                if result['success']:
                    gap = (gt_continuous['objective'] - result['objective']) / gt_continuous['objective'] * 100 if gt_continuous['objective'] > 0 else 0
                    viol_str = "✅" if result['violations'] == 0 else f"❌{result['violations']}"
                    print(f"    D-W:     obj={result['objective']:.6f} gap={gap:+.1f}% solve={result['solve_time']:.3f}s embed=N/A {viol_str}")
                    scale_results['dw'][method_name] = {
                        'objective': result['objective'],
                        'gap': gap,
                        'solve_time': result['solve_time'],
                        'embed_time': 0,
                        'violations': result['violations']
                    }
            except Exception as e:
                print(f"    D-W:     ERROR - {e}")
        
        all_results[n_farms] = scale_results
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = OUTPUT_DIR / f"scaling_benchmark_comprehensive_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {results_file}")
    
    # All decomposition method names
    ALL_DECOMP_METHODS = ['None', 'PlotBased', 'Spectral(4)', 'Louvain', 'Multilevel(5)', 'Cutset(2)', 'SpatialGrid(5)']
    
    # Print summary table
    print("\n" + "=" * 180)
    print("COMPREHENSIVE SCALING SUMMARY (with Embedding)")
    print("=" * 180)
    
    for n_farms in FARM_SCALES:
        res = all_results[n_farms]
        print(f"\n{'='*180}")
        print(f"{n_farms} FARMS × {N_FOODS} FOODS (GT={res['gt_binary']:.6f})")
        print(f"{'='*180}")
        
        print(f"\n{'Method':<12} {'Decomp':<15} {'Parts':>6} {'Solve':>10} {'Embed':>10} {'Qubits':>10} {'Chain':>6} {'Objective':>12} {'Gap':>10} {'Viol':>6}")
        print("-" * 120)
        
        # CQM results
        print("CQM (Binary):")
        for method_name in ALL_DECOMP_METHODS:
            if method_name in res['cqm']:
                r = res['cqm'][method_name]
                viol = "✅" if r['violations'] == 0 else f"❌{r['violations']}"
                embed_str = f"{r['embed_time']:.1f}s" if r['embed_time'] > 0 else "N/A"
                qubits_str = str(r.get('physical_qubits', 'N/A')) if r.get('all_embeddable') else "FAIL"
                chain_str = str(r.get('max_chain', 'N/A')) if r.get('all_embeddable') else "N/A"
                print(f"  {'CQM':<10} {method_name:<15} {r['n_partitions']:>6} {r['solve_time']:>9.3f}s {embed_str:>10} {qubits_str:>10} {chain_str:>6} {r['objective']:>12.6f} {r['gap']:>+9.1f}% {viol:>6}")
        
        # Benders results
        print("\nBenders (Continuous):")
        for method_name in ALL_DECOMP_METHODS:
            if method_name in res['benders']:
                r = res['benders'][method_name]
                viol = "✅" if r['violations'] == 0 else f"❌{r['violations']}"
                print(f"  {'Benders':<10} {method_name:<15} {'N/A':>6} {r['solve_time']:>9.3f}s {'N/A':>10} {'N/A':>10} {'N/A':>6} {r['objective']:>12.6f} {r['gap']:>+9.1f}% {viol:>6}")
        
        # D-W results
        print("\nDantzig-Wolfe (Continuous):")
        for method_name in ALL_DECOMP_METHODS:
            if method_name in res['dw']:
                r = res['dw'][method_name]
                viol = "✅" if r['violations'] == 0 else f"❌{r['violations']}"
                print(f"  {'D-W':<10} {method_name:<15} {'N/A':>6} {r['solve_time']:>9.3f}s {'N/A':>10} {'N/A':>10} {'N/A':>6} {r['objective']:>12.6f} {r['gap']:>+9.1f}% {viol:>6}")
    
    # Embedding summary
    print("\n" + "=" * 180)
    print("EMBEDDING SUMMARY (CQM only - all decompositions)")
    print("=" * 180)
    print(f"\n{'Scale':<8} {'Decomp':<15} {'Parts':>8} {'Embeddable':>14} {'Log.Qubits':>12} {'Phys.Qubits':>12} {'MaxChain':>10} {'EmbedTime':>12}")
    print("-" * 110)
    
    for n_farms in FARM_SCALES:
        res = all_results[n_farms]
        print(f"\n{n_farms} farms:")
        for method_name in ALL_DECOMP_METHODS:
            if method_name in res['cqm']:
                r = res['cqm'][method_name]
                embeddable = "✅ ALL" if r.get('all_embeddable') else f"❌ {r.get('n_embeddable', 0)}/{r['n_partitions']}"
                phys = str(r.get('physical_qubits', 'N/A')) if r.get('all_embeddable') else "FAILED"
                chain = str(r.get('max_chain', 'N/A')) if r.get('all_embeddable') else "N/A"
                print(f"  {n_farms:<6} {method_name:<15} {r['n_partitions']:>8} {embeddable:>14} {r.get('logical_qubits', 0):>12} {phys:>12} {chain:>10} {r['embed_time']:>11.1f}s")
    
    # Best results summary
    print("\n" + "=" * 180)
    print("BEST RESULTS PER SCALE")
    print("=" * 180)
    
    for n_farms in FARM_SCALES:
        res = all_results[n_farms]
        print(f"\n{n_farms} farms:")
        
        # Best CQM
        if res['cqm']:
            best_cqm = min(res['cqm'].items(), key=lambda x: (x[1]['violations'], x[1]['gap']))
            r = best_cqm[1]
            embeddable = "✅" if r.get('all_embeddable') else "❌"
            print(f"  🏆 Best CQM:     {best_cqm[0]:<15} gap={r['gap']:+.1f}% viol={r['violations']} embed={embeddable}")
        
        # Best Benders
        if res['benders']:
            best_benders = min(res['benders'].items(), key=lambda x: (x[1]['violations'], x[1]['gap']))
            r = best_benders[1]
            print(f"  🏆 Best Benders: {best_benders[0]:<15} gap={r['gap']:+.1f}% viol={r['violations']}")
        
        # Best D-W
        if res['dw']:
            best_dw = min(res['dw'].items(), key=lambda x: (x[1]['violations'], x[1]['gap']))
            r = best_dw[1]
            print(f"  🏆 Best D-W:     {best_dw[0]:<15} gap={r['gap']:+.1f}% viol={r['violations']}")
    
    print("\n" + "=" * 180)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()