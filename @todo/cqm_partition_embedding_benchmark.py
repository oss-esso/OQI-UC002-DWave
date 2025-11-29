#!/usr/bin/env python3
"""
CQM Partition Benchmark with Embedding Analysis

This benchmark tests CQM partition methods with:
1. Partition time (decomposition overhead)
2. Embedding analysis (can partitions embed on QPU?)
3. Solve time (Gurobi solving)
4. Objective quality vs ground truth

Tests 25 farms with 27 foods = 702 variables, 710 constraints

Partition methods:
- None (full problem - no decomposition)
- PlotBased (natural domain decomposition)
- Spectral(4) (graph clustering into 4 parts)
- Louvain (community detection)
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

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CQM PARTITION BENCHMARK WITH EMBEDDING ANALYSIS")
print("=" * 80)

# Imports
print("\n[1/4] Importing libraries...")
import_start = time.time()

from dimod import ConstrainedQuadraticModel, Binary, BinaryQuadraticModel
import minorminer

# Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
    print("  [OK] Gurobi available")
except ImportError:
    GUROBI_AVAILABLE = False
    print("  [WARN] Gurobi not available")

# D-Wave for embedding
try:
    from dwave.system import DWaveSampler
    DWAVE_AVAILABLE = True
    print("  [OK] DWaveSampler available")
except ImportError:
    DWAVE_AVAILABLE = False
    print("  [WARN] DWaveSampler not available, using simulated Pegasus")

# Spectral clustering
try:
    from sklearn.cluster import SpectralClustering
    HAS_SKLEARN = True
    print("  [OK] sklearn SpectralClustering available")
except ImportError:
    HAS_SKLEARN = False
    print("  [WARN] sklearn not available")

# Louvain
try:
    from networkx.algorithms.community import louvain_communities
    HAS_LOUVAIN = True
    print("  [OK] Louvain community detection available")
except ImportError:
    HAS_LOUVAIN = False
    print("  [WARN] Louvain not available")

# Real data
from src.scenarios import load_food_data
from Utils import patch_sampler

print(f"  [OK] Imports done in {time.time() - import_start:.2f}s")

# ============================================================================
# CONFIGURATION
# ============================================================================
N_FARMS = 25
N_FOODS = 27
EMBEDDING_TIMEOUT = 60  # seconds per embedding attempt
SOLVE_TIMEOUT = 30  # seconds per partition solve

OUTPUT_DIR = Path(__file__).parent / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_problem_data(n_farms):
    """Load food data and create patch configuration."""
    print(f"\n[2/4] Loading problem data for {n_farms} farms...")
    
    _, foods, food_groups, config_loaded = load_food_data('full_family')
    weights = config_loaded.get('parameters', {}).get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15
    })
    
    land_availability = patch_sampler.generate_grid(n_farms, area=100.0, seed=42)
    patch_names = list(land_availability.keys())
    
    # Build inverse mapping: food -> group
    food_to_group = {}
    for group, foods_list in food_groups.items():
        for food in foods_list:
            food_to_group[food] = group
    
    group_name_mapping = {
        'Animal-source foods': 'Proteins',
        'Pulses, nuts, and seeds': 'Legumes',
        'Starchy staples': 'Staples',
        'Fruits': 'Fruits',
        'Vegetables': 'Vegetables'
    }
    
    # More realistic constraints to force diversity
    # Without max_plots_per_crop, Spinach dominates (benefit=0.43 vs next=0.31)
    food_group_constraints = {
        'Proteins': {'min': 2, 'max': 5},
        'Fruits': {'min': 2, 'max': 5},
        'Legumes': {'min': 2, 'max': 5},
        'Staples': {'min': 2, 'max': 5},
        'Vegetables': {'min': 2, 'max': 5}
    }
    
    # Max plots per crop to force diversity (otherwise Spinach everywhere)
    max_plots_per_crop = 5
    
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
    
    print(f"  Foods: {len(foods)}")
    print(f"  Patches: {len(patch_names)}")
    print(f"  Weights: {weights}")
    print(f"  Max plots per crop: {max_plots_per_crop}")
    
    return {
        'foods': foods,
        'food_groups': food_groups,
        'food_to_group': food_to_group,
        'group_name_mapping': group_name_mapping,
        'food_benefits': food_benefits,
        'weights': weights,
        'land_availability': land_availability,
        'patch_names': patch_names,
        'food_group_constraints': food_group_constraints,
        'max_plots_per_crop': max_plots_per_crop
    }


# ============================================================================
# CQM BUILDING
# ============================================================================

def build_cqm(data):
    """Build the full CQM."""
    foods = data['foods']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    patch_names = data['patch_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
    group_name_mapping = data.get('group_name_mapping', {})
    reverse_mapping = {v: k for k, v in group_name_mapping.items()}
    total_area = sum(land_availability.values())
    
    cqm = ConstrainedQuadraticModel()
    
    # Variables
    Y = {}
    for patch in patch_names:
        for food in foods:
            Y[(patch, food)] = Binary(f"Y_{patch}_{food}")
    
    U = {}
    for food in foods:
        U[food] = Binary(f"U_{food}")
    
    # Objective (negated for minimization)
    objective = 0
    for patch in patch_names:
        patch_area = land_availability[patch]
        for food in foods:
            objective += food_benefits[food] * patch_area * Y[(patch, food)]
    objective = objective / total_area
    cqm.set_objective(-objective)
    
    # Constraint 1: At most one food per patch
    for patch in patch_names:
        cqm.add_constraint(
            sum(Y[(patch, food)] for food in foods) <= 1,
            label=f"one_per_patch_{patch}"
        )
    
    # Constraint 2: U-Y linking
    for food in foods:
        for patch in patch_names:
            cqm.add_constraint(
                U[food] - Y[(patch, food)] >= 0,
                label=f"link_{food}_{patch}"
            )
    
    # Constraint 3: Food group constraints
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        if foods_in_group:
            group_sum = sum(U[f] for f in foods_in_group if f in U)
            if 'min' in limits and limits['min'] > 0:
                cqm.add_constraint(group_sum >= limits['min'], label=f"group_min_{constraint_group}")
            if 'max' in limits:
                cqm.add_constraint(group_sum <= limits['max'], label=f"group_max_{constraint_group}")
    
    return cqm, Y, U


# ============================================================================
# PARTITION METHODS
# ============================================================================

def cqm_to_variable_graph(cqm):
    """Build graph where nodes are variables, edges connect variables in same constraint."""
    G = nx.Graph()
    G.add_nodes_from(cqm.variables)
    
    for label, constraint in cqm.constraints.items():
        constraint_vars = list(constraint.lhs.variables)
        for i, v1 in enumerate(constraint_vars):
            for v2 in constraint_vars[i+1:]:
                if G.has_edge(v1, v2):
                    G[v1][v2]['weight'] += 1
                else:
                    G.add_edge(v1, v2, weight=1)
    
    return G


def partition_none(cqm, data):
    """No partitioning - single partition with all variables."""
    return [set(cqm.variables)], "None"


def partition_plot_based(cqm, data):
    """Partition by patch - each patch gets its own partition, plus U variables."""
    foods = data['foods']
    patch_names = data['patch_names']
    
    partitions = []
    
    # One partition per patch (Y variables only)
    for patch in patch_names:
        patch_vars = set()
        for food in foods:
            var_name = f"Y_{patch}_{food}"
            if var_name in cqm.variables:
                patch_vars.add(var_name)
        if patch_vars:
            partitions.append(patch_vars)
    
    # One partition for all U variables
    u_vars = set()
    for food in foods:
        var_name = f"U_{food}"
        if var_name in cqm.variables:
            u_vars.add(var_name)
    if u_vars:
        partitions.append(u_vars)
    
    return partitions, "PlotBased"


def partition_spectral(cqm, data, n_clusters=4):
    """Spectral clustering on variable graph."""
    if not HAS_SKLEARN:
        return None, "sklearn not available"
    
    G = cqm_to_variable_graph(cqm)
    nodes = list(G.nodes())
    
    if len(nodes) < n_clusters:
        n_clusters = max(2, len(nodes) // 2)
    
    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
    
    try:
        sc = SpectralClustering(n_clusters=n_clusters, 
                                affinity='precomputed',
                                random_state=42,
                                n_init=10)
        labels = sc.fit_predict(adj_matrix + np.eye(len(nodes)) * 0.01)
        
        partitions = defaultdict(set)
        for i, node in enumerate(nodes):
            partitions[labels[i]].add(node)
        
        return list(partitions.values()), f"Spectral({n_clusters})"
    except Exception as e:
        return None, f"Spectral failed: {e}"


def partition_louvain(cqm, data):
    """Louvain community detection on variable graph."""
    if not HAS_LOUVAIN:
        return None, "Louvain not available"
    
    G = cqm_to_variable_graph(cqm)
    
    try:
        communities = louvain_communities(G, seed=42, resolution=1.0)
        partitions = [set(c) for c in communities]
        return partitions, "Louvain"
    except Exception as e:
        return None, f"Louvain failed: {e}"


def partition_master_subproblem(cqm, data):
    """Master (U vars first) + Subproblems (Y vars per patch).
    
    This ordering is SUBOPTIMAL because:
    - Master solves U variables without knowing Y assignments
    - Subproblems are then constrained by potentially bad U choices
    """
    foods = data['foods']
    patch_names = data['patch_names']
    
    partitions = []
    
    # Master partition: all U variables FIRST
    u_vars = set()
    for food in foods:
        var_name = f"U_{food}"
        if var_name in cqm.variables:
            u_vars.add(var_name)
    partitions.append(u_vars)
    
    # Subproblem partitions: Y variables per patch
    for patch in patch_names:
        patch_vars = set()
        for food in foods:
            var_name = f"Y_{patch}_{food}"
            if var_name in cqm.variables:
                patch_vars.add(var_name)
        if patch_vars:
            partitions.append(patch_vars)
    
    return partitions, "MasterSubproblem"


def partition_multilevel(cqm, data, coarsening_factor=2):
    """Multilevel coarsening - groups patches in pairs/triplets.
    
    Similar to ML-QLS style decomposition.
    """
    foods = data['foods']
    patch_names = data['patch_names']
    
    n_patches = len(patch_names)
    group_size = coarsening_factor
    
    partitions = []
    
    # Group patches together
    for i in range(0, n_patches, group_size):
        group_patches = patch_names[i:i+group_size]
        group_vars = set()
        for patch in group_patches:
            for food in foods:
                var_name = f"Y_{patch}_{food}"
                if var_name in cqm.variables:
                    group_vars.add(var_name)
        if group_vars:
            partitions.append(group_vars)
    
    # U variables partition
    u_vars = set()
    for food in foods:
        var_name = f"U_{food}"
        if var_name in cqm.variables:
            u_vars.add(var_name)
    if u_vars:
        partitions.append(u_vars)
    
    return partitions, f"Multilevel({coarsening_factor})"


def partition_cutset(cqm, data, patches_per_cut=2):
    """Cutset-based decomposition - small partitions for each patch pair.
    
    Creates many small partitions that are easier to embed.
    """
    foods = data['foods']
    patch_names = data['patch_names']
    
    partitions = []
    
    # Very fine-grained: each patch or pair gets its own partition
    for i in range(0, len(patch_names), patches_per_cut):
        cut_patches = patch_names[i:i+patches_per_cut]
        cut_vars = set()
        for patch in cut_patches:
            for food in foods:
                var_name = f"Y_{patch}_{food}"
                if var_name in cqm.variables:
                    cut_vars.add(var_name)
        if cut_vars:
            partitions.append(cut_vars)
    
    # U variables at the end
    u_vars = set()
    for food in foods:
        var_name = f"U_{food}"
        if var_name in cqm.variables:
            u_vars.add(var_name)
    if u_vars:
        partitions.append(u_vars)
    
    return partitions, f"Cutset({patches_per_cut})"


def partition_spatial_grid(cqm, data, grid_size=5):
    """Spatial grid decomposition - groups patches by geographic proximity.
    
    Assumes patches are ordered spatially (e.g., patch_0, patch_1, ... in a grid).
    """
    foods = data['foods']
    patch_names = data['patch_names']
    
    n_patches = len(patch_names)
    patches_per_partition = max(1, n_patches // grid_size)
    
    partitions = []
    
    # Group spatially adjacent patches
    for i in range(0, n_patches, patches_per_partition):
        grid_patches = patch_names[i:i+patches_per_partition]
        grid_vars = set()
        for patch in grid_patches:
            for food in foods:
                var_name = f"Y_{patch}_{food}"
                if var_name in cqm.variables:
                    grid_vars.add(var_name)
        if grid_vars:
            partitions.append(grid_vars)
    
    # U variables partition
    u_vars = set()
    for food in foods:
        var_name = f"U_{food}"
        if var_name in cqm.variables:
            u_vars.add(var_name)
    if u_vars:
        partitions.append(u_vars)
    
    return partitions, f"SpatialGrid({grid_size})"


def partition_energy_impact(cqm, data, partition_size=100):
    """Energy-impact based decomposition.
    
    Selects variables with highest impact on objective first.
    Groups high-impact variables together.
    """
    foods = data['foods']
    food_benefits = data['food_benefits']
    patch_names = data['patch_names']
    land_availability = data['land_availability']
    
    # Calculate impact for each Y variable
    var_impacts = []
    for patch in patch_names:
        patch_area = land_availability[patch]
        for food in foods:
            var_name = f"Y_{patch}_{food}"
            if var_name in cqm.variables:
                impact = abs(food_benefits[food] * patch_area)
                var_impacts.append((var_name, impact))
    
    # Sort by impact (descending)
    var_impacts.sort(key=lambda x: x[1], reverse=True)
    
    # Group into partitions
    partitions = []
    current_partition = set()
    
    for var_name, _ in var_impacts:
        current_partition.add(var_name)
        if len(current_partition) >= partition_size:
            partitions.append(current_partition)
            current_partition = set()
    
    if current_partition:
        partitions.append(current_partition)
    
    # U variables partition
    u_vars = set()
    for food in foods:
        var_name = f"U_{food}"
        if var_name in cqm.variables:
            u_vars.add(var_name)
    if u_vars:
        partitions.append(u_vars)
    
    return partitions, f"EnergyImpact({partition_size})"


def partition_food_group_based(cqm, data):
    """Partition by food group - one partition per food group.
    
    WARNING: This cuts one_per_patch constraints and WILL produce violations!
    Included for comparison to show why this approach fails.
    """
    foods = data['foods']
    food_groups = data['food_groups']
    patch_names = data['patch_names']
    
    partitions = []
    
    for group, group_foods in food_groups.items():
        group_vars = set()
        for food in group_foods:
            var_name = f"U_{food}"
            if var_name in cqm.variables:
                group_vars.add(var_name)
            for patch in patch_names:
                var_name = f"Y_{patch}_{food}"
                if var_name in cqm.variables:
                    group_vars.add(var_name)
        if group_vars:
            partitions.append(group_vars)
    
    return partitions, "FoodGroupBased"


# ============================================================================
# EMBEDDING ANALYSIS
# ============================================================================

def get_pegasus_graph():
    """Get Pegasus topology graph."""
    if DWAVE_AVAILABLE:
        try:
            sampler = DWaveSampler()
            return sampler.to_networkx_graph()
        except Exception:
            pass
    
    # Simulate Pegasus P16
    import dwave_networkx as dnx
    return dnx.pegasus_graph(16)


def partition_to_bqm(partition, data):
    """Convert a partition (set of variable names) to a simple BQM for embedding."""
    # Create a simple BQM with just the variables
    # We use the constraint graph structure to add interactions
    bqm = BinaryQuadraticModel('BINARY')
    
    foods = data['foods']
    food_benefits = data['food_benefits']
    patch_names = data['patch_names']
    land_availability = data['land_availability']
    total_area = sum(land_availability.values())
    
    # Add variables with linear biases from objective
    for var_name in partition:
        if var_name.startswith("Y_"):
            parts = var_name.split("_", 2)
            if len(parts) == 3:
                patch = parts[1]
                food = parts[2]
                if food in food_benefits and patch in land_availability:
                    bias = -food_benefits[food] * land_availability[patch] / total_area
                    bqm.add_variable(var_name, bias)
                else:
                    bqm.add_variable(var_name, 0)
        else:
            bqm.add_variable(var_name, 0)
    
    # Add quadratic terms for constraints within partition
    # one_per_patch: sum(Y[patch,*]) <= 1 → penalize pairs
    for patch in patch_names:
        patch_vars = [f"Y_{patch}_{food}" for food in foods if f"Y_{patch}_{food}" in partition]
        if len(patch_vars) > 1:
            for i, v1 in enumerate(patch_vars):
                for v2 in patch_vars[i+1:]:
                    bqm.add_quadratic(v1, v2, 2.0)  # Penalty for both selected
    
    return bqm


def study_embedding(bqm, target_graph, timeout=60):
    """Study embedding feasibility for a BQM."""
    n_vars = len(bqm.variables)
    n_interactions = len(bqm.quadratic)
    
    if n_vars == 0:
        return {
            'embeddable': False,
            'n_vars': 0,
            'n_interactions': 0,
            'embed_time': 0,
            'chain_length_avg': 0,
            'chain_length_max': 0,
            'physical_qubits': 0,
            'error': 'Empty BQM'
        }
    
    # Build source graph
    source_graph = nx.Graph()
    source_graph.add_nodes_from(bqm.variables)
    for (u, v) in bqm.quadratic:
        source_graph.add_edge(u, v)
    
    start_time = time.time()
    try:
        embedding = minorminer.find_embedding(
            source_graph.edges(),
            target_graph.edges(),
            timeout=timeout,
            random_seed=42
        )
        embed_time = time.time() - start_time
        
        if embedding:
            chain_lengths = [len(chain) for chain in embedding.values()]
            return {
                'embeddable': True,
                'n_vars': n_vars,
                'n_interactions': n_interactions,
                'embed_time': embed_time,
                'chain_length_avg': np.mean(chain_lengths),
                'chain_length_max': max(chain_lengths),
                'physical_qubits': sum(chain_lengths),
                'error': None
            }
        else:
            return {
                'embeddable': False,
                'n_vars': n_vars,
                'n_interactions': n_interactions,
                'embed_time': embed_time,
                'chain_length_avg': 0,
                'chain_length_max': 0,
                'physical_qubits': 0,
                'error': 'No embedding found'
            }
    except Exception as e:
        return {
            'embeddable': False,
            'n_vars': n_vars,
            'n_interactions': n_interactions,
            'embed_time': time.time() - start_time,
            'chain_length_avg': 0,
            'chain_length_max': 0,
            'physical_qubits': 0,
            'error': str(e)
        }


# ============================================================================
# GUROBI SOLVERS
# ============================================================================

def solve_full_cqm_gurobi(cqm, data, timeout=60):
    """Solve full CQM with Gurobi (ground truth)."""
    if not GUROBI_AVAILABLE:
        return {'success': False, 'error': 'Gurobi not available'}
    
    foods = data['foods']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    patch_names = data['patch_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
    max_plots_per_crop = data.get('max_plots_per_crop', None)
    group_name_mapping = data.get('group_name_mapping', {})
    reverse_mapping = {v: k for k, v in group_name_mapping.items()}
    total_area = sum(land_availability.values())
    
    model = gp.Model("FullCQM")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    Y = {}
    for patch in patch_names:
        for food in foods:
            Y[(patch, food)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{patch}_{food}")
    
    U = {}
    for food in foods:
        U[food] = model.addVar(vtype=GRB.BINARY, name=f"U_{food}")
    
    obj = 0
    for patch in patch_names:
        patch_area = land_availability[patch]
        for food in foods:
            obj += food_benefits[food] * patch_area * Y[(patch, food)]
    obj = obj / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    for patch in patch_names:
        model.addConstr(gp.quicksum(Y[(patch, food)] for food in foods) <= 1)
    
    for food in foods:
        for patch in patch_names:
            model.addConstr(U[food] >= Y[(patch, food)])
    
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        if foods_in_group:
            group_sum = gp.quicksum(U[f] for f in foods_in_group if f in U)
            if 'min' in limits and limits['min'] > 0:
                model.addConstr(group_sum >= limits['min'])
            if 'max' in limits:
                model.addConstr(group_sum <= limits['max'])
    
    # Max plots per crop (diversity constraint)
    if max_plots_per_crop is not None:
        for food in foods:
            model.addConstr(gp.quicksum(Y[(patch, food)] for patch in patch_names) <= max_plots_per_crop)
    
    start = time.time()
    model.optimize()
    solve_time = time.time() - start
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        solution = {}
        for patch in patch_names:
            for food in foods:
                solution[f"Y_{patch}_{food}"] = int(Y[(patch, food)].X)
        for food in foods:
            solution[f"U_{food}"] = int(U[food].X)
        
        return {
            'objective': model.ObjVal,
            'solve_time': solve_time,
            'solution': solution,
            'status': 'OPTIMAL' if model.Status == GRB.OPTIMAL else 'SUBOPTIMAL',
            'success': True
        }
    
    return {
        'objective': 0,
        'solve_time': solve_time,
        'solution': {},
        'status': f'Status {model.Status}',
        'success': False
    }


def solve_partition_gurobi(partition, data, fixed_vars=None, timeout=30):
    """Solve a single partition with Gurobi."""
    if not GUROBI_AVAILABLE:
        return {'success': False, 'solution': {var: 0 for var in partition}}
    
    if fixed_vars is None:
        fixed_vars = {}
    
    foods = data['foods']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    patch_names = data['patch_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
    group_name_mapping = data.get('group_name_mapping', {})
    reverse_mapping = {v: k for k, v in group_name_mapping.items()}
    total_area = sum(land_availability.values())
    
    model = gp.Model("Partition")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    gp_vars = {}
    for var_name in partition:
        gp_vars[var_name] = model.addVar(vtype=GRB.BINARY, name=var_name)
    
    # Objective
    obj = 0
    for patch in patch_names:
        patch_area = land_availability[patch]
        for food in foods:
            var_name = f"Y_{patch}_{food}"
            if var_name in partition:
                obj += food_benefits[food] * patch_area * gp_vars[var_name]
    obj = obj / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    def get_var(var_name):
        if var_name in gp_vars:
            return gp_vars[var_name]
        elif var_name in fixed_vars:
            return fixed_vars[var_name]
        else:
            return None
    
    # Constraints
    for patch in patch_names:
        patch_y_vars = []
        fixed_count = 0
        skip = False
        
        for food in foods:
            var_name = f"Y_{patch}_{food}"
            var_val = get_var(var_name)
            
            if var_val is None:
                skip = True
                break
            elif isinstance(var_val, (int, float)):
                fixed_count += var_val
            else:
                patch_y_vars.append(var_val)
        
        if not skip and patch_y_vars:
            model.addConstr(gp.quicksum(patch_y_vars) + fixed_count <= 1)
    
    for food in foods:
        u_name = f"U_{food}"
        u_var = get_var(u_name)
        if u_var is None:
            continue
            
        for patch in patch_names:
            y_name = f"Y_{patch}_{food}"
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
    
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        
        group_vars = []
        fixed_sum = 0
        skip = False
        
        for food in foods_in_group:
            var_name = f"U_{food}"
            var_val = get_var(var_name)
            
            if var_val is None:
                skip = True
                break
            elif isinstance(var_val, (int, float)):
                fixed_sum += var_val
            else:
                group_vars.append(var_val)
        
        if not skip:
            if group_vars:
                group_sum = gp.quicksum(group_vars) + fixed_sum
            else:
                group_sum = fixed_sum
            if 'min' in limits and limits['min'] > 0:
                model.addConstr(group_sum >= limits['min'])
            if 'max' in limits:
                model.addConstr(group_sum <= limits['max'])
    
    # Max plots per crop (diversity constraint)
    max_plots_per_crop = data.get('max_plots_per_crop', None)
    if max_plots_per_crop is not None:
        for food in foods:
            y_vars_this_partition = []
            fixed_count = 0
            
            for patch in patch_names:
                var_name = f"Y_{patch}_{food}"
                if var_name in partition:
                    y_vars_this_partition.append(gp_vars[var_name])
                elif var_name in fixed_vars:
                    fixed_count += fixed_vars[var_name]
            
            if y_vars_this_partition:
                model.addConstr(gp.quicksum(y_vars_this_partition) + fixed_count <= max_plots_per_crop)
    
    model.optimize()
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        solution = {var_name: int(gp_vars[var_name].X) for var_name in partition}
        return {'success': True, 'solution': solution, 'objective': model.ObjVal}
    
    return {'success': False, 'solution': {var_name: 0 for var_name in partition}}


def solve_partitioned_cqm(partitions, data, timeout_per_partition=30):
    """Solve CQM by partitions sequentially."""
    all_solutions = {}
    partition_times = []
    
    for i, partition in enumerate(partitions):
        if len(partition) == 0:
            continue
        
        start = time.time()
        result = solve_partition_gurobi(partition, data, fixed_vars=all_solutions, timeout=timeout_per_partition)
        partition_times.append(time.time() - start)
        
        if result['success']:
            all_solutions.update(result['solution'])
        else:
            for var_name in partition:
                all_solutions[var_name] = 0
    
    return all_solutions, sum(partition_times), partition_times


def calculate_objective(solution, data):
    """Calculate objective value from solution."""
    foods = data['foods']
    food_benefits = data['food_benefits']
    patch_names = data['patch_names']
    land_availability = data['land_availability']
    total_area = sum(land_availability.values())
    
    obj = 0
    for patch in patch_names:
        patch_area = land_availability[patch]
        for food in foods:
            var_name = f"Y_{patch}_{food}"
            if solution.get(var_name, 0) == 1:
                obj += food_benefits[food] * patch_area
    
    return obj / total_area


def check_violations(solution, data):
    """Check constraint violations."""
    foods = data['foods']
    food_groups = data['food_groups']
    patch_names = data['patch_names']
    food_group_constraints = data['food_group_constraints']
    max_plots_per_crop = data.get('max_plots_per_crop', None)
    group_name_mapping = data.get('group_name_mapping', {})
    reverse_mapping = {v: k for k, v in group_name_mapping.items()}
    
    violations = []
    
    # One per patch
    for patch in patch_names:
        count = sum(1 for food in foods if solution.get(f"Y_{patch}_{food}", 0) == 1)
        if count > 1:
            violations.append(f"Patch {patch}: {count} foods")
    
    # Food group constraints
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        selected = set()
        for food in foods_in_group:
            if solution.get(f"U_{food}", 0) == 1:
                selected.add(food)
            else:
                for patch in patch_names:
                    if solution.get(f"Y_{patch}_{food}", 0) == 1:
                        selected.add(food)
                        break
        
        count = len(selected)
        if count < limits.get('min', 0):
            violations.append(f"Group {constraint_group}: {count} < min {limits['min']}")
        if count > limits.get('max', 999):
            violations.append(f"Group {constraint_group}: {count} > max {limits['max']}")
    
    # Max plots per crop
    if max_plots_per_crop is not None:
        for food in foods:
            count = sum(1 for patch in patch_names if solution.get(f"Y_{patch}_{food}", 0) == 1)
            if count > max_plots_per_crop:
                violations.append(f"Food {food}: {count} > max {max_plots_per_crop}")
    
    return violations


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print(f"CQM PARTITION BENCHMARK: {N_FARMS} farms × {N_FOODS} foods")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data = load_problem_data(N_FARMS)
    
    # Build CQM
    print("\n[3/4] Building CQM...")
    cqm_start = time.time()
    cqm, Y, U = build_cqm(data)
    cqm_build_time = time.time() - cqm_start
    
    n_vars = len(cqm.variables)
    n_constraints = len(cqm.constraints)
    print(f"  CQM Variables: {n_vars}")
    print(f"  CQM Constraints: {n_constraints}")
    print(f"  Build time: {cqm_build_time:.3f}s")
    
    # Get target graph for embedding
    print("\n  Getting Pegasus graph for embedding analysis...")
    target_graph = get_pegasus_graph()
    print(f"  Pegasus qubits: {target_graph.number_of_nodes()}")
    print(f"  Pegasus couplers: {target_graph.number_of_edges()}")
    
    # Define partition methods - comprehensive list
    partition_methods = [
        ("None", partition_none),
        ("PlotBased", partition_plot_based),
        ("Spectral(4)", lambda c, d: partition_spectral(c, d, 4)),
        ("Louvain", partition_louvain),
        ("MasterSubproblem", partition_master_subproblem),
        ("Multilevel(2)", lambda c, d: partition_multilevel(c, d, 2)),
        ("Multilevel(5)", lambda c, d: partition_multilevel(c, d, 5)),
        ("Cutset(1)", lambda c, d: partition_cutset(c, d, 1)),
        ("Cutset(2)", lambda c, d: partition_cutset(c, d, 2)),
        ("SpatialGrid(5)", lambda c, d: partition_spatial_grid(c, d, 5)),
        ("EnergyImpact(100)", lambda c, d: partition_energy_impact(c, d, 100)),
        ("EnergyImpact(200)", lambda c, d: partition_energy_impact(c, d, 200)),
        ("FoodGroupBased", partition_food_group_based),  # Expected to fail!
    ]
    
    # Solve ground truth
    print("\n[4/4] Running benchmark...")
    print("\n  [Ground Truth] Solving full CQM with Gurobi...")
    gt_result = solve_full_cqm_gurobi(cqm, data, timeout=SOLVE_TIMEOUT)
    print(f"    Objective: {gt_result['objective']:.6f}")
    print(f"    Solve time: {gt_result['solve_time']:.3f}s")
    
    # Results storage
    results = {
        'problem': {
            'n_farms': N_FARMS,
            'n_foods': N_FOODS,
            'n_vars': n_vars,
            'n_constraints': n_constraints,
            'cqm_build_time': cqm_build_time
        },
        'ground_truth': {
            'objective': gt_result['objective'],
            'solve_time': gt_result['solve_time'],
            'status': gt_result['status']
        },
        'methods': {}
    }
    
    # Test each partition method
    for method_name, partition_fn in partition_methods:
        print(f"\n  [{method_name}] Testing...")
        
        # Partition
        partition_start = time.time()
        partitions, actual_name = partition_fn(cqm, data)
        partition_time = time.time() - partition_start
        
        if partitions is None:
            print(f"    SKIPPED: {actual_name}")
            continue
        
        n_partitions = len(partitions)
        partition_sizes = [len(p) for p in partitions]
        
        print(f"    Partitions: {n_partitions}")
        print(f"    Sizes: {partition_sizes[:5]}{'...' if len(partition_sizes) > 5 else ''}")
        print(f"    Partition time: {partition_time:.3f}s")
        
        # Embedding analysis for each partition
        print(f"    Embedding analysis...")
        embedding_results = []
        total_embed_time = 0
        all_embeddable = True
        total_physical_qubits = 0
        max_chain_length = 0
        
        for i, partition in enumerate(partitions):
            bqm = partition_to_bqm(partition, data)
            embed_result = study_embedding(bqm, target_graph, timeout=EMBEDDING_TIMEOUT)
            embedding_results.append(embed_result)
            total_embed_time += embed_result['embed_time']
            
            if embed_result['embeddable']:
                total_physical_qubits += embed_result['physical_qubits']
                max_chain_length = max(max_chain_length, embed_result['chain_length_max'])
            else:
                all_embeddable = False
        
        embeddable_count = sum(1 for e in embedding_results if e['embeddable'])
        embed_status = f"✅ {embeddable_count}/{n_partitions}" if all_embeddable else f"⚠️ {embeddable_count}/{n_partitions}"
        print(f"    Embeddable: {embed_status}")
        print(f"    Total embed time: {total_embed_time:.3f}s")
        if all_embeddable:
            print(f"    Physical qubits: {total_physical_qubits}")
            print(f"    Max chain length: {max_chain_length}")
        
        # Solve with Gurobi
        print(f"    Solving partitions with Gurobi...")
        solve_start = time.time()
        solution, total_solve_time, _ = solve_partitioned_cqm(partitions, data, timeout_per_partition=SOLVE_TIMEOUT)
        solve_time = time.time() - solve_start
        
        # Evaluate
        objective = calculate_objective(solution, data)
        violations = check_violations(solution, data)
        gap = ((gt_result['objective'] - objective) / gt_result['objective'] * 100) if gt_result['objective'] > 0 else 0
        
        viol_status = f"✅ 0" if len(violations) == 0 else f"❌ {len(violations)}"
        print(f"    Objective: {objective:.6f} (gap: {gap:+.1f}%)")
        print(f"    Solve time: {solve_time:.3f}s")
        print(f"    Violations: {viol_status}")
        
        # Store results
        results['methods'][method_name] = {
            'n_partitions': n_partitions,
            'partition_sizes': partition_sizes,
            'partition_time': partition_time,
            'embedding': {
                'embeddable_count': embeddable_count,
                'all_embeddable': all_embeddable,
                'total_embed_time': total_embed_time,
                'total_physical_qubits': total_physical_qubits if all_embeddable else None,
                'max_chain_length': max_chain_length if all_embeddable else None
            },
            'solving': {
                'objective': objective,
                'gap_percent': gap,
                'solve_time': solve_time,
                'n_violations': len(violations)
            },
            'total_time': partition_time + total_embed_time + solve_time
        }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = OUTPUT_DIR / f"cqm_partition_benchmark_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary table
    print("\n" + "=" * 120)
    print("SUMMARY: CQM PARTITION BENCHMARK")
    print("=" * 120)
    print(f"Problem: {N_FARMS} farms × {N_FOODS} foods = {n_vars} variables, {n_constraints} constraints")
    print(f"Ground Truth: obj={gt_result['objective']:.6f}, time={gt_result['solve_time']:.3f}s")
    print()
    
    print(f"{'Method':<15} {'Parts':>6} {'Part Time':>10} {'Embed':>12} {'Embed Time':>12} {'Phys Qubits':>12} {'Solve Time':>12} {'Objective':>10} {'Gap':>8} {'Viol':>6}")
    print("-" * 120)
    
    for method_name, method_data in results['methods'].items():
        emb = method_data['embedding']
        sol = method_data['solving']
        
        embed_str = f"{emb['embeddable_count']}/{method_data['n_partitions']}"
        qubits_str = str(emb['total_physical_qubits']) if emb['all_embeddable'] else "N/A"
        viol_str = "✅" if sol['n_violations'] == 0 else f"❌{sol['n_violations']}"
        
        print(f"{method_name:<15} {method_data['n_partitions']:>6} {method_data['partition_time']:>9.3f}s {embed_str:>12} {emb['total_embed_time']:>11.3f}s {qubits_str:>12} {sol['solve_time']:>11.3f}s {sol['objective']:>10.6f} {sol['gap_percent']:>+7.1f}% {viol_str:>6}")
    
    print("=" * 120)
    
    # Key insights
    print("\n📊 KEY INSIGHTS:")
    print("-" * 80)
    
    # Find best embeddable method
    embeddable_methods = [(name, data) for name, data in results['methods'].items() 
                          if data['embedding']['all_embeddable']]
    if embeddable_methods:
        best_embed = min(embeddable_methods, key=lambda x: x[1]['embedding']['total_physical_qubits'])
        print(f"  🏆 Best for QPU: {best_embed[0]} ({best_embed[1]['embedding']['total_physical_qubits']} physical qubits)")
    
    # Find fastest
    fastest = min(results['methods'].items(), key=lambda x: x[1]['total_time'])
    print(f"  ⚡ Fastest overall: {fastest[0]} ({fastest[1]['total_time']:.3f}s total)")
    
    # Find best quality
    feasible = [(name, data) for name, data in results['methods'].items() 
                if data['solving']['n_violations'] == 0]
    if feasible:
        best_quality = min(feasible, key=lambda x: abs(x[1]['solving']['gap_percent']))
        print(f"  🎯 Best solution quality: {best_quality[0]} (gap: {best_quality[1]['solving']['gap_percent']:+.1f}%)")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
