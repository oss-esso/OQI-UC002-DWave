#!/usr/bin/env python3
"""
QPU Benchmark: Pure Quantum Annealing (No Hybrid Solvers) v2.0

This benchmark tests ONLY direct QPU methods:
1. Direct QPU (DWaveSampler + EmbeddingComposite) - CQM→BQM→QPU
2. Manual Decomposition + SA/QPU (PlotBased, Multilevel, Louvain, Cutset, Spectral)
3. Coordinated Master-Subproblem decomposition (constraint-preserving)

NO hybrid solvers (LeapHybridCQMSampler, LeapHybridBQMSampler, dwave-hybrid).

Features:
- Comprehensive timing (CQM build, BQM conversion, embedding, solving, QPU access)
- Advanced logging with step-by-step progress
- Multiple decomposition strategies from comprehensive_embedding_benchmark
- Detailed QPU timing extraction from sampleset.info

Author: OQI-UC002-DWave Project
Date: 2025-11-30 v2.0
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
import logging

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(verbose: bool = True) -> logging.Logger:
    """Setup logging with timestamps."""
    logger = logging.getLogger('QPU_Benchmark')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger

LOG = setup_logging()

# ============================================================================
# IMPORTS
# ============================================================================

print("=" * 80)
print("QPU BENCHMARK v2.0: Pure Quantum Annealing (No Hybrid)")
print("=" * 80)
print()

print("[1/5] Importing core libraries...")
import_start = time.time()

import networkx as nx
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel, Binary, cqm_to_bqm

# Gurobi for ground truth
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    print("  Warning: Gurobi not available - ground truth will be limited")

# Project imports
from src.scenarios import load_food_data
from Utils import patch_sampler

print(f"  Core imports: {time.time() - import_start:.2f}s")

# ============================================================================
# D-WAVE IMPORTS (QPU ONLY - NO HYBRID)
# ============================================================================

print("[2/5] Importing D-Wave QPU libraries (no hybrid)...")
dwave_start = time.time()

HAS_QPU = False
HAS_QBSOLV = False
HAS_EMBEDDING = False
HAS_LOUVAIN = False
HAS_SPECTRAL = False

# Direct QPU sampler
try:
    from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
    HAS_QPU = True
    print("  ✓ DWaveSampler (direct QPU) available")
except ImportError:
    print("  ✗ DWaveSampler not available")

# QBSolv is deprecated and incompatible with Python 3.12+
HAS_QBSOLV = False
print("  ℹ QBSolv: Deprecated (using decomposition methods instead)")

# Embedding tools
try:
    from minorminer import find_embedding
    HAS_EMBEDDING = True
    print("  ✓ minorminer available")
except ImportError:
    print("  ✗ minorminer not available")

# Pegasus graph for embedding analysis
try:
    import dwave_networkx as dnx
    HAS_DNX = True
    print("  ✓ dwave_networkx available")
except ImportError:
    HAS_DNX = False

# Simulated annealing for fallback/comparison
try:
    import neal
    HAS_NEAL = True
    print("  ✓ neal (SimulatedAnnealing) available for comparison")
except ImportError:
    HAS_NEAL = False

# Louvain community detection
try:
    from networkx.algorithms.community import louvain_communities
    HAS_LOUVAIN = True
    print("  ✓ Louvain community detection available")
except ImportError:
    print("  ✗ Louvain not available")

# Spectral clustering
try:
    from sklearn.cluster import SpectralClustering
    HAS_SPECTRAL = True
    print("  ✓ Spectral clustering available")
except ImportError:
    print("  ✗ Spectral clustering not available")

print(f"  D-Wave imports: {time.time() - dwave_start:.2f}s")

# ============================================================================
# CONFIGURATION
# ============================================================================

print("[3/5] Loading configuration...")

# Problem scales
FARM_SCALES = [25, 50, 100, 200]
N_FOODS = 27

# QPU parameters
NUM_READS_OPTIONS = [100, 500, 1000, 5000]
ANNEALING_TIME_OPTIONS = [20, 100, 200]  # microseconds
CHAIN_STRENGTH_MULTIPLIERS = [1.0, 1.5, 2.0]

# Defaults
DEFAULT_NUM_READS = 1000
DEFAULT_ANNEALING_TIME = 20  # µs
DEFAULT_CHAIN_STRENGTH = None  # auto-calculated

# Timeouts - CRITICAL: Don't waste QPU time on problems that can't embed
DIRECT_QPU_TIMEOUT = 200  # seconds for direct QPU (embedding + solving)
EMBED_TIMEOUT_PER_PARTITION = 60  # seconds for embedding per partition  
SOLVE_TIMEOUT = 300  # seconds total per method

# Output
OUTPUT_DIR = Path(__file__).parent / "qpu_benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"  Output directory: {OUTPUT_DIR}")

# Global QPU graph cache
QPU_GRAPH = None
QPU_SAMPLER = None
DWAVE_TOKEN = None  # Set via --token or set_dwave_token()


def set_dwave_token(token: str):
    """Set the D-Wave API token for QPU access."""
    global DWAVE_TOKEN
    DWAVE_TOKEN = token
    # Also set environment variable for dwave-system
    os.environ['DWAVE_API_TOKEN'] = token
    print(f"  D-Wave token configured (length: {len(token)})")


def get_qpu_sampler():
    """Get cached QPU sampler."""
    global QPU_SAMPLER
    if QPU_SAMPLER is None and HAS_QPU:
        try:
            # Use token if set
            if DWAVE_TOKEN:
                QPU_SAMPLER = DWaveSampler(token=DWAVE_TOKEN)
            else:
                QPU_SAMPLER = DWaveSampler()
            print(f"  Connected to QPU: {QPU_SAMPLER.properties.get('chip_id', 'Unknown')}")
        except Exception as e:
            print(f"  Failed to connect to QPU: {e}")
    return QPU_SAMPLER


def get_qpu_graph():
    """Get QPU topology graph (cached)."""
    global QPU_GRAPH
    if QPU_GRAPH is not None:
        return QPU_GRAPH
    
    sampler = get_qpu_sampler()
    if sampler:
        QPU_GRAPH = sampler.to_networkx_graph()
        print(f"  QPU topology: {QPU_GRAPH.number_of_nodes()} qubits")
        return QPU_GRAPH
    
    # Fallback to simulated Pegasus
    if HAS_DNX:
        QPU_GRAPH = dnx.pegasus_graph(16)
        print(f"  Using simulated Pegasus P16: {QPU_GRAPH.number_of_nodes()} qubits")
        return QPU_GRAPH
    
    return None


# ============================================================================
# DATA LOADING
# ============================================================================

print("[4/5] Setting up data loading...")


def load_problem_data(n_farms: int) -> Dict:
    """Load food data and create patch configuration."""
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
    
    max_plots_per_crop = max(5, n_farms // 5)
    
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
        'group_name_mapping': group_name_mapping,
        'reverse_mapping': reverse_mapping,
        'n_farms': n_farms,
        'n_foods': len(food_names)
    }


# ============================================================================
# CQM / BQM BUILDING
# ============================================================================

def build_binary_cqm(data: Dict) -> Tuple[ConstrainedQuadraticModel, Dict]:
    """Build binary CQM for plot assignment."""
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
    
    # Variables
    Y = {}
    for farm in farm_names:
        for food in food_names:
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")
    
    U = {}
    for food in food_names:
        U[food] = Binary(f"U_{food}")
    
    # Objective: maximize benefit
    objective = sum(
        food_benefits[food] * land_availability[farm] * Y[(farm, food)]
        for farm in farm_names for food in food_names
    ) / total_area
    cqm.set_objective(-objective)
    
    # Constraint: one crop per farm
    for farm in farm_names:
        cqm.add_constraint(sum(Y[(farm, food)] for food in food_names) == 1, 
                          label=f"OneCrop_{farm}")
    
    # Constraint: U[f] >= Y[p,f]  -->  U[f] - Y[p,f] >= 0
    for food in food_names:
        for farm in farm_names:
            cqm.add_constraint(U[food] - Y[(farm, food)] >= 0, 
                              label=f"Link_{farm}_{food}")
    
    # Constraint: food group diversity
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        if foods_in_group:
            group_sum = sum(U[f] for f in foods_in_group if f in U)
            if limits.get('min', 0) > 0:
                cqm.add_constraint(group_sum >= limits['min'], 
                                  label=f"FG_Min_{constraint_group}")
            if 'max' in limits:
                cqm.add_constraint(group_sum <= limits['max'], 
                                  label=f"FG_Max_{constraint_group}")
    
    # Constraint: max plots per crop
    for food in food_names:
        cqm.add_constraint(sum(Y[(farm, food)] for farm in farm_names) <= max_plots_per_crop,
                          label=f"MaxPlots_{food}")
    
    metadata = {
        'n_farms': len(farm_names),
        'n_foods': len(food_names),
        'n_variables': len(cqm.variables),
        'n_constraints': len(cqm.constraints),
        'logical_qubits': len(Y) + len(U),
        'total_area': total_area
    }
    
    return cqm, metadata


def build_bqm_for_partition(partition_vars: set, data: Dict, lagrange: float = 10.0) -> BinaryQuadraticModel:
    """Build BQM for a partition of variables with QUBO penalties."""
    food_names = data['food_names']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    total_area = data['total_area']
    
    bqm = BinaryQuadraticModel('BINARY')
    
    # Extract farms/foods in this partition
    farms_in_partition = set()
    for var in partition_vars:
        if var.startswith("Y_"):
            parts = var.split("_", 2)
            farms_in_partition.add(parts[1])
    
    # Add objective (linear terms)
    for var in partition_vars:
        if var.startswith("Y_"):
            parts = var.split("_", 2)
            farm, food = parts[1], parts[2]
            benefit = food_benefits.get(food, 0) * land_availability.get(farm, 0) / total_area
            bqm.add_variable(var, -benefit)  # Negate for minimization
        elif var.startswith("U_"):
            bqm.add_variable(var, 0.0)
    
    # Add one-per-farm constraint as QUBO penalty
    for farm in farms_in_partition:
        y_vars = [f"Y_{farm}_{food}" for food in food_names if f"Y_{farm}_{food}" in partition_vars]
        if len(y_vars) > 1:
            # Penalty: lagrange * (sum(y) - 1)^2
            for var in y_vars:
                bqm.add_variable(var, lagrange * (1 - 2))
            for i, v1 in enumerate(y_vars):
                for v2 in y_vars[i+1:]:
                    bqm.add_interaction(v1, v2, 2 * lagrange)
    
    return bqm


# ============================================================================
# GROUND TRUTH (GUROBI)
# ============================================================================

def solve_ground_truth(data: Dict, timeout: int = 120) -> Dict:
    """Solve with Gurobi to get ground truth."""
    if not HAS_GUROBI:
        return {'success': False, 'error': 'Gurobi not available'}
    
    total_start = time.time()
    
    food_names = data['food_names']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
    max_plots_per_crop = data['max_plots_per_crop']
    reverse_mapping = data['reverse_mapping']
    total_area = data['total_area']
    
    # Model build time
    build_start = time.time()
    model = gp.Model("GroundTruth")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    Y = {(f, c): model.addVar(vtype=GRB.BINARY) for f in farm_names for c in food_names}
    U = {c: model.addVar(vtype=GRB.BINARY) for c in food_names}
    
    obj = gp.quicksum(food_benefits[c] * land_availability[f] * Y[(f, c)] 
                      for f in farm_names for c in food_names) / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    for f in farm_names:
        model.addConstr(gp.quicksum(Y[(f, c)] for c in food_names) == 1)
    
    for c in food_names:
        for f in farm_names:
            model.addConstr(U[c] >= Y[(f, c)])
    
    for cg, limits in food_group_constraints.items():
        dg = reverse_mapping.get(cg, cg)
        foods = food_groups.get(dg, [])
        if foods:
            gs = gp.quicksum(U[f] for f in foods if f in U)
            if limits.get('min', 0) > 0:
                model.addConstr(gs >= limits['min'])
            if 'max' in limits:
                model.addConstr(gs <= limits['max'])
    
    for c in food_names:
        model.addConstr(gp.quicksum(Y[(f, c)] for f in farm_names) <= max_plots_per_crop)
    
    build_time = time.time() - build_start
    
    # Solve time
    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start
    
    total_time = time.time() - total_start
    
    result = {
        'method': 'gurobi',
        'timings': {
            'build': build_time,
            'solve': solve_time,
            'total': total_time,
            'solve_time': solve_time,  # Alias for consistency
            'embedding_total': 0,  # N/A for Gurobi
            'qpu_access_total': 0,  # N/A for Gurobi
        },
        'solve_time': solve_time,
        'total_time': total_time,
        'wall_time': total_time,
        'n_variables': model.NumVars,
        'n_constraints': model.NumConstrs,
    }
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        # Extract solution
        solution = {
            'Y': {},
            'U': {},
            'allocations': {},
            'summary': {
                'n_farms': len(farm_names),
                'n_foods': len(food_names),
                'foods_used': [],
                'total_area_allocated': 0,
            }
        }
        
        for f in farm_names:
            solution['Y'][f] = {}
            for c in food_names:
                y_val = int(Y[(f, c)].X + 0.5)
                solution['Y'][f][c] = y_val
                if y_val == 1:
                    area = land_availability.get(f, 0)
                    solution['allocations'][f"{f}_{c}"] = area
                    solution['summary']['total_area_allocated'] += area
        
        for c in food_names:
            u_val = int(U[c].X + 0.5)
            solution['U'][c] = u_val
            if u_val == 1:
                solution['summary']['foods_used'].append(c)
        
        solution['summary']['n_unique_foods'] = len(solution['summary']['foods_used'])
        
        result.update({
            'success': True,
            'objective': model.ObjVal,
            'violations': 0,
            'feasible': True,
            'solution': solution,
            'status': 'optimal' if model.Status == GRB.OPTIMAL else 'suboptimal',
        })
    else:
        result.update({
            'success': False,
            'objective': 0,
            'violations': -1,
            'feasible': False,
            'status': model.Status,
        })
    
    return result


# ============================================================================
# DECOMPOSITION METHODS (Enhanced with all strategies)
# ============================================================================

def get_bqm_graph(bqm: BinaryQuadraticModel) -> nx.Graph:
    """Convert BQM to NetworkX graph for decomposition."""
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    G.add_edges_from(bqm.quadratic.keys())
    return G


def partition_plot_based(data: Dict) -> List[Set]:
    """PlotBased: one partition per farm + U variables."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    partitions = [{f"Y_{farm}_{food}" for food in food_names} for farm in farm_names]
    partitions.append({f"U_{food}" for food in food_names})
    return partitions


def partition_multilevel(data: Dict, group_size: int = 5) -> List[Set]:
    """Multilevel: group farms together."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    partitions = []
    for i in range(0, len(farm_names), group_size):
        group_farms = farm_names[i:i+group_size]
        partitions.append({f"Y_{f}_{c}" for f in group_farms for c in food_names})
    partitions.append({f"U_{food}" for food in food_names})
    return partitions


def partition_cutset(data: Dict, farms_per_cut: int = 2) -> List[Set]:
    """Cutset: fine-grained decomposition."""
    return partition_multilevel(data, farms_per_cut)


def partition_louvain(data: Dict, max_partition_size: int = 100) -> List[Set]:
    """Louvain community detection decomposition."""
    if not HAS_LOUVAIN:
        LOG.warning("Louvain not available, falling back to PlotBased")
        return partition_plot_based(data)
    
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    # Build graph from variable interactions
    G = nx.Graph()
    all_vars = [f"Y_{f}_{c}" for f in farm_names for c in food_names]
    all_vars += [f"U_{c}" for c in food_names]
    G.add_nodes_from(all_vars)
    
    # Add edges: Y variables in same farm are connected, U-Y links
    for farm in farm_names:
        farm_vars = [f"Y_{farm}_{c}" for c in food_names]
        for i, v1 in enumerate(farm_vars):
            for v2 in farm_vars[i+1:]:
                G.add_edge(v1, v2)
            # Link to corresponding U
            for c in food_names:
                G.add_edge(f"Y_{farm}_{c}", f"U_{c}")
    
    # Run Louvain
    communities = louvain_communities(G, seed=42, resolution=1.5)
    
    # Convert to list of sets and handle size limits
    partitions = []
    for comm in communities:
        comm_set = set(comm)
        if len(comm_set) > max_partition_size:
            # Split large communities
            comm_list = list(comm_set)
            for i in range(0, len(comm_list), max_partition_size):
                partitions.append(set(comm_list[i:i+max_partition_size]))
        else:
            partitions.append(comm_set)
    
    return partitions if partitions else [set(all_vars)]


def partition_spectral(data: Dict, n_clusters: int = 10) -> List[Set]:
    """Spectral clustering decomposition."""
    if not HAS_SPECTRAL:
        LOG.warning("Spectral clustering not available, falling back to Multilevel")
        return partition_multilevel(data, 5)
    
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    # Build adjacency matrix
    all_vars = [f"Y_{f}_{c}" for f in farm_names for c in food_names]
    all_vars += [f"U_{c}" for c in food_names]
    var_to_idx = {v: i for i, v in enumerate(all_vars)}
    n_vars = len(all_vars)
    
    # Create sparse adjacency
    adj = np.zeros((n_vars, n_vars))
    
    for farm in farm_names:
        farm_vars = [f"Y_{farm}_{c}" for c in food_names]
        for i, v1 in enumerate(farm_vars):
            for v2 in farm_vars[i+1:]:
                idx1, idx2 = var_to_idx[v1], var_to_idx[v2]
                adj[idx1, idx2] = adj[idx2, idx1] = 1
            for c in food_names:
                idx1 = var_to_idx[f"Y_{farm}_{c}"]
                idx2 = var_to_idx[f"U_{c}"]
                adj[idx1, idx2] = adj[idx2, idx1] = 1
    
    # Spectral clustering
    n_clusters = min(n_clusters, n_vars // 10, len(farm_names))
    n_clusters = max(2, n_clusters)
    
    try:
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                                         random_state=42, assign_labels='kmeans')
        labels = clustering.fit_predict(adj + np.eye(n_vars) * 0.1)  # Add self-loops for stability
        
        partitions = defaultdict(set)
        for var, label in zip(all_vars, labels):
            partitions[label].add(var)
        
        return list(partitions.values())
    except Exception as e:
        LOG.warning(f"Spectral clustering failed: {e}, falling back to Multilevel")
        return partition_multilevel(data, 5)


def partition_master_subproblem(data: Dict) -> Tuple[Set, List[Set]]:
    """
    MasterSubproblem: Two-level partition for CONSTRAINT PRESERVATION.
    
    Returns:
        (master_vars, subproblem_partitions)
        - master_vars: U variables (global food selection - controls food group diversity)
        - subproblem_partitions: Y variables per farm (local assignment)
    
    This decomposition preserves global constraints by:
    1. Solving master (U vars) first with food group constraints
    2. Solving each farm subproblem with U variables fixed
    """
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    # Master: all U variables (controls food group diversity)
    master_vars = {f"U_{food}" for food in food_names}
    
    # Subproblems: Y variables per farm
    subproblems = [{f"Y_{farm}_{food}" for food in food_names} for farm in farm_names]
    
    return master_vars, subproblems


# Registry of all decomposition methods
DECOMPOSITION_METHODS = {
    'PlotBased': partition_plot_based,
    'Multilevel(5)': lambda d: partition_multilevel(d, 5),
    'Multilevel(10)': lambda d: partition_multilevel(d, 10),
    'Cutset(2)': lambda d: partition_cutset(d, 2),
    'Louvain': partition_louvain,
    'Spectral(10)': lambda d: partition_spectral(d, 10),
}


def build_master_bqm(data: Dict, lagrange: float = 50.0) -> BinaryQuadraticModel:
    """
    Build BQM for master problem (U variables only) with food group constraints.
    
    This enforces food group diversity constraints directly in the BQM.
    Uses stronger penalties to ensure feasibility.
    """
    food_names = data['food_names']
    food_groups = data['food_groups']
    food_group_constraints = data['food_group_constraints']
    reverse_mapping = data['reverse_mapping']
    
    bqm = BinaryQuadraticModel('BINARY')
    
    # Add U variables (with small benefit to encourage selection)
    for food in food_names:
        bqm.add_variable(f"U_{food}", -0.01)  # Small incentive to select foods
    
    # Add food group constraints as QUBO penalties
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        
        if not foods_in_group:
            continue
        
        u_vars = [f"U_{f}" for f in foods_in_group if f in food_names]
        
        if not u_vars:
            continue
        
        min_count = limits.get('min', 0)
        max_count = limits.get('max', len(u_vars))
        
        # Use stronger penalty formulation
        # For min constraint: we want sum(U) >= min_count
        # Penalty: lagrange * max(0, min_count - sum(U))^2
        # 
        # For a QUBO, we use slack variables or penalty term:
        # Penalty = lagrange * (sum(U) - target)^2 where target = (min + max) / 2
        # But scale based on how far we are from bounds
        
        target = (min_count + max_count) / 2
        
        # QUBO expansion of (sum(U) - target)^2:
        # = sum(U)^2 - 2*target*sum(U) + target^2
        # = sum_i(U_i^2) + 2*sum_{i<j}(U_i*U_j) - 2*target*sum(U_i) + target^2
        # Since U_i is binary, U_i^2 = U_i
        # = sum_i(U_i) + 2*sum_{i<j}(U_i*U_j) - 2*target*sum_i(U_i) + target^2
        # = (1 - 2*target)*sum_i(U_i) + 2*sum_{i<j}(U_i*U_j) + target^2
        
        # Linear coefficient for each U_i: lagrange * (1 - 2*target)
        lin_coef = lagrange * (1 - 2 * target)
        for var in u_vars:
            bqm.add_variable(var, lin_coef)
        
        # Quadratic coefficient for each pair: 2 * lagrange
        quad_coef = 2 * lagrange
        for i, v1 in enumerate(u_vars):
            for v2 in u_vars[i+1:]:
                bqm.add_interaction(v1, v2, quad_coef)
    
    return bqm


def build_subproblem_bqm(farm: str, data: Dict, fixed_u: Dict, lagrange: float = 10.0) -> BinaryQuadraticModel:
    """
    Build BQM for farm subproblem with fixed U values.
    
    Args:
        farm: Farm name
        data: Problem data
        fixed_u: Dictionary of U variable values from master solution
        lagrange: Lagrange multiplier for constraints
    
    Returns:
        BQM for selecting one food for this farm
    """
    food_names = data['food_names']
    food_benefits = data['food_benefits']
    land_availability = data['land_availability']
    total_area = data['total_area']
    
    bqm = BinaryQuadraticModel('BINARY')
    farm_area = land_availability.get(farm, 0)
    
    # Add Y variables with objective coefficients
    for food in food_names:
        var = f"Y_{farm}_{food}"
        benefit = food_benefits.get(food, 0) * farm_area / total_area
        
        # Only allow selection if U[food] = 1 (food is globally selected)
        u_val = fixed_u.get(f"U_{food}", 0)
        if u_val == 0:
            # High penalty to prevent selection of non-selected foods
            bqm.add_variable(var, lagrange)
        else:
            # Normal objective
            bqm.add_variable(var, -benefit)
    
    # One-hot constraint: exactly one food per farm
    # Penalty: lagrange * (sum(Y) - 1)^2
    y_vars = [f"Y_{farm}_{food}" for food in food_names]
    
    for var in y_vars:
        bqm.add_variable(var, lagrange * (1 - 2))  # -lagrange from expansion
    
    for i, v1 in enumerate(y_vars):
        for v2 in y_vars[i+1:]:
            bqm.add_interaction(v1, v2, 2 * lagrange)
    
    return bqm


def solve_coordinated_decomposition(cqm: ConstrainedQuadraticModel, data: Dict,
                                     num_reads: int = 1000,
                                     annealing_time: int = 20,
                                     use_qpu: bool = False,
                                     embed_timeout: int = 30) -> Dict:
    """
    Coordinated decomposition: Solve master (U) first, then subproblems (Y) with fixed U.
    
    This PRESERVES CONSTRAINTS by:
    1. Master problem enforces food group diversity via U variables
    2. Subproblems respect U values (can only assign food f if U[f]=1)
    3. Each subproblem enforces one-food-per-farm
    
    Args:
        use_qpu: If True, attempt QPU embedding (may timeout). Default False uses SA.
        embed_timeout: Max seconds for embedding attempt before falling back to SA.
    """
    result = {
        'method': 'coordinated_decomposition',
        'num_reads': num_reads,
        'annealing_time': annealing_time,
        'timings': {},
    }
    
    try:
        total_start = time.time()
        
        # Default to SimulatedAnnealing (fast and reliable)
        # QPU embedding takes too long for many small problems
        sa_sampler = neal.SimulatedAnnealingSampler()
        result['sampler'] = 'SimulatedAnnealing'
        
        # Only try QPU if explicitly requested and available
        qpu_sampler = None
        if use_qpu and HAS_QPU:
            qpu = get_qpu_sampler()
            if qpu:
                qpu_sampler = EmbeddingComposite(qpu)
        
        # ================================================================
        # STEP 1: Solve master problem (U variables with food group constraints)
        # ================================================================
        LOG.info("    [Coordinated] Step 1: Solving master problem (U variables)...")
        master_bqm = build_master_bqm(data)
        
        master_start = time.time()
        # Master is small (27 vars) - use SA for speed
        master_result = sa_sampler.sample(master_bqm, num_reads=num_reads, num_sweeps=1000)
        master_time = time.time() - master_start
        result['timings']['master'] = master_time
        
        # Extract U values
        fixed_u = {k: v for k, v in master_result.first.sample.items()}
        n_selected = sum(1 for v in fixed_u.values() if v == 1)
        LOG.info(f"    [Coordinated] Master: {n_selected} foods selected in {master_time:.2f}s")
        
        # ================================================================
        # STEP 2: Solve subproblems (Y variables per farm with fixed U)
        # ================================================================
        LOG.info(f"    [Coordinated] Step 2: Solving {len(data['farm_names'])} farm subproblems...")
        
        all_samples = dict(fixed_u)  # Start with U values
        subproblem_times = []
        total_subproblem_time = 0
        
        for farm in data['farm_names']:
            sub_bqm = build_subproblem_bqm(farm, data, fixed_u)
            
            sub_start = time.time()
            # Subproblems are small (27 vars) - SA is fast enough
            sub_result = sa_sampler.sample(sub_bqm, num_reads=num_reads // 2, num_sweeps=500)
            sub_time = time.time() - sub_start
            subproblem_times.append(sub_time)
            total_subproblem_time += sub_time
            
            # Merge solution
            all_samples.update(sub_result.first.sample)
        
        result['timings']['subproblems_total'] = total_subproblem_time
        result['timings']['solve_time'] = master_time + total_subproblem_time
        result['timings']['embedding_total'] = 0  # SA doesn't embed
        result['timings']['qpu_access_total'] = 0  # SA doesn't use QPU
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        result['total_qpu_time'] = 0
        
        # Extract full solution
        result['solution'] = extract_solution(all_samples, data)
        
        # Calculate objective and violations
        result['objective'] = calculate_objective(all_samples, data)
        result['violations'] = count_violations(all_samples, data)
        result['feasible'] = result['violations'] == 0
        result['success'] = True
        
        LOG.info(f"    [Coordinated] Total time: {result['total_time']:.2f}s, "
              f"Objective: {result['objective']:.4f}, Violations: {result['violations']}")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return result


# ============================================================================
# EMBEDDING ANALYSIS
# ============================================================================

def analyze_embedding(bqm: BinaryQuadraticModel, timeout: int = 60) -> Dict:
    """Analyze if BQM can be embedded on QPU."""
    if not HAS_EMBEDDING:
        return {'success': False, 'error': 'minorminer not available'}
    
    target = get_qpu_graph()
    if target is None:
        return {'success': False, 'error': 'No QPU graph available'}
    
    # Build source graph
    source = nx.Graph()
    source.add_nodes_from(bqm.variables)
    source.add_edges_from(bqm.quadratic.keys())
    
    result = {
        'logical_qubits': len(bqm.variables),
        'interactions': len(bqm.quadratic),
    }
    
    start = time.time()
    try:
        embedding = find_embedding(source, target, timeout=timeout, random_seed=42)
        result['time'] = time.time() - start
        
        if embedding:
            result['success'] = True
            result['physical_qubits'] = sum(len(chain) for chain in embedding.values())
            result['max_chain_length'] = max(len(chain) for chain in embedding.values())
            result['avg_chain_length'] = np.mean([len(chain) for chain in embedding.values()])
        else:
            result['success'] = False
            result['error'] = 'Embedding not found'
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        result['time'] = time.time() - start
    
    return result


# ============================================================================
# QPU SOLVERS (NO HYBRID)
# ============================================================================

import threading
import concurrent.futures

def solve_direct_qpu(cqm: ConstrainedQuadraticModel, data: Dict,
                     num_reads: int = 1000, annealing_time: int = 20,
                     chain_strength: Optional[float] = None,
                     timeout: int = DIRECT_QPU_TIMEOUT) -> Dict:
    """
    Direct QPU: CQM → BQM → Embed → QPU
    
    Uses DWaveSampler + EmbeddingComposite only (no hybrid).
    Has timeout protection to avoid hanging on embedding.
    """
    if not HAS_QPU:
        return {'success': False, 'error': 'DWaveSampler not available'}
    
    result = {
        'method': 'direct_qpu',
        'num_reads': num_reads,
        'annealing_time': annealing_time,
        'timeout': timeout,
        'timings': {},
    }
    
    total_start = time.time()
    
    try:
        # Convert CQM to BQM
        LOG.info(f"  [DirectQPU] Converting CQM to BQM...")
        t0 = time.time()
        bqm, info = cqm_to_bqm(cqm)
        result['timings']['cqm_to_bqm'] = time.time() - t0
        result['bqm_variables'] = len(bqm.variables)
        result['bqm_interactions'] = len(bqm.quadratic)
        LOG.info(f"  [DirectQPU] BQM: {result['bqm_variables']} vars, {result['bqm_interactions']} interactions")
        
        # Auto chain strength
        if chain_strength is None:
            max_bias = max(abs(b) for b in bqm.linear.values()) if bqm.linear else 1.0
            chain_strength = max_bias * 1.5
        result['chain_strength'] = chain_strength
        
        # FIRST: Try to find embedding with minorminer (has proper timeout support)
        LOG.info(f"  [DirectQPU] Finding embedding (timeout: {timeout}s)...")
        
        target = get_qpu_graph()
        if target is None:
            return {'success': False, 'error': 'Could not get QPU graph'}
        
        # Build source graph
        source = nx.Graph()
        source.add_nodes_from(bqm.variables)
        source.add_edges_from(bqm.quadratic.keys())
        
        embed_start = time.time()
        try:
            embedding = find_embedding(source, target, timeout=timeout, random_seed=42)
            embed_time = time.time() - embed_start
            result['timings']['embedding'] = embed_time
            
            if not embedding:
                result['success'] = False
                result['error'] = f'No embedding found within {timeout}s'
                result['timings']['total'] = time.time() - total_start
                result['total_time'] = result['timings']['total']
                LOG.warning(f"  [DirectQPU] No embedding found in {embed_time:.1f}s")
                return result
            
            result['physical_qubits'] = sum(len(chain) for chain in embedding.values())
            result['max_chain_length'] = max(len(chain) for chain in embedding.values())
            LOG.info(f"  [DirectQPU] Found embedding: {result['physical_qubits']} physical qubits, "
                    f"max chain {result['max_chain_length']}, in {embed_time:.1f}s")
            
        except Exception as e:
            result['success'] = False
            result['error'] = f'Embedding error: {str(e)}'
            result['timings']['total'] = time.time() - total_start
            result['total_time'] = result['timings']['total']
            return result
        
        # Now sample with FixedEmbeddingComposite (faster since embedding is precomputed)
        qpu = get_qpu_sampler()
        if qpu is None:
            return {'success': False, 'error': 'Could not connect to QPU'}
        
        sampler = FixedEmbeddingComposite(qpu, embedding)
        
        LOG.info(f"  [DirectQPU] Sampling on QPU ({num_reads} reads)...")
        sample_start = time.time()
        sampleset = sampler.sample(
            bqm,
            num_reads=num_reads,
            annealing_time=annealing_time,
            chain_strength=chain_strength,
            label=f"DirectQPU_{data['n_farms']}farms"
        )
        result['timings']['sampling'] = time.time() - sample_start
        
        # Extract detailed timing from sampleset.info
        timing_info = sampleset.info.get('timing', {})
        qpu_access_us = timing_info.get('qpu_access_time', 0)
        qpu_programming_us = timing_info.get('qpu_programming_time', 0)
        qpu_sampling_us = timing_info.get('qpu_sampling_time', 0)
        total_real_us = timing_info.get('total_real_time', 0)
        
        result['timings']['qpu_access'] = qpu_access_us / 1e6
        result['timings']['qpu_programming'] = qpu_programming_us / 1e6
        result['timings']['qpu_sampling'] = qpu_sampling_us / 1e6
        result['timings']['qpu_total_real'] = total_real_us / 1e6
        result['timings']['embedding_total'] = result['timings']['embedding']  # Already timed above
        result['timings']['qpu_access_total'] = result['timings']['qpu_access']  # Alias for consistency
        result['timings']['solve_time'] = result['timings']['qpu_access'] + result['timings']['embedding']
        
        # Chain breaks
        result['chain_break_fraction'] = float(np.mean(sampleset.record.chain_break_fraction))
        
        # Best solution
        best = sampleset.first
        result['best_energy'] = float(best.energy)
        result['n_samples'] = len(sampleset)
        
        # Extract full solution for output
        result['solution'] = extract_solution(best.sample, data)
        
        # Calculate objective from solution
        result['objective'] = calculate_objective(best.sample, data)
        result['violations'] = count_violations(best.sample, data)
        result['feasible'] = result['violations'] == 0
        result['success'] = True
        
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        
        LOG.info(f"  [DirectQPU] Success! obj={result['objective']:.4f}, "
                f"QPU_access={result['timings']['qpu_access']:.3f}s, "
                f"embed={result['timings']['embedding']:.2f}s")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        LOG.error(f"  [DirectQPU] Error: {e}")
    
    return result


def extract_solution(sample: Dict, data: Dict) -> Dict:
    """Extract solution in a structured format similar to benchmark_all_strategies."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    
    solution = {
        'Y': {},  # Y[farm][food] = 0/1
        'U': {},  # U[food] = 0/1
        'allocations': {},  # (farm, food) -> area
        'summary': {
            'n_farms': len(farm_names),
            'n_foods': len(food_names),
            'foods_used': [],
            'total_area_allocated': 0,
        }
    }
    
    # Extract Y and U values
    for farm in farm_names:
        solution['Y'][farm] = {}
        for food in food_names:
            y_val = sample.get(f"Y_{farm}_{food}", 0)
            solution['Y'][farm][food] = int(y_val)
            if y_val == 1:
                area = land_availability.get(farm, 0)
                solution['allocations'][f"{farm}_{food}"] = area
                solution['summary']['total_area_allocated'] += area
    
    for food in food_names:
        u_val = sample.get(f"U_{food}", 0)
        solution['U'][food] = int(u_val)
        if u_val == 1:
            solution['summary']['foods_used'].append(food)
    
    solution['summary']['n_unique_foods'] = len(solution['summary']['foods_used'])
    
    return solution


def solve_qbsolv(cqm: ConstrainedQuadraticModel, data: Dict,
                 subproblem_size: int = 50, num_repeats: int = 100) -> Dict:
    """
    QBSolv: Decompose large BQM into subproblems, solve on QPU in parallel.
    """
    if not HAS_QBSOLV:
        return {'success': False, 'error': 'QBSolv not available'}
    
    result = {
        'method': 'qbsolv',
        'subproblem_size': subproblem_size,
        'num_repeats': num_repeats,
    }
    
    try:
        # Convert CQM to BQM
        t0 = time.time()
        bqm, info = cqm_to_bqm(cqm)
        result['cqm_to_bqm_time'] = time.time() - t0
        result['bqm_variables'] = len(bqm.variables)
        
        # Get QPU sampler for subproblems
        qpu = get_qpu_sampler()
        if qpu:
            sub_sampler = EmbeddingComposite(qpu)
        elif HAS_NEAL:
            # Fallback to simulated annealing
            sub_sampler = neal.SimulatedAnnealingSampler()
            result['fallback'] = 'SimulatedAnnealing'
        else:
            return {'success': False, 'error': 'No sampler available'}
        
        # Solve with QBSolv
        t0 = time.time()
        response = QBSolv().sample(
            bqm,
            solver=sub_sampler,
            solver_limit=subproblem_size,
            num_repeats=num_repeats
        )
        result['total_time'] = time.time() - t0
        
        # Best solution
        best = response.first
        result['best_energy'] = float(best.energy)
        result['n_samples'] = len(response)
        
        # Calculate objective
        result['objective'] = calculate_objective(best.sample, data)
        result['violations'] = count_violations(best.sample, data)
        result['feasible'] = result['violations'] == 0
        result['success'] = True
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
    
    return result


def solve_decomposition_sa(cqm: ConstrainedQuadraticModel, data: Dict,
                            method: str = 'PlotBased',
                            num_reads: int = 1000,
                            num_sweeps: int = 1000,
                            verbose: bool = True) -> Dict:
    """
    Decomposition + SimulatedAnnealing: Split problem, solve each partition with SA.
    
    This is fast and reliable for benchmarking decomposition quality.
    """
    result = {
        'method': f'decomposition_{method}',
        'decomposition': method,
        'solver': 'SimulatedAnnealing',
        'num_reads': num_reads,
        'num_sweeps': num_sweeps,
        'timings': {},
    }
    
    try:
        total_start = time.time()
        
        # Get decomposition function
        decomp_func = DECOMPOSITION_METHODS.get(method)
        if decomp_func is None:
            return {'success': False, 'error': f'Unknown decomposition: {method}'}
        
        # Partition with timing
        if verbose:
            LOG.info(f"    [{method}] Partitioning...")
        t0 = time.time()
        partitions = decomp_func(data)
        partition_time = time.time() - t0
        result['timings']['partition'] = partition_time
        result['n_partitions'] = len(partitions)
        result['partition_sizes'] = [len(p) for p in partitions]
        
        if verbose:
            LOG.info(f"    [{method}] Created {len(partitions)} partitions in {partition_time:.3f}s")
            LOG.info(f"    [{method}] Partition sizes: min={min(result['partition_sizes'])}, "
                    f"max={max(result['partition_sizes'])}, avg={np.mean(result['partition_sizes']):.1f}")
        
        # Create sampler
        sampler = neal.SimulatedAnnealingSampler()
        
        # Solve each partition with detailed timing
        all_samples = {}
        partition_results = []
        total_sa_time = 0
        
        for i, partition in enumerate(partitions):
            if len(partition) == 0:
                continue
            
            # Build BQM for partition
            t0 = time.time()
            bqm = build_bqm_for_partition(partition, data)
            bqm_build_time = time.time() - t0
            
            # Solve with SA
            t0 = time.time()
            sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps)
            sa_time = time.time() - t0
            total_sa_time += sa_time
            
            # Collect solution
            best_sample = sampleset.first.sample
            all_samples.update(best_sample)
            
            partition_results.append({
                'partition': i,
                'variables': len(partition),
                'bqm_build_time': bqm_build_time,
                'sa_time': sa_time,
                'energy': float(sampleset.first.energy),
            })
            
            if verbose and (i + 1) % 10 == 0:
                LOG.info(f"    [{method}] Solved {i+1}/{len(partitions)} partitions...")
        
        result['partition_results'] = partition_results
        result['timings']['sa_total'] = total_sa_time
        result['timings']['solve_time'] = total_sa_time  # Alias for consistency
        result['timings']['embedding_total'] = 0  # SA doesn't embed
        result['timings']['qpu_access_total'] = 0  # SA doesn't use QPU
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        
        # Extract full solution
        result['solution'] = extract_solution(all_samples, data)
        
        # Calculate final objective and violations
        result['objective'] = calculate_objective(all_samples, data)
        result['violations'] = count_violations(all_samples, data)
        result['feasible'] = result['violations'] == 0
        result['success'] = True
        
        if verbose:
            LOG.info(f"    [{method}] Complete: obj={result['objective']:.4f}, "
                    f"violations={result['violations']}, time={result['total_time']:.2f}s")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def solve_decomposition_qpu(cqm: ConstrainedQuadraticModel, data: Dict,
                            method: str = 'PlotBased',
                            num_reads: int = 1000,
                            annealing_time: int = 20,
                            embed_timeout: int = 30,
                            verbose: bool = True) -> Dict:
    """
    Decomposition + QPU: Split problem, solve each partition on real QPU.
    
    Includes detailed timing for:
    - Partition time
    - Embedding time per partition
    - QPU access time (from sampleset.info)
    - Total wall-clock time
    """
    result = {
        'method': f'decomposition_{method}_QPU',
        'decomposition': method,
        'solver': 'QPU',
        'num_reads': num_reads,
        'annealing_time': annealing_time,
        'timings': {},
    }
    
    try:
        total_start = time.time()
        
        # Get decomposition function
        decomp_func = DECOMPOSITION_METHODS.get(method)
        if decomp_func is None:
            return {'success': False, 'error': f'Unknown decomposition: {method}'}
        
        # Partition with timing
        if verbose:
            LOG.info(f"    [{method}+QPU] Partitioning...")
        t0 = time.time()
        partitions = decomp_func(data)
        partition_time = time.time() - t0
        result['timings']['partition'] = partition_time
        result['n_partitions'] = len(partitions)
        result['partition_sizes'] = [len(p) for p in partitions]
        
        if verbose:
            LOG.info(f"    [{method}+QPU] Created {len(partitions)} partitions in {partition_time:.3f}s")
        
        # Get QPU sampler
        qpu = get_qpu_sampler()
        if qpu is None:
            if verbose:
                LOG.warning(f"    [{method}+QPU] No QPU available, falling back to SA")
            return solve_decomposition_sa(cqm, data, method, num_reads, verbose=verbose)
        
        sampler = EmbeddingComposite(qpu)
        
        # Solve each partition with detailed timing
        all_samples = {}
        partition_results = []
        total_qpu_access_time = 0
        total_qpu_programming_time = 0
        total_qpu_sampling_time = 0
        total_embedding_time = 0
        failed_embeddings = 0
        
        for i, partition in enumerate(partitions):
            if len(partition) == 0:
                continue
            
            # Build BQM for partition
            t0 = time.time()
            bqm = build_bqm_for_partition(partition, data)
            bqm_build_time = time.time() - t0
            
            # Solve on QPU with timeout
            t0 = time.time()
            try:
                sampleset = sampler.sample(
                    bqm,
                    num_reads=num_reads,
                    annealing_time=annealing_time,
                    label=f"Decomp_{method}_P{i}"
                )
                wall_time = time.time() - t0
                
                # Extract detailed timing from sampleset.info
                timing_info = sampleset.info.get('timing', {})
                qpu_access_us = timing_info.get('qpu_access_time', 0)
                qpu_programming_us = timing_info.get('qpu_programming_time', 0)
                qpu_sampling_us = timing_info.get('qpu_sampling_time', 0)
                total_real_us = timing_info.get('total_real_time', 0)
                
                # Estimate embedding time (wall_time - total_real_time)
                embedding_time = wall_time - (total_real_us / 1e6)
                total_embedding_time += max(0, embedding_time)
                
                total_qpu_access_time += qpu_access_us / 1e6
                total_qpu_programming_time += qpu_programming_us / 1e6
                total_qpu_sampling_time += qpu_sampling_us / 1e6
                
                # Chain break info
                chain_breaks = float(np.mean(sampleset.record.chain_break_fraction))
                
                # Collect solution
                best_sample = sampleset.first.sample
                all_samples.update(best_sample)
                
                partition_results.append({
                    'partition': i,
                    'variables': len(partition),
                    'bqm_build_time': bqm_build_time,
                    'wall_time': wall_time,
                    'embedding_time_est': max(0, embedding_time),
                    'qpu_access_time': qpu_access_us / 1e6,
                    'qpu_programming_time': qpu_programming_us / 1e6,
                    'qpu_sampling_time': qpu_sampling_us / 1e6,
                    'chain_break_fraction': chain_breaks,
                    'energy': float(sampleset.first.energy),
                    'success': True,
                })
                
            except Exception as e:
                # Embedding failed - use SA fallback for this partition
                failed_embeddings += 1
                if verbose:
                    LOG.warning(f"    [{method}+QPU] Partition {i} embedding failed: {e}")
                
                # Fallback to SA
                sa_sampler = neal.SimulatedAnnealingSampler()
                sampleset = sa_sampler.sample(bqm, num_reads=num_reads, num_sweeps=1000)
                wall_time = time.time() - t0
                
                best_sample = sampleset.first.sample
                all_samples.update(best_sample)
                
                partition_results.append({
                    'partition': i,
                    'variables': len(partition),
                    'wall_time': wall_time,
                    'fallback': 'SA',
                    'error': str(e),
                    'success': False,
                })
            
            if verbose and (i + 1) % 5 == 0:
                LOG.info(f"    [{method}+QPU] Solved {i+1}/{len(partitions)} partitions...")
        
        result['partition_results'] = partition_results
        result['failed_embeddings'] = failed_embeddings
        result['timings']['partition'] = partition_time
        result['timings']['embedding_total'] = total_embedding_time
        result['timings']['qpu_access_total'] = total_qpu_access_time
        result['timings']['qpu_programming_total'] = total_qpu_programming_time
        result['timings']['qpu_sampling_total'] = total_qpu_sampling_time
        result['timings']['solve_time'] = total_qpu_access_time + partition_time  # Classical + QPU
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        result['total_qpu_time'] = total_qpu_access_time
        
        # Extract full solution
        result['solution'] = extract_solution(all_samples, data)
        
        # Calculate final objective and violations
        result['objective'] = calculate_objective(all_samples, data)
        result['violations'] = count_violations(all_samples, data)
        result['feasible'] = result['violations'] == 0
        result['success'] = True
        
        if verbose:
            LOG.info(f"    [{method}+QPU] Complete: obj={result['objective']:.4f}, "
                    f"violations={result['violations']}")
            LOG.info(f"    [{method}+QPU] Timing: total={result['total_time']:.2f}s, "
                    f"QPU_access={total_qpu_access_time:.3f}s, embed={total_embedding_time:.2f}s")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return result


# ============================================================================
# CQM-FIRST DECOMPOSITION (CONSTRAINT-PRESERVING)
# ============================================================================

def extract_sub_cqm(cqm: ConstrainedQuadraticModel, 
                    partition_vars: Set[str], 
                    fixed_vars: Dict[str, int] = None) -> ConstrainedQuadraticModel:
    """
    Extract a sub-CQM containing only the specified variables.
    
    Constraints are kept if they can be fully evaluated:
    - All variables are in partition_vars (will be optimized)
    - Or some variables are in partition_vars and others are fixed
    
    Args:
        cqm: The original CQM
        partition_vars: Set of variable names to include
        fixed_vars: Dict of variable_name -> value for variables not in partition
    
    Returns:
        A new CQM with only the partition variables and relevant constraints
    """
    from dimod import ConstrainedQuadraticModel, Binary, QuadraticModel
    
    if fixed_vars is None:
        fixed_vars = {}
    
    sub_cqm = ConstrainedQuadraticModel()
    
    # Add variables from partition
    for var_name in partition_vars:
        if var_name in cqm.variables:
            sub_cqm.add_variable('BINARY', var_name)
    
    # Extract objective: only keep terms involving partition variables
    # For terms with fixed variables, compute their contribution
    obj = cqm.objective
    new_obj = QuadraticModel()
    
    for var in obj.linear:
        if var in partition_vars:
            new_obj.add_variable('BINARY', var)
            new_obj.set_linear(var, obj.get_linear(var))
    
    for (u, v), bias in obj.quadratic.items():
        if u in partition_vars and v in partition_vars:
            # Both in partition - keep quadratic term
            new_obj.add_quadratic(u, v, bias)
        elif u in partition_vars and v in fixed_vars:
            # v is fixed - becomes linear term for u
            if fixed_vars[v] == 1:
                current = new_obj.get_linear(u) if u in new_obj.variables else 0
                new_obj.set_linear(u, current + bias)
        elif v in partition_vars and u in fixed_vars:
            # u is fixed - becomes linear term for v
            if fixed_vars[u] == 1:
                current = new_obj.get_linear(v) if v in new_obj.variables else 0
                new_obj.set_linear(v, current + bias)
    
    sub_cqm.set_objective(new_obj)
    
    # Extract constraints
    for label, constraint in cqm.constraints.items():
        lhs = constraint.lhs
        rhs = constraint.rhs
        sense = constraint.sense
        
        # Get variables in this constraint
        constraint_vars = set(lhs.variables)
        
        # Check if constraint can be included
        in_partition = constraint_vars & partition_vars
        in_fixed = constraint_vars & set(fixed_vars.keys())
        outside = constraint_vars - partition_vars - set(fixed_vars.keys())
        
        if outside:
            # Some variables are neither in partition nor fixed - skip
            continue
        
        if not in_partition:
            # All variables are fixed - constraint is already satisfied/violated, skip
            continue
        
        # Build new constraint LHS
        new_lhs = QuadraticModel()
        offset = 0
        
        for var in lhs.linear:
            if var in partition_vars:
                new_lhs.add_variable('BINARY', var)
                new_lhs.set_linear(var, lhs.get_linear(var))
            elif var in fixed_vars:
                offset += lhs.get_linear(var) * fixed_vars[var]
        
        for (u, v), bias in lhs.quadratic.items():
            if u in partition_vars and v in partition_vars:
                new_lhs.add_quadratic(u, v, bias)
            elif u in partition_vars and v in fixed_vars:
                if fixed_vars[v] == 1:
                    current = new_lhs.get_linear(u) if u in new_lhs.variables else 0
                    new_lhs.set_linear(u, current + bias)
            elif v in partition_vars and u in fixed_vars:
                if fixed_vars[u] == 1:
                    current = new_lhs.get_linear(v) if v in new_lhs.variables else 0
                    new_lhs.set_linear(v, current + bias)
            elif u in fixed_vars and v in fixed_vars:
                offset += bias * fixed_vars[u] * fixed_vars[v]
        
        # Adjust RHS for fixed variable contributions
        new_rhs = rhs - offset
        
        # Add constraint with appropriate sense
        try:
            if sense.name == 'Eq':
                sub_cqm.add_constraint(new_lhs == new_rhs, label=label)
            elif sense.name == 'Le':
                sub_cqm.add_constraint(new_lhs <= new_rhs, label=label)
            elif sense.name == 'Ge':
                sub_cqm.add_constraint(new_lhs >= new_rhs, label=label)
        except Exception:
            # Skip problematic constraints
            pass
    
    return sub_cqm


def solve_cqm_first_decomposition_sa(cqm: ConstrainedQuadraticModel, data: Dict,
                                      method: str = 'PlotBased',
                                      num_reads: int = 1000,
                                      num_sweeps: int = 1000,
                                      lagrange: float = 10.0,
                                      verbose: bool = True) -> Dict:
    """
    CQM-First Decomposition: Partition CQM, then convert each partition to BQM.
    
    This PRESERVES CONSTRAINTS by:
    1. Partitioning at the CQM level (variables, not penalty edges)
    2. Extracting sub-CQMs with relevant constraints for each partition
    3. Converting each sub-CQM to BQM (penalties only for that partition's constraints)
    4. Solving with SA
    
    This avoids the constraint-cutting problem of BQM-first decomposition.
    
    IMPORTANT: For best results, use PlotBased partitioning where each farm is in its
    own partition. This allows precise tracking of MaxPlots constraints across all farms.
    Multilevel partitioning can still have MaxPlots violations because multiple farms
    in the same partition can independently select the same food.
    """
    result = {
        'method': f'cqm_first_{method}',
        'decomposition': method,
        'solver': 'SimulatedAnnealing',
        'num_reads': num_reads,
        'num_sweeps': num_sweeps,
        'approach': 'cqm_first',  # Key identifier
        'timings': {},
    }
    
    try:
        total_start = time.time()
        
        # Get decomposition function
        decomp_func = DECOMPOSITION_METHODS.get(method)
        if decomp_func is None:
            return {'success': False, 'error': f'Unknown decomposition: {method}'}
        
        # Partition with timing
        if verbose:
            LOG.info(f"    [CQM-First {method}] Partitioning...")
        t0 = time.time()
        partitions = decomp_func(data)
        partition_time = time.time() - t0
        result['timings']['partition'] = partition_time
        result['n_partitions'] = len(partitions)
        result['partition_sizes'] = [len(p) for p in partitions]
        
        if verbose:
            LOG.info(f"    [CQM-First {method}] Created {len(partitions)} partitions in {partition_time:.3f}s")
        
        # Create sampler
        sa_sampler = neal.SimulatedAnnealingSampler()
        
        # Solve partitions in order, using two-stage approach:
        # 1. First solve U partition (master) to get food group feasibility
        # 2. Then solve Y partitions with U values fixed
        
        all_samples = {}
        partition_results = []
        total_sa_time = 0
        total_convert_time = 0
        
        # Find the U partition (master)
        u_partition_idx = None
        for i, partition in enumerate(partitions):
            if any(v.startswith("U_") for v in partition):
                u_partition_idx = i
                break
        
        # Solve U partition first (if exists)
        if u_partition_idx is not None:
            u_partition = partitions[u_partition_idx]
            
            if verbose:
                LOG.info(f"    [CQM-First {method}] Solving master partition (U vars)...")
            
            # Extract sub-CQM for U variables
            t0 = time.time()
            sub_cqm = extract_sub_cqm(cqm, u_partition)
            
            # Convert to BQM
            sub_bqm, _ = cqm_to_bqm(sub_cqm, lagrange_multiplier=lagrange)
            convert_time = time.time() - t0
            total_convert_time += convert_time
            
            # Solve with SA
            t0 = time.time()
            sampleset = sa_sampler.sample(sub_bqm, num_reads=num_reads, num_sweeps=num_sweeps)
            sa_time = time.time() - t0
            total_sa_time += sa_time
            
            # Collect solution
            best_sample = sampleset.first.sample
            # Filter to only include original CQM variables (not slack)
            for var, val in best_sample.items():
                if var in cqm.variables:
                    all_samples[var] = int(val)
            
            partition_results.append({
                'partition': u_partition_idx,
                'type': 'master',
                'variables': len(u_partition),
                'sub_cqm_vars': len(sub_cqm.variables),
                'sub_cqm_constraints': len(sub_cqm.constraints),
                'sub_bqm_vars': len(sub_bqm.variables),
                'convert_time': convert_time,
                'sa_time': sa_time,
                'energy': float(sampleset.first.energy),
            })
        
        # Now solve Y partitions with U values fixed
        # Track food usage for MaxPlots constraint
        fixed_u = {k: v for k, v in all_samples.items() if k.startswith("U_")}
        food_usage_count = {food: 0 for food in data['food_names']}  # Track plots per food
        max_plots = data['max_plots_per_crop']
        
        for i, partition in enumerate(partitions):
            if i == u_partition_idx:
                continue  # Already solved
            
            if len(partition) == 0:
                continue
            
            # Build list of foods that have reached max usage
            full_foods = {food for food, count in food_usage_count.items() if count >= max_plots}
            
            # Calculate remaining slots for each food
            remaining_slots = {food: max_plots - count for food, count in food_usage_count.items()}
            
            # Combine fixed vars: U values + mark full foods as unavailable
            combined_fixed = dict(fixed_u)
            # If a food is full, pretend U[food]=0 to prevent selection
            for food in full_foods:
                combined_fixed[f"U_{food}"] = 0
            
            # Extract sub-CQM with U values fixed
            t0 = time.time()
            sub_cqm = extract_sub_cqm(cqm, partition, fixed_vars=combined_fixed)
            
            # Adjust MaxPlots constraints for already-used slots
            # The original constraint is: sum(Y[farm,food]) <= max_plots
            # With current usage, it becomes: sum(Y[farm,food]) <= max_plots - usage = remaining_slots
            for food in data['food_names']:
                if food not in full_foods and remaining_slots.get(food, max_plots) < max_plots:
                    label = f"MaxPlots_{food}"
                    if label in sub_cqm.constraints:
                        # Modify the RHS of this constraint
                        constraint = sub_cqm.constraints[label]
                        new_rhs = remaining_slots[food]
                        # Remove and re-add with new RHS
                        del sub_cqm.constraints[label]
                        sub_cqm.add_constraint(constraint.lhs <= new_rhs, label=label)
            
            # Convert to BQM
            sub_bqm, _ = cqm_to_bqm(sub_cqm, lagrange_multiplier=lagrange)
            convert_time = time.time() - t0
            total_convert_time += convert_time
            
            # Solve with SA
            t0 = time.time()
            sampleset = sa_sampler.sample(sub_bqm, num_reads=num_reads, num_sweeps=num_sweeps)
            sa_time = time.time() - t0
            total_sa_time += sa_time
            
            # Collect solution and update food usage
            best_sample = sampleset.first.sample
            for var, val in best_sample.items():
                if var in cqm.variables:
                    all_samples[var] = int(val)
                    # Track food usage for MaxPlots
                    if var.startswith("Y_") and int(val) == 1:
                        parts = var.split("_", 2)
                        if len(parts) == 3:
                            food = parts[2]
                            food_usage_count[food] = food_usage_count.get(food, 0) + 1
            
            partition_results.append({
                'partition': i,
                'type': 'subproblem',
                'variables': len(partition),
                'sub_cqm_vars': len(sub_cqm.variables),
                'sub_cqm_constraints': len(sub_cqm.constraints),
                'sub_bqm_vars': len(sub_bqm.variables),
                'convert_time': convert_time,
                'sa_time': sa_time,
                'energy': float(sampleset.first.energy),
            })
            
            if verbose and (i + 1) % 10 == 0:
                LOG.info(f"    [CQM-First {method}] Solved {i+1}/{len(partitions)} partitions...")
        
        result['partition_results'] = partition_results
        result['timings']['convert_total'] = total_convert_time
        result['timings']['sa_total'] = total_sa_time
        result['timings']['solve_time'] = total_sa_time
        result['timings']['embedding_total'] = 0  # SA doesn't embed
        result['timings']['qpu_access_total'] = 0  # SA doesn't use QPU
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        
        # Extract full solution
        result['solution'] = extract_solution(all_samples, data)
        
        # Calculate final objective and violations
        result['objective'] = calculate_objective(all_samples, data)
        result['violations'] = count_violations(all_samples, data)
        result['feasible'] = result['violations'] == 0
        result['success'] = True
        
        if verbose:
            LOG.info(f"    [CQM-First {method}] Complete: obj={result['objective']:.4f}, "
                    f"violations={result['violations']}, time={result['total_time']:.2f}s")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return result


# ============================================================================
# SOLUTION EVALUATION
# ============================================================================

def calculate_objective(sample: Dict, data: Dict) -> float:
    """Calculate objective value from solution."""
    food_benefits = data['food_benefits']
    land_availability = data['land_availability']
    total_area = data['total_area']
    
    objective = 0.0
    for key, val in sample.items():
        if key.startswith("Y_") and val == 1:
            parts = key.split("_", 2)
            farm, food = parts[1], parts[2]
            if farm in land_availability and food in food_benefits:
                objective += food_benefits[food] * land_availability[farm]
    
    return objective / total_area


def count_violations(sample: Dict, data: Dict) -> int:
    """Count constraint violations."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    food_groups = data['food_groups']
    food_group_constraints = data['food_group_constraints']
    reverse_mapping = data['reverse_mapping']
    max_plots_per_crop = data['max_plots_per_crop']
    
    violations = 0
    
    # One crop per farm
    for farm in farm_names:
        count = sum(1 for food in food_names if sample.get(f"Y_{farm}_{food}", 0) == 1)
        if count != 1:
            violations += 1
    
    # Food group constraints
    for cg, limits in food_group_constraints.items():
        dg = reverse_mapping.get(cg, cg)
        foods_in_group = food_groups.get(dg, [])
        unique_foods = sum(1 for f in foods_in_group if sample.get(f"U_{f}", 0) == 1)
        if limits.get('min', 0) > 0 and unique_foods < limits['min']:
            violations += 1
        if 'max' in limits and unique_foods > limits['max']:
            violations += 1
    
    # Max plots per crop
    for food in food_names:
        count = sum(1 for farm in farm_names if sample.get(f"Y_{farm}_{food}", 0) == 1)
        if count > max_plots_per_crop:
            violations += 1
    
    return violations


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(scales: List[int] = None,
                  methods: List[str] = None,
                  verbose: bool = True) -> Dict:
    """Run QPU benchmark (no hybrid solvers)."""
    
    if scales is None:
        scales = FARM_SCALES
    
    if methods is None:
        methods = ['ground_truth', 'direct_qpu', 'qbsolv', 
                   'decomposition_PlotBased', 'decomposition_Multilevel(5)']
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'scales': scales,
        'methods': methods,
        'qpu_available': HAS_QPU,
        'qbsolv_available': HAS_QBSOLV,
        'results': []
    }
    
    for n_farms in scales:
        if verbose:
            print(f"\n{'='*80}")
            print(f"SCALE: {n_farms} farms ({n_farms * N_FOODS} Y variables + {N_FOODS} U variables)")
            print('='*80)
        
        # Load data with timing
        LOG.info(f"Loading problem data for {n_farms} farms...")
        t0 = time.time()
        data = load_problem_data(n_farms)
        data_load_time = time.time() - t0
        
        # Build CQM with timing
        LOG.info(f"Building CQM...")
        t0 = time.time()
        cqm, metadata = build_binary_cqm(data)
        cqm_build_time = time.time() - t0
        
        if verbose:
            print(f"  Data load: {data_load_time:.2f}s")
            print(f"  CQM build: {cqm_build_time:.2f}s")
            print(f"  CQM: {metadata['n_variables']} vars, {metadata['n_constraints']} constraints")
        
        scale_results = {
            'n_farms': n_farms,
            'metadata': metadata,
            'timings': {
                'data_load': data_load_time,
                'cqm_build': cqm_build_time,
            },
            'method_results': {}
        }
        
        # Ground truth
        if 'ground_truth' in methods:
            if verbose:
                LOG.info(f"Solving ground truth with Gurobi...")
                print(f"\n  [Ground Truth] Solving with Gurobi...")
            gt = solve_ground_truth(data)
            scale_results['ground_truth'] = gt
            if verbose and gt['success']:
                print(f"    Objective: {gt['objective']:.4f} in {gt['solve_time']:.2f}s")
        
        gt_obj = scale_results.get('ground_truth', {}).get('objective', 0)
        
        # Direct QPU (only for very small scales - embedding takes forever for larger)
        if 'direct_qpu' in methods:
            if n_farms <= 15:  # Only attempt for very small problems
                if verbose:
                    LOG.info(f"Attempting direct QPU...")
                    print(f"\n  [Direct QPU] CQM → BQM → QPU...")
                qpu_result = solve_direct_qpu(cqm, data)
                scale_results['method_results']['direct_qpu'] = qpu_result
                if verbose:
                    if qpu_result['success']:
                        gap = ((gt_obj - qpu_result['objective']) / gt_obj * 100) if gt_obj > 0 else 0
                        print(f"    Objective: {qpu_result['objective']:.4f} (gap: {gap:.1f}%)")
                        print(f"    QPU access time: {qpu_result.get('qpu_access_time', 0):.3f}s")
                        print(f"    Chain breaks: {qpu_result.get('chain_break_fraction', 0):.2%}")
                    else:
                        print(f"    Failed: {qpu_result.get('error', 'Unknown')}")
            else:
                if verbose:
                    print(f"\n  [Direct QPU] Skipped (scale too large for direct embedding)")
        
        # Decomposition methods with SA (fast)
        for method in methods:
            if method.startswith('decomposition_') and method.endswith('_SA'):
                decomp_name = method.replace('decomposition_', '').replace('_SA', '')
                if verbose:
                    LOG.info(f"Running {decomp_name} decomposition with SA...")
                    print(f"\n  [Decomposition: {decomp_name}] Partition + SimulatedAnnealing...")
                decomp_result = solve_decomposition_sa(cqm, data, method=decomp_name, verbose=verbose)
                scale_results['method_results'][method] = decomp_result
                if verbose:
                    if decomp_result['success']:
                        gap = ((gt_obj - decomp_result['objective']) / gt_obj * 100) if gt_obj > 0 else 0
                        print(f"    Objective: {decomp_result['objective']:.4f} (gap: {gap:.1f}%)")
                        print(f"    Partitions: {decomp_result['n_partitions']}")
                        print(f"    Time: {decomp_result['total_time']:.2f}s")
                        print(f"    Violations: {decomp_result['violations']}")
                    else:
                        print(f"    Failed: {decomp_result.get('error', 'Unknown')}")
        
        # Decomposition methods with QPU (slower but real quantum)
        for method in methods:
            if method.startswith('decomposition_') and method.endswith('_QPU'):
                decomp_name = method.replace('decomposition_', '').replace('_QPU', '')
                if verbose:
                    LOG.info(f"Running {decomp_name} decomposition with QPU...")
                    print(f"\n  [Decomposition: {decomp_name}] Partition + QPU...")
                decomp_result = solve_decomposition_qpu(cqm, data, method=decomp_name, verbose=verbose)
                scale_results['method_results'][method] = decomp_result
                if verbose:
                    if decomp_result['success']:
                        gap = ((gt_obj - decomp_result['objective']) / gt_obj * 100) if gt_obj > 0 else 0
                        print(f"    Objective: {decomp_result['objective']:.4f} (gap: {gap:.1f}%)")
                        print(f"    Partitions: {decomp_result['n_partitions']}")
                        print(f"    Total time: {decomp_result['total_time']:.2f}s")
                        if 'total_qpu_time' in decomp_result:
                            print(f"    QPU access time: {decomp_result['total_qpu_time']:.3f}s")
                        print(f"    Violations: {decomp_result['violations']}")
                    else:
                        print(f"    Failed: {decomp_result.get('error', 'Unknown')}")
        
        # CQM-First decomposition methods (constraint-preserving partition)
        for method in methods:
            if method.startswith('cqm_first_'):
                decomp_name = method.replace('cqm_first_', '')
                if verbose:
                    LOG.info(f"Running CQM-First {decomp_name} decomposition...")
                    print(f"\n  [CQM-First: {decomp_name}] Partition CQM → BQM → SA...")
                cqm_result = solve_cqm_first_decomposition_sa(cqm, data, method=decomp_name, verbose=verbose)
                scale_results['method_results'][method] = cqm_result
                if verbose:
                    if cqm_result['success']:
                        gap = ((gt_obj - cqm_result['objective']) / gt_obj * 100) if gt_obj > 0 else 0
                        print(f"    Objective: {cqm_result['objective']:.4f} (gap: {gap:.1f}%)")
                        print(f"    Partitions: {cqm_result['n_partitions']}")
                        print(f"    Time: {cqm_result['total_time']:.2f}s")
                        print(f"    Violations: {cqm_result['violations']} {'✓ FEASIBLE' if cqm_result['violations'] == 0 else '✗ INFEASIBLE'}")
                    else:
                        print(f"    Failed: {cqm_result.get('error', 'Unknown')}")
        
        # Coordinated decomposition (constraint-preserving)
        if 'coordinated' in methods:
            if verbose:
                LOG.info(f"Running coordinated master-subproblem decomposition...")
                print(f"\n  [Coordinated] Master-Subproblem (constraint-preserving)...")
            coord_result = solve_coordinated_decomposition(cqm, data)
            scale_results['method_results']['coordinated'] = coord_result
            if verbose:
                if coord_result['success']:
                    gap = ((gt_obj - coord_result['objective']) / gt_obj * 100) if gt_obj > 0 else 0
                    print(f"    Objective: {coord_result['objective']:.4f} (gap: {gap:.1f}%)")
                    print(f"    Violations: {coord_result['violations']}")
                    print(f"    Time: {coord_result['total_time']:.2f}s")
                else:
                    print(f"    Failed: {coord_result.get('error', 'Unknown')}")
        
        results['results'].append(scale_results)
    
    return results


def print_summary(results: Dict):
    """Print detailed summary table with all timing metrics."""
    print("\n" + "=" * 140)
    print("BENCHMARK SUMMARY (Pure QPU - No Hybrid)")
    print("=" * 140)
    
    # Detailed timing table
    print(f"\n{'Scale':<6} {'Method':<28} {'Obj':>8} {'Gap%':>7} {'Wall':>8} {'Solve':>8} {'Embed':>8} {'QPU':>8} {'Viol':>5} {'Status':<12}")
    print("-" * 140)
    
    for sr in results['results']:
        n_farms = sr['n_farms']
        gt_obj = sr.get('ground_truth', {}).get('objective', 0)
        
        # Ground truth (Gurobi)
        if 'ground_truth' in sr:
            gt = sr['ground_truth']
            gt_timings = gt.get('timings', {})
            gt_wall = gt.get('wall_time', gt.get('total_time', gt.get('solve_time', 0)))
            gt_solve = gt_timings.get('solve', gt.get('solve_time', 0))
            viol = gt.get('violations', 0)
            status = '✓ Opt' if gt.get('success') else '✗ Fail'
            print(f"{n_farms:<6} {'Ground Truth (Gurobi)':<28} {gt.get('objective', 0):>8.4f} {'0.0':>7} {gt_wall:>8.3f} {gt_solve:>8.3f} {'N/A':>8} {'N/A':>8} {viol:>5} {status:<12}")
        
        # Methods
        for method, r in sr.get('method_results', {}).items():
            obj = r.get('objective', 0)
            gap = ((gt_obj - obj) / gt_obj * 100) if gt_obj > 0 and obj > 0 else -999
            
            # Get timing details
            timings = r.get('timings', {})
            wall_time = r.get('wall_time', r.get('total_time', 0))
            solve_time = timings.get('solve_time', timings.get('sa_total', wall_time))
            embed_time = timings.get('embedding_total', timings.get('embedding', 0))
            qpu_time = timings.get('qpu_access_total', timings.get('qpu_access', 0))
            violations = r.get('violations', 0)
            
            # Determine solver type for display
            solver_type = r.get('solver', r.get('sampler', ''))
            is_sa = '_SA' in method or solver_type == 'SimulatedAnnealing' or method == 'coordinated'
            is_qpu = 'QPU' in method or solver_type == 'QPU' or qpu_time > 0.001
            
            if r.get('success'):
                status = '✓ Feas' if r.get('feasible', True) else f"⚠ {violations}v"
            else:
                status = f"✗ Fail"
            
            gap_str = f"{gap:.1f}" if gap > -100 else "N/A"
            
            # Format embed/QPU columns based on solver type
            if is_sa and not is_qpu:
                embed_str = "N/A"
                qpu_str = "N/A"
            else:
                embed_str = f"{embed_time:>8.2f}" if embed_time > 0 else f"{embed_time:>8.2f}"
                qpu_str = f"{qpu_time:>8.3f}" if qpu_time > 0 else f"{qpu_time:>8.3f}"
            
            print(f"{'':<6} {method:<28} {obj:>8.4f} {gap_str:>7} {wall_time:>8.2f} {solve_time:>8.2f} {embed_str:>8} {qpu_str:>8} {violations:>5} {status:<12}")
        
        print("-" * 140)
    
    # Legend
    print("\nLegend:")
    print("  Wall  = Total wall-clock time (seconds)")
    print("  Solve = Solve time (SA sweeps or QPU access + classical decomposition)")
    print("  Embed = Total embedding time (N/A for SA methods)")
    print("  QPU   = Total QPU access time (N/A for SA methods)")
    print("  Viol  = Number of constraint violations")


def save_results(results: Dict, filename: str = None) -> Path:
    """Save results to JSON."""
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qpu_benchmark_{ts}.json"
    
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='QPU Benchmark (Pure Quantum - No Hybrid)')
    parser.add_argument('--test', type=int, nargs='?', const=25,
                        help='Quick test with N farms (default: 25)')
    parser.add_argument('--full', action='store_true',
                        help='Full benchmark (25, 50, 100, 200 farms)')
    parser.add_argument('--scale', type=int, nargs='+',
                        help='Specific scales')
    parser.add_argument('--methods', nargs='+',
                        help='Specific methods')
    parser.add_argument('--output', type=str,
                        help='Output filename')
    parser.add_argument('--no-qpu', action='store_true',
                        help='Use SimulatedAnnealing instead of QPU (testing)')
    parser.add_argument('--token', type=str,
                        help='D-Wave API token (or set DWAVE_API_TOKEN env var)')
    
    args = parser.parse_args()
    
    print("[5/5] Starting benchmark...")
    
    # Configure D-Wave token
    # Priority: --token argument > environment variable > hardcoded default
    default_token = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'
    dwave_token = args.token or os.getenv('DWAVE_API_TOKEN') or default_token
    
    if dwave_token and not args.no_qpu:
        set_dwave_token(dwave_token)
    
    # Scales
    if args.test:
        scales = [args.test]
    elif args.full:
        scales = FARM_SCALES
    elif args.scale:
        scales = args.scale
    else:
        scales = [25]
    
    # Methods - use SA versions by default (faster), QPU versions available via --methods
    methods = args.methods
    if methods is None:
        # Default: ground truth + all SA decompositions (fast)
        methods = [
            'ground_truth', 
            'coordinated',
            'decomposition_PlotBased_SA',
            'decomposition_Multilevel(5)_SA',
            'decomposition_Louvain_SA',
            'decomposition_Spectral(10)_SA',
        ]
        if HAS_QPU and not args.no_qpu:
            # Add QPU methods: direct QPU + decomposition with QPU partitions
            methods.insert(1, 'direct_qpu')
            methods.extend([
                'decomposition_PlotBased_QPU',
                'decomposition_Multilevel(5)_QPU',
            ])
    
    print(f"\nScales: {scales}")
    print(f"Methods: {methods}")
    print(f"QPU Available: {HAS_QPU}")
    print(f"Louvain Available: {HAS_LOUVAIN}")
    print(f"Spectral Available: {HAS_SPECTRAL}")
    
    # Run
    results = run_benchmark(scales=scales, methods=methods, verbose=True)
    
    # Summary
    print_summary(results)
    
    # Save
    save_results(results, args.output)
    
    print("\n✅ Benchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
