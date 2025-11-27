#!/usr/bin/env python3
"""
Comprehensive Embedding and Solving Benchmark v2.0

Systematically tests ALL combinations of:
- Problem sizes (5, 10, 15, 20, 25, 30 farms)
- Formulations (CQM, BQM from CQM, Sparse Direct BQM)
- Decomposition strategies:
  * None (direct embedding)
  * Louvain community detection
  * Plot-based domain decomposition
  * Multilevel coarsening (ML-QLS style)
  * Sequential cut-set reduction
  * Energy-impact (dwave-hybrid)

For each configuration:
1. Build formulation
2. Apply decomposition (if any)
3. Attempt embedding (300s timeout, continue even if fails)
4. Solve with Gurobi (always, regardless of embedding)
5. Calculate total times (summing partitions for decomposed)

Outputs:
- JSON (detailed results)
- CSV (flattened for analysis)
- Markdown (human-readable summary)

Author: Generated for OQI-UC002-DWave comprehensive benchmark
Date: 2025-11-27 (v2.0 - comprehensive testing matrix)
"""

import os
import sys
import time
import json
import csv
import statistics
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=" * 80)
print("COMPREHENSIVE EMBEDDING AND SOLVING BENCHMARK v2.0")
print("=" * 80)
print("Testing: Formulations × Decompositions × Solvers")
print("=" * 80)

# Imports
print("\n[1/6] Importing libraries...")
import_start = time.time()

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, Real, cqm_to_bqm
from dimod.generators import combinations
import minorminer

# Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("  Warning: Gurobi not available. Classical solving disabled.")

# D-Wave
try:
    from dwave.system import DWaveSampler
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    print("  Warning: DWaveSampler not available. Using Pegasus simulation.")

# Hybrid
try:
    from hybrid.decomposers import EnergyImpactDecomposer
    from hybrid.core import State
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    print("  Warning: dwave-hybrid not available.")

# Louvain
try:
    from networkx.algorithms.community import louvain_communities
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("  Warning: networkx louvain not available.")

# Advanced decomposition strategies
try:
    from advanced_decomposition_strategies import (
        decompose_multilevel,
        decompose_sequential_cutset,
        decompose_spatial_grid,
        analyze_decomposition_quality
    )
    ADVANCED_DECOMP_AVAILABLE = True
except ImportError:
    ADVANCED_DECOMP_AVAILABLE = False
    print("  Warning: Advanced decomposition strategies not available.")

print(f"  [OK] Imports done in {time.time() - import_start:.2f}s")

# Configuration
PROBLEM_SIZES = [5, 10, 15, 20, 25, 30]  # Number of farms to test
N_FOODS = 27
EMBEDDING_TIMEOUT = 300  # 5 minutes per partition - continue even if fails
SOLVE_TIMEOUT = 300  # 5 minutes for Gurobi per partition
SKIP_DENSE_EMBEDDING = False  # NEVER skip - always try embedding

# Formulation types to test
FORMULATIONS = ["CQM", "BQM", "SparseBQM"]

# Decomposition strategies to test
DECOMPOSITIONS = ["None", "Louvain", "PlotBased", "Multilevel", "Cutset", "SpatialGrid", "EnergyImpact"]

# Output directory
OUTPUT_DIR = Path(__file__).parent / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# FORMULATION BUILDERS
# =============================================================================

def build_farm_cqm(n_farms: int, n_foods: int = 27) -> Tuple[ConstrainedQuadraticModel, Dict]:
    """Build Farm CQM with continuous areas + binary selections"""
    print(f"    Building Farm CQM ({n_farms} farms, {n_foods} foods)...")
    
    cqm = ConstrainedQuadraticModel()
    metadata = {"type": "Farm_CQM", "n_farms": n_farms, "n_foods": n_foods}
    
    # Variables
    A = {}  # Continuous area
    Y = {}  # Binary selection
    for f in range(n_farms):
        for c in range(n_foods):
            A[f, c] = Real(f"A_{f}_{c}", lower_bound=0, upper_bound=10)
            Y[f, c] = Binary(f"Y_{f}_{c}")
    
    # Objective: maximize total area
    objective = sum(A[f, c] for f in range(n_farms) for c in range(n_foods))
    cqm.set_objective(-objective)  # Minimize negative = maximize
    
    # Constraints
    land_per_farm = 10.0
    min_area = 0.0001
    
    # 1. Land availability per farm
    for f in range(n_farms):
        cqm.add_constraint(
            sum(A[f, c] for c in range(n_foods)) <= land_per_farm,
            label=f"land_capacity_farm_{f}"
        )
    
    # 2. Min area if selected (A >= min_area * Y => A - min_area * Y >= 0)
    for f in range(n_farms):
        for c in range(n_foods):
            cqm.add_constraint(A[f, c] - min_area * Y[f, c] >= 0, label=f"min_area_{f}_{c}")
    
    # 3. Max area if selected (A <= max * Y => A - max * Y <= 0)
    for f in range(n_farms):
        for c in range(n_foods):
            cqm.add_constraint(A[f, c] - land_per_farm * Y[f, c] <= 0, label=f"max_area_{f}_{c}")
    
    # 4. Simple food group constraint (global)
    total_selections = sum(Y[f, c] for f in range(n_farms) for c in range(n_foods))
    cqm.add_constraint(total_selections >= n_foods // 2, label="min_total_foods")
    cqm.add_constraint(total_selections <= n_foods * n_farms, label="max_total_foods")
    
    metadata.update({
        "variables": len(cqm.variables),
        "constraints": len(cqm.constraints),
        "continuous_vars": n_farms * n_foods,
        "binary_vars": n_farms * n_foods
    })
    
    return cqm, metadata


def build_patch_cqm(n_patches: int, n_foods: int = 27) -> Tuple[ConstrainedQuadraticModel, Dict]:
    """Build Patch CQM with binary-only selections"""
    print(f"    Building Patch CQM ({n_patches} patches, {n_foods} foods)...")
    
    cqm = ConstrainedQuadraticModel()
    metadata = {"type": "Patch_CQM", "n_patches": n_patches, "n_foods": n_foods}
    
    # Variables: binary only
    Y = {}
    for p in range(n_patches):
        for c in range(n_foods):
            Y[p, c] = Binary(f"Y_{p}_{c}")
    
    # Objective: maximize total selections
    objective = sum(Y[p, c] for p in range(n_patches) for c in range(n_foods))
    cqm.set_objective(-objective)
    
    # Constraints
    # 1. One food per patch (or allow multiple with constraint)
    for p in range(n_patches):
        cqm.add_constraint(sum(Y[p, c] for c in range(n_foods)) <= 5, label=f"patch_limit_{p}")
    
    # 2. Global food constraints
    total_selections = sum(Y[p, c] for p in range(n_patches) for c in range(n_foods))
    cqm.add_constraint(total_selections >= n_foods // 2, label="min_foods")
    
    metadata.update({
        "variables": len(cqm.variables),
        "constraints": len(cqm.constraints),
        "binary_vars": n_patches * n_foods
    })
    
    return cqm, metadata


def build_patch_direct_bqm(n_patches: int, n_foods: int = 27) -> Tuple[BinaryQuadraticModel, Dict]:
    """Build Patch BQM directly (minimal slack variables)"""
    print(f"    Building Patch Direct BQM ({n_patches} patches, {n_foods} foods)...")
    
    bqm = BinaryQuadraticModel('BINARY')
    metadata = {"type": "Patch_Direct_BQM", "n_patches": n_patches, "n_foods": n_foods}
    
    # Primary variables
    for p in range(n_patches):
        for c in range(n_foods):
            var = f"Y_{p}_{c}"
            bqm.add_variable(var, -1.0)  # Reward selection
    
    # Penalty for constraint violations (soft constraints)
    penalty = 10.0
    
    # Patch limit constraint: sum(Y[p,:]) <= 5
    for p in range(n_patches):
        vars_in_patch = [f"Y_{p}_{c}" for c in range(n_foods)]
        # Add quadratic penalty for exceeding limit
        for i, v1 in enumerate(vars_in_patch):
            for v2 in vars_in_patch[i+1:]:
                bqm.add_interaction(v1, v2, penalty * 0.5)
    
    metadata.update({
        "variables": len(bqm.variables),
        "linear_terms": len(bqm.linear),
        "quadratic_terms": len(bqm.quadratic),
        "density": len(bqm.quadratic) / (len(bqm.variables) ** 2) if len(bqm.variables) > 0 else 0
    })
    
    return bqm, metadata


def build_patch_ultra_sparse_bqm(n_patches: int, n_foods: int = 27) -> Tuple[BinaryQuadraticModel, Dict]:
    """Build ultra-sparse BQM with minimal quadratic terms"""
    print(f"    Building Patch Ultra-Sparse BQM ({n_patches} patches, {n_foods} foods)...")
    
    bqm = BinaryQuadraticModel('BINARY')
    metadata = {"type": "Patch_UltraSparse_BQM", "n_patches": n_patches, "n_foods": n_foods}
    
    # Only linear terms - maximize selections
    for p in range(n_patches):
        for c in range(n_foods):
            var = f"Y_{p}_{c}"
            bqm.add_variable(var, -1.0)
    
    # Minimal quadratic terms: only adjacent patches
    penalty = 5.0
    for p in range(n_patches - 1):
        for c in range(n_foods):
            v1 = f"Y_{p}_{c}"
            v2 = f"Y_{p+1}_{c}"
            bqm.add_interaction(v1, v2, penalty * 0.1)
    
    metadata.update({
        "variables": len(bqm.variables),
        "linear_terms": len(bqm.linear),
        "quadratic_terms": len(bqm.quadratic),
        "density": len(bqm.quadratic) / (len(bqm.variables) ** 2) if len(bqm.variables) > 0 else 0
    })
    
    return bqm, metadata


def cqm_to_bqm_wrapper(cqm: ConstrainedQuadraticModel, formulation_name: str) -> Tuple[BinaryQuadraticModel, Dict]:
    """Convert CQM to BQM with metadata tracking"""
    print(f"    Converting {formulation_name} to BQM...")
    start = time.time()
    
    result = cqm_to_bqm(cqm, lagrange_multiplier=10.0)
    
    # Handle both old and new API (may return tuple of (bqm, inv_map) or just bqm)
    if isinstance(result, tuple):
        bqm = result[0]
    else:
        bqm = result
    
    metadata = {
        "type": f"{formulation_name}_to_BQM",
        "conversion_time": time.time() - start,
        "variables": len(bqm.variables),
        "linear_terms": len(bqm.linear),
        "quadratic_terms": len(bqm.quadratic),
        "density": len(bqm.quadratic) / (len(bqm.variables) ** 2) if len(bqm.variables) > 0 else 0
    }
    
    return bqm, metadata


# =============================================================================
# DECOMPOSITION STRATEGIES
# =============================================================================

def get_bqm_graph(bqm: BinaryQuadraticModel) -> nx.Graph:
    """Convert BQM to NetworkX graph"""
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    G.add_edges_from(bqm.quadratic.keys())
    return G


def decompose_louvain(bqm: BinaryQuadraticModel, max_partition_size: int = 150) -> List[Set]:
    """Decompose using Louvain community detection"""
    if not LOUVAIN_AVAILABLE:
        return []
    
    G = get_bqm_graph(bqm)
    communities = louvain_communities(G, seed=42)
    
    # Merge small communities
    partitions = []
    current_partition = set()
    
    for community in communities:
        if len(current_partition) + len(community) <= max_partition_size:
            current_partition.update(community)
        else:
            if current_partition:
                partitions.append(current_partition)
            current_partition = set(community)
    
    if current_partition:
        partitions.append(current_partition)
    
    return partitions


def decompose_plot_based(bqm: BinaryQuadraticModel, plots_per_partition: int = 5) -> List[Set]:
    """Decompose by grouping plots together"""
    variables = list(bqm.variables)
    
    # Extract plot indices from variable names (assumes Y_p_c format)
    plot_vars = {}
    for var in variables:
        if var.startswith("Y_"):
            parts = var.split("_")
            if len(parts) >= 3:
                plot_idx = int(parts[1])
                if plot_idx not in plot_vars:
                    plot_vars[plot_idx] = []
                plot_vars[plot_idx].append(var)
    
    # Group plots into partitions
    partitions = []
    plot_indices = sorted(plot_vars.keys())
    
    for i in range(0, len(plot_indices), plots_per_partition):
        partition = set()
        for plot_idx in plot_indices[i:i + plots_per_partition]:
            partition.update(plot_vars[plot_idx])
        partitions.append(partition)
    
    return partitions


def decompose_energy_impact(bqm: BinaryQuadraticModel, partition_size: int = 100) -> List[Set]:
    """Decompose using energy-impact from dwave-hybrid"""
    if not HYBRID_AVAILABLE:
        return []
    
    decomposer = EnergyImpactDecomposer(
        size=partition_size,
        rolling_history=0.85,
        traversal="bfs"
    )
    
    # Create initial state
    initial_sample = {v: 0 for v in bqm.variables}
    state = State.from_sample(initial_sample, bqm)
    
    # Run decomposer
    decomposed_state = decomposer.run(state).result()
    
    # Extract subproblem variables
    if hasattr(decomposed_state, 'subproblem'):
        return [set(decomposed_state.subproblem.variables)]
    
    return []


def extract_sub_bqm(bqm: BinaryQuadraticModel, variables: Set) -> BinaryQuadraticModel:
    """Extract subproblem BQM"""
    sub_bqm = BinaryQuadraticModel('BINARY')
    
    for var in variables:
        if var in bqm.linear:
            sub_bqm.add_variable(var, bqm.linear[var])
    
    for (u, v), bias in bqm.quadratic.items():
        if u in variables and v in variables:
            sub_bqm.add_interaction(u, v, bias)
    
    return sub_bqm


# =============================================================================
# EMBEDDING STUDY
# =============================================================================

def get_target_graph() -> nx.Graph:
    """Get QPU topology (or simulate Pegasus)"""
    if DWAVE_AVAILABLE:
        try:
            sampler = DWaveSampler()
            return sampler.to_networkx_graph()
        except:
            pass
    
    # Simulate Pegasus P16
    print("  Using simulated Pegasus P16 topology")
    import dwave_networkx as dnx
    return dnx.pegasus_graph(16)


def study_embedding(bqm: BinaryQuadraticModel, target_graph: nx.Graph, timeout: int = 300) -> Dict:
    """Study embedding feasibility and timing - ALWAYS attempts embedding regardless of density"""
    n_vars = len(bqm.variables)
    n_edges = len(bqm.quadratic)
    density = n_edges / (n_vars ** 2) if n_vars > 0 else 0
    print(f"      Testing embedding: {n_vars} vars, {n_edges} edges (density={density:.3f})")
    
    result = {
        "attempted": True,
        "success": False,
        "embedding_time": timeout,  # Default to full timeout if failed
        "chain_length_max": None,
        "chain_length_mean": None,
        "num_chains": 0,
        "num_variables": n_vars,
        "num_edges": n_edges,
        "density": density,
        "error": None,
        "skipped": False
    }
    
    # Handle empty or trivial BQMs
    if n_vars == 0:
        result["success"] = True
        result["embedding_time"] = 0
        result["error"] = "Empty BQM - trivially embeddable"
        return result
    
    source_graph = get_bqm_graph(bqm)
    
    try:
        print(f"      Running minorminer (timeout={timeout}s)...", flush=True)
        start = time.time()
        embedding = minorminer.find_embedding(source_graph, target_graph, timeout=timeout, verbose=0)
        elapsed = time.time() - start
        result["embedding_time"] = elapsed
        
        if embedding:
            result["success"] = True
            result["num_chains"] = len(embedding)
            
            chain_lengths = [len(chain) for chain in embedding.values()]
            result["chain_length_max"] = max(chain_lengths)
            result["chain_length_mean"] = statistics.mean(chain_lengths)
            print(f"      [SUCCESS] Embedded in {elapsed:.1f}s (chains: max={result['chain_length_max']}, mean={result['chain_length_mean']:.1f})")
        else:
            result["error"] = f"No embedding found in {elapsed:.1f}s"
            print(f"      [FAILED] No embedding (tried for {elapsed:.1f}s)")
            
    except Exception as e:
        result["error"] = str(e)
        result["embedding_time"] = timeout  # Assume full timeout on error
        print(f"      [ERROR] {e}")
    
    return result


def study_decomposed_embedding(bqm: BinaryQuadraticModel, partitions: List[Set],
                               target_graph: nx.Graph, timeout: int = 300) -> Dict:
    """Study embedding for decomposed problem - always calculates total time"""
    print(f"      Embedding {len(partitions)} partitions...")
    
    result = {
        "num_partitions": len(partitions),
        "partition_sizes": [len(p) for p in partitions],
        "partition_results": [],
        "total_embedding_time": 0,  # Sum of all partition times (success or not)
        "successful_embedding_time": 0,  # Sum of only successful embeddings
        "all_embedded": True,
        "num_successful": 0,
        "num_failed": 0
    }
    
    for i, partition in enumerate(partitions):
        print(f"        Partition {i+1}/{len(partitions)} ({len(partition)} vars)...")
        sub_bqm = extract_sub_bqm(bqm, partition)
        partition_result = study_embedding(sub_bqm, target_graph, timeout)
        partition_result["partition_id"] = i
        
        result["partition_results"].append(partition_result)
        
        # Always add to total time (even failed attempts take time)
        if partition_result.get("embedding_time") is not None:
            result["total_embedding_time"] += partition_result["embedding_time"]
        
        if partition_result["success"]:
            result["num_successful"] += 1
            if partition_result.get("embedding_time") is not None:
                result["successful_embedding_time"] += partition_result["embedding_time"]
        else:
            result["num_failed"] += 1
            result["all_embedded"] = False
    
    # Copy key metrics for convenience
    result["success"] = result["all_embedded"]
    result["embedding_time"] = result["total_embedding_time"]
    
    return result


def solve_decomposed_bqm_with_gurobi(bqm: BinaryQuadraticModel, partitions: List[Set], timeout: int = 600) -> Dict:
    """Solve each partition independently with Gurobi and aggregate results"""
    if not GUROBI_AVAILABLE:
        return {"error": "Gurobi not available", "success": False, "solve_time": 0}
    
    print(f"      Solving {len(partitions)} partitions with Gurobi...")
    
    result = {
        "num_partitions": len(partitions),
        "partition_results": [],
        "total_solve_time": 0,
        "aggregated_objective": 0,
        "all_solved": True
    }
    
    for i, partition in enumerate(partitions):
        sub_bqm = extract_sub_bqm(bqm, partition)
        
        # Solve partition
        partition_solve = solve_bqm_with_gurobi(sub_bqm, timeout)
        
        result["partition_results"].append({
            "partition_id": i,
            "n_vars": len(partition),
            "success": partition_solve.get("success", False),
            "solve_time": partition_solve.get("solve_time", None),
            "objective": partition_solve.get("objective", None),
            "status": partition_solve.get("status", None)
        })
        
        if partition_solve.get("success"):
            result["total_solve_time"] += partition_solve["solve_time"]
            if partition_solve.get("objective") is not None:
                result["aggregated_objective"] += partition_solve["objective"]
        else:
            result["all_solved"] = False
    
    # Add summary info
    result["success"] = result["all_solved"]
    result["solve_time"] = result["total_solve_time"]
    result["objective"] = result["aggregated_objective"] if result["all_solved"] else None
    
    return result


# =============================================================================
# GUROBI SOLVING
# =============================================================================

def solve_cqm_with_gurobi(cqm: ConstrainedQuadraticModel, timeout: int = 600) -> Dict:
    """Solve CQM with Gurobi"""
    if not GUROBI_AVAILABLE:
        return {"error": "Gurobi not available"}
    
    print(f"      Solving CQM ({len(cqm.variables)} vars, {len(cqm.constraints)} constraints)...")
    
    try:
        print("        [1/5] Creating Gurobi model...")
        model = gp.Model("CQM")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', timeout)
        
        # Create variables
        print(f"        [2/5] Adding variables...")
        gurobi_vars = {}
        for var_name in cqm.variables:
            var_info = cqm.vartype(var_name)
            if var_info == 'BINARY':
                gurobi_vars[var_name] = model.addVar(vtype=GRB.BINARY, name=var_name)
            else:  # REAL
                lb = cqm.lower_bound(var_name)
                ub = cqm.upper_bound(var_name)
                gurobi_vars[var_name] = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=var_name)
        
        # Set objective
        print(f"        [3/5] Building objective ({len(cqm.objective.linear)} linear, {len(cqm.objective.quadratic)} quadratic)...")
        obj_expr = 0
        for var_name, coeff in cqm.objective.linear.items():
            obj_expr += coeff * gurobi_vars[var_name]
        
        for (v1, v2), coeff in cqm.objective.quadratic.items():
            obj_expr += coeff * gurobi_vars[v1] * gurobi_vars[v2]
        
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Add constraints
        print(f"        [4/5] Adding {len(cqm.constraints)} constraints...")
        for i, (label, constraint) in enumerate(cqm.constraints.items()):
            if i % 100 == 0 and i > 0:
                print(f"            Progress: {i}/{len(cqm.constraints)}")
            constr_expr = 0
            for var_name, coeff in constraint.lhs.linear.items():
                constr_expr += coeff * gurobi_vars[var_name]
            
            for (v1, v2), coeff in constraint.lhs.quadratic.items():
                constr_expr += coeff * gurobi_vars[v1] * gurobi_vars[v2]
            
            if constraint.sense == '<=':
                model.addConstr(constr_expr <= constraint.rhs, name=label)
            elif constraint.sense == '>=':
                model.addConstr(constr_expr >= constraint.rhs, name=label)
            else:  # ==
                model.addConstr(constr_expr == constraint.rhs, name=label)
        
        # Solve
        print("        [5/5] Optimizing...")
        start = time.time()
        model.optimize()
        solve_time = time.time() - start
        print(f"        [DONE] in {solve_time:.2f}s (status={model.status})")
        
        result = {
            "success": model.status == GRB.OPTIMAL,
            "solve_time": solve_time,
            "objective": model.objVal if model.status == GRB.OPTIMAL else None,
            "status": model.status,
            "mip_gap": model.MIPGap if hasattr(model, 'MIPGap') else None
        }
        
        return result
        
    except Exception as e:
        print(f"        [ERROR] {e}")
        return {"error": str(e)}


def solve_bqm_with_gurobi(bqm: BinaryQuadraticModel, timeout: int = 600) -> Dict:
    """Solve BQM as QUBO with Gurobi"""
    if not GUROBI_AVAILABLE:
        return {"error": "Gurobi not available"}
    
    print(f"      Solving BQM ({len(bqm.variables)} vars, {len(bqm.quadratic)} quad)...")
    
    try:
        print("        Creating model and optimizing...")
        model = gp.Model("BQM")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', timeout)
        
        # Create binary variables
        gurobi_vars = {var: model.addVar(vtype=GRB.BINARY, name=var) for var in bqm.variables}
        
        # Build objective
        obj_expr = bqm.offset
        
        for var, coeff in bqm.linear.items():
            obj_expr += coeff * gurobi_vars[var]
        
        for (v1, v2), coeff in bqm.quadratic.items():
            obj_expr += coeff * gurobi_vars[v1] * gurobi_vars[v2]
        
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Solve
        start = time.time()
        model.optimize()
        solve_time = time.time() - start
        print(f"        [DONE] in {solve_time:.2f}s (status={model.status})")
        
        result = {
            "success": model.status == GRB.OPTIMAL,
            "solve_time": solve_time,
            "objective": model.objVal if model.status == GRB.OPTIMAL else None,
            "status": model.status
        }
        
        return result
        
    except Exception as e:
        print(f"        [ERROR] {e}")
        return {"error": str(e)}


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def save_results_json(results: List[Dict], output_dir: Path, timestamp: str) -> Path:
    """Save detailed results as JSON"""
    output_file = output_dir / f"benchmark_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "config": {
                "problem_sizes": PROBLEM_SIZES,
                "n_foods": N_FOODS,
                "embedding_timeout": EMBEDDING_TIMEOUT,
                "solve_timeout": SOLVE_TIMEOUT,
                "formulations": FORMULATIONS,
                "decompositions": DECOMPOSITIONS
            },
            "results": results
        }, f, indent=2)
    return output_file


def save_results_csv(results: List[Dict], output_dir: Path, timestamp: str) -> Path:
    """Save flattened results as CSV for analysis"""
    output_file = output_dir / f"benchmark_results_{timestamp}.csv"
    
    rows = []
    for r in results:
        # Handle None embedding (for CQM)
        embedding = r.get("embedding") or {}
        solving = r.get("solving") or {}
        metadata = r.get("metadata") or {}
        
        row = {
            "n_farms": r.get("n_farms"),
            "formulation": r.get("formulation"),
            "decomposition": r.get("decomposition", "None"),
            "num_partitions": r.get("num_partitions", 1),
            
            # Variables and structure
            "num_variables": metadata.get("variables", 0),
            "num_quadratic": metadata.get("quadratic_terms", 0),
            "density": metadata.get("density", 0),
            
            # Embedding results (may be None for CQM)
            "embedding_success": embedding.get("success", False) if embedding else False,
            "embedding_time": embedding.get("embedding_time", 0) or 0 if embedding else 0,
            "chain_length_max": embedding.get("chain_length_max") if embedding else None,
            "chain_length_mean": embedding.get("chain_length_mean") if embedding else None,
            
            # Solving results
            "solve_success": solving.get("success", False),
            "solve_time": solving.get("solve_time", 0) or 0,
            "objective": solving.get("objective"),
            
            # Total times
            "total_embedding_time": r.get("total_embedding_time", 0),
            "total_solve_time": r.get("total_solve_time", 0),
            "total_time": r.get("total_time", 0),
        }
        rows.append(row)
    
    # Write CSV (csv module imported at top)
    if rows:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    return output_file


def save_results_markdown(results: List[Dict], output_dir: Path, timestamp: str) -> Path:
    """Generate human-readable Markdown summary"""
    output_file = output_dir / f"benchmark_summary_{timestamp}.md"
    
    lines = [
        f"# Comprehensive Benchmark Summary",
        f"",
        f"**Generated:** {timestamp}",
        f"",
        f"## Configuration",
        f"",
        f"- Problem sizes: {PROBLEM_SIZES}",
        f"- Foods per farm: {N_FOODS}",
        f"- Embedding timeout: {EMBEDDING_TIMEOUT}s",
        f"- Solve timeout: {SOLVE_TIMEOUT}s",
        f"- Formulations: {FORMULATIONS}",
        f"- Decompositions: {DECOMPOSITIONS}",
        f"",
        f"## Results Summary",
        f"",
        f"| n_farms | Formulation | Decomposition | Partitions | Embed? | Embed Time | Solve? | Solve Time | Total Time |",
        f"|---------|-------------|---------------|------------|--------|------------|--------|------------|------------|",
    ]
    
    for r in results:
        # Handle None embedding (for CQM)
        embedding = r.get("embedding") or {}
        solving = r.get("solving") or {}
        
        embed_success = "[OK]" if embedding.get("success") else "[NO]"
        solve_success = "[OK]" if solving.get("success") else "[NO]"
        embed_time = r.get("total_embedding_time", 0) or 0
        solve_time = r.get("total_solve_time", 0) or 0
        total_time = r.get("total_time", 0) or 0
        
        lines.append(
            f"| {r.get('n_farms', 'N/A')} | {r.get('formulation', 'N/A')} | "
            f"{r.get('decomposition', 'None')} | {r.get('num_partitions', 1)} | "
            f"{embed_success} | {embed_time:.1f}s | {solve_success} | {solve_time:.1f}s | {total_time:.1f}s |"
        )
    
    # Statistics
    lines.extend([
        f"",
        f"## Statistics",
        f"",
        f"- Total experiments: {len(results)}",
        f"- Successful embeddings: {sum(1 for r in results if r.get('embedding') and r['embedding'].get('success'))}",
        f"- Successful solves: {sum(1 for r in results if r.get('solving') and r['solving'].get('success'))}",
    ])
    
    # Best configurations by farm size
    lines.extend([
        f"",
        f"## Best Configurations by Problem Size",
        f"",
    ])
    
    for n_farms in PROBLEM_SIZES:
        farm_results = [r for r in results if r.get("n_farms") == n_farms]
        if farm_results:
            # Best by total time (only successful solves)
            successful = [r for r in farm_results if r.get("solving", {}).get("success")]
            if successful:
                best = min(successful, key=lambda x: x.get("total_time", float('inf')))
                lines.append(
                    f"- **{n_farms} farms**: {best.get('formulation')} + {best.get('decomposition', 'None')} "
                    f"(total: {best.get('total_time', 0):.1f}s)"
                )
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    return output_file


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def apply_decomposition(bqm: BinaryQuadraticModel, decomp_name: str) -> Tuple[List[Set], str]:
    """Apply a decomposition strategy and return partitions"""
    if decomp_name == "None" or decomp_name is None:
        return [set(bqm.variables)], "None"
    
    elif decomp_name == "Louvain":
        if not LOUVAIN_AVAILABLE:
            return None, "Louvain not available"
        partitions = decompose_louvain(bqm, max_partition_size=150)
        if not partitions or len(partitions) <= 1:
            return [set(bqm.variables)], "Louvain produced single partition"
        return partitions, None
    
    elif decomp_name == "PlotBased":
        partitions = decompose_plot_based(bqm, plots_per_partition=5)
        if not partitions or len(partitions) <= 1:
            return [set(bqm.variables)], "PlotBased produced single partition"
        return partitions, None
    
    elif decomp_name == "Multilevel":
        if not ADVANCED_DECOMP_AVAILABLE:
            return None, "Multilevel not available"
        partitions = decompose_multilevel(bqm, levels=2, partition_size=100)
        if not partitions or len(partitions) <= 1:
            return [set(bqm.variables)], "Multilevel produced single partition"
        return partitions, None
    
    elif decomp_name == "Cutset":
        if not ADVANCED_DECOMP_AVAILABLE:
            return None, "Cutset not available"
        partitions = decompose_sequential_cutset(bqm, max_cut_size=5, min_partition_size=50)
        if not partitions or len(partitions) <= 1:
            return [set(bqm.variables)], "Cutset produced single partition"
        return partitions, None
    
    elif decomp_name == "SpatialGrid":
        if not ADVANCED_DECOMP_AVAILABLE:
            return None, "SpatialGrid not available"
        partitions = decompose_spatial_grid(bqm, grid_size=5)
        if not partitions or len(partitions) <= 1:
            return [set(bqm.variables)], "SpatialGrid produced single partition"
        return partitions, None
    
    elif decomp_name == "EnergyImpact":
        if not HYBRID_AVAILABLE:
            return None, "EnergyImpact not available"
        partitions = decompose_energy_impact(bqm, partition_size=100)
        if not partitions or len(partitions) <= 1:
            return [set(bqm.variables)], "EnergyImpact produced single partition"
        return partitions, None
    
    else:
        return None, f"Unknown decomposition: {decomp_name}"


def test_single_configuration(
    n_farms: int,
    formulation: str,
    decomposition: str,
    target_graph: nx.Graph,
    experiment_id: int,
    total_experiments: int
) -> Dict:
    """Test a single configuration and return comprehensive results"""
    
    print(f"\n  [{experiment_id}/{total_experiments}] n_farms={n_farms}, form={formulation}, decomp={decomposition}")
    
    result = {
        "n_farms": n_farms,
        "formulation": formulation,
        "decomposition": decomposition,
        "num_partitions": 1,
        "metadata": {},
        "embedding": None,
        "solving": None,
        "total_embedding_time": 0,
        "total_solve_time": 0,
        "total_time": 0,
        "error": None
    }
    
    try:
        # Step 1: Build formulation
        print(f"    Building {formulation}...")
        
        if formulation == "CQM":
            cqm, meta = build_patch_cqm(n_farms)
            result["metadata"] = meta
            
            # CQM: No embedding (continuous), just solve with Gurobi
            print(f"    Solving CQM with Gurobi (no embedding for CQM)...")
            solve_result = solve_cqm_with_gurobi(cqm, SOLVE_TIMEOUT)
            result["solving"] = solve_result
            result["total_solve_time"] = solve_result.get("solve_time", 0) or 0
            result["total_time"] = result["total_solve_time"]
            
            # No decomposition for CQM
            if decomposition != "None":
                result["error"] = "CQM does not support decomposition"
                return result
            
            return result
        
        elif formulation == "BQM":
            # Convert CQM to BQM
            cqm, _ = build_patch_cqm(n_farms)
            bqm, meta = cqm_to_bqm_wrapper(cqm, "Patch_CQM")
            result["metadata"] = meta
            
        elif formulation == "SparseBQM":
            bqm, meta = build_patch_ultra_sparse_bqm(n_farms)
            result["metadata"] = meta
        
        else:
            result["error"] = f"Unknown formulation: {formulation}"
            return result
        
        # Step 2: Apply decomposition
        print(f"    Applying decomposition: {decomposition}...")
        partitions, decomp_error = apply_decomposition(bqm, decomposition)
        
        if partitions is None:
            result["error"] = decomp_error
            # Still try to solve without decomposition
            partitions = [set(bqm.variables)]
        
        result["num_partitions"] = len(partitions)
        print(f"      Created {len(partitions)} partition(s)")
        
        # Step 3: Embedding study (always try, even if dense)
        print(f"    Running embedding study...")
        if len(partitions) == 1:
            embed_result = study_embedding(bqm, target_graph, EMBEDDING_TIMEOUT)
            result["embedding"] = embed_result
            result["total_embedding_time"] = embed_result.get("embedding_time", 0) or 0
        else:
            embed_result = study_decomposed_embedding(bqm, partitions, target_graph, EMBEDDING_TIMEOUT)
            result["embedding"] = embed_result
            result["total_embedding_time"] = embed_result.get("total_embedding_time", 0) or 0
        
        # Step 4: Solve with Gurobi (always, regardless of embedding success)
        print(f"    Solving with Gurobi...")
        if len(partitions) == 1:
            solve_result = solve_bqm_with_gurobi(bqm, SOLVE_TIMEOUT)
            result["solving"] = solve_result
            result["total_solve_time"] = solve_result.get("solve_time", 0) or 0
        else:
            solve_result = solve_decomposed_bqm_with_gurobi(bqm, partitions, SOLVE_TIMEOUT)
            result["solving"] = solve_result
            result["total_solve_time"] = solve_result.get("total_solve_time", 0) or 0
        
        # Step 5: Calculate total time
        result["total_time"] = result["total_embedding_time"] + result["total_solve_time"]
        
        # Summary
        embed_status = "[OK]" if result["embedding"].get("success") else "[NO]"
        solve_status = "[OK]" if result["solving"].get("success") else "[NO]"
        print(f"    Result: Embed={embed_status} ({result['total_embedding_time']:.1f}s), "
              f"Solve={solve_status} ({result['total_solve_time']:.1f}s), "
              f"Total={result['total_time']:.1f}s")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"    [ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    return result


def run_comprehensive_benchmark():
    """Run complete systematic benchmark across all configurations"""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EMBEDDING AND SOLVING BENCHMARK v2.0")
    print("=" * 80)
    
    # Initialize
    print("\n[1/4] Initializing...")
    target_graph = get_target_graph()
    print(f"  [OK] Target graph: {len(target_graph.nodes)} nodes, {len(target_graph.edges)} edges")
    
    # Calculate total experiments
    # CQM only with None decomposition
    # BQM and SparseBQM with all decompositions
    total_experiments = 0
    for n_farms in PROBLEM_SIZES:
        total_experiments += 1  # CQM (no decomp)
        for form in ["BQM", "SparseBQM"]:
            for decomp in DECOMPOSITIONS:
                total_experiments += 1
    
    print(f"  Total configurations to test: {total_experiments}")
    print(f"  Problem sizes: {PROBLEM_SIZES}")
    print(f"  Formulations: {FORMULATIONS}")
    print(f"  Decompositions: {DECOMPOSITIONS}")
    
    # Run all experiments
    print("\n[2/4] Running experiments...")
    all_results = []
    experiment_id = 0
    benchmark_start = time.time()
    
    for n_farms in PROBLEM_SIZES:
        print(f"\n{'='*60}")
        print(f"PROBLEM SIZE: {n_farms} farms × {N_FOODS} foods = {n_farms * N_FOODS} variables")
        print(f"{'='*60}")
        
        # Test CQM (only with None decomposition)
        experiment_id += 1
        result = test_single_configuration(
            n_farms, "CQM", "None", target_graph, experiment_id, total_experiments
        )
        all_results.append(result)
        
        # Test BQM and SparseBQM with all decompositions
        for formulation in ["BQM", "SparseBQM"]:
            for decomposition in DECOMPOSITIONS:
                experiment_id += 1
                result = test_single_configuration(
                    n_farms, formulation, decomposition, target_graph, experiment_id, total_experiments
                )
                all_results.append(result)
    
    benchmark_time = time.time() - benchmark_start
    
    # Save results
    print("\n[3/4] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_file = save_results_json(all_results, OUTPUT_DIR, timestamp)
    print(f"  [OK] JSON: {json_file}")
    
    csv_file = save_results_csv(all_results, OUTPUT_DIR, timestamp)
    print(f"  [OK] CSV: {csv_file}")
    
    md_file = save_results_markdown(all_results, OUTPUT_DIR, timestamp)
    print(f"  [OK] Markdown: {md_file}")
    
    # Final summary
    print("\n[4/4] Summary...")
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Total benchmark time: {benchmark_time:.1f}s ({benchmark_time/60:.1f} min)")
    
    # Statistics
    embed_attempts = sum(1 for r in all_results if r.get("embedding"))
    embed_success = sum(1 for r in all_results if r.get("embedding") and r["embedding"].get("success"))
    solve_success = sum(1 for r in all_results if r.get("solving") and r["solving"].get("success"))
    
    print(f"\nSuccessful embeddings: {embed_success}/{embed_attempts}")
    print(f"Successful solves: {solve_success}/{len(all_results)}")
    
    # Best results by problem size
    print("\nBest configurations by problem size (by total time):")
    for n_farms in PROBLEM_SIZES:
        farm_results = [r for r in all_results if r.get("n_farms") == n_farms]
        successful = [r for r in farm_results if r.get("solving", {}).get("success")]
        if successful:
            best = min(successful, key=lambda x: x.get("total_time", float('inf')))
            print(f"  {n_farms} farms: {best.get('formulation')} + {best.get('decomposition', 'None')} "
                  f"({best.get('total_time', 0):.1f}s total)")
    
    print(f"\nOutput files:")
    print(f"  - {json_file}")
    print(f"  - {csv_file}")
    print(f"  - {md_file}")
    
    return all_results


if __name__ == "__main__":
    try:
        results = run_comprehensive_benchmark()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
