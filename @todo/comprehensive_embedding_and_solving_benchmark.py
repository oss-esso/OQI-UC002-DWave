#!/usr/bin/env python3
"""
Comprehensive Embedding and Solving Benchmark

Implements the COMPREHENSIVE_DECOMPOSITION_PLAN.md strategy:
1. Build multiple formulations (CQM, BQM variants)
2. Test embedding feasibility and timing (no QPU billing)
3. Apply decomposition strategies where needed
4. Solve with Gurobi (classical baseline)
5. Compare embedding time vs solve time

Formulations:
- Farm CQM (continuous + binary)
- Farm BQM (from CQM conversion)
- Patch CQM (binary only)
- Patch BQM (from CQM)
- Patch Direct BQM (minimal slack)
- Patch Ultra-Sparse BQM (ultra-minimal quadratic)

Decomposition Strategies:
- None (direct embedding)
- Louvain graph partitioning
- Plot-based partitioning
- Energy-impact decomposition (dwave-hybrid)

Solvers:
- Gurobi (MINLP/MILP for CQM, QUBO for BQM)
- Embedding study (timing only, no QPU)

Author: Generated for OQI-UC002-DWave comprehensive benchmark
Date: 2025-11-27
"""

import os
import sys
import time
import json
import statistics
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=" * 80)
print("COMPREHENSIVE EMBEDDING AND SOLVING BENCHMARK")
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
        analyze_decomposition_quality
    )
    ADVANCED_DECOMP_AVAILABLE = True
except ImportError:
    ADVANCED_DECOMP_AVAILABLE = False
    print("  Warning: Advanced decomposition strategies not available.")

print(f"  ✅ Imports done in {time.time() - import_start:.2f}s")

# Configuration
PROBLEM_SIZES = [25]  # Number of units (farms/patches) - focus on size 25 for decomposition study
N_FOODS = 27
EMBEDDING_TIMEOUT = 180  # 3 minutes per partition (study_binary_plot_embedding used 120s successfully)
SOLVE_TIMEOUT = 100  # 10 minutes for Gurobi
SKIP_DENSE_EMBEDDING = True  # Skip embedding if density > threshold
DENSITY_THRESHOLD = 0.5  # 30% - problems this dense rarely embed

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
    """Study embedding feasibility and timing"""
    density = len(bqm.quadratic) / (len(bqm.variables) ** 2) if len(bqm.variables) > 0 else 0
    print(f"      Testing embedding: {len(bqm.variables)} vars, {len(bqm.quadratic)} edges (density={density:.3f})")
    
    result = {
        "attempted": True,
        "success": False,
        "embedding_time": None,
        "chain_length_max": None,
        "chain_length_mean": None,
        "num_chains": 0,
        "error": None,
        "skipped": False
    }
    
    # Skip very dense problems - they won't embed
    if SKIP_DENSE_EMBEDDING and density > DENSITY_THRESHOLD:
        result["skipped"] = True
        result["error"] = f"Skipped - density {density:.3f} > threshold {DENSITY_THRESHOLD}"
        print(f"      ⊘ Skipped (too dense: {density:.3f} > {DENSITY_THRESHOLD})")
        return result
    
    source_graph = get_bqm_graph(bqm)
    
    try:
        print(f"      Running minorminer (timeout={timeout}s)...", flush=True)
        start = time.time()
        embedding = minorminer.find_embedding(source_graph, target_graph, timeout=timeout, verbose=0)
        elapsed = time.time() - start
        
        if embedding:
            result["success"] = True
            result["embedding_time"] = elapsed
            result["num_chains"] = len(embedding)
            
            chain_lengths = [len(chain) for chain in embedding.values()]
            result["chain_length_max"] = max(chain_lengths)
            result["chain_length_mean"] = statistics.mean(chain_lengths)
            print(f"      ✓ Embedded in {elapsed:.1f}s (chains: max={result['chain_length_max']}, mean={result['chain_length_mean']:.1f})")
        else:
            result["error"] = f"No embedding found in {elapsed:.1f}s"
            print(f"      ✗ No embedding (tried for {elapsed:.1f}s)")
            
    except Exception as e:
        result["error"] = str(e)
        print(f"      ✗ Error: {e}")
    
    return result
    
    return result


def study_decomposed_embedding(bqm: BinaryQuadraticModel, partitions: List[Set],
                               target_graph: nx.Graph, timeout: int = 300) -> Dict:
    """Study embedding for decomposed problem"""
    print(f"      Embedding {len(partitions)} partitions...")
    
    result = {
        "num_partitions": len(partitions),
        "partition_sizes": [len(p) for p in partitions],
        "partition_results": [],
        "total_embedding_time": 0,
        "all_embedded": True
    }
    
    for i, partition in enumerate(partitions):
        sub_bqm = extract_sub_bqm(bqm, partition)
        partition_result = study_embedding(sub_bqm, target_graph, timeout)
        
        result["partition_results"].append(partition_result)
        if partition_result["success"]:
            result["total_embedding_time"] += partition_result["embedding_time"]
        else:
            result["all_embedded"] = False
    
    return result


def solve_decomposed_bqm_with_gurobi(bqm: BinaryQuadraticModel, partitions: List[Set], timeout: int = 600) -> Dict:
    """Solve each partition independently with Gurobi and aggregate results"""
    if not GUROBI_AVAILABLE:
        return {"error": "Gurobi not available"}
    
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
        print(f"        ✓ Done in {solve_time:.2f}s (status={model.status})")
        
        result = {
            "success": model.status == GRB.OPTIMAL,
            "solve_time": solve_time,
            "objective": model.objVal if model.status == GRB.OPTIMAL else None,
            "status": model.status,
            "mip_gap": model.MIPGap if hasattr(model, 'MIPGap') else None
        }
        
        return result
        
    except Exception as e:
        print(f"        ✗ Error: {e}")
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
        print(f"        ✓ Done in {solve_time:.2f}s (status={model.status})")
        
        result = {
            "success": model.status == GRB.OPTIMAL,
            "solve_time": solve_time,
            "objective": model.objVal if model.status == GRB.OPTIMAL else None,
            "status": model.status
        }
        
        return result
        
    except Exception as e:
        print(f"        ✗ Error: {e}")
        return {"error": str(e)}


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_comprehensive_benchmark():
    """Run complete benchmark following COMPREHENSIVE_DECOMPOSITION_PLAN.md"""
    
    print("\n[2/6] Getting target graph (QPU topology)...")
    target_graph = get_target_graph()
    print(f"  ✅ Target graph: {len(target_graph.nodes)} nodes, {len(target_graph.edges)} edges")
    
    all_results = []
    
    for n_units in PROBLEM_SIZES:
        print(f"\n{'='*80}")
        print(f"TESTING PROBLEM SIZE: {n_units} units")
        print(f"{'='*80}")
        
        # =====================================================================
        # FARM SCENARIO
        # =====================================================================
        print(f"\n[3/6] Farm Scenario ({n_units} farms)...")
        
        # Farm CQM
        print("  Building Farm CQM...")
        farm_cqm, farm_cqm_meta = build_farm_cqm(n_units)
        
        # Solve with Gurobi
        print("    Solving Farm CQM with Gurobi...")
        farm_cqm_gurobi = solve_cqm_with_gurobi(farm_cqm, SOLVE_TIMEOUT)
        
        all_results.append({
            "scenario": "Farm",
            "formulation": "CQM",
            "n_units": n_units,
            "metadata": farm_cqm_meta,
            "decomposition": None,
            "embedding": None,
            "solving": farm_cqm_gurobi
        })
        
        # NOTE: Farm CQM contains continuous variables, cannot convert to BQM
        # Skip BQM conversion for Farm scenario - it's inherently mixed-integer
        print("  (Skipping BQM conversion for Farm - continuous variables not supported)")
        
        # =====================================================================
        # PATCH SCENARIO
        # =====================================================================
        print(f"\n[4/6] Patch Scenario ({n_units} patches)...")
        
        # Patch CQM
        print("  Building Patch CQM...")
        patch_cqm, patch_cqm_meta = build_patch_cqm(n_units)
        patch_cqm_gurobi = solve_cqm_with_gurobi(patch_cqm, SOLVE_TIMEOUT)
        
        all_results.append({
            "scenario": "Patch",
            "formulation": "CQM",
            "n_units": n_units,
            "metadata": patch_cqm_meta,
            "decomposition": None,
            "embedding": None,
            "solving": patch_cqm_gurobi
        })
        
        # Patch BQM (from CQM)
        print("  Converting Patch CQM to BQM...")
        patch_bqm_cqm, patch_bqm_cqm_meta = cqm_to_bqm_wrapper(patch_cqm, "Patch_CQM")
        patch_bqm_cqm_embed = study_embedding(patch_bqm_cqm, target_graph, EMBEDDING_TIMEOUT)
        patch_bqm_cqm_gurobi = solve_bqm_with_gurobi(patch_bqm_cqm, SOLVE_TIMEOUT)
        
        all_results.append({
            "scenario": "Patch",
            "formulation": "BQM_from_CQM",
            "n_units": n_units,
            "metadata": patch_bqm_cqm_meta,
            "decomposition": None,
            "embedding": patch_bqm_cqm_embed,
            "solving": patch_bqm_cqm_gurobi
        })
        
        # Patch Direct BQM
        print("  Building Patch Direct BQM...")
        patch_bqm_direct, patch_bqm_direct_meta = build_patch_direct_bqm(n_units)
        patch_bqm_direct_embed = study_embedding(patch_bqm_direct, target_graph, EMBEDDING_TIMEOUT)
        patch_bqm_direct_gurobi = solve_bqm_with_gurobi(patch_bqm_direct, SOLVE_TIMEOUT)
        
        all_results.append({
            "scenario": "Patch",
            "formulation": "Direct_BQM",
            "n_units": n_units,
            "metadata": patch_bqm_direct_meta,
            "decomposition": None,
            "embedding": patch_bqm_direct_embed,
            "solving": patch_bqm_direct_gurobi
        })
        
        # Patch Ultra-Sparse BQM
        print("  Building Patch Ultra-Sparse BQM...")
        patch_bqm_sparse, patch_bqm_sparse_meta = build_patch_ultra_sparse_bqm(n_units)
        patch_bqm_sparse_embed = study_embedding(patch_bqm_sparse, target_graph, EMBEDDING_TIMEOUT)
        patch_bqm_sparse_gurobi = solve_bqm_with_gurobi(patch_bqm_sparse, SOLVE_TIMEOUT)
        
        all_results.append({
            "scenario": "Patch",
            "formulation": "UltraSparse_BQM",
            "n_units": n_units,
            "metadata": patch_bqm_sparse_meta,
            "decomposition": None,
            "embedding": patch_bqm_sparse_embed,
            "solving": patch_bqm_sparse_gurobi
        })
        
        # Test decompositions on Direct BQM and Ultra-Sparse BQM (the good formulations!)
        # Skip BQM_from_CQM - it's too dense even after decomposition
        print(f"\n  Testing decompositions on embeddable formulations...")
        
        for bqm, meta, name in [
            (patch_bqm_direct, patch_bqm_direct_meta, "Direct_BQM"),
            (patch_bqm_sparse, patch_bqm_sparse_meta, "UltraSparse_BQM"),
        ]:
            print(f"\n    Decomposing {name} ({len(bqm.variables)} vars, {len(bqm.quadratic)} edges, density={meta.get('density', 0):.3f})...")
            
            # Strategy 1: Louvain graph partitioning
            if LOUVAIN_AVAILABLE:
                print(f"      [1/5] Louvain decomposition...")
                louvain_parts = decompose_louvain(bqm, max_partition_size=150)  # Match study_binary_plot_embedding
                if louvain_parts and len(louvain_parts) > 1:
                    print(f"        Created {len(louvain_parts)} partitions")
                    louvain_embed = study_decomposed_embedding(bqm, louvain_parts, target_graph, EMBEDDING_TIMEOUT)
                    louvain_solve = solve_decomposed_bqm_with_gurobi(bqm, louvain_parts, SOLVE_TIMEOUT)
                    all_results.append({
                        "scenario": "Patch",
                        "formulation": name,
                        "n_units": n_units,
                        "metadata": meta,
                        "decomposition": "Louvain",
                        "embedding": louvain_embed,
                        "solving": louvain_solve
                    })
            
            # Strategy 2: Plot-based partitioning
            print(f"      [2/5] Plot-based decomposition...")
            plot_parts = decompose_plot_based(bqm, plots_per_partition=5)
            if plot_parts and len(plot_parts) > 1:
                print(f"        Created {len(plot_parts)} partitions (5 plots each)")
                plot_embed = study_decomposed_embedding(bqm, plot_parts, target_graph, EMBEDDING_TIMEOUT)
                plot_solve = solve_decomposed_bqm_with_gurobi(bqm, plot_parts, SOLVE_TIMEOUT)
                all_results.append({
                    "scenario": "Patch",
                    "formulation": name,
                    "n_units": n_units,
                    "metadata": meta,
                    "decomposition": "PlotBased",
                    "embedding": plot_embed,
                    "solving": plot_solve
                })
            
            # Strategy 3: Energy-impact decomposition
            if HYBRID_AVAILABLE:
                print(f"      [3/5] Energy-impact decomposition...")
                energy_parts = decompose_energy_impact(bqm, partition_size=100)
                if energy_parts and len(energy_parts) > 1:
                    print(f"        Created {len(energy_parts)} partitions")
                    energy_embed = study_decomposed_embedding(bqm, energy_parts, target_graph, EMBEDDING_TIMEOUT)
                    energy_solve = solve_decomposed_bqm_with_gurobi(bqm, energy_parts, SOLVE_TIMEOUT)
                    all_results.append({
                        "scenario": "Patch",
                        "formulation": name,
                        "n_units": n_units,
                        "metadata": meta,
                        "decomposition": "EnergyImpact",
                        "embedding": energy_embed,
                        "solving": energy_solve
                    })
            
            # Strategy 4: Multilevel decomposition (ML-QLS)
            if ADVANCED_DECOMP_AVAILABLE:
                print(f"      [4/5] Multilevel (ML-QLS) decomposition...")
                ml_parts = decompose_multilevel(bqm, levels=2, partition_size=50)
                if ml_parts and len(ml_parts) > 1:
                    print(f"        Created {len(ml_parts)} partitions")
                    ml_embed = study_decomposed_embedding(bqm, ml_parts, target_graph, EMBEDDING_TIMEOUT)
                    ml_solve = solve_decomposed_bqm_with_gurobi(bqm, ml_parts, SOLVE_TIMEOUT)
                    all_results.append({
                        "scenario": "Patch",
                        "formulation": name,
                        "n_units": n_units,
                        "metadata": meta,
                        "decomposition": "Multilevel_MLQLS",
                        "embedding": ml_embed,
                        "solving": ml_solve
                    })
            
            # Strategy 5: Sequential cut-set reduction
            if ADVANCED_DECOMP_AVAILABLE:
                print(f"      [5/5] Sequential cut-set decomposition...")
                cutset_parts = decompose_sequential_cutset(bqm, max_cut_size=5, min_partition_size=100)
                if cutset_parts and len(cutset_parts) > 1:
                    print(f"        Created {len(cutset_parts)} partitions")
                    cutset_embed = study_decomposed_embedding(bqm, cutset_parts, target_graph, EMBEDDING_TIMEOUT)
                    cutset_solve = solve_decomposed_bqm_with_gurobi(bqm, cutset_parts, SOLVE_TIMEOUT)
                    all_results.append({
                        "scenario": "Patch",
                        "formulation": name,
                        "n_units": n_units,
                        "metadata": meta,
                        "decomposition": "SequentialCutSet",
                        "embedding": cutset_embed,
                        "solving": cutset_solve
                    })
            
            # Strategy 5: Sequential cut-set reduction
            if ADVANCED_DECOMP_AVAILABLE:
                print(f"      [5/5] Sequential cut-set decomposition...")
                cutset_parts = decompose_sequential_cutset(bqm, max_cut_size=3, min_partition_size=30)
                if cutset_parts and len(cutset_parts) > 1:
                    print(f"        Created {len(cutset_parts)} partitions")
                    cutset_embed = study_decomposed_embedding(bqm, cutset_parts, target_graph, EMBEDDING_TIMEOUT)
                    all_results.append({
                        "scenario": "Patch",
                        "formulation": name,
                        "n_units": n_units,
                        "metadata": meta,
                        "decomposition": "SequentialCutSet",
                        "embedding": cutset_embed,
                        "solving": solve_bqm_with_gurobi(bqm, SOLVE_TIMEOUT)
                    })
    
    # Save results
    print(f"\n[5/6] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"comprehensive_benchmark_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "problem_sizes": PROBLEM_SIZES,
            "results": all_results
        }, f, indent=2)
    
    print(f"  ✅ Results saved to {output_file}")
    
    # Generate summary
    print(f"\n[6/6] Summary...")
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Output: {output_file}")
    
    # Quick stats
    embedded_count = sum(1 for r in all_results if r.get("embedding") and r["embedding"].get("success"))
    solved_count = sum(1 for r in all_results if r.get("solving") and r["solving"].get("success"))
    
    print(f"\nSuccessful embeddings: {embedded_count}/{sum(1 for r in all_results if r.get('embedding'))}")
    print(f"Successful solves: {solved_count}/{len(all_results)}")
    
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
