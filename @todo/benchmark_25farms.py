#!/usr/bin/env python3
"""
Benchmark for 25 farms - All methods with embedding time, solve time, and objective value.
Outputs results in a format suitable for Excel analysis.
"""

import os
import sys
import time
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=" * 100)
print("BENCHMARK: 25 FARMS - ALL METHODS")
print("=" * 100)

# Imports
print("\nImporting libraries...")
import networkx as nx
from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, cqm_to_bqm
import minorminer
import dwave_networkx as dnx

import gurobipy as gp
from gurobipy import GRB

try:
    from networkx.algorithms.community import louvain_communities
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

try:
    from advanced_decomposition_strategies import (
        decompose_multilevel,
        decompose_sequential_cutset,
        decompose_spatial_grid
    )
    ADVANCED_DECOMP_AVAILABLE = True
except ImportError:
    ADVANCED_DECOMP_AVAILABLE = False
    print("  Warning: Advanced decomposition not available")

print("  Done!")

# Configuration
N_FARMS = 25
N_FOODS = 27
EMBEDDING_TIMEOUT = 300  # 5 minutes
SOLVE_TIMEOUT = 120  # 2 minutes per partition (reduced for faster results)

# Results storage
RESULTS = []


def build_patch_cqm(n_patches: int, n_foods: int = 27) -> ConstrainedQuadraticModel:
    """Build Patch CQM with binary-only selections"""
    cqm = ConstrainedQuadraticModel()
    
    Y = {}
    for p in range(n_patches):
        for c in range(n_foods):
            Y[p, c] = Binary(f"Y_{p}_{c}")
    
    # Objective: maximize total selections (minimize negative)
    objective = sum(Y[p, c] for p in range(n_patches) for c in range(n_foods))
    cqm.set_objective(-objective)
    
    # Constraints
    for p in range(n_patches):
        cqm.add_constraint(sum(Y[p, c] for c in range(n_foods)) <= 5, label=f"patch_limit_{p}")
    
    total_selections = sum(Y[p, c] for p in range(n_patches) for c in range(n_foods))
    cqm.add_constraint(total_selections >= n_foods // 2, label="min_foods")
    
    return cqm


def build_sparse_bqm(n_patches: int, n_foods: int = 27) -> BinaryQuadraticModel:
    """Build ultra-sparse BQM with minimal quadratic terms"""
    bqm = BinaryQuadraticModel('BINARY')
    
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
    
    return bqm


def get_bqm_graph(bqm: BinaryQuadraticModel) -> nx.Graph:
    """Convert BQM to NetworkX graph"""
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    G.add_edges_from(bqm.quadratic.keys())
    return G


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


# Decomposition strategies
def decompose_none(bqm: BinaryQuadraticModel) -> List[Set]:
    return [set(bqm.variables)]


def decompose_louvain(bqm: BinaryQuadraticModel, max_size: int = 100) -> List[Set]:
    if not LOUVAIN_AVAILABLE:
        return [set(bqm.variables)]
    
    G = get_bqm_graph(bqm)
    if len(G.edges) == 0:
        return [set(bqm.variables)]
    
    communities = louvain_communities(G, seed=42)
    
    partitions = []
    current = set()
    for community in communities:
        if len(current) + len(community) <= max_size:
            current.update(community)
        else:
            if current:
                partitions.append(current)
            current = set(community)
    if current:
        partitions.append(current)
    
    return partitions if partitions else [set(bqm.variables)]


def decompose_plot_based(bqm: BinaryQuadraticModel, plots_per_partition: int = 5) -> List[Set]:
    """Group by plot index"""
    plot_vars = {}
    for var in bqm.variables:
        if var.startswith("Y_"):
            parts = var.split("_")
            if len(parts) >= 2:
                try:
                    plot_idx = int(parts[1])
                    if plot_idx not in plot_vars:
                        plot_vars[plot_idx] = set()
                    plot_vars[plot_idx].add(var)
                except ValueError:
                    pass
    
    partitions = []
    plot_indices = sorted(plot_vars.keys())
    
    for i in range(0, len(plot_indices), plots_per_partition):
        partition = set()
        for plot_idx in plot_indices[i:i + plots_per_partition]:
            partition.update(plot_vars[plot_idx])
        if partition:
            partitions.append(partition)
    
    return partitions if partitions else [set(bqm.variables)]


def study_embedding(bqm: BinaryQuadraticModel, target_graph: nx.Graph, timeout: int = 300) -> Dict:
    """Try to embed and return results"""
    n_vars = len(bqm.variables)
    n_edges = len(bqm.quadratic)
    
    if n_vars == 0:
        return {"success": True, "time": 0, "max_chain": 0, "mean_chain": 0}
    
    source_graph = get_bqm_graph(bqm)
    
    try:
        start = time.time()
        embedding = minorminer.find_embedding(source_graph, target_graph, timeout=timeout, verbose=0)
        elapsed = time.time() - start
        
        if embedding:
            chain_lengths = [len(chain) for chain in embedding.values()]
            return {
                "success": True,
                "time": elapsed,
                "max_chain": max(chain_lengths),
                "mean_chain": sum(chain_lengths) / len(chain_lengths)
            }
        else:
            return {"success": False, "time": elapsed, "max_chain": None, "mean_chain": None}
    except Exception as e:
        return {"success": False, "time": timeout, "max_chain": None, "mean_chain": None, "error": str(e)}


def solve_cqm_with_gurobi(cqm: ConstrainedQuadraticModel, timeout: int = 120) -> Dict:
    """Solve CQM with Gurobi"""
    model = gp.Model("CQM")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    
    gurobi_vars = {}
    for var_name in cqm.variables:
        gurobi_vars[var_name] = model.addVar(vtype=GRB.BINARY, name=var_name)
    
    obj_expr = 0
    for var_name, coeff in cqm.objective.linear.items():
        obj_expr += coeff * gurobi_vars[var_name]
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    for label, constraint in cqm.constraints.items():
        constr_expr = 0
        for var_name, coeff in constraint.lhs.linear.items():
            constr_expr += coeff * gurobi_vars[var_name]
        
        sense_str = str(constraint.sense)
        if sense_str == 'Sense.Le' or constraint.sense == '<=':
            model.addConstr(constr_expr <= constraint.rhs, name=label)
        elif sense_str == 'Sense.Ge' or constraint.sense == '>=':
            model.addConstr(constr_expr >= constraint.rhs, name=label)
        elif sense_str == 'Sense.Eq' or constraint.sense == '==':
            model.addConstr(constr_expr == constraint.rhs, name=label)
    
    start = time.time()
    model.optimize()
    solve_time = time.time() - start
    
    return {
        "success": model.status == GRB.OPTIMAL,
        "has_solution": model.SolCount > 0,
        "time": solve_time,
        "objective": model.objVal if model.SolCount > 0 else None,
        "status": model.status
    }


def solve_bqm_with_gurobi(bqm: BinaryQuadraticModel, timeout: int = 120) -> Dict:
    """Solve BQM with Gurobi"""
    model = gp.Model("BQM")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    
    gurobi_vars = {var: model.addVar(vtype=GRB.BINARY, name=str(var)) for var in bqm.variables}
    
    obj_expr = bqm.offset
    for var, coeff in bqm.linear.items():
        obj_expr += coeff * gurobi_vars[var]
    for (v1, v2), coeff in bqm.quadratic.items():
        obj_expr += coeff * gurobi_vars[v1] * gurobi_vars[v2]
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    start = time.time()
    model.optimize()
    solve_time = time.time() - start
    
    return {
        "success": model.status == GRB.OPTIMAL,
        "has_solution": model.SolCount > 0,
        "is_time_limit": model.status == GRB.TIME_LIMIT,
        "time": solve_time,
        "objective": model.objVal if model.SolCount > 0 else None,
        "status": model.status,
        "solution_count": model.SolCount
    }


def run_test(formulation: str, decomposition: str, target_graph: nx.Graph) -> Dict:
    """Run a single test configuration"""
    result = {
        "formulation": formulation,
        "decomposition": decomposition,
        "n_vars": 0,
        "n_quad": 0,
        "n_partitions": 1,
        "embed_success": False,
        "embed_time": None,
        "max_chain": None,
        "mean_chain": None,
        "solve_success": False,
        "solve_time": None,
        "objective": None,
        "total_time": None
    }
    
    print(f"\n  [{formulation}] + [{decomposition}]")
    
    # Build formulation
    if formulation == "CQM":
        print(f"    Building CQM...")
        cqm = build_patch_cqm(N_FARMS, N_FOODS)
        result["n_vars"] = len(cqm.variables)
        result["n_quad"] = len(cqm.objective.quadratic)
        
        # CQM: no embedding, just solve
        print(f"    Solving CQM ({result['n_vars']} vars)...")
        solve_result = solve_cqm_with_gurobi(cqm, SOLVE_TIMEOUT)
        
        result["embed_success"] = "N/A"
        result["embed_time"] = 0
        result["solve_success"] = solve_result["success"] or solve_result["has_solution"]
        result["solve_time"] = solve_result["time"]
        result["objective"] = solve_result["objective"]
        result["total_time"] = solve_result["time"]
        
        print(f"    -> Solve: {'OK' if result['solve_success'] else 'FAIL'} in {result['solve_time']:.2f}s, obj={result['objective']}")
        return result
    
    elif formulation == "BQM":
        print(f"    Building BQM from CQM...")
        cqm = build_patch_cqm(N_FARMS, N_FOODS)
        bqm_result = cqm_to_bqm(cqm, lagrange_multiplier=10.0)
        bqm = bqm_result[0] if isinstance(bqm_result, tuple) else bqm_result
        
    elif formulation == "SparseBQM":
        print(f"    Building Sparse BQM...")
        bqm = build_sparse_bqm(N_FARMS, N_FOODS)
    
    result["n_vars"] = len(bqm.variables)
    result["n_quad"] = len(bqm.quadratic)
    print(f"    BQM: {result['n_vars']} vars, {result['n_quad']} quadratic terms")
    
    # Apply decomposition
    print(f"    Applying decomposition: {decomposition}...")
    if decomposition == "None":
        partitions = decompose_none(bqm)
    elif decomposition == "Louvain":
        partitions = decompose_louvain(bqm)
    elif decomposition == "PlotBased":
        partitions = decompose_plot_based(bqm)
    elif decomposition == "Multilevel" and ADVANCED_DECOMP_AVAILABLE:
        partitions = decompose_multilevel(bqm)
    elif decomposition == "Cutset" and ADVANCED_DECOMP_AVAILABLE:
        partitions = decompose_sequential_cutset(bqm)
    elif decomposition == "SpatialGrid" and ADVANCED_DECOMP_AVAILABLE:
        partitions = decompose_spatial_grid(bqm)
    else:
        print(f"    -> Decomposition not available, using None")
        partitions = decompose_none(bqm)
    
    result["n_partitions"] = len(partitions)
    partition_sizes = [len(p) for p in partitions]
    print(f"    -> {len(partitions)} partition(s): {partition_sizes}")
    
    # Embedding study
    print(f"    Embedding...")
    total_embed_time = 0
    all_embedded = True
    max_chain_overall = 0
    mean_chains = []
    
    for i, partition in enumerate(partitions):
        sub_bqm = extract_sub_bqm(bqm, partition)
        embed_result = study_embedding(sub_bqm, target_graph, EMBEDDING_TIMEOUT)
        total_embed_time += embed_result["time"]
        
        if embed_result["success"]:
            if embed_result["max_chain"]:
                max_chain_overall = max(max_chain_overall, embed_result["max_chain"])
            if embed_result["mean_chain"]:
                mean_chains.append(embed_result["mean_chain"])
            print(f"      Partition {i+1}: OK in {embed_result['time']:.1f}s (chain max={embed_result['max_chain']})")
        else:
            all_embedded = False
            print(f"      Partition {i+1}: FAILED in {embed_result['time']:.1f}s")
    
    result["embed_success"] = all_embedded
    result["embed_time"] = total_embed_time
    result["max_chain"] = max_chain_overall if max_chain_overall > 0 else None
    result["mean_chain"] = sum(mean_chains) / len(mean_chains) if mean_chains else None
    
    # Solve with Gurobi
    print(f"    Solving...")
    total_solve_time = 0
    all_solved = True
    total_objective = 0
    
    for i, partition in enumerate(partitions):
        sub_bqm = extract_sub_bqm(bqm, partition)
        solve_result = solve_bqm_with_gurobi(sub_bqm, SOLVE_TIMEOUT)
        total_solve_time += solve_result["time"]
        
        status = "OK" if solve_result["success"] else ("PARTIAL" if solve_result["has_solution"] else "FAIL")
        if solve_result["has_solution"]:
            total_objective += solve_result["objective"]
            print(f"      Partition {i+1}: {status} in {solve_result['time']:.1f}s, obj={solve_result['objective']:.2f}")
        else:
            all_solved = False
            print(f"      Partition {i+1}: {status} in {solve_result['time']:.1f}s")
    
    result["solve_success"] = all_solved
    result["solve_time"] = total_solve_time
    result["objective"] = total_objective if all_solved else None
    result["total_time"] = total_embed_time + total_solve_time
    
    return result


def print_results_table(results: List[Dict]):
    """Print results in a nice table format"""
    print("\n" + "=" * 130)
    print("RESULTS SUMMARY - 25 FARMS")
    print("=" * 130)
    
    header = f"{'Formulation':<12} {'Decomp':<12} {'Vars':>8} {'Quad':>10} {'Parts':>6} {'Embed':>8} {'EmbTime':>10} {'MaxChain':>9} {'Solve':>8} {'SolveTime':>10} {'Objective':>12} {'TotalTime':>10}"
    print(header)
    print("-" * 130)
    
    for r in results:
        embed_str = "OK" if r["embed_success"] == True else ("N/A" if r["embed_success"] == "N/A" else "FAIL")
        solve_str = "OK" if r["solve_success"] else "FAIL"
        embed_time_str = f"{r['embed_time']:.1f}s" if r['embed_time'] is not None else "N/A"
        solve_time_str = f"{r['solve_time']:.1f}s" if r['solve_time'] is not None else "N/A"
        total_time_str = f"{r['total_time']:.1f}s" if r['total_time'] is not None else "N/A"
        max_chain_str = str(r['max_chain']) if r['max_chain'] is not None else "N/A"
        obj_str = f"{r['objective']:.2f}" if r['objective'] is not None else "N/A"
        
        row = f"{r['formulation']:<12} {r['decomposition']:<12} {r['n_vars']:>8} {r['n_quad']:>10} {r['n_partitions']:>6} {embed_str:>8} {embed_time_str:>10} {max_chain_str:>9} {solve_str:>8} {solve_time_str:>10} {obj_str:>12} {total_time_str:>10}"
        print(row)
    
    print("=" * 130)


def save_results_csv(results: List[Dict], filename: str):
    """Save results to CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {filename}")


def main():
    print(f"\nConfiguration:")
    print(f"  N_FARMS = {N_FARMS}")
    print(f"  N_FOODS = {N_FOODS}")
    print(f"  EMBEDDING_TIMEOUT = {EMBEDDING_TIMEOUT}s")
    print(f"  SOLVE_TIMEOUT = {SOLVE_TIMEOUT}s")
    
    # Get target graph (Pegasus P16)
    print("\nInitializing Pegasus P16 topology...")
    target_graph = dnx.pegasus_graph(16)
    print(f"  Nodes: {len(target_graph.nodes)}, Edges: {len(target_graph.edges)}")
    
    # Test configurations
    configs = [
        # CQM (no decomposition needed - solves directly)
        ("CQM", "None"),
        
        # BQM with various decompositions
        ("BQM", "None"),
        ("BQM", "Louvain"),
        ("BQM", "PlotBased"),
        
        # SparseBQM with various decompositions
        ("SparseBQM", "None"),
        ("SparseBQM", "Louvain"),
        ("SparseBQM", "PlotBased"),
    ]
    
    # Add advanced decompositions if available
    if ADVANCED_DECOMP_AVAILABLE:
        configs.extend([
            ("BQM", "Multilevel"),
            ("BQM", "Cutset"),
            ("BQM", "SpatialGrid"),
            ("SparseBQM", "Multilevel"),
            ("SparseBQM", "Cutset"),
            ("SparseBQM", "SpatialGrid"),
        ])
    
    print(f"\nRunning {len(configs)} configurations...")
    
    results = []
    for i, (formulation, decomposition) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing {formulation} + {decomposition}")
        result = run_test(formulation, decomposition, target_graph)
        results.append(result)
        RESULTS.append(result)
    
    # Print summary table
    print_results_table(results)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"benchmark_25farms_{timestamp}.csv"
    save_results_csv(results, csv_file)
    
    # Also save as JSON for detailed analysis
    json_file = f"benchmark_25farms_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
