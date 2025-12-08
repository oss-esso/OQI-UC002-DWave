#!/usr/bin/env python3
"""
Benchmark Hard QUBO Instances with Gurobi

This script:
1. Generates hard spin glass QUBO instances at multiple scales
2. Solves each with Gurobi (ground truth)
3. Records solve time, gap, and solution quality
4. Fits scaling curves to extract computational complexity
5. Saves results for comparison with quantum solvers

The goal is to establish classical baseline performance before testing QPU.
"""

import sys
sys.path.insert(0, '.')
import numpy as np
import networkx as nx
from collections import defaultdict
import json
import os
import time
from datetime import datetime

from dimod import BinaryQuadraticModel
import gurobipy as gp
from gurobipy import GRB

print("="*100)
print("HARD QUBO BENCHMARK: GUROBI GROUND TRUTH")
print("="*100)

# ============================================================================
# QUBO INSTANCE GENERATORS
# ============================================================================

def generate_spin_glass_qubo(n_spins, edge_density=0.1, frustration_ratio=0.5, seed=42):
    """
    Generate frustrated spin glass QUBO with planted solution.
    
    This creates instances with:
    - Sparse graph (embeddable on QPU)
    - Mixed ferro/antiferromagnetic couplings (frustration)
    - Large integrality gap (hard for classical)
    """
    np.random.seed(seed)
    
    # Generate planted solution
    planted = np.random.choice([-1, 1], size=n_spins)
    planted_binary = ((planted + 1) // 2).astype(int)  # Convert to {0,1}
    
    # Generate sparse random graph
    G = nx.gnp_random_graph(n_spins, edge_density, seed=seed)
    
    # Ensure connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            u = np.random.choice(list(components[0]))
            v = np.random.choice(list(components[i]))
            G.add_edge(u, v)
    
    # Build Ising model
    h = {}  # Linear terms
    J = {}  # Quadratic terms
    
    # Random field terms
    for i in range(n_spins):
        h[i] = np.random.uniform(-0.2, 0.2)
    
    # Couplings with frustration
    for (u, v) in G.edges():
        if np.random.random() < frustration_ratio:
            # Frustrated: coupling opposes planted solution
            J[(u, v)] = -planted[u] * planted[v] * np.random.uniform(0.5, 1.5)
        else:
            # Satisfied: coupling agrees with planted solution
            J[(u, v)] = planted[u] * planted[v] * np.random.uniform(0.5, 1.5)
    
    # Convert Ising to QUBO: s = 2x - 1
    bqm = BinaryQuadraticModel('BINARY')
    
    for i in range(n_spins):
        bqm.add_variable(f"x{i}", 2 * h[i])
    
    for (u, v), j in J.items():
        bqm.add_interaction(f"x{u}", f"x{v}", 4 * j)
        bqm.add_variable(f"x{u}", -2 * j)
        bqm.add_variable(f"x{v}", -2 * j)
    
    return bqm, G, planted_binary, h, J


def generate_ultra_sparse_qubo(n_farms, n_types=6, k_neighbors=4, frustration_prob=0.3, seed=42):
    """
    Generate ultra-sparse farm-type QUBO (from earlier analysis).
    
    This has bounded degree but one-hot penalties make it easier for LP.
    """
    np.random.seed(seed)
    
    # Build spatial adjacency
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {}
    farm_names = []
    
    for i in range(n_farms):
        row, col = i // side, i % side
        farm_names.append(f"F{i}")
        positions[farm_names[-1]] = (row, col)
    
    G = nx.Graph()
    G.add_nodes_from(farm_names)
    
    for f1 in farm_names:
        distances = [(np.sqrt((positions[f1][0]-positions[f2][0])**2 + 
                             (positions[f1][1]-positions[f2][1])**2), f2) 
                    for f2 in farm_names if f1 != f2]
        distances.sort()
        for _, f2 in distances[:k_neighbors]:
            G.add_edge(f1, f2)
    
    # Farm types
    type_names = [f"T{i}" for i in range(n_types)]
    type_benefits = {t: np.random.uniform(0.2, 0.5) for t in type_names}
    
    # Compatibility matrix with frustration
    compatibility = np.zeros((n_types, n_types))
    for i in range(n_types):
        for j in range(i+1, n_types):
            if np.random.random() < frustration_prob:
                compatibility[i,j] = compatibility[j,i] = np.random.uniform(-0.15, -0.05)
            else:
                compatibility[i,j] = compatibility[j,i] = np.random.uniform(-0.05, 0.1)
    
    # Build BQM
    bqm = BinaryQuadraticModel('BINARY')
    same_farm_penalty = 10.0
    
    for farm in farm_names:
        for t in type_names:
            bqm.add_variable(f"x_{farm}_{t}", -type_benefits[t])
    
    for farm in farm_names:
        for i, t1 in enumerate(type_names):
            for j, t2 in enumerate(type_names):
                if i < j:
                    bqm.add_interaction(f"x_{farm}_{t1}", f"x_{farm}_{t2}", same_farm_penalty)
    
    for f1, f2 in G.edges():
        for i, t1 in enumerate(type_names):
            for j, t2 in enumerate(type_names):
                interaction = -compatibility[i, j]
                if abs(interaction) > 1e-6:
                    bqm.add_interaction(f"x_{f1}_{t1}", f"x_{f2}_{t2}", interaction)
    
    return bqm, G


# ============================================================================
# GUROBI SOLVER
# ============================================================================

def solve_qubo_with_gurobi(bqm, time_limit=300, mip_gap=1e-4, threads=None):
    """
    Solve QUBO with Gurobi and collect detailed statistics.
    
    Returns:
        dict with solve stats, solution, gaps, etc.
    """
    m = gp.Model("QUBO")
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPGap', mip_gap)
    if threads:
        m.setParam('Threads', threads)
    
    # Variables
    x = {}
    for v in bqm.variables:
        x[v] = m.addVar(vtype=GRB.BINARY, name=v)
    m.update()
    
    # Objective
    obj = gp.quicksum(bqm.linear[v] * x[v] for v in bqm.linear)
    obj += gp.quicksum(bqm.quadratic[(v1,v2)] * x[v1] * x[v2] for (v1,v2) in bqm.quadratic)
    m.setObjective(obj, GRB.MINIMIZE)
    
    # Solve MIP
    t0 = time.time()
    m.optimize()
    solve_time = time.time() - t0
    
    # Extract solution
    solution = {}
    mip_obj = None
    if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
        mip_obj = m.ObjVal
        for v in m.getVars():
            solution[v.VarName] = int(round(v.X))
    
    # LP relaxation
    rel = m.relax()
    rel.optimize()
    lp_obj = rel.ObjVal if rel.status == GRB.OPTIMAL else None
    
    # Count fractional variables
    frac_count = 0
    if rel.status == GRB.OPTIMAL:
        for v in rel.getVars():
            if abs(v.X - round(v.X)) > 1e-6:
                frac_count += 1
    
    # Integrality gap
    gap = None
    if mip_obj is not None and lp_obj is not None:
        gap = abs(mip_obj - lp_obj) / max(1.0, abs(mip_obj))
    
    return {
        'status': m.status,
        'status_string': {GRB.OPTIMAL: 'OPTIMAL', GRB.TIME_LIMIT: 'TIME_LIMIT', 
                         GRB.INFEASIBLE: 'INFEASIBLE'}.get(m.status, 'UNKNOWN'),
        'mip_obj': mip_obj,
        'lp_obj': lp_obj,
        'integrality_gap': gap,
        'solve_time': solve_time,
        'node_count': m.NodeCount,
        'solution_count': m.SolCount,
        'frac_count': frac_count,
        'frac_ratio': frac_count / len(bqm.variables),
        'solution': solution,
        'mip_gap_achieved': m.MIPGap if m.status == GRB.OPTIMAL else None
    }


# ============================================================================
# BENCHMARK SUITE
# ============================================================================

def benchmark_spin_glass_scaling():
    """Benchmark spin glass instances at multiple scales."""
    
    print("\n" + "="*100)
    print("BENCHMARK 1: SPIN GLASS INSTANCES")
    print("="*100)
    
    # Size configurations: (n_spins, edge_density)
    configs = [
        (20, 0.15),
        (50, 0.12),
        (100, 0.10),
        (200, 0.08),
        (500, 0.05),
        (1000, 0.04),
        (2000, 0.03),
    ]
    
    results = []
    
    print(f"\n{'Size':<8} {'Edges':<10} {'Density':<10} {'MIP Obj':<12} {'LP Obj':<12} {'Gap':<8} {'Time(s)':<10} {'Nodes':<12} {'Status'}")
    print("-"*110)
    
    for n_spins, density in configs:
        # Generate instance
        bqm, G, planted, h, J = generate_spin_glass_qubo(n_spins, density, frustration_ratio=0.5, seed=42)
        
        n_edges = len(bqm.quadratic)
        
        # Solve with Gurobi
        stats = solve_qubo_with_gurobi(bqm, time_limit=300, threads=8)
        
        # Display
        mip_str = f"{stats['mip_obj']:.4f}" if stats['mip_obj'] else "N/A"
        lp_str = f"{stats['lp_obj']:.4f}" if stats['lp_obj'] else "N/A"
        gap_str = f"{stats['integrality_gap']:.2%}" if stats['integrality_gap'] else "N/A"
        
        print(f"{n_spins:<8} {n_edges:<10} {density:<10.3f} {mip_str:<12} {lp_str:<12} {gap_str:<8} {stats['solve_time']:<10.2f} {stats['node_count']:<12} {stats['status_string']}")
        
        # Save result
        results.append({
            'instance_type': 'spin_glass',
            'n_spins': n_spins,
            'n_edges': n_edges,
            'edge_density': density,
            'frustration_ratio': 0.5,
            **{k: v for k, v in stats.items() if k != 'solution'}  # Exclude solution for space
        })
    
    return results


def benchmark_farm_type_scaling():
    """Benchmark farm-type instances at multiple scales."""
    
    print("\n" + "="*100)
    print("BENCHMARK 2: FARM-TYPE ASSIGNMENT INSTANCES")
    print("="*100)
    
    # Size configurations: (n_farms, n_types, k_neighbors)
    configs = [
        (25, 6, 4),
        (50, 6, 4),
        (100, 6, 4),
        (200, 6, 4),
        (500, 6, 4),
        (1000, 6, 4),
    ]
    
    results = []
    
    print(f"\n{'Farms':<8} {'Types':<8} {'Vars':<10} {'MIP Obj':<12} {'LP Obj':<12} {'Gap':<8} {'Time(s)':<10} {'Nodes':<12} {'Status'}")
    print("-"*100)
    
    for n_farms, n_types, k_neighbors in configs:
        # Generate instance
        bqm, G = generate_ultra_sparse_qubo(n_farms, n_types, k_neighbors, frustration_prob=0.3, seed=42)
        
        n_vars = len(bqm.variables)
        
        # Solve with Gurobi
        stats = solve_qubo_with_gurobi(bqm, time_limit=300, threads=8)
        
        # Display
        mip_str = f"{stats['mip_obj']:.4f}" if stats['mip_obj'] else "N/A"
        lp_str = f"{stats['lp_obj']:.4f}" if stats['lp_obj'] else "N/A"
        gap_str = f"{stats['integrality_gap']:.2%}" if stats['integrality_gap'] else "N/A"
        
        print(f"{n_farms:<8} {n_types:<8} {n_vars:<10} {mip_str:<12} {lp_str:<12} {gap_str:<8} {stats['solve_time']:<10.2f} {stats['node_count']:<12} {stats['status_string']}")
        
        # Save result
        results.append({
            'instance_type': 'farm_type',
            'n_farms': n_farms,
            'n_types': n_types,
            'k_neighbors': k_neighbors,
            'n_vars': n_vars,
            **{k: v for k, v in stats.items() if k != 'solution'}
        })
    
    return results


# ============================================================================
# SCALING ANALYSIS
# ============================================================================

def fit_scaling_curve(results, x_key='n_spins', y_key='solve_time'):
    """
    Fit power-law scaling: y = a * x^b
    
    Returns: (a, b, r_squared)
    """
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    
    # Extract data
    x = np.array([r[x_key] for r in results if r[y_key] is not None])
    y = np.array([r[y_key] for r in results if r[y_key] is not None])
    
    if len(x) < 3:
        return None, None, None
    
    # Fit power law: y = a * x^b
    def power_law(x, a, b):
        return a * x**b
    
    try:
        params, _ = curve_fit(power_law, x, y, p0=[1, 1], maxfev=10000)
        a, b = params
        
        y_pred = power_law(x, a, b)
        r2 = r2_score(y, y_pred)
        
        return a, b, r2
    except:
        return None, None, None


def analyze_scaling(results, instance_type):
    """Analyze and display scaling behavior."""
    
    print(f"\n" + "="*100)
    print(f"SCALING ANALYSIS: {instance_type.upper()}")
    print("="*100)
    
    # Determine x variable
    x_key = 'n_spins' if instance_type == 'spin_glass' else 'n_farms'
    
    # Fit curves for different metrics
    metrics = [
        ('solve_time', 'Solve Time'),
        ('node_count', 'Branch-and-Bound Nodes'),
        ('integrality_gap', 'Integrality Gap'),
    ]
    
    print(f"\nPower-law fits: y = a × {x_key}^b\n")
    print(f"{'Metric':<25} {'a (coeff)':<15} {'b (exponent)':<15} {'R²':<10} {'Complexity'}")
    print("-"*80)
    
    scaling_fits = {}
    
    for metric_key, metric_name in metrics:
        a, b, r2 = fit_scaling_curve(results, x_key=x_key, y_key=metric_key)
        
        if a is not None:
            # Classify complexity
            if b < 1.5:
                complexity = "SUB-QUADRATIC ✓"
            elif b < 2.5:
                complexity = "QUADRATIC"
            elif b < 3.5:
                complexity = "CUBIC"
            else:
                complexity = "SUPER-CUBIC ✗"
            
            print(f"{metric_name:<25} {a:<15.6f} {b:<15.3f} {r2:<10.4f} {complexity}")
            
            scaling_fits[metric_key] = {'a': a, 'b': b, 'r2': r2, 'complexity': complexity}
        else:
            print(f"{metric_name:<25} {'N/A':<15} {'N/A':<15} {'N/A':<10} N/A")
    
    return scaling_fits


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def main():
    print(f"\nStarting benchmark at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using Gurobi version {gp.gurobi.version()}")
    
    # Run benchmarks
    spin_glass_results = benchmark_spin_glass_scaling()
    farm_type_results = benchmark_farm_type_scaling()
    
    # Analyze scaling
    spin_glass_scaling = analyze_scaling(spin_glass_results, 'spin_glass')
    farm_type_scaling = analyze_scaling(farm_type_results, 'farm_type')
    
    # Save results
    os.makedirs('@todo/hardness_output', exist_ok=True)
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'gurobi_version': str(gp.gurobi.version()),
        'spin_glass': {
            'results': spin_glass_results,
            'scaling_fits': spin_glass_scaling
        },
        'farm_type': {
            'results': farm_type_results,
            'scaling_fits': farm_type_scaling
        }
    }
    
    out_path = '@todo/hardness_output/gurobi_ground_truth_benchmark.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*100}")
    print(f"Results saved to: {out_path}")
    print(f"{'='*100}")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  Spin Glass instances: {len(spin_glass_results)} scales benchmarked")
    print(f"  Farm Type instances: {len(farm_type_results)} scales benchmarked")
    print(f"  Total runtime: {sum(r['solve_time'] for r in spin_glass_results + farm_type_results):.1f}s")


if __name__ == "__main__":
    main()
