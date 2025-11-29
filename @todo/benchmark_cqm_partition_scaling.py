"""
CQM Partition Scaling Benchmark: Embed Time, Solve Time, Objective up to 200 Plots

This benchmark tests how different CQM partition methods scale with problem size.
We measure:
1. Partition time (analogous to "embed" time - overhead of decomposition)
2. Solve time (actual optimization)
3. Objective value quality vs ground truth
4. Constraint violations

Partition methods tested:
- None (full problem)
- PlotBased (natural decomposition)
- Spectral(4) (best cut ratio)
- MasterSubproblem (two-stage)
"""

import os
import sys
import time
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import warnings
warnings.filterwarnings('ignore')

from dimod import ConstrainedQuadraticModel, Binary
import networkx as nx
import gurobipy as gp

from src.scenarios import load_food_data
from Utils import patch_sampler

# Try imports
try:
    from sklearn.cluster import SpectralClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ============================================================================
# CONFIGURATION
# ============================================================================
PLOT_SIZES = [10, 25, 50, 75, 100, 125, 150, 175, 200]
SOLVE_TIMEOUT = 60  # seconds per method
PARTITION_TIMEOUT = 30  # seconds per partition

print("="*80)
print("CQM PARTITION SCALING BENCHMARK")
print("="*80)
print(f"Plot sizes to test: {PLOT_SIZES}")
print(f"Solve timeout: {SOLVE_TIMEOUT}s per method")
print()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_problem_data(n_farms):
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
    
    food_group_constraints = {
        'Proteins': {'min': 1, 'max': 5},
        'Fruits': {'min': 1, 'max': 5},
        'Legumes': {'min': 1, 'max': 5},
        'Staples': {'min': 1, 'max': 5},
        'Vegetables': {'min': 1, 'max': 5}
    }
    
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
    
    return {
        'foods': foods,
        'food_groups': food_groups,
        'food_to_group': food_to_group,
        'group_name_mapping': group_name_mapping,
        'food_benefits': food_benefits,
        'weights': weights,
        'land_availability': land_availability,
        'patch_names': patch_names,
        'food_group_constraints': food_group_constraints
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


def partition_master_subproblem(cqm, data):
    """Two-level partition: Master (U vars) + Subproblems (Y vars per patch)."""
    foods = data['foods']
    patch_names = data['patch_names']
    
    partitions = []
    
    # Master partition: all U variables
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


# ============================================================================
# SOLVERS
# ============================================================================

def solve_full_cqm_gurobi(cqm, data, timeout=60):
    """Solve full CQM with Gurobi (ground truth)."""
    foods = data['foods']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    patch_names = data['patch_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
    group_name_mapping = data.get('group_name_mapping', {})
    reverse_mapping = {v: k for k, v in group_name_mapping.items()}
    total_area = sum(land_availability.values())
    
    model = gp.Model("FullCQM")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    Y = {}
    for patch in patch_names:
        for food in foods:
            Y[(patch, food)] = model.addVar(vtype=gp.GRB.BINARY, name=f"Y_{patch}_{food}")
    
    U = {}
    for food in foods:
        U[food] = model.addVar(vtype=gp.GRB.BINARY, name=f"U_{food}")
    
    obj = 0
    for patch in patch_names:
        patch_area = land_availability[patch]
        for food in foods:
            obj += food_benefits[food] * patch_area * Y[(patch, food)]
    obj = obj / total_area
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    
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
    
    start = time.time()
    model.optimize()
    solve_time = time.time() - start
    
    if model.Status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.SUBOPTIMAL] and model.SolCount > 0:
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
            'status': 'OPTIMAL' if model.Status == gp.GRB.OPTIMAL else 'SUBOPTIMAL',
            'success': True
        }
    
    return {
        'objective': 0,
        'solve_time': solve_time,
        'solution': {},
        'status': f'Status {model.Status}',
        'success': False
    }


def solve_partition_independently(partition, data, fixed_vars=None, timeout=30):
    """Solve a single partition of the CQM."""
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
        gp_vars[var_name] = model.addVar(vtype=gp.GRB.BINARY, name=var_name)
    
    # Objective
    obj = 0
    for patch in patch_names:
        patch_area = land_availability[patch]
        for food in foods:
            var_name = f"Y_{patch}_{food}"
            if var_name in partition:
                obj += food_benefits[food] * patch_area * gp_vars[var_name]
    obj = obj / total_area
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    
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
    
    model.optimize()
    
    if model.Status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.SUBOPTIMAL] and model.SolCount > 0:
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
        result = solve_partition_independently(partition, data, fixed_vars=all_solutions, timeout=timeout_per_partition)
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
    group_name_mapping = data.get('group_name_mapping', {})
    reverse_mapping = {v: k for k, v in group_name_mapping.items()}
    
    violations = []
    
    for patch in patch_names:
        count = sum(1 for food in foods if solution.get(f"Y_{patch}_{food}", 0) == 1)
        if count > 1:
            violations.append(f"Patch {patch}: {count} foods")
    
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
    
    return violations


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def benchmark_single_size(n_plots):
    """Run benchmark for a single problem size."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {n_plots} plots")
    print(f"{'='*60}")
    
    # Load data
    data = load_problem_data(n_plots)
    n_foods = len(data['foods'])
    n_vars = n_plots * n_foods + n_foods  # Y + U variables
    
    print(f"  Variables: {n_vars} ({n_plots} patches × {n_foods} foods + {n_foods} U vars)")
    
    results = {
        'n_plots': n_plots,
        'n_foods': n_foods,
        'n_vars': n_vars,
        'methods': {}
    }
    
    # Build CQM (for partition methods)
    cqm_start = time.time()
    cqm, _, _ = build_cqm(data)
    cqm_build_time = time.time() - cqm_start
    print(f"  CQM build time: {cqm_build_time:.3f}s")
    print(f"  CQM constraints: {len(cqm.constraints)}")
    results['cqm_build_time'] = cqm_build_time
    results['n_constraints'] = len(cqm.constraints)
    
    # Ground truth (Gurobi direct)
    print(f"\n  [Ground Truth] Solving full problem with Gurobi...")
    gt_result = solve_full_cqm_gurobi(cqm, data, timeout=SOLVE_TIMEOUT)
    results['ground_truth'] = {
        'objective': gt_result['objective'],
        'solve_time': gt_result['solve_time'],
        'status': gt_result['status'],
        'success': gt_result['success']
    }
    print(f"    Objective: {gt_result['objective']:.6f}")
    print(f"    Solve time: {gt_result['solve_time']:.3f}s")
    
    # Partition methods
    partition_methods = [
        ("None", partition_none),
        ("PlotBased", partition_plot_based),
        ("Spectral(4)", lambda c, d: partition_spectral(c, d, 4)),
        ("MasterSubproblem", partition_master_subproblem),
    ]
    
    for method_name, partition_fn in partition_methods:
        print(f"\n  [{method_name}] Partitioning and solving...")
        
        # Partition (embed) time
        partition_start = time.time()
        partitions, actual_name = partition_fn(cqm, data)
        partition_time = time.time() - partition_start
        
        if partitions is None:
            print(f"    SKIPPED: {actual_name}")
            continue
        
        n_partitions = len(partitions)
        
        # Solve time
        solve_start = time.time()
        solution, total_solve_time, _ = solve_partitioned_cqm(partitions, data, timeout_per_partition=PARTITION_TIMEOUT)
        solve_time = time.time() - solve_start
        
        # Evaluate
        objective = calculate_objective(solution, data)
        violations = check_violations(solution, data)
        
        gap = ((gt_result['objective'] - objective) / gt_result['objective'] * 100) if gt_result['objective'] > 0 else 0
        
        results['methods'][method_name] = {
            'n_partitions': n_partitions,
            'partition_time': partition_time,
            'solve_time': solve_time,
            'total_time': partition_time + solve_time,
            'objective': objective,
            'gap_percent': gap,
            'n_violations': len(violations)
        }
        
        status = "✅" if len(violations) == 0 else "❌"
        print(f"    Partitions: {n_partitions}")
        print(f"    Partition time: {partition_time:.3f}s")
        print(f"    Solve time: {solve_time:.3f}s")
        print(f"    Objective: {objective:.6f} (gap: {gap:+.1f}%)")
        print(f"    Violations: {status} {len(violations)}")
    
    return results


def main():
    """Run full benchmark across all sizes."""
    print("\n" + "="*80)
    print("CQM PARTITION SCALING BENCHMARK")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Plot sizes: {PLOT_SIZES}")
    
    # Load food data once to show
    _, foods, _, _ = load_food_data('full_family')
    print(f"Foods: {len(foods)}")
    
    all_results = []
    
    for n_plots in PLOT_SIZES:
        result = benchmark_single_size(n_plots)
        all_results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"cqm_partition_scaling_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary table
    print("\n" + "="*120)
    print("SUMMARY: SCALING BENCHMARK RESULTS")
    print("="*120)
    
    # Header
    print(f"{'Plots':>6} {'Vars':>7} | {'Ground Truth':^20} | {'None':^18} | {'PlotBased':^18} | {'Spectral(4)':^18} | {'MasterSub':^18}")
    print(f"{'':>6} {'':>7} | {'Obj':>8} {'Time':>6} | {'Obj':>6} {'Time':>5} {'Gap':>5} | {'Obj':>6} {'Time':>5} {'Gap':>5} | {'Obj':>6} {'Time':>5} {'Gap':>5} | {'Obj':>6} {'Time':>5} {'Gap':>5}")
    print("-"*120)
    
    for r in all_results:
        n = r['n_plots']
        v = r['n_vars']
        gt = r['ground_truth']
        
        row = f"{n:>6} {v:>7} | {gt['objective']:>8.4f} {gt['solve_time']:>5.2f}s |"
        
        for method in ['None', 'PlotBased', 'Spectral(4)', 'MasterSubproblem']:
            if method in r['methods']:
                m = r['methods'][method]
                viol_mark = "" if m['n_violations'] == 0 else "!"
                row += f" {m['objective']:>6.4f} {m['total_time']:>4.2f}s {m['gap_percent']:>+4.0f}%{viol_mark} |"
            else:
                row += f" {'N/A':>6} {'N/A':>5} {'N/A':>5} |"
        
        print(row)
    
    print("="*120)
    print("Note: '!' after gap indicates constraint violations")
    
    # Print timing summary
    print("\n" + "="*80)
    print("TIMING BREAKDOWN (seconds)")
    print("="*80)
    print(f"{'Plots':>6} | {'CQM Build':>10} | {'GT Solve':>10} | {'None':>10} | {'PlotBased':>10} | {'Spectral':>10} | {'MasterSub':>10}")
    print("-"*80)
    
    for r in all_results:
        n = r['n_plots']
        gt = r['ground_truth']
        
        row = f"{n:>6} | {r['cqm_build_time']:>10.3f} | {gt['solve_time']:>10.3f} |"
        
        for method in ['None', 'PlotBased', 'Spectral(4)', 'MasterSubproblem']:
            if method in r['methods']:
                m = r['methods'][method]
                row += f" {m['total_time']:>10.3f} |"
            else:
                row += f" {'N/A':>10} |"
        
        print(row)
    
    print("="*80)
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
