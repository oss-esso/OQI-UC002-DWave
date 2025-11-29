"""
CQM Partition Benchmark: Test All Decomposition Methods on CQM

This script tests whether we can partition the CQM directly (not BQM)
and solve each partition while respecting constraints.

Key insight: The CQM has constraints that link variables across partitions.
If we partition poorly, we may break these constraint links.

Decomposition strategies:
1. None - Solve full CQM
2. PlotBased - Each patch is a partition (natural structure)
3. Louvain - Community detection on CQM variable graph
4. Spectral - Spectral clustering
5. FoodGroupBased - Partition by food groups
6. MasterSubproblem - Two-stage: U variables (master) + Y variables per patch (subproblems)
"""

import os
import sys
import time
import numpy as np
from collections import defaultdict

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import warnings
warnings.filterwarnings('ignore')

from dimod import ConstrainedQuadraticModel, Binary, cqm_to_bqm
import networkx as nx
import gurobipy as gp

from src.scenarios import load_food_data
from Utils import patch_sampler

# Try imports
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

try:
    from sklearn.cluster import SpectralClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ============================================================================
# CONFIGURATION
# ============================================================================
N_FARMS = 10
SOLVE_TIMEOUT = 30  # seconds per partition

print("="*80)
print("CQM PARTITION BENCHMARK")
print("="*80)
print(f"Problem size: {N_FARMS} farms")
print()

# ============================================================================
# LOAD DATA
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
    
    # Use actual food group names from data
    # Map to simplified constraint names
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
        'food_groups': food_groups,  # Original: group -> [foods]
        'food_to_group': food_to_group,  # Inverse: food -> group
        'group_name_mapping': group_name_mapping,
        'food_benefits': food_benefits,
        'weights': weights,
        'land_availability': land_availability,
        'patch_names': patch_names,
        'food_group_constraints': food_group_constraints
    }


# ============================================================================
# BUILD CQM
# ============================================================================

def build_cqm(data):
    """Build the full CQM."""
    foods = data['foods']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    patch_names = data['patch_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
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
    # Use group_name_mapping to convert constraint names to data names
    group_name_mapping = data.get('group_name_mapping', {})
    reverse_mapping = {v: k for k, v in group_name_mapping.items()}
    
    for constraint_group, limits in food_group_constraints.items():
        # Map constraint group name to data group name
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
# CQM TO GRAPH (for partitioning)
# ============================================================================

def cqm_to_variable_graph(cqm):
    """
    Build a graph where nodes are CQM variables and edges connect 
    variables that appear in the same constraint.
    """
    G = nx.Graph()
    G.add_nodes_from(cqm.variables)
    
    # For each constraint, add edges between all variable pairs
    for label, constraint in cqm.constraints.items():
        # Get variables in this constraint
        constraint_vars = list(constraint.lhs.variables)
        
        # Add edges between all pairs
        for i, v1 in enumerate(constraint_vars):
            for v2 in constraint_vars[i+1:]:
                if G.has_edge(v1, v2):
                    G[v1][v2]['weight'] += 1
                else:
                    G.add_edge(v1, v2, weight=1)
    
    return G


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


def partition_louvain(cqm, data):
    """Louvain community detection on variable graph."""
    if not HAS_LOUVAIN:
        return None, "Louvain not available"
    
    G = cqm_to_variable_graph(cqm)
    partition_dict = community_louvain.best_partition(G, resolution=1.0)
    
    partitions = defaultdict(set)
    for var, comm in partition_dict.items():
        partitions[comm].add(var)
    
    return list(partitions.values()), "Louvain"


def partition_spectral(cqm, data, n_clusters=4):
    """Spectral clustering on variable graph."""
    if not HAS_SKLEARN:
        return None, "sklearn not available"
    
    G = cqm_to_variable_graph(cqm)
    nodes = list(G.nodes())
    
    if len(nodes) < n_clusters:
        n_clusters = len(nodes)
    
    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
    
    try:
        sc = SpectralClustering(n_clusters=n_clusters, 
                                affinity='precomputed',
                                random_state=42)
        labels = sc.fit_predict(adj_matrix + np.eye(len(nodes)) * 0.01)
        
        partitions = defaultdict(set)
        for i, node in enumerate(nodes):
            partitions[labels[i]].add(node)
        
        return list(partitions.values()), f"Spectral({n_clusters})"
    except Exception as e:
        return None, f"Spectral failed: {e}"


def partition_food_group_based(cqm, data):
    """Partition by food group - Y variables grouped by food's group."""
    foods = data['foods']
    food_to_group = data.get('food_to_group', {})
    patch_names = data['patch_names']
    
    partitions = defaultdict(set)
    
    # Group Y variables by food group
    for patch in patch_names:
        for food in foods:
            var_name = f"Y_{patch}_{food}"
            if var_name in cqm.variables:
                group = food_to_group.get(food, 'Other')
                partitions[f"Y_{group}"].add(var_name)
    
    # U variables grouped by food group
    for food in foods:
        var_name = f"U_{food}"
        if var_name in cqm.variables:
            group = food_to_group.get(food, 'Other')
            partitions[f"U_{group}"].add(var_name)
    
    return list(partitions.values()), "FoodGroupBased"


def partition_master_subproblem(cqm, data):
    """
    Two-level partition:
    - Master: All U variables (global food selection)
    - Subproblems: Y variables per patch
    """
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
# ANALYZE CONSTRAINTS CUT
# ============================================================================

def analyze_constraint_cuts(cqm, partitions):
    """Analyze which constraints are cut by the partitioning."""
    # Build variable to partition mapping
    var_to_partition = {}
    for i, partition in enumerate(partitions):
        for var in partition:
            var_to_partition[var] = i
    
    within_constraints = 0
    cut_constraints = 0
    cut_constraint_types = defaultdict(int)
    
    for label, constraint in cqm.constraints.items():
        constraint_vars = list(constraint.lhs.variables)
        partitions_involved = set(var_to_partition.get(v) for v in constraint_vars if v in var_to_partition)
        
        if len(partitions_involved) <= 1:
            within_constraints += 1
        else:
            cut_constraints += 1
            # Classify constraint type
            if 'one_per_patch' in label:
                cut_constraint_types['one_per_patch'] += 1
            elif 'link_' in label:
                cut_constraint_types['U-Y_linking'] += 1
            elif 'group_min' in label or 'group_max' in label:
                cut_constraint_types['food_group'] += 1
            else:
                cut_constraint_types['other'] += 1
    
    return {
        'total_constraints': len(cqm.constraints),
        'within_constraints': within_constraints,
        'cut_constraints': cut_constraints,
        'cut_ratio': cut_constraints / len(cqm.constraints) if cqm.constraints else 0,
        'cut_by_type': dict(cut_constraint_types)
    }


# ============================================================================
# SOLVE METHODS
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
    
    # Variables
    Y = {}
    for patch in patch_names:
        for food in foods:
            Y[(patch, food)] = model.addVar(vtype=gp.GRB.BINARY, name=f"Y_{patch}_{food}")
    
    U = {}
    for food in foods:
        U[food] = model.addVar(vtype=gp.GRB.BINARY, name=f"U_{food}")
    
    # Objective
    obj = 0
    for patch in patch_names:
        patch_area = land_availability[patch]
        for food in foods:
            obj += food_benefits[food] * patch_area * Y[(patch, food)]
    obj = obj / total_area
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    
    # Constraints
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
        'status': 'OPTIMAL' if model.Status == gp.GRB.OPTIMAL else 'SUBOPTIMAL'
    }


def solve_partition_independently(cqm, partition, data, fixed_vars=None, timeout=30):
    """
    Solve a single partition of the CQM.
    
    This builds a Gurobi model with only the variables in the partition,
    including all constraints that can be evaluated.
    """
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
    
    # Create variables only for this partition
    gp_vars = {}
    for var_name in partition:
        gp_vars[var_name] = model.addVar(vtype=gp.GRB.BINARY, name=var_name)
    
    # Build objective (only for partition variables)
    obj = 0
    for patch in patch_names:
        patch_area = land_availability[patch]
        for food in foods:
            var_name = f"Y_{patch}_{food}"
            if var_name in partition:
                obj += food_benefits[food] * patch_area * gp_vars[var_name]
    obj = obj / total_area
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    
    # Helper to get variable value (from partition or fixed)
    def get_var(var_name):
        if var_name in gp_vars:
            return gp_vars[var_name]
        elif var_name in fixed_vars:
            return fixed_vars[var_name]
        else:
            return None  # Not available
    
    # Add constraints
    # 1. At-most-one-per-patch
    for patch in patch_names:
        patch_y_vars = []
        fixed_count = 0
        skip_constraint = False
        
        for food in foods:
            var_name = f"Y_{patch}_{food}"
            var_val = get_var(var_name)
            
            if var_val is None:
                skip_constraint = True
                break
            elif isinstance(var_val, (int, float)):
                fixed_count += var_val
            else:
                patch_y_vars.append(var_val)
        
        if not skip_constraint and patch_y_vars:
            model.addConstr(gp.quicksum(patch_y_vars) + fixed_count <= 1)
    
    # 2. U-Y linking: U[food] >= Y[patch, food]
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
            
            # Both variables available
            if isinstance(u_var, (int, float)) and isinstance(y_var, (int, float)):
                pass  # Both fixed - nothing to add
            elif isinstance(u_var, (int, float)):
                # U is fixed, Y is variable: Y <= U_fixed
                model.addConstr(y_var <= u_var)
            elif isinstance(y_var, (int, float)):
                # Y is fixed, U is variable: U >= Y_fixed
                if y_var == 1:
                    model.addConstr(u_var >= 1)
            else:
                # Both are variables
                model.addConstr(u_var >= y_var)
    
    # 3. Food group constraints
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        
        group_vars = []
        fixed_sum = 0
        skip_constraint = False
        
        for food in foods_in_group:
            var_name = f"U_{food}"
            var_val = get_var(var_name)
            
            if var_val is None:
                skip_constraint = True
                break
            elif isinstance(var_val, (int, float)):
                fixed_sum += var_val
            else:
                group_vars.append(var_val)
        
        if not skip_constraint:
            if group_vars:
                group_sum = gp.quicksum(group_vars) + fixed_sum
            else:
                group_sum = fixed_sum
            if 'min' in limits and limits['min'] > 0:
                model.addConstr(group_sum >= limits['min'])
            if 'max' in limits:
                model.addConstr(group_sum <= limits['max'])
    
    model.optimize()
    
    if model.Status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.SUBOPTIMAL]:
        if model.SolCount > 0:
            solution = {var_name: int(gp_vars[var_name].X) for var_name in partition}
            return {
                'success': True,
                'solution': solution,
                'objective': model.ObjVal,
                'status': model.Status
            }
    
    return {
        'success': False,
        'error': f'Status {model.Status}',
        'solution': {var_name: 0 for var_name in partition}
    }


def solve_partitioned_cqm(cqm, partitions, data, method_name):
    """
    Solve CQM by partitions.
    
    Strategy: Solve partitions sequentially, passing fixed values to subsequent partitions.
    """
    start = time.time()
    
    all_solutions = {}
    partition_results = []
    
    for i, partition in enumerate(partitions):
        if len(partition) == 0:
            continue
        
        result = solve_partition_independently(
            cqm, partition, data, 
            fixed_vars=all_solutions,  # Pass solutions from previous partitions
            timeout=SOLVE_TIMEOUT
        )
        
        partition_results.append({
            'partition': i,
            'n_vars': len(partition),
            'success': result['success'],
            'objective': result.get('objective', 0)
        })
        
        if result['success']:
            all_solutions.update(result['solution'])
        else:
            # Fill with zeros
            for var_name in partition:
                all_solutions[var_name] = 0
    
    total_time = time.time() - start
    
    # Calculate actual objective
    actual_obj = calculate_objective(all_solutions, data)
    
    # Check violations
    violations = check_violations(all_solutions, data)
    
    return {
        'method': method_name,
        'n_partitions': len(partitions),
        'objective': actual_obj,
        'solve_time': total_time,
        'violations': violations,
        'solution': all_solutions,
        'partition_results': partition_results
    }


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
    
    # Check at-most-one-per-patch
    for patch in patch_names:
        count = sum(1 for food in foods if solution.get(f"Y_{patch}_{food}", 0) == 1)
        if count > 1:
            violations.append(f"Patch {patch}: {count} foods (max 1)")
    
    # Check food group constraints
    for constraint_group, limits in food_group_constraints.items():
        data_group = reverse_mapping.get(constraint_group, constraint_group)
        foods_in_group = food_groups.get(data_group, [])
        selected = set()
        for food in foods_in_group:
            # Check if U variable is set OR any Y variable for this food
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

def main():
    print("\n[1/4] Loading data and building CQM...")
    data = load_problem_data(N_FARMS)
    cqm, Y, U = build_cqm(data)
    
    print(f"  Foods: {len(data['foods'])}")
    print(f"  Patches: {len(data['patch_names'])}")
    print(f"  CQM Variables: {len(cqm.variables)}")
    print(f"  CQM Constraints: {len(cqm.constraints)}")
    
    # Define partition methods
    partition_methods = [
        ("None", lambda c, d: partition_none(c, d)),
        ("PlotBased", lambda c, d: partition_plot_based(c, d)),
        ("Louvain", lambda c, d: partition_louvain(c, d)),
        ("Spectral(4)", lambda c, d: partition_spectral(c, d, 4)),
        ("Spectral(8)", lambda c, d: partition_spectral(c, d, 8)),
        ("FoodGroupBased", lambda c, d: partition_food_group_based(c, d)),
        ("MasterSubproblem", lambda c, d: partition_master_subproblem(c, d)),
    ]
    
    # ========================================================================
    # PHASE 1: ANALYZE PARTITIONS
    # ========================================================================
    print("\n[2/4] Analyzing partition methods...")
    
    partition_data = {}
    cut_analysis = {}
    
    for name, partition_fn in partition_methods:
        partitions, actual_name = partition_fn(cqm, data)
        if partitions is None:
            print(f"  {name}: FAILED - {actual_name}")
            continue
        
        partition_data[name] = partitions
        analysis = analyze_constraint_cuts(cqm, partitions)
        cut_analysis[name] = analysis
        
        print(f"\n  {name}:")
        print(f"    Partitions: {len(partitions)}")
        print(f"    Sizes: {sorted([len(p) for p in partitions], reverse=True)[:5]}...")
        print(f"    Cut constraints: {analysis['cut_constraints']}/{analysis['total_constraints']} ({analysis['cut_ratio']*100:.1f}%)")
        if analysis['cut_by_type']:
            print(f"    Cut types: {analysis['cut_by_type']}")
    
    # ========================================================================
    # PHASE 2: SOLVE GROUND TRUTH
    # ========================================================================
    print("\n[3/4] Solving ground truth (full CQM)...")
    
    ground_truth = solve_full_cqm_gurobi(cqm, data, timeout=60)
    print(f"  Objective: {ground_truth['objective']:.6f}")
    print(f"  Solve time: {ground_truth['solve_time']:.3f}s")
    
    # ========================================================================
    # PHASE 3: SOLVE ALL PARTITIONS
    # ========================================================================
    print("\n[4/4] Solving with each partition method...")
    
    results = {'GroundTruth': ground_truth}
    
    for name, partitions in partition_data.items():
        print(f"\n  Solving {name} ({len(partitions)} partitions)...")
        result = solve_partitioned_cqm(cqm, partitions, data, name)
        results[name] = result
        
        print(f"    Objective: {result['objective']:.6f}")
        print(f"    Time: {result['solve_time']:.3f}s")
        print(f"    Violations: {len(result['violations'])}")
        if result['violations']:
            for v in result['violations'][:3]:
                print(f"      - {v}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*100)
    print("SUMMARY: CQM PARTITION BENCHMARK")
    print("="*100)
    print(f"{'Method':<20} {'Parts':>6} {'Cut%':>8} {'Objective':>12} {'Gap%':>8} {'Time':>8} {'Violations':>12}")
    print("-"*100)
    
    gt_obj = ground_truth['objective']
    
    for name in ['GroundTruth'] + list(partition_data.keys()):
        if name not in results:
            continue
        
        r = results[name]
        cuts = cut_analysis.get(name, {})
        
        n_parts = len(partition_data.get(name, [[]])) if name != 'GroundTruth' else 1
        cut_pct = cuts.get('cut_ratio', 0) * 100
        obj = r.get('objective', 0)
        gap = (gt_obj - obj) / gt_obj * 100 if gt_obj > 0 else 0
        time_s = r.get('solve_time', 0)
        n_viol = len(r.get('violations', []))
        
        viol_str = "✅ 0" if n_viol == 0 else f"❌ {n_viol}"
        gap_str = f"{gap:+.1f}%" if name != 'GroundTruth' else "baseline"
        
        print(f"{name:<20} {n_parts:>6} {cut_pct:>7.1f}% {obj:>12.6f} {gap_str:>8} {time_s:>7.3f}s {viol_str:>12}")
    
    print("="*100)
    
    # Key insights
    print("\n📊 KEY INSIGHTS:")
    print("-"*80)
    
    # Find methods with no violations
    feasible_methods = [name for name, r in results.items() 
                       if len(r.get('violations', [])) == 0 and name != 'GroundTruth']
    
    if feasible_methods:
        print(f"  ✅ Feasible partition methods: {', '.join(feasible_methods)}")
    else:
        print("  ⚠️  No partition method produced a fully feasible solution!")
    
    # Find best objective among feasible
    best_obj = 0
    best_method = None
    for name in feasible_methods:
        obj = results[name]['objective']
        if obj > best_obj:
            best_obj = obj
            best_method = name
    
    if best_method:
        gap = (gt_obj - best_obj) / gt_obj * 100
        print(f"  ✅ Best feasible method: {best_method} (obj={best_obj:.6f}, gap={gap:.1f}%)")
    
    # Methods that cut constraints
    cut_methods = [name for name, analysis in cut_analysis.items() 
                  if analysis['cut_ratio'] > 0]
    if cut_methods:
        print(f"  ⚠️  Methods that cut constraints: {', '.join(cut_methods)}")
        print("     → Cutting constraints may lead to infeasible solutions!")
    
    print()
    print("="*80)
    print("CONCLUSION: CQM partition success depends on respecting constraint structure")
    print("="*80)


if __name__ == "__main__":
    main()
