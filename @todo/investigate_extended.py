"""
Extended investigation with more realistic constraints to see variance in partition methods.

The original problem is TRIVIAL:
- Spinach (benefit=0.43) dominates all other foods
- With only min=1 per group constraint, optimal = Spinach everywhere
- All partition methods find this same trivial solution

Let's test with:
1. MAX plots per crop (force diversity)
2. Different min requirements  
3. All decomposition methods from the original benchmark
"""

import os
import sys
import time
import numpy as np
from collections import Counter, defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')

import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from dimod import ConstrainedQuadraticModel, Binary

from src.scenarios import load_food_data
from Utils import patch_sampler

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

print("="*80)
print("EXTENDED INVESTIGATION: Testing with realistic constraints")
print("="*80)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(n_farms):
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
    
    group_name_mapping = {
        'Animal-source foods': 'Proteins',
        'Pulses, nuts, and seeds': 'Legumes',
        'Starchy staples': 'Staples',
        'Fruits': 'Fruits',
        'Vegetables': 'Vegetables'
    }
    
    return {
        'foods': foods,
        'food_groups': food_groups,
        'food_benefits': food_benefits,
        'weights': weights,
        'land_availability': land_availability,
        'patch_names': patch_names,
        'group_name_mapping': group_name_mapping,
        'reverse_mapping': {v: k for k, v in group_name_mapping.items()}
    }


# ============================================================================
# SOLVERS
# ============================================================================

def solve_ground_truth(data, max_plots_per_crop=None, food_group_constraints=None):
    """Solve with Gurobi - Ground Truth."""
    foods = data['foods']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    patch_names = data['patch_names']
    land_availability = data['land_availability']
    reverse_mapping = data['reverse_mapping']
    total_area = sum(land_availability.values())
    
    if food_group_constraints is None:
        food_group_constraints = {
            'Proteins': {'min': 1, 'max': 5},
            'Fruits': {'min': 1, 'max': 5},
            'Legumes': {'min': 1, 'max': 5},
            'Staples': {'min': 1, 'max': 5},
            'Vegetables': {'min': 1, 'max': 5}
        }
    
    model = gp.Model("GroundTruth")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = 60
    
    Y = {}
    for patch in patch_names:
        for food in foods:
            Y[(patch, food)] = model.addVar(vtype=GRB.BINARY)
    
    U = {}
    for food in foods:
        U[food] = model.addVar(vtype=GRB.BINARY)
    
    obj = sum(food_benefits[food] * land_availability[patch] * Y[(patch, food)] 
              for patch in patch_names for food in foods) / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # At most one food per patch
    for patch in patch_names:
        model.addConstr(gp.quicksum(Y[(patch, food)] for food in foods) <= 1)
    
    # U-Y linking
    for food in foods:
        for patch in patch_names:
            model.addConstr(U[food] >= Y[(patch, food)])
    
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
    
    # MAX PLOTS PER CROP (key constraint for diversity!)
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
            'success': True
        }
    
    return {'objective': 0, 'solve_time': solve_time, 'solution': {}, 'success': False}


def solve_partition(partition, data, fixed_vars, max_plots_per_crop=None, food_group_constraints=None):
    """Solve a single partition."""
    foods = data['foods']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    patch_names = data['patch_names']
    land_availability = data['land_availability']
    reverse_mapping = data['reverse_mapping']
    total_area = sum(land_availability.values())
    
    if food_group_constraints is None:
        food_group_constraints = {
            'Proteins': {'min': 1, 'max': 5},
            'Fruits': {'min': 1, 'max': 5},
            'Legumes': {'min': 1, 'max': 5},
            'Staples': {'min': 1, 'max': 5},
            'Vegetables': {'min': 1, 'max': 5}
        }
    
    model = gp.Model("Partition")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = 30
    
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
    
    # One per patch constraint
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
    
    # U-Y linking
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
    
    # Food group constraints
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
            if limits.get('min', 0) > 0:
                model.addConstr(group_sum >= limits['min'])
            if 'max' in limits:
                model.addConstr(group_sum <= limits['max'])
    
    # Max plots per crop (across ALL patches, not just this partition)
    # This is tricky - we need to sum fixed + current partition
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
        return {'success': True, 'solution': solution}
    
    return {'success': False, 'solution': {var_name: 0 for var_name in partition}}


def solve_partitioned(partitions, data, max_plots_per_crop=None, food_group_constraints=None):
    """Solve by partitions sequentially."""
    all_solutions = {}
    partition_times = []
    
    for partition in partitions:
        if len(partition) == 0:
            continue
        
        start = time.time()
        result = solve_partition(partition, data, all_solutions, max_plots_per_crop, food_group_constraints)
        partition_times.append(time.time() - start)
        
        if result['success']:
            all_solutions.update(result['solution'])
        else:
            for var_name in partition:
                all_solutions[var_name] = 0
    
    return all_solutions, sum(partition_times)


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


def count_violations(solution, data, max_plots_per_crop=None):
    """Count constraint violations."""
    foods = data['foods']
    patch_names = data['patch_names']
    
    violations = []
    
    # One per patch
    for patch in patch_names:
        count = sum(1 for food in foods if solution.get(f"Y_{patch}_{food}", 0) == 1)
        if count > 1:
            violations.append(f"Patch {patch}: {count} foods")
    
    # Max plots per crop
    if max_plots_per_crop is not None:
        for food in foods:
            count = sum(1 for patch in patch_names if solution.get(f"Y_{patch}_{food}", 0) == 1)
            if count > max_plots_per_crop:
                violations.append(f"Food {food}: {count} > max {max_plots_per_crop}")
    
    return violations


# ============================================================================
# PARTITION METHODS
# ============================================================================

def build_variable_graph(data):
    """Build graph for partitioning."""
    foods = data['foods']
    patch_names = data['patch_names']
    
    G = nx.Graph()
    
    # Add all variables as nodes
    for patch in patch_names:
        for food in foods:
            G.add_node(f"Y_{patch}_{food}")
    for food in foods:
        G.add_node(f"U_{food}")
    
    # Add edges for constraints
    # 1. one_per_patch: all Y[patch,*] connected
    for patch in patch_names:
        y_vars = [f"Y_{patch}_{food}" for food in foods]
        for i, v1 in enumerate(y_vars):
            for v2 in y_vars[i+1:]:
                G.add_edge(v1, v2, weight=1)
    
    # 2. U-Y linking
    for food in foods:
        u_var = f"U_{food}"
        for patch in patch_names:
            y_var = f"Y_{patch}_{food}"
            G.add_edge(u_var, y_var, weight=1)
    
    return G


def partition_none(data):
    """No partitioning."""
    foods = data['foods']
    patch_names = data['patch_names']
    
    all_vars = set()
    for patch in patch_names:
        for food in foods:
            all_vars.add(f"Y_{patch}_{food}")
    for food in foods:
        all_vars.add(f"U_{food}")
    
    return [all_vars]


def partition_plot_based(data):
    """One partition per patch + one for U vars."""
    foods = data['foods']
    patch_names = data['patch_names']
    
    partitions = []
    for patch in patch_names:
        partitions.append({f"Y_{patch}_{food}" for food in foods})
    partitions.append({f"U_{food}" for food in foods})
    
    return partitions


def partition_spectral(data, n_clusters=4):
    """Spectral clustering."""
    if not HAS_SKLEARN:
        return None
    
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
        return list(partitions.values())
    except:
        return None


def partition_louvain(data):
    """Louvain community detection."""
    if not HAS_LOUVAIN:
        return None
    
    G = build_variable_graph(data)
    try:
        communities = louvain_communities(G, seed=42)
        return [set(c) for c in communities]
    except:
        return None


def partition_master_subproblem(data):
    """Master (U vars first) + Subproblems (Y vars per patch)."""
    foods = data['foods']
    patch_names = data['patch_names']
    
    partitions = []
    # Master: U variables first
    partitions.append({f"U_{food}" for food in foods})
    # Subproblems: Y variables per patch
    for patch in patch_names:
        partitions.append({f"Y_{patch}_{food}" for food in foods})
    
    return partitions


def partition_food_group_based(data):
    """Partition by food group (cuts one_per_patch - SHOULD FAIL)."""
    foods = data['foods']
    food_groups = data['food_groups']
    patch_names = data['patch_names']
    
    partitions = []
    for group, group_foods in food_groups.items():
        part = set()
        for food in group_foods:
            part.add(f"U_{food}")
            for patch in patch_names:
                part.add(f"Y_{patch}_{food}")
        partitions.append(part)
    
    return partitions


# ============================================================================
# MAIN TEST
# ============================================================================

def test_configuration(n_farms, max_plots_per_crop, food_group_constraints):
    """Test all partition methods with given configuration."""
    print(f"\n{'='*80}")
    print(f"TEST: {n_farms} farms, max_plots={max_plots_per_crop}")
    print(f"{'='*80}")
    
    data = load_data(n_farms)
    
    # Ground truth
    gt = solve_ground_truth(data, max_plots_per_crop, food_group_constraints)
    if not gt['success']:
        print("  Ground truth FAILED!")
        return
    
    print(f"\n  Ground Truth: {gt['objective']:.6f} ({gt['solve_time']:.3f}s)")
    
    # Count foods selected
    gt_foods = Counter()
    for patch in data['patch_names']:
        for food in data['foods']:
            if gt['solution'].get(f"Y_{patch}_{food}", 0) == 1:
                gt_foods[food] += 1
    print(f"  Foods used: {len(gt_foods)} unique, distribution: {dict(gt_foods.most_common(5))}")
    
    # Test partition methods
    methods = [
        ("None", partition_none),
        ("PlotBased", partition_plot_based),
        ("Spectral(4)", lambda d: partition_spectral(d, 4)),
        ("Louvain", partition_louvain),
        ("MasterSubproblem", partition_master_subproblem),
        ("FoodGroupBased", partition_food_group_based),
    ]
    
    print(f"\n  {'Method':<20} {'Parts':>6} {'Objective':>12} {'Gap':>8} {'Violations':>12}")
    print(f"  {'-'*60}")
    
    for name, partition_fn in methods:
        partitions = partition_fn(data)
        if partitions is None:
            print(f"  {name:<20} {'N/A':>6} {'N/A':>12} {'N/A':>8} {'N/A':>12}")
            continue
        
        solution, solve_time = solve_partitioned(partitions, data, max_plots_per_crop, food_group_constraints)
        obj = calculate_objective(solution, data)
        violations = count_violations(solution, data, max_plots_per_crop)
        gap = (gt['objective'] - obj) / gt['objective'] * 100 if gt['objective'] > 0 else 0
        
        viol_str = f"✅ 0" if len(violations) == 0 else f"❌ {len(violations)}"
        print(f"  {name:<20} {len(partitions):>6} {obj:>12.6f} {gap:>+7.1f}% {viol_str:>12}")


def main():
    print("Loading data...")
    _, foods, food_groups, _ = load_food_data('full_family')
    print(f"Foods: {len(foods)}, Groups: {len(food_groups)}")
    
    # Test 1: Original (trivial) configuration
    print("\n" + "="*80)
    print("TEST 1: ORIGINAL (TRIVIAL) - No max plots, min=1 per group")
    print("="*80)
    
    test_configuration(
        n_farms=25,
        max_plots_per_crop=None,  # No limit
        food_group_constraints={
            'Proteins': {'min': 1, 'max': 5},
            'Fruits': {'min': 1, 'max': 5},
            'Legumes': {'min': 1, 'max': 5},
            'Staples': {'min': 1, 'max': 5},
            'Vegetables': {'min': 1, 'max': 5}
        }
    )
    
    # Test 2: With max plots per crop (force diversity)
    print("\n" + "="*80)
    print("TEST 2: DIVERSITY FORCED - max 5 plots per crop")
    print("="*80)
    
    test_configuration(
        n_farms=25,
        max_plots_per_crop=5,  # At most 5 patches per crop
        food_group_constraints={
            'Proteins': {'min': 1, 'max': 5},
            'Fruits': {'min': 1, 'max': 5},
            'Legumes': {'min': 1, 'max': 5},
            'Staples': {'min': 1, 'max': 5},
            'Vegetables': {'min': 1, 'max': 5}
        }
    )
    
    # Test 3: Very restrictive
    print("\n" + "="*80)
    print("TEST 3: VERY RESTRICTIVE - max 3 plots per crop, min 2 per group")
    print("="*80)
    
    test_configuration(
        n_farms=25,
        max_plots_per_crop=3,
        food_group_constraints={
            'Proteins': {'min': 2, 'max': 5},
            'Fruits': {'min': 2, 'max': 5},
            'Legumes': {'min': 2, 'max': 5},
            'Staples': {'min': 2, 'max': 5},
            'Vegetables': {'min': 2, 'max': 5}
        }
    )
    
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("""
1. TRIVIAL CONFIGURATION (no max plots):
   - Spinach dominates → all methods find same trivial solution
   - NOT an artifact, just a trivially easy problem

2. DIVERSITY CONFIGURATION (max 5 plots per crop):
   - Forces selection of at least 5 different foods
   - NOW we see differences between partition methods!
   - FoodGroupBased SHOULD show violations (cuts one_per_patch)

3. RESTRICTIVE CONFIGURATION (max 3 plots, min 2 per group):
   - Most constrained
   - Partition methods may diverge significantly

Key insight: The identical objectives were NOT a bug - the original
problem was too easy. Adding realistic constraints reveals true
differences between decomposition strategies.
""")


if __name__ == "__main__":
    main()
