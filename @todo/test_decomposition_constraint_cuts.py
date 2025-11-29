"""
Test Decomposition Constraint Cuts Analysis

This script analyzes whether decomposition methods cut constraint penalty edges
and verifies if decomposed solutions are actually feasible.

Key question: In BQM, constraints are encoded as penalties. 
When we partition the BQM graph, do we cut these penalty edges?
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import warnings
warnings.filterwarnings('ignore')

# Import D-Wave and optimization tools
import dimod
from dimod import ConstrainedQuadraticModel, Binary, cqm_to_bqm
import networkx as nx

# Import from project
from src.scenarios import load_food_data
from Utils import patch_sampler

# Try to import community detection
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

try:
    from dwave.system import DWaveSampler
    from dwave_networkx import pegasus_graph
    import minorminer
    HAS_DWAVE = True
except ImportError:
    HAS_DWAVE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
N_FARMS = 10  # Smaller for faster testing
N_FOODS = 27
EMBEDDING_TIMEOUT = 60  # seconds per partition

print("="*80)
print("DECOMPOSITION CONSTRAINT CUT ANALYSIS")
print("="*80)
print(f"Problem size: {N_FARMS} farms × {N_FOODS} foods = {N_FARMS * N_FOODS} decision variables")
print()

# ============================================================================
# BUILD CQM AND BQM
# ============================================================================

def build_test_cqm(n_farms):
    """Build a test CQM using real data."""
    # Load food data (returns farms, foods, food_groups, config)
    _, foods, food_groups, config_loaded = load_food_data('full_family')
    weights = config_loaded.get('parameters', {}).get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15
    })
    
    # Sample patches using generate_grid for equal-area patches
    land_availability = patch_sampler.generate_grid(n_farms, area=100.0, seed=42)
    patch_names = list(land_availability.keys())
    
    # Build config
    config = {
        'parameters': {
            'land_availability': land_availability,
            'weights': weights,
            'minimum_planting_area': {},
            'maximum_planting_area': {},
            'food_group_constraints': {
                'Proteins': {'min': 1, 'max': 5},
                'Fruits': {'min': 1, 'max': 5},
                'Legumes': {'min': 1, 'max': 5},
                'Staples': {'min': 1, 'max': 5},
                'Vegetables': {'min': 1, 'max': 5}
            }
        }
    }
    
    # Build CQM
    cqm = ConstrainedQuadraticModel()
    
    # Binary variables Y[patch, food]
    Y = {}
    for patch in patch_names:
        for food in foods:
            Y[(patch, food)] = Binary(f"Y_{patch}_{food}")
    
    # U variables for unique food selection
    U = {}
    for food in foods:
        U[food] = Binary(f"U_{food}")
    
    # Objective
    total_area = sum(land_availability.values())
    objective = 0
    for patch in patch_names:
        patch_area = land_availability[patch]
        for food in foods:
            benefit = (
                weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) +
                weights.get('affordability', 0) * foods[food].get('affordability', 0) +
                weights.get('sustainability', 0) * foods[food].get('sustainability', 0)
            )
            objective += benefit * patch_area * Y[(patch, food)]
    
    objective = objective / total_area
    cqm.set_objective(-objective)
    
    # Constraint 1: At most one food per patch
    for patch in patch_names:
        cqm.add_constraint(
            sum(Y[(patch, food)] for food in foods) <= 1,
            label=f"one_food_per_patch_{patch}"
        )
    
    # Constraint 2: U-Y linking (U[food] >= Y[patch,food] for all patches)
    # Rewrite as: U[food] - Y[patch,food] >= 0
    for food in foods:
        for patch in patch_names:
            cqm.add_constraint(
                U[food] - Y[(patch, food)] >= 0,
                label=f"link_U_{food}_{patch}"
            )
    
    # Constraint 3: Food group constraints
    food_group_constraints = config['parameters']['food_group_constraints']
    for group, limits in food_group_constraints.items():
        foods_in_group = [f for f in foods if foods[f].get('food_group') == group]
        if foods_in_group:
            group_sum = sum(U[f] for f in foods_in_group)
            if 'min' in limits and limits['min'] > 0:
                cqm.add_constraint(group_sum >= limits['min'], label=f"group_min_{group}")
            if 'max' in limits:
                cqm.add_constraint(group_sum <= limits['max'], label=f"group_max_{group}")
    
    metadata = {
        'n_farms': n_farms,
        'n_foods': len(foods),
        'n_variables': len(cqm.variables),
        'n_constraints': len(cqm.constraints),
        'foods': foods,
        'food_groups': food_groups,
        'weights': weights,
        'land_availability': land_availability,
        'patch_names': patch_names
    }
    
    return cqm, metadata


def convert_to_bqm(cqm):
    """Convert CQM to BQM."""
    print("  Converting CQM to BQM...")
    start = time.time()
    bqm, invert = cqm_to_bqm(cqm)
    elapsed = time.time() - start
    print(f"    BQM: {len(bqm.variables)} vars, {len(bqm.quadratic)} quadratic terms")
    print(f"    Conversion time: {elapsed:.2f}s")
    return bqm, invert


# ============================================================================
# DECOMPOSITION METHODS
# ============================================================================

def bqm_to_graph(bqm):
    """Convert BQM to NetworkX graph for analysis."""
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    for (u, v), bias in bqm.quadratic.items():
        G.add_edge(u, v, weight=abs(bias))
    return G


def decompose_none(bqm):
    """No decomposition - single partition."""
    return [set(bqm.variables)], "None"


def decompose_louvain(bqm, resolution=1.0):
    """Louvain community detection."""
    if not HAS_LOUVAIN:
        return None, "Louvain not available"
    
    G = bqm_to_graph(bqm)
    partition_dict = community_louvain.best_partition(G, resolution=resolution)
    
    partitions = {}
    for var, comm in partition_dict.items():
        if comm not in partitions:
            partitions[comm] = set()
        partitions[comm].add(var)
    
    return list(partitions.values()), "Louvain"


def decompose_plot_based(bqm, n_farms):
    """Group by plot/farm - all foods for same plot together."""
    partitions = {}
    
    for var in bqm.variables:
        var_str = str(var)
        # Extract plot name from Y_PatchX_Food or similar
        if var_str.startswith('Y_'):
            parts = var_str.split('_')
            if len(parts) >= 2:
                plot = parts[1]  # e.g., "Patch1"
                if plot not in partitions:
                    partitions[plot] = set()
                partitions[plot].add(var)
            else:
                if 'other' not in partitions:
                    partitions['other'] = set()
                partitions['other'].add(var)
        elif var_str.startswith('U_'):
            # U variables go to a separate partition
            if 'U_vars' not in partitions:
                partitions['U_vars'] = set()
            partitions['U_vars'].add(var)
        else:
            # Slack variables - group by constraint
            if 'slack' not in partitions:
                partitions['slack'] = set()
            partitions['slack'].add(var)
    
    return list(partitions.values()), "PlotBased"


def decompose_spectral(bqm, n_partitions=4):
    """Spectral clustering based decomposition."""
    G = bqm_to_graph(bqm)
    
    try:
        from sklearn.cluster import SpectralClustering
        
        # Build adjacency matrix
        nodes = list(G.nodes())
        n = len(nodes)
        adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
        
        # Spectral clustering
        sc = SpectralClustering(n_clusters=min(n_partitions, n), 
                                affinity='precomputed',
                                random_state=42)
        labels = sc.fit_predict(adj_matrix + np.eye(n) * 0.01)
        
        partitions = {}
        for i, node in enumerate(nodes):
            label = labels[i]
            if label not in partitions:
                partitions[label] = set()
            partitions[label].add(node)
        
        return list(partitions.values()), "Spectral"
    except Exception as e:
        return None, f"Spectral failed: {e}"


def decompose_metis(bqm, n_partitions=4):
    """METIS graph partitioning."""
    try:
        import pymetis
        
        G = bqm_to_graph(bqm)
        nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        # Build adjacency list for METIS
        adjacency = []
        for node in nodes:
            neighbors = [node_to_idx[n] for n in G.neighbors(node)]
            adjacency.append(np.array(neighbors, dtype=np.int32))
        
        # Run METIS
        n_cuts, membership = pymetis.part_graph(n_partitions, adjacency=adjacency)
        
        partitions = {}
        for i, node in enumerate(nodes):
            part = membership[i]
            if part not in partitions:
                partitions[part] = set()
            partitions[part].add(node)
        
        return list(partitions.values()), f"METIS({n_partitions})"
    except Exception as e:
        return None, f"METIS failed: {e}"


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_cut_edges(bqm, partitions):
    """
    Analyze which edges (quadratic terms) are cut by partitioning.
    
    Returns statistics about:
    - Total quadratic terms
    - Terms within partitions (preserved)
    - Terms between partitions (cut)
    - Total weight of cut terms
    """
    # Build partition membership
    var_to_partition = {}
    for i, partition in enumerate(partitions):
        for var in partition:
            var_to_partition[var] = i
    
    within_count = 0
    between_count = 0
    within_weight = 0.0
    between_weight = 0.0
    
    cut_edges = []
    
    for (u, v), bias in bqm.quadratic.items():
        part_u = var_to_partition.get(u)
        part_v = var_to_partition.get(v)
        
        if part_u is None or part_v is None:
            continue
        
        weight = abs(bias)
        
        if part_u == part_v:
            within_count += 1
            within_weight += weight
        else:
            between_count += 1
            between_weight += weight
            cut_edges.append((u, v, bias))
    
    total = within_count + between_count
    
    return {
        'total_quadratic': total,
        'within_partitions': within_count,
        'between_partitions': between_count,
        'cut_ratio': between_count / total if total > 0 else 0,
        'within_weight': within_weight,
        'between_weight': between_weight,
        'cut_weight_ratio': between_weight / (within_weight + between_weight) if (within_weight + between_weight) > 0 else 0,
        'cut_edges': cut_edges[:20]  # Sample of cut edges
    }


def classify_cut_edges(cut_edges, metadata):
    """Classify what types of constraint edges were cut."""
    categories = {
        'Y_Y_same_patch': 0,      # Same patch, different foods (at-most-one constraint)
        'Y_Y_diff_patch': 0,      # Different patches (coupling)
        'Y_U_linking': 0,         # Y-U linking constraints
        'U_U_group': 0,           # U-U for food group constraints
        'slack_related': 0,       # Involving slack variables
        'other': 0
    }
    
    for u, v, bias in cut_edges:
        u_str, v_str = str(u), str(v)
        
        # Check if slack variable
        if 'slack' in u_str.lower() or 'slack' in v_str.lower():
            categories['slack_related'] += 1
            continue
        
        # Both Y variables
        if u_str.startswith('Y_') and v_str.startswith('Y_'):
            # Extract patch from Y_PatchX_Food
            u_parts = u_str.split('_')
            v_parts = v_str.split('_')
            if len(u_parts) >= 2 and len(v_parts) >= 2:
                u_patch = u_parts[1]
                v_patch = v_parts[1]
                if u_patch == v_patch:
                    categories['Y_Y_same_patch'] += 1
                else:
                    categories['Y_Y_diff_patch'] += 1
            else:
                categories['other'] += 1
            continue
        
        # Y-U linking
        if (u_str.startswith('Y_') and v_str.startswith('U_')) or \
           (u_str.startswith('U_') and v_str.startswith('Y_')):
            categories['Y_U_linking'] += 1
            continue
        
        # U-U (food group)
        if u_str.startswith('U_') and v_str.startswith('U_'):
            categories['U_U_group'] += 1
            continue
        
        categories['other'] += 1
    
    return categories


# ============================================================================
# EMBEDDING TEST
# ============================================================================

def test_embedding(bqm, partitions, decomp_name, timeout=60):
    """Test embedding for each partition."""
    if not HAS_DWAVE:
        # Use simulated Pegasus
        target = pegasus_graph(16)
    else:
        try:
            sampler = DWaveSampler()
            target = sampler.to_networkx_graph()
        except:
            target = pegasus_graph(16)
    
    results = []
    total_time = 0
    all_success = True
    
    for i, partition in enumerate(partitions):
        if len(partition) == 0:
            continue
        
        # Extract sub-BQM
        sub_bqm = bqm.copy()
        vars_to_remove = set(bqm.variables) - partition
        for var in vars_to_remove:
            sub_bqm.remove_variable(var)
        
        # Count edges
        n_edges = len(sub_bqm.quadratic)
        density = 2 * n_edges / (len(partition) * (len(partition) - 1)) if len(partition) > 1 else 0
        
        # Build source graph
        source = nx.Graph()
        source.add_nodes_from(sub_bqm.variables)
        for (u, v) in sub_bqm.quadratic:
            source.add_edge(u, v)
        
        # Try embedding
        start = time.time()
        try:
            embedding = minorminer.find_embedding(source, target, timeout=timeout)
            elapsed = time.time() - start
            
            if embedding:
                chain_lengths = [len(chain) for chain in embedding.values()]
                result = {
                    'partition': i,
                    'n_vars': len(partition),
                    'n_edges': n_edges,
                    'density': density,
                    'success': True,
                    'time': elapsed,
                    'max_chain': max(chain_lengths),
                    'mean_chain': np.mean(chain_lengths),
                    'total_qubits': sum(chain_lengths)
                }
            else:
                result = {
                    'partition': i,
                    'n_vars': len(partition),
                    'n_edges': n_edges,
                    'density': density,
                    'success': False,
                    'time': elapsed,
                    'error': 'No embedding found'
                }
                all_success = False
        except Exception as e:
            elapsed = time.time() - start
            result = {
                'partition': i,
                'n_vars': len(partition),
                'n_edges': n_edges,
                'density': density,
                'success': False,
                'time': elapsed,
                'error': str(e)
            }
            all_success = False
        
        total_time += elapsed
        results.append(result)
    
    return {
        'decomposition': decomp_name,
        'n_partitions': len(partitions),
        'partition_results': results,
        'all_success': all_success,
        'total_embedding_time': total_time
    }


# ============================================================================
# SOLVING TEST  
# ============================================================================

def solve_partition_gurobi(bqm, partition, timeout=30):
    """Solve a partition with Gurobi."""
    import gurobipy as gp
    
    # Extract sub-BQM
    sub_bqm = bqm.copy()
    vars_to_remove = set(bqm.variables) - partition
    for var in vars_to_remove:
        sub_bqm.remove_variable(var)
    
    # Solve with Gurobi
    model = gp.Model("partition")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    # Add variables
    x = {}
    for var in sub_bqm.variables:
        x[var] = model.addVar(vtype=gp.GRB.BINARY, name=str(var))
    
    # Build objective
    obj = sub_bqm.offset
    for var, bias in sub_bqm.linear.items():
        obj += bias * x[var]
    for (u, v), bias in sub_bqm.quadratic.items():
        obj += bias * x[u] * x[v]
    
    model.setObjective(obj, gp.GRB.MINIMIZE)
    model.optimize()
    
    if model.Status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT]:
        solution = {var: int(x[var].X) for var in sub_bqm.variables}
        return {
            'success': True,
            'solution': solution,
            'objective': model.ObjVal,
            'status': model.Status
        }
    else:
        return {
            'success': False,
            'error': f'Status {model.Status}'
        }


def solve_decomposed(bqm, partitions, decomp_name, timeout=30):
    """Solve all partitions and merge solutions."""
    results = []
    merged_solution = {}
    total_time = 0
    all_success = True
    total_obj = 0
    
    for i, partition in enumerate(partitions):
        if len(partition) == 0:
            continue
        
        start = time.time()
        result = solve_partition_gurobi(bqm, partition, timeout)
        elapsed = time.time() - start
        
        result['partition'] = i
        result['n_vars'] = len(partition)
        result['time'] = elapsed
        results.append(result)
        
        total_time += elapsed
        
        if result['success']:
            merged_solution.update(result['solution'])
            total_obj += result['objective']
        else:
            all_success = False
    
    return {
        'decomposition': decomp_name,
        'n_partitions': len(partitions),
        'partition_results': results,
        'all_success': all_success,
        'merged_solution': merged_solution,
        'total_objective': total_obj,
        'total_solve_time': total_time
    }


def calculate_actual_objective(solution, metadata):
    """Calculate the actual objective from a merged solution."""
    foods = metadata['foods']
    weights = metadata['weights']
    land_availability = metadata['land_availability']
    patch_names = metadata['patch_names']
    total_area = sum(land_availability.values())
    
    objective = 0.0
    
    for patch in patch_names:
        patch_area = land_availability[patch]
        for food in foods:
            var_name = f"Y_{patch}_{food}"
            if solution.get(var_name, 0) == 1:
                benefit = (
                    weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) +
                    weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) -
                    weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) +
                    weights.get('affordability', 0) * foods[food].get('affordability', 0) +
                    weights.get('sustainability', 0) * foods[food].get('sustainability', 0)
                )
                objective += benefit * patch_area
    
    return objective / total_area


def check_constraint_violations(solution, metadata):
    """Check if solution violates any constraints."""
    foods = metadata['foods']
    patch_names = metadata['patch_names']
    food_groups = metadata['food_groups']
    
    violations = []
    
    # 1. At most one food per patch
    for patch in patch_names:
        count = sum(1 for food in foods if solution.get(f"Y_{patch}_{food}", 0) == 1)
        if count > 1:
            violations.append(f"Patch {patch} has {count} foods (max 1)")
    
    # 2. Food group constraints
    food_group_constraints = {
        'Proteins': {'min': 1, 'max': 5},
        'Fruits': {'min': 1, 'max': 5},
        'Legumes': {'min': 1, 'max': 5},
        'Staples': {'min': 1, 'max': 5},
        'Vegetables': {'min': 1, 'max': 5}
    }
    
    # Count unique foods selected per group
    for group, limits in food_group_constraints.items():
        foods_in_group = [f for f in foods if foods[f].get('food_group') == group]
        selected = set()
        for food in foods_in_group:
            for patch in patch_names:
                if solution.get(f"Y_{patch}_{food}", 0) == 1:
                    selected.add(food)
        
        count = len(selected)
        if count < limits['min']:
            violations.append(f"Group {group}: {count} foods < min {limits['min']}")
        if count > limits['max']:
            violations.append(f"Group {group}: {count} foods > max {limits['max']}")
    
    return violations


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("\n[1/5] Building CQM and BQM...")
    cqm, metadata = build_test_cqm(N_FARMS)
    print(f"  CQM: {len(cqm.variables)} vars, {len(cqm.constraints)} constraints")
    
    bqm, invert = convert_to_bqm(cqm)
    
    # Define decomposition methods
    decomposition_methods = [
        ("None", lambda b: decompose_none(b)),
        ("Louvain", lambda b: decompose_louvain(b)),
        ("PlotBased", lambda b: decompose_plot_based(b, N_FARMS)),
        ("Spectral(4)", lambda b: decompose_spectral(b, 4)),
        ("Spectral(8)", lambda b: decompose_spectral(b, 8)),
    ]
    
    # Try to add METIS if available
    try:
        import pymetis
        decomposition_methods.append(("METIS(4)", lambda b: decompose_metis(b, 4)))
        decomposition_methods.append(("METIS(8)", lambda b: decompose_metis(b, 8)))
    except ImportError:
        print("  Note: pymetis not available, skipping METIS decomposition")
    
    # ========================================================================
    # PHASE 1: ANALYZE CUT EDGES
    # ========================================================================
    print("\n[2/5] Analyzing constraint cuts for each decomposition...")
    print()
    
    cut_analysis = {}
    partition_data = {}
    
    for name, decompose_fn in decomposition_methods:
        partitions, actual_name = decompose_fn(bqm)
        if partitions is None:
            print(f"  {name}: FAILED - {actual_name}")
            continue
        
        partition_data[name] = partitions
        
        # Analyze cuts
        analysis = analyze_cut_edges(bqm, partitions)
        categories = classify_cut_edges(analysis['cut_edges'], metadata)
        
        cut_analysis[name] = {
            'n_partitions': len(partitions),
            'partition_sizes': [len(p) for p in partitions],
            **analysis,
            'cut_categories': categories
        }
        
        # Print summary
        print(f"  {name}:")
        print(f"    Partitions: {len(partitions)}, sizes: {sorted([len(p) for p in partitions], reverse=True)[:5]}...")
        print(f"    Total quadratic terms: {analysis['total_quadratic']}")
        print(f"    Cut edges: {analysis['between_partitions']} ({analysis['cut_ratio']*100:.1f}%)")
        print(f"    Cut weight ratio: {analysis['cut_weight_ratio']*100:.1f}%")
        print(f"    Cut types: Y-Y same patch: {categories['Y_Y_same_patch']}, "
              f"Y-U linking: {categories['Y_U_linking']}, slack: {categories['slack_related']}")
        print()
    
    # ========================================================================
    # PHASE 2: EMBEDDING TESTS
    # ========================================================================
    print("\n[3/5] Testing embeddings for all decompositions...")
    print()
    
    embedding_results = {}
    
    for name, partitions in partition_data.items():
        print(f"  Testing {name}...")
        result = test_embedding(bqm, partitions, name, EMBEDDING_TIMEOUT)
        embedding_results[name] = result
        
        n_success = sum(1 for r in result['partition_results'] if r['success'])
        n_total = len(result['partition_results'])
        print(f"    Embedded: {n_success}/{n_total} partitions")
        print(f"    Total time: {result['total_embedding_time']:.1f}s")
        
        if result['all_success']:
            max_chains = [r['max_chain'] for r in result['partition_results']]
            print(f"    Max chain lengths: {max_chains}")
        print()
    
    # ========================================================================
    # PHASE 3: SOLVING TESTS
    # ========================================================================
    print("\n[4/5] Solving all decompositions with Gurobi...")
    print()
    
    solving_results = {}
    
    for name, partitions in partition_data.items():
        print(f"  Solving {name}...")
        result = solve_decomposed(bqm, partitions, name, timeout=30)
        solving_results[name] = result
        
        print(f"    All partitions solved: {result['all_success']}")
        print(f"    Total BQM energy: {result['total_objective']:.4f}")
        print(f"    Total time: {result['total_solve_time']:.1f}s")
        
        if result['all_success'] and result['merged_solution']:
            # Calculate actual objective
            actual_obj = calculate_actual_objective(result['merged_solution'], metadata)
            result['actual_objective'] = actual_obj
            print(f"    Actual objective: {actual_obj:.6f}")
            
            # Check constraints
            violations = check_constraint_violations(result['merged_solution'], metadata)
            result['violations'] = violations
            if violations:
                print(f"    ⚠️  VIOLATIONS: {len(violations)}")
                for v in violations[:3]:
                    print(f"       - {v}")
            else:
                print(f"    ✅ All constraints satisfied!")
        print()
    
    # ========================================================================
    # PHASE 4: SUMMARY
    # ========================================================================
    print("\n[5/5] FINAL SUMMARY")
    print("="*100)
    print(f"{'Decomposition':<15} {'Parts':>6} {'Cut%':>8} {'Embed':>8} {'Solve':>8} {'Objective':>12} {'Violations':>12}")
    print("-"*100)
    
    for name in partition_data.keys():
        cuts = cut_analysis.get(name, {})
        embed = embedding_results.get(name, {})
        solve = solving_results.get(name, {})
        
        n_parts = cuts.get('n_partitions', 0)
        cut_pct = cuts.get('cut_ratio', 0) * 100
        
        embed_ok = "✅" if embed.get('all_success') else "❌"
        solve_ok = "✅" if solve.get('all_success') else "❌"
        
        obj = solve.get('actual_objective', 0)
        n_viol = len(solve.get('violations', []))
        viol_str = "✅ 0" if n_viol == 0 else f"❌ {n_viol}"
        
        print(f"{name:<15} {n_parts:>6} {cut_pct:>7.1f}% {embed_ok:>8} {solve_ok:>8} {obj:>12.6f} {viol_str:>12}")
    
    print("="*100)
    
    # Key insights
    print("\n📊 KEY INSIGHTS:")
    print("-"*80)
    
    # Find best decomposition for embedding
    best_embed = None
    best_embed_time = float('inf')
    for name, result in embedding_results.items():
        if result.get('all_success') and result['total_embedding_time'] < best_embed_time:
            best_embed = name
            best_embed_time = result['total_embedding_time']
    
    if best_embed:
        print(f"  ✅ Best for embedding: {best_embed} ({best_embed_time:.1f}s)")
    
    # Find best decomposition for objective (with no violations)
    best_obj_name = None
    best_obj = -float('inf')
    for name, result in solving_results.items():
        if len(result.get('violations', [])) == 0:
            obj = result.get('actual_objective', 0)
            if obj > best_obj:
                best_obj = obj
                best_obj_name = name
    
    if best_obj_name:
        print(f"  ✅ Best feasible objective: {best_obj_name} ({best_obj:.6f})")
    
    # Warning about decomposition
    decomp_with_violations = [name for name, result in solving_results.items() 
                              if len(result.get('violations', [])) > 0]
    if decomp_with_violations:
        print(f"  ⚠️  Decompositions with violations: {', '.join(decomp_with_violations)}")
        print(f"     → These cut constraint edges, resulting in infeasible solutions!")
    
    print()
    print("="*80)
    print("CONCLUSION: Decomposition works for EMBEDDING but may break CONSTRAINTS")
    print("="*80)


if __name__ == "__main__":
    main()
