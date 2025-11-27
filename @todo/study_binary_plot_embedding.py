#!/usr/bin/env python3
"""
Study Embedding Scaling for BINARY PLOT Formulation

Key insight: By skipping continuous variables (another layer of decomposition),
we can achieve MUCH better embedding results because:
1. No slack variables needed for inequalities
2. Lower BQM density
3. Sparser problem graphs → easier embedding

This script compares:
1. CQM→BQM conversion (baseline, with slack variables)
2. DIRECT BQM construction (no slack variables)
3. Various decomposition strategies

Decomposition Strategies Tested:
- Direct embedding (baseline)
- Energy-impact decomposition (dwave-hybrid)
- Graph partitioning (Louvain communities)
- Subgraph extraction for oversized problems

NO QPU TIME IS BILLED - only embedding is tested!

Author: Generated for OQI-UC002-DWave project
Date: 2025-11-27
"""

import os
import sys
import time
import json
import statistics
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=" * 80)
print("BINARY PLOT EMBEDDING STUDY")
print("=" * 80)
print("Testing decomposition strategies on PURE BINARY formulation")
print("NO QPU time will be billed - embedding study only!")
print("=" * 80)

# Step 1: Imports
print("\n[1/5] Importing libraries...")
import_start = time.time()

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, cqm_to_bqm
from dimod.generators import combinations
import minorminer

# Try to import D-Wave sampler (for target graph)
try:
    from dwave.system import DWaveSampler
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    print("  Warning: DWaveSampler not available. Using Pegasus topology simulation.")

# Try to import dwave-hybrid for decomposers
try:
    from hybrid.decomposers import EnergyImpactDecomposer
    from hybrid.core import State
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    print("  Warning: dwave-hybrid not available. Energy-impact decomposition disabled.")

# Try to import networkx community detection
try:
    from networkx.algorithms.community import louvain_communities
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("  Warning: networkx louvain_communities not available.")

# Try to import neal for simulated annealing (solution quality testing)
try:
    import neal
    NEAL_AVAILABLE = True
except ImportError:
    NEAL_AVAILABLE = False
    print("  Warning: neal not available. Solution quality testing disabled.")

print(f"      ✅ Imports done in {time.time() - import_start:.2f}s")

# Configuration
PROBLEM_SIZES = [5, 10, 25]  # Number of plots to test
N_FOODS = 27  # Fixed number of foods (as in real problem)
EMBEDDING_ATTEMPTS = 1  # Attempts per problem size
EMBEDDING_TIMEOUT = 30  # 5 minutes max per attempt

# Get token
TOKEN = os.getenv('DWAVE_API_TOKEN', '')


# =============================================================================
# BQM CONSTRUCTION METHODS
# =============================================================================

def build_cqm_based_bqm(n_plots: int, n_foods: int = 27) -> Tuple[BinaryQuadraticModel, Dict]:
    """
    Build BQM via CQM→BQM conversion (baseline method with slack variables).
    
    This is the CURRENT approach - has high density due to penalty terms.
    """
    plots = [f"Plot_{i}" for i in range(n_plots)]
    foods = [f"Food_{i}" for i in range(n_foods)]
    
    # Build CQM
    cqm = ConstrainedQuadraticModel()
    
    # Variables: Y[p,c] binary
    Y = {}
    for plot in plots:
        for food in foods:
            var_name = f"Y_{plot}_{food}"
            Y[(plot, food)] = Binary(var_name)
            cqm.add_variable('BINARY', var_name)
    
    # Objective: simple weighted sum (simulating real objective)
    objective = sum(
        (i % 10 + 1) * Y[(p, f)]  # Varying weights
        for i, (p, f) in enumerate((p, f) for p in plots for f in foods)
    )
    cqm.set_objective(-objective)  # Maximize
    
    # Constraint 1: Each plot can have at most 1 crop
    for plot in plots:
        cqm.add_constraint(
            sum(Y[(plot, food)] for food in foods) <= 1,
            label=f"Plot_{plot}_MaxOne"
        )
    
    # Constraint 2: Food group constraints (3 groups, require diversity)
    n_groups = 3
    foods_per_group = n_foods // n_groups
    for g in range(n_groups):
        group_foods = foods[g * foods_per_group:(g + 1) * foods_per_group]
        
        # At least 2 unique foods from this group across all plots
        total = sum(Y[(p, f)] for p in plots for f in group_foods)
        cqm.add_constraint(total >= 2, label=f"FoodGroup_{g}_Min")
        
        # At most n_plots foods from this group (prevent over-specialization)
        cqm.add_constraint(total <= n_plots, label=f"FoodGroup_{g}_Max")
    
    # Convert to BQM (this adds penalty terms and slack variables!)
    bqm, _ = cqm_to_bqm(cqm)
    
    info = {
        'method': 'CQM→BQM',
        'n_plots': n_plots,
        'n_foods': n_foods,
        'n_original_vars': n_plots * n_foods,
        'n_bqm_vars': len(bqm.variables),
        'n_linear': len(bqm.linear),
        'n_quadratic': len(bqm.quadratic),
        'density': 2 * len(bqm.quadratic) / (len(bqm.variables) * (len(bqm.variables) - 1)) 
                   if len(bqm.variables) > 1 else 0,
        'slack_vars_added': len(bqm.variables) - n_plots * n_foods
    }
    
    return bqm, info


def build_direct_bqm(n_plots: int, n_foods: int = 27) -> Tuple[BinaryQuadraticModel, Dict]:
    """
    Build BQM DIRECTLY without CQM conversion.
    
    This is the KEY OPTIMIZATION - no slack variables, much sparser!
    
    We encode constraints as QUBO penalties directly:
    - Plot assignment (at most 1 crop per plot) → one-hot penalty
    - Food diversity (soft constraint) → bonus for diversity
    """
    plots = [f"Plot_{i}" for i in range(n_plots)]
    foods = [f"Food_{i}" for i in range(n_foods)]
    
    bqm = BinaryQuadraticModel('BINARY')
    
    # Add variables with linear biases (objective coefficients)
    for i, plot in enumerate(plots):
        for j, food in enumerate(foods):
            var_name = f"Y_{plot}_{food}"
            # Linear bias: negative = incentive to select (we want to maximize)
            linear_bias = -((i + j) % 10 + 1) / 10.0  # Normalized varying weights
            bqm.add_variable(var_name, linear_bias)
    
    # Constraint 1: Each plot has at most 1 crop
    # Use one-hot encoding penalty: penalty if sum(Y_p_*) > 1
    # Penalty = lambda * (sum_i sum_j>i Y_i * Y_j) for same plot
    penalty_strength = 10.0  # Strong penalty for constraint violation
    
    for plot in plots:
        plot_vars = [f"Y_{plot}_{food}" for food in foods]
        # Add quadratic penalty for all pairs (creates "at most one" constraint)
        for i, v1 in enumerate(plot_vars):
            for v2 in plot_vars[i+1:]:
                bqm.add_quadratic(v1, v2, penalty_strength)
    
    # Constraint 2: Food group diversity (soft constraint via bonus)
    # Reward selecting different foods from same group
    # This is OPTIONAL and creates density - can be commented out
    diversity_bonus = 0.1  # Small bonus for diversity
    n_groups = 3
    foods_per_group = n_foods // n_groups
    
    for g in range(n_groups):
        group_foods = foods[g * foods_per_group:(g + 1) * foods_per_group]
        # Penalize selecting same food on multiple plots (encourage diversity)
        for food in group_foods:
            food_vars = [f"Y_{plot}_{food}" for plot in plots]
            # Penalty for selecting same food multiple times
            for i, v1 in enumerate(food_vars):
                for v2 in food_vars[i+1:]:
                    bqm.add_quadratic(v1, v2, diversity_bonus)
    
    info = {
        'method': 'Direct BQM',
        'n_plots': n_plots,
        'n_foods': n_foods,
        'n_original_vars': n_plots * n_foods,
        'n_bqm_vars': len(bqm.variables),
        'n_linear': len(bqm.linear),
        'n_quadratic': len(bqm.quadratic),
        'density': 2 * len(bqm.quadratic) / (len(bqm.variables) * (len(bqm.variables) - 1)) 
                   if len(bqm.variables) > 1 else 0,
        'slack_vars_added': 0
    }
    
    return bqm, info


def build_ultra_sparse_bqm(n_plots: int, n_foods: int = 27) -> Tuple[BinaryQuadraticModel, Dict]:
    """
    Build an ULTRA-SPARSE BQM with minimal quadratic terms.
    
    Strategy: Use only plot-local penalties (no cross-plot interactions).
    This creates a nearly embeddable problem for large sizes.
    """
    plots = [f"Plot_{i}" for i in range(n_plots)]
    foods = [f"Food_{i}" for i in range(n_foods)]
    
    bqm = BinaryQuadraticModel('BINARY')
    
    # Add variables with linear biases (objective coefficients)
    for i, plot in enumerate(plots):
        for j, food in enumerate(foods):
            var_name = f"Y_{plot}_{food}"
            linear_bias = -((i + j) % 10 + 1) / 10.0
            bqm.add_variable(var_name, linear_bias)
    
    # Constraint: Each plot has at most 1 crop (one-hot within each plot)
    penalty_strength = 10.0
    
    for plot in plots:
        plot_vars = [f"Y_{plot}_{food}" for food in foods]
        # Only penalize within same plot - NO cross-plot interactions!
        for i, v1 in enumerate(plot_vars):
            for v2 in plot_vars[i+1:]:
                bqm.add_quadratic(v1, v2, penalty_strength)
    
    info = {
        'method': 'Ultra-Sparse BQM',
        'n_plots': n_plots,
        'n_foods': n_foods,
        'n_original_vars': n_plots * n_foods,
        'n_bqm_vars': len(bqm.variables),
        'n_linear': len(bqm.linear),
        'n_quadratic': len(bqm.quadratic),
        'density': 2 * len(bqm.quadratic) / (len(bqm.variables) * (len(bqm.variables) - 1)) 
                   if len(bqm.variables) > 1 else 0,
        'slack_vars_added': 0,
        'structure': 'block-diagonal (no cross-plot interactions)'
    }
    
    return bqm, info


# =============================================================================
# DECOMPOSITION STRATEGIES
# =============================================================================

def get_bqm_graph(bqm: BinaryQuadraticModel) -> nx.Graph:
    """Convert BQM to NetworkX graph for analysis."""
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    G.add_edges_from(bqm.quadratic.keys())
    return G


def decompose_louvain(bqm: BinaryQuadraticModel, max_partition_size: int = 150) -> List[Set]:
    """
    Decompose BQM using Louvain community detection.
    
    Returns list of variable sets (partitions).
    """
    if not LOUVAIN_AVAILABLE:
        print("    Louvain not available")
        return [set(bqm.variables)]
    
    G = get_bqm_graph(bqm)
    
    if len(G.nodes) <= max_partition_size:
        return [set(G.nodes)]
    
    # Detect communities
    communities = louvain_communities(G, seed=42)
    partitions = [set(c) for c in communities]
    
    # Split oversized partitions
    final_partitions = []
    for partition in partitions:
        if len(partition) <= max_partition_size:
            final_partitions.append(partition)
        else:
            # Split oversized partition
            partition_list = list(partition)
            for i in range(0, len(partition_list), max_partition_size):
                chunk = set(partition_list[i:i + max_partition_size])
                final_partitions.append(chunk)
    
    return final_partitions


def extract_sub_bqm(bqm: BinaryQuadraticModel, variables: Set) -> BinaryQuadraticModel:
    """Extract a sub-BQM containing only specified variables."""
    sub_bqm = BinaryQuadraticModel('BINARY')
    
    for var in variables:
        if var in bqm.linear:
            sub_bqm.add_variable(var, bqm.linear[var])
    
    for (u, v), bias in bqm.quadratic.items():
        if u in variables and v in variables:
            sub_bqm.add_quadratic(u, v, bias)
    
    return sub_bqm


def decompose_plot_based(bqm: BinaryQuadraticModel, plots_per_partition: int = 5) -> List[Set]:
    """
    Decompose BQM based on plot structure.
    
    Groups variables by their plot origin for natural partitioning.
    """
    # Group variables by plot
    plot_vars = {}
    for var in bqm.variables:
        if var.startswith('Y_'):
            parts = var.split('_')
            if len(parts) >= 3:
                plot = f"{parts[1]}_{parts[2]}" if parts[1] == 'Plot' else parts[1]
                if plot not in plot_vars:
                    plot_vars[plot] = set()
                plot_vars[plot].add(var)
    
    # Group plots into partitions
    plots = list(plot_vars.keys())
    partitions = []
    
    for i in range(0, len(plots), plots_per_partition):
        chunk_plots = plots[i:i + plots_per_partition]
        partition = set()
        for plot in chunk_plots:
            partition.update(plot_vars.get(plot, set()))
        if partition:
            partitions.append(partition)
    
    return partitions


# =============================================================================
# EMBEDDING STUDY
# =============================================================================

def get_target_graph() -> nx.Graph:
    """Get the QPU target graph (Pegasus topology)."""
    if DWAVE_AVAILABLE and TOKEN:
        try:
            print("    Connecting to D-Wave to get hardware graph...")
            sampler = DWaveSampler(token=TOKEN)
            G = nx.Graph()
            G.add_nodes_from(sampler.nodelist)
            G.add_edges_from(sampler.edgelist)
            print(f"    ✅ Got hardware graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
            return G
        except Exception as e:
            print(f"    Warning: Could not connect to D-Wave: {e}")
    
    # Fallback: simulate Pegasus P16 topology
    print("    Using simulated Pegasus P16 topology...")
    try:
        import dwave_networkx as dnx
        G = dnx.pegasus_graph(16)
        print(f"    ✅ Simulated graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G
    except ImportError:
        print("    Warning: dwave_networkx not available, using Chimera fallback")
        import dwave_networkx as dnx
        G = dnx.chimera_graph(16, 16, 4)
        return G


def study_embedding(bqm: BinaryQuadraticModel, target_graph: nx.Graph,
                    timeout: int = 300, attempts: int = 3) -> Dict:
    """
    Study embedding characteristics without running QPU.
    """
    source_graph = get_bqm_graph(bqm)
    
    results = {
        'attempts': [],
        'success_count': 0,
        'fail_count': 0,
        'best_embedding': None,
        'best_chain_stats': None
    }
    
    for attempt in range(attempts):
        print(f"        Attempt {attempt + 1}/{attempts}...", end=" ", flush=True)
        attempt_start = time.time()
        
        try:
            embedding = minorminer.find_embedding(
                source_graph.edges(),
                target_graph.edges(),
                timeout=timeout,
                verbose=0
            )
            
            embed_time = time.time() - attempt_start
            
            if embedding:
                chain_lengths = [len(chain) for chain in embedding.values()]
                physical_qubits = sum(chain_lengths)
                
                attempt_result = {
                    'success': True,
                    'time': embed_time,
                    'physical_qubits': physical_qubits,
                    'chain_lengths': {
                        'min': min(chain_lengths),
                        'max': max(chain_lengths),
                        'mean': statistics.mean(chain_lengths),
                        'median': statistics.median(chain_lengths),
                        'stdev': statistics.stdev(chain_lengths) if len(chain_lengths) > 1 else 0
                    },
                    'qubit_ratio': physical_qubits / len(source_graph.nodes)
                }
                
                results['success_count'] += 1
                
                if (results['best_embedding'] is None or 
                    attempt_result['chain_lengths']['max'] < results['best_chain_stats']['max']):
                    results['best_embedding'] = embedding
                    results['best_chain_stats'] = attempt_result['chain_lengths']
                
                print(f"✓ ({embed_time:.1f}s) - {physical_qubits} qubits, max chain: {max(chain_lengths)}")
            else:
                results['fail_count'] += 1
                attempt_result = {
                    'success': False,
                    'time': embed_time,
                    'error': 'Empty embedding returned'
                }
                print(f"✗ ({embed_time:.1f}s) - Empty embedding")
                
        except Exception as e:
            embed_time = time.time() - attempt_start
            results['fail_count'] += 1
            attempt_result = {
                'success': False,
                'time': embed_time,
                'error': str(e)
            }
            print(f"✗ ({embed_time:.1f}s) - {e}")
        
        results['attempts'].append(attempt_result)
    
    return results


def study_decomposed_embedding(bqm: BinaryQuadraticModel, target_graph: nx.Graph,
                               decomposer_name: str, partitions: List[Set],
                               timeout: int = 300) -> Dict:
    """
    Study embedding for decomposed (partitioned) problem.
    """
    results = {
        'decomposer': decomposer_name,
        'n_partitions': len(partitions),
        'partition_sizes': [len(p) for p in partitions],
        'partition_results': [],
        'all_success': True,
        'total_physical_qubits': 0
    }
    
    for i, partition in enumerate(partitions):
        print(f"      Partition {i+1}/{len(partitions)} ({len(partition)} vars)...", end=" ", flush=True)
        
        sub_bqm = extract_sub_bqm(bqm, partition)
        source_graph = get_bqm_graph(sub_bqm)
        
        if len(source_graph.edges) == 0:
            # No quadratic terms - trivially embeddable
            print(f"✓ (trivial - no quadratic terms)")
            results['partition_results'].append({
                'partition_id': i,
                'n_vars': len(partition),
                'n_edges': 0,
                'success': True,
                'physical_qubits': len(partition),
                'trivial': True
            })
            results['total_physical_qubits'] += len(partition)
            continue
        
        try:
            start = time.time()
            embedding = minorminer.find_embedding(
                source_graph.edges(),
                target_graph.edges(),
                timeout=timeout,
                verbose=0
            )
            embed_time = time.time() - start
            
            if embedding:
                chain_lengths = [len(chain) for chain in embedding.values()]
                physical_qubits = sum(chain_lengths)
                
                print(f"✓ ({embed_time:.1f}s) - {physical_qubits} qubits")
                results['partition_results'].append({
                    'partition_id': i,
                    'n_vars': len(partition),
                    'n_edges': len(source_graph.edges),
                    'success': True,
                    'time': embed_time,
                    'physical_qubits': physical_qubits,
                    'max_chain': max(chain_lengths)
                })
                results['total_physical_qubits'] += physical_qubits
            else:
                print(f"✗ ({embed_time:.1f}s)")
                results['partition_results'].append({
                    'partition_id': i,
                    'n_vars': len(partition),
                    'success': False,
                    'time': embed_time
                })
                results['all_success'] = False
                
        except Exception as e:
            print(f"✗ ({e})")
            results['partition_results'].append({
                'partition_id': i,
                'n_vars': len(partition),
                'success': False,
                'error': str(e)
            })
            results['all_success'] = False
    
    return results


# =============================================================================
# MAIN STUDY
# =============================================================================

def main():
    """Run the comprehensive binary plot embedding study."""
    print("\n[2/5] Setting up study parameters...")
    print(f"      Problem sizes: {PROBLEM_SIZES}")
    print(f"      Foods per problem: {N_FOODS}")
    print(f"      Embedding attempts: {EMBEDDING_ATTEMPTS}")
    print(f"      Timeout per attempt: {EMBEDDING_TIMEOUT}s")
    
    # Get target graph
    print("\n[3/5] Getting target graph...")
    target_graph = get_target_graph()
    
    # Results storage
    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_foods': N_FOODS,
            'embedding_attempts': EMBEDDING_ATTEMPTS,
            'embedding_timeout': EMBEDDING_TIMEOUT,
            'target_graph_nodes': len(target_graph.nodes),
            'target_graph_edges': len(target_graph.edges)
        },
        'bqm_comparison': {},
        'embedding_results': {}
    }
    
    print("\n[4/5] Running embedding studies...")
    
    for n_plots in PROBLEM_SIZES:
        print(f"\n{'='*60}")
        print(f"PROBLEM SIZE: {n_plots} plots × {N_FOODS} foods")
        print(f"{'='*60}")
        
        all_results['embedding_results'][n_plots] = {}
        
        # Build BQMs with different methods
        print("\n  Building BQMs...")
        
        print("    1. CQM→BQM (baseline)...", end=" ", flush=True)
        cqm_bqm, cqm_info = build_cqm_based_bqm(n_plots, N_FOODS)
        print(f"✓ ({cqm_info['n_bqm_vars']} vars, {cqm_info['n_quadratic']} quadratic, density={cqm_info['density']:.3f})")
        
        print("    2. Direct BQM...", end=" ", flush=True)
        direct_bqm, direct_info = build_direct_bqm(n_plots, N_FOODS)
        print(f"✓ ({direct_info['n_bqm_vars']} vars, {direct_info['n_quadratic']} quadratic, density={direct_info['density']:.3f})")
        
        print("    3. Ultra-Sparse BQM...", end=" ", flush=True)
        sparse_bqm, sparse_info = build_ultra_sparse_bqm(n_plots, N_FOODS)
        print(f"✓ ({sparse_info['n_bqm_vars']} vars, {sparse_info['n_quadratic']} quadratic, density={sparse_info['density']:.3f})")
        
        all_results['bqm_comparison'][n_plots] = {
            'cqm_based': cqm_info,
            'direct': direct_info,
            'ultra_sparse': sparse_info
        }
        
        # Test embeddings
        print("\n  Testing embeddings...")
        
        # 1. Direct embedding - CQM-based BQM
        print("\n    Strategy 1: Direct embedding (CQM→BQM)")
        if cqm_info['n_bqm_vars'] <= 500:  # Only try if reasonable size
            cqm_embed = study_embedding(cqm_bqm, target_graph, EMBEDDING_TIMEOUT, EMBEDDING_ATTEMPTS)
            all_results['embedding_results'][n_plots]['cqm_direct'] = {
                'bqm_info': cqm_info,
                'success_rate': cqm_embed['success_count'] / EMBEDDING_ATTEMPTS * 100,
                'results': cqm_embed
            }
        else:
            print(f"      Skipped - too large ({cqm_info['n_bqm_vars']} vars)")
            all_results['embedding_results'][n_plots]['cqm_direct'] = {
                'bqm_info': cqm_info,
                'skipped': True,
                'reason': 'too_large'
            }
        
        # 2. Direct embedding - Direct BQM
        print("\n    Strategy 2: Direct embedding (Direct BQM)")
        direct_embed = study_embedding(direct_bqm, target_graph, EMBEDDING_TIMEOUT, EMBEDDING_ATTEMPTS)
        all_results['embedding_results'][n_plots]['direct_bqm'] = {
            'bqm_info': direct_info,
            'success_rate': direct_embed['success_count'] / EMBEDDING_ATTEMPTS * 100,
            'results': direct_embed
        }
        
        # 3. Direct embedding - Ultra-Sparse BQM
        print("\n    Strategy 3: Direct embedding (Ultra-Sparse BQM)")
        sparse_embed = study_embedding(sparse_bqm, target_graph, EMBEDDING_TIMEOUT, EMBEDDING_ATTEMPTS)
        all_results['embedding_results'][n_plots]['ultra_sparse'] = {
            'bqm_info': sparse_info,
            'success_rate': sparse_embed['success_count'] / EMBEDDING_ATTEMPTS * 100,
            'results': sparse_embed
        }
        
        # 4. Louvain decomposition on Direct BQM
        if LOUVAIN_AVAILABLE and direct_info['n_bqm_vars'] > 150:
            print("\n    Strategy 4: Louvain decomposition (Direct BQM)")
            partitions = decompose_louvain(direct_bqm, max_partition_size=150)
            print(f"      Created {len(partitions)} partitions")
            louvain_results = study_decomposed_embedding(
                direct_bqm, target_graph, 'louvain', partitions, timeout=120
            )
            all_results['embedding_results'][n_plots]['louvain_decomposed'] = louvain_results
        
        # 5. Plot-based decomposition
        if n_plots > 5:
            print("\n    Strategy 5: Plot-based decomposition (Direct BQM)")
            plot_partitions = decompose_plot_based(direct_bqm, plots_per_partition=5)
            print(f"      Created {len(plot_partitions)} partitions (5 plots each)")
            plot_results = study_decomposed_embedding(
                direct_bqm, target_graph, 'plot_based', plot_partitions, timeout=120
            )
            all_results['embedding_results'][n_plots]['plot_based_decomposed'] = plot_results
    
    # Save results
    print("\n[5/5] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"binary_plot_embedding_study_{timestamp}.json"
    
    # Convert sets to lists for JSON serialization
    def convert_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets(i) for i in obj]
        return obj
    
    all_results = convert_sets(all_results)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"      ✅ Results saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nBQM Structure Comparison:")
    print(f"{'Size':<10} {'CQM→BQM Vars':<15} {'Direct Vars':<15} {'Sparse Vars':<15} {'CQM Density':<15} {'Direct Density':<15}")
    print("-" * 85)
    
    for n_plots in PROBLEM_SIZES:
        comp = all_results['bqm_comparison'][n_plots]
        print(f"{n_plots:<10} {comp['cqm_based']['n_bqm_vars']:<15} {comp['direct']['n_bqm_vars']:<15} {comp['ultra_sparse']['n_bqm_vars']:<15} {comp['cqm_based']['density']:<15.3f} {comp['direct']['density']:<15.3f}")
    
    print("\nEmbedding Success Rates:")
    print(f"{'Size':<10} {'CQM Direct':<15} {'Direct BQM':<15} {'Ultra-Sparse':<15} {'Louvain':<15} {'Plot-Based':<15}")
    print("-" * 85)
    
    for n_plots in PROBLEM_SIZES:
        results = all_results['embedding_results'][n_plots]
        
        cqm_rate = results.get('cqm_direct', {}).get('success_rate', 'N/A')
        if isinstance(cqm_rate, (int, float)):
            cqm_rate = f"{cqm_rate:.0f}%"
        
        direct_rate = results.get('direct_bqm', {}).get('success_rate', 'N/A')
        if isinstance(direct_rate, (int, float)):
            direct_rate = f"{direct_rate:.0f}%"
        
        sparse_rate = results.get('ultra_sparse', {}).get('success_rate', 'N/A')
        if isinstance(sparse_rate, (int, float)):
            sparse_rate = f"{sparse_rate:.0f}%"
        
        louvain_result = results.get('louvain_decomposed', {})
        louvain_rate = "100%" if louvain_result.get('all_success') else f"{louvain_result.get('n_partitions', 0)} parts" if louvain_result else "N/A"
        
        plot_result = results.get('plot_based_decomposed', {})
        plot_rate = "100%" if plot_result.get('all_success') else f"{plot_result.get('n_partitions', 0)} parts" if plot_result else "N/A"
        
        print(f"{n_plots:<10} {cqm_rate:<15} {direct_rate:<15} {sparse_rate:<15} {louvain_rate:<15} {plot_rate:<15}")
    
    print("\n✅ Study complete!")
    return all_results


if __name__ == "__main__":
    main()
