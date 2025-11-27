#!/usr/bin/env python3
"""
Decomposition Framework for Comprehensive Benchmark

This module provides a unified interface for all decomposition strategies:
1. Louvain Graph Partitioning
2. Plot-Based Partitioning  
3. Energy-Impact Decomposition (dwave-hybrid)
4. QBSolv (placeholder)

Each decomposer returns a list of BQM partitions that can be solved independently.

Author: Generated for OQI-UC002-DWave
Date: 2025-11-27
"""

import time
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx
import numpy as np

try:
    from networkx.algorithms.community import louvain_communities
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

try:
    from hybrid.decomposers import EnergyImpactDecomposer
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False

from dimod import BinaryQuadraticModel


class DecompositionResult:
    """Container for decomposition results."""
    
    def __init__(self, strategy_name: str, partitions: List[BinaryQuadraticModel], 
                 metadata: Dict):
        self.strategy_name = strategy_name
        self.partitions = partitions
        self.metadata = metadata
        self.decomposition_time = metadata.get('decomposition_time', 0.0)
        
    def __repr__(self):
        return (f"DecompositionResult(strategy='{self.strategy_name}', "
                f"n_partitions={len(self.partitions)}, "
                f"time={self.decomposition_time:.2f}s)")


def get_bqm_graph(bqm: BinaryQuadraticModel) -> nx.Graph:
    """Extract interaction graph from BQM."""
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    G.add_edges_from(bqm.quadratic.keys())
    return G


def extract_sub_bqm(bqm: BinaryQuadraticModel, variables: Set) -> BinaryQuadraticModel:
    """Extract a sub-BQM containing only the specified variables."""
    sub_bqm = BinaryQuadraticModel(bqm.vartype)
    
    # Add linear terms
    for var in variables:
        if var in bqm.linear:
            sub_bqm.add_variable(var, bqm.linear[var])
    
    # Add quadratic terms (only internal edges)
    for (u, v), bias in bqm.quadratic.items():
        if u in variables and v in variables:
            sub_bqm.add_interaction(u, v, bias)
    
    sub_bqm.offset = 0  # Don't transfer offset to subproblems
    
    return sub_bqm


# =============================================================================
# STRATEGY 1: Louvain Graph Partitioning
# =============================================================================

def decompose_louvain(bqm: BinaryQuadraticModel, 
                     max_partition_size: int = 150,
                     min_partition_size: int = 20) -> DecompositionResult:
    """
    Decompose BQM using Louvain community detection.
    
    This keeps densely connected variables together.
    
    Args:
        bqm: Binary Quadratic Model to partition
        max_partition_size: Maximum variables per partition
        min_partition_size: Minimum variables per partition
        
    Returns:
        DecompositionResult with partitions
    """
    if not HAS_LOUVAIN:
        raise ImportError("networkx.algorithms.community required for Louvain decomposition")
    
    start_time = time.time()
    
    G = get_bqm_graph(bqm)
    n_vars = len(bqm.variables)
    
    if n_vars <= max_partition_size:
        # No decomposition needed
        return DecompositionResult(
            strategy_name='louvain',
            partitions=[bqm],
            metadata={
                'decomposition_time': time.time() - start_time,
                'n_partitions': 1,
                'partition_sizes': [n_vars],
                'skipped': True,
                'reason': 'BQM already small enough'
            }
        )
    
    # Try different resolution parameters to get acceptable partition sizes
    best_partitions = None
    best_max_size = float('inf')
    
    for resolution in [2.0, 1.5, 1.0, 0.8, 0.5, 0.3, 0.1]:
        communities = louvain_communities(G, resolution=resolution, seed=42)
        partitions_list = [set(community) for community in communities]
        max_size = max(len(p) for p in partitions_list)
        
        if max_size <= max_partition_size:
            best_partitions = partitions_list
            best_max_size = max_size
            break
        
        if max_size < best_max_size:
            best_partitions = partitions_list
            best_max_size = max_size
    
    if best_partitions is None:
        # Fallback: random balanced partitions
        all_vars = list(bqm.variables)
        np.random.shuffle(all_vars)
        n_partitions = int(np.ceil(n_vars / max_partition_size))
        partition_size = int(np.ceil(n_vars / n_partitions))
        
        best_partitions = []
        for i in range(n_partitions):
            start_idx = i * partition_size
            end_idx = min((i + 1) * partition_size, n_vars)
            best_partitions.append(set(all_vars[start_idx:end_idx]))
    
    # Merge very small partitions
    merged_partitions = []
    current_partition = set()
    
    for partition in sorted(best_partitions, key=len):
        if len(current_partition) + len(partition) <= max_partition_size:
            current_partition.update(partition)
        else:
            if current_partition:
                merged_partitions.append(current_partition)
            current_partition = partition
    
    if current_partition:
        if len(current_partition) < min_partition_size and merged_partitions:
            merged_partitions[-1].update(current_partition)
        else:
            merged_partitions.append(current_partition)
    
    # Extract sub-BQMs
    partition_bqms = []
    for partition_vars in merged_partitions:
        sub_bqm = extract_sub_bqm(bqm, partition_vars)
        partition_bqms.append(sub_bqm)
    
    decomp_time = time.time() - start_time
    
    return DecompositionResult(
        strategy_name='louvain',
        partitions=partition_bqms,
        metadata={
            'decomposition_time': decomp_time,
            'n_partitions': len(partition_bqms),
            'partition_sizes': [len(p.variables) for p in partition_bqms],
            'max_partition_size': max(len(p.variables) for p in partition_bqms),
            'min_partition_size': min(len(p.variables) for p in partition_bqms),
            'total_variables': sum(len(p.variables) for p in partition_bqms),
            'original_variables': n_vars
        }
    )


# =============================================================================
# STRATEGY 2: Plot-Based Partitioning
# =============================================================================

def decompose_plot_based(bqm: BinaryQuadraticModel, 
                        plots_per_partition: int = 5,
                        n_foods: int = 27) -> DecompositionResult:
    """
    Decompose BQM by grouping plots together.
    
    Assumes variables are named: Y_{plot}_{crop}
    
    Args:
        bqm: Binary Quadratic Model to partition
        plots_per_partition: Number of plots per partition
        n_foods: Number of food types (crops)
        
    Returns:
        DecompositionResult with partitions
    """
    start_time = time.time()
    
    # Extract plot names from variable names
    plots = set()
    for var in bqm.variables:
        if var.startswith('Y_'):
            parts = var.split('_')
            if len(parts) >= 2:
                plot_name = parts[1]
                plots.add(plot_name)
    
    plots = sorted(plots)
    n_plots = len(plots)
    
    if n_plots <= plots_per_partition:
        # No decomposition needed
        return DecompositionResult(
            strategy_name='plot_based',
            partitions=[bqm],
            metadata={
                'decomposition_time': time.time() - start_time,
                'n_partitions': 1,
                'partition_sizes': [len(bqm.variables)],
                'skipped': True,
                'reason': f'Only {n_plots} plots, <= {plots_per_partition}'
            }
        )
    
    # Create partitions
    n_partitions = int(np.ceil(n_plots / plots_per_partition))
    plot_partitions = []
    
    for i in range(n_partitions):
        start_idx = i * plots_per_partition
        end_idx = min((i + 1) * plots_per_partition, n_plots)
        partition_plots = plots[start_idx:end_idx]
        plot_partitions.append(partition_plots)
    
    # Extract variables for each partition
    partition_bqms = []
    
    for partition_plots in plot_partitions:
        # Get all variables for these plots
        partition_vars = set()
        for var in bqm.variables:
            if var.startswith('Y_'):
                parts = var.split('_')
                if len(parts) >= 2 and parts[1] in partition_plots:
                    partition_vars.add(var)
        
        sub_bqm = extract_sub_bqm(bqm, partition_vars)
        partition_bqms.append(sub_bqm)
    
    decomp_time = time.time() - start_time
    
    return DecompositionResult(
        strategy_name='plot_based',
        partitions=partition_bqms,
        metadata={
            'decomposition_time': decomp_time,
            'n_partitions': len(partition_bqms),
            'partition_sizes': [len(p.variables) for p in partition_bqms],
            'plots_per_partition': plots_per_partition,
            'plot_partitions': plot_partitions,
            'total_variables': sum(len(p.variables) for p in partition_bqms),
            'original_variables': len(bqm.variables)
        }
    )


# =============================================================================
# STRATEGY 3: Energy-Impact Decomposition
# =============================================================================

def decompose_energy_impact(bqm: BinaryQuadraticModel, 
                           partition_size: int = 100,
                           traversal: str = 'bfs') -> DecompositionResult:
    """
    Decompose BQM using energy-impact decomposition from dwave-hybrid.
    
    Args:
        bqm: Binary Quadratic Model to partition
        partition_size: Target size for each partition
        traversal: Traversal mode ('bfs' or 'pfs')
        
    Returns:
        DecompositionResult with partitions
    """
    if not HAS_HYBRID:
        raise ImportError("dwave-hybrid required for energy-impact decomposition")
    
    start_time = time.time()
    
    n_vars = len(bqm.variables)
    
    if n_vars <= partition_size:
        # No decomposition needed
        return DecompositionResult(
            strategy_name='energy_impact',
            partitions=[bqm],
            metadata={
                'decomposition_time': time.time() - start_time,
                'n_partitions': 1,
                'partition_sizes': [n_vars],
                'skipped': True,
                'reason': 'BQM already small enough'
            }
        )
    
    # Use EnergyImpactDecomposer to identify high-impact subproblems
    decomposer = EnergyImpactDecomposer(
        size=partition_size,
        traversal=traversal,
        rolling_history=0.85
    )
    
    # We need to decompose iteratively until all variables are covered
    remaining_vars = set(bqm.variables)
    partition_var_sets = []
    
    # Create a dummy initial state (all zeros)
    from hybrid import State
    import dimod
    initial_sample = {var: 0 for var in bqm.variables}
    state = State.from_sample(initial_sample, bqm)
    
    max_iterations = int(np.ceil(n_vars / partition_size)) + 2
    
    for iteration in range(max_iterations):
        if not remaining_vars:
            break
        
        # Run decomposer
        next_state = decomposer.next(state)
        
        if next_state.subproblem is None:
            break
        
        # Extract subproblem variables
        subproblem_vars = set(next_state.subproblem.variables)
        
        # Only take variables we haven't seen yet
        new_vars = subproblem_vars & remaining_vars
        
        if new_vars:
            partition_var_sets.append(new_vars)
            remaining_vars -= new_vars
        
        state = next_state
    
    # If there are still remaining variables, add them to last partition or create new one
    if remaining_vars:
        if partition_var_sets:
            partition_var_sets[-1].update(remaining_vars)
        else:
            partition_var_sets.append(remaining_vars)
    
    # Extract sub-BQMs
    partition_bqms = []
    for partition_vars in partition_var_sets:
        sub_bqm = extract_sub_bqm(bqm, partition_vars)
        partition_bqms.append(sub_bqm)
    
    decomp_time = time.time() - start_time
    
    return DecompositionResult(
        strategy_name='energy_impact',
        partitions=partition_bqms,
        metadata={
            'decomposition_time': decomp_time,
            'n_partitions': len(partition_bqms),
            'partition_sizes': [len(p.variables) for p in partition_bqms],
            'traversal': traversal,
            'target_partition_size': partition_size,
            'total_variables': sum(len(p.variables) for p in partition_bqms),
            'original_variables': n_vars
        }
    )


# =============================================================================
# STRATEGY 4: QBSolv (Placeholder)
# =============================================================================

def decompose_qbsolv(bqm: BinaryQuadraticModel, 
                     subproblem_size: int = 100) -> DecompositionResult:
    """
    QBSolv decomposition (PLACEHOLDER - not implemented).
    
    QBSolv automatically decomposes problems but requires D-Wave setup.
    This is a placeholder for future implementation.
    
    Args:
        bqm: Binary Quadratic Model to partition
        subproblem_size: Target subproblem size
        
    Returns:
        DecompositionResult (empty placeholder)
    """
    return DecompositionResult(
        strategy_name='qbsolv',
        partitions=[],
        metadata={
            'decomposition_time': 0.0,
            'n_partitions': 0,
            'partition_sizes': [],
            'status': 'placeholder',
            'reason': 'QBSolv not configured - requires D-Wave QBSolv setup',
            'original_variables': len(bqm.variables)
        }
    )


# =============================================================================
# UNIFIED DECOMPOSITION INTERFACE
# =============================================================================

def decompose_bqm(bqm: BinaryQuadraticModel, 
                 strategy: str,
                 **kwargs) -> DecompositionResult:
    """
    Unified interface for all decomposition strategies.
    
    Args:
        bqm: Binary Quadratic Model to decompose
        strategy: Strategy name ('louvain', 'plot_based', 'energy_impact', 'qbsolv')
        **kwargs: Strategy-specific parameters
        
    Returns:
        DecompositionResult
    """
    strategy = strategy.lower()
    
    if strategy == 'louvain':
        return decompose_louvain(bqm, **kwargs)
    elif strategy == 'plot_based':
        return decompose_plot_based(bqm, **kwargs)
    elif strategy == 'energy_impact':
        return decompose_energy_impact(bqm, **kwargs)
    elif strategy == 'qbsolv':
        return decompose_qbsolv(bqm, **kwargs)
    else:
        raise ValueError(f"Unknown decomposition strategy: {strategy}")


if __name__ == "__main__":
    print("Testing decomposition framework...")
    
    # Create a simple test BQM
    from dimod import BinaryQuadraticModel, Binary
    
    print("\nCreating test BQM (10 plots × 5 foods = 50 variables)...")
    bqm = BinaryQuadraticModel('BINARY')
    
    for plot in range(10):
        for food in range(5):
            var = f"Y_Plot{plot}_Food{food}"
            bqm.add_variable(var, -1.0)  # Encourage selection
    
    # Add some quadratic terms
    for plot in range(10):
        vars_in_plot = [f"Y_Plot{plot}_Food{food}" for food in range(5)]
        for i, v1 in enumerate(vars_in_plot):
            for v2 in vars_in_plot[i+1:]:
                bqm.add_interaction(v1, v2, 2.0)  # Penalty
    
    print(f"   BQM: {len(bqm.variables)} vars, {len(bqm.quadratic)} quadratic terms")
    
    # Test each strategy
    print("\n1. Testing Louvain decomposition...")
    result = decompose_louvain(bqm, max_partition_size=30)
    print(f"   {result}")
    print(f"   Partition sizes: {result.metadata['partition_sizes']}")
    
    print("\n2. Testing Plot-based decomposition...")
    result = decompose_plot_based(bqm, plots_per_partition=3, n_foods=5)
    print(f"   {result}")
    print(f"   Partition sizes: {result.metadata['partition_sizes']}")
    
    if HAS_HYBRID:
        print("\n3. Testing Energy-impact decomposition...")
        result = decompose_energy_impact(bqm, partition_size=25)
        print(f"   {result}")
        print(f"   Partition sizes: {result.metadata['partition_sizes']}")
    else:
        print("\n3. Energy-impact decomposition: dwave-hybrid not available")
    
    print("\n4. Testing QBSolv decomposition (placeholder)...")
    result = decompose_qbsolv(bqm)
    print(f"   {result}")
    print(f"   Status: {result.metadata['status']}")
    
    print("\n✅ All decomposition strategies tested!")
