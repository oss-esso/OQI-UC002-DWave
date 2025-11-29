#!/usr/bin/env python3
"""
Advanced Graph Decomposition Strategies

Implements sophisticated decomposition methods from literature:
1. Multilevel Decomposition (ML-QLS style) - graph coarsening + refinement
2. Sequential Cut-Set Reduction - iterative graph reduction via cut sets

Based on:
- graph_decomp_QLS.tex (Multilevel Quantum Local Search)
- graph_decomp_sequential.tex (Cut-set based reduction)

Author: Generated for OQI-UC002-DWave
Date: 2025-11-27
"""

import networkx as nx
import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from dimod import BinaryQuadraticModel


# =============================================================================
# MULTILEVEL DECOMPOSITION (ML-QLS TYPE)
# =============================================================================

def coarsen_graph_matching(G: nx.Graph, target_size: int = 50) -> Tuple[nx.Graph, Dict]:
    """
    Coarsen graph using maximum weight matching.
    
    Creates hierarchy of smaller graphs by repeatedly matching and merging nodes.
    Similar to multilevel graph partitioning approaches.
    
    Args:
        G: Input graph
        target_size: Target number of nodes in coarsened graph
        
    Returns:
        Coarsened graph and mapping from coarse to fine nodes
    """
    if len(G.nodes) <= target_size:
        return G, {node: {node} for node in G.nodes}
    
    # Find maximum weight matching
    matching = nx.max_weight_matching(G, maxcardinality=True)
    
    # Create coarse graph
    G_coarse = nx.Graph()
    coarse_to_fine = {}
    node_counter = 0
    matched_nodes = set()
    
    # Merge matched pairs
    for u, v in matching:
        coarse_node = f"C{node_counter}"
        coarse_to_fine[coarse_node] = {u, v}
        G_coarse.add_node(coarse_node)
        matched_nodes.update([u, v])
        node_counter += 1
    
    # Add unmatched nodes
    for node in G.nodes:
        if node not in matched_nodes:
            coarse_node = f"C{node_counter}"
            coarse_to_fine[coarse_node] = {node}
            G_coarse.add_node(coarse_node)
            node_counter += 1
    
    # Add edges between coarse nodes
    for c1, fine1 in coarse_to_fine.items():
        for c2, fine2 in coarse_to_fine.items():
            if c1 >= c2:
                continue
            # Check if any nodes in fine1 connect to any in fine2
            for f1 in fine1:
                for f2 in fine2:
                    if G.has_edge(f1, f2):
                        # Aggregate edge weight
                        weight = G[f1][f2].get('weight', 1)
                        if G_coarse.has_edge(c1, c2):
                            G_coarse[c1][c2]['weight'] += weight
                        else:
                            G_coarse.add_edge(c1, c2, weight=weight)
                        break
    
    return G_coarse, coarse_to_fine


def multilevel_coarsening(G: nx.Graph, levels: int = 3, target_size: int = 50) -> List[Tuple[nx.Graph, Dict]]:
    """
    Create multilevel hierarchy by repeated coarsening.
    
    Args:
        G: Input graph
        levels: Number of coarsening levels
        target_size: Target size for coarsest level
        
    Returns:
        List of (graph, mapping) tuples from fine to coarse
    """
    hierarchy = [(G, {node: {node} for node in G.nodes})]
    current_graph = G
    
    for level in range(levels):
        if len(current_graph.nodes) <= target_size:
            break
        
        coarse_graph, mapping = coarsen_graph_matching(current_graph, target_size)
        
        # Combine mappings from previous level
        combined_mapping = {}
        prev_mapping = hierarchy[-1][1]
        
        for coarse_node, medium_nodes in mapping.items():
            fine_nodes = set()
            for medium_node in medium_nodes:
                if medium_node in prev_mapping:
                    fine_nodes.update(prev_mapping[medium_node])
                else:
                    fine_nodes.add(medium_node)
            combined_mapping[coarse_node] = fine_nodes
        
        hierarchy.append((coarse_graph, combined_mapping))
        current_graph = coarse_graph
    
    return hierarchy


def decompose_multilevel(bqm: BinaryQuadraticModel, levels: int = 2, 
                         partition_size: int = 100) -> List[Set]:
    """
    Multilevel decomposition (ML-QLS style).
    
    Strategy:
    1. Coarsen graph into hierarchy
    2. Partition coarsest level
    3. Project partitions back to fine level
    4. Refine boundaries
    
    Args:
        bqm: Binary Quadratic Model
        levels: Number of coarsening levels
        partition_size: Target partition size
        
    Returns:
        List of variable sets (partitions)
    """
    # Convert BQM to graph
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    G.add_edges_from(bqm.quadratic.keys())
    
    if len(G.nodes) <= partition_size:
        return [set(G.nodes)]
    
    # Create hierarchy
    hierarchy = multilevel_coarsening(G, levels=levels, target_size=partition_size)
    
    # Get coarsest graph
    coarsest_graph, coarse_to_fine = hierarchy[-1]
    
    # Partition coarsest graph (simple greedy approach)
    coarse_partitions = []
    remaining_nodes = set(coarsest_graph.nodes)
    
    while remaining_nodes:
        # Start new partition
        partition = set()
        seed = remaining_nodes.pop()
        partition.add(seed)
        
        # Grow partition greedily
        while len(partition) < partition_size and remaining_nodes:
            # Find node with most connections to current partition
            best_node = None
            best_score = -1
            
            for candidate in remaining_nodes:
                score = sum(1 for n in partition if coarsest_graph.has_edge(candidate, n))
                if score > best_score:
                    best_score = score
                    best_node = candidate
            
            if best_node is None:
                break
            
            partition.add(best_node)
            remaining_nodes.discard(best_node)
        
        coarse_partitions.append(partition)
    
    # Project back to fine level
    fine_partitions = []
    for coarse_partition in coarse_partitions:
        fine_partition = set()
        for coarse_node in coarse_partition:
            fine_partition.update(coarse_to_fine[coarse_node])
        fine_partitions.append(fine_partition)
    
    return fine_partitions


# =============================================================================
# SEQUENTIAL CUT-SET REDUCTION
# =============================================================================

def find_minimum_vertex_cut(G: nx.Graph, max_cut_size: int = 5) -> Optional[Set]:
    """
    Find a small vertex cut that separates the graph.
    
    Uses node connectivity algorithms to find separator.
    
    Args:
        G: Input graph
        max_cut_size: Maximum allowed cut set size
        
    Returns:
        Set of nodes forming a cut, or None if no small cut exists
    """
    if not nx.is_connected(G):
        return None
    
    # Try to find small vertex cuts
    try:
        # Get node connectivity
        connectivity = nx.node_connectivity(G)
        
        if connectivity > max_cut_size:
            return None
        
        # Find actual cut set
        # Strategy: try cutting between random node pairs
        nodes = list(G.nodes)
        if len(nodes) < 2:
            return None
        
        # Try multiple random pairs
        for _ in range(min(10, len(nodes))):
            try:
                s = np.random.choice(nodes)
                t = np.random.choice([n for n in nodes if n != s])
                
                cut_set = nx.minimum_node_cut(G, s, t)
                
                if len(cut_set) <= max_cut_size:
                    return set(cut_set)
            except:
                continue
        
        return None
        
    except:
        return None


def partition_graph_by_cut(G: nx.Graph, cut_set: Set) -> List[Set]:
    """
    Partition graph by removing cut set.
    
    Args:
        G: Input graph
        cut_set: Nodes to remove
        
    Returns:
        List of connected components after cut removal
    """
    G_reduced = G.copy()
    G_reduced.remove_nodes_from(cut_set)
    
    components = list(nx.connected_components(G_reduced))
    return components


def decompose_sequential_cutset(bqm: BinaryQuadraticModel, max_cut_size: int = 5,
                                min_partition_size: int = 50) -> List[Set]:
    """
    Sequential cut-set reduction decomposition.
    
    Strategy:
    1. Find minimum vertex cut
    2. Partition graph by removing cut
    3. Recursively decompose larger components
    4. Cut variables are added to adjacent partitions
    
    Args:
        bqm: Binary Quadratic Model
        max_cut_size: Maximum size of cut sets
        min_partition_size: Minimum partition size before stopping
        
    Returns:
        List of variable sets (partitions)
    """
    # Convert BQM to graph
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    
    # Add edges with weights from quadratic terms
    for (u, v), bias in bqm.quadratic.items():
        G.add_edge(u, v, weight=abs(bias))
    
    if len(G.nodes) <= min_partition_size:
        return [set(G.nodes)]
    
    # Find cut
    cut_set = find_minimum_vertex_cut(G, max_cut_size=max_cut_size)
    
    if cut_set is None:
        # No small cut found - use simple partitioning
        nodes = list(G.nodes)
        partitions = []
        for i in range(0, len(nodes), min_partition_size):
            partitions.append(set(nodes[i:i + min_partition_size]))
        return partitions
    
    # Partition by cut
    components = partition_graph_by_cut(G, cut_set)
    
    # Assign cut nodes to components based on edge weight
    cut_assignments = {node: [] for node in cut_set}
    for cut_node in cut_set:
        for comp_idx, component in enumerate(components):
            weight = sum(G[cut_node][n].get('weight', 1) 
                        for n in component if G.has_edge(cut_node, n))
            if weight > 0:
                cut_assignments[cut_node].append((comp_idx, weight))
    
    # Create partitions with cut nodes distributed
    partitions = []
    for comp_idx, component in enumerate(components):
        partition = set(component)
        
        # Add cut nodes strongly connected to this component
        for cut_node, assignments in cut_assignments.items():
            if assignments:
                # Add to component with strongest connection
                best_comp = max(assignments, key=lambda x: x[1])[0]
                if best_comp == comp_idx:
                    partition.add(cut_node)
        
        # Recursively decompose large partitions
        if len(partition) > min_partition_size * 2:
            # Create sub-BQM
            sub_bqm = BinaryQuadraticModel('BINARY')
            for var in partition:
                if var in bqm.linear:
                    sub_bqm.add_variable(var, bqm.linear[var])
            for (u, v), bias in bqm.quadratic.items():
                if u in partition and v in partition:
                    sub_bqm.add_interaction(u, v, bias)
            
            # Recursive decomposition
            sub_partitions = decompose_sequential_cutset(
                sub_bqm, max_cut_size, min_partition_size
            )
            partitions.extend(sub_partitions)
        else:
            partitions.append(partition)
    
    return partitions


# =============================================================================
# SPATIAL GRID DECOMPOSITION
# =============================================================================

def decompose_spatial_grid(bqm: BinaryQuadraticModel, grid_size: int = 3) -> List[Set]:
    """
    Decompose based on spatial grid partitioning of variables.
    
    Handles both numeric indices (Y_0_0) and string identifiers (Y_Patch1_Beef).
    Creates a grid of partitions based on spatial locality.
    
    Args:
        bqm: Binary Quadratic Model
        grid_size: Number of spatial units per grid cell (default 3 for smaller partitions)
        
    Returns:
        List of variable sets (partitions)
    """
    import re
    variables = list(bqm.variables)
    
    # Extract spatial indices from variable names
    # Handle both Y_0_0 (numeric) and Y_Patch1_Beef (string) formats
    spatial_map = {}
    for var in variables:
        if var.startswith("Y_"):
            parts = var.split("_", 2)  # Split into at most 3 parts
            if len(parts) >= 2:
                spatial_part = parts[1]
                
                # Try numeric index first
                try:
                    spatial_idx = int(spatial_part)
                except ValueError:
                    # Extract numeric part from string like "Patch1" -> 1
                    match = re.search(r'(\d+)', spatial_part)
                    if match:
                        spatial_idx = int(match.group(1))
                    else:
                        # Use hash for completely non-numeric strings
                        spatial_idx = hash(spatial_part) % 10000
                
                if spatial_idx not in spatial_map:
                    spatial_map[spatial_idx] = []
                spatial_map[spatial_idx].append(var)
        elif var.startswith("U_"):
            # U variables (unique food indicators) - group separately
            if -1 not in spatial_map:
                spatial_map[-1] = []
            spatial_map[-1].append(var)
    
    if not spatial_map:
        return [set(variables)]
    
    # Create grid partitions
    partitions = []
    spatial_indices = sorted([k for k in spatial_map.keys() if k >= 0])
    
    # Partition into grid cells
    for i in range(0, len(spatial_indices), grid_size):
        partition = set()
        for idx in spatial_indices[i:i + grid_size]:
            partition.update(spatial_map[idx])
        if partition:
            partitions.append(partition)
    
    # Add U variables to each partition (they connect to all Y vars)
    if -1 in spatial_map:
        u_vars = spatial_map[-1]
        # Distribute U variables across partitions
        for i, u_var in enumerate(u_vars):
            if partitions:
                partitions[i % len(partitions)].add(u_var)
    
    return partitions if len(partitions) > 1 else [set(variables)]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def analyze_decomposition_quality(bqm: BinaryQuadraticModel, partitions: List[Set]) -> Dict:
    """
    Analyze quality metrics of a decomposition.
    
    Returns:
        Dict with quality metrics
    """
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    G.add_edges_from(bqm.quadratic.keys())
    
    # Count inter-partition edges
    inter_edges = 0
    intra_edges = 0
    
    for (u, v) in bqm.quadratic.keys():
        u_part = None
        v_part = None
        
        for i, part in enumerate(partitions):
            if u in part:
                u_part = i
            if v in part:
                v_part = i
        
        if u_part is not None and v_part is not None:
            if u_part == v_part:
                intra_edges += 1
            else:
                inter_edges += 1
    
    # Partition sizes
    sizes = [len(p) for p in partitions]
    
    return {
        'num_partitions': len(partitions),
        'partition_sizes': sizes,
        'min_size': min(sizes) if sizes else 0,
        'max_size': max(sizes) if sizes else 0,
        'mean_size': np.mean(sizes) if sizes else 0,
        'inter_partition_edges': inter_edges,
        'intra_partition_edges': intra_edges,
        'edge_cut_ratio': inter_edges / (inter_edges + intra_edges) if (inter_edges + intra_edges) > 0 else 0
    }


if __name__ == "__main__":
    print("Advanced Decomposition Strategies Module")
    print("=" * 60)
    print("Implements:")
    print("  1. Multilevel Decomposition (ML-QLS)")
    print("  2. Sequential Cut-Set Reduction")
    print("\nUse from comprehensive benchmark script.")
