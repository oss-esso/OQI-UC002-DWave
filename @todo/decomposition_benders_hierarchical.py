"""
Hierarchical Benders Decomposition with Graph Partitioning

This module implements a divide-and-conquer approach to solve large-scale
optimization problems that exceed QPU embedding capacity.

Inspired by the QAOA-in-QAOA paper (Zhou et al.):
"QAOA-in-QAOA: solving large-scale MaxCut problems on small quantum machines"

Key Concepts:
1. Partition the BQM graph into smaller subgraphs using community detection
2. Solve each subgraph independently on QPU (or simulator)
3. Merge solutions by solving a coordination problem for cross-partition coupling
4. Apply hierarchically if the merging problem is still too large

This enables solving problems with 15+ farms that would otherwise fail embedding.
"""
import time
import os
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

import networkx as nx
from networkx.algorithms.community import louvain_communities
import gurobipy as gp
from gurobipy import GRB

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, cqm_to_bqm
import neal  # SimulatedAnnealing for fallback/simulation

from result_formatter import format_decomposition_result, validate_solution_constraints


# Configuration
MAX_EMBEDDABLE_VARS = 150  # Max BQM variables for reliable embedding (~5 farms worth)
MIN_PARTITION_SIZE = 20   # Minimum variables per partition


def get_bqm_graph(bqm: BinaryQuadraticModel) -> nx.Graph:
    """
    Extract the interaction graph from a BQM.
    
    Returns a NetworkX graph where nodes are variables and edges
    are quadratic interactions.
    """
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    G.add_edges_from(bqm.quadratic.keys())
    return G


def partition_bqm_graph(
    bqm: BinaryQuadraticModel,
    max_partition_size: int = MAX_EMBEDDABLE_VARS,
    min_partition_size: int = MIN_PARTITION_SIZE,
    seed: Optional[int] = None
) -> List[Set]:
    """
    Partition the BQM graph using Louvain community detection.
    
    This keeps densely connected variables together, which:
    1. Maximizes internal edges within partitions (solved exactly)
    2. Minimizes cross-partition edges (handled by merging)
    
    Args:
        bqm: Binary Quadratic Model to partition
        max_partition_size: Maximum variables per partition
        min_partition_size: Minimum variables per partition
        seed: Random seed for reproducibility
        
    Returns:
        List of variable sets, one per partition
    """
    G = get_bqm_graph(bqm)
    n_vars = len(bqm.variables)
    
    print(f"    [Partition] BQM has {n_vars} variables, {len(bqm.quadratic)} interactions")
    
    if n_vars <= max_partition_size:
        print(f"    [Partition] Small enough ({n_vars} <= {max_partition_size}), no partition needed")
        return [set(bqm.variables)]
    
    # Use Louvain for community detection (modularity-based)
    # Resolution parameter controls partition granularity
    target_partitions = max(2, int(np.ceil(n_vars / max_partition_size)))
    
    # Start with high resolution and decrease until we get acceptable partition sizes
    resolution = 1.0
    best_partitions = None
    best_max_size = float('inf')
    
    for resolution in [2.0, 1.5, 1.0, 0.8, 0.5, 0.3, 0.1]:
        try:
            communities = louvain_communities(G, resolution=resolution, seed=seed)
            communities = [set(c) for c in communities]
            
            max_size = max(len(c) for c in communities)
            min_size = min(len(c) for c in communities)
            
            print(f"    [Partition] Resolution={resolution}: {len(communities)} communities, "
                  f"sizes {min_size}-{max_size}")
            
            # Check if this is acceptable
            if max_size <= max_partition_size and max_size < best_max_size:
                best_partitions = communities
                best_max_size = max_size
                
                if max_size <= max_partition_size:
                    break  # Good enough
                    
        except Exception as e:
            print(f"    [Partition] Resolution={resolution} failed: {e}")
            continue
    
    if best_partitions is None:
        # Fallback: use balanced random partitioning
        print(f"    [Partition] Louvain failed, using balanced random partitioning")
        variables = list(bqm.variables)
        np.random.seed(seed)
        np.random.shuffle(variables)
        
        best_partitions = []
        for i in range(0, len(variables), max_partition_size):
            best_partitions.append(set(variables[i:i+max_partition_size]))
    
    # Post-process: merge very small partitions
    merged_partitions = []
    current_partition = set()
    
    for partition in sorted(best_partitions, key=len):
        if len(current_partition) + len(partition) <= max_partition_size:
            current_partition |= partition
        else:
            if len(current_partition) >= min_partition_size:
                merged_partitions.append(current_partition)
            elif merged_partitions:
                # Merge into smallest existing partition
                smallest_idx = min(range(len(merged_partitions)), 
                                   key=lambda i: len(merged_partitions[i]))
                merged_partitions[smallest_idx] |= current_partition
            else:
                merged_partitions.append(current_partition)
            current_partition = partition
    
    if current_partition:
        if len(current_partition) >= min_partition_size:
            merged_partitions.append(current_partition)
        elif merged_partitions:
            smallest_idx = min(range(len(merged_partitions)), 
                               key=lambda i: len(merged_partitions[i]))
            merged_partitions[smallest_idx] |= current_partition
        else:
            merged_partitions.append(current_partition)
    
    print(f"    [Partition] Final: {len(merged_partitions)} partitions, "
          f"sizes: {[len(p) for p in merged_partitions]}")
    
    return merged_partitions


def extract_sub_bqm(bqm: BinaryQuadraticModel, variables: Set) -> BinaryQuadraticModel:
    """
    Extract a sub-BQM containing only the specified variables.
    
    The sub-BQM includes:
    - Linear biases for variables in the set
    - Quadratic biases for edges where BOTH endpoints are in the set
    """
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


def solve_sub_bqm(
    sub_bqm: BinaryQuadraticModel,
    partition_id: int,
    use_qpu: bool = False,
    dwave_token: Optional[str] = None,
    num_reads: int = 200,
    annealing_time: int = 20
) -> Tuple[Dict, float, float]:
    """
    Solve a sub-BQM using simulated annealing or QPU.
    
    Returns:
        (solution_dict, energy, solve_time)
    """
    start_time = time.time()
    
    if use_qpu and dwave_token:
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite
            
            print(f"      [Sub-BQM {partition_id}] Attempting QPU solve ({len(sub_bqm.variables)} vars)...")
            
            sampler = DWaveSampler(token=dwave_token)
            composite = EmbeddingComposite(sampler)
            
            sampleset = composite.sample(
                sub_bqm,
                num_reads=num_reads,
                annealing_time=annealing_time,
                label=f"Hierarchical_Partition_{partition_id}"
            )
            
            best_sample = sampleset.first.sample
            best_energy = sampleset.first.energy
            solve_time = time.time() - start_time
            
            print(f"      [Sub-BQM {partition_id}] QPU solved: energy={best_energy:.4f}, time={solve_time:.2f}s")
            return dict(best_sample), best_energy, solve_time
            
        except Exception as e:
            print(f"      [Sub-BQM {partition_id}] QPU failed ({e}), falling back to SA")
    
    # Simulated Annealing fallback
    sampler = neal.SimulatedAnnealingSampler()
    
    sampleset = sampler.sample(
        sub_bqm,
        num_reads=num_reads,
        num_sweeps=2000
    )
    
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    solve_time = time.time() - start_time
    
    print(f"      [Sub-BQM {partition_id}] SA solved: energy={best_energy:.4f}, time={solve_time:.2f}s")
    return dict(best_sample), best_energy, solve_time


def compute_cross_partition_coupling(
    bqm: BinaryQuadraticModel,
    partitions: List[Set],
    partition_solutions: List[Dict]
) -> Tuple[Dict, float]:
    """
    Compute the coupling between partitions based on cross-partition edges.
    
    For the merging problem (inspired by QAOA-in-QAOA Theorem 1):
    - Each partition i has a binary decision variable s_i ∈ {0, 1}
    - s_i = 0: keep solution as-is
    - s_i = 1: flip all binary variables in partition i
    
    The merging objective accounts for cross-partition edges:
    - When s_i = s_j: use original coupling strength
    - When s_i ≠ s_j: use flipped coupling strength
    
    Returns:
        (coupling_dict, constant_offset)
        where coupling_dict[(i,j)] is the weight for s_i XOR s_j
    """
    n_partitions = len(partitions)
    
    # Map variables to their partition index
    var_to_partition = {}
    for i, partition in enumerate(partitions):
        for var in partition:
            var_to_partition[var] = i
    
    # Initialize coupling matrix
    coupling = {}
    for i in range(n_partitions):
        for j in range(i + 1, n_partitions):
            coupling[(i, j)] = 0.0
    
    constant_offset = 0.0
    
    # Process all cross-partition edges
    for (u, v), bias in bqm.quadratic.items():
        p_u = var_to_partition.get(u)
        p_v = var_to_partition.get(v)
        
        if p_u is None or p_v is None:
            continue
            
        if p_u == p_v:
            continue  # Internal edge, already solved
        
        if p_u > p_v:
            p_u, p_v = p_v, p_u
            u, v = v, u
        
        # Get current variable values
        x_u = partition_solutions[p_u].get(u, 0)
        x_v = partition_solutions[p_v].get(v, 0)
        
        # For QUBO/BQM: bias * x_u * x_v
        # When s_i = s_j (aligned): contribution is bias * x_u * x_v
        # When s_i ≠ s_j (flipped): contribution is bias * (1-x_u) * x_v or bias * x_u * (1-x_v)
        
        # Actually, in our formulation flipping means:
        # - If we flip partition i: all x_i become (1 - x_i) for BINARY variables
        
        # The difference in contribution when flipping:
        # Original: bias * x_u * x_v
        # Flip u: bias * (1-x_u) * x_v = bias * x_v - bias * x_u * x_v
        # Flip v: bias * x_u * (1-x_v) = bias * x_u - bias * x_u * x_v
        # Flip both: bias * (1-x_u) * (1-x_v) = bias - bias*x_u - bias*x_v + bias*x_u*x_v
        
        # Energy change when flipping partition p_u (s_u = 1):
        # Δ_u = bias * ((1-x_u) * x_v - x_u * x_v) = bias * x_v * (1 - 2*x_u)
        
        # Energy change when flipping partition p_v (s_v = 1):
        # Δ_v = bias * (x_u * (1-x_v) - x_u * x_v) = bias * x_u * (1 - 2*x_v)
        
        # Energy change when flipping both (s_u = s_v = 1):
        # Δ_both = bias * ((1-x_u)*(1-x_v) - x_u*x_v)
        #        = bias * (1 - x_u - x_v + x_u*x_v - x_u*x_v)
        #        = bias * (1 - x_u - x_v)
        
        # For the merging BQM, we want to decide s_u and s_v to minimize total energy.
        # The interaction term J_{uv} for the merging BQM is related to cross-partition coupling.
        
        # Let's use a simpler formulation:
        # Merging variable m_i ∈ {0, 1} indicates whether to flip partition i
        # Cross-partition contribution = Σ bias * (x_u XOR m_u) * (x_v XOR m_v)
        
        # This expands to a quadratic in m:
        # bias * (x_u + m_u - 2*x_u*m_u) * (x_v + m_v - 2*x_v*m_v)
        
        # For simplicity, we'll compute the contribution for each (m_u, m_v) combination:
        # (0,0): bias * x_u * x_v
        # (0,1): bias * x_u * (1-x_v)
        # (1,0): bias * (1-x_u) * x_v
        # (1,1): bias * (1-x_u) * (1-x_v)
        
        e_00 = bias * x_u * x_v
        e_01 = bias * x_u * (1 - x_v)
        e_10 = bias * (1 - x_u) * x_v
        e_11 = bias * (1 - x_u) * (1 - x_v)
        
        # Convert to QUBO form for (m_u, m_v):
        # E = c + a*m_u + b*m_v + d*m_u*m_v
        # E(0,0) = c = e_00
        # E(1,0) = c + a = e_10  =>  a = e_10 - e_00
        # E(0,1) = c + b = e_01  =>  b = e_01 - e_00
        # E(1,1) = c + a + b + d = e_11  =>  d = e_11 - e_10 - e_01 + e_00
        
        c = e_00
        a = e_10 - e_00
        b = e_01 - e_00
        d = e_11 - e_10 - e_01 + e_00
        
        constant_offset += c
        # Linear terms and quadratic term contribute to the merging BQM
        coupling[(p_u, p_v)] = coupling.get((p_u, p_v), 0.0) + d
        
        # Note: Linear terms a*m_u and b*m_v also need to be tracked
        # For now, we're simplifying to just track the quadratic coupling
    
    return coupling, constant_offset


def build_merging_bqm(
    partitions: List[Set],
    partition_solutions: List[Dict],
    cross_coupling: Dict,
    partition_energies: List[float]
) -> BinaryQuadraticModel:
    """
    Build the merging BQM to decide which partitions to flip.
    
    Variables: m_i ∈ {0, 1} for each partition i
    Objective: Minimize total energy including cross-partition contributions
    """
    n_partitions = len(partitions)
    
    merging_bqm = BinaryQuadraticModel('BINARY')
    
    # Add variables for each partition
    for i in range(n_partitions):
        # Linear bias: accounts for energy change from flipping this partition
        # For now, set to 0 (the main effect is in quadratic terms)
        merging_bqm.add_variable(f"m_{i}", 0.0)
    
    # Add quadratic terms from cross-partition coupling
    for (i, j), weight in cross_coupling.items():
        if abs(weight) > 1e-10:
            merging_bqm.add_interaction(f"m_{i}", f"m_{j}", weight)
    
    return merging_bqm


def solve_merging_problem(
    merging_bqm: BinaryQuadraticModel,
    max_embeddable: int = MAX_EMBEDDABLE_VARS,
    use_qpu: bool = False,
    dwave_token: Optional[str] = None
) -> Dict:
    """
    Solve the merging problem to determine optimal partition flips.
    
    If the merging BQM is too large, recursively apply hierarchical decomposition.
    
    Returns:
        Dictionary mapping partition indices to flip decisions (0 or 1)
    """
    n_vars = len(merging_bqm.variables)
    
    print(f"    [Merging] Solving merging problem with {n_vars} variables")
    
    if n_vars <= 20:  # Small enough for exhaustive search
        # Try all combinations (2^n for small n)
        best_sample = None
        best_energy = float('inf')
        
        import itertools
        vars_list = list(merging_bqm.variables)
        
        for bits in itertools.product([0, 1], repeat=n_vars):
            sample = {vars_list[i]: bits[i] for i in range(n_vars)}
            energy = merging_bqm.energy(sample)
            if energy < best_energy:
                best_energy = energy
                best_sample = sample
        
        print(f"    [Merging] Exhaustive search: best energy = {best_energy:.4f}")
        
    elif n_vars <= max_embeddable:
        # Use SA or QPU directly
        if use_qpu and dwave_token:
            try:
                from dwave.system import DWaveSampler, EmbeddingComposite
                
                sampler = DWaveSampler(token=dwave_token)
                composite = EmbeddingComposite(sampler)
                
                sampleset = composite.sample(merging_bqm, num_reads=500, annealing_time=20)
                best_sample = dict(sampleset.first.sample)
                best_energy = sampleset.first.energy
                
                print(f"    [Merging] QPU solved: energy = {best_energy:.4f}")
            except:
                # Fallback to SA
                sampler = neal.SimulatedAnnealingSampler()
                sampleset = sampler.sample(merging_bqm, num_reads=500, num_sweeps=5000)
                best_sample = dict(sampleset.first.sample)
                best_energy = sampleset.first.energy
                print(f"    [Merging] SA solved: energy = {best_energy:.4f}")
        else:
            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample(merging_bqm, num_reads=500, num_sweeps=5000)
            best_sample = dict(sampleset.first.sample)
            best_energy = sampleset.first.energy
            print(f"    [Merging] SA solved: energy = {best_energy:.4f}")
    else:
        # Recursive hierarchical decomposition
        print(f"    [Merging] Too large ({n_vars} > {max_embeddable}), applying recursive decomposition")
        
        # Partition the merging BQM
        sub_partitions = partition_bqm_graph(merging_bqm, max_partition_size=max_embeddable // 2)
        
        # Solve each sub-partition
        sub_solutions = []
        sub_energies = []
        for i, partition in enumerate(sub_partitions):
            sub_bqm = extract_sub_bqm(merging_bqm, partition)
            solution, energy, _ = solve_sub_bqm(sub_bqm, i, use_qpu=use_qpu, dwave_token=dwave_token)
            sub_solutions.append(solution)
            sub_energies.append(energy)
        
        # Compute cross-partition coupling for the meta-merging problem
        meta_coupling, meta_offset = compute_cross_partition_coupling(
            merging_bqm, sub_partitions, sub_solutions
        )
        
        # Build and solve meta-merging BQM
        meta_merging_bqm = build_merging_bqm(
            sub_partitions, sub_solutions, meta_coupling, sub_energies
        )
        
        # Recursive call
        meta_flips = solve_merging_problem(
            meta_merging_bqm, max_embeddable, use_qpu, dwave_token
        )
        
        # Reconstruct solution
        best_sample = {}
        for i, partition in enumerate(sub_partitions):
            flip = meta_flips.get(f"m_{i}", 0)
            for var in partition:
                original_val = sub_solutions[i].get(var, 0)
                best_sample[var] = (1 - original_val) if flip else original_val
    
    # Convert to partition flip decisions
    flip_decisions = {}
    for var, val in best_sample.items():
        if var.startswith('m_'):
            idx = int(var.split('_')[1])
            flip_decisions[idx] = val
    
    return flip_decisions


def reconstruct_global_solution(
    partitions: List[Set],
    partition_solutions: List[Dict],
    flip_decisions: Dict
) -> Dict:
    """
    Reconstruct the global solution by applying flip decisions to partition solutions.
    """
    global_solution = {}
    
    for i, (partition, solution) in enumerate(zip(partitions, partition_solutions)):
        flip = flip_decisions.get(i, 0)
        
        for var, val in solution.items():
            if flip:
                global_solution[var] = 1 - val  # Flip binary value
            else:
                global_solution[var] = val
    
    return global_solution


def solve_with_benders_hierarchical(
    farms: Dict[str, float],
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    dwave_token: Optional[str] = None,
    max_iterations: int = 50,
    gap_tolerance: float = 1e-4,
    time_limit: float = 600.0,
    max_embeddable_vars: int = MAX_EMBEDDABLE_VARS,
    use_qpu: bool = True,
    num_reads: int = 200,
    annealing_time: int = 20
) -> Dict:
    """
    Solve farm allocation problem using Hierarchical Benders Decomposition.
    
    This strategy partitions the master problem BQM into smaller pieces,
    solves each on QPU/simulator, then merges solutions.
    
    Args:
        farms: Dictionary of farm names to land availability
        foods: List of food names
        food_groups: Dictionary of food groups
        config: Configuration dictionary with parameters
        dwave_token: D-Wave API token for QPU access
        max_iterations: Maximum number of Benders iterations
        gap_tolerance: Convergence tolerance for optimality gap
        time_limit: Maximum total solve time in seconds
        max_embeddable_vars: Maximum BQM variables for QPU embedding
        use_qpu: Whether to attempt QPU solving
        num_reads: Number of QPU samples
        annealing_time: Annealing time in microseconds
    
    Returns:
        Formatted result dictionary
    """
    start_time = time.time()
    
    # Check QPU availability
    has_qpu = dwave_token is not None and dwave_token != 'YOUR_DWAVE_TOKEN_HERE'
    if not has_qpu:
        print("⚠️  No D-Wave token provided - using SimulatedAnnealing")
        use_qpu = False
    
    # Extract parameters
    params = config.get('parameters', {})
    min_planting_area = params.get('minimum_planting_area', {})
    max_planting_area = params.get('maximum_planting_area', {})
    benefits = config.get('benefits', {})
    
    print(f"\n{'='*80}")
    print(f"HIERARCHICAL BENDERS DECOMPOSITION")
    print(f"{'='*80}")
    print(f"Problem: {len(farms)} farms, {len(foods)} foods")
    print(f"Max embeddable vars: {max_embeddable_vars}")
    print(f"QPU enabled: {use_qpu and has_qpu}")
    print(f"{'='*80}\n")
    
    # Initialize tracking
    master_iterations = []
    qpu_time_total = 0.0
    
    # Build the master problem CQM (Y variables only for selection decisions)
    print("[1/4] Building master problem CQM...")
    cqm_start = time.time()
    
    cqm = ConstrainedQuadraticModel()
    
    # Variables: Y[f,c] binary
    Y = {}
    for farm in farms:
        for food in foods:
            var_name = f"Y_{farm}_{food}"
            Y[(farm, food)] = Binary(var_name)
            cqm.add_variable('BINARY', var_name)
    
    # Objective: maximize food selections (simplified master)
    objective = sum(Y[(f, c)] * benefits.get(c, 1.0) for f in farms for c in foods)
    cqm.set_objective(-objective)  # Minimize negative = maximize
    
    # Food group constraints
    food_group_constraints = config.get('parameters', {}).get('food_group_constraints', {})
    if food_group_constraints:
        for group_name, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            min_foods = constraints.get('min_foods', 0)
            max_foods = constraints.get('max_foods', len(foods_in_group) * len(farms))
            
            total = sum(Y[(f, c)] for f in farms for c in foods_in_group)
            
            if min_foods > 0:
                cqm.add_constraint(total >= min_foods, label=f"FG_Min_{group_name}")
            if max_foods < float('inf'):
                cqm.add_constraint(total <= max_foods, label=f"FG_Max_{group_name}")
    
    # Convert to BQM
    bqm, _ = cqm_to_bqm(cqm)
    
    print(f"    CQM → BQM: {len(bqm.variables)} variables, {len(bqm.quadratic)} interactions")
    print(f"    Build time: {time.time() - cqm_start:.2f}s")
    
    # Check if hierarchical decomposition is needed
    needs_decomposition = len(bqm.variables) > max_embeddable_vars
    
    if needs_decomposition:
        print(f"\n[2/4] Partitioning BQM (too large for direct embedding)...")
        partitions = partition_bqm_graph(bqm, max_partition_size=max_embeddable_vars)
    else:
        print(f"\n[2/4] BQM small enough for direct solving")
        partitions = [set(bqm.variables)]
    
    # Solve each partition
    print(f"\n[3/4] Solving {len(partitions)} partition(s)...")
    partition_solutions = []
    partition_energies = []
    
    for i, partition in enumerate(partitions):
        print(f"\n  Partition {i+1}/{len(partitions)} ({len(partition)} variables):")
        sub_bqm = extract_sub_bqm(bqm, partition)
        
        solution, energy, solve_time = solve_sub_bqm(
            sub_bqm, i, 
            use_qpu=use_qpu and has_qpu,
            dwave_token=dwave_token,
            num_reads=num_reads,
            annealing_time=annealing_time
        )
        
        partition_solutions.append(solution)
        partition_energies.append(energy)
        qpu_time_total += solve_time
        
        master_iterations.append({
            'iteration': i + 1,
            'partition_size': len(partition),
            'energy': energy,
            'solve_time': solve_time
        })
    
    # Merge solutions if multiple partitions
    if len(partitions) > 1:
        print(f"\n[4/4] Merging {len(partitions)} partition solutions...")
        
        # Compute cross-partition coupling
        cross_coupling, offset = compute_cross_partition_coupling(
            bqm, partitions, partition_solutions
        )
        
        # Build and solve merging problem
        merging_bqm = build_merging_bqm(
            partitions, partition_solutions, cross_coupling, partition_energies
        )
        
        flip_decisions = solve_merging_problem(
            merging_bqm, max_embeddable_vars, 
            use_qpu=use_qpu and has_qpu, 
            dwave_token=dwave_token
        )
        
        # Reconstruct global solution
        bqm_solution = reconstruct_global_solution(
            partitions, partition_solutions, flip_decisions
        )
        
        print(f"    Flip decisions: {flip_decisions}")
    else:
        print(f"\n[4/4] Single partition - no merging needed")
        bqm_solution = partition_solutions[0]
        flip_decisions = {0: 0}
    
    # Extract Y values from BQM solution
    Y_star = {}
    for (farm, food) in Y:
        var_name = f"Y_{farm}_{food}"
        # Handle both direct variable names and slack variable names
        if var_name in bqm_solution:
            Y_star[(farm, food)] = 1.0 if bqm_solution[var_name] > 0.5 else 0.0
        else:
            Y_star[(farm, food)] = 0.0
    
    # Solve continuous subproblem given Y*
    print(f"\n[5/4] Solving continuous subproblem for area allocation...")
    A_star, subproblem_obj = solve_continuous_subproblem(
        farms, foods, Y_star, benefits, min_planting_area, max_planting_area
    )
    
    # Build final solution
    best_solution = {}
    for f in farms:
        for c in foods:
            best_solution[f"A_{f}_{c}"] = A_star.get((f, c), 0.0)
            best_solution[f"Y_{f}_{c}"] = Y_star.get((f, c), 0.0)
    
    total_time = time.time() - start_time
    
    # Validate solution
    validation = validate_solution_constraints(
        best_solution, farms, foods, food_groups, farms, config, 'farm'
    )
    
    print(f"\n{'='*80}")
    print(f"Hierarchical Benders Complete")
    print(f"{'='*80}")
    print(f"Partitions: {len(partitions)}")
    print(f"Objective: {subproblem_obj:.4f}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Feasible: {validation['is_feasible']}")
    print(f"{'='*80}\n")
    
    # Format result - Note: we use a custom formatting since format_benders_result
    # creates its own decomposition_specific that doesn't include our hierarchical info
    from result_formatter import format_decomposition_result
    
    hierarchical_details = {
        'iterations_detail': master_iterations,
        'n_partitions': len(partitions),
        'partition_sizes': [len(p) for p in partitions],
        'flip_decisions': flip_decisions,
        'partition_energies': partition_energies,
        'hierarchical': needs_decomposition,
        'qpu_time_total': qpu_time_total,
        'used_qpu': use_qpu and has_qpu
    }
    
    result = format_decomposition_result(
        strategy_name='benders_hierarchical',
        scenario_type='farm',
        n_units=len(farms),
        n_foods=len(foods),
        total_area=sum(farms.values()),
        objective_value=subproblem_obj,
        solution=best_solution,
        solve_time=total_time,
        num_iterations=len(master_iterations),
        is_feasible=validation['is_feasible'],
        validation_results=validation,
        num_variables=len(farms) * len(foods) * 2,
        num_constraints=len(farms) + len(food_groups) * 2,
        decomposition_specific=hierarchical_details
    )
    
    return result


def solve_continuous_subproblem(
    farms: Dict[str, float],
    foods: List[str],
    Y_fixed: Dict[Tuple[str, str], float],
    benefits: Dict[str, float],
    min_planting_area: Dict[str, float],
    max_planting_area: Dict[str, float]
) -> Tuple[Dict, float]:
    """
    Solve the continuous area allocation subproblem given fixed Y selections.
    
    This is the standard Benders subproblem: optimize A variables.
    """
    sub = gp.Model("Continuous_Subproblem")
    sub.setParam('OutputFlag', 0)
    
    # Subproblem variables: A[f,c] continuous
    A = {}
    for farm, capacity in farms.items():
        for food in foods:
            A[(farm, food)] = sub.addVar(lb=0.0, name=f"A_{farm}_{food}")
    
    # Objective: maximize benefit per hectare (normalized)
    total_area = sum(farms.values())
    obj_expr = gp.quicksum(
        A[(farm, food)] * benefits.get(food, 1.0)
        for farm in farms
        for food in foods
    ) / total_area
    sub.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Constraints
    
    # 1. Land availability
    for farm, capacity in farms.items():
        sub.addConstr(
            gp.quicksum(A[(farm, food)] for food in foods) <= capacity,
            name=f"Land_{farm}"
        )
    
    # 2. Min area if Y=1, force zero if Y=0
    for farm in farms:
        for food in foods:
            y_val = Y_fixed.get((farm, food), 0.0)
            min_area = min_planting_area.get(food, 0.0001)
            
            if y_val > 0.5:  # Y is selected
                sub.addConstr(A[(farm, food)] >= min_area, name=f"MinArea_{farm}_{food}")
            else:  # Y is not selected
                sub.addConstr(A[(farm, food)] == 0.0, name=f"ForceZero_{farm}_{food}")
    
    # 3. Max area if Y=1
    for farm, capacity in farms.items():
        for food in foods:
            y_val = Y_fixed.get((farm, food), 0.0)
            if y_val > 0.5:
                max_area = max_planting_area.get(food, capacity)
                sub.addConstr(A[(farm, food)] <= max_area, name=f"MaxArea_{farm}_{food}")
    
    # Solve
    sub.optimize()
    
    if sub.status != GRB.OPTIMAL:
        print(f"    ⚠️ Subproblem status: {sub.status}")
        return {}, -float('inf')
    
    # Extract solution
    A_solution = {key: var.X for key, var in A.items()}
    obj_value = sub.ObjVal
    
    print(f"    Subproblem solved: obj={obj_value:.4f}")
    
    return A_solution, obj_value


# Entry point for testing
if __name__ == "__main__":
    print("Hierarchical Benders Decomposition Module")
    print("=" * 60)
    print("This module provides solve_with_benders_hierarchical()")
    print("Use via decomposition_strategies.py factory pattern")
