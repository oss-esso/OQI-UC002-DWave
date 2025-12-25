#!/usr/bin/env python3
"""
Hierarchical Quantum-Classical Solver for Large-Scale Crop Rotation

Three-level optimization:
1. Level 1 (Classical): Food aggregation (27→6) + Spatial decomposition
2. Level 2 (Quantum/SA): Solve QPU-sized subproblems with boundary coordination
3. Level 3 (Classical): Post-processing to specific crops + diversity analysis

For TESTING: Uses SimulatedAnnealingSampler instead of QPU
For PRODUCTION: Set use_qpu=True to use D-Wave QPU

Author: OQI-UC002-DWave Project
Date: 2025-12-12
"""

import os
import sys
import time
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict

warnings.filterwarnings('ignore')

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# ============================================================================
# IMPORTS
# ============================================================================

print("[HierarchicalSolver] Loading modules...")

# Food grouping (local module)
from food_grouping import (
    aggregate_foods_to_families,
    refine_family_solution_to_crops,
    analyze_crop_diversity,
    create_family_rotation_matrix,
    FAMILY_ORDER,
    FAMILY_TO_CROPS,
)

# D-Wave libraries
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler

# Check for QPU availability (optional)
try:
    from dwave.system import DWaveCliqueSampler
    HAS_QPU = True
    print("  ✓ QPU available (DWaveCliqueSampler)")
except ImportError:
    HAS_QPU = False
    print("  ✗ QPU not available (using SimulatedAnnealing only)")

# Scenario loading
from src.scenarios import load_food_data

print("[HierarchicalSolver] Ready!")

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    # Decomposition
    'farms_per_cluster': 10,         # Target farms per quantum subproblem
    'max_cluster_vars': 360,         # Max variables per subproblem (10 farms × 6 families × 3 periods)
    'decomposition_method': 'spatial_grid',  # 'spatial_grid', 'multilevel', 'louvain'
    
    # Quantum solving
    'num_reads': 100,                # SA/QPU reads per subproblem
    'num_iterations': 3,             # Boundary coordination iterations
    'annealing_time': 20,            # QPU annealing time (µs)
    
    # Rotation problem
    'n_periods': 3,                  # Rotation periods
    'rotation_gamma': 1.0,           # Rotation synergy weight (increased for importance)
    'diversity_bonus': 0.5,          # Diversity bonus (increased)
    'one_hot_penalty': 0.5,          # Soft one-hot constraint weight (reduced to avoid dominating)
    
    # Post-processing
    'enable_post_processing': True,
    'crops_per_family': 3,           # Average crops per family in refinement
}

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'hierarchical_results'

# ============================================================================
# LEVEL 1: SPATIAL DECOMPOSITION
# ============================================================================

def decompose_spatial_grid(farm_names: List[str], farms_per_cluster: int = 10) -> List[List[str]]:
    """
    Decompose farms into spatial clusters using grid-based assignment.
    
    Assumes farms are named with indices that roughly correspond to spatial positions.
    Creates clusters of approximately farms_per_cluster farms each.
    """
    n_farms = len(farm_names)
    n_clusters = max(1, (n_farms + farms_per_cluster - 1) // farms_per_cluster)
    
    clusters = []
    for i in range(n_clusters):
        start = i * farms_per_cluster
        end = min(start + farms_per_cluster, n_farms)
        cluster = farm_names[start:end]
        if cluster:
            clusters.append(cluster)
    
    return clusters


def decompose_multilevel(farm_names: List[str], farms_per_cluster: int = 10) -> List[List[str]]:
    """
    Multilevel decomposition: hierarchical grouping.
    
    Creates balanced clusters using recursive bisection.
    """
    n_farms = len(farm_names)
    
    if n_farms <= farms_per_cluster:
        return [farm_names]
    
    # Bisect
    mid = n_farms // 2
    left = farm_names[:mid]
    right = farm_names[mid:]
    
    # Recursively decompose
    left_clusters = decompose_multilevel(left, farms_per_cluster)
    right_clusters = decompose_multilevel(right, farms_per_cluster)
    
    return left_clusters + right_clusters


def get_cluster_neighbors(clusters: List[List[str]], farm_names: List[str]) -> Dict[int, List[int]]:
    """
    Determine which clusters are neighbors (for boundary coordination).
    
    Uses farm name proximity as a proxy for spatial proximity.
    """
    # Build farm to cluster mapping
    farm_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for farm in cluster:
            farm_to_cluster[farm] = i
    
    # Build cluster adjacency (consecutive clusters are neighbors in grid layout)
    neighbors = defaultdict(list)
    n_clusters = len(clusters)
    
    for i in range(n_clusters):
        if i > 0:
            neighbors[i].append(i - 1)
        if i < n_clusters - 1:
            neighbors[i].append(i + 1)
    
    return dict(neighbors)


# ============================================================================
# LEVEL 2: QUANTUM/SA SUBPROBLEM SOLVING
# ============================================================================

def build_cluster_bqm(cluster_farms: List[str], 
                       family_data: Dict,
                       boundary_solutions: Dict = None,
                       config: Dict = None) -> Tuple[BinaryQuadraticModel, Dict]:
    """
    Build BQM for a single cluster subproblem.
    
    Args:
        cluster_farms: Farms in this cluster
        family_data: Family-level problem data (6 families)
        boundary_solutions: Solutions from neighboring clusters (for coordination)
        config: Solver configuration
    
    Returns:
        bqm: Binary Quadratic Model
        var_map: Mapping of (farm, family, period) -> variable name
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    families = family_data['food_names']  # 6 families
    food_benefits = family_data['food_benefits']
    land_availability = family_data['land_availability']
    total_area = family_data['total_area']
    
    n_periods = config.get('n_periods', 3)
    rotation_gamma = config.get('rotation_gamma', 0.25)
    diversity_bonus = config.get('diversity_bonus', 0.15)
    one_hot_penalty = config.get('one_hot_penalty', 3.0)
    
    # Create rotation matrix (deterministic)
    R = create_family_rotation_matrix(seed=42)
    
    # Build BQM
    bqm = BinaryQuadraticModel(vartype='BINARY')
    var_map = {}
    
    # Add variables
    for farm in cluster_farms:
        for c_idx, family in enumerate(families):
            for t in range(1, n_periods + 1):
                var_name = f"Y_{farm}_{family}_t{t}"
                var_map[(farm, family, t)] = var_name
                
                # Linear bias: benefit (negative because BQM minimizes)
                area_frac = land_availability.get(farm, 1.0) / total_area
                benefit = food_benefits.get(family, 0.5)
                linear_bias = -benefit * area_frac  # Negative for maximization
                
                bqm.add_variable(var_name, linear_bias)
    
    # Add quadratic terms
    
    # 1. Rotation synergies (temporal)
    for farm in cluster_farms:
        area_frac = land_availability.get(farm, 1.0) / total_area
        for t in range(2, n_periods + 1):
            for c1_idx, fam1 in enumerate(families):
                for c2_idx, fam2 in enumerate(families):
                    synergy = R[c1_idx, c2_idx]
                    var1 = var_map[(farm, fam1, t-1)]
                    var2 = var_map[(farm, fam2, t)]
                    
                    # Negative synergy (maximize = minimize negative)
                    bqm.add_quadratic(var1, var2, -rotation_gamma * synergy * area_frac)
    
    # 2. Spatial synergies (within cluster)
    # Connect consecutive farms in cluster
    for i in range(len(cluster_farms) - 1):
        f1, f2 = cluster_farms[i], cluster_farms[i + 1]
        for t in range(1, n_periods + 1):
            for c1_idx, fam1 in enumerate(families):
                for c2_idx, fam2 in enumerate(families):
                    synergy = R[c1_idx, c2_idx] * 0.3  # Spatial weaker than temporal
                    var1 = var_map[(f1, fam1, t)]
                    var2 = var_map[(f2, fam2, t)]
                    
                    bqm.add_quadratic(var1, var2, -rotation_gamma * 0.5 * synergy)
    
    # 3. One-hot penalty (soft constraint)
    for farm in cluster_farms:
        for t in range(1, n_periods + 1):
            vars_this_period = [var_map[(farm, fam, t)] for fam in families]
            
            # Penalty for deviation from 1: (sum - 1)^2 = sum^2 - 2*sum + 1
            # sum^2 = sum_i sum_j x_i * x_j = sum_i x_i^2 + sum_{i!=j} x_i * x_j
            # x_i^2 = x_i (binary), so sum^2 = sum_i x_i + 2 * sum_{i<j} x_i * x_j
            
            # Add quadratic penalty for pairs
            for i in range(len(vars_this_period)):
                for j in range(i + 1, len(vars_this_period)):
                    bqm.add_quadratic(vars_this_period[i], vars_this_period[j], 
                                     2 * one_hot_penalty)
            
            # Linear: x_i * (1 - 2) = -x_i (from expanding (sum - 1)^2)
            for var in vars_this_period:
                bqm.add_linear(var, -one_hot_penalty)
    
    # 4. Diversity bonus
    # Approximation: bonus for using each family at least once
    for farm in cluster_farms:
        for family in families:
            # Add small bonus if family used in any period
            for t in range(1, n_periods + 1):
                var = var_map[(farm, family, t)]
                bqm.add_linear(var, -diversity_bonus / n_periods)
    
    # 5. Boundary coordination (if neighboring solutions provided)
    if boundary_solutions:
        boundary_strength = 0.5 * rotation_gamma  # Increased from 0.1 to 0.5
        
        for (neighbor_farm, neighbor_family, t), value in boundary_solutions.items():
            if value != 1:
                continue
            
            # Add strong attraction to consistent solutions at boundaries
            for farm in cluster_farms:
                var = var_map.get((farm, neighbor_family, t))
                if var:
                    bqm.add_linear(var, -boundary_strength)
    
    return bqm, var_map


def solve_cluster_sa(bqm: BinaryQuadraticModel, 
                      var_map: Dict,
                      num_reads: int = 100) -> Tuple[Dict, float, float]:
    """
    Solve cluster BQM using Simulated Annealing (for testing without QPU).
    
    Returns:
        solution: Dict of (farm, family, period) -> value
        energy: Solution energy
        solve_time: Time taken
    """
    start_time = time.time()
    
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    
    solve_time = time.time() - start_time
    
    # Extract best solution
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    
    # Convert to (farm, family, period) format
    solution = {}
    reverse_map = {v: k for k, v in var_map.items()}
    
    for var_name, value in best_sample.items():
        if var_name in reverse_map:
            key = reverse_map[var_name]
            solution[key] = int(value)
    
    return solution, best_energy, solve_time


def solve_cluster_qpu(bqm: BinaryQuadraticModel,
                       var_map: Dict,
                       num_reads: int = 100,
                       annealing_time: int = 20,
                       label: str = None) -> Tuple[Dict, float, float, float]:
    """
    Solve cluster BQM using D-Wave QPU.
    
    Returns:
        solution: Dict of (farm, family, period) -> value
        energy: Solution energy
        wall_time: Total wall clock time
        qpu_time: Actual QPU access time
    """
    if not HAS_QPU:
        raise RuntimeError("QPU not available. Use solve_cluster_sa() instead.")
    
    start_time = time.time()
    
    # Get QPU sampler
    token = os.environ.get('DWAVE_API_TOKEN')
    if token:
        sampler = DWaveCliqueSampler(token=token)
    else:
        sampler = DWaveCliqueSampler()
    
    # Solve
    if label:
        sampleset = sampler.sample(bqm, num_reads=num_reads, annealing_time=annealing_time, label=label)
    else:
        sampleset = sampler.sample(bqm, num_reads=num_reads, annealing_time=annealing_time)
    
    wall_time = time.time() - start_time
    
    # Extract QPU timing - detailed breakdown
    timing = sampleset.info.get('timing', {})
    
    # qpu_access_time: Total time QPU was accessed (includes programming, sampling, readout)
    qpu_access_time = timing.get('qpu_access_time', 0) / 1e6  # Convert µs to s
    
    # qpu_sampling_time: Actual quantum annealing time (the "real" QPU compute time)
    qpu_sampling_time = timing.get('qpu_sampling_time', 0) / 1e6  # Convert µs to s
    
    # qpu_anneal_time_per_sample: Time for single anneal
    qpu_anneal_time = timing.get('qpu_anneal_time_per_sample', annealing_time) / 1e6  # µs to s
    
    # qpu_programming_time: Time to program the QPU
    qpu_programming_time = timing.get('qpu_programming_time', 0) / 1e6  # µs to s
    
    # qpu_readout_time_per_sample: Time to read results
    qpu_readout_time = timing.get('qpu_readout_time_per_sample', 0) / 1e6  # µs to s
    
    # Store detailed timing
    detailed_timing = {
        'wall_time': wall_time,
        'qpu_access_time': qpu_access_time,
        'qpu_sampling_time': qpu_sampling_time,
        'qpu_anneal_time_per_sample': qpu_anneal_time,
        'qpu_programming_time': qpu_programming_time,
        'qpu_readout_time_per_sample': qpu_readout_time,
        'num_reads': num_reads,
    }
    
    # For backward compatibility, qpu_time returns qpu_access_time
    qpu_time = qpu_access_time
    
    # Extract best solution
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    
    # Convert to (farm, family, period) format
    solution = {}
    reverse_map = {v: k for k, v in var_map.items()}
    
    for var_name, value in best_sample.items():
        if var_name in reverse_map:
            key = reverse_map[var_name]
            solution[key] = int(value)
    
    return solution, best_energy, wall_time, qpu_time, detailed_timing


# ============================================================================
# BOUNDARY COORDINATION
# ============================================================================

def coordinate_boundaries(clusters: List[List[str]],
                          cluster_solutions: List[Dict],
                          cluster_neighbors: Dict[int, List[int]]) -> List[Dict]:
    """
    Extract boundary information for inter-cluster coordination.
    
    For each cluster, collect solutions from neighboring clusters
    that can be used as soft constraints in the next iteration.
    """
    boundary_info = []
    
    for i, cluster in enumerate(clusters):
        boundary_solutions = {}
        
        # Collect solutions from neighboring clusters
        for neighbor_idx in cluster_neighbors.get(i, []):
            neighbor_sol = cluster_solutions[neighbor_idx]
            
            # Add neighbor's boundary farms (first and last in their cluster)
            neighbor_cluster = clusters[neighbor_idx]
            boundary_farms = [neighbor_cluster[0], neighbor_cluster[-1]]
            
            for key, value in neighbor_sol.items():
                farm, family, period = key
                if farm in boundary_farms:
                    boundary_solutions[key] = value
        
        boundary_info.append(boundary_solutions)
    
    return boundary_info


# ============================================================================
# OBJECTIVE AND VIOLATION CALCULATION
# ============================================================================

def calculate_family_objective(solution: Dict, family_data: Dict, config: Dict = None) -> float:
    """
    Calculate objective value for a family-level solution.
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    families = family_data['food_names']
    farm_names = family_data['farm_names']
    food_benefits = family_data['food_benefits']
    land_availability = family_data['land_availability']
    total_area = family_data['total_area']
    
    n_periods = config.get('n_periods', 3)
    rotation_gamma = config.get('rotation_gamma', 0.25)
    diversity_bonus = config.get('diversity_bonus', 0.15)
    one_hot_penalty = config.get('one_hot_penalty', 3.0)
    
    R = create_family_rotation_matrix(seed=42)
    
    obj = 0.0
    
    # 1. Base benefit
    for farm in farm_names:
        area_frac = land_availability.get(farm, 1.0) / total_area
        for c_idx, family in enumerate(families):
            for t in range(1, n_periods + 1):
                val = solution.get((farm, family, t), 0)
                if val:
                    obj += food_benefits.get(family, 0.5) * area_frac
    
    # 2. Rotation synergies
    for farm in farm_names:
        area_frac = land_availability.get(farm, 1.0) / total_area
        for t in range(2, n_periods + 1):
            for c1_idx, fam1 in enumerate(families):
                for c2_idx, fam2 in enumerate(families):
                    v1 = solution.get((farm, fam1, t-1), 0)
                    v2 = solution.get((farm, fam2, t), 0)
                    if v1 and v2:
                        obj += rotation_gamma * R[c1_idx, c2_idx] * area_frac
    
    # 3. Diversity bonus (normalized by area)
    for farm in farm_names:
        area_frac = land_availability.get(farm, 1.0) / total_area
        for family in families:
            used = any(solution.get((farm, family, t), 0) for t in range(1, n_periods + 1))
            if used:
                obj += diversity_bonus * area_frac
    
    # 4. One-hot penalty (normalized by area)
    for farm in farm_names:
        area_frac = land_availability.get(farm, 1.0) / total_area
        for t in range(1, n_periods + 1):
            count = sum(solution.get((farm, fam, t), 0) for fam in families)
            if count != 1:
                obj -= one_hot_penalty * (count - 1) ** 2 * area_frac
    
    return obj


def count_violations(solution: Dict, family_data: Dict, config: Dict = None) -> int:
    """
    Count constraint violations in a family-level solution.
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    families = family_data['food_names']
    farm_names = family_data['farm_names']
    n_periods = config.get('n_periods', 3)
    
    violations = 0
    
    for farm in farm_names:
        for t in range(1, n_periods + 1):
            count = sum(solution.get((farm, fam, t), 0) for fam in families)
            if count > 2:  # Max 2 families per period
                violations += count - 2
    
    return violations


# ============================================================================
# MAIN HIERARCHICAL SOLVER
# ============================================================================

def solve_hierarchical(data: Dict,
                        config: Dict = None,
                        use_qpu: bool = False,
                        verbose: bool = True) -> Dict:
    """
    Solve large-scale rotation problem using hierarchical decomposition.
    
    Three-level optimization:
    1. Level 1: Aggregate 27 foods → 6 families, decompose farms into clusters
    2. Level 2: Solve each cluster on QPU/SA with boundary coordination
    3. Level 3: Refine to 27 specific crops, analyze diversity
    
    Args:
        data: Problem data (can be 27 foods or 6 families)
        config: Solver configuration
        use_qpu: If True, use D-Wave QPU; if False, use SimulatedAnnealing
        verbose: Print progress
    
    Returns:
        result: Dict with solution, timing, diversity metrics
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    total_start = time.time()
    
    result = {
        'method': 'hierarchical',
        'use_qpu': use_qpu,
        'config': config,
        'timings': {},
        'levels': {},
    }
    
    if verbose:
        print("="*70)
        print("HIERARCHICAL QUANTUM-CLASSICAL SOLVER")
        print("="*70)
        print(f"Mode: {'QPU' if use_qpu else 'SimulatedAnnealing (testing)'}")
    
    # =========================================================================
    # LEVEL 1: Food Aggregation + Spatial Decomposition
    # =========================================================================
    level1_start = time.time()
    
    if verbose:
        print("\n[LEVEL 1] Food Aggregation + Spatial Decomposition")
    
    # Check if data needs aggregation (27 foods → 6 families)
    n_foods = len(data.get('food_names', []))
    
    if n_foods > 6:
        if verbose:
            print(f"  Aggregating {n_foods} foods → 6 families...")
        family_data = aggregate_foods_to_families(data)
        result['levels']['food_aggregation'] = {
            'original_foods': n_foods,
            'families': 6,
            'reduction_factor': n_foods / 6,
        }
    else:
        family_data = data
        result['levels']['food_aggregation'] = {'original_foods': n_foods, 'families': n_foods}
    
    # Spatial decomposition
    farm_names = family_data['farm_names']
    farms_per_cluster = config.get('farms_per_cluster', 10)
    decomp_method = config.get('decomposition_method', 'spatial_grid')
    
    if decomp_method == 'multilevel':
        clusters = decompose_multilevel(farm_names, farms_per_cluster)
    else:
        clusters = decompose_spatial_grid(farm_names, farms_per_cluster)
    
    cluster_neighbors = get_cluster_neighbors(clusters, farm_names)
    
    n_clusters = len(clusters)
    avg_cluster_size = np.mean([len(c) for c in clusters])
    vars_per_cluster = int(avg_cluster_size * 6 * config.get('n_periods', 3))
    
    if verbose:
        print(f"  Decomposed {len(farm_names)} farms → {n_clusters} clusters")
        print(f"  Avg cluster size: {avg_cluster_size:.1f} farms ({vars_per_cluster} variables)")
    
    result['levels']['decomposition'] = {
        'method': decomp_method,
        'n_clusters': n_clusters,
        'avg_cluster_size': avg_cluster_size,
        'vars_per_cluster': vars_per_cluster,
        'cluster_sizes': [len(c) for c in clusters],
    }
    
    level1_time = time.time() - level1_start
    result['timings']['level1_decomposition'] = level1_time
    
    # =========================================================================
    # LEVEL 2: Quantum/SA Solving with Boundary Coordination
    # =========================================================================
    level2_start = time.time()
    
    if verbose:
        print("\n[LEVEL 2] Quantum Solving with Boundary Coordination")
    
    num_iterations = config.get('num_iterations', 3)
    num_reads = config.get('num_reads', 100)
    
    # Initialize cluster solutions
    cluster_solutions = [{}] * n_clusters
    boundary_info = [{}] * n_clusters
    
    total_solve_time = 0
    total_qpu_time = 0
    total_qpu_sampling_time = 0  # Actual quantum annealing time
    total_qpu_programming_time = 0  # QPU programming time
    iteration_results = []
    
    best_global_solution = None
    best_global_objective = -np.inf
    
    for iteration in range(num_iterations):
        if verbose:
            print(f"\n  Iteration {iteration + 1}/{num_iterations}:")
        
        iter_start = time.time()
        iter_solve_time = 0
        iter_qpu_time = 0
        iter_qpu_sampling_time = 0
        iter_qpu_programming_time = 0
        
        # Update boundary info (except first iteration)
        if iteration > 0:
            boundary_info = coordinate_boundaries(clusters, cluster_solutions, cluster_neighbors)
        
        # Solve each cluster
        new_cluster_solutions = []
        
        for i, cluster in enumerate(clusters):
            # Build BQM for this cluster
            bqm, var_map = build_cluster_bqm(
                cluster, family_data, boundary_info[i], config
            )
            
            # Create problem label
            n_farms_total = len(family_data['farm_names'])
            cluster_label = f"Hierarchical_Rotation_{n_farms_total}farms_cluster{i+1}of{n_clusters}_iter{iteration+1}"
            
            # Solve
            if use_qpu and HAS_QPU:
                sol, energy, wall_time, qpu_time, detailed_timing = solve_cluster_qpu(
                    bqm, var_map, num_reads, config.get('annealing_time', 20), label=cluster_label
                )
                iter_qpu_time += qpu_time
                # Accumulate detailed QPU timing
                iter_qpu_sampling_time += detailed_timing.get('qpu_sampling_time', 0)
                iter_qpu_programming_time += detailed_timing.get('qpu_programming_time', 0)
            else:
                sol, energy, wall_time = solve_cluster_sa(bqm, var_map, num_reads)
            
            iter_solve_time += wall_time
            new_cluster_solutions.append(sol)
            
            if verbose and (i + 1) % max(1, n_clusters // 5) == 0:
                print(f"    Cluster {i+1}/{n_clusters}: vars={len(var_map)}, energy={energy:.3f}")
        
        cluster_solutions = new_cluster_solutions
        
        # Combine into global solution
        combined_solution = {}
        for sol in cluster_solutions:
            combined_solution.update(sol)
        
        # Evaluate
        global_obj = calculate_family_objective(combined_solution, family_data, config)
        global_violations = count_violations(combined_solution, family_data, config)
        
        if global_obj > best_global_objective:
            best_global_objective = global_obj
            best_global_solution = combined_solution.copy()
        
        iter_time = time.time() - iter_start
        total_solve_time += iter_solve_time
        total_qpu_time += iter_qpu_time
        total_qpu_sampling_time += iter_qpu_sampling_time
        total_qpu_programming_time += iter_qpu_programming_time
        
        iteration_results.append({
            'iteration': iteration + 1,
            'objective': global_obj,
            'violations': global_violations,
            'solve_time': iter_solve_time,
            'qpu_time': iter_qpu_time,
            'qpu_sampling_time': iter_qpu_sampling_time,
            'qpu_programming_time': iter_qpu_programming_time,
            'total_time': iter_time,
        })
        
        if verbose:
            print(f"    Combined: obj={global_obj:.4f}, violations={global_violations}, time={iter_time:.2f}s")
    
    level2_time = time.time() - level2_start
    result['timings']['level2_quantum'] = level2_time
    result['timings']['level2_solve'] = total_solve_time
    result['timings']['level2_qpu'] = total_qpu_time
    result['timings']['level2_qpu_sampling'] = total_qpu_sampling_time  # Actual annealing time
    result['timings']['level2_qpu_programming'] = total_qpu_programming_time
    
    result['levels']['quantum_solving'] = {
        'iterations': iteration_results,
        'best_objective': best_global_objective,
        'total_solve_time': total_solve_time,
        'total_qpu_time': total_qpu_time,
        'total_qpu_sampling_time': total_qpu_sampling_time,
        'total_qpu_programming_time': total_qpu_programming_time,
    }
    
    # =========================================================================
    # LEVEL 3: Post-Processing (Family → Crops + Diversity)
    # =========================================================================
    level3_start = time.time()
    
    # Store objective BEFORE post-processing (at family level)
    objective_before_postprocessing = best_global_objective
    
    if verbose:
        print("\n[LEVEL 3] Post-Processing: Family → Specific Crops")
    
    # Get original data for post-processing
    original_data = family_data.get('original_data', data)
    
    # Time refinement
    refine_start = time.time()
    crop_solution = refine_family_solution_to_crops(best_global_solution, original_data)
    refine_time = time.time() - refine_start
    
    # Time diversity analysis
    diversity_start = time.time()
    diversity_stats = analyze_crop_diversity(crop_solution, original_data)
    diversity_time = time.time() - diversity_start
    
    level3_time = time.time() - level3_start
    
    result['timings']['level3_postprocessing'] = level3_time
    result['timings']['level3_refinement'] = refine_time
    result['timings']['level3_diversity'] = diversity_time
    
    result['levels']['post_processing'] = {
        'refinement_time': refine_time,
        'diversity_time': diversity_time,
        'total_time': level3_time,
    }
    
    if verbose:
        print(f"  Refinement: {refine_time*1000:.2f}ms")
        print(f"  Diversity analysis: {diversity_time*1000:.2f}ms")
        print(f"  Unique crops: {diversity_stats['total_unique_crops']}/{diversity_stats['max_possible_crops']}")
        print(f"  Shannon diversity: {diversity_stats['shannon_diversity']:.3f}")
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    total_time = time.time() - total_start
    
    result['timings']['total'] = total_time
    result['wall_time'] = total_time  # For compatibility with statistical_comparison_test
    result['qpu_time'] = total_qpu_time  # Total QPU access time (includes embedding, programming, readout)
    result['qpu_sampling_time'] = total_qpu_sampling_time  # Actual quantum annealing time only
    result['qpu_programming_time'] = total_qpu_programming_time  # QPU programming time
    result['family_solution'] = best_global_solution
    result['crop_solution'] = crop_solution
    result['objective'] = best_global_objective
    result['objective_before_postprocessing'] = objective_before_postprocessing  # NEW: Track pre-PP objective
    result['violations'] = count_violations(best_global_solution, family_data, config)
    result['diversity_stats'] = diversity_stats
    result['success'] = True
    
    if verbose:
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"Total time: {total_time:.2f}s")
        print(f"  Level 1 (decomposition): {level1_time:.2f}s")
        print(f"  Level 2 (quantum solve): {level2_time:.2f}s")
        print(f"    - QPU access time: {total_qpu_time:.3f}s")
        print(f"    - QPU sampling (annealing): {total_qpu_sampling_time*1000:.2f}ms")
        print(f"    - QPU programming: {total_qpu_programming_time*1000:.2f}ms")
        print(f"  Level 3 (post-process): {level3_time:.4f}s")
        print(f"\nObjective: {best_global_objective:.4f}")
        print(f"Violations: {result['violations']}")
        print(f"Unique crops: {diversity_stats['total_unique_crops']} / 27")
        print(f"Shannon diversity: {diversity_stats['shannon_diversity']:.3f} / {diversity_stats['max_shannon']:.3f}")
        print("="*70)
    
    return result


# ============================================================================
# TESTING
# ============================================================================

def test_on_rotation_scenario(scenario_name: str, use_qpu: bool = False):
    """
    Test hierarchical solver on a rotation scenario.
    """
    print(f"\nTesting on scenario: {scenario_name}")
    print("-" * 50)
    
    # Load scenario
    farms, foods, food_groups, config = load_food_data(scenario_name)
    
    # Build data dict
    params = config.get('parameters', {})
    weights = params.get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15
    })
    
    land_availability = params.get('land_availability', {})
    farm_names = list(land_availability.keys())
    total_area = sum(land_availability.values())
    
    food_names = list(foods.keys())
    food_benefits = {}
    for food in food_names:
        food_data = foods.get(food, {})
        benefit = sum(food_data.get(attr, 0.5) * w for attr, w in weights.items())
        food_benefits[food] = benefit
    
    data = {
        'foods': foods,
        'food_names': food_names,
        'food_groups': food_groups,
        'food_benefits': food_benefits,
        'weights': weights,
        'land_availability': land_availability,
        'farm_names': farm_names,
        'total_area': total_area,
        'n_farms': len(farm_names),
        'n_foods': len(food_names),
        'config': config,
    }
    
    print(f"Loaded: {len(farm_names)} farms × {len(food_names)} foods")
    
    # Configure solver
    solver_config = DEFAULT_CONFIG.copy()
    solver_config['farms_per_cluster'] = min(10, max(3, len(farm_names) // 5))
    
    # Solve
    result = solve_hierarchical(data, solver_config, use_qpu=use_qpu, verbose=True)
    
    return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hierarchical Quantum-Classical Solver')
    parser.add_argument('--scenario', type=str, default='rotation_medium_100',
                       help='Scenario name (default: rotation_medium_100)')
    parser.add_argument('--qpu', action='store_true',
                       help='Use real QPU (default: use SimulatedAnnealing)')
    parser.add_argument('--farms-per-cluster', type=int, default=10,
                       help='Farms per cluster (default: 10)')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Boundary coordination iterations (default: 3)')
    parser.add_argument('--reads', type=int, default=100,
                       help='SA/QPU reads per subproblem (default: 100)')
    
    args = parser.parse_args()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config['farms_per_cluster'] = args.farms_per_cluster
    config['num_iterations'] = args.iterations
    config['num_reads'] = args.reads
    
    # Run test
    result = test_on_rotation_scenario(args.scenario, use_qpu=args.qpu)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = OUTPUT_DIR / f"hierarchical_{args.scenario}_{timestamp}.json"
    
    # Convert for JSON
    import json
    
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(result), f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
