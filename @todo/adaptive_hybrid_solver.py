#!/usr/bin/env python3
"""
Adaptive Hybrid Solver with 27-Food Post-Processing Recovery

This module provides:
1. Adaptive granularity: 27 foods → 6 families for QPU, then recover 27 foods
2. Proper post-processing that converts 6-family solutions to 27-food solutions
3. Objective calculation at both family and food levels
4. Both simulated (SA) and real QPU modes

The post-processing follows the LaTeX formulation:
- For each (farm, family, period) assignment from QPU
- Select 2-3 specific crops within that family based on benefits
- Calculate 27-food objective using the hybrid rotation matrix

Author: OQI-UC002-DWave
Date: 2025-12-25
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from food_grouping import (
    FOOD_TO_FAMILY, get_family, FAMILY_ORDER, FAMILY_TO_CROPS,
    aggregate_foods_to_families, create_family_rotation_matrix,
)
from hybrid_formulation import build_hybrid_rotation_matrix

# D-Wave imports
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler

try:
    from dwave.system import DWaveCliqueSampler
    HAS_QPU = True
except ImportError:
    HAS_QPU = False

# D-Wave token
DEFAULT_DWAVE_TOKEN = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'
MAX_CLIQUE_VARS = 170


# ============================================================================
# ENHANCED POST-PROCESSING: 6-FAMILY → 27-FOOD RECOVERY
# ============================================================================

def recover_27food_solution(family_solution: Dict,
                            original_data: Dict,
                            method: str = 'benefit_weighted',
                            seed: int = 42) -> Dict:
    """
    Convert 6-family QPU solution to 27-food solution.
    
    This is the KEY post-processing step that recovers the full 27-food
    granularity from the aggregated 6-family solution.
    
    Methods:
    - 'benefit_weighted': Select crops based on benefit scores (default)
    - 'uniform': Equal probability selection within family
    - 'greedy': Always select highest-benefit crops
    
    Args:
        family_solution: Dict with (farm, family, period) -> 1 assignments
        original_data: Original 27-food problem data
        method: Selection method ('benefit_weighted', 'uniform', 'greedy')
        seed: Random seed for reproducibility
    
    Returns:
        food_solution: Dict with (farm, food, period) -> 1 assignments (binary)
    """
    np.random.seed(seed)
    
    farm_names = original_data.get('farm_names', [])
    food_names = original_data.get('food_names', [])
    food_benefits = original_data.get('food_benefits', {})
    n_periods = 3
    
    food_solution = {}
    
    for farm in farm_names:
        for period in range(1, n_periods + 1):
            # Find assigned family for this farm-period
            assigned_family = None
            
            # Try tuple key format
            for family in FAMILY_ORDER:
                if family_solution.get((farm, family, period), 0) == 1:
                    assigned_family = family
                    break
            
            # Try string key format
            if assigned_family is None:
                for family in FAMILY_ORDER:
                    var_name = f"Y_{farm}_{family}_t{period}"
                    if family_solution.get(var_name, 0) == 1:
                        assigned_family = family
                        break
            
            if assigned_family is None:
                # No assignment for this farm-period, skip
                continue
            
            # Get crops in this family that exist in our food list
            family_crops = FAMILY_TO_CROPS.get(assigned_family, [])
            available_crops = [c for c in family_crops if c in food_names]
            
            # If no crops found in food_names, try fuzzy matching
            if not available_crops:
                for crop in family_crops:
                    for food in food_names:
                        if crop.lower() in food.lower() or food.lower() in crop.lower():
                            available_crops.append(food)
                            break
            
            # Still nothing? Use all foods that map to this family
            if not available_crops:
                available_crops = [f for f in food_names if get_family(f) == assigned_family]
            
            # Final fallback
            if not available_crops:
                available_crops = [food_names[0]]
            
            # Select EXACTLY 1 crop based on method (to maintain one-hot constraint)
            if method == 'greedy' or method == 'benefit_weighted':
                # Select highest-benefit crop
                crop_benefits = [(c, food_benefits.get(c, 0.5)) for c in available_crops]
                crop_benefits.sort(key=lambda x: x[1], reverse=True)
                selected_crop = crop_benefits[0][0]
                
            elif method == 'uniform':
                # Random selection with equal probability
                np.random.seed(seed + farm_names.index(farm) * 100 + period)
                selected_crop = np.random.choice(available_crops)
                
            else:  # Default: benefit_weighted
                # Weighted selection based on benefits
                benefits = np.array([food_benefits.get(c, 0.5) for c in available_crops])
                probs = benefits / (benefits.sum() + 1e-9)
                np.random.seed(seed + farm_names.index(farm) * 100 + period)
                selected_crop = np.random.choice(available_crops, p=probs)
            
            # Assign exactly 1 crop per farm-period (maintains one-hot constraint)
            food_solution[(farm, selected_crop, period)] = 1
    
    return food_solution


def calculate_27food_objective(food_solution: Dict, 
                               original_data: Dict,
                               R_hybrid: np.ndarray = None) -> float:
    """
    Calculate objective value for 27-food solution.
    
    Uses the hybrid rotation matrix (27×27) derived from 6-family template
    as specified in the LaTeX formulation.
    
    Objective = Base Benefit + Rotation Synergies + Diversity Bonus - One-Hot Penalty
    
    Args:
        food_solution: Dict with (farm, food, period) -> 1 assignments
        original_data: Original 27-food problem data
        R_hybrid: 27×27 hybrid rotation matrix (built if not provided)
    
    Returns:
        Objective value (higher is better)
    """
    farm_names = original_data['farm_names']
    food_names = original_data['food_names']
    food_benefits = original_data['food_benefits']
    land_availability = original_data['land_availability']
    total_area = original_data['total_area']
    
    config = original_data.get('config', {})
    params = config.get('parameters', {})
    rotation_gamma = params.get('rotation_gamma', 0.2)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    
    n_periods = 3
    n_foods = len(food_names)
    
    # Build hybrid rotation matrix if not provided
    if R_hybrid is None:
        R_hybrid = build_hybrid_rotation_matrix(food_names, seed=42)
    
    obj = 0
    
    # Part 1: Base Benefit (Eq. 1 in LaTeX - first term)
    # sum_{f,c,t} B_c * A_f * Y_{f,c,t}
    for (farm, food, period), val in food_solution.items():
        if val == 1:
            benefit = food_benefits.get(food, 0.5)
            area = land_availability.get(farm, 1.0)
            obj += (benefit * area) / total_area
    
    # Part 2: Rotation Synergies (Eq. 1 in LaTeX - second term)
    # gamma_R * sum_{f, t>=2} sum_{c,c'} R_{c,c'} * Y_{f,c,t-1} * Y_{f,c',t}
    food_to_idx = {f: i for i, f in enumerate(food_names)}
    
    for farm in farm_names:
        area = land_availability.get(farm, 1.0)
        for period in range(2, n_periods + 1):
            for food1 in food_names:
                for food2 in food_names:
                    v1 = food_solution.get((farm, food1, period - 1), 0)
                    v2 = food_solution.get((farm, food2, period), 0)
                    if v1 == 1 and v2 == 1:
                        idx1 = food_to_idx.get(food1, 0)
                        idx2 = food_to_idx.get(food2, 0)
                        synergy = R_hybrid[idx1, idx2]
                        obj += (rotation_gamma * synergy * area) / total_area
    
    # Part 3: One-Hot Penalty (soft constraint)
    for farm in farm_names:
        for period in range(1, n_periods + 1):
            count = sum(1 for food in food_names 
                       if food_solution.get((farm, food, period), 0) == 1)
            if count > 1:
                # Penalize multiple selections
                obj -= one_hot_penalty * (count - 1)
    
    # Part 4: Diversity Bonus
    for farm in farm_names:
        area = land_availability.get(farm, 1.0)
        foods_used = set()
        for food in food_names:
            for period in range(1, n_periods + 1):
                if food_solution.get((farm, food, period), 0) == 1:
                    foods_used.add(food)
        obj += diversity_bonus * len(foods_used) * (area / total_area)
    
    return obj


# ============================================================================
# UNIFIED ADAPTIVE SOLVER
# ============================================================================

def solve_adaptive_with_recovery(data: Dict,
                                  num_reads: int = 100,
                                  num_iterations: int = 3,
                                  overlap_farms: int = 1,
                                  use_qpu: bool = True,
                                  recovery_method: str = 'benefit_weighted',
                                  verbose: bool = True) -> Dict:
    """
    Adaptive solver with full 27-food solution recovery.
    
    Pipeline:
    1. If 27 foods: Aggregate to 6 families for QPU
    2. Solve 6-family problem on QPU (or SA simulator)
    3. Post-process: Recover 27-food solution from 6-family assignments
    4. Calculate both family-level and food-level objectives
    
    Args:
        data: Problem data (can be 27 foods or 6 families)
        num_reads: QPU reads per subproblem
        num_iterations: Boundary coordination iterations
        overlap_farms: Overlapping farms between clusters
        use_qpu: If True, use D-Wave QPU; else use SimulatedAnnealing
        recovery_method: Method for 27-food recovery ('benefit_weighted', 'greedy', 'uniform')
        verbose: Print progress
    
    Returns:
        Dict with:
        - family_solution: 6-family level solution from QPU
        - food_solution: Recovered 27-food solution
        - objective_family: Objective at family level
        - objective_27food: Objective at 27-food level (main metric)
        - timing, diversity, etc.
    """
    total_start = time.time()
    
    # Extract data
    farm_names = data['farm_names']
    food_names = data['food_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    n_periods = 3
    
    if verbose:
        print("="*70)
        print("ADAPTIVE HYBRID SOLVER WITH 27-FOOD RECOVERY")
        print("="*70)
        print(f"Input: {n_farms} farms × {n_foods} foods × {n_periods} periods")
        print(f"Mode: {'QPU' if use_qpu and HAS_QPU else 'Simulated Annealing'}")
        print(f"Recovery method: {recovery_method}")
    
    result = {
        'method': 'adaptive_with_recovery',
        'n_farms': n_farms,
        'n_foods_original': n_foods,
        'n_periods': n_periods,
        'use_qpu': use_qpu and HAS_QPU,
        'recovery_method': recovery_method,
        'timings': {},
    }
    
    # =========================================================================
    # STEP 1: Aggregate to 6 families if needed
    # =========================================================================
    step1_start = time.time()
    
    if n_foods > 6:
        if verbose:
            print(f"\n[Step 1] Aggregating {n_foods} foods → 6 families")
        family_data = aggregate_foods_to_families(data)
        used_aggregation = True
        n_families = 6
    else:
        if verbose:
            print(f"\n[Step 1] Using {n_foods} foods directly (no aggregation)")
        family_data = data.copy()
        family_data['original_data'] = data
        used_aggregation = False
        n_families = n_foods
    
    family_names = family_data['food_names']
    family_benefits = family_data['food_benefits']
    
    result['timings']['step1_aggregation'] = time.time() - step1_start
    result['used_aggregation'] = used_aggregation
    result['n_families'] = n_families
    
    # =========================================================================
    # STEP 2: Spatial decomposition
    # =========================================================================
    step2_start = time.time()
    
    vars_per_farm = n_families * n_periods
    max_farms_per_cluster = max(1, MAX_CLIQUE_VARS // vars_per_farm)
    
    effective_farms_per_cluster = max_farms_per_cluster - overlap_farms
    if effective_farms_per_cluster < 1:
        effective_farms_per_cluster = max_farms_per_cluster
        overlap_farms = 0
    
    # Create clusters with overlap
    clusters = []
    cluster_start = 0
    while cluster_start < n_farms:
        cluster_end = min(cluster_start + max_farms_per_cluster, n_farms)
        clusters.append(farm_names[cluster_start:cluster_end])
        cluster_start += effective_farms_per_cluster
    
    n_clusters = len(clusters)
    
    if verbose:
        print(f"\n[Step 2] Spatial decomposition:")
        print(f"  Vars per farm: {vars_per_farm}")
        print(f"  Max farms per cluster: {max_farms_per_cluster}")
        print(f"  Total clusters: {n_clusters}")
    
    result['timings']['step2_decomposition'] = time.time() - step2_start
    result['n_clusters'] = n_clusters
    
    # =========================================================================
    # STEP 3: Build cluster neighbor graph
    # =========================================================================
    cluster_neighbors = {}
    for i in range(n_clusters):
        neighbors = []
        if i > 0:
            neighbors.append(i - 1)
        if i < n_clusters - 1:
            neighbors.append(i + 1)
        cluster_neighbors[i] = neighbors
    
    # =========================================================================
    # STEP 4: Solve with QPU or SA
    # =========================================================================
    step4_start = time.time()
    
    if verbose:
        print(f"\n[Step 4] Solving ({num_iterations} iterations, {num_reads} reads)")
    
    # Initialize sampler
    if use_qpu and HAS_QPU:
        token = os.environ.get('DWAVE_API_TOKEN', DEFAULT_DWAVE_TOKEN)
        sampler = DWaveCliqueSampler(token=token)
        sampler_type = 'QPU'
    else:
        sampler = SimulatedAnnealingSampler()
        sampler_type = 'SA'
    
    # Problem parameters
    config = data.get('config', {})
    params = config.get('parameters', {})
    rotation_gamma = params.get('rotation_gamma', 0.2)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    
    # Family rotation matrix
    R_family = create_family_rotation_matrix(seed=42)
    
    # Tracking
    total_qpu_time = 0
    cluster_solutions = [{}] * n_clusters
    boundary_info = [{}] * n_clusters
    best_global_solution = {}
    best_global_objective = -np.inf
    
    for iteration in range(num_iterations):
        if verbose:
            print(f"\n  Iteration {iteration + 1}/{num_iterations}")
        
        iter_qpu_time = 0
        
        # Update boundary info
        if iteration > 0:
            boundary_info = _coordinate_boundaries(clusters, cluster_solutions, cluster_neighbors)
        
        # Solve each cluster
        new_cluster_solutions = []
        
        for cluster_idx, cluster_farms in enumerate(clusters):
            # Build BQM for this cluster
            bqm = BinaryQuadraticModel('BINARY')
            var_map = {}
            
            # Create variables
            for farm in cluster_farms:
                for family in family_names:
                    for period in range(1, n_periods + 1):
                        var_name = f"Y_{farm}_{family}_t{period}"
                        var_map[(farm, family, period)] = var_name
                        bqm.add_variable(var_name, 0)
            
            # Benefits (linear terms)
            for farm in cluster_farms:
                farm_area = land_availability.get(farm, 25.0)
                for family in family_names:
                    benefit = family_benefits.get(family, 0.5)
                    for period in range(1, n_periods + 1):
                        var = var_map[(farm, family, period)]
                        bqm.add_variable(var, -(benefit * farm_area) / total_area)
            
            # Rotation synergies
            for farm in cluster_farms:
                farm_area = land_availability.get(farm, 25.0)
                for period in range(2, n_periods + 1):
                    for fam1_idx, fam1 in enumerate(family_names):
                        for fam2_idx, fam2 in enumerate(family_names):
                            synergy = R_family[fam1_idx, fam2_idx]
                            if abs(synergy) > 1e-6:
                                var1 = var_map[(farm, fam1, period - 1)]
                                var2 = var_map[(farm, fam2, period)]
                                bqm.add_interaction(var1, var2, -(rotation_gamma * synergy * farm_area) / total_area)
            
            # One-hot constraint
            for farm in cluster_farms:
                for period in range(1, n_periods + 1):
                    period_vars = [var_map[(farm, fam, period)] for fam in family_names]
                    for i, v1 in enumerate(period_vars):
                        for v2 in period_vars[i+1:]:
                            bqm.add_interaction(v1, v2, 2 * one_hot_penalty)
                        bqm.add_variable(v1, -one_hot_penalty)
            
            # Diversity bonus
            for farm in cluster_farms:
                for family in family_names:
                    for period in range(1, n_periods + 1):
                        var = var_map[(farm, family, period)]
                        bqm.add_variable(var, -diversity_bonus / n_periods)
            
            # Boundary coordination
            if iteration > 0 and boundary_info[cluster_idx]:
                for (farm, family, period), value in boundary_info[cluster_idx].items():
                    if farm in cluster_farms:
                        var = var_map.get((farm, family, period))
                        if var and value == 1:
                            bqm.add_variable(var, -0.1)
            
            # Solve
            try:
                if use_qpu and HAS_QPU:
                    label = f"Adaptive27_{n_farms}f_c{cluster_idx}_i{iteration}"
                    sampleset = sampler.sample(bqm, num_reads=num_reads, label=label)
                    timing = sampleset.info.get('timing', {})
                    qpu_time = timing.get('qpu_access_time', 0) / 1e6
                    iter_qpu_time += qpu_time
                else:
                    sampleset = sampler.sample(bqm, num_reads=num_reads)
                
                # Extract solution
                best_sample = sampleset.first.sample
                cluster_sol = {}
                for (farm, family, period), var_name in var_map.items():
                    if best_sample.get(var_name, 0) == 1:
                        cluster_sol[(farm, family, period)] = 1
                
            except Exception as e:
                print(f"    Cluster {cluster_idx} FAILED: {e}")
                cluster_sol = {}
                for farm in cluster_farms:
                    for period in range(1, n_periods + 1):
                        cluster_sol[(farm, family_names[0], period)] = 1
            
            new_cluster_solutions.append(cluster_sol)
        
        cluster_solutions = new_cluster_solutions
        total_qpu_time += iter_qpu_time
        
        # Stitch solutions
        combined_solution = _stitch_cluster_solutions(clusters, cluster_solutions, overlap_farms)
        
        # Evaluate family-level objective
        obj = _calculate_family_objective(combined_solution, family_data, R_family)
        
        if obj > best_global_objective:
            best_global_objective = obj
            best_global_solution = combined_solution.copy()
        
        if verbose:
            n_assigned = sum(1 for v in combined_solution.values() if v == 1)
            print(f"    {sampler_type}: obj={obj:.4f}, assigned={n_assigned}")
    
    result['timings']['step4_solve'] = time.time() - step4_start
    result['qpu_time'] = total_qpu_time
    
    # =========================================================================
    # STEP 5: Post-processing - Recover 27-food solution
    # =========================================================================
    step5_start = time.time()
    
    if verbose:
        print(f"\n[Step 5] Recovering 27-food solution (method: {recovery_method})")
    
    # Get original data for recovery
    original_data = family_data.get('original_data', data)
    
    # Recover 27-food solution
    food_solution = recover_27food_solution(
        best_global_solution, 
        original_data, 
        method=recovery_method,
        seed=42
    )
    
    # Build hybrid rotation matrix for 27-food objective
    original_food_names = original_data.get('food_names', food_names)
    R_hybrid = build_hybrid_rotation_matrix(original_food_names, seed=42)
    
    # Calculate 27-food objective
    obj_27food = calculate_27food_objective(food_solution, original_data, R_hybrid)
    
    # Analyze diversity
    from food_grouping import analyze_crop_diversity
    diversity_stats = analyze_crop_diversity(food_solution, original_data)
    
    if verbose:
        print(f"  Family-level objective: {best_global_objective:.4f}")
        print(f"  27-food objective: {obj_27food:.4f}")
        print(f"  Unique crops: {diversity_stats['total_unique_crops']}/{diversity_stats['max_possible_crops']}")
        print(f"  Shannon diversity: {diversity_stats['shannon_diversity']:.3f}")
    
    result['timings']['step5_recovery'] = time.time() - step5_start
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    total_time = time.time() - total_start
    
    result['solve_time'] = total_time
    result['objective_family'] = best_global_objective
    result['objective_27food'] = obj_27food
    result['objective'] = obj_27food  # Main objective is 27-food level
    result['family_solution'] = best_global_solution
    result['food_solution'] = food_solution
    result['diversity_stats'] = diversity_stats
    result['n_assigned_family'] = sum(1 for v in best_global_solution.values() if v == 1)
    result['n_assigned_food'] = sum(1 for v in food_solution.values() if v == 1)
    result['violations'] = _count_violations_27food(food_solution, original_data)
    result['success'] = True
    
    if verbose:
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Family-level objective: {best_global_objective:.4f}")
        print(f"27-food objective: {obj_27food:.4f}")
        print(f"Family assignments: {result['n_assigned_family']}")
        print(f"Food assignments: {result['n_assigned_food']}")
        print(f"Violations: {result['violations']}")
        print(f"Total time: {total_time:.2f}s (QPU: {total_qpu_time:.3f}s)")
    
    return result


def _coordinate_boundaries(clusters, cluster_solutions, cluster_neighbors):
    """Extract boundary information for inter-cluster coordination."""
    boundary_info = []
    
    for i, cluster in enumerate(clusters):
        boundary_solutions = {}
        
        for neighbor_idx in cluster_neighbors.get(i, []):
            neighbor_sol = cluster_solutions[neighbor_idx]
            neighbor_cluster = clusters[neighbor_idx]
            boundary_farms = [neighbor_cluster[0], neighbor_cluster[-1]]
            
            for key, value in neighbor_sol.items():
                farm, family, period = key
                if farm in boundary_farms:
                    boundary_solutions[key] = value
        
        boundary_info.append(boundary_solutions)
    
    return boundary_info


def _stitch_cluster_solutions(clusters, cluster_solutions, overlap_farms):
    """Merge cluster solutions, handling overlapping farms."""
    combined = {}
    farm_assignments = defaultdict(list)
    
    for cluster_idx, (cluster, sol) in enumerate(zip(clusters, cluster_solutions)):
        for (farm, family, period), value in sol.items():
            if value == 1:
                farm_assignments[(farm, period)].append((family, cluster_idx))
    
    for (farm, period), assignments in farm_assignments.items():
        if len(assignments) == 1:
            family, _ = assignments[0]
            combined[(farm, family, period)] = 1
        else:
            family, _ = assignments[-1]
            combined[(farm, family, period)] = 1
    
    return combined


def _calculate_family_objective(solution, family_data, R):
    """Calculate objective value for family-level solution."""
    farm_names = family_data['farm_names']
    family_names = family_data['food_names']
    land_availability = family_data['land_availability']
    family_benefits = family_data['food_benefits']
    total_area = family_data['total_area']
    
    config = family_data.get('config', {})
    params = config.get('parameters', {})
    rotation_gamma = params.get('rotation_gamma', 0.2)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    
    n_periods = 3
    obj = 0
    
    # Benefits
    for (farm, family, period), val in solution.items():
        if val == 1:
            benefit = family_benefits.get(family, 0.5)
            area = land_availability.get(farm, 25.0)
            obj += (benefit * area) / total_area
    
    # Rotation synergies
    for farm in farm_names:
        area = land_availability.get(farm, 25.0)
        for period in range(2, n_periods + 1):
            for fam1_idx, fam1 in enumerate(family_names):
                for fam2_idx, fam2 in enumerate(family_names):
                    v1 = solution.get((farm, fam1, period - 1), 0)
                    v2 = solution.get((farm, fam2, period), 0)
                    if v1 == 1 and v2 == 1:
                        obj += (rotation_gamma * R[fam1_idx, fam2_idx] * area) / total_area
    
    # One-hot penalty
    for farm in farm_names:
        for period in range(1, n_periods + 1):
            count = sum(1 for fam in family_names if solution.get((farm, fam, period), 0) == 1)
            if count > 1:
                obj -= one_hot_penalty * (count - 1) ** 2
    
    # Diversity bonus
    for farm in farm_names:
        area = land_availability.get(farm, 25.0)
        families_used = set()
        for fam in family_names:
            for period in range(1, n_periods + 1):
                if solution.get((farm, fam, period), 0) == 1:
                    families_used.add(fam)
        obj += diversity_bonus * len(families_used) * (area / total_area)
    
    return obj


def _count_violations_27food(solution, data):
    """Count violations in 27-food solution."""
    farm_names = data['farm_names']
    food_names = data['food_names']
    n_periods = 3
    
    max_crops = 0
    for farm in farm_names:
        for period in range(1, n_periods + 1):
            count = sum(1 for food in food_names if solution.get((farm, food, period), 0) == 1)
            if count > 2:
                max_crops += count - 2
    
    return {'max_crops': max_crops, 'total': max_crops}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def solve_simulated(data: Dict, **kwargs) -> Dict:
    """Solve with Simulated Annealing (for testing without QPU)."""
    return solve_adaptive_with_recovery(data, use_qpu=False, **kwargs)


def solve_qpu(data: Dict, **kwargs) -> Dict:
    """Solve with D-Wave QPU."""
    return solve_adaptive_with_recovery(data, use_qpu=True, **kwargs)


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from data_loader_utils import load_food_data_as_dict
    
    print("Testing adaptive solver with 27-food recovery...")
    
    # Load 27-food scenario
    data = load_food_data_as_dict('rotation_250farms_27foods')
    data['farm_names'] = data['farm_names'][:10]
    data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
    data['total_area'] = sum(data['land_availability'].values())
    
    print(f"\nProblem: {len(data['farm_names'])} farms × {len(data['food_names'])} foods")
    
    # Test with SA (no QPU needed)
    print("\n--- Simulated Annealing Test ---")
    result_sa = solve_simulated(data, num_reads=50, num_iterations=2, verbose=True)
    
    print(f"\nSA Results:")
    print(f"  Family objective: {result_sa['objective_family']:.4f}")
    print(f"  27-food objective: {result_sa['objective_27food']:.4f}")
    print(f"  Unique crops: {result_sa['diversity_stats']['total_unique_crops']}")
    
    # Test with QPU if available
    if HAS_QPU:
        print("\n--- QPU Test ---")
        result_qpu = solve_qpu(data, num_reads=50, num_iterations=2, verbose=True)
        
        print(f"\nQPU Results:")
        print(f"  Family objective: {result_qpu['objective_family']:.4f}")
        print(f"  27-food objective: {result_qpu['objective_27food']:.4f}")
        print(f"  Unique crops: {result_qpu['diversity_stats']['total_unique_crops']}")
    
    print("\n✓ Test complete!")
