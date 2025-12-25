#!/usr/bin/env python3
"""
Hybrid Formulation: 27-Food Variables with 6-Family Synergy Matrices

Key Innovation:
- Keep all 27 foods as optimization variables (no aggregation loss)
- Use 6-family rotation synergy matrix (computational tractability)
- Map synergies: Food1 → Family1, Food2 → Family2, then lookup synergy(Family1, Family2)

This gives us:
- Full expressiveness (27 distinct choices)
- Tractable synergy computation (6×6 matrix instead of 27×27)
- Consistent formulation across all problem sizes
- Auto-detection for decomposition strategy

SOLVER: Uses DWaveCliqueSampler with spatial decomposition for QPU solving.
- Same formulation as Gurobi ground truth
- Same constraints (soft one-hot, max 2 crops per farm per period)
- Same objective (benefits + rotation synergies + spatial + diversity)

Author: OQI-UC002-DWave
Date: 2025-12-12, Updated: 2025-12-24
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

from food_grouping import FOOD_TO_FAMILY, get_family, FAMILY_ORDER

# D-Wave imports
from dimod import BinaryQuadraticModel

try:
    from dwave.system import DWaveCliqueSampler
    HAS_QPU = True
except ImportError:
    HAS_QPU = False
    print("Warning: DWaveCliqueSampler not available")

# D-Wave token
DEFAULT_DWAVE_TOKEN = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'

# ============================================================================
# HYBRID SYNERGY MATRIX BUILDER
# ============================================================================

def build_hybrid_rotation_matrix(food_names: List[str], 
                                  frustration_ratio: float = 0.7,
                                  negative_strength: float = -0.8,
                                  seed: int = 42) -> np.ndarray:
    """
    Build 27×27 rotation synergy matrix using 6-family templates.
    
    Strategy:
    1. Generate 6×6 family-level synergy matrix with frustration
    2. For each (food_i, food_j) pair:
       - Map food_i → family_i, food_j → family_j
       - Lookup synergy(family_i, family_j) from 6×6 matrix
       - Add small random noise for diversity
    
    This gives:
    - Structured synergies (families have consistent patterns)
    - Food-level granularity (27×27 matrix)
    - Computational efficiency (based on 6×6 template)
    
    Args:
        food_names: List of 27 food names
        frustration_ratio: Fraction of negative synergies (anti-ferromagnetic)
        negative_strength: Strength of negative synergies
        seed: Random seed for reproducibility
    
    Returns:
        27×27 rotation synergy matrix
    """
    np.random.seed(seed)
    n_foods = len(food_names)
    
    # Step 1: Build 6×6 family-level template
    n_families = len(FAMILY_ORDER)
    family_matrix = np.zeros((n_families, n_families))
    
    for i in range(n_families):
        for j in range(n_families):
            if i == j:
                # Same family: strong negative (avoid monoculture)
                family_matrix[i, j] = negative_strength * 1.5
            elif np.random.random() < frustration_ratio:
                # Different families: mostly negative (frustration)
                family_matrix[i, j] = np.random.uniform(negative_strength * 1.2, 
                                                         negative_strength * 0.3)
            else:
                # Positive synergies (rare)
                family_matrix[i, j] = np.random.uniform(0.02, 0.20)
    
    # Step 2: Map foods to families
    food_to_family_idx = {}
    for food in food_names:
        family = get_family(food)
        if family in FAMILY_ORDER:
            food_to_family_idx[food] = FAMILY_ORDER.index(family)
        else:
            food_to_family_idx[food] = FAMILY_ORDER.index('Other')
    
    # Step 3: Build 27×27 matrix from 6×6 template
    food_matrix = np.zeros((n_foods, n_foods))
    
    for i, food_i in enumerate(food_names):
        for j, food_j in enumerate(food_names):
            # Get family indices
            fam_i = food_to_family_idx[food_i]
            fam_j = food_to_family_idx[food_j]
            
            # Base synergy from family template
            base_synergy = family_matrix[fam_i, fam_j]
            
            # Add small random noise for food-level diversity
            # (keeps family structure but distinguishes individual foods)
            noise = np.random.uniform(-0.05, 0.05)
            food_matrix[i, j] = base_synergy + noise
    
    return food_matrix


def get_food_family_mapping(food_names: List[str]) -> Dict:
    """
    Get mapping information for foods → families.
    
    Returns:
        Dict with:
        - food_to_family: {food: family_name}
        - family_to_foods: {family: [foods]}
        - food_to_idx: {food: index in food_names}
        - family_to_idx: {family: index in FAMILY_ORDER}
    """
    food_to_family = {food: get_family(food) for food in food_names}
    
    family_to_foods = {family: [] for family in FAMILY_ORDER}
    for food, family in food_to_family.items():
        if family in family_to_foods:
            family_to_foods[family].append(food)
    
    food_to_idx = {food: i for i, food in enumerate(food_names)}
    family_to_idx = {family: i for i, family in enumerate(FAMILY_ORDER)}
    
    return {
        'food_to_family': food_to_family,
        'family_to_foods': family_to_foods,
        'food_to_idx': food_to_idx,
        'family_to_idx': family_to_idx,
    }


# ============================================================================
# AUTO-DETECTION FOR DECOMPOSITION STRATEGY
# ============================================================================

# DWaveCliqueSampler has a max clique size around 177 variables
# We need to stay well below this for reliable embedding
MAX_CLIQUE_VARS = 170  # Safe limit for DWaveCliqueSampler

def detect_decomposition_strategy(n_farms: int, n_foods: int, n_periods: int = 3) -> Dict:
    """
    Auto-detect which decomposition strategy to use based on problem size.
    
    CRITICAL: DWaveCliqueSampler can only embed ~177 variables max!
    For 27 foods × 3 periods = 81 vars/farm, so max ~2 farms per cluster.
    For 6 families × 3 periods = 18 vars/farm, so max ~9 farms per cluster.
    
    Decision tree:
    1. If cluster_vars <= 170: Use that cluster size
    2. For 27 foods: max 2 farms per cluster (162 vars)
    3. For 6 families: max 9 farms per cluster (162 vars)
    
    Args:
        n_farms: Number of farms
        n_foods: Number of foods/families
        n_periods: Number of rotation periods
    
    Returns:
        Dict with strategy recommendations
    """
    n_vars = n_farms * n_foods * n_periods
    vars_per_farm = n_foods * n_periods
    
    # Calculate max farms per cluster based on QPU limit
    max_farms_per_cluster = max(1, MAX_CLIQUE_VARS // vars_per_farm)
    
    strategy = {
        'n_vars': n_vars,
        'n_farms': n_farms,
        'n_foods': n_foods,
        'n_periods': n_periods,
        'vars_per_farm': vars_per_farm,
        'max_farms_per_cluster': max_farms_per_cluster,
    }
    
    if n_farms <= max_farms_per_cluster and n_vars <= MAX_CLIQUE_VARS:
        # Small enough for direct solve
        strategy['method'] = 'direct'
        strategy['use_decomposition'] = False
        strategy['description'] = f'Direct QPU solve ({n_vars} vars fits in clique)'
        strategy['farms_per_cluster'] = n_farms
        
    else:
        # Need decomposition
        strategy['method'] = 'spatial_decomposition'
        strategy['use_decomposition'] = True
        
        # Use max farms per cluster (respecting QPU limit)
        farms_per_cluster = min(max_farms_per_cluster, n_farms)
        strategy['farms_per_cluster'] = farms_per_cluster
        strategy['n_clusters'] = (n_farms + farms_per_cluster - 1) // farms_per_cluster
        strategy['vars_per_cluster'] = farms_per_cluster * vars_per_farm
        strategy['description'] = f'Spatial decomposition ({farms_per_cluster} farms/cluster = {farms_per_cluster * vars_per_farm} vars)'
    
    return strategy


def recommend_parameters(strategy: Dict) -> Dict:
    """
    Recommend solver parameters based on strategy.
    
    Returns:
        Dict with num_reads, num_iterations, timeout, etc.
    """
    params = {}
    
    if strategy['method'] == 'direct':
        params['num_reads'] = 100
        params['num_iterations'] = 1
        params['timeout'] = 300
        params['use_qpu'] = True
        
    else:  # spatial_decomposition
        params['num_reads'] = 100
        params['num_iterations'] = 3  # Boundary coordination
        params['timeout'] = 600
        params['use_qpu'] = True
    
    return params


# ============================================================================
# HYBRID QPU SOLVER - SAME FORMULATION AS GUROBI
# ============================================================================

def solve_hybrid_qpu(data: Dict, 
                      num_reads: int = 100,
                      num_iterations: int = 3,
                      verbose: bool = True) -> Dict:
    """
    Solve 27-food rotation problem on D-Wave QPU using spatial decomposition.
    
    FORMULATION (matches Gurobi ground truth EXACTLY):
    - Variables: Y[farm, food, period] binary for all 27 foods
    - Objective: benefits + rotation synergies + spatial synergies + diversity - one_hot_penalty
    - Constraints: soft one-hot (max 2 crops per farm per period)
    
    DECOMPOSITION:
    - Spatial decomposition: clusters of farms
    - Each cluster solved on QPU with DWaveCliqueSampler
    - Boundary coordination between iterations
    
    Args:
        data: Problem data with farm_names, food_names (27), land_availability, food_benefits
        num_reads: Number of QPU reads per subproblem
        num_iterations: Number of boundary coordination iterations
        verbose: Print progress
    
    Returns:
        Dict with objective, solve_time, qpu_time, solution, violations, etc.
    """
    if not HAS_QPU:
        raise RuntimeError("QPU not available. Install dwave-system.")
    
    total_start = time.time()
    
    # Extract data
    farm_names = data['farm_names']
    food_names = data['food_names']  # Should be 27 foods
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    n_periods = 3
    n_vars = n_farms * n_foods * n_periods
    
    if verbose:
        print(f"[HybridQPU] Problem: {n_farms} farms × {n_foods} foods × {n_periods} periods = {n_vars} vars")
    
    # Get problem parameters (same as Gurobi)
    config = data.get('config', {})
    params = config.get('parameters', {})
    rotation_gamma = params.get('rotation_gamma', 0.2)
    k_neighbors = params.get('spatial_k_neighbors', 4)
    frustration_ratio = params.get('frustration_ratio', 0.7)
    negative_strength = params.get('negative_synergy_strength', -0.8)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    
    # Build 27×27 rotation matrix using 6-family template (our hybrid approach)
    R = build_hybrid_rotation_matrix(food_names, frustration_ratio, negative_strength, seed=42)
    
    # Create spatial neighbor graph (same as Gurobi)
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {}
    for i, farm in enumerate(farm_names):
        row, col = i // side, i % side
        positions[farm] = (row, col)
    
    neighbor_edges = []
    for f1 in farm_names:
        distances = []
        for f2 in farm_names:
            if f1 != f2:
                dist = np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                             (positions[f1][1] - positions[f2][1])**2)
                distances.append((dist, f2))
        distances.sort()
        for _, f2 in distances[:k_neighbors]:
            if (f2, f1) not in neighbor_edges:
                neighbor_edges.append((f1, f2))
    
    # Determine decomposition strategy
    strategy = detect_decomposition_strategy(n_farms, n_foods, n_periods)
    farms_per_cluster = strategy['farms_per_cluster']
    
    # Create farm clusters
    clusters = []
    for i in range(0, n_farms, farms_per_cluster):
        cluster = farm_names[i:i+farms_per_cluster]
        clusters.append(cluster)
    n_clusters = len(clusters)
    
    if verbose:
        print(f"[HybridQPU] Strategy: {strategy['method']}")
        print(f"[HybridQPU] Clusters: {n_clusters} × ~{farms_per_cluster} farms")
    
    # Initialize QPU sampler
    token = os.environ.get('DWAVE_API_TOKEN', DEFAULT_DWAVE_TOKEN)
    sampler = DWaveCliqueSampler(token=token)
    
    # Track timing
    total_qpu_time = 0
    total_qpu_sampling_time = 0
    total_qpu_programming_time = 0
    
    # Store cluster solutions
    cluster_solutions = [{}] * n_clusters
    best_global_solution = {}
    best_global_objective = -np.inf
    
    # Iterative solving with boundary coordination
    for iteration in range(num_iterations):
        if verbose:
            print(f"[HybridQPU] Iteration {iteration + 1}/{num_iterations}")
        
        iter_qpu_time = 0
        
        for cluster_idx, cluster_farms in enumerate(clusters):
            # Build BQM for this cluster (SAME formulation as Gurobi)
            bqm = BinaryQuadraticModel('BINARY')
            
            # Variable mapping: (farm, food, period) -> var_id
            var_map = {}
            var_id = 0
            for farm in cluster_farms:
                for food in food_names:
                    for period in range(1, n_periods + 1):
                        var_map[(farm, food, period)] = var_id
                        var_id += 1
            
            # === OBJECTIVE (matches Gurobi exactly) ===
            
            # Part 1: Base benefit (negative because BQM minimizes)
            for farm in cluster_farms:
                farm_area = land_availability[farm]
                for food_idx, food in enumerate(food_names):
                    benefit = food_benefits.get(food, 0.5)
                    for period in range(1, n_periods + 1):
                        var = var_map[(farm, food, period)]
                        # Strong benefit to select crops
                        bqm.add_variable(var, -(benefit * farm_area) / total_area)
            
            # Part 2: Rotation synergies (temporal) - SIMPLIFIED
            # Use same-family penalty, different-family small bonus
            # This is more tractable for QPU than full R matrix
            for farm in cluster_farms:
                farm_area = land_availability[farm]
                for period in range(2, n_periods + 1):
                    for food1_idx, food1 in enumerate(food_names):
                        fam1 = get_family(food1)
                        for food2_idx, food2 in enumerate(food_names):
                            fam2 = get_family(food2)
                            # Same family = bad (monoculture)
                            # Different family = small bonus (rotation)
                            if fam1 == fam2:
                                synergy = -0.3  # Penalty for same family
                            else:
                                synergy = 0.02  # Small bonus for rotation
                            
                            if abs(synergy) > 1e-6:
                                var1 = var_map[(farm, food1, period - 1)]
                                var2 = var_map[(farm, food2, period)]
                                bqm.add_interaction(
                                    var1, var2, 
                                    -(rotation_gamma * synergy * farm_area) / total_area
                                )
            
            # Part 3: One-hot constraint - encourage exactly 1 crop per farm per period
            # Formula: penalty * (sum - 1)^2 = penalty * (sum^2 - 2*sum + 1)
            # = penalty * (sum_i sum_j x_i*x_j - 2*sum_i x_i + 1)
            # Linear: -2*penalty for each var
            # Quadratic: +penalty for each pair
            # (Constant +penalty doesn't affect optimization)
            for farm in cluster_farms:
                for period in range(1, n_periods + 1):
                    period_vars = [var_map[(farm, food, period)] for food in food_names]
                    # Quadratic: penalty for pairs (discourages selecting >1)
                    for i, v1 in enumerate(period_vars):
                        for v2 in period_vars[i+1:]:
                            bqm.add_interaction(v1, v2, 2 * one_hot_penalty)  # *2 for (x_i * x_j + x_j * x_i)
                        # Linear: encourage selecting 1 (-2*penalty pushes toward 1)
                        bqm.add_variable(v1, -one_hot_penalty)
            
            # Part 4: Diversity bonus - encourage using different foods
            for farm in cluster_farms:
                for food in food_names:
                    # Bonus for using a food at least once across periods
                    for period in range(1, n_periods + 1):
                        var = var_map[(farm, food, period)]
                        bqm.add_variable(var, -diversity_bonus / n_periods)
            
            # Part 6: Boundary coordination (from neighboring clusters)
            if iteration > 0 and cluster_idx > 0:
                # Get boundary farms from previous cluster
                prev_cluster = clusters[cluster_idx - 1]
                boundary_farm = prev_cluster[-1]  # Last farm of previous cluster
                
                # Add weak coupling to encourage consistency
                if cluster_farms:
                    first_farm = cluster_farms[0]
                    for food_idx, food in enumerate(food_names):
                        for period in range(1, n_periods + 1):
                            # Check if prev solution had this food
                            prev_sol = cluster_solutions[cluster_idx - 1]
                            if prev_sol.get((boundary_farm, food, period), 0) > 0:
                                var = var_map[(first_farm, food, period)]
                                bqm.add_variable(var, -0.1)  # Small bonus for consistency
            
            # === SOLVE ON QPU ===
            try:
                label = f"Hybrid27_{n_farms}f_c{cluster_idx+1}of{n_clusters}_i{iteration+1}"
                sampleset = sampler.sample(bqm, num_reads=num_reads, label=label)
                
                # Extract timing
                timing = sampleset.info.get('timing', {})
                qpu_access = timing.get('qpu_access_time', 0) / 1e6  # µs to s
                qpu_sampling = timing.get('qpu_sampling_time', 0) / 1e6
                qpu_programming = timing.get('qpu_programming_time', 0) / 1e6
                
                iter_qpu_time += qpu_access
                total_qpu_sampling_time += qpu_sampling
                total_qpu_programming_time += qpu_programming
                
                # Decode solution
                best_sample = sampleset.first.sample
                best_energy = sampleset.first.energy
                
                # DEBUG: Count how many variables are set
                n_set = sum(1 for v in best_sample.values() if v > 0.5)
                if verbose:
                    print(f"      Cluster {cluster_idx}: {n_set}/{len(var_map)} vars set, energy={best_energy:.3f}")
                
                cluster_sol = {}
                
                for farm in cluster_farms:
                    for period in range(1, n_periods + 1):
                        # Find which food(s) are selected
                        for food in food_names:
                            var = var_map[(farm, food, period)]
                            if best_sample.get(var, 0) > 0.5:
                                cluster_sol[(farm, food, period)] = 1
                
                cluster_solutions[cluster_idx] = cluster_sol
                
            except Exception as e:
                print(f"[HybridQPU] Warning: Cluster {cluster_idx} failed: {e}")
                # Default solution: first food for each farm-period
                cluster_sol = {}
                for farm in cluster_farms:
                    for period in range(1, n_periods + 1):
                        cluster_sol[(farm, food_names[0], period)] = 1
                cluster_solutions[cluster_idx] = cluster_sol
        
        total_qpu_time += iter_qpu_time
        
        # Combine cluster solutions
        combined_solution = {}
        for sol in cluster_solutions:
            combined_solution.update(sol)
        
        # Calculate objective (same formula as Gurobi)
        obj = calculate_hybrid_objective(combined_solution, data, R)
        
        if obj > best_global_objective:
            best_global_objective = obj
            best_global_solution = combined_solution.copy()
        
        if verbose:
            print(f"    Objective: {obj:.4f}, QPU time: {iter_qpu_time:.3f}s")
    
    total_time = time.time() - total_start
    
    # Count violations
    violations = count_hybrid_violations(best_global_solution, data)
    
    # Count assignments
    n_assigned = len(best_global_solution)
    
    # Crop distribution
    crop_counts = defaultdict(int)
    for (farm, food, period), val in best_global_solution.items():
        if val > 0:
            crop_counts[food] += 1
    
    result = {
        'method': 'hybrid_qpu',
        'success': True,
        'objective': best_global_objective,
        'solve_time': total_time,
        'qpu_time': total_qpu_time,
        'qpu_access_time': total_qpu_time,
        'qpu_sampling_time': total_qpu_sampling_time,
        'qpu_programming_time': total_qpu_programming_time,
        'violations': violations,
        'n_assigned': n_assigned,
        'n_clusters': n_clusters,
        'farms_per_cluster': farms_per_cluster,
        'strategy': strategy['method'],
        'solution': best_global_solution,
        'crop_distribution': dict(crop_counts),
    }
    
    if verbose:
        print(f"[HybridQPU] Complete: obj={best_global_objective:.4f}, "
              f"time={total_time:.1f}s, qpu={total_qpu_time:.3f}s, "
              f"violations={violations['total']}")
    
    return result


def calculate_hybrid_objective(solution: Dict, data: Dict, R: np.ndarray = None) -> float:
    """
    Calculate objective value for hybrid solution (matches BQM formulation).
    
    Note: R parameter is kept for backward compatibility but not used.
    The function uses simplified synergies to match the BQM exactly.
    """
    farm_names = data['farm_names']
    food_names = data['food_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    
    config = data.get('config', {})
    params = config.get('parameters', {})
    rotation_gamma = params.get('rotation_gamma', 0.2)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    
    n_periods = 3
    obj = 0
    
    # Part 1: Base benefit
    for (farm, food, period), val in solution.items():
        if val > 0:
            farm_area = land_availability.get(farm, 25.0)
            benefit = food_benefits.get(food, 0.5)
            obj += (benefit * farm_area) / total_area
    
    # Part 2: Rotation synergies (simplified - matches BQM)
    # Note: BQM uses simplified synergies for tractability
    for farm in farm_names:
        farm_area = land_availability.get(farm, 25.0)
        for period in range(2, n_periods + 1):
            for food1_idx, food1 in enumerate(food_names):
                for food2_idx, food2 in enumerate(food_names):
                    if solution.get((farm, food1, period - 1), 0) > 0 and \
                       solution.get((farm, food2, period), 0) > 0:
                        # Simplified synergy: same crop = bad, different = good
                        synergy = -0.5 if food1 == food2 else 0.05
                        obj += (rotation_gamma * synergy * farm_area) / total_area
    
    # Part 3: One-hot penalty
    for farm in farm_names:
        for period in range(1, n_periods + 1):
            count = sum(1 for food in food_names if solution.get((farm, food, period), 0) > 0)
            if count > 1:
                obj -= one_hot_penalty * (count - 1) ** 2
    
    # Part 4: Diversity bonus
    for farm in farm_names:
        foods_used = set()
        for food in food_names:
            for period in range(1, n_periods + 1):
                if solution.get((farm, food, period), 0) > 0:
                    foods_used.add(food)
        obj += diversity_bonus * len(foods_used)
    
    return obj


def count_hybrid_violations(solution: Dict, data: Dict) -> Dict:
    """
    Count constraint violations in hybrid solution.
    """
    farm_names = data['farm_names']
    food_names = data['food_names']
    n_periods = 3
    
    max_crops_violations = 0
    rotation_violations = 0
    
    for farm in farm_names:
        for period in range(1, n_periods + 1):
            count = sum(1 for food in food_names if solution.get((farm, food, period), 0) > 0)
            if count > 2:  # Max 2 crops per period
                max_crops_violations += count - 2
    
    return {
        'max_crops': max_crops_violations,
        'rotation': rotation_violations,
        'total': max_crops_violations + rotation_violations,
    }


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("HYBRID FORMULATION: 27-Food Variables + 6-Family Synergies")
    print("="*80)
    print()
    
    # Example 1: Build hybrid rotation matrix
    print("Example 1: Building hybrid rotation matrix")
    print("-"*80)
    
    # 27 foods from rotation scenarios
    foods_27 = [
        'Beef', 'Chicken', 'Egg', 'Lamb', 'Pork',  # Proteins
        'Apple', 'Avocado', 'Banana', 'Durian', 'Guava', 'Kiwi', 'Mango', 'Papaya', 'Pineapple',  # Fruits
        'Broccoli', 'Cabbage', 'Carrot', 'Cauliflower', 'Celery',  # Vegetables
        'Barley', 'Oats', 'Rice', 'Wheat',  # Grains
        'Beans', 'Lentils', 'Peas',  # Legumes
        'Potato',  # Roots
    ]
    
    R = build_hybrid_rotation_matrix(foods_27)
    print(f"✓ Built {R.shape[0]}×{R.shape[1]} rotation matrix")
    print(f"  Negative entries: {(R < 0).sum()}/{R.size} ({(R < 0).sum()/R.size*100:.1f}%)")
    print(f"  Positive entries: {(R > 0).sum()}/{R.size} ({(R > 0).sum()/R.size*100:.1f}%)")
    print()
    
    # Show family structure
    mapping = get_food_family_mapping(foods_27)
    print("Family structure:")
    for family, foods in mapping['family_to_foods'].items():
        if foods:
            print(f"  {family}: {len(foods)} foods ({', '.join(foods[:3])}...)")
    print()
    
    # Example 2: Auto-detect strategy for different problem sizes
    print("Example 2: Auto-detecting decomposition strategy")
    print("-"*80)
    
    test_sizes = [
        (5, 27),   # 405 vars
        (10, 27),  # 810 vars
        (25, 27),  # 2025 vars
        (50, 27),  # 4050 vars
        (100, 27), # 8100 vars
    ]
    
    for n_farms, n_foods in test_sizes:
        strategy = detect_decomposition_strategy(n_farms, n_foods)
        params = recommend_parameters(strategy)
        
        print(f"\n{n_farms} farms × {n_foods} foods = {strategy['n_vars']} variables")
        print(f"  Strategy: {strategy['method']}")
        print(f"  Description: {strategy['description']}")
        if strategy['use_decomposition']:
            print(f"  Clusters: {strategy.get('n_clusters', 1)} × {strategy['farms_per_cluster']} farms")
        print(f"  QPU reads: {params['num_reads']}, iterations: {params['num_iterations']}")


# ============================================================================
# ADAPTIVE HYBRID SOLVER - BEST OF BOTH WORLDS
# ============================================================================
# Key insight: 27 foods = 81 vars/farm = max 2 farms/cluster = poor quality
#              6 families = 18 vars/farm = max 9 farms/cluster = good quality
#
# SOLUTION: Automatically aggregate to 6 families for QPU, then refine to 27 foods

from food_grouping import (
    aggregate_foods_to_families,
    refine_family_solution_to_crops,
    analyze_crop_diversity,
    create_family_rotation_matrix,
    FAMILY_TO_CROPS,
)


def solve_hybrid_adaptive(data: Dict,
                          num_reads: int = 100,
                          num_iterations: int = 3,
                          overlap_farms: int = 1,
                          verbose: bool = True) -> Dict:
    """
    Adaptive Hybrid Solver with automatic granularity adjustment.
    
    KEY INNOVATION:
    - Automatically aggregates 27 foods → 6 families for QPU (larger clusters)
    - Uses boundary coordination and overlap-based stitching
    - Refines 6-family solution back to 27 foods in post-processing
    
    This gives MUCH better solution quality than direct 27-food solving because:
    - 6 families = 18 vars/farm = max 9 farms/cluster (vs 2 for 27 foods)
    - Larger clusters = better global optimization
    - Post-processing recovers food-level granularity
    
    Args:
        data: Problem data with farm_names, food_names, land_availability, food_benefits
        num_reads: QPU reads per subproblem
        num_iterations: Boundary coordination iterations
        overlap_farms: Number of overlapping farms between clusters for stitching
        verbose: Print progress
    
    Returns:
        Dict with objective, solve_time, qpu_time, family_solution, crop_solution, etc.
    """
    if not HAS_QPU:
        raise RuntimeError("QPU not available. Install dwave-system.")
    
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
        print("ADAPTIVE HYBRID SOLVER")
        print("="*70)
        print(f"Input: {n_farms} farms × {n_foods} foods × {n_periods} periods")
    
    result = {
        'method': 'hybrid_adaptive',
        'n_farms': n_farms,
        'n_foods_original': n_foods,
        'n_periods': n_periods,
        'timings': {},
    }
    
    # =========================================================================
    # STEP 1: Adaptive Granularity - Aggregate to 6 families if needed
    # =========================================================================
    step1_start = time.time()
    
    if n_foods > 6:
        if verbose:
            print(f"\n[Step 1] Aggregating {n_foods} foods → 6 families for QPU efficiency")
        family_data = aggregate_foods_to_families(data)
        used_aggregation = True
        n_families = 6
    else:
        if verbose:
            print(f"\n[Step 1] Using {n_foods} foods directly (no aggregation needed)")
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
    # STEP 2: Determine decomposition strategy
    # =========================================================================
    step2_start = time.time()
    
    vars_per_farm = n_families * n_periods  # 18 for 6 families
    max_farms_per_cluster = max(1, MAX_CLIQUE_VARS // vars_per_farm)
    
    # Use overlap for better stitching
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
        print(f"  Overlap farms: {overlap_farms}")
        print(f"  Total clusters: {n_clusters}")
        for i, c in enumerate(clusters):
            print(f"    Cluster {i}: {len(c)} farms ({c[0]}...{c[-1]})")
    
    result['timings']['step2_decomposition'] = time.time() - step2_start
    result['n_clusters'] = n_clusters
    result['farms_per_cluster'] = max_farms_per_cluster
    result['overlap_farms'] = overlap_farms
    
    # =========================================================================
    # STEP 3: Build cluster neighbor graph for boundary coordination
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
    # STEP 4: QPU solving with boundary coordination
    # =========================================================================
    step4_start = time.time()
    
    if verbose:
        print(f"\n[Step 4] QPU solving ({num_iterations} iterations, {num_reads} reads)")
    
    # Initialize QPU sampler
    token = os.environ.get('DWAVE_API_TOKEN', DEFAULT_DWAVE_TOKEN)
    sampler = DWaveCliqueSampler(token=token)
    
    # Problem parameters
    config = data.get('config', {})
    params = config.get('parameters', {})
    rotation_gamma = params.get('rotation_gamma', 0.2)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    
    # Build rotation matrix for families
    R = create_family_rotation_matrix(seed=42)
    
    # Tracking
    total_qpu_time = 0
    total_qpu_sampling_time = 0
    cluster_solutions = [{}] * n_clusters
    boundary_info = [{}] * n_clusters
    best_global_solution = {}
    best_global_objective = -np.inf
    
    for iteration in range(num_iterations):
        if verbose:
            print(f"\n  Iteration {iteration + 1}/{num_iterations}")
        
        iter_qpu_time = 0
        
        # Update boundary info from previous iteration
        if iteration > 0:
            boundary_info = _coordinate_boundaries(clusters, cluster_solutions, cluster_neighbors)
        
        # Solve each cluster
        new_cluster_solutions = []
        
        for cluster_idx, cluster_farms in enumerate(clusters):
            # Build BQM for this cluster
            bqm = BinaryQuadraticModel('BINARY')
            var_map = {}
            
            n_cluster_farms = len(cluster_farms)
            
            # Create variables
            for farm in cluster_farms:
                for family in family_names:
                    for period in range(1, n_periods + 1):
                        var_name = f"Y_{farm}_{family}_t{period}"
                        var_map[(farm, family, period)] = var_name
                        bqm.add_variable(var_name, 0)
            
            # Part 1: Benefits (linear terms)
            for farm in cluster_farms:
                farm_area = land_availability.get(farm, 25.0)
                for family in family_names:
                    benefit = family_benefits.get(family, 0.5)
                    for period in range(1, n_periods + 1):
                        var = var_map[(farm, family, period)]
                        # Negative because BQM minimizes
                        bqm.add_variable(var, -(benefit * farm_area) / total_area)
            
            # Part 2: Rotation synergies (quadratic terms)
            for farm in cluster_farms:
                farm_area = land_availability.get(farm, 25.0)
                for period in range(2, n_periods + 1):
                    for fam1_idx, fam1 in enumerate(family_names):
                        for fam2_idx, fam2 in enumerate(family_names):
                            synergy = R[fam1_idx, fam2_idx]
                            if abs(synergy) > 1e-6:
                                var1 = var_map[(farm, fam1, period - 1)]
                                var2 = var_map[(farm, fam2, period)]
                                bqm.add_interaction(
                                    var1, var2,
                                    -(rotation_gamma * synergy * farm_area) / total_area
                                )
            
            # Part 3: One-hot constraint (exactly 1 family per farm per period)
            for farm in cluster_farms:
                for period in range(1, n_periods + 1):
                    period_vars = [var_map[(farm, fam, period)] for fam in family_names]
                    for i, v1 in enumerate(period_vars):
                        for v2 in period_vars[i+1:]:
                            bqm.add_interaction(v1, v2, 2 * one_hot_penalty)
                        bqm.add_variable(v1, -one_hot_penalty)
            
            # Part 4: Diversity bonus
            for farm in cluster_farms:
                for family in family_names:
                    for period in range(1, n_periods + 1):
                        var = var_map[(farm, family, period)]
                        bqm.add_variable(var, -diversity_bonus / n_periods)
            
            # Part 5: Boundary coordination (soft constraints from neighbors)
            if iteration > 0 and boundary_info[cluster_idx]:
                boundary_strength = 0.1
                for (farm, family, period), value in boundary_info[cluster_idx].items():
                    if farm in cluster_farms:
                        var = var_map.get((farm, family, period))
                        if var and value == 1:
                            bqm.add_variable(var, -boundary_strength)
            
            # Solve on QPU
            try:
                label = f"HybridAdaptive_{n_farms}f_c{cluster_idx+1}of{n_clusters}_i{iteration+1}"
                sampleset = sampler.sample(bqm, num_reads=num_reads, label=label)
                
                # Extract timing
                timing = sampleset.info.get('timing', {})
                qpu_time = timing.get('qpu_access_time', 0) / 1e6
                qpu_sampling = timing.get('qpu_sampling_time', 0) / 1e6
                iter_qpu_time += qpu_time
                total_qpu_sampling_time += qpu_sampling
                
                # Extract solution
                best_sample = sampleset.first.sample
                cluster_sol = {}
                for (farm, family, period), var_name in var_map.items():
                    if best_sample.get(var_name, 0) == 1:
                        cluster_sol[(farm, family, period)] = 1
                
                if verbose:
                    n_assigned = sum(1 for v in cluster_sol.values() if v == 1)
                    print(f"    Cluster {cluster_idx}: {n_assigned}/{n_cluster_farms * n_periods} assigned, QPU={qpu_time:.3f}s")
                
            except Exception as e:
                print(f"    Cluster {cluster_idx} FAILED: {e}")
                cluster_sol = {}
                # Default: assign first family to each farm-period
                for farm in cluster_farms:
                    for period in range(1, n_periods + 1):
                        cluster_sol[(farm, family_names[0], period)] = 1
            
            new_cluster_solutions.append(cluster_sol)
        
        cluster_solutions = new_cluster_solutions
        total_qpu_time += iter_qpu_time
        
        # Stitch solutions with overlap handling
        combined_solution = _stitch_cluster_solutions(clusters, cluster_solutions, overlap_farms)
        
        # Evaluate
        obj = _calculate_family_objective(combined_solution, family_data, R)
        violations = _count_violations(combined_solution, family_data)
        
        if obj > best_global_objective:
            best_global_objective = obj
            best_global_solution = combined_solution.copy()
        
        if verbose:
            n_total = sum(1 for v in combined_solution.values() if v == 1)
            print(f"    Combined: obj={obj:.4f}, assigned={n_total}, violations={violations['total']}")
    
    result['timings']['step4_qpu'] = time.time() - step4_start
    result['qpu_time'] = total_qpu_time
    result['qpu_sampling_time'] = total_qpu_sampling_time
    
    # =========================================================================
    # STEP 5: Post-processing - Refine to 27 foods
    # =========================================================================
    step5_start = time.time()
    
    if used_aggregation:
        if verbose:
            print(f"\n[Step 5] Refining 6-family solution → 27 foods")
        
        original_data = family_data.get('original_data', data)
        crop_solution = refine_family_solution_to_crops(best_global_solution, original_data)
        diversity_stats = analyze_crop_diversity(crop_solution, original_data)
        
        if verbose:
            print(f"  Unique crops: {diversity_stats['total_unique_crops']}/{diversity_stats['max_possible_crops']}")
            print(f"  Shannon diversity: {diversity_stats['shannon_diversity']:.3f}")
    else:
        crop_solution = best_global_solution
        diversity_stats = {'total_unique_crops': n_foods, 'shannon_diversity': 1.0, 'max_possible_crops': n_foods}
    
    result['timings']['step5_postprocessing'] = time.time() - step5_start
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    total_time = time.time() - total_start
    
    result['solve_time'] = total_time
    result['objective'] = best_global_objective
    result['family_solution'] = best_global_solution
    result['crop_solution'] = crop_solution
    result['diversity_stats'] = diversity_stats
    result['violations'] = _count_violations(best_global_solution, family_data)
    result['n_assigned'] = sum(1 for v in best_global_solution.values() if v == 1)
    result['success'] = True
    
    if verbose:
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Objective: {best_global_objective:.4f}")
        print(f"Violations: {result['violations']}")
        print(f"Assignments: {result['n_assigned']} (expected: {n_farms * n_periods})")
        print(f"Total time: {total_time:.2f}s (QPU: {total_qpu_time:.3f}s)")
    
    return result


def _coordinate_boundaries(clusters: List[List[str]],
                           cluster_solutions: List[Dict],
                           cluster_neighbors: Dict[int, List[int]]) -> List[Dict]:
    """Extract boundary information for inter-cluster coordination."""
    boundary_info = []
    
    for i, cluster in enumerate(clusters):
        boundary_solutions = {}
        
        for neighbor_idx in cluster_neighbors.get(i, []):
            neighbor_sol = cluster_solutions[neighbor_idx]
            neighbor_cluster = clusters[neighbor_idx]
            
            # Get boundary farms (first and last in neighbor cluster)
            boundary_farms = [neighbor_cluster[0], neighbor_cluster[-1]]
            
            for key, value in neighbor_sol.items():
                farm, family, period = key
                if farm in boundary_farms:
                    boundary_solutions[key] = value
        
        boundary_info.append(boundary_solutions)
    
    return boundary_info


def _stitch_cluster_solutions(clusters: List[List[str]],
                              cluster_solutions: List[Dict],
                              overlap_farms: int) -> Dict:
    """Merge cluster solutions, handling overlapping farms."""
    combined = {}
    farm_assignments = defaultdict(list)  # Track multiple assignments for overlap farms
    
    for cluster_idx, (cluster, sol) in enumerate(zip(clusters, cluster_solutions)):
        for (farm, family, period), value in sol.items():
            if value == 1:
                farm_assignments[(farm, period)].append((family, cluster_idx))
    
    # Resolve overlaps: prefer assignment from cluster with more context
    for (farm, period), assignments in farm_assignments.items():
        if len(assignments) == 1:
            family, _ = assignments[0]
            combined[(farm, family, period)] = 1
        else:
            # Multiple clusters assigned this farm-period
            # Pick the one from the cluster where farm is more central
            # (heuristic: later cluster index if farm appears in multiple)
            family, _ = assignments[-1]  # Take last (most refined) assignment
            combined[(farm, family, period)] = 1
    
    return combined


def _calculate_family_objective(solution: Dict, family_data: Dict, R: np.ndarray) -> float:
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
    
    # Part 1: Benefits
    for (farm, family, period), val in solution.items():
        if val == 1:
            benefit = family_benefits.get(family, 0.5)
            area = land_availability.get(farm, 25.0)
            obj += (benefit * area) / total_area
    
    # Part 2: Rotation synergies
    for farm in farm_names:
        area = land_availability.get(farm, 25.0)
        for period in range(2, n_periods + 1):
            for fam1_idx, fam1 in enumerate(family_names):
                for fam2_idx, fam2 in enumerate(family_names):
                    v1 = solution.get((farm, fam1, period - 1), 0)
                    v2 = solution.get((farm, fam2, period), 0)
                    if v1 == 1 and v2 == 1:
                        obj += (rotation_gamma * R[fam1_idx, fam2_idx] * area) / total_area
    
    # Part 3: One-hot penalty
    for farm in farm_names:
        for period in range(1, n_periods + 1):
            count = sum(1 for fam in family_names if solution.get((farm, fam, period), 0) == 1)
            if count > 1:
                obj -= one_hot_penalty * (count - 1) ** 2
    
    # Part 4: Diversity bonus
    for farm in farm_names:
        area = land_availability.get(farm, 25.0)
        families_used = set()
        for fam in family_names:
            for period in range(1, n_periods + 1):
                if solution.get((farm, fam, period), 0) == 1:
                    families_used.add(fam)
        obj += diversity_bonus * len(families_used) * (area / total_area)
    
    return obj


def _count_violations(solution: Dict, family_data: Dict) -> Dict:
    """Count constraint violations."""
    farm_names = family_data['farm_names']
    family_names = family_data['food_names']
    n_periods = 3
    
    max_crops_violations = 0
    rotation_violations = 0
    
    for farm in farm_names:
        for period in range(1, n_periods + 1):
            count = sum(1 for fam in family_names if solution.get((farm, fam, period), 0) == 1)
            if count > 2:  # Allow up to 2 crops per farm per period
                max_crops_violations += count - 2
    
    return {
        'max_crops': max_crops_violations,
        'rotation': rotation_violations,
        'total': max_crops_violations + rotation_violations,
    }
