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

Author: OQI-UC002-DWave
Date: 2025-12-12
"""

import numpy as np
from typing import Dict, List, Tuple
from food_grouping import FOOD_TO_FAMILY, get_family, FAMILY_ORDER

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

def detect_decomposition_strategy(n_farms: int, n_foods: int, n_periods: int = 3) -> Dict:
    """
    Auto-detect which decomposition strategy to use based on problem size.
    
    Decision tree:
    1. If n_vars <= 450: No decomposition (direct QPU solve)
    2. If 450 < n_vars <= 1800: Spatial decomposition (cluster farms)
    3. If n_vars > 1800: Hierarchical with family-level reduction
    
    Args:
        n_farms: Number of farms
        n_foods: Number of foods/families
        n_periods: Number of rotation periods
    
    Returns:
        Dict with strategy recommendations
    """
    n_vars = n_farms * n_foods * n_periods
    
    strategy = {
        'n_vars': n_vars,
        'n_farms': n_farms,
        'n_foods': n_foods,
        'n_periods': n_periods,
    }
    
    if n_vars <= 450:
        # Small: Direct solve
        strategy['method'] = 'direct'
        strategy['use_decomposition'] = False
        strategy['description'] = 'Direct QPU solve (problem fits in clique)'
        strategy['farms_per_cluster'] = n_farms
        
    elif n_vars <= 1800:
        # Medium: Spatial decomposition
        strategy['method'] = 'spatial_decomposition'
        strategy['use_decomposition'] = True
        strategy['description'] = 'Spatial decomposition (cluster farms, keep all foods)'
        
        # Calculate optimal cluster size (target: ~180-300 vars per cluster)
        target_vars_per_cluster = 270
        farms_per_cluster = max(5, min(15, target_vars_per_cluster // (n_foods * n_periods)))
        strategy['farms_per_cluster'] = farms_per_cluster
        strategy['n_clusters'] = (n_farms + farms_per_cluster - 1) // farms_per_cluster
        
    else:
        # Large: Hierarchical with aggregation
        strategy['method'] = 'hierarchical_aggregation'
        strategy['use_decomposition'] = True
        strategy['description'] = 'Hierarchical: aggregate foods → families, then spatial decomposition'
        strategy['aggregate_to_families'] = True
        strategy['n_families'] = 6
        
        # After aggregation: n_vars = n_farms × 6 × 3
        n_vars_aggregated = n_farms * 6 * n_periods
        farms_per_cluster = max(5, min(15, 270 // (6 * n_periods)))
        strategy['farms_per_cluster'] = farms_per_cluster
        strategy['n_clusters'] = (n_farms + farms_per_cluster - 1) // farms_per_cluster
        strategy['n_vars_aggregated'] = n_vars_aggregated
    
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
        
    elif strategy['method'] == 'spatial_decomposition':
        params['num_reads'] = 100
        params['num_iterations'] = 3  # Boundary coordination
        params['timeout'] = 600
        params['use_qpu'] = True
        
    else:  # hierarchical_aggregation
        params['num_reads'] = 100
        params['num_iterations'] = 3
        params['timeout'] = 900
        params['use_qpu'] = True
        params['enable_post_processing'] = True
    
    return params


# ============================================================================
# EXAMPLE USAGE
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
    
    print()
    print("="*80)
    print("KEY BENEFITS:")
    print("="*80)
    print("""
    ✅ Full 27-food variable space (no aggregation loss)
    ✅ Simplified 6-family synergy structure (tractable)
    ✅ Consistent formulation across all sizes
    ✅ Auto-detection picks optimal strategy
    ✅ Foods within same family have similar (but not identical) synergies
    ✅ Enables fair comparison across all problem sizes
    """)
