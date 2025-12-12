"""
Fast BQM builder wrapper using Cython-optimized C++ code.

Provides drop-in replacement for Python BQM building with 10-100x speedup.
"""

import numpy as np
from dimod import BinaryQuadraticModel
try:
    from src.fast_rotation_bqm import FastRotationBQM
    HAS_FAST_BQM = True
except ImportError:
    HAS_FAST_BQM = False
    print("Warning: fast_rotation_bqm not available. Install with:")
    print("  cd src && python setup_fast_rotation.py build_ext --inplace")


def build_rotation_bqm_fast(farm_names, families, land_availability, total_area,
                             food_benefits, rotation_matrix, spatial_edges=None,
                             rotation_gamma=0.2, diversity_bonus=0.15, one_hot_penalty=3.0,
                             n_periods=3, vartype='BINARY'):
    """
    Build rotation BQM using fast Cython implementation.
    
    Args:
        farm_names: List of farm names
        families: List of family/crop names  
        land_availability: Dict mapping farm -> land area
        total_area: Total land area
        food_benefits: Dict mapping family -> benefit value
        rotation_matrix: numpy array (n_families × n_families) with rotation synergies
        spatial_edges: List of (farm1, farm2) tuples for spatial neighbors
        rotation_gamma: Weight for rotation synergies
        diversity_bonus: Weight for diversity bonus
        one_hot_penalty: Penalty weight for one-hot constraint
        n_periods: Number of rotation periods (default: 3)
        vartype: 'BINARY' or 'SPIN'
        
    Returns:
        dimod.BinaryQuadraticModel
    """
    if not HAS_FAST_BQM:
        raise ImportError("fast_rotation_bqm extension not available")
    
    # Build with fast C++ implementation
    builder = FastRotationBQM(farm_names, families, rotation_matrix, spatial_edges, n_periods)
    bqm_dict = builder.build_bqm(
        land_availability, total_area, food_benefits,
        rotation_gamma, diversity_bonus, one_hot_penalty
    )
    
    # Convert to dimod BQM
    # Note: We maximize, so negate for dimod (which minimizes by default)
    linear = {k: -v for k, v in bqm_dict['linear'].items()}
    quadratic = {k: -v for k, v in bqm_dict['quadratic'].items()}
    offset = -bqm_dict['offset']
    
    bqm = BinaryQuadraticModel(linear, quadratic, offset, vartype=vartype)
    
    return bqm, bqm_dict['var_map']


def benchmark_bqm_building():
    """
    Benchmark fast vs slow BQM building.
    """
    import time
    from dimod import Binary
    
    print("Benchmarking BQM building speed...")
    print("="*60)
    
    # Test problem: 100 farms × 6 families × 3 periods = 1800 variables
    n_farms = 100
    n_families = 6
    n_periods = 3
    
    farm_names = [f"farm_{i}" for i in range(n_farms)]
    families = [f"family_{i}" for i in range(n_families)]
    land_availability = {f: 10.0 for f in farm_names}
    total_area = sum(land_availability.values())
    food_benefits = {f: 0.5 for f in families}
    
    # Create rotation matrix with frustration
    np.random.seed(42)
    R = np.zeros((n_families, n_families))
    for i in range(n_families):
        for j in range(n_families):
            if i == j:
                R[i, j] = -0.8 * 1.5
            elif np.random.random() < 0.7:
                R[i, j] = np.random.uniform(-0.96, -0.24)
            else:
                R[i, j] = np.random.uniform(0.02, 0.20)
    
    # Create spatial edges
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {farm: (i // side, i % side) for i, farm in enumerate(farm_names)}
    spatial_edges = []
    for f1 in farm_names:
        distances = []
        for f2 in farm_names:
            if f1 != f2:
                dist = np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                             (positions[f1][1] - positions[f2][1])**2)
                distances.append((dist, f2))
        distances.sort()
        for _, f2 in distances[:4]:
            if (f2, f1) not in spatial_edges:
                spatial_edges.append((f1, f2))
    
    # Test fast implementation
    if HAS_FAST_BQM:
        start = time.time()
        bqm_fast, var_map = build_rotation_bqm_fast(
            farm_names, families, land_availability, total_area,
            food_benefits, R, spatial_edges
        )
        fast_time = time.time() - start
        print(f"Fast (Cython): {fast_time:.4f}s")
        print(f"  Variables: {len(bqm_fast.variables)}")
        print(f"  Linear terms: {len(bqm_fast.linear)}")
        print(f"  Quadratic terms: {len(bqm_fast.quadratic)}")
    else:
        print("Fast implementation not available")
        return
    
    # Test slow implementation (pure Python) - FULL VERSION
    start = time.time()
    bqm_slow = BinaryQuadraticModel('BINARY')
    var_map_slow = {}
    
    # Create variables
    for farm in farm_names:
        for family in families:
            for t in range(1, n_periods + 1):
                var_name = f"Y_{farm}_{family}_t{t}"
                var_map_slow[(farm, family, t)] = var_name
                bqm_slow.add_variable(var_name, 0.0)
    
    rotation_gamma = 0.2
    diversity_bonus = 0.15
    one_hot_penalty = 3.0
    
    # Part 1: Base benefits
    for farm in farm_names:
        area_frac = land_availability[farm] / total_area
        for c_idx, family in enumerate(families):
            benefit = food_benefits[family]
            for t in range(1, n_periods + 1):
                var = var_map_slow[(farm, family, t)]
                bqm_slow.set_linear(var, bqm_slow.get_linear(var) - benefit * area_frac)
    
    # Part 2: Temporal synergies
    for farm in farm_names:
        area_frac = land_availability[farm] / total_area
        for t in range(2, n_periods + 1):
            for c1_idx, c1 in enumerate(families):
                for c2_idx, c2 in enumerate(families):
                    synergy = R[c1_idx, c2_idx]
                    if abs(synergy) > 1e-6:
                        var1 = var_map_slow[(farm, c1, t-1)]
                        var2 = var_map_slow[(farm, c2, t)]
                        bqm_slow.add_quadratic(var1, var2, -rotation_gamma * synergy * area_frac)
    
    # Part 3: Spatial synergies
    spatial_gamma = rotation_gamma * 0.5
    for f1, f2 in spatial_edges:
        for t in range(1, n_periods + 1):
            for c1_idx, c1 in enumerate(families):
                for c2_idx, c2 in enumerate(families):
                    spatial_synergy = R[c1_idx, c2_idx] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        var1 = var_map_slow[(f1, c1, t)]
                        var2 = var_map_slow[(f2, c2, t)]
                        bqm_slow.add_quadratic(var1, var2, -spatial_gamma * spatial_synergy)
    
    # Part 4: Diversity bonus
    for farm in farm_names:
        for family in families:
            for t in range(1, n_periods + 1):
                var = var_map_slow[(farm, family, t)]
                bqm_slow.set_linear(var, bqm_slow.get_linear(var) - diversity_bonus / n_periods)
    
    # Part 5: One-hot penalty
    for farm in farm_names:
        for t in range(1, n_periods + 1):
            for c_idx, family in enumerate(families):
                var = var_map_slow[(farm, family, t)]
                bqm_slow.set_linear(var, bqm_slow.get_linear(var) + 2.0 * one_hot_penalty)
            for c1_idx, c1 in enumerate(families):
                for c2_idx in range(c1_idx + 1, n_families):
                    c2 = families[c2_idx]
                    var1 = var_map_slow[(farm, c1, t)]
                    var2 = var_map_slow[(farm, c2, t)]
                    bqm_slow.add_quadratic(var1, var2, -2.0 * one_hot_penalty)
    
    slow_time = time.time() - start
    print(f"Slow (Python): {slow_time:.4f}s")
    print(f"  Variables: {len(bqm_slow.variables)}")
    print(f"  Linear terms: {len(bqm_slow.linear)}")
    print(f"  Quadratic terms: {len(bqm_slow.quadratic)}")
    
    print(f"\nSpeedup: {slow_time / fast_time:.1f}x faster")
    print("="*60)


if __name__ == '__main__':
    benchmark_bqm_building()
