"""
Patch for solver_runner_LQ.py to use optimized synergy computation.

This file shows the exact changes needed to integrate SynergyOptimizer
for ~10-100x speedup in quadratic synergy bonus computation.
"""

# ==============================================================================
# CHANGE 1: Add import at the top of the file (after other imports)
# ==============================================================================

# Add after: from tqdm import tqdm

# Try to import Cython version first, fallback to pure Python
try:
    from synergy_optimizer import SynergyOptimizer
    SYNERGY_OPTIMIZER_TYPE = "Cython"
except ImportError:
    try:
        from src.synergy_optimizer_pure import SynergyOptimizer
        SYNERGY_OPTIMIZER_TYPE = "NumPy"
    except ImportError:
        # If neither is available, use None and fall back to original code
        SynergyOptimizer = None
        SYNERGY_OPTIMIZER_TYPE = "Original"


# ==============================================================================
# CHANGE 2: In create_cqm() function - Replace quadratic synergy loop
# ==============================================================================

# FIND THIS CODE (lines ~179-188):
"""
    # Objective function - Quadratic synergy bonus
    pbar.set_description("Adding quadratic synergy bonus")
    for farm in farms:
        # Iterate through synergy matrix
        for crop1, pairs in synergy_matrix.items():
            if crop1 in foods:
                for crop2, boost_value in pairs.items():
                    if crop2 in foods and crop1 < crop2:  # Avoid double counting
                        objective += synergy_bonus_weight * boost_value * Y[(farm, crop1)] * Y[(farm, crop2)]
                        pbar.update(1)
"""

# REPLACE WITH:
"""
    # Objective function - Quadratic synergy bonus
    pbar.set_description(f"Adding quadratic synergy bonus ({SYNERGY_OPTIMIZER_TYPE})")
    
    if SynergyOptimizer is not None:
        # OPTIMIZED: Use precomputed synergy pairs (~10-100x faster)
        optimizer = SynergyOptimizer(synergy_matrix, foods)
        objective += optimizer.build_synergy_terms_dimod(farms, Y, synergy_bonus_weight)
        pbar.update(optimizer.get_n_pairs() * len(farms))
    else:
        # FALLBACK: Original nested loop (slower but works without optimizer)
        for farm in farms:
            for crop1, pairs in synergy_matrix.items():
                if crop1 in foods:
                    for crop2, boost_value in pairs.items():
                        if crop2 in foods and crop1 < crop2:
                            objective += synergy_bonus_weight * boost_value * Y[(farm, crop1)] * Y[(farm, crop2)]
                            pbar.update(1)
"""


# ==============================================================================
# CHANGE 3: In solve_with_pulp() function - Replace synergy pairs building
# ==============================================================================

# FIND THIS CODE (lines ~322-333):
"""
    Z_pulp = {}
    synergy_pairs = []
    for f in farms:
        for crop1, pairs in synergy_matrix.items():
            if crop1 in foods:
                for crop2, boost_value in pairs.items():
                    if crop2 in foods and crop1 < crop2:  # Avoid double counting
                        Z_pulp[(f, crop1, crop2)] = pl.LpVariable(
                            f"Z_{f}_{crop1}_{crop2}", 
                            cat='Binary'
                        )
                        synergy_pairs.append((f, crop1, crop2, boost_value))
"""

# REPLACE WITH:
"""
    # Build synergy pairs for McCormick linearization
    if SynergyOptimizer is not None:
        # OPTIMIZED: Use precomputed synergy pairs
        optimizer = SynergyOptimizer(synergy_matrix, foods)
        synergy_pairs = optimizer.build_synergy_pairs_list(farms)
    else:
        # FALLBACK: Original nested loop
        synergy_pairs = []
        for f in farms:
            for crop1, pairs in synergy_matrix.items():
                if crop1 in foods:
                    for crop2, boost_value in pairs.items():
                        if crop2 in foods and crop1 < crop2:
                            synergy_pairs.append((f, crop1, crop2, boost_value))
    
    # Create Z variables for all synergy pairs
    Z_pulp = {}
    for f, crop1, crop2, boost_value in synergy_pairs:
        Z_pulp[(f, crop1, crop2)] = pl.LpVariable(
            f"Z_{f}_{crop1}_{crop2}", 
            cat='Binary'
        )
"""


# ==============================================================================
# CHANGE 4: In solve_with_pyomo() function - Replace synergy loop
# ==============================================================================

# FIND THIS CODE (lines ~694-700):
"""
        # Quadratic synergy bonus
        for f in m.farms:
            for crop1, pairs in synergy_matrix.items():
                if crop1 in foods:
                    for crop2, boost_value in pairs.items():
                        if crop2 in foods and crop1 < crop2:  # Avoid double counting
                            obj += synergy_bonus_weight * boost_value * m.Y[f, crop1] * m.Y[f, crop2]
"""

# REPLACE WITH (OPTIONAL - Pyomo is already quite fast):
"""
        # Quadratic synergy bonus
        if SynergyOptimizer is not None:
            # OPTIMIZED: Use precomputed synergy pairs
            optimizer = SynergyOptimizer(synergy_matrix, foods)
            for crop1, crop2, boost_value in optimizer.iter_pairs_with_names():
                for f in m.farms:
                    obj += synergy_bonus_weight * boost_value * m.Y[f, crop1] * m.Y[f, crop2]
        else:
            # FALLBACK: Original nested loop
            for f in m.farms:
                for crop1, pairs in synergy_matrix.items():
                    if crop1 in foods:
                        for crop2, boost_value in pairs.items():
                            if crop2 in foods and crop1 < crop2:
                                obj += synergy_bonus_weight * boost_value * m.Y[f, crop1] * m.Y[f, crop2]
"""


# ==============================================================================
# SUMMARY OF CHANGES
# ==============================================================================

print("""
SUMMARY: 4 changes to integrate SynergyOptimizer into solver_runner_LQ.py

1. Add import at top of file (with try/except for graceful fallback)
2. Replace create_cqm() synergy loop with optimizer.build_synergy_terms_dimod()
3. Replace solve_with_pulp() synergy pairs building with optimizer.build_synergy_pairs_list()
4. (Optional) Replace solve_with_pyomo() synergy loop with optimizer.iter_pairs_with_names()

Expected speedup:
- create_cqm(): 10-100x faster (most critical)
- solve_with_pulp(): 5-20x faster for pair generation
- solve_with_pyomo(): 2-5x faster (less critical as Pyomo handles it well)

To compile Cython version:
    pip install cython
    python setup_synergy.py build_ext --inplace

The code will automatically use:
1. Cython version if available (fastest)
2. Pure Python NumPy version if Cython not compiled (medium speed)
3. Original nested loops if optimizer not available (slowest, but works)
""")
