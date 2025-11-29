#!/usr/bin/env python3
"""
Comprehensive Embedding and Solving Benchmark v3.0

Systematically tests ALL combinations of:
- Problem sizes (5, 10, 15, 20, 25, 30 farms)
- Formulations (CQM, BQM from CQM, Sparse Direct BQM)
- Decomposition strategies:
  * None (direct embedding)
  * Louvain community detection
  * Plot-based domain decomposition
  * Multilevel coarsening (ML-QLS style)
  * Sequential cut-set reduction
  * Energy-impact (dwave-hybrid)

For each configuration:
1. Build formulation using REAL food data with nutritional weights
2. Apply decomposition (if any)
3. Attempt embedding (300s timeout, continue even if fails)
4. Solve with Gurobi (always, regardless of embedding)
5. Calculate ACTUAL objectives from solutions (not BQM energy)
6. Calculate total times (summing partitions for decomposed)

Outputs:
- JSON (detailed results)
- CSV (flattened for analysis)
- Markdown (human-readable summary)

Author: Generated for OQI-UC002-DWave comprehensive benchmark
Date: 2025-11-28 (v3.0 - real formulation with nutritional weights)
"""

import os
import sys
import time
import json
import csv
import statistics
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=" * 80)
print("COMPREHENSIVE EMBEDDING AND SOLVING BENCHMARK v2.0")
print("=" * 80)
print("Testing: Formulations × Decompositions × Solvers")
print("=" * 80)

# Imports
print("\n[1/6] Importing libraries...")
import_start = time.time()

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, Real, cqm_to_bqm
from dimod.generators import combinations
import minorminer

# Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("  Warning: Gurobi not available. Classical solving disabled.")

# D-Wave
try:
    from dwave.system import DWaveSampler
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    print("  Warning: DWaveSampler not available. Using Pegasus simulation.")

# Hybrid
try:
    from hybrid.decomposers import EnergyImpactDecomposer
    from hybrid.core import State
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    print("  Warning: dwave-hybrid not available.")

# Louvain
try:
    from networkx.algorithms.community import louvain_communities
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("  Warning: networkx louvain not available.")

# Advanced decomposition strategies
try:
    from advanced_decomposition_strategies import (
        decompose_multilevel,
        decompose_sequential_cutset,
        decompose_spatial_grid,
        analyze_decomposition_quality
    )
    ADVANCED_DECOMP_AVAILABLE = True
except ImportError:
    ADVANCED_DECOMP_AVAILABLE = False
    print("  Warning: Advanced decomposition strategies not available.")

# Real data loading
try:
    from src.scenarios import load_food_data
    from Utils.farm_sampler import generate_farms
    from Utils.patch_sampler import generate_grid
    REAL_DATA_AVAILABLE = True
    print("  [OK] Real data loaders available (scenarios, farm_sampler, patch_sampler)")
except ImportError as e:
    REAL_DATA_AVAILABLE = False
    print(f"  Warning: Real data loading not available: {e}")

# Import create_cqm_plots from solver_runner_BINARY for proper CQM formulation
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "solver_runner_BINARY", 
        os.path.join(os.path.dirname(__file__), '..', 'Benchmark Scripts', 'solver_runner_BINARY.py')
    )
    solver_runner_BINARY = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver_runner_BINARY)
    create_cqm_plots_original = solver_runner_BINARY.create_cqm_plots
    CREATE_CQM_PLOTS_AVAILABLE = True
    print("  [OK] create_cqm_plots available from solver_runner_BINARY")
except Exception as e:
    CREATE_CQM_PLOTS_AVAILABLE = False
    print(f"  Warning: create_cqm_plots not available: {e}")

print(f"  [OK] Imports done in {time.time() - import_start:.2f}s")

# Configuration
PROBLEM_SIZES = [25]  # Testing 25 farms only
N_FOODS = 27
EMBEDDING_TIMEOUT = 30  # 30 seconds per partition (reduced to avoid hanging)
SOLVE_TIMEOUT = 30  # 30 seconds for Gurobi per partition (focus on embedding, not solving)
SKIP_DENSE_EMBEDDING = True  # Skip embedding for CQM and dense BQM (None decomp)

# Formulation types to test
FORMULATIONS = ["CQM", "BQM"]

# Decomposition strategies to test
# IMPORTANT: Graph-based decomposition does NOT produce correct objectives
# because constraints span partitions. This benchmark focuses on EMBEDDING PERFORMANCE.
#
# For correct objectives, use:
#   - CQM with None decomposition (handled by D-Wave hybrid or Gurobi)
#
# Decomposition is useful for:
#   - Studying embedding feasibility and timing
#   - Creating smaller subproblems that CAN embed on QPU
#   - The solutions won't be globally optimal due to constraint violations
#
# Removed: Cutset (creates too many partitions, hangs), SpatialGrid (1 partition)
DECOMPOSITIONS = ["None", "Louvain", "PlotBased"]

# NOTE: For correct objective values, only use None decomposition
VALID_DECOMPOSITIONS_FOR_OBJECTIVE = ["None"]

# Output directory
OUTPUT_DIR = Path(__file__).parent / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# REAL DATA LOADING
# =============================================================================

# Global caches for real data (loaded once)
_CACHED_FOODS = None
_CACHED_FOOD_GROUPS = None
_CACHED_BASE_CONFIG = None
_CACHED_WEIGHTS = None


def load_real_data():
    """Load real food data from scenarios (cached)"""
    global _CACHED_FOODS, _CACHED_FOOD_GROUPS, _CACHED_BASE_CONFIG, _CACHED_WEIGHTS
    
    if _CACHED_FOODS is not None:
        return _CACHED_FOODS, _CACHED_FOOD_GROUPS, _CACHED_BASE_CONFIG, _CACHED_WEIGHTS
    
    if not REAL_DATA_AVAILABLE:
        raise RuntimeError("Real data loading not available - check imports")
    
    print("  Loading real food data from 'full_family' scenario...")
    _, foods, food_groups, base_config = load_food_data('full_family')
    
    weights = base_config['parameters'].get('weights', {
        'nutritional_value': 0.3,
        'nutrient_density': 0.2,
        'environmental_impact': 0.2,
        'affordability': 0.15,
        'sustainability': 0.15
    })
    
    _CACHED_FOODS = foods
    _CACHED_FOOD_GROUPS = food_groups
    _CACHED_BASE_CONFIG = base_config
    _CACHED_WEIGHTS = weights
    
    print(f"    Loaded {len(foods)} foods, {len(food_groups)} food groups")
    print(f"    Weights: {weights}")
    
    return foods, food_groups, base_config, weights


def generate_land_data(n_units: int, total_land: float = 100.0, seed: int = 42) -> Dict[str, float]:
    """Generate land availability data using patch_sampler"""
    if not REAL_DATA_AVAILABLE:
        # Fallback to simple even distribution
        unit_names = [f"Unit_{i}" for i in range(n_units)]
        land_per_unit = total_land / n_units
        return {name: land_per_unit for name in unit_names}
    
    patches = generate_grid(n_farms=n_units, area=total_land, seed=seed)
    return patches


def create_real_config(land_data: Dict[str, float], base_config: Dict) -> Dict:
    """Create configuration for CQM building"""
    total_land = sum(land_data.values())
    max_percentage = base_config['parameters'].get('max_percentage_per_crop', {})
    maximum_planting_area = {crop: max_pct * total_land for crop, max_pct in max_percentage.items()}
    
    return {
        'parameters': {
            'land_availability': land_data,
            'minimum_planting_area': base_config['parameters'].get('minimum_planting_area', {}),
            'maximum_planting_area': maximum_planting_area,
            'food_group_constraints': base_config['parameters'].get('food_group_constraints', {}),
            'weights': base_config['parameters'].get('weights', {}),
        }
    }


def calculate_actual_objective(solution: Dict, foods: Dict, weights: Dict, 
                                land_data: Dict, unit_names: List[str]) -> float:
    """
    Calculate the ACTUAL objective value from a solution (not BQM energy).
    
    The real objective is:
    sum over units and foods of:
        weights['nutritional_value'] * foods[food]['nutritional_value'] * area +
        weights['nutrient_density'] * foods[food]['nutrient_density'] * area -
        weights['environmental_impact'] * foods[food]['environmental_impact'] * area +
        weights['affordability'] * foods[food]['affordability'] * area +
        weights['sustainability'] * foods[food]['sustainability'] * area
    
    Normalized by total land area.
    """
    total_land = sum(land_data.values())
    if total_land == 0:
        return 0.0
    
    actual_objective = 0.0
    food_names = list(foods.keys())
    
    for unit_idx, unit in enumerate(unit_names):
        unit_area = land_data.get(unit, 0)
        
        for food_idx, food in enumerate(food_names):
            # Check for Y variable (binary selection)
            y_key = f"Y_{unit_idx}_{food_idx}"
            y_val = solution.get(y_key, 0)
            
            # Also check for A variable (area) if available
            a_key = f"A_{unit_idx}_{food_idx}"
            a_val = solution.get(a_key, 0)
            
            # Use area if available, otherwise use unit_area * y_val
            if a_val > 0:
                effective_area = a_val
            elif y_val > 0.5:
                effective_area = unit_area  # Assume full allocation if selected
            else:
                effective_area = 0
            
            if effective_area > 0:
                food_data = foods[food]
                crop_value = (
                    weights.get('nutritional_value', 0) * food_data.get('nutritional_value', 0) +
                    weights.get('nutrient_density', 0) * food_data.get('nutrient_density', 0) -
                    weights.get('environmental_impact', 0) * food_data.get('environmental_impact', 0) +
                    weights.get('affordability', 0) * food_data.get('affordability', 0) +
                    weights.get('sustainability', 0) * food_data.get('sustainability', 0)
                )
                actual_objective += effective_area * crop_value
    
    # Normalize by total land
    return actual_objective / total_land


def calculate_actual_objective_from_bqm_solution(solution: Dict, metadata: Dict, n_units: int) -> Optional[float]:
    """
    Calculate the ACTUAL objective from a BQM solution using metadata.
    
    BQM energy includes penalty terms, so we need to recalculate
    using the original nutritional weights.
    
    Handles both variable naming formats:
    - Y_{idx}_{idx} (fallback formulation)
    - Y_{unit_name}_{food_name} (create_cqm_plots format)
    """
    if not solution:
        return None
    
    # Try to get data from metadata
    if "weights" not in metadata or "total_land" not in metadata:
        # Metadata not available, try to load from cache
        try:
            foods, _, _, weights = load_real_data()
            food_names = metadata.get("food_names", list(foods.keys())[:N_FOODS])
        except Exception:
            return None
    else:
        weights = metadata["weights"]
        food_names = metadata.get("food_names", [])
        if not food_names:
            try:
                foods, _, _, _ = load_real_data()
                food_names = list(foods.keys())[:N_FOODS]
            except Exception:
                return None
    
    total_land = metadata.get("total_land", 100.0)
    
    # Load food data
    try:
        foods, _, _, _ = load_real_data()
    except Exception:
        return None
    
    # Generate land data (same seed as formulation builder)
    land_data = generate_land_data(n_units, total_land=total_land)
    unit_names = metadata.get("unit_names", list(land_data.keys()))
    
    actual_objective = 0.0
    
    for p_idx, unit in enumerate(unit_names):
        unit_area = land_data.get(unit, 0)
        
        for c_idx, food in enumerate(food_names):
            # Check for Y variable with multiple naming formats
            # Format 1: Y_{idx}_{idx} (numeric indices)
            y_key_idx = f"Y_{p_idx}_{c_idx}"
            # Format 2: Y_{unit_name}_{food_name} (string names from create_cqm_plots)
            y_key_name = f"Y_{unit}_{food}"
            
            y_val = solution.get(y_key_idx, solution.get(y_key_name, 0))
            
            if y_val > 0.5:  # Selected
                food_data = foods.get(food, {})
                crop_value = (
                    weights.get('nutritional_value', 0) * food_data.get('nutritional_value', 0) +
                    weights.get('nutrient_density', 0) * food_data.get('nutrient_density', 0) -
                    weights.get('environmental_impact', 0) * food_data.get('environmental_impact', 0) +
                    weights.get('affordability', 0) * food_data.get('affordability', 0) +
                    weights.get('sustainability', 0) * food_data.get('sustainability', 0)
                )
                actual_objective += unit_area * crop_value
    
    # Normalize by total land
    return actual_objective / total_land if total_land > 0 else 0.0


# =============================================================================
# FORMULATION BUILDERS
# =============================================================================

def build_farm_cqm(n_farms: int, n_foods: int = 27) -> Tuple[ConstrainedQuadraticModel, Dict]:
    """Build Farm CQM with continuous areas + binary selections using REAL food data"""
    print(f"    Building Farm CQM ({n_farms} farms, {n_foods} foods) with REAL data...")
    
    # Load real data
    foods, food_groups, base_config, weights = load_real_data()
    land_data = generate_land_data(n_farms, total_land=100.0)
    config = create_real_config(land_data, base_config)
    
    unit_names = list(land_data.keys())
    food_names = list(foods.keys())[:n_foods]  # Limit to n_foods
    
    cqm = ConstrainedQuadraticModel()
    metadata = {
        "type": "Farm_CQM", 
        "n_farms": n_farms, 
        "n_foods": len(food_names),
        "unit_names": unit_names,
        "food_names": food_names,
        "weights": weights,
        "total_land": sum(land_data.values())
    }
    
    # Variables
    A = {}  # Continuous area
    Y = {}  # Binary selection
    for f_idx, farm in enumerate(unit_names):
        for c_idx, food in enumerate(food_names):
            land_avail = land_data[farm]
            A[f_idx, c_idx] = Real(f"A_{f_idx}_{c_idx}", lower_bound=0, upper_bound=land_avail)
            Y[f_idx, c_idx] = Binary(f"Y_{f_idx}_{c_idx}")
    
    # Objective: weighted nutritional value (REAL formulation)
    objective = 0
    for f_idx, farm in enumerate(unit_names):
        for c_idx, food in enumerate(food_names):
            food_data = foods[food]
            crop_coeff = (
                weights.get('nutritional_value', 0) * food_data.get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * food_data.get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * food_data.get('environmental_impact', 0) +
                weights.get('affordability', 0) * food_data.get('affordability', 0) +
                weights.get('sustainability', 0) * food_data.get('sustainability', 0)
            )
            objective += crop_coeff * A[f_idx, c_idx]
    
    cqm.set_objective(-objective)  # Minimize negative = maximize
    
    # Constraints
    min_area = 0.0001
    
    # 1. Land availability per farm (use REAL land data)
    for f_idx, farm in enumerate(unit_names):
        land_avail = land_data[farm]
        cqm.add_constraint(
            sum(A[f_idx, c] for c in range(len(food_names))) <= land_avail,
            label=f"land_capacity_farm_{f_idx}"
        )
    
    # 2. Min area if selected (A >= min_area * Y => A - min_area * Y >= 0)
    for f_idx in range(n_farms):
        for c_idx in range(len(food_names)):
            cqm.add_constraint(A[f_idx, c_idx] - min_area * Y[f_idx, c_idx] >= 0, 
                             label=f"min_area_{f_idx}_{c_idx}")
    
    # 3. Max area if selected (A <= land_avail * Y)
    for f_idx, farm in enumerate(unit_names):
        land_avail = land_data[farm]
        for c_idx in range(len(food_names)):
            cqm.add_constraint(A[f_idx, c_idx] - land_avail * Y[f_idx, c_idx] <= 0, 
                             label=f"max_area_{f_idx}_{c_idx}")
    
    # 4. Simple food group constraint (global)
    n_food_vars = len(food_names)
    total_selections = sum(Y[f, c] for f in range(n_farms) for c in range(n_food_vars))
    cqm.add_constraint(total_selections >= n_food_vars // 2, label="min_total_foods")
    cqm.add_constraint(total_selections <= n_food_vars * n_farms, label="max_total_foods")
    
    metadata.update({
        "variables": len(cqm.variables),
        "constraints": len(cqm.constraints),
        "continuous_vars": n_farms * n_food_vars,
        "binary_vars": n_farms * n_food_vars
    })
    
    return cqm, metadata


def build_patch_cqm(n_patches: int, n_foods: int = 27) -> Tuple[ConstrainedQuadraticModel, Dict]:
    """Build Patch CQM using create_cqm_plots from solver_runner_BINARY.
    
    This ensures the formulation matches exactly what PuLP/Gurobi benchmarks use,
    including proper constraint encoding for:
    - At most one food per patch
    - Min/max planting areas
    - Food group diversity (unique foods)
    - U variables for linking
    """
    print(f"    Building Patch CQM ({n_patches} patches, {n_foods} foods) with REAL data...")
    
    # Load real data
    foods_dict, food_groups, base_config, weights = load_real_data()
    land_data = generate_land_data(n_patches, total_land=100.0)
    
    unit_names = list(land_data.keys())
    food_names = list(foods_dict.keys())[:n_foods]
    total_land = sum(land_data.values())
    
    # Limit foods dict to n_foods
    foods_limited = {k: v for i, (k, v) in enumerate(foods_dict.items()) if i < n_foods}
    
    # Create config matching what create_cqm_plots expects
    config = {
        'parameters': {
            'land_availability': land_data,
            'weights': weights,
            'minimum_planting_area': base_config['parameters'].get('minimum_planting_area', {}),
            'maximum_planting_area': {},  # Will be computed from max_percentage
            'food_group_constraints': base_config['parameters'].get('food_group_constraints', {})
        }
    }
    
    # Convert max_percentage_per_crop to maximum_planting_area
    max_pct = base_config['parameters'].get('max_percentage_per_crop', {})
    for crop, pct in max_pct.items():
        if crop in foods_limited:
            config['parameters']['maximum_planting_area'][crop] = pct * total_land
    
    # Use create_cqm_plots if available, otherwise fall back to simple formulation
    if CREATE_CQM_PLOTS_AVAILABLE:
        print(f"    Using create_cqm_plots from solver_runner_BINARY...")
        try:
            cqm, Y, constraint_metadata = create_cqm_plots_original(
                unit_names, foods_limited, food_groups, config
            )
            
            metadata = {
                "type": "Patch_CQM_BINARY", 
                "n_patches": n_patches, 
                "n_foods": len(food_names),
                "unit_names": unit_names,
                "food_names": food_names,
                "weights": weights,
                "total_land": total_land,
                "variables": len(cqm.variables),
                "constraints": len(cqm.constraints),
                "constraint_metadata": constraint_metadata
            }
            
            return cqm, metadata
            
        except Exception as e:
            print(f"    Warning: create_cqm_plots failed ({e}), using fallback...")
    
    # Fallback: simple formulation
    print(f"    Using fallback simple CQM formulation...")
    
    cqm = ConstrainedQuadraticModel()
    metadata = {
        "type": "Patch_CQM", 
        "n_patches": n_patches, 
        "n_foods": len(food_names),
        "unit_names": unit_names,
        "food_names": food_names,
        "weights": weights,
        "total_land": total_land
    }
    
    # Variables: binary only (Y_patchIdx_foodIdx format for compatibility)
    Y = {}
    for p_idx, patch in enumerate(unit_names):
        for c_idx, food in enumerate(food_names):
            Y[(p_idx, c_idx)] = Binary(f"Y_{p_idx}_{c_idx}")
    
    # Objective: weighted nutritional value (REAL formulation)
    objective = 0
    for p_idx, patch in enumerate(unit_names):
        patch_area = land_data[patch]
        for c_idx, food in enumerate(food_names):
            food_data = foods_dict[food]
            crop_coeff = (
                weights.get('nutritional_value', 0) * food_data.get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * food_data.get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * food_data.get('environmental_impact', 0) +
                weights.get('affordability', 0) * food_data.get('affordability', 0) +
                weights.get('sustainability', 0) * food_data.get('sustainability', 0)
            )
            objective += crop_coeff * patch_area * Y[(p_idx, c_idx)]
    
    # Normalize by total land
    objective = objective / total_land
    cqm.set_objective(-objective)  # Minimize negative = maximize
    
    # Constraints - at most one food per patch
    for p_idx in range(n_patches):
        cqm.add_constraint(sum(Y[(p_idx, c)] for c in range(len(food_names))) <= 1, 
                          label=f"Max_Assignment_{p_idx}")
    
    # 2. Global food constraints
    n_food_vars = len(food_names)
    total_selections = sum(Y[p, c] for p in range(n_patches) for c in range(n_food_vars))
    cqm.add_constraint(total_selections >= n_food_vars // 2, label="min_foods")
    
    metadata.update({
        "variables": len(cqm.variables),
        "constraints": len(cqm.constraints),
        "binary_vars": n_patches * n_food_vars
    })
    
    return cqm, metadata


def build_patch_direct_bqm(n_patches: int, n_foods: int = 27) -> Tuple[BinaryQuadraticModel, Dict]:
    """Build Patch BQM directly with REAL nutritional coefficients (minimal slack variables)"""
    print(f"    Building Patch Direct BQM ({n_patches} patches, {n_foods} foods) with REAL data...")
    
    # Load real data
    foods, food_groups, base_config, weights = load_real_data()
    land_data = generate_land_data(n_patches, total_land=100.0)
    
    unit_names = list(land_data.keys())
    food_names = list(foods.keys())[:n_foods]
    total_land = sum(land_data.values())
    
    bqm = BinaryQuadraticModel('BINARY')
    metadata = {
        "type": "Patch_Direct_BQM", 
        "n_patches": n_patches, 
        "n_foods": len(food_names),
        "unit_names": unit_names,
        "food_names": food_names,
        "weights": weights,
        "total_land": total_land
    }
    
    # Primary variables with REAL nutritional coefficients
    for p_idx, patch in enumerate(unit_names):
        patch_area = land_data[patch]
        for c_idx, food in enumerate(food_names):
            var = f"Y_{p_idx}_{c_idx}"
            food_data = foods[food]
            # Linear coefficient: negative of weighted value (to minimize)
            crop_coeff = (
                weights.get('nutritional_value', 0) * food_data.get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * food_data.get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * food_data.get('environmental_impact', 0) +
                weights.get('affordability', 0) * food_data.get('affordability', 0) +
                weights.get('sustainability', 0) * food_data.get('sustainability', 0)
            )
            # Weight by area and normalize
            linear_coeff = -crop_coeff * patch_area / total_land
            bqm.add_variable(var, linear_coeff)
    
    # Penalty for constraint violations (soft constraints)
    penalty = 10.0
    n_food_vars = len(food_names)
    
    # Patch limit constraint: sum(Y[p,:]) <= 5
    for p_idx in range(n_patches):
        vars_in_patch = [f"Y_{p_idx}_{c}" for c in range(n_food_vars)]
        # Add quadratic penalty for exceeding limit
        for i, v1 in enumerate(vars_in_patch):
            for v2 in vars_in_patch[i+1:]:
                bqm.add_interaction(v1, v2, penalty * 0.5)
    
    metadata.update({
        "variables": len(bqm.variables),
        "linear_terms": len(bqm.linear),
        "quadratic_terms": len(bqm.quadratic),
        "density": len(bqm.quadratic) / (len(bqm.variables) ** 2) if len(bqm.variables) > 0 else 0
    })
    
    return bqm, metadata


def build_patch_ultra_sparse_bqm(n_patches: int, n_foods: int = 27) -> Tuple[BinaryQuadraticModel, Dict]:
    """Build sparse BQM with REAL nutritional coefficients and AT-MOST-ONE constraint per patch.
    
    The key insight is that "at most one food per patch" creates O(n_patches * n_foods^2)
    quadratic terms, but this is still much sparser than full pairwise coupling.
    
    For 25 patches × 27 foods:
    - Variables: 675
    - At-most-one quadratic: 25 × (27 choose 2) = 25 × 351 = 8,775 terms
    - Full pairwise: 675 × 674 / 2 = 227,475 terms
    
    That's 26× sparser while encoding the actual constraint!
    
    AT-MOST-ONE constraint: sum(x_i) <= 1
    Penalty form: P * max(0, sum(x_i) - 1)^2
    For binary vars, this is: P * sum(x_i * x_j) for all pairs i < j
    (No linear penalty needed - selecting 0 or 1 is fine, only penalize selecting 2+)
    """
    print(f"    Building Patch Sparse BQM ({n_patches} patches, {n_foods} foods) with REAL data...")
    
    # Load real data
    foods, food_groups, base_config, weights = load_real_data()
    land_data = generate_land_data(n_patches, total_land=100.0)
    
    unit_names = list(land_data.keys())
    food_names = list(foods.keys())[:n_foods]
    total_land = sum(land_data.values())
    
    bqm = BinaryQuadraticModel('BINARY')
    metadata = {
        "type": "Patch_Sparse_BQM", 
        "n_patches": n_patches, 
        "n_foods": len(food_names),
        "unit_names": unit_names,
        "food_names": food_names,
        "weights": weights,
        "total_land": total_land
    }
    
    # Penalty for constraint violations - should dominate objective coefficients
    # Objective coeffs are ~0.01-0.1, so penalty of 2.0 should be sufficient
    PENALTY = 2.0
    
    # Linear terms with REAL nutritional coefficients (no constraint penalty for at-most-one)
    for p_idx, patch in enumerate(unit_names):
        patch_area = land_data[patch]
        for c_idx, food in enumerate(food_names):
            var = f"Y_{p_idx}_{c_idx}"
            food_data = foods[food]
            
            # Objective coefficient (negative because we minimize and want to maximize value)
            crop_coeff = (
                weights.get('nutritional_value', 0) * food_data.get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * food_data.get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * food_data.get('environmental_impact', 0) +
                weights.get('affordability', 0) * food_data.get('affordability', 0) +
                weights.get('sustainability', 0) * food_data.get('sustainability', 0)
            )
            objective_coeff = -crop_coeff * patch_area / total_land
            
            bqm.add_variable(var, objective_coeff)
    
    # At-most-one quadratic terms: for each patch, penalize selecting multiple foods
    # sum(x_i) <= 1 penalty: P * x_i * x_j for all pairs (penalize selecting 2+ foods)
    n_food_vars = len(food_names)
    for p_idx in range(n_patches):
        for c1 in range(n_food_vars):
            for c2 in range(c1 + 1, n_food_vars):
                v1 = f"Y_{p_idx}_{c1}"
                v2 = f"Y_{p_idx}_{c2}"
                bqm.add_interaction(v1, v2, PENALTY)
    
    metadata.update({
        "variables": len(bqm.variables),
        "linear_terms": len(bqm.linear),
        "quadratic_terms": len(bqm.quadratic),
        "density": len(bqm.quadratic) / (len(bqm.variables) ** 2) if len(bqm.variables) > 0 else 0,
        "penalty": PENALTY,
        "constraint": "at_most_one_per_patch"
    })
    
    return bqm, metadata


def cqm_to_bqm_wrapper(cqm: ConstrainedQuadraticModel, formulation_name: str) -> Tuple[BinaryQuadraticModel, Dict]:
    """Convert CQM to BQM with metadata tracking"""
    print(f"    Converting {formulation_name} to BQM...")
    start = time.time()
    
    result = cqm_to_bqm(cqm, lagrange_multiplier=10.0)
    
    # Handle both old and new API (may return tuple of (bqm, inv_map) or just bqm)
    if isinstance(result, tuple):
        bqm = result[0]
    else:
        bqm = result
    
    metadata = {
        "type": f"{formulation_name}_to_BQM",
        "conversion_time": time.time() - start,
        "variables": len(bqm.variables),
        "linear_terms": len(bqm.linear),
        "quadratic_terms": len(bqm.quadratic),
        "density": len(bqm.quadratic) / (len(bqm.variables) ** 2) if len(bqm.variables) > 0 else 0
    }
    
    return bqm, metadata


# =============================================================================
# DECOMPOSITION STRATEGIES
# =============================================================================

def get_bqm_graph(bqm: BinaryQuadraticModel) -> nx.Graph:
    """Convert BQM to NetworkX graph"""
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    G.add_edges_from(bqm.quadratic.keys())
    return G


def decompose_louvain(bqm: BinaryQuadraticModel, max_partition_size: int = 60) -> List[Set]:
    """Decompose using Louvain community detection.
    
    Creates partitions based on community structure with size limits.
    Large communities are split into smaller partitions.
    """
    if not LOUVAIN_AVAILABLE:
        return decompose_plot_based(bqm, plots_per_partition=3)
    
    G = get_bqm_graph(bqm)
    communities = louvain_communities(G, seed=42, resolution=1.5)  # Higher resolution = smaller communities
    
    # Process communities - split large ones, merge small ones
    partitions = []
    current_partition = set()
    
    for community in communities:
        community = set(community)
        
        # If community is too large, split it
        if len(community) > max_partition_size:
            # Split by converting to list and chunking
            comm_list = list(community)
            for i in range(0, len(comm_list), max_partition_size):
                chunk = set(comm_list[i:i + max_partition_size])
                partitions.append(chunk)
        elif len(current_partition) + len(community) <= max_partition_size:
            # Merge small communities
            current_partition.update(community)
        else:
            # Current partition is full, start new one
            if current_partition:
                partitions.append(current_partition)
            current_partition = community.copy()
    
    if current_partition:
        partitions.append(current_partition)
    
    return partitions if partitions else [set(bqm.variables)]


def decompose_plot_based(bqm: BinaryQuadraticModel, plots_per_partition: int = 3) -> List[Set]:
    """Decompose by grouping plots together.
    
    Handles both Y_0_0 (numeric) and Y_Patch1_Beef (string) variable naming formats.
    """
    import re
    variables = list(bqm.variables)
    
    # Extract plot indices from variable names
    plot_vars = {}
    slack_vars = []  # Collect slack/auxiliary variables separately
    
    for var in variables:
        if var.startswith("Y_"):
            parts = var.split("_", 2)  # Split into at most 3 parts
            if len(parts) >= 2:
                plot_part = parts[1]
                
                # Try numeric index first
                try:
                    plot_idx = int(plot_part)
                except ValueError:
                    # Extract numeric part from string like "Patch1" -> 1
                    match = re.search(r'(\d+)', plot_part)
                    if match:
                        plot_idx = int(match.group(1))
                    else:
                        # Use hash for completely non-numeric strings
                        plot_idx = hash(plot_part) % 10000
                
                if plot_idx not in plot_vars:
                    plot_vars[plot_idx] = []
                plot_vars[plot_idx].append(var)
        elif var.startswith("U_"):
            # U variables (unique food indicators) - track separately
            slack_vars.append(var)
        else:
            # Other slack or auxiliary variables
            slack_vars.append(var)
    
    if not plot_vars:
        return [set(variables)]
    
    # Group plots into partitions
    partitions = []
    plot_indices = sorted(plot_vars.keys())
    
    for i in range(0, len(plot_indices), plots_per_partition):
        partition = set()
        for plot_idx in plot_indices[i:i + plots_per_partition]:
            partition.update(plot_vars[plot_idx])
        partitions.append(partition)
    
    # Distribute slack variables among partitions (needed for constraint satisfaction)
    # Strategy: Add slack vars to the partition that has the most connections to them
    for slack_var in slack_vars:
        best_partition_idx = 0
        best_connection_count = 0
        
        for p_idx, partition in enumerate(partitions):
            # Count quadratic connections from slack_var to this partition
            connection_count = sum(
                1 for (u, v) in bqm.quadratic 
                if (u == slack_var and v in partition) or (v == slack_var and u in partition)
            )
            if connection_count > best_connection_count:
                best_connection_count = connection_count
                best_partition_idx = p_idx
        
        partitions[best_partition_idx].add(slack_var)
    
    return partitions


def decompose_energy_impact(bqm: BinaryQuadraticModel, partition_size: int = 60) -> List[Set]:
    """Decompose using energy-impact from dwave-hybrid.
    
    Runs the decomposer iteratively to create multiple partitions covering all variables.
    """
    if not HYBRID_AVAILABLE:
        # Fallback to plot-based if hybrid not available
        return decompose_plot_based(bqm, plots_per_partition=3)
    
    partitions = []
    remaining_vars = set(bqm.variables)
    
    # Create a working BQM that we'll progressively reduce
    while remaining_vars and len(partitions) < 100:  # Safety limit
        # Create sub-BQM with remaining variables
        sub_bqm = BinaryQuadraticModel('BINARY')
        for var in remaining_vars:
            if var in bqm.linear:
                sub_bqm.add_variable(var, bqm.linear[var])
        for (u, v), bias in bqm.quadratic.items():
            if u in remaining_vars and v in remaining_vars:
                sub_bqm.add_interaction(u, v, bias)
        
        if len(sub_bqm.variables) == 0:
            break
            
        # Use smaller partition size for better embedding
        actual_size = min(partition_size, len(sub_bqm.variables))
        
        decomposer = EnergyImpactDecomposer(
            size=actual_size,
            rolling_history=0.85,
            traversal="bfs"
        )
        
        # Create initial state
        initial_sample = {v: 0 for v in sub_bqm.variables}
        state = State.from_sample(initial_sample, sub_bqm)
        
        try:
            # Run decomposer
            decomposed_state = decomposer.run(state).result()
            
            # Extract subproblem variables
            if hasattr(decomposed_state, 'subproblem') and decomposed_state.subproblem:
                partition_vars = set(decomposed_state.subproblem.variables)
                if partition_vars:
                    partitions.append(partition_vars)
                    remaining_vars -= partition_vars
                else:
                    # Decomposer didn't produce useful result - take a chunk
                    chunk = set(list(remaining_vars)[:partition_size])
                    partitions.append(chunk)
                    remaining_vars -= chunk
            else:
                # No subproblem - take a chunk
                chunk = set(list(remaining_vars)[:partition_size])
                partitions.append(chunk)
                remaining_vars -= chunk
        except Exception as e:
            # Decomposer failed - take a chunk
            chunk = set(list(remaining_vars)[:partition_size])
            partitions.append(chunk)
            remaining_vars -= chunk
    
    # Add any remaining variables to the last partition
    if remaining_vars:
        if partitions:
            partitions[-1].update(remaining_vars)
        else:
            partitions.append(remaining_vars)
    
    return partitions if partitions else [set(bqm.variables)]


def extract_sub_bqm(bqm: BinaryQuadraticModel, variables: Set) -> BinaryQuadraticModel:
    """Extract subproblem BQM"""
    sub_bqm = BinaryQuadraticModel('BINARY')
    
    for var in variables:
        if var in bqm.linear:
            sub_bqm.add_variable(var, bqm.linear[var])
    
    for (u, v), bias in bqm.quadratic.items():
        if u in variables and v in variables:
            sub_bqm.add_interaction(u, v, bias)
    
    return sub_bqm


# =============================================================================
# EMBEDDING STUDY
# =============================================================================

def get_target_graph() -> nx.Graph:
    """Get QPU topology (or simulate Pegasus)"""
    if DWAVE_AVAILABLE:
        try:
            sampler = DWaveSampler()
            return sampler.to_networkx_graph()
        except:
            pass
    
    # Simulate Pegasus P16
    print("  Using simulated Pegasus P16 topology")
    import dwave_networkx as dnx
    return dnx.pegasus_graph(16)


def study_embedding(bqm: BinaryQuadraticModel, target_graph: nx.Graph, timeout: int = 300) -> Dict:
    """Study embedding feasibility and timing - ALWAYS attempts embedding regardless of density"""
    n_vars = len(bqm.variables)
    n_edges = len(bqm.quadratic)
    density = n_edges / (n_vars ** 2) if n_vars > 0 else 0
    print(f"      Testing embedding: {n_vars} vars, {n_edges} edges (density={density:.3f})")
    
    result = {
        "attempted": True,
        "success": False,
        "embedding_time": timeout,  # Default to full timeout if failed
        "chain_length_max": None,
        "chain_length_mean": None,
        "num_chains": 0,
        "num_variables": n_vars,
        "num_edges": n_edges,
        "density": density,
        "error": None,
        "skipped": False
    }
    
    # Handle empty or trivial BQMs
    if n_vars == 0:
        result["success"] = True
        result["embedding_time"] = 0
        result["error"] = "Empty BQM - trivially embeddable"
        return result
    
    source_graph = get_bqm_graph(bqm)
    
    try:
        print(f"      Running minorminer (timeout={timeout}s)...", flush=True)
        start = time.time()
        embedding = minorminer.find_embedding(source_graph, target_graph, timeout=timeout, verbose=0)
        elapsed = time.time() - start
        result["embedding_time"] = elapsed
        
        if embedding:
            result["success"] = True
            result["num_chains"] = len(embedding)
            
            chain_lengths = [len(chain) for chain in embedding.values()]
            result["chain_length_max"] = max(chain_lengths)
            result["chain_length_mean"] = statistics.mean(chain_lengths)
            print(f"      [SUCCESS] Embedded in {elapsed:.1f}s (chains: max={result['chain_length_max']}, mean={result['chain_length_mean']:.1f})")
        else:
            result["error"] = f"No embedding found in {elapsed:.1f}s"
            print(f"      [FAILED] No embedding (tried for {elapsed:.1f}s)")
            
    except Exception as e:
        result["error"] = str(e)
        result["embedding_time"] = timeout  # Assume full timeout on error
        print(f"      [ERROR] {e}")
    
    return result


def study_decomposed_embedding(bqm: BinaryQuadraticModel, partitions: List[Set],
                               target_graph: nx.Graph, timeout: int = 300) -> Dict:
    """Study embedding for decomposed problem - always calculates total time"""
    print(f"      Embedding {len(partitions)} partitions...")
    
    result = {
        "num_partitions": len(partitions),
        "partition_sizes": [len(p) for p in partitions],
        "partition_results": [],
        "total_embedding_time": 0,  # Sum of all partition times (success or not)
        "successful_embedding_time": 0,  # Sum of only successful embeddings
        "all_embedded": True,
        "num_successful": 0,
        "num_failed": 0
    }
    
    for i, partition in enumerate(partitions):
        print(f"        Partition {i+1}/{len(partitions)} ({len(partition)} vars)...")
        sub_bqm = extract_sub_bqm(bqm, partition)
        partition_result = study_embedding(sub_bqm, target_graph, timeout)
        partition_result["partition_id"] = i
        
        result["partition_results"].append(partition_result)
        
        # Always add to total time (even failed attempts take time)
        if partition_result.get("embedding_time") is not None:
            result["total_embedding_time"] += partition_result["embedding_time"]
        
        if partition_result["success"]:
            result["num_successful"] += 1
            if partition_result.get("embedding_time") is not None:
                result["successful_embedding_time"] += partition_result["embedding_time"]
        else:
            result["num_failed"] += 1
            result["all_embedded"] = False
    
    # Copy key metrics for convenience
    result["success"] = result["all_embedded"]
    result["embedding_time"] = result["total_embedding_time"]
    
    return result


def solve_decomposed_bqm_with_gurobi(bqm: BinaryQuadraticModel, partitions: List[Set], timeout: int = 600) -> Dict:
    """Solve each partition independently with Gurobi and aggregate results"""
    if not GUROBI_AVAILABLE:
        return {"error": "Gurobi not available", "success": False, "solve_time": 0}
    
    print(f"      Solving {len(partitions)} partitions with Gurobi...")
    
    result = {
        "num_partitions": len(partitions),
        "partition_results": [],
        "total_solve_time": 0,
        "aggregated_objective": 0,
        "all_optimal": True,
        "all_have_solution": True,
        "num_optimal": 0,
        "num_with_solution": 0,
        "num_time_limit": 0
    }
    
    for i, partition in enumerate(partitions):
        sub_bqm = extract_sub_bqm(bqm, partition)
        
        # Solve partition
        partition_solve = solve_bqm_with_gurobi(sub_bqm, timeout)
        
        # Always add solve time (even for failures/timeouts)
        partition_time = partition_solve.get("solve_time", 0) or 0
        result["total_solve_time"] += partition_time
        
        result["partition_results"].append({
            "partition_id": i,
            "n_vars": len(partition),
            "success": partition_solve.get("success", False),
            "has_solution": partition_solve.get("has_solution", False),
            "is_time_limit": partition_solve.get("is_time_limit", False),
            "solve_time": partition_time,
            "objective": partition_solve.get("objective", None),
            "status": partition_solve.get("status", None),
            "solution": partition_solve.get("solution", {})  # FIX: Pass through solution for objective calculation
        })
        
        # Track statistics
        if partition_solve.get("success"):
            result["num_optimal"] += 1
        else:
            result["all_optimal"] = False
            
        if partition_solve.get("has_solution"):
            result["num_with_solution"] += 1
            if partition_solve.get("objective") is not None:
                result["aggregated_objective"] += partition_solve["objective"]
        else:
            result["all_have_solution"] = False
            
        if partition_solve.get("is_time_limit"):
            result["num_time_limit"] += 1
    
    # Add summary info
    # "success" = all optimal; "has_solution" = all partitions have at least one solution
    result["success"] = result["all_optimal"]
    result["has_solution"] = result["all_have_solution"]
    result["solve_time"] = result["total_solve_time"]
    result["objective"] = result["aggregated_objective"] if result["all_have_solution"] else None
    
    return result


# =============================================================================
# GUROBI SOLVING
# =============================================================================

def solve_cqm_with_gurobi(cqm: ConstrainedQuadraticModel, timeout: int = 600) -> Dict:
    """Solve CQM with Gurobi"""
    if not GUROBI_AVAILABLE:
        return {"error": "Gurobi not available"}
    
    print(f"      Solving CQM ({len(cqm.variables)} vars, {len(cqm.constraints)} constraints)...")
    
    try:
        print("        [1/5] Creating Gurobi model...")
        model = gp.Model("CQM")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', timeout)
        
        # Create variables
        print(f"        [2/5] Adding variables...")
        gurobi_vars = {}
        for var_name in cqm.variables:
            var_info = cqm.vartype(var_name)
            if var_info == 'BINARY':
                gurobi_vars[var_name] = model.addVar(vtype=GRB.BINARY, name=var_name)
            else:  # REAL
                lb = cqm.lower_bound(var_name)
                ub = cqm.upper_bound(var_name)
                gurobi_vars[var_name] = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=var_name)
        
        # Set objective
        print(f"        [3/5] Building objective ({len(cqm.objective.linear)} linear, {len(cqm.objective.quadratic)} quadratic)...")
        obj_expr = 0
        for var_name, coeff in cqm.objective.linear.items():
            obj_expr += coeff * gurobi_vars[var_name]
        
        for (v1, v2), coeff in cqm.objective.quadratic.items():
            obj_expr += coeff * gurobi_vars[v1] * gurobi_vars[v2]
        
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Add constraints
        print(f"        [4/5] Adding {len(cqm.constraints)} constraints...")
        for i, (label, constraint) in enumerate(cqm.constraints.items()):
            if i % 100 == 0 and i > 0:
                print(f"            Progress: {i}/{len(cqm.constraints)}")
            constr_expr = 0
            for var_name, coeff in constraint.lhs.linear.items():
                constr_expr += coeff * gurobi_vars[var_name]
            
            for (v1, v2), coeff in constraint.lhs.quadratic.items():
                constr_expr += coeff * gurobi_vars[v1] * gurobi_vars[v2]
            
            # Handle Sense enum from dimod (Sense.Le, Sense.Ge, Sense.Eq) or string comparison
            sense_str = str(constraint.sense)
            if sense_str == 'Sense.Le' or constraint.sense == '<=':
                model.addConstr(constr_expr <= constraint.rhs, name=label)
            elif sense_str == 'Sense.Ge' or constraint.sense == '>=':
                model.addConstr(constr_expr >= constraint.rhs, name=label)
            elif sense_str == 'Sense.Eq' or constraint.sense == '==':
                model.addConstr(constr_expr == constraint.rhs, name=label)
            else:
                raise ValueError(f"Unknown constraint sense: {constraint.sense} (type: {type(constraint.sense)})")
        
        # Solve
        print("        [5/5] Optimizing...")
        start = time.time()
        model.optimize()
        solve_time = time.time() - start
        print(f"        [DONE] in {solve_time:.2f}s (status={model.status})")
        
        result = {
            "success": model.status == GRB.OPTIMAL,
            "solve_time": solve_time,
            "objective": model.objVal if model.status == GRB.OPTIMAL else None,
            "status": model.status,
            "mip_gap": model.MIPGap if hasattr(model, 'MIPGap') else None
        }
        
        return result
        
    except Exception as e:
        print(f"        [ERROR] {e}")
        return {"error": str(e)}


def solve_bqm_with_gurobi(bqm: BinaryQuadraticModel, timeout: int = 600) -> Dict:
    """Solve BQM as QUBO with Gurobi"""
    if not GUROBI_AVAILABLE:
        return {"error": "Gurobi not available"}
    
    print(f"      Solving BQM ({len(bqm.variables)} vars, {len(bqm.quadratic)} quad)...")
    
    try:
        print("        Creating model and optimizing...")
        model = gp.Model("BQM")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', timeout)
        
        # Create binary variables
        gurobi_vars = {var: model.addVar(vtype=GRB.BINARY, name=var) for var in bqm.variables}
        
        # Build objective
        obj_expr = bqm.offset
        
        for var, coeff in bqm.linear.items():
            obj_expr += coeff * gurobi_vars[var]
        
        for (v1, v2), coeff in bqm.quadratic.items():
            obj_expr += coeff * gurobi_vars[v1] * gurobi_vars[v2]
        
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Solve
        start = time.time()
        model.optimize()
        solve_time = time.time() - start
        
        # Check if we have a feasible solution (even if not optimal)
        has_solution = model.SolCount > 0
        is_optimal = model.status == GRB.OPTIMAL
        is_time_limit = model.status == GRB.TIME_LIMIT
        
        # Get objective and solution if we have any solution
        obj_value = None
        solution = {}
        if has_solution:
            obj_value = model.objVal
            # Extract solution for objective recalculation
            for var_name, gvar in gurobi_vars.items():
                solution[var_name] = gvar.X
        
        status_str = "OPTIMAL" if is_optimal else ("TIME_LIMIT" if is_time_limit else f"STATUS_{model.status}")
        solution_str = f", bqm_energy={obj_value:.4f}" if has_solution else ", no solution"
        print(f"        [DONE] in {solve_time:.2f}s ({status_str}{solution_str})")
        
        result = {
            "success": is_optimal,
            "has_solution": has_solution,
            "is_time_limit": is_time_limit,
            "solve_time": solve_time,
            "bqm_energy": obj_value,  # BQM energy includes penalty terms
            "objective": obj_value,  # Will be replaced with actual objective if metadata available
            "solution": solution,  # Store solution for actual objective calculation
            "status": model.status,
            "solution_count": model.SolCount
        }
        
        return result
        
    except Exception as e:
        print(f"        [ERROR] {e}")
        return {"error": str(e), "solve_time": 0}


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def save_results_json(results: List[Dict], output_dir: Path, timestamp: str) -> Path:
    """Save detailed results as JSON"""
    output_file = output_dir / f"benchmark_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "config": {
                "problem_sizes": PROBLEM_SIZES,
                "n_foods": N_FOODS,
                "embedding_timeout": EMBEDDING_TIMEOUT,
                "solve_timeout": SOLVE_TIMEOUT,
                "formulations": FORMULATIONS,
                "decompositions": DECOMPOSITIONS
            },
            "results": results
        }, f, indent=2)
    return output_file


def save_results_csv(results: List[Dict], output_dir: Path, timestamp: str) -> Path:
    """Save flattened results as CSV for analysis"""
    output_file = output_dir / f"benchmark_results_{timestamp}.csv"
    
    rows = []
    for r in results:
        # Handle None embedding (for CQM)
        embedding = r.get("embedding") or {}
        solving = r.get("solving") or {}
        metadata = r.get("metadata") or {}
        
        row = {
            "n_farms": r.get("n_farms"),
            "formulation": r.get("formulation"),
            "decomposition": r.get("decomposition", "None"),
            "num_partitions": r.get("num_partitions", 1),
            
            # Variables and structure
            "num_variables": metadata.get("variables", 0),
            "num_quadratic": metadata.get("quadratic_terms", 0),
            "density": metadata.get("density", 0),
            
            # Embedding results (may be None for CQM)
            "embedding_success": embedding.get("success", False) if embedding else False,
            "embedding_time": embedding.get("embedding_time", 0) or 0 if embedding else 0,
            "chain_length_max": embedding.get("chain_length_max") if embedding else None,
            "chain_length_mean": embedding.get("chain_length_mean") if embedding else None,
            
            # Solving results
            "solve_success": solving.get("success", False),
            "solve_time": solving.get("solve_time", 0) or 0,
            "objective": solving.get("objective"),
            
            # Total times
            "total_embedding_time": r.get("total_embedding_time", 0),
            "total_solve_time": r.get("total_solve_time", 0),
            "total_time": r.get("total_time", 0),
        }
        rows.append(row)
    
    # Write CSV (csv module imported at top)
    if rows:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    return output_file


def save_results_markdown(results: List[Dict], output_dir: Path, timestamp: str) -> Path:
    """Generate human-readable Markdown summary"""
    output_file = output_dir / f"benchmark_summary_{timestamp}.md"
    
    lines = [
        f"# Comprehensive Benchmark Summary",
        f"",
        f"**Generated:** {timestamp}",
        f"",
        f"## Configuration",
        f"",
        f"- Problem sizes: {PROBLEM_SIZES}",
        f"- Foods per farm: {N_FOODS}",
        f"- Embedding timeout: {EMBEDDING_TIMEOUT}s",
        f"- Solve timeout: {SOLVE_TIMEOUT}s",
        f"- Formulations: {FORMULATIONS}",
        f"- Decompositions: {DECOMPOSITIONS}",
        f"",
        f"## Results Summary",
        f"",
        f"| n_farms | Formulation | Decomposition | Partitions | Embed? | Embed Time | Solve? | Solve Time | Total Time |",
        f"|---------|-------------|---------------|------------|--------|------------|--------|------------|------------|",
    ]
    
    for r in results:
        # Handle None embedding (for CQM)
        embedding = r.get("embedding") or {}
        solving = r.get("solving") or {}
        
        embed_success = "[OK]" if embedding.get("success") else "[NO]"
        solve_success = "[OK]" if solving.get("success") else "[NO]"
        embed_time = r.get("total_embedding_time", 0) or 0
        solve_time = r.get("total_solve_time", 0) or 0
        total_time = r.get("total_time", 0) or 0
        
        lines.append(
            f"| {r.get('n_farms', 'N/A')} | {r.get('formulation', 'N/A')} | "
            f"{r.get('decomposition', 'None')} | {r.get('num_partitions', 1)} | "
            f"{embed_success} | {embed_time:.1f}s | {solve_success} | {solve_time:.1f}s | {total_time:.1f}s |"
        )
    
    # Statistics
    lines.extend([
        f"",
        f"## Statistics",
        f"",
        f"- Total experiments: {len(results)}",
        f"- Successful embeddings: {sum(1 for r in results if r.get('embedding') and r['embedding'].get('success'))}",
        f"- Successful solves: {sum(1 for r in results if r.get('solving') and r['solving'].get('success'))}",
    ])
    
    # Best configurations by farm size
    lines.extend([
        f"",
        f"## Best Configurations by Problem Size",
        f"",
    ])
    
    for n_farms in PROBLEM_SIZES:
        farm_results = [r for r in results if r.get("n_farms") == n_farms]
        if farm_results:
            # Best by total time (only successful solves)
            successful = [r for r in farm_results if r.get("solving", {}).get("success")]
            if successful:
                best = min(successful, key=lambda x: x.get("total_time", float('inf')))
                lines.append(
                    f"- **{n_farms} farms**: {best.get('formulation')} + {best.get('decomposition', 'None')} "
                    f"(total: {best.get('total_time', 0):.1f}s)"
                )
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    return output_file


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def apply_decomposition(bqm: BinaryQuadraticModel, decomp_name: str) -> Tuple[List[Set], str]:
    """Apply a decomposition strategy and return partitions"""
    if decomp_name == "None" or decomp_name is None:
        return [set(bqm.variables)], "None"
    
    elif decomp_name == "Louvain":
        if not LOUVAIN_AVAILABLE:
            return None, "Louvain not available"
        partitions = decompose_louvain(bqm, max_partition_size=50)
        if not partitions or len(partitions) <= 1:
            return [set(bqm.variables)], "Louvain produced single partition"
        return partitions, None
    
    elif decomp_name == "PlotBased":
        partitions = decompose_plot_based(bqm, plots_per_partition=5)
        if not partitions or len(partitions) <= 1:
            return [set(bqm.variables)], "PlotBased produced single partition"
        return partitions, None
    
    elif decomp_name == "Multilevel":
        if not ADVANCED_DECOMP_AVAILABLE:
            return None, "Multilevel not available"
        partitions = decompose_multilevel(bqm, levels=2, partition_size=100)
        if not partitions or len(partitions) <= 1:
            return [set(bqm.variables)], "Multilevel produced single partition"
        return partitions, None
    
    elif decomp_name == "Cutset":
        if not ADVANCED_DECOMP_AVAILABLE:
            return None, "Cutset not available"
        partitions = decompose_sequential_cutset(bqm, max_cut_size=5, min_partition_size=50)
        if not partitions or len(partitions) <= 1:
            return [set(bqm.variables)], "Cutset produced single partition"
        return partitions, None
    
    elif decomp_name == "SpatialGrid":
        if not ADVANCED_DECOMP_AVAILABLE:
            return None, "SpatialGrid not available"
        partitions = decompose_spatial_grid(bqm, grid_size=3)
        if not partitions or len(partitions) <= 1:
            return [set(bqm.variables)], "SpatialGrid produced single partition"
        return partitions, None
    
    elif decomp_name == "EnergyImpact":
        if not HYBRID_AVAILABLE:
            return None, "EnergyImpact not available"
        partitions = decompose_energy_impact(bqm, partition_size=100)
        if not partitions or len(partitions) <= 1:
            return [set(bqm.variables)], "EnergyImpact produced single partition"
        return partitions, None
    
    else:
        return None, f"Unknown decomposition: {decomp_name}"


def test_single_configuration(
    n_farms: int,
    formulation: str,
    decomposition: str,
    target_graph: nx.Graph,
    experiment_id: int,
    total_experiments: int
) -> Dict:
    """Test a single configuration and return comprehensive results"""
    
    print(f"\n  [{experiment_id}/{total_experiments}] n_farms={n_farms}, form={formulation}, decomp={decomposition}")
    
    result = {
        "n_farms": n_farms,
        "formulation": formulation,
        "decomposition": decomposition,
        "num_partitions": 1,
        "metadata": {},
        "embedding": None,
        "solving": None,
        "total_embedding_time": 0,
        "total_solve_time": 0,
        "total_time": 0,
        "error": None
    }
    
    try:
        # Step 1: Build formulation
        print(f"    Building {formulation}...")
        
        if formulation == "CQM":
            cqm, meta = build_patch_cqm(n_farms)
            result["metadata"] = meta
            
            # CQM: No embedding (continuous), just solve with Gurobi
            print(f"    Solving CQM with Gurobi (no embedding for CQM)...")
            solve_result = solve_cqm_with_gurobi(cqm, SOLVE_TIMEOUT)
            result["solving"] = solve_result
            result["total_solve_time"] = solve_result.get("solve_time", 0) or 0
            result["total_time"] = result["total_solve_time"]
            
            # No decomposition for CQM
            if decomposition != "None":
                result["error"] = "CQM does not support decomposition"
                return result
            
            return result
        
        elif formulation == "BQM":
            # Convert CQM to BQM
            cqm, _ = build_patch_cqm(n_farms)
            bqm, meta = cqm_to_bqm_wrapper(cqm, "Patch_CQM")
            result["metadata"] = meta
        
        else:
            result["error"] = f"Unknown formulation: {formulation}"
            return result
        
        # Step 2: Apply decomposition
        print(f"    Applying decomposition: {decomposition}...")
        partitions, decomp_error = apply_decomposition(bqm, decomposition)
        
        if partitions is None:
            result["error"] = decomp_error
            # Still try to solve without decomposition
            partitions = [set(bqm.variables)]
        
        result["num_partitions"] = len(partitions)
        print(f"      Created {len(partitions)} partition(s)")
        
        # Step 3: Embedding study
        # Skip embedding for BQM with None decomposition (known to fail for dense 25-farm problems)
        skip_embedding = SKIP_DENSE_EMBEDDING and decomposition == "None" and len(partitions) == 1
        
        if skip_embedding:
            print(f"    Skipping embedding (dense BQM with no decomposition - known to fail)")
            result["embedding"] = {
                "success": False,
                "skipped": True,
                "embedding_time": 0,
                "error": "Skipped - dense BQM without decomposition"
            }
            result["total_embedding_time"] = 0
        elif len(partitions) == 1:
            print(f"    Running embedding study...")
            embed_result = study_embedding(bqm, target_graph, EMBEDDING_TIMEOUT)
            result["embedding"] = embed_result
            result["total_embedding_time"] = embed_result.get("embedding_time", 0) or 0
        else:
            print(f"    Running embedding study...")
            embed_result = study_decomposed_embedding(bqm, partitions, target_graph, EMBEDDING_TIMEOUT)
            result["embedding"] = embed_result
            result["total_embedding_time"] = embed_result.get("total_embedding_time", 0) or 0
        
        # Step 4: Solve with Gurobi (always, regardless of embedding success)
        print(f"    Solving with Gurobi...")
        if len(partitions) == 1:
            solve_result = solve_bqm_with_gurobi(bqm, SOLVE_TIMEOUT)
            result["solving"] = solve_result
            result["total_solve_time"] = solve_result.get("solve_time", 0) or 0
            
            # Calculate ACTUAL objective from solution (not BQM energy)
            if solve_result.get("has_solution") and "solution" in solve_result:
                actual_obj = calculate_actual_objective_from_bqm_solution(
                    solve_result["solution"], meta, n_farms
                )
                if actual_obj is not None:
                    solve_result["actual_objective"] = actual_obj
                    solve_result["bqm_energy"] = solve_result.get("objective")
                    solve_result["objective"] = actual_obj
                    print(f"      Actual objective: {actual_obj:.6f} (BQM energy: {solve_result.get('bqm_energy', 0):.4f})")
        else:
            solve_result = solve_decomposed_bqm_with_gurobi(bqm, partitions, SOLVE_TIMEOUT)
            result["solving"] = solve_result
            result["total_solve_time"] = solve_result.get("total_solve_time", 0) or 0
            
            # For decomposed: MERGE all partition solutions first, then calculate objective once
            # FIX: Previous code calculated objective per partition which was wrong
            if solve_result.get("all_have_solution"):
                # Merge all partition solutions into a single global solution dict
                merged_solution = {}
                for part_result in solve_result.get("partition_results", []):
                    if part_result.get("has_solution") and "solution" in part_result:
                        merged_solution.update(part_result["solution"])
                
                # Calculate actual objective ONCE from merged solution
                actual_obj = calculate_actual_objective_from_bqm_solution(
                    merged_solution, meta, n_farms
                )
                
                if actual_obj is not None:
                    solve_result["actual_objective"] = actual_obj
                    solve_result["bqm_energy"] = solve_result.get("aggregated_objective")
                    solve_result["objective"] = actual_obj
                    print(f"      Actual objective: {actual_obj:.6f} (BQM energy: {solve_result.get('bqm_energy', 0):.4f})")
                    
                    # Warn if decomposition likely caused constraint violations
                    if decomposition not in VALID_DECOMPOSITIONS_FOR_OBJECTIVE:
                        print(f"      ⚠️  WARNING: Decomposition '{decomposition}' may violate constraints (objective unreliable)")
                else:
                    print(f"      Warning: Could not calculate actual objective from merged solution")
        
        # Step 5: Calculate total time
        result["total_time"] = result["total_embedding_time"] + result["total_solve_time"]
        
        # Summary
        embed_status = "[OK]" if result["embedding"].get("success") else "[NO]"
        
        # More detailed solve status
        solving = result["solving"]
        if solving.get("success"):
            solve_status = "[OK]"
        elif solving.get("has_solution"):
            solve_status = "[PARTIAL]"  # Has solution but not proven optimal
        else:
            solve_status = "[NO]"
        
        # Show extra info for decomposed problems
        extra_info = ""
        if len(partitions) > 1 and "num_optimal" in solving:
            extra_info = f" [{solving['num_optimal']}/{len(partitions)} optimal"
            if solving.get("num_time_limit", 0) > 0:
                extra_info += f", {solving['num_time_limit']} timeout"
            extra_info += "]"
        elif solving.get("is_time_limit"):
            extra_info = " [timeout]"
            
        print(f"    Result: Embed={embed_status} ({result['total_embedding_time']:.1f}s), "
              f"Solve={solve_status} ({result['total_solve_time']:.1f}s){extra_info}, "
              f"Total={result['total_time']:.1f}s")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"    [ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    return result


def run_comprehensive_benchmark():
    """Run complete systematic benchmark across all configurations"""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EMBEDDING AND SOLVING BENCHMARK v2.0")
    print("=" * 80)
    
    # Initialize
    print("\n[1/4] Initializing...")
    target_graph = get_target_graph()
    print(f"  [OK] Target graph: {len(target_graph.nodes)} nodes, {len(target_graph.edges)} edges")
    
    # Calculate total experiments
    # CQM only with None decomposition
    # BQM with all decompositions
    total_experiments = 0
    for n_farms in PROBLEM_SIZES:
        total_experiments += 1  # CQM (no decomp)
        for decomp in DECOMPOSITIONS:
            total_experiments += 1  # BQM with each decomposition
    
    print(f"  Total configurations to test: {total_experiments}")
    print(f"  Problem sizes: {PROBLEM_SIZES}")
    print(f"  Formulations: {FORMULATIONS}")
    print(f"  Decompositions: {DECOMPOSITIONS}")
    
    # Run all experiments
    print("\n[2/4] Running experiments...")
    all_results = []
    experiment_id = 0
    benchmark_start = time.time()
    
    for n_farms in PROBLEM_SIZES:
        print(f"\n{'='*60}")
        print(f"PROBLEM SIZE: {n_farms} farms × {N_FOODS} foods = {n_farms * N_FOODS} variables")
        print(f"{'='*60}")
        
        # Test CQM (only with None decomposition)
        experiment_id += 1
        result = test_single_configuration(
            n_farms, "CQM", "None", target_graph, experiment_id, total_experiments
        )
        all_results.append(result)
        
        # Test BQM with all decompositions
        for decomposition in DECOMPOSITIONS:
            experiment_id += 1
            result = test_single_configuration(
                n_farms, "BQM", decomposition, target_graph, experiment_id, total_experiments
            )
            all_results.append(result)
    
    benchmark_time = time.time() - benchmark_start
    
    # Save results
    print("\n[3/4] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_file = save_results_json(all_results, OUTPUT_DIR, timestamp)
    print(f"  [OK] JSON: {json_file}")
    
    csv_file = save_results_csv(all_results, OUTPUT_DIR, timestamp)
    print(f"  [OK] CSV: {csv_file}")
    
    md_file = save_results_markdown(all_results, OUTPUT_DIR, timestamp)
    print(f"  [OK] Markdown: {md_file}")
    
    # Final summary
    print("\n[4/4] Summary...")
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Total benchmark time: {benchmark_time:.1f}s ({benchmark_time/60:.1f} min)")
    
    # Statistics
    embed_attempts = sum(1 for r in all_results if r.get("embedding"))
    embed_success = sum(1 for r in all_results if r.get("embedding") and r["embedding"].get("success"))
    solve_success = sum(1 for r in all_results if r.get("solving") and r["solving"].get("success"))
    
    print(f"\nSuccessful embeddings: {embed_success}/{embed_attempts}")
    print(f"Successful solves: {solve_success}/{len(all_results)}")
    
    # Best results by problem size
    print("\nBest configurations by problem size (by total time):")
    for n_farms in PROBLEM_SIZES:
        farm_results = [r for r in all_results if r.get("n_farms") == n_farms]
        successful = [r for r in farm_results if r.get("solving", {}).get("success")]
        if successful:
            best = min(successful, key=lambda x: x.get("total_time", float('inf')))
            print(f"  {n_farms} farms: {best.get('formulation')} + {best.get('decomposition', 'None')} "
                  f"({best.get('total_time', 0):.1f}s total)")
    
    print(f"\nOutput files:")
    print(f"  - {json_file}")
    print(f"  - {csv_file}")
    print(f"  - {md_file}")
    
    return all_results


if __name__ == "__main__":
    try:
        results = run_comprehensive_benchmark()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
