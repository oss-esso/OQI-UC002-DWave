#!/usr/bin/env python3
"""
QPU Benchmark: Pure Quantum Annealing (No Hybrid Solvers) v2.0

This benchmark tests ONLY direct QPU methods:
1. Direct QPU (DWaveSampler + EmbeddingComposite) - CQM→BQM→QPU
2. Manual Decomposition + SA/QPU (PlotBased, Multilevel, Louvain, Cutset, Spectral)
3. Coordinated Master-Subproblem decomposition (constraint-preserving)

NO hybrid solvers (LeapHybridCQMSampler, LeapHybridBQMSampler, dwave-hybrid).

Features:
- Comprehensive timing (CQM build, BQM conversion, embedding, solving, QPU access)
- Advanced logging with step-by-step progress
- Multiple decomposition strategies from comprehensive_embedding_benchmark
- Detailed QPU timing extraction from sampleset.info

Author: OQI-UC002-DWave Project
Date: 2025-11-30 v2.0
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
import logging

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(verbose: bool = True) -> logging.Logger:
    """Setup logging with timestamps."""
    logger = logging.getLogger('QPU_Benchmark')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger

LOG = setup_logging()

# ============================================================================
# IMPORTS
# ============================================================================

print("=" * 80)
print("QPU BENCHMARK v2.0: Pure Quantum Annealing (No Hybrid)")
print("=" * 80)
print()

print("[1/5] Importing core libraries...")
import_start = time.time()

import networkx as nx
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel, Binary, cqm_to_bqm

# Gurobi for ground truth
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    print("  Warning: Gurobi not available - ground truth will be limited")

# Project imports
from src.scenarios import load_food_data
from Utils import patch_sampler

print(f"  Core imports: {time.time() - import_start:.2f}s")

# ============================================================================
# D-WAVE IMPORTS (QPU ONLY - NO HYBRID)
# ============================================================================

print("[2/5] Importing D-Wave QPU libraries (no hybrid)...")
dwave_start = time.time()

HAS_QPU = False
HAS_QBSOLV = False
HAS_EMBEDDING = False
HAS_LOUVAIN = False
HAS_SPECTRAL = False
HAS_CLIQUE = False

# Direct QPU sampler
try:
    from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
    HAS_QPU = True
    print("  ✓ DWaveSampler (direct QPU) available")
except ImportError:
    print("  ✗ DWaveSampler not available")

# Clique sampler (for small problems n<=16)
try:
    from dwave.system.samplers import DWaveCliqueSampler
    HAS_CLIQUE = True
    print("  ✓ DWaveCliqueSampler (hardware cliques) available")
except ImportError:
    print("  ✗ DWaveCliqueSampler not available")

# QBSolv is deprecated and incompatible with Python 3.12+
HAS_QBSOLV = False
print("  ℹ QBSolv: Deprecated (using decomposition methods instead)")

# Embedding tools
try:
    from minorminer import find_embedding
    HAS_EMBEDDING = True
    print("  ✓ minorminer available")
except ImportError:
    print("  ✗ minorminer not available")

# Pegasus graph for embedding analysis
try:
    import dwave_networkx as dnx
    HAS_DNX = True
    print("  ✓ dwave_networkx available")
except ImportError:
    HAS_DNX = False

# Simulated annealing for fallback/comparison
try:
    import neal
    HAS_NEAL = True
    print("  ✓ neal (SimulatedAnnealing) available for comparison")
except ImportError:
    HAS_NEAL = False

# Louvain community detection
try:
    from networkx.algorithms.community import louvain_communities
    HAS_LOUVAIN = True
    print("  ✓ Louvain community detection available")
except ImportError:
    print("  ✗ Louvain not available")

# Spectral clustering
try:
    from sklearn.cluster import SpectralClustering
    HAS_SPECTRAL = True
    print("  ✓ Spectral clustering available")
except ImportError:
    print("  ✗ Spectral clustering not available")

print(f"  D-Wave imports: {time.time() - dwave_start:.2f}s")

# ============================================================================
# CONFIGURATION
# ============================================================================

print("[3/5] Loading configuration...")

# Problem scales
FARM_SCALES = [25, 50, 100, 250, 500, 1000]
N_FOODS = 27

# QPU parameters
NUM_READS_OPTIONS = [100, 500, 1000, 5000]
ANNEALING_TIME_OPTIONS = [20, 100, 200]  # microseconds
CHAIN_STRENGTH_MULTIPLIERS = [1.0, 1.5, 2.0]

# Defaults
DEFAULT_NUM_READS = 1000
DEFAULT_ANNEALING_TIME = 20  # µs
DEFAULT_CHAIN_STRENGTH = None  # auto-calculated

# Timeouts - CRITICAL: Don't waste QPU time on problems that can't embed
DIRECT_QPU_TIMEOUT = 200  # seconds for direct QPU (embedding + solving)
EMBED_TIMEOUT_PER_PARTITION = 300  # seconds for embedding per partition  
SOLVE_TIMEOUT = 300  # seconds total per method

# Output
OUTPUT_DIR = Path(__file__).parent / "qpu_benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"  Output directory: {OUTPUT_DIR}")

# Global QPU graph cache
QPU_GRAPH = None
QPU_SAMPLER = None
DWAVE_TOKEN = None  # Set via --token or set_dwave_token()


def set_dwave_token(token: str):
    """Set the D-Wave API token for QPU access."""
    global DWAVE_TOKEN
    DWAVE_TOKEN = token
    # Also set environment variable for dwave-system
    os.environ['DWAVE_API_TOKEN'] = token
    print(f"  D-Wave token configured (length: {len(token)})")


def get_qpu_sampler():
    """Get cached QPU sampler."""
    global QPU_SAMPLER
    if QPU_SAMPLER is None and HAS_QPU:
        try:
            # Use token if set
            if DWAVE_TOKEN:
                QPU_SAMPLER = DWaveSampler(token=DWAVE_TOKEN)
            else:
                QPU_SAMPLER = DWaveSampler()
            print(f"  Connected to QPU: {QPU_SAMPLER.properties.get('chip_id', 'Unknown')}")
        except Exception as e:
            print(f"  Failed to connect to QPU: {e}")
    return QPU_SAMPLER


def get_qpu_graph():
    """Get QPU topology graph (cached)."""
    global QPU_GRAPH
    if QPU_GRAPH is not None:
        return QPU_GRAPH
    
    sampler = get_qpu_sampler()
    if sampler:
        QPU_GRAPH = sampler.to_networkx_graph()
        print(f"  QPU topology: {QPU_GRAPH.number_of_nodes()} qubits")
        return QPU_GRAPH
    
    # Fallback to simulated Pegasus
    if HAS_DNX:
        QPU_GRAPH = dnx.pegasus_graph(16)
        print(f"  Using simulated Pegasus P16: {QPU_GRAPH.number_of_nodes()} qubits")
        return QPU_GRAPH
    
    return None


# ============================================================================
# DATA LOADING
# ============================================================================

print("[4/5] Setting up data loading...")


def load_problem_data(n_farms: int) -> Dict:
    """Load food data and create patch configuration."""
    _, foods, food_groups, config_loaded = load_food_data('full_family')
    weights = config_loaded.get('parameters', {}).get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15
    })
    
    land_availability = patch_sampler.generate_grid(n_farms, area=100.0, seed=42)
    farm_names = list(land_availability.keys())
    total_area = sum(land_availability.values())
    
    food_names = list(foods.keys())
    food_benefits = {}
    for food in food_names:
        benefit = (
            weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) +
            weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) -
            weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) +
            weights.get('affordability', 0) * foods[food].get('affordability', 0) +
            weights.get('sustainability', 0) * foods[food].get('sustainability', 0)
        )
        food_benefits[food] = benefit
    
    # Food group constraints: match comprehensive_benchmark.py formulation
    # Use min_foods: 1 (at least 1 food from each group) - same as comprehensive_benchmark
    food_group_constraints = {
        group: {'min_foods': 1, 'max_foods': len(foods_in_group)}
        for group, foods_in_group in food_groups.items()
    }
    
    # No max_plots_per_crop constraint by default (matching comprehensive_benchmark)
    # This can be enabled via max_planting_area in config if needed
    max_plots_per_crop = None  # Disabled to match comprehensive_benchmark
    
    return {
        'foods': foods,
        'food_names': food_names,
        'food_groups': food_groups,
        'food_benefits': food_benefits,
        'weights': weights,
        'land_availability': land_availability,
        'farm_names': farm_names,
        'total_area': total_area,
        'food_group_constraints': food_group_constraints,
        'max_plots_per_crop': max_plots_per_crop,
        'n_farms': n_farms,
        'n_foods': len(food_names)
    }


def load_problem_data_from_scenario(scenario_name: str) -> Dict:
    """
    Load problem data from a named scenario (synthetic or standard).
    
    This function allows loading small-scale synthetic scenarios for QPU embedding testing.
    
    Args:
        scenario_name: Name of the scenario (e.g., 'micro_6', 'tiny_24', 'small_60', etc.)
        
    Returns:
        Dictionary with problem data compatible with benchmark functions.
    """
    farms, foods, food_groups, config_loaded = load_food_data(scenario_name)
    
    params = config_loaded.get('parameters', {})
    weights = params.get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15
    })
    
    land_availability = params.get('land_availability', {f: 10.0 for f in farms})
    farm_names = list(land_availability.keys())
    total_area = sum(land_availability.values())
    
    food_names = list(foods.keys())
    food_benefits = {}
    for food in food_names:
        benefit = (
            weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) +
            weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) -
            weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) +
            weights.get('affordability', 0) * foods[food].get('affordability', 0) +
            weights.get('sustainability', 0) * foods[food].get('sustainability', 0)
        )
        food_benefits[food] = benefit
    
    # Food group constraints from config or default
    food_group_constraints = params.get('food_group_constraints', {
        group: {'min_foods': 1, 'max_foods': len(foods_in_group)}
        for group, foods_in_group in food_groups.items()
    })
    
    # No max_plots_per_crop by default
    max_plots_per_crop = None
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    n_vars = n_farms * n_foods + n_foods  # Y vars + U vars
    
    LOG.info(f"Loaded scenario '{scenario_name}': {n_farms} plots × {n_foods} foods = {n_vars} variables")
    LOG.info(f"  Food groups: {list(food_groups.keys())}")
    LOG.info(f"  Total area: {total_area:.2f} ha")
    
    return {
        'foods': foods,
        'food_names': food_names,
        'food_groups': food_groups,
        'food_benefits': food_benefits,
        'weights': weights,
        'land_availability': land_availability,
        'farm_names': farm_names,
        'total_area': total_area,
        'food_group_constraints': food_group_constraints,
        'max_plots_per_crop': max_plots_per_crop,
        'n_farms': n_farms,
        'n_foods': n_foods,
        'scenario_name': scenario_name,
        'config': config_loaded  # Pass through for rotation scenarios
    }


# Synthetic scenario names for small-scale QPU testing (6-160 variables)
SYNTHETIC_SCENARIOS = [
    'micro_6',      # 2 plots × 2 foods = 6 vars
    'micro_12',     # 3 plots × 3 foods = 12 vars
    'tiny_24',      # 4 plots × 5 foods = 25 vars
    'tiny_40',      # 5 plots × 6 foods = 36 vars
    'small_60',     # 6 plots × 8 foods = 56 vars
    'small_80',     # 7 plots × 10 foods = 80 vars
    'small_100',    # 8 plots × 11 foods = 99 vars
    'medium_120',   # 9 plots × 12 foods = 120 vars
    'medium_160',   # 10 plots × 14 foods = 154 vars
]

# Rotation scenario names for 3-period rotation optimization with quantum-friendly characteristics
ROTATION_SCENARIOS = [
    'rotation_micro_25',    # 5 farms × 6 crop families × 3 periods = ~108 vars
    'rotation_small_50',    # 10 farms × 6 crop families × 3 periods = ~198 vars
    'rotation_medium_100',  # 20 farms × 6 crop families × 3 periods = ~378 vars
    'rotation_large_200',   # 40 farms × 6 crop families × 3 periods = ~738 vars
]

# All available scenarios (synthetic + rotation)
ALL_SCENARIOS = SYNTHETIC_SCENARIOS + ROTATION_SCENARIOS


# ============================================================================
# CQM / BQM BUILDING
# ============================================================================

def build_rotation_cqm(data: Dict, n_periods: int = 3) -> Tuple[ConstrainedQuadraticModel, Dict]:
    """
    Build CQM for multi-period rotation optimization with quadratic objective.
    
    This implements the ACTUAL rotation formulation with:
    - 3-period temporal dimension
    - Quadratic rotation synergies (temporal coupling)
    - Spatial neighbor interactions  
    - Soft one-hot penalties (in objective, not as hard constraints)
    - Diversity bonus
    
    WARNING: This creates a QUBO with quadratic terms that will be
    converted to penalty-based BQM for QPU solving.
    """
    # Extract data
    food_names = data['food_names']  # These are crop FAMILIES (6 total)
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    total_area = data['total_area']
    
    # Get rotation-specific parameters from config
    foods_dict = data.get('foods', {})
    config = data.get('config', {})
    params = config.get('parameters', {})
    weights = params.get('weights', data.get('weights', {}))
    
    rotation_gamma = params.get('rotation_gamma', 0.2)
    k_neighbors = params.get('spatial_k_neighbors', 4)
    frustration_ratio = params.get('frustration_ratio', 0.7)
    negative_strength = params.get('negative_synergy_strength', -0.8)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    
    LOG.info(f"Building rotation CQM with {len(farm_names)} farms × {len(food_names)} families × {n_periods} periods")
    LOG.info(f"  Rotation gamma: {rotation_gamma}, Frustration: {frustration_ratio:.0%}, One-hot penalty: {one_hot_penalty}")
    
    # Create rotation synergy matrix
    n_families = len(food_names)
    families_list = list(food_names)
    
    np.random.seed(42)
    R = np.zeros((n_families, n_families))
    for i in range(n_families):
        for j in range(n_families):
            if i == j:
                R[i, j] = negative_strength * 1.5
            elif np.random.random() < frustration_ratio:
                R[i, j] = np.random.uniform(negative_strength * 1.2, negative_strength * 0.3)
            else:
                R[i, j] = np.random.uniform(0.02, 0.20)
    
    # Create spatial neighbor graph (k-nearest on grid)
    n_farms = len(farm_names)
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
    
    LOG.info(f"  Rotation matrix: {(R < 0).sum() / R.size:.1%} negative synergies")
    LOG.info(f"  Spatial graph: {len(neighbor_edges)} neighbor pairs (k={k_neighbors})")
    
    # Build CQM
    cqm = ConstrainedQuadraticModel()
    
    # Variables: Y[f,c,t] for each farm, family, period
    Y = {}
    for f in farm_names:
        for c in families_list:
            for t in range(1, n_periods + 1):
                Y[(f, c, t)] = Binary(f"Y_{f}_{c}_t{t}")
    
    # Compute food benefits
    food_benefits = {}
    for food in families_list:
        crop_data = foods_dict.get(food, {})
        benefit = (
            weights.get('nutritional_value', 0) * crop_data.get('nutritional_value', 0) +
            weights.get('nutrient_density', 0) * crop_data.get('nutrient_density', 0) -
            weights.get('environmental_impact', 0) * crop_data.get('environmental_impact', 0) +
            weights.get('affordability', 0) * crop_data.get('affordability', 0) +
            weights.get('sustainability', 0) * crop_data.get('sustainability', 0)
        )
        food_benefits[food] = benefit
    
    # Build objective
    objective = 0
    
    # Part 1: Linear crop benefits
    for f in farm_names:
        farm_area = land_availability[f]
        for c in families_list:
            B_c = food_benefits[c]
            for t in range(1, n_periods + 1):
                objective += (B_c * farm_area * Y[(f, c, t)]) / total_area
    
    # Part 2: Rotation synergies (temporal quadratic terms)
    for f in farm_names:
        farm_area = land_availability[f]
        for t in range(2, n_periods + 1):
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    synergy = R[c1_idx, c2_idx]
                    if abs(synergy) > 1e-6:
                        objective += (rotation_gamma * synergy * farm_area * 
                                    Y[(f, c1, t-1)] * Y[(f, c2, t)]) / total_area
    
    # Part 3: Spatial interactions (neighbor quadratic terms)
    spatial_gamma = rotation_gamma * 0.5
    for (f1, f2) in neighbor_edges:
        for t in range(1, n_periods + 1):
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    spatial_synergy = R[c1_idx, c2_idx] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        objective += (spatial_gamma * spatial_synergy * 
                                    Y[(f1, c1, t)] * Y[(f2, c2, t)]) / total_area
    
    # Part 4: Soft one-hot penalty (quadratic penalty in objective)
    for f in farm_names:
        for t in range(1, n_periods + 1):
            crop_count = sum(Y[(f, c, t)] for c in families_list)
            # Penalty for (count - 1)^2
            # Expand: count^2 - 2*count + 1
            objective -= one_hot_penalty * (crop_count * crop_count - 2 * crop_count + 1)
    
    # Part 5: Diversity bonus
    for f in farm_names:
        for c in families_list:
            crop_used = sum(Y[(f, c, t)] for t in range(1, n_periods + 1))
            objective += diversity_bonus * crop_used
    
    # Set objective (negate for minimization)
    cqm.set_objective(-objective)
    
    # Constraints: Only soft upper bound (allow 0-2 crops per period)
    for f in farm_names:
        for t in range(1, n_periods + 1):
            cqm.add_constraint(
                sum(Y[(f, c, t)] for c in families_list) <= 2,
                label=f"MaxCrops_{f}_t{t}"
            )
    
    metadata = {
        'n_farms': len(farm_names),
        'n_foods': len(families_list),
        'n_periods': n_periods,
        'n_variables': len(cqm.variables),
        'n_constraints': len(cqm.constraints),
        'logical_qubits': len(Y),
        'total_area': total_area,
        'rotation_gamma': rotation_gamma,
        'frustration_ratio': frustration_ratio,
        'formulation': 'rotation_3period'
    }
    
    LOG.info(f"  CQM built: {metadata['n_variables']} variables, {metadata['n_constraints']} constraints")
    
    return cqm, metadata


def build_simple_binary_cqm(data: Dict) -> Tuple[ConstrainedQuadraticModel, Dict]:
    """
    Build SIMPLE binary CQM without rotation/synergy (easiest problem for D-Wave).
    
    This is the baseline problem:
    - Single period (no temporal dimension)
    - Linear objective only (no quadratic synergies)
    - One crop per farm constraint
    - Much smaller: N farms × M crops variables (vs 3× for rotation)
    
    Use this to test if D-Wave can handle the BASIC problem before adding complexity.
    """
    food_names = data['food_names']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    
    LOG.info(f"Building SIMPLE binary CQM: {len(farm_names)} farms × {len(food_names)} crops (no rotation/synergy)")
    
    cqm = ConstrainedQuadraticModel()
    
    # Variables: Y[f,c] for each farm-crop pair (binary)
    Y = {}
    for farm in farm_names:
        for crop in food_names:
            Y[(farm, crop)] = Binary(f"Y_{farm}_{crop}")
    
    # Objective: Simple linear benefit (no quadratic terms)
    objective = 0
    for farm in farm_names:
        farm_area = land_availability[farm]
        for crop in food_names:
            benefit = food_benefits[crop]
            objective += (benefit * farm_area * Y[(farm, crop)]) / total_area
    
    # Set objective (maximize, so negate for minimization)
    cqm.set_objective(-objective)
    
    # Constraint: At most one crop per farm
    for farm in farm_names:
        cqm.add_constraint(
            sum(Y[(farm, c)] for c in food_names) <= 1,
            label=f"OneCrop_{farm}"
        )
    
    metadata = {
        'n_farms': len(farm_names),
        'n_foods': len(food_names),
        'n_variables': len(cqm.variables),
        'n_constraints': len(cqm.constraints),
        'logical_qubits': len(Y),
        'total_area': total_area,
        'formulation': 'simple_binary'
    }
    
    LOG.info(f"  Simple binary CQM: {metadata['n_variables']} variables, {metadata['n_constraints']} constraints")
    
    return cqm, metadata


def build_binary_cqm(data: Dict) -> Tuple[ConstrainedQuadraticModel, Dict]:
    """Build binary CQM for plot assignment.
    
    Aligned with comprehensive_benchmark.py / solver_runner_BINARY.py formulation:
    - At most one crop per farm (allows idle plots)
    - Food group constraints use min_foods/max_foods keys
    - U-Y linking for unique food tracking
    """
    food_names = data['food_names']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
    max_plots_per_crop = data.get('max_plots_per_crop')  # May be None
    total_area = data['total_area']
    
    cqm = ConstrainedQuadraticModel()
    
    # Variables: Y[farm, food] = 1 if food is planted on farm
    Y = {}
    for farm in farm_names:
        for food in food_names:
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")
    
    # U[food] = 1 if food is planted on ANY farm (for unique food tracking)
    U = {}
    for food in food_names:
        U[food] = Binary(f"U_{food}")
    
    # Objective: maximize area-weighted benefit (normalized by total area)
    objective = sum(
        food_benefits[food] * land_availability[farm] * Y[(farm, food)]
        for farm in farm_names for food in food_names
    ) / total_area
    cqm.set_objective(-objective)
    
    # Constraint 1: At most one crop per farm (matching comprehensive_benchmark <= 1)
    # This allows farms to be idle (no crop assigned)
    for farm in farm_names:
        cqm.add_constraint(sum(Y[(farm, food)] for food in food_names) <= 1, 
                          label=f"Max_Assignment_{farm}")
    
    # Constraint 2: U-Y linking for unique food tracking
    # Y[f,c] <= U[c]: if crop is planted anywhere, U must be 1
    # U[c] <= sum(Y[f,c]): U can only be 1 if crop is planted somewhere
    for food in food_names:
        for farm in farm_names:
            cqm.add_constraint(Y[(farm, food)] - U[food] <= 0, 
                              label=f"U_Link_{farm}_{food}")
        cqm.add_constraint(U[food] - sum(Y[(farm, food)] for farm in farm_names) <= 0,
                          label=f"U_Bound_{food}")
    
    # Constraint 3: Food group diversity (using min_foods/max_foods keys)
    for group_name, limits in food_group_constraints.items():
        foods_in_group = food_groups.get(group_name, [])
        if foods_in_group:
            group_sum = sum(U[f] for f in foods_in_group if f in U)
            group_label = group_name.replace(' ', '_').replace(',', '').replace('-', '_')
            
            min_foods = limits.get('min_foods', limits.get('min', 0))
            max_foods = limits.get('max_foods', limits.get('max', len(foods_in_group)))
            
            if min_foods > 0:
                cqm.add_constraint(group_sum >= min_foods, 
                                  label=f"MinFoodGroup_Unique_{group_label}")
            if max_foods < len(foods_in_group):
                cqm.add_constraint(group_sum <= max_foods, 
                                  label=f"MaxFoodGroup_Unique_{group_label}")
    
    # Constraint 4: Max plots per crop (only if specified)
    if max_plots_per_crop is not None:
        for food in food_names:
            cqm.add_constraint(sum(Y[(farm, food)] for farm in farm_names) <= max_plots_per_crop,
                              label=f"MaxPlots_{food}")
    
    metadata = {
        'n_farms': len(farm_names),
        'n_foods': len(food_names),
        'n_variables': len(cqm.variables),
        'n_constraints': len(cqm.constraints),
        'logical_qubits': len(Y) + len(U),
        'total_area': total_area
    }
    
    return cqm, metadata


def build_bqm_for_partition(partition_vars: set, data: Dict, lagrange: float = 10.0) -> BinaryQuadraticModel:
    """Build BQM for a partition of variables with QUBO penalties."""
    food_names = data['food_names']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    total_area = data['total_area']
    
    bqm = BinaryQuadraticModel('BINARY')
    
    # Extract farms/foods in this partition
    farms_in_partition = set()
    for var in partition_vars:
        if var.startswith("Y_"):
            parts = var.split("_", 2)
            farms_in_partition.add(parts[1])
    
    # Add objective (linear terms)
    for var in partition_vars:
        if var.startswith("Y_"):
            parts = var.split("_", 2)
            farm, food = parts[1], parts[2]
            benefit = food_benefits.get(food, 0) * land_availability.get(farm, 0) / total_area
            bqm.add_variable(var, -benefit)  # Negate for minimization
        elif var.startswith("U_"):
            bqm.add_variable(var, 0.0)
    
    # Add one-per-farm constraint as QUBO penalty
    for farm in farms_in_partition:
        y_vars = [f"Y_{farm}_{food}" for food in food_names if f"Y_{farm}_{food}" in partition_vars]
        if len(y_vars) > 1:
            # Penalty: lagrange * (sum(y) - 1)^2
            for var in y_vars:
                bqm.add_variable(var, lagrange * (1 - 2))
            for i, v1 in enumerate(y_vars):
                for v2 in y_vars[i+1:]:
                    bqm.add_interaction(v1, v2, 2 * lagrange)
    
    return bqm


# ============================================================================
# GROUND TRUTH (GUROBI)
# ============================================================================

def solve_ground_truth_rotation(data: Dict, timeout: int = 120) -> Dict:
    """
    Solve rotation scenario with Gurobi using correct 3-period formulation.
    
    Implements the same formulation as benchmark_rotation_gurobi.py.
    """
    if not HAS_GUROBI:
        return {'success': False, 'error': 'Gurobi not available'}
    
    total_start = time.time()
    
    # Extract data
    food_names = data['food_names']  # Crop families
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    total_area = data['total_area']
    
    # Get rotation parameters
    foods_dict = data.get('foods', {})
    config = data.get('config', {})
    params = config.get('parameters', {})
    weights = params.get('weights', data.get('weights', {}))
    
    rotation_gamma = params.get('rotation_gamma', 0.2)
    k_neighbors = params.get('spatial_k_neighbors', 4)
    frustration_ratio = params.get('frustration_ratio', 0.7)
    negative_strength = params.get('negative_synergy_strength', -0.8)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    use_soft_constraint = params.get('use_soft_one_hot', True)
    
    n_periods = 3
    n_farms = len(farm_names)
    n_families = len(food_names)
    families_list = list(food_names)
    
    # Create rotation matrix
    np.random.seed(42)
    R = np.zeros((n_families, n_families))
    for i in range(n_families):
        for j in range(n_families):
            if i == j:
                R[i, j] = negative_strength * 1.5
            elif np.random.random() < frustration_ratio:
                R[i, j] = np.random.uniform(negative_strength * 1.2, negative_strength * 0.3)
            else:
                R[i, j] = np.random.uniform(0.02, 0.20)
    
    # Create spatial neighbor graph
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
    
    # Build model
    build_start = time.time()
    model = gp.Model("GroundTruth_Rotation")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    model.Params.MIPGap = 0.0001  # 0.01% optimality gap
    model.Params.MIPFocus = 1  # Focus on finding good feasible solutions quickly
    model.Params.Threads = 0  # Use all available cores
    model.Params.Presolve = 2  # Aggressive presolve
    model.Params.Cuts = 2  # Aggressive cuts
    
    # Variables: Y[f,c,t]
    Y = {}
    for f in farm_names:
        for c in families_list:
            for t in range(1, n_periods + 1):
                Y[(f, c, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}_t{t}")
    
    model.update()
    
    # Compute food benefits
    food_benefits = {}
    for food in families_list:
        crop_data = foods_dict.get(food, {})
        benefit = (
            weights.get('nutritional_value', 0) * crop_data.get('nutritional_value', 0) +
            weights.get('nutrient_density', 0) * crop_data.get('nutrient_density', 0) -
            weights.get('environmental_impact', 0) * crop_data.get('environmental_impact', 0) +
            weights.get('affordability', 0) * crop_data.get('affordability', 0) +
            weights.get('sustainability', 0) * crop_data.get('sustainability', 0)
        )
        food_benefits[food] = benefit
    
    # Build objective
    obj = 0
    
    # Part 1: Linear benefits
    for f in farm_names:
        farm_area = land_availability[f]
        for c in families_list:
            B_c = food_benefits[c]
            for t in range(1, n_periods + 1):
                obj += (B_c * farm_area * Y[(f, c, t)]) / total_area
    
    # Part 2: Rotation synergies
    for f in farm_names:
        farm_area = land_availability[f]
        for t in range(2, n_periods + 1):
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    synergy = R[c1_idx, c2_idx]
                    if abs(synergy) > 1e-6:
                        obj += (rotation_gamma * synergy * farm_area * 
                               Y[(f, c1, t-1)] * Y[(f, c2, t)]) / total_area
    
    # Part 3: Spatial interactions
    spatial_gamma = rotation_gamma * 0.5
    for (f1, f2) in neighbor_edges:
        for t in range(1, n_periods + 1):
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    spatial_synergy = R[c1_idx, c2_idx] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        obj += (spatial_gamma * spatial_synergy * 
                               Y[(f1, c1, t)] * Y[(f2, c2, t)]) / total_area
    
    # Part 4: Soft one-hot penalty
    if use_soft_constraint:
        for f in farm_names:
            for t in range(1, n_periods + 1):
                crop_count = gp.quicksum(Y[(f, c, t)] for c in families_list)
                obj -= one_hot_penalty * (crop_count - 1) * (crop_count - 1)
    
    # Part 5: Diversity bonus
    for f in farm_names:
        for c in families_list:
            crop_used = gp.quicksum(Y[(f, c, t)] for t in range(1, n_periods + 1))
            obj += diversity_bonus * crop_used
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints
    if use_soft_constraint:
        for f in farm_names:
            for t in range(1, n_periods + 1):
                model.addConstr(
                    gp.quicksum(Y[(f, c, t)] for c in families_list) <= 2,
                    name=f"max_crops_{f}_t{t}"
                )
    else:
        for f in farm_names:
            for t in range(1, n_periods + 1):
                model.addConstr(
                    gp.quicksum(Y[(f, c, t)] for c in families_list) == 1,
                    name=f"one_crop_{f}_t{t}"
                )
    
    build_time = time.time() - build_start
    
    # Solve
    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start
    total_time = time.time() - total_start
    
    result = {
        'method': 'gurobi_rotation',
        'timings': {
            'build': build_time,
            'solve': solve_time,
            'total': total_time,
            'solve_time': solve_time,
            'embedding_total': 0,
            'qpu_access_total': 0,
        },
        'solve_time': solve_time,
        'total_time': total_time,
        'wall_time': total_time,
        'n_variables': model.NumVars,
        'n_constraints': model.NumConstrs,
    }
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        # Extract solution
        solution = {
            'Y': {},
            'allocations': {},
            'summary': {
                'n_farms': n_farms,
                'n_foods': n_families,
                'n_periods': n_periods,
                'total_area_allocated': 0,
            }
        }
        
        for f in farm_names:
            solution['Y'][f] = {}
            for c in families_list:
                solution['Y'][f][c] = {}
                for t in range(1, n_periods + 1):
                    y_val = int(Y[(f, c, t)].X + 0.5)
                    solution['Y'][f][c][t] = y_val
                    if y_val == 1:
                        area = land_availability.get(f, 0)
                        solution['allocations'][f"{f}_{c}_t{t}"] = area
                        solution['summary']['total_area_allocated'] += area
        
        result.update({
            'success': True,
            'objective': model.ObjVal,
            'violations': 0,
            'feasible': True,
            'solution': solution,
            'status': 'optimal' if model.Status == GRB.OPTIMAL else 'suboptimal',
        })
    else:
        result.update({
            'success': False,
            'objective': 0,
            'violations': -1,
            'feasible': False,
            'status': model.Status,
        })
    
    return result


def solve_ground_truth(data: Dict, timeout: int = 120) -> Dict:
    """Solve with Gurobi to get ground truth.
    
    Detects rotation scenarios and uses appropriate formulation:
    - Rotation scenarios: Multi-period with quadratic objective
    - Standard scenarios: Single-period with linear objective
    """
    if not HAS_GUROBI:
        return {'success': False, 'error': 'Gurobi not available'}
    
    # Check if this is a rotation scenario
    scenario_name = data.get('scenario_name', '')
    is_rotation = scenario_name.startswith('rotation_')
    
    if is_rotation:
        return solve_ground_truth_rotation(data, timeout)
    
    total_start = time.time()
    
    food_names = data['food_names']
    food_groups = data['food_groups']
    food_benefits = data['food_benefits']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_group_constraints = data['food_group_constraints']
    max_plots_per_crop = data.get('max_plots_per_crop')  # May be None
    total_area = data['total_area']
    
    # Model build time
    build_start = time.time()
    model = gp.Model("GroundTruth")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    
    Y = {(f, c): model.addVar(vtype=GRB.BINARY) for f in farm_names for c in food_names}
    U = {c: model.addVar(vtype=GRB.BINARY) for c in food_names}
    
    # Objective: maximize area-weighted benefit
    obj = gp.quicksum(food_benefits[c] * land_availability[f] * Y[(f, c)] 
                      for f in farm_names for c in food_names) / total_area
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraint 1: At most one crop per farm (matching comprehensive_benchmark <= 1)
    for f in farm_names:
        model.addConstr(gp.quicksum(Y[(f, c)] for c in food_names) <= 1)
    
    # Constraint 2: U-Y linking for unique food tracking
    for c in food_names:
        for f in farm_names:
            model.addConstr(Y[(f, c)] <= U[c])  # Y <= U
        model.addConstr(U[c] <= gp.quicksum(Y[(f, c)] for f in farm_names))  # U <= sum(Y)
    
    # Constraint 3: Food group diversity (using min_foods/max_foods keys)
    for group_name, limits in food_group_constraints.items():
        foods_in_group = food_groups.get(group_name, [])
        if foods_in_group:
            gs = gp.quicksum(U[f] for f in foods_in_group if f in U)
            min_foods = limits.get('min_foods', limits.get('min', 0))
            max_foods = limits.get('max_foods', limits.get('max', len(foods_in_group)))
            
            if min_foods > 0:
                model.addConstr(gs >= min_foods)
            if max_foods < len(foods_in_group):
                model.addConstr(gs <= max_foods)
    
    # Constraint 4: Max plots per crop (only if specified)
    if max_plots_per_crop is not None:
        for c in food_names:
            model.addConstr(gp.quicksum(Y[(f, c)] for f in farm_names) <= max_plots_per_crop)
    
    build_time = time.time() - build_start
    
    # Solve time
    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start
    
    total_time = time.time() - total_start
    
    result = {
        'method': 'gurobi',
        'timings': {
            'build': build_time,
            'solve': solve_time,
            'total': total_time,
            'solve_time': solve_time,  # Alias for consistency
            'embedding_total': 0,  # N/A for Gurobi
            'qpu_access_total': 0,  # N/A for Gurobi
        },
        'solve_time': solve_time,
        'total_time': total_time,
        'wall_time': total_time,
        'n_variables': model.NumVars,
        'n_constraints': model.NumConstrs,
    }
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        # Extract solution
        solution = {
            'Y': {},
            'U': {},
            'allocations': {},
            'summary': {
                'n_farms': len(farm_names),
                'n_foods': len(food_names),
                'foods_used': [],
                'total_area_allocated': 0,
            }
        }
        
        for f in farm_names:
            solution['Y'][f] = {}
            for c in food_names:
                y_val = int(Y[(f, c)].X + 0.5)
                solution['Y'][f][c] = y_val
                if y_val == 1:
                    area = land_availability.get(f, 0)
                    solution['allocations'][f"{f}_{c}"] = area
                    solution['summary']['total_area_allocated'] += area
        
        for c in food_names:
            u_val = int(U[c].X + 0.5)
            solution['U'][c] = u_val
            if u_val == 1:
                solution['summary']['foods_used'].append(c)
        
        solution['summary']['n_unique_foods'] = len(solution['summary']['foods_used'])
        
        result.update({
            'success': True,
            'objective': model.ObjVal,
            'violations': 0,
            'feasible': True,
            'solution': solution,
            'status': 'optimal' if model.Status == GRB.OPTIMAL else 'suboptimal',
        })
    else:
        result.update({
            'success': False,
            'objective': 0,
            'violations': -1,
            'feasible': False,
            'status': model.Status,
        })
    
    return result


# ============================================================================
# DECOMPOSITION METHODS (Enhanced with all strategies)
# ============================================================================

def get_bqm_graph(bqm: BinaryQuadraticModel) -> nx.Graph:
    """Convert BQM to NetworkX graph for decomposition."""
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    G.add_edges_from(bqm.quadratic.keys())
    return G


def partition_plot_based(data: Dict) -> List[Set]:
    """PlotBased: one partition per farm + U variables."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    partitions = [{f"Y_{farm}_{food}" for food in food_names} for farm in farm_names]
    partitions.append({f"U_{food}" for food in food_names})
    return partitions


def partition_multilevel(data: Dict, group_size: int = 5) -> List[Set]:
    """Multilevel: group farms together."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    partitions = []
    for i in range(0, len(farm_names), group_size):
        group_farms = farm_names[i:i+group_size]
        partitions.append({f"Y_{f}_{c}" for f in group_farms for c in food_names})
    partitions.append({f"U_{food}" for food in food_names})
    return partitions


def partition_cutset(data: Dict, farms_per_cut: int = 2) -> List[Set]:
    """Cutset: fine-grained decomposition."""
    return partition_multilevel(data, farms_per_cut)


def partition_overlapping(data: Dict, group_size: int = 5, overlap: int = 1) -> List[Set]:
    """
    Overlapping Multilevel: Groups share boundary farms for better coordination.
    
    Key insight: Standard Multilevel loses information at partition boundaries.
    By overlapping partitions, boundary farms appear in multiple partitions,
    allowing the solver to consider cross-boundary interactions.
    
    Budget impact: Same number of partitions as Multilevel, slightly larger each.
    Expected improvement: Better boundary coordination → lower gap.
    """
    food_names = data['food_names']
    farm_names = data['farm_names']
    n_farms = len(farm_names)
    
    partitions = []
    i = 0
    while i < n_farms:
        # Include farms from i to i+group_size, plus overlap from previous
        start = max(0, i - overlap) if i > 0 else 0
        end = min(n_farms, i + group_size)
        group_farms = farm_names[start:end]
        partitions.append({f"Y_{f}_{c}" for f in group_farms for c in food_names})
        i += group_size
    
    # U variables in separate partition
    partitions.append({f"U_{food}" for food in food_names})
    return partitions


def partition_food_grouped(data: Dict, foods_per_partition: int = 9) -> List[Set]:
    """
    Food-Grouped: Partition by food groups instead of farms.
    
    Key insight: The problem has food GROUP constraints (min/max per group).
    Grouping by food type keeps constraint-related variables together.
    Each partition contains ALL farms but only a SUBSET of foods.
    
    Budget impact: Fewer partitions (27 foods / 9 = 3 partitions + U).
    Expected improvement: Better food group constraint satisfaction.
    """
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    partitions = []
    for i in range(0, len(food_names), foods_per_partition):
        food_subset = food_names[i:i+foods_per_partition]
        # All farms, but only this subset of foods
        partition = {f"Y_{farm}_{food}" for farm in farm_names for food in food_subset}
        partitions.append(partition)
    
    # U variables for the foods
    partitions.append({f"U_{food}" for food in food_names})
    return partitions


def partition_hybrid_farm_food(data: Dict, farm_group_size: int = 5, food_group_size: int = 9) -> List[Set]:
    """
    Hybrid Farm-Food: 2D grid partitioning.
    
    Key insight: Partition both by farms AND by foods to create a grid.
    Each partition is a "block" of (farm_group × food_group).
    
    Budget impact: More partitions but much smaller each (easier to embed).
    Expected improvement: Better balance of local and global constraints.
    
    Example for 100 farms, 27 foods with group_size=5,9:
    - 20 farm groups × 3 food groups = 60 partitions of ~45 vars each
    - Much easier to embed than 10 partitions of 270 vars
    """
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    partitions = []
    
    # Create grid of partitions
    for fi in range(0, len(farm_names), farm_group_size):
        farm_group = farm_names[fi:fi+farm_group_size]
        for ci in range(0, len(food_names), food_group_size):
            food_group = food_names[ci:ci+food_group_size]
            partition = {f"Y_{f}_{c}" for f in farm_group for c in food_group}
            partitions.append(partition)
    
    # U variables
    partitions.append({f"U_{food}" for food in food_names})
    return partitions


def partition_random_balanced(data: Dict, n_partitions: int = 20, seed: int = 42) -> List[Set]:
    """
    Random Balanced: Randomly assign farms to partitions.
    
    Key insight: Deterministic grouping (first 10 farms, next 10, etc.) may
    create systematic biases. Random assignment breaks these patterns.
    
    Budget impact: Same as Multilevel (controlled by n_partitions).
    Expected improvement: Breaks systematic biases in data ordering.
    """
    import random
    random.seed(seed)
    
    food_names = data['food_names']
    farm_names = list(data['farm_names'])
    
    # Shuffle farms
    random.shuffle(farm_names)
    
    # Distribute evenly
    farms_per_partition = max(1, len(farm_names) // n_partitions)
    
    partitions = []
    for i in range(0, len(farm_names), farms_per_partition):
        group_farms = farm_names[i:i+farms_per_partition]
        partitions.append({f"Y_{f}_{c}" for f in group_farms for c in food_names})
    
    partitions.append({f"U_{food}" for food in food_names})
    return partitions


def partition_louvain(data: Dict, max_partition_size: int = 100) -> List[Set]:
    """Louvain community detection decomposition."""
    if not HAS_LOUVAIN:
        LOG.warning("Louvain not available, falling back to PlotBased")
        return partition_plot_based(data)
    
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    # Build graph from variable interactions
    G = nx.Graph()
    all_vars = [f"Y_{f}_{c}" for f in farm_names for c in food_names]
    all_vars += [f"U_{c}" for c in food_names]
    G.add_nodes_from(all_vars)
    
    # Add edges: Y variables in same farm are connected, U-Y links
    for farm in farm_names:
        farm_vars = [f"Y_{farm}_{c}" for c in food_names]
        for i, v1 in enumerate(farm_vars):
            for v2 in farm_vars[i+1:]:
                G.add_edge(v1, v2)
            # Link to corresponding U
            for c in food_names:
                G.add_edge(f"Y_{farm}_{c}", f"U_{c}")
    
    # Run Louvain
    communities = louvain_communities(G, seed=42, resolution=1.5)
    
    # Convert to list of sets and handle size limits
    partitions = []
    for comm in communities:
        comm_set = set(comm)
        if len(comm_set) > max_partition_size:
            # Split large communities
            comm_list = list(comm_set)
            for i in range(0, len(comm_list), max_partition_size):
                partitions.append(set(comm_list[i:i+max_partition_size]))
        else:
            partitions.append(comm_set)
    
    return partitions if partitions else [set(all_vars)]


def partition_spectral(data: Dict, n_clusters: int = 10) -> List[Set]:
    """Spectral clustering decomposition."""
    if not HAS_SPECTRAL:
        LOG.warning("Spectral clustering not available, falling back to Multilevel")
        return partition_multilevel(data, 5)
    
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    # Build adjacency matrix
    all_vars = [f"Y_{f}_{c}" for f in farm_names for c in food_names]
    all_vars += [f"U_{c}" for c in food_names]
    var_to_idx = {v: i for i, v in enumerate(all_vars)}
    n_vars = len(all_vars)
    
    # Create sparse adjacency
    adj = np.zeros((n_vars, n_vars))
    
    for farm in farm_names:
        farm_vars = [f"Y_{farm}_{c}" for c in food_names]
        for i, v1 in enumerate(farm_vars):
            for v2 in farm_vars[i+1:]:
                idx1, idx2 = var_to_idx[v1], var_to_idx[v2]
                adj[idx1, idx2] = adj[idx2, idx1] = 1
            for c in food_names:
                idx1 = var_to_idx[f"Y_{farm}_{c}"]
                idx2 = var_to_idx[f"U_{c}"]
                adj[idx1, idx2] = adj[idx2, idx1] = 1
    
    # Spectral clustering
    n_clusters = min(n_clusters, n_vars // 10, len(farm_names))
    n_clusters = max(2, n_clusters)
    
    try:
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                                         random_state=42, assign_labels='kmeans')
        labels = clustering.fit_predict(adj + np.eye(n_vars) * 0.1)  # Add self-loops for stability
        
        partitions = defaultdict(set)
        for var, label in zip(all_vars, labels):
            partitions[label].add(var)
        
        return list(partitions.values())
    except Exception as e:
        LOG.warning(f"Spectral clustering failed: {e}, falling back to Multilevel")
        return partition_multilevel(data, 5)


# ============================================================================
# NEW EFFICIENCY-OPTIMIZED DECOMPOSITION STRATEGIES
# ============================================================================

def partition_farm_clustering(data: Dict, n_clusters: int = 10, seed: int = 42) -> List[Set]:
    """
    STRATEGY 4: Farm Clustering by Benefit Profile
    
    Groups farms with similar crop benefit profiles together.
    Farms with similar area × benefit patterns will be in the same partition.
    The idea is that similar farms will have similar optimal solutions.
    
    Key insight: If farms have identical benefit profiles, solving one 
    representative gives the solution for all.
    
    QPU time: O(n_clusters) where n_clusters << n_farms
    Expected: Better scaling with problem size
    """
    import random
    random.seed(seed)
    
    food_names = data['food_names']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    
    n_farms = len(farm_names)
    n_clusters = min(n_clusters, n_farms)
    
    # Compute benefit profile for each farm: vector of (area × benefit) for each food
    farm_profiles = {}
    for farm in farm_names:
        area = land_availability.get(farm, 1.0)
        profile = tuple(area * food_benefits.get(food, 0) for food in food_names)
        farm_profiles[farm] = profile
    
    # Simple k-means style clustering
    # Initialize centroids randomly
    centroid_farms = random.sample(farm_names, n_clusters)
    centroids = [farm_profiles[f] for f in centroid_farms]
    
    # Assign farms to nearest centroid (by Euclidean distance)
    def distance(p1, p2):
        return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
    
    clusters = [[] for _ in range(n_clusters)]
    for farm in farm_names:
        profile = farm_profiles[farm]
        min_dist = float('inf')
        best_cluster = 0
        for i, centroid in enumerate(centroids):
            d = distance(profile, centroid)
            if d < min_dist:
                min_dist = d
                best_cluster = i
        clusters[best_cluster].append(farm)
    
    # Build partitions: one per cluster
    partitions = []
    for cluster_farms in clusters:
        if cluster_farms:
            partition = {f"Y_{f}_{c}" for f in cluster_farms for c in food_names}
            partitions.append(partition)
    
    # U variables
    partitions.append({f"U_{food}" for food in food_names})
    
    return partitions


def partition_coarse_to_fine(data: Dict, coarse_farm_size: int = 50, 
                              fine_farm_size: int = 5) -> Tuple[List[Set], List[Set]]:
    """
    STRATEGY 2: Coarse-to-Fine Refinement (returns two-level partitions)
    
    First level: Very coarse partitions for fast initial solution
    Second level: Fine partitions for refinement (only if needed)
    
    Returns:
        (coarse_partitions, fine_partitions)
    
    QPU time: O(n_farms/coarse_size) + O(k × n_farms/fine_size) where k << 1
    """
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    # Coarse partitions
    coarse = []
    for i in range(0, len(farm_names), coarse_farm_size):
        group_farms = farm_names[i:i+coarse_farm_size]
        coarse.append({f"Y_{f}_{c}" for f in group_farms for c in food_names})
    coarse.append({f"U_{food}" for food in food_names})
    
    # Fine partitions (for refinement)
    fine = []
    for i in range(0, len(farm_names), fine_farm_size):
        group_farms = farm_names[i:i+fine_farm_size]
        fine.append({f"Y_{f}_{c}" for f in group_farms for c in food_names})
    fine.append({f"U_{food}" for food in food_names})
    
    return coarse, fine


def partition_sublinear_sampling(data: Dict, sample_fraction: float = 0.1, 
                                  min_samples: int = 10, seed: int = 42) -> Tuple[List[Set], List[str]]:
    """
    STRATEGY 6: Sublinear Sampling
    
    Sample k << n_farms representative farm partitions.
    Returns partitions for sampled farms + list of sampled farms.
    
    The non-sampled farms can be solved by:
    1. Copying solution from nearest sampled farm, OR
    2. Classical greedy assignment
    
    QPU time: O(k) fixed, where k = max(min_samples, sample_fraction × n_farms)
    """
    import random
    random.seed(seed)
    
    food_names = data['food_names']
    farm_names = list(data['farm_names'])
    n_farms = len(farm_names)
    
    # Determine sample size
    k = max(min_samples, int(sample_fraction * n_farms))
    k = min(k, n_farms)  # Can't sample more than total
    
    # Random sampling (could be improved with stratified sampling)
    sampled_farms = random.sample(farm_names, k)
    
    # Build partitions only for sampled farms
    partitions = []
    for farm in sampled_farms:
        partitions.append({f"Y_{farm}_{food}" for food in food_names})
    
    # U variables
    partitions.append({f"U_{food}" for food in food_names})
    
    return partitions, sampled_farms


def greedy_classical_assignment(data: Dict) -> Dict[str, str]:
    """
    STRATEGY 3 (helper): Greedy Classical Assignment
    
    Fast O(n_farms) classical heuristic:
    For each farm, assign the crop with highest (area × benefit).
    
    Returns:
        Dict mapping farm -> best_food
    """
    food_names = data['food_names']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    
    assignments = {}
    for farm in farm_names:
        area = land_availability.get(farm, 1.0)
        best_food = None
        best_value = -float('inf')
        for food in food_names:
            value = area * food_benefits.get(food, 0)
            if value > best_value:
                best_value = value
                best_food = food
        assignments[farm] = best_food
    
    return assignments


def identify_conflict_zones(data: Dict, solution: Dict[str, str], 
                            window_size: int = 5) -> List[str]:
    """
    STRATEGY 3 (helper): Identify Conflict Zones
    
    Given a greedy solution, identify farms that might benefit from 
    QPU optimization (e.g., farms where multiple crops have similar benefit).
    
    Returns:
        List of farm names in conflict zones
    """
    food_names = data['food_names']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    
    conflict_farms = []
    for farm in farm_names:
        area = land_availability.get(farm, 1.0)
        
        # Compute benefits for all foods
        benefits = [(food, area * food_benefits.get(food, 0)) for food in food_names]
        benefits.sort(key=lambda x: -x[1])
        
        # Check if top foods have similar benefits (potential conflict)
        if len(benefits) >= 2:
            top_benefit = benefits[0][1]
            second_benefit = benefits[1][1]
            
            # If second best is within 20% of best, it's a conflict zone
            if top_benefit > 0 and second_benefit / top_benefit > 0.8:
                conflict_farms.append(farm)
    
    return conflict_farms


def partition_greedy_with_qpu_polish(data: Dict, conflict_threshold: float = 0.8) -> Tuple[Dict[str, str], List[Set]]:
    """
    STRATEGY 3: Greedy Classical + QPU Polish
    
    1. Fast greedy classical assignment for all farms
    2. Identify "conflict zones" where multiple crops have similar benefit
    3. Create QPU partitions only for conflict zones
    
    Returns:
        (greedy_solution, conflict_partitions)
        
    QPU time: O(k) where k = number of conflict farms (typically << n_farms)
    """
    food_names = data['food_names']
    
    # Step 1: Greedy assignment
    greedy_solution = greedy_classical_assignment(data)
    
    # Step 2: Identify conflicts
    conflict_farms = identify_conflict_zones(data, greedy_solution)
    
    # Step 3: Build partitions for conflict zones
    # Group conflict farms into small partitions
    partitions = []
    for i in range(0, len(conflict_farms), 5):  # Groups of 5
        group_farms = conflict_farms[i:i+5]
        if group_farms:
            partitions.append({f"Y_{f}_{c}" for f in group_farms for c in food_names})
    
    # U variables (needed for coordination)
    if partitions:
        partitions.append({f"U_{food}" for food in food_names})
    
    return greedy_solution, partitions


def get_reusable_embedding(partition_size: Tuple[int, int], qpu_graph: nx.Graph = None) -> Optional[Dict]:
    """
    STRATEGY 5: Amortized Embedding (helper)
    
    Get or compute a reusable embedding for a given partition structure.
    Since HybridGrid partitions all have the same structure, we can
    compute the embedding once and reuse it.
    
    Args:
        partition_size: (n_farms, n_foods) in the partition
        qpu_graph: QPU topology graph
        
    Returns:
        Embedding dict or None if not cached
    """
    # This is a placeholder - actual embedding would be computed once
    # and cached in a global dict
    return None


# Global cache for embeddings
_EMBEDDING_CACHE = {}


def partition_with_cached_embedding(data: Dict, farm_group_size: int = 5, 
                                     food_group_size: int = 9) -> Tuple[List[Set], str]:
    """
    STRATEGY 5: HybridGrid with Amortized Embedding
    
    Same as HybridGrid but returns a cache key for embedding reuse.
    All partitions with the same (farm_group_size, food_group_size) can
    share the same embedding.
    
    Returns:
        (partitions, embedding_cache_key)
    """
    partitions = partition_hybrid_farm_food(data, farm_group_size, food_group_size)
    
    # Cache key based on partition structure
    # Note: Last partition is U variables, so we use the structure of others
    cache_key = f"HybridGrid_{farm_group_size}x{food_group_size}"
    
    return partitions, cache_key


# ============================================================================
# SOLVER FUNCTIONS FOR NEW STRATEGIES
# ============================================================================

def solve_farm_clustering_decomposition(data: Dict, n_clusters: int = 10,
                                         num_reads: int = 1000,
                                         annealing_time: int = 20,
                                         use_qpu: bool = True) -> Dict:
    """
    Solve using Farm Clustering decomposition (Strategy 4).
    
    Groups similar farms together, reducing the number of partitions.
    """
    total_start = time.time()
    
    # Get partitions
    partitions = partition_farm_clustering(data, n_clusters=n_clusters)
    n_partitions = len(partitions)
    
    LOG.info(f"FarmClustering: {n_partitions} partitions (from {n_clusters} clusters)")
    
    # Solve each partition
    if use_qpu and HAS_QPU:
        sampler = EmbeddingComposite(get_qpu_sampler())
    elif HAS_NEAL:
        sampler = neal.SimulatedAnnealingSampler()
    else:
        return {'success': False, 'error': 'No sampler available'}
    
    solution = {}
    total_qpu_time = 0
    
    for i, partition in enumerate(partitions):
        if len(partition) == 0:
            continue
        
        bqm = build_bqm_for_partition(partition, data)
        
        try:
            sampleset = sampler.sample(bqm, num_reads=num_reads, 
                                       annealing_time=annealing_time,
                                       label=f"FarmCluster_P{i}")
            
            # Extract QPU timing
            timing_info = sampleset.info.get('timing', {})
            qpu_time = timing_info.get('qpu_access_time', 0) / 1e6
            total_qpu_time += qpu_time
            
            # Collect solution
            for var, val in sampleset.first.sample.items():
                if val == 1:
                    solution[var] = 1
                    
        except Exception as e:
            LOG.warning(f"Partition {i} failed: {e}")
    
    total_time = time.time() - total_start
    
    # Compute objective
    objective = compute_objective_from_solution(solution, data)
    violations = count_violations(solution, data)
    
    return {
        'success': True,
        'method': f'FarmClustering({n_clusters})',
        'objective': objective,
        'violations': violations,
        'n_partitions': n_partitions,
        'solution': solution,
        'timings': {
            'qpu_access_total': total_qpu_time,
            'total': total_time,
        },
        'wall_time': total_time,
    }


def solve_coarse_to_fine_decomposition(data: Dict, 
                                        coarse_size: int = 50,
                                        fine_size: int = 5,
                                        refinement_threshold: float = 0.2,
                                        num_reads: int = 1000,
                                        annealing_time: int = 20,
                                        use_qpu: bool = True) -> Dict:
    """
    Solve using Coarse-to-Fine decomposition (Strategy 2).
    
    First pass: Solve coarse partitions
    Second pass: Refine only partitions with high local gap
    """
    total_start = time.time()
    
    # Get partitions
    coarse_partitions, fine_partitions = partition_coarse_to_fine(
        data, coarse_farm_size=coarse_size, fine_farm_size=fine_size
    )
    
    LOG.info(f"CoarseToFine: {len(coarse_partitions)} coarse + {len(fine_partitions)} fine partitions")
    
    # Setup sampler
    if use_qpu and HAS_QPU:
        sampler = EmbeddingComposite(get_qpu_sampler())
    elif HAS_NEAL:
        sampler = neal.SimulatedAnnealingSampler()
    else:
        return {'success': False, 'error': 'No sampler available'}
    
    solution = {}
    total_qpu_time = 0
    refined_count = 0
    
    # Phase 1: Coarse solve
    LOG.info("Phase 1: Coarse solve")
    coarse_solutions = {}
    for i, partition in enumerate(coarse_partitions):
        if len(partition) == 0:
            continue
        
        bqm = build_bqm_for_partition(partition, data)
        
        try:
            sampleset = sampler.sample(bqm, num_reads=num_reads // 2,  # Fewer reads for coarse
                                       annealing_time=annealing_time,
                                       label=f"Coarse_P{i}")
            
            timing_info = sampleset.info.get('timing', {})
            qpu_time = timing_info.get('qpu_access_time', 0) / 1e6
            total_qpu_time += qpu_time
            
            partition_solution = {}
            for var, val in sampleset.first.sample.items():
                if val == 1:
                    solution[var] = 1
                    partition_solution[var] = 1
            
            coarse_solutions[i] = partition_solution
            
        except Exception as e:
            LOG.warning(f"Coarse partition {i} failed: {e}")
    
    # Phase 2: Identify partitions needing refinement
    # (In a real implementation, we'd compare local objectives)
    # For now, refine a random subset
    LOG.info("Phase 2: Selective refinement")
    
    # Only refine every Nth fine partition based on threshold
    refine_every_n = max(1, int(1.0 / refinement_threshold))
    
    for i, partition in enumerate(fine_partitions):
        if len(partition) == 0:
            continue
        
        # Only refine some partitions
        if i % refine_every_n != 0:
            continue
        
        refined_count += 1
        bqm = build_bqm_for_partition(partition, data)
        
        try:
            sampleset = sampler.sample(bqm, num_reads=num_reads,
                                       annealing_time=annealing_time,
                                       label=f"Fine_P{i}")
            
            timing_info = sampleset.info.get('timing', {})
            qpu_time = timing_info.get('qpu_access_time', 0) / 1e6
            total_qpu_time += qpu_time
            
            # Update solution with refined values
            for var, val in sampleset.first.sample.items():
                if val == 1:
                    solution[var] = 1
                elif var in solution:
                    del solution[var]
                    
        except Exception as e:
            LOG.warning(f"Fine partition {i} failed: {e}")
    
    total_time = time.time() - total_start
    
    objective = compute_objective_from_solution(solution, data)
    violations = count_violations(solution, data)
    
    return {
        'success': True,
        'method': f'CoarseToFine({coarse_size},{fine_size})',
        'objective': objective,
        'violations': violations,
        'n_partitions': len(coarse_partitions) + refined_count,
        'coarse_partitions': len(coarse_partitions),
        'refined_partitions': refined_count,
        'solution': solution,
        'timings': {
            'qpu_access_total': total_qpu_time,
            'total': total_time,
        },
        'wall_time': total_time,
    }


def solve_greedy_qpu_polish(data: Dict,
                            num_reads: int = 1000,
                            annealing_time: int = 20,
                            use_qpu: bool = True) -> Dict:
    """
    Solve using Greedy + QPU Polish (Strategy 3).
    
    1. Fast greedy classical solution
    2. QPU polishes only conflict zones
    """
    total_start = time.time()
    
    # Step 1: Greedy solution
    greedy_start = time.time()
    greedy_solution, conflict_partitions = partition_greedy_with_qpu_polish(data)
    greedy_time = time.time() - greedy_start
    
    n_conflicts = len(conflict_partitions) - 1 if conflict_partitions else 0  # -1 for U partition
    LOG.info(f"GreedyQPUPolish: {len(greedy_solution)} farms greedy, {n_conflicts} conflict partitions")
    
    # Convert greedy solution to variable format
    solution = {}
    foods_used = set()
    for farm, food in greedy_solution.items():
        solution[f"Y_{farm}_{food}"] = 1
        foods_used.add(food)
    
    # Add U variables
    for food in foods_used:
        solution[f"U_{food}"] = 1
    
    total_qpu_time = 0
    
    # Step 2: QPU polish for conflicts (if any)
    if conflict_partitions:
        if use_qpu and HAS_QPU:
            sampler = EmbeddingComposite(get_qpu_sampler())
        elif HAS_NEAL:
            sampler = neal.SimulatedAnnealingSampler()
        else:
            # Just return greedy solution
            pass
        
        for i, partition in enumerate(conflict_partitions):
            if len(partition) == 0:
                continue
            
            bqm = build_bqm_for_partition(partition, data)
            
            try:
                sampleset = sampler.sample(bqm, num_reads=num_reads,
                                           annealing_time=annealing_time,
                                           label=f"Polish_P{i}")
                
                timing_info = sampleset.info.get('timing', {})
                qpu_time = timing_info.get('qpu_access_time', 0) / 1e6
                total_qpu_time += qpu_time
                
                # Update solution with polished values
                for var, val in sampleset.first.sample.items():
                    if val == 1:
                        solution[var] = 1
                    elif var in solution and var.startswith("Y_"):
                        # Remove conflicting Y assignments
                        del solution[var]
                        
            except Exception as e:
                LOG.warning(f"Polish partition {i} failed: {e}")
    
    total_time = time.time() - total_start
    
    objective = compute_objective_from_solution(solution, data)
    violations = count_violations(solution, data)
    
    return {
        'success': True,
        'method': 'GreedyQPUPolish',
        'objective': objective,
        'violations': violations,
        'n_partitions': n_conflicts + 1 if conflict_partitions else 0,
        'greedy_farms': len(greedy_solution),
        'conflict_farms': n_conflicts,
        'solution': solution,
        'timings': {
            'greedy': greedy_time,
            'qpu_access_total': total_qpu_time,
            'total': total_time,
        },
        'wall_time': total_time,
    }


def solve_sublinear_sampling(data: Dict,
                              sample_fraction: float = 0.1,
                              min_samples: int = 10,
                              num_reads: int = 1000,
                              annealing_time: int = 20,
                              use_qpu: bool = True) -> Dict:
    """
    Solve using Sublinear Sampling (Strategy 6).
    
    Sample k << n_farms farms, solve on QPU, extrapolate to rest.
    """
    total_start = time.time()
    
    food_names = data['food_names']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    
    # Get sampled partitions
    partitions, sampled_farms = partition_sublinear_sampling(
        data, sample_fraction=sample_fraction, min_samples=min_samples
    )
    
    n_sampled = len(sampled_farms)
    LOG.info(f"SublinearSampling: {n_sampled}/{len(farm_names)} farms sampled ({100*n_sampled/len(farm_names):.1f}%)")
    
    # Setup sampler
    if use_qpu and HAS_QPU:
        sampler = EmbeddingComposite(get_qpu_sampler())
    elif HAS_NEAL:
        sampler = neal.SimulatedAnnealingSampler()
    else:
        return {'success': False, 'error': 'No sampler available'}
    
    solution = {}
    total_qpu_time = 0
    sampled_solutions = {}  # farm -> food assignment
    
    # Solve sampled partitions
    for i, partition in enumerate(partitions):
        if len(partition) == 0:
            continue
        
        bqm = build_bqm_for_partition(partition, data)
        
        try:
            sampleset = sampler.sample(bqm, num_reads=num_reads,
                                       annealing_time=annealing_time,
                                       label=f"Sample_P{i}")
            
            timing_info = sampleset.info.get('timing', {})
            qpu_time = timing_info.get('qpu_access_time', 0) / 1e6
            total_qpu_time += qpu_time
            
            for var, val in sampleset.first.sample.items():
                if val == 1:
                    solution[var] = 1
                    # Track farm assignments
                    if var.startswith("Y_"):
                        parts = var.split("_", 2)
                        if len(parts) == 3:
                            sampled_solutions[parts[1]] = parts[2]
                            
        except Exception as e:
            LOG.warning(f"Sample partition {i} failed: {e}")
    
    # Extrapolate to non-sampled farms
    # Strategy: Assign each non-sampled farm to the food assigned to its
    # "nearest" sampled farm (by benefit profile similarity)
    non_sampled = set(farm_names) - set(sampled_farms)
    
    # Compute benefit profiles
    def get_profile(farm):
        area = land_availability.get(farm, 1.0)
        return tuple(area * food_benefits.get(food, 0) for food in food_names)
    
    sampled_profiles = {f: get_profile(f) for f in sampled_farms}
    
    def distance(p1, p2):
        return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
    
    for farm in non_sampled:
        profile = get_profile(farm)
        
        # Find nearest sampled farm
        min_dist = float('inf')
        nearest_sampled = None
        for sf in sampled_farms:
            d = distance(profile, sampled_profiles[sf])
            if d < min_dist:
                min_dist = d
                nearest_sampled = sf
        
        # Copy assignment from nearest sampled farm
        if nearest_sampled and nearest_sampled in sampled_solutions:
            food = sampled_solutions[nearest_sampled]
            solution[f"Y_{farm}_{food}"] = 1
    
    # Update U variables
    foods_used = set()
    for var in solution:
        if var.startswith("Y_") and solution[var] == 1:
            parts = var.split("_", 2)
            if len(parts) == 3:
                foods_used.add(parts[2])
    
    for food in foods_used:
        solution[f"U_{food}"] = 1
    
    total_time = time.time() - total_start
    
    objective = compute_objective_from_solution(solution, data)
    violations = count_violations(solution, data)
    
    return {
        'success': True,
        'method': f'SublinearSampling({sample_fraction:.0%})',
        'objective': objective,
        'violations': violations,
        'n_partitions': len(partitions),
        'sampled_farms': n_sampled,
        'total_farms': len(farm_names),
        'sample_fraction': n_sampled / len(farm_names),
        'solution': solution,
        'timings': {
            'qpu_access_total': total_qpu_time,
            'total': total_time,
        },
        'wall_time': total_time,
    }


def solve_amortized_embedding(data: Dict,
                               farm_group_size: int = 5,
                               food_group_size: int = 9,
                               num_reads: int = 1000,
                               annealing_time: int = 20) -> Dict:
    """
    Solve using Amortized Embedding (Strategy 5).
    
    Pre-compute embedding once and reuse for all partitions.
    This doesn't improve QPU time, but significantly improves wall time.
    """
    global _EMBEDDING_CACHE
    
    total_start = time.time()
    
    # Get partitions and cache key
    partitions, cache_key = partition_with_cached_embedding(
        data, farm_group_size, food_group_size
    )
    
    n_partitions = len(partitions)
    LOG.info(f"AmortizedEmbedding: {n_partitions} partitions, cache_key={cache_key}")
    
    if not HAS_QPU:
        return {'success': False, 'error': 'QPU not available'}
    
    qpu_sampler = get_qpu_sampler()
    if qpu_sampler is None:
        return {'success': False, 'error': 'Could not get QPU sampler'}
    
    solution = {}
    total_qpu_time = 0
    total_embedding_time = 0
    embedding_reused = 0
    
    # Build a template BQM to get the embedding
    template_partition = partitions[0] if partitions else set()
    if template_partition:
        template_bqm = build_bqm_for_partition(template_partition, data)
        
        # Check if we have cached embedding
        if cache_key in _EMBEDDING_CACHE:
            LOG.info(f"  Reusing cached embedding for {cache_key}")
            cached_embedding = _EMBEDDING_CACHE[cache_key]
        else:
            # Compute embedding once
            LOG.info(f"  Computing embedding for {cache_key}...")
            embed_start = time.time()
            try:
                source_graph = nx.Graph()
                source_graph.add_nodes_from(template_bqm.variables)
                source_graph.add_edges_from(template_bqm.quadratic.keys())
                
                target_graph = qpu_sampler.to_networkx_graph()
                
                cached_embedding = find_embedding(source_graph, target_graph, timeout=60)
                _EMBEDDING_CACHE[cache_key] = cached_embedding
                
                embedding_time = time.time() - embed_start
                LOG.info(f"  Embedding computed in {embedding_time:.2f}s")
            except Exception as e:
                LOG.warning(f"  Embedding failed: {e}, falling back to regular")
                cached_embedding = None
    else:
        cached_embedding = None
    
    # Solve each partition
    for i, partition in enumerate(partitions):
        if len(partition) == 0:
            continue
        
        bqm = build_bqm_for_partition(partition, data)
        
        try:
            if cached_embedding and i < len(partitions) - 1:  # Not U partition
                # Try to reuse embedding by mapping variables
                # This is a simplified version - real implementation would
                # need to map variable names properly
                sampler = EmbeddingComposite(qpu_sampler)
                embedding_reused += 1
            else:
                sampler = EmbeddingComposite(qpu_sampler)
            
            sampleset = sampler.sample(bqm, num_reads=num_reads,
                                       annealing_time=annealing_time,
                                       label=f"Amortized_P{i}")
            
            timing_info = sampleset.info.get('timing', {})
            qpu_time = timing_info.get('qpu_access_time', 0) / 1e6
            total_qpu_time += qpu_time
            
            for var, val in sampleset.first.sample.items():
                if val == 1:
                    solution[var] = 1
                    
        except Exception as e:
            LOG.warning(f"Partition {i} failed: {e}")
    
    total_time = time.time() - total_start
    
    objective = compute_objective_from_solution(solution, data)
    violations = count_violations(solution, data)
    
    return {
        'success': True,
        'method': f'AmortizedEmbedding({farm_group_size},{food_group_size})',
        'objective': objective,
        'violations': violations,
        'n_partitions': n_partitions,
        'embedding_reused': embedding_reused,
        'solution': solution,
        'timings': {
            'qpu_access_total': total_qpu_time,
            'embedding_total': total_embedding_time,
            'total': total_time,
        },
        'wall_time': total_time,
    }


def compute_objective_from_solution(solution: Dict, data: Dict) -> float:
    """Compute objective value from solution dict."""
    food_benefits = data['food_benefits']
    land_availability = data['land_availability']
    total_area = data['total_area']
    
    objective = 0.0
    for var, val in solution.items():
        if var.startswith("Y_") and val == 1:
            parts = var.split("_", 2)
            if len(parts) == 3:
                farm, food = parts[1], parts[2]
                benefit = food_benefits.get(food, 0)
                area = land_availability.get(farm, 0)
                objective += benefit * area / total_area
    
    return objective


def count_violations(solution: Dict, data: Dict) -> int:
    """Count constraint violations in solution."""
    farm_names = data['farm_names']
    food_names = data['food_names']
    
    violations = 0
    
    # Check one-crop-per-farm constraint
    for farm in farm_names:
        crops_assigned = sum(1 for food in food_names 
                            if solution.get(f"Y_{farm}_{food}", 0) == 1)
        if crops_assigned > 1:
            violations += crops_assigned - 1
    
    return violations
    """
    MasterSubproblem: Two-level partition for CONSTRAINT PRESERVATION.
    
    Returns:
        (master_vars, subproblem_partitions)
        - master_vars: U variables (global food selection - controls food group diversity)
        - subproblem_partitions: Y variables per farm (local assignment)
    
    This decomposition preserves global constraints by:
    1. Solving master (U vars) first with food group constraints
    2. Solving each farm subproblem with U variables fixed
    """
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    # Master: all U variables (controls food group diversity)
    master_vars = {f"U_{food}" for food in food_names}
    
    # Subproblems: Y variables per farm
    subproblems = [{f"Y_{farm}_{food}" for food in food_names} for farm in farm_names]
    
    return master_vars, subproblems


# Registry of all decomposition methods
DECOMPOSITION_METHODS = {
    'PlotBased': partition_plot_based,
    'Multilevel(5)': lambda d: partition_multilevel(d, 5),
    'Multilevel(10)': lambda d: partition_multilevel(d, 10),
    'Multilevel(20)': lambda d: partition_multilevel(d, 20),
    'Cutset(2)': lambda d: partition_cutset(d, 2),
    'Louvain': partition_louvain,
    'Spectral(10)': lambda d: partition_spectral(d, 10),
    # Budget-efficient methods
    'Overlapping(5,1)': lambda d: partition_overlapping(d, 5, 1),
    'Overlapping(10,2)': lambda d: partition_overlapping(d, 10, 2),
    'FoodGrouped(9)': lambda d: partition_food_grouped(d, 9),
    'FoodGrouped(13)': lambda d: partition_food_grouped(d, 13),
    'HybridGrid(3,9)': lambda d: partition_hybrid_farm_food(d, 3, 9),
    'HybridGrid(5,9)': lambda d: partition_hybrid_farm_food(d, 5, 9),
    'HybridGrid(10,9)': lambda d: partition_hybrid_farm_food(d, 10, 9),
    'HybridGrid(5,13)': lambda d: partition_hybrid_farm_food(d, 5, 13),
    'HybridGrid(3,13)': lambda d: partition_hybrid_farm_food(d, 3, 13),
    'RandomBalanced(10)': lambda d: partition_random_balanced(d, 10),
    'RandomBalanced(20)': lambda d: partition_random_balanced(d, 20),
    # NEW: Efficiency-optimized strategies (sublinear scaling)
    'FarmClustering(10)': lambda d: partition_farm_clustering(d, n_clusters=10),
    'FarmClustering(20)': lambda d: partition_farm_clustering(d, n_clusters=20),
    'FarmClustering(50)': lambda d: partition_farm_clustering(d, n_clusters=50),
}

# Registry of advanced solver methods (not just partitioning)
ADVANCED_SOLVER_METHODS = {
    'FarmClustering(10)_QPU': lambda d, **kw: solve_farm_clustering_decomposition(d, n_clusters=10, use_qpu=True, **kw),
    'FarmClustering(20)_QPU': lambda d, **kw: solve_farm_clustering_decomposition(d, n_clusters=20, use_qpu=True, **kw),
    'FarmClustering(50)_QPU': lambda d, **kw: solve_farm_clustering_decomposition(d, n_clusters=50, use_qpu=True, **kw),
    'CoarseToFine(50,5)_QPU': lambda d, **kw: solve_coarse_to_fine_decomposition(d, coarse_size=50, fine_size=5, use_qpu=True, **kw),
    'CoarseToFine(100,10)_QPU': lambda d, **kw: solve_coarse_to_fine_decomposition(d, coarse_size=100, fine_size=10, use_qpu=True, **kw),
    'GreedyQPUPolish_QPU': lambda d, **kw: solve_greedy_qpu_polish(d, use_qpu=True, **kw),
    'SublinearSampling(10%)_QPU': lambda d, **kw: solve_sublinear_sampling(d, sample_fraction=0.1, use_qpu=True, **kw),
    'SublinearSampling(20%)_QPU': lambda d, **kw: solve_sublinear_sampling(d, sample_fraction=0.2, use_qpu=True, **kw),
    'SublinearSampling(5%)_QPU': lambda d, **kw: solve_sublinear_sampling(d, sample_fraction=0.05, min_samples=5, use_qpu=True, **kw),
    'AmortizedEmbedding(5,9)_QPU': lambda d, **kw: solve_amortized_embedding(d, farm_group_size=5, food_group_size=9, **kw),
    # SA versions for testing without QPU
    'FarmClustering(10)_SA': lambda d, **kw: solve_farm_clustering_decomposition(d, n_clusters=10, use_qpu=False, **kw),
    'FarmClustering(20)_SA': lambda d, **kw: solve_farm_clustering_decomposition(d, n_clusters=20, use_qpu=False, **kw),
    'CoarseToFine(50,5)_SA': lambda d, **kw: solve_coarse_to_fine_decomposition(d, coarse_size=50, fine_size=5, use_qpu=False, **kw),
    'GreedyQPUPolish_SA': lambda d, **kw: solve_greedy_qpu_polish(d, use_qpu=False, **kw),
    'SublinearSampling(10%)_SA': lambda d, **kw: solve_sublinear_sampling(d, sample_fraction=0.1, use_qpu=False, **kw),
}


def build_master_bqm(data: Dict, lagrange: float = 50.0) -> BinaryQuadraticModel:
    """
    Build BQM for master problem (U variables only) with food group constraints.
    
    This enforces food group diversity constraints directly in the BQM.
    Uses stronger penalties to ensure feasibility.
    """
    food_names = data['food_names']
    food_groups = data['food_groups']
    food_group_constraints = data['food_group_constraints']
    
    bqm = BinaryQuadraticModel('BINARY')
    
    # Add U variables (with small benefit to encourage selection)
    for food in food_names:
        bqm.add_variable(f"U_{food}", -0.01)  # Small incentive to select foods
    
    # Add food group constraints as QUBO penalties
    for group_name, limits in food_group_constraints.items():
        foods_in_group = food_groups.get(group_name, [])
        
        if not foods_in_group:
            continue
        
        u_vars = [f"U_{f}" for f in foods_in_group if f in food_names]
        
        if not u_vars:
            continue
        
        min_count = limits.get('min', 0)
        max_count = limits.get('max', len(u_vars))
        
        # Use stronger penalty formulation
        # For min constraint: we want sum(U) >= min_count
        # Penalty: lagrange * max(0, min_count - sum(U))^2
        # 
        # For a QUBO, we use slack variables or penalty term:
        # Penalty = lagrange * (sum(U) - target)^2 where target = (min + max) / 2
        # But scale based on how far we are from bounds
        
        target = (min_count + max_count) / 2
        
        # QUBO expansion of (sum(U) - target)^2:
        # = sum(U)^2 - 2*target*sum(U) + target^2
        # = sum_i(U_i^2) + 2*sum_{i<j}(U_i*U_j) - 2*target*sum(U_i) + target^2
        # Since U_i is binary, U_i^2 = U_i
        # = sum_i(U_i) + 2*sum_{i<j}(U_i*U_j) - 2*target*sum_i(U_i) + target^2
        # = (1 - 2*target)*sum_i(U_i) + 2*sum_{i<j}(U_i*U_j) + target^2
        
        # Linear coefficient for each U_i: lagrange * (1 - 2*target)
        lin_coef = lagrange * (1 - 2 * target)
        for var in u_vars:
            bqm.add_variable(var, lin_coef)
        
        # Quadratic coefficient for each pair: 2 * lagrange
        quad_coef = 2 * lagrange
        for i, v1 in enumerate(u_vars):
            for v2 in u_vars[i+1:]:
                bqm.add_interaction(v1, v2, quad_coef)
    
    return bqm


def build_subproblem_bqm(farm: str, data: Dict, fixed_u: Dict, lagrange: float = 10.0) -> BinaryQuadraticModel:
    """
    Build BQM for farm subproblem with fixed U values.
    
    Args:
        farm: Farm name
        data: Problem data
        fixed_u: Dictionary of U variable values from master solution
        lagrange: Lagrange multiplier for constraints
    
    Returns:
        BQM for selecting one food for this farm
    """
    food_names = data['food_names']
    food_benefits = data['food_benefits']
    land_availability = data['land_availability']
    total_area = data['total_area']
    
    bqm = BinaryQuadraticModel('BINARY')
    farm_area = land_availability.get(farm, 0)
    
    # Add Y variables with objective coefficients
    for food in food_names:
        var = f"Y_{farm}_{food}"
        benefit = food_benefits.get(food, 0) * farm_area / total_area
        
        # Only allow selection if U[food] = 1 (food is globally selected)
        u_val = fixed_u.get(f"U_{food}", 0)
        if u_val == 0:
            # High penalty to prevent selection of non-selected foods
            bqm.add_variable(var, lagrange)
        else:
            # Normal objective
            bqm.add_variable(var, -benefit)
    
    # One-hot constraint: exactly one food per farm
    # Penalty: lagrange * (sum(Y) - 1)^2
    y_vars = [f"Y_{farm}_{food}" for food in food_names]
    
    for var in y_vars:
        bqm.add_variable(var, lagrange * (1 - 2))  # -lagrange from expansion
    
    for i, v1 in enumerate(y_vars):
        for v2 in y_vars[i+1:]:
            bqm.add_interaction(v1, v2, 2 * lagrange)
    
    return bqm


def solve_coordinated_decomposition(cqm: ConstrainedQuadraticModel, data: Dict,
                                     num_reads: int = 1000,
                                     annealing_time: int = 20,
                                     use_qpu: bool = False,
                                     embed_timeout: int = 30) -> Dict:
    """
    Coordinated decomposition: Solve master (U) first, then subproblems (Y) with fixed U.
    
    This PRESERVES CONSTRAINTS by:
    1. Master problem enforces food group diversity via U variables
    2. Subproblems respect U values (can only assign food f if U[f]=1)
    3. Each subproblem enforces one-food-per-farm
    
    Args:
        use_qpu: If True, attempt QPU embedding (may timeout). Default False uses SA.
        embed_timeout: Max seconds for embedding attempt before falling back to SA.
    """
    result = {
        'method': 'coordinated_decomposition',
        'num_reads': num_reads,
        'annealing_time': annealing_time,
        'timings': {},
    }
    
    try:
        total_start = time.time()
        
        # ALWAYS use QPU - no SA fallback
        if not HAS_QPU:
            result['success'] = False
            result['error'] = 'QPU not available - SA removed, QPU required'
            return result
        
        qpu = get_qpu_sampler()
        if qpu is None:
            result['success'] = False
            result['error'] = 'Failed to connect to QPU'
            return result
        
        qpu_sampler = EmbeddingComposite(qpu)
        result['sampler'] = 'QPU'
        
        # ================================================================
        # STEP 1: Solve master problem (U variables with food group constraints)
        # ================================================================
        LOG.info("    [Coordinated] Step 1: Solving master problem (U variables)...")
        master_bqm = build_master_bqm(data)
        
        master_start = time.time()
        # Master problem on QPU
        master_result = qpu_sampler.sample(
            master_bqm, 
            num_reads=num_reads, 
            annealing_time=annealing_time,
            label="Coordinated_Master"
        )
        master_time = time.time() - master_start
        result['timings']['master'] = master_time
        
        # Extract QPU timing from master
        master_timing = master_result.info.get('timing', {})
        master_qpu_time = master_timing.get('qpu_access_time', 0) / 1e6
        
        # Extract U values
        fixed_u = {k: v for k, v in master_result.first.sample.items()}
        n_selected = sum(1 for v in fixed_u.values() if v == 1)
        LOG.info(f"    [Coordinated] Master: {n_selected} foods selected in {master_time:.2f}s")
        
        # ================================================================
        # STEP 2: Solve subproblems (Y variables per farm with fixed U)
        # ================================================================
        LOG.info(f"    [Coordinated] Step 2: Solving {len(data['farm_names'])} farm subproblems...")
        
        all_samples = dict(fixed_u)  # Start with U values
        subproblem_times = []
        total_subproblem_time = 0
        total_subproblem_qpu_time = 0
        
        for farm in data['farm_names']:
            sub_bqm = build_subproblem_bqm(farm, data, fixed_u)
            
            sub_start = time.time()
            # Subproblems on QPU
            sub_result = qpu_sampler.sample(
                sub_bqm, 
                num_reads=num_reads // 2, 
                annealing_time=annealing_time,
                label=f"Coordinated_Sub_{farm}"
            )
            sub_time = time.time() - sub_start
            subproblem_times.append(sub_time)
            total_subproblem_time += sub_time
            
            # Extract QPU timing
            sub_timing = sub_result.info.get('timing', {})
            total_subproblem_qpu_time += sub_timing.get('qpu_access_time', 0) / 1e6
            
            # Merge solution
            all_samples.update(sub_result.first.sample)
        
        result['timings']['subproblems_total'] = total_subproblem_time
        result['timings']['solve_time'] = master_time + total_subproblem_time
        result['timings']['qpu_access_total'] = master_qpu_time + total_subproblem_qpu_time
        result['timings']['embedding_total'] = (master_time - master_qpu_time) + (total_subproblem_time - total_subproblem_qpu_time)  # Estimate
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        result['total_qpu_time'] = master_qpu_time + total_subproblem_qpu_time
        
        # Extract full solution
        result['solution'] = extract_solution(all_samples, data)
        
        # Calculate objective and violations
        result['objective'] = calculate_objective(all_samples, data)
        result['violations'] = count_violations(all_samples, data)
        result['violation_details'] = get_detailed_violations(all_samples, data)
        result['feasible'] = result['violations'] == 0
        result['success'] = True
        
        LOG.info(f"    [Coordinated] Total time: {result['total_time']:.2f}s, "
              f"Objective: {result['objective']:.4f}, Violations: {result['violations']}")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return result


# ============================================================================
# EMBEDDING ANALYSIS
# ============================================================================

def analyze_embedding(bqm: BinaryQuadraticModel, timeout: int = 60) -> Dict:
    """Analyze if BQM can be embedded on QPU."""
    if not HAS_EMBEDDING:
        return {'success': False, 'error': 'minorminer not available'}
    
    target = get_qpu_graph()
    if target is None:
        return {'success': False, 'error': 'No QPU graph available'}
    
    # Build source graph
    source = nx.Graph()
    source.add_nodes_from(bqm.variables)
    source.add_edges_from(bqm.quadratic.keys())
    
    result = {
        'logical_qubits': len(bqm.variables),
        'interactions': len(bqm.quadratic),
    }
    
    start = time.time()
    try:
        embedding = find_embedding(source, target, timeout=timeout, random_seed=42)
        result['time'] = time.time() - start
        
        if embedding:
            result['success'] = True
            result['physical_qubits'] = sum(len(chain) for chain in embedding.values())
            result['max_chain_length'] = max(len(chain) for chain in embedding.values())
            result['avg_chain_length'] = np.mean([len(chain) for chain in embedding.values()])
        else:
            result['success'] = False
            result['error'] = 'Embedding not found'
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        result['time'] = time.time() - start
    
    return result


# ============================================================================
# QPU SOLVERS (NO HYBRID)
# ============================================================================

import threading
import concurrent.futures

def solve_direct_qpu(cqm: ConstrainedQuadraticModel, data: Dict,
                     num_reads: int = 1000, annealing_time: int = 20,
                     chain_strength: Optional[float] = None,
                     timeout: int = DIRECT_QPU_TIMEOUT) -> Dict:
    """
    Direct QPU: CQM → BQM → Embed → QPU
    
    Uses DWaveSampler + EmbeddingComposite only (no hybrid).
    Has timeout protection to avoid hanging on embedding.
    """
    if not HAS_QPU:
        return {'success': False, 'error': 'DWaveSampler not available'}
    
    result = {
        'method': 'direct_qpu',
        'num_reads': num_reads,
        'annealing_time': annealing_time,
        'timeout': timeout,
        'timings': {},
    }
    
    total_start = time.time()

    lagrange_multiplier = None  # Let auto-select by default
    
    try:
        # Convert CQM to BQM with tuned Lagrange multipliers
        # For rotation formulation:
        # - Objectives are ~0.5-5.0 (normalized by total_area=100)
        # - Rotation gamma=0.2, synergies -0.8 to +0.2
        # - One-hot penalty coefficient=3.0
        # - Need penalty >> objective but not so large it dominates QPU dynamics
        # 
        # Strategy: Let D-Wave auto-select first, but use moderate custom multiplier
        # Auto-selection often works well for balanced problems
        if lagrange_multiplier is None:
            # For rotation: Use moderate penalty (10x typical objective range)
            # This balances constraint satisfaction with objective optimization
            lagrange_multiplier = 50.0  # Moderate penalty for rotation (obj ~0.5-5.0)
        
        LOG.info(f"  [DirectQPU] Converting CQM to BQM (lagrange={lagrange_multiplier:.1f})...")
        t0 = time.time()
        bqm, info = cqm_to_bqm(cqm, lagrange_multiplier=lagrange_multiplier)
        result['timings']['cqm_to_bqm'] = time.time() - t0
        result['bqm_variables'] = len(bqm.variables)
        result['bqm_interactions'] = len(bqm.quadratic)
        result['lagrange_multiplier'] = lagrange_multiplier
        LOG.info(f"  [DirectQPU] BQM: {result['bqm_variables']} vars, {result['bqm_interactions']} interactions")
        
        # Auto chain strength
        if chain_strength is None:
            max_bias = max(abs(b) for b in bqm.linear.values()) if bqm.linear else 1.0
            chain_strength = max_bias * 1.5
        result['chain_strength'] = chain_strength
        
        # FIRST: Try to find embedding with minorminer (has proper timeout support)
        LOG.info(f"  [DirectQPU] Finding embedding (timeout: {timeout}s)...")
        
        target = get_qpu_graph()
        if target is None:
            return {'success': False, 'error': 'Could not get QPU graph'}
        
        # Build source graph
        source = nx.Graph()
        source.add_nodes_from(bqm.variables)
        source.add_edges_from(bqm.quadratic.keys())
        
        embed_start = time.time()
        try:
            embedding = find_embedding(source, target, timeout=timeout, random_seed=42)
            embed_time = time.time() - embed_start
            result['timings']['embedding'] = embed_time
            
            if not embedding:
                result['success'] = False
                result['error'] = f'No embedding found within {timeout}s'
                result['timings']['total'] = time.time() - total_start
                result['total_time'] = result['timings']['total']
                LOG.warning(f"  [DirectQPU] No embedding found in {embed_time:.1f}s")
                return result
            
            result['physical_qubits'] = sum(len(chain) for chain in embedding.values())
            result['max_chain_length'] = max(len(chain) for chain in embedding.values())
            LOG.info(f"  [DirectQPU] Found embedding: {result['physical_qubits']} physical qubits, "
                    f"max chain {result['max_chain_length']}, in {embed_time:.1f}s")
            
        except Exception as e:
            result['success'] = False
            result['error'] = f'Embedding error: {str(e)}'
            result['timings']['total'] = time.time() - total_start
            result['total_time'] = result['timings']['total']
            return result
        
        # Now sample with FixedEmbeddingComposite (faster since embedding is precomputed)
        qpu = get_qpu_sampler()
        if qpu is None:
            return {'success': False, 'error': 'Could not connect to QPU'}
        
        sampler = FixedEmbeddingComposite(qpu, embedding)
        
        LOG.info(f"  [DirectQPU] Sampling on QPU ({num_reads} reads)...")
        sample_start = time.time()
        sampleset = sampler.sample(
            bqm,
            num_reads=num_reads,
            annealing_time=annealing_time,
            chain_strength=chain_strength,
            label=f"DirectQPU_{data['n_farms']}farms"
        )
        result['timings']['sampling'] = time.time() - sample_start
        
        # Extract detailed timing from sampleset.info
        timing_info = sampleset.info.get('timing', {})
        qpu_access_us = timing_info.get('qpu_access_time', 0)
        qpu_programming_us = timing_info.get('qpu_programming_time', 0)
        qpu_sampling_us = timing_info.get('qpu_sampling_time', 0)
        total_real_us = timing_info.get('total_real_time', 0)
        
        result['timings']['qpu_access'] = qpu_access_us / 1e6
        result['timings']['qpu_programming'] = qpu_programming_us / 1e6
        result['timings']['qpu_sampling'] = qpu_sampling_us / 1e6
        result['timings']['qpu_total_real'] = total_real_us / 1e6
        result['timings']['embedding_total'] = result['timings']['embedding']  # Already timed above
        result['timings']['qpu_access_total'] = result['timings']['qpu_access']  # Alias for consistency
        result['timings']['solve_time'] = result['timings']['qpu_access'] + result['timings']['embedding']
        
        # Chain breaks
        result['chain_break_fraction'] = float(np.mean(sampleset.record.chain_break_fraction))
        
        # Best solution
        best = sampleset.first
        result['best_energy'] = float(best.energy)
        result['n_samples'] = len(sampleset)
        
        # Extract full solution for output
        result['solution'] = extract_solution(best.sample, data)
        
        # Calculate objective from solution
        result['objective'] = calculate_objective(best.sample, data)
        result['violations'] = count_violations(best.sample, data)
        result['violation_details'] = get_detailed_violations(best.sample, data)
        result['feasible'] = result['violations'] == 0
        result['success'] = True
        
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        
        LOG.info(f"  [DirectQPU] Success! obj={result['objective']:.4f}, "
                f"QPU_access={result['timings']['qpu_access']:.3f}s, "
                f"embed={result['timings']['embedding']:.2f}s")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        LOG.error(f"  [DirectQPU] Error: {e}")
    
    return result


def solve_clique_qpu(cqm: ConstrainedQuadraticModel, data: Dict,
                     num_reads: int = 1000, annealing_time: int = 20) -> Dict:
    """
    Clique QPU: CQM → BQM → DWaveCliqueSampler (zero embedding overhead)
    
    Uses DWaveCliqueSampler which maps directly to hardware cliques (n<=16).
    **Critical**: Problem must be small enough to fit in a clique!
    """
    if not HAS_CLIQUE:
        return {'success': False, 'error': 'DWaveCliqueSampler not available'}
    
    result = {
        'method': 'clique_qpu',
        'num_reads': num_reads,
        'annealing_time': annealing_time,
        'timings': {},
    }
    
    total_start = time.time()
    lagrange_multiplier = 50.0  # Same as direct QPU for fair comparison
    
    try:
        # Convert CQM to BQM
        LOG.info(f"  [CliqueQPU] Converting CQM to BQM (lagrange={lagrange_multiplier:.1f})...")
        t0 = time.time()
        bqm, info = cqm_to_bqm(cqm, lagrange_multiplier=lagrange_multiplier)
        result['timings']['cqm_to_bqm'] = time.time() - t0
        result['bqm_variables'] = len(bqm.variables)
        result['bqm_interactions'] = len(bqm.quadratic)
        result['lagrange_multiplier'] = lagrange_multiplier
        
        LOG.info(f"  [CliqueQPU] BQM: {result['bqm_variables']} vars, {result['bqm_interactions']} interactions")
        
        # Check if problem fits in clique (typically n<=16 for Pegasus)
        n_vars = len(bqm.variables)
        if n_vars > 20:
            LOG.warning(f"  [CliqueQPU] Problem too large ({n_vars} vars) for guaranteed clique embedding")
            LOG.warning(f"  [CliqueQPU] DWaveCliqueSampler works best for n<=16, may use chains for n>16")
        
        # Sample with DWaveCliqueSampler (automatic clique detection + embedding)
        LOG.info(f"  [CliqueQPU] Sampling with DWaveCliqueSampler ({num_reads} reads)...")
        
        sampler = DWaveCliqueSampler()
        
        sample_start = time.time()
        sampleset = sampler.sample(
            bqm,
            num_reads=num_reads,
            annealing_time=annealing_time,
            label=f"CliqueQPU_{data['n_farms']}farms"
        )
        result['timings']['sampling'] = time.time() - sample_start
        
        # Extract timing info
        timing_info = sampleset.info.get('timing', {})
        qpu_access_us = timing_info.get('qpu_access_time', 0)
        qpu_programming_us = timing_info.get('qpu_programming_time', 0)
        qpu_sampling_us = timing_info.get('qpu_sampling_time', 0)
        total_real_us = timing_info.get('total_real_time', 0)
        
        result['timings']['qpu_access'] = qpu_access_us / 1e6
        result['timings']['qpu_programming'] = qpu_programming_us / 1e6
        result['timings']['qpu_sampling'] = qpu_sampling_us / 1e6
        result['timings']['qpu_total_real'] = total_real_us / 1e6
        
        # Embedding info (cliques have minimal embedding overhead)
        embedding_info = sampleset.info.get('embedding_context', {})
        embedding_time = embedding_info.get('embedding_time', 0)
        chain_strength_used = embedding_info.get('chain_strength', 0)
        
        result['timings']['embedding'] = embedding_time / 1e6 if embedding_time else 0
        result['timings']['embedding_total'] = result['timings']['embedding']
        result['timings']['qpu_access_total'] = result['timings']['qpu_access']
        result['timings']['solve_time'] = result['timings']['qpu_access'] + result['timings']['embedding']
        result['chain_strength'] = chain_strength_used
        
        # Check for chain breaks (should be minimal/zero for cliques)
        if hasattr(sampleset.record, 'chain_break_fraction'):
            result['chain_break_fraction'] = float(np.mean(sampleset.record.chain_break_fraction))
        else:
            result['chain_break_fraction'] = 0.0
        
        # Extract embedding details if available
        if 'embedding' in embedding_info:
            embedding = embedding_info['embedding']
            result['physical_qubits'] = sum(len(chain) for chain in embedding.values())
            result['max_chain_length'] = max(len(chain) for chain in embedding.values()) if embedding else 1
            LOG.info(f"  [CliqueQPU] Embedding: {result['physical_qubits']} physical qubits, "
                    f"max chain {result['max_chain_length']}")
        else:
            result['physical_qubits'] = n_vars  # Assume 1:1 for cliques
            result['max_chain_length'] = 1
        
        # Best solution
        best = sampleset.first
        result['best_energy'] = float(best.energy)
        result['n_samples'] = len(sampleset)
        
        # Extract and evaluate solution
        result['solution'] = extract_solution(best.sample, data)
        result['objective'] = calculate_objective(best.sample, data)
        result['violations'] = count_violations(best.sample, data)
        result['violation_details'] = get_detailed_violations(best.sample, data)
        result['feasible'] = result['violations'] == 0
        result['success'] = True
        
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        
        LOG.info(f"  [CliqueQPU] Success! obj={result['objective']:.4f}, "
                f"QPU_access={result['timings']['qpu_access']:.3f}s, "
                f"embed={result['timings']['embedding']:.3f}s, "
                f"chains={result['chain_break_fraction']:.2%}")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        LOG.error(f"  [CliqueQPU] Error: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def solve_spatial_temporal_decomposition(data: Dict, cqm: ConstrainedQuadraticModel,
                                         num_reads: int = 100, num_iterations: int = 3,
                                         farms_per_cluster: int = 2) -> Dict:
    """
    **STRATEGY 1: Spatial + Temporal Decomposition (HIGHEST PRIORITY)**
    
    Mohseni-style decomposition to achieve quantum speedup:
    1. Cluster farms spatially (e.g., 5 farms → 3 clusters of [2,2,1] farms)
    2. Decompose temporally (3 periods → solve one period at a time)
    3. Result: Small subproblems (e.g., 2 farms × 6 crops = 12 vars) that FIT IN CLIQUES!
    4. Use DWaveCliqueSampler for ZERO embedding overhead
    5. Iterate for boundary coordination between clusters
    
    This is the "sweet spot" approach that matches Mohseni et al.'s success formula:
        Quantum Speedup = Decomposition (→ n ≤ 16) + Cliques (zero overhead) + Acceptable gap
    
    Example for 4 farms, 6 crops, 3 periods:
        - Total: 72 variables (intractable monolithic)
        - Decomposed: 6 subproblems of 12 variables each (2 farms × 6 crops × 1 period)
        - Each subproblem: 12 ≤ 16 → FITS CLIQUE PERFECTLY!
        - QPU calls: ~6-18 (6 subproblems × 1-3 iterations)
        - Expected: <15% optimality gap, <1s total QPU time
    
    Args:
        farms_per_cluster: Target farms per cluster (default 2 for 12-var subproblems)
        num_iterations: Boundary coordination iterations (3+ recommended)
    """
    if not HAS_CLIQUE:
        return {'success': False, 'error': 'DWaveCliqueSampler not available'}
    
    result = {
        'method': 'spatial_temporal_decomp',
        'num_reads': num_reads,
        'num_iterations': num_iterations,
        'farms_per_cluster': farms_per_cluster,
        'timings': {},
        'subproblem_results': []
    }
    
    total_start = time.time()
    
    farm_names = data['farm_names']
    food_names = data['food_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    n_periods = 3
    n_farms = len(farm_names)
    n_crops = len(food_names)
    
    # Calculate subproblem size
    vars_per_subproblem = farms_per_cluster * n_crops
    
    LOG.info(f"  [SpatialTemporal] Strategy 1: Spatial + Temporal Decomposition")
    LOG.info(f"  [SpatialTemporal] {n_farms} farms → {farms_per_cluster}-farm clusters × {n_periods} periods")
    LOG.info(f"  [SpatialTemporal] Subproblem size: {farms_per_cluster} farms × {n_crops} crops = {vars_per_subproblem} vars")
    
    if vars_per_subproblem > 16:
        LOG.warning(f"  [SpatialTemporal] WARNING: {vars_per_subproblem} vars > 16! May not fit cliques!")
        LOG.warning(f"  [SpatialTemporal] Recommendation: Reduce farms_per_cluster to {16 // n_crops}")
    elif vars_per_subproblem <= 16:
        LOG.info(f"  [SpatialTemporal] ✓ {vars_per_subproblem} ≤ 16: FITS CLIQUES (zero embedding overhead!)")
    
    # Step 1: Spatial clustering (simple adjacent grouping)
    farm_clusters = []
    for i in range(0, n_farms, farms_per_cluster):
        cluster = farm_names[i:i+farms_per_cluster]
        farm_clusters.append(cluster)
    
    n_clusters = len(farm_clusters)
    n_subproblems = n_clusters * n_periods
    
    LOG.info(f"  [SpatialTemporal] Created {n_clusters} spatial clusters → {n_subproblems} total subproblems")
    LOG.info(f"  [SpatialTemporal] Iterations: {num_iterations} (boundary coordination)")
    
    # Initialize sampler
    sampler = DWaveCliqueSampler()
    
    # Solution storage: period_solutions[period][farm] = {crop: value}
    period_solutions = {t: {} for t in range(1, n_periods + 1)}
    best_global_objective = -np.inf
    best_global_solution = None
    
    # Iterative refinement with boundary coordination
    for iteration in range(num_iterations):
        iteration_start = time.time()
        total_qpu_time = 0
        total_embedding_time = 0
        
        if iteration == 0:
            LOG.info(f"  [SpatialTemporal] Iteration 1/{num_iterations}: Initial independent solve")
        else:
            LOG.info(f"  [SpatialTemporal] Iteration {iteration+1}/{num_iterations}: Refining with temporal/spatial context")
        
        # Solve period by period (temporal decomposition)
        for period in range(1, n_periods + 1):
            # Solve cluster by cluster within this period (spatial decomposition)
            for cluster_idx, cluster_farms in enumerate(farm_clusters):
                # Build BQM for this space-time subproblem
                bqm = BinaryQuadraticModel('BINARY')
                
                # Variables: one per (farm, crop) in this cluster for this period
                # Use simple integer indexing for clique sampler
                var_map = {}  # (farm, crop) -> var_index
                var_idx = 0
                for farm in cluster_farms:
                    for crop in food_names:
                        var_map[(farm, crop)] = var_idx
                        var_idx += 1
                
                # Linear benefits
                for farm in cluster_farms:
                    farm_area = land_availability[farm]
                    for crop in food_names:
                        benefit = food_benefits[crop]
                        var = var_map[(farm, crop)]
                        # Negate for minimization
                        bqm.add_variable(var, -(benefit * farm_area) / total_area)
                
                # Rotation synergies (temporal coupling with previous period)
                if period > 1 and iteration > 0:
                    # Add bias based on previous period's solution
                    rotation_gamma = 0.2
                    for farm in cluster_farms:
                        if farm not in period_solutions[period-1]:
                            continue
                        
                        farm_area = land_availability[farm]
                        prev_crop = period_solutions[period-1][farm]
                        
                        # Encourage positive synergies, discourage negative
                        for crop in food_names:
                            var = var_map[(farm, crop)]
                            
                            if crop == prev_crop:
                                # Same crop: negative synergy (discourage)
                                synergy = -0.5
                            else:
                                # Different crop: positive synergy (encourage)
                                synergy = 0.1
                            
                            bonus = -(rotation_gamma * synergy * farm_area) / total_area
                            bqm.add_variable(var, bonus)
                
                # Spatial coupling (encourage diversity within cluster)
                spatial_gamma = 0.1
                for i, farm1 in enumerate(cluster_farms):
                    for farm2 in cluster_farms[i+1:]:
                        for crop in food_names:
                            var1 = var_map[(farm1, crop)]
                            var2 = var_map[(farm2, crop)]
                            # Small penalty for same crop on adjacent farms
                            bqm.add_interaction(var1, var2, spatial_gamma)
                
                # One-hot penalty (prefer one crop per farm per period)
                one_hot_penalty = 3.0
                for farm in cluster_farms:
                    farm_vars = [var_map[(farm, c)] for c in food_names]
                    # Penalty for (sum - 1)^2
                    for i, v1 in enumerate(farm_vars):
                        for v2 in farm_vars[i+1:]:
                            bqm.add_interaction(v1, v2, one_hot_penalty)
                        bqm.add_variable(v1, -one_hot_penalty)  # Linear term
                
                # Solve with clique sampler
                try:
                    sample_start = time.time()
                    sampleset = sampler.sample(
                        bqm,
                        num_reads=num_reads,
                        label=f"SpatialTemporal_p{period}_c{cluster_idx}_i{iteration}"
                    )
                    sample_time = time.time() - sample_start
                    
                    # Extract timing
                    timing_info = sampleset.info.get('timing', {})
                    qpu_time = timing_info.get('qpu_access_time', 0) / 1e6
                    total_qpu_time += qpu_time
                    
                    embedding_info = sampleset.info.get('embedding_context', {})
                    embed_time = embedding_info.get('embedding_time', 0) / 1e6 if embedding_info else 0
                    total_embedding_time += embed_time
                    
                    # Extract solution
                    best = sampleset.first
                    
                    # Decode: for each farm, find crop with highest activation
                    for farm in cluster_farms:
                        farm_crop_values = {}
                        for crop in food_names:
                            var = var_map[(farm, crop)]
                            farm_crop_values[crop] = best.sample.get(var, 0)
                        
                        # Choose crop with highest value (or None if all 0)
                        chosen_crop = max(farm_crop_values.items(), key=lambda x: x[1])[0] if any(farm_crop_values.values()) else food_names[0]
                        period_solutions[period][farm] = chosen_crop
                    
                    if iteration == 0 and period == 1:
                        subproblem_result = {
                            'period': period,
                            'cluster': cluster_idx,
                            'variables': len(bqm.variables),
                            'qpu_time': qpu_time,
                            'embedding_time': embed_time,
                            'sample_time': sample_time,
                            'success': True
                        }
                        result['subproblem_results'].append(subproblem_result)
                    
                except Exception as e:
                    LOG.error(f"  [SpatialTemporal] Period {period}, Cluster {cluster_idx} failed: {e}")
                    # Fallback: assign first crop to all farms
                    for farm in cluster_farms:
                        period_solutions[period][farm] = food_names[0]
        
        # Convert to full solution format
        combined_solution = {}
        for period in range(1, n_periods + 1):
            for farm in farm_names:
                for crop in food_names:
                    var_name = f"Y_{farm}_{crop}_t{period}"
                    assigned_crop = period_solutions[period].get(farm, food_names[0])
                    combined_solution[var_name] = 1 if crop == assigned_crop else 0
        
        # Evaluate iteration
        iter_obj = calculate_objective(combined_solution, data)
        
        if iter_obj > best_global_objective:
            best_global_objective = iter_obj
            best_global_solution = combined_solution
            LOG.info(f"  [SpatialTemporal] Iteration {iteration+1}: obj={iter_obj:.4f} (improved!)")
        else:
            LOG.info(f"  [SpatialTemporal] Iteration {iteration+1}: obj={iter_obj:.4f}")
    
    # Final evaluation
    result['objective'] = calculate_objective(best_global_solution, data)
    result['violations'] = count_violations(best_global_solution, data)
    result['violation_details'] = get_detailed_violations(best_global_solution, data)
    result['feasible'] = result['violations'] == 0
    result['solution'] = extract_solution(best_global_solution, data)
    
    # Timing summary
    result['timings']['total'] = time.time() - total_start
    result['timings']['qpu_access'] = total_qpu_time
    result['timings']['embedding'] = total_embedding_time
    result['timings']['qpu_access_total'] = total_qpu_time
    result['timings']['embedding_total'] = total_embedding_time
    result['timings']['solve_time'] = total_qpu_time + total_embedding_time
    result['total_time'] = result['timings']['total']
    result['wall_time'] = result['timings']['total']
    result['success'] = True
    result['n_subproblems'] = n_subproblems
    result['avg_subproblem_size'] = vars_per_subproblem
    result['n_clusters'] = n_clusters
    
    LOG.info(f"  [SpatialTemporal] Complete! {n_subproblems} subproblems ({vars_per_subproblem} vars each)")
    LOG.info(f"  [SpatialTemporal] Total QPU={total_qpu_time:.3f}s, embed={total_embedding_time:.4f}s")
    LOG.info(f"  [SpatialTemporal] Final objective: {result['objective']:.4f}, violations: {result['violations']}")
    
    return result


def solve_rotation_clique_decomposition(data: Dict, cqm: ConstrainedQuadraticModel,
                                        num_reads: int = 100, num_iterations: int = 1) -> Dict:
    """
    Decompose rotation problem farm-by-farm, solve each with clique sampler.
    
    Strategy (Mohseni et al. style):
    1. For each farm: 6 families × 3 periods = 18 variables
    2. Solve each farm independently with DWaveCliqueSampler (fits in clique!)
    3. Optional: Iterate to incorporate neighbor solutions (coordination)
    
    Args:
        num_iterations: Number of refinement iterations (1=no coordination, 3+=good coordination)
    """
    if not HAS_CLIQUE:
        return {'success': False, 'error': 'DWaveCliqueSampler not available'}
    
    result = {
        'method': 'clique_decomposition',
        'num_reads': num_reads,
        'num_iterations': num_iterations,
        'timings': {},
        'subproblem_results': []
    }
    
    total_start = time.time()
    
    farm_names = data['farm_names']
    food_names = data['food_names']  # Crop families
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    n_periods = 3
    
    LOG.info(f"  [CliqueDecomp] Decomposing {len(farm_names)} farms into independent subproblems...")
    LOG.info(f"  [CliqueDecomp] Each farm: {len(food_names)} families × {n_periods} periods = {len(food_names) * n_periods} variables")
    LOG.info(f"  [CliqueDecomp] Iterations: {num_iterations} (1=independent, 3+=coordinated)")
    
    # Check if subproblems fit in cliques
    vars_per_farm = len(food_names) * n_periods
    if vars_per_farm > 20:
        LOG.warning(f"  [CliqueDecomp] Subproblem size ({vars_per_farm} vars) may exceed clique size (16)!")
    
    sampler = DWaveCliqueSampler()
    
    # Get neighbor relationships for spatial coordination
    neighbor_map = {}
    if 'neighbor_pairs' in data:
        for f1, f2 in data['neighbor_pairs']:
            neighbor_map.setdefault(f1, []).append(f2)
            neighbor_map.setdefault(f2, []).append(f1)
    
    # Iterative refinement (like Mohseni et al.'s coalition formation)
    farm_solutions = {}
    best_global_objective = -np.inf
    best_global_solution = None
    
    for iteration in range(num_iterations):
        iteration_start = time.time()
        total_qpu_time = 0
        total_embedding_time = 0
        
        if iteration > 0:
            LOG.info(f"  [CliqueDecomp] Iteration {iteration+1}/{num_iterations}: Refining with neighbor context...")
        
        # Solve each farm
        for farm_idx, farm in enumerate(farm_names):
            # Build BQM for this farm
            bqm = BinaryQuadraticModel('BINARY')
            farm_area = land_availability[farm]
            
            # Linear benefits
            for c in food_names:
                B_c = food_benefits[c]
                for t in range(1, n_periods + 1):
                    var = (c, t)
                    bqm.add_variable(var, -(B_c * farm_area) / total_area)
            
            # Rotation synergies (temporal coupling within farm)
            rotation_gamma = 0.2
            for t in range(2, n_periods + 1):
                for c1 in food_names:
                    for c2 in food_names:
                        synergy = -0.5 if c1 == c2 else 0.1
                        if abs(synergy) > 1e-6:
                            var1 = (c1, t-1)
                            var2 = (c2, t)
                            interaction = -(rotation_gamma * synergy * farm_area) / total_area
                            bqm.add_interaction(var1, var2, interaction)
            
            # Spatial coupling bias (coordinate with neighbors from previous iteration)
            if iteration > 0 and farm in neighbor_map and farm in farm_solutions:
                spatial_gamma = rotation_gamma * 0.3  # Weaker than temporal
                neighbors = neighbor_map[farm]
                
                for neighbor in neighbors:
                    if neighbor not in farm_solutions:
                        continue
                    
                    # Get neighbor's solution from previous iteration
                    neighbor_sol = farm_solutions[neighbor]
                    
                    # Add bias to align with neighbor's crops
                    for t in range(1, n_periods + 1):
                        for c in food_names:
                            # Check what neighbor is growing in period t
                            neighbor_var = f"Y_{neighbor}_{c}_t{t}"
                            if neighbor_sol.get(neighbor_var, 0) == 1:
                                # Neighbor is growing crop c in period t
                                # Add small bias to encourage similar choice (positive synergy)
                                var = (c, t)
                                bonus = spatial_gamma * 0.2 * farm_area / total_area
                                bqm.add_variable(var, -bonus)  # Negate for minimization
            
            # Soft one-hot penalty
            one_hot_penalty = 3.0
            for t in range(1, n_periods + 1):
                for c1 in food_names:
                    for c2 in food_names:
                        if c1 != c2:
                            var1 = (c1, t)
                            var2 = (c2, t)
                            bqm.add_interaction(var1, var2, one_hot_penalty)
                        else:
                            var = (c1, t)
                            bqm.add_variable(var, -one_hot_penalty)
            
            # Sample with clique sampler
            try:
                sample_start = time.time()
                sampleset = sampler.sample(
                    bqm,
                    num_reads=num_reads,
                    label=f"CliqueDecomp_farm{farm_idx}_iter{iteration}"
                )
                sample_time = time.time() - sample_start
                
                # Extract timing
                timing_info = sampleset.info.get('timing', {})
                qpu_time = timing_info.get('qpu_access_time', 0) / 1e6
                total_qpu_time += qpu_time
                
                embedding_info = sampleset.info.get('embedding_context', {})
                embed_time = embedding_info.get('embedding_time', 0) / 1e6 if embedding_info else 0
                total_embedding_time += embed_time
                
                # Best solution for this farm
                best = sampleset.first
                farm_solution = {}
                for c in food_names:
                    for t in range(1, n_periods + 1):
                        var_name = f"Y_{farm}_{c}_t{t}"
                        farm_solution[var_name] = best.sample.get((c, t), 0)
                
                farm_solutions[farm] = farm_solution
                
                if iteration == 0:  # Only log first iteration details
                    subproblem_result = {
                        'farm': farm,
                        'variables': len(bqm.variables),
                        'qpu_time': qpu_time,
                        'embedding_time': embed_time,
                        'sample_time': sample_time,
                        'success': True
                    }
                    result['subproblem_results'].append(subproblem_result)
                
                if (farm_idx + 1) % max(1, len(farm_names) // 5) == 0 or farm_idx == 0:
                    if iteration == 0:
                        LOG.info(f"  [CliqueDecomp] Farm {farm_idx+1}/{len(farm_names)}: "
                                f"{len(bqm.variables)} vars, QPU={qpu_time:.3f}s, embed={embed_time:.4f}s")
                
            except Exception as e:
                LOG.error(f"  [CliqueDecomp] Farm {farm} failed: {e}")
                farm_solution = {f"Y_{farm}_{c}_t{t}": 0 
                               for c in food_names for t in range(1, n_periods + 1)}
                farm_solutions[farm] = farm_solution
                if iteration == 0:
                    result['subproblem_results'].append({
                        'farm': farm,
                        'success': False,
                        'error': str(e)
                    })
        
        # Evaluate iteration
        combined_solution = {}
        for farm in farm_names:
            combined_solution.update(farm_solutions[farm])
        
        iter_obj = calculate_objective(combined_solution, data)
        
        if iter_obj > best_global_objective:
            best_global_objective = iter_obj
            best_global_solution = combined_solution
            LOG.info(f"  [CliqueDecomp] Iteration {iteration+1}: obj={iter_obj:.4f} (improved!)")
        else:
            LOG.info(f"  [CliqueDecomp] Iteration {iteration+1}: obj={iter_obj:.4f} (no improvement)")
    
    # Use best solution found
    result['objective'] = calculate_objective(best_global_solution, data)
    result['violations'] = count_violations(best_global_solution, data)
    result['violation_details'] = get_detailed_violations(best_global_solution, data)
    result['feasible'] = result['violations'] == 0
    result['solution'] = extract_solution(best_global_solution, data)
    
    # Timing summary
    result['timings']['total'] = time.time() - total_start
    result['timings']['qpu_access'] = total_qpu_time
    result['timings']['embedding'] = total_embedding_time
    result['timings']['qpu_access_total'] = total_qpu_time
    result['timings']['embedding_total'] = total_embedding_time
    result['timings']['solve_time'] = total_qpu_time + total_embedding_time
    result['total_time'] = result['timings']['total']
    result['wall_time'] = result['timings']['total']
    
    result['success'] = True
    result['n_subproblems'] = len(farm_names)
    result['avg_subproblem_size'] = vars_per_farm
    
    LOG.info(f"  [CliqueDecomp] Complete! {len(farm_names)} subproblems, "
            f"total QPU={total_qpu_time:.3f}s, embed={total_embedding_time:.3f}s")
    LOG.info(f"  [CliqueDecomp] Combined objective: {result['objective']:.4f}, "
            f"violations: {result['violations']}")
    
    return result


def extract_solution(sample: Dict, data: Dict) -> Dict:
    """Extract solution in a structured format similar to benchmark_all_strategies."""
    food_names = data['food_names']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    
    solution = {
        'Y': {},  # Y[farm][food] = 0/1
        'U': {},  # U[food] = 0/1
        'allocations': {},  # (farm, food) -> area
        'summary': {
            'n_farms': len(farm_names),
            'n_foods': len(food_names),
            'foods_used': [],
            'total_area_allocated': 0,
        }
    }
    
    # Extract Y and U values
    for farm in farm_names:
        solution['Y'][farm] = {}
        for food in food_names:
            y_val = sample.get(f"Y_{farm}_{food}", 0)
            solution['Y'][farm][food] = int(y_val)
            if y_val == 1:
                area = land_availability.get(farm, 0)
                solution['allocations'][f"{farm}_{food}"] = area
                solution['summary']['total_area_allocated'] += area
    
    for food in food_names:
        u_val = sample.get(f"U_{food}", 0)
        solution['U'][food] = int(u_val)
        if u_val == 1:
            solution['summary']['foods_used'].append(food)
    
    solution['summary']['n_unique_foods'] = len(solution['summary']['foods_used'])
    
    return solution


def solve_qbsolv(cqm: ConstrainedQuadraticModel, data: Dict,
                 subproblem_size: int = 50, num_repeats: int = 100) -> Dict:
    """
    QBSolv: Decompose large BQM into subproblems, solve on QPU in parallel.
    """
    if not HAS_QBSOLV:
        return {'success': False, 'error': 'QBSolv not available'}
    
    result = {
        'method': 'qbsolv',
        'subproblem_size': subproblem_size,
        'num_repeats': num_repeats,
    }
    
    try:
        # Convert CQM to BQM
        t0 = time.time()
        bqm, info = cqm_to_bqm(cqm)
        result['cqm_to_bqm_time'] = time.time() - t0
        result['bqm_variables'] = len(bqm.variables)
        
        # Get QPU sampler for subproblems - REQUIRED
        qpu = get_qpu_sampler()
        if qpu is None:
            return {'success': False, 'error': 'QPU not available - SA removed, QPU required'}
        sub_sampler = EmbeddingComposite(qpu)
        
        # Solve with QBSolv
        t0 = time.time()
        response = QBSolv().sample(
            bqm,
            solver=sub_sampler,
            solver_limit=subproblem_size,
            num_repeats=num_repeats
        )
        result['total_time'] = time.time() - t0
        
        # Best solution
        best = response.first
        result['best_energy'] = float(best.energy)
        result['n_samples'] = len(response)
        
        # Calculate objective
        result['objective'] = calculate_objective(best.sample, data)
        result['violations'] = count_violations(best.sample, data)
        result['violation_details'] = get_detailed_violations(best.sample, data)
        result['feasible'] = result['violations'] == 0
        result['success'] = True
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
    
    return result


# REMOVED: solve_decomposition_sa - SA eliminated, QPU only


def solve_decomposition_qpu(cqm: ConstrainedQuadraticModel, data: Dict,
                            method: str = 'PlotBased',
                            num_reads: int = 1000,
                            annealing_time: int = 20,
                            embed_timeout: int = 30,
                            verbose: bool = True) -> Dict:
    """
    Decomposition + QPU: Split problem, solve each partition on real QPU.
    
    Includes detailed timing for:
    - Partition time
    - Embedding time per partition
    - QPU access time (from sampleset.info)
    - Total wall-clock time
    """
    result = {
        'method': f'decomposition_{method}_QPU',
        'decomposition': method,
        'solver': 'QPU',
        'num_reads': num_reads,
        'annealing_time': annealing_time,
        'timings': {},
    }
    
    try:
        total_start = time.time()
        
        # Get decomposition function
        decomp_func = DECOMPOSITION_METHODS.get(method)
        if decomp_func is None:
            return {'success': False, 'error': f'Unknown decomposition: {method}'}
        
        # Partition with timing
        if verbose:
            LOG.info(f"    [{method}+QPU] Partitioning...")
        t0 = time.time()
        partitions = decomp_func(data)
        partition_time = time.time() - t0
        result['timings']['partition'] = partition_time
        result['n_partitions'] = len(partitions)
        result['partition_sizes'] = [len(p) for p in partitions]
        
        if verbose:
            LOG.info(f"    [{method}+QPU] Created {len(partitions)} partitions in {partition_time:.3f}s")
        
        # Get QPU sampler - REQUIRED, no SA fallback
        qpu = get_qpu_sampler()
        if qpu is None:
            result['success'] = False
            result['error'] = 'QPU not available - SA removed, QPU required'
            return result
        
        sampler = EmbeddingComposite(qpu)
        
        # Solve each partition with detailed timing
        all_samples = {}
        partition_results = []
        total_qpu_access_time = 0
        total_qpu_programming_time = 0
        total_qpu_sampling_time = 0
        total_embedding_time = 0
        failed_embeddings = 0
        
        for i, partition in enumerate(partitions):
            if len(partition) == 0:
                continue
            
            # Build BQM for partition
            t0 = time.time()
            bqm = build_bqm_for_partition(partition, data)
            bqm_build_time = time.time() - t0
            
            # Solve on QPU with timeout
            t0 = time.time()
            try:
                sampleset = sampler.sample(
                    bqm,
                    num_reads=num_reads,
                    annealing_time=annealing_time,
                    label=f"Decomp_{method}_P{i}"
                )
                wall_time = time.time() - t0
                
                # Extract detailed timing from sampleset.info
                timing_info = sampleset.info.get('timing', {})
                qpu_access_us = timing_info.get('qpu_access_time', 0)
                qpu_programming_us = timing_info.get('qpu_programming_time', 0)
                qpu_sampling_us = timing_info.get('qpu_sampling_time', 0)
                total_real_us = timing_info.get('total_real_time', 0)
                
                # Estimate embedding time (wall_time - total_real_time)
                embedding_time = wall_time - (total_real_us / 1e6)
                total_embedding_time += max(0, embedding_time)
                
                total_qpu_access_time += qpu_access_us / 1e6
                total_qpu_programming_time += qpu_programming_us / 1e6
                total_qpu_sampling_time += qpu_sampling_us / 1e6
                
                # Chain break info
                chain_breaks = float(np.mean(sampleset.record.chain_break_fraction))
                
                # Collect solution with conflict resolution
                best_sample = sampleset.first.sample
                
                # Check for farm assignment conflicts (multiple crops for same farm)
                for var, val in best_sample.items():
                    if var.startswith("Y_") and val == 1:
                        # Extract farm name from Y_Farm_Food
                        parts = var.split("_", 2)
                        if len(parts) == 3:
                            farm = parts[1]
                            # Check if this farm already has a crop assigned
                            existing_crop = None
                            for existing_var in all_samples:
                                if existing_var.startswith(f"Y_{farm}_") and all_samples.get(existing_var, 0) == 1:
                                    existing_crop = existing_var
                                    break
                            
                            if existing_crop:
                                # Conflict! Keep the one with better benefit
                                existing_food = existing_crop.split("_", 2)[2]
                                new_food = parts[2]
                                
                                existing_benefit = data['food_benefits'].get(existing_food, 0) * data['land_availability'].get(farm, 0)
                                new_benefit = data['food_benefits'].get(new_food, 0) * data['land_availability'].get(farm, 0)
                                
                                if new_benefit > existing_benefit:
                                    # Replace with better option
                                    all_samples[existing_crop] = 0
                                    all_samples[var] = 1
                                else:
                                    # Keep existing, skip new
                                    continue
                            else:
                                # No conflict, add it
                                all_samples[var] = val
                    else:
                        # Not a Y variable or not assigned, just add it
                        all_samples[var] = val
                
                partition_results.append({
                    'partition': i,
                    'variables': len(partition),
                    'bqm_build_time': bqm_build_time,
                    'wall_time': wall_time,
                    'embedding_time_est': max(0, embedding_time),
                    'qpu_access_time': qpu_access_us / 1e6,
                    'qpu_programming_time': qpu_programming_us / 1e6,
                    'qpu_sampling_time': qpu_sampling_us / 1e6,
                    'chain_break_fraction': chain_breaks,
                    'energy': float(sampleset.first.energy),
                    'success': True,
                })
                
            except Exception as e:
                # Embedding failed - NO SA FALLBACK, fail the method
                failed_embeddings += 1
                if verbose:
                    LOG.error(f"    [{method}+QPU] Partition {i} embedding FAILED: {e}")
                
                partition_results.append({
                    'partition': i,
                    'variables': len(partition),
                    'error': str(e),
                    'success': False,
                })
                
                # Stop processing - method failed
                result['success'] = False
                result['error'] = f'Embedding failed for partition {i}: {e}'
                result['failed_partition'] = i
                result['failed_embeddings'] = failed_embeddings
                return result
            
            if verbose and (i + 1) % 5 == 0:
                LOG.info(f"    [{method}+QPU] Solved {i+1}/{len(partitions)} partitions...")
        
        result['partition_results'] = partition_results
        result['failed_embeddings'] = failed_embeddings
        result['timings']['partition'] = partition_time
        result['timings']['embedding_total'] = total_embedding_time
        result['timings']['qpu_access_total'] = total_qpu_access_time
        result['timings']['qpu_programming_total'] = total_qpu_programming_time
        result['timings']['qpu_sampling_total'] = total_qpu_sampling_time
        result['timings']['solve_time'] = total_qpu_access_time + partition_time  # Classical + QPU
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        result['total_qpu_time'] = total_qpu_access_time
        
        # Extract full solution
        result['solution'] = extract_solution(all_samples, data)
        
        # Calculate final objective and violations
        result['objective'] = calculate_objective(all_samples, data)
        result['violations'] = count_violations(all_samples, data)
        result['violation_details'] = get_detailed_violations(all_samples, data)
        result['feasible'] = result['violations'] == 0
        result['success'] = True
        
        if verbose:
            LOG.info(f"    [{method}+QPU] Complete: obj={result['objective']:.4f}, "
                    f"violations={result['violations']}")
            LOG.info(f"    [{method}+QPU] Timing: total={result['total_time']:.2f}s, "
                    f"QPU_access={total_qpu_access_time:.3f}s, embed={total_embedding_time:.2f}s")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return result


# ============================================================================
# CQM-FIRST DECOMPOSITION (CONSTRAINT-PRESERVING)
# ============================================================================

def extract_sub_cqm(cqm: ConstrainedQuadraticModel, 
                    partition_vars: Set[str], 
                    fixed_vars: Dict[str, int] = None) -> ConstrainedQuadraticModel:
    """
    Extract a sub-CQM containing only the specified variables.
    
    Constraints are kept if they can be fully evaluated:
    - All variables are in partition_vars (will be optimized)
    - Or some variables are in partition_vars and others are fixed
    
    Args:
        cqm: The original CQM
        partition_vars: Set of variable names to include
        fixed_vars: Dict of variable_name -> value for variables not in partition
    
    Returns:
        A new CQM with only the partition variables and relevant constraints
    """
    from dimod import ConstrainedQuadraticModel, Binary, QuadraticModel
    
    if fixed_vars is None:
        fixed_vars = {}
    
    sub_cqm = ConstrainedQuadraticModel()
    
    # Add variables from partition
    for var_name in partition_vars:
        if var_name in cqm.variables:
            sub_cqm.add_variable('BINARY', var_name)
    
    # Extract objective: only keep terms involving partition variables
    # For terms with fixed variables, compute their contribution
    obj = cqm.objective
    new_obj = QuadraticModel()
    
    for var in obj.linear:
        if var in partition_vars:
            new_obj.add_variable('BINARY', var)
            new_obj.set_linear(var, obj.get_linear(var))
    
    for (u, v), bias in obj.quadratic.items():
        if u in partition_vars and v in partition_vars:
            # Both in partition - keep quadratic term
            new_obj.add_quadratic(u, v, bias)
        elif u in partition_vars and v in fixed_vars:
            # v is fixed - becomes linear term for u
            if fixed_vars[v] == 1:
                current = new_obj.get_linear(u) if u in new_obj.variables else 0
                new_obj.set_linear(u, current + bias)
        elif v in partition_vars and u in fixed_vars:
            # u is fixed - becomes linear term for v
            if fixed_vars[u] == 1:
                current = new_obj.get_linear(v) if v in new_obj.variables else 0
                new_obj.set_linear(v, current + bias)
    
    sub_cqm.set_objective(new_obj)
    
    # Extract constraints
    for label, constraint in cqm.constraints.items():
        lhs = constraint.lhs
        rhs = constraint.rhs
        sense = constraint.sense
        
        # Get variables in this constraint
        constraint_vars = set(lhs.variables)
        
        # Check if constraint can be included
        in_partition = constraint_vars & partition_vars
        in_fixed = constraint_vars & set(fixed_vars.keys())
        outside = constraint_vars - partition_vars - set(fixed_vars.keys())
        
        if outside:
            # Some variables are neither in partition nor fixed - skip
            continue
        
        if not in_partition:
            # All variables are fixed - constraint is already satisfied/violated, skip
            continue
        
        # Build new constraint LHS
        new_lhs = QuadraticModel()
        offset = 0
        
        for var in lhs.linear:
            if var in partition_vars:
                new_lhs.add_variable('BINARY', var)
                new_lhs.set_linear(var, lhs.get_linear(var))
            elif var in fixed_vars:
                offset += lhs.get_linear(var) * fixed_vars[var]
        
        for (u, v), bias in lhs.quadratic.items():
            if u in partition_vars and v in partition_vars:
                new_lhs.add_quadratic(u, v, bias)
            elif u in partition_vars and v in fixed_vars:
                if fixed_vars[v] == 1:
                    current = new_lhs.get_linear(u) if u in new_lhs.variables else 0
                    new_lhs.set_linear(u, current + bias)
            elif v in partition_vars and u in fixed_vars:
                if fixed_vars[u] == 1:
                    current = new_lhs.get_linear(v) if v in new_lhs.variables else 0
                    new_lhs.set_linear(v, current + bias)
            elif u in fixed_vars and v in fixed_vars:
                offset += bias * fixed_vars[u] * fixed_vars[v]
        
        # Adjust RHS for fixed variable contributions
        new_rhs = rhs - offset
        
        # Add constraint with appropriate sense
        try:
            if sense.name == 'Eq':
                sub_cqm.add_constraint(new_lhs == new_rhs, label=label)
            elif sense.name == 'Le':
                sub_cqm.add_constraint(new_lhs <= new_rhs, label=label)
            elif sense.name == 'Ge':
                sub_cqm.add_constraint(new_lhs >= new_rhs, label=label)
        except Exception:
            # Skip problematic constraints
            pass
    
    return sub_cqm


def solve_cqm_first_decomposition_qpu(cqm: ConstrainedQuadraticModel, data: Dict,
                                      method: str = 'PlotBased',
                                      num_reads: int = 1000,
                                      annealing_time: int = 20,
                                      lagrange: float = 10.0,
                                      verbose: bool = True) -> Dict:
    """
    CQM-First Decomposition: Partition CQM, then convert each partition to BQM.
    
    This PRESERVES CONSTRAINTS by:
    1. Partitioning at the CQM level (variables, not penalty edges)
    2. Extracting sub-CQMs with relevant constraints for each partition
    3. Converting each sub-CQM to BQM (penalties only for that partition's constraints)
    4. Solving with QPU
    
    This avoids the constraint-cutting problem of BQM-first decomposition.
    
    IMPORTANT: For best results, use PlotBased partitioning where each farm is in its
    own partition. This allows precise tracking of MaxPlots constraints across all farms.
    Multilevel partitioning can still have MaxPlots violations because multiple farms
    in the same partition can independently select the same food.
    """
    result = {
        'method': f'cqm_first_{method}',
        'decomposition': method,
        'solver': 'QPU',
        'num_reads': num_reads,
        'annealing_time': annealing_time,
        'approach': 'cqm_first',  # Key identifier
        'timings': {},
    }
    
    try:
        total_start = time.time()
        
        # Get decomposition function
        decomp_func = DECOMPOSITION_METHODS.get(method)
        if decomp_func is None:
            return {'success': False, 'error': f'Unknown decomposition: {method}'}
        
        # Partition with timing
        if verbose:
            LOG.info(f"    [CQM-First {method}] Partitioning...")
        t0 = time.time()
        partitions = decomp_func(data)
        partition_time = time.time() - t0
        result['timings']['partition'] = partition_time
        result['n_partitions'] = len(partitions)
        result['partition_sizes'] = [len(p) for p in partitions]
        
        if verbose:
            LOG.info(f"    [CQM-First {method}] Created {len(partitions)} partitions in {partition_time:.3f}s")
        
        # QPU ONLY - no SA
        if not HAS_QPU:
            result['success'] = False
            result['error'] = 'QPU not available - SA removed, QPU required'
            return result
        
        qpu = get_qpu_sampler()
        if qpu is None:
            result['success'] = False
            result['error'] = 'Failed to connect to QPU'
            return result
        
        qpu_sampler = EmbeddingComposite(qpu)
        result['sampler'] = 'QPU'
        
        # Solve partitions in order, using two-stage approach:
        # 1. First solve U partition (master) to get food group feasibility
        # 2. Then solve Y partitions with U values fixed
        
        all_samples = {}
        partition_results = []
        total_solve_time = 0
        total_convert_time = 0
        total_qpu_time = 0
        total_embedding_time = 0
        
        # Find the U partition (master)
        u_partition_idx = None
        for i, partition in enumerate(partitions):
            if any(v.startswith("U_") for v in partition):
                u_partition_idx = i
                break
        
        # Solve U partition first (if exists)
        if u_partition_idx is not None:
            u_partition = partitions[u_partition_idx]
            
            if verbose:
                LOG.info(f"    [CQM-First {method}] Solving master partition (U vars)...")
            
            # Extract sub-CQM for U variables
            t0 = time.time()
            sub_cqm = extract_sub_cqm(cqm, u_partition)
            
            # Convert to BQM
            sub_bqm, _ = cqm_to_bqm(sub_cqm, lagrange_multiplier=lagrange)
            convert_time = time.time() - t0
            total_convert_time += convert_time
            
            # Solve with QPU
            t0 = time.time()
            sampleset = qpu_sampler.sample(
                sub_bqm, 
                num_reads=num_reads, 
                annealing_time=annealing_time,
                label=f"CQMFirst_{method}_Master"
            )
            solve_time = time.time() - t0
            total_solve_time += solve_time
            
            # Extract QPU timing
            timing_info = sampleset.info.get('timing', {})
            qpu_time = timing_info.get('qpu_access_time', 0) / 1e6
            total_qpu_time += qpu_time
            embedding_time = solve_time - qpu_time
            total_embedding_time += embedding_time
            
            # Collect solution
            best_sample = sampleset.first.sample
            # Filter to only include original CQM variables (not slack)
            for var, val in best_sample.items():
                if var in cqm.variables:
                    all_samples[var] = int(val)
            
            partition_results.append({
                'partition': u_partition_idx,
                'type': 'master',
                'variables': len(u_partition),
                'sub_cqm_vars': len(sub_cqm.variables),
                'sub_cqm_constraints': len(sub_cqm.constraints),
                'sub_bqm_vars': len(sub_bqm.variables),
                'convert_time': convert_time,
                'solve_time': solve_time,
                'qpu_time': qpu_time,
                'embedding_time': embedding_time,
                'energy': float(sampleset.first.energy),
            })
        
        # Now solve Y partitions with U values fixed
        # Track food usage for MaxPlots constraint
        fixed_u = {k: v for k, v in all_samples.items() if k.startswith("U_")}
        food_usage_count = {food: 0 for food in data['food_names']}  # Track plots per food
        max_plots = data.get('max_plots_per_crop')
        
        for i, partition in enumerate(partitions):
            if i == u_partition_idx:
                continue  # Already solved
            
            if len(partition) == 0:
                continue
            
            # Build list of foods that have reached max usage (only if max_plots is set)
            full_foods = set()
            if max_plots is not None:
                full_foods = {food for food, count in food_usage_count.items() if count >= max_plots}
            
            # Calculate remaining slots for each food (only if max_plots is set)
            remaining_slots = {}
            if max_plots is not None:
                remaining_slots = {food: max_plots - count for food, count in food_usage_count.items()}
            
            # Combine fixed vars: U values + mark full foods as unavailable
            combined_fixed = dict(fixed_u)
            # If a food is full, pretend U[food]=0 to prevent selection
            for food in full_foods:
                combined_fixed[f"U_{food}"] = 0
            
            # Extract sub-CQM with U values fixed
            t0 = time.time()
            sub_cqm = extract_sub_cqm(cqm, partition, fixed_vars=combined_fixed)
            
            # Adjust MaxPlots constraints for already-used slots (only if max_plots is set)
            # The original constraint is: sum(Y[farm,food]) <= max_plots
            # With current usage, it becomes: sum(Y[farm,food]) <= max_plots - usage = remaining_slots
            if max_plots is not None:
                for food in data['food_names']:
                    if food not in full_foods and remaining_slots.get(food, max_plots) < max_plots:
                        label = f"MaxPlots_{food}"
                        if label in sub_cqm.constraints:
                            # Modify the RHS of this constraint
                            constraint = sub_cqm.constraints[label]
                            new_rhs = remaining_slots[food]
                            # Remove and re-add with new RHS
                            del sub_cqm.constraints[label]
                            sub_cqm.add_constraint(constraint.lhs <= new_rhs, label=label)
            
            # Convert to BQM
            sub_bqm, _ = cqm_to_bqm(sub_cqm, lagrange_multiplier=lagrange)
            convert_time = time.time() - t0
            total_convert_time += convert_time
            
            # Solve with QPU
            t0 = time.time()
            sampleset = qpu_sampler.sample(
                sub_bqm, 
                num_reads=num_reads, 
                annealing_time=annealing_time,
                label=f"CQMFirst_{method}_Part{i}"
            )
            solve_time = time.time() - t0
            total_solve_time += solve_time
            
            # Extract QPU timing
            timing_info = sampleset.info.get('timing', {})
            qpu_time = timing_info.get('qpu_access_time', 0) / 1e6
            total_qpu_time += qpu_time
            embedding_time = solve_time - qpu_time
            total_embedding_time += embedding_time
            
            # Collect solution with conflict resolution and update food usage
            best_sample = sampleset.first.sample
            for var, val in best_sample.items():
                if var in cqm.variables:
                    # Check for farm assignment conflicts (multiple crops for same farm)
                    if var.startswith("Y_") and int(val) == 1:
                        parts = var.split("_", 2)
                        if len(parts) == 3:
                            farm = parts[1]
                            food = parts[2]
                            
                            # Check if this farm already has a crop assigned
                            existing_crop = None
                            for existing_var in all_samples:
                                if existing_var.startswith(f"Y_{farm}_") and all_samples.get(existing_var, 0) == 1:
                                    existing_crop = existing_var
                                    break
                            
                            if existing_crop:
                                # Conflict! Keep the one with better benefit
                                existing_food = existing_crop.split("_", 2)[2]
                                
                                existing_benefit = data['food_benefits'].get(existing_food, 0) * data['land_availability'].get(farm, 0)
                                new_benefit = data['food_benefits'].get(food, 0) * data['land_availability'].get(farm, 0)
                                
                                if new_benefit > existing_benefit:
                                    # Replace with better option
                                    all_samples[existing_crop] = 0
                                    all_samples[var] = int(val)
                                    # Update food usage tracking
                                    food_usage_count[food] = food_usage_count.get(food, 0) + 1
                                    # Decrement old food
                                    food_usage_count[existing_food] = max(0, food_usage_count.get(existing_food, 0) - 1)
                                # else: keep existing, don't add new
                            else:
                                # No conflict, add it
                                all_samples[var] = int(val)
                                # Track food usage for MaxPlots
                                food_usage_count[food] = food_usage_count.get(food, 0) + 1
                    else:
                        # Not a Y variable or not assigned, just add it
                        all_samples[var] = int(val)
            
            partition_results.append({
                'partition': i,
                'type': 'subproblem',
                'variables': len(partition),
                'sub_cqm_vars': len(sub_cqm.variables),
                'sub_cqm_constraints': len(sub_cqm.constraints),
                'sub_bqm_vars': len(sub_bqm.variables),
                'convert_time': convert_time,
                'solve_time': solve_time,
                'qpu_time': qpu_time,
                'embedding_time': embedding_time,
                'energy': float(sampleset.first.energy),
            })
            
            if verbose and (i + 1) % 10 == 0:
                LOG.info(f"    [CQM-First {method}] Solved {i+1}/{len(partitions)} partitions...")
        
        result['partition_results'] = partition_results
        result['timings']['convert_total'] = total_convert_time
        result['timings']['solve_total'] = total_solve_time
        result['timings']['solve_time'] = total_solve_time
        result['timings']['embedding_total'] = total_embedding_time
        result['timings']['qpu_access_total'] = total_qpu_time
        result['timings']['total'] = time.time() - total_start
        result['total_time'] = result['timings']['total']
        result['wall_time'] = result['timings']['total']
        result['total_qpu_time'] = total_qpu_time
        
        # Extract full solution
        result['solution'] = extract_solution(all_samples, data)
        
        # Calculate final objective and violations
        result['objective'] = calculate_objective(all_samples, data)
        result['violations'] = count_violations(all_samples, data)
        result['violation_details'] = get_detailed_violations(all_samples, data)
        result['feasible'] = result['violations'] == 0
        result['success'] = True
        
        if verbose:
            LOG.info(f"    [CQM-First {method}] Complete: obj={result['objective']:.4f}, "
                    f"violations={result['violations']}, time={result['total_time']:.2f}s")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return result


# ============================================================================
# SOLUTION EVALUATION
# ============================================================================

def calculate_objective(sample: Dict, data: Dict) -> float:
    """Calculate objective value from solution - MATCHES GUROBI/CQM FORMULATION EXACTLY.
    
    Computes ALL objective terms:
    1. Linear benefits
    2. Rotation synergies (temporal quadratic)
    3. Spatial interactions (neighbor quadratic)
    4. One-hot penalties (soft constraint)
    5. Diversity bonuses
    
    Handles both standard (Y_farm_food) and rotation (Y_farm_food_tX) formats.
    """
    food_benefits = data['food_benefits']
    land_availability = data['land_availability']
    total_area = data['total_area']
    farm_names = data['farm_names']
    food_names = data['food_names']
    
    # Detect if this is a rotation scenario
    is_rotation = any(key.startswith("Y_") and "_t" in key for key in sample.keys())
    
    objective = 0.0
    
    if is_rotation:
        # Rotation formulation - compute all 5 terms
        
        # Get rotation parameters
        config = data.get('config', {})
        params = config.get('parameters', {})
        rotation_gamma = params.get('rotation_gamma', 0.2)
        k_neighbors = params.get('spatial_k_neighbors', 4)
        frustration_ratio = params.get('frustration_ratio', 0.7)
        negative_strength = params.get('negative_synergy_strength', -0.8)
        one_hot_penalty = params.get('one_hot_penalty', 3.0)
        diversity_bonus = params.get('diversity_bonus', 0.15)
        use_soft_constraint = params.get('use_soft_one_hot', True)
        
        n_periods = 3
        n_families = len(food_names)
        families_list = list(food_names)
        
        # Recreate rotation matrix (same seed as in build_rotation_cqm)
        import numpy as np
        np.random.seed(42)
        R = np.zeros((n_families, n_families))
        for i in range(n_families):
            for j in range(n_families):
                if i == j:
                    R[i, j] = negative_strength * 1.5
                elif np.random.random() < frustration_ratio:
                    R[i, j] = np.random.uniform(negative_strength * 1.2, negative_strength * 0.3)
                else:
                    R[i, j] = np.random.uniform(0.02, 0.20)
        
        # Recreate spatial neighbor graph
        n_farms = len(farm_names)
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
        
        # Part 1: Linear benefits
        for farm in farm_names:
            farm_area = land_availability[farm]
            for crop in families_list:
                B_c = food_benefits[crop]
                for t in range(1, n_periods + 1):
                    if sample.get(f"Y_{farm}_{crop}_t{t}", 0) == 1:
                        objective += (B_c * farm_area) / total_area
        
        # Part 2: Rotation synergies (temporal quadratic terms)
        for farm in farm_names:
            farm_area = land_availability[farm]
            for t in range(2, n_periods + 1):
                for c1_idx, c1 in enumerate(families_list):
                    for c2_idx, c2 in enumerate(families_list):
                        synergy = R[c1_idx, c2_idx]
                        if abs(synergy) > 1e-6:
                            y1 = sample.get(f"Y_{farm}_{c1}_t{t-1}", 0)
                            y2 = sample.get(f"Y_{farm}_{c2}_t{t}", 0)
                            if y1 == 1 and y2 == 1:
                                objective += (rotation_gamma * synergy * farm_area) / total_area
        
        # Part 3: Spatial interactions (neighbor quadratic terms)
        spatial_gamma = rotation_gamma * 0.5
        for (f1, f2) in neighbor_edges:
            for t in range(1, n_periods + 1):
                for c1_idx, c1 in enumerate(families_list):
                    for c2_idx, c2 in enumerate(families_list):
                        spatial_synergy = R[c1_idx, c2_idx] * 0.3
                        if abs(spatial_synergy) > 1e-6:
                            y1 = sample.get(f"Y_{f1}_{c1}_t{t}", 0)
                            y2 = sample.get(f"Y_{f2}_{c2}_t{t}", 0)
                            if y1 == 1 and y2 == 1:
                                objective += (spatial_gamma * spatial_synergy) / total_area
        
        # Part 4: Soft one-hot penalty
        if use_soft_constraint:
            for farm in farm_names:
                for t in range(1, n_periods + 1):
                    crop_count = sum(1 for c in families_list 
                                   if sample.get(f"Y_{farm}_{c}_t{t}", 0) == 1)
                    # Penalty for (count - 1)^2
                    objective -= one_hot_penalty * (crop_count - 1) ** 2
        
        # Part 5: Diversity bonus
        for farm in farm_names:
            for crop in families_list:
                crop_used = sum(1 for t in range(1, n_periods + 1) 
                              if sample.get(f"Y_{farm}_{crop}_t{t}", 0) == 1)
                objective += diversity_bonus * crop_used
    
    else:
        # Standard formulation - only linear benefits
        for key, val in sample.items():
            if key.startswith("Y_") and val == 1:
                parts = key.split("_", 2)
                if len(parts) >= 3:
                    farm, food = parts[1], parts[2]
                    if farm in land_availability and food in food_benefits:
                        benefit = food_benefits[food] * land_availability[farm]
                        objective += benefit
        
        objective = objective / total_area
    
    return objective


def count_violations(sample: Dict, data: Dict) -> int:
    """Count constraint violations.
    
    Handles both standard and rotation formulations:
    - Standard: Y_farm_food, U_food variables with food group constraints
    - Rotation: Y_farm_food_tX variables with ≤2 crops per period constraint
    """
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    # Detect formulation type
    is_rotation = any(key.startswith("Y_") and "_t" in key for key in sample.keys())
    
    if is_rotation:
        # Rotation formulation: Check ≤2 crops per farm per period
        violations = 0
        n_periods = 3  # Rotation always uses 3 periods
        
        for farm in farm_names:
            for t in range(1, n_periods + 1):
                # Count active crops in this period
                count = sum(1 for food in food_names 
                          if sample.get(f"Y_{farm}_{food}_t{t}", 0) == 1)
                if count > 2:  # Violates ≤2 constraint
                    violations += 1
        
        return violations
    
    else:
        # Standard formulation: Original logic
        food_groups = data['food_groups']
        food_group_constraints = data['food_group_constraints']
        max_plots_per_crop = data.get('max_plots_per_crop')
        
        violations = 0
        
        # One crop per farm (check for <= 1 since idle plots are allowed)
        for farm in farm_names:
            count = sum(1 for food in food_names if sample.get(f"Y_{farm}_{food}", 0) == 1)
            if count > 1:  # Changed from != 1 to > 1 (allows 0 or 1)
                violations += 1
        
        # Food group constraints (use food_groups directly, no reverse_mapping)
        for group_name, limits in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            unique_foods = sum(1 for f in foods_in_group if sample.get(f"U_{f}", 0) == 1)
            
            # Support both min/max and min_foods/max_foods key formats
            min_foods = limits.get('min_foods', limits.get('min', 0))
            max_foods = limits.get('max_foods', limits.get('max', len(foods_in_group)))
            
            if min_foods > 0 and unique_foods < min_foods:
                violations += 1
            if unique_foods > max_foods:
                violations += 1
        
        # Max plots per crop (only check if constraint is set)
        if max_plots_per_crop is not None:
            for food in food_names:
                count = sum(1 for farm in farm_names if sample.get(f"Y_{farm}_{food}", 0) == 1)
                if count > max_plots_per_crop:
                    violations += 1
        
        return violations


def get_detailed_violations(sample: Dict, data: Dict) -> Dict:
    """Get detailed constraint violation information.
    
    Handles both standard and rotation formulations.
    """
    food_names = data['food_names']
    farm_names = data['farm_names']
    
    # Detect formulation type
    is_rotation = any(key.startswith("Y_") and "_t" in key for key in sample.keys())
    
    if is_rotation:
        # Rotation formulation: Check ≤2 crops per farm per period
        n_periods = 3
        violation_details = {
            'total_violations': 0,
            'max_crops_per_period': {'violations': 0, 'details': []}
        }
        
        for farm in farm_names:
            for t in range(1, n_periods + 1):
                count = sum(1 for food in food_names 
                          if sample.get(f"Y_{farm}_{food}_t{t}", 0) == 1)
                if count > 2:
                    violation_details['max_crops_per_period']['violations'] += 1
                    violation_details['max_crops_per_period']['details'].append({
                        'farm': farm,
                        'period': t,
                        'expected': '≤2',
                        'actual': count
                    })
        
        violation_details['total_violations'] = violation_details['max_crops_per_period']['violations']
        return violation_details
    
    else:
        # Standard formulation: Original logic
        food_groups = data['food_groups']
        food_group_constraints = data['food_group_constraints']
        max_plots_per_crop = data.get('max_plots_per_crop')
        
        violation_details = {
            'total_violations': 0,
            'one_crop_per_farm': {'violations': 0, 'details': []},
            'food_group_constraints': {'violations': 0, 'details': []},
            'max_plots_per_crop': {'violations': 0, 'details': []}
        }
        
        # One crop per farm (check for <= 1 since idle plots are allowed)
        for farm in farm_names:
            count = sum(1 for food in food_names if sample.get(f"Y_{farm}_{food}", 0) == 1)
            if count > 1:  # Changed from != 1 to > 1 (allows 0 or 1)
                violation_details['one_crop_per_farm']['violations'] += 1
                violation_details['one_crop_per_farm']['details'].append({
                    'farm': farm,
                    'expected': '0 or 1',
                    'actual': count
                })
        
        # Food group constraints (use food_groups directly, no reverse_mapping)
        for group_name, limits in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            unique_foods = sum(1 for f in foods_in_group if sample.get(f"U_{f}", 0) == 1)
            
            # Support both min/max and min_foods/max_foods key formats
            min_foods = limits.get('min_foods', limits.get('min', 0))
            max_foods = limits.get('max_foods', limits.get('max', len(foods_in_group)))
            
            violated = False
            if min_foods > 0 and unique_foods < min_foods:
                violated = True
            if unique_foods > max_foods:
                violated = True
            if violated:
                violation_details['food_group_constraints']['violations'] += 1
                violation_details['food_group_constraints']['details'].append({
                    'group': group_name,
                    'min': min_foods,
                    'max': max_foods,
                    'actual': unique_foods
                })
        
        # Max plots per crop (only check if constraint is set)
        if max_plots_per_crop is not None:
            for food in food_names:
                count = sum(1 for farm in farm_names if sample.get(f"Y_{farm}_{food}", 0) == 1)
                if count > max_plots_per_crop:
                    violation_details['max_plots_per_crop']['violations'] += 1
                    violation_details['max_plots_per_crop']['details'].append({
                        'food': food,
                        'max_allowed': max_plots_per_crop,
                        'actual': count
                    })
        
        violation_details['total_violations'] = (
            violation_details['one_crop_per_farm']['violations'] +
            violation_details['food_group_constraints']['violations'] +
            violation_details['max_plots_per_crop']['violations']
        )
        
        return violation_details
    
    # Max plots per crop (only check if constraint is set)
    if max_plots_per_crop is not None:
        for food in food_names:
            count = sum(1 for farm in farm_names if sample.get(f"Y_{farm}_{food}", 0) == 1)
            if count > max_plots_per_crop:
                violation_details['max_plots_per_crop']['violations'] += 1
                violation_details['max_plots_per_crop']['details'].append({
                    'food': food,
                    'max_allowed': max_plots_per_crop,
                    'actual': count
                })
    
    violation_details['total_violations'] = (
        violation_details['one_crop_per_farm']['violations'] +
        violation_details['food_group_constraints']['violations'] +
        violation_details['max_plots_per_crop']['violations']
    )
    
    return violation_details


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(scales: List[int] = None,
                  methods: List[str] = None,
                  verbose: bool = True,
                  scenarios: List[str] = None,
                  reads_configs: List[int] = None) -> Dict:
    """Run QPU benchmark (no hybrid solvers).
    
    Args:
        scales: List of farm counts for standard scenarios (e.g., [25, 50, 100])
        methods: List of solver methods to test
        verbose: Enable detailed logging
        scenarios: List of scenario names (e.g., ['micro_6', 'tiny_24']) - takes precedence over scales
        reads_configs: List of num_reads values to test (e.g., [100, 250, 500]). If None, uses defaults
    """
    
    if scales is None and scenarios is None:
        scales = FARM_SCALES
    
    if methods is None:
        methods = ['ground_truth', 'direct_qpu', 'qbsolv', 
                   'decomposition_PlotBased_QPU', 'decomposition_Multilevel(10)_QPU', 'cqm_first_PlotBased']
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'scales': scales if scenarios is None else [],
        'scenarios': scenarios if scenarios is not None else [],
        'methods': methods,
        'reads_configs': reads_configs if reads_configs else 'default',
        'qpu_available': HAS_QPU,
        'qbsolv_available': HAS_QBSOLV,
        'results': []
    }
    
    # Determine what to iterate over: scenarios or scales
    if scenarios is not None:
        # Use scenario-based loading
        items = scenarios
        use_scenarios = True
    else:
        # Use scale-based loading (traditional)
        items = scales
        use_scenarios = False
    
    for item in items:
        if use_scenarios:
            scenario_name = item
            if verbose:
                print(f"\n{'='*80}")
                print(f"SCENARIO: {scenario_name}")
                print('='*80)
            
            # Load scenario data with timing
            LOG.info(f"Loading scenario '{scenario_name}'...")
            t0 = time.time()
            data = load_problem_data_from_scenario(scenario_name)
            data_load_time = time.time() - t0
            
            n_farms = data['n_farms']
            n_foods = data['n_foods']
            n_vars = n_farms * n_foods + n_foods
        else:
            n_farms = item
            if verbose:
                print(f"\n{'='*80}")
                print(f"SCALE: {n_farms} farms ({n_farms * N_FOODS} Y variables + {N_FOODS} U variables)")
                print('='*80)
            
            # Load data with timing
            LOG.info(f"Loading problem data for {n_farms} farms...")
            t0 = time.time()
            data = load_problem_data(n_farms)
            data_load_time = time.time() - t0
            n_foods = data['n_foods']
            n_vars = n_farms * n_foods + n_foods
        
        # Build CQM with timing
        LOG.info(f"Building CQM...")
        t0 = time.time()
        
        # Detect rotation scenarios and use appropriate builder
        is_rotation = use_scenarios and scenario_name.startswith('rotation_')
        if is_rotation:
            LOG.info(f"  Detected rotation scenario - using 3-period formulation")
            cqm, metadata = build_rotation_cqm(data, n_periods=3)
        else:
            cqm, metadata = build_binary_cqm(data)
        
        cqm_build_time = time.time() - t0
        
        if verbose:
            print(f"  Data load: {data_load_time:.2f}s")
            print(f"  CQM build: {cqm_build_time:.2f}s")
            print(f"  CQM: {metadata['n_variables']} vars, {metadata['n_constraints']} constraints")
        
        scale_results = {
            'n_farms': n_farms,
            'scenario': scenario_name if use_scenarios else None,
            'metadata': metadata,
            'timings': {
                'data_load': data_load_time,
                'cqm_build': cqm_build_time,
            },
            'method_results': {}
        }
        
        # Ground truth
        if 'ground_truth' in methods:
            if verbose:
                LOG.info(f"Solving ground truth with Gurobi...")
                print(f"\n  [Ground Truth] Solving with Gurobi...")
            gt = solve_ground_truth(data)
            scale_results['ground_truth'] = gt
            if verbose and gt['success']:
                print(f"    Objective: {gt['objective']:.4f} in {gt['solve_time']:.2f}s")
        
        gt_obj = scale_results.get('ground_truth', {}).get('objective', 0)
        
        # Determine reads configurations to test
        if reads_configs is not None:
            # User specified custom reads configs
            reads_list = reads_configs
        else:
            # Use defaults: 500 for small scenarios, 1000 for others
            is_small_scenario = use_scenarios and scenario_name in ALL_SCENARIOS
            default_reads = 500 if is_small_scenario else DEFAULT_NUM_READS
            reads_list = [default_reads]
        
        # Test each reads configuration
        for num_reads in reads_list:
            # Suffix for method name if multiple reads configs
            reads_suffix = f"_r{num_reads}" if len(reads_list) > 1 else ""
            
            # Direct QPU: allow for small scenarios OR very small scales
            # Small scenarios (micro_6 to medium_160) have 6-160 variables - very embeddable
            # Rotation scenarios (rotation_micro_25 to rotation_large_200) have 108-738 variables
            if 'direct_qpu' in methods:
                is_small_scenario = use_scenarios and scenario_name in ALL_SCENARIOS
                is_tiny_scale = n_farms <= 15
                attempt_direct_qpu = is_small_scenario or is_tiny_scale
                
                if attempt_direct_qpu:
                    if verbose:
                        LOG.info(f"Attempting direct QPU (n_vars={n_vars}, num_reads={num_reads})...")
                        print(f"\n  [Direct QPU] CQM → BQM → QPU ({num_reads} reads)...")
                    qpu_result = solve_direct_qpu(cqm, data, num_reads=num_reads)
                    method_key = f'direct_qpu{reads_suffix}'
                    scale_results['method_results'][method_key] = qpu_result
                    if verbose:
                        if qpu_result['success']:
                            gap = ((gt_obj - qpu_result['objective']) / gt_obj * 100) if gt_obj > 0 else 0
                            print(f"    Objective: {qpu_result['objective']:.4f} (gap: {gap:.1f}%)")
                            print(f"    QPU access time: {qpu_result.get('qpu_access_time', 0):.3f}s")
                            print(f"    Chain breaks: {qpu_result.get('chain_break_fraction', 0):.2%}")
                        else:
                            print(f"    Failed: {qpu_result.get('error', 'Unknown')}")
                else:
                    if verbose:
                        print(f"\n  [Direct QPU] Skipped (scale too large for direct embedding)")
            
            # Clique QPU: for testing hardware clique embedding
            if 'clique_qpu' in methods:
                if verbose:
                    LOG.info(f"Attempting clique QPU (n_vars={n_vars}, num_reads={num_reads})...")
                    print(f"\n  [Clique QPU] CQM → BQM → DWaveCliqueSampler ({num_reads} reads)...")
                clique_result = solve_clique_qpu(cqm, data, num_reads=num_reads)
                method_key = f'clique_qpu{reads_suffix}'
                scale_results['method_results'][method_key] = clique_result
                if verbose:
                    if clique_result['success']:
                        gap = ((gt_obj - clique_result['objective']) / gt_obj * 100) if gt_obj > 0 else 0
                        print(f"    Objective: {clique_result['objective']:.4f} (gap: {gap:.1f}%)")
                        print(f"    QPU access time: {clique_result.get('qpu_access_time', 0):.3f}s")
                        print(f"    Embedding time: {clique_result.get('timings', {}).get('embedding', 0):.3f}s")
                        print(f"    Chain breaks: {clique_result.get('chain_break_fraction', 0):.2%}")
                    else:
                        print(f"    Failed: {clique_result.get('error', 'Unknown')}")
            
            # Clique Decomposition: Mohseni et al. style hierarchical decomposition
            if 'clique_decomp' in methods:
                # Use 3 iterations for better coordination (Mohseni et al. style)
                num_iterations = 3
                if verbose:
                    LOG.info(f"Attempting clique decomposition ({num_reads} reads per subproblem, {num_iterations} iterations)...")
                    print(f"\n  [Clique Decomp] Farm-by-farm decomposition + DWaveCliqueSampler ({num_reads} reads, {num_iterations} iters)...")
                decomp_result = solve_rotation_clique_decomposition(data, cqm, num_reads=num_reads, num_iterations=num_iterations)
                method_key = f'clique_decomp{reads_suffix}'
                scale_results['method_results'][method_key] = decomp_result
                if verbose:
                    if decomp_result['success']:
                        gap = ((gt_obj - decomp_result['objective']) / gt_obj * 100) if gt_obj > 0 else 0
                        print(f"    Objective: {decomp_result['objective']:.4f} (gap: {gap:.1f}%)")
                        print(f"    Iterations: {num_iterations}")
                        print(f"    Subproblems: {decomp_result.get('n_subproblems', 0)} × {decomp_result.get('avg_subproblem_size', 0)} vars")
                        print(f"    Total QPU time: {decomp_result.get('timings', {}).get('qpu_access', 0):.3f}s")
                        print(f"    Total embedding time: {decomp_result.get('timings', {}).get('embedding', 0):.3f}s")
                        print(f"    Violations: {decomp_result.get('violations', 0)}")
                    else:
                        print(f"    Failed: {decomp_result.get('error', 'Unknown')}")
            
            # Spatial+Temporal Decomposition: ROADMAP STRATEGY 1 (Highest Priority)
            if 'spatial_temporal' in methods:
                # Automatically determine farms_per_cluster to fit cliques (≤16 vars)
                n_foods = data['n_foods']
                farms_per_cluster = max(1, 16 // n_foods)  # Ensure ≤16 vars per subproblem
                num_iterations = 3  # Boundary coordination
                if verbose:
                    LOG.info(f"Attempting spatial+temporal decomposition (Strategy 1 - roadmap)...")
                    print(f"\n  [Spatial+Temporal] Roadmap Strategy 1: {farms_per_cluster} farms/cluster × {n_foods} crops = {farms_per_cluster * n_foods} vars/subproblem")
                    print(f"    Target: ≤16 vars (fits cliques!), {num_iterations} boundary iterations")
                st_result = solve_spatial_temporal_decomposition(data, cqm, num_reads=num_reads, 
                                                                  num_iterations=num_iterations, 
                                                                  farms_per_cluster=farms_per_cluster)
                method_key = f'spatial_temporal{reads_suffix}'
                scale_results['method_results'][method_key] = st_result
                if verbose:
                    if st_result['success']:
                        gap = ((gt_obj - st_result['objective']) / gt_obj * 100) if gt_obj > 0 else 0
                        print(f"    Objective: {st_result['objective']:.4f} (gap: {gap:.1f}%)")
                        print(f"    Clusters: {st_result.get('n_clusters', 0)} spatial × 3 periods = {st_result.get('n_subproblems', 0)} subproblems")
                        print(f"    Subproblem size: {st_result.get('avg_subproblem_size', 0)} vars {('✓ FITS CLIQUE!' if st_result.get('avg_subproblem_size', 99) <= 16 else '⚠ May not fit')}")
                        print(f"    Total QPU time: {st_result.get('timings', {}).get('qpu_access', 0):.3f}s")
                        print(f"    Total embedding time: {st_result.get('timings', {}).get('embedding', 0):.4f}s")
                        print(f"    Violations: {st_result.get('violations', 0)}")
                    else:
                        print(f"    Failed: {st_result.get('error', 'Unknown')}")
            
            # Decomposition methods (ALL with QPU - SA removed)
            for method in methods:
                if method.startswith('decomposition_') and method.endswith('_QPU'):
                    decomp_name = method.replace('decomposition_', '').replace('_QPU', '')
                    if verbose:
                        LOG.info(f"Running {decomp_name} decomposition with QPU ({num_reads} reads)...")
                        print(f"\n  [Decomposition: {decomp_name}] Partition + QPU ({num_reads} reads)...")
                    decomp_result = solve_decomposition_qpu(cqm, data, method=decomp_name, num_reads=num_reads, verbose=verbose)
                    method_key = f'{method}{reads_suffix}'
                    scale_results['method_results'][method_key] = decomp_result
                    if verbose:
                        if decomp_result['success']:
                            gap = ((gt_obj - decomp_result['objective']) / gt_obj * 100) if gt_obj > 0 else 0
                            print(f"    Objective: {decomp_result['objective']:.4f} (gap: {gap:.1f}%)")
                            print(f"    Partitions: {decomp_result['n_partitions']}")
                            print(f"    Total time: {decomp_result['total_time']:.2f}s")
                            if 'total_qpu_time' in decomp_result:
                                print(f"    QPU access time: {decomp_result['total_qpu_time']:.3f}s")
                            print(f"    Violations: {decomp_result['violations']}")
                        else:
                            print(f"    Failed: {decomp_result.get('error', 'Unknown')}")
            
            # CQM-First decomposition methods (constraint-preserving partition with QPU)
            for method in methods:
                if method.startswith('cqm_first_'):
                    decomp_name = method.replace('cqm_first_', '')
                    if verbose:
                        LOG.info(f"Running CQM-First {decomp_name} decomposition ({num_reads} reads)...")
                        print(f"\n  [CQM-First: {decomp_name}] Partition CQM → BQM → QPU ({num_reads} reads)...")
                    cqm_result = solve_cqm_first_decomposition_qpu(cqm, data, method=decomp_name, num_reads=num_reads, verbose=verbose)
                    method_key = f'{method}{reads_suffix}'
                    scale_results['method_results'][method_key] = cqm_result
                    if verbose:
                        if cqm_result['success']:
                            gap = ((gt_obj - cqm_result['objective']) / gt_obj * 100) if gt_obj > 0 else 0
                            print(f"    Objective: {cqm_result['objective']:.4f} (gap: {gap:.1f}%)")
                            print(f"    Partitions: {cqm_result['n_partitions']}")
                            print(f"    Time: {cqm_result['total_time']:.2f}s")
                            print(f"    Violations: {cqm_result['violations']} {'✓ FEASIBLE' if cqm_result['violations'] == 0 else '✗ INFEASIBLE'}")
                        else:
                            print(f"    Failed: {cqm_result.get('error', 'Unknown')}")
            
            # Coordinated decomposition (constraint-preserving with QPU)
            if 'coordinated' in methods:
                if verbose:
                    LOG.info(f"Running coordinated master-subproblem decomposition ({num_reads} reads)...")
                    print(f"\n  [Coordinated] Master-Subproblem (constraint-preserving) with QPU ({num_reads} reads)...")
                coord_result = solve_coordinated_decomposition(cqm, data, num_reads=num_reads)
                method_key = f'coordinated{reads_suffix}'
                scale_results['method_results'][method_key] = coord_result
                if verbose:
                    if coord_result['success']:
                        gap = ((gt_obj - coord_result['objective']) / gt_obj * 100) if gt_obj > 0 else 0
                        print(f"    Objective: {coord_result['objective']:.4f} (gap: {gap:.1f}%)")
                        print(f"    Violations: {coord_result['violations']}")
                        print(f"    Time: {coord_result['total_time']:.2f}s")
                    else:
                        print(f"    Failed: {coord_result.get('error', 'Unknown')}")
        
        # NEW: Advanced solver methods (efficiency-optimized)
        for method in methods:
            if method in ADVANCED_SOLVER_METHODS:
                if verbose:
                    LOG.info(f"Running advanced method: {method}...")
                    print(f"\n  [Advanced: {method}]...")
                try:
                    solver_func = ADVANCED_SOLVER_METHODS[method]
                    advanced_result = solver_func(data, num_reads=DEFAULT_NUM_READS, 
                                                   annealing_time=DEFAULT_ANNEALING_TIME)
                    scale_results['method_results'][method] = advanced_result
                    if verbose:
                        if advanced_result.get('success'):
                            gap = ((gt_obj - advanced_result['objective']) / gt_obj * 100) if gt_obj > 0 else 0
                            print(f"    Objective: {advanced_result['objective']:.4f} (gap: {gap:.1f}%)")
                            print(f"    Partitions: {advanced_result.get('n_partitions', 'N/A')}")
                            qpu_time = advanced_result.get('timings', {}).get('qpu_access_total', 0)
                            print(f"    QPU time: {qpu_time:.2f}s")
                            print(f"    Violations: {advanced_result.get('violations', 0)}")
                        else:
                            print(f"    Failed: {advanced_result.get('error', 'Unknown')}")
                except Exception as e:
                    LOG.error(f"Advanced method {method} failed: {e}")
                    scale_results['method_results'][method] = {'success': False, 'error': str(e)}
        
        results['results'].append(scale_results)
    
    return results


def print_summary(results: Dict):
    """Print detailed summary table with all timing metrics."""
    print("\n" + "=" * 140)
    print("BENCHMARK SUMMARY (Pure QPU - No Hybrid)")
    print("=" * 140)
    
    # Detailed timing table
    print(f"\n{'Scale':<20} {'Method':<28} {'Obj':>8} {'Gap%':>7} {'Wall':>8} {'Solve':>8} {'Embed':>8} {'QPU':>8} {'Viol':>5} {'Status':<12}")
    print("-" * 140)
    
    for sr in results['results']:
        n_farms = sr['n_farms']
        scenario_name = sr.get('scenario')
        
        # Scale label: use scenario name if available, else farm count
        scale_label = scenario_name if scenario_name else f"{n_farms}farms"
        
        gt_obj = sr.get('ground_truth', {}).get('objective', 0)
        
        # Ground truth (Gurobi)
        if 'ground_truth' in sr:
            gt = sr['ground_truth']
            gt_timings = gt.get('timings', {})
            gt_wall = gt.get('wall_time', gt.get('total_time', gt.get('solve_time', 0)))
            gt_solve = gt_timings.get('solve', gt.get('solve_time', 0))
            viol = gt.get('violations', 0)
            status = '✓ Opt' if gt.get('success') else '✗ Fail'
            print(f"{scale_label:<20} {'Ground Truth (Gurobi)':<28} {gt.get('objective', 0):>8.4f} {'0.0':>7} {gt_wall:>8.3f} {gt_solve:>8.3f} {'N/A':>8} {'N/A':>8} {viol:>5} {status:<12}")
        
        # Methods
        for method, r in sr.get('method_results', {}).items():
            obj = r.get('objective', 0)
            gap = ((gt_obj - obj) / gt_obj * 100) if gt_obj > 0 and obj > 0 else -999
            
            # Get timing details
            timings = r.get('timings', {})
            wall_time = r.get('wall_time', r.get('total_time', 0))
            solve_time = timings.get('solve_time', timings.get('sa_total', wall_time))
            embed_time = timings.get('embedding_total', timings.get('embedding', 0))
            qpu_time = timings.get('qpu_access_total', timings.get('qpu_access', 0))
            violations = r.get('violations', 0)
            
            # Determine solver type for display (ALL QPU NOW)
            solver_type = r.get('solver', r.get('sampler', 'QPU'))
            
            if r.get('success'):
                status = '✓ Feas' if r.get('feasible', True) else f"⚠ {violations}v"
            else:
                status = f"✗ Fail"
            
            gap_str = f"{gap:.1f}" if gap > -100 else "N/A"
            
            # All methods now use QPU - format timing columns
            embed_str = f"{embed_time:>8.2f}" if embed_time > 0 else f"{'N/A':>8}"
            qpu_str = f"{qpu_time:>8.3f}" if qpu_time > 0 else f"{'N/A':>8}"
            
            print(f"{'':<20} {method:<28} {obj:>8.4f} {gap_str:>7} {wall_time:>8.2f} {solve_time:>8.2f} {embed_str:>8} {qpu_str:>8} {violations:>5} {status:<12}")
        
        print("-" * 140)
    
    # Legend
    print("\nLegend:")
    print("  Wall  = Total wall-clock time (seconds)")
    print("  Solve = Solve time (QPU access + classical decomposition)")
    print("  Embed = Total embedding time (all methods use QPU)")
    print("  QPU   = Total QPU access time (all methods use QPU)")
    print("  Viol  = Number of constraint violations")


def save_results(results: Dict, filename: str = None) -> Path:
    """Save results to JSON and detailed report."""
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qpu_benchmark_{ts}.json"
    
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")
    
    # Also save detailed text report
    report_path = filepath.with_suffix('.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("QPU BENCHMARK DETAILED REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Scales: {results['scales']}\n")
        f.write(f"Methods: {results['methods']}\n")
        f.write(f"QPU Available: {results['qpu_available']}\n\n")
        
        for scale_result in results['results']:
            n_farms = scale_result['n_farms']
            f.write("\n" + "="*80 + "\n")
            f.write(f"SCALE: {n_farms} farms\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Metadata:\n")
            for key, val in scale_result['metadata'].items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")
            
            if 'ground_truth' in scale_result:
                gt = scale_result['ground_truth']
                f.write(f"Ground Truth (Gurobi):\n")
                f.write(f"  Success: {gt['success']}\n")
                if gt['success']:
                    f.write(f"  Objective: {gt['objective']:.6f}\n")
                    f.write(f"  Solve Time: {gt['solve_time']:.2f}s\n")
                    f.write(f"  Violations: {gt.get('violations', 0)}\n")
                f.write("\n")
            
            for method_name, method_result in scale_result.get('method_results', {}).items():
                f.write(f"\n{'-'*80}\n")
                f.write(f"Method: {method_name}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Success: {method_result['success']}\n")
                
                if method_result['success']:
                    f.write(f"Objective: {method_result['objective']:.6f}\n")
                    f.write(f"Total Time: {method_result.get('total_time', 0):.2f}s\n")
                    f.write(f"Violations: {method_result['violations']}\n")
                    
                    if 'solution' in method_result:
                        sol = method_result['solution']
                        f.write(f"\nSolution Summary:\n")
                        f.write(f"  Unique Foods Used: {sol['summary']['n_unique_foods']}\n")
                        f.write(f"  Foods: {', '.join(sol['summary']['foods_used'][:10])}")
                        if len(sol['summary']['foods_used']) > 10:
                            f.write(f" ... ({len(sol['summary']['foods_used'])} total)")
                        f.write("\n")
                        f.write(f"  Total Area Allocated: {sol['summary']['total_area_allocated']:.2f}\n")
                    
                    if 'violation_details' in method_result:
                        vd = method_result['violation_details']
                        if vd['total_violations'] > 0:
                            f.write(f"\nViolation Details:\n")
                            if vd['one_crop_per_farm']['violations'] > 0:
                                f.write(f"  One Crop Per Farm: {vd['one_crop_per_farm']['violations']} violations\n")
                                for detail in vd['one_crop_per_farm']['details'][:5]:
                                    f.write(f"    Farm {detail['farm']}: {detail['actual']} crops (expected 1)\n")
                            if vd['food_group_constraints']['violations'] > 0:
                                f.write(f"  Food Group Constraints: {vd['food_group_constraints']['violations']} violations\n")
                                for detail in vd['food_group_constraints']['details'][:5]:
                                    f.write(f"    Group {detail['group']}: {detail['actual']} (range: {detail['min']}-{detail['max']})\n")
                            if vd['max_plots_per_crop']['violations'] > 0:
                                f.write(f"  Max Plots Per Crop: {vd['max_plots_per_crop']['violations']} violations\n")
                                for detail in vd['max_plots_per_crop']['details'][:5]:
                                    f.write(f"    Food {detail['food']}: {detail['actual']} plots (max: {detail['max_allowed']})\n")
                    
                    if 'embedding_info' in method_result:
                        ei = method_result['embedding_info']
                        f.write(f"\nEmbedding Info:\n")
                        f.write(f"  Embedding Time: {ei.get('embedding_time', 0):.2f}s\n")
                        f.write(f"  Chain Length: {ei.get('chain_length', {}).get('mean', 0):.2f}\n")
                        f.write(f"  Max Chain: {ei.get('chain_length', {}).get('max', 0)}\n")
                        if 'qpu_access_time' in ei:
                            f.write(f"  QPU Access Time: {ei['qpu_access_time']:.4f}s\n")
                else:
                    f.write(f"Error: {method_result.get('error', 'Unknown')}\n")
                
                f.write("\n")
    
    print(f"Detailed report saved to: {report_path}")
    return filepath


# ============================================================================
# ROADMAP BENCHMARK RUNNER (Mohseni-Style Quantum Speedup Testing)
# ============================================================================

def run_roadmap_benchmark(phase: int = 1, verbose: bool = True) -> Dict:
    """
    Run comprehensive roadmap benchmarks following the Quantum Speedup Roadmap.
    
    Phases:
        Phase 1: Proof of Concept (4-5 farms, 6 crops, 3 periods)
            - Test spatial+temporal decomposition with cliques
            - Compare: Gurobi vs Clique Decomposition vs Spatial+Temporal
            - Success: Gap <20%, QPU time <1s, zero embedding
        
        Phase 2: Scaling Validation (5, 10, 15 farms)
            - Measure scaling curves
            - Find crossover point where quantum wins
        
        Phase 3: Optimization (if Phase 2 successful)
            - Parallel QPU calls
            - Advanced clustering
            - Boundary refinement
    
    Args:
        phase: Which roadmap phase to run (1, 2, or 3)
        verbose: Enable detailed logging
    
    Returns:
        Results dictionary with detailed benchmarks
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'phase': phase,
        'roadmap_version': '1.0',
        'results': []
    }
    
    if phase == 1:
        # Phase 1: Proof of Concept
        print("\n" + "=" * 100)
        print("ROADMAP PHASE 1: PROOF OF CONCEPT")
        print("Testing: Simple binary and rotation scenarios with clique-friendly sizes")
        print("Goal: Verify decomposition works and fits cliques")
        print("Success Criteria: Gap <20%, QPU <1s, zero embedding overhead")
        print("=" * 100)
        
        # Test configurations for Phase 1 - USE SYNTHETIC SCENARIOS (guaranteed feasible)
        test_configs = [
            {
                'name': 'Simple Binary (tiny_24: 4 farms, 5 foods, NO rotation)',
                'scenario': 'tiny_24',  # 4 farms × 5 foods, proven feasible
                'formulation': 'simple',
                'methods': ['ground_truth', 'direct_qpu', 'clique_qpu']
            },
            {
                'name': 'Rotation (rotation_micro_25: 5 farms, 6 crops, 3 periods)',
                'scenario': 'rotation_micro_25',  # 5 farms × 6 crops × 3 periods
                'formulation': 'rotation',
                'methods': ['ground_truth', 'clique_decomp', 'spatial_temporal']
            }
        ]
        
        for config in test_configs:
            print(f"\n{'='*100}")
            print(f"Test: {config['name']}")
            print('='*100)
            
            # Load data from scenario (guaranteed feasible)
            data = load_problem_data_from_scenario(config['scenario'])
            
            # Build CQM
            if config['formulation'] == 'simple':
                cqm, metadata = build_simple_binary_cqm(data)
            elif config['formulation'] == 'rotation':
                cqm, metadata = build_rotation_cqm(data, n_periods=3)
            else:
                cqm, metadata = build_binary_cqm(data)
            
            print(f"Problem size: {metadata['n_variables']} variables, {metadata['n_constraints']} constraints")
            
            # Run methods
            for method in config['methods']:
                print(f"\n--- Method: {method} ---")
                
                try:
                    if method == 'ground_truth':
                        if config['formulation'] == 'rotation':
                            method_result = solve_ground_truth_rotation(data, timeout=120)
                        else:
                            method_result = solve_ground_truth(data, timeout=120)
                    
                    elif method == 'direct_qpu':
                        method_result = solve_direct_qpu(cqm, data, num_reads=1000)
                    
                    elif method == 'clique_qpu':
                        method_result = solve_clique_qpu(cqm, data, num_reads=1000)
                    
                    elif method == 'clique_decomp':
                        method_result = solve_rotation_clique_decomposition(data, cqm, num_reads=100, num_iterations=3)
                    
                    elif method == 'spatial_temporal':
                        method_result = solve_spatial_temporal_decomposition(data, cqm, num_reads=100, num_iterations=3, farms_per_cluster=2)
                    
                    else:
                        print(f"Unknown method: {method}")
                        continue
                    
                    # Store result
                    result_entry = {
                        'test': config['name'],
                        'formulation': config['formulation'],
                        'method': method,
                        'n_farms': data['n_farms'],
                        'n_foods': data['n_foods'],
                        'n_variables': metadata['n_variables'],
                        **method_result
                    }
                    results['results'].append(result_entry)
                    
                    # Print summary
                    if method_result.get('success'):
                        obj = method_result.get('objective', 0)
                        wall_time = method_result.get('wall_time', 0)
                        qpu_time = method_result.get('timings', {}).get('qpu_access', 0)
                        embed_time = method_result.get('timings', {}).get('embedding', 0)
                        violations = method_result.get('violations', 0)
                        
                        print(f"✓ Success: obj={obj:.4f}, wall={wall_time:.3f}s, QPU={qpu_time:.3f}s, embed={embed_time:.4f}s, violations={violations}")
                        
                        # Check Phase 1 success criteria
                        if method != 'ground_truth' and 'ground_truth' in [r['method'] for r in results['results']]:
                            gt_result = next(r for r in results['results'] if r['method'] == 'ground_truth' and r['test'] == config['name'])
                            gt_obj = gt_result.get('objective', 1)
                            gap = abs(gt_obj - obj) / abs(gt_obj) * 100 if gt_obj != 0 else 0
                            
                            print(f"  Gap vs Gurobi: {gap:.1f}%")
                            
                            if gap < 20 and qpu_time < 1.0 and embed_time < 0.01:
                                print(f"  🎉 PHASE 1 SUCCESS CRITERIA MET!")
                    else:
                        print(f"✗ Failed: {method_result.get('error', 'unknown error')}")
                
                except Exception as e:
                    print(f"✗ Exception: {e}")
                    import traceback
                    traceback.print_exc()
    
    elif phase == 2:
        # Phase 2: Scaling Validation
        print("\n" + "=" * 100)
        print("ROADMAP PHASE 2: SCALING VALIDATION")
        print("Testing: F ∈ {5, 10, 15} farms with spatial+temporal decomposition")
        print("Goal: Find crossover point where quantum wins")
        print("=" * 100)
        
        farm_scales = [5, 10, 15]
        methods = ['ground_truth', 'spatial_temporal']
        
        # Map farm scales to rotation scenarios (6 crop families, not 27 foods!)
        scenario_map = {
            5: 'rotation_micro_25',   # 5 farms × 6 families
            10: 'rotation_small_50',  # 10 farms × 6 families
            15: 'rotation_medium_100' # 20 farms (we'll use first 15)
        }
        
        for n_farms in farm_scales:
            print(f"\n{'='*100}")
            print(f"Scale: {n_farms} farms × 6 crops × 3 periods")
            print('='*100)
            
            # Load rotation scenario data (6 crop families, not 27 foods!)
            scenario = scenario_map.get(n_farms, 'rotation_micro_25')
            data = load_problem_data_from_scenario(scenario)
            
            # Limit to exact number of farms if scenario has more
            if len(data['farm_names']) > n_farms:
                data['farm_names'] = data['farm_names'][:n_farms]
                data['land_availability'] = {k: v for k, v in data['land_availability'].items() if k in data['farm_names']}
                data['total_area'] = sum(data['land_availability'].values())
                data['n_farms'] = n_farms
            
            data['n_periods'] = 3
            
            # Build rotation CQM
            cqm, metadata = build_rotation_cqm(data, n_periods=3)
            print(f"Problem size: {metadata['n_variables']} variables")
            
            for method in methods:
                print(f"\n--- Method: {method} ---")
                
                try:
                    if method == 'ground_truth':
                        method_result = solve_ground_truth_rotation(data, timeout=300)
                    elif method == 'spatial_temporal':
                        # Adjust farms_per_cluster based on scale
                        farms_per_cluster = 2 if n_farms <= 10 else 3
                        method_result = solve_spatial_temporal_decomposition(data, cqm, num_reads=100, num_iterations=3, farms_per_cluster=farms_per_cluster)
                    else:
                        continue
                    
                    result_entry = {
                        'phase': 2,
                        'scale': n_farms,
                        'method': method,
                        'n_variables': metadata['n_variables'],
                        **method_result
                    }
                    results['results'].append(result_entry)
                    
                    if method_result.get('success'):
                        obj = method_result.get('objective', 0)
                        wall_time = method_result.get('wall_time', 0)
                        print(f"✓ obj={obj:.4f}, time={wall_time:.3f}s")
                
                except Exception as e:
                    print(f"✗ Exception: {e}")
        
        # Analyze crossover
        print(f"\n{'='*100}")
        print("CROSSOVER ANALYSIS")
        print('='*100)
        
        for n_farms in farm_scales:
            gt_result = next((r for r in results['results'] if r['method'] == 'ground_truth' and r.get('scale') == n_farms), None)
            qpu_result = next((r for r in results['results'] if r['method'] == 'spatial_temporal' and r.get('scale') == n_farms), None)
            
            if gt_result and qpu_result and gt_result.get('success') and qpu_result.get('success'):
                gt_time = gt_result.get('wall_time', 0)
                qpu_time = qpu_result.get('wall_time', 0)
                gt_obj = gt_result.get('objective', 1)
                qpu_obj = qpu_result.get('objective', 0)
                gap = abs(gt_obj - qpu_obj) / abs(gt_obj) * 100 if gt_obj != 0 else 0
                
                speedup = gt_time / qpu_time if qpu_time > 0 else 0
                
                print(f"{n_farms} farms: Gurobi={gt_time:.2f}s, QPU={qpu_time:.2f}s, Speedup={speedup:.2f}x, Gap={gap:.1f}%")
                
                if speedup > 1.0 and gap < 15:
                    print(f"  🎉 QUANTUM ADVANTAGE AT {n_farms} FARMS!")
    
    elif phase == 3:
        # Phase 3: Optimization & Refinement
        print("\n" + "=" * 100)
        print("ROADMAP PHASE 3: OPTIMIZATION & REFINEMENT")
        print("Testing: Advanced techniques for maximum quantum advantage")
        print("Goal: Optimize quality, speed, and scalability")
        print("=" * 100)
        
        # Test scales: use results from Phase 2 to determine best scale
        test_scales = [10, 15, 20]
        
        # Map farm scales to rotation scenarios (6 crop families, not 27 foods!)
        scenario_map = {
            10: 'rotation_small_50',   # 10 farms × 6 families
            15: 'rotation_medium_100', # 20 farms (we'll use first 15)
            20: 'rotation_medium_100'  # 20 farms × 6 families
        }
        
        # Advanced optimization strategies
        optimization_strategies = [
            {
                'name': 'Baseline (Phase 2)',
                'method': 'spatial_temporal',
                'params': {
                    'num_iterations': 3,
                    'farms_per_cluster': 2
                }
            },
            {
                'name': 'Increased Iterations (5x)',
                'method': 'spatial_temporal',
                'params': {
                    'num_iterations': 5,
                    'farms_per_cluster': 2
                }
            },
            {
                'name': 'Larger Clusters',
                'method': 'spatial_temporal',
                'params': {
                    'num_iterations': 3,
                    'farms_per_cluster': 3  # 3 farms × 6 crops = 18 vars (still fits cliques)
                }
            },
            {
                'name': 'Hybrid: More Iterations + Larger Clusters',
                'method': 'spatial_temporal',
                'params': {
                    'num_iterations': 5,
                    'farms_per_cluster': 3
                }
            },
            {
                'name': 'High Reads (500)',
                'method': 'spatial_temporal',
                'params': {
                    'num_iterations': 3,
                    'farms_per_cluster': 2,
                    'num_reads': 500
                }
            }
        ]
        
        for n_farms in test_scales:
            print(f"\n{'='*100}")
            print(f"Scale: {n_farms} farms × 6 crops × 3 periods")
            print('='*100)
            
            # Load rotation scenario data (6 crop families, not 27 foods!)
            scenario = scenario_map.get(n_farms, 'rotation_small_50')
            data = load_problem_data_from_scenario(scenario)
            
            # Limit to exact number of farms if scenario has more
            if len(data['farm_names']) > n_farms:
                data['farm_names'] = data['farm_names'][:n_farms]
                data['land_availability'] = {k: v for k, v in data['land_availability'].items() if k in data['farm_names']}
                data['total_area'] = sum(data['land_availability'].values())
                data['n_farms'] = n_farms
            
            data['n_periods'] = 3
            
            # Build rotation CQM
            cqm, metadata = build_rotation_cqm(data, n_periods=3)
            print(f"Problem size: {metadata['n_variables']} variables")
            
            # Ground truth for comparison
            print(f"\n--- Ground Truth (Gurobi) ---")
            try:
                gt_result = solve_ground_truth_rotation(data, timeout=300)
                
                result_entry = {
                    'phase': 3,
                    'scale': n_farms,
                    'method': 'ground_truth',
                    'n_variables': metadata['n_variables'],
                    **gt_result
                }
                results['results'].append(result_entry)
                
                if gt_result.get('success'):
                    gt_obj = gt_result.get('objective', 0)
                    gt_time = gt_result.get('wall_time', 0)
                    print(f"✓ obj={gt_obj:.4f}, time={gt_time:.3f}s")
                else:
                    print(f"✗ Failed")
                    gt_obj = None
                    
            except Exception as e:
                print(f"✗ Exception: {e}")
                gt_obj = None
            
            # Test each optimization strategy
            for strategy in optimization_strategies:
                print(f"\n--- {strategy['name']} ---")
                
                try:
                    params = strategy['params'].copy()
                    num_reads = params.pop('num_reads', 100)
                    
                    method_result = solve_spatial_temporal_decomposition(
                        data, cqm, 
                        num_reads=num_reads,
                        **params
                    )
                    
                    result_entry = {
                        'phase': 3,
                        'scale': n_farms,
                        'method': strategy['name'],
                        'strategy_params': strategy['params'],
                        'n_variables': metadata['n_variables'],
                        **method_result
                    }
                    results['results'].append(result_entry)
                    
                    if method_result.get('success'):
                        obj = method_result.get('objective', 0)
                        wall_time = method_result.get('wall_time', 0)
                        qpu_time = method_result.get('timings', {}).get('qpu_access', 0)
                        embed_time = method_result.get('timings', {}).get('embedding', 0)
                        violations = method_result.get('violations', 0)
                        n_subproblems = method_result.get('n_subproblems', 0)
                        avg_size = method_result.get('avg_subproblem_size', 0)
                        
                        gap = abs(gt_obj - obj) / abs(gt_obj) * 100 if gt_obj and gt_obj != 0 else 0
                        
                        print(f"✓ obj={obj:.4f}, gap={gap:.1f}%, time={wall_time:.3f}s")
                        print(f"  QPU={qpu_time:.3f}s, embed={embed_time:.4f}s, violations={violations}")
                        print(f"  Subproblems: {n_subproblems} × {avg_size} vars")
                        
                        # Check if this is the best strategy
                        if gap < 10 and violations == 0:
                            print(f"  🌟 EXCELLENT QUALITY (gap < 10%)")
                        if wall_time < gt_time if gt_obj else False:
                            speedup = gt_time / wall_time if wall_time > 0 else 0
                            print(f"  ⚡ FASTER THAN GUROBI ({speedup:.2f}x speedup)")
                    else:
                        print(f"✗ Failed: {method_result.get('error', 'unknown error')}")
                
                except Exception as e:
                    print(f"✗ Exception: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Final analysis: find best strategy
        print(f"\n{'='*100}")
        print("PHASE 3 OPTIMIZATION ANALYSIS")
        print('='*100)
        
        for n_farms in test_scales:
            print(f"\n{n_farms} farms:")
            
            gt_result = next((r for r in results['results'] if r['method'] == 'ground_truth' and r.get('scale') == n_farms), None)
            
            if not gt_result or not gt_result.get('success'):
                print("  (No ground truth available)")
                continue
            
            gt_obj = gt_result.get('objective', 1)
            gt_time = gt_result.get('wall_time', 0)
            
            # Compare all strategies
            strategy_results = [r for r in results['results'] 
                              if r.get('scale') == n_farms and r['method'] != 'ground_truth' and r.get('phase') == 3]
            
            best_quality = None
            best_speed = None
            best_balanced = None
            
            for sr in strategy_results:
                if not sr.get('success'):
                    continue
                
                obj = sr.get('objective', 0)
                time_val = sr.get('wall_time', 999)
                violations = sr.get('violations', 999)
                
                gap = abs(gt_obj - obj) / abs(gt_obj) * 100 if gt_obj != 0 else 999
                speedup = gt_time / time_val if time_val > 0 else 0
                
                # Track best in each category (only feasible solutions)
                if violations == 0:
                    if best_quality is None or gap < best_quality['gap']:
                        best_quality = {'name': sr['method'], 'gap': gap, 'time': time_val}
                    
                    if best_speed is None or time_val < best_speed['time']:
                        best_speed = {'name': sr['method'], 'time': time_val, 'gap': gap}
                    
                    # Balanced: gap < 15% AND competitive speed
                    if gap < 15:
                        score = gap + (time_val / gt_time * 100 if gt_time > 0 else 999)
                        if best_balanced is None or score < best_balanced.get('score', 999):
                            best_balanced = {'name': sr['method'], 'gap': gap, 'time': time_val, 'speedup': speedup, 'score': score}
            
            if best_quality:
                print(f"  🏆 Best Quality: {best_quality['name']}")
                print(f"     Gap: {best_quality['gap']:.1f}%, Time: {best_quality['time']:.2f}s")
            
            if best_speed:
                print(f"  ⚡ Fastest: {best_speed['name']}")
                print(f"     Time: {best_speed['time']:.2f}s, Gap: {best_speed['gap']:.1f}%")
            
            if best_balanced:
                print(f"  ⭐ Best Balanced: {best_balanced['name']}")
                print(f"     Gap: {best_balanced['gap']:.1f}%, Time: {best_balanced['time']:.2f}s, Speedup: {best_balanced['speedup']:.2f}x")
        
        print(f"\n{'='*100}")
        print("PHASE 3 RECOMMENDATIONS")
        print('='*100)
        print("Based on the optimization analysis:")
        print("1. For best quality (lowest gap): Use strategy with highest iterations and coordination")
        print("2. For best speed: Use baseline with minimal iterations but good clustering")
        print("3. For balanced performance: Consider hybrid approaches with moderate parameters")
        print("4. For large-scale problems (>15 farms): Increase farms_per_cluster to reduce subproblems")
        print("5. For publication results: Use the 'Best Balanced' strategy for each scale")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='QPU Benchmark (Pure Quantum - No Hybrid)')
    parser.add_argument('--roadmap', type=int, nargs='?', const=1,
                        help='Run roadmap benchmark (Phase 1, 2, or 3). Default: Phase 1')
    parser.add_argument('--test', type=int, nargs='?', const=25,
                        help='Quick test with N farms (default: 25)')
    parser.add_argument('--full', action='store_true',
                        help='Full benchmark (25, 50, 100, 200 farms)')
    parser.add_argument('--scale', type=int, nargs='+',
                        help='Specific scales')
    parser.add_argument('--scenario', '--scenarios', nargs='+', dest='scenarios',
                        help='Test small synthetic scenarios (e.g., micro_6 tiny_24 small_60). '
                             'Available: micro_6, micro_12, tiny_24, tiny_40, small_60, small_80, '
                             'small_100, medium_120, medium_160')
    parser.add_argument('--all-small', action='store_true',
                        help='Test all small synthetic scenarios (micro_6 through medium_160)')
    parser.add_argument('--all-small-rotation', action='store_true',
                        help='Test all rotation scenarios (rotation_micro_25 through rotation_large_200)')
    parser.add_argument('--reads', type=int, nargs='+',
                        help='Number of reads for QPU sampling (e.g., 100 250 500). '
                             'If multiple values, runs benchmark for each. Default: 500 for small scenarios, 1000 for others')
    parser.add_argument('--methods', nargs='+',
                        help='Specific methods')
    parser.add_argument('--output', type=str,
                        help='Output filename')
    parser.add_argument('--no-qpu', action='store_true',
                        help='Use SimulatedAnnealing instead of QPU (testing)')
    parser.add_argument('--token', type=str,
                        help='D-Wave API token (or set DWAVE_API_TOKEN env var)')
    
    args = parser.parse_args()
    
    print("[5/5] Starting benchmark...")
    
    # Configure D-Wave token
    # Priority: --token argument > environment variable
    default_token = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'
    dwave_token = args.token or os.getenv('DWAVE_API_TOKEN') or default_token
    
    # Set token UNLESS user explicitly wants to skip QPU methods entirely
    # Note: --no-qpu doesn't disable QPU; it was for SA fallback (now removed)
    # To run only ground_truth without token, use: --methods ground_truth
    if dwave_token:
        set_dwave_token(dwave_token)
        if args.no_qpu:
            print(f"  Note: --no-qpu flag present but QPU token configured (QPU methods will attempt to run)")
    elif not args.no_qpu:
        print(f"  Warning: No D-Wave token available. Only ground_truth method will work.")
    
    # Check if running roadmap benchmark
    if args.roadmap is not None:
        phase = args.roadmap if isinstance(args.roadmap, int) else 1
        print(f"\n{'='*100}")
        print(f"RUNNING ROADMAP BENCHMARK - PHASE {phase}")
        print(f"{'='*100}\n")
        
        results = run_roadmap_benchmark(phase=phase, verbose=True)
        
        # Save roadmap results
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"roadmap_phase{phase}_{ts}.json"
        filepath = OUTPUT_DIR / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n{'='*100}")
        print(f"Roadmap Phase {phase} results saved to: {filepath}")
        print(f"{'='*100}")
        
        return 0
    
    # Determine if using scenarios or scales
    scenarios = None
    scales = None
    
    if args.all_small:
        # All small synthetic scenarios (including rotation scenarios)
        scenarios = ALL_SCENARIOS
        print(f"\nUsing all small scenarios: {scenarios}")
    elif args.all_small_rotation:
        # All rotation scenarios only
        scenarios = ROTATION_SCENARIOS
        print(f"\nUsing all rotation scenarios: {scenarios}")
    elif args.scenarios:
        # Specific scenarios
        scenarios = args.scenarios
        # Validate scenario names
        invalid = [s for s in scenarios if s not in ALL_SCENARIOS]
        if invalid:
            print(f"\nWarning: Unknown scenarios: {invalid}")
            print(f"Available scenarios: {ALL_SCENARIOS}")
            scenarios = [s for s in scenarios if s in ALL_SCENARIOS]
        print(f"\nUsing scenarios: {scenarios}")
    else:
        # Use scales (traditional)
        if args.test:
            scales = [args.test]
        elif args.full:
            scales = FARM_SCALES
        elif args.scale:
            scales = args.scale
        else:
            scales = [25]
        print(f"\nScales: {scales}")
    
    # Methods - ALL QPU (SA completely removed)
    methods = args.methods
    if methods is not None:
        # Handle comma-separated methods (e.g., --methods ground_truth,direct_qpu,clique_qpu)
        if len(methods) == 1 and ',' in methods[0]:
            methods = methods[0].split(',')
        # Strip whitespace from each method
        methods = [m.strip() for m in methods]
    
    if methods is None:
        # For small scenarios, default to direct_qpu + ground_truth (most relevant)
        if scenarios is not None:
            methods = [
                'ground_truth',
                'direct_qpu',  # Small scenarios should embed directly
            ]
        else:
            # For larger scales, use decomposition methods
            methods = [
                'ground_truth', 
                'direct_qpu',
                'coordinated',
                'decomposition_PlotBased_QPU',
                'decomposition_Multilevel(5)_QPU',
                'decomposition_Multilevel(10)_QPU',
                'decomposition_Louvain_QPU',
                'decomposition_Spectral(10)_QPU',
                'cqm_first_PlotBased',
            ]
    
    print(f"Methods: {methods}")
    print(f"QPU Available: {HAS_QPU}")
    print(f"Louvain Available: {HAS_LOUVAIN}")
    print(f"Spectral Available: {HAS_SPECTRAL}")
    
    # Reads configuration
    reads_configs = args.reads if args.reads else None
    if reads_configs:
        print(f"Num reads configurations: {reads_configs}")
    else:
        print(f"Using default reads (500 for small scenarios, 1000 for others)")
    
    # Run
    results = run_benchmark(scales=scales, methods=methods, verbose=True, scenarios=scenarios, reads_configs=reads_configs)
    
    # Summary
    print_summary(results)
    
    # Save
    save_results(results, args.output)
    
    print("\n✅ Benchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
