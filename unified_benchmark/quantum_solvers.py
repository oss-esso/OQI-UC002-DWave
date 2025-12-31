"""
Quantum/SA Solvers for the unified benchmark.

Implements three quantum-hybrid approaches:
1. qpu-native-6-family: Native small problems (6 families) without aggregation
2. qpu-hierarchical-aggregated: 27-food via aggregate->solve->refine pipeline
3. qpu-hybrid-27-food: 27-food variables with simplified 6-family synergy matrix

All solvers:
- Support both QPU and Simulated Annealing (SA) backends
- Recompute the true MIQP objective on the final 27-food assignment
- Record detailed timing and constraint violations
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from .core import (
    RunEntry,
    TimingInfo,
    DecompositionInfo,
    BenchmarkLogger,
    MIQP_PARAMS,
    create_run_entry,
)
from .scenarios import (
    load_scenario,
    build_rotation_matrix,
    build_spatial_neighbors,
)
from .miqp_scorer import (
    compute_miqp_objective,
    check_constraints,
    convert_family_solution_to_27food,
)

# D-Wave imports
try:
    from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler
    HAS_DIMOD = True
except ImportError:
    HAS_DIMOD = False

try:
    from dwave.system import DWaveCliqueSampler
    HAS_QPU = True
except ImportError:
    HAS_QPU = False

# Family order for 6-family problems
FAMILY_ORDER = ["Legumes", "Grains", "Vegetables", "Roots", "Fruits", "Other"]

# Food to family mapping
try:
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from food_grouping import FOOD_TO_FAMILY, get_family
except ImportError:
    FOOD_TO_FAMILY = {}
    def get_family(food: str) -> str:
        return "Other"


def create_family_rotation_matrix(seed: int = 42) -> np.ndarray:
    """
    Create 6×6 family rotation synergy matrix.
    
    From formulations.tex: R'[g1,g2] represents synergy between families.
    """
    np.random.seed(seed)
    n_families = 6
    R = np.zeros((n_families, n_families))
    
    frustration_ratio = 0.7
    negative_strength = -0.8
    
    for i in range(n_families):
        for j in range(n_families):
            if i == j:
                # Same family: strong negative
                R[i, j] = negative_strength * 1.5
            elif np.random.random() < frustration_ratio:
                # Most pairs: negative
                R[i, j] = np.random.uniform(negative_strength * 1.2, negative_strength * 0.3)
            else:
                # Some pairs: positive
                R[i, j] = np.random.uniform(0.02, 0.20)
    
    return R


def build_hybrid_rotation_matrix(
    food_names: List[str],
    seed: int = 42
) -> np.ndarray:
    """
    Build 27×27 rotation matrix using 6-family template.
    
    This is the "hybrid" approach from hybrid_formulation.py:
    - Build 6×6 family matrix
    - For each food pair, lookup synergy based on families
    - Add small noise for food-level diversity
    """
    np.random.seed(seed)
    n_foods = len(food_names)
    
    # Build 6×6 family template
    R_family = create_family_rotation_matrix(seed)
    
    # Map foods to family indices
    food_to_fam_idx = {}
    for food in food_names:
        family = get_family(food)
        if family in FAMILY_ORDER:
            food_to_fam_idx[food] = FAMILY_ORDER.index(family)
        else:
            food_to_fam_idx[food] = FAMILY_ORDER.index("Other")
    
    # Build food matrix
    R = np.zeros((n_foods, n_foods))
    for i, food_i in enumerate(food_names):
        for j, food_j in enumerate(food_names):
            fam_i = food_to_fam_idx[food_i]
            fam_j = food_to_fam_idx[food_j]
            base_synergy = R_family[fam_i, fam_j]
            noise = np.random.uniform(-0.05, 0.05)
            R[i, j] = base_synergy + noise
    
    return R


# ===========================================================================
# Mode 1: QPU-Native-6-Family
# ===========================================================================

def solve_native_6family(
    scenario_data: Dict[str, Any],
    use_qpu: bool = False,
    num_reads: int = 100,
    timeout: float = 600.0,
    params: Optional[Dict[str, float]] = None,
    verbose: bool = True,
    logger: Optional[BenchmarkLogger] = None,
    seed: int = 42,
    sampleset_dir: Optional[str] = None,
) -> RunEntry:
    """
    Solve a native 6-family problem without aggregation.
    
    This is for scenarios that already have 6 foods/families.
    The solution is then refined to 27 foods for MIQP scoring.
    
    Args:
        scenario_data: Must have n_foods == 6
        use_qpu: Use QPU (True) or SA (False)
        num_reads: Number of samples
        timeout: Wall clock timeout
        params: MIQP parameters
        verbose: Print progress
        logger: BenchmarkLogger
        seed: Random seed
        sampleset_dir: Directory to save samplesets as .pkl files
    
    Returns:
        RunEntry with results
    """
    if not HAS_DIMOD:
        raise RuntimeError("dimod not available")
    
    if params is None:
        params = MIQP_PARAMS.copy()
    
    if logger is None:
        logger = BenchmarkLogger()
    
    # Extract data
    farm_names = scenario_data["farm_names"]
    food_names = scenario_data["food_names"]
    land_availability = scenario_data["land_availability"]
    food_benefits = scenario_data["food_benefits"]
    total_area = scenario_data["total_area"]
    n_periods = scenario_data.get("n_periods", 3)
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    n_vars = n_farms * n_foods * n_periods
    
    if n_foods != 6:
        raise ValueError(f"Native 6-family mode requires 6 foods, got {n_foods}")
    
    # Create run entry
    entry = create_run_entry(
        mode="qpu-native-6-family",
        scenario_name=scenario_data.get("scenario_name", "unknown"),
        n_farms=n_farms,
        n_foods=n_foods,
        n_periods=n_periods,
        sampler="qpu" if use_qpu else "sa",
        backend="DWaveCliqueSampler" if use_qpu else "SimulatedAnnealingSampler",
        num_reads=num_reads,
        timeout_s=timeout,
        seed=seed,
        area_constant=scenario_data.get("area_constant", 1.0),
    )
    entry.timing = TimingInfo()
    
    # Get parameters
    rotation_gamma = params.get("rotation_gamma", 0.2)
    one_hot_penalty = params.get("one_hot_penalty", 3.0)
    diversity_bonus = params.get("diversity_bonus", 0.15)
    
    # Build rotation matrix (6×6)
    R = create_family_rotation_matrix(seed)
    
    logger.model_build_start("qpu-native-6-family", n_vars)
    
    total_start = time.time()
    build_start = time.time()
    
    try:
        # Build BQM
        bqm = BinaryQuadraticModel(vartype='BINARY')
        var_map = {}  # (farm_idx, food_idx, period) -> var_name
        
        # Add variables with linear biases
        for f_idx, farm in enumerate(farm_names):
            area_frac = land_availability[farm] / total_area
            for c_idx, food in enumerate(food_names):
                benefit = food_benefits.get(food, 1.0)
                for t in range(1, n_periods + 1):
                    var_name = f"Y_{f_idx}_{c_idx}_{t}"
                    var_map[(f_idx, c_idx, t)] = var_name
                    
                    # Linear bias: negative benefit (BQM minimizes)
                    linear_bias = -benefit * area_frac
                    # Add diversity bonus
                    linear_bias -= diversity_bonus / n_periods
                    # Add one-hot linear part
                    linear_bias -= one_hot_penalty
                    
                    bqm.add_variable(var_name, linear_bias)
        
        # Add quadratic terms
        
        # Temporal synergies
        for f_idx, farm in enumerate(farm_names):
            area_frac = land_availability[farm] / total_area
            for t in range(2, n_periods + 1):
                for c1_idx in range(n_foods):
                    for c2_idx in range(n_foods):
                        synergy = R[c1_idx, c2_idx]
                        var1 = var_map[(f_idx, c1_idx, t-1)]
                        var2 = var_map[(f_idx, c2_idx, t)]
                        bqm.add_quadratic(var1, var2, -rotation_gamma * synergy * area_frac)
        
        # Spatial synergies (within adjacent farms)
        for f_idx in range(n_farms - 1):
            for t in range(1, n_periods + 1):
                for c1_idx in range(n_foods):
                    for c2_idx in range(n_foods):
                        synergy = R[c1_idx, c2_idx] * 0.3
                        var1 = var_map[(f_idx, c1_idx, t)]
                        var2 = var_map[(f_idx + 1, c2_idx, t)]
                        bqm.add_quadratic(var1, var2, -rotation_gamma * 0.5 * synergy)
        
        # One-hot penalty (quadratic part)
        for f_idx in range(n_farms):
            for t in range(1, n_periods + 1):
                vars_this = [var_map[(f_idx, c, t)] for c in range(n_foods)]
                for i in range(len(vars_this)):
                    for j in range(i + 1, len(vars_this)):
                        bqm.add_quadratic(vars_this[i], vars_this[j], 2 * one_hot_penalty)
        
        # EXPLICIT ROTATION CONSTRAINT: Y_{f,c,t} + Y_{f,c,t+1} <= 1
        # Penalizes same crop in consecutive periods (hard constraint)
        rotation_constraint_penalty = 5.0  # Strong penalty to enforce constraint
        for f_idx in range(n_farms):
            for c_idx in range(n_foods):
                for t in range(1, n_periods):
                    var1 = var_map[(f_idx, c_idx, t)]
                    var2 = var_map[(f_idx, c_idx, t+1)]
                    bqm.add_quadratic(var1, var2, rotation_constraint_penalty)
        
        entry.timing.model_build_time = time.time() - build_start
        logger.model_build_done(entry.timing.model_build_time)
        
        # Solve
        logger.solve_start("QPU" if use_qpu else "SA", timeout)
        solve_start = time.time()
        
        if use_qpu and HAS_QPU:
            token = os.environ.get('DWAVE_API_TOKEN')
            sampler = DWaveCliqueSampler(token=token) if token else DWaveCliqueSampler()
            sampleset = sampler.sample(bqm, num_reads=num_reads)
            
            # Extract detailed QPU timing (all times in microseconds, convert to seconds)
            timing = sampleset.info.get('timing', {})
            entry.timing.qpu_access_time = timing.get('qpu_access_time', 0) / 1e6
            entry.timing.qpu_sampling_time = timing.get('qpu_sampling_time', 0) / 1e6
            entry.timing.embedding_time = timing.get('total_post_processing_time', 0) / 1e6
            
            # Log QPU timing details
            logger.info(f"QPU timing: access={entry.timing.qpu_access_time:.4f}s, "
                       f"sampling={entry.timing.qpu_sampling_time:.4f}s")
            
            # Save sampleset if directory provided
            if sampleset_dir:
                import pickle
                from pathlib import Path
                ss_dir = Path(sampleset_dir)
                ss_dir.mkdir(parents=True, exist_ok=True)
                scenario_name = scenario_data.get("scenario_name", "unknown")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                ss_path = ss_dir / f"sampleset_native6_{scenario_name}_{timestamp}.pkl"
                with open(ss_path, 'wb') as f:
                    pickle.dump({
                        'sampleset': sampleset,
                        'var_map': var_map,
                        'bqm': bqm,
                        'scenario_name': scenario_name,
                        'mode': 'qpu-native-6-family',
                        'num_reads': num_reads,
                        'timestamp': timestamp,
                    }, f)
                logger.info(f"Sampleset saved to {ss_path}")
        else:
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=num_reads)
        
        entry.timing.solve_time = time.time() - solve_start
        
        # Extract best solution
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy
        
        entry.objective_model = -best_energy  # Convert to maximization
        entry.status = "feasible"
        
        logger.solve_done(entry.status, entry.timing.solve_time, entry.objective_model)
        
        # Convert to indexed solution
        reverse_map = {v: k for k, v in var_map.items()}
        solution = {}
        for var_name, value in best_sample.items():
            if var_name in reverse_map and value == 1:
                key = reverse_map[var_name]
                solution[key] = 1
        
        entry.solution = solution
        
        # Refinement: expand 6-family to 27-food for MIQP scoring
        logger.refinement_start("6-to-27 expansion")
        refine_start = time.time()
        
        # Load 27-food scenario data for proper MIQP scoring
        try:
            # Get a 27-food reference
            ref_data = load_scenario("rotation_25farms_27foods", 
                                     area_constant=scenario_data.get("area_constant", 1.0))
            food_names_27 = ref_data["food_names"]
            
            # Expand solution
            expanded_solution = convert_family_solution_to_27food(
                solution, 
                {"food_names": food_names_27, **scenario_data},
                FOOD_TO_FAMILY,
                seed=seed
            )
            
            entry.timing.refinement_time = time.time() - refine_start
            logger.refinement_done(entry.timing.refinement_time)
            
            # Compute MIQP objective on expanded solution
            miqp_start = time.time()
            
            # Create scenario data for 27 foods
            scenario_27 = scenario_data.copy()
            scenario_27["food_names"] = food_names_27
            scenario_27["n_foods"] = len(food_names_27)
            scenario_27["food_benefits"] = ref_data["food_benefits"]
            
            R_27 = build_hybrid_rotation_matrix(food_names_27, seed=seed)
            neighbor_edges = build_spatial_neighbors(farm_names, k_neighbors=params.get("k_neighbors", 4))
            
            entry.objective_miqp = compute_miqp_objective(
                expanded_solution, scenario_27, R=R_27, neighbor_edges=neighbor_edges, params=params
            )
            entry.timing.miqp_recompute_time = time.time() - miqp_start
            
            logger.miqp_recompute(entry.objective_miqp, entry.timing.miqp_recompute_time)
            
        except Exception as e:
            # If refinement fails, use 6-family objective
            entry.objective_miqp = entry.objective_model
            logger.warning(f"Refinement failed, using model objective: {e}")
        
        # Check constraints on original solution
        violations = check_constraints(solution, scenario_data, params)
        entry.constraint_violations = violations
        entry.feasible = violations.total_violations == 0
        
        logger.constraint_check(violations.total_violations, entry.feasible)
        
        entry.timing.total_wall_time = time.time() - total_start
        
    except Exception as e:
        entry.status = "error"
        entry.error_message = str(e)
        entry.timing.total_wall_time = time.time() - total_start
        logger.error(f"Native 6-family solver error: {e}")
    
    return entry


# ===========================================================================
# Mode 2: QPU-Hierarchical-Aggregated
# ===========================================================================

def solve_hierarchical_aggregated(
    scenario_data: Dict[str, Any],
    use_qpu: bool = False,
    num_reads: int = 100,
    num_iterations: int = 3,
    farms_per_cluster: int = 10,
    timeout: float = 600.0,
    params: Optional[Dict[str, float]] = None,
    verbose: bool = True,
    logger: Optional[BenchmarkLogger] = None,
    seed: int = 42,
    sampleset_dir: Optional[str] = None,
) -> RunEntry:
    """
    Solve 27-food problem via aggregate->solve->refine pipeline.
    
    Level 1: Aggregate 27 foods to 6 families (classical)
    Level 2: Solve 6-family problem on QPU/SA with spatial decomposition
    Level 3: Refine back to 27 foods and score with true MIQP
    
    Args:
        scenario_data: Scenario with any number of foods (will aggregate to 6)
        use_qpu: Use QPU (True) or SA (False)
        num_reads: Number of samples per subproblem
        num_iterations: Boundary coordination iterations
        farms_per_cluster: Farms per cluster for decomposition
        timeout: Wall clock timeout
        params: MIQP parameters
        verbose: Print progress
        logger: BenchmarkLogger
        seed: Random seed
    
    Returns:
        RunEntry with results
    """
    if not HAS_DIMOD:
        raise RuntimeError("dimod not available")
    
    if params is None:
        params = MIQP_PARAMS.copy()
    
    if logger is None:
        logger = BenchmarkLogger()
    
    # Extract data
    farm_names = scenario_data["farm_names"]
    food_names_orig = scenario_data["food_names"]
    land_availability = scenario_data["land_availability"]
    food_benefits_orig = scenario_data["food_benefits"]
    total_area = scenario_data["total_area"]
    n_periods = scenario_data.get("n_periods", 3)
    
    n_farms = len(farm_names)
    n_foods_orig = len(food_names_orig)
    
    # Aggregated to 6 families
    n_foods_agg = 6
    n_vars_agg = n_farms * n_foods_agg * n_periods
    
    # Create run entry (report original problem size)
    entry = create_run_entry(
        mode="qpu-hierarchical-aggregated",
        scenario_name=scenario_data.get("scenario_name", "unknown"),
        n_farms=n_farms,
        n_foods=n_foods_orig,
        n_periods=n_periods,
        sampler="qpu" if use_qpu else "sa",
        backend="DWaveCliqueSampler" if use_qpu else "SimulatedAnnealingSampler",
        num_reads=num_reads,
        timeout_s=timeout,
        seed=seed,
        area_constant=scenario_data.get("area_constant", 1.0),
    )
    entry.timing = TimingInfo()
    entry.decomposition = DecompositionInfo()
    
    # Get parameters
    rotation_gamma = params.get("rotation_gamma", 0.2)
    one_hot_penalty = params.get("one_hot_penalty", 3.0)
    diversity_bonus = params.get("diversity_bonus", 0.15)
    
    logger.model_build_start("qpu-hierarchical-aggregated", n_vars_agg)
    logger.info(f"Aggregating {n_foods_orig} foods to {n_foods_agg} families")
    
    total_start = time.time()
    build_start = time.time()
    
    try:
        # ========== LEVEL 1: AGGREGATION ==========
        
        # Aggregate food benefits to family level
        family_benefits = {}
        family_counts = defaultdict(int)
        
        for food in food_names_orig:
            family = get_family(food)
            benefit = food_benefits_orig.get(food, 1.0)
            
            if family not in family_benefits:
                family_benefits[family] = 0.0
            family_benefits[family] += benefit
            family_counts[family] += 1
        
        # Average benefits
        for family in family_benefits:
            if family_counts[family] > 0:
                family_benefits[family] /= family_counts[family]
        
        # Ensure all families have benefits
        for family in FAMILY_ORDER:
            if family not in family_benefits:
                family_benefits[family] = 1.0
        
        # Build 6×6 family rotation matrix
        R = create_family_rotation_matrix(seed)
        
        # ========== LEVEL 2: SPATIAL DECOMPOSITION ==========
        
        # Create clusters
        clusters = []
        for i in range(0, n_farms, farms_per_cluster):
            cluster = farm_names[i:i + farms_per_cluster]
            if cluster:
                clusters.append(cluster)
        
        n_clusters = len(clusters)
        entry.decomposition.method = "spatial_grid"
        entry.decomposition.n_clusters = n_clusters
        entry.decomposition.farms_per_cluster = farms_per_cluster
        entry.decomposition.cluster_sizes = [len(c) for c in clusters]
        entry.decomposition.iterations = num_iterations
        
        logger.info(f"Decomposed into {n_clusters} clusters of ~{farms_per_cluster} farms")
        
        entry.timing.model_build_time = time.time() - build_start
        logger.model_build_done(entry.timing.model_build_time)
        
        # ========== SOLVE CLUSTERS WITH BOUNDARY COORDINATION ==========
        
        logger.solve_start("QPU" if use_qpu else "SA", timeout)
        solve_start = time.time()
        
        # Initialize cluster solutions
        cluster_solutions = [None] * n_clusters
        total_qpu_time = 0.0
        total_qpu_sampling = 0.0
        all_samplesets = []  # Track saved sampleset paths
        
        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            
            for cluster_idx, cluster_farms in enumerate(clusters):
                # Build BQM for this cluster
                bqm = BinaryQuadraticModel(vartype='BINARY')
                var_map = {}
                
                # Variables
                for f_idx, farm in enumerate(cluster_farms):
                    farm_global_idx = farm_names.index(farm)
                    area_frac = land_availability[farm] / total_area
                    
                    for c_idx, family in enumerate(FAMILY_ORDER):
                        benefit = family_benefits.get(family, 1.0)
                        for t in range(1, n_periods + 1):
                            var_name = f"Y_{farm_global_idx}_{c_idx}_{t}"
                            var_map[(farm_global_idx, c_idx, t)] = var_name
                            
                            # Linear bias
                            linear_bias = -benefit * area_frac
                            linear_bias -= diversity_bonus / n_periods
                            linear_bias -= one_hot_penalty
                            
                            bqm.add_variable(var_name, linear_bias)
                
                # Temporal synergies
                for f_idx, farm in enumerate(cluster_farms):
                    farm_global_idx = farm_names.index(farm)
                    area_frac = land_availability[farm] / total_area
                    for t in range(2, n_periods + 1):
                        for c1_idx in range(n_foods_agg):
                            for c2_idx in range(n_foods_agg):
                                synergy = R[c1_idx, c2_idx]
                                var1 = var_map[(farm_global_idx, c1_idx, t-1)]
                                var2 = var_map[(farm_global_idx, c2_idx, t)]
                                bqm.add_quadratic(var1, var2, -rotation_gamma * synergy * area_frac)
                
                # Spatial synergies within cluster
                for f_idx in range(len(cluster_farms) - 1):
                    farm1 = cluster_farms[f_idx]
                    farm2 = cluster_farms[f_idx + 1]
                    farm1_global = farm_names.index(farm1)
                    farm2_global = farm_names.index(farm2)
                    
                    for t in range(1, n_periods + 1):
                        for c1_idx in range(n_foods_agg):
                            for c2_idx in range(n_foods_agg):
                                synergy = R[c1_idx, c2_idx] * 0.3
                                var1 = var_map[(farm1_global, c1_idx, t)]
                                var2 = var_map[(farm2_global, c2_idx, t)]
                                bqm.add_quadratic(var1, var2, -rotation_gamma * 0.5 * synergy)
                
                # One-hot penalty
                for f_idx, farm in enumerate(cluster_farms):
                    farm_global_idx = farm_names.index(farm)
                    for t in range(1, n_periods + 1):
                        vars_this = [var_map[(farm_global_idx, c, t)] for c in range(n_foods_agg)]
                        for i in range(len(vars_this)):
                            for j in range(i + 1, len(vars_this)):
                                bqm.add_quadratic(vars_this[i], vars_this[j], 2 * one_hot_penalty)
                
                # EXPLICIT ROTATION CONSTRAINT: Y_{f,c,t} + Y_{f,c,t+1} <= 1
                rotation_constraint_penalty = 5.0
                for f_idx, farm in enumerate(cluster_farms):
                    farm_global_idx = farm_names.index(farm)
                    for c_idx in range(n_foods_agg):
                        for t in range(1, n_periods):
                            if (farm_global_idx, c_idx, t) in var_map and (farm_global_idx, c_idx, t+1) in var_map:
                                var1 = var_map[(farm_global_idx, c_idx, t)]
                                var2 = var_map[(farm_global_idx, c_idx, t+1)]
                                bqm.add_quadratic(var1, var2, rotation_constraint_penalty)
                
                # Boundary coordination from previous iteration
                if iteration > 0 and cluster_idx > 0:
                    prev_solution = cluster_solutions[cluster_idx - 1]
                    if prev_solution:
                        # Add soft constraints to align boundaries
                        boundary_farm = cluster_farms[0]
                        boundary_global = farm_names.index(boundary_farm)
                        
                        for key, val in prev_solution.items():
                            if val == 1:
                                f_idx, c_idx, t = key
                                if (boundary_global, c_idx, t) in var_map:
                                    var = var_map[(boundary_global, c_idx, t)]
                                    bqm.add_linear(var, -0.5 * rotation_gamma)
                
                # Solve cluster
                if use_qpu and HAS_QPU:
                    token = os.environ.get('DWAVE_API_TOKEN')
                    sampler = DWaveCliqueSampler(token=token) if token else DWaveCliqueSampler()
                    sampleset = sampler.sample(bqm, num_reads=num_reads)
                    
                    timing = sampleset.info.get('timing', {})
                    total_qpu_time += timing.get('qpu_access_time', 0) / 1e6
                    total_qpu_sampling += timing.get('qpu_sampling_time', 0) / 1e6
                    
                    # Save sampleset if directory provided
                    if sampleset_dir:
                        import pickle
                        from pathlib import Path
                        ss_dir = Path(sampleset_dir)
                        ss_dir.mkdir(parents=True, exist_ok=True)
                        scenario_name = scenario_data.get("scenario_name", "unknown")
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        ss_path = ss_dir / f"sampleset_hier_{scenario_name}_iter{iteration}_cluster{cluster_idx}_{timestamp}.pkl"
                        with open(ss_path, 'wb') as f:
                            pickle.dump({
                                'sampleset': sampleset,
                                'var_map': var_map,
                                'bqm': bqm,
                                'scenario_name': scenario_name,
                                'mode': 'qpu-hierarchical-aggregated',
                                'iteration': iteration,
                                'cluster_idx': cluster_idx,
                                'cluster_farms': cluster_farms,
                                'num_reads': num_reads,
                                'timestamp': timestamp,
                            }, f)
                        all_samplesets.append(str(ss_path))
                else:
                    sampler = SimulatedAnnealingSampler()
                    sampleset = sampler.sample(bqm, num_reads=num_reads)
                
                # Extract solution
                best_sample = sampleset.first.sample
                reverse_map = {v: k for k, v in var_map.items()}
                
                cluster_sol = {}
                for var_name, value in best_sample.items():
                    if var_name in reverse_map and value == 1:
                        key = reverse_map[var_name]
                        cluster_sol[key] = 1
                
                cluster_solutions[cluster_idx] = cluster_sol
        
        entry.timing.solve_time = time.time() - solve_start
        entry.timing.qpu_access_time = total_qpu_time if use_qpu else None
        entry.timing.qpu_sampling_time = total_qpu_sampling if use_qpu else None
        
        if use_qpu:
            logger.info(f"QPU timing: access={total_qpu_time:.4f}s, sampling={total_qpu_sampling:.4f}s")
        
        # Merge cluster solutions
        global_solution = {}
        for cluster_sol in cluster_solutions:
            if cluster_sol:
                global_solution.update(cluster_sol)
        
        entry.status = "feasible"
        logger.solve_done(entry.status, entry.timing.solve_time)
        
        # ========== LEVEL 3: REFINEMENT TO 27 FOODS ==========
        
        logger.refinement_start("family-to-crop expansion")
        refine_start = time.time()
        
        # Expand to 27 foods
        # Load reference data for 27 foods
        try:
            ref_data = load_scenario("rotation_25farms_27foods",
                                     area_constant=scenario_data.get("area_constant", 1.0))
            food_names_27 = ref_data["food_names"]
            food_benefits_27 = ref_data["food_benefits"]
        except:
            food_names_27 = food_names_orig
            food_benefits_27 = food_benefits_orig
        
        expanded_solution = convert_family_solution_to_27food(
            global_solution,
            {"food_names": food_names_27, **scenario_data},
            FOOD_TO_FAMILY,
            seed=seed
        )
        
        entry.timing.refinement_time = time.time() - refine_start
        logger.refinement_done(entry.timing.refinement_time)
        
        entry.solution = global_solution  # Store aggregated solution
        
        # ========== MIQP OBJECTIVE RECOMPUTATION ==========
        
        miqp_start = time.time()
        
        scenario_27 = scenario_data.copy()
        scenario_27["food_names"] = food_names_27
        scenario_27["n_foods"] = len(food_names_27)
        scenario_27["food_benefits"] = food_benefits_27
        
        R_27 = build_hybrid_rotation_matrix(food_names_27, seed=seed)
        neighbor_edges = build_spatial_neighbors(farm_names, k_neighbors=params.get("k_neighbors", 4))
        
        entry.objective_miqp = compute_miqp_objective(
            expanded_solution, scenario_27, R=R_27, neighbor_edges=neighbor_edges, params=params
        )
        entry.timing.miqp_recompute_time = time.time() - miqp_start
        
        logger.miqp_recompute(entry.objective_miqp, entry.timing.miqp_recompute_time)
        
        # Check constraints on aggregated solution
        scenario_6 = scenario_data.copy()
        scenario_6["food_names"] = FAMILY_ORDER
        scenario_6["n_foods"] = 6
        
        violations = check_constraints(global_solution, scenario_6, params)
        entry.constraint_violations = violations
        entry.feasible = violations.total_violations == 0
        
        logger.constraint_check(violations.total_violations, entry.feasible)
        
        entry.timing.total_wall_time = time.time() - total_start
        
    except Exception as e:
        entry.status = "error"
        entry.error_message = str(e)
        entry.timing.total_wall_time = time.time() - total_start
        logger.error(f"Hierarchical solver error: {e}")
    
    return entry


# ===========================================================================
# Mode 3: QPU-Hybrid-27-Food
# ===========================================================================

def solve_hybrid_27food(
    scenario_data: Dict[str, Any],
    use_qpu: bool = False,
    num_reads: int = 100,
    num_iterations: int = 3,
    farms_per_cluster: int = 2,  # Very small due to 27 vars per farm-period
    timeout: float = 600.0,
    params: Optional[Dict[str, float]] = None,
    verbose: bool = True,
    logger: Optional[BenchmarkLogger] = None,
    seed: int = 42,
    sampleset_dir: Optional[str] = None,
) -> RunEntry:
    """
    Solve 27-food problem with 27-food variables but 6-family synergy matrix.
    
    This is the "hybrid" approach:
    - Full 27-food variable space (no aggregation)
    - Synergy matrix built from 6-family template
    - Spatial decomposition to fit QPU
    
    Args:
        scenario_data: Scenario with 27 foods
        use_qpu: Use QPU (True) or SA (False)
        num_reads: Number of samples
        num_iterations: Boundary coordination iterations
        farms_per_cluster: Farms per cluster (default 2 for 27-food)
        timeout: Wall clock timeout
        params: MIQP parameters
        verbose: Print progress
        logger: BenchmarkLogger
        seed: Random seed
    
    Returns:
        RunEntry with results
    """
    if not HAS_DIMOD:
        raise RuntimeError("dimod not available")
    
    if params is None:
        params = MIQP_PARAMS.copy()
    
    if logger is None:
        logger = BenchmarkLogger()
    
    # Extract data
    farm_names = scenario_data["farm_names"]
    food_names = scenario_data["food_names"]
    land_availability = scenario_data["land_availability"]
    food_benefits = scenario_data["food_benefits"]
    total_area = scenario_data["total_area"]
    n_periods = scenario_data.get("n_periods", 3)
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    n_vars = n_farms * n_foods * n_periods
    
    # QPU clique sampler limit is ~177 variables
    # For 27 foods × 3 periods = 81 vars per farm, max ~2 farms per cluster
    vars_per_farm = n_foods * n_periods
    max_farms_per_cluster = min(farms_per_cluster, max(1, 170 // vars_per_farm))
    
    # Create run entry
    entry = create_run_entry(
        mode="qpu-hybrid-27-food",
        scenario_name=scenario_data.get("scenario_name", "unknown"),
        n_farms=n_farms,
        n_foods=n_foods,
        n_periods=n_periods,
        sampler="qpu" if use_qpu else "sa",
        backend="DWaveCliqueSampler" if use_qpu else "SimulatedAnnealingSampler",
        num_reads=num_reads,
        timeout_s=timeout,
        seed=seed,
        area_constant=scenario_data.get("area_constant", 1.0),
    )
    entry.timing = TimingInfo()
    entry.decomposition = DecompositionInfo()
    
    # Get parameters
    rotation_gamma = params.get("rotation_gamma", 0.2)
    spatial_gamma = params.get("spatial_gamma", 0.1)
    one_hot_penalty = params.get("one_hot_penalty", 3.0)
    diversity_bonus = params.get("diversity_bonus", 0.15)
    
    logger.model_build_start("qpu-hybrid-27-food", n_vars)
    
    total_start = time.time()
    build_start = time.time()
    
    try:
        # Build 27×27 hybrid rotation matrix (from 6-family template)
        R = build_hybrid_rotation_matrix(food_names, seed=seed)
        
        # Create clusters
        clusters = []
        for i in range(0, n_farms, max_farms_per_cluster):
            cluster = farm_names[i:i + max_farms_per_cluster]
            if cluster:
                clusters.append(cluster)
        
        n_clusters = len(clusters)
        
        entry.decomposition.method = "spatial_hybrid"
        entry.decomposition.n_clusters = n_clusters
        entry.decomposition.farms_per_cluster = max_farms_per_cluster
        entry.decomposition.cluster_sizes = [len(c) for c in clusters]
        entry.decomposition.iterations = num_iterations
        
        logger.info(f"Hybrid decomposition: {n_clusters} clusters × ~{max_farms_per_cluster} farms")
        logger.info(f"Variables per cluster: ~{max_farms_per_cluster * vars_per_farm}")
        
        entry.timing.model_build_time = time.time() - build_start
        logger.model_build_done(entry.timing.model_build_time)
        
        # ========== SOLVE CLUSTERS ==========
        
        logger.solve_start("QPU" if use_qpu else "SA", timeout)
        solve_start = time.time()
        
        cluster_solutions = [None] * n_clusters
        total_qpu_time = 0.0
        total_qpu_sampling = 0.0
        all_samplesets = []  # Track saved sampleset paths
        
        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            
            for cluster_idx, cluster_farms in enumerate(clusters):
                # Build BQM for this cluster
                bqm = BinaryQuadraticModel(vartype='BINARY')
                var_map = {}
                
                # Variables
                for f_local_idx, farm in enumerate(cluster_farms):
                    f_global_idx = farm_names.index(farm)
                    area_frac = land_availability[farm] / total_area
                    
                    for c_idx, food in enumerate(food_names):
                        benefit = food_benefits.get(food, 1.0)
                        for t in range(1, n_periods + 1):
                            var_name = f"Y_{f_global_idx}_{c_idx}_{t}"
                            var_map[(f_global_idx, c_idx, t)] = var_name
                            
                            # Linear bias
                            linear_bias = -benefit * area_frac
                            linear_bias -= diversity_bonus / n_periods
                            linear_bias -= one_hot_penalty
                            
                            bqm.add_variable(var_name, linear_bias)
                
                # Temporal synergies (using hybrid R matrix)
                for farm in cluster_farms:
                    f_global_idx = farm_names.index(farm)
                    area_frac = land_availability[farm] / total_area
                    
                    for t in range(2, n_periods + 1):
                        for c1_idx in range(n_foods):
                            for c2_idx in range(n_foods):
                                synergy = R[c1_idx, c2_idx]
                                if abs(synergy) > 1e-8:
                                    var1 = var_map[(f_global_idx, c1_idx, t-1)]
                                    var2 = var_map[(f_global_idx, c2_idx, t)]
                                    bqm.add_quadratic(var1, var2, -rotation_gamma * synergy * area_frac)
                
                # Spatial synergies within cluster
                for f_local_idx in range(len(cluster_farms) - 1):
                    farm1 = cluster_farms[f_local_idx]
                    farm2 = cluster_farms[f_local_idx + 1]
                    f1_global = farm_names.index(farm1)
                    f2_global = farm_names.index(farm2)
                    
                    for t in range(1, n_periods + 1):
                        for c1_idx in range(n_foods):
                            for c2_idx in range(n_foods):
                                synergy = R[c1_idx, c2_idx]
                                if abs(synergy) > 1e-8:
                                    var1 = var_map[(f1_global, c1_idx, t)]
                                    var2 = var_map[(f2_global, c2_idx, t)]
                                    bqm.add_quadratic(var1, var2, -spatial_gamma * synergy)
                
                # One-hot penalty
                for farm in cluster_farms:
                    f_global_idx = farm_names.index(farm)
                    for t in range(1, n_periods + 1):
                        vars_this = [var_map[(f_global_idx, c, t)] for c in range(n_foods)]
                        for i in range(len(vars_this)):
                            for j in range(i + 1, len(vars_this)):
                                bqm.add_quadratic(vars_this[i], vars_this[j], 2 * one_hot_penalty)
                
                # EXPLICIT ROTATION CONSTRAINT: Y_{f,c,t} + Y_{f,c,t+1} <= 1
                rotation_constraint_penalty = 5.0
                for farm in cluster_farms:
                    f_global_idx = farm_names.index(farm)
                    for c_idx in range(n_foods):
                        for t in range(1, n_periods):
                            if (f_global_idx, c_idx, t) in var_map and (f_global_idx, c_idx, t+1) in var_map:
                                var1 = var_map[(f_global_idx, c_idx, t)]
                                var2 = var_map[(f_global_idx, c_idx, t+1)]
                                bqm.add_quadratic(var1, var2, rotation_constraint_penalty)
                
                # Boundary coordination
                if iteration > 0 and cluster_idx > 0:
                    prev_solution = cluster_solutions[cluster_idx - 1]
                    if prev_solution:
                        boundary_farm = cluster_farms[0]
                        boundary_global = farm_names.index(boundary_farm)
                        
                        for key, val in prev_solution.items():
                            if val == 1:
                                f_idx, c_idx, t = key
                                if (boundary_global, c_idx, t) in var_map:
                                    var = var_map[(boundary_global, c_idx, t)]
                                    bqm.add_linear(var, -0.3 * rotation_gamma)
                
                # Solve
                if use_qpu and HAS_QPU:
                    token = os.environ.get('DWAVE_API_TOKEN')
                    sampler = DWaveCliqueSampler(token=token) if token else DWaveCliqueSampler()
                    sampleset = sampler.sample(bqm, num_reads=num_reads)
                    
                    timing = sampleset.info.get('timing', {})
                    total_qpu_time += timing.get('qpu_access_time', 0) / 1e6
                    total_qpu_sampling += timing.get('qpu_sampling_time', 0) / 1e6
                    
                    # Save sampleset if directory provided
                    if sampleset_dir:
                        import pickle
                        from pathlib import Path
                        ss_dir = Path(sampleset_dir)
                        ss_dir.mkdir(parents=True, exist_ok=True)
                        scenario_name = scenario_data.get("scenario_name", "unknown")
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        ss_path = ss_dir / f"sampleset_hybrid27_{scenario_name}_iter{iteration}_cluster{cluster_idx}_{timestamp}.pkl"
                        with open(ss_path, 'wb') as f:
                            pickle.dump({
                                'sampleset': sampleset,
                                'var_map': var_map,
                                'bqm': bqm,
                                'scenario_name': scenario_name,
                                'mode': 'qpu-hybrid-27-food',
                                'iteration': iteration,
                                'cluster_idx': cluster_idx,
                                'cluster_farms': cluster_farms,
                                'num_reads': num_reads,
                                'timestamp': timestamp,
                            }, f)
                        all_samplesets.append(str(ss_path))
                else:
                    sampler = SimulatedAnnealingSampler()
                    sampleset = sampler.sample(bqm, num_reads=num_reads)
                
                # Extract solution
                best_sample = sampleset.first.sample
                reverse_map = {v: k for k, v in var_map.items()}
                
                cluster_sol = {}
                for var_name, value in best_sample.items():
                    if var_name in reverse_map and value == 1:
                        key = reverse_map[var_name]
                        cluster_sol[key] = 1
                
                cluster_solutions[cluster_idx] = cluster_sol
        
        entry.timing.solve_time = time.time() - solve_start
        entry.timing.qpu_access_time = total_qpu_time if use_qpu else None
        entry.timing.qpu_sampling_time = total_qpu_sampling if use_qpu else None
        
        if use_qpu:
            logger.info(f"QPU timing: access={total_qpu_time:.4f}s, sampling={total_qpu_sampling:.4f}s")
        
        # Merge cluster solutions
        global_solution = {}
        for cluster_sol in cluster_solutions:
            if cluster_sol:
                global_solution.update(cluster_sol)
        
        entry.status = "feasible"
        entry.solution = global_solution
        
        logger.solve_done(entry.status, entry.timing.solve_time)
        
        # ========== MIQP OBJECTIVE RECOMPUTATION ==========
        
        miqp_start = time.time()
        
        neighbor_edges = build_spatial_neighbors(farm_names, k_neighbors=params.get("k_neighbors", 4))
        
        entry.objective_miqp = compute_miqp_objective(
            global_solution, scenario_data, R=R, neighbor_edges=neighbor_edges, params=params
        )
        entry.timing.miqp_recompute_time = time.time() - miqp_start
        
        logger.miqp_recompute(entry.objective_miqp, entry.timing.miqp_recompute_time)
        
        # Check constraints
        violations = check_constraints(global_solution, scenario_data, params)
        entry.constraint_violations = violations
        entry.feasible = violations.total_violations == 0
        
        logger.constraint_check(violations.total_violations, entry.feasible)
        
        entry.timing.total_wall_time = time.time() - total_start
        
    except Exception as e:
        entry.status = "error"
        entry.error_message = str(e)
        entry.timing.total_wall_time = time.time() - total_start
        logger.error(f"Hybrid 27-food solver error: {e}")
    
    return entry


# ===========================================================================
# Unified Solver Interface
# ===========================================================================

def solve(
    mode: str,
    scenario_data: Dict[str, Any],
    use_qpu: bool = False,
    num_reads: int = 100,
    timeout: float = 600.0,
    params: Optional[Dict[str, float]] = None,
    verbose: bool = True,
    logger: Optional[BenchmarkLogger] = None,
    seed: int = 42,
    sampleset_dir: Optional[str] = None,
    **kwargs
) -> RunEntry:
    """
    Unified interface to all solvers.
    
    Args:
        mode: One of 'gurobi-true-ground-truth', 'qpu-native-6-family',
              'qpu-hierarchical-aggregated', 'qpu-hybrid-27-food'
        scenario_data: Scenario data
        use_qpu: Use QPU (for quantum modes)
        num_reads: Number of samples (for quantum modes)
        timeout: Wall clock timeout
        params: MIQP parameters
        verbose: Print progress
        logger: BenchmarkLogger
        seed: Random seed
        sampleset_dir: Directory to save samplesets as .pkl files
        **kwargs: Mode-specific arguments
    
    Returns:
        RunEntry with results
    """
    if mode == "gurobi-true-ground-truth":
        from .gurobi_solver import solve_gurobi_ground_truth
        return solve_gurobi_ground_truth(
            scenario_data, timeout=timeout, params=params,
            verbose=verbose, logger=logger, seed=seed
        )
    
    elif mode == "qpu-native-6-family":
        return solve_native_6family(
            scenario_data, use_qpu=use_qpu, num_reads=num_reads,
            timeout=timeout, params=params, verbose=verbose,
            logger=logger, seed=seed, sampleset_dir=sampleset_dir
        )
    
    elif mode == "qpu-hierarchical-aggregated":
        return solve_hierarchical_aggregated(
            scenario_data, use_qpu=use_qpu, num_reads=num_reads,
            num_iterations=kwargs.get("num_iterations", 3),
            farms_per_cluster=kwargs.get("farms_per_cluster", 10),
            timeout=timeout, params=params, verbose=verbose,
            logger=logger, seed=seed, sampleset_dir=sampleset_dir
        )
    
    elif mode == "qpu-hybrid-27-food":
        return solve_hybrid_27food(
            scenario_data, use_qpu=use_qpu, num_reads=num_reads,
            num_iterations=kwargs.get("num_iterations", 3),
            farms_per_cluster=kwargs.get("farms_per_cluster", 2),
            timeout=timeout, params=params, verbose=verbose,
            logger=logger, seed=seed, sampleset_dir=sampleset_dir
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    # Test quantum solvers
    print("Testing quantum solvers with SA...")
    print("=" * 70)
    
    if not HAS_DIMOD:
        print("dimod not available!")
        exit(1)
    
    # Test native 6-family
    print("\n1. Testing Native 6-Family mode")
    print("-" * 40)
    data = load_scenario("rotation_micro_25")  # 6-family scenario
    print(f"Scenario: {data['n_farms']} farms × {data['n_foods']} foods")
    
    result = solve_native_6family(data, use_qpu=False, num_reads=50)
    print(f"Status: {result.status}")
    print(f"MIQP objective: {result.objective_miqp}")
    print(f"Solve time: {result.timing.solve_time:.2f}s")
    print(f"Feasible: {result.feasible}")
    
    # Test hierarchical
    print("\n2. Testing Hierarchical Aggregated mode")
    print("-" * 40)
    data = load_scenario("rotation_25farms_27foods")
    print(f"Scenario: {data['n_farms']} farms × {data['n_foods']} foods")
    
    result = solve_hierarchical_aggregated(data, use_qpu=False, num_reads=50, num_iterations=2)
    print(f"Status: {result.status}")
    print(f"MIQP objective: {result.objective_miqp}")
    print(f"Solve time: {result.timing.solve_time:.2f}s")
    print(f"Clusters: {result.decomposition.n_clusters}")
    print(f"Feasible: {result.feasible}")
    
    # Test hybrid
    print("\n3. Testing Hybrid 27-Food mode")
    print("-" * 40)
    data = load_scenario("rotation_25farms_27foods")
    print(f"Scenario: {data['n_farms']} farms × {data['n_foods']} foods")
    
    result = solve_hybrid_27food(data, use_qpu=False, num_reads=50, num_iterations=2)
    print(f"Status: {result.status}")
    print(f"MIQP objective: {result.objective_miqp}")
    print(f"Solve time: {result.timing.solve_time:.2f}s")
    print(f"Clusters: {result.decomposition.n_clusters}")
    print(f"Feasible: {result.feasible}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
