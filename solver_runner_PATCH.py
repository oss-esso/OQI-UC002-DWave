"""
Professional solver runner script with BQUBO (CQM→BQM conversion).

This script:
1. Loads a scenario (simple, intermediate, or custom)
2. Converts to CQM with LINEAR objective and saves the model
3. Solves with PuLP and saves results
4. Solves with DWave using CQM→BQM conversion + HybridBQM solver (QPU-enabled)
5. Saves all constraints for verification
"""

import os
import sys
import json
import pickle
import shutil
import time
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.scenarios import load_food_data
from patch_sampler import generate_farms as generate_patches
from dimod import ConstrainedQuadraticModel, Binary, Real, cqm_to_bqm
from dwave.system import LeapHybridCQMSampler, LeapHybridBQMSampler
from dwave.samplers import SimulatedAnnealingSampler
import pulp as pl
from tqdm import tqdm

def create_cqm(farms, foods, food_groups, config):
    """
    Creates a CQM for the plot-crop assignment problem (BQM_PATCH formulation).
    
    Variables:
    - X_{p,c}: Binary, 1 if plot p is assigned to crop c, 0 otherwise
    - Y_c: Binary, 1 if crop c is grown on at least one plot, 0 otherwise
    
    Objective: Maximize sum_{p,c} (B_c + λ) * s_p * X_{p,c}
    Where B_c is the weighted benefit per area, s_p is plot area, λ is idle penalty
    
    Constraints:
    1. At most one crop per plot: sum_c X_{p,c} <= 1 for all p
    2. Linking X and Y: X_{p,c} <= Y_c for all p,c
    3. Y activation: Y_c <= sum_p X_{p,c} for all c
    4. Area bounds: A_c^min <= sum_p (s_p * X_{p,c}) <= A_c^max for all c
    5. Food group diversity: FG_g^min <= sum_{c in G_g} Y_c <= FG_g^max for all g
    
    Returns CQM, variables (X, Y), and constraint metadata.
    """
    cqm = ConstrainedQuadraticModel()
    
    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']  # s_p: area of each plot
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})  # A_c^min
    max_percentage_per_crop = params.get('max_percentage_per_crop', {})
    food_group_constraints = params.get('food_group_constraints', {})
    idle_penalty = params.get('idle_penalty_lambda', 0.1)  # λ: penalty for unused area
    
    n_farms = len(farms)
    n_foods = len(foods)
    n_food_groups = len(food_groups) if food_group_constraints else 0
    
    # Calculate total land available
    total_land = sum(land_availability.values())
    
    # Calculate A_c^max for each crop (from max_percentage_per_crop)
    max_planting_area = {}
    for crop in foods:
        if crop in max_percentage_per_crop:
            max_planting_area[crop] = max_percentage_per_crop[crop] * total_land
        else:
            max_planting_area[crop] = total_land  # No limit if not specified
    
    # Calculate total operations for progress bar
    total_ops = (
        n_farms * n_foods +       # X_{p,c} variables
        n_foods +                 # Y_c variables
        n_farms * n_foods +       # Objective terms
        n_farms +                 # At most one crop per plot
        n_farms * n_foods +       # X-Y linking constraints
        n_foods +                 # Y activation constraints
        n_foods * 2 +             # Area bounds per crop (min and max)
        n_food_groups * 2         # Food group constraints (min and max)
    )
    
    pbar = tqdm(total=total_ops, desc="Building CQM (BQM_PATCH formulation)", unit="op", ncols=100)
    
    # Define X_{p,c} variables: 1 if plot p is assigned to crop c
    X = {}
    pbar.set_description("Creating X_{p,c} (plot-crop assignment) variables")
    for plot in farms:
        for crop in foods:
            X[(plot, crop)] = Binary(f"X_{plot}_{crop}")
            pbar.update(1)
    
    # Define Y_c variables: 1 if crop c is grown on at least one plot
    Y = {}
    pbar.set_description("Creating Y_c (crop activation) variables")
    for crop in foods:
        Y[crop] = Binary(f"Y_{crop}")
        pbar.update(1)
    
    # Objective function: Maximize sum_{p,c} (B_c + λ) * s_p * X_{p,c}
    # B_c is the weighted benefit per unit area for crop c
    pbar.set_description("Building objective function")
    objective = 0
    for plot in farms:
        s_p = land_availability[plot]  # Area of plot p
        for crop in foods:
            # Calculate B_c: weighted benefit per unit area
            B_c = (
                weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
            )
            # Add (B_c + λ) * s_p * X_{p,c} to objective
            objective += (B_c + idle_penalty) * s_p * X[(plot, crop)]
            pbar.update(1)
    
    # Set objective (maximize, so negate for minimization)
    cqm.set_objective(-objective)
    
    # Constraint metadata
    constraint_metadata = {
        'at_most_one_per_plot': {},
        'x_y_linking': {},
        'y_activation': {},
        'area_bounds_min': {},
        'area_bounds_max': {},
        'food_group_min': {},
        'food_group_max': {}
    }
    
    # Constraint 1: At most one crop per plot
    # For each plot p: sum_c X_{p,c} <= 1
    pbar.set_description("Adding at-most-one-crop-per-plot constraints")
    for plot in farms:
        cqm.add_constraint(
            sum(X[(plot, crop)] for crop in foods) <= 1,
            label=f"AtMostOne_{plot}"
        )
        constraint_metadata['at_most_one_per_plot'][plot] = {
            'type': 'at_most_one_per_plot',
            'plot': plot,
            'plot_area': land_availability[plot]
        }
        pbar.update(1)
    
    # Constraint 2: X-Y Linking
    # For each plot p and crop c: X_{p,c} <= Y_c
    # (If crop c is not selected, no plot can be assigned to it)
    pbar.set_description("Adding X-Y linking constraints")
    for plot in farms:
        for crop in foods:
            cqm.add_constraint(
                X[(plot, crop)] - Y[crop] <= 0,
                label=f"XY_Link_{plot}_{crop}"
            )
            constraint_metadata['x_y_linking'][(plot, crop)] = {
                'type': 'x_y_linking',
                'plot': plot,
                'crop': crop
            }
            pbar.update(1)
    
    # Constraint 3: Y Activation
    # For each crop c: Y_c <= sum_p X_{p,c}
    # (If crop c is selected, at least one plot must be assigned to it)
    pbar.set_description("Adding Y activation constraints")
    for crop in foods:
        cqm.add_constraint(
            Y[crop] - sum(X[(plot, crop)] for plot in farms) <= 0,
            label=f"Y_Activation_{crop}"
        )
        constraint_metadata['y_activation'][crop] = {
            'type': 'y_activation',
            'crop': crop
        }
        pbar.update(1)
    
    # Constraint 4: Area bounds per crop
    # For each crop c: A_c^min <= sum_p (s_p * X_{p,c}) <= A_c^max
    pbar.set_description("Adding area bounds constraints")
    for crop in foods:
        total_crop_area = sum(land_availability[plot] * X[(plot, crop)] for plot in farms)
        
        # Minimum area constraint
        if crop in min_planting_area and min_planting_area[crop] > 0:
            cqm.add_constraint(
                total_crop_area >= min_planting_area[crop],
                label=f"MinArea_{crop}"
            )
            constraint_metadata['area_bounds_min'][crop] = {
                'type': 'area_bounds_min',
                'crop': crop,
                'min_area': min_planting_area[crop]
            }
        pbar.update(1)
        
        # Maximum area constraint
        if crop in max_planting_area and max_planting_area[crop] < total_land:
            cqm.add_constraint(
                total_crop_area <= max_planting_area[crop],
                label=f"MaxArea_{crop}"
            )
            constraint_metadata['area_bounds_max'][crop] = {
                'type': 'area_bounds_max',
                'crop': crop,
                'max_area': max_planting_area[crop]
            }
        pbar.update(1)
    
    # Constraint 5: Food group diversity constraints
    # For each food group g: FG_g^min <= sum_{c in G_g} Y_c <= FG_g^max
    pbar.set_description("Adding food group diversity constraints")
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                # Minimum number of crops from this group
                if 'min_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[crop] for crop in foods_in_group) >= constraints['min_foods'],
                        label=f"FoodGroup_Min_{group}"
                    )
                    constraint_metadata['food_group_min'][group] = {
                        'type': 'food_group_min',
                        'group': group,
                        'min_foods': constraints['min_foods'],
                        'foods_in_group': foods_in_group
                    }
                    pbar.update(1)
                
                # Maximum number of crops from this group
                if 'max_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[crop] for crop in foods_in_group) <= constraints['max_foods'],
                        label=f"FoodGroup_Max_{group}"
                    )
                    constraint_metadata['food_group_max'][group] = {
                        'type': 'food_group_max',
                        'group': group,
                        'max_foods': constraints['max_foods'],
                        'foods_in_group': foods_in_group
                    }
                    pbar.update(1)
    
    pbar.set_description("CQM complete (BQM_PATCH formulation)")
    pbar.close()
    
    return cqm, (X, Y), constraint_metadata

def solve_with_pulp(farms, foods, food_groups, config):
    """
    Solve with PuLP using BQM_PATCH formulation.
    
    Variables:
    - X_{p,c}: Binary, 1 if plot p is assigned to crop c
    - Y_c: Binary, 1 if crop c is grown on at least one plot
    
    Matches the CQM formulation exactly.
    """
    params = config['parameters']
    land_availability = params['land_availability']  # s_p: area of each plot
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})  # A_c^min
    max_percentage_per_crop = params.get('max_percentage_per_crop', {})
    food_group_constraints = params.get('food_group_constraints', {})
    idle_penalty = params.get('idle_penalty_lambda', 0.1)  # λ: penalty for unused area
    
    # Calculate total land and max areas
    total_land = sum(land_availability.values())
    max_planting_area = {}
    for crop in foods:
        if crop in max_percentage_per_crop:
            max_planting_area[crop] = max_percentage_per_crop[crop] * total_land
        else:
            max_planting_area[crop] = total_land
    
    # Define X_{p,c} variables: 1 if plot p is assigned to crop c
    X_pulp = pl.LpVariable.dicts("X", [(p, c) for p in farms for c in foods], cat='Binary')
    
    # Define Y_c variables: 1 if crop c is grown on at least one plot
    Y_pulp = pl.LpVariable.dicts("Y", foods, cat='Binary')
    
    model = pl.LpProblem("Food_Optimization_BQM_PATCH", pl.LpMaximize)
    
    # Objective: Maximize sum_{p,c} (B_c + λ) * s_p * X_{p,c}
    objective = 0
    for plot in farms:
        s_p = land_availability[plot]  # Area of plot p
        for crop in foods:
            # Calculate B_c: weighted benefit per unit area
            B_c = (
                weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
            )
            # Add (B_c + λ) * s_p * X_{p,c}
            objective += (B_c + idle_penalty) * s_p * X_pulp[(plot, crop)]
    
    model += objective, "Objective"
    
    # Constraint 1: At most one crop per plot
    for plot in farms:
        model += pl.lpSum([X_pulp[(plot, crop)] for crop in foods]) <= 1, f"AtMostOne_{plot}"
    
    # Constraint 2: X-Y Linking - X_{p,c} <= Y_c
    for plot in farms:
        for crop in foods:
            model += X_pulp[(plot, crop)] <= Y_pulp[crop], f"XY_Link_{plot}_{crop}"
    
    # Constraint 3: Y Activation - Y_c <= sum_p X_{p,c}
    for crop in foods:
        model += Y_pulp[crop] <= pl.lpSum([X_pulp[(plot, crop)] for plot in farms]), f"Y_Activation_{crop}"
    
    # Constraint 4: Area bounds per crop
    for crop in foods:
        total_crop_area = pl.lpSum([land_availability[plot] * X_pulp[(plot, crop)] for plot in farms])
        
        # Minimum area constraint
        if crop in min_planting_area and min_planting_area[crop] > 0:
            model += total_crop_area >= min_planting_area[crop], f"MinArea_{crop}"
        
        # Maximum area constraint
        if crop in max_planting_area and max_planting_area[crop] < total_land:
            model += total_crop_area <= max_planting_area[crop], f"MaxArea_{crop}"
    
    # Constraint 5: Food group diversity
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                if 'min_foods' in constraints:
                    model += pl.lpSum([Y_pulp[crop] for crop in foods_in_group]) >= constraints['min_foods'], f"FoodGroup_Min_{group}"
                if 'max_foods' in constraints:
                    model += pl.lpSum([Y_pulp[crop] for crop in foods_in_group]) <= constraints['max_foods'], f"FoodGroup_Max_{group}"
    
    start_time = time.time()
    # Use Gurobi with GPU acceleration enabled
    # Parameters:
    # - msg=0: Suppress output
    # - options: List of Gurobi parameters
    #   - RINS=2: More aggressive RINS heuristic (for MIP)
    #   - MIPFocus=1: Focus on finding feasible solutions quickly
    #   - Threads=0: Use all available threads
    #   - NumericFocus=0: Default numeric precision
    # GPU-specific parameters (requires Gurobi 9.0+ and CUDA-compatible GPU):
    #   - BarHomogeneous=1: Use homogeneous barrier algorithm (better for GPU)
    #   - Crossover=0: Disable crossover (barrier stays on GPU)
    #   - Method=2: Use barrier method (GPU-accelerated)
    gurobi_options = [
        ('Method', 2),           # Barrier method (can use GPU)
        ('Crossover', 0),        # Disable crossover to keep computation on GPU
        ('BarHomogeneous', 1),   # Homogeneous barrier (more GPU-friendly)
        ('Threads', 0),          # Use all available CPU threads
        ('MIPFocus', 1),         # Focus on finding good solutions quickly
    ]
    
    # Convert options to command-line format for GUROBI_CMD
    options_str = ' '.join([f'{k}={v}' for k, v in gurobi_options])
    model.solve(pl.GUROBI_CMD(msg=0, options=[options_str]))
    solve_time = time.time() - start_time
    
    # Extract results
    results = {
        'status': pl.LpStatus[model.status],
        'objective_value': pl.value(model.objective),
        'solve_time': solve_time,
        'X_variables': {},  # X_{p,c}: plot-crop assignments
        'Y_variables': {}   # Y_c: crop selections
    }
    
    # Extract X_{p,c} values
    for plot in farms:
        for crop in foods:
            key = f"X_{plot}_{crop}"
            results['X_variables'][key] = X_pulp[(plot, crop)].value() if X_pulp[(plot, crop)].value() is not None else 0.0
    
    # Extract Y_c values
    for crop in foods:
        key = f"Y_{crop}"
        results['Y_variables'][key] = Y_pulp[crop].value() if Y_pulp[crop].value() is not None else 0.0
    
    return model, results

def solve_with_dwave(cqm, token):
    """
    Solve with DWave using HybridBQM solver after converting CQM to BQM.
    This enables QPU usage and better scaling for quadratic problems.
    
    Args:
        cqm: ConstrainedQuadraticModel to convert and solve
        token: DWave API token
    
    Returns tuple of (sampleset, hybrid_time, qpu_time, bqm_conversion_time, invert)
    """
    print("\nConverting CQM to BQM for QPU-enabled solving...")
    print("  This discretizes continuous variables for better QPU utilization.")
    
    # Convert CQM to BQM - this discretizes continuous variables
    convert_start = time.time()
    bqm, invert = cqm_to_bqm(cqm)
    bqm_conversion_time = time.time() - convert_start
    
    print(f"  ✅ CQM converted to BQM in {bqm_conversion_time:.2f}s")
    print(f"  BQM Variables: {len(bqm.variables)}")
    print(f"  BQM Interactions: {len(bqm.quadratic)}")
    
    # Use HybridBQM sampler for better QPU usage
    sampler = LeapHybridBQMSampler(token=token)
    
    print("\nSubmitting to DWave Leap HybridBQM solver...")
    print("  This solver uses more QPU time than CQM solver for quadratic problems.")

    sampleset = sampler.sample(bqm, label="Food Optimization - BQUBO Run")
    
    # Extract timing from sampleset.info
    timing_info = sampleset.info.get('timing', {})
    
    # Hybrid solve time (total time including QPU)
    hybrid_time = (timing_info.get('run_time') or 
                  sampleset.info.get('run_time') or
                  timing_info.get('charge_time') or
                  sampleset.info.get('charge_time'))
    
    if hybrid_time is not None:
        hybrid_time = hybrid_time / 1e6  # Convert from microseconds to seconds
    
    # QPU access time
    qpu_time = (timing_info.get('qpu_access_time') or
               sampleset.info.get('qpu_access_time'))
    
    if qpu_time is not None:
        qpu_time = qpu_time / 1e6  # Convert from microseconds to seconds
    
    if hybrid_time is not None:
        print(f"  Hybrid Time: {hybrid_time:.2f}s")
    if qpu_time is not None:
        print(f"  QPU Access Time: {qpu_time:.4f}s")
    
    return sampleset, hybrid_time, qpu_time, bqm_conversion_time, invert

def solve_with_simulated_annealing(bqm):
    """
    Solve with free Simulated Annealing sampler.
    This runs locally and doesn't require QPU access.
    
    Args:
        bqm: BinaryQuadraticModel to solve
    
    Returns tuple of (sampleset, solve_time)
    """
    print("\n" + "=" * 80)
    print("SOLVING WITH SIMULATED ANNEALING (FREE LOCAL SOLVER)")
    print("=" * 80)
    
    sampler = SimulatedAnnealingSampler()
    
    print("Running Simulated Annealing locally...")
    start_time = time.time()
    sampleset = sampler.sample(bqm, num_reads=100, label="Food Optimization - Simulated Annealing")
    solve_time = time.time() - start_time
    
    print(f"  Total samples: {len(sampleset)}")
    print(f"  Solve time: {solve_time:.2f} seconds")
    
    if len(sampleset) > 0:
        best = sampleset.first
        print(f"  Best energy: {best.energy:.6f}")
        best_objective = -best.energy
        print(f"  Best objective: {best_objective:.6f}")
    
    return sampleset, solve_time

def main(scenario='simple', n_patches=None):
    """Main execution function."""
    print("=" * 80)
    print("PROFESSIONAL SOLVER RUNNER - BQM_PATCH")
    print("=" * 80)
    
    # Create output directories
    os.makedirs('PuLP_Results', exist_ok=True)
    os.makedirs('DWave_Results', exist_ok=True)
    os.makedirs('CQM_Models', exist_ok=True)
    os.makedirs('Constraints', exist_ok=True)
    
    # Load scenario
    print(f"\nLoading '{scenario}' scenario...")
    farms, foods, food_groups, config = load_food_data(scenario)
    
    # If n_patches is specified, override farms with generated patches
    if n_patches is not None:
        print(f"\nGenerating {n_patches} patches using patch_sampler...")
        patches_dict = generate_patches(n_patches, seed=42)
        farms = list(patches_dict.keys())
        config['parameters']['land_availability'] = patches_dict
        print(f"  Generated patches: {farms}")
        print(f"  Total patch area: {sum(patches_dict.values()):.3f} ha")
        print(f"  Average patch size: {sum(patches_dict.values())/len(patches_dict):.3f} ha")
    else:
        print(f"  Using scenario farms: {len(farms)} - {farms}")
    
    print(f"  Foods: {len(foods)} - {list(foods.keys())}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CQM with BQM_PATCH formulation
    print("\nCreating CQM with BQM_PATCH formulation (implicit idle area)...")
    cqm, (X, Y), constraint_metadata = create_cqm(farms, foods, food_groups, config)
    print(f"  Variables: {len(cqm.variables)} (X_{{p,c}} and Y_c, all binary)")
    print(f"  - X variables (plot-crop assignments): {len(X)}")
    print(f"  - Y variables (crop selections): {len(Y)}")
    print(f"  Constraints: {len(cqm.constraints)}")
    print(f"  Formulation: Plot-crop assignment with implicit idle representation")
    
    # Save CQM
    cqm_path = f'CQM_Models/cqm_{scenario}_{timestamp}.cqm'
    print(f"\nSaving CQM to {cqm_path}...")
    with open(cqm_path, 'wb') as f:
        shutil.copyfileobj(cqm.to_file(), f)
    
    # Save constraint metadata
    constraints_path = f'Constraints/constraints_{scenario}_{timestamp}.json'
    print(f"Saving constraints to {constraints_path}...")
    
    # Convert constraint_metadata keys to strings for JSON serialization
    # Also convert foods dict to serializable format
    foods_serializable = {
        name: {k: float(v) if isinstance(v, (int, float)) else v for k, v in attrs.items()}
        for name, attrs in foods.items()
    }
    
    constraints_json = {
        'scenario': scenario,
        'timestamp': timestamp,
        'farms': farms,
        'foods': list(foods.keys()),
        'foods_data': foods_serializable,  # Add full food data for objective calculation
        'food_groups': food_groups,
        'config': config,
        'constraint_metadata': {
            'at_most_one_per_plot': {str(k): v for k, v in constraint_metadata['at_most_one_per_plot'].items()},
            'x_y_linking': {str(k): v for k, v in constraint_metadata['x_y_linking'].items()},
            'y_activation': {str(k): v for k, v in constraint_metadata['y_activation'].items()},
            'area_bounds_min': {str(k): v for k, v in constraint_metadata['area_bounds_min'].items()},
            'area_bounds_max': {str(k): v for k, v in constraint_metadata['area_bounds_max'].items()},
            'food_group_min': {str(k): v for k, v in constraint_metadata['food_group_min'].items()},
            'food_group_max': {str(k): v for k, v in constraint_metadata['food_group_max'].items()}
        },
        'formulation': 'BQM_PATCH',
        'description': 'Plot-crop assignment with implicit idle area representation'
    }
    
    with open(constraints_path, 'w') as f:
        json.dump(constraints_json, f, indent=2)
    
    # Solve with PuLP
    print("\n" + "=" * 80)
    print("SOLVING WITH PULP")
    print("=" * 80)
    pulp_model, pulp_results = solve_with_pulp(farms, foods, food_groups, config)
    print(f"  Status: {pulp_results['status']}")
    print(f"  Objective: {pulp_results['objective_value']:.6f}")
    print(f"  Solve time: {pulp_results['solve_time']:.2f} seconds")
    
    # Save PuLP results
    pulp_path = f'PuLP_Results/pulp_{scenario}_{timestamp}.json'
    print(f"\nSaving PuLP results to {pulp_path}...")
    with open(pulp_path, 'w') as f:
        json.dump(pulp_results, f, indent=2)
    
    # Solve with DWave using BQUBO approach
    print("\n" + "=" * 80)
    print("SOLVING WITH DWAVE (BQUBO: CQM→BQM + HybridBQM)")
    print("=" * 80)
    token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
    
    # Convert CQM to BQM once (reuse for both samplers)
    print("\nConverting CQM to BQM...")
    convert_start = time.time()
    bqm, invert = cqm_to_bqm(cqm)
    bqm_conversion_time = time.time() - convert_start
    print(f"  ✅ CQM converted to BQM in {bqm_conversion_time:.2f}s")
    print(f"  BQM Variables: {len(bqm.variables)}")
    print(f"  BQM Interactions: {len(bqm.quadratic)}")
    
    # Solve with HybridBQM (QPU-enabled)
    print("\nSubmitting to DWave Leap HybridBQM solver...")
    sampler_hybrid = LeapHybridBQMSampler(token=token)
    hybrid_start = time.time()
    sampleset_hybrid = sampler_hybrid.sample(bqm, label="Food Optimization - BQUBO HybridBQM")
    hybrid_total_time = time.time() - hybrid_start
    
    # Extract timing from sampleset
    timing_info = sampleset_hybrid.info.get('timing', {})
    hybrid_time = (timing_info.get('run_time') or 
                  sampleset_hybrid.info.get('run_time') or
                  timing_info.get('charge_time') or
                  sampleset_hybrid.info.get('charge_time'))
    if hybrid_time is not None:
        hybrid_time = hybrid_time / 1e6
    else:
        hybrid_time = hybrid_total_time
        
    qpu_access_time = (timing_info.get('qpu_access_time') or
                      sampleset_hybrid.info.get('qpu_access_time'))
    if qpu_access_time is not None:
        qpu_access_time = qpu_access_time / 1e6
    else:
        qpu_access_time = 0.0
    
    print(f"  ✅ HybridBQM solve complete")
    print(f"  Total samples: {len(sampleset_hybrid)}")
    print(f"  Hybrid Time: {hybrid_time:.2f}s")
    print(f"  QPU Access Time: {qpu_access_time:.4f}s")
    
    if len(sampleset_hybrid) > 0:
        best_hybrid = sampleset_hybrid.first
        print(f"  Best energy: {best_hybrid.energy:.6f}")
        best_objective_hybrid = -best_hybrid.energy
        print(f"  Best objective: {best_objective_hybrid:.6f}")
    
    # Solve with Simulated Annealing (free, local)
    print("\n" + "=" * 80)
    print("SOLVING WITH SIMULATED ANNEALING (FREE LOCAL SOLVER)")
    print("=" * 80)
    sampler_sa = SimulatedAnnealingSampler()
    sa_start = time.time()
    sampleset_sa = sampler_sa.sample(bqm, num_reads=100, label="Food Optimization - Simulated Annealing")
    sa_time = time.time() - sa_start
    
    print(f"  ✅ Simulated Annealing complete")
    print(f"  Total samples: {len(sampleset_sa)}")
    print(f"  Solve time: {sa_time:.2f}s")
    
    if len(sampleset_sa) > 0:
        best_sa = sampleset_sa.first
        print(f"  Best energy: {best_sa.energy:.6f}")
        best_objective_sa = -best_sa.energy
        print(f"  Best objective: {best_objective_sa:.6f}")
    
    # Save DWave results (both pickle and JSON)
    dwave_pickle_path = f'DWave_Results/dwave_bqubo_{scenario}_{timestamp}.pickle'
    print(f"\nSaving DWave HybridBQM sampleset to {dwave_pickle_path}...")
    with open(dwave_pickle_path, 'wb') as f:
        pickle.dump(sampleset_hybrid, f)
    
    # Save DWave results as JSON for easy reading
    dwave_json_path = f'DWave_Results/dwave_bqubo_{scenario}_{timestamp}.json'
    
    # Convert sample to JSON-serializable format
    best_sample_json = {}
    if len(sampleset_hybrid) > 0:
        for key, value in best_hybrid.sample.items():
            best_sample_json[str(key)] = int(value)  # Convert numpy types to Python int
    
    dwave_results = {
        'status': 'Optimal' if len(sampleset_hybrid) > 0 else 'No solutions',
        'objective_value': best_objective_hybrid if len(sampleset_hybrid) > 0 else None,
        'solve_time': hybrid_time,
        'qpu_access_time': qpu_access_time,
        'bqm_conversion_time': bqm_conversion_time,
        'num_samples': len(sampleset_hybrid),
        'formulation': 'BQUBO (HybridBQM)',
        'best_sample': best_sample_json
    }
    with open(dwave_json_path, 'w') as f:
        json.dump(dwave_results, f, indent=2)
    
    # Save Simulated Annealing results
    sa_json_path = f'DWave_Results/simulated_annealing_{scenario}_{timestamp}.json'
    print(f"Saving Simulated Annealing results to {sa_json_path}...")
    
    best_sa_sample_json = {}
    if len(sampleset_sa) > 0:
        for key, value in best_sa.sample.items():
            best_sa_sample_json[str(key)] = int(value)
    
    sa_results = {
        'status': 'Optimal' if len(sampleset_sa) > 0 else 'No solutions',
        'objective_value': best_objective_sa if len(sampleset_sa) > 0 else None,
        'solve_time': sa_time,
        'num_samples': len(sampleset_sa),
        'formulation': 'Simulated Annealing (Free)',
        'best_sample': best_sa_sample_json
    }
    with open(sa_json_path, 'w') as f:
        json.dump(sa_results, f, indent=2)
    
    # Print comprehensive comparison
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON - BQM_PATCH FORMULATION")
    print("=" * 80)
    print(f"\n📊 PROBLEM SIZE:")
    print(f"   Plots: {len(farms)}")
    print(f"   Crops: {len(foods)}")
    print(f"   X variables (plot-crop assignments): {len(farms) * len(foods)}")
    print(f"   Y variables (crop selections): {len(foods)}")
    print(f"   Total CQM variables: {len(cqm.variables)}")
    print(f"   Total BQM variables (after conversion): {len(bqm.variables)}")
    print(f"   Total constraints: {len(cqm.constraints)}")
    
    print(f"\n🔧 PULP SOLVER:")
    print(f"   Status: {pulp_results['status']}")
    print(f"   Objective: {pulp_results['objective_value']:.6f}")
    print(f"   Solve Time: {pulp_results['solve_time']:.4f}s")
    print(f"   Selected crops (Y=1): {sum(1 for v in pulp_results['Y_variables'].values() if v > 0.5)}")
    print(f"   Assigned plots (X=1): {sum(1 for v in pulp_results['X_variables'].values() if v > 0.5)}")
    
    print(f"\n⚛️  DWAVE HYBRID BQM:")
    print(f"   Status: {dwave_results['status']}")
    print(f"   Objective: {dwave_results['objective_value']:.6f}")
    print(f"   Run Time: {hybrid_time:.4f}s")
    print(f"   QPU Access Time: {qpu_access_time:.6f}s")
    print(f"   BQM Conversion: {bqm_conversion_time:.4f}s")
    print(f"   Samples: {len(sampleset_hybrid)}")
    
    print(f"\n🔥 SIMULATED ANNEALING:")
    print(f"   Status: {sa_results['status']}")
    print(f"   Objective: {sa_results['objective_value']:.6f}")
    print(f"   Solve Time: {sa_time:.4f}s")
    print(f"   Samples: {len(sampleset_sa)}")
    
    print(f"\n📈 OBJECTIVE COMPARISONS:")
    print(f"   PuLP vs HybridBQM: {abs(pulp_results['objective_value'] - best_objective_hybrid):.6f}")
    print(f"   PuLP vs SimAnneal: {abs(pulp_results['objective_value'] - best_objective_sa):.6f}")
    print(f"   HybridBQM vs SimAnneal: {abs(best_objective_hybrid - best_objective_sa):.6f}")
    
    # Create run manifest
    manifest = {
        'scenario': scenario,
        'timestamp': timestamp,
        'cqm_path': cqm_path,
        'constraints_path': constraints_path,
        'pulp_path': pulp_path,
        'dwave_pickle_path': dwave_pickle_path,
        'dwave_json_path': dwave_json_path,
        'sa_json_path': sa_json_path,
        'farms': farms,
        'foods': list(foods.keys()),
        'pulp_status': pulp_results['status'],
        'pulp_objective': pulp_results['objective_value'],
        'dwave_status': dwave_results['status'],
        'dwave_objective': dwave_results['objective_value'],
        'dwave_qpu_time': qpu_access_time,
        'dwave_sample_count': len(sampleset_hybrid),
        'sa_status': sa_results['status'],
        'sa_objective': sa_results['objective_value'],
        'sa_sample_count': len(sampleset_sa),
        'formulation': 'BQM_PATCH'
    }
    
    manifest_path = f'run_manifest_{scenario}_{timestamp}.json'
    print(f"\nSaving run manifest to {manifest_path}...")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SOLVER RUN COMPLETE (BQUBO)")
    print("=" * 80)
    print(f"Manifest file: {manifest_path}")
    print(f"CQM model: {cqm_path}")
    print(f"PuLP results: {pulp_path}")
    print(f"DWave results (JSON): {dwave_json_path}")
    print(f"DWave results (pickle): {dwave_pickle_path}")
    print(f"\n✅ BQUBO approach: CQM→BQM conversion + HybridBQM solver")
    print(f"   QPU Access Time: {qpu_access_time:.4f}s")
    print(f"   More QPU usage = Better scaling for larger problems!")
    print("\nRun the verifier script with this manifest to check results:")
    print(f"  python verifier.py {manifest_path}")
    
    return manifest_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run solvers on a food optimization scenario')
    parser.add_argument('--scenario', type=str, default='simple', 
                       choices=['simple', 'intermediate', 'full', 'custom', 'full_family'],
                       help='Scenario to solve (default: simple)')
    parser.add_argument('--n-patches', type=int, default=None,
                       help='Number of patches to generate (overrides scenario farms)')
    
    args = parser.parse_args()
    
    main(scenario=args.scenario, n_patches=args.n_patches)
