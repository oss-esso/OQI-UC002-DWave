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
import patch_sampler
import farm_sampler
from dimod import ConstrainedQuadraticModel, Binary, Real, cqm_to_bqm
from dwave.system import LeapHybridCQMSampler, LeapHybridBQMSampler
import pulp as pl
from tqdm import tqdm

def create_cqm(farms, foods, food_groups, config):
    """
    Creates a CQM for the BINARY food optimization problem.
    
    This function supports two land representation methods:
    1. Even Grid: land_availability[farm] represents area per patch (all equal)
    2. Uneven Distribution: land_availability[farm] represents number of 1-acre plots
    
    For binary formulation, each variable represents a 1-acre plantation.
    
    Args:
        farms: List of farm/patch names
        foods: Dictionary of food data
        food_groups: Dictionary of food group mappings  
        config: Configuration including land_availability and generation method
    
    Returns:
        Tuple of (CQM, variables_dict, constraint_metadata)
    """
    cqm = ConstrainedQuadraticModel()
    
    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    n_farms = len(farms)
    n_foods = len(foods)
    n_food_groups = len(food_groups) if food_group_constraints else 0
    
    # Calculate total operations for progress bar
    total_ops = (
        n_farms * n_foods +       # Binary variables only (Y)
        n_farms * n_foods +       # Objective terms
        n_farms +                 # Land availability constraints
        n_farms * n_food_groups * 2  # Food group constraints (min and max)
    )
    
    pbar = tqdm(total=total_ops, desc="Building CQM (Binary formulation)", unit="op", ncols=100)
    
    # Define variables - ONLY BINARY (each plantation is 1 acre if selected, 0 otherwise)
    Y = {}
    
    pbar.set_description("Creating binary variables")
    for farm in farms:
        for food in foods:
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")
            pbar.update(1)
    
    # Objective function - each selected crop contributes its area-weighted value
    pbar.set_description("Building objective")
    total_land_area = sum(land_availability.values())  # Normalization factor
    
    objective = 0
    for farm in farms:
        farm_area = land_availability[farm]  # Area of this farm/plot
        for food in foods:
            # Y is binary: 1 if crop assigned to this unit, 0 if not
            # Each assignment contributes its area-weighted value
            area_weighted_value = farm_area * (
                weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) +
                weights.get('affordability', 0) * foods[food].get('affordability', 0) +
                weights.get('sustainability', 0) * foods[food].get('sustainability', 0)
            )
            objective += area_weighted_value * Y[(farm, food)]
            pbar.update(1)
    
    # Normalize by total land area to make objectives comparable
    objective = objective / total_land_area
    cqm.set_objective(-objective)
    
    # Constraint metadata
    constraint_metadata = {
        'plantation_limit': {},
        'food_group_min': {},
        'food_group_max': {}
    }
    
    # Land unit constraints - each unit can be assigned to at most one crop
    # The area of each unit affects the objective contribution
    pbar.set_description("Adding land unit constraints")
    
    for farm in farms:
        # Each unit can have exactly one crop assigned (or remain idle)
        cqm.add_constraint(
            sum(Y[(farm, food)] for food in foods) <= 1,
            label=f"Max_Assignment_{farm}"
        )
        constraint_metadata['plantation_limit'][farm] = {
            'type': 'land_unit_assignment',
            'farm': farm,
            'area_ha': land_availability[farm]
        }
        pbar.update(1)
    
    # Food group constraints
    pbar.set_description("Adding food group constraints")
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                for farm in farms:
                    if 'min_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(farm, food)] for food in foods_in_group) - constraints['min_foods'] >= 0,
                            label=f"Food_Group_Min_{group}_{farm}"
                        )
                        constraint_metadata['food_group_min'][(group, farm)] = {
                            'type': 'food_group_min',
                            'group': group,
                            'farm': farm,
                            'min_foods': constraints['min_foods'],
                            'foods_in_group': foods_in_group
                        }
                        pbar.update(1)
                    
                    if 'max_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(farm, food)] for food in foods_in_group) - constraints['max_foods'] <= 0,
                            label=f"Food_Group_Max_{group}_{farm}"
                        )
                        constraint_metadata['food_group_max'][(group, farm)] = {
                            'type': 'food_group_max',
                            'group': group,
                            'farm': farm,
                            'max_foods': constraints['max_foods'],
                            'foods_in_group': foods_in_group
                        }
                        pbar.update(1)
    
    pbar.set_description("CQM complete")
    pbar.close()
    
    return cqm, Y, constraint_metadata

def solve_with_pulp(farms, foods, food_groups, config):
    """
    Solve with PuLP using BINARY formulation.
    
    Supports both land generation methods:
    - even_grid: land_availability represents area per plot
    - uneven_distribution: land_availability represents area per farm
    """
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    food_group_constraints = params.get('food_group_constraints', {})
    land_method = params.get('land_generation_method', 'even_grid')
    
    # Binary variables - each represents assignment of a crop to a land unit
    Y_pulp = pl.LpVariable.dicts("Assignment", [(f, c) for f in farms for c in foods], cat='Binary')
    
    total_land_area = sum(land_availability.values())
    
    # Objective: maximize area-weighted value of crop assignments
    goal = 0
    for f in farms:
        farm_area = land_availability[f]
        for c in foods:
            area_weighted_value = farm_area * (
                weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
                weights.get('affordability', 0) * foods[c].get('affordability', 0) +
                weights.get('sustainability', 0) * foods[c].get('sustainability', 0)
            )
            goal += area_weighted_value * Y_pulp[(f, c)]
    
    # Normalize by total land area
    goal = goal / total_land_area
    
    model = pl.LpProblem("Food_Optimization_Binary", pl.LpMaximize)
    
    # Land unit assignment: each unit can be assigned to at most one crop
    # The area of each unit affects the objective contribution
    for f in farms:
        # Each unit can have exactly one crop assigned (or remain idle)
        model += pl.lpSum([Y_pulp[(f, c)] for c in foods]) <= 1, f"Max_Assignment_{f}"
    
    # Food group constraints
    if food_group_constraints:
        for g, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(g, [])
            if foods_in_group:
                for f in farms:
                    if 'min_foods' in constraints:
                        model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]) >= constraints['min_foods'], f"MinFoodGroup_{f}_{g}"
                    if 'max_foods' in constraints:
                        model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]) <= constraints['max_foods'], f"MaxFoodGroup_{f}_{g}"
    
    model += goal, "Objective"
    
    start_time = time.time()
    model.solve(pl.PULP_CBC_CMD(msg=0))
    solve_time = time.time() - start_time
    
    # Extract results
    results = {
        'status': pl.LpStatus[model.status],
        'objective_value': pl.value(model.objective),
        'solve_time': solve_time,
        'plantations': {}  # Binary: 1 if planted, 0 if not
    }
    
    for f in farms:
        for c in foods:
            key = f"{f}_{c}"
            results['plantations'][key] = Y_pulp[(f, c)].value() if Y_pulp[(f, c)].value() is not None else 0.0
    
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

def main(scenario='simple', land_method='even_grid', n_units=None, total_land=100.0):
    """
    Main execution function.
    
    Args:
        scenario: Scenario type ('simple', 'intermediate', etc.)
        land_method: Land generation method ('even_grid' or 'uneven_distribution')
        n_units: Number of units to generate (if None, use scenario default)
        total_land: Total land area in hectares (default: 100.0)
    """
    print("=" * 80)
    print("PROFESSIONAL SOLVER RUNNER - BINARY FORMULATION")
    print("=" * 80)
    print(f"Land Generation Method: {land_method}")
    print(f"Total Land Area: {total_land} ha")
    
    # Create output directories
    os.makedirs('PuLP_Results', exist_ok=True)
    os.makedirs('DWave_Results', exist_ok=True)
    os.makedirs('CQM_Models', exist_ok=True)
    os.makedirs('Constraints', exist_ok=True)
    
    # Load scenario data first to get foods and food groups
    print(f"\nLoading food data for '{scenario}' scenario...")
    _, foods, food_groups, base_config = load_food_data(scenario)
    print(f"  Foods: {len(foods)} - {list(foods.keys())}")
    print(f"  Food Groups: {len(food_groups)} - {list(food_groups.keys())}")
    
    # Generate land data based on method
    print(f"\nGenerating land data using '{land_method}' method...")
    if land_method == 'even_grid':
        # Even grid: all patches have equal size
        n_patches = n_units or 25  # Default to 25 patches if not specified
        land_availability = patch_sampler.generate_grid(
            n_farms=n_patches, 
            area=total_land, 
            seed=42
        )
        farms = list(land_availability.keys())
        print(f"  ✅ Even Grid: {len(farms)} patches of {total_land/len(farms):.3f} ha each")
        print(f"  Formulation: Binary variables X_{{p,c}} ∈ {{0,1}} (discrete optimization)")
        
    elif land_method == 'uneven_distribution':
        # Uneven distribution: farms have realistic size distribution
        n_farms = n_units or 15  # Default to 15 farms if not specified  
        land_availability = farm_sampler.generate_farms(n_farms=n_farms, seed=42)
        
        # Scale to match total_land
        current_total = sum(land_availability.values())
        scale_factor = total_land / current_total
        land_availability = {k: v * scale_factor for k, v in land_availability.items()}
        
        farms = list(land_availability.keys())
        areas = list(land_availability.values())
        print(f"  ✅ Uneven Distribution: {len(farms)} farms")
        print(f"  Farm sizes: min={min(areas):.2f} ha, max={max(areas):.2f} ha, avg={sum(areas)/len(areas):.2f} ha")
        print(f"  Formulation: Continuous area variables A_{{f,c}} ∈ [0, farm_area] (same as solver_runner.py)")
        
    else:
        raise ValueError(f"Unknown land_method: {land_method}. Use 'even_grid' or 'uneven_distribution'")
    
    # Update configuration with generated land data
    config = {
        'parameters': {
            'land_availability': land_availability,
            'minimum_planting_area': base_config['parameters'].get('minimum_planting_area', {}),
            'food_group_constraints': base_config['parameters'].get('food_group_constraints', {}),
            'weights': base_config['parameters'].get('weights', {}),
            'land_generation_method': land_method,
            'total_land_ha': total_land,
            'n_units': len(farms)
        }
    }
    
    print(f"  Total Land: {sum(land_availability.values()):.2f} ha")
    print(f"  Units (farms/plots): {len(farms)}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Choose formulation based on land method
    if land_method == 'even_grid':
        # Binary formulation for even grid
        print(f"\nCreating CQM with BINARY formulation (even grid)...")
        print(f"  Land method: {land_method}")
        print(f"  Total plots: {len(farms)}")
        cqm, Y, constraint_metadata = create_cqm(farms, foods, food_groups, config)
        print(f"  Variables: {len(cqm.variables)} (all binary)")
        print(f"  Constraints: {len(cqm.constraints)}")
        
        # Save CQM for binary formulation
        cqm_path = f'CQM_Models/cqm_binary_{scenario}_{timestamp}.cqm'
        print(f"\nSaving Binary CQM to {cqm_path}...")
        with open(cqm_path, 'wb') as f:
            shutil.copyfileobj(cqm.to_file(), f)
            
    elif land_method == 'uneven_distribution':
        # Continuous formulation for uneven farms (same as solver_runner.py)
        print(f"\nUsing CONTINUOUS formulation (uneven farms)...")
        print(f"  Land method: {land_method}")
        print(f"  Total farms: {len(farms)}")
        print(f"  Formulation: Continuous area variables (same as solver_runner.py)")
        
        # Import continuous solver functions
        from solver_runner import create_cqm as create_cqm_continuous
        
        cqm, Y, constraint_metadata = create_cqm_continuous(farms, foods, food_groups, config)
        print(f"  Variables: {len(cqm.variables)} (continuous)")
        print(f"  Constraints: {len(cqm.constraints)}")
        
        # Save CQM for continuous formulation
        cqm_path = f'CQM_Models/cqm_continuous_{scenario}_{timestamp}.cqm'
        print(f"\nSaving Continuous CQM to {cqm_path}...")
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
        'land_generation_method': land_method,
        'total_land_ha': total_land,
        'n_units': len(farms),
        'farms': farms,
        'foods': list(foods.keys()),
        'foods_data': foods_serializable,  # Add full food data for objective calculation
        'food_groups': food_groups,
        'config': config,
        'constraint_metadata': {
            'plantation_limit': {str(k): v for k, v in constraint_metadata['plantation_limit'].items()},
            'food_group_min': {str(k): v for k, v in constraint_metadata['food_group_min'].items()},
            'food_group_max': {str(k): v for k, v in constraint_metadata['food_group_max'].items()}
        },
        'formulation': 'binary' if land_method == 'even_grid' else 'continuous',
        'variable_type': 'X_{p,c} ∈ {0,1}' if land_method == 'even_grid' else 'A_{f,c} ∈ [0, farm_area]'
    }
    
    with open(constraints_path, 'w') as f:
        json.dump(constraints_json, f, indent=2)
    
    # Solve with PuLP
    print("\n" + "=" * 80)
    if land_method == 'even_grid':
        print("SOLVING WITH PULP (BINARY FORMULATION)")
        print("=" * 80)
        pulp_model, pulp_results = solve_with_pulp(farms, foods, food_groups, config)
        formulation_type = "binary"
    else:
        print("SOLVING WITH PULP (CONTINUOUS FORMULATION)")  
        print("=" * 80)
        # Import and use continuous solver
        from solver_runner import solve_with_pulp as solve_with_pulp_continuous
        pulp_model, pulp_results = solve_with_pulp_continuous(farms, foods, food_groups, config)
        formulation_type = "continuous"
        
    print(f"  Formulation: {formulation_type}")
    print(f"  Status: {pulp_results['status']}")
    print(f"  Objective: {pulp_results['objective_value']:.6f}")
    print(f"  Solve time: {pulp_results['solve_time']:.2f} seconds")
    
    # Save PuLP results
    pulp_path = f'PuLP_Results/pulp_{formulation_type}_{scenario}_{timestamp}.json'
    print(f"\nSaving PuLP results to {pulp_path}...")
    with open(pulp_path, 'w') as f:
        json.dump(pulp_results, f, indent=2)
    
    # Solve with DWave (only for binary formulation)
    if land_method == 'even_grid':
        print("\n" + "=" * 80)
        print("SOLVING WITH DWAVE (BQUBO: CQM→BQM + HybridBQM)")
        print("=" * 80)
        token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
        sampleset, dwave_solve_time, qpu_access_time, bqm_conversion_time, invert = solve_with_dwave(cqm, token)
        
        # BQM samplesets don't have feasibility - all samples are valid (constraints are penalties)
        print(f"  Total samples: {len(sampleset)}")
        print(f"  Total solve time: {dwave_solve_time:.2f} seconds")
        print(f"  BQM conversion time: {bqm_conversion_time:.2f} seconds")
        print(f"  QPU access time: {qpu_access_time:.4f} seconds")
        
        if len(sampleset) > 0:
            best = sampleset.first
            print(f"  Best energy: {best.energy:.6f}")
            best_objective = -best.energy
            print(f"  Best objective: {best_objective:.6f}")
    else:
        print("\n" + "=" * 80)
        print("SKIPPING DWAVE (Continuous formulation not quantum-compatible)")
        print("=" * 80)
        sampleset = None
        dwave_solve_time = 0
        qpu_access_time = 0
        bqm_conversion_time = 0
        best_objective = None
    
    # Save DWave results (only for binary formulation)
    if land_method == 'even_grid' and sampleset is not None:
        dwave_pickle_path = f'DWave_Results/dwave_bqubo_{scenario}_{timestamp}.pickle'
        print(f"\nSaving DWave sampleset to {dwave_pickle_path}...")
        with open(dwave_pickle_path, 'wb') as f:
            pickle.dump(sampleset, f)
        
        # Save DWave results as JSON for easy reading
        dwave_json_path = f'DWave_Results/dwave_bqubo_{scenario}_{timestamp}.json'
        dwave_results = {
            'status': 'Optimal' if len(sampleset) > 0 else 'No solutions',
            'objective_value': best_objective if len(sampleset) > 0 else None,
            'solve_time': dwave_solve_time,
            'qpu_access_time': qpu_access_time,
            'bqm_conversion_time': bqm_conversion_time,
            'num_samples': len(sampleset) if sampleset else 0,
            'formulation': 'BQUBO (binary only)'
        }
        with open(dwave_json_path, 'w') as f:
            json.dump(dwave_results, f, indent=2)
    else:
        dwave_pickle_path = None
        dwave_json_path = None
    
    # Create run manifest
    manifest = {
        'scenario': scenario,
        'timestamp': timestamp,
        'cqm_path': cqm_path,
        'constraints_path': constraints_path,
        'pulp_path': pulp_path,
        'dwave_pickle_path': dwave_pickle_path,
        'dwave_json_path': dwave_json_path,
        'farms': farms,
        'foods': list(foods.keys()),
        'pulp_status': pulp_results['status'],
        'pulp_objective': pulp_results['objective_value'],
        'dwave_status': dwave_results['status'],
        'dwave_objective': dwave_results['objective_value'],
        'dwave_qpu_time': qpu_access_time,
        'dwave_sample_count': len(sampleset),
        'formulation': 'binary_plantation'
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
    
    parser = argparse.ArgumentParser(description='Run solvers on a food optimization scenario with configurable land generation')
    parser.add_argument('--scenario', type=str, default='simple', 
                       choices=['simple', 'intermediate', 'full', 'custom', 'full_family'],
                       help='Scenario to solve (default: simple)')
    parser.add_argument('--land-method', type=str, default='even_grid',
                       choices=['even_grid', 'uneven_distribution'], 
                       help='Land generation method: even_grid (equal patches) or uneven_distribution (realistic farms) (default: even_grid)')
    parser.add_argument('--n-units', type=int, default=None,
                       help='Number of units to generate (patches for even_grid, farms for uneven_distribution). If not specified, uses defaults.')
    parser.add_argument('--total-land', type=float, default=100.0,
                       help='Total land area in hectares (default: 100.0)')
    
    args = parser.parse_args()
    
    main(scenario=args.scenario, land_method=args.land_method, n_units=args.n_units, total_land=args.total_land)
