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
from dimod import ConstrainedQuadraticModel, Binary, Real, cqm_to_bqm
from dwave.system import LeapHybridCQMSampler, LeapHybridBQMSampler
import pulp as pl
from tqdm import tqdm

def create_cqm(farms, foods, food_groups, config):
    """
    Creates a CQM for the BINARY food optimization problem.
    Each farm-crop combination is either planted (1 acre) or not (0).
    This formulation uses only Binary variables for true BQUBO compatibility.
    
    Returns CQM, variables, and constraint metadata.
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
    
    # Objective function - each selected crop contributes 1 acre worth of value
    pbar.set_description("Building objective")
    total_possible_plantations = n_farms * n_foods  # Normalization factor
    
    objective = 0
    for farm in farms:
        for food in foods:
            # Y is binary: 1 if planted (1 acre), 0 if not
            # Each plantation contributes its weighted value
            objective += (
                weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) * Y[(farm, food)] +
                weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) * Y[(farm, food)] -
                weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) * Y[(farm, food)] +
                weights.get('affordability', 0) * foods[food].get('affordability', 0) * Y[(farm, food)] +
                weights.get('sustainability', 0) * foods[food].get('sustainability', 0) * Y[(farm, food)]
            )
            pbar.update(1)
    
    # Normalize by total possible plantations
    objective = objective / total_possible_plantations
    cqm.set_objective(-objective)
    
    # Constraint metadata
    constraint_metadata = {
        'plantation_limit': {},
        'food_group_min': {},
        'food_group_max': {}
    }
    
    # Plantation limit constraints - each farm can have at most 'land_availability' plantations
    # Since each plantation is 1 acre, land_availability represents max number of crops
    pbar.set_description("Adding plantation limit constraints")
    for farm in farms:
        max_plantations = int(land_availability[farm])  # Max number of 1-acre plantations
        cqm.add_constraint(
            sum(Y[(farm, food)] for food in foods) <= max_plantations,
            label=f"Max_Plantations_{farm}"
        )
        constraint_metadata['plantation_limit'][farm] = {
            'type': 'plantation_limit',
            'farm': farm,
            'max_plantations': max_plantations
        }
        pbar.update(1)
    
    # Food group constraints - GLOBAL across all farms
    pbar.set_description("Adding food group constraints")
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                # Global minimum: across ALL farms, at least min_foods from this group
                if 'min_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[(farm, food)] for farm in farms for food in foods_in_group) - constraints['min_foods'] >= 0,
                        label=f"Food_Group_Min_{group}_Global"
                    )
                    constraint_metadata['food_group_min'][group] = {
                        'type': 'food_group_min_global',
                        'group': group,
                        'min_foods': constraints['min_foods'],
                        'foods_in_group': foods_in_group,
                        'scope': 'global'
                    }
                    pbar.update(1)
                
                # Global maximum: across ALL farms, at most max_foods from this group
                if 'max_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[(farm, food)] for farm in farms for food in foods_in_group) - constraints['max_foods'] <= 0,
                        label=f"Food_Group_Max_{group}_Global"
                    )
                    constraint_metadata['food_group_max'][group] = {
                        'type': 'food_group_max_global',
                        'group': group,
                        'max_foods': constraints['max_foods'],
                        'foods_in_group': foods_in_group,
                        'scope': 'global'
                    }
                    pbar.update(1)
    
    pbar.set_description("CQM complete")
    pbar.close()
    
    return cqm, Y, constraint_metadata

def solve_with_pulp(farms, foods, food_groups, config):
    """Solve with PuLP using BINARY formulation (1 acre plantations)."""
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    food_group_constraints = params.get('food_group_constraints', {})
    
    # Only binary variables - each represents a 1-acre plantation
    Y_pulp = pl.LpVariable.dicts("Plantation", [(f, c) for f in farms for c in foods], cat='Binary')
    
    total_possible_plantations = len(farms) * len(foods)
    
    # Objective: maximize weighted value of selected plantations (each is 1 acre)
    goal = (
        weights.get('nutritional_value', 0) * pl.lpSum([(foods[c].get('nutritional_value', 0) * Y_pulp[(f, c)]) for f in farms for c in foods]) / total_possible_plantations +
        weights.get('nutrient_density', 0) * pl.lpSum([(foods[c].get('nutrient_density', 0) * Y_pulp[(f, c)]) for f in farms for c in foods]) / total_possible_plantations -
        weights.get('environmental_impact', 0) * pl.lpSum([(foods[c].get('environmental_impact', 0) * Y_pulp[(f, c)]) for f in farms for c in foods]) / total_possible_plantations +
        weights.get('affordability', 0) * pl.lpSum([(foods[c].get('affordability', 0) * Y_pulp[(f, c)]) for f in farms for c in foods]) / total_possible_plantations +
        weights.get('sustainability', 0) * pl.lpSum([(foods[c].get('sustainability', 0) * Y_pulp[(f, c)]) for f in farms for c in foods]) / total_possible_plantations
    )
    
    model = pl.LpProblem("Food_Optimization_Binary", pl.LpMaximize)
    
    # Plantation limit: each farm can have at most land_availability plantations (1 acre each)
    for f in farms:
        max_plantations = int(land_availability[f])
        model += pl.lpSum([Y_pulp[(f, c)] for c in foods]) <= max_plantations, f"Max_Plantations_{f}"
    
    # Food group constraints
    if food_group_constraints:
        for g, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(g, [])
            if foods_in_group:
                # Global constraints: across ALL farms
                if 'min_foods' in constraints:
                    model += pl.lpSum([Y_pulp[(f, c)] for f in farms for c in foods_in_group]) >= constraints['min_foods'], f"MinFoodGroup_Global_{g}"
                if 'max_foods' in constraints:
                    model += pl.lpSum([Y_pulp[(f, c)] for f in farms for c in foods_in_group]) <= constraints['max_foods'], f"MaxFoodGroup_Global_{g}"
    
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

def main(scenario='simple'):
    """Main execution function."""
    print("=" * 80)
    print("PROFESSIONAL SOLVER RUNNER")
    print("=" * 80)
    
    # Create output directories
    os.makedirs('PuLP_Results', exist_ok=True)
    os.makedirs('DWave_Results', exist_ok=True)
    os.makedirs('CQM_Models', exist_ok=True)
    os.makedirs('Constraints', exist_ok=True)
    
    # Load scenario
    print(f"\nLoading '{scenario}' scenario...")
    farms, foods, food_groups, config = load_food_data(scenario)
    print(f"  Farms: {len(farms)} - {farms}")
    print(f"  Foods: {len(foods)} - {list(foods.keys())}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CQM with binary formulation
    print("\nCreating CQM with BINARY formulation (1-acre plantations)...")
    cqm, Y, constraint_metadata = create_cqm(farms, foods, food_groups, config)
    print(f"  Variables: {len(cqm.variables)} (all binary)")
    print(f"  Constraints: {len(cqm.constraints)}")
    print(f"  Formulation: Each farm-crop = 1 acre if selected, 0 if not")
    
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
            'plantation_limit': {str(k): v for k, v in constraint_metadata['plantation_limit'].items()},
            'food_group_min': {str(k): v for k, v in constraint_metadata['food_group_min'].items()},
            'food_group_max': {str(k): v for k, v in constraint_metadata['food_group_max'].items()}
        },
        'formulation': 'binary',
        'plantation_size_acres': 1
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
    
    # Save DWave results (both pickle and JSON)
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
        'num_samples': len(sampleset),
        'formulation': 'BQUBO (binary only)'
    }
    with open(dwave_json_path, 'w') as f:
        json.dump(dwave_results, f, indent=2)
    
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
    
    parser = argparse.ArgumentParser(description='Run solvers on a food optimization scenario')
    parser.add_argument('--scenario', type=str, default='simple', 
                       choices=['simple', 'intermediate', 'full', 'custom', 'full_family'],
                       help='Scenario to solve (default: simple)')
    
    args = parser.parse_args()
    
    main(scenario=args.scenario)
