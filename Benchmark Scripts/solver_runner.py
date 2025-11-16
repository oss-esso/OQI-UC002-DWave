"""
Professional solver runner script.

This script:
1. Loads a scenario (simple, intermediate, or custom)
2. Converts to CQM and saves the model
3. Solves with PuLP and saves results
4. Solves with DWave and saves results
5. Saves all constraints for verification
"""

import pulp as pl
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, Binary, Real
from src.scenarios import load_food_data
import os
import sys
import json
import pickle
import shutil
import time
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)


def create_cqm(farms, foods, food_groups, config):
    """
    Creates a CQM for the food optimization problem.
    Returns CQM, variables, and constraint metadata.
    """
    cqm = ConstrainedQuadraticModel()

    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})

    # Define variables
    A = {}
    Y = {}

    for farm in tqdm(farms, desc="Creating variables"):
        for food in foods:
            A[(farm, food)] = Real(
                f"A_{farm}_{food}", lower_bound=0, upper_bound=land_availability[farm])
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")

    # Objective function
    total_area = sum(land_availability[farm] for farm in farms)

    objective = sum(
        weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) * A[(farm, food)] +
        weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) * A[(farm, food)] -
        weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) * A[(farm, food)] +
        weights.get('affordability', 0) * foods[food].get('affordability', 0) * A[(farm, food)] +
        weights.get('sustainability', 0) *
        foods[food].get('sustainability', 0) * A[(farm, food)]
        for farm in tqdm(farms) for food in foods
    )

    cqm.set_objective(-objective)

    # Constraint metadata
    constraint_metadata = {
        'land_availability': {},
        'min_area_if_selected': {},
        'max_area_if_selected': {},
        'food_group_min': {},
        'food_group_max': {}
    }

    # Land availability constraints
    for farm in tqdm(farms, desc="Adding land availability constraints"):
        cqm.add_constraint(
            sum(A[(farm, food)]
                for food in foods) - land_availability[farm] <= 0,
            label=f"Land_Availability_{farm}"
        )
        constraint_metadata['land_availability'][farm] = {
            'type': 'land_availability',
            'farm': farm,
            'max_land': land_availability[farm]
        }

    # Linking constraints
    for farm in tqdm(farms, desc="Adding linking constraints"):
        for food in foods:
            A_min = min_planting_area.get(food, 0)

            cqm.add_constraint(
                A[(farm, food)] - A_min * Y[(farm, food)] >= 0,
                label=f"Min_Area_If_Selected_{farm}_{food}"
            )
            constraint_metadata['min_area_if_selected'][(farm, food)] = {
                'type': 'min_area_if_selected',
                'farm': farm,
                'food': food,
                'min_area': A_min
            }

            cqm.add_constraint(
                A[(farm, food)] - land_availability[farm] * Y[(farm, food)] <= 0,
                label=f"Max_Area_If_Selected_{farm}_{food}"
            )
            constraint_metadata['max_area_if_selected'][(farm, food)] = {
                'type': 'max_area_if_selected',
                'farm': farm,
                'food': food,
                'max_land': land_availability[farm]
            }

    # Food group constraints - GLOBAL across all farms
    if food_group_constraints:
        for group, constraints in tqdm(food_group_constraints.items(), desc="Adding food group constraints"):
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                # Global minimum: across ALL farms, at least min_foods from this group
                if 'min_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[(farm, food)] for farm in farms for food in foods_in_group) -
                        constraints['min_foods'] >= 0,
                        label=f"Food_Group_Min_{group}_Global"
                    )
                    constraint_metadata['food_group_min'][group] = {
                        'type': 'food_group_min_global',
                        'group': group,
                        'min_foods': constraints['min_foods'],
                        'foods_in_group': foods_in_group,
                        'scope': 'global'
                    }

                # Global maximum: across ALL farms, at most max_foods from this group
                if 'max_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[(farm, food)] for farm in farms for food in foods_in_group) -
                        constraints['max_foods'] <= 0,
                        label=f"Food_Group_Max_{group}_Global"
                    )
                    constraint_metadata['food_group_max'][group] = {
                        'type': 'food_group_max_global',
                        'group': group,
                        'max_foods': constraints['max_foods'],
                        'foods_in_group': foods_in_group,
                        'scope': 'global'
                    }

    return cqm, A, Y, constraint_metadata


def solve_with_pulp(farms, foods, food_groups, config):
    """Solve with PuLP and return model and results."""
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})

    A_pulp = pl.LpVariable.dicts(
        "Area", [(f, c) for f in farms for c in foods], lowBound=0)
    Y_pulp = pl.LpVariable.dicts(
        "Choose", [(f, c) for f in farms for c in foods], cat='Binary')

    total_area = sum(land_availability[f] for f in farms)

    goal = (
        weights.get('nutritional_value', 0) * pl.lpSum([(foods[c].get('nutritional_value', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('nutrient_density', 0) * pl.lpSum([(foods[c].get('nutrient_density', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area -
        weights.get('environmental_impact', 0) * pl.lpSum([(foods[c].get('environmental_impact', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('affordability', 0) * pl.lpSum([(foods[c].get('affordability', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('sustainability', 0) * pl.lpSum([(foods[c].get(
            'sustainability', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area
    )

    model = pl.LpProblem("Food_Optimization", pl.LpMaximize)

    for f in farms:
        model += pl.lpSum([A_pulp[(f, c)] for c in foods]
                          ) <= land_availability[f], f"Max_Area_{f}"

    for f in farms:
        for c in foods:
            A_min = min_planting_area.get(c, 0)
            model += A_pulp[(f, c)] >= A_min * \
                Y_pulp[(f, c)], f"MinArea_{f}_{c}"
            model += A_pulp[(f, c)] <= land_availability[f] * \
                Y_pulp[(f, c)], f"MaxArea_{f}_{c}"

    if food_group_constraints:
        for g, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(g, [])
            if foods_in_group:
                # Global constraints: across ALL farms
                if 'min_foods' in constraints:
                    model += pl.lpSum([Y_pulp[(f, c)] for f in farms for c in foods_in_group]
                                      ) >= constraints['min_foods'], f"MinFoodGroup_Global_{g}"
                if 'max_foods' in constraints:
                    model += pl.lpSum([Y_pulp[(f, c)] for f in farms for c in foods_in_group]
                                      ) <= constraints['max_foods'], f"MaxFoodGroup_Global_{g}"

    model += goal, "Objective"

    start_time = time.time()
    # Use Gurobi with GPU acceleration and aggressive parallelization
    # GPU-specific parameters (requires Gurobi 9.0+ and CUDA-compatible GPU):
    #   - Method=2: Use barrier method (GPU-accelerated)
    #   - Crossover=0: Disable crossover to keep computation on GPU
    #   - BarHomogeneous=1: Use homogeneous barrier algorithm (better for GPU)
    #   - Threads=0: Use all available CPU threads for parallel processing
    #   - MIPFocus=1: Focus on finding good solutions quickly
    #   - Presolve=2: Aggressive presolve
    gurobi_options = [
        ('Method', 2),           # Barrier method (GPU-accelerated)
        ('Crossover', 0),        # Disable crossover to keep computation on GPU
        ('BarHomogeneous', 1),   # Homogeneous barrier (more GPU-friendly)
        ('Threads', 0),          # Use all available CPU threads for parallelization
        ('MIPFocus', 1),         # Focus on finding good solutions quickly
        ('Presolve', 2),         # Aggressive presolve
    ]
    
    try:
        # Try using GUROBI API directly for better GPU support
        print("  Using Gurobi API with GPU acceleration and parallelization...")
        solver = pl.GUROBI(msg=0, timeLimit=100)
        # Set parameters directly on the solver
        for param, value in gurobi_options:
            solver.optionsDict[param] = value
        model.solve(solver)
    except Exception as e:
        # Fallback to GUROBI_CMD if direct API is not available
        print(f"  Gurobi API failed ({str(e)[:50]}...), using GUROBI_CMD...")
        # GUROBI_CMD expects options as a list of "key=value" strings
        options_list = [f'{k}={v}' for k, v in gurobi_options]
        model.solve(pl.GUROBI_CMD(msg=0, options=options_list))
    solve_time = time.time() - start_time

    # Extract results
    results = {
        'status': pl.LpStatus[model.status],
        'objective_value': pl.value(model.objective),
        'solve_time': solve_time,
        'areas': {},
        'selections': {}
    }

    for f in farms:
        for c in foods:
            key = f"{f}_{c}"
            results['areas'][key] = A_pulp[(f, c)].value(
            ) if A_pulp[(f, c)].value() is not None else 0.0
            results['selections'][key] = Y_pulp[(f, c)].value(
            ) if Y_pulp[(f, c)].value() is not None else 0.0

    return model, results


def solve_with_dwave(cqm, token):
    """Solve with DWave and return sampleset."""
    sampler = LeapHybridCQMSampler(token=token)

    print("Submitting to DWave Leap hybrid solver...")
    
    sampleset = sampler.sample_cqm(
        cqm, label="Food Optimization - Professional Run")
    solve_time = sampleset.info.get('charge_time', 0) / 1e6  # Convert to seconds

    return sampleset, solve_time


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

    # Create CQM
    print("\nCreating CQM...")
    cqm, A, Y, constraint_metadata = create_cqm(
        farms, foods, food_groups, config)
    print(f"  Variables: {len(cqm.variables)}")
    print(f"  Constraints: {len(cqm.constraints)}")

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
        name: {k: float(v) if isinstance(v, (int, float))
               else v for k, v in attrs.items()}
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
            'land_availability': {str(k): v for k, v in constraint_metadata['land_availability'].items()},
            'min_area_if_selected': {str(k): v for k, v in constraint_metadata['min_area_if_selected'].items()},
            'max_area_if_selected': {str(k): v for k, v in constraint_metadata['max_area_if_selected'].items()},
            'food_group_min': {str(k): v for k, v in constraint_metadata['food_group_min'].items()},
            'food_group_max': {str(k): v for k, v in constraint_metadata['food_group_max'].items()}
        }
    }

    with open(constraints_path, 'w') as f:
        json.dump(constraints_json, f, indent=2)

    # Solve with PuLP
    print("\n" + "=" * 80)
    print("SOLVING WITH PULP")
    print("=" * 80)
    pulp_model, pulp_results = solve_with_pulp(
        farms, foods, food_groups, config)
    print(f"  Status: {pulp_results['status']}")
    print(f"  Objective: {pulp_results['objective_value']:.6f}")
    print(f"  Solve time: {pulp_results['solve_time']:.2f} seconds")

    # Save PuLP results
    pulp_path = f'PuLP_Results/pulp_{scenario}_{timestamp}.json'
    print(f"\nSaving PuLP results to {pulp_path}...")
    with open(pulp_path, 'w') as f:
        json.dump(pulp_results, f, indent=2)

    # Solve with DWave
    print("\n" + "=" * 80)
    print("SOLVING WITH DWAVE")
    print("=" * 80)
    token = os.getenv('DWAVE_API_TOKEN',
                      '45FS-7b81782896495d7c6a061bda257a9d9b03b082cd')
    sampleset, dwave_solve_time = solve_with_dwave(cqm, token)

    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
    print(
        f"  Feasible solutions: {len(feasible_sampleset)} of {len(sampleset)}")
    print(f"  Solve time: {dwave_solve_time:.2f} seconds")

    if feasible_sampleset:
        best = feasible_sampleset.first
        print(f"  Best energy: {best.energy:.6f}")

    # Save DWave results
    dwave_path = f'DWave_Results/dwave_{scenario}_{timestamp}.pickle'
    print(f"\nSaving DWave results to {dwave_path}...")
    with open(dwave_path, 'wb') as f:
        pickle.dump(sampleset, f)

    # Create run manifest
    manifest = {
        'scenario': scenario,
        'timestamp': timestamp,
        'cqm_path': cqm_path,
        'constraints_path': constraints_path,
        'pulp_path': pulp_path,
        'dwave_path': dwave_path,
        'farms': farms,
        'foods': list(foods.keys()),
        'pulp_status': pulp_results['status'],
        'pulp_objective': pulp_results['objective_value'],
        'dwave_feasible_count': len(feasible_sampleset),
        'dwave_total_count': len(sampleset)
    }

    manifest_path = f'run_manifest_{scenario}_{timestamp}.json'
    print(f"\nSaving run manifest to {manifest_path}...")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("SOLVER RUN COMPLETE")
    print("=" * 80)
    print(f"Manifest file: {manifest_path}")
    print("\nRun the verifier script with this manifest to check results:")
    print(f"  python verifier.py {manifest_path}")

    return manifest_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run solvers on a food optimization scenario')
    parser.add_argument('--scenario', type=str, default='simple',
                        choices=['simple', 'intermediate', 'full',
                                 'custom', '30farms', '60farms', '90farms', '250farms', '350farms', '500farms_full', '1000farms_full', '2000farms_full', 'full_family'],
                        help='Scenario to solve (default: simple)')

    args = parser.parse_args()

    main(scenario=args.scenario)
