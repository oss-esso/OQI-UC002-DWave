"""
Professional solver runner script with Linear-Quadratic objective.

This script:
1. Loads a scenario (simple, intermediate, or custom)
2. Converts to CQM with linear-quadratic objective (linear area + quadratic synergy bonus)
3. Saves the model
4. Solves with PuLP and saves results
5. Solves with Pyomo and saves results
6. (DWave solving enabled for CQM)
7. Saves all constraints for verification

The objective function combines:
- Linear term: Based on area allocation weighted by food attributes
- Quadratic term: Synergy bonus for planting similar crops (same food_group) on the same farm
"""

import os
import sys
import json
import pickle
import shutil
import time
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.scenarios import load_food_data
from dimod import ConstrainedQuadraticModel, Binary, Real
from dwave.system import LeapHybridCQMSampler
import pulp as pl
from tqdm import tqdm

# Try to import synergy optimizer (Cython first, then pure Python)
try:
    from synergy_optimizer import SynergyOptimizer
    SYNERGY_OPTIMIZER_TYPE = "Cython"
except ImportError:
    try:
        from src.synergy_optimizer_pure import SynergyOptimizer
        SYNERGY_OPTIMIZER_TYPE = "NumPy"
    except ImportError:
        SynergyOptimizer = None
        SYNERGY_OPTIMIZER_TYPE = "Original"

# Try to import Pyomo for solving
try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    print("Warning: Pyomo not available. Install with: pip install pyomo")

def extract_solution_summary(solution, farms, foods, land_availability):
    """
    Extract a summary of the solution showing crop selections and area allocations (LQ formulation).
    
    Args:
        solution: Dictionary with variable assignments (A_{farm}_{crop} and Y_{farm}_{crop})
        farms: List of farm names
        foods: Dictionary of food data
        land_availability: Dictionary mapping farm to area
        
    Returns:
        dict: Summary with crops selected, areas, and farm assignments
    """
    crops_selected = set()
    farm_assignments = []
    total_allocated = 0.0
    
    for crop in foods:
        # Calculate total area allocated to this crop
        total_area = 0.0
        assigned_farms = []
        
        for farm in farms:
            a_var = f"A_{farm}_{crop}"
            y_var = f"Y_{farm}_{crop}"
            area = solution.get(a_var, 0)
            selected = solution.get(y_var, 0)
            
            if area > 1e-6:  # Only include if actually allocated
                total_area += area
                assigned_farms.append({
                    'farm': farm,
                    'area': area,
                    'selected': selected
                })
        
        if total_area > 1e-6:  # Only include if actually allocated
            crops_selected.add(crop)
            farm_assignments.append({
                'crop': crop,
                'total_area': total_area,
                'n_farms': len(assigned_farms),
                'farms': assigned_farms
            })
            total_allocated += total_area
    
    total_available = sum(land_availability.values())
    idle_area = total_available - total_allocated
    
    return {
        'crops_selected': list(crops_selected),
        'n_crops': len(crops_selected),
        'farm_assignments': farm_assignments,
        'total_allocated': total_allocated,
        'total_available': total_available,
        'idle_area': idle_area,
        'utilization': total_allocated / total_available if total_available > 0 else 0
    }

def create_cqm(farms, foods, food_groups, config):
    """
    Creates a CQM for the food optimization problem with linear-quadratic objective.
    
    The objective combines:
    - Linear term: Proportional to allocated area A with weighted food attributes
    - Quadratic term: Synergy bonus for planting similar crops (same food_group) on same farm
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data
        food_groups: Dictionary of food groups
        config: Configuration dictionary
    
    Returns CQM, variables, and constraint metadata.
    """
    cqm = ConstrainedQuadraticModel()
    
    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    synergy_matrix = params.get('synergy_matrix', {})
    synergy_bonus_weight = weights.get('synergy_bonus', 0.1)
    
    n_farms = len(farms)
    n_foods = len(foods)
    n_food_groups = len(food_groups) if food_group_constraints else 0
    
    # Count synergy pairs for progress bar
    n_synergy_pairs = 0
    for crop1, pairs in synergy_matrix.items():
        n_synergy_pairs += len(pairs)
    n_synergy_pairs = n_synergy_pairs // 2  # Each pair counted twice
    
    # Calculate total operations for progress bar
    total_ops = (
        n_farms * n_foods * 2 +  # Variables (A and Y)
        n_farms * n_foods +       # Linear objective terms
        n_farms * n_synergy_pairs +  # Quadratic synergy terms
        n_farms +                 # Land availability constraints
        n_farms * n_foods * 2 +   # Linking constraints (2 per farm-food pair)
        n_farms * n_food_groups * 2  # Food group constraints (min and max)
    )
    
    pbar = tqdm(total=total_ops, desc="Building CQM with linear-quadratic objective", unit="op", ncols=100)
    
    # Define variables
    A = {}
    Y = {}
    
    pbar.set_description("Creating area and binary variables")
    for farm in farms:
        for food in foods:
            A[(farm, food)] = Real(f"A_{farm}_{food}", lower_bound=0, upper_bound=land_availability[farm])
            pbar.update(1)
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")
            pbar.update(1)
    
    # Objective function - Linear term
    pbar.set_description("Building linear objective")
    objective = 0
    for farm in farms:
        for food in foods:
            objective += (
                weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) * A[(farm, food)] +
                weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) * A[(farm, food)] -
                weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) * A[(farm, food)] +
                weights.get('affordability', 0) * foods[food].get('affordability', 0) * A[(farm, food)] +
                weights.get('sustainability', 0) * foods[food].get('sustainability', 0) * A[(farm, food)]
            )
            pbar.update(1)
    
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
            # Iterate through synergy matrix
            for crop1, pairs in synergy_matrix.items():
                if crop1 in foods:
                    for crop2, boost_value in pairs.items():
                        if crop2 in foods and crop1 < crop2:  # Avoid double counting
                            objective += synergy_bonus_weight * boost_value * Y[(farm, crop1)] * Y[(farm, crop2)]
                            pbar.update(1)
    
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
    pbar.set_description("Adding land constraints")
    for farm in farms:
        cqm.add_constraint(
            sum(A[(farm, food)] for food in foods) - land_availability[farm] <= 0,
            label=f"Land_Availability_{farm}"
        )
        constraint_metadata['land_availability'][farm] = {
            'type': 'land_availability',
            'farm': farm,
            'max_land': land_availability[farm]
        }
        pbar.update(1)
    
    # Linking constraints
    pbar.set_description("Adding linking constraints")
    for farm in farms:
        for food in foods:
            A_min = min_planting_area.get(food, 0)
            # CRITICAL FIX: If no minimum area is specified, use small epsilon (0.001 ha)
            # This prevents Y=1 when A=0 (which would incorrectly claim synergy bonuses)
            if A_min == 0:
                A_min = 0.0001  # 0.0001 hectares = 1 square meter minimum
            
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
            pbar.update(1)
            
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
                    pbar.update(1)
                
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
                    pbar.update(1)
    
    pbar.set_description("CQM complete")
    pbar.close()
    
    return cqm, A, Y, constraint_metadata

def solve_with_pulp(farms, foods, food_groups, config):
    """
    Solve with PuLP using linear-quadratic objective.
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data
        food_groups: Dictionary of food groups
        config: Configuration dictionary
    
    Returns model and results.
    """
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    synergy_matrix = params.get('synergy_matrix', {})
    synergy_bonus_weight = weights.get('synergy_bonus', 0.1)
    
    print(f"\nCreating PuLP model with linear-quadratic objective...")
    print(f"  Note: PuLP uses linearized form of quadratic synergy bonus")
    
    # Decision variables
    A_pulp = pl.LpVariable.dicts("Area", [(f, c) for f in farms for c in foods], lowBound=0)
    Y_pulp = pl.LpVariable.dicts("Choose", [(f, c) for f in farms for c in foods], cat='Binary')
    
    # Additional variables for linearized quadratic terms (McCormick relaxation)
    # For each Y[f, c1] * Y[f, c2] product, we create a new binary variable Z[f, c1, c2]
    
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
                        if crop2 in foods and crop1 < crop2:  # Avoid double counting
                            synergy_pairs.append((f, crop1, crop2, boost_value))
    
    # Create Z variables for all synergy pairs
    Z_pulp = {}
    for f, crop1, crop2, boost_value in synergy_pairs:
        Z_pulp[(f, crop1, crop2)] = pl.LpVariable(
            f"Z_{f}_{crop1}_{crop2}", 
            cat='Binary'
        )
    
    # Create model
    model = pl.LpProblem("Food_Optimization_LQ_PuLP", pl.LpMaximize)
    
    # Objective function - Linear term
    objective_terms = []
    for f in farms:
        for c in foods:
            coeff = (
                weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
                weights.get('affordability', 0) * foods[c].get('affordability', 0) +
                weights.get('sustainability', 0) * foods[c].get('sustainability', 0)
            )
            objective_terms.append(coeff * A_pulp[(f, c)])
    
    # Objective function - Linearized quadratic synergy bonus
    # Use Z variables instead of Y * Y products
    synergy_terms = []
    for f, crop1, crop2, boost_value in synergy_pairs:
        synergy_terms.append(synergy_bonus_weight * boost_value * Z_pulp[(f, crop1, crop2)])
    
    goal = pl.lpSum(objective_terms) + pl.lpSum(synergy_terms)
    model += goal, "Objective"
    
    # Linearization constraints for Z[f, c1, c2] = Y[f, c1] * Y[f, c2]
    # McCormick relaxation: Z <= Y1, Z <= Y2, Z >= Y1 + Y2 - 1
    for f, crop1, crop2, _ in synergy_pairs:
        model += Z_pulp[(f, crop1, crop2)] <= Y_pulp[(f, crop1)], f"Z_upper1_{f}_{crop1}_{crop2}"
        model += Z_pulp[(f, crop1, crop2)] <= Y_pulp[(f, crop2)], f"Z_upper2_{f}_{crop1}_{crop2}"
        model += Z_pulp[(f, crop1, crop2)] >= Y_pulp[(f, crop1)] + Y_pulp[(f, crop2)] - 1, f"Z_lower_{f}_{crop1}_{crop2}"
    
    # Land availability constraints
    for f in farms:
        model += pl.lpSum([A_pulp[(f, c)] for c in foods]) <= land_availability[f], f"Max_Area_{f}"
    
    # Linking constraints (binary selection)
    for f in farms:
        for c in foods:
            A_min = min_planting_area.get(c, 0)
            # CRITICAL FIX: If no minimum area is specified, use small epsilon (0.001 ha)
            # This prevents Y=1 when A=0 (which would incorrectly claim synergy bonuses)
            if A_min == 0:
                A_min = 0.001  # 0.001 hectares = 10 square meters minimum
            model += A_pulp[(f, c)] >= A_min * Y_pulp[(f, c)], f"MinArea_{f}_{c}"
            model += A_pulp[(f, c)] <= land_availability[f] * Y_pulp[(f, c)], f"MaxArea_{f}_{c}"
    
    # Food group constraints
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
    
    # Solve
    print("  Solving with Gurobi...")
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
    
    # Extract results and calculate total area
    total_area = 0.0
    areas = {}
    selections = {}
    
    for f in farms:
        for c in foods:
            key = f"{f}_{c}"
            area_val = A_pulp[(f, c)].value() if A_pulp[(f, c)].value() is not None else 0.0
            select_val = Y_pulp[(f, c)].value() if Y_pulp[(f, c)].value() is not None else 0.0
            
            areas[key] = area_val
            selections[key] = select_val
            total_area += area_val
    
    # Calculate normalized objective (objective per unit area)
    obj_value = pl.value(model.objective)
    normalized_objective = obj_value / total_area if total_area > 1e-6 else 0.0
    
    # Create solution dict with proper prefixes for validation and summary extraction
    # Use A_ and Y_ prefixes to distinguish area and selection variables
    solution = {}
    for key, val in areas.items():
        solution[f"A_{key}"] = val
    for key, val in selections.items():
        solution[f"Y_{key}"] = val
    
    # Validate solution against constraints
    validation_results = validate_solution_constraints(
        solution, farms, foods, food_groups, land_availability, config
    )
    
    # Generate solution summary
    solution_summary = extract_solution_summary(solution, farms, foods, land_availability)
    
    results = {
        'status': pl.LpStatus[model.status],
        'objective_value': obj_value,
        'normalized_objective': normalized_objective,
        'total_area': total_area,
        'solve_time': solve_time,
        'areas': areas,
        'selections': selections,
        'solution_summary': solution_summary,
        'validation': validation_results
    }
    
    print(f"  Total area allocated: {total_area:.2f}")
    print(f"  Raw objective: {obj_value:.4f}")
    print(f"  Normalized objective: {normalized_objective:.6f}")
    validation_status = "PASSED" if validation_results['is_feasible'] else "FAILED"
    print(f"  Constraint validation: {validation_status}")
    if not validation_results['is_feasible']:
        print(f"    Violations: {validation_results['n_violations']}")
        for violation in validation_results['violations'][:3]:  # Show first 3 violations
            print(f"      - {violation}")
        if validation_results['n_violations'] > 3:
            print(f"      ... and {validation_results['n_violations'] - 3} more")
    
    return model, results

def validate_solution_constraints(solution, farms, foods, food_groups, land_availability, config):
    """
    Validate if a solution satisfies all original CQM constraints for LQ formulation.
    
    Args:
        solution: Dictionary with variable assignments (A_{farm}_{food} and Y_{farm}_{food})
        farms: List of farm names
        foods: Dictionary of food data
        food_groups: Dictionary of food groups
        land_availability: Dictionary mapping farm to area
        config: Configuration dictionary with parameters
        
    Returns:
        dict: Validation results with violations and constraint checks
    """
    params = config['parameters']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    violations = []
    constraint_checks = {
        'land_availability': {'passed': 0, 'failed': 0, 'violations': []},
        'linking_constraints': {'passed': 0, 'failed': 0, 'violations': []},
        'food_group_constraints': {'passed': 0, 'failed': 0, 'violations': []}
    }
    
    # 1. Check: Land availability per farm
    for farm in farms:
        farm_total = sum(solution.get(f"A_{farm}_{crop}", 0) for crop in foods)
        farm_capacity = land_availability[farm]
        
        if farm_total > farm_capacity + 0.01:  # Absolute tolerance
            violation = f"{farm}: {farm_total:.4f} ha > {farm_capacity:.4f} ha capacity"
            violations.append(violation)
            constraint_checks['land_availability']['violations'].append(violation)
            constraint_checks['land_availability']['failed'] += 1
        else:
            constraint_checks['land_availability']['passed'] += 1
    
    # 2. Check: Linking constraints A and Y
    # Constraint: A >= min_area * Y  AND  A <= farm_capacity * Y
    for farm in farms:
        farm_capacity = land_availability[farm]
        for crop in foods:
            a_val = solution.get(f"A_{farm}_{crop}", 0)
            y_val = solution.get(f"Y_{farm}_{crop}", 0)
            min_area = min_planting_area.get(crop, 0)
            
            # Treat Y values close to 0 or 1 as binary (for solvers with numerical imprecision)
            # Y > 0.9 is considered selected (Y=1)
            # Y < 0.1 is considered not selected (Y=0)
            # 0.1 <= Y <= 0.9 is considered a violation (Y should be binary)
            
            if y_val > 0.9:  # Selected (Y ≈ 1)
                if a_val < min_area * 0.999:  # Relative tolerance
                    violation = f"A_{farm}_{crop}={a_val:.4f} < min_area={min_area:.4f} (Y={y_val:.2f}≈1)"
                    violations.append(violation)
                    constraint_checks['linking_constraints']['violations'].append(violation)
                    constraint_checks['linking_constraints']['failed'] += 1
                elif a_val > farm_capacity + 0.001:
                    violation = f"A_{farm}_{crop}={a_val:.4f} > farm_capacity={farm_capacity:.4f} (Y={y_val:.2f}≈1)"
                    violations.append(violation)
                    constraint_checks['linking_constraints']['violations'].append(violation)
                    constraint_checks['linking_constraints']['failed'] += 1
                else:
                    constraint_checks['linking_constraints']['passed'] += 1
            elif y_val < 0.1:  # Not selected (Y ≈ 0)
                if a_val > 0.001:
                    violation = f"A_{farm}_{crop}={a_val:.4f} but Y_{farm}_{crop}={y_val:.4f}≈0 (should be 0)"
                    violations.append(violation)
                    constraint_checks['linking_constraints']['violations'].append(violation)
                    constraint_checks['linking_constraints']['failed'] += 1
                else:
                    constraint_checks['linking_constraints']['passed'] += 1
            else:  # Y is not binary (0.1 <= Y <= 0.9)
                # This indicates the solver is not properly enforcing binary constraints
                violation = f"Y_{farm}_{crop}={y_val:.4f} is not binary (should be 0 or 1)"
                violations.append(violation)
                constraint_checks['linking_constraints']['violations'].append(violation)
                constraint_checks['linking_constraints']['failed'] += 1
    
    # 3. Check: Food group constraints - GLOBAL across all farms
    if food_group_constraints:
        for group_name, group_data in food_group_constraints.items():
            if group_name in food_groups:
                crops_in_group = food_groups[group_name]
                
                # Count total selections across ALL farms (global constraint)
                # Round Y values: >0.9 counts as selected, <0.1 counts as not selected
                n_selected_global = sum(
                    1 if solution.get(f"Y_{farm}_{crop}", 0) > 0.9 else 0
                    for farm in farms 
                    for crop in crops_in_group
                )
                
                min_foods = group_data.get('min_foods', 0)
                max_foods = group_data.get('max_foods', len(crops_in_group) * len(farms))
                
                if n_selected_global < min_foods:
                    violation = f"Group {group_name}: {n_selected_global} selections < min={min_foods} (global)"
                    violations.append(violation)
                    constraint_checks['food_group_constraints']['violations'].append(violation)
                    constraint_checks['food_group_constraints']['failed'] += 1
                elif n_selected_global > max_foods:
                    violation = f"Group {group_name}: {n_selected_global} selections > max={max_foods} (global)"
                    violations.append(violation)
                    constraint_checks['food_group_constraints']['violations'].append(violation)
                    constraint_checks['food_group_constraints']['failed'] += 1
                else:
                    constraint_checks['food_group_constraints']['passed'] += 1
    
    # Calculate summary statistics
    total_checks = sum(check['passed'] + check['failed'] for check in constraint_checks.values())
    total_passed = sum(check['passed'] for check in constraint_checks.values())
    total_failed = sum(check['failed'] for check in constraint_checks.values())
    
    return {
        'is_feasible': len(violations) == 0,
        'n_violations': len(violations),
        'violations': violations,
        'constraint_checks': constraint_checks,
        'summary': {
            'total_checks': total_checks,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'pass_rate': total_passed / total_checks if total_checks > 0 else 0
        }
    }

def solve_with_dwave(cqm, token):
    """Solve with DWave and return sampleset."""
    sampler = LeapHybridCQMSampler(token=token)
    
    print("Submitting to DWave Leap hybrid solver...")
    sampleset = sampler.sample_cqm(cqm, label="Food Optimization - Professional Run")
    
    # Extract timing information correctly
    # charge_time is in seconds (not microseconds)
    solve_time = sampleset.info.get('run_time', 0) / 1e6  # Convert milliseconds to seconds
    
    return sampleset, solve_time

def solve_with_pyomo(farms, foods, food_groups, config):
    """
    Solve with Pyomo using linear-quadratic objective.
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data
        food_groups: Dictionary of food groups
        config: Configuration dictionary
    
    Returns model and results.
    """
    if not PYOMO_AVAILABLE:
        print("ERROR: Pyomo is not installed. Skipping Pyomo solver.")
        return None, {
            'status': 'Not Available',
            'objective_value': None,
            'solve_time': 0.0,
            'areas': {},
            'selections': {},
            'error': 'Pyomo not installed'
        }
    
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    synergy_matrix = params.get('synergy_matrix', {})
    synergy_bonus_weight = weights.get('synergy_bonus', 0.1)
    
    print(f"\nCreating Pyomo model with linear-quadratic objective...")
    
    # Create model
    model = pyo.ConcreteModel(name="Food_Optimization_LQ_Pyomo")
    
    # Sets
    model.farms = pyo.Set(initialize=farms)
    model.foods = pyo.Set(initialize=list(foods.keys()))
    
    # Variables
    model.A = pyo.Var(model.farms, model.foods, domain=pyo.NonNegativeReals,
                      bounds=lambda m, f, c: (0, land_availability[f]))
    model.Y = pyo.Var(model.farms, model.foods, domain=pyo.Binary)
    
    # Objective function
    def objective_rule(m):
        # Linear term
        obj = 0
        for f in m.farms:
            for c in m.foods:
                coeff = (
                    weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
                    weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
                    weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
                    weights.get('affordability', 0) * foods[c].get('affordability', 0) +
                    weights.get('sustainability', 0) * foods[c].get('sustainability', 0)
                )
                obj += coeff * m.A[f, c]
        
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
                            if crop2 in foods and crop1 < crop2:  # Avoid double counting
                                obj += synergy_bonus_weight * boost_value * m.Y[f, crop1] * m.Y[f, crop2]
        
        return obj
    
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    # Land availability constraints
    def land_constraint_rule(m, f):
        return sum(m.A[f, c] for c in m.foods) <= land_availability[f]
    model.land_constraint = pyo.Constraint(model.farms, rule=land_constraint_rule)
    
    # Linking constraints (binary selection)
    def min_area_rule(m, f, c):
        A_min = min_planting_area.get(c, 0)
        # CRITICAL FIX: If no minimum area is specified, use small epsilon (0.001 ha)
        # This prevents Y=1 when A=0 (which would incorrectly claim synergy bonuses)
        if A_min == 0:
            A_min = 0.001  # 0.001 hectares = 10 square meters minimum
        return m.A[f, c] >= A_min * m.Y[f, c]
    model.min_area = pyo.Constraint(model.farms, model.foods, rule=min_area_rule)
    
    def max_area_rule(m, f, c):
        return m.A[f, c] <= land_availability[f] * m.Y[f, c]
    model.max_area = pyo.Constraint(model.farms, model.foods, rule=max_area_rule)
    
    # Food group constraints - GLOBAL across all farms
    if food_group_constraints:
        def min_food_group_rule(m, g):
            foods_in_group = food_groups.get(g, [])
            min_foods = food_group_constraints[g].get('min_foods', None)
            if min_foods is not None and foods_in_group:
                return sum(m.Y[f, c] for f in m.farms for c in foods_in_group if c in m.foods) >= min_foods
            else:
                return pyo.Constraint.Skip
        
        def max_food_group_rule(m, g):
            foods_in_group = food_groups.get(g, [])
            max_foods = food_group_constraints[g].get('max_foods', None)
            if max_foods is not None and foods_in_group:
                return sum(m.Y[f, c] for f in m.farms for c in foods_in_group if c in m.foods) <= max_foods
            else:
                return pyo.Constraint.Skip
        
        model.min_food_group = pyo.Constraint(
            list(food_group_constraints.keys()), 
            rule=min_food_group_rule
        )
        model.max_food_group = pyo.Constraint(
            list(food_group_constraints.keys()), 
            rule=max_food_group_rule
        )
    
    # Try to find an available MIQP/MIQCP solver
    solver_name = None
    solver = None
    
    print("  Searching for MIQP/MIQCP solver (required for binary+quadratic)...")
    
    # CRITICAL: We need MIQP (Mixed-Integer Quadratic Programming) solvers
    # IPOPT is NOT suitable - it's a continuous NLP solver that treats binaries as [0,1] continuous!
    # Proper priority: Commercial MIQP solvers > Open-source MIQCP solvers
    # - Gurobi, CPLEX: Commercial, excellent for MIQP
    # - SCIP: Open-source, supports MIQCP
    # - CBC, GLPK: Only support linear (no quadratic), will linearize
    # - IPOPT: Continuous only - CANNOT handle binary variables!
    
    solver_options = [
        'gurobi',  # Commercial MIQP - best performance
        'cplex',   # Commercial MIQP - excellent
        'scip',    # Open-source MIQCP - good
        'cbc',     # Open-source MIP only (no Q) - requires linearization
        'glpk',    # Open-source MIP only (no Q) - requires linearization
    ]
    
    for solver_opt in solver_options:
        try:
            test_solver = pyo.SolverFactory(solver_opt)
            if test_solver.available():
                solver_name = solver_opt
                solver = test_solver
                print(f"  * Found solver: {solver_name}")
                if solver_name in ['cbc', 'glpk']:
                    print(f"  ! Warning: {solver_name} doesn't support quadratic objectives natively")
                    print(f"    Pyomo will attempt automatic linearization (may be suboptimal)")
                break
        except Exception as e:
            continue
    
    if solver is None:
        print("  X ERROR: No suitable MIQP/MIP solver found.")
        print("  ")
        print("  CRITICAL: IPOPT was intentionally excluded because it's a continuous")
        print("            NLP solver that CANNOT handle binary variables properly!")
        print("  ")
        print("  Recommended solvers for LQ (in priority order):")
        print("    1. Gurobi (commercial, free academic license):")
        print("       conda install -c gurobi gurobi")
        print("    2. CPLEX (commercial, free academic license):")
        print("       conda install -c ibmdecisionoptimization cplex")
        print("    3. SCIP (open-source MIQCP):")
        print("       conda install -c conda-forge pyscipopt")
        print("    4. CBC (open-source MIP, will linearize quadratic):")
        print("       conda install -c conda-forge coincbc")
        print("  ")
        return model, {
            'status': 'No Solver',
            'objective_value': None,
            'solve_time': 0.0,
            'areas': {},
            'selections': {},
            'error': 'No MIQP/MIP solver available. IPOPT excluded (continuous NLP only).'
        }
    
    # Solve
    print(f"  Solving with {solver_name}...")
    start_time = time.time()
    
    try:
        results = solver.solve(model, tee=False)
        solve_time = time.time() - start_time
        
        # Extract results and calculate total area
        status = str(results.solver.status)
        termination = str(results.solver.termination_condition)
        
        total_area = 0.0
        areas = {}
        selections = {}
        
        for f in farms:
            for c in foods:
                key = f"{f}_{c}"
                area_val = pyo.value(model.A[f, c]) if model.A[f, c].value is not None else 0.0
                select_val_raw = pyo.value(model.Y[f, c]) if model.Y[f, c].value is not None else 0.0
                
                # Clean up numerical noise for area values
                if abs(area_val) < 1e-6:
                    area_val = 0.0
                
                # Round Y values to nearest binary (0 or 1)
                # For MIQP solvers like Gurobi, Y should already be exactly 0 or 1
                # But we round for robustness: values >0.5 -> 1, values <=0.5 -> 0
                if select_val_raw > 0.5:
                    select_val = 1.0
                else:
                    select_val = 0.0
                
                areas[key] = area_val
                selections[key] = select_val
                total_area += area_val
        
        # Calculate normalized objective (objective per unit area)
        obj_value = pyo.value(model.obj) if pyo.value(model.obj) is not None else None
        normalized_objective = obj_value / total_area if (obj_value is not None and total_area > 1e-6) else 0.0
        
        # Create solution dict with proper prefixes for validation and summary extraction
        # Use A_ and Y_ prefixes to distinguish area and selection variables
        solution = {}
        for key, val in areas.items():
            solution[f"A_{key}"] = val
        for key, val in selections.items():
            solution[f"Y_{key}"] = val
        
        # Validate solution against constraints
        validation_results = validate_solution_constraints(
            solution, farms, foods, food_groups, land_availability, config
        )
        
        # Generate solution summary
        solution_summary = extract_solution_summary(solution, farms, foods, land_availability)
        
        output = {
            'status': f"{status} ({termination})",
            'solver': solver_name,
            'objective_value': obj_value,
            'normalized_objective': normalized_objective,
            'total_area': total_area,
            'solve_time': solve_time,
            'areas': areas,
            'selections': selections,
            'solution_summary': solution_summary,
            'validation': validation_results
        }
        
        print(f"  Total area allocated: {total_area:.2f}")
        print(f"  Raw objective: {obj_value:.4f}" if obj_value else "  Raw objective: None")
        print(f"  Normalized objective: {normalized_objective:.6f}")
        print(f"  Crops selected: {solution_summary['n_crops']}")
        validation_status = "PASSED" if validation_results['is_feasible'] else "FAILED"
        print(f"  Constraint validation: {validation_status}")
        if not validation_results['is_feasible']:
            print(f"    Violations: {validation_results['n_violations']}")
            for violation in validation_results['violations'][:3]:  # Show first 3 violations
                print(f"      - {violation}")
            if validation_results['n_violations'] > 3:
                print(f"      ... and {validation_results['n_violations'] - 3} more")
        
        return model, output
        
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"  ERROR during solving: {str(e)}")
        return model, {
            'status': 'Error',
            'solver': solver_name,
            'objective_value': None,
            'solve_time': solve_time,
            'areas': {},
            'selections': {},
            'error': str(e)
        }

def main(scenario='simple'):
    """Main execution function."""
    print("=" * 80)
    print("LINEAR-QUADRATIC SOLVER RUNNER")
    print("=" * 80)
    
    # Create output directories
    os.makedirs('PuLP_Results_LQ', exist_ok=True)
    os.makedirs('DWave_Results_LQ', exist_ok=True)
    os.makedirs('CQM_Models_LQ', exist_ok=True)
    os.makedirs('Constraints_LQ', exist_ok=True)
    
    # Load scenario
    print(f"\nLoading '{scenario}' scenario...")
    farms, foods, food_groups, config = load_food_data(scenario)
    print(f"  Farms: {len(farms)} - {farms}")
    print(f"  Foods: {len(foods)} - {list(foods.keys())}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CQM with linear-quadratic objective
    print("\nCreating CQM with linear-quadratic objective...")
    cqm, A, Y, constraint_metadata = create_cqm(
        farms, foods, food_groups, config
    )
    print(f"  Variables: {len(cqm.variables)}")
    print(f"  Constraints: {len(cqm.constraints)}")
    print(f"  Objective: Linear + Quadratic synergy bonus")
    
    # Save CQM
    cqm_path = os.path.join(project_root, 'CQM_Models_LQ', f'cqm_lq_{scenario}_{timestamp}.cqm')
    os.makedirs(os.path.dirname(cqm_path), exist_ok=True)
    print(f"\nSaving CQM to {cqm_path}...")
    with open(cqm_path, 'wb') as f:
        shutil.copyfileobj(cqm.to_file(), f)
    
    # Save constraint metadata
    constraints_path = os.path.join(project_root, 'Constraints_LQ', f'constraints_lq_{scenario}_{timestamp}.json')
    os.makedirs(os.path.dirname(constraints_path), exist_ok=True)
    print(f"Saving constraints to {constraints_path}...")
    
    # Convert constraint_metadata keys to strings for JSON serialization
    # Also convert foods dict to serializable format
    foods_serializable = {
        name: {k: float(v) if isinstance(v, (int, float)) else v for k, v in attrs.items()}
        for name, attrs in foods.items()
    }
    
    # Serialize config properly
    config_serializable = {
        'parameters': {
            k: (dict(v) if isinstance(v, dict) else v)
            for k, v in config['parameters'].items()
        }
    }
    
    constraints_json = {
        'scenario': scenario,
        'timestamp': timestamp,
        'farms': farms,
        'foods': list(foods.keys()),
        'foods_data': foods_serializable,
        'food_groups': food_groups,
        'config': config_serializable,
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
    print("SOLVING WITH PULP (Linear-Quadratic Objective)")
    print("=" * 80)
    pulp_model, pulp_results = solve_with_pulp(farms, foods, food_groups, config)
    print(f"  Status: {pulp_results['status']}")
    print(f"  Objective: {pulp_results['objective_value']:.6f}")
    print(f"  Solve time: {pulp_results['solve_time']:.2f} seconds")
    
    # Save PuLP results
    pulp_path = os.path.join(project_root, 'PuLP_Results_LQ', f'pulp_lq_{scenario}_{timestamp}.json')
    os.makedirs(os.path.dirname(pulp_path), exist_ok=True)
    print(f"\nSaving PuLP results to {pulp_path}...")
    with open(pulp_path, 'w') as f:
        json.dump(pulp_results, f, indent=2)
    
    # Solve with Pyomo
    print("\n" + "=" * 80)
    print("SOLVING WITH PYOMO (Linear-Quadratic Objective)")
    print("=" * 80)
    pyomo_model, pyomo_results = solve_with_pyomo(farms, foods, food_groups, config)
    
    if pyomo_results.get('error'):
        print(f"  Status: {pyomo_results['status']}")
        print(f"  Error: {pyomo_results.get('error')}")
    else:
        print(f"  Solver: {pyomo_results.get('solver', 'Unknown')}")
        print(f"  Status: {pyomo_results['status']}")
        if pyomo_results['objective_value'] is not None:
            print(f"  Objective: {pyomo_results['objective_value']:.6f}")
        print(f"  Solve time: {pyomo_results['solve_time']:.2f} seconds")
    
    # Save Pyomo results
    pyomo_path = os.path.join(project_root, 'PuLP_Results_LQ', f'pyomo_lq_{scenario}_{timestamp}.json')
    os.makedirs(os.path.dirname(pyomo_path), exist_ok=True)
    print(f"\nSaving Pyomo results to {pyomo_path}...")
    with open(pyomo_path, 'w') as f:
        json.dump(pyomo_results, f, indent=2)
    
    # Solve with DWave
    print("\n" + "=" * 80)
    print("SOLVING WITH DWAVE (Linear-Quadratic Objective)")
    print("=" * 80)
    
    #token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
    token = None
    if token:
        try:
            sampleset, dwave_solve_time = solve_with_dwave(cqm, token)
            
            feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
            print(f"  Feasible solutions: {len(feasible_sampleset)} of {len(sampleset)}")
            print(f"  Total solve time: {dwave_solve_time:.2f} seconds")
            
            if feasible_sampleset:
                best = feasible_sampleset.first
                print(f"  Best energy: {best.energy:.6f}")
                
                # Extract timing info correctly
                # DWave timing info is in the sampleset.info dictionary
                qpu_time = sampleset.info.get('qpu_access_time', 0)  / 1e6
                charge_time = sampleset.info.get('charge_time', 0)   / 1e6
                run_time = dwave_solve_time   
                
                if qpu_time > 0:
                    print(f"  QPU access time: {qpu_time:.4f} seconds")
                if charge_time > 0:
                    print(f"  Charge time: {charge_time:.4f} seconds")
                if run_time > 0:
                    print(f"  Run time: {run_time:.4f} seconds")
            else:
                print("  WARNING: No feasible solutions found")
            
            # Save DWave results
            dwave_path = os.path.join(project_root, 'DWave_Results_LQ', f'dwave_lq_{scenario}_{timestamp}.pickle')
            os.makedirs(os.path.dirname(dwave_path), exist_ok=True)
            print(f"\nSaving DWave results to {dwave_path}...")
            with open(dwave_path, 'wb') as f:
                pickle.dump(sampleset, f)
            
        except Exception as e:
            print(f"  ERROR: DWave solving failed: {str(e)}")
            dwave_path = None
    else:
        print("  DWave API token not found in environment")
        print("  Set DWAVE_API_TOKEN environment variable to enable DWave solving")
        dwave_path = None
    
    # Compare results if multiple solvers succeeded
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    if pulp_results['status'] == 'Optimal':
        print(f"  PuLP:    {pulp_results['objective_value']:.6f}  |  {pulp_results['solve_time']:.2f}s")
    
    if pyomo_results.get('objective_value') is not None:
        print(f"  Pyomo:   {pyomo_results['objective_value']:.6f}  |  {pyomo_results['solve_time']:.2f}s")
    
    if dwave_path and feasible_sampleset:
        dwave_obj = -best.energy  # Convert energy back to objective
        print(f"  DWave:   {dwave_obj:.6f}  |  {dwave_solve_time:.2f}s")
    
    print("\n" + "=" * 80)
    print("SOLVER RUN COMPLETE")
    print("=" * 80)
    print(f"CQM saved to: {cqm_path}")
    print(f"Constraints saved to: {constraints_path}")
    print(f"PuLP results saved to: {pulp_path}")
    print(f"Pyomo results saved to: {pyomo_path}")
    if dwave_path:
        print(f"DWave results saved to: {dwave_path}")
    print(f"\nObjective: Linear area allocation + Quadratic synergy bonus")
    
    return cqm_path, constraints_path, pulp_path, pyomo_path, dwave_path if dwave_path else None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run solvers with linear-quadratic objective on a food optimization scenario'
    )
    parser.add_argument('--scenario', type=str, default='simple', 
                       choices=['simple', 'intermediate', 'full', 'custom', 'full_family'],
                       help='Scenario to solve (default: simple)')
    
    args = parser.parse_args()
    
    main(scenario=args.scenario)
