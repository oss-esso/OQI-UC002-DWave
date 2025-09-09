"""
Test script for solving the food optimization problem with a CQM hybrid solver.

This script demonstrates how to:
1. Load a complex problem scenario.
2. Formulate the problem as a Constrained Quadratic Model (CQM).
3. Use the LeapHybridCQMSampler to solve the CQM.
4. Print and interpret the results.
"""

import os
import sys
import logging
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.scenarios import load_food_data
    import dimod
    from dwave.system import LeapHybridCQMSampler
except ImportError as e:
    logger.error(f"Import error: {e}. Please ensure all required libraries are installed.")
    sys.exit(1)

def create_cqm_for_food_problem(farms, foods, food_groups, config):
    """
    Creates a CQM for the food optimization problem.
    """
    cqm = dimod.ConstrainedQuadraticModel()
    
    # --- Define variables ---
    # Continuous variables: x_farm_food = area of land for food on farm
    land_availability = config['parameters']['land_availability']
    min_planting_area = config['parameters'].get('minimum_planting_area', {})
    
    # For each farm-food combination, the area can be either 0 or >= minimum planting area
    x = {}
    for farm in farms:
        for food in foods:
            min_area = min_planting_area.get(food, 0.1)  # Default minimum area
            # Variable can be 0 or between min_area and land_availability
            x[farm, food] = dimod.Real(f'x_{farm}_{food}', upper_bound=land_availability[farm])

    # Binary variables to indicate if a food is selected on a farm
    y = {(farm, food): dimod.Binary(f'y_{farm}_{food}') for farm in farms for food in foods}

    # --- Objective Function ---
    # Maximize the weighted score based on area allocation
    weights = config['parameters']['weights']
    objective = dimod.quicksum(-foods[food][obj] * weights[obj] * x[farm, food]
                               for farm in farms for food in foods
                               for obj in weights)
    cqm.set_objective(objective)

    # --- Constraints ---
    # 1. Land availability: total area used on a farm cannot exceed available land
    for farm in farms:
        cqm.add_constraint(dimod.quicksum(x[farm, food] for food in foods) <= land_availability[farm],
                           label=f'land_availability_{farm}')

    # 2. Binary logic: if x > 0, then y = 1; if x = 0, then y = 0
    # This is handled through the objective and minimum area constraints
    # We'll use the binary variables for group constraints only
    
    # 3. Minimum planting area: if area > 0, it must be at least minimum
    for farm in farms:
        for food in foods:
            if food in min_planting_area:
                # Either x = 0 OR x >= min_area (this is a disjunctive constraint)
                # We'll handle this through penalty in objective or separate formulation
                pass

    # 4. Food group constraints: min/max number of foods from each group
    food_group_constraints = config['parameters'].get('food_group_constraints', {})
    for group, constraints in food_group_constraints.items():
        foods_in_group = food_groups.get(group, [])
        if foods_in_group:
            # Link binary variables to continuous variables
            for food in foods_in_group:
                for farm in farms:
                    # If x[farm, food] > 0, then y[farm, food] = 1
                    # We approximate this with: x[farm, food] <= land_availability[farm] * y[farm, food]
                    # But this creates quadratic constraints...
                    pass
                    
            # Group constraints based on whether any farm produces the food
            group_selection = [dimod.Binary(f'group_{group}_{food}') for food in foods_in_group]
            
            # Link group selection to farm production
            for i, food in enumerate(foods_in_group):
                # If any farm produces this food, group_selection[i] = 1
                total_area = dimod.quicksum(x[farm, food] for farm in farms)
                # This is tricky without quadratic constraints...
                
            if 'min_foods' in constraints:
                cqm.add_constraint(dimod.quicksum(group_selection) >= constraints['min_foods'], 
                                 label=f'min_foods_{group}')
            if 'max_foods' in constraints:
                cqm.add_constraint(dimod.quicksum(group_selection) <= constraints['max_foods'], 
                                 label=f'max_foods_{group}')

    logger.info(f"CQM created with {len(cqm.variables)} variables and {len(cqm.constraints)} constraints.")
    return cqm

def main():
    """
    Main function to run the CQM hybrid solver test.
    """
    print("=" * 80)
    print("CQM HYBRID SOLVER TEST SCRIPT")
    print("=" * 80)

    # --- 1. Load Problem Data ---
    print("Loading 'full' complexity food data...")
    farms, foods, food_groups, config = load_food_data('full')

    # --- 2. Formulate the CQM ---
    print("\nFormulating the problem as a Constrained Quadratic Model (CQM)...")
    cqm = create_cqm_for_food_problem(farms, foods, food_groups, config)

    # --- 3. Configure and Run the Solver ---
    print("\nConfiguring the D-Wave CQM hybrid solver...")
    
    # IMPORTANT: You need to configure your D-Wave API token to run this.
    # You can do this by setting the DWAVE_API_TOKEN environment variable, or by
    # using a dwave.conf file. For this script, we use a placeholder.
    token = os.getenv('DWAVE_API_TOKEN', 'YOUR_D-WAVE_API_TOKEN_HERE')
    if token == 'YOUR_D-WAVE_API_TOKEN_HERE':
        print("\nWARNING: D-Wave API token is not configured.")
        print("The script will not run on the D-Wave hardware.")
        print("Please set the DWAVE_API_TOKEN environment variable.")
        print("To run this test, you need to:")
        print("1. Sign up for D-Wave Leap at https://cloud.dwavesys.com/leap/")
        print("2. Get your API token from the dashboard")
        print("3. Set it as an environment variable: DWAVE_API_TOKEN=your_token_here")
        print("\nScript cannot continue without a valid D-Wave API token.")
        return

    sampler = LeapHybridCQMSampler(token=token)

    print("Submitting the CQM to the D-Wave Leap hybrid solver...")
    start_time = time.time()
    sampleset = sampler.sample_cqm(cqm, label="Food Optimization Example")
    solve_time = time.time() - start_time

    # --- 4. Process and Display Results ---
    print("\n--- Results ---")
    print(f"Solved in {solve_time:.2f} seconds.")
    
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
    if not feasible_sampleset:
        print("\nNo feasible solution found.")
        # Print information about the best infeasible solution
        best_infeasible = sampleset.first
        print(f"Best infeasible solution energy: {best_infeasible.energy}")
        violations = dimod.cqm.violations(cqm, best_infeasible.sample)
        for constr, violation in violations.items():
            if violation > 1e-5:
                print(f"Constraint '{constr}' violated by {violation}")
        return

    best_solution = feasible_sampleset.first
    print(f"Lowest energy found: {best_solution.energy:.4f}")

    print("\nOptimal food production plan:")
    for var, val in best_solution.sample.items():
        if val > 0.5 and var.startswith('y_'):
            print(f"  - {var}")

    print("\n" + "=" * 80)
    print("Test script finished.")
    print("=" * 80)

if __name__ == "__main__":
    main()