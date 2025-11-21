#!/usr/bin/env python3
"""
Debug Dantzig-Wolfe Infeasibility

Investigate why RMP is infeasible with initial columns.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from benchmark_utils_decomposed import generate_farm_data, create_config
from src.scenarios import load_food_data
import gurobipy as gp
from gurobipy import GRB


def debug_dw_infeasibility():
    """Debug why Dantzig-Wolfe RMP is infeasible."""
    
    print("="*80)
    print("DEBUGGING DANTZIG-WOLFE INFEASIBILITY")
    print("="*80)
    
    # Generate test problem
    farm_data = generate_farm_data(n_units=5, total_land=100.0)
    foods, food_groups, config = create_config(farm_data['land_data'])
    
    # Load benefits
    _, _, _, base_config = load_food_data('full_family')
    config['benefits'] = {}
    for food in foods:
        weights = config.get('parameters', {}).get('weights', {})
        benefit = sum(
            base_config.get('nutrients', {}).get(food, {}).get(attr, 0) * weight
            for attr, weight in weights.items()
        )
        config['benefits'][food] = benefit if benefit > 0 else 100.0
    
    farms = farm_data['land_data']
    benefits = config['benefits']
    min_planting_area = config.get('parameters', {}).get('minimum_planting_area', {})
    max_planting_area = config.get('parameters', {}).get('maximum_planting_area', {})
    
    print(f"\nProblem: {len(farms)} farms, {len(foods)} foods")
    print(f"Total land: {sum(farms.values()):.2f}")
    
    # Check food group constraints
    print("\nFood Group Constraints:")
    food_group_constraints = config.get('parameters', {}).get('food_group_constraints', {})
    for group_name, constraints in food_group_constraints.items():
        min_foods = constraints.get('min_foods', 0)
        max_foods = constraints.get('max_foods', float('inf'))
        foods_in_group = food_groups.get(group_name, [])
        print(f"  {group_name}: min={min_foods}, max={max_foods}, foods={len(foods_in_group)}")
    
    # Generate initial columns like DW does
    print("\nGenerating initial columns...")
    columns = []
    sorted_foods = sorted(foods, key=lambda f: benefits.get(f, 1.0), reverse=True)
    
    for farm, capacity in farms.items():
        for start_idx in range(0, min(len(foods), 9), 3):
            allocation = {}
            selection = {}
            remaining_capacity = capacity
            
            for food in sorted_foods[start_idx:start_idx+3]:
                min_area = min_planting_area.get(food, 0.0001)
                max_area = min(max_planting_area.get(food, capacity), capacity / 2)
                alloc = min(max_area, remaining_capacity / 2)
                
                if alloc >= min_area and remaining_capacity >= min_area:
                    allocation[(farm, food)] = alloc
                    selection[(farm, food)] = 1.0
                    remaining_capacity -= alloc
            
            if allocation:
                col_obj = sum(allocation.get(key, 0.0) * benefits.get(key[1], 1.0) / 100.0 for key in allocation)
                columns.append({
                    'farm': farm,
                    'allocation': allocation,
                    'selection': selection,
                    'objective': col_obj
                })
    
    print(f"  Created {len(columns)} initial columns")
    
    # Analyze columns
    print("\nColumn Analysis:")
    for k, col in enumerate(columns[:5]):  # Show first 5
        foods_selected = [key[1] for key in col['selection'].keys()]
        print(f"  Column {k}: Farm {col['farm']}, Foods: {foods_selected}")
    
    # Check total selections per food group
    print("\nFood Group Coverage in Initial Columns:")
    for group_name, constraints in food_group_constraints.items():
        foods_in_group = food_groups.get(group_name, [])
        total_selections = 0
        
        for col in columns:
            for key in col['selection'].keys():
                if key[1] in foods_in_group:
                    total_selections += col['selection'][key]
        
        min_required = constraints.get('min_foods', 0)
        print(f"  {group_name}: {total_selections:.0f} selections (min required: {min_required})")
        if total_selections < min_required:
            print(f"    ❌ INFEASIBLE: Need {min_required - total_selections:.0f} more selections!")
    
    # Try to solve RMP
    print("\nTrying to solve RMP...")
    model = gp.Model("RMP_Debug")
    model.setParam('OutputFlag', 1)
    
    # Variables
    lambda_vars = {}
    for k, col in enumerate(columns):
        lambda_vars[k] = model.addVar(lb=0.0, ub=1.0, name=f"lambda_{k}")
    
    # Objective
    obj_expr = gp.quicksum(lambda_vars[k] * col['objective'] for k, col in enumerate(columns))
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Convexity
    model.addConstr(gp.quicksum(lambda_vars[k] for k in lambda_vars) <= len(farms), name="Convexity")
    
    # Land constraints
    for farm in farms:
        farm_usage = gp.quicksum(
            lambda_vars[k] * sum(col['allocation'].get((farm, f), 0.0) for f in foods)
            for k, col in enumerate(columns)
        )
        model.addConstr(farm_usage <= farms[farm], name=f"Land_{farm}")
    
    # Food group constraints
    for group_name, constraints in food_group_constraints.items():
        foods_in_group = food_groups.get(group_name, [])
        min_foods = constraints.get('min_foods', 0)
        
        total_selected = gp.quicksum(
            lambda_vars[k] * sum(col['selection'].get((f, c), 0.0) for f in farms for c in foods_in_group)
            for k, col in enumerate(columns)
        )
        
        if min_foods > 0:
            model.addConstr(total_selected >= min_foods, name=f"FG_Min_{group_name}")
    
    # Solve
    model.optimize()
    
    print(f"\nModel Status: {model.status}")
    
    if model.status == GRB.INFEASIBLE:
        print("\n❌ MODEL IS INFEASIBLE")
        print("\nComputing IIS (Irreducible Inconsistent Subsystem)...")
        model.computeIIS()
        
        print("\nConstraints in IIS:")
        for constr in model.getConstrs():
            if constr.IISConstr:
                print(f"  - {constr.ConstrName}")
        
        print("\n💡 SOLUTION:")
        print("  Option 1: Relax food group min constraints in RMP (soft constraints)")
        print("  Option 2: Generate more diverse initial columns covering all food groups")
        print("  Option 3: Remove food group constraints from RMP entirely")
    
    elif model.status == GRB.OPTIMAL:
        print(f"\n✅ MODEL IS FEASIBLE")
        print(f"Objective: {model.ObjVal:.4f}")


if __name__ == "__main__":
    debug_dw_infeasibility()
