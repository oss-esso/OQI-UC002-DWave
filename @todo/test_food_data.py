#!/usr/bin/env python3
"""
Test actual food data loading and Gurobi solve to identify constraint issues.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scenarios import load_food_data
from Utils import patch_sampler
import gurobipy as gp
from gurobipy import GRB

print("="*80)
print("Testing actual food data loading")
print("="*80)

# Test loading different scenarios
scenarios_to_test = [
    ('full_family', 4),  # Standard load_problem_data approach
    ('micro_6', None),   # Synthetic scenario
    ('tiny_24', None),   # Synthetic scenario  
]

for scenario, n_farms in scenarios_to_test:
    print(f"\n{'='*80}")
    print(f"Scenario: {scenario}")
    print('='*80)
    
    try:
        if n_farms:
            # Load using standard approach
            _, foods, food_groups, config = load_food_data(scenario)
            land_availability = patch_sampler.generate_grid(n_farms, area=100.0, seed=42)
            params = config.get('parameters', {})
        else:
            # Load from scenario config
            farms, foods, food_groups, config = load_food_data(scenario)
            params = config.get('parameters', {})
            land_availability = params.get('land_availability', {f: 10.0 for f in farms})
        
        farm_names = list(land_availability.keys())
        food_names = list(foods.keys())
        
        print(f"Farms: {len(farm_names)}")
        print(f"Foods: {len(food_names)}")
        print(f"Food groups: {list(food_groups.keys())}")
        print(f"Total area: {sum(land_availability.values()):.2f}")
        
        # Get food group constraints from params
        food_group_constraints = params.get('food_group_constraints', {
            group: {'min_foods': 1, 'max_foods': len(foods_in_group)}
            for group, foods_in_group in food_groups.items()
        })
        
        print(f"\nFood group constraints:")
        for group, constraint in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            print(f"  {group}: min={constraint.get('min_foods', 0)}, max={constraint.get('max_foods', len(foods_in_group))}, available={len(foods_in_group)}")
        
        # Build simple Gurobi model
        model = gp.Model(f"test_{scenario}")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 10
        
        # Variables
        Y = {(f, c): model.addVar(vtype=GRB.BINARY) for f in farm_names for c in food_names}
        U = {c: model.addVar(vtype=GRB.BINARY) for c in food_names}
        
        # Objective (simple: maximize number of farms used)
        obj = gp.quicksum(Y[(f, c)] for f in farm_names for c in food_names)
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # Constraint 1: At most one crop per farm
        for f in farm_names:
            model.addConstr(gp.quicksum(Y[(f, c)] for c in food_names) <= 1, name=f"one_per_farm_{f}")
        
        # Constraint 2: U-Y linking
        for c in food_names:
            for f in farm_names:
                model.addConstr(Y[(f, c)] <= U[c], name=f"link_{f}_{c}")
            model.addConstr(U[c] <= gp.quicksum(Y[(f, c)] for f in farm_names), name=f"bound_{c}")
        
        # Constraint 3: Food group constraints
        for group_name, limits in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            if foods_in_group:
                gs = gp.quicksum(U[f] for f in foods_in_group if f in U)
                min_foods = limits.get('min_foods', limits.get('min', 0))
                max_foods = limits.get('max_foods', limits.get('max', len(foods_in_group)))
                
                if min_foods > 0:
                    model.addConstr(gs >= min_foods, name=f"min_{group_name}")
                if max_foods < len(foods_in_group):
                    model.addConstr(gs <= max_foods, name=f"max_{group_name}")
        
        print(f"\nModel: {model.NumVars} vars, {model.NumConstrs} constraints")
        
        # Solve
        model.optimize()
        
        print(f"Status: {model.Status}")
        
        if model.Status == GRB.OPTIMAL:
            print(f"✓ OPTIMAL: obj={model.ObjVal:.2f}")
            
            # Show solution
            farms_used = sum(1 for f in farm_names if any(Y[(f, c)].X > 0.5 for c in food_names))
            foods_used = [c for c in food_names if U[c].X > 0.5]
            
            print(f"  Farms used: {farms_used}/{len(farm_names)}")
            print(f"  Foods used: {len(foods_used)}/{len(food_names)}")
            
            # Check food group constraints
            for group_name, limits in food_group_constraints.items():
                foods_in_group = food_groups.get(group_name, [])
                if foods_in_group:
                    count = sum(1 for f in foods_in_group if U.get(f) and U[f].X > 0.5)
                    min_req = limits.get('min_foods', 0)
                    max_req = limits.get('max_foods', len(foods_in_group))
                    status = "✓" if min_req <= count <= max_req else "✗"
                    print(f"  {status} {group_name}: {count} (req: {min_req}-{max_req})")
                    
        elif model.Status == GRB.INFEASIBLE:
            print(f"✗ INFEASIBLE - Computing IIS...")
            model.computeIIS()
            print("Conflicting constraints:")
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"  • {c.ConstrName}")
        else:
            print(f"✗ Status: {model.Status}")
        
    except Exception as e:
        print(f"✗ Error loading {scenario}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("This test checks if the food data and constraints are compatible with Gurobi.")
print("If INFEASIBLE, the food group constraints may be too strict for the problem.")
