"""
Test script to verify PuLP and Pyomo formulations are identical
by using very tight solver tolerances.
"""
import os
import sys
# Add project root and Benchmark Scripts to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

from Utils.farm_sampler import generate_farms
from src.scenarios import load_food_data
import pulp as pl
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def create_pulp_model_tight(farms, foods, food_groups, config):
    """Create PuLP model with VERY tight tolerances"""
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    synergy_matrix = params.get('synergy_matrix', {})
    synergy_bonus_weight = weights.get('synergy_bonus', 0.1)
    
    # Decision variables
    A_pulp = pl.LpVariable.dicts("Area", [(f, c) for f in farms for c in foods], lowBound=0)
    Y_pulp = pl.LpVariable.dicts("Choose", [(f, c) for f in farms for c in foods], cat='Binary')
    
    # Linearization variables
    Z_pulp = {}
    synergy_pairs = []
    for f in farms:
        for crop1, pairs in synergy_matrix.items():
            if crop1 in foods:
                for crop2, boost_value in pairs.items():
                    if crop2 in foods and crop1 < crop2:
                        Z_pulp[(f, crop1, crop2)] = pl.LpVariable(f"Z_{f}_{crop1}_{crop2}", cat='Binary')
                        synergy_pairs.append((f, crop1, crop2, boost_value))
    
    model = pl.LpProblem("LQ_Test", pl.LpMaximize)
    
    # Objective
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
    
    synergy_terms = []
    for f, crop1, crop2, boost_value in synergy_pairs:
        synergy_terms.append(synergy_bonus_weight * boost_value * Z_pulp[(f, crop1, crop2)])
    
    model += pl.lpSum(objective_terms) + pl.lpSum(synergy_terms), "Objective"
    
    # McCormick constraints
    for f, crop1, crop2, _ in synergy_pairs:
        model += Z_pulp[(f, crop1, crop2)] <= Y_pulp[(f, crop1)]
        model += Z_pulp[(f, crop1, crop2)] <= Y_pulp[(f, crop2)]
        model += Z_pulp[(f, crop1, crop2)] >= Y_pulp[(f, crop1)] + Y_pulp[(f, crop2)] - 1
    
    # Land constraints
    for f in farms:
        model += pl.lpSum([A_pulp[(f, c)] for c in foods]) <= land_availability[f]
    
    # Linking constraints
    for f in farms:
        for c in foods:
            A_min = min_planting_area.get(c, 0)
            model += A_pulp[(f, c)] >= A_min * Y_pulp[(f, c)]
            model += A_pulp[(f, c)] <= land_availability[f] * Y_pulp[(f, c)]
    
    # Food group constraints
    if food_group_constraints:
        for g, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(g, [])
            if foods_in_group:
                for f in farms:
                    if 'min_foods' in constraints:
                        model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]) >= constraints['min_foods']
                    if 'max_foods' in constraints:
                        model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]) <= constraints['max_foods']
    
    print("\n  PuLP Model Stats:")
    print(f"    Variables: {len(A_pulp) + len(Y_pulp) + len(Z_pulp)}")
    print(f"    - Area (A): {len(A_pulp)}")
    print(f"    - Binary (Y): {len(Y_pulp)}")
    print(f"    - Linearization (Z): {len(Z_pulp)}")
    print(f"    Synergy pairs: {len(synergy_pairs)}")
    
    return model, A_pulp, Y_pulp, Z_pulp

def create_pyomo_model(farms, foods, food_groups, config):
    """Create Pyomo model"""
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    synergy_matrix = params.get('synergy_matrix', {})
    synergy_bonus_weight = weights.get('synergy_bonus', 0.1)
    
    model = pyo.ConcreteModel(name="LQ_Test_Pyomo")
    
    model.farms = pyo.Set(initialize=farms)
    model.foods = pyo.Set(initialize=list(foods.keys()))
    
    model.A = pyo.Var(model.farms, model.foods, domain=pyo.NonNegativeReals,
                      bounds=lambda m, f, c: (0, land_availability[f]))
    model.Y = pyo.Var(model.farms, model.foods, domain=pyo.Binary)
    
    def objective_rule(m):
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
        
        # Quadratic synergy
        for f in m.farms:
            for crop1, pairs in synergy_matrix.items():
                if crop1 in foods:
                    for crop2, boost_value in pairs.items():
                        if crop2 in foods and crop1 < crop2:
                            obj += synergy_bonus_weight * boost_value * m.Y[f, crop1] * m.Y[f, crop2]
        return obj
    
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    def land_constraint_rule(m, f):
        return sum(m.A[f, c] for c in m.foods) <= land_availability[f]
    model.land_constraint = pyo.Constraint(model.farms, rule=land_constraint_rule)
    
    def min_area_rule(m, f, c):
        A_min = min_planting_area.get(c, 0)
        return m.A[f, c] >= A_min * m.Y[f, c]
    model.min_area = pyo.Constraint(model.farms, model.foods, rule=min_area_rule)
    
    def max_area_rule(m, f, c):
        return m.A[f, c] <= land_availability[f] * m.Y[f, c]
    model.max_area = pyo.Constraint(model.farms, model.foods, rule=max_area_rule)
    
    if food_group_constraints:
        def min_food_group_rule(m, f, g):
            foods_in_group = food_groups.get(g, [])
            min_foods = food_group_constraints[g].get('min_foods', None)
            if min_foods is not None and foods_in_group:
                return sum(m.Y[f, c] for c in foods_in_group if c in m.foods) >= min_foods
            else:
                return pyo.Constraint.Skip
        
        def max_food_group_rule(m, f, g):
            foods_in_group = food_groups.get(g, [])
            max_foods = food_group_constraints[g].get('max_foods', None)
            if max_foods is not None and foods_in_group:
                return sum(m.Y[f, c] for c in foods_in_group if c in m.foods) <= max_foods
            else:
                return pyo.Constraint.Skip
        
        model.min_food_group = pyo.Constraint(model.farms, list(food_group_constraints.keys()), rule=min_food_group_rule)
        model.max_food_group = pyo.Constraint(model.farms, list(food_group_constraints.keys()), rule=max_food_group_rule)
    
    print("\n  Pyomo Model Stats:")
    print(f"    Variables: {len(list(model.A)) + len(list(model.Y))}")
    print(f"    - Area (A): {len(list(model.A))}")
    print(f"    - Binary (Y): {len(list(model.Y))}")
    
    return model

def test_with_simple():
    """Test with simple scenario"""
    print("="*80)
    print("TESTING WITH SIMPLE SCENARIO")
    print("="*80)
    
    farms, foods, food_groups, config = load_food_data('simple')
    print(f"Farms: {len(farms)}, Foods: {len(foods)}")
    
    # PuLP with VERY tight tolerance
    print("\n" + "-"*80)
    print("PULP (Linearized) - TIGHT TOLERANCE")
    print("-"*80)
    pulp_model, A, Y, Z = create_pulp_model_tight(farms, foods, food_groups, config)
    
    solver = pl.PULP_CBC_CMD(
        msg=1,  # Show output
        timeLimit=600,
        gapRel=0.0001,  # 0.01% gap - VERY tight
        threads=4
    )
    pulp_model.solve(solver)
    pulp_obj = pl.value(pulp_model.objective)
    print(f"\n  PuLP Objective: {pulp_obj:.10f}")
    print(f"  Status: {pl.LpStatus[pulp_model.status]}")
    
    # Pyomo with tight tolerance
    print("\n" + "-"*80)
    print("PYOMO (Quadratic) - TIGHT TOLERANCE")
    print("-"*80)
    pyomo_model = create_pyomo_model(farms, foods, food_groups, config)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(pyomo_model, tee=True, options={'limits/gap': 0.0001})
    pyomo_obj = pyo.value(pyomo_model.obj)
    print(f"\n  Pyomo Objective: {pyomo_obj:.10f}")
    print(f"  Status: {results.solver.status}")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"  PuLP:  {pulp_obj:.10f}")
    print(f"  Pyomo: {pyomo_obj:.10f}")
    print(f"  Diff:  {abs(pulp_obj - pyomo_obj):.10f} ({abs(pulp_obj - pyomo_obj) / pyomo_obj * 100:.6f}%)")
    
    if abs(pulp_obj - pyomo_obj) / pyomo_obj < 0.001:  # Less than 0.1%
        print("\n  ✅ PASS: Objectives match within 0.1%")
        print("  The formulations are equivalent!")
    else:
        print("\n  ⚠️  WARNING: Objectives differ by more than 0.1%")
        print("  This suggests a potential formulation issue.")

if __name__ == "__main__":
    test_with_simple()
