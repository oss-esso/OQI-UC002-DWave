"""Debug script to see raw Pyomo variable values from Gurobi."""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import pyomo.environ as pyo
from src.scenarios import load_food_data

# Load scenario
farms, foods, food_groups, config = load_food_data('simple')
land_availability = config['parameters']['land_availability']

# Create simple model
model = pyo.ConcreteModel()
model.farms = pyo.Set(initialize=farms)
model.foods = pyo.Set(initialize=list(foods.keys()))
model.A = pyo.Var(model.farms, model.foods, domain=pyo.NonNegativeReals)
model.Y = pyo.Var(model.farms, model.foods, domain=pyo.Binary)

# Simple objective: maximize area of Soybeans only
def obj_rule(m):
    return sum(m.A[f, 'Soybeans'] for f in m.farms)
model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

# Land constraints
def land_rule(m, f):
    return sum(m.A[f, c] for c in m.foods) <= land_availability[f]
model.land_con = pyo.Constraint(model.farms, rule=land_rule)

# Linking: A >= epsilon * Y (forces Y=0 when A=0)
# Linking: A <= M * Y (forces A=0 when Y=0)
def link_min_rule(m, f, c):
    epsilon = 0.001  # Small minimum area
    return m.A[f, c] >= epsilon * m.Y[f, c]
def link_max_rule(m, f, c):
    return m.A[f, c] <= land_availability[f] * m.Y[f, c]
model.link_min_con = pyo.Constraint(model.farms, model.foods, rule=link_min_rule)
model.link_max_con = pyo.Constraint(model.farms, model.foods, rule=link_max_rule)

# Solve
solver = pyo.SolverFactory('gurobi')
results = solver.solve(model, tee=False)

print("Raw values from Gurobi:")
print("-" * 60)
for f in farms:
    for c in list(foods.keys()):
        a_val = pyo.value(model.A[f, c])
        y_val = pyo.value(model.Y[f, c])
        if a_val is not None and a_val > 1e-10:
            print(f"  {f}_{c}: A={a_val:.10f}, Y={y_val:.10f}")
        elif y_val is not None and y_val > 1e-10:
            print(f"  {f}_{c}: A={a_val if a_val else 0:.10f}, Y={y_val:.10f}")

print("\nSummary:")
total_area = sum(pyo.value(model.A[f, c]) or 0 for f in farms for c in foods)
selected_count = sum(1 for f in farms for c in foods if (pyo.value(model.Y[f, c]) or 0) > 0.5)
print(f"Total area: {total_area:.2f}")
print(f"Crops selected (Y>0.5): {selected_count}")
