import sys
import os
import time
import pulp as pl
import numpy as np
from typing import Dict, List, Tuple, Any

farms = ['Farm1', 'Farm2']
crops = ['Wheat', 'Corn', 'Soy', 'Tomato']

food_groups = {
    'Grains': ['Wheat', 'Corn'],
    'Legumes': ['Soy'],
    'Vegetables': ['Tomato']
}

N = {'Wheat': 0.7, 'Corn': 0.9, 'Soy': 0.5, 'Tomato': 0.8}
D = {'Wheat': 0.6, 'Corn': 0.85, 'Soy': 0.55, 'Tomato': 0.9}
E = {'Wheat': 0.4, 'Corn': 0.3, 'Soy': 0.5, 'Tomato': 0.2}
P = {'Wheat': 0.7, 'Corn': 0.5, 'Soy': 0.6, 'Tomato': 0.9}

L = {'Farm1': 100, 'Farm2': 150}
A_min = {'Wheat': 5, 'Corn': 4, 'Soy': 3, 'Tomato': 2}

FG_min = {'Grains': 1, 'Legumes': 1, 'Vegetables': 1}
FG_max = {'Grains': 2, 'Legumes': 1, 'Vegetables': 1}

weights = {'w_1': 0.25, 'w_2': 0.25, 'w_3': 0.25, 'w_4': 0.25}


A = pl.LpVariable.dicts("Area", [(f, c) for f in farms for c in crops], lowBound=0)
Y = pl.LpVariable.dicts("Choose", [(f, c) for f in farms for c in crops], cat='Binary')

total_area = pl.lpSum(L[f] for f in farms)

goal = (
    weights['w_1'] * pl.lpSum([(N[c] * A[(f, c)]) for f in farms for c in crops]) / total_area +
    weights['w_2'] * pl.lpSum([(D[c] * A[(f, c)]) for f in farms for c in crops]) / total_area -
    weights['w_3'] * pl.lpSum([(E[c] * A[(f, c)]) for f in farms for c in crops]) / total_area +
    weights['w_4'] * pl.lpSum([(P[c] * A[(f, c)]) for f in farms for c in crops]) / total_area
)

model = pl.LpProblem("Crop_Allocation_Optimization", pl.LpMaximize)

for f in farms:
    model += pl.lpSum([A[(f, c)] for c in crops]) <= L[f], f"Max_Area_{f}"

# Link Y (binary selection) with A (continuous area)
for f in farms:
    for c in crops:
        model += A[(f, c)] >= A_min[c] * Y[(f, c)], f"MinArea_{f}_{c}"
        model += A[(f, c)] <= L[f] * Y[(f, c)], f"MaxArea_{f}_{c}"

for g, crops_group in food_groups.items():
    for f in farms:
        model += pl.lpSum([Y[(f, c)] for c in crops_group]) >= FG_min[g], f"MinFoodGroup_{f}_{g}"
        model += pl.lpSum([Y[(f, c)] for c in crops_group]) <= FG_max[g], f"MaxFoodGroup_{f}_{g}"

model += goal, "Objective"

model.solve(pl.PULP_CBC_CMD(msg=0))

print("="*80)
print("COMPLETE SOLUTION DUMP")
print("="*80)
print(f"Status: {pl.LpStatus[model.status]}")
print(f"Objective Value: {pl.value(model.objective):.6f}")

print("\n" + "="*80)
print("ALL VARIABLES")
print("="*80)

for f in farms:
    print(f"\n{f}:")
    print(f"{'Crop':<10} {'Y (binary)':<15} {'A (area)':<15} {'A_min':<10}")
    print("-" * 50)
    for c in crops:
        y_val = Y[(f, c)].value() if Y[(f, c)].value() is not None else 0.0
        a_val = A[(f, c)].value() if A[(f, c)].value() is not None else 0.0
        print(f"{c:<10} {y_val:<15.2f} {a_val:<15.2f} {A_min[c]:<10}")

print("\n" + "="*80)
print("CONSTRAINT VERIFICATION")
print("="*80)

# Check if linking constraints are satisfied
print("\nLinking Constraint Check (A >= A_min * Y):")
for f in farms:
    for c in crops:
        y_val = Y[(f, c)].value() if Y[(f, c)].value() is not None else 0.0
        a_val = A[(f, c)].value() if A[(f, c)].value() is not None else 0.0
        required_min = A_min[c] * y_val
        status = "GOOD" if a_val >= required_min - 0.001 else "VIOLATED"
        if y_val > 0.5:  # Only show selected crops
            print(f"  {f}, {c}: A={a_val:.2f} >= {required_min:.2f} (A_min*Y) {status}")

print("\nFood Group Constraints:")
for f in farms:
    print(f"\n{f}:")
    for g, crops_group in food_groups.items():
        count = sum(1 for c in crops_group if Y[(f, c)].value() and Y[(f, c)].value() > 0.5)
        selected = [c for c in crops_group if Y[(f, c)].value() and Y[(f, c)].value() > 0.5]
        status = "GOOD" if FG_min[g] <= count <= FG_max[g] else "VIOLATED"
        print(f"  {g}: {count} selected (range: {FG_min[g]}-{FG_max[g]}) {status}")
        print(f"    Selected crops: {selected}")
