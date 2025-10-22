import pickle
import os
import pulp as pl

# --- Data Definitions (similar to dwave-test.py) ---
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

A_min = {'Wheat': 5, 'Corn': 4, 'Soy': 3, 'Tomato': 2}
L = {'Farm1': 100, 'Farm2': 150}

FG_min = {'Grains': 1, 'Legumes': 1, 'Vegetables': 1}
FG_max = {'Grains': 2, 'Legumes': 1, 'Vegetables': 1}

weights = {'w_1': 0.25, 'w_2': 0.25, 'w_3': 0.25, 'w_4': 0.25}

# --- Load Sampleset ---
file_path = 'DWave_Results/sampleset.pickle'
if not os.path.exists(file_path):
    print(f"Error: Sampleset file not found at {file_path}")
    print("Please run dwave-test.py first to generate the sampleset.")
    exit()

with open(file_path, 'rb') as f:
    sampleset = pickle.load(f)

# --- Analysis Code ---
feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

print(f"{len(feasible_sampleset)} feasible solutions of {len(sampleset)}.")

print("\nBest feasible solution:")
if len(feasible_sampleset) > 0:
    best_sample = feasible_sampleset.first
    
    # Calculate the efficiency from the solution
    N = {'Wheat': 0.7, 'Corn': 0.9, 'Soy': 0.5, 'Tomato': 0.8}
    D = {'Wheat': 0.6, 'Corn': 0.85, 'Soy': 0.55, 'Tomato': 0.9}
    E = {'Wheat': 0.4, 'Corn': 0.3, 'Soy': 0.5, 'Tomato': 0.2}
    P = {'Wheat': 0.7, 'Corn': 0.5, 'Soy': 0.6, 'Tomato': 0.9}
    weights = {'w_1': 0.25, 'w_2': 0.25, 'w_3': 0.25, 'w_4': 0.25}
    total_area = sum(L[f] for f in farms)
    
    numerator = 0
    for f in farms:
        for c in crops:
            a_name = f"A_{f}_{c}"
            if a_name in best_sample.sample:
                a_val = best_sample.sample[a_name]
                numerator += (
                    weights['w_1'] * N[c] * a_val +
                    weights['w_2'] * D[c] * a_val -
                    weights['w_3'] * E[c] * a_val +
                    weights['w_4'] * P[c] * a_val
                )
    
    efficiency = numerator / total_area
    print(f"Optimal Efficiency: {efficiency:.6f}")
    print(f"CQM Energy (negative objective): {best_sample.energy:.6f}\n")

    for f in farms:
        print(f"--- {f} ---")
        for c in crops:
            y_name = f"Y_{f}_{c}"
            a_name = f"A_{f}_{c}"
            
            if y_name in best_sample.sample and a_name in best_sample.sample:
                y_val = best_sample.sample[y_name]
                a_val = best_sample.sample[a_name]
                
                if y_val > 0.5:
                    required_min = A_min[c] * y_val
                    status = "GOOD" if a_val >= required_min - 0.001 else "VIOLATED"
                    print(f"  {f}, {c}: A={a_val:.2f} >= {required_min:.2f} (A_min*Y) {status}")

        for g, crops_group in food_groups.items():
            y_names_in_group = [f"Y_{f}_{c}" for c in crops_group]
            count = sum(1 for y_name in y_names_in_group if y_name in best_sample.sample and best_sample.sample[y_name] > 0.5)
            
            selected_crops = [c for c in crops_group if f"Y_{f}_{c}" in best_sample.sample and best_sample.sample[f"Y_{f}_{c}"] > 0.5]

            status = "GOOD" if FG_min[g] <= count <= FG_max[g] else "VIOLATED"
            print(f"  {g}: {count} selected (range: {FG_min[g]}-{FG_max[g]}) {status}")
            if selected_crops:
                print(f"    Selected crops: {selected_crops}")

# --- Timing Information ---
times = sampleset.info

print("\nDWave Timing Information:")
if isinstance(times, dict):
    for key, value in times.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value*1e-3:.2f} ms")
        else:
            print(f"  {key}: {value}")
else:
    print("No timing information available.")

# --- Solve with PuLP for Comparison ---
print("\n" + "="*80)
print("SOLVING WITH PULP FOR COMPARISON")
print("="*80)

A_pulp = pl.LpVariable.dicts("Area", [(f, c) for f in farms for c in crops], lowBound=0)
Y_pulp = pl.LpVariable.dicts("Choose", [(f, c) for f in farms for c in crops], cat='Binary')

total_area = sum(L[f] for f in farms)

goal = (
    weights['w_1'] * pl.lpSum([(N[c] * A_pulp[(f, c)]) for f in farms for c in crops]) / total_area +
    weights['w_2'] * pl.lpSum([(D[c] * A_pulp[(f, c)]) for f in farms for c in crops]) / total_area -
    weights['w_3'] * pl.lpSum([(E[c] * A_pulp[(f, c)]) for f in farms for c in crops]) / total_area +
    weights['w_4'] * pl.lpSum([(P[c] * A_pulp[(f, c)]) for f in farms for c in crops]) / total_area
)

model = pl.LpProblem("Crop_Allocation_Optimization", pl.LpMaximize)

for f in farms:
    model += pl.lpSum([A_pulp[(f, c)] for c in crops]) <= L[f], f"Max_Area_{f}"

for f in farms:
    for c in crops:
        model += A_pulp[(f, c)] >= A_min[c] * Y_pulp[(f, c)], f"MinArea_{f}_{c}"
        model += A_pulp[(f, c)] <= L[f] * Y_pulp[(f, c)], f"MaxArea_{f}_{c}"

for g, crops_group in food_groups.items():
    for f in farms:
        model += pl.lpSum([Y_pulp[(f, c)] for c in crops_group]) >= FG_min[g], f"MinFoodGroup_{f}_{g}"
        model += pl.lpSum([Y_pulp[(f, c)] for c in crops_group]) <= FG_max[g], f"MaxFoodGroup_{f}_{g}"

model += goal, "Objective"

model.solve(pl.PULP_CBC_CMD(msg=0))

pulp_objective = pl.value(model.objective)
pulp_status = pl.LpStatus[model.status]

print(f"PuLP Status: {pulp_status}")
print(f"PuLP Objective Value: {pulp_objective:.6f}")

# --- Create Comparison Table ---
print("\n" + "="*80)
print("FINAL COMPARISON: DWave vs PuLP")
print("="*80)

print("\nObjective Value:")
print(f"  DWave:  {efficiency:.6f}")
print(f"  PuLP:   {pulp_objective:.6f}")
if abs(efficiency - pulp_objective) < 0.0001:
    print(f"  Status: OK IDENTICAL")
else:
    print(f"  Status: NO DIFFERENT (diff: {abs(efficiency - pulp_objective):.6f})")

for f in farms:
    print(f"\n{f} ({L[f]} ha):")
    print(f"  {'Crop':<10} | {'DWave':<8} | {'PuLP':<8} | {'Match':<6}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    
    for c in crops:
        # Get DWave values
        a_name = f"A_{f}_{c}"
        dwave_val = 0.0
        if a_name in best_sample.sample:
            dwave_val = best_sample.sample[a_name]
        
        # Get PuLP values
        pulp_val = 0.0
        if A_pulp[(f, c)].value() is not None:
            pulp_val = A_pulp[(f, c)].value()
        
        # Check match
        match = "OK" if abs(dwave_val - pulp_val) < 0.01 else "NO"
        
        print(f"  {c:<10} | {dwave_val:<8.2f} | {pulp_val:<8.2f} | {match:<6}")

print("\n" + "="*80)
if abs(efficiency - pulp_objective) < 0.0001:
    print("Both solvers found the SAME solution!")
else:
    print("WARNING: Solutions differ!")
print("="*80)
