from dimod import ConstrainedQuadraticModel, Binary, Real
from dwave.system import LeapHybridCQMSampler   # hybrid solver (requires D-Wave account)
import math
import pickle

#Data

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


eps = 1e-6

cqm = ConstrainedQuadraticModel()

A = {}
Y = {}

for f in farms:
    for c in crops:
        A[(f, c)] = Real(f"A_{f}_{c}", lower_bound=0, upper_bound=L[f])
        Y[(f, c)] = Binary(f"Y_{f}_{c}")



# Direct objective formulation: maximize the weighted sum
# Since we're dividing by a constant (total_area), we can maximize the numerator directly
# This is equivalent to maximizing numerator/total_area

total_area = sum(L[f] for f in farms)

objective = sum(
    weights['w_1'] * N[c] * A[(f, c)] +
    weights['w_2'] * D[c] * A[(f, c)] -
    weights['w_3'] * E[c] * A[(f, c)] +
    weights['w_4'] * P[c] * A[(f, c)]
    for f in farms for c in crops
)

#Land availability constraints
for f in farms:
    cqm.add_constraint(
        sum(A[(f, c)] for c in crops) - L[f] <= 0,
        label=f"Land_Availability_{f}"
    )

#Linking A and Y variables
for f in farms:
    for c in crops:
        cqm.add_constraint(A[(f, c)] - A_min[c] * Y[(f, c)] >= 0, label=f"Min_Area_If_Selected_{f}_{c}")  # If Y=1, area must be at least A_min
        cqm.add_constraint(A[(f, c)] - L[f] * Y[(f, c)] <= 0, label=f"Max_Area_If_Selected_{f}_{c}")       # If Y=0, area must be 0


#Food group constraints
for g, crops_group in food_groups.items():
    for f in farms:
        cqm.add_constraint(
            sum(Y[(f, c)] for c in crops_group) - FG_min[g] >= 0,
            label=f"Food_Group_Min_{g}_{f}"
        )
        cqm.add_constraint(
            sum(Y[(f, c)] for c in crops_group) - FG_max[g] <= 0,
            label=f"Food_Group_Max_{g}_{f}"
        )

#Solve the CQM
# Note: CQM minimizes by default, so we negate the objective to maximize
cqm.set_objective(-objective)

sampler = LeapHybridCQMSampler(token = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
sampleset = sampler.sample_cqm(cqm)

# Save the sampleset to a file
with open('DWave_Results/sampleset.pickle', 'wb') as f:
    pickle.dump(sampleset, f)

print("Sampleset saved to DWave_Results/sampleset.pickle")

