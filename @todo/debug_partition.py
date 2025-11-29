"""Quick debug script for partition solver"""
import sys
sys.path.insert(0, '..')

from test_cqm_partition_benchmark import *

# Load data and build CQM  
print("Loading data...")
data = load_problem_data(10)
cqm, Y, U = build_cqm(data)

print(f"Foods: {len(data['foods'])}")
print(f"Patches: {len(data['patch_names'])}")

# Test 'None' partition (all variables)
partitions, name = partition_none(cqm, data)
print(f"\nPartition has {len(partitions[0])} variables")
print(f"CQM has {len(cqm.variables)} variables")

# Solve partition
print("\nSolving partition...")
result = solve_partition_independently(cqm, partitions[0], data, timeout=30)
print(f"Result: {result}")
print(f"Result success: {result['success']}")
if 'objective' in result:
    print(f"Result objective: {result['objective']}")
if 'error' in result:
    print(f"Error: {result['error']}")

# Check solution
sol = result['solution']
y_set = sum(1 for k, v in sol.items() if k.startswith('Y_') and v == 1)
u_set = sum(1 for k, v in sol.items() if k.startswith('U_') and v == 1)
print(f"Y vars set to 1: {y_set}")
print(f"U vars set to 1: {u_set}")

# Check what foods were selected
selected_y = [k for k, v in sol.items() if k.startswith('Y_') and v == 1]
selected_u = [k for k, v in sol.items() if k.startswith('U_') and v == 1]
print(f"\nSelected Y: {selected_y[:5]}...")
print(f"Selected U: {selected_u[:5]}...")

# Now solve the ground truth and compare
print("\n\nSolving ground truth...")
gt = solve_full_cqm_gurobi(cqm, data, timeout=60)
print(f"GT objective: {gt['objective']}")

gt_sol = gt['solution']
gt_y_set = sum(1 for k, v in gt_sol.items() if k.startswith('Y_') and v == 1)
gt_u_set = sum(1 for k, v in gt_sol.items() if k.startswith('U_') and v == 1)
print(f"GT Y vars set to 1: {gt_y_set}")
print(f"GT U vars set to 1: {gt_u_set}")

gt_selected_y = [k for k, v in gt_sol.items() if k.startswith('Y_') and v == 1]
gt_selected_u = [k for k, v in gt_sol.items() if k.startswith('U_') and v == 1]
print(f"\nGT Selected Y: {gt_selected_y[:5]}...")
print(f"GT Selected U: {gt_selected_u[:5]}...")
