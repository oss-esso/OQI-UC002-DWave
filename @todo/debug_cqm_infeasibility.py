#!/usr/bin/env python3
"""
Debug CQM Infeasibility Issue

Test the Patch CQM formulation to see why it's infeasible.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dimod import ConstrainedQuadraticModel, Binary
import gurobipy as gp
from gurobipy import GRB


def build_simple_patch_cqm(n_patches=5, n_foods=27):
    """Build the exact same CQM as comprehensive benchmark"""
    print(f"Building Patch CQM ({n_patches} patches, {n_foods} foods)...")
    
    cqm = ConstrainedQuadraticModel()
    
    # Variables: binary only
    Y = {}
    for p in range(n_patches):
        for c in range(n_foods):
            Y[p, c] = Binary(f"Y_{p}_{c}")
    
    # Objective: maximize total selections
    objective = sum(Y[p, c] for p in range(n_patches) for c in range(n_foods))
    cqm.set_objective(-objective)
    
    # Constraints
    # 1. Each patch can select up to 5 foods
    for p in range(n_patches):
        cqm.add_constraint(sum(Y[p, c] for c in range(n_foods)) <= 5, label=f"patch_limit_{p}")
    
    # 2. Global minimum selections
    total_selections = sum(Y[p, c] for p in range(n_patches) for c in range(n_foods))
    cqm.add_constraint(total_selections >= n_foods // 2, label="min_foods")
    
    print(f"  Variables: {len(cqm.variables)}")
    print(f"  Constraints: {len(cqm.constraints)}")
    print(f"  Constraint details:")
    for label in cqm.constraints:
        print(f"    - {label}")
    
    return cqm, Y


def solve_with_gurobi(cqm):
    """Solve CQM with Gurobi"""
    print("\nSolving with Gurobi...")
    
    model = gp.Model("Debug_CQM")
    model.setParam('OutputFlag', 1)  # Enable output to see what's happening
    
    # Create variables
    gurobi_vars = {}
    for var_name in cqm.variables:
        gurobi_vars[var_name] = model.addVar(vtype=GRB.BINARY, name=var_name)
    
    # Set objective
    obj_expr = 0
    for var_name, coeff in cqm.objective.linear.items():
        obj_expr += coeff * gurobi_vars[var_name]
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Add constraints
    for label, constraint in cqm.constraints.items():
        constr_expr = 0
        for var_name, coeff in constraint.lhs.linear.items():
            constr_expr += coeff * gurobi_vars[var_name]
        
        sense_str = str(constraint.sense)
        rhs_val = constraint.rhs
        
        if str(constraint.sense) == 'Sense.Le' or constraint.sense == '<=':
            model.addConstr(constr_expr <= constraint.rhs, name=label)
        elif str(constraint.sense) == 'Sense.Ge' or constraint.sense == '>=':
            model.addConstr(constr_expr >= constraint.rhs, name=label)
        elif str(constraint.sense) == 'Sense.Eq' or constraint.sense == '==':
            model.addConstr(constr_expr == constraint.rhs, name=label)
        else:
            print(f"  WARNING: Unknown sense type: {constraint.sense} (type: {type(constraint.sense)})")
        
        # Debug: print constraint details
        num_terms = len(constraint.lhs.linear)
        print(f"  Added constraint '{label}': {num_terms} terms {sense_str} {rhs_val}")
    
    # Update model to process all additions
    model.update()
    
    # Print model details before solving
    print(f"\nModel details before solve:")
    print(f"  Variables: {model.NumVars}")
    print(f"  Constraints: {model.NumConstrs}")
    print(f"  Objective sense: {'MINIMIZE' if model.ModelSense == 1 else 'MAXIMIZE'}")
    
    # Check objective coefficients
    obj_coeffs = [(v.VarName, v.Obj) for v in model.getVars()[:10]]
    print(f"  Sample objective coefficients: {obj_coeffs}")
    
    # Solve
    model.optimize()
    
    print(f"\nSolution status: {model.status}")
    print(f"Status name: {get_status_name(model.status)}")
    
    if model.status == GRB.INFEASIBLE:
        print("\n" + "="*80)
        print("MODEL IS INFEASIBLE - Computing IIS...")
        print("="*80)
        model.computeIIS()
        print("\nIrreducible Inconsistent Subsystem (IIS):")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"  Constraint: {c.ConstrName}")
        print()
    
    elif model.status == GRB.OPTIMAL:
        print(f"\nOptimal objective: {model.objVal:.6f}")
        print(f"Active variables: {sum(1 for v in model.getVars() if v.X > 0.5)}")
        
        # Show some solution details
        selections_by_patch = {p: 0 for p in range(5)}
        for v in model.getVars():
            if v.X > 0.5:
                # Parse Y_p_c
                parts = v.VarName.split('_')
                if len(parts) == 3 and parts[0] == 'Y':
                    patch = int(parts[1])
                    selections_by_patch[patch] += 1
        
        print("\nSelections per patch:")
        for p, count in selections_by_patch.items():
            print(f"  Patch {p}: {count} foods")
        print(f"  Total: {sum(selections_by_patch.values())} foods")
    
    return model


def get_status_name(status_code):
    """Convert Gurobi status code to name"""
    status_map = {
        GRB.OPTIMAL: 'OPTIMAL',
        GRB.INFEASIBLE: 'INFEASIBLE',
        GRB.INF_OR_UNBD: 'INFEASIBLE_OR_UNBOUNDED',
        GRB.UNBOUNDED: 'UNBOUNDED',
        GRB.TIME_LIMIT: 'TIME_LIMIT',
        2: 'OPTIMAL',
        3: 'INFEASIBLE',
        4: 'INF_OR_UNBD',
        5: 'UNBOUNDED',
        9: 'TIME_LIMIT'
    }
    return status_map.get(status_code, f'UNKNOWN_{status_code}')


if __name__ == "__main__":
    print("="*80)
    print("DEBUG: CQM INFEASIBILITY TEST")
    print("="*80)
    
    cqm, Y = build_simple_patch_cqm(n_patches=5, n_foods=27)
    model = solve_with_gurobi(cqm)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
