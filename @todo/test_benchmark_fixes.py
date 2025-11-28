#!/usr/bin/env python3
"""
Quick test of the benchmark fixes for CQM and BQM solving.
Tests only 5 farms with a few configurations to verify fixes.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, cqm_to_bqm
import gurobipy as gp
from gurobipy import GRB


def build_patch_cqm(n_patches: int, n_foods: int = 27):
    """Build Patch CQM with binary-only selections"""
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
    for p in range(n_patches):
        cqm.add_constraint(sum(Y[p, c] for c in range(n_foods)) <= 5, label=f"patch_limit_{p}")
    
    total_selections = sum(Y[p, c] for p in range(n_patches) for c in range(n_foods))
    cqm.add_constraint(total_selections >= n_foods // 2, label="min_foods")
    
    return cqm


def solve_cqm_with_gurobi(cqm, timeout=60):
    """Solve CQM with Gurobi - FIXED version"""
    print(f"  Solving CQM ({len(cqm.variables)} vars, {len(cqm.constraints)} constraints)...")
    
    model = gp.Model("CQM")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    
    # Create variables
    gurobi_vars = {}
    for var_name in cqm.variables:
        gurobi_vars[var_name] = model.addVar(vtype=GRB.BINARY, name=var_name)
    
    # Set objective
    obj_expr = 0
    for var_name, coeff in cqm.objective.linear.items():
        obj_expr += coeff * gurobi_vars[var_name]
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Add constraints - FIXED: handle Sense enum
    for label, constraint in cqm.constraints.items():
        constr_expr = 0
        for var_name, coeff in constraint.lhs.linear.items():
            constr_expr += coeff * gurobi_vars[var_name]
        
        sense_str = str(constraint.sense)
        if sense_str == 'Sense.Le' or constraint.sense == '<=':
            model.addConstr(constr_expr <= constraint.rhs, name=label)
        elif sense_str == 'Sense.Ge' or constraint.sense == '>=':
            model.addConstr(constr_expr >= constraint.rhs, name=label)
        elif sense_str == 'Sense.Eq' or constraint.sense == '==':
            model.addConstr(constr_expr == constraint.rhs, name=label)
    
    # Solve
    start = time.time()
    model.optimize()
    solve_time = time.time() - start
    
    has_solution = model.SolCount > 0
    is_optimal = model.status == GRB.OPTIMAL
    
    print(f"    Status: {'OPTIMAL' if is_optimal else model.status}, "
          f"Time: {solve_time:.2f}s, "
          f"Obj: {model.objVal if has_solution else 'N/A'}")
    
    return {
        "success": is_optimal,
        "has_solution": has_solution,
        "solve_time": solve_time,
        "objective": model.objVal if has_solution else None,
        "status": model.status
    }


def solve_bqm_with_gurobi(bqm, timeout=60):
    """Solve BQM with Gurobi - IMPROVED version"""
    print(f"  Solving BQM ({len(bqm.variables)} vars, {len(bqm.quadratic)} quad)...")
    
    model = gp.Model("BQM")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    
    # Create binary variables
    gurobi_vars = {var: model.addVar(vtype=GRB.BINARY, name=var) for var in bqm.variables}
    
    # Build objective
    obj_expr = bqm.offset
    for var, coeff in bqm.linear.items():
        obj_expr += coeff * gurobi_vars[var]
    for (v1, v2), coeff in bqm.quadratic.items():
        obj_expr += coeff * gurobi_vars[v1] * gurobi_vars[v2]
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Solve
    start = time.time()
    model.optimize()
    solve_time = time.time() - start
    
    has_solution = model.SolCount > 0
    is_optimal = model.status == GRB.OPTIMAL
    is_time_limit = model.status == GRB.TIME_LIMIT
    
    status_str = "OPTIMAL" if is_optimal else ("TIME_LIMIT" if is_time_limit else f"STATUS_{model.status}")
    print(f"    Status: {status_str}, Time: {solve_time:.2f}s, "
          f"Obj: {model.objVal if has_solution else 'N/A'}, "
          f"Solutions: {model.SolCount}")
    
    return {
        "success": is_optimal,
        "has_solution": has_solution,
        "is_time_limit": is_time_limit,
        "solve_time": solve_time,
        "objective": model.objVal if has_solution else None,
        "status": model.status,
        "solution_count": model.SolCount
    }


if __name__ == "__main__":
    print("=" * 80)
    print("QUICK TEST: CQM and BQM Solving Fixes")
    print("=" * 80)
    
    # Test 1: CQM (should now work!)
    print("\n[TEST 1] CQM with 5 patches...")
    cqm = build_patch_cqm(5)
    result = solve_cqm_with_gurobi(cqm)
    assert result["success"], f"CQM should be OPTIMAL, got status {result['status']}"
    print("  ✓ CQM PASSED")
    
    # Test 2: Convert to BQM and solve
    print("\n[TEST 2] BQM (from CQM) with 5 patches...")
    cqm = build_patch_cqm(5)
    print("  Converting CQM to BQM...")
    bqm_result = cqm_to_bqm(cqm, lagrange_multiplier=10.0)
    bqm = bqm_result[0] if isinstance(bqm_result, tuple) else bqm_result
    print(f"    BQM has {len(bqm.variables)} vars, {len(bqm.quadratic)} quadratic terms")
    result = solve_bqm_with_gurobi(bqm, timeout=30)
    print(f"  {'✓' if result['success'] or result['has_solution'] else '✗'} BQM {'PASSED' if result['success'] else 'has solution' if result['has_solution'] else 'FAILED'}")
    
    # Test 3: Larger problem (may timeout)
    print("\n[TEST 3] BQM with 10 patches (may hit timeout)...")
    cqm = build_patch_cqm(10)
    print("  Converting CQM to BQM...")
    bqm_result = cqm_to_bqm(cqm, lagrange_multiplier=10.0)
    bqm = bqm_result[0] if isinstance(bqm_result, tuple) else bqm_result
    print(f"    BQM has {len(bqm.variables)} vars, {len(bqm.quadratic)} quadratic terms")
    result = solve_bqm_with_gurobi(bqm, timeout=30)
    if result['success']:
        print("  ✓ BQM OPTIMAL")
    elif result['has_solution']:
        print("  ~ BQM has feasible solution (not proven optimal)")
    else:
        print("  ✗ BQM no solution found")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
