# Analysis: DWave CQM Constraint Violations

## Problem

The DWave HybridCQM Sampler is producing solutions with constraint violations:
- 8 plots have 2 crops assigned (should be ≤ 1)
- 4 Y variables activated without corresponding X variables
- 117% land utilization (overallocation!)

## Root Cause

The PATCH formulation uses **binary assignment variables**:
- X_{p,c} = 1 if plot p assigned to crop c
- Constraint: sum_c X_{p,c} <= 1 (at most one crop per plot)

**The issue**: CQM solvers may treat inequality constraints as "soft" during optimization,
especially when they conflict with maximizing the objective. The solver is prioritizing
objective maximization over strict constraint satisfaction.

## Why Grid_Refinement.py Works Better

Grid_Refinement.py uses the **continuous formulation** from solver_runner.py:
- A_{f,c} = continuous area variable (how much area of crop c on farm f)
- Constraint: sum_c A_{f,c} <= land_availability[f]

**Key difference**: 
1. Continuous variables are easier for solvers to handle
2. Area-based formulation is more natural for the problem
3. No discrete assignment conflicts

## Options

### Option 1: Use Continuous Formulation (RECOMMENDED)
- Switch to solver_runner.py formulation for all solvers
- Uses continuous area variables
- More natural for the problem domain
- Better solver behavior

### Option 2: Add Stronger Constraints to PATCH
- Make constraints equality-based where possible
- Add penalty terms to objective
- May still have issues with CQM solver

### Option 3: Use BQM with Proper Lagrange Multipliers
- Convert CQM to BQM with lambda=10.0
- We know this works from testing
- But loses native CQM benefits

## Recommendation

**Switch to continuous formulation (Option 1)**:
1. It's more natural for the problem (allocating area, not assigning plots)
2. Solvers handle it better (no discrete assignment conflicts)
3. Grid_Refinement.py shows this approach works well
4. We can still test grid refinement by varying farm/patch counts

This means:
- Farm scenario: solver_runner.py (continuous areas)
- Patch scenario: ALSO use solver_runner.py, just with more patches
- Grid refinement = testing with 10, 25, 50, 100 patches
