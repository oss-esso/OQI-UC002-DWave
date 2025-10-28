# Gurobi QUBO Comparison Script - User Guide

## Overview

This educational script (`gurobi_qubo_comparison.py`) demonstrates how to solve agricultural optimization problems using Gurobi's quadratic optimizer. It compares 4 different formulations to help you understand the impact of problem structure on solver performance.

## What It Does

The script:
1. **Generates** 4 problem formulations with 10 plots
2. **Converts** each to QUBO (Quadratic Unconstrained Binary Optimization) format
3. **Solves** using Gurobi's optimizer
4. **Compares** results side-by-side with detailed metrics

## The Four Formulations

### 1. BQUBO (Original)
- **Uniform 1-acre plots**
- Simple, well-behaved coefficients
- Always feasible ✓

### 2. Scaled BQUBO (NEW!)
- **Variable-sized plots**
- Benefits scaled by plot area
- Maintains good coefficient properties
- **Recommended approach for variable plots**

### 3. PATCH (with idle penalty)
- Explicit area tracking
- Includes penalty for unused land
- May violate constraints due to coefficient scaling

### 4. PATCH_NO_IDLE
- Same as PATCH but without idle penalty
- Slightly better feasibility

## Prerequisites

```bash
# Install Gurobi (requires license - free for academics)
pip install gurobipy

# Other required packages (should already be installed)
pip install dimod numpy
```

## How to Run

```bash
# From project root directory
conda activate oqi  # or your environment name
python gurobi_qubo_comparison.py
```

## Example Output

```
================================================================================
  GUROBI QUBO COMPARISON: Educational Demo
================================================================================

This script compares 4 agricultural optimization formulations:
  1. BQUBO - Original (uniform 1-acre plots)
  2. Scaled BQUBO - New (variable-sized plots with benefit scaling)
  3. PATCH - Plot assignment (with idle penalty)
  4. PATCH_NO_IDLE - Plot assignment (no idle penalty)

...

  Summary Table:
  ------------------------------------------------------------------------------------------
  Formulation          Status     Energy          Time (s)   Feasible  
  ------------------------------------------------------------------------------------------
  bqubo                optimal    -2.456789       0.012      ✓ YES     
  scaled_bqubo         optimal    -5.234567       0.015      ✓ YES     
  patch                optimal    12.345678       0.023      ✗ NO      
  patch_no_idle        optimal    8.765432        0.019      ✗ NO      
  ------------------------------------------------------------------------------------------
```

## Understanding the Results

### Metrics Explained

- **Status**: Whether Gurobi found an optimal solution
- **Energy**: BQM objective value (lower is better)
- **Time**: Solve time in seconds
- **Feasible**: Whether the solution satisfies all CQM constraints

### Coefficient Scaling

The script shows coefficient ranges for each formulation:

```
Coefficient Scaling Comparison:
  BQUBO          : Range = 0.15, Mean = 0.17, Std = 0.05
  Scaled BQUBO   : Range = 2.45, Mean = 0.52, Std = 0.68
  PATCH          : Range = 276.26, Mean = 0.54, Std = 37.20
  PATCH_NO_IDLE  : Range = 227.37, Mean = 0.44, Std = 30.62
```

**Key Insight**: PATCH has 1,782× larger coefficient range than BQUBO! This causes constraint violations.

## Learning Points

### 1. Why Scaled BQUBO Works Better

**PATCH Approach** (Problem):
```python
# Area appears in constraint: sum(area[p] * x[p,c])
# When squared for penalty: area[p] * area[q] → O(area²)
# Result: Coefficient explosion!
```

**Scaled BQUBO Approach** (Solution):
```python
# Scale objective: benefit[c] * area[p] * x[p,c]
# Area appears linearly, not quadratically
# Result: Controlled coefficient growth
```

### 2. QUBO Formulation Basics

A QUBO problem has the form:
```
minimize: x^T Q x
where x ∈ {0,1}^n
```

The script shows how to:
- Extract linear terms (h_i)
- Extract quadratic terms (J_ij)
- Build Gurobi model
- Solve and extract solution

### 3. Constraint Feasibility

Just because a BQM solution has low energy doesn't mean it's feasible! The script demonstrates:
- Checking CQM constraints
- Identifying violations
- Understanding penalty term trade-offs

## Code Structure

```python
def create_scaled_bqubo_cqm(...):
    # Shows how to build Scaled BQUBO formulation
    
def bqm_to_gurobi_qubo(...):
    # Demonstrates QUBO conversion and solving
    
def check_cqm_feasibility(...):
    # Shows how to validate constraints
    
def main():
    # Orchestrates the full comparison
```

## Common Issues

### "Gurobi not available"
- Install Gurobi: `pip install gurobipy`
- Get free academic license: https://www.gurobi.com/academia/

### "Could not import project modules"
- Run from project root directory
- Ensure all dependencies installed: `pip install -r requirements.txt`

### All formulations show infeasible
- This is expected for PATCH formulations
- BQUBO and Scaled BQUBO should be feasible
- If not, check problem generation

## Next Steps

1. **Modify** plot sizes in `generate_patches()` to see impact
2. **Adjust** weights in configuration to change objectives
3. **Add** your own formulation to the comparison
4. **Scale up** to more plots (change `n_plots = 10` to larger number)

## References

- **Technical Report**: See `technical_report.tex` for detailed analysis
- **Diagnostic Script**: See `diagnose_bqm_constraint_violations.py` for comprehensive testing
- **Memory File**: See `.github/instructions/memory.instruction.md` for project findings

## Questions?

The script is heavily commented to be educational. Read through the code to understand:
- How QUBOs work
- How Gurobi solves them
- Why coefficient scaling matters
- How to validate solutions

Happy optimizing! 🚀
