---
applyTo: '**'
---

# Project Memory: OQI-UC002-DWave Constraint Violation Analysis

## Key Findings from Diagnostic Analysis (Oct 28, 2025)

### Problem Statement
The PATCH formulation violates constraints when using D-Wave's Hybrid BQM solver, while BQUBO does not. Investigation revealed the root cause.

### Critical Discovery: Coefficient Scale Catastrophe
- **BQUBO**: Quadratic coefficient range = 0.155 (well-scaled)
- **PATCH**: Quadratic coefficient range = 276.26 (1,782x larger!)
- **Energy scales**: BQUBO ~3.5, PATCH ~1,735 (490x larger)

### Root Cause
When `cqm_to_bqm()` converts constraints to penalties:
1. PATCH's heterogeneous patch areas create massive coefficient variation
2. Auto-selected Lagrange multipliers are insufficient
3. Constraint coupling (15.79% overlap vs 3% in BQUBO) amplifies the problem
4. Penalty terms dominate and drown out the objective function

### Solution Direction
**Primary**: Use `LeapHybridCQMSampler` directly (avoid BQM conversion)
**Alternative**: Scale BQUBO to support variable-sized plots (like PATCH) but with proper normalization

### Terminology
- **Farms** (BQUBO) = **Plots** (PATCH) - same concept, different names
- BQUBO uses uniform 1-acre units; PATCH uses heterogeneous areas

## Current Status
- Diagnostic script completed with progress bars
- 10-variable analysis shows clear coefficient scaling issues
- Next: 50-variable run, LaTeX report, scaled BQUBO implementation, Gurobi QUBO comparison script

## Violation Healing Analysis (Jan 21, 2026)

### Context
Applied the same violation "healing" analysis from the rotation scenario to the benchmark data (small/large/comprehensive scales).

### Key Findings

**Feasibility Summary:**
- Total method-scale combinations: 44
- Feasible (0 violations): 28 (63.6%)
- Infeasible (>0 violations): 16 (36.4%)

**Methods Always Feasible:**
- `cqm_first_PlotBased`
- `Louvain_QPU`

**Methods With Some Violations:**
- `Spectral(10)_QPU`, `coordinated`, `Multilevel(10)_QPU`, `PlotBased_QPU`, `Multilevel(5)_QPU`, `HybridGrid(5,9)_QPU`, `HybridGrid(10,9)_QPU`

**Critical Insight:**
Unlike the rotation scenario, in the benchmark data:
- Most infeasible solutions (14/16) have **NEGATIVE gaps** (QPU < Gurobi)
- Only 2 entries have positive gaps (QPU > Gurobi)
- For those 2 positive-gap entries, violations explain **95.1%** of the gap

**Interpretation:**
Violations are NOT helping QPU "cheat" to higher objectives. Instead, violations indicate difficulty in the optimization:
- Problems are harder for QPU to solve
- Both solution quality AND feasibility suffer together
- Feasible solutions have better average performance than infeasible ones

**Analysis Script:** `analyze_benchmark_violation_healing.py`
**Output Files:** 
- `phase3_results_plots/benchmark_violation_healing_analysis.png/pdf`
- `phase3_results_plots/benchmark_violation_healing_table.png`
- `phase3_results_plots/benchmark_violation_healing_data.csv`

## Gurobi Scaling Benchmark (Jan 26, 2026)

### Context
Created benchmark comparing Gurobi performance on continuous (LP) vs binary (MIP) formulations to demonstrate why the original problem is trivially solvable while QUBO conversion creates complexity.

### Key Results

**Continuous LP Formulation (Original Problem):**
- 1,000 vars (36 patches): 0.11s optimal
- 10,000 vars (369 patches): 0.21s optimal
- 100,000 vars (3,703 patches): 1.34s optimal
- 1,000,000 vars (37,036 patches): 26.1s optimal

**Binary MIP Formulation (QUBO-Required):**
- 1,000 vars: 0.12s optimal
- 10,000 vars: 2.12s optimal
- 100,000 vars: 57.96s optimal
- 1,000,000 vars: >1,698s TIMEOUT

**Key Insight:**
- Continuous (LP) scales polynomially - 1M variables in 26 seconds
- Binary (MIP) has exponential worst-case complexity - times out at 1M
- Speedup ratio: 1.1x -> 10x -> 43x -> 65x+
- The complexity barrier comes from discretization, not the problem itself

**Benchmark Script:** `@todo/gurobi_scaling_benchmark.py`
**Output Files:** `@todo/gurobi_scaling_benchmark_*.json`

### Report Updates (content_report.tex)
1. Added Gurobi Performance section with LP vs MIP comparison table
2. Added Violation Impact Analysis by Decomposition Strategy section
