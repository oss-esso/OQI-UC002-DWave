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
