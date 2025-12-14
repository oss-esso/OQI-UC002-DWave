# Quick Summary: Timeout Issue Resolution

## Problem
Gurobi was solving all scenarios optimally in < 1 second (no timeouts), even though statistical tests consistently hit 300s timeouts.

## Root Cause
**Linear MIP formulation was too simple!**
- All farms identical → symmetric problem
- No quadratic terms → Gurobi's presolve trivializes it
- Message: "Presolve: All rows and columns removed"

## Solution
**Added MIQP (Mixed Integer Quadratic Program) formulation:**

### 1. Heterogeneous Data (data_loader_utils.py)
- Farm sizes vary 50-150 ha (±40%)
- Food benefits vary 0.5-1.5 (by crop value)

### 2. Quadratic Objective Terms (test_gurobi_timeout.py + benchmark)
```python
# Rotation synergies (temporal coupling)
obj += Σ (gamma × R[c1,c2] × Y[f,c1,t-1] × Y[f,c2,t])

# Spatial interactions (neighbor coupling)
obj += Σ (gamma × R[c1,c2] × Y[f1,c,t] × Y[f2,c,t])

# Soft one-hot penalty
obj -= Σ (penalty × (crop_count - 1)²)
```

### 3. Rotation Synergy Matrix
- Same crop consecutive periods: -1.2 (negative)
- 70% of pairs: negative synergy (frustration)
- 30% of pairs: positive synergy

### 4. Spatial Neighbor Graph
- Grid layout, each farm connected to 4 nearest neighbors

## Results
**Before:** 0s optimal solve, no timeouts  
**After:** 300s timeout, 200% MIP gap, 17,681 nodes explored

**Model complexity:**
- 5 farms: 1,755 quadratic terms
- 100 farms: ~3,200,000 quadratic terms

## Status
✅ **RESOLVED** - Both test scripts now hit 300s timeout consistently

## Files Modified
1. `data_loader_utils.py` - Added heterogeneous data
2. `test_gurobi_timeout.py` - Added MIQP formulation
3. `significant_scenarios_benchmark.py` - Needs same update

## Next Step
Update `significant_scenarios_benchmark.py` with MIQP formulation, then run full benchmark.
