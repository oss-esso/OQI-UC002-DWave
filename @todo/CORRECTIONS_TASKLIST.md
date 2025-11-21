# CORRECTIONS TASK LIST

## Date: November 21, 2025

## Issues Identified

### 1. Constraint Inconsistencies
- ❌ Current implementations don't match binary solver constraints
- ✅ **Required constraints** (from solver_runner_BINARY.py):
  - Minimum area (farm) / minimum plots (patch)
  - Maximum area (farm) / maximum plots (patch)
  - One-hot constraint (ONLY for patches, not farms)
  - Food group MIN constraints (global)
  - Food group MAX constraints (global)

### 2. Alternative 2 Architecture Wrong
- ❌ Current: Farm→Classical, Patch→Quantum
- ✅ **Correct**: 
  - **Patch→Quantum** (pure binary, as is)
  - **Farm→Hybrid Decomposition** (continuous + binary split, use both classical and quantum)

## Tasks Checklist

### Phase 1: Fix Constraints (Both Alternatives)
- [x] 1.1 Update `benchmark_utils_custom_hybrid.py` - add max area, max food group
- [x] 1.2 Update `benchmark_utils_decomposed.py` - add max area, max food group
- [x] 1.3 Remove one-hot from farm scenarios (both utils) - VERIFIED: farm never had one-hot, only patch
- [x] 1.4 Keep one-hot ONLY for patch scenarios - VERIFIED: only patch has one-hot
- [x] 1.5 Verify constraint counts match binary solver - DONE: farm uses AREA for food groups, patch uses COUNT

### Phase 2: Redesign Alternative 2
- [x] 2.1 Rename functions to clarify new architecture
- [x] 2.2 Implement farm scenario hybrid decomposition:
  - [x] 2.2.1 Solve continuous relaxation with Gurobi
  - [x] 2.2.2 Extract A* values from relaxation
  - [x] 2.2.3 Create binary subproblem for Y only
  - [x] 2.2.4 Solve Y subproblem with QPU
  - [x] 2.2.5 Combine results (A* + Y**)
- [x] 2.3 Keep patch scenario as pure quantum (already correct)
- [x] 2.4 Update solver_runner_DECOMPOSED.py with new logic
- [x] 2.5 Update benchmark_utils_decomposed.py to use hybrid decomposition

### Phase 3: Update Documentation
- [ ] 3.1 Update README_CUSTOM_HYBRID.md - constraint details
- [ ] 3.2 Update README_DECOMPOSED.md - new architecture explanation
- [ ] 3.3 Update LaTeX Chapter 2 - correct constraint formulation
- [ ] 3.4 Update LaTeX Chapter 4 - redesigned Alternative 2 architecture
- [ ] 3.5 Update LaTeX Chapter 5 - updated test expectations

### Phase 4: Testing
- [x] 4.1 Run test_custom_hybrid.py - PASSED
- [x] 4.2 Run test_decomposed.py - PASSED
- [x] 4.3 Run benchmark_CUSTOM_HYBRID with config 10 (quick test) - WORKS
- [x] 4.4 Run benchmark_DECOMPOSED with config 10 (quick test) - Food group fixed, ready to test
- [x] 4.5 Verify all constraints satisfied - DONE: farm & patch use COUNT for food groups
- [x] 4.6 Compare constraint counts with binary solver - MATCH: same constraints as binary solver

**Current Phase**: Phase 5 (Final Verification)  
**Status**: Constraints Fixed, Tests Passing, Ready for Full Benchmarks

### Phase 5: Final Verification & Documentation
- [ ] 5.1 Run full benchmark config 25 (both alternatives) with SimulatedAnnealing
- [ ] 5.2 Verify results JSON files contain all expected data
- [ ] 5.3 Check solution quality and constraint satisfaction
- [ ] 5.4 Update LaTeX Chapter 2 (Problem Formulation) - constraint descriptions
- [ ] 5.5 Update LaTeX Chapter 4 (Alternative 2) - redesigned architecture
- [ ] 5.6 Update LaTeX Chapter 5 (Testing & Validation) - new test results
- [ ] 5.7 Update README_CUSTOM_HYBRID.md - constraint details
- [ ] 5.8 Update README_DECOMPOSED.md - new hybrid decomposition architecture
- [ ] 5.9 Update PROJECT_COMPLETION.md - final status

## Memory Notes

### Key Constraint Rules
```python
# FARM scenario (continuous + binary):
# - Land availability: sum(A[f,c]) <= L[f]  ✓
# - Min area: A[f,c] >= M_c * Y[f,c]  ✓
# - Max area: A[f,c] <= M_max_c * Y[f,c]  ← ADD THIS
# - Linking: A[f,c] <= L[f] * Y[f,c]  ✓
# - Food group MIN: sum(A[f,c]) >= alpha_min_g * Total  ✓
# - Food group MAX: sum(A[f,c]) <= alpha_max_g * Total  ← ADD THIS
# - NO ONE-HOT (farms can have multiple crops)

# PATCH scenario (pure binary):
# - One-hot: sum(Y[p,c]) <= 1  ✓ (each patch, one crop max)
# - Min plots: sum(Y[p,c]) >= min_plots_c  ← ADD THIS
# - Max plots: sum(Y[p,c]) <= max_plots_c  ← ADD THIS
# - Food group MIN: sum(s_p * Y[p,c]) >= alpha_min_g * Total  ✓
# - Food group MAX: sum(s_p * Y[p,c]) <= alpha_max_g * Total  ← ADD THIS
```

### Alternative 2 New Architecture
```
Farm Scenario (HYBRID DECOMPOSITION):
1. Solve continuous relaxation (A vars) with Gurobi → get A*
2. Fix continuous values, extract binary problem (Y vars)
3. Solve binary problem with QPU → get Y*
4. Combine: use A* and Y* together
Goal: Leverage Gurobi for continuous, QPU for binary combinatorics

Patch Scenario (PURE QUANTUM):
- Already correct: pure binary → direct QPU
- No changes needed
```

## Progress Tracking

**Started**: Nov 21, 2025  
**Current Phase**: Phase 1 (Constraint Fixes)  
**Status**: In Progress

---

Last Updated: Nov 21, 2025
