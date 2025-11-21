# CRITICAL ISSUES - INVESTIGATION REPORT

**Date**: November 21, 2025, 12:15 PM  
**Status**: Issues Identified and Fixed

---

## ✅ ISSUE 1: Patch Scenario Not Running with SimulatedAnnealing

### Problem
Patch scenario showed "Skipped" instead of running with neal fallback when no D-Wave token provided.

### Root Cause
In `benchmark_utils_decomposed.py` line 115:
```python
if not dwave_token:
    print(f"⊘ (no token)")
    return {'solver': 'decomposed_qpu', 'status': 'Skipped', 'success': False}
```

This check prevented the function from reaching `solve_with_decomposed_qpu()` which has the SimulatedAnnealing fallback logic.

### Fix Applied
**File**: `benchmark_utils_decomposed.py`

**Changed**:
```python
# REMOVED the early return check
# Now goes directly to CQM → BQM conversion and solve_with_decomposed_qpu()
```

The function now:
1. Converts CQM → BQM regardless of token
2. Calls `solve_with_decomposed_qpu(bqm, token, ...)`
3. Inside that function, the token check triggers SimulatedAnnealing fallback

### Additional Fix
**File**: `solver_runner_DECOMPOSED.py` line 1287

**Enhanced token detection**:
```python
# OLD:
use_simulated_annealing = (token is None or token == 'YOUR_DWAVE_TOKEN_HERE')

# NEW:
use_simulated_annealing = (
    token is None or 
    token == 'YOUR_DWAVE_TOKEN_HERE' or
    (isinstance(token, str) and token.strip() == '')
)
```

Now catches empty strings and None values properly.

### Expected Behavior After Fix
```
[PATCH SCENARIO: 25 patches - QUANTUM OPTIMIZATION]
Loading food data...
✓ (675 vars, 89 constraints)
  Running solvers:
    [decomposed_qpu] converting...

================================================================================
SOLVING WITH SIMULATED ANNEALING (Testing Mode - No QPU)
================================================================================
  Running Simulated Annealing...
    ✓ Simulated Annealing complete in 5.23s

PATCH Scenario (quantum_only):
  Units: 25
  Variables: 675
  Constraints: 89
  Solvers:
    decomposed_qpu (simulated_annealing):
      Status: Optimal | Obj: 1.2345 | Time: 5.23s
```

---

## ✅ ISSUE 2: Custom Hybrid Binary Solving Clarification

### Investigation
The user asked: "find out why in the custom hybrid gurobi is not solving the binary problem"

### Finding: **NOT A BUG - Working as Designed**

The custom hybrid workflow (`solver_runner_CUSTOM_HYBRID.py`) works differently than the decomposed hybrid:

#### Alternative 1: Custom Hybrid Workflow
```
Input: CQM (mixed continuous + binary)
  ↓
Convert: CQM → BQM (all variables discretized to binary)
  ↓
Solve: dwave-hybrid framework
  - Decomposes BQM into subproblems
  - Races: TabuSampler | SimulatedAnnealing | QPU
  - Iterates until convergence
  ↓
Output: BQM solution → inverted back to CQM space
```

**Key Point**: The continuous variables (A) are **discretized** during CQM → BQM conversion. The hybrid workflow then solves the fully-binary BQM using:
- Energy Impact Decomposer (selects high-impact variables)
- QPU (or SimulatedAnnealing) solves binary subproblems
- Splat Composer rebuilds full solution
- Iteration refines solution

**Gurobi is NOT used in Alternative 1** - it relies entirely on the dwave-hybrid framework.

#### Alternative 2: Decomposed Hybrid (Farm)
```
Phase 1: Gurobi solves continuous relaxation
  - A ∈ ℝ, Y ∈ [0,1]
  - Get A*, Y_relaxed
  ↓
Phase 2: QPU solves binary subproblem
  - Fix A = A*
  - Solve for binary Y ∈ {0,1}
  - Get Y**
  ↓
Phase 3: Combine (A*, Y**)
```

**Key Point**: Alternative 2 explicitly uses Gurobi for continuous variables in Phase 1, then QPU for binary Y in Phase 2.

### Conclusion
**Custom Hybrid is working correctly**. It doesn't use Gurobi for binary solving because it's designed as a pure dwave-hybrid workflow. The BQM includes discretized versions of both A and Y variables, solved together by the racing samplers.

If the user wants Gurobi to solve binary problems, they should use **Alternative 2** (decomposed hybrid).

---

## 📋 NEXT STEPS

### Phase 1: Test Fixes (IMMEDIATE)
1. [ ] Run `comprehensive_benchmark_DECOMPOSED.py` with config 10
2. [ ] Verify patch scenario runs with SimulatedAnnealing
3. [ ] Verify JSON output matches reference format
4. [ ] Run config 25 to confirm full benchmark works

### Phase 2: Implement Multiple Decomposition Strategies
As outlined in `DECOMPOSITION_ENHANCEMENT_TASKLIST.md`:

1. **Benders Decomposition** (2-3 hours)
   - Master: Binary Y problem (MILP)
   - Subproblem: Continuous A given Y* (LP)
   - Iterative with Benders cuts

2. **Dantzig-Wolfe Decomposition** (2-3 hours)
   - Restricted Master Problem
   - Column generation
   - Pricing subproblem

3. **ADMM** (2-3 hours)
   - Split into A-subproblem and Y-subproblem
   - Alternating optimization
   - Dual variable updates

### Phase 3: Standardize JSON Output
Create `result_formatter.py` to ensure all strategies output in the reference format with:
- `status`, `objective_value`, `hybrid_time`, `qpu_time`
- `is_feasible`, `num_samples`, `success`
- `solution_plantations` (full variable mapping)
- `validation` (constraint checks)

---

## 🧪 TESTING COMMANDS

### Test Patch Fix
```powershell
cd d:\Projects\OQI-UC002-DWave\@todo
$env:PYTHONIOENCODING='utf-8'
python comprehensive_benchmark_DECOMPOSED.py --config 10
```

Expected: Both Farm and Patch complete successfully with SimulatedAnnealing.

### Test with Actual Token (if available)
```powershell
$env:DWAVE_API_TOKEN='your-token-here'
python comprehensive_benchmark_DECOMPOSED.py --config 10
```

Expected: Patch uses actual QPU, Farm uses Gurobi + QPU hybrid.

---

## 📊 CURRENT STATUS

| Issue | Status | Fix Applied | Testing Required |
|-------|--------|-------------|------------------|
| Patch SimulatedAnnealing fallback | ✅ FIXED | Yes | Yes - run benchmark |
| Token detection robustness | ✅ FIXED | Yes | Yes - test empty string |
| Custom hybrid binary solving | ℹ️ NOT A BUG | N/A | Document behavior |

---

## 🎯 RECOMMENDATIONS

### Immediate (Today)
1. ✅ Test the patch fix with config 10
2. ✅ Verify JSON output format
3. ✅ Start implementing Benders decomposition

### Short-term (This Week)
4. Implement all 4 decomposition strategies
5. Create unified strategy interface
6. Run comprehensive comparison benchmarks
7. Generate performance analysis report

### Medium-term (Next Week)
8. Update LaTeX Chapter 4 with new strategies
9. Create performance comparison plots
10. Document best-practice recommendations

---

**Last Updated**: November 21, 2025, 12:15 PM  
**Next Action**: Run test benchmark to verify patch fix
