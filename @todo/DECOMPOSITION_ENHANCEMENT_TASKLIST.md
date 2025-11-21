# Decomposition Enhancement Task List

**Created**: November 21, 2025  
**Status**: In Progress  
**Goal**: Enhance Alternative 2 with multiple decomposition strategies and fix issues

---

## 🔴 CRITICAL ISSUES (Fix First)

### Issue 1: Patch Scenario Not Running with SimulatedAnnealing ❌
**Problem**: Patch scenario shows "Skipped" instead of running with neal fallback  
**Location**: `benchmark_utils_decomposed.py` - `solve_with_decomposed_qpu()`  
**Root Cause**: Token check is preventing fallback from triggering properly  
**Fix Required**:
- [ ] Update token detection logic to match custom_hybrid pattern
- [ ] Ensure patch CQM → BQM conversion works
- [ ] Test with SimulatedAnnealing fallback
- [ ] Verify results are saved in correct JSON format

### Issue 2: Custom Hybrid Binary Problem Not Being Solved ❌
**Problem**: Custom hybrid might not be solving binary subproblem with Gurobi  
**Location**: `solver_runner_CUSTOM_HYBRID.py`  
**Investigation Needed**:
- [ ] Check if binary variables are being created correctly
- [ ] Verify Gurobi is being called for binary subproblem
- [ ] Review CQM to BQM conversion in custom hybrid
- [ ] Check if BQM is being solved or bypassed

---

## 📋 PHASE 1: Fix Critical Issues

### Task 1.1: Fix Patch SimulatedAnnealing Fallback ⚠️
**Priority**: CRITICAL  
**Estimated Time**: 30 minutes

**Steps**:
1. [ ] Read `benchmark_utils_decomposed.py`
2. [ ] Locate `solve_with_decomposed_qpu()` function
3. [ ] Update token detection:
   ```python
   use_simulated_annealing = (
       token is None or 
       token == 'YOUR_DWAVE_TOKEN_HERE' or
       token.strip() == ''
   )
   ```
4. [ ] Add proper error handling and logging
5. [ ] Test with config 10
6. [ ] Verify JSON output matches reference format

**Success Criteria**:
- Patch scenario runs with SimulatedAnnealing
- Results saved in correct JSON format (like attached examples)
- Both Farm and Patch complete successfully

### Task 1.2: Investigate Custom Hybrid Binary Solving ⚠️
**Priority**: CRITICAL  
**Estimated Time**: 45 minutes

**Steps**:
1. [ ] Read `solver_runner_CUSTOM_HYBRID.py`
2. [ ] Trace hybrid workflow construction
3. [ ] Check if binary subproblem is created
4. [ ] Verify QPU/Simulated sampler is invoked
5. [ ] Add debug logging to track execution path
6. [ ] Compare with working decomposed implementation

**Success Criteria**:
- Identify exact point where binary solving fails/is skipped
- Document root cause
- Propose fix

---

## 📋 PHASE 2: Implement Multiple Decomposition Strategies

### Overview: Farm Scenario Decomposition Variants

We'll implement 4 decomposition strategies for the FARM scenario:

| Strategy | Description | Solver Type |
|----------|-------------|-------------|
| **Current** | Gurobi relaxation → QPU binary | Hybrid (Gurobi + QPU) |
| **Benders** | Master problem (Y) ↔ Subproblem (A) | Iterative classical |
| **Dantzig-Wolfe** | Column generation with restricted master | Iterative classical |
| **ADMM** | Alternating Direction Method of Multipliers | Iterative hybrid |

### Task 2.1: Implement Benders Decomposition 🔵
**Priority**: HIGH  
**Estimated Time**: 2-3 hours

**Algorithm**:
```
Master Problem (MILP): Optimize Y variables + cuts
Subproblem (LP): For fixed Y*, optimize A variables
Iterate: Add Benders cuts to master until convergence
```

**Steps**:
1. [ ] Create `decomposition_benders.py`
2. [ ] Implement master problem (binary Y variables)
3. [ ] Implement subproblem (continuous A given Y*)
4. [ ] Implement Benders cut generation
5. [ ] Add convergence criteria (gap tolerance)
6. [ ] Integrate with benchmark framework
7. [ ] Test with config 10, 25

**Success Criteria**:
- Converges to optimal/near-optimal solution
- Faster than current hybrid for some problem sizes
- Results saved in standard JSON format

### Task 2.2: Implement Dantzig-Wolfe Decomposition 🔵
**Priority**: HIGH  
**Estimated Time**: 2-3 hours

**Algorithm**:
```
Restricted Master Problem: Select from column pool
Pricing Subproblem: Generate new columns
Iterate: Add columns until no improvement
```

**Steps**:
1. [ ] Create `decomposition_dantzig_wolfe.py`
2. [ ] Implement restricted master problem
3. [ ] Implement pricing subproblem
4. [ ] Implement column generation logic
5. [ ] Add termination criteria
6. [ ] Integrate with benchmark framework
7. [ ] Test with config 10, 25

**Success Criteria**:
- Generates valid columns
- Converges to solution
- Competitive solve time

### Task 2.3: Implement ADMM Decomposition 🔵
**Priority**: MEDIUM  
**Estimated Time**: 2-3 hours

**Algorithm**:
```
Split variables: A-subproblem and Y-subproblem
Iterate:
  1. Solve A-subproblem (Gurobi/QPU)
  2. Solve Y-subproblem (Gurobi/QPU)
  3. Update dual variables
Until convergence
```

**Steps**:
1. [ ] Create `decomposition_admm.py`
2. [ ] Implement A-subproblem solver
3. [ ] Implement Y-subproblem solver
4. [ ] Implement dual variable updates
5. [ ] Add convergence monitoring
6. [ ] Integrate with benchmark framework
7. [ ] Test with config 10, 25

**Success Criteria**:
- ADMM iterations converge
- Final solution is feasible
- Potential for QPU acceleration

### Task 2.4: Create Unified Decomposition Interface 🔵
**Priority**: HIGH  
**Estimated Time**: 1 hour

**Steps**:
1. [ ] Create `decomposition_strategies.py`
2. [ ] Define abstract base class `DecompositionStrategy`
3. [ ] Implement common interface:
   ```python
   def solve(self, farms, foods, food_groups, config, **kwargs):
       pass  # Returns standard result dict
   ```
4. [ ] Register all strategies in a factory
5. [ ] Update benchmark to support strategy selection

**Success Criteria**:
- All strategies use same interface
- Easy to add new strategies
- Benchmark can run all strategies with `--strategies all`

---

## 📋 PHASE 3: Standardize JSON Output Format

### Task 3.1: Analyze Reference JSON Structure ⚠️
**Priority**: HIGH  
**Estimated Time**: 30 minutes

**Steps**:
1. [ ] Read attached `config_25_run_1.json` files
2. [ ] Document required fields:
   - [ ] `status`, `objective_value`, `hybrid_time`, `qpu_time`
   - [ ] `is_feasible`, `num_samples`, `success`
   - [ ] `n_units`, `total_area`, `n_foods`
   - [ ] `n_variables`, `n_constraints`
   - [ ] `total_covered_area`
   - [ ] `solution_plantations` (full Y variable mapping)
   - [ ] `validation` (constraint checks)
3. [ ] Create JSON schema definition
4. [ ] Compare with current output format

### Task 3.2: Implement Standardized Result Builder 🔵
**Priority**: HIGH  
**Estimated Time**: 1 hour

**Steps**:
1. [ ] Create `result_formatter.py`
2. [ ] Implement `format_farm_result()` function
3. [ ] Implement `format_patch_result()` function
4. [ ] Add validation section builder
5. [ ] Test output matches reference JSON exactly

**Success Criteria**:
- All results match reference JSON structure
- Validation section included
- Can parse back from JSON successfully

---

## 📋 PHASE 4: Benchmarking & Comparison

### Task 4.1: Create Comprehensive Benchmark Script 🔵
**Priority**: MEDIUM  
**Estimated Time**: 1 hour

**Steps**:
1. [ ] Create `comprehensive_benchmark_ALL_DECOMPOSITIONS.py`
2. [ ] Support command-line args:
   - `--strategies`: comma-separated list or "all"
   - `--config`: problem size (10, 25, etc.)
   - `--runs`: number of repetitions
3. [ ] Run all strategies on same problem instance
4. [ ] Save results in organized directory structure
5. [ ] Generate comparison table (CSV/Markdown)

### Task 4.2: Performance Analysis 🔵
**Priority**: MEDIUM  
**Estimated Time**: 2 hours

**Steps**:
1. [ ] Run all strategies on config 10, 25
2. [ ] Compare metrics:
   - [ ] Solve time
   - [ ] Solution quality (objective value)
   - [ ] Feasibility rate
   - [ ] Number of iterations (for iterative methods)
3. [ ] Create comparison plots
4. [ ] Document findings in `DECOMPOSITION_COMPARISON.md`

---

## 📋 PHASE 5: Documentation & LaTeX Updates

### Task 5.1: Update Chapter 4 with New Strategies 🔵
**Priority**: LOW  
**Estimated Time**: 2 hours

**Steps**:
1. [ ] Add sections for each decomposition strategy
2. [ ] Include algorithm pseudocode
3. [ ] Add performance comparison tables
4. [ ] Update architecture diagrams

### Task 5.2: Create Implementation Memory 🔵
**Priority**: HIGH  
**Estimated Time**: 30 minutes

**Steps**:
1. [ ] Create `DECOMPOSITION_MEMORY.md`
2. [ ] Document each strategy's implementation
3. [ ] Include mathematical formulation
4. [ ] Add code examples and usage

---

## 🎯 SUCCESS METRICS

### Critical (Must Have)
- [x] Patch scenario runs with SimulatedAnnealing fallback
- [ ] Results match reference JSON format exactly
- [ ] All decomposition strategies converge to feasible solutions
- [ ] Custom hybrid binary solving issue identified and fixed

### High Priority (Should Have)
- [ ] 4 decomposition strategies implemented and tested
- [ ] Comparison benchmark runs successfully
- [ ] Performance analysis complete

### Medium Priority (Nice to Have)
- [ ] LaTeX documentation updated
- [ ] Visualization of strategy comparison
- [ ] Automated testing for all strategies

---

## 📁 FILE STRUCTURE

```
@todo/
├── decomposition_strategies/
│   ├── __init__.py
│   ├── base.py                    # Abstract base class
│   ├── current_hybrid.py          # Existing Gurobi + QPU
│   ├── benders.py                 # Benders decomposition
│   ├── dantzig_wolfe.py           # Dantzig-Wolfe
│   └── admm.py                    # ADMM
├── result_formatter.py            # JSON standardization
├── comprehensive_benchmark_ALL.py # All strategies benchmark
├── DECOMPOSITION_MEMORY.md        # Implementation reference
└── DECOMPOSITION_COMPARISON.md    # Performance analysis
```

---

## 🚀 EXECUTION PLAN

### Day 1 (Today - 4 hours)
1. ✅ Fix patch SimulatedAnnealing fallback (30 min)
2. ✅ Investigate custom hybrid issue (45 min)
3. ✅ Standardize JSON output (1 hour)
4. ✅ Implement Benders decomposition (2 hours)

### Day 2 (4 hours)
5. Implement Dantzig-Wolfe (2 hours)
6. Implement ADMM (2 hours)

### Day 3 (3 hours)
7. Create unified interface (1 hour)
8. Run comprehensive benchmarks (1 hour)
9. Performance analysis (1 hour)

### Day 4 (2 hours)
10. Documentation updates (2 hours)

---

**Last Updated**: November 21, 2025, 12:00 PM  
**Status**: Phase 1 starting now
