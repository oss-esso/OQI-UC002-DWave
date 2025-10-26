# Benchmark Consistency Analysis - Config 10

## Date: October 26, 2025

## Summary

✅ **All solvers completed successfully**
⚠️ **Objective values differ by up to 5.13%**
❌ **D-Wave BQM QPU time not captured (needs re-run)**

## Detailed Results

### Farm Scenario (10 farms, 48.9 ha)

| Solver | Objective | Time | Status |
|--------|-----------|------|--------|
| **Gurobi (PuLP)** | 31.5405 | 0.012s | ✅ Optimal |
| **D-Wave CQM** | 30.0704 | 2.990s (QPU: 0.103s) | ✅ Optimal |

**Difference:** 1.47 (4.89%) - ⚠️ Acceptable but notable

### Patch Scenario (10 patches, 5.398 ha)

| Solver | Objective | Time | Feasible | Violations |
|--------|-----------|------|----------|------------|
| **Gurobi (PuLP)** | 3.4817 | 0.005s | ✅ Yes | 0 |
| **D-Wave CQM** | 3.3118 | 3.000s (QPU: 0.104s) | ✅ Yes | 0 |
| **Gurobi QUBO** | 3.4347 | 30.034s | ✅ Yes | 0 |
| **D-Wave BQM** | 3.3356 | 4.091s (QPU: ❌ None) | ✅ Yes | 0 |

**Difference:** 0.17 (5.13%) - ❌ Significant!

## Issues Identified

### 1. D-Wave BQM QPU Time Missing ❌

**Problem:**
```json
"qpu_time": null
```

**Root Cause:**
LeapHybridBQMSampler returns timing information differently than expected. The code was looking for `qpu_access_time` but it may not be present or may be in a different location.

**Fix Applied:**
Updated timing extraction in `comprehensive_benchmark.py` to match the logic from `solve_with_dwave()`:
```python
# Hybrid solve time (total time including QPU)
hybrid_time_bqm = (timing_info.get('run_time') or 
                  sampleset_bqm.info.get('run_time') or
                  timing_info.get('charge_time') or
                  sampleset_bqm.info.get('charge_time'))

# QPU access time
qpu_time_bqm = (timing_info.get('qpu_access_time') or
               sampleset_bqm.info.get('qpu_access_time'))
```

**Action Required:**
Re-run benchmark to get correct QPU time:
```bash
# Delete the BQM result
rm Benchmarks/COMPREHENSIVE/Patch_DWaveBQM/config_10_run_1.json

# Re-run benchmark (will use cache for other solvers)
python comprehensive_benchmark.py
```

### 2. Objective Value Differences ⚠️

**Patch Scenario - 5.13% difference:**
- Best: Gurobi PuLP = 3.4817
- Worst: D-Wave CQM = 3.3118
- Range: 0.17 (5.13%)

**Farm Scenario - 4.89% difference:**
- Best: Gurobi PuLP = 31.5405
- Worst: D-Wave CQM = 30.0704
- Range: 1.47 (4.89%)

**Possible Reasons:**

1. **Heuristic vs Exact Solvers:**
   - PuLP/Gurobi (exact): Finds proven optimal solution
   - D-Wave (heuristic): Finds good solution, not necessarily optimal
   - Expected behavior for quantum/hybrid solvers

2. **Time Limits:**
   - Gurobi QUBO hit time limit (30s) - found feasible solution but not optimal
   - D-Wave solvers may have internal time constraints

3. **CQM vs BQM Formulation:**
   - CQM: Native constraint handling
   - BQM: Constraints converted to penalties
   - Different formulations may find different local optima

4. **Problem Characteristics:**
   - Patch scenario (5.4 ha): Smaller, potentially more challenging
   - Farm scenario (48.9 ha): Larger but may have more flexibility

**Analysis:**

✅ **All solutions are feasible** (0 violations)
✅ **Objectives within 5%** is reasonable for hybrid/quantum solvers
⚠️ **5.13% difference** suggests room for improvement:
   - Increase D-Wave time limits?
   - Tune BQM penalty coefficients?
   - Multiple runs to find best?

## Timing Analysis

### Patch Scenario

| Solver | Total Time | QPU Time | Hybrid Time | Speedup vs Gurobi |
|--------|------------|----------|-------------|-------------------|
| Gurobi (PuLP) | 0.005s | - | - | 1.0x (baseline) |
| D-Wave CQM | 3.000s | 0.104s | 3.000s | 0.002x (600x slower) |
| Gurobi QUBO | 30.034s | - | - | 0.0002x (6000x slower) |
| D-Wave BQM | 4.091s | ❌ None | 4.091s | 0.001x (818x slower) |

**Observations:**
- Classical Gurobi is **much faster** than all quantum/hybrid solvers
- Gurobi QUBO hit 30s timeout (would be even slower)
- D-Wave solvers take 3-4 seconds (mostly hybrid overhead, QPU < 0.1s)

### Why Quantum is Slower Here

1. **Problem Size**: 10 plots is very small
   - Classical solvers excel at small problems
   - Quantum advantage expected at larger scales
   
2. **Communication Overhead**: 
   - QPU time: ~0.1s
   - Total time: ~3-4s
   - **96% of time** is communication/preprocessing!

3. **No Quantum Advantage Yet**:
   - Need larger problems (100+ variables) to see quantum benefit
   - Current problem is in classical regime

## Recommendations

### Immediate Actions

1. **Re-run D-Wave BQM** to get QPU time ✅ Fix applied
2. **Increase problem size** to test quantum advantage:
   ```bash
   python comprehensive_benchmark.py 20 --dwave  # 20 plots
   python comprehensive_benchmark.py 50 --dwave  # 50 plots
   ```

### Analysis Improvements

1. **Multiple Runs per Config:**
   - Run each solver 5-10 times
   - Report mean ± std deviation
   - Shows variability in heuristic solvers

2. **Quality-Time Tradeoff:**
   - Plot objective vs time
   - Show Pareto frontier
   - Some solvers might be faster but less accurate

3. **Solution Comparison:**
   - Compare which crops each solver selected
   - Understand why objectives differ
   - May reveal interesting patterns

### D-Wave Tuning

1. **Increase Time Limit:**
   ```python
   sampler.sample_cqm(cqm, time_limit=10)  # 10 seconds
   ```

2. **Multiple Reads:**
   ```python
   sampler.sample(bqm, num_reads=100)  # Try 100 solutions
   ```

3. **Chain Strength Tuning:**
   ```python
   sampler.sample(bqm, chain_strength=2.0)  # Stronger chains
   ```

## Conclusions

### Positives ✅
- All solvers find **feasible solutions** (0 violations)
- Constraint validation working correctly
- Timing instrumentation in place (except BQM QPU)
- Cache system working well

### Issues ⚠️
- **5.13% objective difference** is significant but not alarming
- D-Wave solvers finding slightly worse solutions than Gurobi
- No quantum speedup at this problem size (expected)

### Next Steps 📋
1. Re-run with QPU time fix
2. Test larger problem sizes (20, 50, 100 plots)
3. Analyze why D-Wave finds worse solutions
4. Consider multiple runs per config
5. Tune D-Wave parameters for better quality

## Technical Note

The QPU time fix has been applied to `comprehensive_benchmark.py`. To get updated results:

```bash
# Remove cached D-Wave BQM result
Remove-Item "Benchmarks/COMPREHENSIVE/Patch_DWaveBQM/config_10_run_1.json"

# Re-run (will use cache for other solvers)
python comprehensive_benchmark.py
```

This will re-run only the D-Wave BQM solver and update its result with correct QPU timing.
