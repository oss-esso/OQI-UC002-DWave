# Benchmark Configuration Summary

## Current Status: READY TO RUN

All scripts configured with correct parameters as requested.

## Gurobi Parameters (Applied to all test scripts)

```python
GUROBI_CONFIG = {
    'timeout': 300,              # 5 minutes HARD LIMIT
    'mip_gap': 0.01,            # 1% - stop within 1% of optimum
    'mip_focus': 1,             # Find good feasible solutions quickly
    'improve_start_time': 30,   # Stop if no improvement for 30s
}
```

### Parameter Explanation

1. **TimeLimit = 300s**
   - HARD LIMIT: Gurobi will stop after exactly 5 minutes
   - Ensures consistent timeout behavior across all problem sizes

2. **MIPGap = 0.01 (1%)**
   - Stop when solution is within 1% of theoretical optimum
   - Balance between solution quality and runtime
   - More aggressive than 10% but still practical

3. **MIPFocus = 1**
   - Prioritize finding good feasible solutions quickly
   - Don't waste time proving optimality on hard problems
   - Best for problems where feasibility is challenging

4. **ImproveStartTime = 30s**
   - Stop if no improvement for 30 consecutive seconds
   - Prevents wasted time on plateaued search
   - Works in conjunction with TimeLimit

### Expected Behavior

| Problem Size | Expected Outcome |
|--------------|------------------|
| **5 farms (90 vars)** | May solve optimally < 1s (too easy) |
| **10 farms (180 vars)** | May timeout or hit improvement limit |
| **20 farms (360 vars)** | **Should timeout at 300s** ✅ |
| **25+ farms (2000+ vars)** | **Should timeout at 300s** ✅ |

**Key Point**: The 300s timeout should be hit consistently for rotation problems with 20+ farms, as these are NP-hard and Gurobi struggles with the constraint structure.

## Run Modes

### Mode 1: Gurobi-Only Timeout Verification
```bash
conda activate oqi
cd @todo
python test_gurobi_timeout.py
```

**Purpose**: Verify timeout behavior without consuming QPU credits
**Output**: `gurobi_timeout_verification/gurobi_timeout_test_TIMESTAMP.csv`

### Mode 2: Full Benchmark (Gurobi → QPU)
```bash
conda activate oqi
cd @todo
python significant_scenarios_benchmark.py
```

**Purpose**: Complete comparison with pause between methods
**Output**: `significant_scenarios_results/benchmark_results_TIMESTAMP.csv`

**Features**:
- Runs Gurobi FIRST for each scenario
- Displays results and PAUSES
- Press ENTER to confirm and run QPU
- Allows verification of Gurobi results before consuming QPU credits

## Files Modified

1. **significant_scenarios_benchmark.py**
   - Main benchmark script
   - Runs Gurobi first, then QPU (with pause)
   - Tracks timeout hits and stopping reasons

2. **test_gurobi_timeout.py** (NEW)
   - Gurobi-only verification script
   - Tests timeout behavior across all 6 scenarios
   - No QPU credits consumed

## What to Watch For

### Good Indicators ✅
- Timeout consistently at ~300s for 20+ farms
- ImproveStartTime stops some scenarios < 300s (if finding good solutions early)
- Small problems (5 farms) may solve optimally quickly (expected)

### Warning Signs ⚠️
- Timeout "disappears" for large problems (shouldn't happen with 300s hard limit)
- All scenarios finish < 30s (parameters not applied correctly)
- No timeouts at all (something wrong with configuration)

## Verification Steps

1. **First: Run Gurobi-only test**
   ```bash
   python test_gurobi_timeout.py
   ```
   Expected: 3-4 scenarios hit 300s timeout

2. **Review results**
   Check CSV for timeout patterns

3. **If timeouts confirmed: Run full benchmark**
   ```bash
   python significant_scenarios_benchmark.py
   ```

4. **After each Gurobi run**
   - Review objective value
   - Check timeout status
   - Press ENTER to run QPU
   - Compare results

## Expected Timeline

### Gurobi-Only Test
- **5 farms**: < 1s (optimal)
- **10 farms**: ~30-60s (may hit improvement limit)
- **20 farms**: ~300s (TIMEOUT)
- **25 farms**: ~300s (TIMEOUT)
- **50 farms**: ~300s (TIMEOUT)
- **100 farms**: ~300s (TIMEOUT)
- **Total**: ~18-20 minutes

### Full Benchmark (Gurobi + QPU)
- Gurobi: ~20 minutes (as above)
- QPU: ~40-60 minutes (depending on problem size)
- **Total**: ~60-80 minutes

## Notes

- **Timeout is EXPECTED and DESIRED** for rotation problems
- This demonstrates QPU advantage: solving problems Gurobi can't
- Gap calculation: `(gurobi_obj - qpu_obj) / gurobi_obj × 100%`
  - If Gurobi timeouts, its objective is the "best found so far"
  - Not guaranteed optimal, so gap may be misleading
  - Better metric: **speedup** (gurobi_time / qpu_time)

---

**Ready to run!** Start with `test_gurobi_timeout.py` to verify configuration before running full benchmark.
