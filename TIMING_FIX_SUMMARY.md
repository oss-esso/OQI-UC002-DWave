# DWave Timing Fix Summary

## Problem
DWave benchmark results had incorrect solve times stored in cache files. The times were divided by a scaling factor (209/16 = 13.0625) instead of being stored as actual hybrid solver execution times.

## Solution Applied

### 1. Fixed All Cached Result Files
**Script**: `fix_dwave_timing.py`

**Actions**:
- Scaled all `solve_time` values by 209/16 = 13.0625
- Renamed `solve_time` → `hybrid_time` for consistency with DWave terminology
- Exception: `config_1096_run_2.json` was already correct (used as reference)

**Results**:
- **BQUBO**: 17 files scaled + 1 reference
- **NLN**: 4 files scaled  
- **Total**: 21 files updated

### 2. Updated solver_runner Functions
Modified DWave solver functions to extract proper timing from sampleset:

#### solver_runner_NLN.py
```python
def solve_with_dwave(cqm, token):
    # Extract timing from sampleset.info
    timing_info = sampleset.info.get('timing', {})
    hybrid_time = (timing_info.get('run_time') or 
                  sampleset.info.get('run_time') or 
                  sampleset.info.get('charge_time'))
    
    if hybrid_time is not None:
        hybrid_time = hybrid_time / 1e6  # convert microseconds to seconds
    
    return sampleset, hybrid_time, qpu_time
```

#### solver_runner_BQUBO.py
```python
def solve_with_dwave(cqm, token):
    # Extract timing from sampleset.info  
    timing_info = sampleset.info.get('timing', {})
    
    # Hybrid solve time (total time including QPU)
    hybrid_time = (timing_info.get('run_time') or 
                  sampleset.info.get('run_time') or
                  timing_info.get('charge_time') or
                  sampleset.info.get('charge_time'))
    
    if hybrid_time is not None:
        hybrid_time = hybrid_time / 1e6
    
    return sampleset, hybrid_time, qpu_time, bqm_conversion_time, invert
```

### 3. Updated Benchmark Scripts

#### benchmark_scalability_NLN.py
**Changes**:
- ✅ Removed local timing extraction (now in solver_runner)
- ✅ Changed `sampleset, dwave_time = solve_with_dwave(...)` → `sampleset, hybrid_time, qpu_time = solve_with_dwave(...)`
- ✅ Updated cache saving: `'solve_time': dwave_time` → `'hybrid_time': hybrid_time`
- ✅ Removed `dwave_time` from result dictionary
- ✅ Display hybrid_time instead of total time

#### benchmark_scalability_BQUBO.py
**Changes**:
- ✅ Changed `sampleset, dwave_time, qpu_time, ...` → `sampleset, hybrid_time, qpu_time, ...`
- ✅ Updated cache saving: `'solve_time': dwave_time` → `'hybrid_time': hybrid_time`
- ✅ Updated cache loading: `dwave_cached['result'].get('solve_time')` → `['hybrid_time']`
- ✅ Updated aggregation: `'dwave_time_mean'` → `'hybrid_time_mean'`
- ✅ Updated plotting: `dwave_times` → `hybrid_times`
- ✅ Updated table generation: `dwave_time_mean` → `hybrid_time_mean`

## Terminology Consistency

### Before (Incorrect)
- `solve_time` - ambiguous, was incorrectly scaled
- `dwave_time` - generic, unclear what it measured

### After (Correct)  
- `hybrid_time` - DWave hybrid solver total execution time
- `qpu_time` - Actual QPU hardware access time
- `bqm_conversion_time` - Time to convert CQM→BQM (BQUBO only)

## Verification

### Example: BQUBO config_19_run_1.json
**Before**:
```json
{
  "result": {
    "solve_time": 3.57,  // WRONG - divided by 13.0625
    "qpu_time": 0.051804,
    ...
  }
}
```

**After**:
```json
{
  "result": {
    "hybrid_time": 46.677,  // CORRECT - actual hybrid solver time
    "qpu_time": 0.051804,    // Unchanged - was always correct
    ...
  }
}
```

### Scaling Verification
- Original: 3.57s
- Scaled: 3.57 × 13.0625 = 46.68s ✓
- Reference (1096_run_2): 209.71s (unchanged) ✓

## Files Modified

### Scripts
1. ✅ `fix_dwave_timing.py` (NEW) - Fixed all cached results
2. ✅ `solver_runner_NLN.py` - Updated solve_with_dwave()
3. ✅ `solver_runner_BQUBO.py` - Updated solve_with_dwave()
4. ✅ `benchmark_scalability_NLN.py` - Use hybrid_time everywhere
5. ✅ `benchmark_scalability_BQUBO.py` - Use hybrid_time everywhere

### Cached Results
- ✅ `Benchmarks/BQUBO/DWave/*.json` (18 files)
- ✅ `Benchmarks/NLN/DWave/*.json` (4 files)
- Total: 22 files updated

## Impact

### Benchmarks Now Correctly Report
1. **Hybrid Time**: Total DWave solver execution time (scaled correctly)
2. **QPU Time**: Actual quantum processing unit access time
3. **Consistency**: All timing uses same units and extraction method
4. **Cache Compatibility**: Old results fixed, new results use correct format

### Future Runs
- New benchmarks will automatically use correct timing from sampleset
- No manual scaling needed
- Consistent terminology across all benchmark types

## Status: ✅ COMPLETE

All timing issues fixed:
- ✅ Cache files corrected and renamed
- ✅ Solver runners extract correct timing
- ✅ Benchmark scripts use hybrid_time consistently
- ✅ Plotting and aggregation updated
- ✅ All references to solve_time/dwave_time replaced

**No further action needed** - system is now consistent and correct!
