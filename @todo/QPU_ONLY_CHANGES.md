# QPU-Only Benchmark Changes

## Summary

**All Simulated Annealing (SA) has been COMPLETELY REMOVED. All methods now use QPU exclusively.**

## Major Changes

### 1. **Eliminated SimulatedAnnealingSampler Everywhere**
   - ❌ **REMOVED**: `solve_decomposition_sa()` function
   - ❌ **REMOVED**: All `neal.SimulatedAnnealingSampler()` instances
   - ❌ **REMOVED**: All SA fallbacks when QPU embedding fails
   - ✅ **NOW**: Methods fail if QPU unavailable or embedding fails

### 2. **Coordinated Decomposition - QPU Only**
   **Before**: Used SA for master + subproblems
   ```python
   sa_sampler = neal.SimulatedAnnealingSampler()
   master_result = sa_sampler.sample(master_bqm, ...)
   ```
   
   **After**: Uses QPU for master + subproblems
   ```python
   qpu_sampler = EmbeddingComposite(get_qpu_sampler())
   master_result = qpu_sampler.sample(master_bqm, annealing_time=20, ...)
   ```
   
   **Result**: QPU timing now properly extracted and reported

### 3. **QPU Decomposition - No Fallback**
   **Before**: If partition embedding failed → fallback to SA
   ```python
   except Exception as e:
       sa_sampler = neal.SimulatedAnnealingSampler()
       sampleset = sa_sampler.sample(bqm, ...)  # WRONG!
   ```
   
   **After**: If partition embedding fails → method fails
   ```python
   except Exception as e:
       result['success'] = False
       result['error'] = f'Embedding failed: {e}'
       return result  # FAIL FAST
   ```

### 4. **CQM-First Decomposition - QPU Only**
   **Before**: SA for master U partition and Y partitions
   **After**: QPU for both master and all Y partitions with proper timing extraction

### 5. **Default Methods Updated**
   **Before**:
   ```python
   methods = [
       'ground_truth',
       'coordinated',
       'decomposition_PlotBased_SA',      # ❌ REMOVED
       'decomposition_Multilevel(5)_SA',  # ❌ REMOVED
       'decomposition_Louvain_SA',        # ❌ REMOVED
       'decomposition_Spectral(10)_SA',   # ❌ REMOVED
   ]
   ```
   
   **After**:
   ```python
   methods = [
       'ground_truth',
       'direct_qpu',                       # ✅ QPU
       'coordinated',                      # ✅ QPU
       'decomposition_PlotBased_QPU',      # ✅ QPU
       'decomposition_Multilevel(5)_QPU',  # ✅ QPU
       'decomposition_Louvain_QPU',        # ✅ QPU
       'cqm_first_PlotBased',              # ✅ QPU
   ]
   ```

## Timing Extraction Fixes

### QPU Timing Now Properly Extracted

**Coordinated Method**:
```python
# Master problem
master_timing = master_result.info.get('timing', {})
master_qpu_time = master_timing.get('qpu_access_time', 0) / 1e6

# Subproblems
sub_timing = sub_result.info.get('timing', {})
total_subproblem_qpu_time += sub_timing.get('qpu_access_time', 0) / 1e6

# Final result
result['total_qpu_time'] = master_qpu_time + total_subproblem_qpu_time
result['timings']['qpu_access_total'] = master_qpu_time + total_subproblem_qpu_time
result['timings']['embedding_total'] = (wall_time - qpu_time)  # Estimate
```

**CQM-First Method**:
```python
sampleset = qpu_sampler.sample(
    sub_bqm, 
    num_reads=num_reads, 
    annealing_time=annealing_time,
    label=f"CQMFirst_{method}_Part{i}"
)
# QPU timing extracted from sampleset.info['timing']
```

**Decomposition Methods**:
```python
timing_info = sampleset.info.get('timing', {})
qpu_access_us = timing_info.get('qpu_access_time', 0)
qpu_programming_us = timing_info.get('qpu_programming_time', 0)
qpu_sampling_us = timing_info.get('qpu_sampling_time', 0)

total_qpu_access_time += qpu_access_us / 1e6
result['total_qpu_time'] = total_qpu_access_time
```

## Error Handling

### QPU Required - No Fallback
All methods now check for QPU availability:
```python
if not HAS_QPU:
    result['success'] = False
    result['error'] = 'QPU not available - SA removed, QPU required'
    return result

qpu = get_qpu_sampler()
if qpu is None:
    result['success'] = False
    result['error'] = 'Failed to connect to QPU'
    return result
```

### Embedding Failures
Embedding failures now cause method failure instead of silent SA fallback:
```python
except Exception as e:
    # NO SA FALLBACK - fail the method
    result['success'] = False
    result['error'] = f'Embedding failed for partition {i}: {e}'
    return result
```

## Expected Behavior

### ✅ What You'll See Now:
1. **QPU timing in ALL methods** (not N/A)
2. **Embedding time in ALL methods** (not N/A)  
3. **Methods fail if QPU unavailable** (no silent SA fallback)
4. **Consistent QPU usage** across all decomposition strategies

### 🎯 Benchmark Output:
```
Scale  Method                            Obj    Gap%     Wall    Solve    Embed      QPU  Viol Status
25     Ground Truth (Gurobi)          0.3151     0.0    0.023    0.009      N/A      N/A     0 ✓ Opt
       direct_qpu                     0.xxxx   -xx.x   xx.xxx   xx.xxx   xx.xxx   xx.xxx     x ✓/⚠
       decomposition_PlotBased_QPU    0.xxxx   -xx.x   xx.xxx   xx.xxx   xx.xxx   xx.xxx     x ✓/⚠
       decomposition_Multilevel(5)_QPU 0.xxxx   -xx.x   xx.xxx   xx.xxx   xx.xxx   xx.xxx     x ✓/⚠
       cqm_first_PlotBased            0.xxxx   -xx.x   xx.xxx   xx.xxx   xx.xxx   xx.xxx     x ✓/⚠
       coordinated                    0.xxxx   -xx.x   xx.xxx   xx.xxx   xx.xxx   xx.xxx     x ✓/⚠
```

**NO MORE N/A for QPU timing!** 🎉

## Testing

Run the benchmark:
```powershell
cd "d:\Projects\OQI-UC002-DWave\@todo"
python qpu_benchmark.py --full --methods ground_truth direct_qpu coordinated decomposition_PlotBased_QPU decomposition_Multilevel(5)_QPU decomposition_Louvain_QPU cqm_first_PlotBased
```

Expected:
- ✅ All methods show QPU timing
- ✅ All methods show embedding timing  
- ✅ Methods fail gracefully if QPU unavailable
- ❌ No SA fallback anywhere
- ❌ No "SimulatedAnnealing" in sampler field

## Files Modified

- `d:\Projects\OQI-UC002-DWave\@todo\qpu_benchmark.py`
  - Removed `solve_decomposition_sa()` function completely
  - Updated `solve_coordinated_decomposition()` to use QPU
  - Updated `solve_decomposition_qpu()` to fail on embedding errors
  - Updated `solve_cqm_first_decomposition_sa()` to use QPU (despite name)
  - Updated default methods list
  - Updated method dispatch logic
  - Updated summary display logic
  - Updated legend

---

**Status**: ✅ **COMPLETE - ALL SA REMOVED, QPU ONLY**
