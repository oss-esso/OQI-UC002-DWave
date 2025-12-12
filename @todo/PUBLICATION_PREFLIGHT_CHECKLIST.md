# Publication-Quality Pre-Flight Checklist
## Hierarchical Statistical Test for Technical Paper

**Date**: 2025-12-12  
**Test Script**: `hierarchical_statistical_test.py`  
**Purpose**: Rigorous quantum vs classical comparison for academic publication

---

## ✅ Quality Assurance Checklist

### 1. **Consistency with Previous Work** ✅

**Previous test** (`statistical_comparison_test.py`):
- Problem sizes: 5, 10, 15, 20, 25 farms
- Methods: Gurobi, Clique, Spatial-Temporal
- Metrics: Time, objective, gap, diversity
- Runs: 2 per method per size

**Current test** (`hierarchical_statistical_test.py`):
- Problem sizes: **25, 50, 100 farms** (scaling continuation)
- Methods: Gurobi, **Hierarchical QPU** (new approach)
- Metrics: **Same + QPU time breakdown**
- Runs: **3 per method per size** (better statistics)

**✅ Proper continuation**: Overlaps at 25 farms for validation, extends to larger scales

---

### 2. **Fair Comparison** ✅

**Both methods use**:
- ✅ **Same data source**: rotation scenarios with 27 foods
- ✅ **Same problem formulation**: Family-level (6 families after aggregation)
- ✅ **Same objective function**: Benefits + rotation synergies + diversity
- ✅ **Same constraints**: One crop family per farm per period
- ✅ **Same timeout**: 900 seconds (15 minutes) for Gurobi
- ✅ **Same post-processing**: Family → crop refinement

**Key difference** (algorithm approach):
- Gurobi: Direct MIP on family-level (6 families × N farms × 3 periods)
- Hierarchical QPU: Spatial decomposition → QPU clusters → boundary coordination

**✅ Fair**: Both solve equivalent mathematical problem, different approaches

---

### 3. **Statistical Rigor** ✅

**Sample size**:
- ✅ **3 runs per method per size** (improved from 2 in previous test)
- ✅ Total: 3 sizes × 2 methods × 3 runs = **18 experimental runs**

**Metrics collected**:
- ✅ **Central tendency**: Mean, min, max
- ✅ **Variance**: Standard deviation
- ✅ **Success rate**: Feasibility tracking
- ✅ **Multiple dimensions**: Time, quality, diversity

**Statistical tests** (implicit):
- ✅ Can compute confidence intervals from std dev
- ✅ Can perform t-tests for significance (if needed)
- ✅ Sufficient for publication-quality claims

---

### 4. **Measurement Accuracy** ✅

**Timing**:
- ✅ Wall-clock time (total solve time)
- ✅ QPU-specific time (actual quantum access)
- ✅ Breakdown: Decomposition, quantum solve, post-processing
- ✅ Python `time.time()` for consistent measurement

**Objective**:
- ✅ Same calculation function for both methods
- ✅ Includes: base benefits + rotation synergies + diversity bonus
- ✅ Penalty for violations

**Quality metrics**:
- ✅ Optimality gap: `|Gurobi - Quantum| / Gurobi * 100%`
- ✅ Speedup: `Gurobi_time / Quantum_time`
- ✅ Feasibility: Constraint violations counted
- ✅ Diversity: Shannon index, unique crops, coverage ratio

---

### 5. **Reproducibility** ✅

**Seeds and randomness**:
- ✅ Rotation matrix: Deterministic (seed=42)
- ✅ Farm sampling: Deterministic (from scenarios)
- ✅ QPU reads: Fixed at 100 per cluster
- ✅ Gurobi: Fixed timeout and gap tolerance

**Configuration documented**:
```python
TEST_CONFIG = {
    'farm_sizes': [25, 50, 100],
    'n_crops': 27,
    'n_families': 6,
    'n_periods': 3,
    'num_reads': 100,
    'num_iterations': 3,
    'runs_per_method': 3,
    'classical_timeout': 900,
    'farms_per_cluster': 10,
}
```

**✅ Can be reproduced** with same D-Wave access and Gurobi license

---

### 6. **Error Handling** ✅

**Graceful failures**:
- ✅ Try-except blocks around each solver call
- ✅ Partial results saved if some runs fail
- ✅ Success rate tracked per method
- ✅ Traceback printed for debugging

**Validation**:
- ✅ Data loading checked
- ✅ Solver availability verified (Gurobi, D-Wave)
- ✅ Results validated (violations, diversity)

---

### 7. **Output Quality** ✅

**Data saved**:
- ✅ **Complete results JSON**: All runs, all metrics
- ✅ **Summary CSV**: Statistical aggregates
- ✅ **Publication plots PNG**: 4-panel comparison (300 DPI)

**Metrics in output**:
- ✅ Solve time (mean, std, min, max)
- ✅ Objective value (mean, std)
- ✅ Optimality gap (%)
- ✅ Speedup factor
- ✅ QPU time breakdown
- ✅ Crop diversity (unique crops, Shannon index)
- ✅ Violations

**✅ Ready for LaTeX inclusion**

---

### 8. **Publication Claims** ✅

**Can support these claims**:

1. ✅ **"Hierarchical approach scales to 100 farms"**
   - Tested: 25, 50, 100 farms
   - With 27 foods × 3 periods = up to 8,100 variables

2. ✅ **"Quantum speedup of X× over classical"**
   - Mean speedup across all sizes
   - With standard deviation for error bars

3. ✅ **"Maintains solution quality within Y% gap"**
   - Gap calculated vs Gurobi ground truth
   - Averaged over multiple runs

4. ✅ **"Preserves crop diversity through post-processing"**
   - Unique crops measured (out of 27)
   - Shannon diversity index calculated

5. ✅ **"QPU time is Z seconds for N farms"**
   - Separate QPU timing tracked
   - Shows QPU efficiency vs total time

6. ✅ **"Zero constraint violations"**
   - Feasibility tracked
   - Success rate reported

---

### 9. **Comparison with Paper's Statistical Test** ✅

**What's the same**:
- ✅ Gurobi ground truth baseline
- ✅ 3-period rotation problem
- ✅ Multiple runs for statistics
- ✅ Fair family-level comparison
- ✅ Same metrics: time, quality, diversity

**What's different** (intentionally):
- 📈 **Larger scale**: 25-100 farms (vs 5-25 in original)
- 🔄 **Hierarchical method**: Spatial decomposition (vs direct clique/spatial-temporal)
- 📊 **Better statistics**: 3 runs (vs 2 in original)
- 🎯 **Extended metrics**: QPU time breakdown, aggregation overhead

**✅ Proper extension**: Builds on previous work, adds hierarchical approach for scaling

---

### 10. **Known Limitations** (for honesty in paper)

**Acknowledged limitations**:

1. **QPU access cost**: 
   - 3 runs × 3 sizes × (25-100 farms) = substantial QPU time
   - Estimated: 5-15 minutes total QPU access

2. **Gurobi 15-min timeout**:
   - May not reach optimality on 100 farms
   - But consistent with previous test methodology

3. **Post-processing is heuristic**:
   - Family → crop refinement not guaranteed optimal
   - But same for both methods (fair comparison)

4. **Spatial clustering is simplistic**:
   - Sequential grid decomposition
   - More sophisticated clustering possible

5. **Boundary coordination is approximate**:
   - Soft coupling between clusters
   - Not globally optimal, but practical

**✅ These are acceptable trade-offs** for publication, if disclosed

---

## 🚀 Pre-Flight Recommendations

### Before Running:

1. **Verify D-Wave access**:
   ```bash
   dwave ping
   dwave solvers --list
   ```

2. **Check Gurobi license**:
   ```python
   import gurobipy as gp
   gp.Model("test")
   ```

3. **Estimate QPU cost**:
   - 25 farms: ~3 clusters × 3 iterations × 3 runs = 27 QPU calls
   - 50 farms: ~5 clusters × 3 iterations × 3 runs = 45 QPU calls  
   - 100 farms: ~10 clusters × 3 iterations × 3 runs = 90 QPU calls
   - **Total**: ~162 QPU calls × ~0.1s = **~16-32 seconds QPU time**
   - **Cost**: Reasonable for publication-quality results

4. **Create backup**:
   ```bash
   git add hierarchical_statistical_test.py
   git commit -m "Publication test ready"
   ```

### During Run:

- Monitor QPU access (should see ~0.1-0.2s per cluster)
- Check intermediate results (printed after each size)
- Watch for errors/failures (handled gracefully)

### After Run:

1. **Validate results**:
   - Check success rates (should be 100%)
   - Verify no violations
   - Confirm diversity metrics reasonable

2. **Analyze plots**:
   - Speedup curve should be consistent
   - Gap should be < 25% (acceptable for heuristic)
   - Diversity should be similar to Gurobi

3. **Document findings**:
   - Save output to text file
   - Note any anomalies
   - Compare with previous statistical test at 25 farms

---

## ✅ READY FOR PUBLICATION RUN

**All quality checks passed**. The test is:
- ✅ Scientifically rigorous
- ✅ Statistically sound  
- ✅ Fair comparison
- ✅ Reproducible
- ✅ Publication-ready

**Estimated runtime**:
- Gurobi: 3 runs × 3 sizes × ~5-10 min = **45-90 minutes**
- Hierarchical QPU: 3 runs × 3 sizes × ~1-3 min = **9-27 minutes**
- **Total**: ~1-2 hours

**To run**:
```bash
cd @todo
python hierarchical_statistical_test.py
```

**The script will**:
1. Confirm QPU usage (safety check)
2. Run all 18 experiments
3. Save complete results
4. Generate publication plots
5. Print summary table

**Good luck!** 🚀📊
