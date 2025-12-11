# Phase 3 QPU Subproblem Count Analysis

## Configuration Summary

**Test Scales:** 3 scales (10, 15, 20 farms)  
**Optimization Strategies:** 5 strategies  
**Total Configurations:** 3 scales × 5 strategies = **15 test runs**

---

## Strategy Configurations

| Strategy | Iterations | Farms/Cluster | Reads |
|----------|-----------|---------------|-------|
| 1. Baseline (Phase 2) | 3 | 2 | 100 |
| 2. Increased Iterations | 5 | 2 | 100 |
| 3. Larger Clusters | 3 | 3 | 100 |
| 4. Hybrid | 5 | 3 | 100 |
| 5. High Reads | 3 | 2 | 500 |

---

## Subproblem Calculation Formula

For each test configuration:

```
n_farms = number of farms
farms_per_cluster = farms per spatial cluster
n_periods = 3 (always)
n_iterations = number of coordination iterations

# Spatial clustering
n_clusters = ceil(n_farms / farms_per_cluster)

# Total subproblems per iteration
subproblems_per_iteration = n_clusters × n_periods

# Total across all iterations
total_subproblems = subproblems_per_iteration × n_iterations
```

---

## Detailed Breakdown by Scale

### Scale 1: 10 farms

| Strategy | Farms/Cluster | Clusters | Periods | Iter | Subproblems/Iter | Total Subproblems |
|----------|---------------|----------|---------|------|------------------|-------------------|
| Baseline | 2 | 5 | 3 | 3 | 15 | **45** |
| Increased Iter | 2 | 5 | 3 | 5 | 15 | **75** |
| Larger Clusters | 3 | 4 | 3 | 3 | 12 | **36** |
| Hybrid | 3 | 4 | 3 | 5 | 12 | **60** |
| High Reads | 2 | 5 | 3 | 3 | 15 | **45** |

**Subtotal for 10 farms:** 45 + 75 + 36 + 60 + 45 = **261 subproblems**

---

### Scale 2: 15 farms

| Strategy | Farms/Cluster | Clusters | Periods | Iter | Subproblems/Iter | Total Subproblems |
|----------|---------------|----------|---------|------|------------------|-------------------|
| Baseline | 2 | 8 | 3 | 3 | 24 | **72** |
| Increased Iter | 2 | 8 | 3 | 5 | 24 | **120** |
| Larger Clusters | 3 | 5 | 3 | 3 | 15 | **45** |
| Hybrid | 3 | 5 | 3 | 5 | 15 | **75** |
| High Reads | 2 | 8 | 3 | 3 | 24 | **72** |

**Subtotal for 15 farms:** 72 + 120 + 45 + 75 + 72 = **384 subproblems**

---

### Scale 3: 20 farms

| Strategy | Farms/Cluster | Clusters | Periods | Iter | Subproblems/Iter | Total Subproblems |
|----------|---------------|----------|---------|------|------------------|-------------------|
| Baseline | 2 | 10 | 3 | 3 | 30 | **90** |
| Increased Iter | 2 | 10 | 3 | 5 | 30 | **150** |
| Larger Clusters | 3 | 7 | 3 | 3 | 21 | **63** |
| Hybrid | 3 | 7 | 3 | 5 | 21 | **105** |
| High Reads | 2 | 10 | 3 | 3 | 30 | **90** |

**Subtotal for 20 farms:** 90 + 150 + 63 + 105 + 90 = **498 subproblems**

---

## TOTAL QPU SUBPROBLEMS FOR PHASE 3

**Grand Total:** 261 + 384 + 498 = **1,143 QPU subproblems**

Plus **Ground Truth runs:** 3 (one per scale, using Gurobi, not QPU)

---

## Estimated Runtime Analysis

### Per Subproblem Timing (from Phase 2 results):
- QPU access time: ~0.036s per subproblem (Clique sampler)
- Embedding time: ~0.0000s (zero embedding overhead for cliques)
- Overhead (Python, data transfer): ~0.01s per subproblem

**Estimated time per subproblem:** ~0.05s

### Total Estimated Runtime:

| Component | Count | Time/Unit | Total Time |
|-----------|-------|-----------|------------|
| QPU subproblems | 1,143 | 0.05s | **57 seconds** |
| Ground truth (Gurobi) | 3 | 300s (timeout) | **900 seconds** |
| Python overhead | 15 configs | 5s | **75 seconds** |
| **TOTAL ESTIMATED** | | | **~17 minutes** |

### Breakdown by Scale:

- **10 farms:** 261 subproblems × 0.05s = ~13s QPU + 300s Gurobi = **~5 min total**
- **15 farms:** 384 subproblems × 0.05s = ~19s QPU + 300s Gurobi = **~6 min total**
- **20 farms:** 498 subproblems × 0.05s = ~25s QPU + 300s Gurobi = **~6 min total**

**Total:** ~17 minutes

---

## QPU Token/Credit Usage

**D-Wave Free Tier:**
- Limit: 2000 problems/month
- Phase 3 usage: **1,143 problems**
- Remaining: **857 problems** (42% of quota)

**QPU Time:**
- Estimated: 1,143 × 0.036s = **41 seconds**
- Free tier QPU time limit: Typically 20 minutes/month
- Usage: **~3.4%** of monthly QPU time quota

---

## Optimization Opportunities

If Phase 3 is too slow, you can reduce subproblems by:

1. **Test fewer scales:** Test only 10 and 20 farms (skip 15)
   - Reduces to: 261 + 498 = **759 subproblems** (-33%)

2. **Remove "Increased Iterations" strategy:** Only 3 iterations max
   - Reduces to: ~**800 subproblems** (-30%)

3. **Skip ground truth on large scales:** Only run Gurobi for 10 farms
   - Saves: 2 × 300s = **10 minutes**

4. **Combine optimizations:** Test [10, 20] with 4 strategies
   - Reduces to: ~**600 subproblems, ~12 minutes total**

---

## Conclusion

**Phase 3 will submit 1,143 QPU subproblems** across 15 test configurations, taking approximately **17 minutes** to complete.

This is well within D-Wave free tier limits and should demonstrate:
- ✅ Which optimization strategy works best at each scale
- ✅ Quality vs speed tradeoffs
- ✅ Scalability of the decomposition approach
- ✅ Publication-ready performance metrics
