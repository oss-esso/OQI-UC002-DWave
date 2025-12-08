# Quantum Advantage Quest: Summary & Recommendations

**Date**: December 8, 2025  
**Analysis**: Gurobi Ground Truth Benchmark Results

## Executive Summary

We successfully benchmarked "quantum-friendly" reformulations against Gurobi to establish ground truth and assess classical hardness. The results reveal a **critical finding**: while these formulations are quantum-tractable (embeddable on QPU), they remain **too easy for classical solvers** (0% integrality gap).

## Benchmark Results

### Spin Glass Instances

| Size (n) | Status | MIP Objective | LP Objective | Gap | Time | Complexity |
|----------|--------|---------------|--------------|-----|------|------------|
| 20 | OPTIMAL | -18.04 | -18.04 | **0%** | 0.06s | O(n^4.4) |
| 50 | OPTIMAL | -92.23 | -92.23 | **0%** | 0.08s | ↓ |
| 100 | OPTIMAL | -224.90 | -224.90 | **0%** | 92.8s | ↓ |
| 200 | TIME_LIMIT | -502.90 | N/A | N/A | 300s | ↓ |
| 1000 | TIME_LIMIT | -4935.86 | N/A | N/A | 300s | ↓ |

**Extrapolated**: n=1000 → 139 hours, n=5000 → 18 years (!)

### Farm Type Instances

| Farms (n) | Status | MIP Objective | LP Objective | Gap | Time | Complexity |
|-----------|--------|---------------|--------------|-----|------|------------|
| 25 | OPTIMAL | -13.03 | -13.03 | **0%** | 4.3s | O(n^2.9) |
| 50 | OPTIMAL | -26.16 | -26.16 | **0%** | 80.7s | ↓ |
| 100 | OPTIMAL | -52.44 | -52.44 | **0%** | 244s | ↓ |
| 200 | TIME_LIMIT | -105.38 | N/A | N/A | 300s | ↓ |
| 1000 | TIME_LIMIT | -524.07 | N/A | N/A | 301s | ↓ |

**Extrapolated**: n=1000 → 77 hours, n=2000 → 583 hours

## The Hardness Paradox

```
╔══════════════════════════════════════════════════╗
║  CURRENT STATUS: NOT IDEAL FOR QUANTUM ADVANTAGE ║
╚══════════════════════════════════════════════════╝

Classical:  EASY (0% gap) → HARDER (poly time) → HARD (exp. time)
            ✓ We're here              Goal →

Quantum:    IMPOSSIBLE → HARD → FEASIBLE
                                ✓ We're here
```

**What we achieved:**
- ✅ Quantum-tractable (bounded degree ~40, embeddable)
- ✅ Harder than original (requires branching, minutes vs <1s)

**What we're missing:**
- ❌ Classical integrality gap (0% = LP finds integer solution)
- ❌ Exponential classical scaling (have polynomial O(n³-n⁴))

## Why Instances Are Too Easy

### 1. One-Hot Penalty Too Strong
```python
penalty = 10.0  # Choose exactly one type per farm
# LP naturally respects this → no fractional variables
```

### 2. Insufficient Frustration
```
Current: 30% frustrated edges
Needed:  50-70% (spin glass phase transition)

Coupling strengths: ±0.15 (too weak)
Needed: ±1.0 (strong competition)
```

### 3. Problem Structure Too Simple
- Assignment problem with local interactions
- Classical branch-and-bound exploits structure efficiently
- No global constraints forcing complex tradeoffs

## Connection to Your Existing Work

### Your Rotation Formulation is 80% There!

| Feature | Status | Impact |
|---------|--------|--------|
| Quadratic terms | ✅ Y[f,c,t-1] × Y[f,c,t] | Native QUBO |
| Temporal sparsity | ✅ Adjacent periods only | Bounded interactions |
| Real agronomic data | ✅ Rotation matrix | Meaningful |
| **Spatial interactions** | ❌ Missing | Need neighbor farms |
| **Frustrated interactions** | ❌ All R ≥ 0 | Need negative synergies |
| **Bounded degree** | ❌ 27² per farm | Too dense |

### What's Missing

**Your rotation matrix has NO negative synergies:**
- All values: 0.0 to 0.078 (purely beneficial)
- No competition → No frustration → LP is tight
- Classical solver wins easily

**Agronomically justified negative synergies:**
1. Same-family consecutive planting (nutrient depletion)
2. Disease carryover (tomato → potato)
3. Allelopathy (walnut → vegetables)
4. Pest harbor (corn → corn)

## Recommendations

### **Option A: Enhanced Rotation Formulation** ⭐ RECOMMENDED

**Why**: Combines quantum tractability + agricultural meaning + potential hardness

**Modifications needed:**

```python
# 1. Add negative synergies to rotation matrix
R_modified[crop1, crop2] = R[crop1, crop2] - penalty  # if same family
                         = R[crop1, crop2] - penalty  # if disease risk
                         = R[crop1, crop2]            # otherwise

# 2. Add spatial interactions (neighbor farms)
for (farm1, farm2) in spatial_neighbors:
    for (crop1, crop2) in crop_pairs:
        synergy = compute_spatial_synergy(crop1, crop2)
        # Positive: pollination, pest control
        # Negative: pest spread, resource competition

# 3. Reduce crop choices per farm (or use decomposition)
n_crop_families = 6  # Instead of 27 individual crops
```

**Expected outcome:**
- Integrality gap: 5-15% (target)
- Classical time: hours for n=200+
- Quantum feasible: degree ~30-50
- Preserves agricultural meaning ✅

### **Option B: Hardcore Spin Glass**

Generate genuinely NP-hard instances from:
- 3-SAT at phase transition (clause/variable ratio ~4.2)
- Graph 3-coloring on random graphs
- MaxCut with planted hardness

**Pros**: Guaranteed hard for classical  
**Cons**: Loses agricultural meaning entirely

### **Option C: Accept Current Formulation**

Focus on:
- QPU embedding demonstration
- Runtime comparison at scale (n=1000+)
- Hybrid quantum-classical workflows
- Acknowledge classical advantage for n<1000

## Action Plan (Option A)

### Phase 1: Enhance Rotation Matrix (1 week)
1. Load existing rotation matrix
2. Add negative synergies for competing crops
3. Validate with agronomist/literature
4. Target: 30-50% negative entries

### Phase 2: Add Spatial Structure (1 week)
1. Implement k-nearest neighbor farm graph
2. Add spatial synergy/conflict terms
3. Test multiple k values (k=4, 6, 8)
4. Target: max degree <50

### Phase 3: Benchmark Classical Hardness (3 days)
1. Generate instances n=25, 50, 100, 200, 500
2. Run Gurobi with 1hr time limit
3. Measure integrality gap, nodes, time
4. Target: >5% gap for n≥100

### Phase 4: QPU Benchmark (1 week)
1. Embed on Advantage_system6
2. Compare QPU vs SA vs Gurobi
3. Analyze time-to-solution
4. Document quantum advantage regime (if found)

## Expected Timeline

- Week 1-2: Rotation formulation enhancement
- Week 3: Classical hardness validation
- Week 4: QPU benchmarking
- Week 5: Analysis & writeup

**Total**: ~5 weeks to complete quantum advantage assessment

## Key Metrics for Success

| Metric | Current | Target | Rationale |
|--------|---------|--------|-----------|
| Integrality Gap | 0% | >5% | Proves LP struggles |
| Classical Time (n=200) | 5min | >1hr | Forces timeout |
| Max Degree | 40 | <50 | Embeddable on Pegasus |
| Agricultural Meaning | Lost | Preserved | Real-world relevance |
| Negative Synergies | 0% | 30-50% | Frustration source |

## Conclusion

Your rotation formulation is **the most promising path forward** because:
1. Already has quadratic structure ✅
2. Preserves agricultural meaning ✅
3. Can add frustration realistically ✅
4. You've already implemented it ✅

**Simple modifications** (negative synergies + spatial interactions) can push it into the quantum advantage regime while keeping real-world relevance.

The synthetic spin glass approach proved that **bounded degree is achievable**, but sacrificed problem meaning. The rotation formulation lets us have both.

## Files Generated

- `@todo/hardness_output/gurobi_ground_truth_benchmark.json` - Full benchmark results
- `@todo/comprehensive_analysis.py` - Analysis script
- `@todo/QUANTUM_ADVANTAGE_REFORMULATION.md` - Technical details
- `@todo/QUANTUM_ADVANTAGE_SUMMARY.md` - This document

---

**Next Step**: Modify rotation matrix to add negative synergies and re-benchmark.
