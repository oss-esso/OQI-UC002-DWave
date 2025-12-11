# Quantum Speedup Roadmap - Usage Guide

This guide explains how to run the complete roadmap benchmarks implemented in `qpu_benchmark.py`.

## Overview

The roadmap follows the Mohseni et al. (2024) approach to achieve quantum speedup through:
1. **Decomposition** to small subproblems (≤16 variables)
2. **Clique embedding** for zero overhead
3. **Acceptable quality gap** (<15%)

## Quick Start

### Phase 1: Proof of Concept (4 farms)

```bash
# Run complete Phase 1 benchmark (recommended starting point)
python qpu_benchmark.py --roadmap 1 --token YOUR_DWAVE_TOKEN

# This tests:
# - Simple binary problem (4 farms × 6 crops, no rotation)
# - Rotation problem (4 farms × 6 crops × 3 periods)
# - Methods: Gurobi, Direct QPU, Clique QPU, Clique Decomp, Spatial+Temporal
```

**Success Criteria for Phase 1:**
- ✅ Optimality gap < 20% vs Gurobi
- ✅ Total QPU time < 1 second
- ✅ Embedding time ≈ 0 (cliques!)
- ✅ All constraints satisfied

### Phase 2: Scaling Validation (5, 10, 15 farms)

```bash
# Run Phase 2 (only after Phase 1 succeeds)
python qpu_benchmark.py --roadmap 2 --token YOUR_DWAVE_TOKEN

# This measures scaling and finds crossover point
```

**Success Criteria for Phase 2:**
- ✅ Quantum faster than Gurobi at F ≥ 12-15 farms
- ✅ Optimality gap < 15%
- ✅ Linear scaling (not exponential)

### Phase 3: Optimization (if Phase 2 successful)

```bash
# Run Phase 3 (advanced optimization techniques)
python qpu_benchmark.py --roadmap 3 --token YOUR_DWAVE_TOKEN
```

## Available Methods

### Simple Binary Problem (No Rotation/Synergy)

**Easiest baseline** - Test if D-Wave can handle basic assignment problem:

```bash
# Test simple problem with various methods
python qpu_benchmark.py --test 4 \
  --methods ground_truth direct_qpu clique_qpu \
  --token YOUR_DWAVE_TOKEN
```

**Problem:**
- N farms × M crops variables
- Linear objective only
- One crop per farm constraint
- NO temporal dimension, NO synergies

### Rotation Problem (With Temporal Synergies)

**Harder problem** - Full 3-period rotation optimization:

```bash
# Test rotation scenarios
python qpu_benchmark.py --scenario rotation_micro_25 \
  --methods ground_truth clique_decomp spatial_temporal \
  --token YOUR_DWAVE_TOKEN
```

**Problem:**
- N farms × M crops × 3 periods variables
- Quadratic rotation synergies (temporal coupling)
- Spatial neighbor interactions
- Diversity bonuses

## Detailed Method Descriptions

### 1. `ground_truth`
- **Solver:** Gurobi (classical optimizer)
- **Use:** Establishes optimal solution for comparison
- **Time:** Exponential growth with problem size
- **Quality:** Optimal (0% gap)

### 2. `direct_qpu`
- **Solver:** DWaveSampler + EmbeddingComposite
- **Use:** Direct QPU embedding (CQM → BQM → QPU)
- **Limit:** Only works for small problems (<100 vars)
- **Overhead:** High embedding time

### 3. `clique_qpu`
- **Solver:** DWaveCliqueSampler
- **Use:** Direct clique embedding (n ≤ 16)
- **Limit:** Only for tiny problems (≤16 vars)
- **Overhead:** ZERO embedding (uses hardware cliques)

### 4. `clique_decomp`
- **Solver:** Farm-by-farm + DWaveCliqueSampler
- **Use:** Decompose by farm (6 crops × 3 periods = 18 vars/farm)
- **Iterations:** 3 (boundary coordination)
- **Best for:** Rotation problems with moderate farms

### 5. `spatial_temporal` ⭐ **ROADMAP STRATEGY 1**
- **Solver:** Spatial clustering + Temporal decomposition + Cliques
- **Use:** The "sweet spot" approach from roadmap
- **Decomposition:**
  - Spatial: 5 farms → 3 clusters ([2,2,1] farms)
  - Temporal: 3 periods → solve one at a time
  - Result: 2 farms × 6 crops = **12 vars** per subproblem ✅ FITS CLIQUES!
- **Iterations:** 3 (boundary coordination)
- **Best for:** All rotation problems

## Example Workflows

### Workflow 1: Test Simple Problem First

```bash
# Step 1: Verify Gurobi works
python qpu_benchmark.py --test 4 --methods ground_truth

# Step 2: Test if D-Wave can handle simple problem
python qpu_benchmark.py --test 4 \
  --methods ground_truth direct_qpu clique_qpu \
  --token YOUR_DWAVE_TOKEN

# Step 3: If successful, try rotation problem
python qpu_benchmark.py --scenario rotation_micro_25 \
  --methods ground_truth spatial_temporal \
  --token YOUR_DWAVE_TOKEN
```

### Workflow 2: Full Roadmap Execution

```bash
# Phase 1: Proof of concept (4 farms)
python qpu_benchmark.py --roadmap 1 --token YOUR_DWAVE_TOKEN

# If Phase 1 succeeds (gap <20%, QPU <1s):
python qpu_benchmark.py --roadmap 2 --token YOUR_DWAVE_TOKEN

# If Phase 2 shows quantum advantage:
python qpu_benchmark.py --roadmap 3 --token YOUR_DWAVE_TOKEN
```

### Workflow 3: Custom Scaling Test

```bash
# Test specific farm counts
python qpu_benchmark.py --scale 4 6 8 10 \
  --methods ground_truth spatial_temporal \
  --reads 100 500 1000 \
  --token YOUR_DWAVE_TOKEN
```

## Understanding Results

### Metrics to Watch

1. **Optimality Gap** (% vs Gurobi)
   - <10%: Excellent ✅
   - 10-15%: Good ✅
   - 15-20%: Acceptable ⚠️
   - >20%: Poor ❌

2. **QPU Access Time** (seconds)
   - <0.5s: Excellent ✅
   - 0.5-1s: Good ✅
   - 1-2s: Acceptable ⚠️
   - >2s: Poor ❌

3. **Embedding Time** (seconds)
   - <0.01s: Perfect (cliques!) ✅
   - 0.01-0.1s: Good ✅
   - 0.1-1s: Acceptable ⚠️
   - >1s: Poor (not using cliques?) ❌

4. **Constraint Violations**
   - 0: Feasible ✅
   - 1-3: Minor ⚠️
   - >3: Infeasible ❌

### Reading Output

```
[Spatial+Temporal] Roadmap Strategy 1: 2 farms/cluster × 6 crops = 12 vars/subproblem
  Target: ≤16 vars (fits cliques!), 3 boundary iterations
  ✓ 12 ≤ 16: FITS CLIQUES (zero embedding overhead!)
  Created 3 spatial clusters → 9 total subproblems
  
  Iteration 1/3: obj=0.8542 (improved!)
  Iteration 2/3: obj=0.8731 (improved!)
  Iteration 3/3: obj=0.8745 (improved!)
  
  Complete! 9 subproblems (12 vars each)
  Total QPU=0.287s, embed=0.0032s
  Final objective: 0.8745, violations: 0
  
  Objective: 0.8745 (gap: 8.3%)  ← GAP VS GUROBI
  Clusters: 3 spatial × 3 periods = 9 subproblems
  Subproblem size: 12 vars ✓ FITS CLIQUE!  ← ZERO EMBEDDING!
  Total QPU time: 0.287s  ← FAST!
  Total embedding time: 0.0032s  ← ESSENTIALLY ZERO!
  Violations: 0  ← FEASIBLE!
```

**This is success!** Gap <10%, QPU <1s, embedding ≈0, feasible.

## Troubleshooting

### Problem: "DWaveCliqueSampler not available"

```bash
# Install missing dependencies
pip install dwave-system dwave-samplers
```

### Problem: "No D-Wave token available"

```bash
# Option 1: Pass token as argument
python qpu_benchmark.py --roadmap 1 --token YOUR_TOKEN

# Option 2: Set environment variable
export DWAVE_API_TOKEN="YOUR_TOKEN"
python qpu_benchmark.py --roadmap 1

# Option 3: Configure dwave (persistent)
dwave config create
```

### Problem: "Subproblem too large (18 vars > 16)"

The clique sampler works best for n ≤ 16. If your problem has:
- 6 crops × 3 periods = 18 vars/farm → Use `clique_decomp` (still works, minor chains)
- Reduce farms_per_cluster: `farms_per_cluster=1` → 6 crops = 6 vars ✅

### Problem: "Gap > 20%"

Try:
1. Increase iterations: `num_iterations=5`
2. Increase reads: `--reads 500 1000`
3. Use simpler problem first (binary, not rotation)
4. Check if Gurobi solution is actually optimal

### Problem: "Embedding time > 1s"

You're not using cliques! Check:
1. Subproblem size ≤ 16? 
2. Using `DWaveCliqueSampler`?
3. Try `clique_qpu` or `spatial_temporal` methods

## Cost Estimation

**QPU usage is very affordable:**

- **Phase 1 (4 farms):** ~$0.01-0.02 per run
- **Phase 2 (5-15 farms):** ~$0.05-0.10 per run
- **Phase 3 (optimization):** ~$0.20-0.50 per run

**Total for complete roadmap:** $5-20

**Per test run:**
- Small (≤10 farms): ~$0.01
- Medium (10-20 farms): ~$0.05
- Large (20-50 farms): ~$0.10-0.20

## Next Steps

1. **Start with Phase 1:**
   ```bash
   python qpu_benchmark.py --roadmap 1 --token YOUR_TOKEN
   ```

2. **Check success criteria:**
   - Gap < 20%? ✅ → Proceed to Phase 2
   - Gap > 30%? ❌ → Try simpler binary problem first

3. **Analyze bottlenecks:**
   - High embedding time? → Not using cliques
   - High QPU time? → Too many reads or subproblems
   - High gap? → Increase iterations or reads

4. **Scale up systematically:**
   - Phase 1 (4 farms) → Phase 2 (5-15 farms) → Phase 3 (optimization)

## Success Story (Optimistic)

**After running the roadmap, you might achieve:**

```
Phase 1 Results (4 farms):
  ✅ Gap: 8.3% (target: <20%) 
  ✅ QPU time: 0.29s (target: <1s)
  ✅ Embedding: 0.003s (target: ≈0)
  ✅ Feasible: 0 violations
  
Phase 2 Results (scaling):
  ✅ Crossover at F=12 farms: Quantum 2.1x faster than Gurobi
  ✅ Gap maintained: 12.1% average
  ✅ Linear scaling confirmed
  
→ QUANTUM ADVANTAGE DEMONSTRATED! 🎉
→ Ready for publication
```

---

**Let's achieve quantum speedup!** 🚀

The math says it should work. The roadmap is complete. Now we validate empirically.

