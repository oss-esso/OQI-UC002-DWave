# Quantum Speedup Implementation Roadmap
## Achieving Mohseni-Style Results for Rotation Optimization

**Date:** December 10, 2025  
**Goal:** Adapt rotation problem to achieve quantum speedup using decomposition + clique embedding  
**Based on:** Critical analysis of Mohseni et al. (2024) verification

---

## 🎯 The Core Insight

**Mohseni's success formula:**
```
Quantum Speedup = Decomposition (→ n ≤ 16) + DWaveCliqueSampler (zero overhead) + Moderate quality loss
```

**Our challenge:**
- Current: 90-900 variable monolithic problem → 87% optimality gap
- Target: Decompose to ≤16 variable subproblems → maintain >80% quality

---

## 🚀 Five Proposed Strategies (Ranked by Promise)

### ⭐ Strategy 1: Spatial + Temporal Decomposition (HIGHEST PRIORITY)

**The Sweet Spot Approach**

#### Problem Transformation
- **Before:** 90 vars (5 farms × 6 crops × 3 periods)
- **After:** 6 subproblems of 12 vars each (2 farms × 6 crops × 1 period)

#### Algorithm
```
1. Cluster farms spatially: 5 farms → 3 clusters ([F1,F2], [F3,F4], [F5])
2. For each period t=1,2,3:
   For each cluster k=1,2,3:
      Variables: Y[f,c,t] for f in cluster_k, all c
      Size: 2 farms × 6 crops = 12 variables ✓ FITS CLIQUE!
      Solve with DWaveCliqueSampler (zero embedding)
      Boundary: coordinate with adjacent clusters
3. Refine boundaries between clusters (iterative)
```

#### Why This Works
- ✅ Fits clique perfectly (12 ≤ 16)
- ✅ Preserves temporal rotation coupling (within period)
- ✅ Preserves spatial coupling (within cluster)
- ✅ Zero embedding overhead
- ✅ Direct analog to Mohseni's hierarchical approach

#### Test Plan (Week 1-2)
```python
# Baseline: 4 farms, 6 crops, 3 periods
F = 4  # 2 clusters of 2 farms each
C = 6
T = 3

# Subproblem count
subproblems = 2 clusters × 3 periods = 6 subproblems
vars_per_subproblem = 2 × 6 = 12  # Fits clique!

# Implementation steps
1. Implement farm clustering (simple: adjacent pairs)
2. Build subproblem QUBOs (12×12 matrices)
3. Solve with DWaveCliqueSampler
4. Add boundary coordination (fix edge farms)
5. Iterate until convergence

# Metrics to track
- QPU calls: expect ~6-20 (6 subproblems × 1-3 iterations)
- QPU time per call: expect ~20-30ms
- Total QPU time: expect ~200-600ms
- Embedding time: 0 (cliques!)
- Optimality gap vs Gurobi: target <15%
```

#### Expected Outcome
- **Optimistic:** 10% gap, 500ms total QPU time, zero embedding → **SPEEDUP!**
- **Realistic:** 15% gap, competitive with Gurobi at F≥10
- **Pessimistic:** 25% gap, boundary coordination overhead too high

---

### Strategy 2: Temporal Decomposition (FALLBACK)

**Simpler but Less Optimal**

#### Problem Transformation
- **Before:** 90 vars (5 farms × 6 crops × 3 periods)
- **After:** 3 subproblems of 30 vars each (5 farms × 6 crops × 1 period)

#### Algorithm
```
For t=1,2,3:
   Variables: Y[f,c,t] for all f,c
   Size: 5 × 6 = 30 variables (too large for clique)
   Use LeapHybridCQMSampler instead
   Fix previous period for rotation coupling
```

#### Why This Is Easier
- ✅ Simple to implement (period-by-period)
- ✅ Preserves all spatial coupling
- ⚠️ Doesn't fit cliques (need hybrid solver)
- ⚠️ Still has embedding overhead

#### Test Plan (Week 1)
Quick implementation to establish baseline for decomposition approach.

---

### Strategy 3: Rolling Horizon with Quantum Core

**For Larger Problems**

#### Problem Transformation
- **Before:** 150 vars (10 farms × 5 crops × 3 periods)
- **After:** Sliding window of 2 periods at a time (10 × 5 × 2 = 100 vars)

#### Algorithm
```
For t=1 to T-1:
   Optimize periods [t, t+1] with hybrid solver
   Commit period t
   Roll forward to t+1
```

#### Best For
- Larger problems (F ≥ 10 farms)
- When full temporal coupling is critical
- Hybrid solver benchmarking

---

### Strategy 4: Hierarchical Variable Aggregation

**Two-Level Coarse-to-Fine**

#### Problem Transformation
```
Level 1 (Coarse): Aggregate 6 families → 3 super-families
   Variables: 5 farms × 3 super × 3 periods = 45 vars
   Solve with hybrid

Level 2 (Fine): For each super-family allocated to a farm:
   Variables: 2-3 crops per super-family
   Size: 5 farms × 2 crops = 10 vars ✓ FITS CLIQUE!
   Solve with DWaveCliqueSampler
```

#### Why Interesting
- ✅ Refinement step fits cliques
- ✅ Two-level hierarchy matches Mohseni
- ⚠️ Requires domain knowledge for aggregation
- ⚠️ Coarse step may prune good solutions

---

### Strategy 5: Constraint Relaxation + Iterative Tightening

**Advanced Technique**

#### Algorithm
```
1. Start with soft penalties (unconstrained QUBO)
2. Decompose to fit cliques (12-16 vars)
3. Solve all subproblems with DWaveCliqueSampler
4. Check constraint violations
5. Increase penalties for violations
6. Repeat until feasible
```

#### Risk
- May never find feasible solution
- Penalty tuning is tricky

---

## 📊 Testing Protocol

### Phase 1: Proof of Concept (Weeks 1-2)

**Goal:** Does decomposition + clique embedding work at all?

**Test Case:**
- 4 farms, 6 crops, 3 periods (72 vars total)
- Decompose to 2×3 = 6 subproblems of 12 vars each

**Baseline Comparisons:**
1. Gurobi monolithic (optimal solution)
2. Gurobi temporal decomposition (quality upper bound)
3. D-Wave clique decomposition (our approach)

**Success Criteria:**
- [ ] Optimality gap < 20% vs Gurobi
- [ ] QPU time < 1 second total
- [ ] Embedding time = 0 (cliques)
- [ ] All constraints satisfied

**Implementation Checklist:**
```python
# Week 1: Infrastructure
[ ] Implement farm clustering (spatial)
[ ] Build QUBO for single subproblem (12 vars)
[ ] Test DWaveCliqueSampler on one subproblem
[ ] Verify clique fitting (no chains)

# Week 2: Full Pipeline
[ ] Solve all 6 subproblems independently
[ ] Implement boundary coordination
[ ] Add iterative refinement
[ ] Benchmark vs Gurobi
[ ] Measure optimality gap, timing
```

### Phase 2: Scaling Validation (Weeks 3-4)

**Goal:** How does it scale?

**Test Sizes:**
- F=5 farms: 90 vars → 9 subproblems (3 clusters × 3 periods)
- F=10 farms: 180 vars → 15 subproblems (5 clusters × 3 periods)
- F=15 farms: 270 vars → 24 subproblems (8 clusters × 3 periods)

**Metrics:**
- Optimality gap vs problem size
- Total QPU time vs problem size
- Gurobi time vs quantum time (crossover point?)

**Expected Crossover:**
- At F≈15-20, Gurobi starts slowing down (exponential growth)
- Quantum approach scales linearly (more subproblems)
- **Hypothesis:** Quantum wins at F≥20 if gap <15%

### Phase 3: Optimization & Refinement (Weeks 5-6)

**Goal:** Squeeze out every percentage point

**Techniques:**
1. **Better clustering:** Use Louvain algorithm on spatial adjacency
2. **Boundary optimization:** Local search at cluster boundaries
3. **Adaptive penalties:** Tune rotation bonuses per iteration
4. **Parallel QPU calls:** Submit all subproblems simultaneously
5. **Warm starting:** Use previous period solution as initial state

---

## 🎯 Success Metrics

### Tier 1: Minimum Viable Success ✅
- Optimality gap < 20% vs Gurobi
- Zero embedding overhead (cliques confirmed)
- Faster than monolithic D-Wave (87% gap baseline)

### Tier 2: Competitive Performance 🎖️
- Optimality gap < 15% vs Gurobi
- Total time competitive with Gurobi at F=10
- Faster than Gurobi at F≥15

### Tier 3: Quantum Advantage 🏆
- Optimality gap < 10% vs Gurobi
- Faster than Gurobi for all F≥10
- Scalable to F=50+ farms
- Publishable results matching Mohseni's impact

---

## 💰 Cost Estimation

### QPU Usage (Per Test Run)

**Small Scale (F=4):**
- 6 subproblems × 100 reads × 20μs = 12,000 QPU-μs
- ~3-5 iterations = 36,000-60,000 QPU-μs
- Cost: ~$0.01-0.02 per full test

**Medium Scale (F=10):**
- 15 subproblems × 100 reads × 20μs = 30,000 QPU-μs
- ~5-10 iterations = 150,000-300,000 QPU-μs
- Cost: ~$0.05-0.10 per full test

**Total Budget for Phase 1-3:**
- ~100-200 test runs across all phases
- Total cost: $5-20 (very affordable!)

---

## 🛠️ Implementation Priorities

### Week 1: Foundation (Strategy 1 - Spatial+Temporal)
```
Day 1-2: Infrastructure
  - Farm clustering implementation
  - QUBO builder for 12-var subproblems
  - DWaveCliqueSampler integration

Day 3-4: Single Subproblem
  - Test one subproblem (2 farms, 6 crops, 1 period)
  - Verify clique embedding (check chains=0)
  - Validate solution quality

Day 5: Full Decomposition
  - Solve all 6 subproblems independently
  - Merge solutions
  - Compare vs Gurobi monolithic
```

### Week 2: Refinement
```
Day 1-2: Boundary Coordination
  - Implement boundary fixing
  - Iterative refinement loop
  - Convergence criteria

Day 3-4: Testing & Benchmarking
  - Run full pipeline 20+ times
  - Measure optimality gap distribution
  - Timing breakdown (QPU, Python, coordination)

Day 5: Analysis & Reporting
  - Compare all metrics vs Gurobi
  - Identify bottlenecks
  - Decision: proceed to scaling or pivot?
```

### Week 3-4: Scaling (If Phase 1 Successful)
- Test F ∈ {5, 10, 15} farms
- Measure scaling curves
- Identify crossover point

### Week 5-6: Optimization (If Phase 2 Promising)
- Advanced clustering
- Parallel QPU calls
- Publication-quality benchmarks

---

## 🚨 Risk Mitigation

### Risk 1: Decomposition Destroys Quality (>30% gap)
**Mitigation:**
- Fall back to Strategy 2 (temporal only, hybrid solver)
- Try Strategy 4 (hierarchical aggregation)
- Accept hybrid solver as "good enough"

### Risk 2: Boundary Coordination Overhead Too High
**Mitigation:**
- Reduce iterations (accept slightly worse quality)
- Use parallel QPU calls (reduce wall-clock time)
- Pre-compute good boundary conditions

### Risk 3: Clique Embedding Fails (n>16)
**Mitigation:**
- Adjust cluster size (1 farm per cluster → n=6)
- Use hybrid for subproblems that don't fit
- Accept mixed approach (some clique, some chains)

---

## 📈 Expected Timeline & Deliverables

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Strategy 1 Implementation | Working clique-based solver for 4 farms |
| 2 | Benchmarking & Refinement | Optimality gap < 20% demonstrated |
| 3 | Scaling to F=10 | Scaling curves, crossover analysis |
| 4 | Optimization | Best-case performance achieved |
| 5-6 | Publication Prep | Paper draft, final benchmarks |

---

## 🎓 Learning from Mohseni

**What They Did Right:**
1. ✅ Matched algorithm to hardware (cliques)
2. ✅ Decomposed to fit constraints
3. ✅ Accepted approximate solutions
4. ✅ Used iterative refinement
5. ✅ Compared to heuristics (not exact solvers)

**What We Must Do:**
1. ✅ Decompose rotation to ≤16 vars per subproblem
2. ✅ Use DWaveCliqueSampler exclusively
3. ✅ Accept 10-15% gap (not exact optimum)
4. ✅ Iterate to refine boundaries
5. ✅ Benchmark fairly (acknowledge gap vs Gurobi)

**The Key Difference:**
- **Mohseni:** Problem naturally decomposes hierarchically
- **Us:** Must engineer decomposition strategy
- **Solution:** Spatial clustering + temporal splitting = artificial hierarchy

---

## 🏁 Go/No-Go Decision Points

### After Week 1:
- **GO if:** Clique embedding confirmed, single subproblem solves correctly
- **NO-GO if:** Can't fit 12 vars in clique, or constraints unsatisfiable

### After Week 2:
- **GO if:** Gap < 25% vs Gurobi, clear path to improvement
- **PIVOT if:** Gap > 30%, try Strategy 2 or 4
- **NO-GO if:** Fundamental issues with decomposition approach

### After Week 4:
- **PUBLISH if:** Gap < 15% and faster than Gurobi at F≥15
- **CONTINUE if:** Promising but needs optimization
- **CONCLUDE if:** No quantum advantage found, document lessons learned

---

## 📚 Success Story (Optimistic Scenario)

**Imagine Week 6:**

"We successfully adapted the rotation problem using spatial+temporal decomposition:
- ✅ 12 variables per subproblem (perfect clique fit)
- ✅ Zero embedding overhead
- ✅ 12% average optimality gap vs Gurobi
- ✅ Faster than Gurobi for F≥12 farms
- ✅ Scales linearly to F=50+ farms

**Result:** Quantum advantage demonstrated for crop rotation scheduling, following Mohseni's decomposition paradigm. Paper submitted to journal."

---

**Let's make it happen!** 🚀

The key is starting small (F=4), validating the clique approach works, then scaling systematically. The math says it should work - now we need to prove it empirically.
