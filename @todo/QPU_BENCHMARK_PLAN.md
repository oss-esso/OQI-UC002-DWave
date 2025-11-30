# QPU Benchmark Plan: Binary CQM Formulation Testing

## Executive Summary

Based on the comprehensive scaling benchmark results, this document outlines the plan for testing the **binary CQM formulation** directly on D-Wave quantum processing units (QPUs). We will benchmark multiple approaches including decomposition strategies and D-Wave's native hybrid solvers.

---

## 1. Key Findings from Simulation Benchmarks

### Hardware Constraints
| Scale | Logical Qubits | PlotBased Phys. Qubits | D-Wave Advantage Capacity | Feasibility |
|-------|---------------|------------------------|---------------------------|-------------|
| 25 farms | 675 | 2,600 | ~5,600 | ✅ Fits |
| 50 farms | 1,350 | 5,200 | ~5,600 | ⚠️ Borderline |
| 100 farms | 2,700 | 10,400 | ~5,600 | ❌ Too large |
| 200 farms | 5,400 | 20,800 | ~5,600 | ❌ Too large |

### Best Performing Decomposition Methods (CQM Binary)
| Method | Embeddable | Gap | Violations | Chain Length | Recommendation |
|--------|------------|-----|------------|--------------|----------------|
| **PlotBased** | ✅ All scales | 0% | 0 | 5 | **Primary choice** |
| **Cutset(2)** | ✅ All scales | 0% | 0 | 6 | Good fallback |
| **Multilevel(5)** | ✅ All scales | 0% | 0 | 7 | Good fallback |
| Spectral(4) | ❌ Fails at 50+ | 0% | 0 | 9 | Not recommended |
| Louvain | ✅ All scales | 13-85% | 2-3 | 5 | Not recommended |
| None | ❌ All scales | 0% | 0 | N/A | Not recommended |

---

## 2. Methods to Benchmark on QPU

We will test the following approaches for solving large binary CQM problems:

### 2.1 Direct QPU Submission (Small Scale Only)
**Method:** `DWaveSampler + EmbeddingComposite`
- Convert CQM to BQM with penalty-based constraints
- Embed directly onto QPU topology
- **Limitation:** Only feasible for 25 farms (2,600 qubits)

```python
from dwave.system import DWaveSampler, EmbeddingComposite

sampler = EmbeddingComposite(DWaveSampler())
bqm, info = cqm_to_bqm(cqm)
sampleset = sampler.sample(bqm, num_reads=1000)
```

### 2.2 Leap Hybrid CQM Solver (Cloud-Based)
**Method:** `LeapHybridCQMSampler`
- D-Wave's managed hybrid solver for CQMs
- Handles constraints natively (no penalty conversion)
- Automatically decomposes and routes to QPU
- **Capacity:** Up to 5,000,000 variables, 100,000 constraints

```python
from dwave.system import LeapHybridCQMSampler

sampler = LeapHybridCQMSampler()
sampleset = sampler.sample_cqm(cqm, time_limit=180, label="QPU Benchmark")
```

### 2.3 Leap Hybrid BQM Solver (Cloud-Based)
**Method:** `LeapHybridBQMSampler`
- For penalty-based BQM representation
- Large-scale problem support
- Hybrid classical-quantum optimization

```python
from dwave.system import LeapHybridBQMSampler

sampler = LeapHybridBQMSampler()
bqm, info = cqm_to_bqm(cqm)
sampleset = sampler.sample(bqm, time_limit=180)
```

### 2.4 dwave-hybrid Framework (Custom Workflows)
**Method:** Custom hybrid workflow using `dwave-hybrid`
- Build custom decomposition → QPU → recombination pipelines
- Uses `EnergyImpactDecomposer` or `RandomSubproblemDecomposer`
- Flexible control over QPU usage

```python
import hybrid
from dwave.system import DWaveSampler

# Define workflow
iteration = hybrid.RacingBranches(
    hybrid.InterruptableTabuSampler(),
    hybrid.EnergyImpactDecomposer(size=50)
    | hybrid.QPUSubproblemAutoEmbeddingSampler()
    | hybrid.SplatComposer()
) | hybrid.ArgMin()

workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)
```

### 2.5 Manual Decomposition + QPU
**Method:** Our proven decomposition strategies + direct QPU
- PlotBased, Multilevel(5), Cutset(2) decomposition
- Solve each partition on QPU
- Recombine solutions

```python
# Decompose
partitions = decompose_problem(cqm, method='PlotBased')

# Solve each partition on QPU
for partition_cqm in partitions:
    bqm, _ = cqm_to_bqm(partition_cqm)
    sampler = EmbeddingComposite(DWaveSampler())
    result = sampler.sample(bqm, num_reads=1000)

# Recombine
final_solution = recombine_solutions(partition_results)
```

---

## 3. Benchmark Test Matrix

### 3.1 Scale vs Method Matrix

| Scale | Direct QPU | Hybrid CQM | Hybrid BQM | dwave-hybrid | PlotBased+QPU | Multilevel+QPU |
|-------|------------|------------|------------|--------------|---------------|----------------|
| 25 farms | ✅ Test | ✅ Test | ✅ Test | ✅ Test | ✅ Test | ✅ Test |
| 50 farms | ⚠️ Try | ✅ Test | ✅ Test | ✅ Test | ✅ Test | ✅ Test |
| 100 farms | ❌ Skip | ✅ Test | ✅ Test | ✅ Test | ✅ Test | ✅ Test |
| 200 farms | ❌ Skip | ✅ Test | ✅ Test | ✅ Test | ✅ Test | ✅ Test |

### 3.2 Parameters to Vary

| Parameter | Values to Test | Purpose |
|-----------|---------------|---------|
| `num_reads` | 100, 500, 1000, 5000 | Solution quality vs time |
| `annealing_time` | 20µs, 100µs, 200µs | QPU sampling time |
| `chain_strength` | auto, 1.5×, 2× | Chain break handling |
| `time_limit` (hybrid) | 60s, 180s, 300s | Hybrid solver runtime |

---

## 4. Implementation Plan

### Phase 1: Setup & Validation (Week 1)

**Tasks:**
1. [ ] Verify D-Wave Leap API credentials and quota
2. [ ] Confirm solver availability (`DWaveSampler`, `LeapHybridCQMSampler`)
3. [ ] Create test script with 25-farm baseline problem
4. [ ] Validate CQM → BQM conversion produces correct penalties
5. [ ] Test embedding on 25-farm problem (ensure it fits)

**Deliverables:**
- Working connection to D-Wave Leap
- Baseline problem file validated
- Initial embedding test results

### Phase 2: Direct QPU Testing (Week 2)

**Tasks:**
1. [ ] Run 25-farm problem directly on QPU
2. [ ] Test different `num_reads` values (100, 500, 1000)
3. [ ] Test different `annealing_time` values
4. [ ] Record chain break fractions
5. [ ] Compare solution quality to classical ground truth

**Metrics to Collect:**
- QPU time (µs)
- Total wall clock time
- Chain break fraction
- Solution quality (gap to GT)
- Constraint violations

### Phase 3: Hybrid Solver Testing (Week 2-3)

**Tasks:**
1. [ ] Test `LeapHybridCQMSampler` on 25, 50, 100, 200 farms
2. [ ] Test `LeapHybridBQMSampler` on same scales
3. [ ] Compare solution quality and runtime
4. [ ] Test different `time_limit` values (60s, 180s, 300s)

**Key Questions:**
- Does hybrid CQM outperform hybrid BQM?
- What's the minimum time for good solutions?
- How does quota consumption scale?

### Phase 4: dwave-hybrid Framework Testing (Week 3)

**Tasks:**
1. [ ] Implement custom workflow with `EnergyImpactDecomposer`
2. [ ] Test `RandomSubproblemDecomposer` with various sizes
3. [ ] Compare custom workflow to Leap hybrid solvers
4. [ ] Tune decomposition parameters for best results

**Workflow Configurations:**
```python
# Configuration A: Energy Impact + Tabu Racing
workflow_a = hybrid.RacingBranches(
    hybrid.InterruptableTabuSampler(),
    hybrid.EnergyImpactDecomposer(size=50)
    | hybrid.QPUSubproblemAutoEmbeddingSampler()
    | hybrid.SplatComposer()
) | hybrid.ArgMin()

# Configuration B: Random Decomposition + SA Racing
workflow_b = hybrid.RacingBranches(
    hybrid.SimulatedAnnealingSubproblemSampler(),
    hybrid.RandomSubproblemDecomposer(size=75)
    | hybrid.QPUSubproblemAutoEmbeddingSampler()
    | hybrid.SplatComposer()
) | hybrid.ArgMin()
```

### Phase 5: Manual Decomposition + QPU (Week 4)

**Tasks:**
1. [ ] Implement PlotBased decomposition → QPU pipeline
2. [ ] Implement Multilevel(5) decomposition → QPU pipeline
3. [ ] Test solution recombination strategies
4. [ ] Compare to Leap hybrid solvers
5. [ ] Measure QPU time per partition

**Implementation:**
```python
def solve_with_decomposition(cqm, method='PlotBased', num_reads=1000):
    """Solve CQM using decomposition + direct QPU."""
    # 1. Decompose
    partitions = decompose_cqm(cqm, method=method)
    
    # 2. Solve each partition
    results = []
    for part_cqm in partitions:
        bqm, penalty_info = cqm_to_bqm(part_cqm)
        sampler = EmbeddingComposite(DWaveSampler())
        result = sampler.sample(bqm, num_reads=num_reads)
        results.append(result)
    
    # 3. Recombine
    final = recombine_partition_solutions(results)
    return final
```

### Phase 6: Analysis & Comparison (Week 5)

**Tasks:**
1. [ ] Compile all benchmark results
2. [ ] Generate comparison tables and charts
3. [ ] Identify best method per scale
4. [ ] Calculate QPU time and quota efficiency
5. [ ] Write final recommendations

---

## 5. Success Metrics

### 5.1 Solution Quality Metrics
| Metric | Target | How to Measure |
|--------|--------|----------------|
| Gap to Ground Truth | < 5% | `(solution - GT) / GT * 100` |
| Constraint Violations | 0 | Count infeasible constraints |
| Chain Break Fraction | < 5% | D-Wave sampleset info |

### 5.2 Performance Metrics
| Metric | Target | How to Measure |
|--------|--------|----------------|
| QPU Time | Report | Sampleset timing info |
| Total Wall Time | < 5 min (25 farms) | Python timing |
| Quota Efficiency | Report | Leap dashboard |

### 5.3 Scalability Metrics
| Metric | Target | How to Measure |
|--------|--------|----------------|
| Max Embeddable Scale | Report | Embedding success/failure |
| Time Scaling | Sub-exponential | Log-log plot |

---

## 6. Risk Mitigation

### 6.1 Embedding Failures
**Risk:** Problem too large for QPU
**Mitigation:** 
- Use decomposition (PlotBased proven to work)
- Fall back to hybrid solvers
- Reduce problem scale

### 6.2 Chain Breaks
**Risk:** High chain break fraction degrades solution quality
**Mitigation:**
- Increase chain strength (1.5× - 2× default)
- Use shorter chains (PlotBased has chain length 5)
- Post-process with local search

### 6.3 Poor Solution Quality
**Risk:** QPU solutions worse than classical
**Mitigation:**
- Increase `num_reads`
- Try different annealing schedules
- Use hybrid solvers with longer time limits

### 6.4 Quota Exhaustion
**Risk:** Running out of Leap minutes
**Mitigation:**
- Start with small scales
- Use simulated annealing for debugging
- Monitor quota usage per experiment

---

## 7. Resource Requirements

### 7.1 D-Wave Access
- Leap account with QPU access
- Estimated quota: ~60 minutes for full benchmark
- Advantage system preferred (5,600+ qubits)

### 7.2 Software Dependencies
```
dwave-ocean-sdk >= 6.0.0
dwave-system
dwave-hybrid
dimod
minorminer
dwave-networkx
```

### 7.3 Test Data
- 25, 50, 100, 200 farm configurations
- Ground truth solutions from Gurobi
- Pre-computed embeddings (optional, for speed)

---

## 8. Output Artifacts

### 8.1 Benchmark Script
`qpu_benchmark.py` - Main benchmark runner

### 8.2 Results Files
```
qpu_benchmark_results/
├── direct_qpu_25farms.json
├── hybrid_cqm_results.json
├── hybrid_bqm_results.json
├── dwave_hybrid_workflow_results.json
├── decomposition_qpu_results.json
└── summary_comparison.json
```

### 8.3 Analysis Report
`QPU_BENCHMARK_REPORT.md` - Final analysis and recommendations

---

## 9. Quick Start: Minimal Test

Before running the full benchmark, validate with this minimal test:

```python
#!/usr/bin/env python3
"""Minimal QPU validation test."""

from dwave.system import LeapHybridCQMSampler, DWaveSampler
from dimod import ConstrainedQuadraticModel, Binary

# Check solver availability
print("Checking D-Wave access...")

try:
    sampler = DWaveSampler()
    print(f"  ✅ QPU: {sampler.properties['chip_id']}")
    print(f"  Qubits: {sampler.properties['num_qubits']}")
except Exception as e:
    print(f"  ❌ Direct QPU not available: {e}")

try:
    hybrid_sampler = LeapHybridCQMSampler()
    print(f"  ✅ Hybrid CQM Solver available")
    print(f"  Max variables: {hybrid_sampler.properties['maximum_number_of_variables']}")
except Exception as e:
    print(f"  ❌ Hybrid CQM not available: {e}")

print("\n✅ Ready to run QPU benchmarks!")
```

---

## 10. Recommended Execution Order

1. **Day 1:** Run minimal validation test
2. **Day 2-3:** Direct QPU on 25 farms (Phase 2)
3. **Day 4-5:** Hybrid CQM solver on all scales (Phase 3a)
4. **Day 6-7:** Hybrid BQM solver on all scales (Phase 3b)
5. **Day 8-9:** dwave-hybrid workflows (Phase 4)
6. **Day 10-12:** Decomposition + QPU (Phase 5)
7. **Day 13-14:** Analysis and report (Phase 6)

---

## 11. Contact & Support

- D-Wave Support: [support@dwavesys.com](mailto:support@dwavesys.com)
- Leap Forum: [https://support.dwavesys.com/hc/en-us/community/topics](https://support.dwavesys.com/hc/en-us/community/topics)
- Ocean SDK Docs: [https://docs.dwavequantum.com](https://docs.dwavequantum.com)

---

*Document created: November 30, 2025*
*Based on: Comprehensive Scaling Benchmark Results*
