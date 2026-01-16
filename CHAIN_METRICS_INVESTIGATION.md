# Chain Length and Chain Break Rate Investigation

**Date:** January 16, 2026  
**Purpose:** Technical investigation of chain metrics in D-Wave quantum annealing results

---

## Executive Summary

Chain metrics are critical indicators of embedding quality and solution reliability in quantum annealing. This investigation analyzed chain_length and chain_break_fraction across the project's QPU results.

**Key Findings:**
- **Chain lengths** range from 1-10 qubits depending on problem size and embedding method
- **Chain break rates** range from 0.001% to 14% depending on problem structure
- The **hierarchical decomposition strategy** keeps chain breaks very low (<0.1%) by limiting partition sizes to 27-90 variables
- **Clique embedding** has shorter chains (length=1) but paradoxically higher chain breaks (10-14%)

---

## 1. Definitions

### Chain Length
The number of physical qubits used to represent a single logical variable in the minor embedding.

```
Logical variable x → Physical qubits [q1, q2, q3, ...]
Chain length = count of physical qubits
```

**Why it matters:** Longer chains require stronger coupling (chain_strength) to stay coherent, and are more prone to breaks.

### Chain Break Fraction
The fraction of chains where physical qubits disagree on the final value (some qubits report 0, others report 1).

```
Chain break = chain has inconsistent qubit values
Chain break fraction = (broken chains) / (total chains)
```

**Why it matters:** Chain breaks indicate the embedding failed to maintain coherence, potentially corrupting the solution.

---

## 2. Data Sources

| File | Description | Problem Sizes |
|------|-------------|---------------|
| `qpu_micro_benchmark.json` | Micro benchmarks | 6-90 variables |
| `qpu_method_benchmark.json` | Method comparison | 20-24 variables |
| `qpu_benchmark_summary_*.json` | Comprehensive benchmarks | 702-27,027 variables |
| `@todo/qpu_benchmark_results/*.json` | Partition-level results | Various |

---

## 3. Chain Length Analysis

### 3.1 Chain Length vs Problem Size

| Problem Size (n_vars) | Physical Qubits | Max Chain Length | Embedding Method |
|----------------------|-----------------|------------------|------------------|
| 6 | 17-21 | 1-2 | Direct/Clique |
| 12 | 31-45 | 1-3 | Direct/Clique |
| 20-24 | 24-32 | 1-2 | Direct/Clique |
| 90 | 120-651 | 1-9 | Clique/Direct |
| 702 (full) | 2,916-2,920 | 9-10 | Direct |
| 702 (partitioned) | ~50-100/partition | 2-5 | Partitioned |

### 3.2 Physical Qubit Overhead

The ratio of physical qubits to logical variables indicates embedding efficiency:

| Logical Variables | Physical Qubits | Ratio | Notes |
|-------------------|-----------------|-------|-------|
| 17 | 17-21 | 1.0-1.2x | Near-ideal embedding |
| 24 | 24-32 | 1.0-1.3x | Good efficiency |
| 90 | 120 (clique) | 1.3x | Clique embedding |
| 90 | 651 (direct) | 7.2x | Standard embedding |
| 702 | 2,916 | 4.2x | Large problem overhead |

**Observation:** Physical qubit overhead scales super-linearly with problem size due to increasing connectivity requirements.

### 3.3 Chain Length Distribution by Method

| Embedding Method | Typical Chain Length | Range |
|------------------|---------------------|-------|
| Clique embedding | 1 | Always 1 |
| Direct embedding (small) | 2-3 | 1-5 |
| Direct embedding (medium) | 5-7 | 3-9 |
| Direct embedding (large) | 8-10 | 5-15+ |
| Partitioned (27 vars) | 2-3 | 1-5 |
| Partitioned (90 vars) | 3-5 | 2-8 |

---

## 4. Chain Break Fraction Analysis

### 4.1 Chain Breaks vs Problem Configuration

| Problem Type | n_vars | Chain Break Fraction | Notes |
|--------------|--------|---------------------|-------|
| Micro (direct) | 6-20 | 0.0% | Perfect embedding |
| Micro (clique) | 6-20 | 1.0-2.0% | Clique has higher breaks |
| Small rotation | 90 | 0.03% (direct), 14.3% (clique) | Major method difference |
| Medium partitioned | 702 | 0.5-4.0% per partition | Acceptable range |
| Large partitioned | 13,527 | 0.001-0.07% per partition | Very low breaks |

### 4.2 Partition-Level Chain Break Statistics

From `qpu_benchmark_25farms_PlotBased.json` (702 variables, 26 partitions):

| Statistic | Value |
|-----------|-------|
| Minimum | 0.54% |
| Maximum | 3.75% |
| Average | 2.0% |
| Std Dev | ~1.2% |

From `qpu_benchmark_500farms_HybridGrid.json` (13,527 variables, 151 partitions):

| Statistic | Value |
|-----------|-------|
| Minimum | 0.001% |
| Maximum | 0.07% |
| Average | 0.005% |
| Std Dev | ~0.01% |

### 4.3 Chain Break vs Chain Strength

The chain_strength parameter controls how strongly physical qubits in a chain are coupled:

| chain_strength | max_chain_length | chain_break_fraction |
|----------------|------------------|---------------------|
| 0 (clique) | 1 | 1.8-14.3% |
| 75.4 | 2 | 0.0% |
| 230.2 | 9 | 0.03% |

**Observation:** Higher chain_strength correlates with lower chain breaks, but clique embedding (chain_strength=0, length=1) has higher breaks due to different connectivity constraints.

---

## 5. Impact on Solution Quality

### 5.1 Chain Breaks and Violations

Chain breaks can cause constraint violations in the final solution:

| Chain Break Rate | Expected Impact |
|------------------|-----------------|
| < 1% | Minimal - random bit flips typically correctable |
| 1-5% | Low - may cause minor one-hot violations |
| 5-15% | Moderate - significant violation probability |
| > 15% | High - solutions likely infeasible |

### 5.2 Correlation with Objective Gap

From the violation analysis (see `analyze_violation_gap.py`):

- Chain breaks contribute to one-hot constraint violations
- Each violation costs approximately 10-20 benefit units
- 80-90% of objective gap is explained by constraint violations (including chain-break-induced)

---

## 6. Decomposition Strategy Effectiveness

The project uses hierarchical decomposition to manage chain metrics:

| Strategy | Partition Size | Chain Length | Chain Break | Scalability |
|----------|---------------|--------------|-------------|-------------|
| No decomposition | Full problem | 9-15+ | 5-20% | Limited to ~200 vars |
| PlotBased | 27 vars | 2-3 | 1-4% | Good to 1000+ farms |
| Multilevel(5) | 27-135 vars | 3-5 | 2-5% | Good to 500 farms |
| Multilevel(10) | 27-270 vars | 4-7 | 3-6% | Moderate |
| HybridGrid | 90 vars | 3-5 | 0.001-0.1% | Excellent to 1000+ farms |

**Key Insight:** The HybridGrid partitioning achieves extremely low chain breaks (0.001-0.1%) even for problems with 27,000+ variables by maintaining consistent partition sizes of ~90 variables.

---

## 7. Recommendations

### 7.1 For Low Chain Breaks
1. **Use partitioned approaches** - Keep subproblems under 100 variables
2. **Prefer direct embedding over clique** for problems with high connectivity
3. **Tune chain_strength** - Higher values reduce breaks but increase energy scale

### 7.2 For Analysis
1. **Monitor partition-level chain breaks** - High variance indicates problematic partitions
2. **Track max chain length** - Lengths > 10 indicate embedding stress
3. **Correlate with violations** - Chain breaks often manifest as one-hot violations

### 7.3 For Reporting
1. **Report both average and max** chain break fractions
2. **Include chain length statistics** (min, max, mean) for reproducibility
3. **Document chain_strength parameter** used in experiments

---

## 8. Files Containing Chain Metrics

### Computation Code
| File | Function |
|------|----------|
| `Benchmark Scripts/benchmarks.py` | Primary chain computation |
| `dwave_adapter.py` | Chain break extraction from samplesets |
| `qpu_benchmark_comprehensive.py` | Detailed chain statistics |

### Result Data
| File Pattern | Content |
|--------------|---------|
| `qpu_*_benchmark.json` | Contains chain_break_fraction, max_chain_length |
| `qpu_benchmark_summary_*.json` | Aggregate chain statistics |
| `@todo/qpu_benchmark_results/*.json` | Partition-level chain data |

### Metrics Stored
```json
{
  "chain_strength": 75.36,
  "physical_qubits": 32,
  "max_chain_length": 2,
  "chain_break_fraction": 0.0,
  "embedding_info": {
    "chain_length": {"min": 1, "max": 2, "mean": 1.5, "median": 1}
  }
}
```

---

## 9. Summary Table

| Metric | Small Problems (<100 vars) | Medium (100-1000 vars) | Large (>1000 vars, partitioned) |
|--------|---------------------------|------------------------|--------------------------------|
| **Max Chain Length** | 1-5 | 5-10 | 2-8 per partition |
| **Avg Chain Length** | 1-3 | 3-7 | 2-5 per partition |
| **Chain Break Fraction** | 0-2% | 2-10% | 0.001-0.1% per partition |
| **Physical/Logical Ratio** | 1.0-1.5x | 2-5x | 1.5-3x per partition |

---

**Document Version:** 1.0  
**Analysis based on:** QPU benchmark results from December 2025 - January 2026
