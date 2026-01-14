# Project Codebase Integrity Check - Verification Report

**Generated:** January 14, 2026  
**Report Reference:** `@todo/report/content_report.tex`

---

## 1. Core Problem Formulations

### 1.1. Formulation A (Binary Crop Allocation)

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `@todo/cqm_partition_embedding_benchmark.py` (lines 470-600), `@todo/hierarchical_quantum_solver.py` |

**Key Implementation Details:**

- **Function Name:** `build_binary_cqm()`, `build_cqm_formulation_a()`
- **Decision Variables:**
  - `Y[farm, food]` — binary assignment variable
  - `U[food]` — unique food indicator
- **Constraints Match:**
  - ✅ At most one crop per farm
  - ✅ U-Y linking constraints
  - ✅ Food group diversity
  - ✅ Max plots per crop

**Notes:** The description mentions "27 crops", which is parameterized via scenario configuration. Scenarios like `full_family` use 27 crops while `rotation_*` scenarios use 6 families.

---

### 1.2. Formulation B (Hierarchical Multi-Period Rotation)

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `@todo/hierarchical_quantum_solver.py`, `@todo/cqm_partition_embedding_benchmark.py`, `@todo/native_family_bqm.py` |

**Key Implementation Details:**

- **Function Name:** `build_rotation_cqm()`
- **Decision Variables:** 3D variables `Y[(f, c, t)]` across farms, families, and periods
- **Objective Components (all present):**
  - ✅ **Benefit** (linear)
  - ✅ **Temporal Synergy** (quadratic)
  - ✅ **Spatial Synergy** (quadratic)
  - ✅ **One-Hot Penalty** (quadratic)
  - ✅ **Diversity Bonus** (linear)
- **Rotation Matrix R:**
  - ✅ Monoculture penalty: `R[i,i] = negative_strength * 1.5`
  - ✅ Frustration ratio: 70-88% (default)
- **Parameters:**
  - ✅ 6 crop families: `Fruits, Grains, Legumes, Leafy_Vegetables, Root_Vegetables, Proteins`
  - ✅ 3 periods (T=3)
  - ✅ k=4 nearest spatial neighbors

**Notes:** None — implementation closely matches the documented formulation.

---

## 2. Decomposition Strategies

All 8 decomposition methods are implemented in `@todo/cqm_partition_embedding_benchmark.py` with a registry at lines 2305-2345.

### 2.1. Direct QPU Embedding

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `@todo/cqm_partition_embedding_benchmark.py` |

**Description:** Implemented as `direct_qpu_submission()`. Converts CQM→BQM, finds embedding with minorminer, uses `DWaveSampler` to sample on real QPU. Includes timeout protection and chain strength auto-tuning.

---

### 2.2. Plot-Based Decomposition

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `@todo/cqm_partition_embedding_benchmark.py` (line 273) |

**Description:** Implemented as `plot_based_partition()`. Creates one partition per farm (Y variables for that farm) plus a separate partition for all U variables.

---

### 2.3. Multilevel Partitioning

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `@todo/cqm_partition_embedding_benchmark.py`, `@todo/advanced_decomposition_strategies.py` (lines 26-100) |

**Description:** Implemented as `multilevel_partition()`. Groups k farms together into clusters. Variants: `Multilevel(5)`, `Multilevel(10)`, `Multilevel(20)`. Advanced version includes graph coarsening using maximum weight matching.

---

### 2.4. Louvain Community Detection

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `@todo/cqm_partition_embedding_benchmark.py`, `@todo/decomposition_framework.py` (lines 81-130) |

**Description:** Implemented as `louvain_partition()`. Uses `community_louvain.best_partition()` with resolution parameter tuning. Builds interaction graph from variable connections, then partitions using modularity maximization.

---

### 2.5. CQM-First Decomposition

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `@todo/cqm_partition_embedding_benchmark.py` |

**Description:** Implemented as `cqm_first_partition()`. Partitions at CQM level BEFORE BQM conversion. Extracts sub-CQMs with relevant constraints per partition using `extract_sub_cqm_for_partition()`. Solves U partition (master) first, then Y partitions with fixed U values.

---

### 2.6. Coordinated Master-Subproblem

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `@todo/cqm_partition_embedding_benchmark.py`, `@todo/hierarchical_quantum_solver.py` |

**Description:** Two-level decomposition: Master = all U variables (food selection), Sub = Y variables per farm. (1) Solves master problem for food group diversity, (2) Fixes U values, (3) Solves each farm subproblem respecting U assignments.

---

### 2.7. Spectral Clustering

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `@todo/cqm_partition_embedding_benchmark.py` (line 302) |

**Description:** Implemented as `spectral_partition()`. Builds adjacency matrix from variable interactions, uses `sklearn.cluster.SpectralClustering` with precomputed affinity. Uses eigenvectors of graph Laplacian for partitioning. Falls back to Multilevel if sklearn unavailable.

---

### 2.8. HybridGrid Decomposition

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `@todo/cqm_partition_embedding_benchmark.py` |

**Description:** Implemented as `hybridgrid_partition()`. Creates 2D grid partitioning over both farms AND foods. Variants: `HybridGrid(3,9)`, `HybridGrid(5,9)`, `HybridGrid(10,9)`, `HybridGrid(5,13)`, `HybridGrid(3,13)`.

---

## 3. Data and Preprocessing

### 3.1. Crop Database

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `Inputs/Food-Environment-Data-Updated.xlsx` |

**Details:**
- **27 crops confirmed** — Excel file contains exactly 27 rows
- **5 food groups confirmed:**
  - Animal-source foods (5): Beef, Chicken, Egg, Lamb, Pork
  - Fruits (9): Apple, Avocado, Banana, Durian, Guava, Mango, Orange, Papaya, Watermelon
  - Pulses, nuts, and seeds (4): Chickpeas, Peanuts, Tempeh, Tofu
  - Starchy staples (2): Corn, Potato
  - Vegetables (7): Cabbage, Cucumber, Eggplant, Long bean, Pumpkin, Spinach, Tomatoes
- **Columns (7):** `Food_Name`, `food_group`, `nutritional_value`, `nutrient_density`, `sustainability`, `environmental_impact`, `affordability`
- **Data source:** GAIN (Global Alliance for Improved Nutrition) for Bangladesh and Indonesia

**Supporting files:**
- `Inputs/Environmental_Impact_Mean.csv` — Mean environmental impact per food group
- `Inputs/rotation_synergy_matrix.csv` — 6×6 rotation synergy matrix

---

### 3.2. Crop Family Aggregation

| Attribute | Result |
|-----------|--------|
| **Status** | ⚠️ PARTIALLY FOUND |
| **Confidence** | **Medium** |
| **Location(s)** | `@todo/cqm_partition_embedding_benchmark.py` (lines 2077-2145), `src/synergy_optimizer.py` |

**Details:**
- Aggregation to 6 families is defined **inline** within rotation scenario loaders
- No standalone `aggregate_crops` function found
- **6 families:** Fruits, Grains, Legumes, Leafy_Vegetables, Root_Vegetables, Proteins

**Discrepancies:**
- Family names vary slightly between files (`Leafy_Vegetables` vs `Vegetables`)
- The 6 families don't directly map to the 5 food groups — designed for QUBO formulation

---

### 3.3. Farm Data and Spatial Layout

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `Utils/scenario_loader.py` (farm sizes), `@todo/hierarchical_quantum_solver.py` (spatial graph) |

**Farm Size Distribution:**
- `generate_farm_sizes()` uses skewed distribution based on Global South agricultural survey data
- Size classes: <1ha (45%), 1-2ha (20%), 2-5ha (15%), 5-10ha (8%), 10-20ha (5%), >20ha (7%)

**Spatial Layout:**
- `build_spatial_neighbors()` function
- Farms arranged on 2D grid (side = √n_farms)
- Each farm connected to k=4 nearest neighbors (default)
- Uses Euclidean distance for neighbor selection

---

## 4. Solver and Hardware Configuration

### 4.1. D-Wave QPU Configuration

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `@todo/cqm_partition_embedding_benchmark.py`, `@todo/hierarchical_quantum_solver.py` |

**Components Found:**

| Parameter | Status | Values Found |
|-----------|--------|--------------|
| Advantage quantum annealer | ✅ | Explicit references |
| Pegasus topology | ✅ | P16 specified |
| `DWaveCliqueSampler` | ✅ | `@todo/cqm_partition_embedding_benchmark.py` |
| `MinorMiner` | ✅ | `@todo/cqm_partition_embedding_benchmark.py` |
| `EmbeddingComposite` | ✅ | `@todo/cqm_partition_embedding_benchmark.py` |
| `annealing_time` | ✅ | Default: 20 µs |
| `num_reads` | ✅ | 50-1000 (varies by scenario) |
| `chain_strength` | ✅ | Auto-calculated (None) |

---

### 4.2. Gurobi Classical Baseline

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **Medium** |
| **Location(s)** | `src/classical_optimizer.py`, `Utils/gurobi_baseline.py` |

**Configuration Found:**

| Parameter | Status | Report Value | Code Value |
|-----------|--------|--------------|------------|
| Version | ⚠️ | 12.0.1 | 12.0.3 (logs) |
| TimeLimit | ✅ | 300s | 300-600s (configurable) |
| MIPGap | ✅ | 1% | 0.01 |
| MIPFocus | ✅ | 1 | 1 |
| Threads | ✅ | - | 0 (all cores) |
| Presolve | ✅ | - | 2 (aggressive) |

**Discrepancy:** Report mentions Gurobi 12.0.1, but logs show 12.0.3 was actually used.

---

## 5. BQM Conversion

| Attribute | Result |
|-----------|--------|
| **Status** | ✅ FOUND |
| **Confidence** | **High** |
| **Location(s)** | `Utils/cqm_analysis.py`, `@todo/native_family_bqm.py`, `@todo/fast_bqm_wrapper.pyx` |

### Conversion Mechanisms:

**A. D-Wave `cqm_to_bqm()` Function:**
- Location: `Utils/cqm_analysis.py`
- Uses `dimod.cqm_to_bqm()` with configurable `lagrange_multiplier`
- Test multipliers: `[10, 100, 1000, 10000, 50000]`

**B. Custom BQM Builders:**
- Location: `@todo/native_family_bqm.py`, `@todo/fast_bqm_wrapper.pyx`
- Explicit h/J construction with full penalty encoding

### Linear Biases (h) and Quadratic Couplings (J):

**From `@todo/native_family_bqm.py`:**
```python
# Linear biases
linear[var_name] = benefit * area_frac
linear[var_name] -= 2.0 * one_hot_penalty  # One-hot contribution

# Quadratic couplings
quadratic[key] += rotation_gamma * synergy * area_frac  # Temporal
quadratic[key] += spatial_gamma * spatial_synergy       # Spatial
quadratic[key] += 2.0 * one_hot_penalty                 # One-hot
```

### Penalty Weights (λ):

| Constraint | Default Value | Source |
|------------|---------------|--------|
| One-hot penalty | 3.0 | `@todo/native_family_bqm.py` |
| Rotation penalty | 10.0 | `@todo/hierarchical_quantum_solver.py` |
| cqm_to_bqm multiplier | 10000× max_obj_coeff | `Utils/cqm_analysis.py` |

**Note:** Report describes adaptive penalty tuning (20% increments until 95% feasibility), but code uses fixed defaults. Adaptive logic is described in comments but not fully automated.

---

## Summary

| Category | Items | Found | Partial | Not Found |
|----------|-------|-------|---------|-----------|
| **1. Formulations** | 2 | 2 | 0 | 0 |
| **2. Decomposition** | 8 | 8 | 0 | 0 |
| **3. Data/Preprocessing** | 3 | 2 | 1 | 0 |
| **4. Solver Config** | 2 | 2 | 0 | 0 |
| **5. BQM Conversion** | 1 | 1 | 0 | 0 |
| **TOTAL** | **16** | **15** | **1** | **0** |

### Key Discrepancies Identified (Now Fixed):

1. **Crop Family Aggregation (3.2):** ✅ **FIXED** — Report now clarifies aggregation is implemented inline in scenario loader.

2. **Gurobi Version (4.2):** ✅ **FIXED** — Report updated to 12.0.3 to match actual version used.

3. **Timeout Values (4.2):** ✅ **FIXED** — Report now specifies different timeouts for different experiment types (600s for main benchmarks, 100s for standard tests, 60s for QPU validation).

4. **Penalty Tuning (5):** ✅ **FIXED** — Report now accurately states that fixed penalty weights are used ($\lambda_{\text{oh}} = 3.0$, $\lambda_r = 10.0$) rather than iterative adaptive tuning.

---

**Verification Status: ✅ PASSED**

All critical artifacts are present and documented. Previously identified discrepancies have been corrected in the report to accurately reflect the implementation.
