# Formulation Audit — Which Problem Does Each Solver Solve?

**Purpose**: Map every data file to its problem formulation, solver model type, time limits,
and identify which classical runs need to be re-done without time limits.

---

## 1. Formulation Definitions (from `content_report.tex`)

### Variant A — Single-Period Binary Crop Allocation (Studies 1 & 2)

- **Decision variables**: `Y_{p,c} ∈ {0,1}` (binary patch-crop assignment) + `U_c ∈ {0,1}` (global crop usage indicator)
- **Objective**: **Linear** — `max Σ B_c × L_p × Y_{p,c} / A_total` (area-weighted benefit sum)
- **Constraints**: one-hot per patch, Y-U linking, food group diversity (min/max crops per group), area bounds
- **Gurobi model**: BIP (Binary Integer Program) — all binary variables, linear objective
- **Variable count**: `n_patches × 27 + 27` (patches × crops + U indicators)
- **Solver behaviour**: Gurobi solves in < 1.2s for all scales up to 1000 patches. Trivial for classical.

### Variant B — Multi-Period Rotation (Study 3)

- **Decision variables**: `Y_{f,c,t} ∈ {0,1}` (farm-crop-period assignment), 6 families × 3 periods or 27 crops × 3 periods
- **Objective**: **Quadratic (MIQP)** — 5 terms:
  1. Linear base benefit
  2. **Quadratic** temporal rotation synergies (`γ_rot × R[c1,c2] × Y_{f,c1,t-1} × Y_{f,c2,t}`)
  3. **Quadratic** spatial neighbor interactions (`γ_spat × S[c1,c2] × Y_{f1,c1,t} × Y_{f2,c2,t}`)
  4. **Quadratic** soft one-hot penalty (`-λ × (Σ_c Y_{f,c,t} - 1)²`)
  5. Linear diversity bonus
- **Constraints**: `1 ≤ Σ_c Y_{f,c,t} ≤ 2` hard bounds (NOT `= 1`); exact one-hot is only a soft penalty in the objective
- **Gurobi model**: MIQP (Mixed-Integer Quadratic Program) — binary vars + quadratic objective
- **Variable count**: `n_farms × n_foods × 3` (6-family: `18F`, 27-food: `81F`)
- **Solver behaviour**: Gurobi struggles with quadratic terms. Timeouts common for > 50 farms.

---

## 2. Data File → Formulation Mapping

### Study 1: Hybrid Solver Benchmarking

| Data File | Generator Script | Formulation | Gurobi Model | Time Limit | Scenarios |
|-----------|-----------------|-------------|-------------|------------|-----------|
| `comprehensive_benchmark_configs_dwave_20251130_212742.json` | `Benchmark Scripts/comprehensive_benchmark.py` | **Variant A (BP)** | BIP (PuLP) | None (PuLP default) | 10–1000 patches × 27 crops |

**Solvers in file**: Gurobi (PuLP), D-Wave CQM, D-Wave BQM, Gurobi QUBO
**Gurobi time for 1000 patches**: 1.15s (trivial — no timeout needed)
**Variables**: `n_patches × 27 + 27` = 297 to 27,027

**⟹ No re-run needed**: Gurobi solves Variant A optimally in < 2s for all scales.

---

### Study 2: Pure QPU Decomposition

| Data File(s) | Generator Script | Formulation | Gurobi Model | Time Limit | Scenarios |
|--------------|-----------------|-------------|-------------|------------|-----------|
| 75× `qpu_benchmark_*.json` (in `@todo/qpu_benchmark_results/` and migrated `data/benchmark_results/study2/`) | `@todo/qpu_benchmark.py` | **Variant A (BP)** | BIP (Gurobi direct) | 120s | 10–1000 patches × 27 crops |

**Gurobi ground truth**: All solved to optimality in < 5s. The 120s timeout was never reached.
**QPU methods**: Direct QPU, PlotBased, Multilevel(5/10), Louvain, Spectral(10), HybridGrid, Coordinated, CQM-First
**Variables**: 297 to 27,027

**⟹ No re-run needed**: Gurobi ground truth is provably optimal. QPU data is from real D-Wave hardware (cannot be re-run).

---

### Study 3: Quantum Rotation (Hierarchical Decomposition)

| Data File | Generator Script | Formulation | Gurobi Model | Time Limit | Scenarios |
|-----------|-----------------|-------------|-------------|------------|-----------|
| `gurobi_baseline_60s.json` | `unified_benchmark.py --mode gurobi-true-ground-truth --timeout 60` | **Variant B** | **MIQP** | **60s** | 13 rotation scenarios (5–200 farms, 6/27 foods) |
| `qpu_hier_repaired.json` | `unified_benchmark.py --mode qpu-hierarchical-aggregated` | **Variant B** | N/A (QPU) | 600s wall-clock | Same 13 scenarios |
| 13× `gurobi_timeout_test_*.json` | `@todo/test_gurobi_timeout.py` / `test_gurobi_timeout_WORKING.py` | **Variant B** | **MIQP** | **100s** or **300s** | 6–20 rotation scenarios |

**Key findings**:

1. **`gurobi_baseline_60s.json`**: 60s timeout. 9/13 scenarios hit timeout (only 4 solved to feasibility under 60s). This was the "classical baseline" used to claim QPU advantage.

2. **`gurobi_timeout_test_*.json`**: Multiple runs with different timeouts:
   - `_20251214_174332.json`: 6 scenarios, **no timeout** (0.07–0.01s → all optimal)
   - `_20251214_180751.json`: 6 scenarios, **no timeout** (all optimal)
   - `_20251214_184357.json`: 6 scenarios, **300s timeout** (all timeout)
   - `_20251215_144413.json`: 10 scenarios, **100s timeout** (1 optimal, 9 timeout)
   - `_20251215_152938.json`: 20 scenarios, **100s timeout** (1 optimal, rest timeout)
   - `_20251222_*` through `_20251224_*`: 20 scenarios each, **100s timeout**, improved solver params
   - **Latest** (`_20251224_103144.json`): 20 scenarios, 100s timeout, aggressive Gurobi params (`Presolve=2, Cuts=2, ImproveStartTime=30`)

3. **`qpu_hier_repaired.json`**: 13 runs using real D-Wave QPU. 5→200 farms, 6/27 foods. Solutions repaired and scored against MIQP objective.

**⟹ CRITICAL: Gurobi re-runs needed for Study 3**:
- Current "ground truth" was run with **60s timeout** → 9/13 scenarios suboptimal
- Timeout tests used **100s** → most still timing out
- For rigorous benchmark: **Gurobi must be run with NO time limit** (or very generous, e.g., 3600s)

---

### Decomposition Scaling Benchmark (Appendix)

| Data File | Generator Script | Formulation | Gurobi Model | Time Limit | Scenarios |
|-----------|-----------------|-------------|-------------|------------|-----------|
| `decomposition_scaling_results.json` | `Benchmarks/decomposition_scaling/benchmark_decomposition_scaling.py` | **Both A & B** | N/A (partition timing only) | N/A | 5–1000 farms |
| `solver_comparison_results.json` | `Benchmarks/decomposition_scaling/benchmark_solvers_comparison.py` | **Both A & B** | BIP (A) / MIQP (B) | **300s** | 5–10000 farms |

**Variant A** in solver_comparison: Gurobi solves all sizes in < 5s (up to 10,000 farms). No re-run needed.

**Variant B** in solver_comparison:
- 27 crops × 3 periods (NOT 6 families)
- Gurobi: 300s timeout. Hits timeout at 200, 1000, 2000 farms (status=9). Skipped at 5000+.
- PT-ICM: Only run up to 200 farms.

**⟹ Re-run needed for Variant B**: Gurobi_full hits 300s timeout for ≥ 200 farms.

---

## 3. Formulation Inconsistencies Discovered

### 3a. Rotation Matrix Source

| Script | Rotation Matrix | Source |
|--------|----------------|--------|
| `unified_benchmark.py` | Synthetic (`np.random.seed(42)`, `frustration=0.7`, `negative_strength=-0.8`) | Built inline in `core.py` |
| `@todo/qpu_benchmark.py` | Synthetic (same seed + params) | Built inline |
| `@todo/test_gurobi_timeout.py` | Synthetic (same seed + params) | Built inline |
| `benchmark_solvers_comparison.py` | ~~Real CSV~~ → **Synthetic** (same seed + params) | **Fixed 2026-03-27** |

**Status ✅ RESOLVED**: `benchmark_solvers_comparison.py` now uses `build_rotation_matrix(n=27, frustration_ratio=0.7, negative_strength=-0.8, seed=42)` matching all other scripts.

### 3b. Spatial Interaction Model

| Script | Spatial Model | Effective coeff |
|--------|--------------|----------------|
| `unified_benchmark/gurobi_solver.py` | k-NN (k=4) on integer grid | ~~0.03·R (no /A_total)~~ → **0.1·0.3·R/A_total** |
| `unified_benchmark/miqp_scorer.py` | k-NN (k=4) on integer grid | ~~0.1·R/A_total (missing 0.3)~~ → **0.1·0.3·R/A_total** |
| `@todo/qpu_benchmark.py` | k-NN (k=4) on integer grid | 0.1·0.3·R/A_total ✓ |
| `benchmark_solvers_comparison.py` | ~~No spatial term~~ → **k-NN (k=4) on integer grid** | **0.1·0.3·R/A_total** |

**Formula per `content_report.tex`**: `γ_spat · S[c1,c2]/A_total` where `S = 0.3·R`, `γ_spat = 0.1`
→ effective coefficient = **`0.1 × 0.3 × R[c1,c2] / A_total = 0.03 × R / A_total`**

**Status ✅ RESOLVED** (2026-03-27):
- `gurobi_solver.py`: added `/total_area`, now uses `spatial_gamma * 0.3 * synergy / total_area`
- `miqp_scorer.py`: added `0.3` factor, now uses `spatial_gamma * 0.3 * (synergy / total_area)`
- `benchmark_solvers_comparison.py`: spatial term added to `build_variant_b_bqm()`, `_build_sub_bqm_B_from_partition()`, and `_extract_cqm_objective_B()`

### 3c. Variable Structure in Variant B

| Script | Foods | Variables per farm |
|--------|-------|--------------------|
| `unified_benchmark.py` (native 6-family) | 6 families | 18 |
| `unified_benchmark.py` (hierarchical) | 6 families (solve) → 27 crops (score) | 18 (solve), scored on 27 |
| `unified_benchmark.py` (hybrid 27-food) | 27 crops | 81 |
| `@todo/qpu_benchmark.py` (rotation) | 6 families | 18 |
| `benchmark_solvers_comparison.py` | **27 crops** | 81 |

**Issue**: The decomposition benchmark uses 27 individual crops for Variant B, while Study 3 uses 6 food families. These are different problem sizes.

### 3e. Strict One-Hot Constraint Missing in Variant B

**All Variant B scripts** use `1 ≤ Σ_c Y_{f,c,t} ≤ 2` as hard constraints, **not** `= 1`.
The exact assignment of one crop per farm per period is enforced only via a soft penalty term `−λ × (Σ_c Y_{f,c,t} − 1)²` in the objective, with `λ = 3.0`.

This is intentional — explicit comment in `qpu_benchmark.py` line 564: `"Only soft upper bound (allow 0-2 crops per period)"`.

**Consequence**: Gurobi considers a 2-crop assignment *feasible*. It will choose 2 crops if the marginal benefit of the second crop exceeds the `λ = 3.0` penalty. Solutions with 2-crop assignments are reported as `one_hot_violations` by `miqp_scorer.py` but are *not* infeasible from the solver's perspective.

The semantically correct formulation would use a hard equality `Σ_c Y_{f,c,t} = 1 ∀ f, t`.



| Parameter | `unified_benchmark.py` | `test_gurobi_timeout.py` | `benchmark_solvers_comparison.py` |
|-----------|----------------------|-------------------------|----------------------------------|
| `rotation_gamma` | 0.2 | 0.2 | ~~0.5~~ → **0.2** |
| `spatial_gamma` | 0.1 | 0.1 | ~~N/A~~ → **0.1** |
| `one_hot_penalty` | 3.0 | 3.0 | 3.0 |
| `diversity_bonus` | 0.15 | 0.15 | N/A in BQM (BQM full delegated to `solve_gurobi_ground_truth`) |
| `k_neighbors` | 4 | 4 | **4** |
| `frustration_ratio` | 0.7 | 0.7 | **0.7** |
| `MIPGap` | 0.01 | 0.01 | N/A |

**Status ✅ RESOLVED** (2026-03-27): `benchmark_solvers_comparison.py` now reads `ROTATION_GAMMA` and `SPATIAL_GAMMA` from `MIQP_PARAMS` in `unified_benchmark/core.py`.

**Additional changes to `benchmark_solvers_comparison.py`**:
- `solve_gurobi_full_B()` now delegates to `solve_gurobi_ground_truth()` (true MIQP, includes diversity bonus)
- `solve_gurobi_decomposed_B()` now reports MIQP objective via `_extract_cqm_objective_B()` (previously reported raw QUBO energy)
- Temporal normalization fixed: was `γ_rot * A_f * R / A_tot²` (double), now `γ_rot * A_f/A_tot * R`

---

## 4. Classical Re-Run Plan

### What needs re-running

| Priority | Script | Formulation | Current Timeout | Issue |
|----------|--------|-------------|----------------|-------|
| **HIGH** | `unified_benchmark.py --mode gurobi-true-ground-truth` | Variant B (MIQP) | 60s | 9/13 scenarios suboptimal — this is the main "ground truth" for Study 3 |
| **HIGH** | `@todo/test_gurobi_timeout.py` | Variant B (MIQP) | 100s | Most scenarios still timing out at 100s |
| **MEDIUM** | `benchmark_solvers_comparison.py` (Variant B) | Variant B (MIQP, 27 crops) | 300s | Hits timeout for ≥ 200 farms |
| **NONE** | `comprehensive_benchmark.py` | Variant A (BIP) | None | Already optimal in < 2s |
| **NONE** | `qpu_benchmark.py` (Gurobi GT) | Variant A (BIP) | 120s | Already optimal in < 5s |

### What CANNOT be re-run (quantum data — keep as-is)

| Data File | Source | Notes |
|-----------|--------|-------|
| `qpu_hier_repaired.json` | D-Wave QPU | Real QPU results. No longer have access. |
| 75× `qpu_benchmark_*.json` | D-Wave QPU | Real QPU decomposition results. |
| `comprehensive_benchmark_*.json` (D-Wave entries) | D-Wave CQM/BQM | Real Leap hybrid results. |

### Recommended re-run parameters

For a rigorous benchmark, Gurobi should be run with:
- **Timeout**: `0` (no limit) or `3600s` (1 hour safety cap)
- **MIPGap**: `0.0` (prove optimality) or `0.001` (0.1% for very large instances)
- **MIPFocus**: `0` (balanced, let Gurobi decide)
- **Presolve**: `-1` (automatic)
- **Threads**: `0` (all available)
- Record: `objective`, `MIP gap`, `runtime`, `status`, `node count`, `bound`

### Scenarios to cover

**Study 3 (Variant B, 6 families)**:
| Scenario | Farms | Foods | Variables | Current best Gurobi status |
|----------|-------|-------|-----------|----------------------------|
| `rotation_micro_25` | 5 | 6 | 90 | optimal (0.08s) |
| `rotation_small_50` | 10 | 6 | 180 | feasible/timeout |
| `rotation_15farms_6foods` | 15 | 6 | 270 | feasible |
| `rotation_medium_100` | 20 | 6 | 360 | feasible/timeout |
| `rotation_25farms_6foods` | 25 | 6 | 450 | timeout |
| `rotation_large_200` | 40 | 6 | 720 | timeout |
| `rotation_50farms_6foods` | 50 | 6 | 900 | timeout |
| `rotation_75farms_6foods` | 75 | 6 | 1,350 | timeout |
| `rotation_100farms_6foods` | 100 | 6 | 1,800 | timeout |
| `rotation_150farms_6foods` | 150 | 6 | 2,700 | timeout |

**Study 3 (Variant B, 27 crops)**:
| Scenario | Farms | Foods | Variables | Current best Gurobi status |
|----------|-------|-------|-----------|----------------------------|
| `rotation_25farms_27foods` | 25 | 27 | 2,025 | timeout |
| `rotation_50farms_27foods` | 50 | 27 | 4,050 | timeout |
| `rotation_100farms_27foods` | 100 | 27 | 8,100 | timeout |
| `rotation_200farms_27foods` | 200 | 27 | 16,200 | timeout |
| `rotation_250farms_27foods` | 250 | 27 | 20,250 | timeout |
| `rotation_350farms_27foods` | 350 | 27 | 28,350 | timeout |
| `rotation_500farms_27foods` | 500 | 27 | 40,500 | timeout |
| `rotation_1000farms_27foods` | 1000 | 27 | 81,000 | timeout |

**Decomposition benchmark (Variant B, 27 crops × 3 periods)**:
| n_farms | Variables | Current Gurobi status |
|---------|-----------|----------------------|
| 5 | 405 | optimal (0.28s) |
| 10 | 810 | optimal (4.5s) |
| 25 | 2,025 | optimal (20.7s) |
| 50 | 4,050 | optimal (59.8s) |
| 100 | 8,100 | optimal (1.9s) ← suspicious |
| 200 | 16,200 | timeout (300s) |
| 500 | 40,500 | optimal (160.9s) ← suspicious? |
| 1000 | 81,000 | timeout (300s) |
| 2000 | 162,000 | timeout (300s) |
| 5000 | 405,000 | SKIPPED |
| 10000 | 810,000 | SKIPPED |

---

## 5. Summary

| Study | Formulation | Gurobi Model | Classical Results Adequate? | Action |
|-------|-------------|-------------|---------------------------|--------|
| 1 (Hybrid) | Variant A (BP, linear) | BIP | **YES** — all optimal in < 2s | None |
| 2 (QPU Decomp) | Variant A (BP, linear) | BIP | **YES** — all optimal in < 5s | None |
| 3 (Rotation) | Variant B (MIQP, quadratic) | MIQP | **NO** — 9/13 hit 60s timeout | **Re-run with no timeout** |
| App (Decomp scaling) | Both A & B | BIP / MIQP | A: YES. B: **NO** (300s timeout) | **Re-run Variant B with no timeout** |
