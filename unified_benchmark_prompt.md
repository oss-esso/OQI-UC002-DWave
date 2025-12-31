# Prompt for Unified Benchmark Script

**To:** Coding Agent

**From:** Project Lead

**Subject:** Mandate for a Unified, Honest Benchmark Script

## 1. Background (why this exists)

Previous benchmarking efforts compared simplified quantum models to equally simplified classical "ground truths." Those results are invalid. All prior benchmark scripts and reports are deprecated. Only formulations.tex (Formulation 1: True Ground Truth MIQP) and RUTHLESS_ANALYSIS_AND_BENCHMARK_PLAN.md define the math and the standard of rigor. This prompt replaces every earlier mandate.

## 2. Quick checklist (do this before running)

- Enforce equal, fixed area for every plot/farm in every scenario (override any heterogenous areas on load).
- Use the True Ground Truth MIQP objective from formulations.tex to score every solution, even after any aggregation/refinement steps.
- Emit a single JSON file with full metadata, timings, and constraint checks for every run.
- Prefer QPU where available; use simulated annealing (SA) as the default fallback for testing and when QPU access is limited. Always record which sampler was used.
- Avoid all deprecated baselines: no aggregating Gurobi ground truth, no simplified objectives.

## 3. Core requirements for unified_benchmark.py

You will build one script that unifies all valid logic. It must be clear, maintainable, and scientifically defensible.

### 3.1 General setup

- Formulation: Implement strictly the MIQP in formulations.tex (Formulation 1). No surrogate objectives.
- Equal area: On scenario load, set every plot/farm area to a common constant (choose 1.0 unless a scenario-specific constant is provided). Recompute any area-derived coefficients accordingly.
- Scenario coverage: Support all rotation_* scenarios defined in src/scenarios.py. Include both native 6-family and full 27-food scenarios (e.g., rotation_micro_25, rotation_small_50, rotation_medium_100, rotation_large_200, rotation_250farms_27foods, rotation_350farms_27foods, rotation_500farms_27foods; extend if more are available).
- CLI: Provide --mode, --scenario, --output-json, --timeout, --sampler (qpu|sa), and any decomposition knobs per mode. Default timeout: 600 s unless overridden.
- Output: Single JSON capturing all runs (see schema below). Always recompute the MIQP objective on the final 27-food crop-level assignment (even if the solve used aggregation or simplified synergies).

### 3.2 Benchmark modes (must support all)

1. gurobi-true-ground-truth
   - Purpose: definitive classical baseline on the full MIQP, no aggregation.
   - Source logic: adapt @todo/test_gurobi_timeout.py (full 27-food MIQP, soft one-hot, rotation/spatial synergies). Do not use any aggregating ground-truth code from hierarchical_statistical_test.py.
   - Requirements: enforce equal areas; honor global timeout; capture MIP gap, status, and incumbent objective; export full timing breakdown.

2. qpu-native-6-family
   - Purpose: native small problems (6 families) without aggregation.
   - Source logic: adapt clique or spatial-temporal decomposition from @todo/statistical_comparison_test.py.
   - Post-processing: refine each family decision into specific crops; recompute MIQP objective on the refined 27-food solution; include constraint violations.
   - Sampler: QPU if available; SA permitted for testing; record sampler, num_reads, chain strength, embedding stats.

3. qpu-hierarchical-aggregated
   - Purpose: 27-food problems solved via aggregate -> solve -> refine pipeline.
   - Source logic: @todo/hierarchical_quantum_solver.py (Level 1 aggregate 27 -> 6, Level 2 solve on QPU/SA, Level 3 refine back to 27 crops).
   - Evaluation: always recompute MIQP objective on the refined 27-food solution; log any heuristic refinement gaps and constraint violations.

4. qpu-hybrid-27-food
   - Purpose: 27-food variables with simplified synergy matrix (6-family template) and spatial decomposition.
   - Source logic: @todo/hybrid_formulation.py using solve_hybrid_qpu with 27-food variable space and 6-family-based R.
   - Requirements: equal areas enforced; recompute MIQP objective post-solve; record decomposition sizes, sampler, and timing.

### 3.3 Scenarios and data loading

- Accept any rotation_* scenario in src/scenarios.py.
- Provide a default suite mixing 6-family and 27-food scales (examples above). Document how to extend the list.
- Validate inputs: fail fast if a scenario lacks required data (e.g., Excel-derived crop lists) and report a clear error.

## 4. Output, logging, and monitoring

### 4.1 JSON schema (one file for all runs)

- Top-level: { "schema_version": "1.0", "runs": [ ... ] }.
- Each run entry MUST include:
  - mode, scenario_name, n_farms, n_foods, n_vars.
  - sampler (qpu|sa), backend (name/version), num_reads (if applicable).
  - timeout_s (wall), status (optimal, feasible, timeout, error), mip_gap (if available).
  - objective_miqp (recomputed true MIQP), objective_model (raw model objective if different), constraint_violations (count and brief breakdown), feasible (bool).
  - timing object with at least: total_wall_time, model_build_time, solve_time, postprocess_time, qpu_access_time, embedding_time (use 0 or null if not applicable), refinement_time.
  - decomposition metadata where relevant: cluster sizes, iterations, boundary sync stats.
  - seed (if set), timestamp_utc, git_commit (if available), hostname.

### 4.2 Logging and monitoring

- Stream concise progress logs: scenario load, area normalization applied, model build, submit, solve, refine, objective recompute, JSON write.
- Capture sampler-specific diagnostics: embedding size/chain lengths for QPU; energy histograms or best-so-far trace for SA if available.
- On any failure/timeout, emit a structured error entry in JSON with diagnostic message and partial timings.

## 5. Runtime policy

- Timeouts: default 600 s wall per run unless overridden. Apply to both classical and quantum/SA solves. Record actual wall time.
- Equal area enforcement: mandatory in all modes and scenarios before model build. Document the chosen constant in logs and JSON.
- Objective scoring: always recompute the true MIQP objective on the final 27-food assignment. Never report only an aggregated or simplified objective.
- Sampler fallback: if QPU unavailable, run SA with identical problem setup; label clearly. Never silently drop a run.
- Determinism: allow a --seed to make SA/clique sampling reproducible when possible; log when seeding is applied.

## 6. Implementation guardrails

- Do not reuse any aggregating "ground truth" code from hierarchical_statistical_test.py or other deprecated scripts.
- Preserve scientific honesty: no hidden simplifications, no unreported preprocessing, no omitted constraints.
- Keep code modular: separate scenario loading, area normalization, model build, solve, refinement, scoring, and reporting.
- Add lightweight in-code comments only where logic is non-obvious (decomposition boundaries, refinement rules, objective recomputation).

## 7. Execution notes for the agent (act like a senior physicist)

- Think in terms of conservation laws: every variable introduced must flow through build -> solve -> refinement -> scoring without loss or silent approximation.
- Measure everything: wall time, QPU access, embedding, refinement, and constraint checks. Prefer fewer, high-quality runs over noisy batches.
- When using SA for testing, match the QPU problem encoding exactly; only the sampler changes.
- Validate outputs by recomputing the MIQP objective and constraint residuals before writing JSON.

## 8. Deliverable

- A single unified_benchmark.py that honors all requirements above, produces the specified JSON, and can be run unattended with clear logs.
