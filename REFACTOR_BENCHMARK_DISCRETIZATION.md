# Refactoring `comprehensive_benchmark.py` for Direct Comparability

## 1. Goal

The objective is to modify `comprehensive_benchmark.py` to ensure that for each benchmark size, the continuous ("farm") and discretized ("patch") scenarios are directly comparable. This is achieved by adopting the data generation strategy from `Grid_Refinement.py`, where each discretized scenario is created with the **exact same total land area** as its continuous counterpart.

This change enhances the benchmark's integrity by comparing the performance of solvers on problems that are true discretized approximations of each other, rather than on independently generated samples.

**Key Change:** Instead of generating independent sets of farms and patches, we will generate a set of "continuous" farm scenarios first, and then for each one, generate a corresponding "discretized" patch scenario with the same total land area.

---

## 2. Analysis of Required Changes

The core logic resides in `comprehensive_benchmark.py`. The main functions to modify are:
- `generate_sample_data()`: This function must be updated to create paired continuous and discretized samples.
- `run_comprehensive_benchmark()`: The main loop needs to be adjusted to process these paired samples.
- `run_patch_scenario()`: May require a minor adjustment to handle the new data structure, although the solver calls remain the same.

The solvers themselves (`solver_runner.py` and `solver_runner_PATCH.py`) do not need to be changed.

---

## 3. Step-by-Step Implementation Guide

### Step 3.1: Modify `generate_sample_data`

The current function generates two independent lists. The new function will generate one list of tuples, where each tuple contains a paired `(farm_sample, patch_sample)`.

**Action:** Replace the existing `generate_sample_data` function in `comprehensive_benchmark.py` with the following implementation.

```python
def generate_sample_data(config_values: List[int], seed_offset: int = 0) -> List[Tuple[Dict, Dict]]:
    """
    Generate paired samples of continuous (farm) and discretized (patch) scenarios.

    For each configuration value, this function first generates a continuous "farm" sample.
    It then generates a corresponding "patch" sample that is scaled to have the
    exact same total land area, ensuring the two are directly comparable.

    Args:
        config_values: List of configuration values (number of units to generate).
        seed_offset: Offset for random seed to ensure variety.

    Returns:
        A list of tuples, where each tuple is `(farm_sample, patch_sample)`.
    """
    print(f"\n{'='*80}")
    print(f"GENERATING PAIRED SAMPLES FOR CONFIGS: {config_values}")
    print(f"{ '='*80}")

    paired_samples = []

    for i, n_units in enumerate(config_values):
        print(f"\n--- Generating pair for {n_units} units ---")
        # 1. Generate the continuous farm sample
        seed = 42 + seed_offset + i * 100
        farms = generate_farms_large(n_farms=n_units, seed=seed)
        total_area = sum(farms.values())
        
        farm_sample = {
            'sample_id': i,
            'type': 'farm',
            'data': farms,
            'total_area': total_area,
            'n_units': n_units,
            'seed': seed
        }
        print(f"  ✓ Continuous: {farm_sample['n_units']} farms, total area {total_area:.2f} ha")

        # 2. Generate the discretized patch sample and scale it
        patch_seed = seed + 50  # Use a different seed for patch structure
        patches_unscaled = generate_patches_small(n_farms=n_units, seed=patch_seed)
        
        # Scale patches to match the continuous version's total area
        current_patch_total = sum(patches_unscaled.values())
        scale_factor = total_area / current_patch_total if current_patch_total > 0 else 0
        patches_scaled = {k: v * scale_factor for k, v in patches_unscaled.items()}

        patch_sample = {
            'sample_id': i,
            'type': 'patch',
            'data': patches_scaled,
            'total_area': total_area,  # Same as the farm sample
            'n_units': n_units,
            'seed': patch_seed
        }
        print(f"  ✓ Discretized: {patch_sample['n_units']} patches, scaled to total area {sum(patches_scaled.values()):.2f} ha")

        paired_samples.append((farm_sample, patch_sample))

    print(f"\nGenerated {len(paired_samples)} paired samples.")
    return paired_samples
```

### Step 3.2: Update `run_comprehensive_benchmark`

This function needs to be updated to iterate over the new paired samples instead of two separate lists.

**Action:** Modify the `run_comprehensive_benchmark` function as follows.

```python
def run_comprehensive_benchmark(config_values: List[int], dwave_token: Optional[str] = None) -> Dict:
    """
    Run the comprehensive benchmark for the given configuration values.
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE BENCHMARK - CONFIGS: {config_values}")
    print(f"{ '='*80}")
    
    start_time = time.time()
    
    # Generate PAIRED sample data
    paired_samples = generate_sample_data(config_values)
    
    farm_results = []
    patch_results = []

    # Iterate over the paired samples
    for farm_sample, patch_sample in paired_samples:
        print(f"\n{'='*80}")
        print(f"PROCESSING SCENARIO FOR {farm_sample['n_units']} UNITS (Sample ID: {farm_sample['sample_id']})")
        print(f"{ '='*80}")

        # Run Farm Scenario (Continuous)
        try:
            farm_result = run_farm_scenario(farm_sample, dwave_token)
            farm_results.append(farm_result)
        except Exception as e:
            print(f"  ❌ Farm sample {farm_sample['sample_id']} failed: {e}")

        # Run Patch Scenario (Discretized)
        try:
            patch_result = run_patch_scenario(patch_sample, dwave_token)
            patch_results.append(patch_result)
        except Exception as e:
            print(f"  ❌ Patch sample {patch_sample['sample_id']} failed: {e}")

    total_time = time.time() - start_time
    
    # Compile comprehensive results (the rest of the function remains the same)
    benchmark_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'config_values': config_values,
            'total_runtime': total_time,
            'dwave_enabled': dwave_token is not None,
            'scenarios': ['farm', 'patch'],
            'solvers': {
                'farm': ['gurobi', 'dwave_cqm'],
                'patch': ['gurobi', 'dwave_cqm', 'gurobi_qubo', 'dwave_bqm']
            }
        },
        'farm_results': farm_results,
        'patch_results': patch_results,
        'summary': {
            'farm_samples_completed': len(farm_results),
            'patch_samples_completed': len(patch_results),
            'total_solver_runs': sum(len(r['solvers']) for r in farm_results + patch_results)
        }
    }
    
    return benchmark_results
```

### Step 3.3: Objective Value Normalization (Optional but Recommended)

The analysis in `Grid_Refinement.py` notes that the objective value from the "patch" formulation might need to be normalized to be comparable to the "farm" (continuous) formulation. While the benchmark records multiple metrics (solve time, success rate), having comparable objective values is crucial for quality analysis.

The continuous formulation in `solver_runner.py` produces a normalized objective (a benefit score per hectare), whereas the patch formulation in `solver_runner_PATCH.py` produces a total benefit score.

**Action:** To ensure objectives are comparable, you can add a normalization step inside `run_patch_scenario`. However, since the primary goal of the benchmark is to compare solver *performance* (time, scalability), it is also valid to leave the objectives as they are and perform normalization during post-processing (e.g., in the plotting scripts).

For completeness, here is how you would add it to the benchmark script itself. In `run_patch_scenario`, after getting a result from a solver, you would normalize it:

```python
# Example inside run_patch_scenario, after a solver returns a result
# (e.g., after the `gurobi_result` is created)

if gurobi_result.get('objective_value') is not None:
    raw_objective = gurobi_result['objective_value']
    total_area = sample_data['total_area']
    if total_area > 0:
        normalized_objective = raw_objective / total_area
        gurobi_result['objective_value_normalized'] = normalized_objective
        gurobi_result['note'] = 'objective_value is raw; normalized value is also provided.'
```
This is an optional enhancement. The primary change is the paired data generation.

---

## 4. Summary of Changes

By implementing the modifications in steps 3.1 and 3.2, `comprehensive_benchmark.py` will:
1.  Generate pairs of continuous and discretized problems that are directly comparable.
2.  Run all solvers on these paired problems.
3.  Store the results in the same format as before, allowing existing analysis and plotting scripts to function with minimal changes.

This refactoring significantly improves the quality of the benchmark by ensuring a fair and direct comparison between the continuous and discretized modeling approaches.
