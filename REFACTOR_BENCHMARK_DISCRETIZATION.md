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
    print(f"{'='*80}")

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
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Generate PAIRED sample data
    paired_samples = generate_sample_data(config_values)
    
    farm_results = []
    patch_results = []

    # Iterate over the paired samples
    for farm_sample, patch_sample in paired_samples:
        print(f"\n{'='*80}")
        print(f"PROCESSING SCENARIO FOR {farm_sample['n_units']} UNITS (Sample ID: {farm_sample['sample_id']})")
        print(f"{'='*80}")

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

---

## 5. Unifying the Discretized Formulation to a Simpler Binary Model

### 5.1. Goal

The current "patch" formulation in `solver_runner_PATCH.py` uses two sets of binary variables (`X_p,c` for assignment and `Y_c` for selection), which adds significant complexity to the model. The formulation in `solver_runner_BQUBO.py` is a pure binary model that is much simpler, using only a single set of variables `Y_p,c` to represent a fixed-size (e.g., 1-acre) plantation.

This step refactors the benchmark to use this simpler, pure-BQUBO formulation for the discretized scenario. This provides a clearer comparison between a complex continuous model and a simple binary model.

### 5.2. Formulation Comparison

- **Old Patch Formulation (`solver_runner_PATCH.py`):**
  - **Variables:** `X_p,c` (binary, assigns crop `c` to plot `p`) and `Y_c` (binary, selects crop `c`).
  - **Objective:** `sum(Benefit * Area_p * X_p,c)` - depends on the variable area of each plot.
  - **Complexity:** High. Requires linking constraints between `X` and `Y` variables, making the model larger and more complex than necessary for a pure BQUBO problem.

- **New Binary Formulation (from `solver_runner_BQUBO.py`):**
  - **Variables:** `Y_p,c` (binary, represents a single, fixed-size plantation of crop `c` on farm `p`).
  - **Objective:** `sum(Benefit * Y_p,c)` - each variable has a fixed contribution.
  - **Complexity:** Low. No linking variables are needed. Constraints directly limit the number of plantations per farm.

### 5.3. Action Plan

To implement this, we will create a new solver runner file for the binary formulation and modify the benchmark to use it for the "patch" scenario.

**Step 5.3.1: Create `solver_runner_BINARY.py`**

1.  Make a copy of `solver_runner_BQUBO.py` and name it `solver_runner_BINARY.py`.
2.  This new file will contain the pure binary `create_cqm` function.

**Step 5.3.2: Modify `comprehensive_benchmark.py`**

1.  **Import the new solver:** Change the import from `solver_runner_PATCH` to the new binary solver.

    ```python
    # REMOVE the old import
    # import solver_runner_PATCH as solver_patch

    # ADD the new import
    import solver_runner_BINARY as solver_binary
    ```

2.  **Update `run_patch_scenario` to `run_binary_scenario`:** Rename the function to reflect the new formulation and modify its contents to use the `solver_binary` module.

    ```python
    # Rename this function
    def run_binary_scenario(sample_data: Dict, dwave_token: Optional[str] = None) -> Dict:
        """
        Run Binary Scenario: A pure BQUBO problem where each variable represents
        a 1-acre plantation.
        """
        print(f"\n  BINARY SCENARIO - Sample {sample_data['sample_id']}")
        # The land data now represents the MAX NUMBER of 1-acre plantations
        land_data_continuous = sample_data['data']
        land_data_binary = {farm: int(round(area)) for farm, area in land_data_continuous.items()}
        total_plantations = sum(land_data_binary.values())

        print(f"     {sample_data['n_units']} farms, {total_plantations} total 1-acre plantations")

        # Create problem setup
        foods, food_groups, config = create_food_config(land_data_binary, 'binary')

        # Create CQM using BINARY formulation
        cqm_start = time.time()
        # Note: We now call the CQM builder from the new solver module
        cqm, Y, constraint_metadata = solver_binary.create_cqm(list(land_data_binary.keys()), foods, food_groups, config)
        cqm_time = time.time() - cqm_start

        results = {
            'sample_id': sample_data['sample_id'],
            'scenario_type': 'binary', # Changed from 'patch'
            'n_units': sample_data['n_units'],
            'total_area': sample_data['total_area'],
            'n_foods': len(foods),
            'n_variables': len(cqm.variables),
            'n_constraints': len(cqm.constraints),
            'cqm_time': cqm_time,
            'solvers': {}
        }

        # --- Gurobi (PuLP) Solver --- 
        # This now solves the pure binary problem
        print(f"     Running Gurobi (Binary MILP)...")
        try:
            pulp_start = time.time()
            # Use the pulp solver from the new binary runner
            pulp_model, pulp_results = solver_binary.solve_with_pulp(list(land_data_binary.keys()), foods, food_groups, config)
            pulp_time = time.time() - pulp_start
            
            gurobi_result = {
                'status': pulp_results['status'],
                'objective_value': pulp_results.get('objective_value'),
                'solve_time': pulp_time,
                # ... other result fields
            }
            results['solvers']['gurobi'] = gurobi_result
        except Exception as e:
            # ... error handling

        # --- BQM Conversion and Solvers (DWave BQM, Gurobi QUBO) ---
        # The rest of the function proceeds as before, but now operates on a 
        # much simpler CQM that is already in a BQUBO-friendly format.
        # ... (bqm conversion, dwave_bqm solver, gurobi_qubo solver)

        return results
    ```

3.  **Update the main benchmark loop:** The call to `run_patch_scenario` should be changed to `run_binary_scenario`.

    ```python
    # In run_comprehensive_benchmark loop
    # ...
    # Run Binary Scenario (Discretized)
    try:
        binary_result = run_binary_scenario(patch_sample, dwave_token)
        patch_results.append(binary_result)
    except Exception as e:
        print(f"  ❌ Binary sample {patch_sample['sample_id']} failed: {e}")
    ```

### 5.4. Final Outcome

By implementing this change, the benchmark will compare two distinct but related problems:
1.  **Continuous (`farm_scenario`):** A complex, continuous-variable optimization problem solved with traditional methods (Gurobi LP) and D-Wave's CQM solver.
2.  **Binary (`binary_scenario`):** A simplified, pure-binary (BQUBO) problem solved with a MILP solver (Gurobi), a native QUBO solver (Gurobi `optimods`), and D-Wave's BQM solver.

This provides a much clearer and more valuable benchmark structure.
