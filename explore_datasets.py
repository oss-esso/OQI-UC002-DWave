#!/usr/bin/env python3
"""
Dataset Explorer for QPU vs Gurobi Benchmark Analysis
=====================================================

This script loads all benchmark datasets and provides:
1. Comprehensive structure analysis
2. Data summary statistics
3. Aggregation strategies for plotting

Data Sources (from plots_for_report.md):
- qpu_hier_repaired.json: Hierarchical QPU with post-processing repair
- gurobi_baseline_60s.json: Classical Gurobi solver (60s timeout)
- gurobi_timeout_test_300s.json: Extended Gurobi runs (300s timeout)
- qpu_benchmark_summary_*.json: Batch orchestration summaries
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Any
import glob

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).parent
PRIMARY_FILES = {
    "qpu_hier": "qpu_hier_repaired.json",
    "gurobi_60s": "gurobi_baseline_60s.json",
    "gurobi_300s": "gurobi_timeout_test_300s.json",
    "qpu_hybrid": "qpu_hybrid_27food.json",
    "qpu_native": "qpu_native_results.json",
}

# ============================================================================
# Data Loading
# ============================================================================


def load_json_safe(filepath: Path) -> dict | None:
    """Load JSON file with error handling."""
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON decode error in {filepath.name}: {e}")
        return None


def load_all_datasets() -> dict[str, dict]:
    """Load all primary datasets."""
    datasets = {}
    print("\n" + "=" * 70)
    print("📂 LOADING DATASETS")
    print("=" * 70)

    for key, filename in PRIMARY_FILES.items():
        filepath = DATA_DIR / filename
        data = load_json_safe(filepath)
        if data:
            datasets[key] = data
            n_runs = len(data.get("runs", []))
            print(f"  ✓ {key:15} → {filename:35} ({n_runs} runs)")
        else:
            print(f"  ✗ {key:15} → {filename:35} (not found)")

    # Load benchmark summaries
    summary_files = list(DATA_DIR.glob("qpu_benchmark_summary_*.json"))
    if summary_files:
        # Use the most recent one
        latest = max(summary_files, key=lambda p: p.stat().st_mtime)
        data = load_json_safe(latest)
        if data:
            datasets["qpu_summary"] = data
            print(f"  ✓ {'qpu_summary':15} → {latest.name}")

    return datasets


# ============================================================================
# Schema Analysis
# ============================================================================


def analyze_schema(data: dict, name: str) -> dict:
    """Analyze the schema of a dataset."""
    schema = {
        "name": name,
        "top_level_keys": list(data.keys()),
        "schema_version": data.get("schema_version", "unknown"),
        "generated_at": data.get("generated_at", "unknown"),
    }

    runs = data.get("runs", [])
    if runs:
        schema["n_runs"] = len(runs)
        # Sample first run for structure
        sample = runs[0]
        schema["run_keys"] = list(sample.keys())

        # Extract problem configs
        configs = []
        for run in runs:
            config = (
                run.get("n_farms", 0),
                run.get("n_foods", 0),
                run.get("n_periods", 0),
            )
            configs.append(config)
        schema["unique_configs"] = sorted(set(configs))

    return schema


def print_schema_summary(schemas: dict[str, dict]) -> None:
    """Print formatted schema summary."""
    print("\n" + "=" * 70)
    print("📋 DATASET SCHEMAS")
    print("=" * 70)

    for name, schema in schemas.items():
        print(f"\n{'─' * 50}")
        print(f"📊 {name.upper()}")
        print(f"{'─' * 50}")
        print(f"  Schema Version: {schema['schema_version']}")
        print(f"  Generated At:   {schema['generated_at']}")
        print(f"  Number of Runs: {schema.get('n_runs', 0)}")

        if "run_keys" in schema:
            print(f"  Run Keys ({len(schema['run_keys'])}):")
            # Group keys by category
            timing_keys = [k for k in schema["run_keys"] if "time" in k.lower()]
            objective_keys = [k for k in schema["run_keys"] if "objective" in k.lower()]
            violation_keys = [k for k in schema["run_keys"] if "violation" in k.lower()]
            config_keys = [
                k for k in schema["run_keys"] if k.startswith("n_") or k == "scenario_name"
            ]
            other_keys = [
                k
                for k in schema["run_keys"]
                if k not in timing_keys + objective_keys + violation_keys + config_keys
            ]

            print(f"    Config:     {config_keys}")
            print(f"    Timing:     {timing_keys}")
            print(f"    Objectives: {objective_keys}")
            print(f"    Violations: {violation_keys}")
            print(f"    Other:      {other_keys[:10]}{'...' if len(other_keys) > 10 else ''}")

        if "unique_configs" in schema:
            print(f"  Problem Configurations (farms × foods × periods):")
            for config in schema["unique_configs"]:
                n_vars = config[0] * config[1] * config[2]
                print(f"    {config[0]:4} × {config[1]:2} × {config[2]} = {n_vars:6} variables")


# ============================================================================
# Data Extraction & Aggregation
# ============================================================================


def extract_run_metrics(run: dict) -> dict:
    """Extract key metrics from a single run."""
    # Handle nested timing dict
    timing = run.get("timing", {})
    violations = run.get("constraint_violations", {})
    
    return {
        # Problem configuration
        "scenario": run.get("scenario_name", "unknown"),
        "n_farms": run.get("n_farms", 0),
        "n_foods": run.get("n_foods", 0),
        "n_periods": run.get("n_periods", 3),
        "n_vars": run.get("n_vars", 0),
        "formulation": "6-family" if run.get("n_foods", 0) <= 6 else "27-food",
        # Results
        "status": run.get("status", "unknown"),
        "feasible": run.get("feasible", False),
        "objective": run.get("objective_miqp"),  # Keep None to detect missing
        "mip_gap": run.get("mip_gap"),
        # Timing (all in seconds) - from nested timing dict
        "total_time": timing.get("total_wall_time", 0),
        "solve_time": timing.get("solve_time", 0),
        "qpu_time": timing.get("qpu_access_time", 0),
        "qpu_sampling": timing.get("qpu_sampling_time", 0),
        "refinement_time": timing.get("refinement_time", 0),
        "timeout": run.get("timeout_s", 60),
        # Violations - from nested constraint_violations dict
        "one_hot_violations": violations.get("one_hot_violations", 0),
        "rotation_violations": violations.get("rotation_violations", 0),
        "total_violations": violations.get("total_violations", 0),
        # Decomposition info
        "n_clusters": run.get("decomposition", {}).get("n_clusters", 1),
        # Solver info
        "mode": run.get("mode", "unknown"),
        "sampler": run.get("sampler", "unknown"),
        "backend": run.get("backend", "unknown"),
    }


def build_comparison_table(datasets: dict) -> list[dict]:
    """Build unified comparison table across all datasets."""
    rows = []

    for dataset_name, data in datasets.items():
        runs = data.get("runs", [])
        for run in runs:
            metrics = extract_run_metrics(run)
            metrics["source"] = dataset_name
            rows.append(metrics)

    return rows


def aggregate_by_scenario(rows: list[dict]) -> dict[str, dict]:
    """Aggregate metrics by scenario for cross-method comparison."""
    by_scenario = defaultdict(dict)

    for row in rows:
        scenario = row["scenario"]
        source = row["source"]

        by_scenario[scenario]["n_farms"] = row["n_farms"]
        by_scenario[scenario]["n_foods"] = row["n_foods"]
        by_scenario[scenario]["n_vars"] = row["n_vars"]
        by_scenario[scenario]["formulation"] = row["formulation"]

        # Store source-specific metrics
        by_scenario[scenario][f"{source}_objective"] = row["objective"]
        by_scenario[scenario][f"{source}_time"] = row["total_time"]
        by_scenario[scenario][f"{source}_feasible"] = row["feasible"]
        by_scenario[scenario][f"{source}_qpu_time"] = row.get("qpu_time", 0)
        by_scenario[scenario][f"{source}_violations"] = (
            row["one_hot_violations"] + row["rotation_violations"]
        )

    return dict(by_scenario)


def print_comparison_summary(aggregated: dict) -> None:
    """Print formatted comparison summary."""
    print("\n" + "=" * 70)
    print("📊 CROSS-METHOD COMPARISON")
    print("=" * 70)

    # Sort by n_vars
    sorted_scenarios = sorted(aggregated.items(), key=lambda x: x[1].get("n_vars", 0))

    print(f"\n{'Scenario':<30} {'Vars':>7} {'Form':>8} | {'Gurobi':>10} {'QPU':>10} {'Gap%':>7}")
    print("-" * 85)

    for scenario, data in sorted_scenarios:
        n_vars = data.get("n_vars", 0)
        formulation = data.get("formulation", "?")[:8]

        gurobi_obj = data.get("gurobi_60s_objective", data.get("gurobi_300s_objective"))
        qpu_obj = data.get("qpu_hier_objective", data.get("qpu_hybrid_objective"))

        gurobi_str = f"{gurobi_obj:.1f}" if gurobi_obj else "—"
        qpu_str = f"{qpu_obj:.1f}" if qpu_obj else "—"

        if gurobi_obj and qpu_obj and gurobi_obj != 0:
            gap = abs(qpu_obj - gurobi_obj) / abs(gurobi_obj) * 100
            gap_str = f"{gap:.1f}%"
        else:
            gap_str = "—"

        print(f"{scenario:<30} {n_vars:>7} {formulation:>8} | {gurobi_str:>10} {qpu_str:>10} {gap_str:>7}")


# ============================================================================
# Aggregation Strategies for Plotting
# ============================================================================


def print_aggregation_strategies(rows: list[dict]) -> None:
    """Print recommended aggregation strategies for plotting."""
    print("\n" + "=" * 70)
    print("📈 RECOMMENDED AGGREGATION STRATEGIES FOR PLOTTING")
    print("=" * 70)

    strategies = """
┌─────────────────────────────────────────────────────────────────────┐
│ STRATEGY 1: BY PROBLEM SIZE (n_vars)                                │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose: Scaling analysis - how does performance change with size?  │
│                                                                     │
│ X-axis: n_vars (log scale)                                          │
│ Y-axis: Metric of interest                                          │
│ Grouping: By solver (Gurobi vs QPU)                                 │
│                                                                     │
│ Metrics to plot:                                                    │
│   • Solve time → Shows computational scaling                        │
│   • Objective value → Shows solution quality scaling                │
│   • Speedup ratio → Shows where QPU beats classical                 │
│   • Gap % → Shows optimality degradation with scale                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ STRATEGY 2: BY FORMULATION (6-family vs 27-food)                    │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose: Compare formulation complexity effects                     │
│                                                                     │
│ Split data into two formulation groups, then compare:               │
│   • Time efficiency per formulation                                 │
│   • QPU advantage region per formulation                            │
│   • Violation rates per formulation                                 │
│                                                                     │
│ Key insight: 6-family embeds better, 27-food needs decomposition    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ STRATEGY 3: TIMING DECOMPOSITION                                    │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose: Understand where time is spent                             │
│                                                                     │
│ Stacked bar chart components:                                       │
│   • qpu_access_time (pure quantum)                                  │
│   • total_time - qpu_access_time (classical overhead)               │
│                                                                     │
│ Key insight: Classical coordination dominates for small problems    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ STRATEGY 4: VIOLATION IMPACT ANALYSIS                               │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose: Correlate violations with objective gap                    │
│                                                                     │
│ Scatter plot:                                                       │
│   X-axis: Total violations (one_hot + rotation)                     │
│   Y-axis: Gap % vs Gurobi                                           │
│   Color: By formulation                                             │
│   Size: By n_vars                                                   │
│                                                                     │
│ Key insight: ~80-90% of gap explained by violations                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ STRATEGY 5: QUANTUM ADVANTAGE HEATMAP                               │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose: Identify advantage zones in problem space                  │
│                                                                     │
│ 2D heatmap:                                                         │
│   X-axis: n_farms (binned)                                          │
│   Y-axis: n_foods (6 or 27)                                         │
│   Color: Speedup factor or gap %                                    │
│                                                                     │
│ Key insight: QPU advantage emerges for specific (farms, foods) combos│
└─────────────────────────────────────────────────────────────────────┘
"""
    print(strategies)


def print_data_structure_for_plotting(rows: list[dict]) -> None:
    """Print the recommended DataFrame structure for plotting."""
    print("\n" + "=" * 70)
    print("🗂️  RECOMMENDED DATAFRAME STRUCTURE")
    print("=" * 70)

    structure = """
For unified analysis, create a merged DataFrame with this structure:

┌──────────────┬──────────────┬──────────────────────────────────────────┐
│ Column       │ Type         │ Description                              │
├──────────────┼──────────────┼──────────────────────────────────────────┤
│ scenario     │ str          │ Unique problem identifier (join key)     │
│ n_farms      │ int          │ Number of farms in problem               │
│ n_foods      │ int          │ Number of foods (6 or 27)                │
│ n_periods    │ int          │ Planning periods (usually 3)             │
│ n_vars       │ int          │ Total binary variables                   │
│ formulation  │ str          │ "6-family" or "27-food"                  │
├──────────────┼──────────────┼──────────────────────────────────────────┤
│ gurobi_obj   │ float        │ Gurobi objective value                   │
│ gurobi_time  │ float        │ Gurobi solve time (s)                    │
│ gurobi_gap   │ float        │ Gurobi MIP gap if timeout                │
│ gurobi_status│ str          │ "optimal" / "timeout" / "infeasible"     │
├──────────────┼──────────────┼──────────────────────────────────────────┤
│ qpu_obj      │ float        │ QPU objective value (post-repair)        │
│ qpu_time     │ float        │ Total QPU wall time                      │
│ qpu_pure     │ float        │ Pure QPU access time                     │
│ qpu_feasible │ bool         │ Whether solution is feasible             │
│ violations   │ int          │ Total constraint violations              │
├──────────────┼──────────────┼──────────────────────────────────────────┤
│ gap_pct      │ float        │ |qpu_obj - gurobi_obj| / |gurobi_obj| %  │
│ speedup      │ float        │ gurobi_time / qpu_time                   │
│ advantage    │ bool         │ speedup > 1 AND gap_pct < 20             │
└──────────────┴──────────────┴──────────────────────────────────────────┘

SAMPLE CODE TO BUILD THIS:

```python
import pandas as pd

# Load datasets
with open('qpu_hier_repaired.json') as f:
    qpu_data = json.load(f)
with open('gurobi_baseline_60s.json') as f:
    gurobi_data = json.load(f)

# Build DataFrames
qpu_df = pd.DataFrame([extract_run_metrics(r) for r in qpu_data['runs']])
gurobi_df = pd.DataFrame([extract_run_metrics(r) for r in gurobi_data['runs']])

# Merge on scenario
merged = qpu_df.merge(
    gurobi_df, 
    on=['scenario', 'n_farms', 'n_foods', 'n_vars'],
    suffixes=('_qpu', '_gurobi')
)

# Compute derived metrics
merged['gap_pct'] = abs(merged['objective_qpu'] - merged['objective_gurobi']) / abs(merged['objective_gurobi']) * 100
merged['speedup'] = merged['total_time_gurobi'] / merged['total_time_qpu']
merged['advantage'] = (merged['speedup'] > 1) & (merged['gap_pct'] < 20)
```
"""
    print(structure)


# ============================================================================
# Sample Data Preview
# ============================================================================


def print_sample_data(datasets: dict) -> None:
    """Print sample data from each dataset."""
    print("\n" + "=" * 70)
    print("🔍 SAMPLE DATA PREVIEW")
    print("=" * 70)

    for name, data in datasets.items():
        runs = data.get("runs", [])
        if not runs:
            continue

        print(f"\n{'─' * 50}")
        print(f"📊 {name.upper()} - First 3 runs")
        print(f"{'─' * 50}")

        for i, run in enumerate(runs[:3]):
            metrics = extract_run_metrics(run)
            obj_str = f"{metrics['objective']:.2f}" if metrics['objective'] is not None else "N/A"
            print(f"\n  Run {i + 1}: {metrics['scenario']}")
            print(f"    Config:    {metrics['n_farms']} farms × {metrics['n_foods']} foods = {metrics['n_vars']} vars")
            print(f"    Objective: {obj_str}")
            print(f"    Time:      {metrics['total_time']:.3f}s (QPU: {metrics['qpu_time']:.4f}s)")
            print(f"    Feasible:  {metrics['feasible']}")
            print(f"    Violations: {metrics['one_hot_violations']} one-hot, {metrics['rotation_violations']} rotation")


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Main entry point."""
    print("\n" + "═" * 70)
    print("  QPU vs GUROBI BENCHMARK DATA EXPLORER")
    print("  Quantum Optimization for Crop Rotation Planning")
    print("═" * 70)

    # Load all datasets
    datasets = load_all_datasets()

    if not datasets:
        print("\n❌ No datasets found! Check that JSON files exist.")
        return

    # Analyze schemas
    schemas = {name: analyze_schema(data, name) for name, data in datasets.items()}
    print_schema_summary(schemas)

    # Build unified comparison table
    rows = build_comparison_table(datasets)
    print(f"\n📊 Total runs across all datasets: {len(rows)}")

    # Aggregate by scenario
    aggregated = aggregate_by_scenario(rows)
    print_comparison_summary(aggregated)

    # Print sample data
    print_sample_data(datasets)

    # Print aggregation strategies
    print_aggregation_strategies(rows)
    print_data_structure_for_plotting(rows)

    # Summary statistics
    print("\n" + "=" * 70)
    print("📈 SUMMARY STATISTICS")
    print("=" * 70)

    sources = defaultdict(list)
    for row in rows:
        sources[row["source"]].append(row)

    for source, source_rows in sources.items():
        n_feasible = sum(1 for r in source_rows if r["feasible"])
        avg_time = sum(r["total_time"] for r in source_rows) / len(source_rows)
        var_range = (
            min(r["n_vars"] for r in source_rows),
            max(r["n_vars"] for r in source_rows),
        )

        print(f"\n  {source.upper()}:")
        print(f"    Runs:        {len(source_rows)}")
        print(f"    Feasible:    {n_feasible}/{len(source_rows)} ({100*n_feasible/len(source_rows):.0f}%)")
        print(f"    Avg Time:    {avg_time:.3f}s")
        print(f"    Var Range:   {var_range[0]} - {var_range[1]}")


if __name__ == "__main__":
    main()
