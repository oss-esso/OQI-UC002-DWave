#!/usr/bin/env python3
"""
Unified Benchmark Script for Crop Rotation Optimization

This is the SINGLE authoritative benchmark script that:
1. Implements the TRUE MIQP from formulations.tex
2. Enforces equal areas across all farms
3. Recomputes MIQP objective on all solutions
4. Outputs structured JSON with full metadata

Modes:
- gurobi-true-ground-truth: Full MIQP, no aggregation (classical baseline)
- qpu-native-6-family: Native 6-family BQM (small problems)
- qpu-hierarchical-aggregated: 27→6 aggregate, solve, refine
- qpu-hybrid-27-food: 27-food variables with 6-family synergy template

Usage:
    python unified_benchmark.py --mode gurobi-true-ground-truth --scenario rotation_micro_25
    python unified_benchmark.py --mode qpu-native-6-family --sampler sa --scenario rotation_micro_25
    python unified_benchmark.py --mode qpu-hierarchical-aggregated --sampler sa --scenario rotation_25farms_27foods
    python unified_benchmark.py --mode qpu-hybrid-27-food --sampler sa --scenario rotation_25farms_27foods
    python unified_benchmark.py --run-all --sampler sa --output-json results.json

Author: OQI-UC002-DWave Project
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add project root and unified_benchmark to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "unified_benchmark"))

from unified_benchmark.core import (
    SCHEMA_VERSION,
    MODES,
    DEFAULT_TIMEOUT,
    MIQP_PARAMS,
    BenchmarkLogger,
    save_benchmark_results,
)
from unified_benchmark.scenarios import (
    load_scenario,
    get_available_scenarios,
    DEFAULT_AREA_CONSTANT,
)
from unified_benchmark.quantum_solvers import solve, HAS_DIMOD, HAS_QPU


# Default scenarios for each mode
# Comprehensive scaling from 5 to 200 farms, both 6-food and 27-food
DEFAULT_SCENARIOS = {
    "gurobi-true-ground-truth": [
        # 6-food scenarios
        "rotation_micro_25",           # 5 farms × 6 foods
        "rotation_small_50",           # 10 farms × 6 foods
        "rotation_15farms_6foods",     # 15 farms × 6 foods
        "rotation_medium_100",         # 20 farms × 6 foods
        "rotation_25farms_6foods",     # 25 farms × 6 foods
        "rotation_50farms_6foods",     # 50 farms × 6 foods
        "rotation_75farms_6foods",     # 75 farms × 6 foods
        "rotation_100farms_6foods",    # 100 farms × 6 foods
        "rotation_large_200",          # 200 farms × 6 foods
        # 27-food scenarios
        "rotation_25farms_27foods",    # 25 farms × 27 foods
        "rotation_50farms_27foods",    # 50 farms × 27 foods
        "rotation_100farms_27foods",   # 100 farms × 27 foods
        "rotation_200farms_27foods",   # 200 farms × 27 foods
    ],
    "qpu-native-6-family": [
        # Native BQM for 6-food problems (best for small-medium)
        "rotation_micro_25",           # 5 farms × 6 foods = 90 vars
        "rotation_small_50",           # 10 farms × 6 foods = 180 vars
        "rotation_15farms_6foods",     # 15 farms × 6 foods = 270 vars
        "rotation_medium_100",         # 20 farms × 6 foods = 360 vars
        "rotation_25farms_6foods",     # 25 farms × 6 foods = 450 vars
        "rotation_50farms_6foods",     # 50 farms × 6 foods = 900 vars
        "rotation_75farms_6foods",     # 75 farms × 6 foods = 1350 vars
        "rotation_100farms_6foods",    # 100 farms × 6 foods = 1800 vars
        "rotation_large_200",          # 200 farms × 6 foods = 3600 vars
    ],
    "qpu-hierarchical-aggregated": [
        # Hierarchical decomposition with 27→6 aggregation
        # Good for medium-large 27-food problems
        "rotation_micro_25",           # 5 farms (will aggregate to 6)
        "rotation_small_50",           # 10 farms × 6 foods
        "rotation_15farms_6foods",     # 15 farms × 6 foods
        "rotation_medium_100",         # 20 farms × 6 foods
        "rotation_25farms_6foods",     # 25 farms × 6 foods
        "rotation_50farms_6foods",     # 50 farms × 6 foods
        "rotation_75farms_6foods",     # 75 farms × 6 foods
        "rotation_100farms_6foods",    # 100 farms × 6 foods
        "rotation_large_200",          # 200 farms × 6 foods
        # 27-food scenarios
        "rotation_25farms_27foods",    # 25 farms × 27 foods
        "rotation_50farms_27foods",    # 50 farms × 27 foods
        "rotation_100farms_27foods",   # 100 farms × 27 foods
        "rotation_200farms_27foods",   # 200 farms × 27 foods
    ],
    "qpu-hybrid-27-food": [
        # Hybrid solver for large 27-food problems
        "rotation_25farms_27foods",    # 25 farms × 27 foods = 2025 vars
        "rotation_50farms_27foods",    # 50 farms × 27 foods = 4050 vars
        "rotation_100farms_27foods",   # 100 farms × 27 foods = 8100 vars
        "rotation_200farms_27foods",   # 200 farms × 27 foods = 16200 vars
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Benchmark for Crop Rotation Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Gurobi ground truth on a small scenario
  python unified_benchmark.py --mode gurobi-true-ground-truth --scenario rotation_micro_25

  # Run quantum solver with SA
  python unified_benchmark.py --mode qpu-hierarchical-aggregated --sampler sa --scenario rotation_25farms_27foods

  # Run all modes on default scenarios with SA
  python unified_benchmark.py --run-all --sampler sa --output-json benchmark_results.json

  # Run single mode on multiple scenarios
  python unified_benchmark.py --mode qpu-native-6-family --sampler sa --scenario rotation_micro_25 rotation_small_50

  # List available scenarios
  python unified_benchmark.py --list-scenarios
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", "-m",
        choices=MODES,
        help="Benchmark mode to run"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all modes on their default scenarios"
    )
    
    # Scenario selection
    parser.add_argument(
        "--scenario", "-s",
        nargs="+",
        help="Scenario(s) to run"
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit"
    )
    
    # Sampler selection
    parser.add_argument(
        "--sampler",
        choices=["qpu", "sa"],
        default="sa",
        help="Sampler to use for quantum modes (default: sa)"
    )
    
    # Output
    parser.add_argument(
        "--output-json", "-o",
        default=None,
        help="Output JSON file path (default: benchmark_TIMESTAMP.json)"
    )
    
    # Parameters
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per run in seconds (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--num-reads",
        type=int,
        default=100,
        help="Number of reads for quantum/SA samplers (default: 100)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Boundary coordination iterations for decomposition (default: 3)"
    )
    parser.add_argument(
        "--farms-per-cluster",
        type=int,
        default=None,
        help="Farms per cluster for decomposition (default: auto)"
    )
    parser.add_argument(
        "--area-constant",
        type=float,
        default=DEFAULT_AREA_CONSTANT,
        help=f"Equal area constant for all farms (default: {DEFAULT_AREA_CONSTANT})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--sampleset-dir",
        type=str,
        default=None,
        help="Directory to save samplesets as .pkl files (for QPU runs)"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (minimal output)"
    )
    
    return parser.parse_args()


def list_scenarios():
    """Print available scenarios."""
    print("\nAvailable Scenarios")
    print("=" * 60)
    
    scenarios = get_available_scenarios()
    
    print("\n6-Family Scenarios (native):")
    for s in scenarios["6_family"]:
        try:
            data = load_scenario(s)
            print(f"  {s}: {data['n_farms']} farms × {data['n_foods']} foods = {data['n_vars']} vars")
        except Exception as e:
            print(f"  {s}: (error loading: {e})")
    
    print("\n27-Food Scenarios:")
    for s in scenarios["27_food"]:
        try:
            data = load_scenario(s)
            print(f"  {s}: {data['n_farms']} farms × {data['n_foods']} foods = {data['n_vars']} vars")
        except Exception as e:
            print(f"  {s}: (error loading: {e})")
    
    print("\nSynthetic Scenarios (gap filling):")
    for s in list(scenarios["synthetic"])[:5]:  # Show first 5
        try:
            data = load_scenario(s)
            print(f"  {s}: {data['n_farms']} farms × {data['n_foods']} foods = {data['n_vars']} vars")
        except Exception as e:
            print(f"  {s}: (error loading: {e})")
    if len(scenarios["synthetic"]) > 5:
        print(f"  ... and {len(scenarios['synthetic']) - 5} more")
    
    print()


def run_benchmark(
    mode: str,
    scenarios: list,
    use_qpu: bool,
    num_reads: int,
    timeout: float,
    area_constant: float,
    seed: int,
    num_iterations: int = 3,
    farms_per_cluster: int = None,
    verbose: bool = True,
    logger: BenchmarkLogger = None,
    sampleset_dir: str = None,
) -> list:
    """
    Run benchmark for a mode on multiple scenarios.
    
    Returns list of RunEntry objects.
    """
    if logger is None:
        logger = BenchmarkLogger()
    
    results = []
    
    for scenario_name in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {mode} on {scenario_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Load scenario with equal areas
            data = load_scenario(scenario_name, area_constant=area_constant)
            logger.scenario_load(scenario_name, data["n_farms"], data["n_foods"])
            logger.info(f"Total area: {data['total_area']:.2f} (= {data['n_farms']} farms × {area_constant:.2f})")
            
            # Determine farms_per_cluster if not specified
            if farms_per_cluster is None:
                if mode == "qpu-hybrid-27-food":
                    # 27 foods × 3 periods = 81 vars/farm, max ~2 farms
                    fpc = 2
                elif mode == "qpu-hierarchical-aggregated":
                    # 6 families × 3 periods = 18 vars/farm, max ~9 farms
                    fpc = 10
                else:
                    fpc = 10
            else:
                fpc = farms_per_cluster
            
            # Run solver
            result = solve(
                mode=mode,
                scenario_data=data,
                use_qpu=use_qpu,
                num_reads=num_reads,
                timeout=timeout,
                verbose=verbose,
                logger=logger,
                seed=seed,
                num_iterations=num_iterations,
                farms_per_cluster=fpc,
                sampleset_dir=sampleset_dir,
            )
            
            results.append(result)
            
            # Print summary
            logger.info(f"\nResult Summary:")
            logger.info(f"  Status: {result.status}")
            logger.info(f"  MIQP Objective: {result.objective_miqp}")
            logger.info(f"  Model Objective: {result.objective_model}")
            logger.info(f"  Total Time: {result.timing.total_wall_time:.2f}s")
            logger.info(f"  Feasible: {result.feasible}")
            
        except Exception as e:
            logger.error(f"Failed to run {scenario_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def main():
    args = parse_args()
    
    # List scenarios and exit
    if args.list_scenarios:
        list_scenarios()
        return 0
    
    # Setup logging
    import logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logger = BenchmarkLogger(level=log_level)
    
    # Check dependencies
    logger.info("Unified Benchmark for Crop Rotation Optimization")
    logger.info(f"Schema version: {SCHEMA_VERSION}")
    logger.info(f"dimod available: {HAS_DIMOD}")
    logger.info(f"QPU available: {HAS_QPU}")
    
    if not HAS_DIMOD:
        logger.error("dimod not available. Install with: pip install dimod")
        return 1
    
    if args.sampler == "qpu" and not HAS_QPU:
        logger.warning("QPU not available, falling back to SA")
        args.sampler = "sa"
    
    use_qpu = args.sampler == "qpu"
    
    # Determine what to run
    if args.run_all:
        # Run all modes on default scenarios
        modes_to_run = MODES
        scenario_map = DEFAULT_SCENARIOS
    elif args.mode:
        # Run single mode
        modes_to_run = [args.mode]
        if args.scenario:
            scenario_map = {args.mode: args.scenario}
        else:
            scenario_map = {args.mode: DEFAULT_SCENARIOS.get(args.mode, ["rotation_micro_25"])}
    else:
        logger.error("Must specify --mode or --run-all")
        return 1
    
    # Collect all results
    all_results = []
    
    for mode in modes_to_run:
        scenarios = scenario_map.get(mode, [])
        if not scenarios:
            logger.warning(f"No scenarios for mode {mode}, skipping")
            continue
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"MODE: {mode}")
        logger.info(f"Scenarios: {scenarios}")
        logger.info(f"{'#'*70}")
        
        results = run_benchmark(
            mode=mode,
            scenarios=scenarios,
            use_qpu=use_qpu,
            num_reads=args.num_reads,
            timeout=args.timeout,
            area_constant=args.area_constant,
            seed=args.seed,
            num_iterations=args.num_iterations,
            farms_per_cluster=args.farms_per_cluster,
            verbose=not args.quiet,
            logger=logger,
            sampleset_dir=args.sampleset_dir,
        )
        
        all_results.extend(results)
    
    # Save results
    if all_results:
        if args.output_json:
            output_path = args.output_json
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"benchmark_{timestamp}.json"
        
        save_benchmark_results(all_results, output_path)
        logger.json_write(output_path, len(all_results))
        
        # Print final summary
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Mode':<35} {'Scenario':<30} {'Status':<10} {'MIQP Obj':>12} {'Time':>8}")
        print("-" * 95)
        
        for r in all_results:
            obj_str = f"{r.objective_miqp:.4f}" if r.objective_miqp is not None else "N/A"
            time_str = f"{r.timing.total_wall_time:.1f}s" if r.timing else "N/A"
            print(f"{r.mode:<35} {r.scenario_name:<30} {r.status:<10} {obj_str:>12} {time_str:>8}")
        
        print("-" * 95)
        print(f"\nResults saved to: {output_path}")
    else:
        logger.warning("No results to save")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
