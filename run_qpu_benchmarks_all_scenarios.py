#!/usr/bin/env python3
"""
Run QPU benchmarks for all scenarios from gurobi_timeout_test CSV file.

This script runs all 3 QPU methods (native-6-family, hierarchical-aggregated, hybrid-27-food)
on all 20 scenarios from the CSV benchmark results.

Usage:
    python run_qpu_benchmarks_all_scenarios.py
    python run_qpu_benchmarks_all_scenarios.py --sampler qpu  # Use real QPU instead of SA
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path

# All scenarios from the CSV file
SCENARIOS = [
    "rotation_micro_25",
    "rotation_small_50",
    "rotation_15farms_6foods",
    "rotation_medium_100",
    "rotation_25farms_6foods",
    "rotation_large_200",
    "rotation_50farms_6foods",
    "rotation_75farms_6foods",
    "rotation_100farms_6foods",
    "rotation_25farms_27foods",
    "rotation_150farms_6foods",
    "rotation_50farms_27foods",
    "rotation_75farms_27foods",
    "rotation_100farms_27foods",
    "rotation_150farms_27foods",
    "rotation_200farms_27foods",
    "rotation_250farms_27foods",
    "rotation_350farms_27foods",
    "rotation_500farms_27foods",
    "rotation_1000farms_27foods",
]

# All 3 QPU modes
QPU_MODES = [
    "qpu-native-6-family",
    "qpu-hierarchical-aggregated",
    "qpu-hybrid-27-food",
]

def run_benchmark(mode: str, scenario: str, sampler: str = "sa", timeout: float = 300):
    """Run a single benchmark."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_{mode.replace('-','_')}_{scenario}_{timestamp}.json"
    
    cmd = [
        sys.executable,
        "unified_benchmark.py",
        "--mode", mode,
        "--sampler", sampler,
        "--scenario", scenario,
        "--timeout", str(timeout),
        "--output-json", output_file,
    ]
    
    print(f"\n{'='*70}")
    print(f"Running: {mode} on {scenario}")
    print(f"Output: {output_file}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=timeout + 120,  # Extra buffer for setup
        )
        return {
            "mode": mode,
            "scenario": scenario,
            "success": result.returncode == 0,
            "output_file": output_file,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "mode": mode,
            "scenario": scenario,
            "success": False,
            "output_file": output_file,
            "error": "timeout",
        }
    except Exception as e:
        return {
            "mode": mode,
            "scenario": scenario,
            "success": False,
            "output_file": output_file,
            "error": str(e),
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run QPU benchmarks on all scenarios")
    parser.add_argument("--sampler", choices=["sa", "qpu"], default="sa",
                       help="Sampler to use (default: sa)")
    parser.add_argument("--timeout", type=float, default=300,
                       help="Timeout per benchmark (default: 300s)")
    parser.add_argument("--modes", nargs="+", choices=QPU_MODES, default=QPU_MODES,
                       help="QPU modes to run (default: all 3)")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS,
                       help="Scenarios to run (default: all 20)")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"qpu_benchmark_summary_{timestamp}.json"
    
    print(f"\n{'#'*70}")
    print(f"# QPU Benchmark Run - {timestamp}")
    print(f"# Sampler: {args.sampler}")
    print(f"# Modes: {len(args.modes)}")
    print(f"# Scenarios: {len(args.scenarios)}")
    print(f"# Total runs: {len(args.modes) * len(args.scenarios)}")
    print(f"{'#'*70}\n")
    
    all_results = []
    success_count = 0
    fail_count = 0
    
    for mode in args.modes:
        print(f"\n\n{'*'*70}")
        print(f"* MODE: {mode}")
        print(f"{'*'*70}")
        
        for scenario in args.scenarios:
            result = run_benchmark(mode, scenario, args.sampler, args.timeout)
            all_results.append(result)
            
            if result.get("success"):
                success_count += 1
                print(f"✓ {mode} on {scenario}: SUCCESS")
            else:
                fail_count += 1
                print(f"✗ {mode} on {scenario}: FAILED ({result.get('error', 'unknown')})")
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "sampler": args.sampler,
        "timeout": args.timeout,
        "modes": args.modes,
        "scenarios": args.scenarios,
        "total_runs": len(all_results),
        "success_count": success_count,
        "fail_count": fail_count,
        "results": all_results,
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n\n{'#'*70}")
    print(f"# SUMMARY")
    print(f"# Total: {len(all_results)} | Success: {success_count} | Failed: {fail_count}")
    print(f"# Summary saved to: {summary_file}")
    print(f"{'#'*70}")

if __name__ == "__main__":
    main()
