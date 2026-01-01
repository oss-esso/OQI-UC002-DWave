#!/usr/bin/env python3
"""
Quick QPU benchmark runner - comprehensive scenarios up to 200 farms.
Uses all 3 QPU methods:
  - qpu-native-6-family (6-food scenarios)
  - qpu-hierarchical-aggregated (6-food and 27-food scenarios)  
  - qpu-hybrid-27-food (27-food scenarios only)

Saves samplesets to .pkl files.
"""

import subprocess
import sys
import json
import csv
import re
import pickle
from datetime import datetime
from pathlib import Path

# Increase CSV field size limit for large solution dictionaries
csv.field_size_limit(10_000_000)

PKL_OUTPUT_DIR = Path("qpu_samplesets")

# Comprehensive scenarios from CSV + extended to 200 farms
SCENARIOS_6FOOD = [
    "rotation_micro_25",           # 5 farms × 6 foods
    "rotation_small_50",           # 10 farms × 6 foods
    "rotation_15farms_6foods",     # 15 farms × 6 foods
    "rotation_medium_100",         # 20 farms × 6 foods
    "rotation_25farms_6foods",     # 25 farms × 6 foods
    "rotation_50farms_6foods",     # 50 farms × 6 foods
    "rotation_75farms_6foods",     # 75 farms × 6 foods
    "rotation_100farms_6foods",    # 100 farms × 6 foods
    "rotation_large_200",          # 200 farms × 6 foods
]

SCENARIOS_27FOOD = [
    "rotation_25farms_27foods",    # 25 farms × 27 foods
    "rotation_50farms_27foods",    # 50 farms × 27 foods
    "rotation_100farms_27foods",   # 100 farms × 27 foods
    "rotation_200farms_27foods",   # 200 farms × 27 foods
]

def run_benchmark(mode, scenario, timeout=100):
    """Run a single benchmark."""
    timestamp = datetime.now().strftime("%H%M%S")
    output_file = f"qpu_{mode.split('-')[1]}_{scenario}_{timestamp}.json"
    
    cmd = [
        sys.executable, "unified_benchmark.py",
        "--mode", mode,
        "--sampler", "qpu",  # Use real QPU
        "--scenario", scenario,
        "--timeout", str(timeout),
        "--output-json", output_file,
        "--sampleset-dir", str(PKL_OUTPUT_DIR),  # Save samplesets to .pkl
    ]
    
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {mode} → {scenario}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, timeout=timeout + 120)
        return result.returncode == 0, output_file
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout + 120}s")
        return False, None
    except Exception as e:
        print(f"ERROR: {e}")
        return False, None

def main():
    start_time = datetime.now()
    print(f"\n{'#'*60}")
    print(f"# QPU BENCHMARK - Started {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# All 3 QPU methods | Samplesets: .pkl saved")
    print(f"{'#'*60}")
    
    # Create output directory for pkl files
    PKL_OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Build comprehensive run list
    runs = []
    
    # 1. qpu-native-6-family: all 6-food scenarios
    for scenario in SCENARIOS_6FOOD:
        runs.append(("qpu-native-6-family", scenario))
    
    # 2. qpu-hierarchical-aggregated: all scenarios (6-food + 27-food)
    for scenario in SCENARIOS_6FOOD + SCENARIOS_27FOOD:
        runs.append(("qpu-hierarchical-aggregated", scenario))
    
    # 3. qpu-hybrid-27-food: only 27-food scenarios
    for scenario in SCENARIOS_27FOOD:
        runs.append(("qpu-hybrid-27-food", scenario))
    
    print(f"\nTotal runs planned: {len(runs)}")
    print(f"  qpu-native-6-family: {len(SCENARIOS_6FOOD)} scenarios")
    print(f"  qpu-hierarchical-aggregated: {len(SCENARIOS_6FOOD) + len(SCENARIOS_27FOOD)} scenarios")
    print(f"  qpu-hybrid-27-food: {len(SCENARIOS_27FOOD)} scenarios")
    
    results = []
    success = 0
    
    for i, (mode, scenario) in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}]", end="")
        ok, outfile = run_benchmark(mode, scenario, timeout=100)
        results.append({"mode": mode, "scenario": scenario, "success": ok, "file": outfile})
        if ok:
            success += 1
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print(f"\n\n{'#'*60}")
    print(f"# COMPLETE - {end_time.strftime('%H:%M:%S')}")
    print(f"# Duration: {duration:.1f} minutes")
    print(f"# Success: {success}/{len(runs)}")
    print(f"{'#'*60}")
    
    # Save summary
    summary_file = f"qpu_batch_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump({"runs": results, "success": success, "total": len(runs)}, f, indent=2)
    print(f"Summary: {summary_file}")

if __name__ == "__main__":
    main()
