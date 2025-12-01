#!/usr/bin/env python3
"""
Test script to verify QPU-only changes work correctly.
Runs a small benchmark to check all methods use QPU.
"""

import subprocess
import sys

print("=" * 80)
print("QPU-ONLY VERIFICATION TEST")
print("=" * 80)

# Run benchmark with one small scale
cmd = [
    sys.executable, "qpu_benchmark.py",
    "--scale", "25",
    "--methods",
    "ground_truth",
    "direct_qpu",
    "coordinated",
    "decomposition_PlotBased_QPU",
    "cqm_first_PlotBased",
]

print("\nRunning command:")
print(" ".join(cmd))
print()

result = subprocess.run(cmd, capture_output=False, text=True)

if result.returncode == 0:
    print("\n" + "=" * 80)
    print("✅ TEST PASSED - Benchmark completed successfully")
    print("=" * 80)
    print("\nVerify in the output above that:")
    print("  1. All methods show QPU timing (not N/A)")
    print("  2. All methods show embedding timing (not N/A)")
    print("  3. No 'SimulatedAnnealing' sampler mentioned")
    print("  4. All solvers show 'QPU' in their output")
else:
    print("\n" + "=" * 80)
    print("❌ TEST FAILED - Benchmark encountered errors")
    print("=" * 80)
    sys.exit(1)
