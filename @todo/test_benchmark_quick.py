#!/usr/bin/env python3
"""
Quick test of comprehensive benchmark with smaller problem sizes
"""

import os
import sys

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the benchmark script and override configuration
import comprehensive_embedding_and_solving_benchmark as benchmark

# Override configuration for quick test
benchmark.PROBLEM_SIZES = [5]  # Just one size
benchmark.EMBEDDING_TIMEOUT = 10  # 10 seconds only
benchmark.SOLVE_TIMEOUT = 30  # 30 seconds
benchmark.DECOMPOSITIONS = ["None", "PlotBased"]  # Just 2 decompositions

print("=" * 80)
print("QUICK TEST - Reduced configuration for verification")
print("=" * 80)
print(f"Problem sizes: {benchmark.PROBLEM_SIZES}")
print(f"Decompositions: {benchmark.DECOMPOSITIONS}")
print(f"Embedding timeout: {benchmark.EMBEDDING_TIMEOUT}s")
print("=" * 80)

# Run benchmark
if __name__ == "__main__":
    try:
        results = benchmark.run_comprehensive_benchmark()
        print("\n" + "=" * 80)
        print("QUICK TEST PASSED!")
        print("=" * 80)
        print(f"Completed {len(results)} experiments")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
