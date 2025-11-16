#!/usr/bin/env python3
"""
Quick test script for comprehensive benchmark
Tests with 1 sample to verify everything works
"""

import subprocess
import sys
import os

def main():
    """Run a quick test of the comprehensive benchmark."""
    
    print("="*80)
    print("COMPREHENSIVE BENCHMARK - QUICK TEST")
    print("="*80)
    print("\nThis will run a quick test with 1 sample (no D-Wave)")
    print("Expected runtime: ~5-10 minutes (mostly Gurobi QUBO)")
    print()
    
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Test cancelled")
        sys.exit(0)
    
    # Get the Python executable path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run the benchmark
    cmd = [
        sys.executable,
        os.path.join(script_dir, "comprehensive_benchmark.py"),
        "1"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("="*80)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*80)
        print("✅ TEST PASSED - Comprehensive benchmark works!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Check results in: Benchmarks/COMPREHENSIVE/")
        print("  2. Run with more samples: python comprehensive_benchmark.py 5")
        print("  3. Run with configs: python comprehensive_benchmark.py --configs")
        print("  4. Enable D-Wave: python comprehensive_benchmark.py 5 --dwave")
        
    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("⚠️  TEST INTERRUPTED BY USER")
        print("="*80)
        sys.exit(1)

if __name__ == "__main__":
    main()
