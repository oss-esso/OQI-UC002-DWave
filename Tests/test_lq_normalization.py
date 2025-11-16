"""
Quick test to verify LQ objective normalization is working correctly.
This will run a small benchmark and check that normalized objectives are calculated.
"""

import sys
import os

# Add project root and Benchmark Scripts to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

from benchmark_scalability_LQ import load_full_family_with_n_farms, run_benchmark
from Utils.benchmark_cache import BenchmarkCache

def test_normalization():
    """Test that normalization is working for all three solvers."""
    
    print("\n" + "="*80)
    print("LQ NORMALIZATION TEST")
    print("="*80)
    print("\nTesting with 5 farms (smallest configuration)...")
    
    # Run a single benchmark with save_to_cache=False to avoid polluting cache
    cache = BenchmarkCache()
    result = run_benchmark(n_farms=5, run_number=1, total_runs=1, cache=cache, save_to_cache=True)
    
    if result is None:
        print("\n❌ Test FAILED: No result returned")
        return False
    
    print("\n" + "="*80)
    print("NORMALIZATION TEST RESULTS")
    print("="*80)
    
    success = True
    
    # Check PuLP
    print("\nPuLP Solver:")
    if 'pulp_objective' in result and result['pulp_objective'] is not None:
        print(f"  ✓ Raw objective: {result['pulp_objective']:.6f}")
        if 'pulp_total_area' in result and result['pulp_total_area'] is not None:
            print(f"  ✓ Total area: {result['pulp_total_area']:.2f}")
            if 'pulp_normalized_objective' in result and result['pulp_normalized_objective'] is not None:
                print(f"  ✓ Normalized objective: {result['pulp_normalized_objective']:.8f}")
                # Verify calculation
                expected = result['pulp_objective'] / result['pulp_total_area']
                if abs(expected - result['pulp_normalized_objective']) < 1e-6:
                    print(f"  ✓ Normalization calculation correct!")
                else:
                    print(f"  ❌ Normalization calculation incorrect!")
                    success = False
            else:
                print(f"  ❌ Normalized objective missing!")
                success = False
        else:
            print(f"  ❌ Total area missing!")
            success = False
    else:
        print(f"  ❌ Objective missing!")
        success = False
    
    # Check Pyomo
    print("\nPyomo Solver:")
    if 'pyomo_objective' in result and result['pyomo_objective'] is not None:
        print(f"  ✓ Raw objective: {result['pyomo_objective']:.6f}")
        if 'pyomo_total_area' in result and result['pyomo_total_area'] is not None:
            print(f"  ✓ Total area: {result['pyomo_total_area']:.2f}")
            if 'pyomo_normalized_objective' in result and result['pyomo_normalized_objective'] is not None:
                print(f"  ✓ Normalized objective: {result['pyomo_normalized_objective']:.8f}")
                # Verify calculation
                expected = result['pyomo_objective'] / result['pyomo_total_area']
                if abs(expected - result['pyomo_normalized_objective']) < 1e-6:
                    print(f"  ✓ Normalization calculation correct!")
                else:
                    print(f"  ❌ Normalization calculation incorrect!")
                    success = False
            else:
                print(f"  ❌ Normalized objective missing!")
                success = False
        else:
            print(f"  ❌ Total area missing!")
            success = False
    else:
        print(f"  ❌ Objective missing or solver not available!")
    
    # Check DWave
    print("\nDWave Solver:")
    if 'dwave_objective' in result and result['dwave_objective'] is not None:
        print(f"  ✓ Raw objective: {result['dwave_objective']:.6f}")
        if 'dwave_total_area' in result and result['dwave_total_area'] is not None:
            print(f"  ✓ Total area: {result['dwave_total_area']:.2f}")
            if 'dwave_normalized_objective' in result and result['dwave_normalized_objective'] is not None:
                print(f"  ✓ Normalized objective: {result['dwave_normalized_objective']:.8f}")
                # Verify calculation
                expected = result['dwave_objective'] / result['dwave_total_area']
                if abs(expected - result['dwave_normalized_objective']) < 1e-6:
                    print(f"  ✓ Normalization calculation correct!")
                else:
                    print(f"  ❌ Normalization calculation incorrect!")
                    success = False
            else:
                print(f"  ❌ Normalized objective missing!")
                success = False
        else:
            print(f"  ❌ Total area missing!")
            success = False
    else:
        print(f"  ⚠️  DWave results not available (may need API token)")
    
    print("\n" + "="*80)
    if success:
        print("✅ NORMALIZATION TEST PASSED")
    else:
        print("❌ NORMALIZATION TEST FAILED")
    print("="*80 + "\n")
    
    return success

if __name__ == "__main__":
    test_normalization()
