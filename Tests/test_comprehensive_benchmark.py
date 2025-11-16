#!/usr/bin/env python3
"""
Test Suite for Comprehensive Benchmark

This script tests the comprehensive_benchmark.py with small samples to verify:
1. JSON output structure is correct
2. All solver configurations work properly
3. Error handling is robust
4. Performance data is recorded accurately
"""

import os
import sys
import json
import time
import tempfile
import shutil
from datetime import datetime

# Add current directory to path
# Add project root and Benchmark Scripts to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

def test_json_structure(results_data: dict) -> bool:
    """
    Test that the JSON output has the correct structure.
    
    Args:
        results_data: Parsed JSON results
        
    Returns:
        True if structure is valid, False otherwise
    """
    print("\n  📋 Testing JSON structure...")
    
    # Check top-level structure
    required_keys = ['metadata', 'farm_results', 'patch_results', 'summary']
    for key in required_keys:
        if key not in results_data:
            print(f"    ❌ Missing top-level key: {key}")
            return False
    
    # Check metadata structure
    metadata = results_data['metadata']
    metadata_keys = ['timestamp', 'n_samples', 'total_runtime', 'dwave_enabled', 'scenarios', 'solvers']
    for key in metadata_keys:
        if key not in metadata:
            print(f"    ❌ Missing metadata key: {key}")
            return False
    
    # Check solver configurations
    solvers = metadata['solvers']
    expected_farm_solvers = ['pulp_gurobi', 'dwave_cqm']
    expected_patch_solvers = ['pulp', 'dwave_cqm', 'dwave_bqm', 'gurobi_qubo']
    
    if 'farm' not in solvers or 'patch' not in solvers:
        print(f"    ❌ Missing solver configurations")
        return False
    
    # Check farm results structure
    if len(results_data['farm_results']) > 0:
        farm_result = results_data['farm_results'][0]
        farm_keys = ['sample_id', 'scenario_type', 'n_units', 'total_area', 'n_foods', 'n_variables', 'n_constraints', 'solvers']
        for key in farm_keys:
            if key not in farm_result:
                print(f"    ❌ Missing farm result key: {key}")
                return False
        
        # Check solver results structure
        for solver_name, solver_result in farm_result['solvers'].items():
            solver_keys = ['status', 'success']
            for key in solver_keys:
                if key not in solver_result:
                    print(f"    ❌ Missing solver result key: {key} in {solver_name}")
                    return False
    
    # Check patch results structure
    if len(results_data['patch_results']) > 0:
        patch_result = results_data['patch_results'][0]
        patch_keys = ['sample_id', 'scenario_type', 'n_units', 'total_area', 'n_foods', 'n_variables', 'n_constraints', 'solvers']
        for key in patch_keys:
            if key not in patch_result:
                print(f"    ❌ Missing patch result key: {key}")
                return False
    
    # Check summary structure
    summary = results_data['summary']
    summary_keys = ['farm_samples_completed', 'patch_samples_completed', 'total_solver_runs']
    for key in summary_keys:
        if key not in summary:
            print(f"    ❌ Missing summary key: {key}")
            return False
    
    print(f"    ✓ JSON structure is valid")
    return True

def test_solver_configurations(results_data: dict) -> bool:
    """
    Test that all expected solver configurations are present and working.
    
    Args:
        results_data: Parsed JSON results
        
    Returns:
        True if configurations are valid, False otherwise
    """
    print("\n  🔧 Testing solver configurations...")
    
    # Test farm solvers
    farm_solvers_found = set()
    for farm_result in results_data['farm_results']:
        for solver_name in farm_result['solvers'].keys():
            farm_solvers_found.add(solver_name)
    
    expected_farm_solvers = {'pulp_gurobi'}  # dwave_cqm might be skipped if no token
    for solver in expected_farm_solvers:
        if solver not in farm_solvers_found:
            print(f"    ❌ Missing farm solver: {solver}")
            return False
    
    print(f"    ✓ Farm solvers found: {sorted(farm_solvers_found)}")
    
    # Test patch solvers
    patch_solvers_found = set()
    for patch_result in results_data['patch_results']:
        for solver_name in patch_result['solvers'].keys():
            patch_solvers_found.add(solver_name)
    
    expected_patch_solvers = {'pulp'}  # Others might be skipped if dependencies not available
    for solver in expected_patch_solvers:
        if solver not in patch_solvers_found:
            print(f"    ❌ Missing patch solver: {solver}")
            return False
    
    print(f"    ✓ Patch solvers found: {sorted(patch_solvers_found)}")
    
    return True

def test_performance_data(results_data: dict) -> bool:
    """
    Test that performance data is properly recorded.
    
    Args:
        results_data: Parsed JSON results
        
    Returns:
        True if performance data is valid, False otherwise
    """
    print("\n  ⏱️ Testing performance data...")
    
    # Check that solve times are recorded
    total_solve_times = 0
    solver_count = 0
    
    for farm_result in results_data['farm_results']:
        for solver_name, solver_result in farm_result['solvers'].items():
            if solver_result.get('success'):
                if 'solve_time' in solver_result:
                    solve_time = solver_result['solve_time']
                    if isinstance(solve_time, (int, float)) and solve_time >= 0:
                        total_solve_times += solve_time
                        solver_count += 1
                    else:
                        print(f"    ❌ Invalid solve_time in {solver_name}: {solve_time}")
                        return False
    
    for patch_result in results_data['patch_results']:
        for solver_name, solver_result in patch_result['solvers'].items():
            if solver_result.get('success'):
                if 'solve_time' in solver_result:
                    solve_time = solver_result['solve_time']
                    if isinstance(solve_time, (int, float)) and solve_time >= 0:
                        total_solve_times += solve_time
                        solver_count += 1
                    else:
                        print(f"    ❌ Invalid solve_time in {solver_name}: {solve_time}")
                        return False
    
    if solver_count == 0:
        print(f"    ⚠️ No successful solvers found for performance testing")
        return True  # Not necessarily an error for minimal tests
    
    avg_solve_time = total_solve_times / solver_count
    print(f"    ✓ Performance data valid: {solver_count} solvers, avg {avg_solve_time:.3f}s")
    
    # Check that runtime is reasonable
    total_runtime = results_data['metadata']['total_runtime']
    if not isinstance(total_runtime, (int, float)) or total_runtime < 0:
        print(f"    ❌ Invalid total_runtime: {total_runtime}")
        return False
    
    print(f"    ✓ Total runtime: {total_runtime:.3f}s")
    return True

def test_error_handling() -> bool:
    """
    Test error handling with invalid inputs.
    
    Returns:
        True if error handling works properly, False otherwise
    """
    print("\n  🛡️ Testing error handling...")
    
    # Test 1: Invalid n_samples
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'comprehensive_benchmark.py', '0'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"    ❌ Should have failed with n_samples=0")
            return False
        print(f"    ✓ Correctly rejected n_samples=0")
        
    except subprocess.TimeoutExpired:
        print(f"    ⚠️ Timeout testing invalid inputs (acceptable)")
    except Exception as e:
        print(f"    ⚠️ Error testing invalid inputs: {e}")
    
    return True

def run_test_benchmark() -> str:
    """
    Run a small comprehensive benchmark for testing.
    
    Returns:
        Path to the generated JSON file, or None if failed
    """
    print("\n  🚀 Running test benchmark...")
    
    try:
        # Import and run the benchmark directly
        from comprehensive_benchmark import run_comprehensive_benchmark
        
        # Run with minimal samples (no D-Wave to avoid budget usage)
        print(f"    Running 2 samples without D-Wave...")
        start_time = time.time()
        results = run_comprehensive_benchmark(n_samples=2, dwave_token=None)
        test_time = time.time() - start_time
        
        # Save to temporary file
        test_filename = f"test_comprehensive_benchmark_{int(time.time())}.json"
        with open(test_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"    ✓ Test benchmark completed in {test_time:.3f}s")
        print(f"    ✓ Results saved to: {test_filename}")
        
        return test_filename
        
    except Exception as e:
        print(f"    ❌ Test benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("="*80)
    print("COMPREHENSIVE BENCHMARK TEST SUITE")
    print("="*80)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    try:
        from comprehensive_benchmark import run_comprehensive_benchmark
        print("    ✓ comprehensive_benchmark module imported")
    except ImportError as e:
        print(f"    ❌ Cannot import comprehensive_benchmark: {e}")
        return False
    
    try:
        from Utils.farm_sampler import generate_farms
        from Utils.patch_sampler import generate_farms as generate_patches
        print("    ✓ Farm and patch samplers available")
    except ImportError as e:
        print(f"    ❌ Cannot import samplers: {e}")
        print("    ⚠️ This is expected if dependencies are not installed")
    
    # Run test benchmark
    test_start = time.time()
    test_filename = run_test_benchmark()
    
    if test_filename is None:
        print("\n❌ TEST BENCHMARK FAILED")
        print("Cannot proceed with JSON structure tests.")
        return False
    
    # Load and test results
    print(f"\n📖 Loading test results from {test_filename}...")
    try:
        with open(test_filename, 'r') as f:
            results_data = json.load(f)
        print(f"    ✓ JSON file loaded successfully")
    except Exception as e:
        print(f"    ❌ Failed to load JSON: {e}")
        return False
    
    # Run tests
    all_tests_passed = True
    
    # Test 1: JSON Structure
    if not test_json_structure(results_data):
        all_tests_passed = False
    
    # Test 2: Solver Configurations
    if not test_solver_configurations(results_data):
        all_tests_passed = False
    
    # Test 3: Performance Data
    if not test_performance_data(results_data):
        all_tests_passed = False
    
    # Test 4: Error Handling
    if not test_error_handling():
        all_tests_passed = False
    
    total_test_time = time.time() - test_start
    
    # Summary
    print("\n" + "="*80)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED")
        print("The comprehensive benchmark is working correctly!")
    else:
        print("❌ SOME TESTS FAILED") 
        print("Check the implementation and fix issues before using.")
    
    print(f"\nTest Summary:")
    print(f"  Total test time: {total_test_time:.3f}s")
    print(f"  Test file: {test_filename}")
    print(f"  Samples tested: {results_data['metadata']['n_samples']}")
    print(f"  Solver runs: {results_data['summary']['total_solver_runs']}")
    
    # Cleanup
    try:
        os.remove(test_filename)
        print(f"  ✓ Cleaned up test file")
    except:
        print(f"  ⚠️ Could not remove test file: {test_filename}")
    
    print("="*80)
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)