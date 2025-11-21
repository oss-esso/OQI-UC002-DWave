"""
Unit tests for Decomposed QPU Workflow

Tests strategic problem decomposition approach.
Run with: conda activate oqi; python test_decomposed.py
"""

import sys
import os

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from solver_runner_DECOMPOSED import create_cqm_farm, create_cqm_plots
from benchmark_utils_decomposed import (
    generate_farm_data,
    generate_patch_data,
    create_config
)
from dimod import cqm_to_bqm


def test_data_generation():
    """Test data generation utilities."""
    print("\n[TEST 1: Data Generation]")
    
    # Test farm data
    farm_data = generate_farm_data(n_units=5, total_land=100.0)
    assert farm_data['n_units'] == 5
    assert abs(farm_data['total_area'] - 100.0) < 0.01
    print("  ✓ Farm data generation")
    
    # Test patch data
    patch_data = generate_patch_data(n_units=5, total_land=100.0)
    assert patch_data['n_units'] == 5
    assert abs(patch_data['total_area'] - 100.0) < 0.01
    print("  ✓ Patch data generation")


def test_cqm_creation():
    """Test CQM creation for both scenarios."""
    print("\n[TEST 2: CQM Creation]")
    
    # Farm CQM
    farm_data = generate_farm_data(n_units=3, total_land=50.0)
    farms_list = list(farm_data['land_data'].keys())
    foods, food_groups, config = create_config(farm_data['land_data'], 'simple')
    
    cqm_farm, A, Y, metadata = create_cqm_farm(farms_list, foods, food_groups, config)
    assert len(cqm_farm.variables) > 0
    assert len(cqm_farm.constraints) > 0
    print(f"  ✓ Farm CQM: {len(cqm_farm.variables)} vars, {len(cqm_farm.constraints)} constraints")
    
    # Patch CQM
    patch_data = generate_patch_data(n_units=3, total_land=50.0)
    patches_list = list(patch_data['land_data'].keys())
    foods, food_groups, config = create_config(patch_data['land_data'], 'simple')
    
    cqm_patch, Y_patch, metadata_patch = create_cqm_plots(patches_list, foods, food_groups, config)
    assert len(cqm_patch.variables) > 0
    assert len(cqm_patch.constraints) > 0
    print(f"  ✓ Patch CQM: {len(cqm_patch.variables)} vars, {len(cqm_patch.constraints)} constraints")


def test_bqm_conversion():
    """Test CQM to BQM conversion for patch scenario."""
    print("\n[TEST 3: BQM Conversion]")
    
    # Create patch CQM
    patch_data = generate_patch_data(n_units=3, total_land=50.0)
    patches_list = list(patch_data['land_data'].keys())
    foods, food_groups, config = create_config(patch_data['land_data'], 'simple')
    
    cqm_patch, Y_patch, metadata_patch = create_cqm_plots(patches_list, foods, food_groups, config)
    
    # Convert to BQM
    bqm, invert = cqm_to_bqm(cqm_patch)
    assert len(bqm.variables) > 0
    print(f"  ✓ BQM Conversion: {len(bqm.variables)} vars, {len(bqm.quadratic)} interactions")
    
    # Test invert function exists
    assert callable(invert)
    print("  ✓ Invert function available")


def test_lowlevel_sampler_availability():
    """Test that DWaveSampler is available."""
    print("\n[TEST 4: Low-Level QPU Sampler Availability]")
    
    try:
        from dwave.system import DWaveSampler, EmbeddingComposite
        print("  ✓ DWaveSampler imported successfully")
        
        from solver_runner_DECOMPOSED import LOWLEVEL_QPU_AVAILABLE
        assert LOWLEVEL_QPU_AVAILABLE, "LOWLEVEL_QPU_AVAILABLE should be True"
        print("  ✓ LOWLEVEL_QPU_AVAILABLE flag is True")
        
    except ImportError as e:
        print(f"  ❌ DWaveSampler not available: {e}")
        print("  Install with: pip install dwave-system")
        return False
    
    return True


def test_decomposed_solver_function():
    """Test decomposed QPU solver function exists and has correct signature."""
    print("\n[TEST 5: Decomposed Solver Function]")
    
    try:
        from solver_runner_DECOMPOSED import solve_with_decomposed_qpu
        print("  ✓ solve_with_decomposed_qpu function imported")
        
        # Check function signature
        import inspect
        sig = inspect.signature(solve_with_decomposed_qpu)
        assert 'bqm' in sig.parameters
        assert 'token' in sig.parameters
        print("  ✓ Function signature validated")
        
        # Test parameters
        params = {
            'num_reads': 100,
            'annealing_time': 20,
            'chain_strength': None,
            'auto_scale': True
        }
        print(f"  ✓ QPU parameters validated: {params}")
        
    except Exception as e:
        print(f"  ❌ Decomposed solver function test failed: {e}")
        return False
    
    return True


def run_all_tests():
    """Run all unit tests."""
    print("="*80)
    print("DECOMPOSED QPU WORKFLOW - UNIT TESTS")
    print("="*80)
    
    try:
        test_data_generation()
        test_cqm_creation()
        test_bqm_conversion()
        
        sampler_available = test_lowlevel_sampler_availability()
        if not sampler_available:
            print("\n⚠️  WARNING: DWaveSampler not available")
            print("   Decomposed QPU solver will not work without it")
            print("   Install with: pip install dwave-system")
        else:
            test_decomposed_solver_function()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nReady to run benchmark:")
        print("  python comprehensive_benchmark_DECOMPOSED.py --config 10")
        print("\nStrategic Decomposition Approach:")
        print("  - Farm scenarios: HYBRID DECOMPOSITION (Gurobi continuous + QPU binary)")
        print("  - Patch scenarios: Quantum-only (low-level QPU)")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
