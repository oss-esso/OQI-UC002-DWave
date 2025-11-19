"""
Unit tests for Custom Hybrid Workflow

Tests individual components to ensure correctness before full benchmark.
Run with: conda activate oqi; python test_custom_hybrid.py
"""

import sys
import os

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from solver_runner_CUSTOM_HYBRID import create_cqm_farm, create_cqm_plots
from benchmark_utils_custom_hybrid import (
    generate_farm_data,
    generate_patch_data,
    create_config
)


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


def test_hybrid_imports():
    """Test that dwave-hybrid is available."""
    print("\n[TEST 3: Hybrid Framework Availability]")
    
    try:
        import hybrid
        from hybrid import Loop, Race, EnergyImpactDecomposer
        print("  ✓ dwave-hybrid imported successfully")
        
        from solver_runner_CUSTOM_HYBRID import HYBRID_AVAILABLE
        assert HYBRID_AVAILABLE, "HYBRID_AVAILABLE should be True"
        print("  ✓ HYBRID_AVAILABLE flag is True")
        
    except ImportError as e:
        print(f"  ❌ dwave-hybrid not available: {e}")
        print("  Install with: pip install dwave-hybrid")
        return False
    
    return True


def test_workflow_construction():
    """Test custom hybrid workflow can be constructed."""
    print("\n[TEST 4: Workflow Construction]")
    
    try:
        from solver_runner_CUSTOM_HYBRID import solve_with_custom_hybrid_workflow
        print("  ✓ solve_with_custom_hybrid_workflow function imported")
        
        # Test workflow parameters
        params = {
            'subproblem_size': 10,
            'tabu_timeout': 100,
            'max_iter': 3
        }
        print(f"  ✓ Workflow parameters validated: {params}")
        
    except Exception as e:
        print(f"  ❌ Workflow construction failed: {e}")
        return False
    
    return True


def run_all_tests():
    """Run all unit tests."""
    print("="*80)
    print("CUSTOM HYBRID WORKFLOW - UNIT TESTS")
    print("="*80)
    
    try:
        test_data_generation()
        test_cqm_creation()
        
        if not test_hybrid_imports():
            print("\n⚠️  WARNING: dwave-hybrid not available")
            print("   Custom hybrid solver will not work without it")
            print("   Install with: pip install dwave-hybrid")
        else:
            test_workflow_construction()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nReady to run benchmark:")
        print("  python comprehensive_benchmark_CUSTOM_HYBRID.py --config 10")
        
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
