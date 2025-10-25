#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for Quantum/Classical Optimization Comparison

This script runs a comprehensive comparison between:
- Farm Scenario: Full-scale farm optimization (Gurobi, D-Wave CQM)
- Patch Scenario: Smaller patch optimization (Gurobi, D-Wave CQM, Gurobi QUBO, D-Wave BQM)

Total solver configurations tested: 6
1. Farm + Gurobi
2. Farm + D-Wave CQM
3. Patch + Gurobi
4. Patch + D-Wave CQM  
5. Patch + Gurobi QUBO
6. Patch + D-Wave BQM

Plot lines: 9 total (3 D-Wave × 2 timing modes + 3 Gurobi × 1 timing mode)

Results are saved to a comprehensive JSON file for analysis and plotting.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import farm and patch generators
from farm_sampler import generate_farms as generate_farms_large
from patch_sampler import generate_farms as generate_patches_small
from src.scenarios import load_food_data

# Import solvers
from solver_runner_PATCH import (
    create_cqm, solve_with_pulp, solve_with_dwave, 
    solve_with_simulated_annealing, solve_with_gurobi_qubo
)
from dimod import cqm_to_bqm

def generate_sample_data(n_samples: int, seed_offset: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate n_samples of both farms and patches in parallel.
    
    Args:
        n_samples: Number of samples to generate
        seed_offset: Offset for random seed to ensure variety
        
    Returns:
        Tuple of (farms_list, patches_list) with areas
    """
    print(f"\n{'='*80}")
    print(f"GENERATING {n_samples} SAMPLES OF FARMS AND PATCHES")
    print(f"{'='*80}")
    
    farms_list = []
    patches_list = []
    
    def generate_farm_sample(sample_idx):
        """Generate a single farm sample."""
        seed = 42 + seed_offset + sample_idx * 100
        farms = generate_farms_large(n_farms=10 + sample_idx * 2, seed=seed)
        total_area = sum(farms.values())
        return {
            'sample_id': sample_idx,
            'type': 'farm',
            'data': farms,
            'total_area': total_area,
            'n_units': len(farms),
            'seed': seed
        }
    
    def generate_patch_sample(sample_idx):
        """Generate a single patch sample."""
        seed = 42 + seed_offset + sample_idx * 100 + 50
        patches = generate_patches_small(n_farms=15 + sample_idx * 3, seed=seed)
        total_area = sum(patches.values())
        return {
            'sample_id': sample_idx,
            'type': 'patch', 
            'data': patches,
            'total_area': total_area,
            'n_units': len(patches),
            'seed': seed
        }
    
    # Generate samples in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit farm generation tasks
        farm_futures = [executor.submit(generate_farm_sample, i) for i in range(n_samples)]
        patch_futures = [executor.submit(generate_patch_sample, i) for i in range(n_samples)]
        
        # Collect farm results
        print(f"  Generating {n_samples} farm samples...")
        for future in as_completed(farm_futures):
            try:
                result = future.result()
                farms_list.append(result)
                print(f"    ✓ Farm sample {result['sample_id']}: {result['n_units']} farms, {result['total_area']:.1f} ha")
            except Exception as e:
                print(f"    ❌ Farm sample failed: {e}")
        
        # Collect patch results  
        print(f"  Generating {n_samples} patch samples...")
        for future in as_completed(patch_futures):
            try:
                result = future.result()
                patches_list.append(result)
                print(f"    ✓ Patch sample {result['sample_id']}: {result['n_units']} patches, {result['total_area']:.1f} ha")
            except Exception as e:
                print(f"    ❌ Patch sample failed: {e}")
    
    # Sort by sample_id for consistency
    farms_list.sort(key=lambda x: x['sample_id'])
    patches_list.sort(key=lambda x: x['sample_id'])
    
    print(f"\n  ✅ Generated {len(farms_list)} farm samples and {len(patches_list)} patch samples")
    return farms_list, patches_list

def create_food_config(land_data: Dict[str, float], scenario_type: str = 'comprehensive') -> Tuple[Dict, Dict, Dict]:
    """
    Create food and configuration data compatible with solvers.
    
    Args:
        land_data: Dictionary of land availability
        scenario_type: Type of scenario to create
        
    Returns:
        Tuple of (foods, food_groups, config)
    """
    # Load food data from Excel or use fallback
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, "Inputs", "Combined_Food_Data.xlsx")
    
    if os.path.exists(excel_path):
        try:
            foods, food_groups = load_food_data(excel_path)
        except Exception as e:
            print(f"    Warning: Excel loading failed ({e}), using fallback")
            foods, food_groups = create_fallback_foods()
    else:
        foods, food_groups = create_fallback_foods()
    
    # Create configuration
    config = {
        'parameters': {
            'land_availability': land_data,
            'minimum_planting_area': {food: 0.0 for food in foods},
            'max_percentage_per_crop': {food: 0.4 for food in foods},
            'food_group_constraints': {},  # Simplified for comprehensive testing
            'weights': {
                'nutritional_value': 0.25,
                'nutrient_density': 0.2,
                'environmental_impact': 0.25,
                'affordability': 0.15,
                'sustainability': 0.15
            },
            'idle_penalty_lambda': 0.1
        }
    }
    
    return foods, food_groups, config

def create_fallback_foods():
    """Create fallback food data if Excel is not available."""
    foods = {
        'Wheat': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.6, 'affordability': 0.9, 'sustainability': 0.7},
        'Corn': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.5, 'affordability': 0.8, 'sustainability': 0.6},
        'Rice': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.7, 'affordability': 0.7, 'sustainability': 0.8},
        'Soybeans': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.4, 'affordability': 0.6, 'sustainability': 0.9},
        'Potatoes': {'nutritional_value': 0.5, 'nutrient_density': 0.4, 'environmental_impact': 0.8, 'affordability': 0.9, 'sustainability': 0.6}
    }
    food_groups = {
        'grains': ['Wheat', 'Corn', 'Rice'],
        'proteins': ['Soybeans'],
        'vegetables': ['Potatoes']
    }
    return foods, food_groups

def run_farm_scenario(sample_data: Dict, dwave_token: Optional[str] = None) -> Dict:
    """
    Run Farm Scenario: DWaveCQM and PuLP/Gurobi solvers.
    
    Args:
        sample_data: Farm sample data
        dwave_token: D-Wave API token (optional)
        
    Returns:
        Dictionary with results for both solvers
    """
    print(f"\n  🏢 FARM SCENARIO - Sample {sample_data['sample_id']}")
    print(f"     {sample_data['n_units']} farms, {sample_data['total_area']:.1f} ha")
    
    # Create problem setup
    land_data = sample_data['data']
    foods, food_groups, config = create_food_config(land_data, 'farm')
    
    # Create CQM
    cqm_start = time.time()
    cqm, (X, Y), constraint_metadata = create_cqm(land_data, foods, food_groups, config)
    cqm_time = time.time() - cqm_start
    
    results = {
        'sample_id': sample_data['sample_id'],
        'scenario_type': 'farm',
        'n_units': sample_data['n_units'],
        'total_area': sample_data['total_area'],
        'n_foods': len(foods),
        'n_variables': len(cqm.variables),
        'n_constraints': len(cqm.constraints),
        'cqm_time': cqm_time,
        'solvers': {}
    }
    
    # 1. Gurobi Solver
    print(f"     Running Gurobi...")
    try:
        pulp_start = time.time()
        pulp_model, pulp_results = solve_with_pulp(land_data, foods, food_groups, config)
        pulp_time = time.time() - pulp_start
        
        results['solvers']['gurobi'] = {
            'status': pulp_results['status'],
            'objective_value': pulp_results.get('objective_value'),
            'solve_time': pulp_time,
            'solver_time': pulp_results.get('solve_time', pulp_time),
            'success': pulp_results['status'] == 'Optimal'
        }
        print(f"       ✓ Gurobi: {pulp_results['status']} in {pulp_time:.3f}s")
        
    except Exception as e:
        print(f"       ❌ Gurobi failed: {e}")
        results['solvers']['gurobi'] = {'status': 'Error', 'error': str(e), 'success': False}
    
    # 2. DWave CQM Solver (if token available)
    if dwave_token:
        print(f"     Running DWave CQM...")
        try:
            dwave_start = time.time()
            sampleset, qpu_time = solve_with_dwave(cqm, dwave_token)
            dwave_time = time.time() - dwave_start
            
            success = len(sampleset) > 0
            objective_value = None
            if success:
                best = sampleset.first
                # For farm scenario, we can calculate objective from CQM sample
                objective_value = -best.energy  # Assuming energy is negative of objective
            
            results['solvers']['dwave_cqm'] = {
                'status': 'Optimal' if success else 'No solution',
                'objective_value': objective_value,
                'solve_time': dwave_time,
                'qpu_time': qpu_time,
                'hybrid_time': dwave_time - (qpu_time or 0),
                'success': success
            }
            print(f"       ✓ DWave CQM: {'Optimal' if success else 'No solution'} in {dwave_time:.3f}s")
            
        except Exception as e:
            print(f"       ❌ DWave CQM failed: {e}")
            results['solvers']['dwave_cqm'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     DWave CQM: SKIPPED (no token)")
        results['solvers']['dwave_cqm'] = {'status': 'Skipped', 'success': False}
    
    return results

def run_patch_scenario(sample_data: Dict, dwave_token: Optional[str] = None) -> Dict:
    """
    Run Patch Scenario: PuLP, DWaveCQM, DWaveBQM, and Gurobi QUBO solvers.
    
    Args:
        sample_data: Patch sample data  
        dwave_token: D-Wave API token (optional)
        
    Returns:
        Dictionary with results for all solvers
    """
    print(f"\n  🏞️  PATCH SCENARIO - Sample {sample_data['sample_id']}")
    print(f"     {sample_data['n_units']} patches, {sample_data['total_area']:.1f} ha")
    
    # Create problem setup
    land_data = sample_data['data']
    foods, food_groups, config = create_food_config(land_data, 'patch')
    
    # Create CQM
    cqm_start = time.time()
    cqm, (X, Y), constraint_metadata = create_cqm(land_data, foods, food_groups, config)
    cqm_time = time.time() - cqm_start
    
    results = {
        'sample_id': sample_data['sample_id'],
        'scenario_type': 'patch',
        'n_units': sample_data['n_units'],
        'total_area': sample_data['total_area'],
        'n_foods': len(foods),
        'n_variables': len(cqm.variables),
        'n_constraints': len(cqm.constraints),
        'cqm_time': cqm_time,
        'solvers': {}
    }
    
    # 1. Gurobi Solver
    print(f"     Running Gurobi...")
    try:
        pulp_start = time.time()
        pulp_model, pulp_results = solve_with_pulp(land_data, foods, food_groups, config)
        pulp_time = time.time() - pulp_start
        
        results['solvers']['gurobi'] = {
            'status': pulp_results['status'],
            'objective_value': pulp_results.get('objective_value'),
            'solve_time': pulp_time,
            'solver_time': pulp_results.get('solve_time', pulp_time),
            'success': pulp_results['status'] == 'Optimal'
        }
        print(f"       ✓ Gurobi: {pulp_results['status']} in {pulp_time:.3f}s")
        
    except Exception as e:
        print(f"       ❌ Gurobi failed: {e}")
        results['solvers']['gurobi'] = {'status': 'Error', 'error': str(e), 'success': False}
    
    # 2. DWave CQM Solver (if token available)
    if dwave_token:
        print(f"     Running DWave CQM...")
        try:
            dwave_cqm_start = time.time()
            sampleset_cqm, qpu_time_cqm = solve_with_dwave(cqm, dwave_token)
            dwave_cqm_time = time.time() - dwave_cqm_start
            
            success = len(sampleset_cqm) > 0
            objective_value = None
            if success:
                best = sampleset_cqm.first
                objective_value = -best.energy
            
            results['solvers']['dwave_cqm'] = {
                'status': 'Optimal' if success else 'No solution',
                'objective_value': objective_value,
                'solve_time': dwave_cqm_time,
                'qpu_time': qpu_time_cqm,
                'hybrid_time': dwave_cqm_time - (qpu_time_cqm or 0),
                'success': success
            }
            print(f"       ✓ DWave CQM: {'Optimal' if success else 'No solution'} in {dwave_cqm_time:.3f}s")
            
        except Exception as e:
            print(f"       ❌ DWave CQM failed: {e}")
            results['solvers']['dwave_cqm'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     DWave CQM: SKIPPED (no token)")
        results['solvers']['dwave_cqm'] = {'status': 'Skipped', 'success': False}
    
    # Convert CQM to BQM for BQM-based solvers
    bqm = None
    invert = None
    bqm_conversion_time = None
    
    if dwave_token:  # Only convert if we'll use BQM solvers
        print(f"     Converting CQM to BQM...")
        try:
            bqm_start = time.time()
            bqm, invert = cqm_to_bqm(cqm)
            bqm_conversion_time = time.time() - bqm_start
            print(f"       ✓ BQM: {len(bqm.variables)} vars, {len(bqm.quadratic)} interactions ({bqm_conversion_time:.3f}s)")
        except Exception as e:
            print(f"       ❌ BQM conversion failed: {e}")
    
    # 3. DWave BQM Solver (if BQM available)
    if bqm is not None and dwave_token:
        print(f"     Running DWave BQM...")
        try:
            from dwave.system import LeapHybridBQMSampler
            sampler = LeapHybridBQMSampler(token=dwave_token)
            
            dwave_bqm_start = time.time()
            sampleset_bqm = sampler.sample(bqm, label="Comprehensive Benchmark - BQM")
            dwave_bqm_time = time.time() - dwave_bqm_start
            
            success = len(sampleset_bqm) > 0
            objective_value = None
            qpu_time_bqm = None
            
            if success:
                best = sampleset_bqm.first
                objective_value = -best.energy
                timing_info = sampleset_bqm.info.get('timing', {})
                qpu_time_bqm = timing_info.get('qpu_access_time')
                if qpu_time_bqm:
                    qpu_time_bqm = qpu_time_bqm / 1e6  # Convert to seconds
            
            results['solvers']['dwave_bqm'] = {
                'status': 'Optimal' if success else 'No solution',
                'objective_value': objective_value,
                'solve_time': dwave_bqm_time,
                'qpu_time': qpu_time_bqm,
                'hybrid_time': dwave_bqm_time - (qpu_time_bqm or 0),
                'bqm_conversion_time': bqm_conversion_time,
                'success': success
            }
            print(f"       ✓ DWave BQM: {'Optimal' if success else 'No solution'} in {dwave_bqm_time:.3f}s")
            
        except Exception as e:
            print(f"       ❌ DWave BQM failed: {e}")
            results['solvers']['dwave_bqm'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     DWave BQM: SKIPPED (no BQM available)")
        results['solvers']['dwave_bqm'] = {'status': 'Skipped', 'success': False}
    
    # 4. Gurobi QUBO Solver (if BQM available)
    if bqm is not None:
        print(f"     Running Gurobi QUBO...")
        try:
            gurobi_result = solve_with_gurobi_qubo(bqm)
            
            results['solvers']['gurobi_qubo'] = {
                'status': gurobi_result['status'],
                'objective_value': gurobi_result['objective_value'],
                'solve_time': gurobi_result['solve_time'],
                'bqm_energy': gurobi_result['bqm_energy'],
                'bqm_conversion_time': bqm_conversion_time,
                'success': gurobi_result['status'] == 'Optimal'
            }
            print(f"       ✓ Gurobi QUBO: {gurobi_result['status']} in {gurobi_result['solve_time']:.3f}s")
            
        except Exception as e:
            print(f"       ❌ Gurobi QUBO failed: {e}")
            results['solvers']['gurobi_qubo'] = {'status': 'Error', 'error': str(e), 'success': False}
    else:
        print(f"     Gurobi QUBO: SKIPPED (no BQM available)")
        results['solvers']['gurobi_qubo'] = {'status': 'Skipped', 'success': False}
    
    return results

def run_comprehensive_benchmark(n_samples: int, dwave_token: Optional[str] = None) -> Dict:
    """
    Run the comprehensive benchmark with n_samples for both scenarios.
    
    Args:
        n_samples: Number of samples to test
        dwave_token: D-Wave API token (optional)
        
    Returns:
        Complete benchmark results dictionary
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE BENCHMARK - {n_samples} SAMPLES")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Generate sample data
    farms_list, patches_list = generate_sample_data(n_samples)
    
    # Run Farm Scenarios
    print(f"\n{'='*80}")
    print(f"FARM SCENARIOS ({len(farms_list)} samples)")
    print(f"{'='*80}")
    
    farm_results = []
    for farm_sample in farms_list:
        try:
            result = run_farm_scenario(farm_sample, dwave_token)
            farm_results.append(result)
        except Exception as e:
            print(f"  ❌ Farm sample {farm_sample['sample_id']} failed: {e}")
    
    # Run Patch Scenarios  
    print(f"\n{'='*80}")
    print(f"PATCH SCENARIOS ({len(patches_list)} samples)")
    print(f"{'='*80}")
    
    patch_results = []
    for patch_sample in patches_list:
        try:
            result = run_patch_scenario(patch_sample, dwave_token)
            patch_results.append(result)
        except Exception as e:
            print(f"  ❌ Patch sample {patch_sample['sample_id']} failed: {e}")
    
    total_time = time.time() - start_time
    
    # Compile comprehensive results
    benchmark_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_samples': n_samples,
            'total_runtime': total_time,
            'dwave_enabled': dwave_token is not None,
            'scenarios': ['farm', 'patch'],
            'solvers': {
                'farm': ['gurobi', 'dwave_cqm'],
                'patch': ['gurobi', 'dwave_cqm', 'gurobi_qubo', 'dwave_bqm']
            }
        },
        'farm_results': farm_results,
        'patch_results': patch_results,
        'summary': {
            'farm_samples_completed': len(farm_results),
            'patch_samples_completed': len(patch_results),
            'total_solver_runs': sum(len(r['solvers']) for r in farm_results + patch_results)
        }
    }
    
    return benchmark_results

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive benchmark comparing quantum and classical solvers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python comprehensive_benchmark.py 5               # 5 samples, no D-Wave
  python comprehensive_benchmark.py 10 --dwave      # 10 samples with D-Wave
  python comprehensive_benchmark.py 3 --output my_results.json
        '''
    )
    
    parser.add_argument('n_samples', type=int, 
                       help='Number of samples to generate for each scenario')
    parser.add_argument('--dwave', action='store_true',
                       help='Enable D-Wave solvers (requires DWAVE_API_TOKEN)')
    parser.add_argument('--output', type=str, 
                       default=None,
                       help='Output JSON filename (default: auto-generated)')
    parser.add_argument('--token', type=str,
                       help='D-Wave API token (overrides environment variable)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.n_samples < 1:
        print("Error: n_samples must be at least 1")
        sys.exit(1)
    
    if args.n_samples > 50:
        print("Warning: Large number of samples may take significant time and D-Wave budget")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Set up D-Wave token
    dwave_token = None
    if args.dwave:
        if args.token:
            dwave_token = args.token
        else:
            dwave_token = os.getenv('DWAVE_API_TOKEN')
        
        if not dwave_token:
            print("Warning: D-Wave enabled but no token found. Set DWAVE_API_TOKEN environment variable or use --token")
            print("Continuing with classical solvers only...")
        else:
            print(f"✓ D-Wave token configured (length: {len(dwave_token)})")
    
    # Generate output filename
    if args.output:
        output_filename = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dwave_suffix = "_dwave" if dwave_token else "_classical"
        output_filename = f"comprehensive_benchmark_{args.n_samples}samples{dwave_suffix}_{timestamp}.json"
    
    print(f"\nRunning comprehensive benchmark:")
    print(f"  Samples: {args.n_samples}")
    print(f"  D-Wave: {'Enabled' if dwave_token else 'Disabled'}")
    print(f"  Output: {output_filename}")
    
    # Run benchmark
    try:
        results = run_comprehensive_benchmark(args.n_samples, dwave_token)
        
        # Save results
        print(f"\n{'='*80}")
        print("SAVING RESULTS")
        print(f"{'='*80}")
        
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {output_filename}")
        print(f"   Total runtime: {results['metadata']['total_runtime']:.1f} seconds")
        print(f"   Farm samples: {results['summary']['farm_samples_completed']}")
        print(f"   Patch samples: {results['summary']['patch_samples_completed']}")
        print(f"   Total solver runs: {results['summary']['total_solver_runs']}")
        
        print(f"\nNext steps:")
        print(f"  1. Run plotting: python plot_comprehensive_results.py {output_filename}")
        print(f"  2. Analyze results in the JSON file")
        
    except KeyboardInterrupt:
        print(f"\n\n❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()