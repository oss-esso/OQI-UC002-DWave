#!/usr/bin/env python3
"""
Extract QPU benchmark data and convert to plottable format.
"""
import json
import pandas as pd
from pathlib import Path
import glob

# Directories
qpu_results_dir = Path(__file__).parent / 'qpu_benchmark_results'
output_dir = Path(__file__).parent / 'hardness_analysis_results'

def extract_roadmap_data():
    """Extract GUROBI data from roadmap JSON files - combine all phases into one dataset."""
    roadmap_files = sorted(glob.glob(str(qpu_results_dir / 'roadmap_*.json')))
    
    all_results = []
    
    for file_path in roadmap_files:
        print(f"Processing {Path(file_path).name}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        phase = data.get('phase', 0)
        
        for result in data.get('results', []):
            method = result.get('method', '')
            
            # ONLY extract Gurobi results
            if 'gurobi' not in method.lower():
                continue
            
            # Extract timing info
            timings = result.get('timings', {})
            if isinstance(timings, str):
                continue
            
            # Get n_farms from scale or n_farms field
            n_farms = result.get('n_farms', result.get('scale', 0))
            n_foods = result.get('n_foods', 6)  # Default for rotation
            
            # Get solve time
            solve_time = timings.get('solve', timings.get('solve_time', result.get('solve_time', 0)))
            if solve_time is None:
                solve_time = result.get('solve_time', 0)
            
            # Get wall time
            wall_time = result.get('wall_time', result.get('total_time', solve_time))
            
            all_results.append({
                'source': f'roadmap_phase{phase}',
                'scenario': result.get('test', f'Phase{phase}_scale{n_farms}'),
                'method': method,
                'n_farms': n_farms,
                'n_vars': result.get('n_variables', 0),
                'n_foods': n_foods,
                'solve_time': solve_time,
                'build_time': timings.get('build', 0),
                'obj_value': result.get('objective', 0),
                'gap': 0,
                'n_quadratic': 0,
                'n_constraints': result.get('n_constraints', 0),
                'status': str(result.get('status', 'OPTIMAL')).upper(),
                'time_category': categorize_time(solve_time),
                'farms_per_food': n_farms / max(n_foods, 1),
                'total_area': n_farms * 1.0,
                'test_type': 'Roadmap Gurobi'
            })
    
    return all_results

def extract_test_output_files():
    """Extract GUROBI data from hierarchical and statistical test JSON files."""
    results = []
    base_dir = Path(__file__).parent
    
    # Parse statistical comparison results
    statistical_dir = base_dir / 'statistical_comparison_results'
    statistical_files = sorted(glob.glob(str(statistical_dir / 'statistical_comparison_*.json')))
    
    if statistical_files:
        print(f"Parsing {len(statistical_files)} statistical comparison files...")
        for sfile in statistical_files[-1:]:  # Get latest
            with open(sfile, 'r') as f:
                data = json.load(f)
            
            # Extract ground_truth (Gurobi) results for each size
            for size_key, size_data in data.get('results_by_size', {}).items():
                n_farms = size_data.get('n_farms', int(size_key))
                n_vars = size_data.get('n_variables', 0)
                
                ground_truth = size_data.get('methods', {}).get('ground_truth', {})
                runs = ground_truth.get('runs', [])
                
                for run in runs:
                    if not run.get('success', False):
                        continue
                    
                    results.append({
                        'source': 'statistical_comparison',
                        'scenario': f'statistical_{n_farms}farms',
                        'method': 'gurobi',
                        'n_farms': n_farms,
                        'n_vars': n_vars,
                        'n_foods': 6,
                        'solve_time': run.get('solve_time', run.get('wall_time', 0)),
                        'build_time': 0,
                        'obj_value': run.get('objective', 0),
                        'gap': run.get('gap', run.get('mip_gap', 0)),
                        'n_quadratic': 0,
                        'n_constraints': 0,
                        'status': 'OPTIMAL' if run.get('optimal', False) else 'SUBOPTIMAL',
                        'time_category': categorize_time(run.get('solve_time', run.get('wall_time', 0))),
                        'farms_per_food': n_farms / 6,
                        'total_area': n_farms * 1.0,
                        'test_type': 'Statistical Gurobi'
                    })
    
    # Parse hierarchical test results
    hierarchical_dir = base_dir / 'hierarchical_statistical_results'
    hierarchical_files = sorted(glob.glob(str(hierarchical_dir / 'hierarchical_results_*.json')))
    
    if hierarchical_files:
        print(f"Parsing {len(hierarchical_files)} hierarchical test files...")
        for hfile in hierarchical_files[-1:]:  # Get latest
            with open(hfile, 'r') as f:
                data = json.load(f)
            
            # Hierarchical JSON structure: {farm_size: {gurobi: [...], hierarchical_qpu: [...]}}
            for size_key, size_data in data.items():
                if not isinstance(size_data, dict):
                    continue
                
                data_info = size_data.get('data_info', {})
                n_farms = data_info.get('n_farms', int(size_key))
                n_vars = data_info.get('n_variables', 0)
                
                gurobi_runs = size_data.get('gurobi', [])
                
                for run in gurobi_runs:
                    if not run.get('success', False):
                        continue
                    
                    results.append({
                        'source': 'hierarchical_test',
                        'scenario': f'hierarchical_{n_farms}farms',
                        'method': 'gurobi',
                        'n_farms': n_farms,
                        'n_vars': n_vars,
                        'n_foods': 6,
                        'solve_time': run.get('solve_time', 0),
                        'build_time': 0,
                        'obj_value': run.get('objective', 0),
                        'gap': run.get('gap', 0),
                        'n_quadratic': 0,
                        'n_constraints': 0,
                        'status': 'OPTIMAL' if run.get('gap', 0) < 0.01 else 'SUBOPTIMAL',
                        'time_category': categorize_time(run.get('solve_time', 0)),
                        'farms_per_food': n_farms / 6,
                        'total_area': n_farms * 1.0,
                        'test_type': 'Hierarchical Gurobi'
                    })
    
    return results

def categorize_time(solve_time):
    """Categorize solve time."""
    if solve_time < 10:
        return 'FAST'
    elif solve_time < 100:
        return 'MEDIUM'
    else:
        return 'TIMEOUT'

def main():
    print("="*80)
    print("EXTRACTING GUROBI BENCHMARK DATA FROM ALL SOURCES")
    print("="*80)
    
    # Extract roadmap GUROBI data (combine all phases)
    print("\n[1/2] Extracting Roadmap GUROBI data (all phases combined)...")
    roadmap_data = extract_roadmap_data()
    print(f"  Found {len(roadmap_data)} roadmap Gurobi results")
    
    # Extract hierarchical and statistical GUROBI data
    print("\n[2/2] Checking for hierarchical and statistical GUROBI data...")
    other_data = extract_test_output_files()
    print(f"  Found {len(other_data)} other Gurobi results")
    
    # Combine all
    all_data = roadmap_data + other_data
    
    if not all_data:
        print("\nNo data found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    output_file = output_dir / 'additional_gurobi_results.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved {len(df)} Gurobi results to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY BY TEST TYPE")
    print("="*80)
    for test_type in sorted(df['test_type'].unique()):
        subset = df[df['test_type'] == test_type]
        print(f"\n{test_type}:")
        print(f"  Count: {len(subset)}")
        if len(subset) > 0:
            print(f"  Farm range: {subset['n_farms'].min()}-{subset['n_farms'].max()}")
            print(f"  Solve time range: {subset['solve_time'].min():.2f}-{subset['solve_time'].max():.2f}s")
            print(f"  Time categories:")
            for cat in ['FAST', 'MEDIUM', 'SLOW', 'TIMEOUT']:
                count = len(subset[subset['time_category'] == cat])
                if count > 0:
                    print(f"    {cat}: {count}")

if __name__ == '__main__':
    main()
