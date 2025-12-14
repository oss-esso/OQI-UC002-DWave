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
    """Extract data from roadmap JSON files."""
    roadmap_files = sorted(glob.glob(str(qpu_results_dir / 'roadmap_*.json')))
    
    all_results = []
    
    for file_path in roadmap_files:
        print(f"Processing {Path(file_path).name}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        phase = data.get('phase', 0)
        
        for result in data.get('results', []):
            method = result.get('method', '')
            
            # Skip Gurobi results (we already have those)
            if 'gurobi' in method.lower():
                continue
            
            # Extract timing info
            timings = result.get('timings', {})
            if isinstance(timings, str):
                # Parse if it's a string representation
                continue
            
            all_results.append({
                'source': f'roadmap_phase{phase}',
                'scenario': result.get('test', ''),
                'method': method,
                'n_farms': result.get('n_farms', 0),
                'n_vars': result.get('n_variables', 0),
                'n_foods': result.get('n_foods', 0),
                'solve_time': timings.get('solve_time', result.get('solve_time', 0)),
                'build_time': timings.get('build', 0),
                'qpu_time': timings.get('qpu_access_total', 0),
                'embedding_time': timings.get('embedding_total', 0),
                'obj_value': result.get('objective', 0),
                'gap': 0,  # QPU doesn't have MIP gap
                'n_quadratic': result.get('bqm_interactions', 0),
                'n_constraints': result.get('n_constraints', 0),
                'status': 'OPTIMAL' if result.get('feasible', False) else 'INFEASIBLE',
                'time_category': categorize_time(timings.get('solve_time', result.get('solve_time', 0))),
                'farms_per_food': result.get('n_farms', 0) / max(result.get('n_foods', 1), 1),
                'total_area': result.get('n_farms', 0) * 1.0,  # Assume 1 ha/farm
                'test_type': 'Roadmap QPU'
            })
    
    return all_results

def extract_qpu_benchmark_data():
    """Extract data from general QPU benchmark JSON files."""
    qpu_files = sorted(glob.glob(str(qpu_results_dir / 'qpu_benchmark_*.json')))
    
    all_results = []
    
    for file_path in qpu_files[-5:]:  # Get latest 5 files
        print(f"Processing {Path(file_path).name}...")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if it has results
            if 'results' not in data:
                continue
            
            for result in data.get('results', []):
                method = result.get('method', '')
                
                # Skip Gurobi
                if 'gurobi' in method.lower():
                    continue
                
                # Determine test type based on method
                if 'hierarchical' in method.lower():
                    test_type = 'Hierarchical QPU'
                elif 'statistical' in method.lower() or 'comparison' in result.get('scenario', '').lower():
                    test_type = 'Statistical QPU'
                else:
                    test_type = 'QPU Benchmark'
                
                timings = result.get('timings', {})
                
                all_results.append({
                    'source': 'qpu_benchmark',
                    'scenario': result.get('scenario', result.get('test', '')),
                    'method': method,
                    'n_farms': result.get('n_farms', 0),
                    'n_vars': result.get('n_variables', 0),
                    'n_foods': result.get('n_foods', 0),
                    'solve_time': timings.get('solve_time', result.get('solve_time', 0)),
                    'build_time': timings.get('build', 0),
                    'qpu_time': timings.get('qpu_access_total', 0),
                    'embedding_time': timings.get('embedding_total', 0),
                    'obj_value': result.get('objective', 0),
                    'gap': 0,
                    'n_quadratic': result.get('bqm_interactions', 0),
                    'n_constraints': result.get('n_constraints', 0),
                    'status': 'OPTIMAL' if result.get('feasible', False) else 'INFEASIBLE',
                    'time_category': categorize_time(timings.get('solve_time', result.get('solve_time', 0))),
                    'farms_per_food': result.get('n_farms', 0) / max(result.get('n_foods', 1), 1),
                    'total_area': result.get('n_farms', 0) * 1.0,
                    'test_type': test_type
                })
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
    
    return all_results

def categorize_time(solve_time):
    """Categorize solve time."""
    if solve_time < 10:
        return 'FAST'
    elif solve_time < 100:
        return 'MEDIUM'
    elif solve_time < 300:
        return 'SLOW'
    else:
        return 'TIMEOUT'

def main():
    print("="*80)
    print("EXTRACTING QPU BENCHMARK DATA")
    print("="*80)
    
    # Extract roadmap data
    print("\n[1/2] Extracting Roadmap data...")
    roadmap_data = extract_roadmap_data()
    print(f"  Found {len(roadmap_data)} roadmap results")
    
    # Extract general QPU benchmark data
    print("\n[2/2] Extracting QPU benchmark data...")
    qpu_data = extract_qpu_benchmark_data()
    print(f"  Found {len(qpu_data)} QPU benchmark results")
    
    # Combine all
    all_data = roadmap_data + qpu_data
    
    if not all_data:
        print("\nNo QPU data found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    output_file = output_dir / 'qpu_results_integrated.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved {len(df)} QPU results to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY BY TEST TYPE")
    print("="*80)
    for test_type in df['test_type'].unique():
        subset = df[df['test_type'] == test_type]
        print(f"\n{test_type}:")
        print(f"  Count: {len(subset)}")
        print(f"  Farm range: {subset['n_farms'].min()}-{subset['n_farms'].max()}")
        print(f"  Solve time: {subset['solve_time'].min():.2f}-{subset['solve_time'].max():.2f}s")
        print(f"  Methods: {', '.join(subset['method'].unique())}")

if __name__ == '__main__':
    main()
