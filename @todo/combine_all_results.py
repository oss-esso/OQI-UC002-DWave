#!/usr/bin/env python3
"""
Combine all benchmark results into unified comprehensive analysis.
Includes: hardness analysis, roadmap phases, statistical tests, hierarchical tests.
"""
import json
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List

def parse_roadmap_json(filepath: Path) -> List[Dict]:
    """Parse roadmap phase JSON results."""
    results = []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        phase = data.get('phase', 'unknown')
        for test in data.get('tests', []):
            scenario = test.get('scenario', '')
            n_vars = test.get('problem_size', {}).get('n_variables', 0)
            n_farms = test.get('problem_size', {}).get('n_farms', 0)
            n_foods = test.get('problem_size', {}).get('n_foods', 0)
            
            for method_name, method_data in test.get('results', {}).items():
                if method_data and method_data.get('status') == 'success':
                    results.append({
                        'source': f'roadmap_phase{phase}',
                        'scenario': scenario,
                        'method': method_name,
                        'n_farms': n_farms,
                        'n_vars': n_vars,
                        'n_foods': n_foods,
                        'solve_time': method_data.get('wall_time', 0),
                        'qpu_time': method_data.get('qpu_time', 0),
                        'embed_time': method_data.get('embedding_time', 0),
                        'obj_value': method_data.get('objective', None),
                        'violations': method_data.get('violations', 0),
                        'gap': method_data.get('gap', None),
                        'status': 'success'
                    })
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    
    return results

def parse_statistical_output(filepath: Path) -> List[Dict]:
    """Parse statistical test output."""
    results = []
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
        
        # Extract test blocks
        test_pattern = r'TESTING: (\d+) farms.*?Variables: (\d+)'
        method_pattern = r'--- Method: (\w+) ---.*?Run \d+/\d+.*?✓ Success: obj=([\d.]+), wall=([\d.]+)s'
        
        for test_match in re.finditer(test_pattern, content, re.DOTALL):
            n_farms = int(test_match.group(1))
            n_vars = int(test_match.group(2))
            test_block = content[test_match.start():test_match.start() + 5000]  # Next 5000 chars
            
            for method_match in re.finditer(method_pattern, test_block):
                method = method_match.group(1)
                obj = float(method_match.group(2))
                wall_time = float(method_match.group(3))
                
                results.append({
                    'source': 'statistical_test',
                    'scenario': f'{n_farms}farms',
                    'method': method,
                    'n_farms': n_farms,
                    'n_vars': n_vars,
                    'n_foods': 6,  # Assumed
                    'solve_time': wall_time,
                    'obj_value': obj,
                    'status': 'success'
                })
    except Exception as e:
        print(f"Error parsing statistical test: {e}")
    
    return results

def parse_hierarchical_output(filepath: Path) -> List[Dict]:
    """Parse hierarchical statistical test output."""
    results = []
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
        
        # Similar parsing logic
        test_pattern = r'Run \d+/\d+: (\d+) farms.*?obj=([\d.]+).*?time=([\d.]+)s'
        
        for match in re.finditer(test_pattern, content):
            n_farms = int(match.group(1))
            obj = float(match.group(2))
            time = float(match.group(3))
            
            results.append({
                'source': 'hierarchical_test',
                'scenario': f'{n_farms}farms',
                'method': 'hierarchical_decomp',
                'n_farms': n_farms,
                'n_vars': n_farms * 6 * 3,  # Estimated
                'n_foods': 6,
                'solve_time': time,
                'obj_value': obj,
                'status': 'success'
            })
    except Exception as e:
        print(f"Error parsing hierarchical test: {e}")
    
    return results

def main():
    base_dir = Path(__file__).parent
    results_dir = base_dir / 'hardness_analysis_results'
    qpu_dir = base_dir / 'qpu_benchmark_results'
    
    all_results = []
    
    print("="*80)
    print("COLLECTING ALL BENCHMARK RESULTS")
    print("="*80)
    
    # 1. Load hardness analysis (Gurobi only)
    print("\n[1/5] Loading hardness analysis (constant area per farm)...")
    try:
        df_hardness = pd.read_csv(results_dir / 'hardness_analysis_results.csv')
        for _, row in df_hardness.iterrows():
            all_results.append({
                'source': 'hardness_analysis',
                'scenario': f'{int(row["n_farms"])}farms_100ha_total',
                'method': 'gurobi',
                'n_farms': int(row['n_farms']),
                'n_vars': int(row['n_vars']),
                'n_foods': 6,
                'solve_time': row['solve_time'],
                'build_time': row.get('build_time', 0),
                'obj_value': row.get('obj_value', None),
                'gap': row.get('gap', None),
                'n_quadratic': int(row.get('n_quadratic', 0)),
                'n_constraints': int(row.get('n_constraints', 0)),
                'status': row['status'],
                'time_category': row['time_category']
            })
        print(f"  ✓ Loaded {len(df_hardness)} hardness test points")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # 2. Load roadmap phases
    print("\n[2/5] Loading roadmap phase results...")
    roadmap_files = sorted(qpu_dir.glob('roadmap_phase*.json'))
    for filepath in roadmap_files:
        phase_results = parse_roadmap_json(filepath)
        all_results.extend(phase_results)
        print(f"  ✓ {filepath.name}: {len(phase_results)} test cases")
    
    # 3. Load statistical tests
    print("\n[3/5] Loading statistical test results...")
    stat_file = base_dir / 'statistical_test_output.txt'
    if stat_file.exists():
        stat_results = parse_statistical_output(stat_file)
        all_results.extend(stat_results)
        print(f"  ✓ Statistical tests: {len(stat_results)} runs")
    else:
        print(f"  ⚠ File not found: {stat_file}")
    
    # 4. Load hierarchical tests
    print("\n[4/5] Loading hierarchical test results...")
    hier_file = base_dir / 'hierarchical_statistical_output.txt'
    if hier_file.exists():
        hier_results = parse_hierarchical_output(hier_file)
        all_results.extend(hier_results)
        print(f"  ✓ Hierarchical tests: {len(hier_results)} runs")
    else:
        print(f"  ⚠ File not found: {hier_file}")
    
    # 5. Combine and save
    print("\n[5/5] Combining results...")
    df_all = pd.DataFrame(all_results)
    
    # Add derived columns
    if 'n_farms' in df_all.columns and 'n_foods' in df_all.columns:
        df_all['farms_per_food'] = df_all['n_farms'] / df_all['n_foods']
    
    # Save combined results
    output_csv = results_dir / 'combined_all_results.csv'
    df_all.to_csv(output_csv, index=False)
    print(f"\n✓ Combined results saved: {output_csv}")
    print(f"  Total: {len(df_all)} test cases")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nBy source:")
    print(df_all.groupby('source').size())
    print(f"\nBy method:")
    print(df_all.groupby('method').size())
    print(f"\nProblem sizes:")
    print(f"  Farms: {df_all['n_farms'].min()}-{df_all['n_farms'].max()}")
    print(f"  Variables: {df_all['n_vars'].min()}-{df_all['n_vars'].max()}")
    
    return df_all

if __name__ == '__main__':
    df = main()
