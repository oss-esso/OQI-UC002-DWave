#!/usr/bin/env python3
"""
Extract and consolidate significant scenarios with QPU results from all benchmark files.

Based on hardness analysis findings:
- Critical threshold at ~10 farms
- Structure (rotation + diversity) matters more than scale
- Phase 1 (simple binary) is tractable, Phase 2+ are hard
- Hierarchical QPU shows promise at scale

This script:
1. Identifies the most significant scenarios
2. Extracts existing Gurobi AND QPU results
3. Creates a consolidated comparison file
"""

import json
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'significant_scenarios'
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("EXTRACTING SIGNIFICANT SCENARIOS WITH QPU RESULTS")
print("="*80)

# ============================================================================
# SIGNIFICANT SCENARIOS DEFINITIONS (based on hardness analysis)
# ============================================================================

SIGNIFICANT_SCENARIOS = {
    # Category 1: "The 10-Farm Cliff" - transition zone
    'cliff_easy_4farms': {
        'description': 'Simple binary (4 farms) - should be FAST',
        'n_farms': 4,
        'n_foods': 6,
        'n_periods': 3,
        'expected_behavior': 'FAST',
        'hardness_reason': 'Below cliff threshold, simple formulation'
    },
    'cliff_transition_10farms': {
        'description': 'Rotation with 10 farms - RIGHT at the cliff',
        'n_farms': 10,
        'n_foods': 6,
        'n_periods': 3,
        'expected_behavior': 'VARIABLE (can be fast or timeout)',
        'hardness_reason': 'Critical threshold - formulation determines outcome'
    },
    'cliff_hard_15farms': {
        'description': 'Rotation with 15 farms - past the cliff',
        'n_farms': 15,
        'n_foods': 6,
        'n_periods': 3,
        'expected_behavior': 'TIMEOUT',
        'hardness_reason': 'Above cliff, rotation constraints make it hard'
    },
    
    # Category 2: Scale comparison (QPU advantage zone)
    'scale_small_5farms': {
        'description': '5 farms with full constraints',
        'n_farms': 5,
        'n_foods': 6,
        'n_periods': 3,
        'expected_behavior': 'TIMEOUT (with diversity+rotation)',
        'hardness_reason': 'Small but constrained'
    },
    'scale_medium_20farms': {
        'description': '20 farms with full constraints',
        'n_farms': 20,
        'n_foods': 6,
        'n_periods': 3,
        'expected_behavior': 'TIMEOUT',
        'hardness_reason': 'Medium scale, classical struggles'
    },
    'scale_large_25farms': {
        'description': '25 farms - D-Wave hierarchical test',
        'n_farms': 25,
        'n_foods': 27,
        'n_periods': 3,
        'expected_behavior': 'TIMEOUT for Gurobi, feasible for QPU',
        'hardness_reason': 'Large scale with aggregation'
    },
    'scale_xlarge_50farms': {
        'description': '50 farms - D-Wave hierarchical test',
        'n_farms': 50,
        'n_foods': 27,
        'n_periods': 3,
        'expected_behavior': 'TIMEOUT for Gurobi, feasible for QPU',
        'hardness_reason': 'Very large, classical intractable'
    },
    'scale_xxlarge_100farms': {
        'description': '100 farms - ultimate test',
        'n_farms': 100,
        'n_foods': 27,
        'n_periods': 3,
        'expected_behavior': 'TIMEOUT for Gurobi',
        'hardness_reason': 'Maximum scale tested'
    },
}

# ============================================================================
# EXTRACT RESULTS FROM EXISTING FILES
# ============================================================================

all_results = []

# --- 1. Extract from Statistical Comparison files ---
print("\n[1] Extracting from statistical comparison files...")
stat_files = sorted(glob.glob(str(BASE_DIR / 'statistical_comparison_results' / '*.json')))

for fpath in stat_files:
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
        
        for size_key, size_data in data.get('results_by_size', {}).items():
            n_farms = size_data.get('n_farms', int(size_key))
            n_vars = size_data.get('n_variables', 0)
            
            # Gurobi (ground_truth)
            gt = size_data.get('methods', {}).get('ground_truth', {})
            for run in gt.get('runs', []):
                if run.get('success'):
                    all_results.append({
                        'source_file': Path(fpath).name,
                        'scenario_id': f'statistical_{n_farms}farms',
                        'method': 'gurobi',
                        'n_farms': n_farms,
                        'n_foods': 6,
                        'n_vars': n_vars,
                        'solve_time': run.get('solve_time', run.get('wall_time', 0)),
                        'objective': run.get('objective', 0),
                        'gap': run.get('gap', run.get('mip_gap', 0)),
                        'success': run.get('success', False),
                        'qpu_time': 0,
                        'embedding_time': 0,
                    })
            
            # QPU methods
            for method_name in ['clique_decomp', 'spatial_temporal']:
                method_data = size_data.get('methods', {}).get(method_name, {})
                for run in method_data.get('runs', []):
                    if run.get('success'):
                        all_results.append({
                            'source_file': Path(fpath).name,
                            'scenario_id': f'statistical_{n_farms}farms',
                            'method': method_name,
                            'n_farms': n_farms,
                            'n_foods': 6,
                            'n_vars': n_vars,
                            'solve_time': run.get('wall_time', 0),
                            'objective': run.get('objective', 0),
                            'gap': 0,
                            'success': run.get('success', False),
                            'qpu_time': run.get('qpu_time', run.get('solve_time', 0)),
                            'embedding_time': run.get('embedding_time', 0),
                        })
    except Exception as e:
        print(f"  Error parsing {fpath}: {e}")

print(f"  Found {len(all_results)} results from statistical files")

# --- 2. Extract from Hierarchical files ---
print("\n[2] Extracting from hierarchical statistical results...")
hier_files = sorted(glob.glob(str(BASE_DIR / 'hierarchical_statistical_results' / '*.json')))

for fpath in hier_files:
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
        
        for size_key, size_data in data.items():
            if not isinstance(size_data, dict):
                continue
            
            data_info = size_data.get('data_info', {})
            n_farms = data_info.get('n_farms', int(size_key))
            n_vars = data_info.get('n_variables', 0)
            n_foods = data_info.get('n_foods', 27)
            
            # Gurobi results
            for run in size_data.get('gurobi', []):
                if run.get('success'):
                    all_results.append({
                        'source_file': Path(fpath).name,
                        'scenario_id': f'hierarchical_{n_farms}farms',
                        'method': 'gurobi',
                        'n_farms': n_farms,
                        'n_foods': n_foods,
                        'n_vars': n_vars,
                        'solve_time': run.get('solve_time', 0),
                        'objective': run.get('objective', 0),
                        'gap': run.get('gap', 0),
                        'success': run.get('success', False),
                        'qpu_time': 0,
                        'embedding_time': 0,
                    })
            
            # Hierarchical QPU results
            for run in size_data.get('hierarchical_qpu', []):
                timings = run.get('timings', {})
                result_data = run.get('result', {})
                
                all_results.append({
                    'source_file': Path(fpath).name,
                    'scenario_id': f'hierarchical_{n_farms}farms',
                    'method': 'hierarchical_qpu',
                    'n_farms': n_farms,
                    'n_foods': n_foods,
                    'n_vars': n_vars,
                    'solve_time': timings.get('total', 0),
                    'objective': result_data.get('objective', 0),
                    'gap': 0,
                    'success': result_data.get('success', False),
                    'qpu_time': timings.get('qpu_access_total', 0),
                    'embedding_time': timings.get('embedding_total', 0),
                })
    except Exception as e:
        print(f"  Error parsing {fpath}: {e}")

print(f"  Total results so far: {len(all_results)}")

# --- 3. Extract from Roadmap files ---
print("\n[3] Extracting from roadmap phase files...")
roadmap_files = sorted(glob.glob(str(BASE_DIR / 'qpu_benchmark_results' / 'roadmap_*.json')))

for fpath in roadmap_files:
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
        
        for result in data.get('results', []):
            method = result.get('method', 'unknown')
            n_farms = result.get('scale', 0)
            
            if 'gurobi' in method.lower():
                timings = result.get('timings', {})
                all_results.append({
                    'source_file': Path(fpath).name,
                    'scenario_id': f'roadmap_phase{data.get("phase", 0)}_{n_farms}farms',
                    'method': 'gurobi',
                    'n_farms': n_farms,
                    'n_foods': 6,
                    'n_vars': result.get('n_variables', 0),
                    'solve_time': timings.get('solve_time', timings.get('solve', 0)),
                    'objective': result.get('objective', 0),
                    'gap': 0,
                    'success': result.get('success', False),
                    'qpu_time': 0,
                    'embedding_time': 0,
                })
            elif any(x in method.lower() for x in ['cqm', 'partition', 'hybrid', 'qpu']):
                timings = result.get('timings', {})
                all_results.append({
                    'source_file': Path(fpath).name,
                    'scenario_id': f'roadmap_phase{data.get("phase", 0)}_{n_farms}farms',
                    'method': method,
                    'n_farms': n_farms,
                    'n_foods': 6,
                    'n_vars': result.get('n_variables', 0),
                    'solve_time': timings.get('total', timings.get('solve_time', 0)),
                    'objective': result.get('objective', 0),
                    'gap': 0,
                    'success': result.get('success', False),
                    'qpu_time': timings.get('qpu_access_total', timings.get('qpu_time', 0)),
                    'embedding_time': timings.get('embedding_total', 0),
                })
    except Exception as e:
        print(f"  Error parsing {fpath}: {e}")

print(f"  Total results: {len(all_results)}")

# --- 4. Extract from benchmark_results ---
print("\n[4] Extracting from general benchmark results...")
bench_files = sorted(glob.glob(str(BASE_DIR / 'benchmark_results' / '*.json')))

for fpath in bench_files[-3:]:  # Last 3 files
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
        
        # Handle different structures
        if isinstance(data, list):
            results_list = data
        elif isinstance(data, dict):
            results_list = data.get('results', [])
        else:
            continue
        
        for result in results_list:
            if not isinstance(result, dict):
                continue
            method = result.get('method', result.get('solver', 'unknown'))
            n_farms = result.get('n_farms', result.get('n_units', 0))
            
            all_results.append({
                'source_file': Path(fpath).name,
                'scenario_id': f'benchmark_{n_farms}farms',
                'method': method,
                'n_farms': n_farms,
                'n_foods': result.get('n_foods', 27),
                'n_vars': result.get('n_variables', 0),
                'solve_time': result.get('solve_time', result.get('wall_time', 0)),
                'objective': result.get('objective', result.get('objective_value', 0)),
                'gap': result.get('gap', 0),
                'success': result.get('success', False),
                'qpu_time': result.get('qpu_time', 0),
                'embedding_time': result.get('embedding_time', 0),
            })
    except Exception as e:
        print(f"  Error parsing {fpath}: {e}")

print(f"  Final total: {len(all_results)} results")

# ============================================================================
# CREATE DATAFRAME AND SAVE
# ============================================================================

df = pd.DataFrame(all_results)

# Remove duplicates
df = df.drop_duplicates(subset=['scenario_id', 'method', 'n_farms', 'objective'])

# Categorize by time (100s threshold)
def categorize_time(t):
    if t < 10:
        return 'FAST'
    elif t < 100:
        return 'MEDIUM'
    else:
        return 'TIMEOUT'

df['time_category'] = df['solve_time'].apply(categorize_time)

# Save all results
df.to_csv(OUTPUT_DIR / 'all_extracted_results.csv', index=False)
print(f"\n✓ Saved {len(df)} results to: {OUTPUT_DIR / 'all_extracted_results.csv'}")

# ============================================================================
# CREATE SIGNIFICANT SCENARIOS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SIGNIFICANT SCENARIOS SUMMARY")
print("="*80)

# Save scenario definitions
with open(OUTPUT_DIR / 'scenario_definitions.json', 'w') as f:
    json.dump(SIGNIFICANT_SCENARIOS, f, indent=2)
print(f"\n✓ Saved scenario definitions to: {OUTPUT_DIR / 'scenario_definitions.json'}")

# Create comparison table
comparison_data = []

for scenario_id, scenario_info in SIGNIFICANT_SCENARIOS.items():
    n_farms = scenario_info['n_farms']
    
    # Find matching results
    df_match = df[df['n_farms'] == n_farms]
    
    # Get Gurobi result
    gurobi_results = df_match[df_match['method'] == 'gurobi']
    if len(gurobi_results) > 0:
        gurobi_best = gurobi_results.loc[gurobi_results['objective'].idxmax()]
    else:
        gurobi_best = None
    
    # Get QPU results
    qpu_methods = ['clique_decomp', 'spatial_temporal', 'hierarchical_qpu', 'cqm_partition', 'hybrid']
    qpu_results = df_match[df_match['method'].str.contains('|'.join(qpu_methods), case=False, na=False)]
    if len(qpu_results) > 0:
        qpu_best = qpu_results.loc[qpu_results['objective'].idxmax()]
    else:
        qpu_best = None
    
    comparison_data.append({
        'scenario_id': scenario_id,
        'description': scenario_info['description'],
        'n_farms': n_farms,
        'n_foods': scenario_info['n_foods'],
        'expected': scenario_info['expected_behavior'],
        'gurobi_obj': gurobi_best['objective'] if gurobi_best is not None else None,
        'gurobi_time': gurobi_best['solve_time'] if gurobi_best is not None else None,
        'gurobi_status': gurobi_best['time_category'] if gurobi_best is not None else None,
        'qpu_obj': qpu_best['objective'] if qpu_best is not None else None,
        'qpu_time': qpu_best['solve_time'] if qpu_best is not None else None,
        'qpu_method': qpu_best['method'] if qpu_best is not None else None,
        'qpu_status': qpu_best['time_category'] if qpu_best is not None else None,
    })

df_comparison = pd.DataFrame(comparison_data)
df_comparison.to_csv(OUTPUT_DIR / 'significant_scenarios_comparison.csv', index=False)
print(f"✓ Saved comparison to: {OUTPUT_DIR / 'significant_scenarios_comparison.csv'}")

# Print summary
print("\n" + "-"*80)
print("GUROBI vs QPU COMPARISON")
print("-"*80)
print(df_comparison.to_string(index=False))

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY BY METHOD")
print("="*80)

for method in df['method'].unique():
    method_df = df[df['method'] == method]
    print(f"\n{method}: {len(method_df)} results")
    print(f"  Farm range: {method_df['n_farms'].min()}-{method_df['n_farms'].max()}")
    print(f"  Solve time: {method_df['solve_time'].min():.2f}s - {method_df['solve_time'].max():.2f}s")
    if method_df['qpu_time'].sum() > 0:
        print(f"  QPU time: {method_df['qpu_time'].min():.2f}s - {method_df['qpu_time'].max():.2f}s")
    print(f"  Success rate: {method_df['success'].sum()/len(method_df)*100:.1f}%")

print("\n" + "="*80)
print("EXTRACTION COMPLETE")
print("="*80)
