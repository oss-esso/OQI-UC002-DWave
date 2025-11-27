#!/usr/bin/env python3
"""
Analyze and summarize comprehensive benchmark results

Generates readable reports showing:
1. Embedding success rates by formulation
2. Solve times comparison
3. Decomposition effectiveness
4. Density vs embedding success correlation
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_latest_results():
    """Load the most recent benchmark results"""
    results_dir = Path(__file__).parent / "benchmark_results"
    json_files = list(results_dir.glob("comprehensive_benchmark_*.json"))
    
    if not json_files:
        print("No benchmark results found!")
        return None
    
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest.name}")
    
    with open(latest) as f:
        return json.load(f)

def analyze_results(data):
    """Generate comprehensive analysis"""
    results = data['results']
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK ANALYSIS")
    print("=" * 80)
    print(f"Timestamp: {data['timestamp']}")
    print(f"Problem sizes: {data['problem_sizes']}")
    print(f"Total experiments: {len(results)}")
    
    # Convert to DataFrame for easier analysis
    df_data = []
    for r in results:
        row = {
            'scenario': r['scenario'],
            'formulation': r['formulation'],
            'n_units': r['n_units'],
            'decomposition': r.get('decomposition', 'None'),
        }
        
        # Handle metadata - convert 'N/A' to None for numeric fields
        meta = r.get('metadata', {})
        row['variables'] = meta.get('variables', meta.get('n_bqm_vars', None))
        row['constraints'] = meta.get('constraints', None)
        row['quadratic_terms'] = meta.get('quadratic_terms', None)
        row['density'] = meta.get('density', None)
        
        # Replace 'N/A' with None
        for key in ['variables', 'constraints', 'quadratic_terms', 'density']:
            if row.get(key) == 'N/A':
                row[key] = None
        
        # Embedding results
        row['embed_success'] = None
        row['embed_skipped'] = False
        row['embed_time'] = None
        row['embed_error'] = None
        row['num_partitions'] = None
        row['partition_details'] = None
        
        if r.get('embedding'):
            emb = r['embedding']
            if isinstance(emb, dict):
                if 'all_embedded' in emb:  # Decomposed
                    row['embed_success'] = emb.get('all_embedded', False)
                    row['embed_time'] = emb.get('total_embedding_time', 300.0)  # Use 300s for failed
                    row['num_partitions'] = emb.get('num_partitions', None)
                    
                    # Extract partition-level details
                    if 'partition_results' in emb:
                        partition_info = []
                        for i, part in enumerate(emb['partition_results']):
                            p_info = {
                                'id': i,
                                'success': part.get('success', False),
                                'time': part.get('embedding_time', 300.0) if part.get('success') else 300.0,
                                'skipped': part.get('skipped', False)
                            }
                            partition_info.append(p_info)
                        row['partition_details'] = partition_info
                else:  # Regular
                    row['embed_success'] = emb.get('success', False)
                    row['embed_skipped'] = emb.get('skipped', False)
                    # Use 300s for skipped/failed embeddings
                    if row['embed_skipped'] or not row['embed_success']:
                        row['embed_time'] = 300.0
                    else:
                        row['embed_time'] = emb.get('embedding_time', None)
                    row['embed_error'] = emb.get('error', None)
        
        # Solving results
        row['solve_success'] = None
        row['solve_time'] = None
        row['objective'] = None
        row['solve_status'] = None
        
        if r.get('solving'):
            solve = r['solving']
            row['solve_success'] = solve.get('success', False)
            row['solve_time'] = solve.get('solve_time', None)
            row['objective'] = solve.get('objective', None)
            row['solve_status'] = solve.get('status', None)
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Convert numeric columns
    numeric_cols = ['variables', 'constraints', 'quadratic_terms', 'density', 'embed_time', 'solve_time']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Analysis 1: Formulation Comparison
    print("\n" + "=" * 80)
    print("FORMULATION COMPARISON")
    print("=" * 80)
    
    form_stats = df.groupby(['scenario', 'formulation']).agg({
        'density': 'mean',
        'variables': 'mean',
        'quadratic_terms': 'mean'
    }).round(3)
    print("\nProblem Structure by Formulation:")
    print(form_stats.to_string())
    
    # Analysis 2: Embedding Success
    print("\n" + "=" * 80)
    print("EMBEDDING SUCCESS RATES")
    print("=" * 80)
    
    embed_df = df[df['embed_success'].notna()].copy()
    if not embed_df.empty:
        embed_stats = embed_df.groupby(['formulation', 'n_units']).agg({
            'embed_success': 'mean',
            'embed_time': 'mean',
            'embed_skipped': 'sum'
        }).round(2)
        print("\nSuccess Rate by Formulation and Size:")
        print(embed_stats.to_string())
        
        # Density correlation
        print("\n\nDensity vs Embedding Success:")
        density_corr = embed_df.groupby(pd.cut(embed_df['density'], bins=[0, 0.1, 0.3, 0.5, 1.0])).agg({
            'embed_success': ['count', 'mean'],
            'density': 'mean'
        }).round(3)
        print(density_corr.to_string())
    
    # Analysis 3: Solve Time Comparison
    print("\n" + "=" * 80)
    print("SOLVE TIME COMPARISON (seconds)")
    print("=" * 80)
    
    solve_df = df[df['solve_success'].notna()].copy()
    if not solve_df.empty:
        solve_pivot = solve_df.pivot_table(
            values='solve_time',
            index=['scenario', 'formulation'],
            columns='n_units',
            aggfunc='mean'
        ).round(2)
        print("\nAverage Solve Time by Size:")
        print(solve_pivot.to_string())
    
    # Analysis 4: Key Findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Best embeddings
    successful_embeds = embed_df[embed_df['embed_success'] == True]
    if not successful_embeds.empty:
        fastest_embed = successful_embeds.loc[successful_embeds['embed_time'].idxmin()]
        print(f"\n✓ Fastest successful embedding:")
        print(f"  - Formulation: {fastest_embed['formulation']}")
        print(f"  - Size: {fastest_embed['n_units']} units")
        print(f"  - Time: {fastest_embed['embed_time']:.2f}s")
        print(f"  - Density: {fastest_embed['density']:.3f}")
    
    # Best solves
    successful_solves = solve_df[solve_df['solve_success'] == True]
    if not successful_solves.empty:
        fastest_solve = successful_solves.loc[successful_solves['solve_time'].idxmin()]
        print(f"\n✓ Fastest successful solve:")
        print(f"  - Formulation: {fastest_solve['formulation']}")
        print(f"  - Size: {fastest_solve['n_units']} units")
        print(f"  - Time: {fastest_solve['solve_time']:.4f}s")
    
    # Density insights
    print("\n✓ Density insights:")
    dense_probs = df[df['density'] > 0.3]
    sparse_probs = df[df['density'] <= 0.1]
    print(f"  - Dense problems (>0.3): {len(dense_probs)} - {(embed_df[embed_df['density'] > 0.3]['embed_success'].mean() * 100):.0f}% embedded")
    print(f"  - Sparse problems (≤0.1): {len(sparse_probs)} - {(embed_df[embed_df['density'] <= 0.1]['embed_success'].mean() * 100):.0f}% embedded")
    
    # Decomposition effectiveness
    decomp_df = df[df['decomposition'] != 'None']
    if not decomp_df.empty:
        print(f"\n✓ Decomposition strategies tested: {decomp_df['decomposition'].nunique()}")
        decomp_success = decomp_df.groupby('decomposition')['embed_success'].mean()
        print("  - Success rates by strategy:")
        for strategy, rate in decomp_success.items():
            print(f"    {strategy}: {rate*100:.0f}%")
    
    print("\n" + "=" * 80)
    print("DETAILED SUMMARY TABLE")
    print("=" * 80)
    
    # Create detailed summary with decomposition info
    detailed_summary = []
    for _, row in df.iterrows():
        entry = {
            'Scenario': row['scenario'],
            'Formulation': row['formulation'],
            'Size': row['n_units'],
            'Decomposition': row['decomposition'],
            'Density': f"{row['density']:.3f}" if pd.notna(row['density']) else 'N/A',
            'Embed_Success': 'Yes' if row.get('embed_success') == True else ('No' if row.get('embed_success') == False else 'Skip'),
            'Embed_Time': f"{row['embed_time']:.1f}s" if pd.notna(row['embed_time']) else '300.0s',
            'Solve_Time': f"{row['solve_time']:.3f}s" if pd.notna(row['solve_time']) else 'N/A',
            'Objective': f"{row['objective']:.2f}" if pd.notna(row['objective']) else 'N/A',
        }
        
        # Add partition info if available
        if row.get('num_partitions'):
            entry['Partitions'] = row['num_partitions']
        else:
            entry['Partitions'] = 1
        
        detailed_summary.append(entry)
    
    summary_df = pd.DataFrame(detailed_summary)
    
    # Group by formulation and decomposition to show clearer patterns
    print("\n" + "=" * 80)
    print("EMBEDDING RESULTS BY FORMULATION AND DECOMPOSITION")
    print("=" * 80)
    
    for size in sorted(df['n_units'].unique()):
        size_df = summary_df[summary_df['Size'] == size]
        if not size_df.empty:
            print(f"\n{'='*80}")
            print(f"SIZE {size} UNITS")
            print(f"{'='*80}")
            print(size_df[['Formulation', 'Decomposition', 'Density', 'Partitions', 
                          'Embed_Success', 'Embed_Time', 'Solve_Time', 'Objective']].to_string(index=False))
    
    # Show partition-level details for decomposed strategies
    print("\n" + "=" * 80)
    print("PARTITION-LEVEL EMBEDDING DETAILS")
    print("=" * 80)
    
    decomp_df = df[df['decomposition'] != 'None'].copy()
    for _, row in decomp_df.iterrows():
        if row.get('partition_details'):
            print(f"\n{row['formulation']} | {row['decomposition']} | Size {row['n_units']}")
            print(f"  Total partitions: {len(row['partition_details'])}")
            for part in row['partition_details']:
                status = '✓ EMBEDDED' if part['success'] else ('⊘ SKIPPED' if part['skipped'] else '✗ FAILED')
                time_str = f"{part['time']:.1f}s" if part['success'] else f"{part['time']:.1f}s (timeout)"
                print(f"    Partition {part['id']}: {status} - {time_str}")
    
    return df

if __name__ == "__main__":
    data = load_latest_results()
    if data:
        df = analyze_results(data)
        
        # Save analysis
        output_file = Path(__file__).parent / "benchmark_results" / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Analysis saved to: {output_file}")
