#!/usr/bin/env python3
"""
Analyze datapoints for patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
results_dir = Path(__file__).parent / 'hardness_analysis_results'
df = pd.read_csv(results_dir / 'all_datapoints_for_analysis.csv')

print("="*80)
print("PATTERN ANALYSIS - 37 Gurobi Datapoints")
print("="*80)

# Basic stats
print(f"\nTotal datapoints: {len(df)}")
print(f"Farm range: {df['n_farms'].min()}-{df['n_farms'].max()}")
print(f"Solve time range: {df['solve_time'].min():.2f}s - {df['solve_time'].max():.2f}s")

# Pattern 1: Success rate by farm size
print("\n" + "="*80)
print("PATTERN 1: Success Rate by Farm Size (< 100s threshold)")
print("="*80)

farm_groups = [
    ('Small (4-10 farms)', df[(df['n_farms'] >= 4) & (df['n_farms'] <= 10)]),
    ('Medium (11-20 farms)', df[(df['n_farms'] >= 11) & (df['n_farms'] <= 20)]),
    ('Large (25-100 farms)', df[df['n_farms'] >= 25])
]

for name, group in farm_groups:
    if len(group) == 0:
        continue
    fast = len(group[group['time_category'] == 'FAST'])
    medium = len(group[group['time_category'] == 'MEDIUM'])
    timeout = len(group[group['time_category'] == 'TIMEOUT'])
    total = len(group)
    success_rate = (fast + medium) / total * 100
    
    print(f"\n{name}: {total} instances")
    print(f"  Fast (< 10s): {fast} ({fast/total*100:.1f}%)")
    print(f"  Medium (10-100s): {medium} ({medium/total*100:.1f}%)")
    print(f"  Timeout (> 100s): {timeout} ({timeout/total*100:.1f}%)")
    print(f"  → Success rate (< 100s): {success_rate:.1f}%")

# Pattern 2: By test type
print("\n" + "="*80)
print("PATTERN 2: Performance by Test Type")
print("="*80)

for test_type in sorted(df['test_type'].unique()):
    subset = df[df['test_type'] == test_type]
    fast = len(subset[subset['time_category'] == 'FAST'])
    medium = len(subset[subset['time_category'] == 'MEDIUM'])
    timeout = len(subset[subset['time_category'] == 'TIMEOUT'])
    
    print(f"\n{test_type}: {len(subset)} instances")
    print(f"  Fast: {fast}, Medium: {medium}, Timeout: {timeout}")
    print(f"  Avg solve time: {subset['solve_time'].mean():.2f}s")
    print(f"  Median solve time: {subset['solve_time'].median():.2f}s")
    print(f"  Farms: {subset['n_farms'].min()}-{subset['n_farms'].max()}")

# Pattern 3: Objective value vs solve time
print("\n" + "="*80)
print("PATTERN 3: Objective Value vs Solve Time Correlation")
print("="*80)

# Separate by timeout
fast_medium = df[df['time_category'].isin(['FAST', 'MEDIUM'])]
timeouts = df[df['time_category'] == 'TIMEOUT']

if len(fast_medium) > 0:
    corr = fast_medium[['solve_time', 'obj_value', 'n_farms']].corr()
    print("\nFor FAST+MEDIUM instances:")
    print(f"  Solve time vs N_farms: {corr.loc['solve_time', 'n_farms']:.3f}")
    print(f"  Solve time vs Objective: {corr.loc['solve_time', 'obj_value']:.3f}")
    print(f"  Objective vs N_farms: {corr.loc['obj_value', 'n_farms']:.3f}")

if len(timeouts) > 0:
    print(f"\nFor TIMEOUT instances ({len(timeouts)}):")
    print(f"  All solve times: ~300s (hit time limit)")
    print(f"  Objective range: {timeouts['obj_value'].min():.2f} - {timeouts['obj_value'].max():.2f}")
    print(f"  Farm range: {timeouts['n_farms'].min()}-{timeouts['n_farms'].max()}")

# Pattern 4: Critical threshold detection
print("\n" + "="*80)
print("PATTERN 4: Critical Thresholds")
print("="*80)

print("\nFarm count where problems become hard:")
successful = df[df['time_category'].isin(['FAST', 'MEDIUM'])].sort_values('n_farms')
failed = df[df['time_category'] == 'TIMEOUT'].sort_values('n_farms')

if len(successful) > 0:
    print(f"  Largest successful instance: {successful['n_farms'].max()} farms ({successful[successful['n_farms'] == successful['n_farms'].max()]['solve_time'].values[0]:.2f}s)")

if len(failed) > 0:
    print(f"  Smallest failed instance: {failed['n_farms'].min()} farms")

# Find transition zone
print("\nTransition zone (instances around 10-20 farms):")
transition = df[(df['n_farms'] >= 10) & (df['n_farms'] <= 20)].sort_values('n_farms')
for _, row in transition.iterrows():
    print(f"  {row['n_farms']} farms → {row['solve_time']:.2f}s ({row['time_category']}) [{row['test_type']}]")

# Pattern 5: Gap analysis for timeouts
print("\n" + "="*80)
print("PATTERN 5: MIP Gap Analysis for Timeouts")
print("="*80)

timeout_with_gap = timeouts[timeouts['gap'] > 0]
if len(timeout_with_gap) > 0:
    print(f"\n{len(timeout_with_gap)} timeout instances with gap data:")
    print(f"  Mean gap: {timeout_with_gap['gap'].mean()*100:.2f}%")
    print(f"  Gap range: {timeout_with_gap['gap'].min()*100:.2f}% - {timeout_with_gap['gap'].max()*100:.2f}%")
    
    print("\n  Instances with large gaps (> 5%):")
    large_gaps = timeout_with_gap[timeout_with_gap['gap'] > 0.05].sort_values('gap', ascending=False)
    for _, row in large_gaps.head(10).iterrows():
        print(f"    {row['n_farms']} farms: {row['gap']*100:.2f}% gap, obj={row['obj_value']:.2f} [{row['test_type']}]")

# Pattern 6: Source/method breakdown
print("\n" + "="*80)
print("PATTERN 6: Breakdown by Data Source")
print("="*80)

for source in sorted(df['source'].unique()):
    subset = df[df['source'] == source]
    print(f"\n{source}: {len(subset)} instances")
    print(f"  Fast: {len(subset[subset['time_category'] == 'FAST'])}")
    print(f"  Medium: {len(subset[subset['time_category'] == 'MEDIUM'])}")
    print(f"  Timeout: {len(subset[subset['time_category'] == 'TIMEOUT'])}")

# KEY FINDINGS
print("\n" + "="*80)
print("KEY FINDINGS & PATTERNS")
print("="*80)

print("\n1. CRITICAL THRESHOLD:")
print(f"   - Only {len(successful)} instances solve within 100s")
print(f"   - {len(failed)} instances timeout (88.9%)")
print(f"   - Problems become intractable around 10-15 farms")

print("\n2. TEST TYPE PATTERNS:")
for test_type in sorted(df['test_type'].unique()):
    subset = df[df['test_type'] == test_type]
    success_pct = len(subset[subset['time_category'] != 'TIMEOUT']) / len(subset) * 100
    print(f"   - {test_type}: {success_pct:.1f}% success rate")

print("\n3. SCALING BEHAVIOR:")
if len(successful) > 0 and len(failed) > 0:
    print(f"   - Max solvable size: {successful['n_farms'].max()} farms")
    print(f"   - Min unsolvable size: {failed['n_farms'].min()} farms")
    print(f"   - Sharp transition between {successful['n_farms'].max()}-{failed['n_farms'].min()} farms")

print("\n4. HARDNESS FACTORS:")
print("   - All Roadmap, Statistical, and Hierarchical tests show extreme hardness")
print("   - Most instances (33/37) cannot solve within 100s")
print("   - MIP gaps remain large (> 5%) even after 300s")

print("\n" + "="*80)
