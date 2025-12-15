#!/usr/bin/env python3
"""
Analyze scaling by number of variables (not farms).
Shows we're missing many data points between tests.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
stat_file = Path('statistical_comparison_results/statistical_comparison_20251211_180707.json')
hier_file = Path('hierarchical_statistical_results/hierarchical_results_20251212_124349.json')

with open(stat_file) as f:
    stat_results = json.load(f)

with open(hier_file) as f:
    hier_results = json.load(f)

print("="*80)
print("VARIABLE COUNT ANALYSIS")
print("="*80)
print()

# Collect all data points by variable count
data_points = []

# Statistical test
for size in [5, 10, 15, 20]:
    size_str = str(size)
    if size_str not in stat_results['results_by_size']:
        continue
    
    data = stat_results['results_by_size'][size_str]
    n_vars = data['n_variables']
    
    # Ground truth
    gt_runs = data['methods']['ground_truth']['runs']
    gt_success = [r for r in gt_runs if r.get('success', False)]
    if gt_success:
        gt_obj = np.mean([r['objective'] for r in gt_success])
        gt_time = np.mean([r['wall_time'] for r in gt_success])
    else:
        continue
    
    # Quantum methods
    for method_name in ['clique_decomp', 'spatial_temporal']:
        method_runs = data['methods'][method_name]['runs']
        q_success = [r for r in method_runs if r.get('success', False)]
        
        if q_success:
            q_obj = np.mean([r['objective'] for r in q_success])
            q_time = np.mean([r['wall_time'] for r in q_success])
            q_qpu = np.mean([r.get('qpu_time', 0) for r in q_success])
            
            gap = abs(q_obj - gt_obj) / abs(gt_obj) * 100 if gt_obj != 0 else 0
            speedup = gt_time / q_time if q_time > 0 else 0
            
            data_points.append({
                'n_vars': n_vars,
                'n_farms': size,
                'method': method_name.replace('_', ' ').title(),
                'test': 'Statistical',
                'formulation': '6 families (native)',
                'gurobi_obj': gt_obj,
                'quantum_obj': q_obj,
                'gap': gap,
                'speedup': speedup,
                'qpu_time': q_qpu,
            })

# Hierarchical test
for size in [25, 50, 100]:
    if str(size) not in hier_results:
        continue
    
    data = hier_results[str(size)]
    n_vars = data['data_info']['n_variables_aggregated']  # After aggregation
    n_vars_original = data['data_info']['n_variables']  # Before aggregation
    
    # Gurobi
    gt_runs = data['gurobi']
    gt_obj = np.mean([r['objective'] for r in gt_runs])
    gt_time = np.mean([r['solve_time'] for r in gt_runs])
    
    # Quantum
    stats = data['statistics']['hierarchical_qpu']
    q_obj = stats['objective_mean']
    q_time = stats['time_mean']
    q_qpu = stats['qpu_time_mean']
    
    gap = abs(q_obj - gt_obj) / abs(gt_obj) * 100 if gt_obj != 0 else 0
    speedup = gt_time / q_time if q_time > 0 else 0
    
    data_points.append({
        'n_vars': n_vars,
        'n_vars_original': n_vars_original,
        'n_farms': size,
        'method': 'Hierarchical Quantum',
        'test': 'Hierarchical',
        'formulation': '27 foods → 6 families',
        'gurobi_obj': gt_obj,
        'quantum_obj': q_obj,
        'gap': gap,
        'speedup': speedup,
        'qpu_time': q_qpu,
    })

df = pd.DataFrame(data_points)

print("DATA POINTS BY VARIABLE COUNT:")
print("="*80)
print()
print(df[['n_vars', 'n_farms', 'method', 'formulation', 'gap', 'speedup', 'qpu_time']].to_string(index=False))
print()

print("="*80)
print("VARIABLE COUNT DISTRIBUTION:")
print("="*80)
print()
print("Statistical Test (native 6 families):")
print(f"  Variable counts: {sorted(df[df['test']=='Statistical']['n_vars'].unique())}")
print(f"  Range: {df[df['test']=='Statistical']['n_vars'].min()}-{df[df['test']=='Statistical']['n_vars'].max()}")
print()
print("Hierarchical Test (aggregated 27→6):")
print(f"  Variable counts: {sorted(df[df['test']=='Hierarchical']['n_vars'].unique())}")
print(f"  Range: {df[df['test']=='Hierarchical']['n_vars'].min()}-{df[df['test']=='Hierarchical']['n_vars'].max()}")
print()

print("="*80)
print("MISSING DATA POINTS:")
print("="*80)
print()
print("We have: 90, 180, 270, 360 (statistical), then jump to 450, 900, 1800 (hierarchical)")
print()
print("Missing intermediate points where we should test:")
print("  - 450 vars with NATIVE 6-family formulation (25 farms)")
print("  - 540 vars (30 farms)")
print("  - 630 vars (35 farms)")
print("  - 720 vars (40 farms)")
print("  - ...up to 1800 vars")
print()
print("This would show if gap increase is due to:")
print("  A) Variable count scaling (smooth increase)")
print("  B) Formulation change (sudden jump at 450)")
print()

# Create proper scaling plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Gap vs Variables (with formulation marker)
ax = axes[0, 0]
for test_type in df['test'].unique():
    test_df = df[df['test'] == test_type]
    marker = 'o' if test_type == 'Statistical' else 's'
    for method in test_df['method'].unique():
        method_df = test_df[test_df['method'] == method].sort_values('n_vars')
        label = f"{method} ({test_type})"
        ax.plot(method_df['n_vars'], method_df['gap'], marker=marker, 
                label=label, linewidth=2, markersize=8)

ax.axvline(x=360, color='red', linestyle='--', alpha=0.5, linewidth=2, 
           label='Formulation Change (360→450 vars)')
ax.set_xlabel('Number of Variables', fontsize=12)
ax.set_ylabel('Optimality Gap (%)', fontsize=12)
ax.set_title('Gap vs Variables (Shows Formulation Jump)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Speedup vs Variables
ax = axes[0, 1]
for test_type in df['test'].unique():
    test_df = df[df['test'] == test_type]
    marker = 'o' if test_type == 'Statistical' else 's'
    for method in test_df['method'].unique():
        method_df = test_df[test_df['method'] == method].sort_values('n_vars')
        label = f"{method} ({test_type})"
        ax.plot(method_df['n_vars'], method_df['speedup'], marker=marker, 
                label=label, linewidth=2, markersize=8)

ax.axvline(x=360, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Number of Variables', fontsize=12)
ax.set_ylabel('Speedup Factor (×)', fontsize=12)
ax.set_title('Speedup vs Variables', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: QPU Time vs Variables
ax = axes[1, 0]
for test_type in df['test'].unique():
    test_df = df[df['test'] == test_type]
    marker = 'o' if test_type == 'Statistical' else 's'
    for method in test_df['method'].unique():
        method_df = test_df[test_df['method'] == method].sort_values('n_vars')
        label = f"{method} ({test_type})"
        ax.plot(method_df['n_vars'], method_df['qpu_time'], marker=marker, 
                label=label, linewidth=2, markersize=8)

ax.axvline(x=360, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.set_xlabel('Number of Variables', fontsize=12)
ax.set_ylabel('QPU Time (seconds)', fontsize=12)
ax.set_title('QPU Time Scaling (Linear)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: Objective comparison
ax = axes[1, 1]
for test_type in df['test'].unique():
    test_df = df[df['test'] == test_type]
    marker = 'o' if test_type == 'Statistical' else 's'
    
    # Group by n_vars (take first method for each)
    grouped = test_df.groupby('n_vars').first().reset_index()
    ax.plot(grouped['n_vars'], grouped['gurobi_obj'], marker=marker, 
            label=f'Gurobi ({test_type})', linewidth=2, markersize=8, linestyle='--')
    ax.plot(grouped['n_vars'], grouped['quantum_obj'], marker=marker, 
            label=f'Quantum ({test_type})', linewidth=2, markersize=8)

ax.axvline(x=360, color='red', linestyle='--', alpha=0.5, linewidth=2, 
           label='Formulation Change')
ax.set_xlabel('Number of Variables', fontsize=12)
ax.set_ylabel('Objective Value', fontsize=12)
ax.set_title('Objective Values (Note: Gurobi drops at formulation change)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_file = Path('variable_count_scaling_analysis.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Scaling plots saved to: {output_file}")

plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
print(f"✓ PDF version saved to: {output_file.with_suffix('.pdf')}")

# Save data
csv_file = Path('variable_scaling_data.csv')
df.to_csv(csv_file, index=False)
print(f"✓ Data saved to: {csv_file}")

print()
print("="*80)
print("RECOMMENDATION:")
print("="*80)
print()
print("To properly understand scaling, we need MORE DATA POINTS:")
print()
print("Suggested additional tests (all with native 6-family formulation):")
print("  - 25 farms = 450 vars")
print("  - 30 farms = 540 vars")
print("  - 40 farms = 720 vars")
print("  - 50 farms = 900 vars (compare to hierarchical 900)")
print()
print("This would:")
print("  1. Show continuous scaling from 90 → 1800 variables")
print("  2. Remove formulation confound")
print("  3. Reveal if gap increase is linear or sudden")
print("  4. Enable proper scaling law analysis")
