#!/usr/bin/env python3
"""
Create comprehensive scaling plot for hardness analysis results.
Updated for constant area PER FARM normalization (1 ha/farm).
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_dir = Path(__file__).parent / 'hardness_analysis_results'
df = pd.read_csv(results_dir / 'hardness_analysis_results.csv')

print("="*80)
print("CREATING COMPREHENSIVE HARDNESS SCALING PLOT")
print("="*80)

# Filter valid results
df = df[df['status'] != 'ERROR'].copy()

# Add derived metrics
df['quadratic_density'] = df['n_quadratic'] / df['n_vars']
df['constraints_per_var'] = df['n_constraints'] / df['n_vars']

print(f"\nLoaded {len(df)} valid test points")
print(f"Variable range: {df['n_vars'].min()}-{df['n_vars'].max()}")
print(f"Farm range: {df['n_farms'].min()}-{df['n_farms'].max()}")
print(f"Solve time range: {df['solve_time'].min():.2f}-{df['solve_time'].max():.2f}s")

# Create 2x3 subplot figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comprehensive Hardness Analysis: Gurobi Performance Scaling\n(Constant Area Per Farm: 1 ha/farm, 6 Food Families, 3 Periods)', 
             fontsize=16, fontweight='bold', y=0.995)

# Color coding by time category
color_map = {
    'FAST': 'green',
    'MEDIUM': 'orange',
    'SLOW': 'red',
    'TIMEOUT': 'darkred'
}
colors = [color_map.get(cat, 'gray') for cat in df['time_category']]

# ============================================================================
# Plot 1: Solve Time vs Number of Farms
# ============================================================================
ax = axes[0, 0]
scatter = ax.scatter(df['n_farms'], df['solve_time'], 
                     c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
ax.plot(df['n_farms'], df['solve_time'], 'k--', alpha=0.3, linewidth=1)

# Add hardness zones
ax.axhspan(0, 10, alpha=0.1, color='green', label='FAST (<10s)')
ax.axhspan(10, 100, alpha=0.1, color='orange', label='MEDIUM (10-100s)')
ax.axhspan(100, 300, alpha=0.1, color='red', label='SLOW (>100s)')

ax.set_xlabel('Number of Farms', fontsize=13, fontweight='bold')
ax.set_ylabel('Solve Time (seconds)', fontsize=13, fontweight='bold')
ax.set_title('Solve Time vs Problem Size', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# ============================================================================
# Plot 2: Solve Time vs Farms/Food Ratio
# ============================================================================
ax = axes[0, 1]
scatter = ax.scatter(df['farms_per_food'], df['solve_time'], 
                     c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
ax.plot(df['farms_per_food'], df['solve_time'], 'k--', alpha=0.3, linewidth=1)

# Mark critical thresholds
ax.axvline(4.2, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Transition (~4.2)')
ax.axvline(10, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Hard zone (>10)')

ax.set_xlabel('Farms per Food Family', fontsize=13, fontweight='bold')
ax.set_ylabel('Solve Time (seconds)', fontsize=13, fontweight='bold')
ax.set_title('Solve Time vs Farms/Food Ratio', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# ============================================================================
# Plot 3: MIP Gap vs Problem Size
# ============================================================================
ax = axes[0, 2]
# Filter out None gaps
df_gap = df[df['gap'].notna()].copy()
scatter = ax.scatter(df_gap['n_vars'], df_gap['gap'] * 100, 
                     c=[color_map.get(cat, 'gray') for cat in df_gap['time_category']], 
                     s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
ax.plot(df_gap['n_vars'], df_gap['gap'] * 100, 'k--', alpha=0.3, linewidth=1)

ax.axhline(1, color='green', linestyle='--', alpha=0.5, linewidth=2, label='1% target')
ax.set_xlabel('Number of Variables', fontsize=13, fontweight='bold')
ax.set_ylabel('MIP Gap (%)', fontsize=13, fontweight='bold')
ax.set_title('Solution Quality (MIP Gap)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# ============================================================================
# Plot 4: Quadratic Density vs Solve Time
# ============================================================================
ax = axes[1, 0]
scatter = ax.scatter(df['quadratic_density'], df['solve_time'], 
                     c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

ax.set_xlabel('Quadratic Terms per Variable', fontsize=13, fontweight='bold')
ax.set_ylabel('Solve Time (seconds)', fontsize=13, fontweight='bold')
ax.set_title('Quadratic Density vs Hardness', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# ============================================================================
# Plot 5: Build Time vs Solve Time
# ============================================================================
ax = axes[1, 1]
scatter = ax.scatter(df['build_time'], df['solve_time'], 
                     c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

# Add diagonal line (build = solve)
max_time = max(df['build_time'].max(), df['solve_time'].max())
ax.plot([0, max_time], [0, max_time], 'k--', alpha=0.3, linewidth=1, label='Build=Solve')

ax.set_xlabel('Build Time (seconds)', fontsize=13, fontweight='bold')
ax.set_ylabel('Solve Time (seconds)', fontsize=13, fontweight='bold')
ax.set_title('Model Construction vs Solving', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# ============================================================================
# Plot 6: Total Area Scaling with Farms
# ============================================================================
ax = axes[1, 2]
scatter = ax.scatter(df['n_farms'], df['total_area'], 
                     c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

# Add expected scaling line (1 ha/farm)
expected_area = df['n_farms'].values * 1.0
ax.plot(df['n_farms'], expected_area, 'b--', alpha=0.5, linewidth=2, label='Expected: 1 ha/farm')
ax.fill_between(df['n_farms'], expected_area * 0.95, expected_area * 1.05, 
                alpha=0.1, color='blue', label='±5% tolerance')

ax.set_xlabel('Number of Farms', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Area (hectares)', fontsize=13, fontweight='bold')
ax.set_title('Area Scaling: Constant Per Farm', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# ============================================================================
# Custom legend for time categories
# ============================================================================
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', edgecolor='black', label='FAST (<10s)'),
    Patch(facecolor='orange', edgecolor='black', label='MEDIUM (10-100s)'),
    Patch(facecolor='red', edgecolor='black', label='SLOW (>100s)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
           fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.02, 1, 0.99])

# Save plot
output_plot = results_dir / 'comprehensive_hardness_scaling_PER_FARM.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\n✓ Comprehensive plot saved to: {output_plot}")

plt.savefig(output_plot.with_suffix('.pdf'), bbox_inches='tight')
print(f"✓ PDF version saved to: {output_plot.with_suffix('.pdf')}")

plt.close()

# ============================================================================
# Print Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE STATISTICS")
print("="*80)

print("\n--- By Time Category ---")
for category in ['FAST', 'MEDIUM', 'SLOW', 'TIMEOUT']:
    cat_df = df[df['time_category'] == category]
    if len(cat_df) > 0:
        print(f"\n{category} ({len(cat_df)} instances):")
        print(f"  Farms: {cat_df['n_farms'].min()}-{cat_df['n_farms'].max()} (mean={cat_df['n_farms'].mean():.1f})")
        print(f"  Variables: {cat_df['n_vars'].min()}-{cat_df['n_vars'].max()} (mean={cat_df['n_vars'].mean():.1f})")
        print(f"  Farms/Food: {cat_df['farms_per_food'].min():.2f}-{cat_df['farms_per_food'].max():.2f} (mean={cat_df['farms_per_food'].mean():.2f})")
        print(f"  Solve time: {cat_df['solve_time'].min():.2f}-{cat_df['solve_time'].max():.2f}s (mean={cat_df['solve_time'].mean():.2f}s)")
        print(f"  Quadratics: {cat_df['n_quadratic'].min()}-{cat_df['n_quadratic'].max()} (mean={cat_df['n_quadratic'].mean():.0f})")
        print(f"  MIP Gap: {cat_df['gap'].min()*100:.2f}%-{cat_df['gap'].max()*100:.2f}% (mean={cat_df['gap'].mean()*100:.2f}%)")

print("\n" + "="*80)
print("PROBLEM SCALING CHARACTERISTICS")
print("="*80)

# Compute correlations
print("\nCorrelations with Solve Time:")
correlations = {
    'Number of Farms': df[['n_farms', 'solve_time']].corr().iloc[0, 1],
    'Number of Variables': df[['n_vars', 'solve_time']].corr().iloc[0, 1],
    'Farms/Food Ratio': df[['farms_per_food', 'solve_time']].corr().iloc[0, 1],
    'Quadratic Terms': df[['n_quadratic', 'solve_time']].corr().iloc[0, 1],
    'Quadratic Density': df[['quadratic_density', 'solve_time']].corr().iloc[0, 1],
    'Constraints': df[['n_constraints', 'solve_time']].corr().iloc[0, 1],
}

for metric, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {metric:<25} r = {corr:>6.3f}  {'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak'}")

print("\n" + "="*80)
print("QPU TARGET RECOMMENDATIONS")
print("="*80)

# Identify QPU sweet spot
medium_df = df[df['time_category'] == 'MEDIUM']
print(f"\nMEDIUM difficulty range (ideal for QPU demonstration):")
print(f"  Farms: {medium_df['n_farms'].min()}-{medium_df['n_farms'].max()}")
print(f"  Variables: {medium_df['n_vars'].min()}-{medium_df['n_vars'].max()}")
print(f"  Solve time: {medium_df['solve_time'].min():.1f}-{medium_df['solve_time'].max():.1f}s")
print(f"  Instances: {len(medium_df)}")

print("\nSpecific recommended test cases:")
for _, row in medium_df.iterrows():
    print(f"  {int(row['n_farms']):3d} farms: {int(row['n_vars']):4d} vars, "
          f"{row['solve_time']:6.2f}s solve, {int(row['n_quadratic']):5d} quads, "
          f"{row['gap']*100:4.2f}% gap")

print("\n" + "="*80)
print("AREA NORMALIZATION VALIDATION")
print("="*80)
print(f"Target area: 100.0 ha")
print(f"Actual area range: {df['total_area'].min():.2f} - {df['total_area'].max():.2f} ha")
print(f"Mean area: {df['total_area'].mean():.2f} ha")
print(f"Std deviation: {df['total_area'].std():.2f} ha")
print(f"Within ±5% tolerance: {((df['total_area'] >= 95) & (df['total_area'] <= 105)).sum()}/{len(df)} instances")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
