#!/usr/bin/env python3
"""
Create comprehensive integrated scaling plot comparing different normalizations.
Integrates multiple result files with different marker shapes.
Includes objective value and solve time subplots.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_dir = Path(__file__).parent / 'hardness_analysis_results'

# Load new results (constant area per farm)
df_new = pd.read_csv(results_dir / 'hardness_analysis_results.csv')
df_new['test_type'] = 'Per-Farm Area (1 ha/farm)'
df_new['normalization'] = 'per_farm'

# Load old results (constant total area) if exists
df_combined_list = [df_new]

try:
    df_old = pd.read_csv(results_dir / 'combined_all_results.csv')
    df_old['test_type'] = 'Total Area (100 ha)'
    df_old['normalization'] = 'total_area'
    
    # Ensure consistent columns
    if 'total_area' not in df_old.columns:
        df_old['total_area'] = 100.0
    
    df_combined_list.append(df_old)
    has_comparison = True
except FileNotFoundError:
    has_comparison = False

# Load QPU results
try:
    df_additional = pd.read_csv(results_dir / 'additional_gurobi_results.csv')
    df_combined_list.append(df_additional)
    print(f"  Loaded {len(df_additional)} additional Gurobi results")
except FileNotFoundError:
    print("  No additional Gurobi data found")

# Combine all datasets
df_combined = pd.concat(df_combined_list, ignore_index=True)
has_comparison = len(df_combined_list) > 1

print("="*80)
print("CREATING INTEGRATED COMPREHENSIVE HARDNESS SCALING PLOT")
print("="*80)

# Filter valid results
df_combined = df_combined[df_combined['status'] != 'ERROR'].copy()

# Add derived metrics if not present
if 'quadratic_density' not in df_combined.columns:
    df_combined['quadratic_density'] = df_combined['n_quadratic'] / df_combined['n_vars']
if 'constraints_per_var' not in df_combined.columns:
    df_combined['constraints_per_var'] = df_combined['n_constraints'] / df_combined['n_vars']

print(f"\nDatasets loaded:")
for test_type in df_combined['test_type'].unique():
    subset = df_combined[df_combined['test_type'] == test_type]
    print(f"  - {test_type}: {len(subset)} data points")
    print(f"    Farm range: {subset['n_farms'].min()}-{subset['n_farms'].max()}")
    print(f"    Solve time range: {subset['solve_time'].min():.2f}-{subset['solve_time'].max():.2f}s")

# Create 3x3 subplot figure for comprehensive analysis
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Comprehensive Integrated Hardness Analysis: Multiple Test Configurations\n' + 
             'Gurobi Performance Scaling (6 Food Families, 3 Periods)', 
             fontsize=18, fontweight='bold', y=0.995)

# Color coding by time category
color_map = {
    'FAST': 'green',
    'MEDIUM': 'orange',
    'SLOW': 'red',
    'TIMEOUT': 'darkred'
}

# Marker shapes by test type
marker_map = {
    'Per-Farm Area (1 ha/farm)': 'o',   # Circle - new normalization
    'Total Area (100 ha)': 's',         # Square - old normalization
    'Roadmap Gurobi': '^',              # Triangle up
    'Hierarchical Gurobi': 'D',         # Diamond  
    'Statistical Gurobi': '*'           # Star
}

# ============================================================================
# Plot 1 (Top Left): Solve Time vs Number of Farms
# ============================================================================
ax = fig.add_subplot(gs[0, 0])

for test_type in df_combined['test_type'].unique():
    df_src = df_combined[df_combined['test_type'] == test_type]
    colors = [color_map.get(cat, 'gray') for cat in df_src['time_category']]
    marker = marker_map.get(test_type, 'o')
    
    ax.scatter(df_src['n_farms'], df_src['solve_time'], 
               c=colors, marker=marker, s=120, alpha=0.7, 
               edgecolors='black', linewidth=1.5, label=test_type)
    ax.plot(df_src['n_farms'], df_src['solve_time'], 
            '--', alpha=0.3, linewidth=1)

# Add hardness zones
ax.axhspan(0, 10, alpha=0.1, color='green')
ax.axhspan(10, 100, alpha=0.1, color='orange')
ax.axhspan(100, 300, alpha=0.1, color='red')

ax.set_xlabel('Number of Farms', fontsize=13, fontweight='bold')
ax.set_ylabel('Solve Time (seconds)', fontsize=13, fontweight='bold')
ax.set_title('Solve Time vs Problem Size', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='upper left')

# ============================================================================
# Plot 2 (Top Center): Solve Time vs Farms/Food Ratio
# ============================================================================
ax = fig.add_subplot(gs[0, 1])

for test_type in df_combined['test_type'].unique():
    df_src = df_combined[df_combined['test_type'] == test_type]
    colors = [color_map.get(cat, 'gray') for cat in df_src['time_category']]
    marker = marker_map.get(test_type, 'o')
    
    ax.scatter(df_src['farms_per_food'], df_src['solve_time'], 
               c=colors, marker=marker, s=120, alpha=0.7, 
               edgecolors='black', linewidth=1.5)
    ax.plot(df_src['farms_per_food'], df_src['solve_time'], 
            '--', alpha=0.3, linewidth=1)

# Mark critical thresholds
ax.axvline(4.2, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Transition (~4.2)')

ax.set_xlabel('Farms per Food Family', fontsize=13, fontweight='bold')
ax.set_ylabel('Solve Time (seconds)', fontsize=13, fontweight='bold')
ax.set_title('Solve Time vs Farms/Food Ratio', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# ============================================================================
# Plot 3 (Top Right): Objective Value vs Number of Farms
# ============================================================================
ax = fig.add_subplot(gs[0, 2])

for test_type in df_combined['test_type'].unique():
    df_src = df_combined[df_combined['test_type'] == test_type]
    colors = [color_map.get(cat, 'gray') for cat in df_src['time_category']]
    marker = marker_map.get(test_type, 'o')
    
    ax.scatter(df_src['n_farms'], df_src['obj_value'], 
               c=colors, marker=marker, s=120, alpha=0.7, 
               edgecolors='black', linewidth=1.5)
    ax.plot(df_src['n_farms'], df_src['obj_value'], 
            '--', alpha=0.3, linewidth=1)

ax.set_xlabel('Number of Farms', fontsize=13, fontweight='bold')
ax.set_ylabel('Objective Value', fontsize=13, fontweight='bold')
ax.set_title('Solution Quality vs Problem Size', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# ============================================================================
# Plot 4 (Middle Left): MIP Gap vs Problem Size
# ============================================================================
ax = fig.add_subplot(gs[1, 0])

for test_type in df_combined['test_type'].unique():
    df_src = df_combined[df_combined['test_type'] == test_type]
    df_gap = df_src[df_src['gap'].notna()].copy()
    colors = [color_map.get(cat, 'gray') for cat in df_gap['time_category']]
    marker = marker_map.get(test_type, 'o')
    
    ax.scatter(df_gap['n_vars'], df_gap['gap'] * 100, 
               c=colors, marker=marker, s=120, alpha=0.7, 
               edgecolors='black', linewidth=1.5)

ax.axhline(1, color='green', linestyle='--', alpha=0.5, linewidth=2, label='1% target')
ax.set_xlabel('Number of Variables', fontsize=13, fontweight='bold')
ax.set_ylabel('MIP Gap (%)', fontsize=13, fontweight='bold')
ax.set_title('Solution Quality (MIP Gap)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# ============================================================================
# Plot 5 (Middle Center): Quadratic Density vs Solve Time
# ============================================================================
ax = fig.add_subplot(gs[1, 1])

for test_type in df_combined['test_type'].unique():
    df_src = df_combined[df_combined['test_type'] == test_type]
    colors = [color_map.get(cat, 'gray') for cat in df_src['time_category']]
    marker = marker_map.get(test_type, 'o')
    
    ax.scatter(df_src['quadratic_density'], df_src['solve_time'], 
               c=colors, marker=marker, s=120, alpha=0.7, 
               edgecolors='black', linewidth=1.5)

ax.set_xlabel('Quadratic Terms per Variable', fontsize=13, fontweight='bold')
ax.set_ylabel('Solve Time (seconds)', fontsize=13, fontweight='bold')
ax.set_title('Quadratic Density vs Hardness', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# ============================================================================
# Plot 6 (Middle Right): Build Time vs Solve Time
# ============================================================================
ax = fig.add_subplot(gs[1, 2])

for test_type in df_combined['test_type'].unique():
    df_src = df_combined[df_combined['test_type'] == test_type]
    colors = [color_map.get(cat, 'gray') for cat in df_src['time_category']]
    marker = marker_map.get(test_type, 'o')
    
    ax.scatter(df_src['build_time'], df_src['solve_time'], 
               c=colors, marker=marker, s=120, alpha=0.7, 
               edgecolors='black', linewidth=1.5)

# Add diagonal line (build = solve)
max_time = max(df_combined['build_time'].max(), df_combined['solve_time'].max())
ax.plot([0, max_time], [0, max_time], 'k--', alpha=0.3, linewidth=1, label='Build=Solve')

ax.set_xlabel('Build Time (seconds)', fontsize=13, fontweight='bold')
ax.set_ylabel('Solve Time (seconds)', fontsize=13, fontweight='bold')
ax.set_title('Model Construction vs Solving', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# ============================================================================
# Plot 7 (Bottom Left): Total Area Scaling
# ============================================================================
ax = fig.add_subplot(gs[2, 0])

for test_type in df_combined['test_type'].unique():
    df_src = df_combined[df_combined['test_type'] == test_type]
    colors = [color_map.get(cat, 'gray') for cat in df_src['time_category']]
    marker = marker_map.get(test_type, 'o')
    
    ax.scatter(df_src['n_farms'], df_src['total_area'], 
               c=colors, marker=marker, s=120, alpha=0.7, 
               edgecolors='black', linewidth=1.5)
    
    # Add expected scaling lines based on area values
    # If area varies with farms, show scaling line
    if df_src['total_area'].std() > 10:  # Variable area
        expected_area = df_src['n_farms'].values * (df_src['total_area'].mean() / df_src['n_farms'].mean())
        ax.plot(df_src['n_farms'], expected_area, '--', 
                alpha=0.5, linewidth=2, label=f'{test_type}: Variable area')
    else:  # Constant area
        if len(df_src) > 0:
            constant_val = df_src['total_area'].iloc[0]
            ax.axhline(constant_val, linestyle='--', 
                       alpha=0.5, linewidth=2, label=f'{test_type}: {constant_val:.0f} ha')

ax.set_xlabel('Number of Farms', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Area (hectares)', fontsize=13, fontweight='bold')
ax.set_title('Area Normalization Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# ============================================================================
# Plot 8 (Bottom Center): Solve Time Distribution Histogram
# ============================================================================
ax = fig.add_subplot(gs[2, 1])

for test_type in df_combined['test_type'].unique():
    df_src = df_combined[df_combined['test_type'] == test_type]
    ax.hist(df_src['solve_time'], bins=20, alpha=0.5, 
            label=test_type, edgecolor='black')

ax.set_xlabel('Solve Time (seconds)', fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax.set_title('Solve Time Distribution', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# ============================================================================
# Plot 9 (Bottom Right): Objective Value Distribution
# ============================================================================
ax = fig.add_subplot(gs[2, 2])

for test_type in df_combined['test_type'].unique():
    df_src = df_combined[df_combined['test_type'] == test_type]
    ax.hist(df_src['obj_value'], bins=20, alpha=0.5, 
            label=test_type, edgecolor='black')

ax.set_xlabel('Objective Value', fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax.set_title('Objective Value Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# ============================================================================
# Custom legend for time categories and marker shapes
# ============================================================================
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Time category legend (colors)
time_legend_elements = [
    Patch(facecolor='green', edgecolor='black', label='FAST (<10s)'),
    Patch(facecolor='orange', edgecolor='black', label='MEDIUM (10-100s)'),
    Patch(facecolor='red', edgecolor='black', label='SLOW (>100s)'),
    Patch(facecolor='darkred', edgecolor='black', label='TIMEOUT')
]

# Test type legend (markers) - only show types with data
marker_legend_elements = []
for test_type in df_combined['test_type'].unique():
    marker = marker_map.get(test_type, 'o')
    if test_type == 'Per-Farm Area (1 ha/farm)':
        marker_legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markeredgecolor='black', markersize=10, label=test_type))
    elif test_type == 'Total Area (100 ha)':
        marker_legend_elements.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                   markeredgecolor='black', markersize=10, label=test_type))
    elif test_type == 'Roadmap Gurobi':
        marker_legend_elements.append(
            Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
                   markeredgecolor='black', markersize=10, label=test_type))
    elif test_type == 'Hierarchical Gurobi':
        marker_legend_elements.append(
            Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', 
                   markeredgecolor='black', markersize=8, label=test_type))
    elif test_type == 'Statistical Gurobi':
        marker_legend_elements.append(
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
                   markeredgecolor='black', markersize=12, label=test_type))

# Add both legends
legend1 = fig.legend(handles=time_legend_elements, loc='lower left', ncol=4, 
                     fontsize=11, frameon=True, title='Time Categories',
                     bbox_to_anchor=(0.02, -0.01))
if len(marker_legend_elements) > 0:
    legend2 = fig.legend(handles=marker_legend_elements, loc='lower right', 
                         ncol=len(marker_legend_elements), 
                         fontsize=11, frameon=True, title='Normalization Strategy',
                         bbox_to_anchor=(0.98, -0.01))

# Save plot
output_plot = results_dir / 'comprehensive_hardness_scaling_INTEGRATED.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\n✓ Integrated comprehensive plot saved to: {output_plot}")

plt.savefig(output_plot.with_suffix('.pdf'), bbox_inches='tight')
print(f"✓ PDF version saved to: {output_plot.with_suffix('.pdf')}")

plt.close()

# ============================================================================
# Print Comparison Statistics
# ============================================================================
print("\n" + "="*80)
print("COMPARATIVE STATISTICS")
print("="*80)

if has_comparison:
    for test_type in df_combined['test_type'].unique():
        df_src = df_combined[df_combined['test_type'] == test_type]
        print(f"\n--- {test_type} ---")
        print(f"Data points: {len(df_src)}")
        print(f"Farm range: {df_src['n_farms'].min()}-{df_src['n_farms'].max()}")
        print(f"Area range: {df_src['total_area'].min():.1f}-{df_src['total_area'].max():.1f} ha")
        print(f"Solve time: {df_src['solve_time'].min():.2f}-{df_src['solve_time'].max():.2f}s (mean={df_src['solve_time'].mean():.2f}s)")
        print(f"Objective: {df_src['obj_value'].min():.3f}-{df_src['obj_value'].max():.3f} (mean={df_src['obj_value'].mean():.3f})")
        
        # Count by category
        print("\nBy category:")
        for category in ['FAST', 'MEDIUM', 'SLOW', 'TIMEOUT']:
            count = len(df_src[df_src['time_category'] == category])
            if count > 0:
                print(f"  {category}: {count} instances")

print("\n" + "="*80)
print("INTEGRATED ANALYSIS COMPLETE")
print("="*80)
