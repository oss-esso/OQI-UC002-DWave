#!/usr/bin/env python3
"""
Beautiful hardness scaling plot with enhanced aesthetics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
results_dir = Path(__file__).parent / 'hardness_analysis_results'

# Load all datasets
df_list = []

# Load original results
try:
    df_per_farm = pd.read_csv(results_dir / 'hardness_results_per_farm_area.csv')
    df_per_farm['test_type'] = 'Per-Farm Area (1 ha/farm)'
    df_list.append(df_per_farm)
    print(f"Loaded {len(df_per_farm)} per-farm area results")
except FileNotFoundError:
    print("No per-farm area data found")

try:
    df_total = pd.read_csv(results_dir / 'hardness_results_total_area.csv')
    df_total['test_type'] = 'Total Area (100 ha)'
    df_list.append(df_total)
    print(f"Loaded {len(df_total)} total area results")
except FileNotFoundError:
    print("No total area data found")

# Load additional Gurobi results
try:
    df_additional = pd.read_csv(results_dir / 'additional_gurobi_results.csv')
    df_list.append(df_additional)
    print(f"Loaded {len(df_additional)} additional Gurobi results")
except FileNotFoundError:
    print("No additional Gurobi data found")

# Combine all
df = pd.concat(df_list, ignore_index=True)

# Re-categorize with 100s threshold
def categorize_time(t):
    if t < 10:
        return 'FAST'
    elif t < 100:
        return 'MEDIUM'
    else:
        return 'TIMEOUT'

df['time_category'] = df['solve_time'].apply(categorize_time)

# Export full dataset for analysis
output_csv = results_dir / 'all_datapoints_for_analysis.csv'
df.to_csv(output_csv, index=False)
print(f"\n✓ Exported {len(df)} datapoints to: {output_csv}")

# Create beautiful plot
fig, ax = plt.subplots(figsize=(18, 12))
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# Enhanced color scheme
colors = {
    'FAST': '#27ae60',      # Forest green
    'MEDIUM': '#f39c12',    # Vibrant orange  
    'TIMEOUT': '#c0392b'    # Crimson red
}

# Marker shapes
markers = {
    'Per-Farm Area (1 ha/farm)': 'o',   # Circle
    'Total Area (100 ha)': 's',         # Square
    'Roadmap Gurobi': '^',              # Triangle
    'Hierarchical Gurobi': 'D',         # Diamond
    'Statistical Gurobi': '*'           # Star
}

# Plot each test type
for test_type in sorted(df['test_type'].unique()):
    df_test = df[df['test_type'] == test_type]
    
    # Plot each category with different colors
    for category in ['FAST', 'MEDIUM', 'TIMEOUT']:
        df_cat = df_test[df_test['time_category'] == category]
        if len(df_cat) == 0:
            continue
            
        ax.scatter(
            df_cat['n_farms'],
            df_cat['solve_time'],
            c=colors[category],
            marker=markers.get(test_type, 'o'),
            s=250,
            alpha=0.75,
            edgecolors='black',
            linewidths=2.5,
            label=f'{test_type} - {category}' if test_type == 'Per-Farm Area (1 ha/farm)' else '',
            zorder=3
        )

# Add horizontal threshold lines
ax.axhline(y=10, color='#27ae60', linestyle='--', linewidth=2.5, alpha=0.4, label='Fast threshold (10s)')
ax.axhline(y=100, color='#c0392b', linestyle='--', linewidth=2.5, alpha=0.4, label='Timeout threshold (100s)')

# Add shaded regions
ax.axhspan(0, 10, alpha=0.05, color='#27ae60', zorder=1)
ax.axhspan(10, 100, alpha=0.05, color='#f39c12', zorder=1)
ax.axhspan(100, df['solve_time'].max() * 1.1, alpha=0.05, color='#c0392b', zorder=1)

# Logarithmic scale
ax.set_yscale('log')
ax.set_ylim(0.1, df['solve_time'].max() * 1.2)

# Grid
ax.grid(True, alpha=0.3, linestyle=':', linewidth=1.2, color='#95a5a6', which='both')
ax.set_axisbelow(True)

# Labels and title
ax.set_xlabel('Number of Farms', fontsize=18, fontweight='bold', color='#2c3e50')
ax.set_ylabel('Solve Time (seconds, log scale)', fontsize=18, fontweight='bold', color='#2c3e50')
ax.set_title('Comprehensive Hardness Scaling Analysis\nGurobi MIP Solver Performance Across 75 Test Instances', 
             fontsize=22, fontweight='bold', pad=25, color='#2c3e50')

# Legend for markers (test types)
from matplotlib.lines import Line2D
marker_legend = []
for test_type in sorted(df['test_type'].unique()):
    marker = markers.get(test_type, 'o')
    marker_legend.append(
        Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=14, linewidth=2.5, label=test_type)
    )

# Legend for colors (categories)
color_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['FAST'],
           markeredgecolor='black', markersize=14, linewidth=2.5, label='Fast (< 10s)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['MEDIUM'],
           markeredgecolor='black', markersize=14, linewidth=2.5, label='Medium (10-100s)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['TIMEOUT'],
           markeredgecolor='black', markersize=14, linewidth=2.5, label='Timeout (> 100s)')
]

# Create two legends
legend1 = ax.legend(handles=marker_legend, loc='upper left', fontsize=11, framealpha=0.95,
                    title='Test Type', title_fontsize=12, shadow=True)
legend2 = ax.legend(handles=color_legend, loc='lower right', fontsize=11, framealpha=0.95,
                    title='Solve Time Category', title_fontsize=12, shadow=True)
ax.add_artist(legend1)

# Tick params
ax.tick_params(labelsize=13, width=2, length=6)

# Tight layout
plt.tight_layout()

# Save
output_png = results_dir / 'beautiful_hardness_scaling.png'
output_pdf = results_dir / 'beautiful_hardness_scaling.pdf'
plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')

print(f"\n✓ Beautiful plot saved to:")
print(f"  PNG: {output_png}")
print(f"  PDF: {output_pdf}")

# Summary stats by category
print("\n" + "="*80)
print("SUMMARY STATISTICS BY CATEGORY (100s threshold)")
print("="*80)

for category in ['FAST', 'MEDIUM', 'TIMEOUT']:
    df_cat = df[df['time_category'] == category]
    if len(df_cat) == 0:
        continue
    print(f"\n{category}: {len(df_cat)} instances")
    print(f"  Farm range: {df_cat['n_farms'].min()}-{df_cat['n_farms'].max()}")
    print(f"  Solve time: {df_cat['solve_time'].min():.2f}-{df_cat['solve_time'].max():.2f}s")
    print(f"  By test type:")
    for test_type in df_cat['test_type'].unique():
        count = len(df_cat[df_cat['test_type'] == test_type])
        print(f"    {test_type}: {count}")

plt.show()
