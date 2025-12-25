#!/usr/bin/env python3
"""
Plot all benchmark results: solve time, objective value, and gap from quantum to classical
Combines data from:
- Timeout test (test_gurobi_timeout.py)  
- Hierarchical test (hierarchical_statistical_test.py)
- Comprehensive scaling test (comprehensive_scaling_test.py)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Set up professional plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 12)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# =============================================================================
# Load Data
# =============================================================================

# 1. Comprehensive scaling test results
scaling_file = Path("scaling_test_results/scaling_test_1766570771.json")
with open(scaling_file) as f:
    scaling_data = json.load(f)

# 2. Hierarchical test results  
hierarchical_csv = Path("hierarchical_statistical_results/summary_20251224_095741.csv")
hierarchical_df = pd.read_csv(hierarchical_csv)

# =============================================================================
# Process Data into DataFrames
# =============================================================================

# Comprehensive scaling data
scaling_records = []
for entry in scaling_data:
    scaling_records.append({
        'n_vars': entry['n_vars'],
        'source': 'Comprehensive',
        'variant': entry['formulation'],
        'gurobi_time': entry['gurobi_time'],
        'gurobi_obj': entry['gurobi_obj'],
        'gurobi_gap': entry['gurobi_gap'],
        'quantum_time': entry['quantum_time'],
        'quantum_obj': entry['quantum_obj'],
        'qpu_time': entry['qpu_time'],
        'gap_to_classical': entry['gap'],
        'speedup': entry['speedup']
    })
scaling_df = pd.DataFrame(scaling_records)

# Hierarchical data
hierarchical_records = []
for _, row in hierarchical_df.iterrows():
    hierarchical_records.append({
        'n_vars': row['n_vars'],
        'source': 'Hierarchical',
        'variant': 'Hierarchical QPU',
        'gurobi_time': row['gurobi_time'],
        'gurobi_obj': row['gurobi_obj'],
        'quantum_time': row['quantum_time'],
        'quantum_obj': row['quantum_obj'],
        'qpu_time': row['qpu_time'],
        'gap_to_classical': row['gap_percent'],
        'speedup': row['speedup']
    })
hierarchical_df_processed = pd.DataFrame(hierarchical_records)

# Combine all data
all_data = pd.concat([scaling_df, hierarchical_df_processed], ignore_index=True)

# =============================================================================
# Create Plots
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Color scheme
colors = {
    'Native 6-Family': '#1f77b4',      # Blue
    '27->6 Aggregated': '#ff7f0e',     # Orange
    '27-Food Hybrid': '#2ca02c',        # Green
    'Hierarchical QPU': '#d62728',      # Red
    'Gurobi': '#9467bd',                # Purple
}

markers = {
    'Native 6-Family': 'o',
    '27->6 Aggregated': 's',
    '27-Food Hybrid': '^',
    'Hierarchical QPU': 'D',
    'Gurobi': 'x',
}

# =============================================================================
# Plot 1: Solve Time vs Variables
# =============================================================================
ax1 = axes[0, 0]

# Plot quantum times by variant
for variant in all_data['variant'].unique():
    subset = all_data[all_data['variant'] == variant].sort_values('n_vars')
    ax1.plot(subset['n_vars'], subset['quantum_time'], 
             marker=markers.get(variant, 'o'), 
             color=colors.get(variant, 'gray'),
             label=f'{variant} (Quantum)', linewidth=2, markersize=8)

# Plot Gurobi times (use one representative)
gurobi_data = scaling_df[scaling_df['variant'] == 'Native 6-Family'].sort_values('n_vars')
ax1.plot(gurobi_data['n_vars'], gurobi_data['gurobi_time'],
         marker='x', color='#9467bd', linestyle='--',
         label='Gurobi (300s timeout)', linewidth=2, markersize=10)

# Add hierarchical Gurobi
hier_data = hierarchical_df_processed.sort_values('n_vars')
ax1.plot(hier_data['n_vars'], hier_data['gurobi_time'],
         marker='+', color='#8c564b', linestyle=':',
         label='Gurobi (Hierarchical)', linewidth=2, markersize=10)

ax1.set_xlabel('Number of Variables')
ax1.set_ylabel('Solve Time (seconds)')
ax1.set_title('Solve Time vs Problem Size')
ax1.legend(loc='upper left', framealpha=0.9)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)

# =============================================================================
# Plot 2: Objective Value vs Variables
# =============================================================================
ax2 = axes[0, 1]

# Plot quantum objectives by variant
for variant in all_data['variant'].unique():
    subset = all_data[all_data['variant'] == variant].sort_values('n_vars')
    ax2.plot(subset['n_vars'], subset['quantum_obj'], 
             marker=markers.get(variant, 'o'), 
             color=colors.get(variant, 'gray'),
             label=f'{variant} (Quantum)', linewidth=2, markersize=8)

# Plot Gurobi objectives
gurobi_data = scaling_df[scaling_df['variant'] == 'Native 6-Family'].sort_values('n_vars')
ax2.plot(gurobi_data['n_vars'], gurobi_data['gurobi_obj'],
         marker='x', color='#9467bd', linestyle='--',
         label='Gurobi (Native 6)', linewidth=2, markersize=10)

# Add hierarchical Gurobi
ax2.plot(hier_data['n_vars'], hier_data['gurobi_obj'],
         marker='+', color='#8c564b', linestyle=':',
         label='Gurobi (Hierarchical)', linewidth=2, markersize=10)

ax2.set_xlabel('Number of Variables')
ax2.set_ylabel('Objective Value (normalized)')
ax2.set_title('Objective Value vs Problem Size')
ax2.legend(loc='upper left', framealpha=0.9)
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)

# =============================================================================
# Plot 3: Gap (Quantum vs Classical) vs Variables
# =============================================================================
ax3 = axes[1, 0]

# Plot gap by variant
for variant in all_data['variant'].unique():
    subset = all_data[all_data['variant'] == variant].sort_values('n_vars')
    ax3.plot(subset['n_vars'], subset['gap_to_classical'], 
             marker=markers.get(variant, 'o'), 
             color=colors.get(variant, 'gray'),
             label=variant, linewidth=2, markersize=8)

ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax3.set_xlabel('Number of Variables')
ax3.set_ylabel('Gap to Classical (%)')
ax3.set_title('Solution Quality Gap: (Gurobi - Quantum) / Gurobi × 100')
ax3.legend(loc='best', framealpha=0.9)
ax3.set_xscale('log')
ax3.grid(True, alpha=0.3)

# Add annotation
ax3.annotate('Higher = Quantum finds worse solutions\n(Gurobi timeout at 300s affects results)',
             xy=(0.5, 0.95), xycoords='axes fraction',
             fontsize=9, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# =============================================================================
# Plot 4: Speedup vs Variables
# =============================================================================
ax4 = axes[1, 1]

# Plot speedup by variant
for variant in all_data['variant'].unique():
    subset = all_data[all_data['variant'] == variant].sort_values('n_vars')
    ax4.plot(subset['n_vars'], subset['speedup'], 
             marker=markers.get(variant, 'o'), 
             color=colors.get(variant, 'gray'),
             label=variant, linewidth=2, markersize=8)

ax4.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Break-even')
ax4.set_xlabel('Number of Variables')
ax4.set_ylabel('Speedup (Gurobi time / Quantum time)')
ax4.set_title('Speedup vs Problem Size')
ax4.legend(loc='best', framealpha=0.9)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)

# Add annotation for speedup > 1
ax4.annotate('Speedup > 1: Quantum faster\nSpeedup < 1: Classical faster',
             xy=(0.02, 0.02), xycoords='axes fraction',
             fontsize=9, ha='left', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# =============================================================================
# Final Layout
# =============================================================================
plt.suptitle('Comprehensive Benchmark Results: Quantum vs Classical Solvers\n'
             'Rotation Crop Planning Problem', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save plots
output_dir = Path("benchmark_plots")
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / "comprehensive_benchmark_all.png", dpi=150, bbox_inches='tight')
plt.savefig(output_dir / "comprehensive_benchmark_all.pdf", bbox_inches='tight')

print(f"✅ Plots saved to {output_dir}/")

# =============================================================================
# Print Summary Table
# =============================================================================
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

# Create summary
summary = all_data.groupby(['n_vars', 'variant']).agg({
    'gurobi_time': 'mean',
    'quantum_time': 'mean',
    'gurobi_obj': 'mean',
    'quantum_obj': 'mean',
    'gap_to_classical': 'mean',
    'speedup': 'mean'
}).round(2)

print(summary.to_string())

# Additional statistics
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print("\n📊 Data Sources:")
print(f"   - Comprehensive scaling test: {len(scaling_df)} data points")
print(f"   - Hierarchical test: {len(hierarchical_df_processed)} data points")

print("\n⏱️ Solve Times:")
max_quantum = all_data['quantum_time'].max()
max_gurobi = all_data['gurobi_time'].max()
print(f"   - Max Quantum time: {max_quantum:.1f}s")
print(f"   - Max Gurobi time: {max_gurobi:.1f}s (timeout)")

print("\n📈 Speedup Range:")
print(f"   - Min speedup: {all_data['speedup'].min():.2f}×")
print(f"   - Max speedup: {all_data['speedup'].max():.2f}×")
print(f"   - Mean speedup: {all_data['speedup'].mean():.2f}×")

print("\n🎯 Solution Quality Gap:")
print(f"   - Min gap: {all_data['gap_to_classical'].min():.1f}%")
print(f"   - Max gap: {all_data['gap_to_classical'].max():.1f}%")
print(f"   - Mean gap: {all_data['gap_to_classical'].mean():.1f}%")

# Show the plot
plt.show()
