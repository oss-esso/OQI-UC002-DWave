#!/usr/bin/env python3
"""
Plot all benchmark results: solve time, objective value, gap, and violations
Combines data from:
- Comprehensive scaling test with full solution details
- Hierarchical test
- Previous tests

Author: OQI-UC002-DWave
Date: 2025-12-24
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

# Set up professional plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 9

# =============================================================================
# Load Data from ALL sources
# =============================================================================

BASE_DIR = Path(__file__).parent

# 1. Load comprehensive scaling test results (old)
scaling_files = sorted(glob(str(BASE_DIR / "scaling_test_results/scaling_test_*.json")))
scaling_data = []
for f in scaling_files[-3:]:  # Last 3 files
    try:
        with open(f) as fp:
            data = json.load(fp)
            if isinstance(data, list):
                scaling_data.extend(data)
    except:
        pass

# 2. Load comprehensive full results (new)
full_result_files = sorted(glob(str(BASE_DIR / "comprehensive_full_results/comprehensive_results_*.csv")))
full_data = []
for f in full_result_files:
    try:
        df = pd.read_csv(f)
        full_data.append(df)
    except:
        pass

# 3. Load hierarchical test results
hierarchical_csv = BASE_DIR / "hierarchical_statistical_results/summary_20251224_095741.csv"
if hierarchical_csv.exists():
    hierarchical_df = pd.read_csv(hierarchical_csv)
else:
    hierarchical_df = pd.DataFrame()

# =============================================================================
# Process and Combine Data
# =============================================================================

all_records = []

# Process scaling data (old format)
for entry in scaling_data:
    if isinstance(entry, dict):
        all_records.append({
            'n_vars': entry.get('n_vars', 0),
            'source': 'Comprehensive',
            'variant': entry.get('formulation', entry.get('variant', 'Unknown')),
            'gurobi_time': entry.get('gurobi_time', 0),
            'gurobi_obj': entry.get('gurobi_obj', 0),
            'gurobi_gap': entry.get('gurobi_gap', 0),
            'gurobi_violations': entry.get('gurobi_violations', 0),
            'quantum_time': entry.get('quantum_time', 0),
            'quantum_obj': entry.get('quantum_obj', 0),
            'qpu_time': entry.get('qpu_time', 0),
            'quantum_violations': entry.get('quantum_violations', 0),
            'gap_to_classical': entry.get('gap', 0),
            'speedup': entry.get('speedup', 0),
        })

# Process full results (new format)
for df in full_data:
    for _, row in df.iterrows():
        all_records.append({
            'n_vars': row.get('n_vars', 0),
            'source': 'Comprehensive-Full',
            'variant': row.get('scenario', 'Unknown'),
            'gurobi_time': row.get('gurobi_time', 0),
            'gurobi_obj': row.get('gurobi_obj', 0),
            'gurobi_gap': row.get('gurobi_gap', 0),
            'gurobi_violations': row.get('gurobi_violations', 0),
            'quantum_time': row.get('quantum_time', 0),
            'quantum_obj': row.get('quantum_obj', 0),
            'qpu_time': row.get('qpu_time', 0),
            'quantum_violations': row.get('quantum_violations', 0),
            'gap_to_classical': row.get('gap', 0),
            'speedup': row.get('speedup', 0),
        })

# Process hierarchical data
if not hierarchical_df.empty:
    for _, row in hierarchical_df.iterrows():
        all_records.append({
            'n_vars': row.get('n_vars', 0),
            'source': 'Hierarchical',
            'variant': 'Hierarchical QPU',
            'gurobi_time': row.get('gurobi_time', 0),
            'gurobi_obj': row.get('gurobi_obj', 0),
            'gurobi_gap': 0,
            'gurobi_violations': 0,
            'quantum_time': row.get('quantum_time', 0),
            'quantum_obj': row.get('quantum_obj', 0),
            'qpu_time': row.get('qpu_time', 0),
            'quantum_violations': 0,
            'gap_to_classical': row.get('gap_percent', 0),
            'speedup': row.get('speedup', 0),
        })

# Create DataFrame
all_data = pd.DataFrame(all_records)

if all_data.empty:
    print("❌ No data found! Please run the tests first.")
    exit(1)

# Remove duplicates and sort
all_data = all_data.drop_duplicates(subset=['n_vars', 'variant'])
all_data = all_data.sort_values('n_vars')

print(f"Loaded {len(all_data)} data points")
print(f"Variables range: {all_data['n_vars'].min()} - {all_data['n_vars'].max()}")
print(f"Sources: {all_data['source'].unique()}")
print(f"Variants: {all_data['variant'].unique()}")

# =============================================================================
# Create Plots (2x3 grid)
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Color scheme by source/variant type
def get_color_marker(variant):
    """Get color and marker based on variant type."""
    if '6-Family' in variant or '6foods' in variant:
        return '#1f77b4', 'o', 'blue'  # Blue
    elif 'Aggregated' in variant:
        return '#ff7f0e', 's', 'orange'  # Orange  
    elif 'Hybrid' in variant or '27foods' in variant:
        return '#2ca02c', '^', 'green'  # Green
    elif 'Hierarchical' in variant:
        return '#d62728', 'D', 'red'  # Red
    else:
        return '#9467bd', 'x', 'purple'  # Purple

# =============================================================================
# Plot 1: Solve Time vs Variables
# =============================================================================
ax1 = axes[0, 0]

# Group by variant and plot
for variant in all_data['variant'].unique():
    subset = all_data[all_data['variant'] == variant].sort_values('n_vars')
    color, marker, _ = get_color_marker(variant)
    ax1.plot(subset['n_vars'], subset['quantum_time'], 
             marker=marker, color=color, label=f'{variant} (Q)', 
             linewidth=1.5, markersize=6, alpha=0.8)

# Add Gurobi reference line (average across variants)
gurobi_times = all_data.groupby('n_vars')['gurobi_time'].mean().reset_index()
ax1.plot(gurobi_times['n_vars'], gurobi_times['gurobi_time'],
         'k--', label='Gurobi (avg)', linewidth=2, markersize=8)

ax1.set_xlabel('Number of Variables')
ax1.set_ylabel('Solve Time (seconds)')
ax1.set_title('Solve Time vs Problem Size')
ax1.legend(loc='upper left', fontsize=8, ncol=2)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)

# =============================================================================
# Plot 2: Objective Value vs Variables
# =============================================================================
ax2 = axes[0, 1]

for variant in all_data['variant'].unique():
    subset = all_data[all_data['variant'] == variant].sort_values('n_vars')
    color, marker, _ = get_color_marker(variant)
    # Quantum
    ax2.plot(subset['n_vars'], subset['quantum_obj'], 
             marker=marker, color=color, label=f'{variant} (Q)',
             linewidth=1.5, markersize=6, alpha=0.8)
    # Gurobi (dashed)
    ax2.plot(subset['n_vars'], subset['gurobi_obj'], 
             marker=marker, color=color, linestyle='--', 
             linewidth=1, markersize=4, alpha=0.5)

ax2.set_xlabel('Number of Variables')
ax2.set_ylabel('Objective Value')
ax2.set_title('Objective Value vs Problem Size\n(solid=Quantum, dashed=Gurobi)')
ax2.legend(loc='upper left', fontsize=8, ncol=2)
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)

# =============================================================================
# Plot 3: Gap (Quantum vs Classical) vs Variables
# =============================================================================
ax3 = axes[0, 2]

for variant in all_data['variant'].unique():
    subset = all_data[all_data['variant'] == variant].sort_values('n_vars')
    color, marker, _ = get_color_marker(variant)
    ax3.plot(subset['n_vars'], subset['gap_to_classical'], 
             marker=marker, color=color, label=variant,
             linewidth=1.5, markersize=6, alpha=0.8)

ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax3.set_xlabel('Number of Variables')
ax3.set_ylabel('Gap to Classical (%)')
ax3.set_title('Solution Quality Gap\n(Gurobi - Quantum) / Gurobi × 100')
ax3.legend(loc='best', fontsize=8)
ax3.set_xscale('log')
ax3.grid(True, alpha=0.3)

# =============================================================================
# Plot 4: Violations vs Variables (NEW!)
# =============================================================================
ax4 = axes[1, 0]

for variant in all_data['variant'].unique():
    subset = all_data[all_data['variant'] == variant].sort_values('n_vars')
    color, marker, _ = get_color_marker(variant)
    
    # Quantum violations
    ax4.plot(subset['n_vars'], subset['quantum_violations'], 
             marker=marker, color=color, label=f'{variant} (Q)',
             linewidth=1.5, markersize=6, alpha=0.8)
    
    # Gurobi violations (dashed)
    ax4.plot(subset['n_vars'], subset['gurobi_violations'], 
             marker=marker, color=color, linestyle='--',
             linewidth=1, markersize=4, alpha=0.5)

ax4.axhline(y=0, color='green', linestyle='-', linewidth=1, alpha=0.7, label='Valid (0)')
ax4.set_xlabel('Number of Variables')
ax4.set_ylabel('Constraint Violations')
ax4.set_title('Constraint Violations vs Problem Size\n(solid=Quantum, dashed=Gurobi)')
ax4.legend(loc='best', fontsize=8)
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)

# Annotation
ax4.annotate('Higher violations may explain\nbetter objectives (relaxed constraints)',
             xy=(0.02, 0.98), xycoords='axes fraction',
             fontsize=8, ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# =============================================================================
# Plot 5: Speedup vs Variables
# =============================================================================
ax5 = axes[1, 1]

for variant in all_data['variant'].unique():
    subset = all_data[all_data['variant'] == variant].sort_values('n_vars')
    color, marker, _ = get_color_marker(variant)
    ax5.plot(subset['n_vars'], subset['speedup'], 
             marker=marker, color=color, label=variant,
             linewidth=1.5, markersize=6, alpha=0.8)

ax5.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Break-even')
ax5.set_xlabel('Number of Variables')
ax5.set_ylabel('Speedup (Gurobi time / Quantum time)')
ax5.set_title('Speedup vs Problem Size')
ax5.legend(loc='best', fontsize=8)
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.grid(True, alpha=0.3)

# =============================================================================
# Plot 6: Gap vs Violations (Correlation)
# =============================================================================
ax6 = axes[1, 2]

# Scatter plot: x=violations, y=gap, colored by variant
for variant in all_data['variant'].unique():
    subset = all_data[all_data['variant'] == variant]
    color, marker, _ = get_color_marker(variant)
    ax6.scatter(subset['quantum_violations'], subset['gap_to_classical'],
               c=color, marker=marker, label=variant, s=60, alpha=0.7)

ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax6.axvline(x=0, color='green', linestyle='--', linewidth=1, alpha=0.7)
ax6.set_xlabel('Quantum Constraint Violations')
ax6.set_ylabel('Gap to Classical (%)')
ax6.set_title('Gap vs Violations\n(Does higher gap come from violations?)')
ax6.legend(loc='best', fontsize=8)
ax6.grid(True, alpha=0.3)

# Annotation
ax6.annotate('Points with violations > 0\nmay have artificially better objectives',
             xy=(0.98, 0.02), xycoords='axes fraction',
             fontsize=8, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

# =============================================================================
# Final Layout
# =============================================================================
plt.suptitle('Comprehensive Benchmark: Quantum vs Classical Solvers\n'
             'Crop Rotation Planning - Full Analysis with Violations', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# Save plots
output_dir = BASE_DIR / "benchmark_plots"
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / "benchmark_with_violations.png", dpi=150, bbox_inches='tight')
plt.savefig(output_dir / "benchmark_with_violations.pdf", bbox_inches='tight')

print(f"\n✅ Plots saved to {output_dir}/")

# =============================================================================
# Print Summary Statistics
# =============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\n📊 Data Points by Source:")
print(all_data.groupby('source').size())

print("\n📊 Data Points by Variant:")
print(all_data.groupby('variant').size())

print("\n⏱️ Solve Time Range:")
print(f"   Quantum: {all_data['quantum_time'].min():.1f}s - {all_data['quantum_time'].max():.1f}s")
print(f"   Gurobi:  {all_data['gurobi_time'].min():.1f}s - {all_data['gurobi_time'].max():.1f}s")

print("\n📈 Speedup:")
print(f"   Min: {all_data['speedup'].min():.2f}×")
print(f"   Max: {all_data['speedup'].max():.2f}×")
print(f"   Mean: {all_data['speedup'].mean():.2f}×")

print("\n🎯 Gap to Classical:")
print(f"   Min: {all_data['gap_to_classical'].min():.1f}%")
print(f"   Max: {all_data['gap_to_classical'].max():.1f}%")
print(f"   Mean: {all_data['gap_to_classical'].mean():.1f}%")

print("\n⚠️ Violations:")
print(f"   Quantum violations: {all_data['quantum_violations'].sum():.0f} total")
print(f"   Gurobi violations:  {all_data['gurobi_violations'].sum():.0f} total")

# Show the plot
plt.show()
