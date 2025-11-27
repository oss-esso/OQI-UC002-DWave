#!/usr/bin/env python3
"""
Generate comparison plots for comprehensive benchmark results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load latest results
results_dir = Path(__file__).parent / "benchmark_results"
json_files = list(results_dir.glob("comprehensive_benchmark_*.json"))
latest = max(json_files, key=lambda p: p.stat().st_mtime)

with open(latest) as f:
    data = json.load(f)

results = data['results']

# Convert to DataFrame
df_data = []
for r in results:
    row = {
        'formulation': r['formulation'],
        'n_units': r['n_units'],
        'decomposition': r.get('decomposition', 'None'),
        'density': r.get('metadata', {}).get('density'),
        'solve_time': r.get('solving', {}).get('solve_time'),
    }
    
    # Embedding success
    emb = r.get('embedding')
    if emb and isinstance(emb, dict):
        row['embed_success'] = emb.get('success', False) if not emb.get('skipped') else None
        row['embed_time'] = emb.get('embedding_time')
    
    df_data.append(row)

df = pd.DataFrame(df_data)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Density vs Embedding Success
ax1 = axes[0, 0]
embed_df = df[df['embed_success'].notna() & df['density'].notna()].copy()
colors = ['green' if x else 'red' for x in embed_df['embed_success']]
ax1.scatter(embed_df['density'], embed_df['n_units'], c=colors, s=100, alpha=0.6)
ax1.axvline(x=0.3, color='orange', linestyle='--', label='Density threshold (30%)')
ax1.set_xlabel('Density', fontsize=12)
ax1.set_ylabel('Problem Size (units)', fontsize=12)
ax1.set_title('Embedding Success by Density', fontsize=14, fontweight='bold')
ax1.legend(['Threshold', 'Failed', 'Success'])
ax1.grid(True, alpha=0.3)

# Plot 2: Solve Time by Formulation
ax2 = axes[0, 1]
solve_df = df[df['solve_time'].notna() & (df['solve_time'] < 100)].copy()
solve_pivot = solve_df.groupby(['formulation', 'n_units'])['solve_time'].mean().reset_index()
for form in solve_pivot['formulation'].unique():
    form_data = solve_pivot[solve_pivot['formulation'] == form]
    ax2.plot(form_data['n_units'], form_data['solve_time'], marker='o', label=form)
ax2.set_xlabel('Problem Size (units)', fontsize=12)
ax2.set_ylabel('Solve Time (seconds)', fontsize=12)
ax2.set_title('Gurobi Solve Time by Formulation', fontsize=14, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Plot 3: Decomposition Strategy Comparison
ax3 = axes[1, 0]
decomp_df = df[df['decomposition'] != 'None'].copy()
decomp_summary = decomp_df.groupby('decomposition').agg({
    'formulation': 'count',
    'embed_success': lambda x: x.sum() if x.notna().any() else 0
}).reset_index()
decomp_summary.columns = ['Strategy', 'Total', 'Successful']

x_pos = range(len(decomp_summary))
ax3.bar([i - 0.2 for i in x_pos], decomp_summary['Total'], width=0.4, label='Total Attempts', alpha=0.7)
ax3.bar([i + 0.2 for i in x_pos], decomp_summary['Successful'], width=0.4, label='Successful Embeds', alpha=0.7)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(decomp_summary['Strategy'], rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Decomposition Strategy Performance', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Density Distribution by Formulation
ax4 = axes[1, 1]
density_df = df[df['density'].notna()].copy()
formulations = density_df['formulation'].unique()
for form in formulations:
    form_data = density_df[density_df['formulation'] == form]['density']
    if len(form_data) > 0:
        ax4.hist(form_data, alpha=0.5, label=form, bins=20)
ax4.axvline(x=0.3, color='red', linestyle='--', linewidth=2, label='Embed threshold')
ax4.set_xlabel('Density', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Density Distribution by Formulation', fontsize=14, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'comprehensive_benchmark_plots.png', dpi=300, bbox_inches='tight')
print(f"✅ Plots saved to: {results_dir / 'comprehensive_benchmark_plots.png'}")
plt.show()
