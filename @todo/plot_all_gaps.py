#!/usr/bin/env python3
"""
Plot gap from ALL scaling test result files to identify which one matches the plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Get all CSV files
results_dir = Path("/Users/edoardospigarolo/Documents/OQI-UC002-DWave/@todo/scaling_test_results")
csv_files = sorted(results_dir.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    # Get file modification time
    mtime = os.path.getmtime(f)
    from datetime import datetime
    dt = datetime.fromtimestamp(mtime)
    print(f"  {f.name} - {dt.strftime('%Y-%m-%d %H:%M')}")

# Create a figure with subplots for each file
n_files = len(csv_files)
n_cols = 3
n_rows = (n_files + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
axes = axes.flatten()

colors = {
    'Native 6-Family': '#1f77b4',
    '27->6 Aggregated': '#7f7f7f', 
    '27-Food Hybrid': '#2ca02c',
}

markers = {
    'Native 6-Family': 'o',
    '27->6 Aggregated': 's',
    '27-Food Hybrid': '^',
}

for idx, csv_file in enumerate(csv_files):
    ax = axes[idx]
    
    try:
        df = pd.read_csv(csv_file)
        
        # Get file date
        mtime = os.path.getmtime(csv_file)
        dt = datetime.fromtimestamp(mtime)
        
        # Plot each formulation
        for form in df['formulation'].unique():
            form_df = df[df['formulation'] == form].sort_values('n_vars')
            color = colors.get(form, 'gray')
            marker = markers.get(form, 'x')
            ax.plot(form_df['n_vars'], form_df['gap'], 
                   marker=marker, color=color, label=form, linewidth=2, markersize=8)
        
        # Add 20% target line
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20% target')
        
        ax.set_xlabel('Variables')
        ax.set_ylabel('Gap (%)')
        ax.set_title(f"{csv_file.stem}\n{dt.strftime('%Y-%m-%d %H:%M')}", fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim(0, 120)
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.set_title(f"{csv_file.stem}\nError: {e}", fontsize=8)
        ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', transform=ax.transAxes)

# Hide empty subplots
for idx in range(len(csv_files), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/edoardospigarolo/Documents/OQI-UC002-DWave/@todo/ALL_GAPS_COMPARISON.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved to: ALL_GAPS_COMPARISON.png")
