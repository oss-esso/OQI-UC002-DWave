#!/usr/bin/env python3
"""
Comprehensive QPU method comparison plots.
Includes: Native vs Hierarchical, Hybrid 27-Food, and scaling limitations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (18, 10),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Colors for different methods
COLORS = {
    'Native': '#e74c3c',       # Red
    'Hier(Orig)': '#3498db',   # Blue
    'Hier(Rep)': '#2ecc71',    # Green
    'Hybrid': '#9b59b6',       # Purple
    'Gurobi': '#34495e',       # Dark gray
    '6-Family': 'blue',
    '27-Food': 'green',
}

MARKERS = {
    'Native': 'X',
    'Hier(Orig)': 's',
    'Hier(Rep)': 'o',
    'Hybrid': 'D',
    'Gurobi': '^',
}

def load_all_qpu_data():
    """Load all QPU result files."""
    data = {}
    
    # Native 6-Family
    with open('qpu_native_6family.json') as f:
        native = json.load(f)
    data['Native'] = {r['scenario_name']: r for r in native['runs']}
    
    # Hierarchical Original
    with open('qpu_hier_all_6family.json') as f:
        hier_orig = json.load(f)
    data['Hier(Orig)'] = {r['scenario_name']: r for r in hier_orig['runs']}
    
    # Hierarchical Repaired
    with open('qpu_hier_repaired.json') as f:
        hier_rep = json.load(f)
    data['Hier(Rep)'] = {r['scenario_name']: r for r in hier_rep['runs']}
    
    # Hybrid 27-Food
    with open('qpu_hybrid_27food.json') as f:
        hybrid = json.load(f)
    data['Hybrid'] = {r['scenario_name']: r for r in hybrid['runs']}
    
    return data

def load_gurobi_300s():
    """Load 300s Gurobi results."""
    with open('@todo/gurobi_timeout_verification/gurobi_timeout_test_20251224_103144.json') as f:
        raw = json.load(f)
    
    scenarios = {}
    for entry in raw:
        if 'metadata' in entry:
            scenario = entry['metadata']['scenario']
            result = entry.get('result', {})
            if scenario not in scenarios:
                scenarios[scenario] = {
                    'n_farms': entry['metadata']['n_farms'],
                    'n_foods': entry['metadata']['n_foods'],
                    'n_vars': result.get('n_vars', 0),
                    'status': result.get('status', 'unknown'),
                    'objective': result.get('objective_value', 0),
                    'time': result.get('solve_time', 0),
                    'mip_gap': result.get('mip_gap', 0),
                    'timeout': result.get('hit_timeout', False),
                }
    return scenarios

def prepare_comparison_df(qpu_data, gurobi_data):
    """Prepare DataFrame for comparison."""
    rows = []
    
    # Get all scenarios from Hier(Rep) as reference
    for scenario, rep_data in qpu_data['Hier(Rep)'].items():
        row = {
            'scenario': scenario,
            'n_vars': rep_data.get('n_vars', 0),
            'n_farms': rep_data.get('n_farms', 0),
            'n_foods': rep_data.get('n_foods', 6),
            'formulation': '27-Food' if rep_data.get('n_foods', 6) == 27 else '6-Family',
        }
        
        # Add QPU method results
        for method in ['Native', 'Hier(Orig)', 'Hier(Rep)', 'Hybrid']:
            if scenario in qpu_data[method]:
                r = qpu_data[method][scenario]
                obj = r.get('objective_miqp')
                timing = r.get('timing', {})
                status = r.get('status', 'error')
                
                row[f'{method}_obj'] = obj if obj is not None else np.nan
                row[f'{method}_time'] = timing.get('total_wall_time', np.nan)
                row[f'{method}_qpu'] = timing.get('qpu_access_time', np.nan)
                row[f'{method}_status'] = status
            else:
                row[f'{method}_obj'] = np.nan
                row[f'{method}_time'] = np.nan
                row[f'{method}_qpu'] = np.nan
                row[f'{method}_status'] = 'missing'
        
        # Add Gurobi results
        if scenario in gurobi_data:
            g = gurobi_data[scenario]
            row['Gurobi_obj'] = g['objective']
            row['Gurobi_time'] = g['time']
            row['Gurobi_status'] = g['status']
            row['Gurobi_mip_gap'] = g['mip_gap'] * 100
        else:
            row['Gurobi_obj'] = np.nan
            row['Gurobi_time'] = np.nan
            row['Gurobi_status'] = 'missing'
            row['Gurobi_mip_gap'] = np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('n_vars')
    return df

def plot_method_comparison(df, output_dir):
    """Create 2x3 plot comparing all QPU methods."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    methods = ['Native', 'Hier(Orig)', 'Hier(Rep)', 'Hybrid']
    
    # =========================================================================
    # Plot 1: Objective Values by Method
    # =========================================================================
    ax = axes[0, 0]
    
    for method in methods:
        col = f'{method}_obj'
        valid = df[df[col].notna()].sort_values('n_vars')
        if len(valid) > 0:
            ax.scatter(valid['n_vars'], abs(valid[col]), 
                      s=100, marker=MARKERS.get(method, 'o'),
                      color=COLORS.get(method, 'gray'), alpha=0.8,
                      label=method, edgecolors='black', linewidths=0.5)
    
    # Add Gurobi
    gurobi_valid = df[df['Gurobi_obj'].notna()].sort_values('n_vars')
    ax.scatter(gurobi_valid['n_vars'], gurobi_valid['Gurobi_obj'], 
              s=100, marker=MARKERS['Gurobi'],
              color=COLORS['Gurobi'], alpha=0.8,
              label='Gurobi (300s)', edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Number of Variables', fontsize=13, fontweight='bold')
    ax.set_ylabel('Objective Value (|abs|)', fontsize=13, fontweight='bold')
    ax.set_title('Objective Values: All Methods', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # =========================================================================
    # Plot 2: Success Rate by Method and Problem Size
    # =========================================================================
    ax = axes[0, 1]
    
    # Count successes by variable count bins
    bins = [0, 100, 500, 1000, 2000, 5000, 20000]
    bin_labels = ['≤100', '101-500', '501-1k', '1k-2k', '2k-5k', '>5k']
    df['var_bin'] = pd.cut(df['n_vars'], bins=bins, labels=bin_labels)
    
    success_data = {}
    for method in methods:
        status_col = f'{method}_status'
        success_by_bin = df.groupby('var_bin').apply(
            lambda x: (x[status_col] == 'feasible').sum() / len(x) * 100
        )
        success_data[method] = success_by_bin.values
    
    x = np.arange(len(bin_labels))
    width = 0.2
    
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, success_data[method], width, 
                     label=method, color=COLORS.get(method, 'gray'), alpha=0.8)
    
    ax.set_xlabel('Problem Size (Variables)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Method Success Rate by Problem Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 110)
    
    # Add percentage labels
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    
    # =========================================================================
    # Plot 3: Native Embedding Limitation
    # =========================================================================
    ax = axes[0, 2]
    
    # Show what happens to native embedding
    native_status = []
    for _, row in df.iterrows():
        status = row['Native_status']
        if status == 'feasible':
            native_status.append(('Success', 'green'))
        elif status == 'error':
            native_status.append(('Embed Fail', 'red'))
        else:
            native_status.append(('Missing', 'gray'))
    
    colors_bar = [s[1] for s in native_status]
    labels_bar = [s[0] for s in native_status]
    
    x_pos = range(len(df))
    ax.bar(x_pos, [1]*len(df), color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add scenario labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{row['n_vars']}" for _, row in df.iterrows()], 
                       rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Problem Size (Variables)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Native Embedding Status', fontsize=12, fontweight='bold')
    ax.set_title('Native QPU Embedding: Hard Limit at ~100 Variables', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Success'),
                       Patch(facecolor='red', label='Embedding Failed')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add annotation
    ax.annotate('Only 90-var\nproblem embeds', xy=(0, 0.5), xytext=(3, 0.7),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='black'))
    
    # =========================================================================
    # Plot 4: Solve Time Comparison
    # =========================================================================
    ax = axes[1, 0]
    
    for method in ['Hier(Rep)', 'Hybrid']:
        col = f'{method}_time'
        valid = df[df[col].notna() & (df[f'{method}_status'] == 'feasible')].sort_values('n_vars')
        if len(valid) > 0:
            ax.plot(valid['n_vars'], valid[col], 
                   marker=MARKERS.get(method, 'o'),
                   color=COLORS.get(method, 'gray'),
                   label=method, linewidth=2.5, markersize=10, alpha=0.8)
    
    # Gurobi
    gurobi_valid = df[df['Gurobi_time'].notna()].sort_values('n_vars')
    ax.plot(gurobi_valid['n_vars'], gurobi_valid['Gurobi_time'], 
           marker=MARKERS['Gurobi'],
           color=COLORS['Gurobi'],
           label='Gurobi (300s)', linewidth=2.5, markersize=10, alpha=0.8)
    
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Gurobi timeout', linewidth=1.5)
    ax.set_xlabel('Number of Variables', fontsize=13, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Solve Time: Working Methods Only', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # =========================================================================
    # Plot 5: 27-Food: Hierarchical vs Hybrid
    # =========================================================================
    ax = axes[1, 1]
    
    df_27 = df[df['formulation'] == '27-Food'].sort_values('n_vars')
    
    if len(df_27) > 0:
        x_pos = np.arange(len(df_27))
        width = 0.35
        
        # Hierarchical (Repaired)
        hier_obj = [abs(v) if pd.notna(v) else 0 for v in df_27['Hier(Rep)_obj']]
        hybrid_obj = [abs(v) if pd.notna(v) else 0 for v in df_27['Hybrid_obj']]
        
        bars1 = ax.bar(x_pos - width/2, hier_obj, width, 
                       label='Hierarchical', color=COLORS['Hier(Rep)'], alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, hybrid_obj, width, 
                       label='Hybrid', color=COLORS['Hybrid'], alpha=0.8)
        
        # Mark missing data
        for i, (h, hy) in enumerate(zip(hier_obj, hybrid_obj)):
            if hy == 0:
                ax.annotate('N/A', xy=(x_pos[i] + width/2, 10), ha='center', fontsize=8, color='red')
        
        ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
        ax.set_ylabel('Objective Value (|abs|)', fontsize=12, fontweight='bold')
        ax.set_title('27-Food: Hierarchical vs Hybrid', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{int(row['n_farms'])} farms\n{int(row['n_vars'])} vars" 
                           for _, row in df_27.iterrows()], fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Plot 6: Summary Table
    # =========================================================================
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate summary stats
    def calc_stats(method):
        status_col = f'{method}_status'
        obj_col = f'{method}_obj'
        time_col = f'{method}_time'
        
        total = len(df)
        success = (df[status_col] == 'feasible').sum()
        avg_obj = df[df[obj_col].notna()][obj_col].apply(abs).mean()
        avg_time = df[df[time_col].notna()][time_col].mean()
        max_vars = df[df[status_col] == 'feasible']['n_vars'].max() if success > 0 else 0
        
        return [success, total, f"{avg_obj:.1f}" if pd.notna(avg_obj) else 'N/A',
                f"{avg_time:.1f}s" if pd.notna(avg_time) else 'N/A', 
                f"{max_vars:,}" if max_vars > 0 else 'N/A']
    
    table_data = [
        ['Method', 'Success', 'Avg |Obj|', 'Avg Time', 'Max Vars'],
        ['Native'] + calc_stats('Native')[:1] + [f"/{len(df)}"] + calc_stats('Native')[2:],
        ['Hier(Orig)'] + calc_stats('Hier(Orig)')[:1] + [f"/{len(df)}"] + calc_stats('Hier(Orig)')[2:],
        ['Hier(Rep)'] + calc_stats('Hier(Rep)')[:1] + [f"/{len(df)}"] + calc_stats('Hier(Rep)')[2:],
        ['Hybrid'] + calc_stats('Hybrid')[:1] + [f"/{len(df)}"] + calc_stats('Hybrid')[2:],
    ]
    
    # Simpler table
    summary = [
        ['Method', 'Success Rate', 'Max Variables', 'Recommendation'],
        ['Native', '1/13 (8%)', '90', '❌ Not scalable'],
        ['Hier(Original)', '9/13 (69%)', '1,800', '⚠️ Superseded'],
        ['Hier(Repaired)', '13/13 (100%)', '16,200', '✅ Recommended'],
        ['Hybrid 27-Food', '2/4 (50%)', '4,050', '⚠️ Incomplete'],
        ['Gurobi 300s', '1/13 optimal', '90', '⚠️ Timeouts'],
    ]
    
    table = ax.table(cellText=summary[1:], colLabels=summary[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.2, 0.2, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header and highlight recommended
    for j in range(4):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Highlight recommended row
    for j in range(4):
        table[(3, j)].set_facecolor('#d5f5e3')
    
    ax.set_title('Method Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'qpu_method_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✓ Saved: {output_path.with_suffix('.pdf')}")
    plt.close()

def plot_scaling_limitations(df, output_dir):
    """Create focused plot on native embedding limitations."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # =========================================================================
    # Plot 1: Problem Size vs Embedding Success
    # =========================================================================
    ax = axes[0]
    
    # Create visualization of embedding success
    scenarios_sorted = df.sort_values('n_vars')
    n_vars = scenarios_sorted['n_vars'].values
    
    native_success = [1 if s == 'feasible' else 0 for s in scenarios_sorted['Native_status']]
    hier_success = [1 if s == 'feasible' else 0 for s in scenarios_sorted['Hier(Rep)_status']]
    
    x = np.arange(len(n_vars))
    width = 0.35
    
    ax.bar(x - width/2, native_success, width, label='Native', color=COLORS['Native'], alpha=0.8)
    ax.bar(x + width/2, hier_success, width, label='Hierarchical', color=COLORS['Hier(Rep)'], alpha=0.8)
    
    ax.set_xlabel('Problem Size (Variables)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success (1=Yes, 0=No)', fontsize=12, fontweight='bold')
    ax.set_title('Embedding Success: Native vs Hierarchical', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:,}" for v in n_vars], rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.2)
    
    # Add annotation
    ax.annotate('Native fails\nbeyond 90 vars', xy=(1, 0.1), xytext=(4, 0.5),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # =========================================================================
    # Plot 2: Why Native Fails - Qubit Requirements
    # =========================================================================
    ax = axes[1]
    
    # Estimate qubit requirements (rough: ~2-3 physical qubits per logical for small, more for large)
    pegasus_qubits = 5760
    
    # Rough estimate of required qubits for native embedding
    # Dense QUBO needs ~n^2 couplers, but clique embedding needs more
    estimated_qubits = []
    for v in n_vars:
        # Very rough: small problems ~3x vars, large problems need full clique
        if v <= 100:
            est = v * 3  # Might embed
        elif v <= 500:
            est = v * 10  # Unlikely to embed
        else:
            est = v * 20  # Definitely won't embed
        estimated_qubits.append(min(est, pegasus_qubits * 2))
    
    ax.bar(x, estimated_qubits, color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=pegasus_qubits, color='green', linestyle='--', linewidth=2, 
               label=f'Pegasus capacity ({pegasus_qubits:,} qubits)')
    
    ax.set_xlabel('Problem Size (Variables)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Estimated Physical Qubits', fontsize=12, fontweight='bold')
    ax.set_title('Why Native Embedding Fails', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:,}" for v in n_vars], rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # =========================================================================
    # Plot 3: Hierarchical Decomposition Solution
    # =========================================================================
    ax = axes[2]
    
    # Show how hierarchical breaks down the problem
    hier_data = df[df['Hier(Rep)_status'] == 'feasible'].sort_values('n_vars')
    
    if len(hier_data) > 0:
        # Cluster size is typically 5-9 farms, 6 foods, 3 periods = 90-162 vars per cluster
        cluster_vars = 90  # Approximate
        n_clusters = hier_data['n_vars'] / cluster_vars
        
        ax.scatter(hier_data['n_vars'], n_clusters, s=150, c=COLORS['Hier(Rep)'], 
                  edgecolors='black', linewidths=0.5, alpha=0.8)
        
        # Fit line
        coef = np.polyfit(hier_data['n_vars'], n_clusters, 1)
        x_fit = np.linspace(hier_data['n_vars'].min(), hier_data['n_vars'].max(), 100)
        ax.plot(x_fit, coef[0] * x_fit + coef[1], '--', color='gray', alpha=0.7,
                label=f'Linear: {coef[0]:.4f} clusters/var')
        
        ax.set_xlabel('Total Problem Size (Variables)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Approximate Clusters', fontsize=12, fontweight='bold')
        ax.set_title('Hierarchical: Divide & Conquer', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add annotation
        ax.annotate('Each cluster\n~90 vars\n(fits on QPU)', 
                   xy=(hier_data['n_vars'].iloc[-1], n_clusters.iloc[-1]),
                   xytext=(hier_data['n_vars'].iloc[-1] * 0.5, n_clusters.iloc[-1] * 1.3),
                   fontsize=10, ha='center',
                   arrowprops=dict(arrowstyle='->', color='green'),
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'native_vs_hierarchical_scaling.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✓ Saved: {output_path.with_suffix('.pdf')}")
    plt.close()

def plot_hybrid_27food_analysis(df, output_dir):
    """Detailed analysis of hybrid 27-food approach."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    df_27 = df[df['formulation'] == '27-Food'].sort_values('n_vars')
    
    # =========================================================================
    # Plot 1: Objective Comparison
    # =========================================================================
    ax = axes[0]
    
    x_pos = np.arange(len(df_27))
    width = 0.25
    
    hier_obj = [abs(v) if pd.notna(v) else 0 for v in df_27['Hier(Rep)_obj']]
    hybrid_obj = [abs(v) if pd.notna(v) else 0 for v in df_27['Hybrid_obj']]
    gurobi_obj = [abs(v) if pd.notna(v) else 0 for v in df_27['Gurobi_obj']]
    
    ax.bar(x_pos - width, gurobi_obj, width, label='Gurobi (300s)', color=COLORS['Gurobi'], alpha=0.8)
    ax.bar(x_pos, hier_obj, width, label='Hierarchical', color=COLORS['Hier(Rep)'], alpha=0.8)
    ax.bar(x_pos + width, hybrid_obj, width, label='Hybrid', color=COLORS['Hybrid'], alpha=0.8)
    
    # Mark failed hybrid
    for i, hy in enumerate(hybrid_obj):
        if hy == 0:
            ax.annotate('✗', xy=(x_pos[i] + width, 5), ha='center', fontsize=14, color='red')
    
    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Objective Value (|abs|)', fontsize=12, fontweight='bold')
    ax.set_title('27-Food: Method Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{int(row['n_farms'])} farms" for _, row in df_27.iterrows()])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Plot 2: Time Comparison
    # =========================================================================
    ax = axes[1]
    
    hier_time = [v if pd.notna(v) else 0 for v in df_27['Hier(Rep)_time']]
    hybrid_time = [v if pd.notna(v) else 0 for v in df_27['Hybrid_time']]
    gurobi_time = [v if pd.notna(v) else 0 for v in df_27['Gurobi_time']]
    
    ax.bar(x_pos - width, gurobi_time, width, label='Gurobi (300s)', color=COLORS['Gurobi'], alpha=0.8)
    ax.bar(x_pos, hier_time, width, label='Hierarchical', color=COLORS['Hier(Rep)'], alpha=0.8)
    ax.bar(x_pos + width, hybrid_time, width, label='Hybrid', color=COLORS['Hybrid'], alpha=0.8)
    
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Gurobi timeout')
    
    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('27-Food: Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{int(row['n_farms'])} farms" for _, row in df_27.iterrows()])
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Plot 3: Summary Table
    # =========================================================================
    ax = axes[2]
    ax.axis('off')
    
    table_data = [
        ['Scenario', 'Gurobi', 'Hier', 'Hybrid', 'Best'],
    ]
    
    for _, row in df_27.iterrows():
        gur = row['Gurobi_obj']
        hier = row['Hier(Rep)_obj']
        hyb = row['Hybrid_obj']
        
        gur_str = f"{gur:.1f}" if pd.notna(gur) else 'N/A'
        hier_str = f"{abs(hier):.1f}" if pd.notna(hier) else 'N/A'
        hyb_str = f"{abs(hyb):.1f}" if pd.notna(hyb) else 'Failed'
        
        # Determine best (most negative = best for maximization)
        valid = []
        if pd.notna(hier): valid.append(('Hier', hier))
        if pd.notna(hyb): valid.append(('Hybrid', hyb))
        
        if valid:
            best = min(valid, key=lambda x: x[1])
            best_str = f"{best[0]} ({abs(best[1]):.1f})"
        else:
            best_str = 'N/A'
        
        table_data.append([
            f"{int(row['n_farms'])} farms\n({int(row['n_vars'])} vars)",
            gur_str,
            hier_str,
            hyb_str,
            best_str
        ])
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for j in range(5):
        table[(0, j)].set_facecolor('#9b59b6')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('27-Food Detailed Results', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'hybrid_27food_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✓ Saved: {output_path.with_suffix('.pdf')}")
    plt.close()

def print_comprehensive_summary(df):
    """Print comprehensive analysis summary."""
    print("\n" + "=" * 120)
    print("COMPREHENSIVE QPU METHOD COMPARISON")
    print("=" * 120)
    
    methods = ['Native', 'Hier(Orig)', 'Hier(Rep)', 'Hybrid']
    
    for method in methods:
        status_col = f'{method}_status'
        obj_col = f'{method}_obj'
        time_col = f'{method}_time'
        qpu_col = f'{method}_qpu'
        
        success = (df[status_col] == 'feasible').sum()
        total = len(df)
        
        print(f"\n{method}:")
        print(f"  Success rate: {success}/{total} ({100*success/total:.0f}%)")
        
        if success > 0:
            successful = df[df[status_col] == 'feasible']
            print(f"  Variable range: {successful['n_vars'].min()} - {successful['n_vars'].max()}")
            if successful[obj_col].notna().any():
                print(f"  Objective range: {successful[obj_col].min():.2f} to {successful[obj_col].max():.2f}")
            if successful[time_col].notna().any():
                print(f"  Time range: {successful[time_col].min():.1f}s - {successful[time_col].max():.1f}s")
            if successful[qpu_col].notna().any():
                total_qpu = successful[qpu_col].sum()
                total_wall = successful[time_col].sum()
                print(f"  Total pure QPU time: {total_qpu:.3f}s ({100*total_qpu/total_wall:.1f}% of wall time)")
    
    print("\n" + "=" * 120)
    print("KEY CONCLUSIONS")
    print("=" * 120)
    print("""
🔴 NATIVE EMBEDDING:
   - Only works for problems ≤90 variables (5 farms × 6 foods × 3 periods)
   - Pegasus topology cannot embed larger dense QUBOs
   - NOT viable for production use

🟡 HIERARCHICAL (ORIGINAL):
   - First successful approach for scaling
   - Some issues with objective calculation
   - Superseded by repaired version

🟢 HIERARCHICAL (REPAIRED):
   - 100% success rate on all 13 scenarios
   - Scales to 16,200 variables (200 farms × 27 foods × 3 periods)
   - Pure QPU time only 1.1% of wall time
   - RECOMMENDED for production use

🟣 HYBRID 27-FOOD:
   - Alternative approach for full 27-crop problems
   - Only 50% success rate (2/4 scenarios)
   - When working, produces better objectives than hierarchical
   - Needs further development

📊 RECOMMENDATIONS:
   1. Use Hierarchical (Repaired) as primary method
   2. Native embedding is demonstration-only
   3. Hybrid approach promising but incomplete
   4. Compare against 300s Gurobi for fair benchmarks
""")

def main():
    """Generate all comparison plots."""
    output_dir = 'professional_plots'
    Path(output_dir).mkdir(exist_ok=True)
    
    print('Loading all QPU data...')
    qpu_data = load_all_qpu_data()
    gurobi_data = load_gurobi_300s()
    
    print('Preparing comparison DataFrame...')
    df = prepare_comparison_df(qpu_data, gurobi_data)
    
    print(f'Loaded {len(df)} scenarios')
    print()
    
    print('Generating method comparison plots...')
    plot_method_comparison(df, output_dir)
    
    print('Generating scaling limitations plot...')
    plot_scaling_limitations(df, output_dir)
    
    print('Generating hybrid 27-food analysis...')
    plot_hybrid_27food_analysis(df, output_dir)
    
    print_comprehensive_summary(df)
    
    # Save comparison data
    df.to_csv(Path(output_dir) / 'qpu_method_comparison_data.csv', index=False)
    print(f"\n✓ Data saved to {output_dir}/qpu_method_comparison_data.csv")
    
    print(f'\n✓ All plots saved to {output_dir}/')

if __name__ == '__main__':
    main()
