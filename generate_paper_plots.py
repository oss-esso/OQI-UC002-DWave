#!/usr/bin/env python3
"""
Generate professional plots for the quantum advantage paper.
Creates publication-quality figures comparing QPU vs Gurobi performance.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import os

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette
COLORS = {
    'qpu': '#2ecc71',       # Green for QPU
    'qpu_access': '#27ae60', # Darker green for QPU access
    'gurobi': '#e74c3c',    # Red for Gurobi
    'classical': '#3498db',  # Blue for classical overhead
    'timeout': '#95a5a6',    # Gray for timeout
    'highlight': '#f39c12',  # Orange for highlights
}

def load_results():
    """Load QPU and Gurobi results."""
    with open('qpu_hier_repaired.json') as f:
        qpu = json.load(f)
    with open('gurobi_baseline_60s.json') as f:
        gurobi = json.load(f)
    return qpu, gurobi

def prepare_data(qpu, gurobi):
    """Prepare data for plotting."""
    qpu_by = {r['scenario_name']: r for r in qpu['runs']}
    gur_by = {r['scenario_name']: r for r in gurobi['runs']}
    
    data = []
    for scenario in qpu_by.keys():
        q = qpu_by[scenario]
        g = gur_by.get(scenario, {})
        
        timing = q.get('timing', {})
        g_timing = g.get('timing', {})
        
        data.append({
            'scenario': scenario,
            'n_farms': q['n_farms'],
            'n_foods': q['n_foods'],
            'n_vars': q['n_vars'],
            'qpu_total_time': timing.get('total_wall_time', 0),
            'qpu_access_time': timing.get('qpu_access_time', 0),
            'qpu_sampling_time': timing.get('qpu_sampling_time', 0),
            'qpu_objective': q.get('objective_miqp', 0),
            'gurobi_time': g_timing.get('total_wall_time', 0),
            'gurobi_objective': g.get('objective_miqp', 0),
            'gurobi_status': g.get('status', 'N/A'),
            'gurobi_timeout': g.get('status') == 'timeout',
        })
    
    # Sort by number of variables
    data.sort(key=lambda x: x['n_vars'])
    return data

def plot_time_comparison(data, output_dir):
    """Plot 1: Time comparison QPU vs Gurobi."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scenarios = [d['scenario'].replace('rotation_', '').replace('_', ' ') for d in data]
    n_vars = [d['n_vars'] for d in data]
    qpu_times = [d['qpu_total_time'] for d in data]
    gurobi_times = [d['gurobi_time'] for d in data]
    timeouts = [d['gurobi_timeout'] for d in data]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # QPU bars
    bars1 = ax.bar(x - width/2, qpu_times, width, label='D-Wave QPU (Hierarchical)', 
                   color=COLORS['qpu'], edgecolor='black', linewidth=0.5)
    
    # Gurobi bars with timeout indicators
    bars2 = ax.bar(x + width/2, gurobi_times, width, label='Gurobi 12.0',
                   color=[COLORS['timeout'] if t else COLORS['gurobi'] for t in timeouts],
                   edgecolor='black', linewidth=0.5)
    
    # Add timeout markers
    for i, (bar, timeout) in enumerate(zip(bars2, timeouts)):
        if timeout:
            ax.annotate('TIMEOUT', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=7, color='red', fontweight='bold',
                       rotation=90)
    
    ax.set_xlabel('Scenario (sorted by problem size)')
    ax.set_ylabel('Solve Time (seconds)')
    ax.set_title('Solve Time Comparison: D-Wave QPU vs Gurobi\n(Gurobi timeout = 60s)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}\n({v} vars)' for s, v in zip(scenarios, n_vars)], 
                       rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(1, 500)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.pdf'))
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'))
    plt.close()
    print('Created: time_comparison.pdf/png')

def plot_qpu_time_breakdown(data, output_dir):
    """Plot 2: QPU time breakdown showing quantum vs classical overhead."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scenarios = [d['scenario'].replace('rotation_', '').replace('_', ' ') for d in data]
    n_vars = [d['n_vars'] for d in data]
    
    qpu_access = [d['qpu_access_time'] for d in data]
    classical_overhead = [d['qpu_total_time'] - d['qpu_access_time'] for d in data]
    
    x = np.arange(len(scenarios))
    width = 0.6
    
    # Stacked bars
    bars1 = ax.bar(x, qpu_access, width, label='Pure QPU Access Time', 
                   color=COLORS['qpu_access'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, classical_overhead, width, bottom=qpu_access,
                   label='Classical Overhead (embedding, coordination)', 
                   color=COLORS['classical'], edgecolor='black', linewidth=0.5)
    
    # Add percentage labels
    for i, (qpu, total) in enumerate(zip(qpu_access, [d['qpu_total_time'] for d in data])):
        pct = (qpu / total) * 100 if total > 0 else 0
        ax.annotate(f'{pct:.1f}%', xy=(i, qpu/2), ha='center', va='center', 
                   fontsize=8, color='white', fontweight='bold')
    
    ax.set_xlabel('Scenario (sorted by problem size)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('QPU Time Breakdown: Quantum vs Classical Computation\n(Percentages show pure QPU fraction)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}\n({v} vars)' for s, v in zip(scenarios, n_vars)], 
                       rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qpu_time_breakdown.pdf'))
    plt.savefig(os.path.join(output_dir, 'qpu_time_breakdown.png'))
    plt.close()
    print('Created: qpu_time_breakdown.pdf/png')

def plot_scaling_analysis(data, output_dir):
    """Plot 3: Scaling analysis - time vs variables."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_vars = np.array([d['n_vars'] for d in data])
    qpu_total = np.array([d['qpu_total_time'] for d in data])
    qpu_access = np.array([d['qpu_access_time'] for d in data])
    gurobi_times = np.array([d['gurobi_time'] for d in data])
    timeouts = [d['gurobi_timeout'] for d in data]
    
    # Left plot: Total time scaling
    ax1 = axes[0]
    ax1.scatter(n_vars, qpu_total, s=100, c=COLORS['qpu'], label='QPU Total Time', 
                edgecolors='black', linewidths=0.5, zorder=3)
    ax1.scatter(n_vars, gurobi_times, s=100, 
                c=[COLORS['timeout'] if t else COLORS['gurobi'] for t in timeouts],
                label='Gurobi Time', marker='s', edgecolors='black', linewidths=0.5, zorder=3)
    
    # Fit lines
    log_vars = np.log(n_vars)
    log_qpu = np.log(qpu_total)
    coef_qpu = np.polyfit(log_vars, log_qpu, 1)
    fit_qpu = np.exp(coef_qpu[1]) * n_vars ** coef_qpu[0]
    ax1.plot(n_vars, fit_qpu, '--', color=COLORS['qpu'], alpha=0.7, 
             label=f'QPU fit: O(n^{coef_qpu[0]:.2f})')
    
    ax1.axhline(y=60, color='red', linestyle=':', alpha=0.5, label='Gurobi timeout (60s)')
    
    ax1.set_xlabel('Number of Variables')
    ax1.set_ylabel('Solve Time (seconds)')
    ax1.set_title('Time Scaling vs Problem Size')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Pure QPU time scaling
    ax2 = axes[1]
    ax2.scatter(n_vars, qpu_access, s=100, c=COLORS['qpu_access'], 
                label='Pure QPU Access Time', edgecolors='black', linewidths=0.5, zorder=3)
    
    # Linear fit for QPU access
    coef_access = np.polyfit(n_vars, qpu_access, 1)
    fit_access = coef_access[0] * n_vars + coef_access[1]
    ax2.plot(n_vars, fit_access, '--', color=COLORS['qpu_access'], alpha=0.7,
             label=f'Linear fit: {coef_access[0]*1000:.3f}ms/var')
    
    ax2.set_xlabel('Number of Variables')
    ax2.set_ylabel('Pure QPU Time (seconds)')
    ax2.set_title('Pure QPU Access Time Scales Linearly')
    ax2.set_xscale('log')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_analysis.pdf'))
    plt.savefig(os.path.join(output_dir, 'scaling_analysis.png'))
    plt.close()
    print('Created: scaling_analysis.pdf/png')

def plot_speedup_analysis(data, output_dir):
    """Plot 4: Speedup analysis where Gurobi times out."""
    # Filter to timeout cases
    timeout_data = [d for d in data if d['gurobi_timeout']]
    
    if not timeout_data:
        print('No timeout cases found, skipping speedup plot')
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = [d['scenario'].replace('rotation_', '').replace('_', ' ') for d in timeout_data]
    n_vars = [d['n_vars'] for d in timeout_data]
    qpu_times = [d['qpu_total_time'] for d in timeout_data]
    gurobi_times = [d['gurobi_time'] for d in timeout_data]  # All 60s timeout
    
    # Compute "speedup" (Gurobi timeout / QPU time)
    speedups = [g / q if q > 0 else 0 for g, q in zip(gurobi_times, qpu_times)]
    
    x = np.arange(len(scenarios))
    
    colors = [COLORS['qpu'] if s > 1 else COLORS['gurobi'] for s in speedups]
    bars = ax.bar(x, speedups, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Parity (1x)')
    
    # Add speedup labels
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        label = f'{speedup:.1f}x' if speedup >= 1 else f'{1/speedup:.1f}x slower'
        color = 'green' if speedup >= 1 else 'red'
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    ax.set_xlabel('Scenario (Gurobi timeout cases only)')
    ax.set_ylabel('Speedup Factor (Gurobi timeout / QPU time)')
    ax.set_title('QPU Speedup on Problems Where Gurobi Times Out\n(Values >1 indicate QPU advantage)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}\n({v} vars)' for s, v in zip(scenarios, n_vars)], 
                       rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_analysis.pdf'))
    plt.savefig(os.path.join(output_dir, 'speedup_analysis.png'))
    plt.close()
    print('Created: speedup_analysis.pdf/png')

def plot_comprehensive_summary(data, output_dir):
    """Plot 5: Comprehensive 2x2 summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    n_vars = np.array([d['n_vars'] for d in data])
    qpu_total = np.array([d['qpu_total_time'] for d in data])
    qpu_access = np.array([d['qpu_access_time'] for d in data])
    gurobi_times = np.array([d['gurobi_time'] for d in data])
    timeouts = [d['gurobi_timeout'] for d in data]
    
    # (0,0) Time comparison
    ax = axes[0, 0]
    ax.scatter(n_vars, qpu_total, s=80, c=COLORS['qpu'], label='QPU Total', 
               edgecolors='black', linewidths=0.5, zorder=3, marker='o')
    ax.scatter(n_vars, gurobi_times, s=80,
               c=[COLORS['timeout'] if t else COLORS['gurobi'] for t in timeouts],
               label='Gurobi', marker='s', edgecolors='black', linewidths=0.5, zorder=3)
    ax.axhline(y=60, color='red', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of Variables')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('(a) Solve Time Comparison')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # (0,1) QPU time breakdown pie
    ax = axes[0, 1]
    total_qpu_access = sum(d['qpu_access_time'] for d in data)
    total_classical = sum(d['qpu_total_time'] - d['qpu_access_time'] for d in data)
    
    sizes = [total_qpu_access, total_classical]
    labels = [f'Pure QPU\n({total_qpu_access:.1f}s, {100*total_qpu_access/(total_qpu_access+total_classical):.1f}%)',
              f'Classical Overhead\n({total_classical:.1f}s, {100*total_classical/(total_qpu_access+total_classical):.1f}%)']
    colors_pie = [COLORS['qpu_access'], COLORS['classical']]
    
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct='', startangle=90,
           wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    ax.set_title('(b) Total QPU Time Breakdown\n(All 13 scenarios combined)')
    
    # (1,0) Pure QPU scaling
    ax = axes[1, 0]
    ax.scatter(n_vars, qpu_access, s=80, c=COLORS['qpu_access'], 
               edgecolors='black', linewidths=0.5, zorder=3)
    coef = np.polyfit(n_vars, qpu_access, 1)
    fit_line = coef[0] * n_vars + coef[1]
    ax.plot(n_vars, fit_line, '--', color=COLORS['qpu_access'], alpha=0.7,
            label=f'Linear: {coef[0]*1000:.4f}ms/var')
    ax.set_xlabel('Number of Variables')
    ax.set_ylabel('Pure QPU Access Time (seconds)')
    ax.set_title('(c) Pure QPU Time Scales Linearly')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # (1,1) Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary table
    n_scenarios = len(data)
    n_qpu_faster = sum(1 for d in data if d['qpu_total_time'] < d['gurobi_time'] or d['gurobi_timeout'])
    n_timeouts = sum(1 for d in data if d['gurobi_timeout'])
    total_qpu_time = sum(d['qpu_total_time'] for d in data)
    total_gurobi_time = sum(d['gurobi_time'] for d in data)
    max_vars = max(d['n_vars'] for d in data)
    
    table_data = [
        ['Metric', 'Value'],
        ['Total Scenarios', f'{n_scenarios}'],
        ['QPU Faster/Equivalent', f'{n_qpu_faster}/{n_scenarios}'],
        ['Gurobi Timeouts', f'{n_timeouts}/{n_scenarios}'],
        ['Largest Problem Solved', f'{max_vars:,} variables'],
        ['Total QPU Time', f'{total_qpu_time:.1f}s'],
        ['Total Gurobi Time', f'{total_gurobi_time:.1f}s'],
        ['Pure QPU Time (all)', f'{total_qpu_access:.1f}s ({100*total_qpu_access/total_qpu_time:.1f}%)'],
    ]
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.5, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Color header
    for j in range(2):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('(d) Summary Statistics', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_summary.pdf'))
    plt.savefig(os.path.join(output_dir, 'comprehensive_summary.png'))
    plt.close()
    print('Created: comprehensive_summary.pdf/png')

def plot_variable_split_analysis(data, output_dir):
    """Plot 6: Analysis by problem configuration (farms x foods x periods)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by food count (6 vs 27)
    data_6food = [d for d in data if d['n_foods'] == 6]
    data_27food = [d for d in data if d['n_foods'] == 27]
    
    # Left: 6-family problems
    ax = axes[0]
    if data_6food:
        farms_6 = [d['n_farms'] for d in data_6food]
        qpu_6 = [d['qpu_total_time'] for d in data_6food]
        gurobi_6 = [d['gurobi_time'] for d in data_6food]
        timeouts_6 = [d['gurobi_timeout'] for d in data_6food]
        
        x = np.arange(len(farms_6))
        width = 0.35
        
        ax.bar(x - width/2, qpu_6, width, label='QPU', color=COLORS['qpu'], 
               edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, gurobi_6, width, label='Gurobi',
               color=[COLORS['timeout'] if t else COLORS['gurobi'] for t in timeouts_6],
               edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Number of Farms')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('6-Family Configuration\n(6 crops × 3 periods = 18 vars/farm)')
        ax.set_xticks(x)
        ax.set_xticklabels(farms_6)
        ax.legend()
        ax.set_yscale('log')
    
    # Right: 27-food problems
    ax = axes[1]
    if data_27food:
        farms_27 = [d['n_farms'] for d in data_27food]
        qpu_27 = [d['qpu_total_time'] for d in data_27food]
        gurobi_27 = [d['gurobi_time'] for d in data_27food]
        timeouts_27 = [d['gurobi_timeout'] for d in data_27food]
        
        x = np.arange(len(farms_27))
        width = 0.35
        
        ax.bar(x - width/2, qpu_27, width, label='QPU', color=COLORS['qpu'],
               edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, gurobi_27, width, label='Gurobi',
               color=[COLORS['timeout'] if t else COLORS['gurobi'] for t in timeouts_27],
               edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Number of Farms')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('27-Food Configuration\n(27 crops × 3 periods = 81 vars/farm)')
        ax.set_xticks(x)
        ax.set_xticklabels(farms_27)
        ax.legend()
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variable_split_analysis.pdf'))
    plt.savefig(os.path.join(output_dir, 'variable_split_analysis.png'))
    plt.close()
    print('Created: variable_split_analysis.pdf/png')

def plot_qpu_efficiency(data, output_dir):
    """Plot 7: QPU efficiency - ratio of pure QPU time to total time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scenarios = [d['scenario'].replace('rotation_', '').replace('_', ' ') for d in data]
    n_vars = [d['n_vars'] for d in data]
    
    efficiency = [100 * d['qpu_access_time'] / d['qpu_total_time'] 
                  if d['qpu_total_time'] > 0 else 0 for d in data]
    
    colors = plt.cm.RdYlGn([e/5 for e in efficiency])  # Scale 0-5% to colormap
    
    bars = ax.bar(range(len(scenarios)), efficiency, color=colors, 
                  edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel('QPU Efficiency (%)\n(Pure QPU Time / Total Time)')
    ax.set_title('QPU Efficiency Across Problem Sizes\n(Higher = more time spent in quantum computation)')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([f'{s}\n({v} vars)' for s, v in zip(scenarios, n_vars)], 
                       rotation=45, ha='right', fontsize=8)
    
    # Add percentage labels
    for bar, eff in zip(bars, efficiency):
        ax.annotate(f'{eff:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_ylim(0, max(efficiency) * 1.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qpu_efficiency.pdf'))
    plt.savefig(os.path.join(output_dir, 'qpu_efficiency.png'))
    plt.close()
    print('Created: qpu_efficiency.pdf/png')

def main():
    """Generate all plots."""
    output_dir = 'professional_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print('Loading results...')
    qpu, gurobi = load_results()
    data = prepare_data(qpu, gurobi)
    
    print(f'Loaded {len(data)} scenarios')
    print()
    
    print('Generating plots...')
    plot_time_comparison(data, output_dir)
    plot_qpu_time_breakdown(data, output_dir)
    plot_scaling_analysis(data, output_dir)
    plot_speedup_analysis(data, output_dir)
    plot_comprehensive_summary(data, output_dir)
    plot_variable_split_analysis(data, output_dir)
    plot_qpu_efficiency(data, output_dir)
    
    print()
    print(f'All plots saved to {output_dir}/')

if __name__ == '__main__':
    main()
