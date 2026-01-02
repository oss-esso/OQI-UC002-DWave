#!/usr/bin/env python3
"""
Assess the impact of constraint violations on QPU solution quality.
Quantify how much violations contribute to the objective gap.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load QPU results
with open('qpu_hier_repaired.json') as f:
    qpu_data = json.load(f)

# Load Gurobi results
with open('@todo/gurobi_timeout_verification/gurobi_timeout_test_20251224_103144.json') as f:
    gurobi_raw = json.load(f)

# Parse Gurobi
gurobi = {}
for entry in gurobi_raw:
    if 'metadata' in entry:
        sc = entry['metadata']['scenario']
        result = entry.get('result', {})
        gurobi[sc] = {
            'objective': result.get('objective_value', 0),
            'n_vars': result.get('n_vars', 0),
            'status': result.get('status', 'unknown'),
            'mip_gap': result.get('mip_gap', 0),
        }

print('=' * 120)
print('VIOLATION IMPACT ASSESSMENT')
print('=' * 120)

# Analyze each scenario
results = []
for r in qpu_data['runs']:
    sc = r['scenario_name']
    n_vars = r['n_vars']
    n_farms = r['n_farms']
    n_foods = r['n_foods']
    n_periods = r['n_periods']
    
    qpu_obj = r['objective_miqp']
    viols = r['constraint_violations']
    one_hot_viols = viols.get('one_hot_violations', 0)
    rotation_viols = viols.get('rotation_violations', 0)
    total_viols = viols.get('total_violations', 0)
    
    # Get Gurobi reference
    gur_obj = gurobi.get(sc, {}).get('objective', 0)
    
    # Calculate gap
    gap = abs(qpu_obj) - gur_obj if gur_obj else 0
    
    # Calculate violation rate
    total_slots = n_farms * n_periods  # Each farm-period should have at least 1 crop
    violation_rate = one_hot_viols / total_slots * 100 if total_slots > 0 else 0
    
    # Estimate lost benefit per violation
    # Assumption: Average benefit per farm-period is roughly objective / (farms * periods)
    avg_benefit_per_slot = gur_obj / total_slots if total_slots > 0 and gur_obj > 0 else 0
    estimated_lost_benefit = one_hot_viols * avg_benefit_per_slot
    
    results.append({
        'scenario': sc,
        'n_vars': n_vars,
        'n_farms': n_farms,
        'n_foods': n_foods,
        'total_slots': total_slots,
        'qpu_obj': qpu_obj,
        'gurobi_obj': gur_obj,
        'gap': gap,
        'one_hot_viols': one_hot_viols,
        'rotation_viols': rotation_viols,
        'total_viols': total_viols,
        'violation_rate': violation_rate,
        'avg_benefit_per_slot': avg_benefit_per_slot,
        'estimated_lost_benefit': estimated_lost_benefit,
    })

df = pd.DataFrame(results)

# Print detailed analysis
print("\nDETAILED VIOLATION IMPACT BY SCENARIO")
print("-" * 120)
header = f"{'Scenario':<30} {'Slots':>6} {'Viols':>6} {'Rate%':>7} {'Gurobi':>10} {'QPU':>10} {'Gap':>10} {'Est.Loss':>10}"
print(header)
print("-" * 120)

for _, row in df.iterrows():
    print(f"{row['scenario']:<30} {row['total_slots']:>6} {row['one_hot_viols']:>6} "
          f"{row['violation_rate']:>6.1f}% {row['gurobi_obj']:>10.2f} {row['qpu_obj']:>10.2f} "
          f"{row['gap']:>10.2f} {row['estimated_lost_benefit']:>10.2f}")

print("-" * 120)
print(f"{'TOTAL':<30} {df['total_slots'].sum():>6} {df['one_hot_viols'].sum():>6} "
      f"{df['one_hot_viols'].sum()/df['total_slots'].sum()*100:>6.1f}% "
      f"{df['gurobi_obj'].sum():>10.2f} {df['qpu_obj'].sum():>10.2f} "
      f"{df['gap'].sum():>10.2f} {df['estimated_lost_benefit'].sum():>10.2f}")

# Key statistics
print("\n" + "=" * 120)
print("KEY STATISTICS")
print("=" * 120)

total_slots = df['total_slots'].sum()
total_viols = df['one_hot_viols'].sum()
total_gap = df['gap'].sum()

print(f"\nOverall Violation Rate: {total_viols}/{total_slots} = {total_viols/total_slots*100:.2f}%")
print(f"Average Violations per Scenario: {total_viols/len(df):.1f}")
print(f"Average Violation Rate: {df['violation_rate'].mean():.2f}%")

# Correlation analysis
corr_viols_gap = df['one_hot_viols'].corr(df['gap'])
corr_rate_gap = df['violation_rate'].corr(df['gap'])

print(f"\nCorrelation (violations vs gap): {corr_viols_gap:.4f}")
print(f"Correlation (violation rate vs gap): {corr_rate_gap:.4f}")

# Gap per violation
gap_per_viol = total_gap / total_viols if total_viols > 0 else 0
print(f"\nAverage Gap per Violation: {gap_per_viol:.2f}")

# What if we "correct" the QPU objective by the violation rate?
print("\n" + "=" * 120)
print("VIOLATION-ADJUSTED COMPARISON")
print("=" * 120)

print("\nIdea: If violations cause the gap, what would QPU objective be if feasible?")
print("Approach 1: Subtract estimated lost benefit from |QPU objective|")
print("Approach 2: Scale QPU objective by (1 - violation_rate)")
print()

print(f"{'Scenario':<30} {'Gurobi':>10} {'QPU(raw)':>10} {'Adj(v1)':>10} {'Adj(v2)':>10} {'Gap(raw)':>10} {'Gap(adj)':>10}")
print("-" * 110)

adj_gaps = []
for _, row in df.iterrows():
    raw_gap = row['gap']
    
    # Approach 1: Subtract lost benefit
    adj1 = abs(row['qpu_obj']) - row['estimated_lost_benefit']
    
    # Approach 2: Scale by feasible rate
    feasible_rate = 1 - (row['violation_rate'] / 100)
    adj2 = abs(row['qpu_obj']) * feasible_rate
    
    adj_gap = adj1 - row['gurobi_obj']
    adj_gaps.append(adj_gap)
    
    print(f"{row['scenario']:<30} {row['gurobi_obj']:>10.2f} {abs(row['qpu_obj']):>10.2f} "
          f"{adj1:>10.2f} {adj2:>10.2f} {raw_gap:>10.2f} {adj_gap:>10.2f}")

print("-" * 110)
print(f"{'Average gap reduction':<30} {'':>10} {'':>10} {'':>10} {'':>10} "
      f"{df['gap'].mean():>10.2f} {np.mean(adj_gaps):>10.2f}")

# What percentage of gap is explained by violations?
explained_pct = (df['gap'].sum() - sum(adj_gaps)) / df['gap'].sum() * 100 if df['gap'].sum() > 0 else 0
print(f"\nPercentage of gap explained by violations: {explained_pct:.1f}%")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Violation rate by scenario
ax = axes[0, 0]
colors = ['red' if r > 10 else 'orange' if r > 5 else 'green' for r in df['violation_rate']]
bars = ax.bar(range(len(df)), df['violation_rate'], color=colors, alpha=0.8, edgecolor='black')
ax.set_xlabel('Scenario (sorted by size)', fontsize=12, fontweight='bold')
ax.set_ylabel('Violation Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('One-Hot Violation Rate by Scenario', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(df)))
ax.set_xticklabels([f"{v:,}" for v in df['n_vars']], rotation=45, ha='right', fontsize=8)
ax.axhline(y=df['violation_rate'].mean(), color='blue', linestyle='--', label=f"Avg: {df['violation_rate'].mean():.1f}%")
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Violations vs Slots (shows scaling)
ax = axes[0, 1]
ax.scatter(df['total_slots'], df['one_hot_viols'], s=100, c='coral', alpha=0.8, edgecolors='black')
# Fit line
coef = np.polyfit(df['total_slots'], df['one_hot_viols'], 1)
x_fit = np.linspace(df['total_slots'].min(), df['total_slots'].max(), 100)
ax.plot(x_fit, coef[0] * x_fit + coef[1], '--', color='gray', 
        label=f"Rate: {coef[0]*100:.1f}% of slots")
ax.set_xlabel('Total Slots (Farms × Periods)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Violations', fontsize=12, fontweight='bold')
ax.set_title('Violations Scale with Problem Size', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Gap breakdown
ax = axes[0, 2]
x = np.arange(len(df))
width = 0.35
ax.bar(x - width/2, df['gap'], width, label='Total Gap', color='red', alpha=0.8)
ax.bar(x + width/2, df['estimated_lost_benefit'], width, label='Est. Violation Impact', color='blue', alpha=0.8)
ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
ax.set_ylabel('Objective Units', fontsize=12, fontweight='bold')
ax.set_title('Gap vs Estimated Violation Impact', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{v:,}" for v in df['n_vars']], rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Adjusted objective comparison
ax = axes[1, 0]
x = np.arange(len(df))
width = 0.25
ax.bar(x - width, df['gurobi_obj'], width, label='Gurobi', color='green', alpha=0.8)
ax.bar(x, df['qpu_obj'].abs(), width, label='QPU (raw)', color='red', alpha=0.8)
adj_obj = df['qpu_obj'].abs() - df['estimated_lost_benefit']
ax.bar(x + width, adj_obj, width, label='QPU (adjusted)', color='blue', alpha=0.8)
ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
ax.set_ylabel('Objective Value', fontsize=12, fontweight='bold')
ax.set_title('Objective: Raw vs Violation-Adjusted', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{v:,}" for v in df['n_vars']], rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Plot 5: Ratio comparison
ax = axes[1, 1]
raw_ratio = df['qpu_obj'].abs() / df['gurobi_obj']
adj_ratio = adj_obj / df['gurobi_obj']
ax.plot(df['n_vars'], raw_ratio, 'ro-', label='Raw QPU/Gurobi', markersize=10, linewidth=2)
ax.plot(df['n_vars'], adj_ratio, 'bs-', label='Adjusted QPU/Gurobi', markersize=10, linewidth=2)
ax.axhline(y=1.0, color='green', linestyle='--', label='Parity (1.0)', linewidth=1.5)
ax.set_xlabel('Number of Variables', fontsize=12, fontweight='bold')
ax.set_ylabel('Objective Ratio', fontsize=12, fontweight='bold')
ax.set_title('Objective Ratio: Before vs After Adjustment', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 6: Summary statistics
ax = axes[1, 2]
ax.axis('off')

summary_text = f"""
VIOLATION IMPACT SUMMARY

Total scenarios: {len(df)}
Total farm-period slots: {total_slots:,}
Total one-hot violations: {total_viols:,}

Overall violation rate: {total_viols/total_slots*100:.2f}%
Violation rate range: {df['violation_rate'].min():.1f}% - {df['violation_rate'].max():.1f}%

Gap Analysis:
  Total raw gap: {df['gap'].sum():.1f}
  Gap per violation: {gap_per_viol:.2f}
  Correlation (viols vs gap): {corr_viols_gap:.4f}

After Adjustment:
  Avg raw ratio (QPU/Gurobi): {raw_ratio.mean():.2f}x
  Avg adjusted ratio: {adj_ratio.mean():.2f}x
  Gap explained by violations: {explained_pct:.1f}%

INTERPRETATION:
Violations account for {explained_pct:.0f}% of the
objective gap. The remaining gap may be due to:
- Decomposition approximation errors
- Boundary effects between clusters
- Stochastic sampling variance
"""

ax.text(0.05, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()

output_dir = Path('professional_plots')
plt.savefig(output_dir / 'violation_impact_assessment.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'violation_impact_assessment.pdf', bbox_inches='tight')
print(f"\n✓ Saved plots to {output_dir}/violation_impact_assessment.*")
plt.close()

# Save data
df.to_csv(output_dir / 'violation_impact_data.csv', index=False)
print(f"✓ Saved data to {output_dir}/violation_impact_data.csv")

# Final conclusion
print("\n" + "=" * 120)
print("CONCLUSIONS")
print("=" * 120)
print(f"""
1. VIOLATION RATE: {total_viols/total_slots*100:.1f}% of farm-period slots have no crop assigned
   - This is a relatively low rate but compounds across scenarios
   
2. GAP EXPLANATION: Violations explain approximately {explained_pct:.0f}% of the objective gap
   - Each violation contributes ~{gap_per_viol:.1f} units to the gap
   
3. AFTER ADJUSTMENT: 
   - Raw QPU/Gurobi ratio: {raw_ratio.mean():.2f}x average
   - Adjusted ratio: {adj_ratio.mean():.2f}x average
   - Significant improvement but gap remains
   
4. REMAINING GAP CAUSES:
   - Decomposition boundary effects
   - Cluster coordination imperfections  
   - Stochastic nature of quantum annealing
   - Possible local optima in subproblems

5. RECOMMENDATIONS:
   - Increase one-hot penalty weights to reduce violation rate
   - Add post-processing repair for remaining violations
   - Consider tighter cluster coordination
   - Report both raw and adjusted metrics
""")
