#!/usr/bin/env python3
"""
Deeper investigation of the objective gap.
Since violations only explain 7%, what else causes the 93%?
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
with open('qpu_hier_repaired.json') as f:
    qpu_data = json.load(f)

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
            'status': result.get('status', 'unknown'),
            'mip_gap': result.get('mip_gap', 0),
            'hit_timeout': result.get('hit_timeout', False),
        }

print('=' * 120)
print('DEEP DIVE: UNDERSTANDING THE OBJECTIVE GAP')
print('=' * 120)

# Key insight: QPU objectives are NEGATIVE (costs/penalties), Gurobi are POSITIVE (benefits)
print("\n1. SIGN ANALYSIS")
print("-" * 80)

for r in qpu_data['runs'][:5]:
    sc = r['scenario_name']
    qpu_obj = r['objective_miqp']
    gur_obj = gurobi.get(sc, {}).get('objective', 0)
    print(f"  {sc}: QPU = {qpu_obj:.2f}, Gurobi = {gur_obj:.2f}")
    print(f"    QPU sign: {'NEGATIVE' if qpu_obj < 0 else 'POSITIVE'}")
    print(f"    Gurobi sign: {'NEGATIVE' if gur_obj < 0 else 'POSITIVE'}")

print("\n  OBSERVATION: QPU objectives are NEGATIVE, Gurobi are POSITIVE!")
print("  This suggests fundamentally different objective formulations!")

# Check if this is a sign convention issue
print("\n2. OBJECTIVE FORMULATION CHECK")
print("-" * 80)

print("  Hypothesis: Are we comparing apples to oranges?")
print("  - Gurobi might be maximizing benefit (positive values = good)")
print("  - QPU might be minimizing QUBO energy (negative values = good for minimization)")
print()

# If we negate QPU objectives
print("  If QPU objectives were negated (sign flip):")
for r in qpu_data['runs'][:5]:
    sc = r['scenario_name']
    qpu_obj = -r['objective_miqp']  # NEGATE
    gur_obj = gurobi.get(sc, {}).get('objective', 0)
    ratio = qpu_obj / gur_obj if gur_obj != 0 else 0
    print(f"    {sc}: QPU = {qpu_obj:.2f}, Gurobi = {gur_obj:.2f}, Ratio = {ratio:.2f}x")

print("\n3. THE REAL ISSUE: WHAT IS 'objective_miqp'?")
print("-" * 80)

# Look at how objective_miqp is calculated
# The MIQP objective is the benefit function: sum(x * benefits)
# If QPU returns negative values, either:
#   a) The benefit calculation is different
#   b) There's a sign convention issue
#   c) The QPU is optimizing a transformed problem

print("""
  The MIQP objective should be: Σ benefit_coefficients × allocation
  
  For Gurobi: Maximizing benefit → positive objective = good
  For QPU: The QUBO is created for minimization
           objective_miqp might be re-calculated from the binary solution
           
  Let's check the violation details to understand better...
""")

# Analyze violation details
print("\n4. VIOLATION DETAIL ANALYSIS")
print("-" * 80)

for r in qpu_data['runs'][:3]:
    sc = r['scenario_name']
    viols = r['constraint_violations']
    details = viols.get('details', [])
    
    print(f"\n  {sc}:")
    print(f"    Total violations: {viols.get('total_violations', 0)}")
    print(f"    One-hot violations: {viols.get('one_hot_violations', 0)}")
    print(f"    Rotation violations: {viols.get('rotation_violations', 0)}")
    print(f"    Details (first 5):")
    for d in details[:5]:
        print(f"      - {d}")

print("\n5. UNDERSTANDING 'min_crops' VIOLATIONS")
print("-" * 80)

print("""
  All violations are type 'min_crops':
  - This means: A farm-period has count=0 crops assigned
  - Expected: At least 1 crop per farm per period
  
  This is NOT a one-hot violation in the traditional sense!
  It's a "no selection made" violation - the QPU didn't assign ANY crop.
  
  Impact: If no crop is assigned to a farm-period:
  - Lost benefit from that slot
  - But also no penalty from that slot
  - Net effect depends on the benefit function
""")

# Calculate actual impact
print("\n6. RECALCULATING IMPACT CORRECTLY")
print("-" * 80)

results = []
for r in qpu_data['runs']:
    sc = r['scenario_name']
    n_farms = r['n_farms']
    n_periods = r['n_periods']
    total_slots = n_farms * n_periods
    
    qpu_obj = r['objective_miqp']
    gur_obj = gurobi.get(sc, {}).get('objective', 0)
    viols = r['constraint_violations'].get('one_hot_violations', 0)
    
    # The violation means: farm-period with no crop
    # This loses the benefit that would have come from that slot
    # Average benefit per slot = Gurobi_obj / total_slots
    avg_benefit = gur_obj / total_slots if total_slots > 0 else 0
    
    # If QPU had assigned crops to all slots, objective would increase by:
    potential_gain = viols * avg_benefit
    
    # "Corrected" QPU objective (if violations were fixed)
    # Since QPU obj is negative, we ADD the potential gain (less negative = better)
    corrected_qpu = qpu_obj + potential_gain  # This will be less negative
    
    # The actual |QPU| comparable to Gurobi would be:
    comparable_qpu = abs(qpu_obj) - potential_gain
    
    results.append({
        'scenario': sc,
        'gurobi': gur_obj,
        'qpu_raw': qpu_obj,
        'qpu_abs': abs(qpu_obj),
        'violations': viols,
        'potential_gain': potential_gain,
        'qpu_corrected': corrected_qpu,
        'comparable': comparable_qpu,
        'ratio_raw': abs(qpu_obj) / gur_obj if gur_obj > 0 else 0,
        'ratio_corrected': comparable_qpu / gur_obj if gur_obj > 0 else 0,
    })

df = pd.DataFrame(results)

print(f"\n{'Scenario':<30} {'Gurobi':>10} {'|QPU|':>10} {'Viols':>6} {'PotGain':>10} {'Comparable':>12} {'Ratio':>8}")
print("-" * 100)
for _, row in df.iterrows():
    print(f"{row['scenario']:<30} {row['gurobi']:>10.2f} {row['qpu_abs']:>10.2f} "
          f"{row['violations']:>6} {row['potential_gain']:>10.2f} {row['comparable']:>12.2f} {row['ratio_corrected']:>8.2f}x")

print("\n7. THE REAL COMPARISON")
print("-" * 80)

avg_raw_ratio = df['ratio_raw'].mean()
avg_corrected_ratio = df['ratio_corrected'].mean()

print(f"  Average raw ratio |QPU|/Gurobi: {avg_raw_ratio:.2f}x")
print(f"  Average corrected ratio: {avg_corrected_ratio:.2f}x")
print(f"  Improvement from correction: {(avg_raw_ratio - avg_corrected_ratio)/avg_raw_ratio*100:.1f}%")

# But wait - the ratio is still ~3.5x even after correction
# This means violations explain only part of the gap

total_gap = (df['qpu_abs'] - df['gurobi']).sum()
violation_explained = df['potential_gain'].sum()
pct_explained = violation_explained / total_gap * 100

print(f"\n  Total objective gap: {total_gap:.2f}")
print(f"  Gap explained by violations: {violation_explained:.2f} ({pct_explained:.1f}%)")
print(f"  Remaining unexplained gap: {total_gap - violation_explained:.2f} ({100-pct_explained:.1f}%)")

print("\n8. WHAT CAUSES THE REMAINING 93% OF THE GAP?")
print("-" * 80)

print("""
  Since violations only explain ~7% of the gap, the remaining ~93% must be due to:
  
  1. DECOMPOSITION APPROXIMATION (likely major factor)
     - Hierarchical method solves subproblems independently
     - Global optimum ≠ sum of local optima
     - Boundary effects between clusters
     
  2. DIFFERENT OPTIMIZATION LANDSCAPE
     - QUBO formulation transforms the problem
     - Penalties for soft constraints change the landscape
     - Local minima in QUBO differ from MIQP optima
     
  3. STOCHASTIC SAMPLING
     - Quantum annealing finds low-energy states
     - Not guaranteed to find global minimum
     - Multiple samples might help but still probabilistic
     
  4. CHAIN BREAKS & EMBEDDING NOISE
     - Physical embedding introduces noise
     - Chain breaks corrupt solutions
     - Minor embedding not perfect
     
  5. PENALTY WEIGHT CALIBRATION
     - Constraint penalties may not be optimal
     - Trade-off between feasibility and optimality
""")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Objective comparison
ax = axes[0, 0]
x = np.arange(len(df))
width = 0.25
ax.bar(x - width, df['gurobi'], width, label='Gurobi', color='green', alpha=0.8)
ax.bar(x, df['qpu_abs'], width, label='|QPU| (raw)', color='red', alpha=0.8)
ax.bar(x + width, df['comparable'], width, label='|QPU| (corrected)', color='blue', alpha=0.8)
ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
ax.set_ylabel('Objective Value', fontsize=12, fontweight='bold')
ax.set_title('Objective Comparison: Raw vs Corrected', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{int(r['violations'])}" for r in results], fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Ratio comparison
ax = axes[0, 1]
ax.bar(x - 0.2, df['ratio_raw'], 0.4, label='Raw ratio', color='red', alpha=0.8)
ax.bar(x + 0.2, df['ratio_corrected'], 0.4, label='Corrected ratio', color='blue', alpha=0.8)
ax.axhline(y=1.0, color='green', linestyle='--', label='Parity (1.0)', linewidth=2)
ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
ax.set_ylabel('Ratio (QPU / Gurobi)', fontsize=12, fontweight='bold')
ax.set_title('Ratio Analysis: Violations Have Minor Impact', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{int(r['violations'])}" for r in results], fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Gap breakdown pie chart
ax = axes[0, 2]
sizes = [pct_explained, 100 - pct_explained]
labels = [f'Violations\n({pct_explained:.0f}%)', f'Other factors\n({100-pct_explained:.0f}%)']
colors = ['coral', 'lightblue']
explode = (0.05, 0)
ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.0f%%',
       shadow=True, startangle=90, textprops={'fontsize': 11})
ax.set_title('Gap Attribution', fontsize=14, fontweight='bold')

# Plot 4: Scaling analysis
ax = axes[1, 0]
ax.scatter(df['gurobi'], df['qpu_abs'], s=100, c='red', alpha=0.7, label='Raw', edgecolors='black')
ax.scatter(df['gurobi'], df['comparable'], s=100, c='blue', alpha=0.7, label='Corrected', edgecolors='black')
max_val = max(df['qpu_abs'].max(), df['gurobi'].max())
ax.plot([0, max_val], [0, max_val], 'g--', label='Parity', linewidth=2)
ax.set_xlabel('Gurobi Objective', fontsize=12, fontweight='bold')
ax.set_ylabel('QPU Objective', fontsize=12, fontweight='bold')
ax.set_title('QPU vs Gurobi: Violation Correction Impact', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Violation rate vs gap percentage
ax = axes[1, 1]
viol_pcts = []
for _, row in df.iterrows():
    total_gap = row['qpu_abs'] - row['gurobi']
    if total_gap > 0:
        viol_pct = row['potential_gain'] / total_gap * 100
    else:
        viol_pct = 0
    viol_pcts.append(viol_pct)

ax.bar(range(len(df)), viol_pcts, color='coral', alpha=0.8, edgecolor='black')
ax.axhline(y=np.mean(viol_pcts), color='blue', linestyle='--', label=f'Avg: {np.mean(viol_pcts):.1f}%')
ax.set_xlabel('Scenario (by violations)', fontsize=12, fontweight='bold')
ax.set_ylabel('% of Gap Explained by Violations', fontsize=12, fontweight='bold')
ax.set_title('Violation Impact by Scenario', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(df)))
ax.set_xticklabels([f"{int(r['violations'])}" for r in results], fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 20)

# Plot 6: Summary
ax = axes[1, 2]
ax.axis('off')

summary = f"""
DEEP DIVE FINDINGS

1. VIOLATION IMPACT IS MINOR
   - Violations explain only {pct_explained:.0f}% of gap
   - Correction improves ratio from {avg_raw_ratio:.2f}x to {avg_corrected_ratio:.2f}x
   - Still ~3.5x gap remains after correction

2. THE 93% UNEXPLAINED GAP
   Main causes (in likely order of importance):
   
   a) DECOMPOSITION APPROXIMATION
      Hierarchical method ≠ global optimization
      
   b) QUBO TRANSFORMATION
      Energy landscape differs from MIQP
      
   c) LOCAL MINIMA
      Quantum annealing may not find global min
      
   d) EMBEDDING NOISE
      Chain breaks and physical imperfections

3. KEY INSIGHT
   Fixing violations would NOT make QPU
   competitive with Gurobi on solution quality.
   The fundamental gap is algorithmic, not
   due to constraint satisfaction failures.

4. IMPLICATIONS
   - Post-processing repair has limited value
   - Better decomposition strategies needed
   - Consider hybrid classical-quantum approaches
"""

ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()

output_dir = Path('professional_plots')
plt.savefig(output_dir / 'gap_deep_dive.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'gap_deep_dive.pdf', bbox_inches='tight')
print(f"\n✓ Saved plots to {output_dir}/gap_deep_dive.*")
plt.close()
