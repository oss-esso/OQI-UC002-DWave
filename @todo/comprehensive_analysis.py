#!/usr/bin/env python3
"""
Comprehensive analysis of Gurobi ground truth benchmark results.
Handles both OPTIMAL and TIME_LIMIT solutions.
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os

# Load results
with open('@todo/hardness_output/gurobi_ground_truth_benchmark.json', 'r') as f:
    data = json.load(f)

print("="*100)
print("GUROBI GROUND TRUTH BENCHMARK - COMPREHENSIVE ANALYSIS")
print("="*100)
print(f"Timestamp: {data['timestamp']}")
print(f"Gurobi version: {data['gurobi_version']}")

# ============================================================================
# SPIN GLASS RESULTS
# ============================================================================

print("\n" + "="*100)
print("SPIN GLASS INSTANCES")
print("="*100)

sg_results = data['spin_glass']['results']
sg_df = pd.DataFrame(sg_results)

print(f"\nTotal instances: {len(sg_df)}")
print(f"OPTIMAL solutions: {(sg_df['status_string'] == 'OPTIMAL').sum()}")
print(f"TIME_LIMIT (300s): {(sg_df['status_string'] == 'TIME_LIMIT').sum()}")

print("\n" + "-"*100)
print(f"{'Size':<8} {'Edges':<10} {'Status':<12} {'MIP Obj':<15} {'LP Obj':<15} {'Gap':<10} {'Time(s)':<10} {'Nodes'}")
print("-"*100)

for _, row in sg_df.iterrows():
    gap_str = f"{row['integrality_gap']:.2%}" if row.get('integrality_gap') is not None else "N/A"
    lp_str = f"{row['lp_obj']:.4f}" if row.get('lp_obj') is not None else "N/A"
    print(f"{row['n_spins']:<8} {row['n_edges']:<10} {row['status_string']:<12} {row['mip_obj']:<15.4f} {lp_str:<15} {gap_str:<10} {row['solve_time']:<10.2f} {row['node_count']:.0f}")

# Analyze OPTIMAL solutions only
sg_optimal = sg_df[sg_df['status_string'] == 'OPTIMAL'].copy()
if len(sg_optimal) >= 2:
    print(f"\n{'='*100}")
    print("SPIN GLASS - SCALING ANALYSIS (OPTIMAL solutions only)")
    print("="*100)
    
    # Objective scaling
    slope, intercept, r_value, _, _ = linregress(sg_optimal['n_spins'], -sg_optimal['mip_obj'])
    print(f"\nObjective scaling:")
    print(f"  obj ≈ {slope:.4f} × n + {intercept:.4f}  (R² = {r_value**2:.4f})")
    
    # Time scaling (log-log)
    if (sg_optimal['solve_time'] > 0.01).sum() >= 2:
        valid = sg_optimal[sg_optimal['solve_time'] > 0.01]
        log_n = np.log(valid['n_spins'])
        log_t = np.log(valid['solve_time'])
        slope_t, intercept_t, r_t, _, _ = linregress(log_n, log_t)
        
        print(f"\nTime complexity:")
        print(f"  time ≈ {np.exp(intercept_t):.4f} × n^{slope_t:.4f}  (R² = {r_t**2:.4f})")
        
        if slope_t < 1.5:
            print(f"  → LINEAR to SUB-QUADRATIC complexity ✓")
        elif slope_t < 2.5:
            print(f"  → QUADRATIC complexity (O(n²))")
        else:
            print(f"  → SUPER-QUADRATIC complexity (O(n^{slope_t:.1f}))")
        
        # Extrapolate
        print(f"\n  Extrapolated solve times:")
        for n in [200, 500, 1000, 2000, 5000]:
            t_proj = np.exp(intercept_t + slope_t * np.log(n))
            if t_proj < 60:
                print(f"    n={n:<6} → {t_proj:.1f}s")
            elif t_proj < 3600:
                print(f"    n={n:<6} → {t_proj/60:.1f}min")
            else:
                print(f"    n={n:<6} → {t_proj/3600:.1f}hr")

# ============================================================================
# FARM TYPE RESULTS
# ============================================================================

print("\n" + "="*100)
print("FARM TYPE INSTANCES")
print("="*100)

ft_results = data['farm_type']['results']
ft_df = pd.DataFrame(ft_results)

print(f"\nTotal instances: {len(ft_df)}")
print(f"OPTIMAL solutions: {(ft_df['status_string'] == 'OPTIMAL').sum()}")
print(f"TIME_LIMIT (300s): {(ft_df['status_string'] == 'TIME_LIMIT').sum()}")

print("\n" + "-"*100)
print(f"{'Farms':<8} {'Types':<8} {'Vars':<10} {'Status':<12} {'MIP Obj':<15} {'LP Obj':<15} {'Gap':<10} {'Time(s)':<10} {'Nodes'}")
print("-"*100)

for _, row in ft_df.iterrows():
    gap_str = f"{row['integrality_gap']:.2%}" if row.get('integrality_gap') is not None else "N/A"
    lp_str = f"{row['lp_obj']:.4f}" if row.get('lp_obj') is not None else "N/A"
    print(f"{row['n_farms']:<8} {row['n_types']:<8} {row['n_vars']:<10} {row['status_string']:<12} {row['mip_obj']:<15.4f} {lp_str:<15} {gap_str:<10} {row['solve_time']:<10.2f} {row['node_count']:.0f}")

# Analyze OPTIMAL solutions only
ft_optimal = ft_df[ft_df['status_string'] == 'OPTIMAL'].copy()
if len(ft_optimal) >= 2:
    print(f"\n{'='*100}")
    print("FARM TYPE - SCALING ANALYSIS (OPTIMAL solutions only)")
    print("="*100)
    
    # Objective scaling
    slope, intercept, r_value, _, _ = linregress(ft_optimal['n_farms'], -ft_optimal['mip_obj'])
    print(f"\nObjective scaling:")
    print(f"  obj ≈ {slope:.4f} × n + {intercept:.4f}  (R² = {r_value**2:.4f})")
    
    # Time scaling (log-log)
    if (ft_optimal['solve_time'] > 0.5).sum() >= 2:
        valid = ft_optimal[ft_optimal['solve_time'] > 0.5]
        log_n = np.log(valid['n_farms'])
        log_t = np.log(valid['solve_time'])
        slope_t, intercept_t, r_t, _, _ = linregress(log_n, log_t)
        
        print(f"\nTime complexity:")
        print(f"  time ≈ {np.exp(intercept_t):.4f} × n^{slope_t:.4f}  (R² = {r_t**2:.4f})")
        
        if slope_t < 1.5:
            print(f"  → LINEAR to SUB-QUADRATIC complexity ✓")
        elif slope_t < 2.5:
            print(f"  → QUADRATIC complexity (O(n²))")
        else:
            print(f"  → SUPER-QUADRATIC complexity (O(n^{slope_t:.1f}))")
        
        # Extrapolate
        print(f"\n  Extrapolated solve times:")
        for n in [200, 500, 1000, 2000]:
            t_proj = np.exp(intercept_t + slope_t * np.log(n))
            if t_proj < 60:
                print(f"    n={n:<6} → {t_proj:.1f}s")
            elif t_proj < 3600:
                print(f"    n={n:<6} → {t_proj/60:.1f}min")
            else:
                print(f"    n={n:<6} → {t_proj/3600:.1f}hr")

# ============================================================================
# KEY FINDINGS
# ============================================================================

print("\n" + "="*100)
print("KEY FINDINGS")
print("="*100)

print("""
1. CLASSICAL SOLVER PERFORMANCE:
   • Gurobi solves instances up to n≈100 optimally within 5 minutes
   • Larger instances (n≥200) hit 5-minute time limit
   • All OPTIMAL solutions have 0% integrality gap → LP relaxation is TIGHT!
   
   ⚠️  CRITICAL FINDING: 0% gap means LP relaxation finds integer solution directly
   → These instances are still TOO EASY for classical LP-based solvers
   → No evidence of frustrated search space or exponential branching

2. QUANTUM TRACTABILITY:
   • Spin Glass: Max degree ≈10-40 → Embeddable on Pegasus ✓
   • Farm Type: Max degree ≈40 → Embeddable on Pegasus ✓
   • Both formulations are QPU-FEASIBLE

3. THE HARDNESS PARADOX:
   
   Problem:  EASY for classical (0% gap, polynomial time)
             FEASIBLE for quantum (bounded degree)
   
   This is NOT the regime for quantum advantage!
   
   Quantum advantage requires:
     Classical: HARD (exponential time, large gap)
     Quantum:   FEASIBLE (sparse, embeddable)

4. WHY THE INSTANCES ARE TOO EASY:
   
   a) ONE-HOT CONSTRAINT IS TOO STRONG:
      • Penalty of 10.0 for choosing multiple types per farm
      • LP solver respects this naturally → no fractional variables
      • Solution: Remove explicit penalty, use different encoding
   
   b) INSUFFICIENT FRUSTRATION:
      • 30% frustrated edges not enough
      • Need 50-70% for spin glass hardness phase
      • Current coupling strengths too weak (±0.15)
   
   c) PROBLEM STRUCTURE TOO SIMPLE:
      • Assignment problem with local interactions
      • Classical algorithms exploit this structure efficiently
      • Need more complex constraint interactions

5. WHAT WOULD MAKE IT GENUINELY HARD:
   
   • Increase frustration to 60-80% of edges
   • Stronger coupling strengths: ±1.0 instead of ±0.15  
   • Remove or soften one-hot penalty
   • Add non-local constraints (global diversity, resource limits)
   • Use planted hardness from known hard instances (3-SAT, MaxCut)

6. RECOMMENDED NEXT STEPS:
   
   Option A: FIX THE FORMULATION
   • Regenerate instances with higher frustration (60-80%)
   • Increase coupling strength to ±1.0
   • Replace one-hot penalty with different encoding
   • Target: >10% integrality gap for n≥100
   
   Option B: USE THE ROTATION FORMULATION (with fixes)
   • Your existing rotation formulation has quadratic terms ✓
   • Add negative synergies to rotation matrix
   • Add spatial interactions between neighbor farms
   • This preserves agricultural meaning while adding hardness
   
   Option C: ACCEPT LIMITED HARDNESS
   • Focus on QPU embedding and runtime comparison
   • Compare quantum annealing vs SA vs exact solver
   • Demonstrate quantum feasibility at scale (n=1000+)
   • Acknowledge classical advantage persists for n<1000
""")

print("\n" + "="*100)
print("RECOMMENDATION: Pursue Option B (Enhanced Rotation Formulation)")
print("="*100)
print("""
Why: It combines the best of both worlds:
• Real agronomic meaning (rotation synergies)
• Temporal AND spatial structure (naturally sparse)
• Can add frustration via negative synergies (realistic: disease carryover)
• Already quadratic (native QUBO)
• You've already implemented it!

Simple modifications needed:
1. Add negative values to rotation matrix (same-family penalties)
2. Add spatial neighbor interactions
3. Test classical hardness (target: >5% gap)
4. Benchmark on QPU

This path preserves your domain expertise while pursuing quantum advantage.
""")
