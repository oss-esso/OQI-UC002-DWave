#!/usr/bin/env python3
"""Extract key metrics from CSV for report generation"""
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv('hardness_analysis_results/hardness_analysis_results.csv')

print("# Performance Metrics for Report\n")
print("## Main Table")
print("\n| Farms | Vars | Area/Farm | Total Area | Solve(s) | Build(s) | Quads | Gap% | Category |")
print("|------:|-----:|----------:|-----------:|---------:|---------:|------:|-----:|----------|")

for _, row in df.iterrows():
    area_per_farm = row['total_area'] / row['n_farms']
    gap_pct = row['gap'] * 100
    print(f"| {int(row['n_farms'])} | {int(row['n_vars'])} | {area_per_farm:.1f} | {row['total_area']:.1f} | {row['solve_time']:.2f} | {row['build_time']:.2f} | {int(row['n_quadratic'])} | {gap_pct:.2f} | {row['time_category']} |")

print("\n## Summary by Category\n")

for category in ['FAST', 'MEDIUM', 'SLOW', 'TIMEOUT']:
    cat_df = df[df['time_category'] == category]
    if len(cat_df) > 0:
        print(f"### {category} ({len(cat_df)} instances)\n")
        print("| Metric | Min | Max | Mean | Std |")
        print("|--------|----:|----:|-----:|----:|")
        print(f"| Farms           | {cat_df['n_farms'].min():.0f} | {cat_df['n_farms'].max():.0f} | {cat_df['n_farms'].mean():.2f} | {cat_df['n_farms'].std():.2f} |")
        print(f"| Variables       | {cat_df['n_vars'].min():.0f} | {cat_df['n_vars'].max():.0f} | {cat_df['n_vars'].mean():.2f} | {cat_df['n_vars'].std():.2f} |")
        print(f"| Farms/Food      | {cat_df['farms_per_food'].min():.2f} | {cat_df['farms_per_food'].max():.2f} | {cat_df['farms_per_food'].mean():.2f} | {cat_df['farms_per_food'].std():.2f} |")
        print(f"| Total Area (ha) | {cat_df['total_area'].min():.2f} | {cat_df['total_area'].max():.2f} | {cat_df['total_area'].mean():.2f} | {cat_df['total_area'].std():.2f} |")
        print(f"| Solve Time (s)  | {cat_df['solve_time'].min():.2f} | {cat_df['solve_time'].max():.2f} | {cat_df['solve_time'].mean():.2f} | {cat_df['solve_time'].std():.2f} |")
        print(f"| Quadratics      | {cat_df['n_quadratic'].min():.0f} | {cat_df['n_quadratic'].max():.0f} | {cat_df['n_quadratic'].mean():.0f} | {cat_df['n_quadratic'].std():.0f} |")
        print(f"| MIP Gap (%)     | {cat_df['gap'].min()*100:.2f} | {cat_df['gap'].max()*100:.2f} | {cat_df['gap'].mean()*100:.2f} | {cat_df['gap'].std()*100:.2f} |\n")

# Correlations
print("## Correlations with Solve Time\n")
print("| Metric | Correlation (r) |")
print("|--------|----------------:|")
corr_vars = ['n_farms', 'farms_per_food', 'n_constraints', 'n_vars', 'n_quadratic', 'build_time']
for var in corr_vars:
    corr = df[['solve_time', var]].corr().iloc[0, 1]
    print(f"| {var.replace('_', ' ').title()} | {corr:.3f} |")
