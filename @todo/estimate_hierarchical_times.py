#!/usr/bin/env python3
"""
Estimate Hierarchical QPU Time for All 20 Test Scenarios

Based on empirical data from hierarchical_statistical_test.py results:
- 25 farms: 34.31s total, 0.60s QPU (5 clusters, 15 QPU calls)
- 50 farms: 69.61s total, 1.19s QPU (10 clusters, 30 QPU calls)
- 100 farms: 136.00s total, 2.38s QPU (20 clusters, 60 QPU calls)

Estimates time for all 20 Gurobi timeout test scenarios (90 to 81,000 vars).
"""

import json
import pandas as pd
from pathlib import Path

# Load empirical data from hierarchical tests
EMPIRICAL_DATA = {
    25: {'total_time': 34.31, 'qpu_time': 0.60, 'clusters': 5, 'qpu_calls': 15},
    50: {'total_time': 69.61, 'qpu_time': 1.19, 'clusters': 10, 'qpu_calls': 30},
    100: {'total_time': 136.00, 'qpu_time': 2.38, 'clusters': 20, 'qpu_calls': 60},
}

# Configuration from hierarchical test
FARMS_PER_CLUSTER = 5
NUM_ITERATIONS = 3  # Boundary coordination iterations

# All 20 test scenarios from the Gurobi timeout test
ALL_SCENARIOS = [
    {'name': 'rotation_micro_25', 'n_farms': 5, 'n_foods': 6, 'n_periods': 3, 'n_vars': 90},
    {'name': 'rotation_small_50', 'n_farms': 10, 'n_foods': 6, 'n_periods': 3, 'n_vars': 180},
    {'name': 'rotation_15farms_6foods', 'n_farms': 15, 'n_foods': 6, 'n_periods': 3, 'n_vars': 270},
    {'name': 'rotation_medium_100', 'n_farms': 20, 'n_foods': 6, 'n_periods': 3, 'n_vars': 360},
    {'name': 'rotation_25farms_6foods', 'n_farms': 25, 'n_foods': 6, 'n_periods': 3, 'n_vars': 450},
    {'name': 'rotation_large_200', 'n_farms': 40, 'n_foods': 6, 'n_periods': 3, 'n_vars': 720},
    {'name': 'rotation_50farms_6foods', 'n_farms': 50, 'n_foods': 6, 'n_periods': 3, 'n_vars': 900},
    {'name': 'rotation_75farms_6foods', 'n_farms': 75, 'n_foods': 6, 'n_periods': 3, 'n_vars': 1350},
    {'name': 'rotation_100farms_6foods', 'n_farms': 100, 'n_foods': 6, 'n_periods': 3, 'n_vars': 1800},
    {'name': 'rotation_25farms_27foods', 'n_farms': 25, 'n_foods': 27, 'n_periods': 3, 'n_vars': 2025},
    {'name': 'rotation_150farms_6foods', 'n_farms': 150, 'n_foods': 6, 'n_periods': 3, 'n_vars': 2700},
    {'name': 'rotation_50farms_27foods', 'n_farms': 50, 'n_foods': 27, 'n_periods': 3, 'n_vars': 4050},
    {'name': 'rotation_75farms_27foods', 'n_farms': 75, 'n_foods': 27, 'n_periods': 3, 'n_vars': 6075},
    {'name': 'rotation_100farms_27foods', 'n_farms': 100, 'n_foods': 27, 'n_periods': 3, 'n_vars': 8100},
    {'name': 'rotation_150farms_27foods', 'n_farms': 150, 'n_foods': 27, 'n_periods': 3, 'n_vars': 12150},
    {'name': 'rotation_200farms_27foods', 'n_farms': 200, 'n_foods': 27, 'n_periods': 3, 'n_vars': 16200},
    {'name': 'rotation_250farms_27foods', 'n_farms': 250, 'n_foods': 27, 'n_periods': 3, 'n_vars': 20250},
    {'name': 'rotation_350farms_27foods', 'n_farms': 350, 'n_foods': 27, 'n_periods': 3, 'n_vars': 28350},
    {'name': 'rotation_500farms_27foods', 'n_farms': 500, 'n_foods': 27, 'n_periods': 3, 'n_vars': 40500},
    {'name': 'rotation_1000farms_27foods', 'n_farms': 1000, 'n_foods': 27, 'n_periods': 3, 'n_vars': 81000},
]


def estimate_time_for_scenario(n_farms):
    """
    Estimate total time and QPU time for a given number of farms.
    
    Uses linear interpolation/extrapolation based on empirical data.
    Key assumption: Time scales linearly with number of clusters.
    """
    # Calculate clusters
    n_clusters = (n_farms + FARMS_PER_CLUSTER - 1) // FARMS_PER_CLUSTER
    qpu_calls = n_clusters * NUM_ITERATIONS
    
    # If we have exact empirical data, use it
    if n_farms in EMPIRICAL_DATA:
        data = EMPIRICAL_DATA[n_farms]
        return {
            'total_time': data['total_time'],
            'qpu_time': data['qpu_time'],
            'n_clusters': data['clusters'],
            'qpu_calls': data['qpu_calls'],
            'source': 'empirical'
        }
    
    # Otherwise, use linear interpolation/extrapolation
    # Based on: time_per_cluster = total_time / n_clusters
    
    # Calculate average time per cluster from empirical data
    time_per_cluster_samples = []
    qpu_per_cluster_samples = []
    
    for farms, data in EMPIRICAL_DATA.items():
        time_per_cluster_samples.append(data['total_time'] / data['clusters'])
        qpu_per_cluster_samples.append(data['qpu_time'] / data['clusters'])
    
    # Use average (could also use weighted average or regression)
    avg_time_per_cluster = sum(time_per_cluster_samples) / len(time_per_cluster_samples)
    avg_qpu_per_cluster = sum(qpu_per_cluster_samples) / len(qpu_per_cluster_samples)
    
    # Estimate for this scenario
    estimated_total = avg_time_per_cluster * n_clusters
    estimated_qpu = avg_qpu_per_cluster * n_clusters
    
    return {
        'total_time': estimated_total,
        'qpu_time': estimated_qpu,
        'n_clusters': n_clusters,
        'qpu_calls': qpu_calls,
        'source': 'estimated'
    }


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m ({seconds:.0f}s)"
    else:
        hours = seconds / 3600
        mins = (seconds % 3600) / 60
        return f"{hours:.1f}h ({mins:.0f}m)"


def main():
    print("="*80)
    print("HIERARCHICAL QPU TIME ESTIMATES FOR ALL 20 TEST SCENARIOS")
    print("="*80)
    
    print("\nEmpirical Data (from hierarchical_statistical_test.py):")
    print("-"*80)
    for farms, data in sorted(EMPIRICAL_DATA.items()):
        print(f"  {farms:3d} farms: {data['total_time']:6.1f}s total, "
              f"{data['qpu_time']:5.2f}s QPU "
              f"({data['clusters']} clusters, {data['qpu_calls']} QPU calls)")
    
    # Calculate time per cluster statistics
    time_per_cluster = [d['total_time'] / d['clusters'] for d in EMPIRICAL_DATA.values()]
    qpu_per_cluster = [d['qpu_time'] / d['clusters'] for d in EMPIRICAL_DATA.values()]
    
    avg_time_per_cluster = sum(time_per_cluster) / len(time_per_cluster)
    avg_qpu_per_cluster = sum(qpu_per_cluster) / len(qpu_per_cluster)
    
    print(f"\nScaling Factors:")
    print(f"  Average time per cluster: {avg_time_per_cluster:.2f}s")
    print(f"  Average QPU per cluster: {avg_qpu_per_cluster:.3f}s")
    print(f"  Farms per cluster: {FARMS_PER_CLUSTER}")
    print(f"  Boundary iterations: {NUM_ITERATIONS}")
    
    print("\n" + "="*80)
    print("ESTIMATES FOR ALL 20 SCENARIOS")
    print("="*80)
    
    # Estimate for all scenarios
    results = []
    total_time_sum = 0
    total_qpu_sum = 0
    
    for scenario in ALL_SCENARIOS:
        n_farms = scenario['n_farms']
        n_vars = scenario['n_vars']
        
        estimate = estimate_time_for_scenario(n_farms)
        
        results.append({
            'scenario': scenario['name'],
            'n_farms': n_farms,
            'n_vars': n_vars,
            'n_clusters': estimate['n_clusters'],
            'qpu_calls': estimate['qpu_calls'],
            'total_time_s': estimate['total_time'],
            'qpu_time_s': estimate['qpu_time'],
            'source': estimate['source']
        })
        
        total_time_sum += estimate['total_time']
        total_qpu_sum += estimate['qpu_time']
        
        source_marker = "✓" if estimate['source'] == 'empirical' else "~"
        
        print(f"{source_marker} {scenario['name']:<30} "
              f"{n_farms:4d} farms, {n_vars:5d} vars -> "
              f"{estimate['n_clusters']:3d} clusters, "
              f"{estimate['qpu_calls']:3d} QPU calls -> "
              f"{format_time(estimate['total_time']):>12} total, "
              f"{format_time(estimate['qpu_time']):>10} QPU")
    
    print("="*80)
    print(f"TOTAL ESTIMATED TIME FOR ALL 20 SCENARIOS:")
    print(f"  Total Runtime: {format_time(total_time_sum):>12}")
    print(f"  Total QPU:     {format_time(total_qpu_sum):>12}")
    print(f"  QPU Efficiency: {(total_qpu_sum / total_time_sum * 100):.1f}% of total time")
    print("="*80)
    
    # Save to CSV
    df = pd.DataFrame(results)
    output_file = Path(__file__).parent / 'hierarchical_time_estimates.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Additional analysis
    print("\n" + "="*80)
    print("ANALYSIS BY PROBLEM SIZE")
    print("="*80)
    
    # Group by size categories
    small = [r for r in results if r['n_vars'] < 1000]
    medium = [r for r in results if 1000 <= r['n_vars'] < 10000]
    large = [r for r in results if r['n_vars'] >= 10000]
    
    for category_name, category in [('Small (< 1000 vars)', small), 
                                     ('Medium (1k-10k vars)', medium), 
                                     ('Large (10k+ vars)', large)]:
        if category:
            total_cat = sum(r['total_time_s'] for r in category)
            qpu_cat = sum(r['qpu_time_s'] for r in category)
            print(f"\n{category_name}: {len(category)} scenarios")
            print(f"  Total time: {format_time(total_cat)}")
            print(f"  QPU time:   {format_time(qpu_cat)}")


if __name__ == '__main__':
    main()
