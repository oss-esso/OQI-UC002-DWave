#!/usr/bin/env python3
"""
Debug script to compare rotation matrix generation between tests.
"""
import numpy as np

def generate_comprehensive_matrix(n_foods=6):
    """Rotation matrix from comprehensive_scaling_test.py"""
    np.random.seed(42)
    frustration_ratio = 0.7
    negative_strength = -0.8
    R = np.zeros((n_foods, n_foods))
    
    for i in range(n_foods):
        for j in range(n_foods):
            if i == j:
                R[i, j] = negative_strength * 1.5
            elif np.random.random() < frustration_ratio:
                R[i, j] = np.random.uniform(negative_strength * 1.2, negative_strength * 0.3)
            else:
                R[i, j] = np.random.uniform(0.02, 0.20)
    
    return R

def generate_statistical_matrix(n_families=6):
    """Rotation matrix from statistical_comparison_test.py"""
    frustration_ratio = 0.7
    negative_strength = -0.8
    
    np.random.seed(42)
    R = np.zeros((n_families, n_families))
    for i in range(n_families):
        for j in range(n_families):
            if i == j:
                R[i, j] = negative_strength * 1.5
            elif np.random.random() < frustration_ratio:
                R[i, j] = np.random.uniform(negative_strength * 1.2, negative_strength * 0.3)
            else:
                R[i, j] = np.random.uniform(0.02, 0.20)
    
    return R

print("="*80)
print("ROTATION MATRIX COMPARISON")
print("="*80)

R_comp = generate_comprehensive_matrix(6)
R_stat = generate_statistical_matrix(6)

print("\n[1] Comprehensive Test Matrix (6×6):")
print(R_comp)
print(f"\nStats: mean={R_comp.mean():.4f}, min={R_comp.min():.4f}, max={R_comp.max():.4f}")
print(f"Negative entries: {(R_comp < 0).sum()}/{R_comp.size} = {100*(R_comp < 0).sum()/R_comp.size:.1f}%")

print("\n[2] Statistical Test Matrix (6×6):")
print(R_stat)
print(f"\nStats: mean={R_stat.mean():.4f}, min={R_stat.min():.4f}, max={R_stat.max():.4f}")
print(f"Negative entries: {(R_stat < 0).sum()}/{R_stat.size} = {100*(R_stat < 0).sum()/R_stat.size:.1f}%")

print("\n[3] Difference:")
diff = np.abs(R_comp - R_stat)
print(f"Max absolute difference: {diff.max():.10f}")
print(f"Are they identical? {np.allclose(R_comp, R_stat)}")

if not np.allclose(R_comp, R_stat):
    print("\n⚠️ MATRICES ARE DIFFERENT!")
    print("Differences:")
    for i in range(6):
        for j in range(6):
            if not np.isclose(R_comp[i,j], R_stat[i,j]):
                print(f"  R[{i},{j}]: comp={R_comp[i,j]:.6f}, stat={R_stat[i,j]:.6f}, diff={R_comp[i,j]-R_stat[i,j]:.6f}")
else:
    print("\n✅ MATRICES ARE IDENTICAL!")
