#!/usr/bin/env python3
"""
Verify the actual monoculture penalty effect in our linear formulation.
"""

import numpy as np

# Current parameters from codebase
beta = -0.8  # negative_strength
monoculture_multiplier = 1.5  # From build_rotation_matrix
rotation_gamma = 0.2  # From all solvers

# Calculate monoculture penalty
R_cc = beta * monoculture_multiplier
print("="*70)
print("MONOCULTURE PENALTY ANALYSIS")
print("="*70)
print(f"\nCurrent Parameters:")
print(f"  β (negative_strength): {beta}")
print(f"  Monoculture multiplier: {monoculture_multiplier}")
print(f"  rotation_gamma: {rotation_gamma}")
print(f"\nCalculated Penalty:")
print(f"  R_c,c = β × multiplier = {beta} × {monoculture_multiplier} = {R_cc}")
print(f"\nEffective Objective Penalty:")
print(f"  penalty = rotation_gamma × R_c,c")
print(f"          = {rotation_gamma} × {R_cc}")
print(f"          = {rotation_gamma * R_cc}")

# Simulate objective contribution
base_benefit = 1.0
area_frac = 1.0  # Normalized

print(f"\n{'='*70}")
print("OBJECTIVE VALUE COMPARISON (area_frac = 1.0)")
print("="*70)

# Without rotation effect (isolated benefit)
obj_base = base_benefit * area_frac
print(f"\nBase benefit (no rotation): {obj_base:.4f}")

# With good rotation (small positive synergy)
good_rotation_bonus = 0.1
obj_good_rotation = base_benefit * area_frac + rotation_gamma * good_rotation_bonus * area_frac
print(f"With good rotation (+{good_rotation_bonus}): {obj_good_rotation:.4f} ({(obj_good_rotation/obj_base - 1)*100:+.1f}%)")

# With monoculture penalty
obj_monoculture = base_benefit * area_frac + rotation_gamma * R_cc * area_frac
reduction_pct = (1 - obj_monoculture / obj_base) * 100
print(f"With monoculture ({R_cc}): {obj_monoculture:.4f} ({-reduction_pct:.1f}%)")

print(f"\n{'='*70}")
print("TARGET: 17-20% PENALTY")
print("="*70)

# What parameters give 17-20%?
target_penalties = [0.17, 0.20]

print("\nOption 1: Adjust β (keep rotation_gamma=0.2, multiplier=1.5)")
for target in target_penalties:
    # penalty = rotation_gamma * beta * multiplier * area_frac
    # target = -rotation_gamma * beta * multiplier
    needed_beta = -target / (rotation_gamma * monoculture_multiplier)
    print(f"  For {target*100:.0f}% penalty: β = {needed_beta:.3f}")

print("\nOption 2: Adjust rotation_gamma (keep β=-0.8, multiplier=1.5)")
for target in target_penalties:
    needed_gamma = target / (-R_cc)
    print(f"  For {target*100:.0f}% penalty: rotation_gamma = {needed_gamma:.3f}")

print("\nOption 3: Adjust multiplier (keep β=-0.8, rotation_gamma=0.2)")
for target in target_penalties:
    needed_mult = -target / (rotation_gamma * beta)
    print(f"  For {target*100:.0f}% penalty: multiplier = {needed_mult:.3f}")

print(f"\n{'='*70}")
print("EXPONENTIAL INTERPRETATION (USER'S CONCERN)")
print("="*70)

print("\nIf formulation was exponential (yield = base × exp(R_c,c)):")
exp_effect_current = np.exp(R_cc)
print(f"  Current: exp({R_cc:.2f}) = {exp_effect_current:.4f} ({(1-exp_effect_current)*100:.1f}% loss)")

print("\nFor 17-20% loss in exponential model:")
for target_loss in [0.17, 0.20]:
    target_retention = 1 - target_loss
    needed_R = np.log(target_retention)
    needed_beta_exp = needed_R / monoculture_multiplier
    print(f"  {target_loss*100:.0f}% loss: need R = {needed_R:.3f}, β = {needed_beta_exp:.3f}")

print("\n⚠️  BUT OUR FORMULATION IS LINEAR, NOT EXPONENTIAL!")
print("   The exponential values don't apply to our model.")

print(f"\n{'='*70}")
print("CONCLUSION")
print("="*70)
print(f"\nCurrent setup gives {reduction_pct:.1f}% objective penalty for monoculture.")
print(f"Target is 17-20% penalty.")
print(f"\n✓ Current parameters are reasonable (within ~4% of target)")
print(f"✓ For exact 18% penalty: set rotation_gamma = 0.15")
print(f"✓ User's exponential concern doesn't apply to our linear formulation")
