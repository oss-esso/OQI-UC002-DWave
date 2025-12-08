# Quantum Advantage Reformulation Analysis

## Executive Summary

The original crop allocation problem is **trivial for classical solvers** (0% integrality gap) but **intractable for quantum** (BQM degree grows linearly to ~2000). This is the **opposite** of what we need for quantum advantage.

This document analyzes reformulation strategies to flip this situation.

## Current Problem Analysis

| Metric | Value | Implication |
|--------|-------|-------------|
| Integrality Gap | **0%** | LP relaxation is tight → solved at root |
| Fractional Variables | **0** | No branching needed |
| BQM Max Degree | 42 → 2022 | Grows linearly with plots |
| BQM Quadratic Terms | 7K → 14M | Explodes quadratically |

**Verdict**: EASY for classical, IMPOSSIBLE for quantum (at scale)

## Why Current Formulation is Easy for Classical

1. **Totally Unimodular-like Structure**: The constraint matrix has special structure (assignment + covering constraints) that makes LP relaxation integral.

2. **No Frustrated Interactions**: Each variable independently contributes to objective; no competing forces.

3. **Linear Objective**: No quadratic terms in original problem → clean LP.

## Why Current Formulation is Hard for Quantum

1. **CQM→BQM Conversion**: Constraint penalties create dense quadratic interactions.

2. **U-Y Linking Constraints**: Each `U[food]` connects to ALL `Y[farm, food]` → degree = n_farms.

3. **Food Group Constraints**: Each `U[food]` connects to all foods in group.

## Reformulation Strategies

### Strategy 1: Spatial Synergy Model (Native QUBO)

**Idea**: Replace linear constraints with quadratic synergy/conflict terms.

```
Original: max Σ benefit[f,c] * Y[f,c]   s.t. constraints

Reformulation: max Σ benefit[f,c] * Y[f,c] 
                 + Σ synergy[c1,c2] * Y[f1,c1] * Y[f2,c2]  (neighbors)
                 - Σ conflict[c1,c2] * Y[f1,c1] * Y[f2,c2] (same farm)
```

**Sparsity Control**: Only adjacent farms interact → degree bounded by k_neighbors × n_crops.

**Problem**: Still high degree (~200) due to n_crops² same-farm penalties.

### Strategy 2: Farm-Type Abstraction

**Idea**: Replace n_crops (27) choices with n_types (4-8) farm strategies.

```
Variables: x[f,t] = 1 if farm f uses strategy t

Degree = (n_types - 1) + k_neighbors × n_types
       = 5 + 4 × 6 = 29  ← EMBEDDABLE!
```

**Results** (n_types=6, k=4):
| Farms | Vars | Quadratic | Max Degree | Embeddable? |
|-------|------|-----------|------------|-------------|
| 100   | 600  | 8,070     | 40         | YES (chain~3) |
| 500   | 3000 | 38,820    | 40         | YES (chain~3) |
| 1000  | 6000 | 76,890    | 40         | YES (chain~3) |

**Problem**: Still 0% integrality gap (one-hot encoding is LP-friendly).

### Strategy 3: Spin Glass with Frustrated Interactions

**Idea**: Use Ising spin glass structure with planted frustration.

```
Variables: s[i] ∈ {-1, +1} (Ising spins)

Energy: E = Σ h[i] * s[i] + Σ J[i,j] * s[i] * s[j]

Frustration: 50% of couplings J oppose the planted solution
```

**Results**:
| Spins | Edges | Max Degree | Gap | Hardness |
|-------|-------|------------|-----|----------|
| 50    | ~180  | 15         | 25% | HARD ✓ |
| 100   | ~450  | 18         | 30% | HARD ✓ |
| 200   | ~1400 | 28         | 35% | VERY HARD ✓ |
| 500   | ~6000 | 35         | 40% | VERY HARD ✓ |

**This achieves our goal**: HARD classical (40% gap), FEASIBLE quantum (degree ~35).

## Recommended Reformulation

### Crop Allocation as Frustrated Spin Glass

**Binary Decision**: For each farm, choose between two crop strategies (A vs B).
- `s[f] = +1`: Farm f uses strategy A
- `s[f] = -1`: Farm f uses strategy B

**Interactions (Ising couplings)**:
1. **Ferromagnetic (J < 0)**: Neighboring farms benefit from same strategy (companion planting).
2. **Antiferromagnetic (J > 0)**: Neighboring farms benefit from different strategies (biodiversity, pest resistance).
3. **Mixed**: Some pairs have competing preferences → FRUSTRATION.

**Spatial Sparsity**: Only k-nearest neighbors interact → bounded degree.

**Mathematical Formulation**:
```
minimize: E = Σ h[f] * s[f] + Σ J[f1,f2] * s[f1] * s[f2]

where:
  h[f] = benefit(A) - benefit(B) for farm f
  J[f1,f2] = synergy(same_strategy) - synergy(different_strategy)
  
  Only (f1, f2) pairs that are spatial neighbors have J ≠ 0
```

**QUBO Conversion**: Standard Ising-to-QUBO transformation (s = 2x - 1).

## Expected Outcomes

| Metric | Original | Reformulated |
|--------|----------|--------------|
| Classical Gap | 0% | 20-40% |
| Fractional Vars | 0% | 30-50% |
| BQM Max Degree | ~2000 | ~35 |
| Embeddable (1000 farms) | NO | YES |
| Classical Time | <1s | Minutes-Hours |
| Quantum Feasible | NO | YES |

## Trade-offs

**Lost**:
- Granularity: 2 strategies vs 27 crops per farm
- Exact constraints: Food group diversity becomes soft penalty

**Gained**:
- Quantum tractability: Bounded degree, embeddable
- Classical hardness: Large integrality gap, frustrated interactions
- Potential for quantum advantage

## Next Steps

1. **Implement**: Build spin glass formulation with real crop data
2. **Validate**: Verify classical hardness (Gurobi runtime, branching)
3. **Benchmark**: Compare QPU vs SA vs exact on same instances
4. **Iterate**: Tune frustration ratio, edge density for hardest instances

## Files Generated

- `@todo/quantum_advantage_analysis.py` - Reformulation ideas
- `@todo/sparse_qubo_prototype.py` - First sparse QUBO attempt  
- `@todo/ultra_sparse_qubo.py` - Farm-type abstraction
- `@todo/hardness_output/comprehensive_hardness_report.json` - Full analysis
- `@todo/hardness_output/sparse_qubo_comparison.json` - Structure comparison
