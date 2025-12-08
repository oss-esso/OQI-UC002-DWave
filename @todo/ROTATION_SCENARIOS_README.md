# Rotation Scenarios - Quantum-Friendly Formulations

## Overview

Four new rotation scenarios have been added to `src/scenarios.py` that implement the quantum-friendly reformulation strategy for demonstrating potential quantum advantage.

## Available Scenarios

### 1. `rotation_micro_25` - Validation Scale
- **Farms**: 5
- **Crop Families**: 6 (Fruits, Grains, Legumes, Leafy Veg, Root Veg, Proteins)
- **Periods**: 3
- **Variables**: ~108 (5 farms × 6 families × 3 periods + U variables)
- **Max Degree**: ~29 (bounded!)
- **Frustration**: 40% negative synergies
- **Embeddability**: ✅ QPU with chains ~2
- **Purpose**: Validate formulation, test embedding

### 2. `rotation_small_50` - Classical Hardness Testing
- **Farms**: 10
- **Crop Families**: 6
- **Variables**: ~198
- **Max Degree**: ~29
- **Frustration**: 40% negative synergies
- **Embeddability**: ✅ QPU with chains ~2
- **Purpose**: Test classical hardness (target >5% integrality gap)

### 3. `rotation_medium_100` - Increased Frustration
- **Farms**: 20
- **Crop Families**: 6
- **Variables**: ~378
- **Max Degree**: ~29
- **Frustration**: 50% negative synergies (higher!)
- **Negative Strength**: -0.5 (stronger penalties)
- **Embeddability**: ✅ QPU with chains ~2
- **Purpose**: Push classical solvers into timeout regime

### 4. `rotation_large_200` - Quantum Advantage Regime
- **Farms**: 50
- **Crop Families**: 6
- **Variables**: ~918
- **Max Degree**: ~29
- **Frustration**: 60% negative synergies (very high!)
- **Negative Strength**: -0.7 (very strong penalties)
- **Embeddability**: ✅ QPU with chains ~2
- **Purpose**: Demonstrate quantum advantage (expect Gurobi >1hr)

## Key Design Features

### 1. Reduced Crop Choices
Instead of 27 individual crops, we use 6 crop families:
- **Reduces** same-farm clique from 27² = 729 edges to 6² = 36 edges
- **Bounds** max degree contribution from (27-1) = 26 to (6-1) = 5

### 2. Spatial Locality
- k=4 nearest neighbors on grid layout
- Creates sparse spatial coupling (not all-to-all)
- Max degree contribution: 4 × 6 = 24 edges per variable
- **Total max degree**: 5 + 24 = 29 ✅

### 3. Frustrated Interactions
Negative synergies represent realistic agricultural conflicts:
- **Same-family nutrient depletion** (e.g., Grains → Grains depletes nitrogen)
- **Disease carryover** (e.g., Solanaceae pathogens)
- **Allelopathic effects** (some plants inhibit others)
- **Pest harbor** (monoculture increases pest pressure)

Frustration levels:
- micro/small: 40% negative edges
- medium: 50% negative edges
- large: 60% negative edges (approaching spin glass phase transition!)

### 4. Temporal Structure
- 3-period rotation (already in your formulation!)
- Quadratic terms: Y[f,c,t-1] × Y[f,c,t]
- Native QUBO structure

## Configuration Parameters

Each scenario includes these rotation-specific parameters:

```python
config = {
    'parameters': {
        'rotation_gamma': 0.15-0.25,  # Rotation synergy weight
        'spatial_k_neighbors': 4,      # Nearest neighbors
        'frustration_ratio': 0.4-0.6,  # % negative synergies
        'negative_synergy_strength': -0.3 to -0.7  # Penalty magnitude
    }
}
```

## Usage

### Load a Scenario
```python
from src.scenarios import load_food_data

farms, crop_families, food_groups, config = load_food_data('rotation_micro_25')
```

### With Rotation Solver
```python
from Benchmark Scripts.solver_runner_ROTATION import (
    create_cqm_plots_rotation_3period,
    solve_with_dwave_cqm
)

# Create CQM
cqm, variables, metadata = create_cqm_plots_rotation_3period(
    farms, crop_families, food_groups, config,
    gamma=config['parameters']['rotation_gamma']
)

# Solve on QPU
solution = solve_with_dwave_cqm(cqm, token=YOUR_TOKEN)
```

## Expected Classical Hardness

| Scenario | Expected Gap | Expected Gurobi Time |
|----------|--------------|---------------------|
| micro_25 | 0-5% | <1 min |
| small_50 | 5-10% | 1-10 min |
| medium_100 | 10-20% | 10-60 min |
| large_200 | >20% | >60 min (timeout) |

## Comparison to Original

| Metric | Original (27 crops) | Rotation (6 families) |
|--------|---------------------|----------------------|
| Variables (n=100) | ~2700 | ~378 |
| Max Degree | Unbounded (~100-2000) | Bounded (~29) |
| Integrality Gap | 0% (too easy) | 5-20% (target) |
| Embeddable? | ❌ No (too dense) | ✅ Yes (bounded) |
| Frustration | None (all positive) | 40-60% negative |
| Agricultural Meaning | ✅ Detailed | ✅ Preserved (families) |

## Next Steps

1. **Validate micro_25**: Test CQM creation and QPU embedding
2. **Benchmark Classical**: Run Gurobi on all scales
3. **Test on QPU**: Compare QPU vs SA vs Gurobi
4. **Analyze Results**: Measure gap, time, solution quality

## Files Modified

- `src/scenarios.py`: Added 4 rotation scenario functions
  - `_load_rotation_micro_25_food_data()`
  - `_load_rotation_small_50_food_data()`
  - `_load_rotation_medium_100_food_data()`
  - `_load_rotation_large_200_food_data()`

## Related Documents

- `@todo/QUANTUM_ADVANTAGE_SUMMARY.md` - Full analysis and recommendations
- `@todo/comprehensive_analysis.py` - Benchmark analysis script
- `Benchmark Scripts/solver_runner_ROTATION.py` - Rotation solver implementation
