#!/usr/bin/env python3
"""
Analysis: How to reformulate the Food Optimization Problem for Potential Quantum Advantage

Current Problem Characteristics:
- EASY for classical: 0% integrality gap, totally unimodular-like structure
- HARD for quantum: Dense BQM graph after CQM→BQM conversion (high degree)

Goal: Find reformulations that are:
- HARD for classical: Large integrality gap, many fractional variables, exponential branching
- EASIER for quantum: Sparse interaction graph, low degree, native QUBO structure

Key Insight: Quantum annealers excel at problems that are:
1. Naturally quadratic (no constraint penalties needed)
2. Sparse connectivity (embeddable on hardware)
3. Have frustrated interactions (no clear greedy solution)
4. Have rough energy landscapes (many local minima)

Classical solvers struggle with:
1. Large integrality gaps (weak LP relaxation)
2. Symmetry (many equivalent solutions)
3. Problems where greedy fails badly
4. Combinatorial explosion without good bounds
"""

import numpy as np
from collections import defaultdict

print("="*100)
print("QUANTUM ADVANTAGE REFORMULATION ANALYSIS")
print("="*100)
print()

# ============================================================================
# IDEA 1: Quadratic Synergy/Conflict Model (Native QUBO)
# ============================================================================

print("IDEA 1: QUADRATIC SYNERGY/CONFLICT MODEL")
print("-"*80)
print("""
Instead of linear constraints, encode crop interactions as QUADRATIC terms:

Original (Linear):
  max Σ benefit[f,c] * Y[f,c]
  s.t. Σ Y[f,c] ≤ 1  (one crop per farm)

Reformulation (Quadratic - Native QUBO):
  max Σ benefit[f,c] * Y[f,c] 
      + Σ synergy[c1,c2] * Y[f1,c1] * Y[f2,c2]  (neighboring farms)
      - Σ conflict[c1,c2] * Y[f1,c1] * Y[f2,c2]  (same farm)

Why this helps quantum:
- NO constraint penalties needed (constraints become quadratic terms)
- Synergies create FRUSTRATED interactions → hard for greedy
- Can be sparse if synergies are local (neighboring farms only)

Why this hurts classical:
- Quadratic objective → non-convex
- LP relaxation becomes weak (quadratic terms relax to zero)
- Branch-and-bound tree explodes

Implementation: Use spatial adjacency for synergies
- Farm f1 and f2 are neighbors → synergy if compatible crops
- Same-farm conflict: Y[f,c1] * Y[f,c2] = 0 for c1 ≠ c2
""")

# ============================================================================
# IDEA 2: Multi-Period Planning with Rotation Constraints
# ============================================================================

print("\nIDEA 2: MULTI-PERIOD CROP ROTATION MODEL")
print("-"*80)
print("""
Add temporal dimension with history-dependent constraints:

Variables: Y[f,c,t] = 1 if farm f grows crop c in period t

Constraints:
- Can't grow same crop twice in a row: Y[f,c,t] + Y[f,c,t+1] ≤ 1
- Must rotate between crop families
- Some crops BENEFIT from predecessors (legumes → nitrogen fixing)

Quadratic form:
  max Σ benefit[c] * Y[f,c,t]
      + Σ rotation_bonus[c1,c2] * Y[f,c1,t] * Y[f,c2,t+1]

Why this helps quantum:
- Natural quadratic structure (rotation bonuses)
- Temporal constraints create sparse interactions (only adjacent periods)
- History dependence creates complex energy landscape

Why this hurts classical:
- Exponentially many rotation sequences
- Weak LP relaxation for temporal dependencies
- Dynamic programming doesn't apply well with global diversity constraints
""")

# ============================================================================
# IDEA 3: Maximum Weight Independent Set (MWIS) Reformulation
# ============================================================================

print("\nIDEA 3: MAXIMUM WEIGHT INDEPENDENT SET REFORMULATION")
print("-"*80)
print("""
MWIS is a classic NP-hard problem where quantum shows promise!

Construct a conflict graph G:
- Nodes: All (farm, crop) pairs with weight = benefit
- Edges: Connect incompatible assignments
  - Same farm, different crops → edge (can't both be 1)
  - Crops that compete for resources → edge
  - Violates diversity constraints → edge

Problem: Find maximum weight independent set in G

QUBO formulation (native!):
  max Σ w[i] * x[i] - penalty * Σ x[i] * x[j]  for edges (i,j)

Why this helps quantum:
- MWIS is naturally QUBO (no slack variables!)
- Graph structure can be designed to be sparse
- Well-studied for quantum advantage

Why this hurts classical:
- MWIS is NP-hard even to approximate
- LP relaxation is notoriously weak
- Greedy gives arbitrarily bad solutions

Key insight: Design the conflict graph to be:
- Sparse (low degree) → easy to embed
- Unit disk or planar → extra structure
- Non-bipartite → frustrated, hard instances
""")

# ============================================================================
# IDEA 4: Quadratic Unconstrained Binary Optimization (Direct QUBO)
# ============================================================================

print("\nIDEA 4: DIRECT QUBO WITH ISING-LIKE INTERACTIONS")
print("-"*80)
print("""
Model crop allocation as an Ising spin system:

Spin variables: s[f,c] ∈ {-1, +1}
  s[f,c] = +1 → farm f grows crop c
  s[f,c] = -1 → farm f doesn't grow crop c

Energy (minimize):
  E = - Σ h[f,c] * s[f,c]                    (linear: crop benefits)
      - Σ J[f1,c1,f2,c2] * s[f1,c1] * s[f2,c2]  (quadratic: interactions)

Design J (coupling matrix) to encode:
- NEGATIVE J (ferromagnetic): crops that benefit from being together
- POSITIVE J (antiferromagnetic): crops that compete/conflict
- ZERO J: no interaction (keeps graph sparse!)

Why this helps quantum:
- Native Ising model → NO penalty conversion
- J matrix can be designed to match hardware topology
- Frustrated interactions → quantum tunneling advantage

Why this hurts classical:
- Ising model with mixed J is NP-hard
- Spin glass instances have exponentially many local minima
- Simulated annealing gets stuck in local minima
""")

# ============================================================================
# IDEA 5: Sparse Parity Constraints (Quantum Error Correction Style)
# ============================================================================

print("\nIDEA 5: PARITY-BASED DIVERSITY CONSTRAINTS")
print("-"*80)
print("""
Instead of counting constraints, use XOR/parity constraints:

Original: "At least 2 vegetables" → Σ U[veg_i] ≥ 2

Parity version: "Odd number of certain vegetables"
  U[veg1] ⊕ U[veg2] ⊕ U[veg3] = 1

QUBO encoding of XOR:
  (x ⊕ y ⊕ z = 1) → 4xy + 4yz + 4xz - 2x - 2y - 2z + 1

Why this helps quantum:
- Parity constraints create HIGHLY FRUSTRATED systems
- No good classical rounding (0.5 + 0.5 + 0.5 ≠ valid)
- Related to quantum error correction → natural for quantum

Why this hurts classical:
- LP relaxation gives 0.5 everywhere (uninformative)
- Random rounding fails (50% wrong)
- Exponential branching
""")

# ============================================================================
# IDEA 6: Portfolio-Style Covariance Model
# ============================================================================

print("\nIDEA 6: RISK-AWARE PORTFOLIO OPTIMIZATION")
print("-"*80)
print("""
Treat crop selection like portfolio optimization with COVARIANCE:

Variables: x[c] = fraction of land for crop c (continuous → discretize)

Objective:
  max  Σ expected_return[c] * x[c]           (expected yield)
     - λ * Σ Cov[c1,c2] * x[c1] * x[c2]      (risk from correlation)

Discretize: x[c] → Σ 2^(-k) * b[c,k]  (binary expansion)

Why this helps quantum:
- Portfolio optimization with covariance is QUADRATIC
- Well-studied quantum algorithms exist
- Natural diversification pressure

Why this hurts classical:
- Non-convex when discretized
- Covariance creates dense but structured interactions
- Multiple Pareto-optimal solutions

Real-world meaning:
- Crops with correlated failures (same disease, weather sensitivity)
- Diversification reduces catastrophic risk
- Quadratic risk term is scientifically meaningful!
""")

# ============================================================================
# RECOMMENDATION
# ============================================================================

print("\n" + "="*100)
print("RECOMMENDATION: COMBINE IDEAS 1 + 3 + 6")
print("="*100)
print("""
Best reformulation strategy:

1. USE SPATIAL SYNERGIES (Idea 1)
   - Define farm adjacency graph (sparse!)
   - Add quadratic synergy terms for neighboring farms
   - Crop rotation benefits between adjacent plots

2. FRAME AS MWIS ON CONFLICT GRAPH (Idea 3)
   - Build sparse conflict graph
   - Use spatial locality to limit edges
   - Natural QUBO formulation

3. ADD RISK COVARIANCE (Idea 6)
   - Crop yield correlations (weather, pests)
   - Quadratic risk penalty for similar crops
   - Meaningful diversification pressure

4. CRITICAL: DESIGN FOR SPARSE BQM
   - Limit interactions to k-nearest neighbors
   - Use hierarchical structure (villages → farms → plots)
   - Target max degree < 15 for Pegasus embedding

Expected outcome:
- Classical: HARD (weak LP, quadratic non-convex)
- Quantum: FEASIBLE (sparse QUBO, embeddable, frustrated)
""")

print("\nNext step: Implement and benchmark a prototype!")
