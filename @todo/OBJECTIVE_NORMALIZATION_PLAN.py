"""
Analysis of Objective Function Normalization Across Decomposition Methods

Current Issue:
--------------
All methods use: obj = sum(A[f,c] * benefit[c] / 100.0)

But this means:
- If one method uses 80% of land and another uses 100%, they're not comparable
- Need to normalize by actual land used OR by total available land

Standard Form (Benefit per Hectare):
-------------------------------------
obj = sum(A[f,c] * benefit[c]) / total_area

Where:
- A[f,c] = allocation in hectares for farm f, crop c
- benefit[c] = benefit value for crop c (from nutrients × weights)
- total_area = sum of all farm capacities

This gives us "benefit per hectare" which is comparable across different land utilizations.

Implementation Plan:
-------------------
1. Add total_area parameter to all solve functions
2. Update objective calculations to divide by total_area
3. Ensure column objectives in Dantzig-Wolfe also normalized
4. Update final objective reporting

Files to Update:
---------------
- decomposition_benders.py
- decomposition_benders_qpu.py
- decomposition_dantzig_wolfe.py
- decomposition_dantzig_wolfe_qpu.py
- decomposition_admm.py
- decomposition_admm_qpu.py
"""

# Current formula in all methods:
# obj_expr = sum(A[(f,c)] * benefits.get(c, 1.0) / 100.0)

# Should be:
# obj_expr = sum(A[(f,c)] * benefits.get(c, 1.0)) / total_area

# For columns in Dantzig-Wolfe:
# col_obj = sum(allocation[(f,c)] * benefits.get(c, 1.0)) / total_area

print("Objective function standardization plan ready for implementation")
