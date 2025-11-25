#!/usr/bin/env python3
"""
Objective Function Implementations Across Decomposition Strategies

This module documents all objective function formulations used in the
various decomposition strategies for the agricultural land allocation problem.

Each strategy optimizes the same fundamental objective but may reformulate it
differently based on the decomposition approach.

PROBLEM FORMULATION
===================

Decision Variables:
- A_{f,c}: Continuous area (hectares) allocated to food c on farm f
- Y_{f,c}: Binary indicator (1 if food c is planted on farm f, 0 otherwise)

Parameters:
- b_c: Benefit score for food c (nutritional + environmental + economic value)
- T: Total available land area (sum of all farm capacities)
- L_f: Land capacity of farm f
- min_area_c: Minimum planting area for food c (if selected)
- max_area_c: Maximum planting area for food c

Base Objective (All Strategies):
    maximize Z = (1/T) * sum_{f,c} b_c * A_{f,c}

This represents the average benefit per hectare across all farms.
"""

import numpy as np
from typing import Dict, List, Tuple


# =============================================================================
# BENEFIT CALCULATION (Common to All Strategies)
# =============================================================================

def calculate_benefits(foods: List[str], foods_dict: Dict, weights: Dict) -> Dict[str, float]:
    """
    Calculate composite benefit score for each food.
    
    Formula:
        b_c = w_1 * nutritional_value_c + w_2 * nutrient_density_c
            + w_3 * environmental_impact_c + w_4 * affordability_c
            + w_5 * sustainability_c
    
    where w_i are the weight parameters and sum to 1.0
    
    LaTeX:
        b_c = \sum_{i=1}^{5} w_i \cdot s_{c,i}
    
    Args:
        foods: List of food names
        foods_dict: Dictionary with food properties
        weights: Dictionary of weight factors
    
    Returns:
        Dictionary mapping food name to benefit score
    """
    benefits = {}
    
    default_weights = {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15
    }
    w = {**default_weights, **weights}
    
    for food in foods:
        food_data = foods_dict.get(food, {})
        
        # Extract scores (default to 0.5 if missing)
        nv = food_data.get('nutritional_value', 0.5)
        nd = food_data.get('nutrient_density', 0.5)
        ei = food_data.get('environmental_impact', 0.5)
        af = food_data.get('affordability', 0.5)
        su = food_data.get('sustainability', 0.5)
        
        # Weighted sum
        b = (w['nutritional_value'] * nv +
             w['nutrient_density'] * nd +
             w['environmental_impact'] * ei +
             w['affordability'] * af +
             w['sustainability'] * su)
        
        benefits[food] = b
    
    return benefits


# =============================================================================
# BENDERS DECOMPOSITION OBJECTIVE
# =============================================================================

def benders_master_objective(Y: Dict[Tuple[str, str], float], eta: float) -> float:
    """
    Benders Master Problem Objective.
    
    The master problem optimizes:
        maximize eta
        
    where eta is a proxy variable bounded by Benders cuts.
    
    LaTeX:
        \max \eta
        
        \text{subject to:}
        \eta \leq z^{(k)} \quad \forall k \in \{1, ..., K\}
        
    where z^{(k)} is the subproblem objective at iteration k.
    
    Args:
        Y: Binary selection variables (fixed in master)
        eta: Objective proxy variable
    
    Returns:
        eta value (master objective)
    """
    return eta


def benders_subproblem_objective(
    A: Dict[Tuple[str, str], float],
    benefits: Dict[str, float],
    total_area: float
) -> float:
    """
    Benders Subproblem Objective.
    
    Given fixed Y*, maximize:
        Z_sub = (1/T) * sum_{f,c} b_c * A_{f,c}
        
    LaTeX:
        Z_{sub}(Y^*) = \frac{1}{T} \sum_{f \in \mathcal{F}} \sum_{c \in \mathcal{C}} b_c \cdot A_{f,c}
        
        \text{subject to:}
        A_{f,c} = 0 \quad \text{if } Y^*_{f,c} = 0
        A_{f,c} \geq a^{min}_c \quad \text{if } Y^*_{f,c} = 1
    
    Args:
        A: Area allocation variables
        benefits: Benefit scores per food
        total_area: Total available land
    
    Returns:
        Normalized objective value
    """
    obj = sum(A[key] * benefits.get(key[1], 1.0) for key in A)
    return obj / total_area if total_area > 0 else 0.0


# =============================================================================
# DANTZIG-WOLFE DECOMPOSITION OBJECTIVE
# =============================================================================

def dantzig_wolfe_rmp_objective(
    columns: List[Dict],
    lambda_weights: Dict[int, float]
) -> float:
    """
    Dantzig-Wolfe Restricted Master Problem (RMP) Objective.
    
    The RMP selects from a pool of columns (allocation patterns):
        maximize sum_k lambda_k * c_k
        
    where c_k is the objective contribution of column k.
    
    LaTeX:
        \max \sum_{k \in \mathcal{K}} \lambda_k \cdot c_k
        
        \text{where } c_k = \frac{1}{T} \sum_{(f,c) \in col_k} b_c \cdot A^{(k)}_{f,c}
        
        \text{subject to:}
        \sum_{k} \lambda_k \leq |\mathcal{F}| \quad \text{(convexity)}
    
    Args:
        columns: List of column dictionaries with 'objective' key
        lambda_weights: Weight for each column
    
    Returns:
        RMP objective value
    """
    obj = sum(lambda_weights.get(k, 0) * col['objective'] 
              for k, col in enumerate(columns))
    return obj


def dantzig_wolfe_pricing_objective(
    A: Dict[Tuple[str, str], float],
    benefits: Dict[str, float],
    duals: Dict[str, float],
    total_area: float
) -> float:
    """
    Dantzig-Wolfe Pricing Subproblem Objective.
    
    Generate new columns by solving:
        maximize (benefit - dual costs)
        
    The reduced cost determines if a new column should be added.
    
    LaTeX:
        \bar{c} = \frac{1}{T} \sum_{f,c} b_c \cdot A_{f,c} - \sum_f \pi_f
        
    where π_f are dual prices from the land constraints.
    
    If reduced cost < 0, add the column to improve the RMP.
    
    Args:
        A: Candidate allocation pattern
        benefits: Benefit scores
        duals: Dual prices from RMP
        total_area: Total land area
    
    Returns:
        Reduced cost (negative means improving column)
    """
    benefit_term = sum(A[key] * benefits.get(key[1], 1.0) for key in A) / total_area
    dual_term = sum(duals.values())
    return benefit_term - dual_term


# =============================================================================
# ADMM DECOMPOSITION OBJECTIVE
# =============================================================================

def admm_a_subproblem_objective(
    A: Dict[Tuple[str, str], float],
    Y: Dict[Tuple[str, str], float],
    U: Dict[Tuple[str, str], float],
    benefits: Dict[str, float],
    rho: float,
    total_area: float
) -> float:
    """
    ADMM A-Subproblem (Continuous Variables) Objective.
    
    Optimize continuous allocations with augmented Lagrangian:
        maximize (1/T) * sum b_c * A_{f,c} - penalty_term
        
    where penalty enforces A-Y consensus.
    
    LaTeX:
        \mathcal{L}_A = \frac{1}{T} \sum_{f,c} b_c \cdot A_{f,c} 
                        - \sum_{f,c} U_{f,c} (A_{f,c} - Y_{f,c})
                        - \frac{\rho}{2} \sum_{f,c} (A_{f,c} - Y_{f,c})^2
                        
        \text{Simplified:}
        \max \frac{1}{T} \sum_{f,c} b_c \cdot A_{f,c} 
             - \frac{\rho}{2} \|A - Y + U\|_2^2
    
    Args:
        A: Area allocations
        Y: Binary selections (from previous Y-update)
        U: Scaled dual variables
        benefits: Benefit scores
        rho: ADMM penalty parameter
        total_area: Total land
    
    Returns:
        Augmented Lagrangian value for A-subproblem
    """
    benefit = sum(A[key] * benefits.get(key[1], 1.0) for key in A) / total_area
    penalty = (rho / 2) * sum((A[key] - Y[key] + U[key])**2 for key in A)
    return benefit - penalty


def admm_y_subproblem_objective(
    Y: Dict[Tuple[str, str], float],
    A: Dict[Tuple[str, str], float],
    U: Dict[Tuple[str, str], float],
    rho: float
) -> float:
    """
    ADMM Y-Subproblem (Binary Variables) Objective.
    
    Optimize binary selections to minimize consensus violation:
        minimize (rho/2) * ||A - Y + U||^2
        
    LaTeX:
        \min \frac{\rho}{2} \sum_{f,c} (A_{f,c} - Y_{f,c} + U_{f,c})^2
        
        \text{Equivalent to:}
        \min \frac{\rho}{2} \|A - Y + U\|_2^2
        
        \text{subject to:}
        Y_{f,c} \in \{0, 1\}
        \text{food group constraints}
    
    Args:
        Y: Binary selections
        A: Area allocations (from previous A-update)
        U: Scaled dual variables
        rho: ADMM penalty parameter
    
    Returns:
        Consensus penalty (to be minimized)
    """
    penalty = (rho / 2) * sum((A[key] - Y[key] + U[key])**2 for key in Y)
    return penalty


def admm_dual_update(
    U: Dict[Tuple[str, str], float],
    A: Dict[Tuple[str, str], float],
    Y: Dict[Tuple[str, str], float],
    rho: float
) -> Dict[Tuple[str, str], float]:
    """
    ADMM Dual Variable Update.
    
    Update dual variables to enforce A-Y consensus:
        U^{k+1} = U^k + rho * (A^{k+1} - Y^{k+1})
        
    LaTeX:
        U^{(k+1)}_{f,c} = U^{(k)}_{f,c} + \rho (A^{(k+1)}_{f,c} - Y^{(k+1)}_{f,c})
    
    Args:
        U: Current dual variables
        A: Updated A values
        Y: Updated Y values
        rho: ADMM penalty parameter
    
    Returns:
        Updated dual variables
    """
    U_new = {}
    for key in U:
        U_new[key] = U[key] + rho * (A[key] - Y[key])
    return U_new


def admm_residuals(
    A: Dict[Tuple[str, str], float],
    Y: Dict[Tuple[str, str], float],
    Y_prev: Dict[Tuple[str, str], float],
    rho: float
) -> Tuple[float, float]:
    """
    ADMM Primal and Dual Residuals.
    
    Convergence is checked using:
        - Primal residual: ||A - Y||_2
        - Dual residual: ||rho * (Y^k - Y^{k-1})||_2
        
    LaTeX:
        r^{(k)} = \|A^{(k)} - Y^{(k)}\|_2 \quad \text{(primal)}
        s^{(k)} = \rho \|Y^{(k)} - Y^{(k-1)}\|_2 \quad \text{(dual)}
        
        \text{Convergence when:}
        r^{(k)} < \epsilon_{pri} \text{ and } s^{(k)} < \epsilon_{dual}
    
    Args:
        A: Current A values
        Y: Current Y values
        Y_prev: Previous Y values
        rho: ADMM penalty
    
    Returns:
        (primal_residual, dual_residual)
    """
    primal = np.sqrt(sum((A[key] - Y[key])**2 for key in A))
    dual = np.sqrt(sum((rho * (Y[key] - Y_prev.get(key, 0)))**2 for key in Y))
    return primal, dual


# =============================================================================
# CURRENT HYBRID (GUROBI + QPU) OBJECTIVE
# =============================================================================

def current_hybrid_objective(
    A: Dict[Tuple[str, str], float],
    benefits: Dict[str, float],
    total_area: float
) -> float:
    """
    Current Hybrid Strategy Objective.
    
    Two-phase approach:
    1. Gurobi solves LP relaxation to get continuous A values
    2. QPU/Sampler solves for binary Y given A hints
    
    Final objective:
        Z = (1/T) * sum_{f,c} b_c * A_{f,c}
        
    LaTeX:
        Z = \frac{1}{T} \sum_{f \in \mathcal{F}} \sum_{c \in \mathcal{C}} b_c \cdot A_{f,c}
        
        \text{Phase 1 (LP Relaxation):}
        \max Z \text{ s.t. } 0 \leq A_{f,c} \leq L_f Y_{f,c}, \; Y_{f,c} \in [0,1]
        
        \text{Phase 2 (Binary):}
        \text{Round/sample } Y^* \in \{0,1\}^{|\mathcal{F}| \times |\mathcal{C}|}
    
    Args:
        A: Final area allocations
        benefits: Benefit scores
        total_area: Total land
    
    Returns:
        Normalized objective value
    """
    obj = sum(A[key] * benefits.get(key[1], 1.0) for key in A)
    return obj / total_area if total_area > 0 else 0.0


# =============================================================================
# UNIFIED OBJECTIVE EVALUATION
# =============================================================================

def evaluate_objective(
    solution: Dict[str, float],
    farms: Dict[str, float],
    foods: List[str],
    benefits: Dict[str, float]
) -> float:
    """
    Evaluate objective for any solution format.
    
    This is the canonical objective evaluation used for comparing
    solutions across all decomposition strategies.
    
    LaTeX:
        Z = \frac{1}{T} \sum_{f \in \mathcal{F}} \sum_{c \in \mathcal{C}} b_c \cdot A_{f,c}
        
        \text{where } T = \sum_{f \in \mathcal{F}} L_f
    
    Args:
        solution: Dictionary with A_{farm}_{food} keys
        farms: Dictionary of farm capacities
        foods: List of food names
        benefits: Benefit scores
    
    Returns:
        Normalized objective value in [0, 1] range
    """
    total_area = sum(farms.values())
    if total_area == 0:
        return 0.0
    
    obj = 0.0
    for farm in farms:
        for food in foods:
            key = f"A_{farm}_{food}"
            area = solution.get(key, 0.0)
            benefit = benefits.get(food, 1.0)
            obj += area * benefit
    
    return obj / total_area


# =============================================================================
# CONSTRAINT SUMMARY
# =============================================================================

"""
COMMON CONSTRAINTS (All Strategies)
====================================

1. Land Availability (per farm):
   LaTeX: \sum_{c \in \mathcal{C}} A_{f,c} \leq L_f \quad \forall f \in \mathcal{F}
   
2. Linking Constraint - Min Area:
   LaTeX: A_{f,c} \geq a^{min}_c \cdot Y_{f,c} \quad \forall f, c
   
3. Linking Constraint - Max Area:
   LaTeX: A_{f,c} \leq L_f \cdot Y_{f,c} \quad \forall f, c
   
4. Food Group Constraints (global count):
   LaTeX: n^{min}_g \leq \sum_{f,c : c \in G_g} Y_{f,c} \leq n^{max}_g \quad \forall g

5. Variable Domains:
   LaTeX: A_{f,c} \geq 0, \quad Y_{f,c} \in \{0, 1\}
"""


if __name__ == "__main__":
    print("=" * 80)
    print("OBJECTIVE FUNCTION IMPLEMENTATIONS")
    print("=" * 80)
    print(__doc__)
    
    print("\n" + "=" * 80)
    print("STRATEGY SUMMARY")
    print("=" * 80)
    
    strategies = [
        ("Benders", "Master: max eta | Subproblem: max sum(b*A)/T"),
        ("Dantzig-Wolfe", "RMP: max sum(λ*c) | Pricing: max (b*A/T - duals)"),
        ("ADMM", "A-sub: max b*A/T - penalty | Y-sub: min consensus penalty"),
        ("Current Hybrid", "LP relaxation + QPU binary rounding")
    ]
    
    for name, formula in strategies:
        print(f"\n{name}:")
        print(f"  {formula}")
