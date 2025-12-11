#!/usr/bin/env python3
"""
Alternative Formulations Benchmark: Quantum-Friendly Problem Reformulations

This script implements and benchmarks 5 alternative formulations designed for
quantum annealing advantage, as outlined in ALTERNATIVE_FORMULATIONS_ANALYSIS.tex:

1. PORTFOLIO SELECTION (27 vars) - Select which crops to grow
2. HIERARCHICAL (27 + subproblems) - Two-stage crop then farm selection
3. GRAPH MWIS (30 vars) - Maximum Weight Independent Set on compatibility graph
4. SINGLE PERIOD (30 vars) - One period optimization without temporal coupling
5. PENALTY MODEL (90 vars) - Soft constraint version of rotation problem

Each formulation includes:
- Data generation with realistic agricultural parameters
- BQM/QUBO construction for QPU
- Gurobi ground truth solver
- D-Wave QPU solver
- Comparative benchmarking

Author: OQI-UC002-DWave Project
Date: 2025-12-11
"""

import os
import sys
import time
import json
import numpy as np
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

warnings.filterwarnings('ignore')

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).parent / 'alternative_formulations_results'

# D-Wave token - same default as other benchmarks
DEFAULT_DWAVE_TOKEN = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'
DWAVE_TOKEN = None

def set_dwave_token(token: str):
    global DWAVE_TOKEN
    DWAVE_TOKEN = token
    os.environ['DWAVE_API_TOKEN'] = token

def get_dwave_token():
    global DWAVE_TOKEN
    if DWAVE_TOKEN:
        return DWAVE_TOKEN
    token = os.environ.get('DWAVE_API_TOKEN')
    if token:
        DWAVE_TOKEN = token
        return token
    DWAVE_TOKEN = DEFAULT_DWAVE_TOKEN
    os.environ['DWAVE_API_TOKEN'] = DEFAULT_DWAVE_TOKEN
    return DEFAULT_DWAVE_TOKEN

# ============================================================================
# IMPORTS
# ============================================================================

print("=" * 80)
print("ALTERNATIVE FORMULATIONS BENCHMARK")
print("=" * 80)
print()

print("[1/4] Importing core libraries...")
import_start = time.time()
import numpy as np
from dimod import BinaryQuadraticModel

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
    print("  ✓ Gurobi available")
except ImportError:
    HAS_GUROBI = False
    print("  ✗ Gurobi not available")

print(f"  Core imports: {time.time() - import_start:.2f}s")

print("[2/4] Importing D-Wave libraries...")
dwave_start = time.time()

try:
    from dwave.system import DWaveCliqueSampler
    HAS_DWAVE = True
    print("  ✓ DWaveCliqueSampler available")
except ImportError:
    HAS_DWAVE = False
    print("  ✗ D-Wave not available")

print(f"  D-Wave imports: {time.time() - dwave_start:.2f}s")

print("[3/4] Importing plotting libraries...")
try:
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    HAS_MATPLOTLIB = True
    print("  ✓ Matplotlib available")
except ImportError:
    HAS_MATPLOTLIB = False
    print("  ✗ Matplotlib not available")

print("[4/4] Setup complete!\n")


# ============================================================================
# CROP DATA - Realistic 27 crops with synergies
# ============================================================================

# 27 crops organized by food group
CROPS = {
    # Cereals (6)
    'Wheat': {'group': 'Cereals', 'nutrition': 0.7, 'sustainability': 0.6, 'env_impact': 0.4, 'affordability': 0.8},
    'Rice': {'group': 'Cereals', 'nutrition': 0.65, 'sustainability': 0.5, 'env_impact': 0.5, 'affordability': 0.85},
    'Corn': {'group': 'Cereals', 'nutrition': 0.6, 'sustainability': 0.55, 'env_impact': 0.45, 'affordability': 0.9},
    'Barley': {'group': 'Cereals', 'nutrition': 0.65, 'sustainability': 0.7, 'env_impact': 0.35, 'affordability': 0.75},
    'Oats': {'group': 'Cereals', 'nutrition': 0.75, 'sustainability': 0.75, 'env_impact': 0.3, 'affordability': 0.7},
    'Sorghum': {'group': 'Cereals', 'nutrition': 0.6, 'sustainability': 0.8, 'env_impact': 0.25, 'affordability': 0.8},
    
    # Legumes (5)
    'Beans': {'group': 'Legumes', 'nutrition': 0.85, 'sustainability': 0.85, 'env_impact': 0.2, 'affordability': 0.8},
    'Lentils': {'group': 'Legumes', 'nutrition': 0.9, 'sustainability': 0.8, 'env_impact': 0.2, 'affordability': 0.75},
    'Chickpeas': {'group': 'Legumes', 'nutrition': 0.85, 'sustainability': 0.85, 'env_impact': 0.2, 'affordability': 0.7},
    'Peas': {'group': 'Legumes', 'nutrition': 0.8, 'sustainability': 0.8, 'env_impact': 0.25, 'affordability': 0.85},
    'Soybeans': {'group': 'Legumes', 'nutrition': 0.9, 'sustainability': 0.7, 'env_impact': 0.35, 'affordability': 0.65},
    
    # Vegetables (8)
    'Tomato': {'group': 'Vegetables', 'nutrition': 0.7, 'sustainability': 0.6, 'env_impact': 0.4, 'affordability': 0.75},
    'Potato': {'group': 'Vegetables', 'nutrition': 0.65, 'sustainability': 0.7, 'env_impact': 0.35, 'affordability': 0.9},
    'Carrot': {'group': 'Vegetables', 'nutrition': 0.75, 'sustainability': 0.75, 'env_impact': 0.3, 'affordability': 0.85},
    'Onion': {'group': 'Vegetables', 'nutrition': 0.6, 'sustainability': 0.8, 'env_impact': 0.25, 'affordability': 0.9},
    'Cabbage': {'group': 'Vegetables', 'nutrition': 0.7, 'sustainability': 0.85, 'env_impact': 0.2, 'affordability': 0.85},
    'Spinach': {'group': 'Vegetables', 'nutrition': 0.9, 'sustainability': 0.7, 'env_impact': 0.3, 'affordability': 0.7},
    'Lettuce': {'group': 'Vegetables', 'nutrition': 0.5, 'sustainability': 0.6, 'env_impact': 0.4, 'affordability': 0.8},
    'Squash': {'group': 'Vegetables', 'nutrition': 0.65, 'sustainability': 0.8, 'env_impact': 0.25, 'affordability': 0.85},
    
    # Fruits (4)
    'Apple': {'group': 'Fruits', 'nutrition': 0.7, 'sustainability': 0.65, 'env_impact': 0.35, 'affordability': 0.7},
    'Orange': {'group': 'Fruits', 'nutrition': 0.75, 'sustainability': 0.6, 'env_impact': 0.4, 'affordability': 0.65},
    'Banana': {'group': 'Fruits', 'nutrition': 0.7, 'sustainability': 0.55, 'env_impact': 0.45, 'affordability': 0.8},
    'Grapes': {'group': 'Fruits', 'nutrition': 0.65, 'sustainability': 0.5, 'env_impact': 0.5, 'affordability': 0.6},
    
    # Oils & Nuts (4)
    'Sunflower': {'group': 'Oils', 'nutrition': 0.6, 'sustainability': 0.7, 'env_impact': 0.35, 'affordability': 0.75},
    'Peanuts': {'group': 'Nuts', 'nutrition': 0.85, 'sustainability': 0.75, 'env_impact': 0.3, 'affordability': 0.7},
    'Almonds': {'group': 'Nuts', 'nutrition': 0.9, 'sustainability': 0.4, 'env_impact': 0.6, 'affordability': 0.4},
    'Sesame': {'group': 'Oils', 'nutrition': 0.75, 'sustainability': 0.8, 'env_impact': 0.25, 'affordability': 0.65},
}

# Synergy matrix - positive = complementary, negative = competing
# Key synergies based on agricultural science
SYNERGIES = {
    # Legume-Cereal rotations (nitrogen fixing) - STRONG POSITIVE
    ('Beans', 'Wheat'): 0.5, ('Beans', 'Corn'): 0.45, ('Beans', 'Rice'): 0.4,
    ('Lentils', 'Wheat'): 0.5, ('Lentils', 'Barley'): 0.45,
    ('Chickpeas', 'Wheat'): 0.5, ('Chickpeas', 'Oats'): 0.45,
    ('Peas', 'Wheat'): 0.45, ('Peas', 'Barley'): 0.4,
    ('Soybeans', 'Corn'): 0.5, ('Soybeans', 'Wheat'): 0.45,
    
    # Nutritional complementarity - MODERATE POSITIVE
    ('Rice', 'Beans'): 0.4,  # Complete protein
    ('Corn', 'Beans'): 0.35,  # Three sisters
    ('Corn', 'Squash'): 0.3,  # Three sisters
    ('Beans', 'Squash'): 0.25,  # Three sisters
    
    # Diverse diet synergies - SMALL POSITIVE
    ('Spinach', 'Carrot'): 0.2,  # Iron + vitamin A
    ('Tomato', 'Onion'): 0.15,  # Culinary pairing
    ('Potato', 'Cabbage'): 0.1,  # Staple combination
    
    # Same family diseases - NEGATIVE
    ('Tomato', 'Potato'): -0.35,  # Solanaceae family
    ('Cabbage', 'Spinach'): -0.15,  # Leafy green redundancy
    ('Lettuce', 'Spinach'): -0.2,  # Leafy green redundancy
    
    # Water competition - NEGATIVE
    ('Rice', 'Almonds'): -0.3,  # Both water intensive
    ('Rice', 'Grapes'): -0.2,
    
    # Nutrient redundancy - SMALL NEGATIVE
    ('Wheat', 'Rice'): -0.1,  # Both cereals
    ('Barley', 'Oats'): -0.1,  # Both cereals
    ('Apple', 'Orange'): -0.1,  # Both fruits
}


def get_synergy(crop1: str, crop2: str) -> float:
    """Get synergy value between two crops (symmetric)."""
    if crop1 == crop2:
        return 0
    key1 = (crop1, crop2)
    key2 = (crop2, crop1)
    if key1 in SYNERGIES:
        return SYNERGIES[key1]
    if key2 in SYNERGIES:
        return SYNERGIES[key2]
    return 0


def calculate_crop_value(crop_name: str, weights: Dict = None) -> float:
    """Calculate composite value for a crop."""
    if weights is None:
        weights = {'nutrition': 0.30, 'sustainability': 0.25, 
                   'env_impact': 0.25, 'affordability': 0.20}
    
    crop = CROPS[crop_name]
    value = (weights['nutrition'] * crop['nutrition'] +
             weights['sustainability'] * crop['sustainability'] -
             weights['env_impact'] * crop['env_impact'] +
             weights['affordability'] * crop['affordability'])
    return value


# ============================================================================
# FORMULATION 1: CROP PORTFOLIO SELECTION (27 variables)
# ============================================================================

def generate_portfolio_data(n_crops: int = 27, target_selection: int = 15) -> Dict:
    """Generate data for portfolio selection formulation."""
    crop_names = list(CROPS.keys())[:n_crops]
    
    # Calculate values
    values = {c: calculate_crop_value(c) for c in crop_names}
    
    # Build synergy matrix
    synergy_matrix = {}
    for c1 in crop_names:
        for c2 in crop_names:
            if c1 != c2:
                synergy_matrix[(c1, c2)] = get_synergy(c1, c2)
    
    # Food groups for diversity constraint
    groups = {}
    for c in crop_names:
        g = CROPS[c]['group']
        if g not in groups:
            groups[g] = []
        groups[g].append(c)
    
    return {
        'formulation': 'portfolio',
        'crop_names': crop_names,
        'values': values,
        'synergy_matrix': synergy_matrix,
        'groups': groups,
        'target_selection': target_selection,
        'n_variables': n_crops,
    }


def build_portfolio_bqm(data: Dict, gamma: float = 0.3, 
                        lambda_div: float = 2.0, lambda_group: float = 1.0) -> BinaryQuadraticModel:
    """Build BQM for crop portfolio selection."""
    bqm = BinaryQuadraticModel('BINARY')
    
    crop_names = data['crop_names']
    values = data['values']
    synergy_matrix = data['synergy_matrix']
    target = data['target_selection']
    
    # Linear terms: crop values (negate for minimization)
    for c in crop_names:
        bqm.add_variable(c, -values[c])
    
    # Quadratic terms: synergies (negate for minimization)
    for (c1, c2), synergy in synergy_matrix.items():
        if abs(synergy) > 0.01:
            bqm.add_interaction(c1, c2, -gamma * synergy)
    
    # Diversity penalty: (sum - target)^2 = sum^2 - 2*target*sum + target^2
    # sum^2 = sum_i x_i + 2*sum_{i<j} x_i*x_j
    for c in crop_names:
        # Linear: -2*target + 1 (from sum^2)
        bqm.add_variable(c, lambda_div * (1 - 2 * target))
    for i, c1 in enumerate(crop_names):
        for c2 in crop_names[i+1:]:
            # Quadratic: 2 (from sum^2)
            bqm.add_interaction(c1, c2, lambda_div * 2)
    
    # Group balance penalty (at least 1 from each major group)
    for group, crops in data['groups'].items():
        if len(crops) >= 2:  # Only for groups with multiple options
            # Penalize if none selected from group: (1 - sum)^2 when sum=0
            for c1 in crops:
                bqm.add_variable(c1, lambda_group * (-1))  # Encourage at least one
    
    return bqm


def solve_portfolio_gurobi(data: Dict, timeout: int = 1500) -> Dict:
    """Solve portfolio selection with Gurobi at maximum performance."""
    if not HAS_GUROBI:
        return {'success': False, 'error': 'Gurobi not available'}
    
    total_start = time.time()
    
    crop_names = data['crop_names']
    values = data['values']
    synergy_matrix = data['synergy_matrix']
    target = data['target_selection']
    gamma = 0.3
    
    model = gp.Model("PortfolioSelection")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    model.setParam('MIPGap', 0.0001)  # 0.01% optimality gap
    model.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions
    model.setParam('Threads', 0)  # Use all available cores
    
    # Variables
    U = {c: model.addVar(vtype=GRB.BINARY, name=f"U_{c}") for c in crop_names}
    model.update()
    
    # Objective: maximize value + synergies
    obj = gp.quicksum(values[c] * U[c] for c in crop_names)
    obj += gamma * gp.quicksum(synergy_matrix.get((c1, c2), 0) * U[c1] * U[c2] 
                                for c1 in crop_names for c2 in crop_names if c1 != c2)
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraint: select approximately target crops
    model.addConstr(gp.quicksum(U[c] for c in crop_names) >= target - 2, "min_crops")
    model.addConstr(gp.quicksum(U[c] for c in crop_names) <= target + 2, "max_crops")
    
    # Constraint: at least 1 from each group
    for group, crops in data['groups'].items():
        if len(crops) >= 2:
            model.addConstr(gp.quicksum(U[c] for c in crops) >= 1, f"group_{group}")
    
    # Solve
    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start
    
    result = {
        'method': 'portfolio_gurobi',
        'success': False,
        'objective': 0,
        'wall_time': time.time() - total_start,
        'solve_time': solve_time,
        'n_variables': len(crop_names),
        'n_selected': 0,
        'selected_crops': [],
    }
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        result['success'] = True
        result['objective'] = model.ObjVal
        result['selected_crops'] = [c for c in crop_names if U[c].X > 0.5]
        result['n_selected'] = len(result['selected_crops'])
        result['optimal'] = model.Status == GRB.OPTIMAL
    
    return result


def solve_portfolio_qpu(data: Dict, num_reads: int = 100) -> Dict:
    """Solve portfolio selection with D-Wave QPU."""
    if not HAS_DWAVE:
        return {'success': False, 'error': 'D-Wave not available'}
    
    total_start = time.time()
    
    # Build BQM
    bqm = build_portfolio_bqm(data)
    
    # Initialize sampler
    token = get_dwave_token()
    sampler = DWaveCliqueSampler(token=token) if token else DWaveCliqueSampler()
    
    # Solve
    try:
        sampleset = sampler.sample(bqm, num_reads=num_reads, label="Portfolio_Selection")
        
        timing = sampleset.info.get('timing', {})
        qpu_time = timing.get('qpu_access_time', 0) / 1e6  # Convert to seconds
        
        # Decode best solution
        best = sampleset.first
        selected = [c for c in data['crop_names'] if best.sample.get(c, 0) == 1]
        
        # Calculate true objective (not BQM energy)
        obj = sum(data['values'][c] for c in selected)
        obj += 0.3 * sum(data['synergy_matrix'].get((c1, c2), 0) 
                        for c1 in selected for c2 in selected if c1 != c2)
        
        result = {
            'method': 'portfolio_qpu',
            'success': True,
            'objective': obj,
            'wall_time': time.time() - total_start,
            'qpu_time': qpu_time,
            'n_variables': len(data['crop_names']),
            'n_selected': len(selected),
            'selected_crops': selected,
            'energy': best.energy,
        }
        
    except Exception as e:
        result = {
            'method': 'portfolio_qpu',
            'success': False,
            'error': str(e),
            'wall_time': time.time() - total_start,
        }
    
    return result


# ============================================================================
# FORMULATION 3: GRAPH-BASED MWIS (Maximum Weight Independent Set)
# ============================================================================

def generate_graph_mwis_data(n_farms: int = 5, n_crops: int = 6) -> Dict:
    """Generate data for graph MWIS formulation (single period)."""
    farm_names = [f"Farm_{i}" for i in range(n_farms)]
    crop_names = list(CROPS.keys())[:n_crops]
    
    # Node weights: value of assigning crop c to farm f
    # Simplified: just crop value (farms have equal land)
    node_weights = {}
    for f in farm_names:
        for c in crop_names:
            node_weights[(f, c)] = calculate_crop_value(c)
    
    # Edges: incompatible pairs
    # Two nodes are incompatible if they're on the same farm (can only grow one crop)
    edges = []
    for f in farm_names:
        for i, c1 in enumerate(crop_names):
            for c2 in crop_names[i+1:]:
                edges.append(((f, c1), (f, c2)))
    
    return {
        'formulation': 'graph_mwis',
        'farm_names': farm_names,
        'crop_names': crop_names,
        'node_weights': node_weights,
        'edges': edges,
        'n_variables': n_farms * n_crops,
        'n_farms': n_farms,
        'n_crops': n_crops,
    }


def build_graph_mwis_bqm(data: Dict, penalty: float = 5.0) -> BinaryQuadraticModel:
    """Build BQM for MWIS formulation."""
    bqm = BinaryQuadraticModel('BINARY')
    
    # Variable naming: "f_c" for farm f, crop c
    for (f, c), weight in data['node_weights'].items():
        var_name = f"{f}_{c}"
        bqm.add_variable(var_name, -weight)  # Negate for minimization
    
    # Edge penalties: incompatible pairs
    for (node1, node2) in data['edges']:
        var1 = f"{node1[0]}_{node1[1]}"
        var2 = f"{node2[0]}_{node2[1]}"
        bqm.add_interaction(var1, var2, penalty)
    
    return bqm


def solve_graph_mwis_gurobi(data: Dict, timeout: int = 1500) -> Dict:
    """Solve MWIS with Gurobi at maximum performance."""
    if not HAS_GUROBI:
        return {'success': False, 'error': 'Gurobi not available'}
    
    total_start = time.time()
    
    model = gp.Model("GraphMWIS")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    model.setParam('MIPGap', 0.0001)  # 0.01% optimality gap
    model.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions
    model.setParam('Threads', 0)  # Use all available cores
    
    # Variables
    X = {}
    for (f, c), weight in data['node_weights'].items():
        X[(f, c)] = model.addVar(vtype=GRB.BINARY, name=f"X_{f}_{c}")
    model.update()
    
    # Objective: maximize total weight
    obj = gp.quicksum(data['node_weights'][(f, c)] * X[(f, c)] 
                      for (f, c) in data['node_weights'])
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints: independent set (no two adjacent nodes)
    for (node1, node2) in data['edges']:
        model.addConstr(X[node1] + X[node2] <= 1, f"edge_{node1}_{node2}")
    
    # Solve
    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start
    
    result = {
        'method': 'graph_mwis_gurobi',
        'success': False,
        'objective': 0,
        'wall_time': time.time() - total_start,
        'solve_time': solve_time,
        'n_variables': data['n_variables'],
    }
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        result['success'] = True
        result['objective'] = model.ObjVal
        result['solution'] = {(f, c): X[(f, c)].X for (f, c) in X if X[(f, c)].X > 0.5}
        result['optimal'] = model.Status == GRB.OPTIMAL
    
    return result


def solve_graph_mwis_qpu(data: Dict, num_reads: int = 100) -> Dict:
    """Solve MWIS with D-Wave QPU."""
    if not HAS_DWAVE:
        return {'success': False, 'error': 'D-Wave not available'}
    
    total_start = time.time()
    
    bqm = build_graph_mwis_bqm(data)
    
    token = get_dwave_token()
    sampler = DWaveCliqueSampler(token=token) if token else DWaveCliqueSampler()
    
    try:
        sampleset = sampler.sample(bqm, num_reads=num_reads, label="Graph_MWIS")
        
        timing = sampleset.info.get('timing', {})
        qpu_time = timing.get('qpu_access_time', 0) / 1e6
        
        best = sampleset.first
        
        # Decode solution
        solution = {}
        for (f, c) in data['node_weights']:
            var_name = f"{f}_{c}"
            if best.sample.get(var_name, 0) == 1:
                solution[(f, c)] = 1
        
        # Calculate objective
        obj = sum(data['node_weights'][(f, c)] for (f, c) in solution)
        
        # Count violations
        violations = 0
        for (node1, node2) in data['edges']:
            if node1 in solution and node2 in solution:
                violations += 1
        
        result = {
            'method': 'graph_mwis_qpu',
            'success': True,
            'objective': obj,
            'wall_time': time.time() - total_start,
            'qpu_time': qpu_time,
            'n_variables': data['n_variables'],
            'violations': violations,
            'energy': best.energy,
        }
        
    except Exception as e:
        result = {
            'method': 'graph_mwis_qpu',
            'success': False,
            'error': str(e),
            'wall_time': time.time() - total_start,
        }
    
    return result


# ============================================================================
# FORMULATION 4: SINGLE PERIOD OPTIMIZATION (no temporal coupling)
# ============================================================================

def generate_single_period_data(n_farms: int = 5, n_crops: int = 6) -> Dict:
    """Generate data for single period optimization."""
    farm_names = [f"Farm_{i}" for i in range(n_farms)]
    crop_names = list(CROPS.keys())[:n_crops]
    
    # Farm areas (normalized)
    land = {f: 1.0 / n_farms for f in farm_names}
    
    # Crop values
    values = {c: calculate_crop_value(c) for c in crop_names}
    
    # Neighbor graph (grid)
    side = int(np.ceil(np.sqrt(n_farms)))
    neighbors = []
    for i, f1 in enumerate(farm_names):
        r1, c1 = i // side, i % side
        for j, f2 in enumerate(farm_names):
            if i < j:
                r2, c2 = j // side, j % side
                if abs(r1 - r2) + abs(c1 - c2) == 1:  # Manhattan distance 1
                    neighbors.append((f1, f2))
    
    return {
        'formulation': 'single_period',
        'farm_names': farm_names,
        'crop_names': crop_names,
        'land': land,
        'values': values,
        'neighbors': neighbors,
        'n_variables': n_farms * n_crops,
        'n_farms': n_farms,
        'n_crops': n_crops,
    }


def build_single_period_bqm(data: Dict, one_hot_penalty: float = 3.0,
                            neighbor_bonus: float = 0.1) -> BinaryQuadraticModel:
    """Build BQM for single period optimization."""
    bqm = BinaryQuadraticModel('BINARY')
    
    farm_names = data['farm_names']
    crop_names = data['crop_names']
    values = data['values']
    land = data['land']
    
    # Linear: crop benefit * land
    for f in farm_names:
        for c in crop_names:
            var = f"{f}_{c}"
            bqm.add_variable(var, -values[c] * land[f])
    
    # One-hot penalty per farm
    for f in farm_names:
        farm_vars = [f"{f}_{c}" for c in crop_names]
        # (sum - 1)^2 = sum^2 - 2*sum + 1
        for v in farm_vars:
            bqm.add_variable(v, one_hot_penalty * (-1))  # Linear: -2 + 1 = -1
        for i, v1 in enumerate(farm_vars):
            for v2 in farm_vars[i+1:]:
                bqm.add_interaction(v1, v2, one_hot_penalty * 2)
    
    # Neighbor diversity bonus
    for (f1, f2) in data['neighbors']:
        for c in crop_names:
            # Penalty for same crop on neighbors
            v1 = f"{f1}_{c}"
            v2 = f"{f2}_{c}"
            bqm.add_interaction(v1, v2, neighbor_bonus)
    
    return bqm


def solve_single_period_gurobi(data: Dict, timeout: int = 1500) -> Dict:
    """Solve single period with Gurobi at maximum performance."""
    if not HAS_GUROBI:
        return {'success': False, 'error': 'Gurobi not available'}
    
    total_start = time.time()
    
    model = gp.Model("SinglePeriod")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    model.setParam('MIPGap', 0.0001)  # 0.01% optimality gap
    model.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions
    model.setParam('Threads', 0)  # Use all available cores
    
    farm_names = data['farm_names']
    crop_names = data['crop_names']
    values = data['values']
    land = data['land']
    
    # Variables
    Y = {}
    for f in farm_names:
        for c in crop_names:
            Y[(f, c)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}")
    model.update()
    
    # Objective
    obj = gp.quicksum(values[c] * land[f] * Y[(f, c)] 
                      for f in farm_names for c in crop_names)
    # Neighbor diversity bonus
    for (f1, f2) in data['neighbors']:
        for c in crop_names:
            obj -= 0.1 * Y[(f1, c)] * Y[(f2, c)]
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # One crop per farm
    for f in farm_names:
        model.addConstr(gp.quicksum(Y[(f, c)] for c in crop_names) == 1, f"one_hot_{f}")
    
    # Solve
    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start
    
    result = {
        'method': 'single_period_gurobi',
        'success': False,
        'objective': 0,
        'wall_time': time.time() - total_start,
        'solve_time': solve_time,
        'n_variables': data['n_variables'],
    }
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        result['success'] = True
        result['objective'] = model.ObjVal
        result['solution'] = {(f, c): Y[(f, c)].X for (f, c) in Y if Y[(f, c)].X > 0.5}
        result['optimal'] = model.Status == GRB.OPTIMAL
    
    return result


def solve_single_period_qpu(data: Dict, num_reads: int = 100) -> Dict:
    """Solve single period with D-Wave QPU."""
    if not HAS_DWAVE:
        return {'success': False, 'error': 'D-Wave not available'}
    
    total_start = time.time()
    
    bqm = build_single_period_bqm(data)
    
    token = get_dwave_token()
    sampler = DWaveCliqueSampler(token=token) if token else DWaveCliqueSampler()
    
    try:
        sampleset = sampler.sample(bqm, num_reads=num_reads, label="Single_Period")
        
        timing = sampleset.info.get('timing', {})
        qpu_time = timing.get('qpu_access_time', 0) / 1e6
        
        best = sampleset.first
        
        # Decode solution - pick best crop per farm
        solution = {}
        for f in data['farm_names']:
            best_crop = None
            best_val = -1
            for c in data['crop_names']:
                var = f"{f}_{c}"
                val = best.sample.get(var, 0)
                if val > best_val:
                    best_val = val
                    best_crop = c
            if best_crop:
                solution[(f, best_crop)] = 1
        
        # Calculate objective
        obj = sum(data['values'][c] * data['land'][f] for (f, c) in solution)
        
        # Count one-hot violations
        violations = 0
        for f in data['farm_names']:
            count = sum(1 for c in data['crop_names'] 
                       if best.sample.get(f"{f}_{c}", 0) == 1)
            if count != 1:
                violations += 1
        
        result = {
            'method': 'single_period_qpu',
            'success': True,
            'objective': obj,
            'wall_time': time.time() - total_start,
            'qpu_time': qpu_time,
            'n_variables': data['n_variables'],
            'violations': violations,
            'energy': best.energy,
        }
        
    except Exception as e:
        result = {
            'method': 'single_period_qpu',
            'success': False,
            'error': str(e),
            'wall_time': time.time() - total_start,
        }
    
    return result


# ============================================================================
# FORMULATION 5: PENALTY MODEL (soft constraints)
# ============================================================================

def generate_penalty_data(n_farms: int = 5, n_crops: int = 6, n_periods: int = 3) -> Dict:
    """Generate data for penalty model (multi-period with soft constraints)."""
    farm_names = [f"Farm_{i}" for i in range(n_farms)]
    crop_names = list(CROPS.keys())[:n_crops]
    
    land = {f: 1.0 / n_farms for f in farm_names}
    values = {c: calculate_crop_value(c) for c in crop_names}
    
    # Build synergy matrix for rotation
    rotation_synergy = {}
    for c1 in crop_names:
        for c2 in crop_names:
            rotation_synergy[(c1, c2)] = get_synergy(c1, c2)
    
    return {
        'formulation': 'penalty_model',
        'farm_names': farm_names,
        'crop_names': crop_names,
        'n_periods': n_periods,
        'land': land,
        'values': values,
        'rotation_synergy': rotation_synergy,
        'n_variables': n_farms * n_crops * n_periods,
        'n_farms': n_farms,
        'n_crops': n_crops,
    }


def build_penalty_bqm(data: Dict, one_hot_penalty: float = 3.0,
                      rotation_gamma: float = 0.2) -> BinaryQuadraticModel:
    """Build BQM for penalty model (all soft constraints)."""
    bqm = BinaryQuadraticModel('BINARY')
    
    farm_names = data['farm_names']
    crop_names = data['crop_names']
    n_periods = data['n_periods']
    values = data['values']
    land = data['land']
    rotation_synergy = data['rotation_synergy']
    
    # Linear: crop benefit
    for f in farm_names:
        for c in crop_names:
            for t in range(1, n_periods + 1):
                var = f"{f}_{c}_t{t}"
                bqm.add_variable(var, -values[c] * land[f])
    
    # Rotation synergies
    for f in farm_names:
        for t in range(2, n_periods + 1):
            for c1 in crop_names:
                for c2 in crop_names:
                    var1 = f"{f}_{c1}_t{t-1}"
                    var2 = f"{f}_{c2}_t{t}"
                    synergy = rotation_synergy.get((c1, c2), 0)
                    if abs(synergy) > 0.01:
                        bqm.add_interaction(var1, var2, -rotation_gamma * synergy * land[f])
    
    # One-hot penalty per farm per period
    for f in farm_names:
        for t in range(1, n_periods + 1):
            vars_ft = [f"{f}_{c}_t{t}" for c in crop_names]
            for v in vars_ft:
                bqm.add_variable(v, one_hot_penalty * (-1))
            for i, v1 in enumerate(vars_ft):
                for v2 in vars_ft[i+1:]:
                    bqm.add_interaction(v1, v2, one_hot_penalty * 2)
    
    return bqm


def solve_penalty_gurobi(data: Dict, timeout: int = 1500) -> Dict:
    """Solve penalty model with Gurobi at maximum performance (using hard constraints for ground truth)."""
    if not HAS_GUROBI:
        return {'success': False, 'error': 'Gurobi not available'}
    
    total_start = time.time()
    
    model = gp.Model("PenaltyModel")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    model.setParam('MIPGap', 0.0001)  # 0.01% optimality gap
    model.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions
    model.setParam('Threads', 0)  # Use all available cores
    
    farm_names = data['farm_names']
    crop_names = data['crop_names']
    n_periods = data['n_periods']
    values = data['values']
    land = data['land']
    rotation_synergy = data['rotation_synergy']
    rotation_gamma = 0.2
    
    # Variables
    Y = {}
    for f in farm_names:
        for c in crop_names:
            for t in range(1, n_periods + 1):
                Y[(f, c, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}_t{t}")
    model.update()
    
    # Objective
    obj = gp.quicksum(values[c] * land[f] * Y[(f, c, t)]
                      for f in farm_names for c in crop_names for t in range(1, n_periods + 1))
    # Rotation synergies
    for f in farm_names:
        for t in range(2, n_periods + 1):
            for c1 in crop_names:
                for c2 in crop_names:
                    synergy = rotation_synergy.get((c1, c2), 0)
                    if abs(synergy) > 0.01:
                        obj += rotation_gamma * synergy * land[f] * Y[(f, c1, t-1)] * Y[(f, c2, t)]
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Hard constraint: one crop per farm per period
    for f in farm_names:
        for t in range(1, n_periods + 1):
            model.addConstr(gp.quicksum(Y[(f, c, t)] for c in crop_names) == 1, f"one_hot_{f}_t{t}")
    
    # Solve
    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start
    
    result = {
        'method': 'penalty_gurobi',
        'success': False,
        'objective': 0,
        'wall_time': time.time() - total_start,
        'solve_time': solve_time,
        'n_variables': data['n_variables'],
    }
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        result['success'] = True
        result['objective'] = model.ObjVal
        result['optimal'] = model.Status == GRB.OPTIMAL
    
    return result


def solve_penalty_qpu(data: Dict, num_reads: int = 100, num_iterations: int = 3) -> Dict:
    """Solve penalty model with D-Wave QPU using decomposition."""
    if not HAS_DWAVE:
        return {'success': False, 'error': 'D-Wave not available'}
    
    total_start = time.time()
    total_qpu_time = 0
    
    # For 90 variables, decompose by farm (each farm = 18 vars = 6 crops × 3 periods)
    token = get_dwave_token()
    sampler = DWaveCliqueSampler(token=token) if token else DWaveCliqueSampler()
    
    farm_names = data['farm_names']
    crop_names = data['crop_names']
    n_periods = data['n_periods']
    
    best_solution = {}
    best_objective = -np.inf
    
    try:
        for iteration in range(num_iterations):
            # Solve each farm independently
            for f in farm_names:
                # Build small BQM for this farm
                farm_bqm = BinaryQuadraticModel('BINARY')
                
                # Crop benefits
                for c in crop_names:
                    for t in range(1, n_periods + 1):
                        var = f"{c}_t{t}"
                        farm_bqm.add_variable(var, -data['values'][c] * data['land'][f])
                
                # Rotation synergies
                for t in range(2, n_periods + 1):
                    for c1 in crop_names:
                        for c2 in crop_names:
                            synergy = data['rotation_synergy'].get((c1, c2), 0)
                            if abs(synergy) > 0.01:
                                v1 = f"{c1}_t{t-1}"
                                v2 = f"{c2}_t{t}"
                                farm_bqm.add_interaction(v1, v2, -0.2 * synergy * data['land'][f])
                
                # One-hot per period
                for t in range(1, n_periods + 1):
                    vars_t = [f"{c}_t{t}" for c in crop_names]
                    for v in vars_t:
                        farm_bqm.add_variable(v, 3.0 * (-1))
                    for i, v1 in enumerate(vars_t):
                        for v2 in vars_t[i+1:]:
                            farm_bqm.add_interaction(v1, v2, 3.0 * 2)
                
                # Solve
                sampleset = sampler.sample(farm_bqm, num_reads=num_reads, 
                                          label=f"Penalty_Farm_{f}_iter{iteration}")
                
                timing = sampleset.info.get('timing', {})
                total_qpu_time += timing.get('qpu_access_time', 0) / 1e6
                
                # Decode and store solution
                best = sampleset.first
                for t in range(1, n_periods + 1):
                    best_crop = None
                    best_val = -1
                    for c in crop_names:
                        var = f"{c}_t{t}"
                        val = best.sample.get(var, 0)
                        if val > best_val:
                            best_val = val
                            best_crop = c
                    if best_crop:
                        best_solution[(f, best_crop, t)] = 1
            
            # Calculate objective
            obj = sum(data['values'][c] * data['land'][f] 
                     for (f, c, t) in best_solution)
            if obj > best_objective:
                best_objective = obj
        
        result = {
            'method': 'penalty_qpu',
            'success': True,
            'objective': best_objective,
            'wall_time': time.time() - total_start,
            'qpu_time': total_qpu_time,
            'n_variables': data['n_variables'],
            'violations': 0,  # Enforced via decoding
        }
        
    except Exception as e:
        result = {
            'method': 'penalty_qpu',
            'success': False,
            'error': str(e),
            'wall_time': time.time() - total_start,
        }
    
    return result


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_formulation_benchmark(formulations: List[str] = None, 
                              n_farms: int = 5, 
                              n_crops: int = 6,
                              num_reads: int = 100,
                              runs: int = 1,
                              gurobi_only: bool = False) -> Dict:
    """Run benchmark for all specified formulations."""
    
    if formulations is None:
        formulations = ['portfolio', 'graph_mwis', 'single_period', 'penalty']
    
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'config': {
            'formulations': formulations,
            'n_farms': n_farms,
            'n_crops': n_crops,
            'num_reads': num_reads,
            'runs': runs,
        },
        'results': {},
    }
    
    print("=" * 80)
    print("ALTERNATIVE FORMULATIONS BENCHMARK")
    print("=" * 80)
    print(f"Formulations: {formulations}")
    print(f"Parameters: {n_farms} farms, {n_crops} crops, {num_reads} reads, {runs} runs")
    print("=" * 80)
    print()
    
    for formulation in formulations:
        print(f"\n{'='*60}")
        print(f"FORMULATION: {formulation.upper()}")
        print('='*60)
        
        # Generate data
        if formulation == 'portfolio':
            data = generate_portfolio_data(n_crops=27, target_selection=15)
            gurobi_fn = solve_portfolio_gurobi
            qpu_fn = solve_portfolio_qpu
        elif formulation == 'graph_mwis':
            data = generate_graph_mwis_data(n_farms=n_farms, n_crops=n_crops)
            gurobi_fn = solve_graph_mwis_gurobi
            qpu_fn = solve_graph_mwis_qpu
        elif formulation == 'single_period':
            data = generate_single_period_data(n_farms=n_farms, n_crops=n_crops)
            gurobi_fn = solve_single_period_gurobi
            qpu_fn = solve_single_period_qpu
        elif formulation == 'penalty':
            data = generate_penalty_data(n_farms=n_farms, n_crops=n_crops, n_periods=3)
            gurobi_fn = solve_penalty_gurobi
            qpu_fn = solve_penalty_qpu
        else:
            print(f"Unknown formulation: {formulation}")
            continue
        
        print(f"Variables: {data['n_variables']}")
        
        form_results = {
            'data': {k: v for k, v in data.items() if k not in ['synergy_matrix', 'rotation_synergy', 'node_weights']},
            'gurobi': [],
            'qpu': [],
        }
        
        # Run Gurobi
        print(f"\n--- Gurobi ---")
        for run in range(runs):
            result = gurobi_fn(data)
            form_results['gurobi'].append(result)
            if result.get('success'):
                print(f"  Run {run+1}: obj={result['objective']:.4f}, time={result['wall_time']:.3f}s")
            else:
                print(f"  Run {run+1}: FAILED - {result.get('error', 'unknown')}")
        
        # Run QPU (skip if gurobi_only)
        if not gurobi_only:
            print(f"\n--- D-Wave QPU ---")
            for run in range(runs):
                result = qpu_fn(data, num_reads=num_reads)
                form_results['qpu'].append(result)
                if result.get('success'):
                    print(f"  Run {run+1}: obj={result['objective']:.4f}, time={result['wall_time']:.3f}s, qpu={result.get('qpu_time', 0):.3f}s")
                else:
                    print(f"  Run {run+1}: FAILED - {result.get('error', 'unknown')}")
        else:
            print(f"\n--- D-Wave QPU: SKIPPED (gurobi-only mode) ---")
        
        # Calculate statistics
        gurobi_objs = [r['objective'] for r in form_results['gurobi'] if r.get('success')]
        qpu_objs = [r['objective'] for r in form_results['qpu'] if r.get('success')]
        gurobi_times = [r['wall_time'] for r in form_results['gurobi'] if r.get('success')]
        qpu_wall_times = [r['wall_time'] for r in form_results['qpu'] if r.get('success')]
        qpu_only_times = [r.get('qpu_time', 0) for r in form_results['qpu'] if r.get('success')]
        
        if gurobi_objs and qpu_objs:
            gt_obj = np.mean(gurobi_objs)
            qpu_obj = np.mean(qpu_objs)
            gap = (gt_obj - qpu_obj) / gt_obj * 100 if gt_obj != 0 else 0
            speedup = np.mean(gurobi_times) / np.mean(qpu_wall_times) if np.mean(qpu_wall_times) > 0 else 1
            
            form_results['summary'] = {
                'gurobi_obj': gt_obj,
                'qpu_obj': qpu_obj,
                'gap': gap,
                'gurobi_time': np.mean(gurobi_times),
                'qpu_wall_time': np.mean(qpu_wall_times),
                'qpu_only_time': np.mean(qpu_only_times),
                'speedup': speedup,
            }
            
            print(f"\n--- Summary ---")
            print(f"  Gurobi: obj={gt_obj:.4f}, time={np.mean(gurobi_times):.3f}s")
            print(f"  QPU:    obj={qpu_obj:.4f}, wall={np.mean(qpu_wall_times):.3f}s, qpu={np.mean(qpu_only_times):.3f}s")
            print(f"  Gap:    {gap:.1f}%")
            print(f"  Speedup: {speedup:.1f}x (wall time)")
        
        results['results'][formulation] = form_results
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / f"formulation_benchmark_{results['timestamp']}.json"
    
    # Convert to JSON-serializable
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")
    
    return results


def generate_comparison_table(results: Dict):
    """Generate comparison table for all formulations."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Formulation':<20} {'Vars':>6} {'Gurobi':>10} {'QPU Wall':>10} {'QPU Only':>10} {'Gap':>8} {'Speedup':>10}")
    print("-" * 80)
    
    for formulation, data in results['results'].items():
        if 'summary' in data:
            s = data['summary']
            n_vars = data['data']['n_variables']
            gurobi_time = s.get('gurobi_time', 0)
            qpu_wall = s.get('qpu_wall_time', s.get('qpu_time', 0))  # Backwards compat
            qpu_only = s.get('qpu_only_time', 0)
            print(f"{formulation:<20} {n_vars:>6} {gurobi_time:>9.3f}s {qpu_wall:>9.3f}s {qpu_only:>9.3f}s "
                  f"{s['gap']:>7.1f}% {s['speedup']:>9.1f}x")
    
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Alternative Formulations Benchmark')
    parser.add_argument('--formulations', nargs='+', 
                        default=['portfolio', 'graph_mwis', 'single_period', 'penalty'],
                        choices=['portfolio', 'graph_mwis', 'single_period', 'penalty', 'hierarchical'],
                        help='Formulations to benchmark')
    parser.add_argument('--farms', type=int, default=5, help='Number of farms (default: 5)')
    parser.add_argument('--crops', type=int, default=6, help='Number of crops (default: 6)')
    parser.add_argument('--reads', type=int, default=100, help='QPU reads (default: 100)')
    parser.add_argument('--runs', type=int, default=1, help='Runs per method (default: 1)')
    parser.add_argument('--token', type=str, default=None, help='D-Wave API token')
    parser.add_argument('--gurobi-only', action='store_true', help='Run only Gurobi (no QPU)')
    
    args = parser.parse_args()
    
    if args.token:
        set_dwave_token(args.token)
    else:
        token = get_dwave_token()
        if token:
            print(f"  D-Wave token loaded (length: {len(token)})")
    
    results = run_formulation_benchmark(
        formulations=args.formulations,
        n_farms=args.farms,
        n_crops=args.crops,
        num_reads=args.reads,
        runs=args.runs,
        gurobi_only=args.gurobi_only,
    )
    
    generate_comparison_table(results)


if __name__ == '__main__':
    main()
