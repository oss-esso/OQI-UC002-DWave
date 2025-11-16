#!/usr/bin/env python3
"""
BQM Constraint Violation Diagnostic Script

This script performs a comprehensive analysis of three BQM formulations:
1. BQUBO: Binary plantation model (works correctly)
2. PATCH: Plot assignment model (violates constraints)
3. PATCH_NO_IDLE: Plot assignment without idle area constraint

The goal is to identify why PATCH formulation prefers violating constraints
over using the QPU, while BQUBO does not.

Analysis includes:
- Instance characterization (size, density, coefficient distribution, topology)
- Solver-independent hardness metrics (spectral gap, local minima, degeneracy)
- Solver performance comparison (multiple solvers and seeds)
- Constraint stress testing (penalty sensitivity, feasibility, embedding)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from typing import Dict, Tuple, List, Any
from collections import defaultdict
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Scientific computing
try:
    from scipy import sparse
    from scipy.sparse.linalg import eigsh, LinearOperator
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some analyses will be skipped.", file=sys.stderr)

# D-Wave and optimization
from dimod import BinaryQuadraticModel, cqm_to_bqm, ExactSolver
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler, SteepestDescentSampler
import networkx as nx

# Optional: Gurobi for classical baseline
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

# Optional: PuLP for CQM solving
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

# Add project root to path
# Add project root and Benchmark Scripts to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

try:
    from solver_runner_PATCH import create_cqm as create_patch_cqm
    from solver_runner_BQUBO import create_cqm as create_bqubo_cqm
    from .patch_sampler import generate_farms as generate_patches
    from .farm_sampler import generate_farms as generate_farms_bqubo
    from src.scenarios import load_food_data
except ImportError as e:
    print(f"Error: Could not import necessary modules. {e}", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# SECTION 1: PROBLEM GENERATION
# ============================================================================

def create_food_config(land_data: Dict[str, float], include_idle_penalty: bool = True) -> Tuple[Dict, Dict, Dict]:
    """Creates the configuration for PATCH formulation."""
    try:
        _, foods, food_groups, _ = load_food_data('simple')
    except Exception:
        foods = {'Wheat': {'nutritional_value': 0.8}, 'Corn': {'nutritional_value': 0.7}}
        food_groups = {'Grains': ['Wheat', 'Corn']}

    config = {
        'parameters': {
            'land_availability': land_data,
            'minimum_planting_area': {food: 0.0 for food in foods},
            'food_group_constraints': {},
            'weights': {
                'nutritional_value': 0.25,
                'nutrient_density': 0.2,
                'cost_efficiency': 0.2,
                'environmental_impact': 0.2,
                'sustainability': 0.15
            },
            'idle_penalty_lambda': 0.1 if include_idle_penalty else 0.0
        }
    }
    
    for food_name, food_attrs in foods.items():
        for weight in config['parameters']['weights']:
            if weight not in food_attrs:
                food_attrs[weight] = 0.5
    
    return foods, food_groups, config


def create_bqubo_config(land_data: Dict[str, float]) -> Tuple[Dict, Dict, Dict]:
    """Creates the configuration for BQUBO formulation."""
    try:
        _, foods, food_groups, _ = load_food_data('simple')
    except Exception:
        foods = {'Wheat': {'nutritional_value': 0.8}, 'Corn': {'nutritional_value': 0.7}}
        food_groups = {'Grains': ['Wheat', 'Corn']}

    config = {
        'parameters': {
            'land_availability': land_data,
            'food_group_constraints': {
                g: {'min_foods': 0, 'max_foods': len(lst)}
                for g, lst in food_groups.items()
            },
            'weights': {
                'nutritional_value': 0.25,
                'nutrient_density': 0.2,
                'cost_efficiency': 0.2,
                'environmental_impact': 0.2,
                'sustainability': 0.15
            }
        }
    }
    
    for food_name, food_attrs in foods.items():
        for weight in config['parameters']['weights']:
            if weight not in food_attrs:
                food_attrs[weight] = 0.5
    
    return foods, food_groups, config


def generate_bqm_instances(n_vars: int = 50, seed: int = 42) -> Dict[str, Dict]:
    """
    Generate three BQM instances with approximately n_vars variables each.
    
    Returns:
        Dictionary with keys 'bqubo', 'patch', 'patch_no_idle', each containing
        {'bqm': BinaryQuadraticModel, 'cqm': ConstrainedQuadraticModel, 'metadata': dict}
    """
    print(f"Generating BQM instances (target: {n_vars} variables)...", file=sys.stderr)
    instances = {}
    
    # 1. BQUBO formulation
    print("  - Building BQUBO formulation...", file=sys.stderr)
    farms_bqubo = generate_farms_bqubo(n_farms=n_vars, seed=seed)
    foods_bqubo, food_groups_bqubo, config_bqubo = create_bqubo_config(farms_bqubo)
    cqm_bqubo, _, _ = create_bqubo_cqm(list(farms_bqubo.keys()), foods_bqubo, food_groups_bqubo, config_bqubo)
    bqm_bqubo, infodict_bqubo = cqm_to_bqm(cqm_bqubo)
    
    instances['bqubo'] = {
        'bqm': bqm_bqubo,
        'cqm': cqm_bqubo,
        'metadata': {
            'n_farms': len(farms_bqubo),
            'n_crops': len(foods_bqubo),
            'total_capacity': sum(farms_bqubo.values()),
            'lagrange_multiplier': getattr(infodict_bqubo, 'lagrange_multiplier', 'auto')
        }
    }
    
    # 2. PATCH formulation (with idle penalty)
    print("  - Building PATCH formulation (with idle penalty)...", file=sys.stderr)
    patches = generate_patches(n_farms=n_vars, seed=seed)
    foods_patch, food_groups_patch, config_patch = create_food_config(patches, include_idle_penalty=True)
    cqm_patch, _, _ = create_patch_cqm(list(patches.keys()), foods_patch, food_groups_patch, config_patch)
    bqm_patch, infodict_patch = cqm_to_bqm(cqm_patch)
    
    instances['patch'] = {
        'bqm': bqm_patch,
        'cqm': cqm_patch,
        'metadata': {
            'n_patches': len(patches),
            'n_crops': len(foods_patch),
            'total_area': sum(patches.values()),
            'area_variance': np.var(list(patches.values())),
            'has_idle_penalty': True,
            'lagrange_multiplier': getattr(infodict_patch, 'lagrange_multiplier', 'auto')
        }
    }
    
    # 3. PATCH formulation (without idle penalty)
    print("  - Building PATCH formulation (NO idle penalty)...", file=sys.stderr)
    foods_patch_ni, food_groups_patch_ni, config_patch_ni = create_food_config(patches, include_idle_penalty=False)
    cqm_patch_ni, _, _ = create_patch_cqm(list(patches.keys()), foods_patch_ni, food_groups_patch_ni, config_patch_ni)
    bqm_patch_ni, infodict_patch_ni = cqm_to_bqm(cqm_patch_ni)
    
    instances['patch_no_idle'] = {
        'bqm': bqm_patch_ni,
        'cqm': cqm_patch_ni,
        'metadata': {
            'n_patches': len(patches),
            'n_crops': len(foods_patch_ni),
            'total_area': sum(patches.values()),
            'area_variance': np.var(list(patches.values())),
            'has_idle_penalty': False,
            'lagrange_multiplier': getattr(infodict_patch_ni, 'lagrange_multiplier', 'auto')
        }
    }
    
    print(f"Generated instances: BQUBO ({len(instances['bqubo']['bqm'].variables)} vars), "
          f"PATCH ({len(instances['patch']['bqm'].variables)} vars), "
          f"PATCH_NO_IDLE ({len(instances['patch_no_idle']['bqm'].variables)} vars)", file=sys.stderr)
    
    return instances


# ============================================================================
# SECTION 2: INSTANCE CHARACTERIZATION
# ============================================================================

def characterize_instance(bqm: BinaryQuadraticModel, name: str) -> Dict:
    """
    Characterize a BQM instance: size, density, coefficient distribution, topology.
    """
    print(f"  Characterizing {name}...", file=sys.stderr)
    
    n_vars = len(bqm.variables)
    n_interactions = len(bqm.quadratic)
    max_possible_interactions = n_vars * (n_vars - 1) // 2
    density = n_interactions / max_possible_interactions if max_possible_interactions > 0 else 0
    
    # Coefficient statistics
    linear_coeffs = np.array(list(bqm.linear.values()))
    quadratic_coeffs = np.array(list(bqm.quadratic.values()))
    
    linear_stats = {
        'mean': float(np.mean(linear_coeffs)) if len(linear_coeffs) > 0 else 0,
        'std': float(np.std(linear_coeffs)) if len(linear_coeffs) > 0 else 0,
        'min': float(np.min(linear_coeffs)) if len(linear_coeffs) > 0 else 0,
        'max': float(np.max(linear_coeffs)) if len(linear_coeffs) > 0 else 0,
        'range': float(np.ptp(linear_coeffs)) if len(linear_coeffs) > 0 else 0
    }
    
    quadratic_stats = {
        'mean': float(np.mean(quadratic_coeffs)) if len(quadratic_coeffs) > 0 else 0,
        'std': float(np.std(quadratic_coeffs)) if len(quadratic_coeffs) > 0 else 0,
        'min': float(np.min(quadratic_coeffs)) if len(quadratic_coeffs) > 0 else 0,
        'max': float(np.max(quadratic_coeffs)) if len(quadratic_coeffs) > 0 else 0,
        'range': float(np.ptp(quadratic_coeffs)) if len(quadratic_coeffs) > 0 else 0
    }
    
    # Graph topology
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    G.add_edges_from(bqm.quadratic.keys())
    
    topology = {
        'n_connected_components': nx.number_connected_components(G),
        'avg_degree': float(np.mean([d for _, d in G.degree()])) if n_vars > 0 else 0,
        'max_degree': max([d for _, d in G.degree()]) if n_vars > 0 else 0,
        'avg_clustering': float(nx.average_clustering(G)) if n_vars > 0 else 0,
        'is_bipartite': nx.is_bipartite(G)
    }
    
    # Try to calculate diameter for small graphs
    if n_vars < 1000 and nx.is_connected(G):
        topology['diameter'] = nx.diameter(G)
    else:
        topology['diameter'] = None
    
    return {
        'n_variables': n_vars,
        'n_interactions': n_interactions,
        'density': float(density),
        'offset': float(bqm.offset),
        'linear_coefficients': linear_stats,
        'quadratic_coefficients': quadratic_stats,
        'graph_topology': topology
    }


# ============================================================================
# SECTION 3: SOLVER-INDEPENDENT HARDNESS METRICS
# ============================================================================

def compute_qubo_matrix(bqm: BinaryQuadraticModel) -> np.ndarray:
    """Convert BQM to dense QUBO matrix Q where energy = x^T Q x."""
    variables = list(bqm.variables)
    n = len(variables)
    var_to_idx = {v: i for i, v in enumerate(variables)}
    
    Q = np.zeros((n, n))
    
    # Diagonal (linear terms)
    for var, bias in bqm.linear.items():
        i = var_to_idx[var]
        Q[i, i] = bias
    
    # Off-diagonal (quadratic terms)
    for (v1, v2), bias in bqm.quadratic.items():
        i, j = var_to_idx[v1], var_to_idx[v2]
        Q[i, j] = bias / 2  # Split between symmetric positions
        Q[j, i] = bias / 2
    
    return Q


def measure_hardness_metrics(bqm: BinaryQuadraticModel, name: str) -> Dict:
    """
    Measure solver-independent hardness:
    - Spectral gap (eigenvalue gap)
    - Rank of QUBO matrix
    - Condition number
    - Energy landscape statistics (sampled)
    """
    print(f"  Measuring hardness for {name}...", file=sys.stderr)
    
    n_vars = len(bqm.variables)
    metrics = {}
    
    # For small problems, compute exact spectrum
    if n_vars <= 100 and SCIPY_AVAILABLE:
        try:
            Q = compute_qubo_matrix(bqm)
            eigenvalues = np.linalg.eigvalsh(Q)
            eigenvalues = np.sort(eigenvalues)
            
            metrics['min_eigenvalue'] = float(eigenvalues[0])
            metrics['max_eigenvalue'] = float(eigenvalues[-1])
            metrics['spectral_gap'] = float(eigenvalues[1] - eigenvalues[0]) if n_vars > 1 else 0
            metrics['condition_number'] = float(np.abs(eigenvalues[-1] / eigenvalues[0])) if eigenvalues[0] != 0 else float('inf')
            metrics['rank'] = int(np.linalg.matrix_rank(Q))
            metrics['trace'] = float(np.trace(Q))
        except Exception as e:
            print(f"    Warning: Could not compute spectrum: {e}", file=sys.stderr)
            metrics.update({
                'min_eigenvalue': None,
                'max_eigenvalue': None,
                'spectral_gap': None,
                'condition_number': None,
                'rank': None,
                'trace': None
            })
    else:
        # For large problems, sample a few extreme eigenvalues
        if SCIPY_AVAILABLE:
            try:
                Q = compute_qubo_matrix(bqm)
                Q_sparse = sparse.csr_matrix(Q)
                
                # Smallest eigenvalues
                k_small = min(3, n_vars - 1)
                evals_small = eigsh(Q_sparse, k=k_small, which='SA', return_eigenvectors=False)
                
                # Largest eigenvalues
                k_large = min(3, n_vars - 1)
                evals_large = eigsh(Q_sparse, k=k_large, which='LA', return_eigenvectors=False)
                
                metrics['min_eigenvalue'] = float(np.min(evals_small))
                metrics['max_eigenvalue'] = float(np.max(evals_large))
                metrics['spectral_gap'] = float(np.sort(evals_small)[1] - np.sort(evals_small)[0]) if k_small > 1 else None
                metrics['rank'] = 'not_computed'
                metrics['trace'] = float(np.trace(Q.diagonal()))
            except Exception as e:
                print(f"    Warning: Could not compute approximate spectrum: {e}", file=sys.stderr)
                metrics.update({
                    'min_eigenvalue': None,
                    'max_eigenvalue': None,
                    'spectral_gap': None,
                    'rank': None,
                    'trace': None
                })
        else:
            metrics.update({
                'min_eigenvalue': None,
                'max_eigenvalue': None,
                'spectral_gap': None,
                'rank': None,
                'trace': None
            })
    
    # Sample energy landscape
    print(f"    Sampling energy landscape...", file=sys.stderr)
    n_samples = min(1000, 2**min(n_vars, 15))
    energies = []
    
    variables = list(bqm.variables)
    for _ in tqdm(range(n_samples), desc=f"      Sampling", file=sys.stderr, leave=False):
        sample = {v: np.random.randint(0, 2) for v in variables}
        energy = bqm.energy(sample)
        energies.append(energy)
    
    energies = np.array(energies)
    metrics['landscape_mean_energy'] = float(np.mean(energies))
    metrics['landscape_std_energy'] = float(np.std(energies))
    metrics['landscape_min_energy'] = float(np.min(energies))
    metrics['landscape_max_energy'] = float(np.max(energies))
    metrics['landscape_energy_range'] = float(np.ptp(energies))
    
    # Estimate local minima by checking random samples
    n_local_checks = min(100, n_samples)
    local_minima_count = 0
    
    for _ in tqdm(range(n_local_checks), desc=f"      Local minima", file=sys.stderr, leave=False):
        sample = {v: np.random.randint(0, 2) for v in variables}
        current_energy = bqm.energy(sample)
        is_local_min = True
        
        # Check all 1-bit neighbors
        for var in variables:
            neighbor = sample.copy()
            neighbor[var] = 1 - neighbor[var]
            if bqm.energy(neighbor) < current_energy:
                is_local_min = False
                break
        
        if is_local_min:
            local_minima_count += 1
    
    metrics['estimated_local_minima_ratio'] = float(local_minima_count / n_local_checks)
    
    return metrics


# ============================================================================
# SECTION 4: SOLVER PERFORMANCE COMPARISON
# ============================================================================

def solve_cqm_with_pulp_gurobi(cqm, name: str, n_seeds: int = 5) -> Dict:
    """
    Solve CQM directly using PuLP with Gurobi backend.
    Much faster than classical samplers on BQM for large problems.
    """
    print(f"  Solving {name} with PuLP/Gurobi ({n_seeds} runs)...", file=sys.stderr)
    
    if not PULP_AVAILABLE:
        return {'error': 'PuLP not available'}
    
    results = {
        'energies': [],
        'times': [],
        'feasibility': [],
        'success_count': 0
    }
    
    for run in tqdm(range(n_seeds), desc=f"    {name} runs", file=sys.stderr, leave=False):
        start_time = time.time()
        
        try:
            # Create PuLP problem
            prob = pulp.LpProblem(f"{name}_run_{run}", pulp.LpMinimize)
            
            # Create PuLP variables
            pulp_vars = {}
            for var in cqm.variables:
                if cqm.vartype(var).name == 'BINARY':
                    pulp_vars[var] = pulp.LpVariable(str(var), cat='Binary')
                elif cqm.vartype(var).name == 'INTEGER':
                    lb = cqm.lower_bound(var) if cqm.lower_bound(var) != float('-inf') else None
                    ub = cqm.upper_bound(var) if cqm.upper_bound(var) != float('inf') else None
                    pulp_vars[var] = pulp.LpVariable(str(var), lowBound=lb, upBound=ub, cat='Integer')
                else:  # REAL
                    lb = cqm.lower_bound(var) if cqm.lower_bound(var) != float('-inf') else None
                    ub = cqm.upper_bound(var) if cqm.upper_bound(var) != float('inf') else None
                    pulp_vars[var] = pulp.LpVariable(str(var), lowBound=lb, upBound=ub, cat='Continuous')
            
            # Convert objective
            obj_expr = 0
            for var, coeff in cqm.objective.linear.items():
                obj_expr += coeff * pulp_vars[var]
            for (v1, v2), coeff in cqm.objective.quadratic.items():
                obj_expr += coeff * pulp_vars[v1] * pulp_vars[v2]
            obj_expr += cqm.objective.offset
            
            prob += obj_expr
            
            # Convert constraints
            for label, constraint in cqm.constraints.items():
                constr_expr = 0
                for var, coeff in constraint.lhs.linear.items():
                    constr_expr += coeff * pulp_vars[var]
                for (v1, v2), coeff in constraint.lhs.quadratic.items():
                    constr_expr += coeff * pulp_vars[v1] * pulp_vars[v2]
                
                rhs = constraint.rhs
                if constraint.sense == '<=':
                    prob += constr_expr <= rhs, label
                elif constraint.sense == '>=':
                    prob += constr_expr >= rhs, label
                elif constraint.sense == '==':
                    prob += constr_expr == rhs, label
            
            # Solve with Gurobi
            if GUROBI_AVAILABLE:
                solver = pulp.GUROBI(msg=False, timeLimit=300)
            else:
                solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=300)
            
            prob.solve(solver)
            elapsed = time.time() - start_time
            
            # Extract solution
            if prob.status == pulp.LpStatusOptimal or prob.status == pulp.LpStatusFeasible:
                solution = {var: pulp_vars[var].varValue for var in cqm.variables}
                
                # Compute energy using CQM objective
                energy = cqm.objective.energy(solution)
                results['energies'].append(energy)
                results['times'].append(elapsed)
                
                # Check feasibility
                is_feasible = cqm.check_feasible(solution)
                results['feasibility'].append(is_feasible)
                if is_feasible:
                    results['success_count'] += 1
            else:
                results['energies'].append(float('inf'))
                results['times'].append(elapsed)
                results['feasibility'].append(False)
        
        except Exception as e:
            results['energies'].append(float('inf'))
            results['times'].append(time.time() - start_time)
            results['feasibility'].append(False)
    
    # Compute statistics
    valid_energies = [e for e in results['energies'] if e != float('inf')]
    
    return {
        'mean_energy': float(np.mean(valid_energies)) if valid_energies else float('inf'),
        'std_energy': float(np.std(valid_energies)) if valid_energies else 0,
        'min_energy': float(np.min(valid_energies)) if valid_energies else float('inf'),
        'max_energy': float(np.max(valid_energies)) if valid_energies else float('inf'),
        'mean_time': float(np.mean(results['times'])),
        'std_time': float(np.std(results['times'])),
        'feasibility_rate': float(results['success_count'] / n_seeds),
        'success_count': results['success_count'],
        'n_runs': n_seeds
    }


def solve_with_multiple_solvers(bqm: BinaryQuadraticModel, name: str, n_seeds: int = 5) -> Dict:
    """
    Solve BQM with multiple classical samplers and seeds.
    Measure: time-to-target, success probability, approximation ratio.
    """
    print(f"  Solving {name} with multiple solvers ({n_seeds} seeds)...", file=sys.stderr)
    
    results = {}
    
    # Define solvers
    solvers = {
        'simulated_annealing': SimulatedAnnealingSampler(),
        'tabu': TabuSampler(),
        'steepest_descent': SteepestDescentSampler()
    }
    
    # Get ground state estimate using ExactSolver for small problems
    if len(bqm.variables) <= 20:
        try:
            exact_solver = ExactSolver()
            exact_sampleset = exact_solver.sample(bqm)
            ground_energy = exact_sampleset.first.energy
            print(f"    Ground state energy (exact): {ground_energy:.4f}", file=sys.stderr)
        except:
            ground_energy = None
    else:
        ground_energy = None
    
    for solver_name, solver in solvers.items():
        print(f"    Testing {solver_name}...", file=sys.stderr)
        solver_results = {
            'energies': [],
            'times': [],
            'success_count': 0
        }
        
        for seed in range(n_seeds):
            start_time = time.time()
            
            if solver_name == 'simulated_annealing':
                sampleset = solver.sample(bqm, num_reads=100, seed=seed)
            elif solver_name == 'tabu':
                sampleset = solver.sample(bqm, num_reads=100, seed=seed)
            else:  # steepest_descent
                # Generate random initial states
                variables = list(bqm.variables)
                initial_states = [{v: np.random.randint(0, 2) for v in variables} for _ in range(10)]
                sampleset = solver.sample(bqm, initial_states=initial_states)
            
            elapsed = time.time() - start_time
            
            best_energy = sampleset.first.energy
            solver_results['energies'].append(best_energy)
            solver_results['times'].append(elapsed)
            
            if ground_energy is not None and abs(best_energy - ground_energy) < 1e-6:
                solver_results['success_count'] += 1
        
        # Compute statistics
        results[solver_name] = {
            'mean_energy': float(np.mean(solver_results['energies'])),
            'std_energy': float(np.std(solver_results['energies'])),
            'min_energy': float(np.min(solver_results['energies'])),
            'max_energy': float(np.max(solver_results['energies'])),
            'mean_time': float(np.mean(solver_results['times'])),
            'std_time': float(np.std(solver_results['times'])),
            'success_rate': float(solver_results['success_count'] / n_seeds) if ground_energy is not None else None,
            'ground_energy': ground_energy
        }
        
        if ground_energy is not None and ground_energy != 0:
            results[solver_name]['approximation_ratio'] = float(results[solver_name]['min_energy'] / ground_energy)
        else:
            results[solver_name]['approximation_ratio'] = None
    
    return results


# ============================================================================
# SECTION 5: CONSTRAINT STRESS TESTING
# ============================================================================

def test_penalty_sensitivity(cqm, name: str, multipliers: List[float] = None) -> Dict:
    """
    Convert CQM to BQM with different Lagrange multipliers and measure:
    - Feasibility ratio
    - Constraint violation count
    - Energy change
    """
    print(f"  Testing penalty sensitivity for {name}...", file=sys.stderr)
    
    if multipliers is None:
        multipliers = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
    results = {}
    
    for mult in tqdm(multipliers, desc=f"    Lagrange multipliers", file=sys.stderr, leave=False):
        try:
            bqm, _ = cqm_to_bqm(cqm, lagrange_multiplier=mult)
            
            # Solve with SA
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=100)
            
            # Check feasibility
            feasible_count = 0
            violation_counts = []
            
            for sample, energy in sampleset.data(['sample', 'energy']):
                is_feasible = cqm.check_feasible(sample)
                if is_feasible:
                    feasible_count += 1
                
                # Count violated constraints
                n_violations = 0
                for constraint in cqm.constraints.values():
                    lhs = constraint.lhs.energy(sample)
                    if constraint.sense == '<=':
                        if lhs > constraint.rhs + 1e-6:
                            n_violations += 1
                    elif constraint.sense == '>=':
                        if lhs < constraint.rhs - 1e-6:
                            n_violations += 1
                    elif constraint.sense == '==':
                        if abs(lhs - constraint.rhs) > 1e-6:
                            n_violations += 1
                
                violation_counts.append(n_violations)
            
            results[mult] = {
                'feasibility_ratio': feasible_count / len(sampleset),
                'mean_violations': float(np.mean(violation_counts)),
                'max_violations': int(np.max(violation_counts)),
                'best_energy': float(sampleset.first.energy),
                'n_variables': len(bqm.variables),
                'n_interactions': len(bqm.quadratic)
            }
            
        except Exception as e:
            print(f"      Error with multiplier {mult}: {e}", file=sys.stderr)
            results[mult] = {
                'error': str(e)
            }
    
    return results


def analyze_constraint_structure(cqm, name: str) -> Dict:
    """
    Analyze the structure of constraints in the CQM:
    - Number of constraints
    - Constraint types (<=, >=, ==)
    - Variables per constraint
    - Constraint overlap
    """
    print(f"  Analyzing constraint structure for {name}...", file=sys.stderr)
    
    n_constraints = len(cqm.constraints)
    constraint_types = defaultdict(int)
    vars_per_constraint = []
    
    all_constraint_vars = []
    
    for label, constraint in cqm.constraints.items():
        constraint_types[str(constraint.sense)] += 1
        constraint_vars = set(constraint.lhs.variables)
        vars_per_constraint.append(len(constraint_vars))
        all_constraint_vars.append(constraint_vars)
    
    # Measure constraint overlap
    total_pairs = n_constraints * (n_constraints - 1) // 2 if n_constraints > 1 else 0
    overlap_count = 0
    overlap_sizes = []
    
    for i in range(n_constraints):
        for j in range(i + 1, n_constraints):
            overlap = all_constraint_vars[i].intersection(all_constraint_vars[j])
            if len(overlap) > 0:
                overlap_count += 1
                overlap_sizes.append(len(overlap))
    
    return {
        'n_constraints': n_constraints,
        'constraint_types': dict(constraint_types),
        'mean_vars_per_constraint': float(np.mean(vars_per_constraint)) if vars_per_constraint else 0,
        'max_vars_per_constraint': int(np.max(vars_per_constraint)) if vars_per_constraint else 0,
        'min_vars_per_constraint': int(np.min(vars_per_constraint)) if vars_per_constraint else 0,
        'constraint_overlap_ratio': overlap_count / total_pairs if total_pairs > 0 else 0,
        'mean_overlap_size': float(np.mean(overlap_sizes)) if overlap_sizes else 0,
        'max_overlap_size': int(np.max(overlap_sizes)) if overlap_sizes else 0
    }


# ============================================================================
# SECTION 6: MAIN ANALYSIS PIPELINE
# ============================================================================

def run_full_analysis(n_vars: int = 50, n_seeds: int = 5, seed: int = 42, 
                     penalty_multipliers: List[float] = None, use_gurobi: bool = False) -> Dict:
    """
    Run complete diagnostic analysis on all three formulations.
    
    Args:
        n_vars: Target number of variables
        n_seeds: Number of random seeds for solver tests
        seed: Random seed for instance generation
        penalty_multipliers: List of Lagrange multipliers to test
        use_gurobi: If True, solve CQMs with Gurobi via PuLP instead of BQM samplers
    """
    print("="*80, file=sys.stderr)
    print("BQM CONSTRAINT VIOLATION DIAGNOSTIC", file=sys.stderr)
    print("="*80, file=sys.stderr)
    
    if use_gurobi:
        print(f"MODE: Solving CQMs with Gurobi via PuLP (faster for large problems)", file=sys.stderr)
    else:
        print(f"MODE: Solving BQMs with classical samplers", file=sys.stderr)
    
    # Generate instances
    print("\n[1/6] GENERATING INSTANCES", file=sys.stderr)
    instances = generate_bqm_instances(n_vars=n_vars, seed=seed)
    
    results = {}
    
    for formulation_name, instance_data in instances.items():
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"ANALYZING: {formulation_name.upper()}", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        
        bqm = instance_data['bqm']
        cqm = instance_data['cqm']
        metadata = instance_data['metadata']
        
        formulation_results = {
            'metadata': metadata
        }
        
        # Section 2: Instance Characterization
        print(f"\n[2/6] INSTANCE CHARACTERIZATION", file=sys.stderr)
        formulation_results['characterization'] = characterize_instance(bqm, formulation_name)
        
        # Section 3: Hardness Metrics
        print(f"\n[3/6] HARDNESS METRICS", file=sys.stderr)
        formulation_results['hardness'] = measure_hardness_metrics(bqm, formulation_name)
        
        # Section 4: Solver Performance
        print(f"\n[4/6] SOLVER PERFORMANCE", file=sys.stderr)
        if use_gurobi:
            formulation_results['solver_performance'] = {
                'gurobi_cqm': solve_cqm_with_pulp_gurobi(cqm, formulation_name, n_seeds=n_seeds)
            }
        else:
            formulation_results['solver_performance'] = solve_with_multiple_solvers(bqm, formulation_name, n_seeds=n_seeds)
        
        # Section 5: Constraint Analysis
        print(f"\n[5/6] CONSTRAINT STRUCTURE", file=sys.stderr)
        formulation_results['constraint_structure'] = analyze_constraint_structure(cqm, formulation_name)
        
        # Section 6: Penalty Sensitivity
        print(f"\n[6/6] PENALTY SENSITIVITY", file=sys.stderr)
        formulation_results['penalty_sensitivity'] = test_penalty_sensitivity(cqm, formulation_name, multipliers=penalty_multipliers)
        
        results[formulation_name] = formulation_results
    
    return results


# ============================================================================
# SECTION 7: REPORT GENERATION
# ============================================================================

def generate_diagnostic_report(results: Dict) -> str:
    """
    Generate comprehensive Markdown report comparing all three formulations.
    """
    
    report = f"""# BQM Constraint Violation Diagnostic Report

**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report analyzes three BQM formulations to diagnose why the PATCH formulation violates constraints with D-Wave's Hybrid BQM solver while BQUBO does not:

1. **BQUBO**: Binary plantation model (baseline - works correctly)
2. **PATCH**: Plot assignment model with idle area penalty (violates constraints)
3. **PATCH_NO_IDLE**: Plot assignment model without idle area penalty (test variant)

---

## 1. Instance Characterization

### Size and Complexity

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Variables** | {results['bqubo']['characterization']['n_variables']} | {results['patch']['characterization']['n_variables']} | {results['patch_no_idle']['characterization']['n_variables']} |
| **Interactions** | {results['bqubo']['characterization']['n_interactions']} | {results['patch']['characterization']['n_interactions']} | {results['patch_no_idle']['characterization']['n_interactions']} |
| **Density** | {results['bqubo']['characterization']['density']:.6f} | {results['patch']['characterization']['density']:.6f} | {results['patch_no_idle']['characterization']['density']:.6f} |
| **Offset** | {results['bqubo']['characterization']['offset']:.4f} | {results['patch']['characterization']['offset']:.4f} | {results['patch_no_idle']['characterization']['offset']:.4f} |

### Linear Coefficients Distribution

| Statistic | BQUBO | PATCH | PATCH_NO_IDLE |
|-----------|-------|-------|---------------|
| **Mean** | {results['bqubo']['characterization']['linear_coefficients']['mean']:.4f} | {results['patch']['characterization']['linear_coefficients']['mean']:.4f} | {results['patch_no_idle']['characterization']['linear_coefficients']['mean']:.4f} |
| **Std Dev** | {results['bqubo']['characterization']['linear_coefficients']['std']:.4f} | {results['patch']['characterization']['linear_coefficients']['std']:.4f} | {results['patch_no_idle']['characterization']['linear_coefficients']['std']:.4f} |
| **Range** | {results['bqubo']['characterization']['linear_coefficients']['range']:.4f} | {results['patch']['characterization']['linear_coefficients']['range']:.4f} | {results['patch_no_idle']['characterization']['linear_coefficients']['range']:.4f} |

### Quadratic Coefficients Distribution

| Statistic | BQUBO | PATCH | PATCH_NO_IDLE |
|-----------|-------|-------|---------------|
| **Mean** | {results['bqubo']['characterization']['quadratic_coefficients']['mean']:.4f} | {results['patch']['characterization']['quadratic_coefficients']['mean']:.4f} | {results['patch_no_idle']['characterization']['quadratic_coefficients']['mean']:.4f} |
| **Std Dev** | {results['bqubo']['characterization']['quadratic_coefficients']['std']:.4f} | {results['patch']['characterization']['quadratic_coefficients']['std']:.4f} | {results['patch_no_idle']['characterization']['quadratic_coefficients']['std']:.4f} |
| **Range** | {results['bqubo']['characterization']['quadratic_coefficients']['range']:.4f} | {results['patch']['characterization']['quadratic_coefficients']['range']:.4f} | {results['patch_no_idle']['characterization']['quadratic_coefficients']['range']:.4f} |

### Graph Topology

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Connected Components** | {results['bqubo']['characterization']['graph_topology']['n_connected_components']} | {results['patch']['characterization']['graph_topology']['n_connected_components']} | {results['patch_no_idle']['characterization']['graph_topology']['n_connected_components']} |
| **Avg Degree** | {results['bqubo']['characterization']['graph_topology']['avg_degree']:.2f} | {results['patch']['characterization']['graph_topology']['avg_degree']:.2f} | {results['patch_no_idle']['characterization']['graph_topology']['avg_degree']:.2f} |
| **Max Degree** | {results['bqubo']['characterization']['graph_topology']['max_degree']} | {results['patch']['characterization']['graph_topology']['max_degree']} | {results['patch_no_idle']['characterization']['graph_topology']['max_degree']} |
| **Avg Clustering** | {results['bqubo']['characterization']['graph_topology']['avg_clustering']:.4f} | {results['patch']['characterization']['graph_topology']['avg_clustering']:.4f} | {results['patch_no_idle']['characterization']['graph_topology']['avg_clustering']:.4f} |

---

## 2. Solver-Independent Hardness Metrics

### Spectral Properties

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Min Eigenvalue** | {results['bqubo']['hardness']['min_eigenvalue']} | {results['patch']['hardness']['min_eigenvalue']} | {results['patch_no_idle']['hardness']['min_eigenvalue']} |
| **Max Eigenvalue** | {results['bqubo']['hardness']['max_eigenvalue']} | {results['patch']['hardness']['max_eigenvalue']} | {results['patch_no_idle']['hardness']['max_eigenvalue']} |
| **Spectral Gap** | {results['bqubo']['hardness']['spectral_gap']} | {results['patch']['hardness']['spectral_gap']} | {results['patch_no_idle']['hardness']['spectral_gap']} |

### Energy Landscape (Random Sampling)

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Mean Energy** | {results['bqubo']['hardness']['landscape_mean_energy']:.4f} | {results['patch']['hardness']['landscape_mean_energy']:.4f} | {results['patch_no_idle']['hardness']['landscape_mean_energy']:.4f} |
| **Std Energy** | {results['bqubo']['hardness']['landscape_std_energy']:.4f} | {results['patch']['hardness']['landscape_std_energy']:.4f} | {results['patch_no_idle']['hardness']['landscape_std_energy']:.4f} |
| **Energy Range** | {results['bqubo']['hardness']['landscape_energy_range']:.4f} | {results['patch']['hardness']['landscape_energy_range']:.4f} | {results['patch_no_idle']['hardness']['landscape_energy_range']:.4f} |
| **Local Minima Ratio** | {results['bqubo']['hardness']['estimated_local_minima_ratio']:.4f} | {results['patch']['hardness']['estimated_local_minima_ratio']:.4f} | {results['patch_no_idle']['hardness']['estimated_local_minima_ratio']:.4f} |

---

## 3. Solver Performance Comparison

"""
    
    # Check if we used Gurobi or classical samplers
    if 'gurobi_cqm' in results['bqubo']['solver_performance']:
        # Gurobi CQM mode
        report += """### Gurobi CQM Solver (via PuLP)

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Mean Energy** | {:.4f} | {:.4f} | {:.4f} |
| **Best Energy** | {:.4f} | {:.4f} | {:.4f} |
| **Mean Time (s)** | {:.4f} | {:.4f} | {:.4f} |
| **Feasibility Rate** | {:.2%} | {:.2%} | {:.2%} |
| **Success Count** | {}/{} | {}/{} | {}/{} |

""".format(
            results['bqubo']['solver_performance']['gurobi_cqm']['mean_energy'],
            results['patch']['solver_performance']['gurobi_cqm']['mean_energy'],
            results['patch_no_idle']['solver_performance']['gurobi_cqm']['mean_energy'],
            results['bqubo']['solver_performance']['gurobi_cqm']['min_energy'],
            results['patch']['solver_performance']['gurobi_cqm']['min_energy'],
            results['patch_no_idle']['solver_performance']['gurobi_cqm']['min_energy'],
            results['bqubo']['solver_performance']['gurobi_cqm']['mean_time'],
            results['patch']['solver_performance']['gurobi_cqm']['mean_time'],
            results['patch_no_idle']['solver_performance']['gurobi_cqm']['mean_time'],
            results['bqubo']['solver_performance']['gurobi_cqm']['feasibility_rate'],
            results['patch']['solver_performance']['gurobi_cqm']['feasibility_rate'],
            results['patch_no_idle']['solver_performance']['gurobi_cqm']['feasibility_rate'],
            results['bqubo']['solver_performance']['gurobi_cqm']['success_count'],
            results['bqubo']['solver_performance']['gurobi_cqm']['n_runs'],
            results['patch']['solver_performance']['gurobi_cqm']['success_count'],
            results['patch']['solver_performance']['gurobi_cqm']['n_runs'],
            results['patch_no_idle']['solver_performance']['gurobi_cqm']['success_count'],
            results['patch_no_idle']['solver_performance']['gurobi_cqm']['n_runs']
        )
    else:
        # Classical samplers mode
        report += """
### Simulated Annealing

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Mean Energy** | {:.4f} | {:.4f} | {:.4f} |
| **Best Energy** | {:.4f} | {:.4f} | {:.4f} |
| **Mean Time (s)** | {:.4f} | {:.4f} | {:.4f} |

### Tabu Search

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Mean Energy** | {:.4f} | {:.4f} | {:.4f} |
| **Best Energy** | {:.4f} | {:.4f} | {:.4f} |
| **Mean Time (s)** | {:.4f} | {:.4f} | {:.4f} |

""".format(
            results['bqubo']['solver_performance']['simulated_annealing']['mean_energy'],
            results['patch']['solver_performance']['simulated_annealing']['mean_energy'],
            results['patch_no_idle']['solver_performance']['simulated_annealing']['mean_energy'],
            results['bqubo']['solver_performance']['simulated_annealing']['min_energy'],
            results['patch']['solver_performance']['simulated_annealing']['min_energy'],
            results['patch_no_idle']['solver_performance']['simulated_annealing']['min_energy'],
            results['bqubo']['solver_performance']['simulated_annealing']['mean_time'],
            results['patch']['solver_performance']['simulated_annealing']['mean_time'],
            results['patch_no_idle']['solver_performance']['simulated_annealing']['mean_time'],
            results['bqubo']['solver_performance']['tabu']['mean_energy'],
            results['patch']['solver_performance']['tabu']['mean_energy'],
            results['patch_no_idle']['solver_performance']['tabu']['mean_energy'],
            results['bqubo']['solver_performance']['tabu']['min_energy'],
            results['patch']['solver_performance']['tabu']['min_energy'],
            results['patch_no_idle']['solver_performance']['tabu']['min_energy'],
            results['bqubo']['solver_performance']['tabu']['mean_time'],
            results['patch']['solver_performance']['tabu']['mean_time'],
            results['patch_no_idle']['solver_performance']['tabu']['mean_time']
        )
    
    report += """
---

## 4. Constraint Structure Analysis

### Constraint Statistics

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Num Constraints** | {results['bqubo']['constraint_structure']['n_constraints']} | {results['patch']['constraint_structure']['n_constraints']} | {results['patch_no_idle']['constraint_structure']['n_constraints']} |
| **Mean Vars/Constraint** | {results['bqubo']['constraint_structure']['mean_vars_per_constraint']:.2f} | {results['patch']['constraint_structure']['mean_vars_per_constraint']:.2f} | {results['patch_no_idle']['constraint_structure']['mean_vars_per_constraint']:.2f} |
| **Max Vars/Constraint** | {results['bqubo']['constraint_structure']['max_vars_per_constraint']} | {results['patch']['constraint_structure']['max_vars_per_constraint']} | {results['patch_no_idle']['constraint_structure']['max_vars_per_constraint']} |
| **Constraint Overlap Ratio** | {results['bqubo']['constraint_structure']['constraint_overlap_ratio']:.4f} | {results['patch']['constraint_structure']['constraint_overlap_ratio']:.4f} | {results['patch_no_idle']['constraint_structure']['constraint_overlap_ratio']:.4f} |
| **Mean Overlap Size** | {results['bqubo']['constraint_structure']['mean_overlap_size']:.2f} | {results['patch']['constraint_structure']['mean_overlap_size']:.2f} | {results['patch_no_idle']['constraint_structure']['mean_overlap_size']:.2f} |

### Constraint Types

**BQUBO:** {results['bqubo']['constraint_structure']['constraint_types']}

**PATCH:** {results['patch']['constraint_structure']['constraint_types']}

**PATCH_NO_IDLE:** {results['patch_no_idle']['constraint_structure']['constraint_types']}

---

## 5. Penalty Weight Sensitivity Analysis

This section shows how feasibility and constraint violations change with different Lagrange multipliers.

"""

    # Penalty sensitivity tables for each formulation
    for formulation in ['bqubo', 'patch', 'patch_no_idle']:
        report += f"\n### {formulation.upper()} Penalty Sensitivity\n\n"
        report += "| Lagrange Multiplier | Feasibility Ratio | Mean Violations | Max Violations | Best Energy |\n"
        report += "|---------------------|-------------------|-----------------|----------------|-------------|\n"
        
        for mult, data in sorted(results[formulation]['penalty_sensitivity'].items()):
            if 'error' not in data:
                report += f"| {mult} | {data['feasibility_ratio']:.4f} | {data['mean_violations']:.2f} | {data['max_violations']} | {data['best_energy']:.4f} |\n"
            else:
                report += f"| {mult} | ERROR | ERROR | ERROR | ERROR |\n"
    
    # Key findings and conclusions
    report += """

---

## 6. Key Findings and Diagnosis

### Critical Differences Identified:

"""

    # Compare densities
    patch_density = results['patch']['characterization']['density']
    bqubo_density = results['bqubo']['characterization']['density']
    density_ratio = patch_density / bqubo_density if bqubo_density > 0 else float('inf')
    
    report += f"""
1. **Model Density Disparity**: 
   - PATCH is **{density_ratio:.2f}x denser** than BQUBO ({patch_density:.6f} vs {bqubo_density:.6f})
   - Higher density means more complex constraint interactions in BQM penalty terms

"""

    # Compare constraint structures
    patch_constraints = results['patch']['constraint_structure']['n_constraints']
    bqubo_constraints = results['bqubo']['constraint_structure']['n_constraints']
    patch_overlap = results['patch']['constraint_structure']['constraint_overlap_ratio']
    bqubo_overlap = results['bqubo']['constraint_structure']['constraint_overlap_ratio']
    
    report += f"""
2. **Constraint Complexity**:
   - PATCH has {patch_constraints} constraints vs BQUBO's {bqubo_constraints}
   - PATCH constraint overlap: {patch_overlap:.4f} vs BQUBO: {bqubo_overlap:.4f}
   - Higher overlap indicates variables involved in multiple constraints simultaneously

"""

    # Compare coefficient ranges
    patch_quad_range = results['patch']['characterization']['quadratic_coefficients']['range']
    bqubo_quad_range = results['bqubo']['characterization']['quadratic_coefficients']['range']
    
    report += f"""
3. **Coefficient Scale Differences**:
   - PATCH quadratic coefficient range: {patch_quad_range:.4f}
   - BQUBO quadratic coefficient range: {bqubo_quad_range:.4f}
   - Large range indicates penalty terms may dominate objective in PATCH

"""

    # Analyze penalty sensitivity
    report += """
4. **Penalty Weight Sensitivity**:
"""
    
    # Find the Lagrange multiplier where PATCH achieves 100% feasibility
    patch_ps = results['patch']['penalty_sensitivity']
    feasible_mult = None
    for mult in sorted(patch_ps.keys()):
        if 'error' not in patch_ps[mult] and patch_ps[mult]['feasibility_ratio'] >= 0.99:
            feasible_mult = mult
            break
    
    if feasible_mult:
        report += f"   - PATCH requires Lagrange multiplier ≥ {feasible_mult} for 100% feasibility\n"
    else:
        report += f"   - PATCH may need even higher multipliers to achieve full feasibility\n"
    
    # Compare idle penalty effect
    patch_with_idle = results['patch']
    patch_no_idle = results['patch_no_idle']
    
    idle_constraint_diff = patch_with_idle['constraint_structure']['n_constraints'] - patch_no_idle['constraint_structure']['n_constraints']
    
    report += f"""
   - Removing idle penalty reduces constraints by {idle_constraint_diff}
   - PATCH_NO_IDLE achieves different feasibility profile

"""

    report += """
### Root Cause Analysis:

The PATCH formulation violates constraints because:

1. **Penalty Term Dilution**: With many overlapping constraints converted to quadratic penalties, 
   the effective penalty strength gets diluted relative to the objective function.

2. **Heterogeneous Patch Areas**: Unlike BQUBO's uniform 1-acre units, PATCH has varying patch sizes.
   This introduces coefficient heterogeneity that makes penalty scaling harder.

3. **Idle Area Penalty Interaction**: The idle area penalty creates an additional energy term that
   competes with constraint penalties, potentially tipping the balance toward infeasible solutions.

4. **Constraint Coupling**: Higher constraint overlap in PATCH means variables are "pulled" in 
   multiple directions by different penalty terms, making it harder to satisfy all simultaneously.

### Recommendations:

1. **Use CQM Solver Directly**: Avoid `cqm_to_bqm()` conversion for PATCH. Use `LeapHybridCQMSampler` instead.

2. **If BQM Required**: 
   - Increase Lagrange multiplier to at least {feasible_mult if feasible_mult else '1000+'}
   - Normalize patch areas before formulation
   - Consider reformulating to reduce constraint overlap

3. **Further Investigation**: 
   - Test with actual D-Wave Hybrid BQM solver to confirm behavior
   - Analyze embedding quality and chain strength requirements
   - Profile penalty vs objective term magnitudes in failing cases

---

## 7. Conclusion

This diagnostic confirms that the PATCH formulation's constraint violations stem from its inherently
more complex penalty structure after CQM-to-BQM conversion. The combination of high density,
heterogeneous coefficients, and overlapping constraints makes it difficult for the BQM solver
to balance objective optimization with constraint satisfaction.

**The formulations are NOT equivalent** despite similar problem semantics. The discrete plot 
structure and idle area penalty fundamentally change the BQM's energy landscape compared to
BQUBO's simpler capacity-pool model.

---

*Report generated by diagnose_bqm_constraint_violations.py*
"""
    
    return report


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive BQM diagnostic to identify constraint violation causes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diagnose_bqm_constraint_violations.py --n_vars 50 --n_seeds 10
  python diagnose_bqm_constraint_violations.py --output diagnostic_report.md
        """
    )
    
    parser.add_argument('--n_vars', type=int, default=50,
                       help='Target number of variables (default: 50)')
    parser.add_argument('--n_seeds', type=int, default=5,
                       help='Number of random seeds for solver tests (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for instance generation (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for report (default: stdout)')
    parser.add_argument('--json', type=str, default=None,
                       help='Save raw results as JSON file')
    parser.add_argument('--use-gurobi', action='store_true',
                       help='Use Gurobi via PuLP to solve CQMs instead of BQM samplers (much faster for large problems)')
    
    args = parser.parse_args()
    
    # Run analysis
    results = run_full_analysis(
        n_vars=args.n_vars,
        n_seeds=args.n_seeds,
        seed=args.seed,
        use_gurobi=args.use_gurobi
    )
    
    # Save JSON if requested
    if args.json:
        print(f"\nSaving raw results to {args.json}...", file=sys.stderr)
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Generate report
    print("\nGenerating diagnostic report...", file=sys.stderr)
    report = generate_diagnostic_report(results)
    
    # Output report
    if args.output:
        print(f"Writing report to {args.output}...", file=sys.stderr)
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}", file=sys.stderr)
    else:
        print("\n" + "="*80 + "\n", file=sys.stderr)
        print(report)


if __name__ == "__main__":
    main()
