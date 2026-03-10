#!/usr/bin/env python3
"""
Benchmark: Whole-Problem vs Decomposed Solving

Compares three solving approaches on each problem variant:
  1. Gurobi solving the full problem
  2. Gurobi solving each sub-problem independently, then merging
  3. Parallel Tempering with Isoenergetic Cluster Moves (PT-ICM)
     solving each sub-problem independently, then merging

For Variant A (Study 2.A): Binary Patch allocation, 27 crops
  - Decompositions: PlotBased, Multilevel(5), HybridGrid(5,9)

For Variant B (Study 2.B): Multi-period rotation, 6 families × 3 periods
  - Decompositions: Clique (farm-by-farm), SpatialTemporal(5)

Output: JSON results.
"""

import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Utils.patch_sampler import generate_grid
from src.scenarios import load_food_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
LOG = logging.getLogger(__name__)

import gurobipy as gp
from gurobipy import GRB
import dimod
from dimod import BinaryQuadraticModel

# ============================================================================
# Common utilities
# ============================================================================

# Load 27-crop data — must match QPU benchmark (full_family scenario)
try:
    _f0, _foods_dict, _fg0, _cfg0 = load_food_data("full_family")
    FOOD_NAMES_27 = list(_foods_dict.keys()) if isinstance(_foods_dict, dict) else list(_foods_dict)
    # Use the same weights as qpu_benchmark.py / load_problem_data()
    _WEIGHTS = _cfg0.get('parameters', {}).get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.20,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15,
    })
    FOOD_BENEFITS = {}
    FOOD_GROUPS: Dict[str, List[str]] = {}
    if isinstance(_foods_dict, dict):
        for cname, attrs in _foods_dict.items():
            FOOD_BENEFITS[cname] = (
                _WEIGHTS.get('nutritional_value', 0.25) * attrs.get("nutritional_value", 0.5)
                + _WEIGHTS.get('nutrient_density', 0.20) * attrs.get("nutrient_density", 0.5)
                - _WEIGHTS.get('environmental_impact', 0.25) * attrs.get("environmental_impact", 0.5)
                + _WEIGHTS.get('affordability', 0.15) * attrs.get("affordability", 0.5)
                + _WEIGHTS.get('sustainability', 0.15) * attrs.get("sustainability", 0.5)
            )
        FOOD_GROUPS = _fg0
except Exception:
    FOOD_NAMES_27 = [f"Crop_{i}" for i in range(27)]
    FOOD_BENEFITS = {c: np.random.default_rng(42).random() for c in FOOD_NAMES_27}
    FOOD_GROUPS = {"GroupA": FOOD_NAMES_27[:9], "GroupB": FOOD_NAMES_27[9:18],
                   "GroupC": FOOD_NAMES_27[18:27]}


FAMILIES_6 = ["Fruits", "Grains", "Legumes", "Leafy_Vegetables",
              "Root_Vegetables", "Proteins"]
N_PERIODS = 3

# ============================================================================
# Variant A — build CQM / Gurobi model
# ============================================================================

def build_variant_a_gurobi(farm_names: List[str], food_names: List[str],
                           land: Dict[str, float]) -> Tuple[gp.Model, dict]:
    """Build the Binary Patch Gurobi model (Variant A full)."""
    m = gp.Model("VariantA_full")
    m.setParam("OutputFlag", 0)

    total_area = sum(land.values())
    Y = {}
    for f in farm_names:
        for c in food_names:
            Y[f, c] = m.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}")

    U = {}
    for c in food_names:
        U[c] = m.addVar(vtype=GRB.BINARY, name=f"U_{c}")

    # Objective
    obj = gp.LinExpr()
    for f in farm_names:
        area_f = land.get(f, 1.0)
        for c in food_names:
            benefit = FOOD_BENEFITS.get(c, 0.5)
            obj += benefit * area_f / total_area * Y[f, c]
    m.setObjective(obj, GRB.MAXIMIZE)

    # One crop per farm
    for f in farm_names:
        m.addConstr(gp.quicksum(Y[f, c] for c in food_names) <= 1,
                    name=f"one_crop_{f}")

    # U-Y linking
    for f in farm_names:
        for c in food_names:
            m.addConstr(U[c] >= Y[f, c], name=f"link_{f}_{c}")

    # Food group diversity (min_foods=1, matching QPU benchmark)
    for gname, gcrops in FOOD_GROUPS.items():
        valid = [c for c in gcrops if c in food_names]
        if valid:
            m.addConstr(gp.quicksum(U[c] for c in valid) >= 1,
                        name=f"fg_min_{gname}")

    m.update()
    return m, {"Y": Y, "U": U, "food_names": food_names, "farm_names": farm_names}


def build_variant_a_sub_gurobi(sub_farms: List[str], food_names: List[str],
                                land: Dict[str, float]) -> Tuple[gp.Model, dict]:
    """Build a subproblem Gurobi model for a subset of farms (no food-group constraints)."""
    m = gp.Model("VariantA_sub")
    m.setParam("OutputFlag", 0)
    total_area = sum(land.values())  # global total for consistent normalization

    Y = {}
    for f in sub_farms:
        for c in food_names:
            Y[f, c] = m.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}")

    obj = gp.LinExpr()
    for f in sub_farms:
        area_f = land.get(f, 1.0)
        for c in food_names:
            benefit = FOOD_BENEFITS.get(c, 0.5)
            obj += benefit * area_f / total_area * Y[f, c]
    m.setObjective(obj, GRB.MAXIMIZE)

    for f in sub_farms:
        m.addConstr(gp.quicksum(Y[f, c] for c in food_names) <= 1,
                    name=f"one_crop_{f}")
    m.update()
    return m, {"Y": Y, "food_names": food_names, "farm_names": sub_farms}


# ============================================================================
# Variant B — build rotation BQM (QUBO)
# ============================================================================

def _build_rotation_matrix(seed: int = 42) -> np.ndarray:
    """6×6 family rotation synergy matrix (frustrated)."""
    rng = np.random.default_rng(seed)
    n = len(FAMILIES_6)
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                R[i, j] = -0.3  # Self-rotation penalty
            elif rng.random() < 0.7:
                R[i, j] = rng.uniform(-0.8, -0.1)
            else:
                R[i, j] = rng.uniform(0.1, 0.5)
    return (R + R.T) / 2


ROT_MATRIX = _build_rotation_matrix()


def build_variant_b_bqm(farm_names: List[str], land: Dict[str, float],
                         rotation_gamma: float = 0.5,
                         one_hot_penalty: float = 3.0,
                         diversity_bonus: float = 0.15) -> BinaryQuadraticModel:
    """Build full Variant-B rotation BQM."""
    total_area = sum(land.values())
    bqm = BinaryQuadraticModel(vartype="BINARY")
    families = FAMILIES_6
    R = ROT_MATRIX

    for f in farm_names:
        area_frac = land.get(f, 1.0) / total_area
        # Family benefit scores
        benefits = {"Fruits": 0.65, "Grains": 0.72, "Legumes": 0.80,
                    "Leafy_Vegetables": 0.70, "Root_Vegetables": 0.60,
                    "Proteins": 0.68}
        for ci, fam in enumerate(families):
            for t in range(1, N_PERIODS + 1):
                vn = f"Y_{f}_{fam}_t{t}"
                linear = -benefits.get(fam, 0.5) * area_frac
                linear -= diversity_bonus / N_PERIODS
                bqm.add_variable(vn, linear)

        # Rotation synergies (temporal)
        for t in range(2, N_PERIODS + 1):
            for ci, f1 in enumerate(families):
                for cj, f2 in enumerate(families):
                    synergy = R[ci, cj]
                    v1 = f"Y_{f}_{f1}_t{t-1}"
                    v2 = f"Y_{f}_{f2}_t{t}"
                    bqm.add_quadratic(v1, v2,
                                      -rotation_gamma * synergy * area_frac)

        # One-hot penalty per period
        for t in range(1, N_PERIODS + 1):
            vt = [f"Y_{f}_{fam}_t{t}" for fam in families]
            for i in range(len(vt)):
                for j in range(i + 1, len(vt)):
                    bqm.add_quadratic(vt[i], vt[j], 2 * one_hot_penalty)
                bqm.add_linear(vt[i], -one_hot_penalty)

    # Spatial synergies (consecutive farms)
    for idx in range(len(farm_names) - 1):
        f1, f2 = farm_names[idx], farm_names[idx + 1]
        for t in range(1, N_PERIODS + 1):
            for ci, fam1 in enumerate(families):
                for cj, fam2 in enumerate(families):
                    syn = R[ci, cj] * 0.3
                    v1 = f"Y_{f1}_{fam1}_t{t}"
                    v2 = f"Y_{f2}_{fam2}_t{t}"
                    bqm.add_quadratic(v1, v2, -rotation_gamma * 0.5 * syn)

    return bqm


def build_variant_b_sub_bqm(sub_farms: List[str], land: Dict[str, float],
                              **kwargs) -> BinaryQuadraticModel:
    """Build a rotation BQM for a farm sub-cluster."""
    return build_variant_b_bqm(sub_farms, land, **kwargs)


def build_variant_b_gurobi(bqm: BinaryQuadraticModel) -> Tuple[gp.Model, dict]:
    """Convert a BQM to a Gurobi QUBO model."""
    m = gp.Model("VariantB_QUBO")
    m.setParam("OutputFlag", 0)

    x = {}
    for v in bqm.variables:
        x[v] = m.addVar(vtype=GRB.BINARY, name=str(v))

    obj = gp.QuadExpr()
    for v in bqm.variables:
        obj += bqm.get_linear(v) * x[v]
    for (u, v), bias in bqm.quadratic.items():
        obj += bias * x[u] * x[v]
    obj += bqm.offset

    m.setObjective(obj, GRB.MINIMIZE)
    m.update()
    return m, {"x": x}


# ============================================================================
# Parallel Tempering with Isoenergetic Cluster Moves (PT-ICM)
# ============================================================================

class ParallelTemperingICM:
    """
    Parallel Tempering with Isoenergetic Cluster Moves for Ising / QUBO.

    Based on Zhu, Ochoa, Katzgraber (PRL 115, 077201, 2015).
    Algorithm:
      1. Maintain *n_replicas* spin configurations at different temperatures.
      2. Each sweep: single-spin Metropolis updates on every replica.
      3. After each sweep: attempt replica exchange between adjacent temperatures.
      4. Periodically: perform isoenergetic cluster moves (ICM) between
         pairs of replicas using Wolff-style cluster identification.
    """

    def __init__(self, n_replicas: int = 8, sweeps: int = 1000,
                 icm_interval: int = 10,
                 beta_min: float = 0.1, beta_max: float = 5.0,
                 seed: int = 42):
        self.n_replicas = n_replicas
        self.sweeps = sweeps
        self.icm_interval = icm_interval
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.rng = np.random.default_rng(seed)

    # --- internal helpers ---

    def _bqm_to_arrays(self, bqm: BinaryQuadraticModel):
        """Convert BQM to dense numpy arrays for fast evaluation."""
        variables = sorted(bqm.variables)
        idx = {v: i for i, v in enumerate(variables)}
        n = len(variables)
        h = np.zeros(n)
        J = np.zeros((n, n))
        for v in variables:
            h[idx[v]] = bqm.get_linear(v)
        for (u, v), bias in bqm.quadratic.items():
            i, j = idx[u], idx[v]
            J[i, j] = bias
            J[j, i] = bias
        return variables, h, J

    def _energy(self, state: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
        return float(h @ state + 0.5 * state @ J @ state)

    def _delta_flip(self, state: np.ndarray, h: np.ndarray,
                    J: np.ndarray, i: int) -> float:
        """Energy change from flipping spin i (binary 0/1)."""
        new_val = 1 - state[i]
        diff = new_val - state[i]  # +1 or -1
        delta = diff * h[i] + diff * (J[i] @ state) - diff * J[i, i] * state[i]
        # Correct for self-interaction
        delta += 0.5 * J[i, i] * (new_val**2 - state[i]**2)
        return float(delta)

    def _metropolis_sweep(self, state: np.ndarray, h: np.ndarray,
                          J: np.ndarray, beta: float) -> float:
        """Single Metropolis sweep over all spins."""
        n = len(state)
        order = self.rng.permutation(n)
        for i in order:
            de = self._delta_flip(state, h, J, int(i))
            if de < 0 or self.rng.random() < np.exp(-beta * de):
                state[i] = 1 - state[i]
        return self._energy(state, h, J)

    def _replica_exchange(self, states, energies, betas):
        """Attempt pairwise replica exchanges between adjacent temperatures."""
        n = len(betas)
        for k in range(n - 1):
            db = (betas[k + 1] - betas[k]) * (energies[k] - energies[k + 1])
            if db >= 0 or self.rng.random() < np.exp(db):
                states[k], states[k + 1] = states[k + 1].copy(), states[k].copy()
                energies[k], energies[k + 1] = energies[k + 1], energies[k]

    def _icm(self, states, energies, h, J, betas):
        """
        Isoenergetic Cluster Move between random replica pair.

        Pick two replicas (i, j). Build a Wolff cluster in replica i
        using the difference state. Replace the cluster spins in
        replica i with replica j's values (and vice-versa), accepting
        only if the move is exactly isoenergetic (ΔE = 0) or within
        tolerance.
        """
        n_rep = len(states)
        if n_rep < 2:
            return
        pair = self.rng.choice(n_rep, size=2, replace=False)
        ri, rj = int(pair[0]), int(pair[1])
        s_i, s_j = states[ri], states[rj]
        n = len(s_i)
        diff_mask = (s_i != s_j)
        diff_indices = np.where(diff_mask)[0]
        if len(diff_indices) == 0:
            return

        # Build Wolff cluster on the "difference graph"
        # Cluster starts from a random differing site
        start = int(self.rng.choice(diff_indices))
        cluster = set()
        stack = [start]
        while stack:
            site = stack.pop()
            if site in cluster:
                continue
            cluster.add(site)
            for nb in range(n):
                if nb in cluster or nb == site:
                    continue
                if not diff_mask[nb]:
                    continue
                # Bond probability based on interaction
                bond_energy = abs(J[site, nb])
                if bond_energy > 0:
                    p_bond = 1.0 - np.exp(-2.0 * betas[ri] * bond_energy)
                    if self.rng.random() < max(0, p_bond):
                        stack.append(nb)

        if not cluster:
            return

        # Swap cluster spins between replicas
        cluster_list = list(cluster)
        new_si = s_i.copy()
        new_sj = s_j.copy()
        for site in cluster_list:
            new_si[site] = s_j[site]
            new_sj[site] = s_i[site]

        # Accept if energy change is small (approximate isoenergetic)
        e_new_i = self._energy(new_si, h, J)
        e_new_j = self._energy(new_sj, h, J)
        de = (e_new_i + e_new_j) - (energies[ri] + energies[rj])
        # Accept with Metropolis criterion on combined energy
        avg_beta = 0.5 * (betas[ri] + betas[rj])
        if de <= 0 or self.rng.random() < np.exp(-avg_beta * de):
            states[ri] = new_si
            states[rj] = new_sj
            energies[ri] = e_new_i
            energies[rj] = e_new_j

    def solve(self, bqm: BinaryQuadraticModel,
              num_reads: int = 1) -> Tuple[Dict[str, int], float, float]:
        """
        Solve BQM using PT-ICM.

        Returns: (best_sample, best_energy, wall_time_s)
        """
        variables, h, J = self._bqm_to_arrays(bqm)
        n = len(variables)

        betas = np.geomspace(self.beta_min, self.beta_max, self.n_replicas)

        best_energy = float("inf")
        best_state = None

        for _read in range(num_reads):
            # Initialize replicas randomly
            states = [self.rng.integers(0, 2, size=n).astype(float)
                       for _ in range(self.n_replicas)]
            energies = [self._energy(s, h, J) for s in states]

            t0 = time.perf_counter()
            for sweep in range(self.sweeps):
                # Metropolis sweep on each replica
                for r in range(self.n_replicas):
                    energies[r] = self._metropolis_sweep(
                        states[r], h, J, betas[r])

                # Replica exchange
                self._replica_exchange(states, energies, betas)

                # ICM
                if sweep > 0 and sweep % self.icm_interval == 0:
                    self._icm(states, energies, h, J, betas)

            wall = time.perf_counter() - t0

            # Track best
            for r in range(self.n_replicas):
                if energies[r] < best_energy:
                    best_energy = energies[r]
                    best_state = states[r].copy()

        sample = {v: int(best_state[i]) for i, v in enumerate(variables)}
        return sample, best_energy + bqm.offset, wall


# ============================================================================
# Partitioning (same as decomposition_scaling but imported here inline)
# ============================================================================

def _partition_plot_based_A(farm_names, food_names):
    return [{f"Y_{f}_{c}" for c in food_names} for f in farm_names] + \
           [{f"U_{c}" for c in food_names}]

def _partition_multilevel_A(farm_names, food_names, gs=5):
    parts = []
    for i in range(0, len(farm_names), gs):
        g = farm_names[i:i + gs]
        parts.append({f"Y_{f}_{c}" for f in g for c in food_names})
    parts.append({f"U_{c}" for c in food_names})
    return parts

def _partition_hybrid_grid_A(farm_names, food_names, fg=5, cg=9):
    parts = []
    for fi in range(0, len(farm_names), fg):
        fgrp = farm_names[fi:fi + fg]
        for ci in range(0, len(food_names), cg):
            cgrp = food_names[ci:ci + cg]
            parts.append({f"Y_{f}_{c}" for f in fgrp for c in cgrp})
    parts.append({f"U_{c}" for c in food_names})
    return parts

def _partition_clique_B(farm_names):
    return [{f"Y_{f}_{fam}_t{t}" for fam in FAMILIES_6 for t in range(1, N_PERIODS+1)}
            for f in farm_names]

def _partition_spatial_temporal_B(farm_names, fpc=5):
    parts = []
    for i in range(0, len(farm_names), fpc):
        cl = farm_names[i:i + fpc]
        for t in range(1, N_PERIODS + 1):
            parts.append({f"Y_{f}_{fam}_t{t}" for f in cl for fam in FAMILIES_6})
    return parts


# ============================================================================
# Solver helpers
# ============================================================================

def solve_gurobi_full_A(farm_names, food_names, land, timeout=300):
    """Solve full Variant-A with Gurobi."""
    m, info = build_variant_a_gurobi(farm_names, food_names, land)
    m.setParam("TimeLimit", timeout)
    t0 = time.perf_counter()
    m.optimize()
    wall = time.perf_counter() - t0
    obj = m.ObjVal if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else None
    return {"solver": "Gurobi_full", "wall_time": wall, "objective": obj,
            "status": m.Status, "n_vars": m.NumVars}


def solve_gurobi_decomposed_A(farm_names, food_names, land, partitioner, timeout=300):
    """Solve Variant-A: Gurobi per sub-problem + merge."""
    parts = partitioner(farm_names, food_names)
    # Identify farm partitions (those that have Y_ variables)
    total_obj = 0.0
    t0 = time.perf_counter()
    sub_times = []
    for part in parts:
        # Extract farm names in this partition
        sub_farms = set()
        for v in part:
            if v.startswith("Y_"):
                pieces = v.split("_", 2)
                if len(pieces) >= 2:
                    sub_farms.add(pieces[1])
        sub_farms = [f for f in farm_names if f in sub_farms]
        if not sub_farms:
            continue
        sm, sinfo = build_variant_a_sub_gurobi(sub_farms, food_names, land)
        sm.setParam("TimeLimit", timeout)
        st0 = time.perf_counter()
        sm.optimize()
        sub_times.append(time.perf_counter() - st0)
        if sm.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            total_obj += sm.ObjVal
    wall = time.perf_counter() - t0
    return {"solver": "Gurobi_decomposed", "wall_time": wall,
            "objective": total_obj, "n_partitions": len(parts),
            "sub_times": sub_times}


def solve_gurobi_full_B(farm_names, land, timeout=300):
    """Solve full Variant-B with Gurobi (via QUBO)."""
    bqm = build_variant_b_bqm(farm_names, land)
    m, info = build_variant_b_gurobi(bqm)
    m.setParam("TimeLimit", timeout)
    t0 = time.perf_counter()
    m.optimize()
    wall = time.perf_counter() - t0
    obj = m.ObjVal if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else None
    return {"solver": "Gurobi_full", "wall_time": wall, "objective": obj,
            "status": m.Status, "n_vars": m.NumVars}


def solve_gurobi_decomposed_B(farm_names, land, partitioner, timeout=300):
    """Solve Variant-B: Gurobi per sub-BQM + merge."""
    # For partitions we need to build separate BQMs per cluster
    t0 = time.perf_counter()
    # Infer clusters from partition
    parts = partitioner(farm_names)
    total_energy = 0.0
    sub_times = []
    for part in parts:
        sub_farms = set()
        for v in part:
            if v.startswith("Y_"):
                pieces = v.split("_")
                if len(pieces) >= 2:
                    sub_farms.add(pieces[1])
        sub_farms = [f for f in farm_names if f in sub_farms]
        if not sub_farms:
            continue
        sub_bqm = build_variant_b_sub_bqm(sub_farms, land)
        sm, _ = build_variant_b_gurobi(sub_bqm)
        sm.setParam("TimeLimit", timeout)
        st0 = time.perf_counter()
        sm.optimize()
        sub_times.append(time.perf_counter() - st0)
        if sm.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            total_energy += sm.ObjVal
    wall = time.perf_counter() - t0
    return {"solver": "Gurobi_decomposed", "wall_time": wall,
            "objective": total_energy, "n_partitions": len(parts),
            "sub_times": sub_times}


def _extract_cqm_objective_A(sample: Dict[str, int], sub_farms: List[str],
                              food_names: List[str], land: Dict[str, float]) -> Tuple[float, int, int]:
    """Compute Variant-A CQM objective and count violations.
    
    Returns:
        (objective, n_violations, n_slots)
        - n_violations: farms with 0 or >1 crops assigned
        - n_slots: total farms (each farm should have exactly 1 crop)
    """
    total_area = sum(land.values())
    obj = 0.0
    n_violations = 0
    n_slots = len(sub_farms)
    
    for f in sub_farms:
        area_f = land.get(f, 1.0)
        crops_assigned = 0
        for c in food_names:
            vn = f"Y_{f}_{c}"
            if sample.get(vn, 0) == 1:
                obj += FOOD_BENEFITS.get(c, 0.5) * area_f / total_area
                crops_assigned += 1
        # Violation: farm has 0 or >1 crops assigned
        if crops_assigned != 1:
            n_violations += 1
    
    return obj, n_violations, n_slots


def solve_pticm_decomposed_A(farm_names, food_names, land, partitioner,
                              pt_kwargs=None):
    """Solve Variant-A sub-problems with PT-ICM + merge."""
    if pt_kwargs is None:
        pt_kwargs = {}
    parts = partitioner(farm_names, food_names)
    pt = ParallelTemperingICM(**pt_kwargs)
    total_obj = 0.0
    total_violations = 0
    total_slots = 0
    t0 = time.perf_counter()
    sub_times = []
    for part in parts:
        sub_farms = set()
        for v in part:
            if v.startswith("Y_"):
                pieces = v.split("_", 2)
                if len(pieces) >= 2:
                    sub_farms.add(pieces[1])
        sub_farms = [f for f in farm_names if f in sub_farms]
        if not sub_farms:
            continue
        # Build sub-BQM for these farms (Variant A QUBO)
        sub_bqm = _build_variant_a_bqm(sub_farms, food_names, land)
        sample, energy, _ = pt.solve(sub_bqm, num_reads=1)
        sub_times.append(time.perf_counter() - t0)
        # Extract CQM-equivalent objective and violations from sample
        obj, viols, slots = _extract_cqm_objective_A(sample, sub_farms, food_names, land)
        total_obj += obj
        total_violations += viols
        total_slots += slots
    wall = time.perf_counter() - t0
    
    # Compute violation rate and healed objective
    violation_rate = (total_violations / total_slots * 100) if total_slots > 0 else 0.0
    # Healed objective: estimate benefit lost due to violations
    avg_benefit_per_slot = total_obj / (total_slots - total_violations) if (total_slots - total_violations) > 0 else 0.0
    healed_obj = total_obj + total_violations * avg_benefit_per_slot
    
    return {"solver": "PT-ICM_decomposed", "wall_time": wall,
            "objective": total_obj, "n_partitions": len(parts),
            "sub_times": sub_times,
            "violations": total_violations, "total_slots": total_slots,
            "violation_rate_pct": violation_rate, "healed_objective": healed_obj}


def _extract_violations_B(sample: Dict[str, int], sub_farms: List[str]) -> Tuple[int, int]:
    """Count one-hot and rotation violations for Variant B.
    
    Returns:
        (one_hot_violations, total_farm_periods)
        - one_hot_violations: farm-period slots with 0 or >1 crops
    """
    n_violations = 0
    total_slots = len(sub_farms) * N_PERIODS
    
    for f in sub_farms:
        for t in range(1, N_PERIODS + 1):
            crops_assigned = 0
            for c in FAMILIES_6:
                vn = f"Y_{f}_{c}_t{t}"  # Note: variable format is Y_{farm}_{family}_t{period}
                if sample.get(vn, 0) == 1:
                    crops_assigned += 1
            if crops_assigned != 1:
                n_violations += 1
    
    return n_violations, total_slots


def _extract_cqm_objective_B(sample: Dict[str, int], sub_farms: List[str],
                              land: Dict[str, float]) -> float:
    """Compute Variant-B CQM objective from sample (benefit + rotation synergy)."""
    total_area = sum(land.values())
    obj = 0.0
    families = FAMILIES_6
    
    # Family benefit scores (same as in build_variant_b_bqm)
    benefits = {"Fruits": 0.65, "Grains": 0.72, "Legumes": 0.80,
                "Leafy_Vegetables": 0.70, "Root_Vegetables": 0.60,
                "Proteins": 0.68}
    
    # Linear benefit term
    for f in sub_farms:
        area_f = land.get(f, 1.0)
        for t in range(1, N_PERIODS + 1):
            for c in families:
                vn = f"Y_{f}_{c}_t{t}"  # Note: variable format is Y_{farm}_{family}_t{period}
                if sample.get(vn, 0) == 1:
                    obj += benefits.get(c, 0.5) * area_f / total_area
    
    # Rotation synergy term (consecutive periods)
    for f in sub_farms:
        area_f = land.get(f, 1.0)
        for t in range(1, N_PERIODS):
            for i, c1 in enumerate(families):
                for j, c2 in enumerate(families):
                    v1 = f"Y_{f}_{c1}_t{t}"
                    v2 = f"Y_{f}_{c2}_t{t+1}"
                    if sample.get(v1, 0) == 1 and sample.get(v2, 0) == 1:
                        syn = ROT_MATRIX[i, j]
                        obj += syn * 0.5 * area_f / total_area  # rotation_gamma=0.5
    
    return obj


def solve_pticm_decomposed_B(farm_names, land, partitioner, pt_kwargs=None):
    """Solve Variant-B sub-problems with PT-ICM + merge."""
    if pt_kwargs is None:
        pt_kwargs = {}
    parts = partitioner(farm_names)
    pt = ParallelTemperingICM(**pt_kwargs)
    total_obj = 0.0
    total_violations = 0
    total_slots = 0
    t0 = time.perf_counter()
    sub_times = []
    all_samples = {}  # Collect all samples for CQM objective extraction
    
    for part in parts:
        sub_farms = set()
        for v in part:
            if v.startswith("Y_"):
                pieces = v.split("_")
                if len(pieces) >= 2:
                    sub_farms.add(pieces[1])
        sub_farms = [f for f in farm_names if f in sub_farms]
        if not sub_farms:
            continue
        sub_bqm = build_variant_b_sub_bqm(sub_farms, land)
        sample, energy, _ = pt.solve(sub_bqm, num_reads=1)
        sub_times.append(time.perf_counter() - t0)
        all_samples.update(sample)
        
        # Count violations for this sub-problem
        viols, slots = _extract_violations_B(sample, sub_farms)
        total_violations += viols
        total_slots += slots
        
        # Extract CQM-equivalent objective (not raw QUBO energy)
        total_obj += _extract_cqm_objective_B(sample, sub_farms, land)
    
    wall = time.perf_counter() - t0
    
    # Compute violation rate and healed objective
    violation_rate = (total_violations / total_slots * 100) if total_slots > 0 else 0.0
    feasible_slots = total_slots - total_violations
    avg_benefit_per_slot = total_obj / feasible_slots if feasible_slots > 0 else 0.0
    healed_obj = total_obj + total_violations * avg_benefit_per_slot
    
    return {"solver": "PT-ICM_decomposed", "wall_time": wall,
            "objective": total_obj, "n_partitions": len(parts),
            "sub_times": sub_times,
            "violations": total_violations, "total_slots": total_slots,
            "violation_rate_pct": violation_rate, "healed_objective": healed_obj}


def _build_variant_a_bqm(sub_farms, food_names, land) -> BinaryQuadraticModel:
    """Build a QUBO for Variant-A sub-problem (one-hot + benefit)."""
    total_area = sum(land.values())
    bqm = BinaryQuadraticModel(vartype="BINARY")
    penalty = 3.0

    for f in sub_farms:
        area_f = land.get(f, 1.0)
        for c in food_names:
            bne = FOOD_BENEFITS.get(c, 0.5)
            bqm.add_variable(f"Y_{f}_{c}", -bne * area_f / total_area)

        # One-hot penalty
        vs = [f"Y_{f}_{c}" for c in food_names]
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                bqm.add_quadratic(vs[i], vs[j], 2 * penalty)
            bqm.add_linear(vs[i], -penalty)

    return bqm


# ============================================================================
# Main benchmark driver
# ============================================================================

FARM_SIZES = [5, 10, 25, 50, 100, 200]

DECOMPOSITIONS_A = {
    "PlotBased": _partition_plot_based_A,
    "Multilevel(5)": lambda fn, cn: _partition_multilevel_A(fn, cn, 5),
    "HybridGrid(5,9)": lambda fn, cn: _partition_hybrid_grid_A(fn, cn, 5, 9),
}

DECOMPOSITIONS_B = {
    "Clique": _partition_clique_B,
    "SpatialTemporal(5)": lambda fn: _partition_spatial_temporal_B(fn, 5),
}


def run_all():
    results = []
    food_names = FOOD_NAMES_27[:27]

    pt_small = {"n_replicas": 6, "sweeps": 500, "icm_interval": 10,
                "beta_min": 0.1, "beta_max": 4.0, "seed": 42}
    pt_medium = {"n_replicas": 8, "sweeps": 800, "icm_interval": 10,
                 "beta_min": 0.1, "beta_max": 5.0, "seed": 42}

    for n_farms in FARM_SIZES:
        land = generate_grid(n_farms, area=100.0, seed=42)
        farm_names = list(land.keys())
        n_vars_a = n_farms * len(food_names) + len(food_names)
        n_vars_b = n_farms * len(FAMILIES_6) * N_PERIODS

        pt_kw = pt_small if n_farms <= 50 else pt_medium

        LOG.info(f"\n{'='*60}")
        LOG.info(f"n_farms={n_farms}  vars_A={n_vars_a}  vars_B={n_vars_b}")
        LOG.info(f"{'='*60}")

        # --- Variant A ---
        LOG.info("[A] Gurobi full")
        r = solve_gurobi_full_A(farm_names, food_names, land)
        r.update(variant="A", decomposition="none", n_farms=n_farms, n_vars=n_vars_a)
        results.append(r)

        for dname, dfn in DECOMPOSITIONS_A.items():
            LOG.info(f"[A] Gurobi decomposed ({dname})")
            r = solve_gurobi_decomposed_A(farm_names, food_names, land, dfn)
            r.update(variant="A", decomposition=dname, n_farms=n_farms, n_vars=n_vars_a)
            results.append(r)

            LOG.info(f"[A] PT-ICM decomposed ({dname})")
            r = solve_pticm_decomposed_A(farm_names, food_names, land, dfn, pt_kw)
            r.update(variant="A", decomposition=dname, n_farms=n_farms, n_vars=n_vars_a)
            results.append(r)

        # --- Variant B ---
        LOG.info("[B] Gurobi full")
        r = solve_gurobi_full_B(farm_names, land)
        r.update(variant="B", decomposition="none", n_farms=n_farms, n_vars=n_vars_b)
        results.append(r)

        for dname, dfn in DECOMPOSITIONS_B.items():
            LOG.info(f"[B] Gurobi decomposed ({dname})")
            r = solve_gurobi_decomposed_B(farm_names, land, dfn)
            r.update(variant="B", decomposition=dname, n_farms=n_farms, n_vars=n_vars_b)
            results.append(r)

            LOG.info(f"[B] PT-ICM decomposed ({dname})")
            r = solve_pticm_decomposed_B(farm_names, land, dfn, pt_kw)
            r.update(variant="B", decomposition=dname, n_farms=n_farms, n_vars=n_vars_b)
            results.append(r)

    return results


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = run_all()

    # Serialize: convert any numpy types
    def _ser(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    out_file = out_dir / "solver_comparison_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=_ser)
    LOG.info(f"Results written to {out_file}")

    # Print summary table
    print(f"\n{'Variant':>3s} {'Solver':>22s} {'Decomp':>18s} {'Farms':>6s} "
          f"{'Obj':>12s} {'Healed':>10s} {'Viols':>6s} {'V%':>6s} {'Time(s)':>10s}")
    print("-" * 105)
    for r in all_results:
        obj_str = f"{r.get('objective', 'N/A'):.6f}" if r.get("objective") is not None else "TIMEOUT"
        healed_str = f"{r.get('healed_objective', '-'):.6f}" if r.get("healed_objective") is not None else "-"
        viols_str = str(r.get('violations', '-'))
        vrate_str = f"{r.get('violation_rate_pct', 0):.1f}" if r.get('violation_rate_pct') is not None else "-"
        print(f"{r['variant']:>3s} {r['solver']:>22s} {r['decomposition']:>18s} "
              f"{r['n_farms']:>6d} {obj_str:>12s} {healed_str:>10s} {viols_str:>6s} {vrate_str:>6s} {r['wall_time']:>10.4f}")
