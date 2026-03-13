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
import math
import pandas as pd

try:
    _f0, _foods_dict, _fg0, _cfg0 = load_food_data("full_family")
    FOOD_NAMES_27 = list(_foods_dict.keys()) if isinstance(_foods_dict, dict) else list(_foods_dict)
    _PARAMS = _cfg0.get('parameters', {})
    _WEIGHTS = _PARAMS.get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.20,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15,
    })
    FOOD_BENEFITS = {}
    FOOD_GROUPS: Dict[str, List[str]] = {}
    FOOD_GROUP_CONSTRAINTS: Dict[str, Dict[str, int]] = {}
    MIN_PLANTING_AREA: Dict[str, float] = {}
    MAX_PERCENTAGE_PER_CROP: Dict[str, float] = {}
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
        FOOD_GROUP_CONSTRAINTS = _PARAMS.get('food_group_constraints', {
            g: {'min_foods': 2, 'max_foods': len(lst)}
            for g, lst in _fg0.items()
        })
        MIN_PLANTING_AREA = _PARAMS.get('minimum_planting_area', {
            c: 0.01 for c in FOOD_NAMES_27
        })
        MAX_PERCENTAGE_PER_CROP = _PARAMS.get('max_percentage_per_crop', {
            c: 0.4 for c in FOOD_NAMES_27
        })
except Exception:
    FOOD_NAMES_27 = [f"Crop_{i}" for i in range(27)]
    FOOD_BENEFITS = {c: np.random.default_rng(42).random() for c in FOOD_NAMES_27}
    FOOD_GROUPS = {"GroupA": FOOD_NAMES_27[:9], "GroupB": FOOD_NAMES_27[9:18],
                   "GroupC": FOOD_NAMES_27[18:27]}
    FOOD_GROUP_CONSTRAINTS = {g: {'min_foods': 2, 'max_foods': len(v)}
                              for g, v in FOOD_GROUPS.items()}
    MIN_PLANTING_AREA = {c: 0.01 for c in FOOD_NAMES_27}
    MAX_PERCENTAGE_PER_CROP = {c: 0.4 for c in FOOD_NAMES_27}


# Variant B uses 27 crops × 3 periods (same crops as Variant A)
N_PERIODS = 3
ROTATION_GAMMA = 0.5  # matches rotation_benchmark.py DEFAULT_GAMMA

# Load real rotation matrix from CSV
def _load_rotation_matrix() -> Optional[pd.DataFrame]:
    csv_path = PROJECT_ROOT / "rotation_data" / "rotation_crop_matrix.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0)
    return None

ROTATION_MATRIX_DF = _load_rotation_matrix()

# ============================================================================
# Variant A — build CQM / Gurobi model
# ============================================================================

def build_variant_a_gurobi(farm_names: List[str], food_names: List[str],
                           land: Dict[str, float]) -> Tuple[gp.Model, dict]:
    """Build the Binary Patch Gurobi model (Variant A full).

    Matches create_cqm_plots() from solver_runner_BINARY.py:
      - One-crop-per-farm (<=1)
      - U-Y linking: lower (Y<=U) AND upper (U <= sum Y)
      - Min plots per crop (conditional on U)
      - Max plots per crop (from max_percentage_per_crop)
      - Food group diversity: min_foods=2, max_foods per group
    """
    m = gp.Model("VariantA_full")
    m.setParam("OutputFlag", 0)

    total_area = sum(land.values())
    plot_area = total_area / len(farm_names) if farm_names else 1.0

    Y = {}
    for f in farm_names:
        for c in food_names:
            Y[f, c] = m.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}")

    U = {}
    for c in food_names:
        U[c] = m.addVar(vtype=GRB.BINARY, name=f"U_{c}")

    # Objective: maximize area-weighted benefit / total_area
    obj = gp.LinExpr()
    for f in farm_names:
        area_f = land.get(f, 1.0)
        for c in food_names:
            benefit = FOOD_BENEFITS.get(c, 0.5)
            obj += benefit * area_f / total_area * Y[f, c]
    m.setObjective(obj, GRB.MAXIMIZE)

    # Constraint: one crop per farm
    for f in farm_names:
        m.addConstr(gp.quicksum(Y[f, c] for c in food_names) <= 1,
                    name=f"one_crop_{f}")

    # U-Y linking: lower bound (Y_{f,c} <= U_c)
    for f in farm_names:
        for c in food_names:
            m.addConstr(Y[f, c] <= U[c], name=f"link_lower_{f}_{c}")

    # U-Y linking: upper bound (U_c <= sum_f Y_{f,c})
    for c in food_names:
        m.addConstr(U[c] <= gp.quicksum(Y[f, c] for f in farm_names),
                    name=f"link_upper_{c}")

    # Min plots per crop (conditional on U): sum_f Y >= min_plots * U
    for c in food_names:
        min_area = MIN_PLANTING_AREA.get(c, 0.0)
        if min_area > 0:
            min_plots = math.ceil(min_area / plot_area)
            if min_plots > 1:
                m.addConstr(
                    gp.quicksum(Y[f, c] for f in farm_names) >= min_plots * U[c],
                    name=f"min_plots_{c}")

    # Max plots per crop: sum_f Y <= max_plots
    for c in food_names:
        max_pct = MAX_PERCENTAGE_PER_CROP.get(c, 0.4)
        max_plots = math.floor(max_pct * total_area / plot_area)
        if max_plots < len(farm_names):
            m.addConstr(
                gp.quicksum(Y[f, c] for f in farm_names) <= max_plots,
                name=f"max_plots_{c}")

    # Food group diversity: min_foods and max_foods per group
    for gname, gcrops in FOOD_GROUPS.items():
        valid = [c for c in gcrops if c in food_names]
        if not valid:
            continue
        gc = FOOD_GROUP_CONSTRAINTS.get(gname, {})
        min_f = gc.get('min_foods', 2)
        max_f = gc.get('max_foods', len(valid))
        m.addConstr(gp.quicksum(U[c] for c in valid) >= min_f,
                    name=f"fg_min_{gname}")
        if max_f < len(valid):
            m.addConstr(gp.quicksum(U[c] for c in valid) <= max_f,
                        name=f"fg_max_{gname}")

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
# Variant B — build rotation BQM (QUBO) — 27 crops × 3 periods
# Matches create_cqm_plots_rotation_3period from solver_runner_ROTATION.py
# ============================================================================

def _get_rotation_value(crop_prev: str, crop_curr: str) -> float:
    """Look up R(crop_prev, crop_curr) from the real rotation CSV matrix."""
    if ROTATION_MATRIX_DF is None:
        return 0.0
    try:
        return float(ROTATION_MATRIX_DF.loc[crop_prev, crop_curr])
    except KeyError:
        return 0.0


def build_variant_b_bqm(farm_names: List[str], land: Dict[str, float],
                         food_names: Optional[List[str]] = None,
                         rotation_gamma: float = ROTATION_GAMMA,
                         one_hot_penalty: float = 3.0) -> BinaryQuadraticModel:
    """Build full Variant-B rotation BQM (27 crops × 3 periods).

    Matches create_cqm_plots_rotation_3period from solver_runner_ROTATION.py:
      - Linear: a_p * B_c * Y / A_tot  (summed across periods)
      - Quadratic: gamma * a_p * R(c,c') * Y_{p,c,t-1} * Y_{p,c',t} / A_tot²
        (double normalization: /A_tot inside synergy + /A_tot on whole objective)
      - One-hot penalty per farm-period
      - No diversity bonus, no spatial synergy
    """
    if food_names is None:
        food_names = FOOD_NAMES_27
    total_area = sum(land.values())
    bqm = BinaryQuadraticModel(vartype="BINARY")

    for f in farm_names:
        area_f = land.get(f, 1.0)
        # Linear benefit terms: a_p * B_c / A_tot (per period — from the CQM
        # the objective is (sum of linear + quadratic) / A_tot, which is
        # maximized via CQM's -objective, here minimized directly in BQM)
        for c in food_names:
            B_c = FOOD_BENEFITS.get(c, 0.5)
            for t in range(1, N_PERIODS + 1):
                vn = f"Y_{f}_{c}_t{t}"
                # Negative because CQM maximizes (sets -obj), BQM minimizes
                linear = -(area_f * B_c) / total_area
                bqm.add_variable(vn, linear)

        # Rotation synergy: gamma * a_p * R(c,c') / A_tot  (then whole /A_tot)
        # Net coefficient per pair: -gamma * a_p * R / A_tot²
        for t in range(2, N_PERIODS + 1):
            for c_prev in food_names:
                for c_curr in food_names:
                    R_cc = _get_rotation_value(c_prev, c_curr)
                    if R_cc == 0.0:
                        continue
                    v1 = f"Y_{f}_{c_prev}_t{t-1}"
                    v2 = f"Y_{f}_{c_curr}_t{t}"
                    # Negative: maximization -> minimization
                    coeff = -(rotation_gamma * area_f * R_cc) / (total_area * total_area)
                    bqm.add_quadratic(v1, v2, coeff)

        # One-hot penalty per period: exactly one crop per farm per period
        for t in range(1, N_PERIODS + 1):
            vt = [f"Y_{f}_{c}_t{t}" for c in food_names]
            for i in range(len(vt)):
                for j in range(i + 1, len(vt)):
                    bqm.add_quadratic(vt[i], vt[j], 2 * one_hot_penalty)
                bqm.add_linear(vt[i], -one_hot_penalty)

    return bqm


def build_variant_b_sub_bqm(sub_farms: List[str], land: Dict[str, float],
                              food_names: Optional[List[str]] = None,
                              **kwargs) -> BinaryQuadraticModel:
    """Build a rotation BQM for a farm sub-cluster."""
    return build_variant_b_bqm(sub_farms, land, food_names=food_names, **kwargs)


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

def _partition_clique_B(farm_names, food_names=None):
    if food_names is None:
        food_names = FOOD_NAMES_27
    return [{f"Y_{f}_{c}_t{t}" for c in food_names for t in range(1, N_PERIODS+1)}
            for f in farm_names]

def _partition_spatial_temporal_B(farm_names, fpc=5, food_names=None):
    if food_names is None:
        food_names = FOOD_NAMES_27
    parts = []
    for i in range(0, len(farm_names), fpc):
        cl = farm_names[i:i + fpc]
        for t in range(1, N_PERIODS + 1):
            parts.append({f"Y_{f}_{c}_t{t}" for f in cl for c in food_names})
    return parts


def _parse_B_partition(part: Set[str], farm_names: List[str],
                        food_names: List[str]) -> Dict[str, Set[int]]:
    """Parse a Variant-B partition's variable names into {farm: {periods}}.

    Variable format: Y_{farm}_{crop}_t{period}
    """
    from collections import defaultdict
    farm_periods: Dict[str, Set[int]] = defaultdict(set)
    farm_set = set(farm_names)
    for v in part:
        if not v.startswith("Y_"):
            continue
        rest = v[2:]  # strip "Y_"
        # Try splitting: rest = "{farm}_{crop}_t{period}"
        # Farm names are like "Patch1", "Patch2", etc. — find first "_" after farm
        idx_us = rest.index("_")
        fname = rest[:idx_us]
        if fname in farm_set:
            suffix = rest[idx_us + 1:]
            idx_t = suffix.rfind("_t")
            if idx_t >= 0:
                try:
                    period = int(suffix[idx_t + 2:])
                    farm_periods[fname].add(period)
                except ValueError:
                    pass
    return dict(farm_periods)


def _build_sub_bqm_B_from_partition(farm_periods: Dict[str, Set[int]],
                                      land: Dict[str, float],
                                      food_names: List[str],
                                      one_hot_penalty: float = 3.0) -> BinaryQuadraticModel:
    """Build a Variant-B sub-BQM restricted to specified (farm, periods) pairs."""
    total_area = sum(land.values())
    bqm = BinaryQuadraticModel(vartype="BINARY")

    for f, periods in farm_periods.items():
        area_f = land.get(f, 1.0)

        for t in sorted(periods):
            # Linear benefit
            for c in food_names:
                vn = f"Y_{f}_{c}_t{t}"
                linear = -(area_f * FOOD_BENEFITS.get(c, 0.5)) / total_area
                bqm.add_variable(vn, linear)

            # One-hot penalty
            vt = [f"Y_{f}_{c}_t{t}" for c in food_names]
            for i in range(len(vt)):
                for j in range(i + 1, len(vt)):
                    bqm.add_quadratic(vt[i], vt[j], 2 * one_hot_penalty)
                bqm.add_linear(vt[i], -one_hot_penalty)

        # Rotation synergies (only between consecutive periods BOTH in partition)
        sorted_periods = sorted(periods)
        for idx in range(len(sorted_periods) - 1):
            t_prev = sorted_periods[idx]
            t_curr = sorted_periods[idx + 1]
            if t_curr != t_prev + 1:
                continue
            for c_prev in food_names:
                for c_curr in food_names:
                    R_cc = _get_rotation_value(c_prev, c_curr)
                    if R_cc == 0.0:
                        continue
                    v1 = f"Y_{f}_{c_prev}_t{t_prev}"
                    v2 = f"Y_{f}_{c_curr}_t{t_curr}"
                    coeff = -(ROTATION_GAMMA * area_f * R_cc) / (total_area * total_area)
                    bqm.add_quadratic(v1, v2, coeff)

    return bqm


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


def _extract_sub_farms_foods(part: Set[str], farm_names: List[str],
                              food_names: List[str]) -> Tuple[List[str], List[str]]:
    """Extract sub-farms and sub-crops from a partition's variable names."""
    farm_set = set(farm_names)
    food_set = set(food_names)
    sub_farms = set()
    sub_foods = set()
    for v in part:
        if v.startswith("Y_"):
            rest = v[2:]  # Strip "Y_"
            idx = rest.index("_")
            fname = rest[:idx]
            if fname in farm_set:
                sub_farms.add(fname)
                crop_part = rest[idx + 1:]
                if crop_part in food_set:
                    sub_foods.add(crop_part)
    sf = [f for f in farm_names if f in sub_farms]
    sc = [c for c in food_names if c in sub_foods]
    return sf, sc if sc else food_names  # fallback to all crops if none extracted


def solve_gurobi_decomposed_A(farm_names, food_names, land, partitioner, timeout=300):
    """Solve Variant-A: Gurobi per sub-problem + merge."""
    parts = partitioner(farm_names, food_names)
    total_obj = 0.0
    t0 = time.perf_counter()
    sub_times = []
    for part in parts:
        sub_farms, sub_foods = _extract_sub_farms_foods(part, farm_names, food_names)
        if not sub_farms:
            continue
        sm, sinfo = build_variant_a_sub_gurobi(sub_farms, sub_foods, land)
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


def solve_gurobi_full_B(farm_names, land, food_names=None, timeout=300):
    """Solve full Variant-B with Gurobi (via QUBO)."""
    bqm = build_variant_b_bqm(farm_names, land, food_names=food_names)
    m, info = build_variant_b_gurobi(bqm)
    m.setParam("TimeLimit", timeout)
    t0 = time.perf_counter()
    m.optimize()
    wall = time.perf_counter() - t0
    obj = m.ObjVal if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else None
    return {"solver": "Gurobi_full", "wall_time": wall, "objective": obj,
            "status": m.Status, "n_vars": m.NumVars}


def solve_gurobi_decomposed_B(farm_names, land, partitioner, food_names=None, timeout=300):
    """Solve Variant-B: Gurobi per sub-BQM + merge."""
    if food_names is None:
        food_names = FOOD_NAMES_27
    t0 = time.perf_counter()
    parts = partitioner(farm_names, food_names)
    total_energy = 0.0
    sub_times = []
    for part in parts:
        fp = _parse_B_partition(part, farm_names, food_names)
        if not fp:
            continue
        sub_bqm = _build_sub_bqm_B_from_partition(fp, land, food_names)
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
        sub_farms, sub_foods = _extract_sub_farms_foods(part, farm_names, food_names)
        if not sub_farms:
            continue
        sub_bqm = _build_variant_a_bqm(sub_farms, sub_foods, land)
        sample, energy, _ = pt.solve(sub_bqm, num_reads=1)
        sub_times.append(time.perf_counter() - t0)
        obj, viols, slots = _extract_cqm_objective_A(sample, sub_farms, sub_foods, land)
        total_obj += obj
        total_violations += viols
        total_slots += slots
    wall = time.perf_counter() - t0
    
    violation_rate = (total_violations / total_slots * 100) if total_slots > 0 else 0.0
    avg_benefit_per_slot = total_obj / (total_slots - total_violations) if (total_slots - total_violations) > 0 else 0.0
    healed_obj = total_obj + total_violations * avg_benefit_per_slot
    
    return {"solver": "PT-ICM_decomposed", "wall_time": wall,
            "objective": total_obj, "n_partitions": len(parts),
            "sub_times": sub_times,
            "violations": total_violations, "total_slots": total_slots,
            "violation_rate_pct": violation_rate, "healed_objective": healed_obj}


def _extract_violations_B(sample: Dict[str, int], sub_farms: List[str],
                          food_names: Optional[List[str]] = None,
                          periods: Optional[List[int]] = None) -> Tuple[int, int]:
    """Count one-hot violations for Variant B.
    
    Returns:
        (one_hot_violations, total_farm_periods)
    """
    if food_names is None:
        food_names = FOOD_NAMES_27
    if periods is None:
        periods = list(range(1, N_PERIODS + 1))
    n_violations = 0
    total_slots = len(sub_farms) * len(periods)
    
    for f in sub_farms:
        for t in periods:
            crops_assigned = 0
            for c in food_names:
                vn = f"Y_{f}_{c}_t{t}"
                if sample.get(vn, 0) == 1:
                    crops_assigned += 1
            if crops_assigned != 1:
                n_violations += 1
    
    return n_violations, total_slots


def _extract_cqm_objective_B(sample: Dict[str, int], sub_farms: List[str],
                              land: Dict[str, float],
                              food_names: Optional[List[str]] = None,
                              periods: Optional[List[int]] = None) -> float:
    """Compute Variant-B CQM objective from sample (benefit + rotation synergy).
    
    Matches create_cqm_plots_rotation_3period normalization:
      linear:    a_p * B_c / A_tot   (per period)
      synergy:   gamma * a_p * R / A_tot  then whole /A_tot  => gamma * a_p * R / A_tot²
    """
    if food_names is None:
        food_names = FOOD_NAMES_27
    if periods is None:
        periods = list(range(1, N_PERIODS + 1))
    total_area = sum(land.values())
    obj = 0.0
    
    # Linear benefit term
    for f in sub_farms:
        area_f = land.get(f, 1.0)
        for t in periods:
            for c in food_names:
                vn = f"Y_{f}_{c}_t{t}"
                if sample.get(vn, 0) == 1:
                    obj += FOOD_BENEFITS.get(c, 0.5) * area_f / total_area
    
    # Rotation synergy term (consecutive periods in the provided list)
    sorted_p = sorted(periods)
    for f in sub_farms:
        area_f = land.get(f, 1.0)
        for idx in range(len(sorted_p) - 1):
            t_prev, t_curr = sorted_p[idx], sorted_p[idx + 1]
            if t_curr != t_prev + 1:
                continue
            for c_prev in food_names:
                v1 = f"Y_{f}_{c_prev}_t{t_prev}"
                if sample.get(v1, 0) != 1:
                    continue
                for c_curr in food_names:
                    v2 = f"Y_{f}_{c_curr}_t{t_curr}"
                    if sample.get(v2, 0) != 1:
                        continue
                    R_cc = _get_rotation_value(c_prev, c_curr)
                    obj += ROTATION_GAMMA * area_f * R_cc / (total_area * total_area)
    
    return obj


def solve_pticm_decomposed_B(farm_names, land, partitioner, food_names=None,
                              pt_kwargs=None):
    """Solve Variant-B sub-problems with PT-ICM + merge."""
    if food_names is None:
        food_names = FOOD_NAMES_27
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
        fp = _parse_B_partition(part, farm_names, food_names)
        if not fp:
            continue
        sub_bqm = _build_sub_bqm_B_from_partition(fp, land, food_names)
        sample, energy, _ = pt.solve(sub_bqm, num_reads=1)
        sub_times.append(time.perf_counter() - t0)
        
        sub_farms = [f for f in farm_names if f in fp]
        # Collect all periods present in this partition
        all_periods = sorted(set().union(*fp.values()))
        viols, slots = _extract_violations_B(sample, sub_farms, food_names, periods=all_periods)
        total_violations += viols
        total_slots += slots
        
        total_obj += _extract_cqm_objective_B(sample, sub_farms, land, food_names, periods=all_periods)
    
    wall = time.perf_counter() - t0
    
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

FARM_SIZES = [5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
PT_ICM_MAX_FARMS = 200  # PT-ICM is too slow beyond this with 27 crops × 3 periods

DECOMPOSITIONS_A = {
    "PlotBased": _partition_plot_based_A,
    "Multilevel(5)": lambda fn, cn: _partition_multilevel_A(fn, cn, 5),
    "HybridGrid(5,9)": lambda fn, cn: _partition_hybrid_grid_A(fn, cn, 5, 9),
}

DECOMPOSITIONS_B = {
    "Clique": _partition_clique_B,
    "SpatialTemporal(5)": lambda fn, cn: _partition_spatial_temporal_B(fn, 5, cn),
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
        n_vars_b = n_farms * len(food_names) * N_PERIODS

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

            if n_farms <= PT_ICM_MAX_FARMS:
                LOG.info(f"[A] PT-ICM decomposed ({dname})")
                r = solve_pticm_decomposed_A(farm_names, food_names, land, dfn, pt_kw)
                r.update(variant="A", decomposition=dname, n_farms=n_farms, n_vars=n_vars_a)
                results.append(r)
            else:
                LOG.info(f"[A] PT-ICM decomposed ({dname}) SKIPPED (n_farms > {PT_ICM_MAX_FARMS})")

        # --- Variant B (27 crops × 3 periods — much larger BQMs) ---
        VARIANT_B_MAX_FARMS = 2000  # SpatialTemporal(5) Gurobi too slow beyond this
        if n_farms > VARIANT_B_MAX_FARMS:
            LOG.info(f"[B] ALL SKIPPED (n_farms={n_farms} > {VARIANT_B_MAX_FARMS})")
            for dname in ["none"] + list(DECOMPOSITIONS_B.keys()):
                results.append({"solver": "Gurobi_full" if dname == "none" else f"Gurobi_decomposed",
                                "wall_time": 0, "objective": None,
                                "variant": "B", "decomposition": dname,
                                "n_farms": n_farms, "n_vars": n_vars_b, "status": "SKIPPED"})
        else:
            # Skip Variant B full Gurobi for very large instances
            if n_vars_b <= 200000:
                LOG.info("[B] Gurobi full")
                r = solve_gurobi_full_B(farm_names, land, food_names=food_names)
                r.update(variant="B", decomposition="none", n_farms=n_farms, n_vars=n_vars_b)
                results.append(r)
            else:
                LOG.info(f"[B] Gurobi full SKIPPED (n_vars={n_vars_b} > 200000)")
                results.append({"solver": "Gurobi_full", "wall_time": 0, "objective": None,
                                "variant": "B", "decomposition": "none",
                                "n_farms": n_farms, "n_vars": n_vars_b, "status": "SKIPPED"})

            for dname, dfn in DECOMPOSITIONS_B.items():
                LOG.info(f"[B] Gurobi decomposed ({dname})")
                r = solve_gurobi_decomposed_B(farm_names, land, dfn, food_names=food_names)
                r.update(variant="B", decomposition=dname, n_farms=n_farms, n_vars=n_vars_b)
                results.append(r)

                if n_farms <= PT_ICM_MAX_FARMS:
                    LOG.info(f"[B] PT-ICM decomposed ({dname})")
                    r = solve_pticm_decomposed_B(farm_names, land, dfn, food_names=food_names,
                                                  pt_kwargs=pt_kw)
                    r.update(variant="B", decomposition=dname, n_farms=n_farms, n_vars=n_vars_b)
                    results.append(r)
                else:
                    LOG.info(f"[B] PT-ICM decomposed ({dname}) SKIPPED (n_farms > {PT_ICM_MAX_FARMS})")

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
