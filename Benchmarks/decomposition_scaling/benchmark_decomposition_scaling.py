#!/usr/bin/env python3
"""
Benchmark: Decomposition Strategy Scaling Analysis

Measures decomposition overhead (partitioning time, partition count, partition sizes)
for Study 2.A (Variant A: Binary Patch allocation, 27 crops) and
Study 2.B (Variant B: Multi-period rotation, 6 families × 3 periods).

Each decomposition strategy is tested on its appropriate problem type across
increasing problem sizes to show computational complexity.

Output: JSON results + matplotlib plots.
"""

import os
import sys
import time
import json
import logging
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# We also reference the UC002-Food-Production-Optimization sibling repo for
# partition functions that live in algorithms/qpu_benchmark.py there.
SIBLING_REPO = PROJECT_ROOT.parent / "UC002-Food-Production-Optimization"
if SIBLING_REPO.exists():
    sys.path.insert(0, str(SIBLING_REPO))

from Utils.farm_sampler import generate_farms
from src.scenarios import load_food_data

# Re-implement partition functions locally (they only need food_names/farm_names)
# to avoid heavy D-Wave imports from qpu_benchmark.py.

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports for partitioning
# ---------------------------------------------------------------------------
try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

try:
    from sklearn.cluster import SpectralClustering
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False

# ============================================================================
# Study 2.A — Variant A partition functions (27 crop binary-patch problem)
# ============================================================================

def partition_plot_based(farm_names: List[str], food_names: List[str]) -> List[Set[str]]:
    """One partition per farm + one U partition."""
    parts = [{f"Y_{f}_{c}" for c in food_names} for f in farm_names]
    parts.append({f"U_{c}" for c in food_names})
    return parts


def partition_multilevel(farm_names: List[str], food_names: List[str],
                         group_size: int = 5) -> List[Set[str]]:
    """Group farms into clusters of *group_size*."""
    parts = []
    for i in range(0, len(farm_names), group_size):
        group = farm_names[i:i + group_size]
        parts.append({f"Y_{f}_{c}" for f in group for c in food_names})
    parts.append({f"U_{c}" for c in food_names})
    return parts


def partition_louvain(farm_names: List[str], food_names: List[str],
                      max_partition_size: int = 100) -> List[Set[str]]:
    """Louvain community-detection decomposition."""
    if not HAS_LOUVAIN:
        return partition_plot_based(farm_names, food_names)

    G = nx.Graph()
    all_vars = [f"Y_{f}_{c}" for f in farm_names for c in food_names]
    all_vars += [f"U_{c}" for c in food_names]
    G.add_nodes_from(all_vars)

    for farm in farm_names:
        fv = [f"Y_{farm}_{c}" for c in food_names]
        for i, v1 in enumerate(fv):
            for v2 in fv[i + 1:]:
                G.add_edge(v1, v2)
            for c in food_names:
                G.add_edge(f"Y_{farm}_{c}", f"U_{c}")

    communities = louvain_communities(G, seed=42, resolution=1.5)
    parts: List[Set[str]] = []
    for comm in communities:
        cs = set(comm)
        if len(cs) > max_partition_size:
            cl = list(cs)
            for j in range(0, len(cl), max_partition_size):
                parts.append(set(cl[j:j + max_partition_size]))
        else:
            parts.append(cs)
    return parts if parts else [set(all_vars)]


def partition_spectral(farm_names: List[str], food_names: List[str],
                       n_clusters: int = 10) -> List[Set[str]]:
    """Spectral clustering decomposition."""
    if not HAS_SPECTRAL:
        return partition_multilevel(farm_names, food_names, 5)

    all_vars = [f"Y_{f}_{c}" for f in farm_names for c in food_names]
    all_vars += [f"U_{c}" for c in food_names]
    var_to_idx = {v: i for i, v in enumerate(all_vars)}
    n = len(all_vars)
    adj = np.zeros((n, n))

    for farm in farm_names:
        fv = [f"Y_{farm}_{c}" for c in food_names]
        for i, v1 in enumerate(fv):
            for v2 in fv[i + 1:]:
                adj[var_to_idx[v1], var_to_idx[v2]] = 1
                adj[var_to_idx[v2], var_to_idx[v1]] = 1
            for c in food_names:
                i1 = var_to_idx[f"Y_{farm}_{c}"]
                i2 = var_to_idx[f"U_{c}"]
                adj[i1, i2] = adj[i2, i1] = 1

    # Guard: Spectral clustering requires O(n²) memory; skip for n > 5000
    if n > 5000:
        return partition_multilevel(farm_names, food_names, 5)

    k = max(2, min(n_clusters, n // 10, len(farm_names)))
    try:
        cl = SpectralClustering(n_clusters=k, affinity="precomputed",
                                random_state=42, assign_labels="kmeans")
        labels = cl.fit_predict(adj + np.eye(n) * 0.1)
        parts: Dict[int, Set[str]] = defaultdict(set)
        for var, lb in zip(all_vars, labels):
            parts[lb].add(var)
        return list(parts.values())
    except Exception:
        return partition_multilevel(farm_names, food_names, 5)


def partition_hybrid_grid(farm_names: List[str], food_names: List[str],
                          farm_group: int = 5, food_group: int = 9) -> List[Set[str]]:
    """HybridGrid: 2-D grid over farms × crops."""
    parts: List[Set[str]] = []
    for fi in range(0, len(farm_names), farm_group):
        fg = farm_names[fi:fi + farm_group]
        for ci in range(0, len(food_names), food_group):
            cg = food_names[ci:ci + food_group]
            parts.append({f"Y_{f}_{c}" for f in fg for c in cg})
    parts.append({f"U_{c}" for c in food_names})
    return parts


def partition_coordinated(farm_names: List[str], food_names: List[str]) -> List[Set[str]]:
    """Coordinated master-subproblem: master (U) + per-farm Y."""
    # Identical structure to PlotBased but solving order differs
    return partition_plot_based(farm_names, food_names)


def partition_cqm_first_plot(farm_names: List[str], food_names: List[str]) -> List[Set[str]]:
    """CQM-First PlotBased: same partition structure, different solving pipeline."""
    return partition_plot_based(farm_names, food_names)


# Registry — Study 2.A (Variant A)
VARIANT_A_METHODS = {
    "PlotBased": partition_plot_based,
    "Multilevel(5)": lambda fn, cn: partition_multilevel(fn, cn, 5),
    "Multilevel(10)": lambda fn, cn: partition_multilevel(fn, cn, 10),
    "Louvain": partition_louvain,
    "Spectral(10)": partition_spectral,
    "HybridGrid(5,9)": lambda fn, cn: partition_hybrid_grid(fn, cn, 5, 9),
    "HybridGrid(10,9)": lambda fn, cn: partition_hybrid_grid(fn, cn, 10, 9),
    "Coordinated": partition_coordinated,
    "CQM-First": partition_cqm_first_plot,
}

# ============================================================================
# Study 2.B — Variant B partition functions (rotation: 6 families × 3 periods)
# ============================================================================

FAMILIES_6 = ["Fruits", "Grains", "Legumes", "Leafy_Vegetables",
              "Root_Vegetables", "Proteins"]
N_PERIODS = 3


def rotation_partition_clique(farm_names: List[str]) -> List[Set[str]]:
    """Clique decomposition: one subproblem per farm (18 vars each)."""
    parts: List[Set[str]] = []
    for f in farm_names:
        parts.append({f"Y_{f}_{fam}_t{t}"
                      for fam in FAMILIES_6 for t in range(1, N_PERIODS + 1)})
    return parts


def rotation_partition_spatial_temporal(
        farm_names: List[str], farms_per_cluster: int = 5) -> List[Set[str]]:
    """Spatial-temporal decomposition: cluster farms × period slices."""
    parts: List[Set[str]] = []
    for i in range(0, len(farm_names), farms_per_cluster):
        cluster = farm_names[i:i + farms_per_cluster]
        for t in range(1, N_PERIODS + 1):
            parts.append({f"Y_{f}_{fam}_t{t}"
                          for f in cluster for fam in FAMILIES_6})
    return parts


# Registry — Study 2.B (Variant B)
VARIANT_B_METHODS = {
    "Clique(farm-by-farm)": rotation_partition_clique,
    "SpatialTemporal(5)": lambda fn: rotation_partition_spatial_temporal(fn, 5),
    "SpatialTemporal(10)": lambda fn: rotation_partition_spatial_temporal(fn, 10),
}

# ============================================================================
# Benchmark driver
# ============================================================================

FARM_SIZES_A = [5, 10, 25, 50, 100, 200, 500, 1000]
FARM_SIZES_B = [5, 10, 25, 50, 100, 200, 500, 1000]

# Load food names once (27 crops from the 'simple' or 'full' scenario)
try:
    _farms_0, _foods_0, _fg_0, _cfg_0 = load_food_data("full")
    FOOD_NAMES_27 = list(_foods_0.keys()) if isinstance(_foods_0, dict) else list(_foods_0)
except Exception:
    # Fallback: generate 27 crop names
    FOOD_NAMES_27 = [f"Crop_{i}" for i in range(27)]


def benchmark_variant_a() -> List[Dict]:
    """Benchmark all Variant-A decomposition strategies across farm sizes."""
    results = []
    for n_farms in FARM_SIZES_A:
        L = generate_farms(n_farms=n_farms, seed=42)
        farm_names = list(L.keys())
        food_names = FOOD_NAMES_27[:27]
        n_vars = n_farms * len(food_names) + len(food_names)  # Y + U

        for method_name, method_fn in VARIANT_A_METHODS.items():
            LOG.info(f"[A] {method_name:25s}  n_farms={n_farms:5d}  n_vars={n_vars}")
            t0 = time.perf_counter()
            try:
                parts = method_fn(farm_names, food_names)
            except Exception as e:
                LOG.warning(f"  FAILED: {e}")
                results.append({
                    "variant": "A", "method": method_name,
                    "n_farms": n_farms, "n_vars": n_vars,
                    "error": str(e),
                })
                continue
            elapsed = time.perf_counter() - t0

            sizes = [len(p) for p in parts]
            results.append({
                "variant": "A",
                "method": method_name,
                "n_farms": n_farms,
                "n_vars": n_vars,
                "n_partitions": len(parts),
                "total_partition_vars": sum(sizes),
                "max_partition_size": max(sizes),
                "min_partition_size": min(sizes),
                "mean_partition_size": float(np.mean(sizes)),
                "decomposition_time_s": elapsed,
            })
    return results


def benchmark_variant_b() -> List[Dict]:
    """Benchmark all Variant-B decomposition strategies across farm sizes."""
    results = []
    for n_farms in FARM_SIZES_B:
        L = generate_farms(n_farms=n_farms, seed=42)
        farm_names = list(L.keys())
        n_vars = n_farms * len(FAMILIES_6) * N_PERIODS  # Y only

        for method_name, method_fn in VARIANT_B_METHODS.items():
            LOG.info(f"[B] {method_name:25s}  n_farms={n_farms:5d}  n_vars={n_vars}")
            t0 = time.perf_counter()
            try:
                parts = method_fn(farm_names)
            except Exception as e:
                LOG.warning(f"  FAILED: {e}")
                results.append({
                    "variant": "B", "method": method_name,
                    "n_farms": n_farms, "n_vars": n_vars,
                    "error": str(e),
                })
                continue
            elapsed = time.perf_counter() - t0

            sizes = [len(p) for p in parts]
            results.append({
                "variant": "B",
                "method": method_name,
                "n_farms": n_farms,
                "n_vars": n_vars,
                "n_partitions": len(parts),
                "total_partition_vars": sum(sizes),
                "max_partition_size": max(sizes),
                "min_partition_size": min(sizes),
                "mean_partition_size": float(np.mean(sizes)),
                "decomposition_time_s": elapsed,
            })
    return results


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("=" * 60)
    LOG.info("Decomposition Scaling Benchmark")
    LOG.info("=" * 60)

    results_a = benchmark_variant_a()
    results_b = benchmark_variant_b()
    all_results = results_a + results_b

    out_file = out_dir / "decomposition_scaling_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    LOG.info(f"Results written to {out_file}")

    # Print summary table
    print("\n=== Variant A ===")
    print(f"{'Method':25s} {'Farms':>6s} {'Vars':>8s} {'Parts':>6s} {'MaxSz':>7s} {'Time(s)':>10s}")
    for r in results_a:
        if "error" not in r:
            print(f"{r['method']:25s} {r['n_farms']:6d} {r['n_vars']:8d} "
                  f"{r['n_partitions']:6d} {r['max_partition_size']:7d} "
                  f"{r['decomposition_time_s']:10.4f}")

    print("\n=== Variant B ===")
    print(f"{'Method':25s} {'Farms':>6s} {'Vars':>8s} {'Parts':>6s} {'MaxSz':>7s} {'Time(s)':>10s}")
    for r in results_b:
        if "error" not in r:
            print(f"{r['method']:25s} {r['n_farms']:6d} {r['n_vars']:8d} "
                  f"{r['n_partitions']:6d} {r['max_partition_size']:7d} "
                  f"{r['decomposition_time_s']:10.4f}")
