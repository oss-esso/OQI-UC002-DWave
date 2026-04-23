#!/usr/bin/env python3
"""
Generate all plots from the decomposition scaling and solver comparison benchmarks.

Reads:
  - decomposition_scaling_results.json
  - solver_comparison_results.json
  - (optional) ../../qpu_hier_repaired.json
  - (optional) ../../@todo/gurobi_timeout_verification/gurobi_timeout_test_*.json

Produces PDF plots in the same directory.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUT_DIR = Path(__file__).parent
COLORS_A = {
    "PlotBased": "#1f77b4",
    "Multilevel(5)": "#ff7f0e",
    "Multilevel(10)": "#2ca02c",
    "Louvain": "#d62728",
    "Spectral(10)": "#9467bd",
    "HybridGrid(5,9)": "#8c564b",
    "HybridGrid(10,9)": "#e377c2",
    "Coordinated": "#7f7f7f",
    "CQM-First": "#bcbd22",
}
COLORS_B = {
    "Clique(plot-by-plot)": "#1f77b4",
    "SpatialTemporal(5)": "#ff7f0e",
    "SpatialTemporal(10)": "#2ca02c",
}
SOLVER_STYLES = {
    "Gurobi_full": {"color": "#e41a1c", "ls": "-", "marker": "o"},
    "Gurobi_decomposed": {"color": "#377eb8", "ls": "--", "marker": "s"},
    "PT-ICM_decomposed": {"color": "#4daf4a", "ls": "-.", "marker": "^"},
}


# ============================================================================
# Plot 1: Decomposition time vs problem size
# ============================================================================

def plot_decomposition_time(data, variant, title_suffix, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = defaultdict(lambda: ([], []))
    for r in data:
        if r["variant"] != variant or "error" in r:
            continue
        methods[r["method"]][0].append(r["n_farms"])
        methods[r["method"]][1].append(r["decomposition_time_s"])

    colors = COLORS_A if variant == "A" else COLORS_B
    for mname, (xs, ys) in sorted(methods.items()):
        c = colors.get(mname, "#333333")
        ax.plot(xs, ys, marker="o", label=mname, color=c, markersize=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of plots")
    ax.set_ylabel("Decomposition time (s)")
    ax.set_title(f"Decomposition Overhead — {title_suffix}")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)
    print(f"  -> {fname}")


# ============================================================================
# Plot 2: Number of partitions vs problem size
# ============================================================================

def plot_partition_count(data, variant, title_suffix, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = defaultdict(lambda: ([], []))
    for r in data:
        if r["variant"] != variant or "error" in r:
            continue
        methods[r["method"]][0].append(r["n_farms"])
        methods[r["method"]][1].append(r["n_partitions"])

    colors = COLORS_A if variant == "A" else COLORS_B
    for mname, (xs, ys) in sorted(methods.items()):
        c = colors.get(mname, "#333333")
        ax.plot(xs, ys, marker="s", label=mname, color=c, markersize=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of plots")
    ax.set_ylabel("Number of partitions")
    ax.set_title(f"Partition Count Scaling — {title_suffix}")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)
    print(f"  -> {fname}")


# ============================================================================
# Plot 3: Max partition size vs problem size
# ============================================================================

def plot_max_partition_size(data, variant, title_suffix, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = defaultdict(lambda: ([], []))
    for r in data:
        if r["variant"] != variant or "error" in r:
            continue
        methods[r["method"]][0].append(r["n_farms"])
        methods[r["method"]][1].append(r["max_partition_size"])

    colors = COLORS_A if variant == "A" else COLORS_B
    for mname, (xs, ys) in sorted(methods.items()):
        c = colors.get(mname, "#333333")
        ax.plot(xs, ys, marker="^", label=mname, color=c, markersize=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of plots")
    ax.set_ylabel("Max sub-problem size (variables)")
    ax.set_title(f"Largest Sub-Problem — {title_suffix}")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)
    print(f"  -> {fname}")


# ============================================================================
# Plot 4: Solver comparison — wall time (ALL decomposition methods)
# ============================================================================

# Extended color/style map for all solver+decomposition combinations
SOLVER_DECOMP_STYLES = {
    # Gurobi full (baseline)
    "Gurobi_full": {"color": "#e41a1c", "ls": "-", "marker": "o", "lw": 2},
    # Gurobi decomposed
    "Gurobi_decomposed|PlotBased": {"color": "#377eb8", "ls": "--", "marker": "s"},
    "Gurobi_decomposed|Multilevel(5)": {"color": "#4daf4a", "ls": "--", "marker": "^"},
    "Gurobi_decomposed|HybridGrid(5,9)": {"color": "#984ea3", "ls": "--", "marker": "D"},
    "Gurobi_decomposed|Clique": {"color": "#377eb8", "ls": "--", "marker": "s"},
    "Gurobi_decomposed|SpatialTemporal(5)": {"color": "#4daf4a", "ls": "--", "marker": "^"},
    "Gurobi_decomposed|Hierarchical(spatial_grid,9)": {"color": "#377eb8", "ls": "--", "marker": "s"},
    # PT-ICM decomposed
    "PT-ICM_decomposed|PlotBased": {"color": "#ff7f00", "ls": "-.", "marker": "v"},
    "PT-ICM_decomposed|Multilevel(5)": {"color": "#ffff33", "ls": "-.", "marker": "<"},
    "PT-ICM_decomposed|HybridGrid(5,9)": {"color": "#a65628", "ls": "-.", "marker": ">"},
    "PT-ICM_decomposed|Clique": {"color": "#ff7f00", "ls": "-.", "marker": "v"},
    "PT-ICM_decomposed|SpatialTemporal(5)": {"color": "#f781bf", "ls": "-.", "marker": "<"},
}


def plot_solver_time_all(data, variant, fname):
    """Plot wall time for ALL solver+decomposition combinations on one figure."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    series = defaultdict(lambda: ([], []))

    for r in data:
        if r["variant"] != variant:
            continue
        solver = r["solver"]
        decomp = r["decomposition"]
        # Create unique key for each solver+decomposition
        if decomp == "none":
            key = solver
            label = solver
        else:
            key = f"{solver}|{decomp}"
            label = f"{solver} [{decomp}]"

        series[key][0].append(r["n_farms"])
        series[key][1].append(r["wall_time"])

    for key, (xs, ys) in sorted(series.items()):
        sty = SOLVER_DECOMP_STYLES.get(key, {"color": "#333", "ls": "-", "marker": "o"})
        label = key.replace("|", " [") + "]" if "|" in key else key
        # Sort by x for proper line drawing
        sorted_pairs = sorted(zip(xs, ys))
        xs_s, ys_s = zip(*sorted_pairs) if sorted_pairs else ([], [])
        ax.plot(xs_s, ys_s, label=label, markersize=5, **sty)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of plots")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"Solver Comparison — Variant {variant} (All Decompositions)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)
    print(f"  -> {fname}")


def plot_solver_time(data, variant, decomp_name, fname):
    """Plot wall time for a specific decomposition (kept for backward compat)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    solvers = defaultdict(lambda: ([], []))
    for r in data:
        if r["variant"] != variant:
            continue
        key = r["solver"]
        if r["decomposition"] != "none" and r["decomposition"] != decomp_name:
            continue
        if r["decomposition"] == "none" and "decomposed" in key:
            continue
        solvers[key][0].append(r["n_farms"])
        solvers[key][1].append(r["wall_time"])

    for sname, (xs, ys) in sorted(solvers.items()):
        sty = SOLVER_STYLES.get(sname, {"color": "#333", "ls": "-", "marker": "o"})
        sorted_pairs = sorted(zip(xs, ys))
        xs_s, ys_s = zip(*sorted_pairs) if sorted_pairs else ([], [])
        ax.plot(xs_s, ys_s, label=sname, **sty, markersize=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of plots")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"Solver Comparison — Variant {variant}, {decomp_name}")
    ax.legend(framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)
    print(f"  -> {fname}")


# ============================================================================
# Plot 5: Solver comparison — objective quality (ALL decomposition methods)
# ============================================================================

def plot_solver_quality_all(data, variant, fname):
    """Plot objective for ALL solver+decomposition combinations on one figure."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    series = defaultdict(lambda: ([], []))

    for r in data:
        if r["variant"] != variant:
            continue
        if r.get("objective") is None:
            continue
        solver = r["solver"]
        decomp = r["decomposition"]
        if decomp == "none":
            key = solver
        else:
            key = f"{solver}|{decomp}"

        series[key][0].append(r["n_farms"])
        series[key][1].append(r["objective"])

    for key, (xs, ys) in sorted(series.items()):
        sty = SOLVER_DECOMP_STYLES.get(key, {"color": "#333", "ls": "-", "marker": "o"})
        label = key.replace("|", " [") + "]" if "|" in key else key
        sorted_pairs = sorted(zip(xs, ys))
        xs_s, ys_s = zip(*sorted_pairs) if sorted_pairs else ([], [])
        ax.plot(xs_s, np.abs(ys_s), label=label, markersize=5, **sty)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of plots")
    ax.set_ylabel("Objective value")
    ax.set_title(f"Solution Quality — Variant {variant} (All Decompositions)")
    ax.legend(loc="best", framealpha=0.9, fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)
    print(f"  -> {fname}")


def plot_solver_quality(data, variant, decomp_name, fname):
    """Plot objective for a specific decomposition (kept for backward compat)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    solvers = defaultdict(lambda: ([], []))
    for r in data:
        if r["variant"] != variant:
            continue
        if r.get("objective") is None:
            continue
        key = r["solver"]
        if r["decomposition"] != "none" and r["decomposition"] != decomp_name:
            continue
        if r["decomposition"] == "none" and "decomposed" in key:
            continue
        solvers[key][0].append(r["n_farms"])
        solvers[key][1].append(r["objective"])

    for sname, (xs, ys) in sorted(solvers.items()):
        sty = SOLVER_STYLES.get(sname, {"color": "#333", "ls": "-", "marker": "o"})
        sorted_pairs = sorted(zip(xs, ys))
        xs_s, ys_s = zip(*sorted_pairs) if sorted_pairs else ([], [])
        ax.plot(xs_s, ys_s, label=sname, **sty, markersize=5)

    ax.set_xscale("log")
    ax.set_xlabel("Number of plots")
    ax.set_ylabel("Objective value")
    ax.set_title(f"Solution Quality — Variant {variant}, {decomp_name}")
    ax.legend(framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)
    print(f"  -> {fname}")


# ============================================================================
# Plot 6: Combined comparison panel
# ============================================================================

def plot_combined_panel(data_solver, fname):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: Variant A time
    ax = axes[0, 0]
    for r in data_solver:
        if r["variant"] != "A":
            continue
        sty = SOLVER_STYLES.get(r["solver"], {"color": "#333", "ls": "-", "marker": "o"})
        label = f"{r['solver']}({r['decomposition']})" if r["decomposition"] != "none" else r["solver"]
        ax.scatter(r["n_farms"], r["wall_time"], color=sty["color"],
                   marker=sty["marker"], s=30, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("plots"); ax.set_ylabel("Time (s)")
    ax.set_title("Variant A — Solve Time")
    ax.grid(True, alpha=0.3)

    # Top-right: Variant A quality
    ax = axes[0, 1]
    for r in data_solver:
        if r["variant"] != "A" or r.get("objective") is None:
            continue
        sty = SOLVER_STYLES.get(r["solver"], {"color": "#333", "ls": "-", "marker": "o"})
        ax.scatter(r["n_farms"], r["objective"], color=sty["color"],
                   marker=sty["marker"], s=30, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("plots"); ax.set_ylabel("Objective")
    ax.set_title("Variant A — Solution Quality")
    ax.grid(True, alpha=0.3)

    # Bottom-left: Variant B time
    ax = axes[1, 0]
    for r in data_solver:
        if r["variant"] != "B":
            continue
        sty = SOLVER_STYLES.get(r["solver"], {"color": "#333", "ls": "-", "marker": "o"})
        ax.scatter(r["n_farms"], r["wall_time"], color=sty["color"],
                   marker=sty["marker"], s=30, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("plots"); ax.set_ylabel("Time (s)")
    ax.set_title("Variant B — Solve Time")
    ax.grid(True, alpha=0.3)

    # Bottom-right: Variant B quality
    ax = axes[1, 1]
    for r in data_solver:
        if r["variant"] != "B" or r.get("objective") is None:
            continue
        sty = SOLVER_STYLES.get(r["solver"], {"color": "#333", "ls": "-", "marker": "o"})
        ax.scatter(r["n_farms"], r["objective"], color=sty["color"],
                   marker=sty["marker"], s=30, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("plots"); ax.set_ylabel("Objective (energy)")
    ax.set_title("Variant B — Solution Quality")
    ax.grid(True, alpha=0.3)

    # Manual legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=s["color"], marker=s["marker"],
                       ls=s["ls"], label=n) for n, s in SOLVER_STYLES.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, framealpha=0.9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)
    print(f"  -> {fname}")


# ============================================================================
# Study 2.B three-group comparison data loaders
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent  # OQI-UC002-DWave root
STUDY2B_FARMS = [25, 50, 100, 200]
STUDY2B_FARM_SET = set(STUDY2B_FARMS)
STUDY2B_DECOMP_LABEL = "Hierarchical(spatial_grid,9)"


def load_qpu_data_27food():
    """Load QPU 27-food Variant B runs from qpu_hier_repaired.json.

    Returns list of dicts: {n_farms, n_vars, wall_time, benefit}
    where benefit = -objective_miqp (sign-corrected to positive QUBO sense).
    """
    path = PROJECT_ROOT / "qpu_hier_repaired.json"
    if not path.exists():
        return []
    with open(path) as f:
        raw = json.load(f)
    runs = raw.get("runs", raw) if isinstance(raw, dict) else raw
    out = []
    for r in runs:
        if r.get("n_foods", 0) != 27:
            continue
        if r.get("n_farms") not in STUDY2B_FARM_SET:
            continue
        timing = r.get("timing", {})
        out.append({
            "n_farms": r["n_farms"],
            "n_vars": r["n_vars"],
            "wall_time": timing.get("total_wall_time", 0.0),
            "benefit": -(r.get("objective_miqp") or 0.0),  # sign-corrected
            "violations": r.get("constraint_violations", {}).get("total_violations", 0),
        })
    return out


def load_qpu_data_6food():
    """Load QPU 6-food Variant B runs from qpu_hier_repaired.json.

    Returns list of dicts: {n_farms, n_vars, wall_time, benefit}
    """
    path = PROJECT_ROOT / "qpu_hier_repaired.json"
    if not path.exists():
        return []
    with open(path) as f:
        raw = json.load(f)
    runs = raw.get("runs", raw) if isinstance(raw, dict) else raw
    # De-duplicate: keep the latest (highest wall_time) per n_farms
    best = {}
    for r in runs:
        if r.get("n_foods", 0) != 6:
            continue
        nf = r["n_farms"]
        timing = r.get("timing", {})
        wt = timing.get("total_wall_time", 0.0)
        if nf not in best or wt > best[nf]["wall_time"]:
            best[nf] = {
                "n_farms": nf,
                "n_vars": r["n_vars"],
                "wall_time": wt,
                "benefit": -(r.get("objective_miqp") or 0.0),
                "violations": r.get("constraint_violations", {}).get("total_violations", 0),
            }
    return list(best.values())


def load_gurobi_timeout_data():
    """Load latest Gurobi timeout verification data.

    Returns two lists: (runs_27food, runs_6food), each with
    {n_farms, n_vars, wall_time (=solve_time), objective, n_foods}
    """
    timeout_dir = PROJECT_ROOT / "@todo" / "gurobi_timeout_verification"
    if not timeout_dir.exists():
        return [], []
    files = sorted(timeout_dir.glob("gurobi_timeout_test_*.json"))
    if not files:
        return [], []
    with open(files[-1]) as f:
        raw = json.load(f)
    entries = raw if isinstance(raw, list) else raw.get("runs", [])
    runs_27, runs_6 = [], []
    for e in entries:
        meta = e.get("metadata", {})
        res = e.get("result", {})
        nf = meta.get("n_farms", 0)
        nfoods = meta.get("n_foods", 0)
        entry = {
            "n_farms": nf,
            "n_vars": res.get("n_vars", nf * nfoods * meta.get("n_periods", 3)),
            "wall_time": res.get("solve_time", 0.0),
            "objective": res.get("objective_value"),
            "n_foods": nfoods,
        }
        if nfoods == 27:
            runs_27.append(entry)
        elif nfoods == 6:
            runs_6.append(entry)
    return runs_27, runs_6


def _solver_comparison_variant_b(data):
    """Extract Variant B rows (skipping SKIPPED entries) from solver_comparison data."""
    rows = [
        r
        for r in data
        if r.get("variant") == "B"
        and r.get("status") != "SKIPPED"
        and r.get("n_farms") in STUDY2B_FARM_SET
    ]
    return rows


# ============================================================================
# Study 2.B wall-clock time comparison — Track A (27-food)
# ============================================================================

# Style definitions for three-group comparison
THREE_GROUP_STYLES = {
    # Gurobi full
    "Gurobi full (27-food)":        {"color": "#e41a1c", "ls": "-",  "marker": "o", "lw": 2, "zorder": 5},
    "Gurobi full (6-food)":         {"color": "#d62728", "ls": ":",  "marker": "o", "lw": 2, "zorder": 5},
    # QPU hierarchical
    "QPU 27-food":                   {"color": "#ff7f0e", "ls": "-",  "marker": "D", "lw": 2, "zorder": 4},
    "QPU 6-food":                    {"color": "#e6550d", "ls": ":",  "marker": "D", "lw": 2, "zorder": 4},
    # Gurobi decomposed hierarchical
    "Gurobi decomp. Hier":           {"color": "#377eb8", "ls": "--", "marker": "s", "lw": 1.5},
    "Gurobi decomp. ST(5)":          {"color": "#4daf4a", "ls": "--", "marker": "^", "lw": 1.5},
    "Gurobi decomp. Clique (6-food)":{"color": "#1f77b4", "ls": "-.", "marker": "s", "lw": 1.5},
    "Gurobi decomp. ST(5) (6-food)": {"color": "#2ca02c", "ls": "-.", "marker": "^", "lw": 1.5},
}


def _sort_xy(xs, ys):
    pairs = sorted(zip(xs, ys))
    return (list(zip(*pairs)) if pairs else ([], []))


def plot_study2b_time_track_a(data_solver, output_dir=None):
    """Track A: wall-clock time comparison for 27-food Variant B.

    Groups: Gurobi full | QPU 27-food | Gurobi decomposed hierarchical.
    """
    out = output_dir or OUT_DIR
    fig, ax = plt.subplots(figsize=(9, 5.5))

    b_rows = _solver_comparison_variant_b(data_solver)

    # --- Gurobi full (27-food) from solver_comparison ---
    gf_xs, gf_ys = [], []
    for r in b_rows:
        if r["solver"] == "Gurobi_full" and r["decomposition"] == "none":
            gf_xs.append(r["n_farms"]); gf_ys.append(r["wall_time"])
    if gf_xs:
        xs_s, ys_s = _sort_xy(gf_xs, gf_ys)
        sty = THREE_GROUP_STYLES["Gurobi full (27-food)"]
        ax.plot(xs_s, ys_s, label="Gurobi full", markersize=6, **sty)

    # --- QPU 27-food ---
    qpu27 = load_qpu_data_27food()
    if qpu27:
        xs_q = [r["n_farms"] for r in sorted(qpu27, key=lambda x: x["n_farms"])]
        ys_q = [r["wall_time"] for r in sorted(qpu27, key=lambda x: x["n_farms"])]
        sty = THREE_GROUP_STYLES["QPU 27-food"]
        ax.plot(xs_q, ys_q, label="QPU (27-crop, hierarchical)", markersize=6, **sty)

    # --- Gurobi decomposed hierarchical (spatial_grid, cluster=9) ---
    cl_xs, cl_ys = [], []
    for r in b_rows:
        if r["solver"] == "Gurobi_decomposed" and r["decomposition"] == STUDY2B_DECOMP_LABEL:
            cl_xs.append(r["n_farms"]); cl_ys.append(r["wall_time"])
    if cl_xs:
        xs_s, ys_s = _sort_xy(cl_xs, cl_ys)
        sty = THREE_GROUP_STYLES["Gurobi decomp. Hier"]
        ax.plot(xs_s, ys_s, label="Gurobi decomp. (hierarchical, spatial-grid 9)", markersize=5, **sty)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of plots (27 crops × 3 periods = 81F variables)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Study 2.B — Solve Time Comparison (27-crop, Track A)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fname = "fig_study2b_time_track_a.pdf"
    fig.savefig(out / fname)
    plt.close(fig)
    print(f"  -> {fname}")


# ============================================================================
# Study 2.B wall-clock time comparison — Track B (6-food + 27-food combined)
# ============================================================================

def plot_study2b_time_track_b(data_solver, output_dir=None):
    """Track B: wall-clock time comparison — both 6-food and 27-food Variant B.

    Combines: Gurobi full (6-food from timeout, 27-food from solver_comparison),
    QPU (6-food, 27-food), Gurobi decomposed 27-food (hierarchical).
    """
    out = output_dir or OUT_DIR
    fig, ax = plt.subplots(figsize=(10, 6))

    b_rows = _solver_comparison_variant_b(data_solver)
    gf27_timeout, gf6_timeout = load_gurobi_timeout_data()

    # --- Gurobi full 27-food (solver_comparison) ---
    gf_xs, gf_ys = [], []
    for r in b_rows:
        if r["solver"] == "Gurobi_full" and r["decomposition"] == "none":
            gf_xs.append(r["n_farms"]); gf_ys.append(r["wall_time"])
    if gf_xs:
        xs_s, ys_s = _sort_xy(gf_xs, gf_ys)
        sty = THREE_GROUP_STYLES["Gurobi full (27-food)"]
        ax.plot(xs_s, ys_s, label="Gurobi full (27-crop)", markersize=6, **sty)

    # --- Gurobi full 6-food (timeout verification) ---
    if gf6_timeout:
        xs_s, ys_s = _sort_xy(
            [r["n_farms"] for r in gf6_timeout],
            [r["wall_time"] for r in gf6_timeout],
        )
        sty = THREE_GROUP_STYLES["Gurobi full (6-food)"]
        ax.plot(xs_s, ys_s, label="Gurobi full (6-food, 200s)", markersize=6, **sty)

    # --- QPU 27-food ---
    qpu27 = load_qpu_data_27food()
    if qpu27:
        xs_q, ys_q = _sort_xy(
            [r["n_farms"] for r in qpu27], [r["wall_time"] for r in qpu27])
        sty = THREE_GROUP_STYLES["QPU 27-food"]
        ax.plot(xs_q, ys_q, label="QPU (27-crop, hierarchical)", markersize=6, **sty)

    # --- QPU 6-food ---
    qpu6 = load_qpu_data_6food()
    if qpu6:
        xs_q, ys_q = _sort_xy(
            [r["n_farms"] for r in qpu6], [r["wall_time"] for r in qpu6])
        sty = THREE_GROUP_STYLES["QPU 6-food"]
        ax.plot(xs_q, ys_q, label="QPU (6-food, hierarchical)", markersize=6, **sty)

    # --- Gurobi decomposed hierarchical (27-food) ---
    cl_xs, cl_ys = [], []
    for r in b_rows:
        if r["solver"] == "Gurobi_decomposed" and r["decomposition"] == STUDY2B_DECOMP_LABEL:
            cl_xs.append(r["n_farms"]); cl_ys.append(r["wall_time"])
    if cl_xs:
        xs_s, ys_s = _sort_xy(cl_xs, cl_ys)
        sty = THREE_GROUP_STYLES["Gurobi decomp. Hier"]
        ax.plot(xs_s, ys_s, label="Gurobi decomp. hierarchical (27-crop)", markersize=5, **sty)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of plots")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Study 2.B — Solve Time Comparison (6-food + 27-crop, Track B)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fname = "fig_study2b_time_track_b.pdf"
    fig.savefig(out / fname)
    plt.close(fig)
    print(f"  -> {fname}")


# ============================================================================
# Study 2.B solution quality comparison — Track A (27-food within solver_comparison)
# ============================================================================

def plot_study2b_quality_track_a(data_solver, output_dir=None):
    """Track A: solution quality comparison for 27-food Variant B.

    Two sub-panels:
      Left:  Absolute objective (Gurobi full vs decomposed, same scenarios)
      Right: Decomposed/full ratio + QPU benefit/Gurobi ratio (from timeout data)
    """
    out = output_dir or OUT_DIR
    b_rows = _solver_comparison_variant_b(data_solver)

    # Build lookup for Gurobi full objective
    full_obj = {}
    for r in b_rows:
        if r["solver"] == "Gurobi_full" and r["decomposition"] == "none":
            if r.get("objective") is not None:
                full_obj[r["n_farms"]] = r["objective"]

    # Gurobi decomposed objectives
    decomp_hier = {}
    for r in b_rows:
        if r["solver"] != "Gurobi_decomposed" or r.get("objective") is None:
            continue
        if r["decomposition"] == STUDY2B_DECOMP_LABEL:
            decomp_hier[r["n_farms"]] = r["objective"]

    # QPU vs Gurobi timeout (scenario-matched comparison)
    qpu27 = load_qpu_data_27food()
    gf27_timeout, _ = load_gurobi_timeout_data()
    timeout_by_plots = {r["n_farms"]: r["objective"] for r in gf27_timeout
                        if r.get("objective") is not None}

    fig, (ax_abs, ax_ratio) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: absolute objectives (Gurobi scale, 27-food solver_comparison) ---
    if full_obj:
        plots_sorted = sorted(full_obj.keys())
        sty = THREE_GROUP_STYLES["Gurobi full (27-food)"]
        ax_abs.plot(plots_sorted, [full_obj[f] for f in plots_sorted],
                    label="Gurobi full", markersize=6, **sty)

    if decomp_hier:
        xs_s, ys_s = _sort_xy(list(decomp_hier.keys()), list(decomp_hier.values()))
        sty = THREE_GROUP_STYLES["Gurobi decomp. Hier"]
        ax_abs.plot(xs_s, ys_s, label="Gurobi decomp. (hierarchical)", markersize=5, **sty)

    ax_abs.set_xscale("log")
    ax_abs.set_yscale("log")
    ax_abs.set_xlabel("Number of plots")
    ax_abs.set_ylabel("MIQP objective value")
    ax_abs.set_title("Solution Quality — Gurobi full vs decomposed (27-crop)")
    ax_abs.legend(framealpha=0.9)
    ax_abs.grid(True, which="both", alpha=0.3)

    # --- Right: quality ratio relative to formulation-specific Gurobi full ---
    ratio_plots_hier, ratio_hier = [], []
    for nf, do in sorted(decomp_hier.items()):
        if nf in full_obj and full_obj[nf]:
            ratio_plots_hier.append(nf)
            ratio_hier.append(do / full_obj[nf])

    # QPU benefit / Gurobi timeout objective (scenario-matched)
    qpu_plots, qpu_ratios = [], []
    for r in sorted(qpu27, key=lambda x: x["n_farms"]):
        nf = r["n_farms"]
        if nf in timeout_by_plots and timeout_by_plots[nf]:
            qpu_plots.append(nf)
            qpu_ratios.append(r["benefit"] / timeout_by_plots[nf])

    if ratio_hier:
        sty = THREE_GROUP_STYLES["Gurobi decomp. Hier"]
        ax_ratio.plot(ratio_plots_hier, ratio_hier,
                      label="Gurobi decomp. hierarchical / full", markersize=5, **sty)

    if qpu_ratios:
        sty = THREE_GROUP_STYLES["QPU 27-food"]
        ax_ratio.plot(qpu_plots, qpu_ratios,
                      label="QPU benefit / Gurobi MIQP†", markersize=6, **sty)

    ax_ratio.axhline(1.0, color="#888", ls=":", lw=1)
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("Number of plots")
    ax_ratio.set_ylabel("Objective ratio (vs. Gurobi full)")
    ax_ratio.set_title("Quality Ratio — decomposed / full\n†QPU benefit / Gurobi MIQP (diff. scales)")
    ax_ratio.legend(framealpha=0.9, fontsize=8)
    ax_ratio.grid(True, which="both", alpha=0.3)

    fig.suptitle("Study 2.B — Solution Quality Comparison (27-crop, Track A)", y=1.01)
    fig.tight_layout()
    fname = "fig_study2b_quality_track_a.pdf"
    fig.savefig(out / fname)
    plt.close(fig)
    print(f"  -> {fname}")


# ============================================================================
# Study 2.B quality comparison — Track B adds 6-food QPU vs Gurobi
# ============================================================================

def plot_study2b_quality_track_b(data_solver, output_dir=None):
    """Track B: quality comparison — 27-food (solver_comparison) + 6-food (timeout)."""
    out = output_dir or OUT_DIR
    b_rows = _solver_comparison_variant_b(data_solver)

    full_obj_27 = {r["n_farms"]: r["objective"]
                   for r in b_rows
                   if r["solver"] == "Gurobi_full" and r["decomposition"] == "none"
                   and r.get("objective") is not None}

    gf27_timeout, gf6_timeout = load_gurobi_timeout_data()
    timeout27 = {r["n_farms"]: r["objective"] for r in gf27_timeout if r.get("objective") is not None}
    timeout6 = {r["n_farms"]: r["objective"] for r in gf6_timeout if r.get("objective") is not None}

    qpu27 = load_qpu_data_27food()
    qpu6 = load_qpu_data_6food()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: absolute objectives (solver_comparison + timeout formulations) ---
    ax = axes[0]
    if full_obj_27:
        plots_s = sorted(full_obj_27.keys())
        sty = THREE_GROUP_STYLES["Gurobi full (27-food)"]
        ax.plot(plots_s, [full_obj_27[f] for f in plots_s],
                label="Gurobi full 27-crop (solver_comp)", markersize=5, **sty)

    if timeout6:
        xs_s, ys_s = _sort_xy(list(timeout6.keys()), list(timeout6.values()))
        sty = THREE_GROUP_STYLES["Gurobi full (6-food)"]
        ax.plot(xs_s, ys_s, label="Gurobi full 6-food (timeout)", markersize=5, **sty)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of plots")
    ax.set_ylabel("MIQP objective value")
    ax.set_title("Absolute Quality: Gurobi full (both formulations)")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # --- Right: QPU quality vs Gurobi full (per formulation, same scenario) ---
    ax = axes[1]
    # QPU 27-food vs Gurobi 27-food timeout (scenario-matched)
    qpu27_plots, qpu27_ratios = [], []
    for r in sorted(qpu27, key=lambda x: x["n_farms"]):
        nf = r["n_farms"]
        ref = timeout27.get(nf)
        if ref:
            qpu27_plots.append(nf)
            qpu27_ratios.append(r["benefit"] / ref)

    # QPU 6-food vs Gurobi 6-food timeout
    qpu6_plots, qpu6_ratios = [], []
    for r in sorted(qpu6, key=lambda x: x["n_farms"]):
        nf = r["n_farms"]
        ref = timeout6.get(nf)
        if ref:
            qpu6_plots.append(nf)
            qpu6_ratios.append(r["benefit"] / ref)

    if qpu27_ratios:
        sty = THREE_GROUP_STYLES["QPU 27-food"]
        ax.plot(qpu27_plots, qpu27_ratios, label="QPU 27-crop / Gurobi MIQP†", markersize=6, **sty)
    if qpu6_ratios:
        sty = THREE_GROUP_STYLES["QPU 6-food"]
        ax.plot(qpu6_plots, qpu6_ratios, label="QPU 6-food / Gurobi MIQP†", markersize=6, **sty)

    ax.axhline(1.0, color="#888", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Number of plots")
    ax.set_ylabel("QPU benefit / Gurobi MIQP objective†")
    ax.set_title("QPU vs Gurobi Quality Ratio (both formulations)\n†Different normalizations — ratio > 1 ≠ QPU outperforms")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Study 2.B — Solution Quality Comparison (Track B: 6-food + 27-crop)", y=1.01)
    fig.tight_layout()
    fname = "fig_study2b_quality_track_b.pdf"
    fig.savefig(out / fname)
    plt.close(fig)
    print(f"  -> {fname}")



# ============================================================================
# Study 2.B standalone absolute quality — Gurobi decomposed vs full
# ============================================================================

def plot_study2b_quality_decomposed_abs(data_solver, output_dir=None):
    """Absolute MIQP objective: Gurobi full vs hierarchical decomposed for Variant B.

    Single-panel with raw objective values on log-log axes so scale differences
    between decompositions are immediately visible.
    """
    out = output_dir or OUT_DIR
    b_rows = _solver_comparison_variant_b(data_solver)

    full_obj  = {}
    hier_obj = {}
    for r in b_rows:
        if r.get("objective") is None:
            continue
        nf = r["n_farms"]
        if r["solver"] == "Gurobi_full" and r["decomposition"] == "none":
            full_obj[nf] = r["objective"]
        elif r["solver"] == "Gurobi_decomposed":
            if r["decomposition"] == STUDY2B_DECOMP_LABEL:
                hier_obj[nf] = r["objective"]

    fig, ax = plt.subplots(figsize=(8, 5))

    if full_obj:
        xs_s, ys_s = _sort_xy(list(full_obj.keys()), list(full_obj.values()))
        sty = THREE_GROUP_STYLES["Gurobi full (27-food)"]
        ax.plot(xs_s, ys_s, label="Gurobi full", markersize=6, **sty)

    if hier_obj:
        xs_s, ys_s = _sort_xy(list(hier_obj.keys()), list(hier_obj.values()))
        sty = THREE_GROUP_STYLES["Gurobi decomp. Hier"]
        ax.plot(xs_s, ys_s, label="Gurobi decomp. (hierarchical)", markersize=5, **sty)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of plots (27 crops × 3 periods)")
    ax.set_ylabel("MIQP objective value")
    ax.set_title("Study 2.B — Solution Quality: Gurobi Full vs Decomposed (Variant B, 27-crop)")
    ax.legend(framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fname = "fig_study2b_quality_decomposed_abs.pdf"
    fig.savefig(out / fname)
    plt.close(fig)
    print(f"  -> {fname}")


# ============================================================================
# Study 2.B comprehensive healed + violations plot (like hybrid_performance)
# ============================================================================
# This is the "no space limit" version — mirrors the style of:
#   study1_hybrid_performance.pdf   (D-Wave CQM/BQM)
#   qpu_benchmark_comprehensive.pdf (QPU decomposition benchmark)
#
# Left panel:  wall-clock time (log-log)
# Right panel: |obj − Gurobi_full| with healed lines (dashed) + violation bars
#              on secondary axis — dual-Y layout identical to the PDF references.
# ============================================================================

def _extract_b_healed_series(data_solver):
    """Pull raw, healed, and violation data from solver_comparison Variant B rows.

    Returns dict keyed by label → {n_farms: {raw, healed, viols, time}}
    """
    b_rows = _solver_comparison_variant_b(data_solver)
    # Full Gurobi reference (for gap calculation)
    full_obj = {r["n_farms"]: r["objective"]
                for r in b_rows
                if r["solver"] == "Gurobi_full"
                and r["decomposition"] == "none"
                and r.get("objective") is not None}

    series = {}

    # --- Gurobi full ---
    gf_data = {}
    for r in b_rows:
        if r["solver"] != "Gurobi_full" or r["decomposition"] != "none":
            continue
        nf = r["n_farms"]
        gf_data[nf] = {
            "raw": r.get("objective"),
            "healed": r.get("healed_objective"),
            "viols": r.get("violations", 0) or 0,
            "time": r.get("wall_time", 0),
        }
    if gf_data:
        series["Gurobi (Full)"] = gf_data

    # --- Gurobi decomposed hierarchical ---
    cl_data = {}
    for r in b_rows:
        if r["solver"] != "Gurobi_decomposed" or r["decomposition"] != STUDY2B_DECOMP_LABEL:
            continue
        nf = r["n_farms"]
        cl_data[nf] = {
            "raw": r.get("objective"),
            "healed": r.get("healed_objective"),
            "viols": r.get("violations", 0) or 0,
            "time": r.get("wall_time", 0),
        }
    if cl_data:
        series["Gurobi decomp. (Hier)"] = cl_data

    # --- QPU 27-food ---
    qpu27 = load_qpu_data_27food()
    if qpu27:
        qpu_data = {}
        for r in sorted(qpu27, key=lambda x: x["n_farms"]):
            nf = r["n_farms"]
            qpu_data[nf] = {
                "raw": r["benefit"],
                "healed": None,          # QPU: no separate healed in current data
                "viols": r.get("violations", 0) or 0,
                "time": r["wall_time"],
            }
        series["QPU (27-crop, hier.)"] = qpu_data

    return full_obj, series


# Color + style palette matching the reference PDFs
_HEALED_STYLES = {
    "Gurobi (Full)":            {"color": "#e41a1c", "marker": "o", "lw": 2},
    "Gurobi decomp. (Hier)":    {"color": "#377eb8", "marker": "s", "lw": 1.5},
    "QPU (27-crop, hier.)":     {"color": "#ff7f0e", "marker": "D", "lw": 2},
}
_VIOL_COLORS = {
    "Gurobi (Full)":            "#fb9a99",
    "Gurobi decomp. (Hier)":    "#a6cee3",
    "QPU (27-crop, hier.)":     "#fdbf6f",
}


def plot_study2b_healed_violations(data_solver, output_dir=None):
    """Comprehensive three-panel plot for Study 2.B.

    Panel 1: wall-clock time (log-log).
    Panel 2: objective values (all 4 traces) with violation histograms on a
             secondary axis.
    Panel 3: healed gap to healed Gurobi full baseline (comparison methods).
    """
    out = output_dir or OUT_DIR

    full_obj, series = _extract_b_healed_series(data_solver)
    if not series:
        print("  SKIP fig_study2b_healed_violations.pdf (no data)")
        return

    fig, (ax_t, ax_obj, ax_healed) = plt.subplots(1, 3, figsize=(20, 6))
    ax_v = ax_obj.twinx()  # secondary axis for violations

    labels_order = list(_HEALED_STYLES.keys())

    all_plots = sorted({nf for s in series.values() for nf in s})
    n_series = sum(1 for k in labels_order if k in series)
    bar_offsets = np.linspace(-0.30, 0.30, max(n_series, 1))

    compare_labels = [
            "Gurobi decomp. (Hier)",
        "QPU (27-crop, hier.)",
    ]

    # Healed baseline is Gurobi full healed if available, else raw.
    full_healed = {}
    full_series = series.get("Gurobi (Full)", {})
    for nf, v in full_series.items():
        healed_ref = v.get("healed")
        if healed_ref is None:
            healed_ref = v.get("raw")
        if healed_ref is not None:
            full_healed[nf] = healed_ref

    # ── Left: time ─────────────────────────────────────────────────────────
    for label in labels_order:
        if label not in series:
            continue
        sty = _HEALED_STYLES[label]
        data = series[label]
        xs_s, ys_s = _sort_xy(list(data.keys()), [v["time"] for v in data.values()])
        clipped = [(x, y) for x, y in zip(xs_s, ys_s) if 20 <= x <= 200]
        if clipped:
            xs_s, ys_s = zip(*clipped)
            ax_t.plot(xs_s, ys_s, label=label, markersize=5,
                        color=sty["color"], marker=sty["marker"], lw=sty["lw"])

    ax_t.set_xscale("log"); ax_t.set_yscale("log")
    ax_t.set_xlabel("Number of plots"); ax_t.set_ylabel("Wall-clock time (s)")
    ax_t.set_title("Solve Time")
    ax_t.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax_t.grid(True, which="both", alpha=0.3)

    # ── Middle: objective values + violations ──────────────────────────────
    bar_idx = 0
    for label in labels_order:
        if label not in series:
            continue
        sty = _HEALED_STYLES[label]
        data = series[label]
        xs, ys, viol_xs, viol_ys = [], [], [], []
        for nf in sorted(data.keys()):
            if not (20 <= nf <= 200):          # ← clip panel 2
                continue
            raw = data[nf].get("raw")
            viols = data[nf].get("viols", 0) or 0
            if raw is None:
                continue
            xs.append(nf)
            ys.append(raw)
            if viols > 0:
                viol_xs.append(nf)
                viol_ys.append(viols)
        if xs:
            ax_obj.plot(
                xs, ys,
                color=sty["color"], marker=sty["marker"],
                ls="-", lw=sty["lw"], markersize=5,
                label=label,
            )

        # violation bars
        if viol_xs and viol_ys:
            offset_factor = 10 ** bar_offsets[bar_idx]
            bar_x = [nf * offset_factor for nf in viol_xs]
            bar_w = [nf * offset_factor * 0.14 for nf in viol_xs]
            ax_v.bar(bar_x, viol_ys, width=bar_w,
                     color=_VIOL_COLORS.get(label, "#ccc"), alpha=0.55,
                     label=f"{label} viols")

        bar_idx += 1

    ax_obj.set_xscale("log")
    ax_obj.set_yscale("symlog", linthresh=1e-4)
    ax_obj.set_ylim(-2.0, 1e3)                 # ← upper y-limit 1e3
    ax_obj.set_xlim(20, 220)
    ax_obj.set_xlabel("Number of plots")
    ax_obj.set_ylabel("Objective value", fontsize=9)
    ax_obj.set_title("Objective + Constraint Violations")
    ax_obj.grid(True, which="both", alpha=0.3)
    #ax_obj.set_xticks([20, 25, 50, 100, 200])
    #ax_obj.set_xticklabels(["20", "25", "50", "100", "200"], rotation=30, ha="right")

    # Secondary Y-axis for violations
    ax_v.set_ylabel("Constraint Violations")
    ax_v.set_xscale("log")
    ax_v.set_yscale("log")
    ax_v.set_ylim(1, 1e4)

    # ── Right: healed gap to healed Gurobi full baseline ──────────────────
    for label in compare_labels:
        if label not in series:
            continue
        sty = _HEALED_STYLES[label]
        data = series[label]
        xs, ys = [], []
        for nf in sorted(data.keys()):
            if not (20 <= nf <= 200):          # ← clip panel 3
                continue
            v = data[nf]
            healed = v.get("healed")
            if healed is None:
                healed = v.get("raw")          # fallback for QPU
            ref_h = full_healed.get(nf)
            if healed is None or ref_h is None:
                continue
            xs.append(nf)
            ys.append(abs(healed - ref_h))

        if xs:
            h_label = f"{label} (healed)"
            if label == "QPU (27-crop, hier.)":
                h_label = "QPU (healed n/a, raw used)"
            ax_healed.plot(
                xs, ys,
                color=sty["color"], marker=sty["marker"],
                ls="--", lw=sty["lw"], markersize=5, alpha=0.9,
                label=h_label,
            )

    ax_healed.axhline(0, color="#888", ls=":", lw=1)
    ax_healed.set_xscale("log")
    ax_healed.set_yscale("symlog", linthresh=1e-4)
    ax_healed.set_ylim(bottom=0, top=1e3)      # ← upper y-limit 1e3
    ax_healed.set_xlim(20, 220)                # ← clip panel 3 axis
    ax_healed.set_xlabel("Number of plots")
    ax_healed.set_ylabel("|Healed objective − Gurobi full healed|", fontsize=9)
    ax_healed.set_title("Healed Gap vs Healed Baseline")
    ax_healed.grid(True, which="both", alpha=0.3)
    #ax_healed.set_xticks([20, 25, 50, 100, 200])
    #ax_healed.set_xticklabels(["20", "25", "50", "100", "200"], rotation=30, ha="right")

    # Combined legend at bottom
    lines_obj, labels_obj = ax_obj.get_legend_handles_labels()
    lines_v, labels_v = ax_v.get_legend_handles_labels()
    lines_h, labels_h = ax_healed.get_legend_handles_labels()
    fig.legend(
        lines_obj + lines_v + lines_h,
        labels_obj + labels_v + labels_h,
        loc="lower center", ncol=5, framealpha=0.9, fontsize=7,
        bbox_to_anchor=(0.5, -0.13),
    )

    fig.suptitle(
        "Study 2.B — QPU Decomposition Benchmark — Comprehensive (Variant B, 27-crop)",
        fontsize=11,
    )
    fig.tight_layout()
    fname = "fig_study2b_healed_violations.pdf"
    fig.savefig(out / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {fname}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Generating plots...")

    # Load data
    decomp_file = OUT_DIR / "decomposition_scaling_results.json"
    solver_file = OUT_DIR / "solver_comparison_results.json"

    if decomp_file.exists():
        with open(decomp_file) as f:
            data_decomp = json.load(f)

        plot_decomposition_time(data_decomp, "A", "Variant A (27-crop BP)", "fig_decomp_time_A.pdf")
        plot_decomposition_time(data_decomp, "B", "Variant B (Rotation)", "fig_decomp_time_B.pdf")
        plot_partition_count(data_decomp, "A", "Variant A", "fig_partition_count_A.pdf")
        plot_partition_count(data_decomp, "B", "Variant B", "fig_partition_count_B.pdf")
        plot_max_partition_size(data_decomp, "A", "Variant A", "fig_max_part_size_A.pdf")
        plot_max_partition_size(data_decomp, "B", "Variant B", "fig_max_part_size_B.pdf")
    else:
        print(f"  SKIP decomposition plots ({decomp_file} not found)")

    if solver_file.exists():
        with open(solver_file) as f:
            data_solver = json.load(f)

        # Combined plots showing ALL decomposition methods together (main figures)
        plot_solver_time_all(data_solver, "A", "fig_solver_time_A_all.pdf")
        plot_solver_quality_all(data_solver, "A", "fig_solver_quality_A_all.pdf")
        plot_solver_time_all(data_solver, "B", "fig_solver_time_B_all.pdf")
        plot_solver_quality_all(data_solver, "B", "fig_solver_quality_B_all.pdf")

        # Per-decomposition plots (supplementary)
        for decomp in ["PlotBased", "Multilevel(5)", "HybridGrid(5,9)"]:
            safe = decomp.replace("(", "").replace(")", "").replace(",", "_")
            plot_solver_time(data_solver, "A", decomp, f"fig_solver_time_A_{safe}.pdf")
            plot_solver_quality(data_solver, "A", decomp, f"fig_solver_quality_A_{safe}.pdf")

        for decomp in [STUDY2B_DECOMP_LABEL]:
            safe = decomp.replace("(", "").replace(")", "").replace(",", "_")
            plot_solver_time(data_solver, "B", decomp, f"fig_solver_time_B_{safe}.pdf")
            plot_solver_quality(data_solver, "B", decomp, f"fig_solver_quality_B_{safe}.pdf")

        plot_combined_panel(data_solver, "fig_combined_panel.pdf")

        # --- Study 2.B three-group comparison (Track A: 27-food) ---
        print("\nGenerating Study 2.B three-group comparison plots...")
        plot_study2b_time_track_a(data_solver)
        plot_study2b_quality_track_a(data_solver)

        # --- Study 2.B Track B: 6-food + 27-food combined ---
        print("\nGenerating Study 2.B Track B (6-food + 27-food) plots...")
        plot_study2b_time_track_b(data_solver)
        plot_study2b_quality_track_b(data_solver)

        # --- Study 2.B: standalone absolute quality for Gurobi decomposed ---
        print("\nGenerating Study 2.B absolute quality (decomposed vs full)...")
        plot_study2b_quality_decomposed_abs(data_solver)

        # --- Study 2.B: comprehensive healed+violations plot (no space limit) ---
        print("\nGenerating Study 2.B comprehensive healed+violations plot...")
        plot_study2b_healed_violations(data_solver)
    else:
        print(f"  SKIP solver plots ({solver_file} not found)")

    print("Done.")
