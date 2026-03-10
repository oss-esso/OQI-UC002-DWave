#!/usr/bin/env python3
"""
Generate all plots from the decomposition scaling and solver comparison benchmarks.

Reads:
  - decomposition_scaling_results.json
  - solver_comparison_results.json

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
    "Clique(farm-by-farm)": "#1f77b4",
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
    ax.set_xlabel("Number of farms")
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
    ax.set_xlabel("Number of farms")
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
    ax.set_xlabel("Number of farms")
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
    ax.set_xlabel("Number of farms")
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
    ax.set_xlabel("Number of farms")
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
    ax.set_xlabel("Number of farms")
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
    ax.set_xlabel("Number of farms")
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
    ax.set_xlabel("Farms"); ax.set_ylabel("Time (s)")
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
    ax.set_xlabel("Farms"); ax.set_ylabel("Objective")
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
    ax.set_xlabel("Farms"); ax.set_ylabel("Time (s)")
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
    ax.set_xlabel("Farms"); ax.set_ylabel("Objective (energy)")
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

        for decomp in ["Clique", "SpatialTemporal(5)"]:
            safe = decomp.replace("(", "").replace(")", "").replace(",", "_")
            plot_solver_time(data_solver, "B", decomp, f"fig_solver_time_B_{safe}.pdf")
            plot_solver_quality(data_solver, "B", decomp, f"fig_solver_quality_B_{safe}.pdf")

        plot_combined_panel(data_solver, "fig_combined_panel.pdf")
    else:
        print(f"  SKIP solver plots ({solver_file} not found)")

    print("Done.")
