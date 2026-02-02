#!/usr/bin/env python3
"""
Single-Plot PDF Generator for OQI-UC002-DWave Project Report.

This script generates individual PDF files for each subplot from the original
multi-panel figures. All plots use significantly larger font sizes for better
readability in presentations and reports.

Output folder: phase3_single_plots/

Author: OQI-UC002-DWave Project
Date: 2025-02-02
"""

from __future__ import annotations

import itertools
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

# =============================================================================
# IMPORT DATA LOADING AND COLOR DEFINITIONS FROM ORIGINAL SCRIPT
# =============================================================================

# We import these from the original script to maintain consistency
from generate_all_report_plots import (
    # Color palettes
    QUALITATIVE_COLORS,
    SOLVER_COLORS,
    FORMULATION_COLORS,
    METHOD_COLORS,
    MARKERS,
    # Data loading functions
    load_qpu_hierarchical,
    load_gurobi_60s,
    load_gurobi_300s,
    DEFAULT_QPU_HIER_PATH,
    DEFAULT_GUROBI_60S_PATH,
    DEFAULT_GUROBI_300S_FILE,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    # Data preparation functions
    load_violation_impact_data,
    load_gap_deep_dive_data,
    _load_qpu_benchmark_data,
    _extract_benchmark_metrics,
    _BENCHMARK_COLORS,
    _BENCHMARK_MARKERS,
)

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

SINGLE_PLOT_OUTPUT_DIR = PROJECT_ROOT / "phase3_single_plots"

# =============================================================================
# LARGE FONT STYLE CONFIGURATION
# =============================================================================


def setup_large_font_style() -> None:
    """
    Configure matplotlib for single plots with LARGE fonts.
    
    Font sizes are significantly increased for better readability
    in presentations and printed reports.
    """
    rcParams.update(
        {
            # Font configuration - LARGE sizes
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "font.size": 18,  # Base font size (was 12)
            "axes.labelsize": 22,  # Axis labels (was 14)
            "axes.titlesize": 26,  # Title (was 16)
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "legend.fontsize": 18,  # Legend (was 11)
            "xtick.labelsize": 18,  # X tick labels (was 11)
            "ytick.labelsize": 18,  # Y tick labels (was 11)
            # Figure quality
            "figure.dpi": 150,
            "figure.constrained_layout.use": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "savefig.transparent": False,
            # Grid configuration
            "axes.grid": True,
            "axes.grid.which": "major",
            "axes.grid.axis": "both",
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
            "grid.linewidth": 1.0,
            "axes.axisbelow": True,
            # Axes configuration
            "axes.linewidth": 1.8,
            "axes.labelpad": 12,
            "axes.titlepad": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            # Tick configuration
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.major.width": 1.8,
            "ytick.major.width": 1.8,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "xtick.minor.width": 1.2,
            "ytick.minor.width": 1.2,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            # Line and marker properties
            "lines.linewidth": 2.5,
            "lines.markersize": 10,
            "lines.markeredgewidth": 1.5,
            "patch.linewidth": 1.0,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "legend.shadow": False,
            # Error bars
            "errorbar.capsize": 4,
        }
    )


def save_single_plot(fig: plt.Figure, filename: str, output_dir: Path | None = None) -> Path:
    """Save a single plot as PDF only."""
    if output_dir is None:
        output_dir = SINGLE_PLOT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{filename}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"   [OK] Saved: {output_path}")
    plt.close(fig)
    return output_path


# =============================================================================
# DATA PREPARATION FUNCTIONS
# =============================================================================


def prepare_scaling_data_60s() -> pd.DataFrame:
    """Prepare merged QPU + Gurobi 60s data for scaling plots."""
    qpu_df = load_qpu_hierarchical()
    gurobi_df = load_gurobi_60s()
    
    results = []
    for _, row in qpu_df.iterrows():
        sc = row["scenario_name"]
        gur = gurobi_df[gurobi_df["scenario_name"] == sc]
        if len(gur) == 0:
            continue
        gur = gur.iloc[0]
        
        results.append({
            "scenario_name": sc,
            "n_vars": row["n_vars"],
            "n_farms": row["n_farms"],
            "n_foods": row["n_foods"],
            "formulation": row["formulation"],
            "qpu_objective": abs(row["objective_miqp"]),
            "gurobi_objective": gur["objective_miqp"],
            "qpu_total_time": row["total_wall_time"],
            "qpu_access_time": row["qpu_access_time"],
            "gurobi_time": gur["total_wall_time"],
            "gurobi_timeout": gur["hit_timeout"],
        })
    
    return pd.DataFrame(results).sort_values("n_vars").reset_index(drop=True)


def prepare_scaling_data_300s() -> pd.DataFrame:
    """Prepare merged QPU + Gurobi 300s data for scaling plots."""
    qpu_df = load_qpu_hierarchical()
    gurobi_df = load_gurobi_300s()
    
    results = []
    for _, row in qpu_df.iterrows():
        sc = row["scenario_name"]
        gur = gurobi_df[gurobi_df["scenario_name"] == sc]
        if len(gur) == 0:
            continue
        gur = gur.iloc[0]
        
        qpu_obj = abs(row["objective_miqp"])
        gur_obj = gur["objective_miqp"]
        gap = ((qpu_obj - gur_obj) / gur_obj * 100) if gur_obj > 0 else 0
        speedup = gur["solve_time"] / row["total_wall_time"] if row["total_wall_time"] > 0 else 0
        
        results.append({
            "scenario_name": sc,
            "n_vars": row["n_vars"],
            "n_farms": row["n_farms"],
            "n_foods": row["n_foods"],
            "formulation": row["formulation"],
            "qpu_objective": qpu_obj,
            "gurobi_objective": gur_obj,
            "gap": gap,
            "speedup": speedup,
            "qpu_total_time": row["total_wall_time"],
            "qpu_access_time": row["qpu_access_time"],
            "gurobi_time": gur["solve_time"],
            "gurobi_timeout": gur["hit_timeout"],
            "gurobi_mip_gap": gur.get("mip_gap", 0) * 100,
        })
    
    return pd.DataFrame(results).sort_values("n_vars").reset_index(drop=True)


# =============================================================================
# SECTION 1: COMPREHENSIVE SCALING (6 plots - vars and farms versions)
# =============================================================================


def plot_scaling_objectives_by_vars(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Objectives comparison by number of variables."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            ax.plot(
                form_df["n_vars"],
                form_df["gurobi_objective"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Gurobi)",
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
            ax.plot(
                form_df["n_vars"],
                form_df["qpu_objective"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (QPU)",
                linewidth=3,
                markersize=10,
                alpha=0.6,
                linestyle="--",
            )
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Objective Value (|abs|)")
    ax.set_title("Solution Quality: Classical vs Quantum (by Variables)")
    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "scaling_objectives_by_vars", output_dir)


def plot_scaling_objectives_by_farms(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Objectives comparison by number of farms."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_farms")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            ax.plot(
                form_df["n_farms"],
                form_df["gurobi_objective"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Gurobi)",
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
            ax.plot(
                form_df["n_farms"],
                form_df["qpu_objective"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (QPU)",
                linewidth=3,
                markersize=10,
                alpha=0.6,
                linestyle="--",
            )
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Objective Value (|abs|)")
    ax.set_title("Solution Quality: Classical vs Quantum (by Farms)")
    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "scaling_objectives_by_farms", output_dir)


def plot_scaling_time_comparison(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Plot 2: Time comparison bars - Gurobi vs QPU Total."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x_pos = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(
        x_pos - width / 2,
        df["gurobi_time"],
        width,
        label="Gurobi",
        color=SOLVER_COLORS["gurobi"],
        alpha=0.8,
    )
    bars2 = ax.bar(
        x_pos + width / 2,
        df["qpu_total_time"],
        width,
        label="QPU Total",
        color=SOLVER_COLORS["qpu"],
        alpha=0.8,
    )
    
    # Mark timeouts
    for i, (bar, timeout) in enumerate(zip(bars1, df["gurobi_timeout"])):
        if timeout:
            ax.annotate(
                "T",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=14,
                color="red",
                fontweight="bold",
            )
    
    ax.set_xlabel("Test Configuration")
    ax.set_ylabel("Solve Time (seconds)")
    ax.set_title("Solve Time: Gurobi vs QPU")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{row['formulation'][:3]}\n{row['n_vars']}v" for _, row in df.iterrows()],
        rotation=45,
        ha="right",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=60, color="red", linestyle="--", alpha=0.3, label="Timeout (60s)")
    
    return save_single_plot(fig, "scaling_time_comparison", output_dir)


def plot_scaling_qpu_breakdown_by_vars(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """QPU Time breakdown by number of variables."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            ax.plot(
                form_df["n_vars"],
                form_df["qpu_total_time"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Total)",
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
            ax.plot(
                form_df["n_vars"],
                form_df["qpu_access_time"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Pure QPU)",
                linewidth=3,
                markersize=10,
                alpha=0.6,
                linestyle="--",
            )
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("QPU Time Breakdown (by Variables)")
    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    
    return save_single_plot(fig, "scaling_qpu_breakdown_by_vars", output_dir)


def plot_scaling_qpu_breakdown_by_farms(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """QPU Time breakdown by number of farms."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_farms")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            ax.plot(
                form_df["n_farms"],
                form_df["qpu_total_time"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Total)",
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
            ax.plot(
                form_df["n_farms"],
                form_df["qpu_access_time"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Pure QPU)",
                linewidth=3,
                markersize=10,
                alpha=0.6,
                linestyle="--",
            )
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("QPU Time Breakdown (by Farms)")
    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    
    return save_single_plot(fig, "scaling_qpu_breakdown_by_farms", output_dir)


# =============================================================================
# SECTION 2: SPLIT FORMULATION ANALYSIS (12 plots - vars and farms versions)
# =============================================================================


def plot_split_objectives_by_vars(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Objective Values by formulation (by variables)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            ax.plot(
                form_df["n_vars"],
                form_df["gurobi_objective"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Gurobi)",
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
            ax.plot(
                form_df["n_vars"],
                form_df["qpu_objective"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (QPU)",
                linewidth=3,
                markersize=10,
                alpha=0.6,
                linestyle="--",
            )
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Objective Value")
    ax.set_title("Solution Quality (by Variables)")
    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    return save_single_plot(fig, "split_objectives_by_vars", output_dir)


def plot_split_objectives_by_farms(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Objective Values by formulation (by farms)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_farms")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            ax.plot(
                form_df["n_farms"],
                form_df["gurobi_objective"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Gurobi)",
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
            ax.plot(
                form_df["n_farms"],
                form_df["qpu_objective"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (QPU)",
                linewidth=3,
                markersize=10,
                alpha=0.6,
                linestyle="--",
            )
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Objective Value")
    ax.set_title("Solution Quality (by Farms)")
    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    return save_single_plot(fig, "split_objectives_by_farms", output_dir)


def plot_split_optimality_gap_by_vars(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Optimality Gap by variables."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            ax.plot(
                form_df["n_vars"],
                form_df["gap"],
                marker=MARKERS.get(formulation, "o"),
                color=FORMULATION_COLORS.get(formulation, "gray"),
                label=formulation,
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
    ax.axhline(y=100, color="orange", linestyle="--", alpha=0.5, label="100% gap", linewidth=2)
    ax.axhline(y=500, color="red", linestyle="--", alpha=0.5, label="500% gap", linewidth=2)
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("QPU Gap from Gurobi (%)")
    ax.set_title("Optimality Gap (by Variables)")
    ax.legend(loc="best", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    return save_single_plot(fig, "split_optimality_gap_by_vars", output_dir)


def plot_split_optimality_gap_by_farms(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Optimality Gap by farms."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_farms")
        if len(form_df) > 0:
            ax.plot(
                form_df["n_farms"],
                form_df["gap"],
                marker=MARKERS.get(formulation, "o"),
                color=FORMULATION_COLORS.get(formulation, "gray"),
                label=formulation,
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
    ax.axhline(y=100, color="orange", linestyle="--", alpha=0.5, label="100% gap", linewidth=2)
    ax.axhline(y=500, color="red", linestyle="--", alpha=0.5, label="500% gap", linewidth=2)
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("QPU Gap from Gurobi (%)")
    ax.set_title("Optimality Gap (by Farms)")
    ax.legend(loc="best", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    return save_single_plot(fig, "split_optimality_gap_by_farms", output_dir)


def plot_split_solve_time_by_vars(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Solve Time by variables."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            ax.plot(
                form_df["n_vars"],
                form_df["gurobi_time"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Gurobi)",
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
            ax.plot(
                form_df["n_vars"],
                form_df["qpu_total_time"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (QPU)",
                linewidth=3,
                markersize=10,
                alpha=0.6,
                linestyle="--",
            )
    ax.axhline(y=100, color="red", linestyle=":", alpha=0.5, label="Gurobi timeout (100s)", linewidth=2)
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Solve Time (seconds)")
    ax.set_title("Time Scaling (by Variables)")
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    return save_single_plot(fig, "split_solve_time_by_vars", output_dir)


def plot_split_solve_time_by_farms(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Solve Time by farms."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_farms")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            ax.plot(
                form_df["n_farms"],
                form_df["gurobi_time"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Gurobi)",
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
            ax.plot(
                form_df["n_farms"],
                form_df["qpu_total_time"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (QPU)",
                linewidth=3,
                markersize=10,
                alpha=0.6,
                linestyle="--",
            )
    ax.axhline(y=100, color="red", linestyle=":", alpha=0.5, label="Gurobi timeout (100s)", linewidth=2)
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Solve Time (seconds)")
    ax.set_title("Time Scaling (by Farms)")
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    return save_single_plot(fig, "split_solve_time_by_farms", output_dir)


def plot_split_speedup_by_vars(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Speedup by variables."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            ax.plot(
                form_df["n_vars"],
                form_df["speedup"],
                marker=MARKERS.get(formulation, "o"),
                color=FORMULATION_COLORS.get(formulation, "gray"),
                label=formulation,
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Break-even", linewidth=2)
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Speedup (Gurobi/QPU)")
    ax.set_title("Speedup Analysis (by Variables)")
    ax.legend(loc="best", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    return save_single_plot(fig, "split_speedup_by_vars", output_dir)


def plot_split_speedup_by_farms(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Speedup by farms."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_farms")
        if len(form_df) > 0:
            ax.plot(
                form_df["n_farms"],
                form_df["speedup"],
                marker=MARKERS.get(formulation, "o"),
                color=FORMULATION_COLORS.get(formulation, "gray"),
                label=formulation,
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Break-even", linewidth=2)
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Speedup (Gurobi/QPU)")
    ax.set_title("Speedup Analysis (by Farms)")
    ax.legend(loc="best", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    return save_single_plot(fig, "split_speedup_by_farms", output_dir)


def plot_split_pure_qpu_time_by_vars(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Pure QPU Time by variables."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            ax.scatter(
                form_df["n_vars"],
                form_df["qpu_access_time"],
                s=150,
                marker=MARKERS.get(formulation, "o"),
                color=color,
                alpha=0.8,
                label=f"{formulation}",
                edgecolors="black",
                linewidths=0.5,
            )
            if len(form_df) >= 2:
                coef = np.polyfit(form_df["n_vars"].values, form_df["qpu_access_time"].values, 1)
                x_fit = np.linspace(form_df["n_vars"].min(), form_df["n_vars"].max(), 100)
                y_fit = coef[0] * x_fit + coef[1]
                ax.plot(x_fit, y_fit, "--", color=color, alpha=0.7,
                        label=f"{formulation} fit: {coef[0]*1000:.4f}ms/var")
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Pure QPU Time (seconds)")
    ax.set_title("Pure QPU Time (by Variables)")
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "split_pure_qpu_time_by_vars", output_dir)


def plot_split_pure_qpu_time_by_farms(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Pure QPU Time by farms."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_farms")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            ax.scatter(
                form_df["n_farms"],
                form_df["qpu_access_time"],
                s=150,
                marker=MARKERS.get(formulation, "o"),
                color=color,
                alpha=0.8,
                label=f"{formulation}",
                edgecolors="black",
                linewidths=0.5,
            )
            if len(form_df) >= 2:
                coef = np.polyfit(form_df["n_farms"].values, form_df["qpu_access_time"].values, 1)
                x_fit = np.linspace(form_df["n_farms"].min(), form_df["n_farms"].max(), 100)
                y_fit = coef[0] * x_fit + coef[1]
                ax.plot(x_fit, y_fit, "--", color=color, alpha=0.7,
                        label=f"{formulation} fit: {coef[0]*1000:.1f}ms/farm")
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Pure QPU Time (seconds)")
    ax.set_title("Pure QPU Time (by Farms)")
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "split_pure_qpu_time_by_farms", output_dir)


def plot_split_gurobi_mip_gap_by_vars(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Gurobi MIP Gap by variables."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            ax.semilogy(
                form_df["n_vars"],
                form_df["gurobi_mip_gap"],
                marker=MARKERS.get(formulation, "o"),
                color=FORMULATION_COLORS.get(formulation, "gray"),
                label=formulation,
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
    ax.axhline(y=100, color="orange", linestyle="--", alpha=0.5, label="100% MIP gap")
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Gurobi MIP Gap (%)")
    ax.set_title("Classical Solver Difficulty (by Variables)")
    ax.legend(loc="best", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    return save_single_plot(fig, "split_gurobi_mip_gap_by_vars", output_dir)


def plot_split_gurobi_mip_gap_by_farms(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Gurobi MIP Gap by farms."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_farms")
        if len(form_df) > 0:
            ax.semilogy(
                form_df["n_farms"],
                form_df["gurobi_mip_gap"],
                marker=MARKERS.get(formulation, "o"),
                color=FORMULATION_COLORS.get(formulation, "gray"),
                label=formulation,
                linewidth=3,
                markersize=12,
                alpha=0.8,
            )
    ax.axhline(y=100, color="orange", linestyle="--", alpha=0.5, label="100% MIP gap")
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Gurobi MIP Gap (%)")
    ax.set_title("Classical Solver Difficulty (by Farms)")
    ax.legend(loc="best", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    return save_single_plot(fig, "split_gurobi_mip_gap_by_farms", output_dir)


# =============================================================================
# SECTION 3: OBJECTIVE GAP ANALYSIS (6 plots)
# =============================================================================


def plot_gap_absolute_comparison(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Gap Plot 1: Absolute Objective Comparison (bars)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(
        x - width / 2,
        df["gurobi_objective"],
        width,
        label="Gurobi (300s)",
        color=SOLVER_COLORS["gurobi"],
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        df["qpu_objective"],
        width,
        label="QPU Hier.",
        color=SOLVER_COLORS["qpu"],
        alpha=0.8,
    )
    
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Objective Value")
    ax.set_title("Objective Values: Gurobi vs QPU")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['n_vars']}" for _, row in df.iterrows()], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")
    
    return save_single_plot(fig, "gap_absolute_comparison", output_dir)


def plot_gap_ratio(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Gap Plot 2: Objective Ratio (QPU/Gurobi) bars."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ratio = df["qpu_objective"] / df["gurobi_objective"].replace(0, np.nan)
    colors_ratio = ["green" if r < 2 else "orange" if r < 5 else "red" for r in ratio.fillna(0)]
    
    ax.bar(range(len(df)), ratio.fillna(0), color=colors_ratio, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(y=1, color="black", linestyle="--", alpha=0.7, label="Equal")
    ax.axhline(y=2, color="orange", linestyle="--", alpha=0.5, label="2x ratio")
    ax.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="5x ratio")
    
    ax.set_xlabel("Scenario (by # variables)")
    ax.set_ylabel("Objective Ratio (QPU / Gurobi)")
    ax.set_title("QPU Objective / Gurobi Objective")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(
        [f"{row['formulation'][:3]}\n{row['n_vars']}" for _, row in df.iterrows()],
        rotation=45,
        ha="right",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    return save_single_plot(fig, "gap_ratio", output_dir)


def plot_gap_vs_mip_gap(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Gap Plot 3: Gap vs Gurobi MIP Gap correlation scatter."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for formulation in ["6-Food", "27-Food (Agg.)"]:
        form_df = df[df["formulation"] == formulation]
        ax.scatter(
            form_df["gurobi_mip_gap"],
            form_df["gap"],
            s=150,
            marker=MARKERS.get(formulation, "o"),
            color=FORMULATION_COLORS.get(formulation, "gray"),
            alpha=0.8,
            label=formulation,
            edgecolors="black",
            linewidths=0.5,
        )
    
    ax.set_xlabel("Gurobi MIP Gap (%)")
    ax.set_ylabel("QPU Gap from Gurobi (%)")
    ax.set_title("Correlation: Problem Hardness vs QPU Gap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    return save_single_plot(fig, "gap_vs_mip_gap", output_dir)


def plot_gap_6family_scaling(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Gap Plot 4: 6-Food detailed objective scaling."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    form_df = df[df["formulation"] == "6-Food"].sort_values("n_vars")
    
    if len(form_df) > 0:
        ax.plot(
            form_df["n_farms"],
            form_df["gurobi_objective"],
            "o-",
            color=SOLVER_COLORS["gurobi"],
            label="Gurobi",
            linewidth=3,
            markersize=12,
        )
        ax.plot(
            form_df["n_farms"],
            form_df["qpu_objective"],
            "s--",
            color=SOLVER_COLORS["qpu"],
            label="QPU",
            linewidth=3,
            markersize=12,
        )
    
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Objective Value")
    ax.set_title("6-Food: Objective Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "gap_6family_scaling", output_dir)


def plot_gap_27food_scaling(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Gap Plot 5: 27-Food (Agg.) detailed objective scaling."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    form_df = df[df["formulation"] == "27-Food (Agg.)"].sort_values("n_vars")
    
    if len(form_df) > 0:
        ax.plot(
            form_df["n_farms"],
            form_df["gurobi_objective"],
            "o-",
            color=SOLVER_COLORS["gurobi"],
            label="Gurobi",
            linewidth=3,
            markersize=12,
        )
        ax.plot(
            form_df["n_farms"],
            form_df["qpu_objective"],
            "s--",
            color=SOLVER_COLORS["qpu"],
            label="QPU",
            linewidth=3,
            markersize=12,
        )
    
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Objective Value")
    ax.set_title("27-Food: Objective Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "gap_27food_scaling", output_dir)


def plot_gap_summary_table(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Gap Plot 6: Summary Statistics Table."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    
    stats_6fam = df[df["formulation"] == "6-Food"]
    stats_27food = df[df["formulation"] == "27-Food (Agg.)"]
    
    table_data = [
        ["Metric", "6-Food", "27-Food (Agg.)"],
        ["Scenarios", f"{len(stats_6fam)}", f"{len(stats_27food)}"],
    ]
    
    # Variable range
    if len(stats_6fam) > 0:
        var_6fam = f'{stats_6fam["n_vars"].min()}-{stats_6fam["n_vars"].max()}'
    else:
        var_6fam = "N/A"
    if len(stats_27food) > 0:
        var_27food = f'{stats_27food["n_vars"].min()}-{stats_27food["n_vars"].max()}'
    else:
        var_27food = "N/A"
    table_data.append(["Variable Range", var_6fam, var_27food])
    
    # Avg QPU Gap
    gap_6fam = f'{stats_6fam["gap"].mean():.1f}%' if len(stats_6fam) > 0 else "N/A"
    gap_27food = f'{stats_27food["gap"].mean():.1f}%' if len(stats_27food) > 0 else "N/A"
    table_data.append(["Avg QPU Gap", gap_6fam, gap_27food])
    
    # Avg Gurobi MIP Gap
    mip_6fam = f'{stats_6fam["gurobi_mip_gap"].mean():.0f}%' if len(stats_6fam) > 0 else "N/A"
    mip_27food = f'{stats_27food["gurobi_mip_gap"].mean():.0f}%' if len(stats_27food) > 0 else "N/A"
    table_data.append(["Avg Gurobi MIP Gap", mip_6fam, mip_27food])
    
    # Avg Obj Ratio
    if len(stats_6fam) > 0:
        ratio_6fam = (stats_6fam["qpu_objective"] / stats_6fam["gurobi_objective"].replace(0, np.nan)).mean()
        ratio_6fam_str = f"{ratio_6fam:.2f}x" if not np.isnan(ratio_6fam) else "N/A"
    else:
        ratio_6fam_str = "N/A"
    if len(stats_27food) > 0:
        ratio_27food = (stats_27food["qpu_objective"] / stats_27food["gurobi_objective"].replace(0, np.nan)).mean()
        ratio_27food_str = f"{ratio_27food:.2f}x" if not np.isnan(ratio_27food) else "N/A"
    else:
        ratio_27food_str = "N/A"
    table_data.append(["Avg Obj Ratio", ratio_6fam_str, ratio_27food_str])
    
    # Gurobi Timeouts
    timeout_6fam = f'{stats_6fam["gurobi_timeout"].sum()}/{len(stats_6fam)}' if len(stats_6fam) > 0 else "N/A"
    timeout_27food = f'{stats_27food["gurobi_timeout"].sum()}/{len(stats_27food)}' if len(stats_27food) > 0 else "N/A"
    table_data.append(["Gurobi Timeouts", timeout_6fam, timeout_27food])
    
    # Avg QPU Time
    qpu_time_6fam = f'{stats_6fam["qpu_total_time"].mean():.1f}s' if len(stats_6fam) > 0 else "N/A"
    qpu_time_27food = f'{stats_27food["qpu_total_time"].mean():.1f}s' if len(stats_27food) > 0 else "N/A"
    table_data.append(["Avg QPU Time", qpu_time_6fam, qpu_time_27food])
    
    # Avg Gurobi Time
    gur_time_6fam = f'{stats_6fam["gurobi_time"].mean():.1f}s' if len(stats_6fam) > 0 else "N/A"
    gur_time_27food = f'{stats_27food["gurobi_time"].mean():.1f}s' if len(stats_27food) > 0 else "N/A"
    table_data.append(["Avg Gurobi Time", gur_time_6fam, gur_time_27food])
    
    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.3, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1.4, 2.0)
    
    # Color header
    for j in range(3):
        table[(0, j)].set_facecolor("#3498db")
        table[(0, j)].set_text_props(color="white", fontweight="bold", fontsize=18)
    
    ax.set_title("Summary by Formulation", pad=30)
    
    return save_single_plot(fig, "gap_summary_table", output_dir)


# =============================================================================
# SECTION 4: VIOLATION IMPACT (3 plots)
# =============================================================================


def plot_violation_rate(df: pd.DataFrame | None = None, output_dir: Path | None = None) -> Path:
    """Violation Plot 1: Violation rate by scenario."""
    if df is None:
        df = load_violation_impact_data()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ["red" if r > 10 else "orange" if r > 5 else "green" for r in df["violation_rate"]]
    ax.bar(range(len(df)), df["violation_rate"], color=colors, alpha=0.8, edgecolor="black")
    ax.set_xlabel("Scenario (sorted by size)")
    ax.set_ylabel("Violation Rate (%)")
    ax.set_title("One-Hot Violation Rate by Scenario")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{v:,}" for v in df["n_vars"]], rotation=45, ha="right")
    ax.axhline(
        y=df["violation_rate"].mean(),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Avg: {df['violation_rate'].mean():.1f}%",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    return save_single_plot(fig, "violation_rate", output_dir)


def plot_violation_gap_breakdown(df: pd.DataFrame | None = None, output_dir: Path | None = None) -> Path:
    """Violation Plot 2: Gap vs estimated violation impact."""
    if df is None:
        df = load_violation_impact_data()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["gap"], width, label="Total Gap", color="red", alpha=0.8)
    ax.bar(x + width / 2, df["estimated_lost_benefit"], width, label="Est. Violation Impact", color="blue", alpha=0.8)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Objective Units")
    ax.set_title("Gap vs Estimated Violation Impact")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:,}" for v in df["n_vars"]], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    return save_single_plot(fig, "violation_gap_breakdown", output_dir)


def plot_violation_adjusted_objective(df: pd.DataFrame | None = None, output_dir: Path | None = None) -> Path:
    """Violation Plot 3: Objective comparison (raw vs adjusted)."""
    if df is None:
        df = load_violation_impact_data()
    
    adj_obj = df["qpu_obj_abs"] - df["estimated_lost_benefit"]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["gurobi_obj"], width, label="Gurobi", color="green", alpha=0.8)
    ax.bar(x, df["qpu_obj_abs"], width, label="QPU (raw)", color="red", alpha=0.8)
    ax.bar(x + width, adj_obj, width, label="QPU (adjusted)", color="blue", alpha=0.8)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Objective Value")
    ax.set_title("Objective: Raw vs Violation-Adjusted")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:,}" for v in df["n_vars"]], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    return save_single_plot(fig, "violation_adjusted_objective", output_dir)


# =============================================================================
# SECTION 5: GAP DEEP DIVE (6 plots)
# =============================================================================


def plot_deepdive_objective_comparison(df: pd.DataFrame | None = None, output_dir: Path | None = None) -> Path:
    """Deep Dive Plot 1: Objective comparison (Gurobi vs Raw vs Corrected)."""
    if df is None:
        df = load_gap_deep_dive_data()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["gurobi"], width, label="Gurobi", color="green", alpha=0.8)
    ax.bar(x, df["qpu_abs"], width, label="|QPU| (raw)", color="red", alpha=0.8)
    ax.bar(x + width, df["comparable"], width, label="|QPU| (corrected)", color="blue", alpha=0.8)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Objective Value")
    ax.set_title("Objective Comparison: Raw vs Corrected")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in df["violations"]])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    return save_single_plot(fig, "deepdive_objective_comparison", output_dir)


def plot_deepdive_ratio_analysis(df: pd.DataFrame | None = None, output_dir: Path | None = None) -> Path:
    """Deep Dive Plot 2: Ratio analysis (raw vs corrected)."""
    if df is None:
        df = load_gap_deep_dive_data()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["ratio_raw"], 0.4, label="Raw ratio", color="red", alpha=0.8)
    ax.bar(x + 0.2, df["ratio_corrected"], 0.4, label="Corrected ratio", color="blue", alpha=0.8)
    ax.axhline(y=1.0, color="green", linestyle="--", label="Parity (1.0)", linewidth=2.5)
    ax.set_xlabel("Scenario (by violations)")
    ax.set_ylabel("Ratio (QPU / Gurobi)")
    ax.set_title("Ratio Analysis: Violations Have Minor Impact")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in df["violations"]])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    return save_single_plot(fig, "deepdive_ratio_analysis", output_dir)


def plot_deepdive_gap_attribution(df: pd.DataFrame | None = None, output_dir: Path | None = None) -> Path:
    """Deep Dive Plot 3: Gap attribution pie chart."""
    if df is None:
        df = load_gap_deep_dive_data()
    
    total_gap = (df["qpu_abs"] - df["gurobi"]).sum()
    violation_explained = df["potential_gain"].sum()
    pct_explained = violation_explained / total_gap * 100 if total_gap > 0 else 0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sizes = [pct_explained, 100 - pct_explained]
    labels = [f"Violations\n({pct_explained:.0f}%)", f"Other factors\n({100 - pct_explained:.0f}%)"]
    colors_pie = ["coral", "lightblue"]
    explode = (0.05, 0)
    
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors_pie,
        autopct="%1.0f%%",
        shadow=True,
        startangle=90,
        textprops={"fontsize": 18},
    )
    ax.set_title("Gap Attribution")
    
    return save_single_plot(fig, "deepdive_gap_attribution", output_dir)


def plot_deepdive_scatter(df: pd.DataFrame | None = None, output_dir: Path | None = None) -> Path:
    """Deep Dive Plot 4: QPU vs Gurobi scatter with parity line."""
    if df is None:
        df = load_gap_deep_dive_data()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(df["gurobi"], df["qpu_abs"], s=150, c="red", alpha=0.7, label="Raw", edgecolors="black")
    ax.scatter(df["gurobi"], df["comparable"], s=150, c="blue", alpha=0.7, label="Corrected", edgecolors="black")
    max_val = max(df["qpu_abs"].max(), df["gurobi"].max())
    ax.plot([0, max_val], [0, max_val], "g--", label="Parity", linewidth=2.5)
    ax.set_xlabel("Gurobi Objective")
    ax.set_ylabel("QPU Objective")
    ax.set_title("QPU vs Gurobi: Violation Correction Impact")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "deepdive_scatter", output_dir)


def plot_deepdive_violation_impact_by_scenario(df: pd.DataFrame | None = None, output_dir: Path | None = None) -> Path:
    """Deep Dive Plot 5: Violation impact percentage by scenario."""
    if df is None:
        df = load_gap_deep_dive_data()
    
    viol_pcts = []
    for _, row in df.iterrows():
        scenario_gap = row["qpu_abs"] - row["gurobi"]
        if scenario_gap > 0:
            viol_pct = row["potential_gain"] / scenario_gap * 100
        else:
            viol_pct = 0
        viol_pcts.append(viol_pct)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.bar(range(len(df)), viol_pcts, color="coral", alpha=0.8, edgecolor="black")
    ax.axhline(y=np.mean(viol_pcts), color="blue", linestyle="--", linewidth=2, label=f"Avg: {np.mean(viol_pcts):.1f}%")
    ax.set_xlabel("Scenario (by violations)")
    ax.set_ylabel("% of Gap Explained by Violations")
    ax.set_title("Violation Impact by Scenario")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{int(v)}" for v in df["violations"]])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 20)
    
    return save_single_plot(fig, "deepdive_violation_impact_by_scenario", output_dir)


def plot_deepdive_summary_findings(df: pd.DataFrame | None = None, output_dir: Path | None = None) -> Path:
    """Deep Dive Plot 6: Summary findings text box."""
    if df is None:
        df = load_gap_deep_dive_data()
    
    avg_raw_ratio = df["ratio_raw"].mean()
    avg_corrected_ratio = df["ratio_corrected"].mean()
    total_gap = (df["qpu_abs"] - df["gurobi"]).sum()
    violation_explained = df["potential_gain"].sum()
    pct_explained = violation_explained / total_gap * 100 if total_gap > 0 else 0
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")
    
    summary = f"""
DEEP DIVE FINDINGS

1. VIOLATION IMPACT IS MINOR
   - Violations explain only {pct_explained:.0f}% of gap
   - Correction improves ratio from {avg_raw_ratio:.2f}x to {avg_corrected_ratio:.2f}x
   - Still ~3.5x gap remains after correction

2. THE {100 - pct_explained:.0f}% UNEXPLAINED GAP
   Main causes (in likely order of importance):

   a) DECOMPOSITION APPROXIMATION
      Hierarchical method ≠ global optimization

   b) QUBO TRANSFORMATION
      Energy landscape differs from MIQP

   c) LOCAL MINIMA
      Quantum annealing may not find global min

   d) EMBEDDING NOISE
      Chain breaks and physical imperfections

3. KEY INSIGHT
   Fixing violations would NOT make QPU
   competitive with Gurobi on solution quality.
   The fundamental gap is algorithmic, not
   due to constraint satisfaction failures.

4. IMPLICATIONS
   - Post-processing repair has limited value
   - Better decomposition strategies needed
   - Consider hybrid classical-quantum approaches
"""
    
    ax.text(
        0.5,
        0.5,
        summary,
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    ax.set_title("Summary Findings")
    
    return save_single_plot(fig, "deepdive_summary_findings", output_dir)


# =============================================================================
# SECTION 6: QPU ADVANTAGE CORRECTED (6 plots)
# =============================================================================


def _prepare_qpu_advantage_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare data for QPU advantage plots."""
    qpu_df = load_qpu_hierarchical()
    gurobi_df = load_gurobi_300s()
    
    results = []
    for _, row in qpu_df.iterrows():
        sc = row["scenario_name"]
        gur_row = gurobi_df[gurobi_df["scenario_name"] == sc]
        if len(gur_row) == 0:
            continue
        gur_row = gur_row.iloc[0]
        
        n_foods = row["n_foods"]
        n_farms = row["n_farms"]
        n_periods = row["n_periods"]
        qpu_benefit = row["benefit"]
        gur_obj = gur_row["objective_miqp"]
        total_viols = row.get("total_violations", 0)
        total_slots = n_farms * n_periods
        
        results.append({
            "scenario": sc,
            "n_vars": row["n_vars"],
            "n_farms": n_farms,
            "n_foods": n_foods,
            "formulation": "27-Food (Agg.)" if n_foods == 27 else "6-Food",
            "qpu_obj_raw": row["objective_miqp"],
            "qpu_benefit": qpu_benefit,
            "gurobi_obj": gur_obj,
            "benefit_advantage": qpu_benefit - gur_obj,
            "benefit_ratio": qpu_benefit / gur_obj if gur_obj > 0 else 0,
            "violations": total_viols,
            "violation_rate": total_viols / total_slots * 100 if total_slots > 0 else 0,
            "qpu_wall_time": row.get("total_wall_time", 0),
            "qpu_pure_time": row.get("qpu_access_time", 0),
            "gurobi_time": gur_row["solve_time"],
            "gurobi_timeout": gur_row["hit_timeout"],
            "gurobi_mip_gap": gur_row.get("mip_gap", 0) * 100,
        })
    
    df = pd.DataFrame(results).sort_values("n_vars").reset_index(drop=True)
    df_6fam = df[df["formulation"] == "6-Food"]
    df_27food = df[df["formulation"] == "27-Food (Agg.)"]
    
    return df, df_6fam, df_27food


def plot_advantage_benefit_comparison(output_dir: Path | None = None) -> Path:
    """Advantage Plot 1: Benefit comparison (QPU vs Gurobi)."""
    df, _, _ = _prepare_qpu_advantage_data()
    
    C_QPU = "#1f77b4"
    C_QPU_DARK = "#0d4f8b"
    C_GUROBI = "#2ca02c"
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["gurobi_obj"], width, label="Gurobi",
           color=C_GUROBI, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, df["qpu_benefit"], width, label="QPU (Hierarchical)",
           color=C_QPU, alpha=0.85, edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel("Problem Size (Variables)")
    ax.set_ylabel("Benefit Value (higher = better)")
    ax.set_title("QPU Achieves Higher Benefit Than Gurobi")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:,}" for v in df["n_vars"]], rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    
    for i, row_ in df.iterrows():
        if row_["benefit_ratio"] > 1:
            ax.annotate(f"+{row_['benefit_ratio']:.1f}x",
                        xy=(i + width / 2, row_["qpu_benefit"]),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=14, color=C_QPU_DARK, fontweight="bold")
    
    return save_single_plot(fig, "advantage_benefit_comparison", output_dir)


def plot_advantage_benefit_ratio(output_dir: Path | None = None) -> Path:
    """Advantage Plot 2: Benefit ratio by formulation."""
    df, df_6fam, df_27food = _prepare_qpu_advantage_data()
    
    C_6FAM = "#3498db"
    C_27FOOD = "#e74c3c"
    C_NEUTRAL = "#7f7f7f"
    C_QPU = "#1f77b4"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(df_6fam["n_vars"], df_6fam["benefit_ratio"], s=180, c=C_6FAM,
               marker="o", label="6-Food", edgecolors="black", linewidths=0.5, alpha=0.8)
    ax.scatter(df_27food["n_vars"], df_27food["benefit_ratio"], s=180, c=C_27FOOD,
               marker="s", label="27-Food (Agg.)", edgecolors="black", linewidths=0.5, alpha=0.8)
    ax.axhline(y=1.0, color=C_NEUTRAL, linestyle="--", linewidth=2.5, label="Parity (1.0)")
    ax.fill_between([0, 20000], 1, 10, alpha=0.1, color=C_QPU, label="QPU advantage region")
    
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("QPU / Gurobi Benefit Ratio")
    ax.set_title("QPU Advantage Increases with Problem Size")
    ax.legend(loc="upper left")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 6)
    
    return save_single_plot(fig, "advantage_benefit_ratio", output_dir)


def plot_advantage_time_comparison(output_dir: Path | None = None) -> Path:
    """Advantage Plot 3: Time comparison (log-log)."""
    df, df_6fam, df_27food = _prepare_qpu_advantage_data()
    
    C_GUROBI = "#2ca02c"
    C_QPU = "#1f77b4"
    C_QPU_DARK = "#0d4f8b"
    C_HIGHLIGHT = "#d62728"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(df_6fam["n_vars"], df_6fam["gurobi_time"], s=150, c=C_GUROBI,
               marker="o", label="Gurobi (6-Food)", alpha=0.8)
    ax.scatter(df_27food["n_vars"], df_27food["gurobi_time"], s=150, c="#1a6b1a",
               marker="s", label="Gurobi (27-Food Agg.)", alpha=0.8)
    ax.scatter(df_6fam["n_vars"], df_6fam["qpu_wall_time"], s=150, c=C_QPU,
               marker="o", label="QPU (6-Food)", alpha=0.8)
    ax.scatter(df_27food["n_vars"], df_27food["qpu_wall_time"], s=150, c=C_QPU_DARK,
               marker="s", label="QPU (27-Food Agg.)", alpha=0.8)
    ax.axhline(y=300, color=C_HIGHLIGHT, linestyle="--", linewidth=2.5, label="Gurobi timeout (300s)")
    
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Solve Time (seconds)")
    ax.set_title("Solve Time Comparison")
    ax.legend(loc="upper left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "advantage_time_comparison", output_dir)


def plot_advantage_violations_vs_benefit(output_dir: Path | None = None) -> Path:
    """Advantage Plot 4: Violations vs benefit advantage."""
    df, _, _ = _prepare_qpu_advantage_data()
    
    C_QPU = "#1f77b4"
    C_NEUTRAL = "#7f7f7f"
    C_QPU_LIGHT = "#a6cee3"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sizes = df["n_vars"].values
    norm_sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min()) if sizes.max() > sizes.min() else np.zeros_like(sizes)
    colors_by_size = plt.cm.viridis(norm_sizes)
    
    ax.scatter(df["violations"], df["benefit_advantage"],
               c=colors_by_size, s=200,
               edgecolors="black", linewidths=0.5, alpha=0.8)
    
    for size_label, norm_val in [("Small", 0.0), ("Medium", 0.5), ("Large", 1.0)]:
        ax.scatter([], [], c=[plt.cm.viridis(norm_val)], s=120, label=f"{size_label} size")
    
    ax.axhline(y=0, color=C_NEUTRAL, linestyle="--", linewidth=2)
    ax.fill_between([0, 200], 0, 500, alpha=0.1, color=C_QPU)
    
    ax.set_xlabel("Number of Violations")
    ax.set_ylabel("QPU Benefit Advantage (QPU - Gurobi)")
    ax.set_title("Violations Trade-off: Higher Benefit Despite Violations")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    
    ax.annotate("ALL points above zero:\nQPU always better!",
                xy=(80, 200), fontsize=16, ha="center",
                bbox=dict(boxstyle="round", facecolor=C_QPU_LIGHT, alpha=0.8))
    
    return save_single_plot(fig, "advantage_violations_vs_benefit", output_dir)


def plot_advantage_pure_qpu_scaling(output_dir: Path | None = None) -> Path:
    """Advantage Plot 5: Pure QPU time scaling with linear fit."""
    df, df_6fam, df_27food = _prepare_qpu_advantage_data()
    
    C_6FAM = "#3498db"
    C_27FOOD = "#e74c3c"
    C_NEUTRAL = "#7f7f7f"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(df_6fam["n_vars"], df_6fam["qpu_pure_time"] * 1000, s=180, c=C_6FAM,
               marker="o", label="6-Food", edgecolors="black", linewidths=0.5)
    ax.scatter(df_27food["n_vars"], df_27food["qpu_pure_time"] * 1000, s=180, c=C_27FOOD,
               marker="s", label="27-Food (Agg.)", edgecolors="black", linewidths=0.5)
    
    all_vars = df["n_vars"].values
    all_times = df["qpu_pure_time"].values * 1000
    coef = np.polyfit(all_vars, all_times, 1)
    x_fit = np.linspace(all_vars.min(), all_vars.max(), 100)
    ax.plot(x_fit, coef[0] * x_fit + coef[1], "--", color=C_NEUTRAL, linewidth=2.5,
            label=f"Linear fit: {coef[0]:.3f}ms/var")
    
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Pure QPU Time (milliseconds)")
    ax.set_title("Pure QPU Time Scales Linearly")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "advantage_pure_qpu_scaling", output_dir)


def plot_advantage_summary_table(output_dir: Path | None = None) -> Path:
    """Advantage Plot 6: Summary statistics table."""
    df, df_6fam, df_27food = _prepare_qpu_advantage_data()
    
    C_QPU = "#1f77b4"
    C_QPU_LIGHT = "#a6cee3"
    C_GUROBI_LIGHT = "#b2df8a"
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")
    
    summary_data = [
        ["Metric", "6-Food", "27-Food (Agg.)", "Overall"],
        ["Scenarios", str(len(df_6fam)), str(len(df_27food)), str(len(df))],
        ["Avg Gurobi Benefit", f"{df_6fam['gurobi_obj'].mean():.1f}",
         f"{df_27food['gurobi_obj'].mean():.1f}", f"{df['gurobi_obj'].mean():.1f}"],
        ["Avg QPU Benefit", f"{df_6fam['qpu_benefit'].mean():.1f}",
         f"{df_27food['qpu_benefit'].mean():.1f}", f"{df['qpu_benefit'].mean():.1f}"],
        ["Avg Benefit Ratio", f"{df_6fam['benefit_ratio'].mean():.2f}x",
         f"{df_27food['benefit_ratio'].mean():.2f}x", f"{df['benefit_ratio'].mean():.2f}x"],
        ["Avg Violation Rate", f"{df_6fam['violation_rate'].mean():.1f}%",
         f"{df_27food['violation_rate'].mean():.1f}%", f"{df['violation_rate'].mean():.1f}%"],
        ["Gurobi Timeouts", f"{df_6fam['gurobi_timeout'].sum()}/{len(df_6fam)}",
         f"{df_27food['gurobi_timeout'].sum()}/{len(df_27food)}", f"{df['gurobi_timeout'].sum()}/{len(df)}"],
        ["Avg QPU Time", f"{df_6fam['qpu_wall_time'].mean():.1f}s",
         f"{df_27food['qpu_wall_time'].mean():.1f}s", f"{df['qpu_wall_time'].mean():.1f}s"],
        ["Pure QPU %", f"{df_6fam['qpu_pure_time'].sum() / df_6fam['qpu_wall_time'].sum() * 100:.1f}%",
         f"{df_27food['qpu_pure_time'].sum() / df_27food['qpu_wall_time'].sum() * 100:.1f}%",
         f"{df['qpu_pure_time'].sum() / df['qpu_wall_time'].sum() * 100:.1f}%"],
    ]
    
    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc="center", loc="center", colWidths=[0.35, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.3, 2.0)
    
    for j in range(4):
        table[(0, j)].set_facecolor(C_QPU)
        table[(0, j)].set_text_props(color="white", fontweight="bold", fontsize=16)
    for j in range(4):
        table[(4, j)].set_facecolor(C_QPU_LIGHT)
        table[(6, j)].set_facecolor(C_GUROBI_LIGHT)
    
    ax.set_title("Summary: QPU Outperforms Gurobi", pad=30)
    
    return save_single_plot(fig, "advantage_summary_table", output_dir)


# =============================================================================
# SECTION 7: QPU BENCHMARK PLOTS (12 plots total: 4 each for small/large/comprehensive)
# =============================================================================


def plot_benchmark_small_time(output_dir: Path | None = None) -> Path:
    """Benchmark Small Plot 1: Time comparison."""
    qpu_data = _load_qpu_benchmark_data()
    metrics = _extract_benchmark_metrics(qpu_data, "small_scale")
    
    small_methods = [
        "PlotBased_QPU", "Multilevel(5)_QPU", "Multilevel(10)_QPU",
        "Louvain_QPU", "Spectral(10)_QPU", "cqm_first_PlotBased",
        "HybridGrid(5,9)_QPU",
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if "Gurobi" in metrics:
        times = [t for t in metrics["Gurobi"]["wall_time"] if t > 0]
        farms = [
            metrics["Gurobi"]["n_farms"][i]
            for i, t in enumerate(metrics["Gurobi"]["wall_time"]) if t > 0
        ]
        if times:
            ax.semilogy(farms, times, "o-", linewidth=3, markersize=12,
                       color=_BENCHMARK_COLORS["Gurobi"], label="Gurobi", alpha=0.9)
    
    for method in small_methods:
        if method in metrics and metrics[method]["n_farms"]:
            max_farms = 50 if "HybridGrid" in method else float("inf")
            qpu_times = [
                t for i, t in enumerate(metrics[method]["qpu_time"])
                if t > 0 and metrics[method]["n_farms"][i] <= max_farms
            ]
            farms = [
                metrics[method]["n_farms"][i]
                for i, t in enumerate(metrics[method]["qpu_time"])
                if t > 0 and metrics[method]["n_farms"][i] <= max_farms
            ]
            if qpu_times:
                ax.semilogy(
                    farms, qpu_times, marker=_BENCHMARK_MARKERS.get(method, "o"),
                    linestyle="-", linewidth=2.5, markersize=10,
                    color=_BENCHMARK_COLORS.get(method, "#888"), label=method, alpha=0.8,
                )
    
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Time (s)")
    ax.set_title("Small Scale: Time Comparison")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    
    return save_single_plot(fig, "benchmark_small_time", output_dir)


def plot_benchmark_small_quality(output_dir: Path | None = None) -> Path:
    """Benchmark Small Plot 2: Solution quality."""
    qpu_data = _load_qpu_benchmark_data()
    metrics = _extract_benchmark_metrics(qpu_data, "small_scale")
    
    small_methods = [
        "PlotBased_QPU", "Multilevel(5)_QPU", "Multilevel(10)_QPU",
        "Louvain_QPU", "Spectral(10)_QPU", "cqm_first_PlotBased",
        "HybridGrid(5,9)_QPU",
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if "Gurobi" in metrics:
        ax.plot(
            metrics["Gurobi"]["n_farms"], metrics["Gurobi"]["objective"],
            "o-", linewidth=3, markersize=14, color=_BENCHMARK_COLORS["Gurobi"],
            label="Gurobi (Optimal)", alpha=0.9,
        )
    
    for method in small_methods:
        if method in metrics and metrics[method]["n_farms"]:
            max_farms = 50 if "HybridGrid" in method else float("inf")
            farms = [f for f in metrics[method]["n_farms"] if f <= max_farms]
            objs = [
                metrics[method]["objective"][i]
                for i, f in enumerate(metrics[method]["n_farms"]) if f <= max_farms
            ]
            if farms:
                ax.plot(
                    farms, objs,
                    marker=_BENCHMARK_MARKERS.get(method, "o"), linestyle="-",
                    linewidth=2.5, markersize=10, color=_BENCHMARK_COLORS.get(method, "#888"),
                    label=method, alpha=0.8,
                )
    
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Objective Value")
    ax.set_title("Small Scale: Solution Quality")
    ax.set_yscale("log")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "benchmark_small_quality", output_dir)


def plot_benchmark_small_gap(output_dir: Path | None = None) -> Path:
    """Benchmark Small Plot 3: Optimality gap."""
    qpu_data = _load_qpu_benchmark_data()
    metrics = _extract_benchmark_metrics(qpu_data, "small_scale")
    
    small_methods = [
        "PlotBased_QPU", "Multilevel(5)_QPU", "Multilevel(10)_QPU",
        "Louvain_QPU", "Spectral(10)_QPU", "cqm_first_PlotBased",
        "HybridGrid(5,9)_QPU",
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for method in small_methods:
        if method in metrics and metrics[method]["n_farms"]:
            max_farms = 50 if "HybridGrid" in method else float("inf")
            farms = [f for f in metrics[method]["n_farms"] if f <= max_farms]
            gaps = [
                metrics[method]["gap"][i]
                for i, f in enumerate(metrics[method]["n_farms"]) if f <= max_farms
            ]
            if farms:
                ax.plot(
                    farms, gaps,
                    marker=_BENCHMARK_MARKERS.get(method, "o"), linestyle="-",
                    linewidth=2.5, markersize=10, color=_BENCHMARK_COLORS.get(method, "#888"),
                    label=method, alpha=0.8,
                )
    
    ax.axhline(y=0, color="green", linestyle="--", linewidth=2, alpha=0.7, label="Optimal")
    ax.axhline(y=10, color="orange", linestyle=":", linewidth=2, alpha=0.5, label="10% Gap")
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Gap from Optimal (%)")
    ax.set_title("Small Scale: Optimality Gap")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "benchmark_small_gap", output_dir)


def plot_benchmark_small_violations(output_dir: Path | None = None) -> Path:
    """Benchmark Small Plot 4: Constraint violations."""
    qpu_data = _load_qpu_benchmark_data()
    metrics = _extract_benchmark_metrics(qpu_data, "small_scale")
    
    small_methods = [
        "PlotBased_QPU", "Multilevel(5)_QPU", "Multilevel(10)_QPU",
        "Louvain_QPU", "Spectral(10)_QPU", "cqm_first_PlotBased",
        "HybridGrid(5,9)_QPU",
    ]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    all_farms = sorted({f for m in metrics.values() for f in m["n_farms"] if f <= 50})
    x = np.arange(len(all_farms))
    width = 0.12
    plotted = 0
    
    for method in small_methods:
        if method in metrics and metrics[method]["n_farms"]:
            max_farms = 50 if "HybridGrid" in method else float("inf")
            viols = [
                metrics[method]["violations"][metrics[method]["n_farms"].index(f)]
                if f in metrics[method]["n_farms"] and f <= max_farms else 0
                for f in all_farms
            ]
            method_farms = [f for f in metrics[method]["n_farms"] if f <= max_farms]
            if method_farms:
                ax.bar(
                    x + plotted * width, viols, width,
                    label=method, color=_BENCHMARK_COLORS.get(method, "#888"), alpha=0.8,
                )
                plotted += 1
    
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Constraint Violations")
    ax.set_title("Small Scale: Feasibility")
    ax.set_xticks(x + width * (plotted - 1) / 2)
    ax.set_xticklabels(all_farms)
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    
    return save_single_plot(fig, "benchmark_small_violations", output_dir)


def plot_benchmark_large_time(output_dir: Path | None = None) -> Path:
    """Benchmark Large Plot 1: Time comparison."""
    qpu_data = _load_qpu_benchmark_data()
    metrics = _extract_benchmark_metrics(qpu_data, "large_scale")
    
    large_methods = [
        "Multilevel(10)_QPU", "cqm_first_PlotBased", "coordinated",
        "HybridGrid(5,9)_QPU", "HybridGrid(10,9)_QPU",
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if "Gurobi" in metrics:
        farms = [f for f in metrics["Gurobi"]["n_farms"] if f >= 200]
        times = [
            metrics["Gurobi"]["wall_time"][i]
            for i, f in enumerate(metrics["Gurobi"]["n_farms"]) if f >= 200
        ]
        if farms and any(t > 0 for t in times):
            ax.semilogy(
                farms, times, "o-", linewidth=3, markersize=12,
                color=_BENCHMARK_COLORS["Gurobi"], label="Gurobi", alpha=0.9,
            )
    
    for method in large_methods:
        if method in metrics and metrics[method]["n_farms"]:
            farms = [f for f in metrics[method]["n_farms"] if f >= 200]
            times = [
                metrics[method]["qpu_time"][i]
                for i, f in enumerate(metrics[method]["n_farms"]) if f >= 200
            ]
            if farms and any(t > 0 for t in times):
                ax.semilogy(
                    farms, times, marker=_BENCHMARK_MARKERS.get(method, "o"),
                    linestyle="-", linewidth=2.5, markersize=10,
                    color=_BENCHMARK_COLORS.get(method, "#888"), label=method, alpha=0.8,
                )
    
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Time (s)")
    ax.set_title("Large Scale: Time Comparison")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    
    return save_single_plot(fig, "benchmark_large_time", output_dir)


def plot_benchmark_large_quality(output_dir: Path | None = None) -> Path:
    """Benchmark Large Plot 2: Solution quality."""
    qpu_data = _load_qpu_benchmark_data()
    metrics = _extract_benchmark_metrics(qpu_data, "large_scale")
    
    large_methods = [
        "Multilevel(10)_QPU", "cqm_first_PlotBased", "coordinated",
        "HybridGrid(5,9)_QPU", "HybridGrid(10,9)_QPU",
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if "Gurobi" in metrics:
        farms = [f for f in metrics["Gurobi"]["n_farms"] if f >= 200]
        objs = [
            metrics["Gurobi"]["objective"][i]
            for i, f in enumerate(metrics["Gurobi"]["n_farms"]) if f >= 200
        ]
        if farms:
            ax.plot(
                farms, objs, "o-", linewidth=3, markersize=14,
                color=_BENCHMARK_COLORS["Gurobi"], label="Gurobi (Optimal)", alpha=0.9,
            )
    
    for method in large_methods:
        if method in metrics and metrics[method]["n_farms"]:
            farms = [f for f in metrics[method]["n_farms"] if f >= 200]
            objs = [
                metrics[method]["objective"][i]
                for i, f in enumerate(metrics[method]["n_farms"]) if f >= 200
            ]
            if farms:
                ax.plot(
                    farms, objs, marker=_BENCHMARK_MARKERS.get(method, "o"),
                    linestyle="-", linewidth=2.5, markersize=10,
                    color=_BENCHMARK_COLORS.get(method, "#888"), label=method, alpha=0.8,
                )
    
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Objective Value")
    ax.set_title("Large Scale: Solution Quality")
    ax.set_yscale("log")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "benchmark_large_quality", output_dir)


def plot_benchmark_large_gap(output_dir: Path | None = None) -> Path:
    """Benchmark Large Plot 3: Optimality gap."""
    qpu_data = _load_qpu_benchmark_data()
    metrics = _extract_benchmark_metrics(qpu_data, "large_scale")
    
    large_methods = [
        "Multilevel(10)_QPU", "cqm_first_PlotBased", "coordinated",
        "HybridGrid(5,9)_QPU", "HybridGrid(10,9)_QPU",
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for method in large_methods:
        if method in metrics and metrics[method]["n_farms"]:
            farms = [f for f in metrics[method]["n_farms"] if f >= 200]
            gaps = [
                metrics[method]["gap"][i]
                for i, f in enumerate(metrics[method]["n_farms"]) if f >= 200
            ]
            if farms:
                ax.plot(
                    farms, gaps, marker=_BENCHMARK_MARKERS.get(method, "o"),
                    linestyle="-", linewidth=2.5, markersize=10,
                    color=_BENCHMARK_COLORS.get(method, "#888"), label=method, alpha=0.8,
                )
    
    ax.axhline(y=0, color="green", linestyle="--", linewidth=2, alpha=0.7, label="Optimal")
    ax.axhline(y=10, color="orange", linestyle=":", linewidth=2, alpha=0.5, label="10% Gap")
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Gap from Optimal (%)")
    ax.set_title("Large Scale: Optimality Gap")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "benchmark_large_gap", output_dir)


def plot_benchmark_large_violations(output_dir: Path | None = None) -> Path:
    """Benchmark Large Plot 4: Constraint violations."""
    qpu_data = _load_qpu_benchmark_data()
    metrics = _extract_benchmark_metrics(qpu_data, "large_scale")
    
    large_methods = [
        "Multilevel(10)_QPU", "cqm_first_PlotBased", "coordinated",
        "HybridGrid(5,9)_QPU", "HybridGrid(10,9)_QPU",
    ]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    large_farms = sorted({f for m in metrics.values() for f in m["n_farms"] if f >= 200})
    x = np.arange(len(large_farms))
    width = 0.15
    plotted = 0
    
    for method in large_methods:
        if method in metrics and metrics[method]["n_farms"]:
            viols = [
                metrics[method]["violations"][metrics[method]["n_farms"].index(f)]
                if f in metrics[method]["n_farms"] else 0
                for f in large_farms
            ]
            ax.bar(
                x + plotted * width, viols, width,
                label=method, color=_BENCHMARK_COLORS.get(method, "#888"), alpha=0.8,
            )
            plotted += 1
    
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Constraint Violations")
    ax.set_title("Large Scale: Feasibility")
    ax.set_xticks(x + width * (plotted - 1) / 2)
    ax.set_xticklabels(large_farms)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    
    return save_single_plot(fig, "benchmark_large_violations", output_dir)


def plot_benchmark_comprehensive_time(output_dir: Path | None = None) -> Path:
    """Benchmark Comprehensive Plot 1: Time comparison all scales."""
    qpu_data = _load_qpu_benchmark_data()
    metrics_small = _extract_benchmark_metrics(qpu_data, "small_scale")
    metrics_large = _extract_benchmark_metrics(qpu_data, "large_scale")
    
    all_metrics: dict[str, dict[str, list]] = {}
    for m in [metrics_small, metrics_large]:
        for method, data in m.items():
            if method not in all_metrics:
                all_metrics[method] = {k: [] for k in data}
            for key in data:
                all_metrics[method][key].extend(data[key])
    
    qpu_methods = [
        "PlotBased_QPU", "Multilevel(5)_QPU", "Multilevel(10)_QPU",
        "Louvain_QPU", "Spectral(10)_QPU", "cqm_first_PlotBased",
        "coordinated", "HybridGrid(5,9)_QPU", "HybridGrid(10,9)_QPU",
    ]
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    if "Gurobi" in all_metrics:
        times = [t for t in all_metrics["Gurobi"]["wall_time"] if t > 0]
        farms = [
            all_metrics["Gurobi"]["n_farms"][i]
            for i, t in enumerate(all_metrics["Gurobi"]["wall_time"]) if t > 0
        ]
        if times:
            ax.semilogy(farms, times, "o-", linewidth=3, markersize=12,
                        color=_BENCHMARK_COLORS["Gurobi"], label="Gurobi", alpha=0.9)
    
    for method in qpu_methods:
        if method in all_metrics and all_metrics[method]["n_farms"]:
            qpu_times = [t for t in all_metrics[method]["qpu_time"] if t > 0]
            farms = [
                all_metrics[method]["n_farms"][i]
                for i, t in enumerate(all_metrics[method]["qpu_time"]) if t > 0
            ]
            if qpu_times:
                ax.semilogy(
                    farms, qpu_times, marker=_BENCHMARK_MARKERS.get(method, "o"),
                    linestyle="-", linewidth=2, markersize=8,
                    color=_BENCHMARK_COLORS.get(method, "#888"), label=method, alpha=0.7,
                )
    
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Time (s)")
    ax.set_title("Comprehensive: Time Comparison (All Scales)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    
    return save_single_plot(fig, "benchmark_comprehensive_time", output_dir)


def plot_benchmark_comprehensive_quality(output_dir: Path | None = None) -> Path:
    """Benchmark Comprehensive Plot 2: Solution quality all scales."""
    qpu_data = _load_qpu_benchmark_data()
    metrics_small = _extract_benchmark_metrics(qpu_data, "small_scale")
    metrics_large = _extract_benchmark_metrics(qpu_data, "large_scale")
    
    all_metrics: dict[str, dict[str, list]] = {}
    for m in [metrics_small, metrics_large]:
        for method, data in m.items():
            if method not in all_metrics:
                all_metrics[method] = {k: [] for k in data}
            for key in data:
                all_metrics[method][key].extend(data[key])
    
    qpu_methods = [
        "PlotBased_QPU", "Multilevel(5)_QPU", "Multilevel(10)_QPU",
        "Louvain_QPU", "Spectral(10)_QPU", "cqm_first_PlotBased",
        "coordinated", "HybridGrid(5,9)_QPU", "HybridGrid(10,9)_QPU",
    ]
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    if "Gurobi" in all_metrics:
        ax.plot(
            all_metrics["Gurobi"]["n_farms"], all_metrics["Gurobi"]["objective"],
            "o-", linewidth=3, markersize=12, color=_BENCHMARK_COLORS["Gurobi"],
            label="Gurobi (Optimal)", alpha=0.9,
        )
    
    for method in qpu_methods:
        if method in all_metrics and all_metrics[method]["n_farms"]:
            ax.plot(
                all_metrics[method]["n_farms"], all_metrics[method]["objective"],
                marker=_BENCHMARK_MARKERS.get(method, "o"), linestyle="-",
                linewidth=2, markersize=8, color=_BENCHMARK_COLORS.get(method, "#888"),
                label=method, alpha=0.7,
            )
    
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Objective Value")
    ax.set_title("Comprehensive: Solution Quality (All Scales)")
    ax.set_yscale("log")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "benchmark_comprehensive_quality", output_dir)


def plot_benchmark_comprehensive_gap(output_dir: Path | None = None) -> Path:
    """Benchmark Comprehensive Plot 3: Optimality gap all methods."""
    qpu_data = _load_qpu_benchmark_data()
    metrics_small = _extract_benchmark_metrics(qpu_data, "small_scale")
    metrics_large = _extract_benchmark_metrics(qpu_data, "large_scale")
    
    all_metrics: dict[str, dict[str, list]] = {}
    for m in [metrics_small, metrics_large]:
        for method, data in m.items():
            if method not in all_metrics:
                all_metrics[method] = {k: [] for k in data}
            for key in data:
                all_metrics[method][key].extend(data[key])
    
    qpu_methods = [
        "PlotBased_QPU", "Multilevel(5)_QPU", "Multilevel(10)_QPU",
        "Louvain_QPU", "Spectral(10)_QPU", "cqm_first_PlotBased",
        "coordinated", "HybridGrid(5,9)_QPU", "HybridGrid(10,9)_QPU",
    ]
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    for method in qpu_methods:
        if method in all_metrics and all_metrics[method]["n_farms"]:
            ax.plot(
                all_metrics[method]["n_farms"], all_metrics[method]["gap"],
                marker=_BENCHMARK_MARKERS.get(method, "o"), linestyle="-",
                linewidth=2, markersize=8, color=_BENCHMARK_COLORS.get(method, "#888"),
                label=method, alpha=0.7,
            )
    
    ax.axhline(y=0, color="green", linestyle="--", linewidth=2, alpha=0.7, label="Optimal")
    ax.axhline(y=10, color="orange", linestyle=":", linewidth=2, alpha=0.5, label="10% Gap")
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Gap from Optimal (%)")
    ax.set_title("Comprehensive: Optimality Gap (All Methods)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    
    return save_single_plot(fig, "benchmark_comprehensive_gap", output_dir)


def plot_benchmark_comprehensive_violations(output_dir: Path | None = None) -> Path:
    """Benchmark Comprehensive Plot 4: Constraint violations all methods."""
    qpu_data = _load_qpu_benchmark_data()
    metrics_small = _extract_benchmark_metrics(qpu_data, "small_scale")
    metrics_large = _extract_benchmark_metrics(qpu_data, "large_scale")
    
    all_metrics: dict[str, dict[str, list]] = {}
    for m in [metrics_small, metrics_large]:
        for method, data in m.items():
            if method not in all_metrics:
                all_metrics[method] = {k: [] for k in data}
            for key in data:
                all_metrics[method][key].extend(data[key])
    
    qpu_methods = [
        "PlotBased_QPU", "Multilevel(5)_QPU", "Multilevel(10)_QPU",
        "Louvain_QPU", "Spectral(10)_QPU", "cqm_first_PlotBased",
    ]
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    all_farms = sorted({f for m in all_metrics.values() for f in m["n_farms"]})
    x = np.arange(len(all_farms))
    width = 0.12
    plotted = 0
    
    for method in qpu_methods:
        if method in all_metrics and all_metrics[method]["n_farms"]:
            viols = [
                all_metrics[method]["violations"][all_metrics[method]["n_farms"].index(f)]
                if f in all_metrics[method]["n_farms"] else 0
                for f in all_farms
            ]
            ax.bar(
                x + plotted * width, viols, width,
                label=method, color=_BENCHMARK_COLORS.get(method, "#888"), alpha=0.8,
            )
            plotted += 1
    
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Violations")
    ax.set_title("Comprehensive: Constraint Violations")
    ax.set_xticks(x + width * (plotted - 1) / 2)
    ax.set_xticklabels(all_farms)
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    
    return save_single_plot(fig, "benchmark_comprehensive_violations", output_dir)


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================


def generate_all_single_plots() -> dict[str, bool]:
    """Generate all individual PDF plots."""
    setup_large_font_style()
    
    results: dict[str, bool] = {}
    output_dir = SINGLE_PLOT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING SINGLE-PLOT PDFs")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load shared data
    print("Loading shared data...")
    try:
        df_60s = prepare_scaling_data_60s()
        print(f"  [OK] 60s data: {len(df_60s)} scenarios")
    except Exception as e:
        print(f"  [FAIL] Failed to load 60s data: {e}")
        df_60s = None
    
    try:
        df_300s = prepare_scaling_data_300s()
        print(f"  [OK] 300s data: {len(df_300s)} scenarios")
    except Exception as e:
        print(f"  [FAIL] Failed to load 300s data: {e}")
        df_300s = None
    
    print()
    
    # Define all plot functions
    plot_functions = [
        # Section 1: Comprehensive Scaling (4 plots - vars and farms)
        ("scaling_objectives_by_vars", lambda: plot_scaling_objectives_by_vars(df_60s, output_dir)),
        ("scaling_objectives_by_farms", lambda: plot_scaling_objectives_by_farms(df_60s, output_dir)),
        ("scaling_time_comparison", lambda: plot_scaling_time_comparison(df_60s, output_dir)),
        ("scaling_qpu_breakdown_by_vars", lambda: plot_scaling_qpu_breakdown_by_vars(df_60s, output_dir)),
        ("scaling_qpu_breakdown_by_farms", lambda: plot_scaling_qpu_breakdown_by_farms(df_60s, output_dir)),
        
        # Section 2: Split Formulation (12 plots - vars and farms)
        ("split_objectives_by_vars", lambda: plot_split_objectives_by_vars(df_300s, output_dir)),
        ("split_objectives_by_farms", lambda: plot_split_objectives_by_farms(df_300s, output_dir)),
        ("split_optimality_gap_by_vars", lambda: plot_split_optimality_gap_by_vars(df_300s, output_dir)),
        ("split_optimality_gap_by_farms", lambda: plot_split_optimality_gap_by_farms(df_300s, output_dir)),
        ("split_solve_time_by_vars", lambda: plot_split_solve_time_by_vars(df_300s, output_dir)),
        ("split_solve_time_by_farms", lambda: plot_split_solve_time_by_farms(df_300s, output_dir)),
        ("split_speedup_by_vars", lambda: plot_split_speedup_by_vars(df_300s, output_dir)),
        ("split_speedup_by_farms", lambda: plot_split_speedup_by_farms(df_300s, output_dir)),
        ("split_pure_qpu_time_by_vars", lambda: plot_split_pure_qpu_time_by_vars(df_300s, output_dir)),
        ("split_pure_qpu_time_by_farms", lambda: plot_split_pure_qpu_time_by_farms(df_300s, output_dir)),
        ("split_gurobi_mip_gap_by_vars", lambda: plot_split_gurobi_mip_gap_by_vars(df_300s, output_dir)),
        ("split_gurobi_mip_gap_by_farms", lambda: plot_split_gurobi_mip_gap_by_farms(df_300s, output_dir)),
        
        # Section 3: Gap Analysis (6 plots)
        ("gap_absolute_comparison", lambda: plot_gap_absolute_comparison(df_300s, output_dir)),
        ("gap_ratio", lambda: plot_gap_ratio(df_300s, output_dir)),
        ("gap_vs_mip_gap", lambda: plot_gap_vs_mip_gap(df_300s, output_dir)),
        ("gap_6family_scaling", lambda: plot_gap_6family_scaling(df_300s, output_dir)),
        ("gap_27food_scaling", lambda: plot_gap_27food_scaling(df_300s, output_dir)),
        ("gap_summary_table", lambda: plot_gap_summary_table(df_300s, output_dir)),
        
        # Section 4: Violation Impact (3 plots)
        ("violation_rate", lambda: plot_violation_rate(None, output_dir)),
        ("violation_gap_breakdown", lambda: plot_violation_gap_breakdown(None, output_dir)),
        ("violation_adjusted_objective", lambda: plot_violation_adjusted_objective(None, output_dir)),
        
        # Section 5: Deep Dive (6 plots)
        ("deepdive_objective_comparison", lambda: plot_deepdive_objective_comparison(None, output_dir)),
        ("deepdive_ratio_analysis", lambda: plot_deepdive_ratio_analysis(None, output_dir)),
        ("deepdive_gap_attribution", lambda: plot_deepdive_gap_attribution(None, output_dir)),
        ("deepdive_scatter", lambda: plot_deepdive_scatter(None, output_dir)),
        ("deepdive_violation_impact_by_scenario", lambda: plot_deepdive_violation_impact_by_scenario(None, output_dir)),
        ("deepdive_summary_findings", lambda: plot_deepdive_summary_findings(None, output_dir)),
        
        # Section 6: QPU Advantage (6 plots)
        ("advantage_benefit_comparison", lambda: plot_advantage_benefit_comparison(output_dir)),
        ("advantage_benefit_ratio", lambda: plot_advantage_benefit_ratio(output_dir)),
        ("advantage_time_comparison", lambda: plot_advantage_time_comparison(output_dir)),
        ("advantage_violations_vs_benefit", lambda: plot_advantage_violations_vs_benefit(output_dir)),
        ("advantage_pure_qpu_scaling", lambda: plot_advantage_pure_qpu_scaling(output_dir)),
        ("advantage_summary_table", lambda: plot_advantage_summary_table(output_dir)),
        
        # Section 7: Benchmark Small Scale (4 plots)
        ("benchmark_small_time", lambda: plot_benchmark_small_time(output_dir)),
        ("benchmark_small_quality", lambda: plot_benchmark_small_quality(output_dir)),
        ("benchmark_small_gap", lambda: plot_benchmark_small_gap(output_dir)),
        ("benchmark_small_violations", lambda: plot_benchmark_small_violations(output_dir)),
        
        # Section 7: Benchmark Large Scale (4 plots)
        ("benchmark_large_time", lambda: plot_benchmark_large_time(output_dir)),
        ("benchmark_large_quality", lambda: plot_benchmark_large_quality(output_dir)),
        ("benchmark_large_gap", lambda: plot_benchmark_large_gap(output_dir)),
        ("benchmark_large_violations", lambda: plot_benchmark_large_violations(output_dir)),
        
        # Section 7: Benchmark Comprehensive (4 plots)
        ("benchmark_comprehensive_time", lambda: plot_benchmark_comprehensive_time(output_dir)),
        ("benchmark_comprehensive_quality", lambda: plot_benchmark_comprehensive_quality(output_dir)),
        ("benchmark_comprehensive_gap", lambda: plot_benchmark_comprehensive_gap(output_dir)),
        ("benchmark_comprehensive_violations", lambda: plot_benchmark_comprehensive_violations(output_dir)),
    ]
    
    # Generate each plot
    for name, func in plot_functions:
        print(f"Generating: {name}...")
        try:
            func()
            results[name] = True
        except Exception as e:
            results[name] = False
            print(f"  [FAIL] {name}: {e}")
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(results.values())
    total = len(results)
    print(f"Completed: {successful}/{total} plots generated successfully")
    print()
    
    failed = [name for name, success in results.items() if not success]
    if failed:
        print("Failed plots:")
        for name in failed:
            print(f"  - {name}")
    
    print()
    print(f"Output directory: {output_dir}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return results


def main() -> None:
    """Main entry point."""
    generate_all_single_plots()


if __name__ == "__main__":
    main()
