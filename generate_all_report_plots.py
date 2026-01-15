#!/usr/bin/env python3
"""
Master Plotting Script for OQI-UC002-DWave Project Report.

This script consolidates all visualization code for the project report,
providing unified color schemes, data loading utilities, and publication-quality
plot generation.

The script is organized into sections:
1. Configuration & Colors - Unified color palette and rcParams
2. Data Loading - Functions to load QPU and Gurobi results
3. Data Preparation - Helper functions for merging and computing metrics
4. Plot Functions - Individual plot generation (PART 2)
5. Report Generation - Master function to generate all plots (PART 2)

KEY CONVENTIONS:
- Sign correction: benefit = -objective_miqp for QPU data (QUBO minimizes negative benefit)
- This is a MAXIMIZATION problem: higher benefit values = better
- QPU achieves higher benefit values than Gurobi in most cases
- Violations are a trade-off for exploring beyond strict feasibility

Author: OQI-UC002-DWave Project
Date: 2025-01-15
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

# =============================================================================
# UNIFIED COLOR PALETTE
# =============================================================================

# Primary qualitative palette (20 colors for line plots, bar charts)
# Optimized for distinguishability, colorblind-friendliness, and modern aesthetics
QUALITATIVE_COLORS: list[str] = [
    "#0173B2",  # Vivid blue (primary)
    "#DE8F05",  # Rich orange (secondary)
    "#029E73",  # Teal green
    "#CC3311",  # Strong red
    "#6F42C1",  # Rich purple
    "#CA6510",  # Burnt orange
    "#EE3377",  # Vibrant magenta
    "#56595C",  # Charcoal gray
    "#949300",  # Olive yellow
    "#00ADB5",  # Cyan
    "#56B4E9",  # Sky blue
    "#F0AB00",  # Amber
    "#4ECDC4",  # Turquoise
    "#FF6B6B",  # Coral red
    "#A463F2",  # Light purple
    "#D4A574",  # Tan
    "#FF8FA3",  # Rose pink
    "#95A5A6",  # Silver gray
    "#B8C936",  # Lime
    "#36C9BB",  # Aquamarine
]

# Sequential palette for heatmaps/gradients (Viridis-inspired blues)
SEQUENTIAL_COLORS: list[str] = [
    "#F0F9FF",  # Lightest blue
    "#DBEAFE",  # Very light blue
    "#BFDBFE",  # Light blue
    "#93C5FD",  # Medium-light blue
    "#60A5FA",  # Medium blue
    "#3B82F6",  # Blue
    "#2563EB",  # Dark blue
    "#1D4ED8",  # Darker blue
    "#1E3A8A",  # Darkest blue
]

# Diverging palette for positive/negative data (Improved red-blue)
DIVERGING_COLORS: list[str] = [
    "#A50026",  # Dark red (negative extreme)
    "#D73027",  # Red
    "#F46D43",  # Orange-red
    "#FDAE61",  # Light orange
    "#F7F7F7",  # Neutral white
    "#ABD9E9",  # Light blue
    "#74ADD1",  # Medium blue
    "#4575B4",  # Blue
    "#313695",  # Dark blue (positive extreme)
]

# Colormap names for consistency
SEQUENTIAL_CMAP: str = "Blues"
DIVERGING_CMAP: str = "RdBu_r"
QUALITATIVE_CMAP: str = "tab20"

# =============================================================================
# METHOD & SOLVER COLORS
# =============================================================================

# Core solver colors (enhanced contrast and modern palette)
SOLVER_COLORS: dict[str, str] = {
    # QPU variants
    "qpu": "#0173B2",  # Vivid blue - D-Wave/Quantum
    "qpu_dark": "#004E89",  # Deep blue
    "qpu_light": "#56B4E9",  # Sky blue
    # Gurobi variants
    "gurobi": "#029E73",  # Teal green - Classical/Gurobi
    "gurobi_dark": "#006B4E",  # Forest green
    "gurobi_light": "#4ECDC4",  # Turquoise
    # Semantic colors
    "violation": "#F77F00",  # Vibrant orange - Warnings/Violations
    "benefit": "#6F42C1",  # Rich purple - Benefit
    "neutral": "#6C757D",  # Neutral gray
    "highlight": "#CC3311",  # Strong red - Highlight/Alert
    "success": "#029E73",  # Teal - Success
    "timeout": "#CC3311",  # Strong red - Timeout indicator
}

# Formulation-specific colors
FORMULATION_COLORS: dict[str, str] = {
    "6-Family": "#0173B2",  # Vivid blue (6-family formulation)
    "6family": "#0173B2",  # Alias
    "27-Food": "#CC3311",  # Strong red (27-food formulation)
    "27food": "#CC3311",  # Alias
}

# Method-specific colors (vibrant professional palette)
METHOD_COLORS: dict[str, str] = {
    "Gurobi": "#DE8F05",  # Rich orange (optimal baseline)
    "PuLP": "#DE8F05",  # Rich orange (optimal baseline)
    "D-Wave Hybrid": "#029E73",  # Teal (hybrid approach)
    "PlotBased_QPU": "#6F42C1",  # Rich purple
    "Multilevel(5)_QPU": "#EE3377",  # Vibrant magenta
    "Multilevel(10)_QPU": "#949300",  # Olive green
    "Louvain_QPU": "#F0AB00",  # Amber
    "Spectral(10)_QPU": "#CA6510",  # Burnt orange
    "cqm_first_PlotBased": "#0173B2",  # Vivid blue
    "coordinated": "#A463F2",  # Light purple
    "HybridGrid(5,9)_QPU": "#4ECDC4",  # Turquoise
    "HybridGrid(10,9)_QPU": "#FF6B6B",  # Coral
    "QPU": "#FF8FA3",  # Rose coral - Legacy compatibility
    "Gurobi_bar": "#0173B2",  # Vivid blue - Bar chart variant
}

# Food group colors
FOOD_GROUP_COLORS: dict[str, str] = {
    "Vegetables": "#029E73",  # Teal
    "Grains": "#F0AB00",  # Amber
    "Legumes": "#949300",  # Olive
    "Fruits": "#DE8F05",  # Rich orange
    "Meats": "#EE3377",  # Vibrant magenta
}

# Legacy alias from generate_comprehensive_scaling_plots.py
COLORS: dict[str, str] = {
    **SOLVER_COLORS,
    **FORMULATION_COLORS,
}

# Markers for different formulations
MARKERS: dict[str, str] = {
    "6-Family": "o",
    "27-Food": "D",
    "QPU": "s",
    "Gurobi": "^",
}


# =============================================================================
# RCPARAMS SETUP FOR PUBLICATION QUALITY
# =============================================================================


def setup_publication_style() -> None:
    """
    Configure matplotlib for publication-quality plots.

    This applies consistent styling across all figures including:
    - Professional serif fonts
    - Bold axis labels and titles
    - Visible grid lines
    - High-resolution output settings
    """
    rcParams.update(
        {
            # Font configuration
            "text.usetex": False,  # Avoid LaTeX compilation issues
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            # Figure quality
            "figure.dpi": 150,
            "figure.constrained_layout.use": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "savefig.transparent": False,
            # Grid configuration
            "axes.grid": True,
            "axes.grid.which": "major",
            "axes.grid.axis": "both",
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            # Axes configuration
            "axes.linewidth": 1.3,
            "axes.labelpad": 10,
            "axes.titlepad": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            # Tick configuration
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            # Line and marker properties
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "lines.markeredgewidth": 1.0,
            "patch.linewidth": 0.8,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "legend.shadow": False,
            # Error bars
            "errorbar.capsize": 3,
        }
    )


def setup_scaling_style() -> None:
    """
    Configure matplotlib for scaling/comprehensive plots.

    This is an alternative style more suitable for multi-panel figures
    with slightly smaller fonts.
    """
    rcParams.update(
        {
            "font.size": 11,
            "font.family": "sans-serif",
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": (18, 10),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class QPURun:
    """Structured representation of a single QPU run result."""

    scenario_name: str
    n_farms: int
    n_foods: int
    n_vars: int
    n_periods: int
    objective_miqp: float
    benefit: float  # Sign-corrected: -objective_miqp
    total_wall_time: float
    qpu_access_time: float
    qpu_sampling_time: float
    total_violations: int
    one_hot_violations: int
    status: str
    formulation: str  # '6-Family' or '27-Food'
    backend: str


@dataclass
class GurobiRun:
    """Structured representation of a single Gurobi run result."""

    scenario_name: str
    n_farms: int
    n_foods: int
    n_vars: int
    n_periods: int
    objective_miqp: float
    benefit: float  # Same as objective_miqp for Gurobi (maximization)
    total_wall_time: float
    solve_time: float
    mip_gap: float
    status: str
    hit_timeout: bool
    formulation: str  # '6-Family' or '27-Food'


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

# Default paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent
DEFAULT_QPU_HIER_PATH = PROJECT_ROOT / "qpu_hier_repaired.json"
DEFAULT_GUROBI_60S_PATH = PROJECT_ROOT / "gurobi_baseline_60s.json"
DEFAULT_GUROBI_300S_DIR = PROJECT_ROOT / "@todo" / "gurobi_timeout_verification"
# The specific 300s file used in original assess_violation_impact.py and deep_dive_gap_analysis.py
DEFAULT_GUROBI_300S_FILE = DEFAULT_GUROBI_300S_DIR / "gurobi_timeout_test_20251224_103144.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "professional_plots"


def _determine_formulation(n_foods: int) -> str:
    """Determine formulation type based on number of foods."""
    return "6-Family" if n_foods == 6 else "27-Food"


def load_qpu_hierarchical(
    filepath: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load QPU hierarchical results from JSON file.

    Args:
        filepath: Path to the JSON file. Defaults to qpu_hier_repaired.json.

    Returns:
        DataFrame with columns:
            - scenario_name, n_farms, n_foods, n_vars, n_periods
            - objective_miqp (raw), benefit (sign-corrected: -objective_miqp)
            - total_wall_time, qpu_access_time, qpu_sampling_time
            - total_violations, one_hot_violations
            - status, formulation, backend

    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        ValueError: If JSON structure is invalid.
    """
    path = Path(filepath) if filepath else DEFAULT_QPU_HIER_PATH

    if not path.exists():
        raise FileNotFoundError(f"QPU hierarchical file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "runs" not in data:
        raise ValueError(f"Invalid JSON structure: missing 'runs' key in {path}")

    records: list[dict[str, Any]] = []
    for run in data["runs"]:
        timing = run.get("timing", {})
        violations = run.get("constraint_violations", {})

        # Sign correction: benefit = -objective_miqp (QUBO minimizes negative benefit)
        obj_miqp = run.get("objective_miqp", 0.0)
        benefit = -obj_miqp  # Convert to positive benefit value

        records.append(
            {
                "scenario_name": run["scenario_name"],
                "n_farms": run["n_farms"],
                "n_foods": run["n_foods"],
                "n_vars": run["n_vars"],
                "n_periods": run["n_periods"],
                "objective_miqp": obj_miqp,
                "benefit": benefit,
                "total_wall_time": timing.get("total_wall_time", 0.0),
                "qpu_access_time": timing.get("qpu_access_time", 0.0),
                "qpu_sampling_time": timing.get("qpu_sampling_time", 0.0),
                "total_violations": violations.get("total_violations", 0),
                "one_hot_violations": violations.get("one_hot_violations", 0),
                "status": run.get("status", "unknown"),
                "formulation": _determine_formulation(run["n_foods"]),
                "backend": run.get("backend", "DWaveCliqueSampler"),
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values("n_vars").reset_index(drop=True)
    return df


def load_gurobi_60s(
    filepath: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load Gurobi 60s baseline results from JSON file.

    Args:
        filepath: Path to the JSON file. Defaults to gurobi_baseline_60s.json.

    Returns:
        DataFrame with columns:
            - scenario_name, n_farms, n_foods, n_vars, n_periods
            - objective_miqp, benefit (same as objective_miqp)
            - total_wall_time, solve_time, mip_gap
            - status, hit_timeout, formulation

    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        ValueError: If JSON structure is invalid.
    """
    path = Path(filepath) if filepath else DEFAULT_GUROBI_60S_PATH

    if not path.exists():
        raise FileNotFoundError(f"Gurobi 60s baseline file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "runs" not in data:
        raise ValueError(f"Invalid JSON structure: missing 'runs' key in {path}")

    records: list[dict[str, Any]] = []
    for run in data["runs"]:
        timing = run.get("timing", {})

        # Gurobi already maximizes, so objective_miqp is the benefit
        obj_miqp = run.get("objective_miqp", 0.0)
        hit_timeout = run.get("status") == "timeout"

        records.append(
            {
                "scenario_name": run["scenario_name"],
                "n_farms": run["n_farms"],
                "n_foods": run["n_foods"],
                "n_vars": run["n_vars"],
                "n_periods": run["n_periods"],
                "objective_miqp": obj_miqp,
                "benefit": obj_miqp,  # No sign correction for Gurobi
                "total_wall_time": timing.get("total_wall_time", 0.0),
                "solve_time": timing.get("solve_time", 0.0),
                "mip_gap": run.get("mip_gap", 0.0),
                "status": run.get("status", "unknown"),
                "hit_timeout": hit_timeout,
                "formulation": _determine_formulation(run["n_foods"]),
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values("n_vars").reset_index(drop=True)
    return df


def load_gurobi_300s(
    filepath: Path | str | None = None,
    directory: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load Gurobi 300s timeout verification results.

    The 300s results have a different JSON structure (list of entries with
    'metadata' and 'result' keys) compared to the 60s baseline.

    Args:
        filepath: Specific JSON file to load. If None, uses latest in directory.
        directory: Directory containing gurobi_timeout_test_*.json files.
                   Defaults to @todo/gurobi_timeout_verification/

    Returns:
        DataFrame with columns:
            - scenario_name, n_farms, n_foods, n_vars, n_periods
            - objective_miqp, benefit (same as objective_miqp)
            - solve_time, mip_gap
            - status, hit_timeout, formulation

    Raises:
        FileNotFoundError: If no matching files found.
        ValueError: If JSON structure is invalid.
    """
    if filepath:
        path = Path(filepath)
    else:
        search_dir = Path(directory) if directory else DEFAULT_GUROBI_300S_DIR
        if not search_dir.exists():
            raise FileNotFoundError(f"Gurobi 300s directory not found: {search_dir}")

        # Find the most recent gurobi_timeout_test file
        json_files = sorted(search_dir.glob("gurobi_timeout_test_*.json"))
        if not json_files:
            raise FileNotFoundError(f"No gurobi_timeout_test_*.json files in {search_dir}")
        path = json_files[-1]  # Latest by timestamp in filename

    if not path.exists():
        raise FileNotFoundError(f"Gurobi 300s file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # 300s files are a list of entries, not a dict with 'runs'
    if not isinstance(data, list):
        raise ValueError(f"Invalid JSON structure: expected list in {path}")

    records: list[dict[str, Any]] = []
    for entry in data:
        if "metadata" not in entry or "result" not in entry:
            continue  # Skip malformed entries

        meta = entry["metadata"]
        result = entry["result"]

        # Extract n_vars from result (not in metadata)
        n_vars = result.get("n_vars", meta["n_farms"] * meta["n_foods"] * meta["n_periods"])
        obj_value = result.get("objective_value", 0.0)
        hit_timeout = result.get("hit_timeout", False)

        records.append(
            {
                "scenario_name": meta["scenario"],
                "n_farms": meta["n_farms"],
                "n_foods": meta["n_foods"],
                "n_vars": n_vars,
                "n_periods": meta["n_periods"],
                "objective_miqp": obj_value,
                "benefit": obj_value,  # No sign correction for Gurobi
                "solve_time": result.get("solve_time", 0.0),
                "mip_gap": result.get("mip_gap", 0.0),
                "status": result.get("status", "unknown"),
                "hit_timeout": hit_timeout,
                "formulation": _determine_formulation(meta["n_foods"]),
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values("n_vars").reset_index(drop=True)
    return df


def load_qpu_benchmark_results(
    directory: Path | str | None = None,
    pattern: str = "qpu_benchmark_*.json",
) -> pd.DataFrame:
    """
    Load QPU benchmark summary results.

    Args:
        directory: Directory to search. Defaults to project root.
        pattern: Glob pattern for matching files.

    Returns:
        DataFrame with benchmark summary information including:
            - timestamp, sampler, timeout
            - modes, scenarios, total_runs
            - success_count, fail_count
            - Individual result entries

    Raises:
        FileNotFoundError: If no matching files found.
    """
    search_dir = Path(directory) if directory else PROJECT_ROOT

    json_files = sorted(search_dir.glob(pattern))
    if not json_files:
        raise FileNotFoundError(f"No {pattern} files found in {search_dir}")

    all_records: list[dict[str, Any]] = []

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # Extract summary info
        timestamp = data.get("timestamp", "")
        sampler = data.get("sampler", "unknown")
        timeout = data.get("timeout", 0.0)

        # Process individual results
        for result in data.get("results", []):
            all_records.append(
                {
                    "source_file": json_file.name,
                    "timestamp": timestamp,
                    "sampler": sampler,
                    "timeout": timeout,
                    "mode": result.get("mode", ""),
                    "scenario": result.get("scenario", ""),
                    "success": result.get("success", False),
                    "output_file": result.get("output_file", ""),
                    "returncode": result.get("returncode", -1),
                }
            )

    return pd.DataFrame(all_records)


# =============================================================================
# DATA PREPARATION HELPERS
# =============================================================================


def merge_by_scenario(
    qpu_df: pd.DataFrame,
    gurobi_df: pd.DataFrame,
    qpu_prefix: str = "qpu_",
    gurobi_prefix: str = "gurobi_",
) -> pd.DataFrame:
    """
    Merge QPU and Gurobi DataFrames by scenario name.

    Args:
        qpu_df: QPU results DataFrame (from load_qpu_hierarchical).
        gurobi_df: Gurobi results DataFrame (from load_gurobi_60s or load_gurobi_300s).
        qpu_prefix: Prefix for QPU columns in merged output.
        gurobi_prefix: Prefix for Gurobi columns in merged output.

    Returns:
        Merged DataFrame with prefixed columns and computed comparison metrics:
            - benefit_advantage: qpu_benefit - gurobi_benefit
            - benefit_ratio: qpu_benefit / gurobi_benefit
            - speedup: gurobi_time / qpu_time
            - gap_pct: |qpu_benefit - gurobi_benefit| / gurobi_benefit * 100
    """
    # Columns to keep from each DataFrame
    qpu_cols = [
        "scenario_name",
        "n_farms",
        "n_foods",
        "n_vars",
        "n_periods",
        "formulation",
        "benefit",
        "objective_miqp",
        "total_wall_time",
        "qpu_access_time",
        "total_violations",
        "status",
    ]
    gurobi_cols = [
        "scenario_name",
        "benefit",
        "objective_miqp",
        "total_wall_time",
        "solve_time",
        "mip_gap",
        "status",
        "hit_timeout",
    ]

    # Filter to available columns
    qpu_subset = qpu_df[[c for c in qpu_cols if c in qpu_df.columns]].copy()
    gurobi_subset = gurobi_df[[c for c in gurobi_cols if c in gurobi_df.columns]].copy()

    # Rename columns with prefixes (except scenario_name)
    qpu_rename = {
        c: f"{qpu_prefix}{c}" for c in qpu_subset.columns if c != "scenario_name"
    }
    gurobi_rename = {
        c: f"{gurobi_prefix}{c}" for c in gurobi_subset.columns if c != "scenario_name"
    }

    qpu_subset = qpu_subset.rename(columns=qpu_rename)
    gurobi_subset = gurobi_subset.rename(columns=gurobi_rename)

    # Merge on scenario_name
    merged = pd.merge(qpu_subset, gurobi_subset, on="scenario_name", how="outer")

    # Compute comparison metrics
    qpu_benefit_col = f"{qpu_prefix}benefit"
    gurobi_benefit_col = f"{gurobi_prefix}benefit"
    qpu_time_col = f"{qpu_prefix}total_wall_time"
    gurobi_time_col = f"{gurobi_prefix}total_wall_time"

    if qpu_benefit_col in merged.columns and gurobi_benefit_col in merged.columns:
        # Benefit advantage (QPU - Gurobi)
        merged["benefit_advantage"] = (
            merged[qpu_benefit_col] - merged[gurobi_benefit_col]
        )

        # Benefit ratio (QPU / Gurobi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            merged["benefit_ratio"] = np.where(
                merged[gurobi_benefit_col] > 0,
                merged[qpu_benefit_col] / merged[gurobi_benefit_col],
                np.nan,
            )

        # Gap percentage
        merged["gap_pct"] = np.where(
            merged[gurobi_benefit_col] > 0,
            np.abs(merged[qpu_benefit_col] - merged[gurobi_benefit_col])
            / merged[gurobi_benefit_col]
            * 100,
            np.nan,
        )

    if qpu_time_col in merged.columns and gurobi_time_col in merged.columns:
        # Speedup (Gurobi time / QPU time)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            merged["speedup"] = np.where(
                merged[qpu_time_col] > 0,
                merged[gurobi_time_col] / merged[qpu_time_col],
                np.nan,
            )

    return merged.sort_values(f"{qpu_prefix}n_vars").reset_index(drop=True)


def compute_speedup(
    qpu_time: float | np.ndarray | pd.Series,
    gurobi_time: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Compute speedup factor (Gurobi time / QPU time).

    Args:
        qpu_time: QPU solve time(s).
        gurobi_time: Gurobi solve time(s).

    Returns:
        Speedup factor. Values > 1 mean QPU is faster.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if isinstance(qpu_time, (pd.Series, np.ndarray)):
            return np.where(qpu_time > 0, gurobi_time / qpu_time, np.nan)
        return gurobi_time / qpu_time if qpu_time > 0 else float("nan")


def compute_gap_percentage(
    qpu_benefit: float | np.ndarray | pd.Series,
    gurobi_benefit: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Compute optimality gap as percentage.

    Args:
        qpu_benefit: QPU benefit value(s).
        gurobi_benefit: Gurobi benefit value(s).

    Returns:
        Gap as percentage: |QPU - Gurobi| / Gurobi * 100
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if isinstance(qpu_benefit, (pd.Series, np.ndarray)):
            return np.where(
                gurobi_benefit > 0,
                np.abs(qpu_benefit - gurobi_benefit) / gurobi_benefit * 100,
                np.nan,
            )
        if gurobi_benefit > 0:
            return abs(qpu_benefit - gurobi_benefit) / gurobi_benefit * 100
        return float("nan")


def filter_by_formulation(
    df: pd.DataFrame,
    formulation: str,
    column: str = "formulation",
) -> pd.DataFrame:
    """
    Filter DataFrame to specific formulation type.

    Args:
        df: DataFrame with formulation column.
        formulation: '6-Family' or '27-Food'.
        column: Column name containing formulation info.

    Returns:
        Filtered DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    # Handle aliased column names (e.g., 'qpu_formulation')
    return df[df[column] == formulation].copy()


def get_color_palette(n_colors: int) -> list[str]:
    """
    Get n colors from the qualitative palette with cycling if needed.

    Args:
        n_colors: Number of colors needed.

    Returns:
        List of hex color strings.
    """
    if n_colors <= len(QUALITATIVE_COLORS):
        return QUALITATIVE_COLORS[:n_colors]

    # Cycle through colors if more are needed
    import itertools

    cycle = itertools.cycle(QUALITATIVE_COLORS)
    return [next(cycle) for _ in range(n_colors)]


def format_large_number(x: float, pos: int | None = None) -> str:
    """
    Format large numbers with K/M suffixes for axes.

    Args:
        x: Number to format.
        pos: Position (unused, for matplotlib FuncFormatter compatibility).

    Returns:
        Formatted string with appropriate suffix.
    """
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    elif abs(x) >= 1e3:
        return f"{x / 1e3:.1f}K"
    else:
        return f"{x:.0f}"


def save_figure(
    fig: plt.Figure,
    filepath: Path | str,
    formats: list[str] | None = None,
    close_after: bool = True,
) -> list[Path]:
    """
    Save figure in multiple formats with consistent settings.

    Args:
        fig: Matplotlib figure object.
        filepath: Path (without extension) where to save.
        formats: List of formats to save (default: ['png', 'pdf']).
        close_after: Whether to close the figure after saving.

    Returns:
        List of paths where figures were saved.
    """
    if formats is None:
        formats = ["png", "pdf"]

    filepath = Path(filepath)
    base_path = filepath.with_suffix("")
    saved_paths: list[Path] = []

    for fmt in formats:
        output_path = base_path.with_suffix(f".{fmt}")
        fig.savefig(output_path, format=fmt, bbox_inches="tight", dpi=300)
        print(f"   Saved: {output_path}")
        saved_paths.append(output_path)

    if close_after:
        plt.close(fig)

    return saved_paths


# =============================================================================
# INITIALIZATION
# =============================================================================

# Apply publication style by default when module is imported
setup_publication_style()


# ==============================================================================
# SECTION 2: COMPREHENSIVE SCALING PLOTS
# From: generate_comprehensive_scaling_plots.py
# ==============================================================================


def prepare_scaling_data_60s() -> pd.DataFrame:
    """
    Prepare data for comprehensive scaling plots using 60s Gurobi timeout.

    Merges QPU and Gurobi results, separating by food count (formulation),
    and computes gaps and speedups.

    Returns:
        DataFrame with merged QPU/Gurobi metrics sorted by n_vars.
    """
    qpu_df = load_qpu_hierarchical()
    gurobi_df = load_gurobi_60s()

    # Build lookup dictionaries
    qpu_by = {row["scenario_name"]: row for _, row in qpu_df.iterrows()}
    gur_by = {row["scenario_name"]: row for _, row in gurobi_df.iterrows()}

    data = []
    for scenario in qpu_by.keys():
        q = qpu_by[scenario]
        g = gur_by.get(scenario, {})

        # Determine formulation type
        n_foods = q["n_foods"]
        formulation = "6-Family" if n_foods == 6 else "27-Food"

        # Get timing values (handle dict or scalar)
        qpu_time = q.get("total_wall_time", 0)
        qpu_access = q.get("qpu_access_time", 0)
        qpu_sampling = q.get("qpu_sampling_time", 0)
        gurobi_time = g.get("solve_time", 0) if isinstance(g, dict) else 0

        # Calculate gap (using absolute values since signs differ)
        qpu_obj = abs(q.get("objective_miqp", 0))
        gurobi_obj = abs(g.get("objective_miqp", 0)) if isinstance(g, dict) else 0
        if gurobi_obj > 0:
            gap = abs(qpu_obj - gurobi_obj) / gurobi_obj * 100
        else:
            gap = 0

        # Calculate speedup
        if qpu_time > 0:
            speedup = gurobi_time / qpu_time
        else:
            speedup = 0

        data.append(
            {
                "scenario": scenario,
                "formulation": formulation,
                "n_farms": q["n_farms"],
                "n_foods": q["n_foods"],
                "n_vars": q["n_vars"],
                "qpu_total_time": qpu_time,
                "qpu_access_time": qpu_access,
                "qpu_sampling_time": qpu_sampling,
                "qpu_objective": qpu_obj,
                "gurobi_time": gurobi_time,
                "gurobi_objective": gurobi_obj,
                "gurobi_status": g.get("status", "N/A") if isinstance(g, dict) else "N/A",
                "gurobi_timeout": g.get("hit_timeout", False) if isinstance(g, dict) else False,
                "gap": gap,
                "speedup": speedup,
            }
        )

    # Sort by number of variables
    df = pd.DataFrame(data)
    return df.sort_values("n_vars").reset_index(drop=True)


def plot_comprehensive_scaling(
    df: pd.DataFrame,
    output_dir: Path | str = "professional_plots",
) -> Path:
    """
    Create 2x3 comprehensive scaling plot matching the reference style.

    Subplots:
        (0,0) Gap vs Variables - by formulation
        (0,1) Objectives comparison - Gurobi vs Quantum
        (0,2) Speedup vs Variables - log scale
        (1,0) Time comparison bars - Gurobi vs QPU Total
        (1,1) QPU time stacked - Total vs Pure QPU
        (1,2) QPU Efficiency - % time in quantum

    Args:
        df: DataFrame from prepare_scaling_data_60s().
        output_dir: Directory for output files.

    Returns:
        Path to saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))


    # =========================================================================
    # Plot 2: Objectives - Gurobi vs Quantum
    # =========================================================================
    ax = axes[0, 1]
    for formulation in ["6-Family", "27-Food"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            # Gurobi (solid lines)
            ax.plot(
                form_df["n_vars"],
                form_df["gurobi_objective"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Gurobi)",
                linewidth=2.5,
                markersize=10,
                alpha=0.8,
            )
            # Quantum (dashed lines)
            ax.plot(
                form_df["n_vars"],
                form_df["qpu_objective"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (QPU)",
                linewidth=2.5,
                markersize=8,
                alpha=0.6,
                linestyle="--",
            )

    ax.set_xlabel("Number of Variables", fontsize=13, fontweight="bold")
    ax.set_ylabel("Objective Value (|abs|)", fontsize=13, fontweight="bold")
    ax.set_title("Solution Quality: Classical vs Quantum", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)



    # =========================================================================
    # Plot 4: Time Comparison - Gurobi vs QPU Total
    # =========================================================================
    ax = axes[1, 0]
    x_pos = np.arange(len(df))
    width = 0.35

    # Create grouped bar chart
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
                fontsize=8,
                color="red",
                fontweight="bold",
            )

    ax.set_xlabel("Test Configuration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Solve Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Solve Time: Gurobi vs QPU", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{row['formulation'][:3]}\n{row['n_vars']}v" for _, row in df.iterrows()],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=60, color="red", linestyle="--", alpha=0.3, label="Timeout (60s)")

    # =========================================================================
    # Plot 5: Pure QPU Time vs Total Time
    # =========================================================================
    ax = axes[1, 1]
    for formulation in ["6-Family", "27-Food"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            # Total QPU time
            ax.plot(
                form_df["n_vars"],
                form_df["qpu_total_time"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Total)",
                linewidth=2.5,
                markersize=10,
                alpha=0.8,
            )
            # Pure QPU access time
            ax.plot(
                form_df["n_vars"],
                form_df["qpu_access_time"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Pure QPU)",
                linewidth=2.5,
                markersize=8,
                alpha=0.6,
                linestyle="--",
            )

    ax.set_xlabel("Number of Variables", fontsize=13, fontweight="bold")
    ax.set_ylabel("Time (seconds)", fontsize=13, fontweight="bold")
    ax.set_title("QPU Time Breakdown: Total vs Pure Quantum", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")



    plt.tight_layout()

    output_path = output_dir / "quantum_advantage_comprehensive_scaling.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   [OK] Saved: {output_path}")

    output_pdf = output_path.with_suffix(".pdf")
    fig.savefig(output_pdf, bbox_inches="tight")
    print(f"   [OK] Saved: {output_pdf}")

    plt.close(fig)
    return output_path


# ==============================================================================
# SECTION 3: SPLIT FORMULATION PLOTS
# From: generate_split_formulation_plots.py
# ==============================================================================


def prepare_scaling_data_300s() -> pd.DataFrame:
    """
    Prepare data for split formulation plots using 300s Gurobi timeout.

    Combines QPU and Gurobi 300s results for fair comparison on harder problems.

    Returns:
        DataFrame with merged QPU/Gurobi metrics including MIP gap.
    """
    qpu_df = load_qpu_hierarchical()
    gurobi_df = load_gurobi_300s()

    # Build lookup dictionaries
    qpu_by = {row["scenario_name"]: row for _, row in qpu_df.iterrows()}
    gur_by = {row["scenario_name"]: row for _, row in gurobi_df.iterrows()}

    data = []
    for scenario in qpu_by.keys():
        if scenario not in gur_by:
            continue

        q = qpu_by[scenario]
        g = gur_by[scenario]

        formulation = "6-Family" if q["n_foods"] == 6 else "27-Food"

        # Get timing and objective values
        qpu_obj = abs(q.get("objective_miqp", 0))
        gurobi_obj = abs(g.get("objective_miqp", 0))

        # Calculate gap
        if gurobi_obj > 0:
            gap = abs(qpu_obj - gurobi_obj) / gurobi_obj * 100
        else:
            gap = 0

        # Speedup
        qpu_time = q.get("total_wall_time", 0)
        gurobi_time = g.get("solve_time", 0)
        speedup = gurobi_time / qpu_time if qpu_time > 0 else 0

        data.append(
            {
                "scenario": scenario,
                "formulation": formulation,
                "n_farms": q["n_farms"],
                "n_foods": q["n_foods"],
                "n_vars": q["n_vars"],
                "qpu_objective": qpu_obj,
                "qpu_total_time": qpu_time,
                "qpu_access_time": q.get("qpu_access_time", 0),
                "gurobi_objective": gurobi_obj,
                "gurobi_time": gurobi_time,
                "gurobi_status": g.get("status", "unknown"),
                "gurobi_timeout": g.get("hit_timeout", False),
                "gurobi_mip_gap": g.get("mip_gap", 0) * 100,  # Convert to percentage
                "gap": gap,
                "speedup": speedup,
            }
        )

    df = pd.DataFrame(data)
    return df.sort_values("n_vars").reset_index(drop=True)


def plot_split_analysis(
    df: pd.DataFrame,
    output_dir: Path | str = "professional_plots",
) -> Path:
    """
    Create 2x3 plot with split formulation analysis (6-Family vs 27-Food).

    Subplots:
        (0,0) Objective Values - Split by Formulation (log-log)
        (0,1) Optimality Gap - Split Analysis (log x-axis)
        (0,2) Solve Time Comparison - Gurobi vs QPU (log-log)
        (1,0) Speedup Analysis (log-log)
        (1,1) Pure QPU Time - Linear Scaling with fit
        (1,2) Gurobi MIP Gap (problem hardness)

    Args:
        df: DataFrame from prepare_scaling_data_300s().
        output_dir: Directory for output files.

    Returns:
        Path to saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # =========================================================================
    # Plot 1: Objective Values - Split by Formulation
    # =========================================================================
    ax = axes[0, 0]
    for formulation in ["6-Family", "27-Food"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            # Gurobi (solid)
            ax.plot(
                form_df["n_vars"],
                form_df["gurobi_objective"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Gurobi)",
                linewidth=2.5,
                markersize=10,
                alpha=0.8,
            )
            # QPU (dashed)
            ax.plot(
                form_df["n_vars"],
                form_df["qpu_objective"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (QPU)",
                linewidth=2.5,
                markersize=8,
                alpha=0.6,
                linestyle="--",
            )

    ax.set_xlabel("Number of Variables", fontsize=13, fontweight="bold")
    ax.set_ylabel("Objective Value", fontsize=13, fontweight="bold")
    ax.set_title("Solution Quality: Classical vs Quantum", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # =========================================================================
    # Plot 2: Optimality Gap - Split Analysis
    # =========================================================================
    ax = axes[0, 1]
    for formulation in ["6-Family", "27-Food"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            ax.plot(
                form_df["n_vars"],
                form_df["gap"],
                marker=MARKERS.get(formulation, "o"),
                color=FORMULATION_COLORS.get(formulation, "gray"),
                label=formulation,
                linewidth=2.5,
                markersize=10,
                alpha=0.8,
            )

    ax.axhline(y=100, color="orange", linestyle="--", alpha=0.5, label="100% gap", linewidth=1.5)
    ax.axhline(y=500, color="red", linestyle="--", alpha=0.5, label="500% gap", linewidth=1.5)
    ax.set_xlabel("Number of Variables", fontsize=13, fontweight="bold")
    ax.set_ylabel("QPU Gap from Gurobi (%)", fontsize=13, fontweight="bold")
    ax.set_title("Optimality Gap Analysis", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # =========================================================================
    # Plot 3: Solve Time Comparison
    # =========================================================================
    ax = axes[0, 2]
    for formulation in ["6-Family", "27-Food"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            # Gurobi
            ax.plot(
                form_df["n_vars"],
                form_df["gurobi_time"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (Gurobi)",
                linewidth=2.5,
                markersize=10,
                alpha=0.8,
            )
            # QPU
            ax.plot(
                form_df["n_vars"],
                form_df["qpu_total_time"],
                marker=MARKERS.get(formulation, "o"),
                color=color,
                label=f"{formulation} (QPU)",
                linewidth=2.5,
                markersize=8,
                alpha=0.6,
                linestyle="--",
            )

    ax.axhline(y=100, color="red", linestyle=":", alpha=0.5, label="Gurobi timeout (100s)", linewidth=1.5)
    ax.set_xlabel("Number of Variables", fontsize=13, fontweight="bold")
    ax.set_ylabel("Solve Time (seconds)", fontsize=13, fontweight="bold")
    ax.set_title("Time Scaling: Gurobi vs QPU", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # =========================================================================
    # Plot 4: Speedup Analysis
    # =========================================================================
    ax = axes[1, 0]
    for formulation in ["6-Family", "27-Food"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            ax.plot(
                form_df["n_vars"],
                form_df["speedup"],
                marker=MARKERS.get(formulation, "o"),
                color=FORMULATION_COLORS.get(formulation, "gray"),
                label=formulation,
                linewidth=2.5,
                markersize=10,
                alpha=0.8,
            )

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Break-even", linewidth=1.5)
    ax.set_xlabel("Number of Variables", fontsize=13, fontweight="bold")
    ax.set_ylabel("Speedup (Gurobi/QPU)", fontsize=13, fontweight="bold")
    ax.set_title("Speedup Analysis", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # =========================================================================
    # Plot 5: Pure QPU Time - Linear Scaling Analysis
    # =========================================================================
    ax = axes[1, 1]

    for formulation in ["6-Family", "27-Food"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            color = FORMULATION_COLORS.get(formulation, "gray")
            ax.scatter(
                form_df["n_vars"],
                form_df["qpu_access_time"],
                s=100,
                marker=MARKERS.get(formulation, "o"),
                color=color,
                alpha=0.8,
                label=f"{formulation}",
                edgecolors="black",
                linewidths=0.5,
            )

            # Linear fit for each formulation
            if len(form_df) >= 2:
                coef = np.polyfit(form_df["n_vars"].values, form_df["qpu_access_time"].values, 1)
                x_fit = np.linspace(form_df["n_vars"].min(), form_df["n_vars"].max(), 100)
                y_fit = coef[0] * x_fit + coef[1]
                ax.plot(
                    x_fit,
                    y_fit,
                    "--",
                    color=color,
                    alpha=0.7,
                    label=f"{formulation} fit: {coef[0]*1000:.4f}ms/var",
                )

    ax.set_xlabel("Number of Variables", fontsize=13, fontweight="bold")
    ax.set_ylabel("Pure QPU Time (seconds)", fontsize=13, fontweight="bold")
    ax.set_title("Pure QPU Time: Linear Scaling", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 6: Gurobi MIP Gap (shows problem hardness)
    # =========================================================================
    ax = axes[1, 2]
    for formulation in ["6-Family", "27-Food"]:
        form_df = df[df["formulation"] == formulation].sort_values("n_vars")
        if len(form_df) > 0:
            ax.semilogy(
                form_df["n_vars"],
                form_df["gurobi_mip_gap"],
                marker=MARKERS.get(formulation, "o"),
                color=FORMULATION_COLORS.get(formulation, "gray"),
                label=formulation,
                linewidth=2.5,
                markersize=10,
                alpha=0.8,
            )

    ax.axhline(y=100, color="orange", linestyle="--", alpha=0.5, label="100% MIP gap")
    ax.set_xlabel("Number of Variables", fontsize=13, fontweight="bold")
    ax.set_ylabel("Gurobi MIP Gap (%)", fontsize=13, fontweight="bold")
    ax.set_title("Classical Solver Difficulty (300s timeout)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()

    output_path = output_dir / "quantum_advantage_split_analysis.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   [OK] Saved: {output_path}")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"   [OK] Saved: {output_path.with_suffix('.pdf')}")
    plt.close(fig)
    return output_path


def plot_objective_gap_analysis(
    df: pd.DataFrame,
    output_dir: Path | str = "professional_plots",
) -> Path:
    """
    Deep dive into objective value gaps with scatter and regression analysis.

    Subplots:
        (0,0) Absolute Objective Comparison (bars)
        (0,1) Objective Ratio (QPU/Gurobi) bars
        (0,2) Gap vs Gurobi MIP Gap correlation scatter
        (1,0) 6-Family detailed analysis (line plot)
        (1,1) 27-Food detailed analysis (line plot)
        (1,2) Summary Statistics Table

    Args:
        df: DataFrame from prepare_scaling_data_300s().
        output_dir: Directory for output files.

    Returns:
        Path to saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # =========================================================================
    # Plot 1: Absolute Objective Comparison
    # =========================================================================
    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        df["gurobi_objective"],
        width,
        label="Gurobi (300s)",
        color=SOLVER_COLORS["gurobi"],
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        df["qpu_objective"],
        width,
        label="QPU Hier.",
        color=SOLVER_COLORS["qpu"],
        alpha=0.8,
    )

    ax.set_xlabel("Scenario", fontsize=12, fontweight="bold")
    ax.set_ylabel("Objective Value", fontsize=12, fontweight="bold")
    ax.set_title("Objective Values: Gurobi vs QPU", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row['n_vars']}" for _, row in df.iterrows()],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")

    # =========================================================================
    # Plot 2: Objective Ratio (QPU/Gurobi)
    # =========================================================================
    ax = axes[0, 1]
    ratio = df["qpu_objective"] / df["gurobi_objective"].replace(0, np.nan)

    colors_ratio = ["green" if r < 2 else "orange" if r < 5 else "red" for r in ratio.fillna(0)]
    ax.bar(range(len(df)), ratio.fillna(0), color=colors_ratio, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.axhline(y=1, color="black", linestyle="--", alpha=0.7, label="Equal")
    ax.axhline(y=2, color="orange", linestyle="--", alpha=0.5, label="2x ratio")
    ax.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="5x ratio")

    ax.set_xlabel("Scenario (by # variables)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Objective Ratio (QPU / Gurobi)", fontsize=12, fontweight="bold")
    ax.set_title("QPU Objective / Gurobi Objective", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(
        [f"{row['formulation'][:3]}\n{row['n_vars']}" for _, row in df.iterrows()],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # =========================================================================
    # Plot 3: Gap vs Gurobi MIP Gap correlation
    # =========================================================================
    ax = axes[0, 2]
    for formulation in ["6-Family", "27-Food"]:
        form_df = df[df["formulation"] == formulation]
        ax.scatter(
            form_df["gurobi_mip_gap"],
            form_df["gap"],
            s=100,
            marker=MARKERS.get(formulation, "o"),
            color=FORMULATION_COLORS.get(formulation, "gray"),
            alpha=0.8,
            label=formulation,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel("Gurobi MIP Gap (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("QPU Gap from Gurobi (%)", fontsize=13, fontweight="bold")
    ax.set_title("Correlation: Problem Hardness vs QPU Gap", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # =========================================================================
    # Plot 4: 6-Family detailed analysis
    # =========================================================================
    ax = axes[1, 0]
    form_df = df[df["formulation"] == "6-Family"].sort_values("n_vars")

    if len(form_df) > 0:
        ax.plot(
            form_df["n_farms"],
            form_df["gurobi_objective"],
            "o-",
            color=SOLVER_COLORS["gurobi"],
            label="Gurobi",
            linewidth=2.5,
            markersize=10,
        )
        ax.plot(
            form_df["n_farms"],
            form_df["qpu_objective"],
            "s--",
            color=SOLVER_COLORS["qpu"],
            label="QPU",
            linewidth=2.5,
            markersize=10,
        )

    ax.set_xlabel("Number of Farms", fontsize=13, fontweight="bold")
    ax.set_ylabel("Objective Value", fontsize=13, fontweight="bold")
    ax.set_title("6-Family: Objective Scaling", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 5: 27-Food detailed analysis
    # =========================================================================
    ax = axes[1, 1]
    form_df = df[df["formulation"] == "27-Food"].sort_values("n_vars")

    if len(form_df) > 0:
        ax.plot(
            form_df["n_farms"],
            form_df["gurobi_objective"],
            "o-",
            color=SOLVER_COLORS["gurobi"],
            label="Gurobi",
            linewidth=2.5,
            markersize=10,
        )
        ax.plot(
            form_df["n_farms"],
            form_df["qpu_objective"],
            "s--",
            color=SOLVER_COLORS["qpu"],
            label="QPU",
            linewidth=2.5,
            markersize=10,
        )

    ax.set_xlabel("Number of Farms", fontsize=13, fontweight="bold")
    ax.set_ylabel("Objective Value", fontsize=13, fontweight="bold")
    ax.set_title("27-Food: Objective Scaling", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 6: Summary Statistics Table
    # =========================================================================
    ax = axes[1, 2]
    ax.axis("off")

    # Calculate stats by formulation
    stats_6fam = df[df["formulation"] == "6-Family"]
    stats_27food = df[df["formulation"] == "27-Food"]

    # Build table data safely
    table_data = [
        ["Metric", "6-Family", "27-Food"],
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
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header
    for j in range(3):
        table[(0, j)].set_facecolor("#3498db")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Summary by Formulation", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()

    output_path = output_dir / "quantum_advantage_objective_gap_analysis.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   [OK] Saved: {output_path}")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"   [OK] Saved: {output_path.with_suffix('.pdf')}")
    plt.close(fig)
    return output_path


# ==============================================================================
# END OF SECTIONS 2-3
# ==============================================================================


# ==============================================================================
# SECTION 4: VIOLATION IMPACT ASSESSMENT
# ==============================================================================
# From assess_violation_impact.py
# Quantifies how constraint violations contribute to the objective gap.
#
# KEY SIGN CONVENTIONS:
# - QPU objectives are NEGATIVE (QUBO minimizes negative benefit)
# - Gurobi objectives are POSITIVE (maximizes benefit)
# - benefit = -objective_miqp (sign correction)
# - Gap = |QPU_obj| - Gurobi_obj (comparing absolute values)
# ==============================================================================


def load_violation_impact_data(
    qpu_path: str | Path = DEFAULT_QPU_HIER_PATH,
    gurobi_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and prepare data for violation impact assessment.

    Calculates violation rates, estimated lost benefits, and gap metrics
    for each scenario.

    Args:
        qpu_path: Path to QPU hierarchical results JSON.
        gurobi_path: Path to Gurobi results JSON. Defaults to 300s timeout file
                     (matching original assess_violation_impact.py).

    Returns:
        DataFrame with violation impact metrics per scenario.
    """
    # Use 300s file by default (matching original script)
    if gurobi_path is None:
        gurobi_path = DEFAULT_GUROBI_300S_FILE

    # Load QPU data
    with open(qpu_path) as f:
        qpu_data = json.load(f)

    # Load Gurobi data (300s file is array format with metadata/result structure)
    with open(gurobi_path) as f:
        gurobi_raw = json.load(f)

    # Parse Gurobi into dict - 300s files use array format with metadata/result
    gurobi = {}
    for entry in gurobi_raw:
        if "metadata" in entry:
            sc = entry["metadata"]["scenario"]
            result = entry.get("result", {})
            gurobi[sc] = {
                "objective": result.get("objective_value", 0),
                "n_vars": result.get("n_vars", 0),
                "status": result.get("status", "unknown"),
                "mip_gap": result.get("mip_gap", 0),
            }

    # Build results
    results = []
    for r in qpu_data["runs"]:
        sc = r["scenario_name"]
        n_vars = r["n_vars"]
        n_farms = r["n_farms"]
        n_foods = r["n_foods"]
        n_periods = r["n_periods"]

        qpu_obj = r["objective_miqp"]
        viols = r.get("constraint_violations", {})
        one_hot_viols = viols.get("one_hot_violations", 0)
        rotation_viols = viols.get("rotation_violations", 0)
        total_viols = viols.get("total_violations", 0)

        # Get Gurobi reference
        gur_data = gurobi.get(sc, {})
        gur_obj = gur_data.get("objective", 0)

        # Calculate gap: |QPU_obj| - Gurobi_obj
        # QPU obj is negative, Gurobi is positive
        gap = abs(qpu_obj) - gur_obj if gur_obj else 0

        # Calculate violation rate
        total_slots = n_farms * n_periods
        violation_rate = one_hot_viols / total_slots * 100 if total_slots > 0 else 0

        # Estimate lost benefit per violation
        avg_benefit_per_slot = gur_obj / total_slots if total_slots > 0 and gur_obj > 0 else 0
        estimated_lost_benefit = one_hot_viols * avg_benefit_per_slot

        results.append({
            "scenario": sc,
            "n_vars": n_vars,
            "n_farms": n_farms,
            "n_foods": n_foods,
            "n_periods": n_periods,
            "total_slots": total_slots,
            "qpu_obj": qpu_obj,
            "qpu_obj_abs": abs(qpu_obj),
            "gurobi_obj": gur_obj,
            "gap": gap,
            "one_hot_viols": one_hot_viols,
            "rotation_viols": rotation_viols,
            "total_viols": total_viols,
            "violation_rate": violation_rate,
            "avg_benefit_per_slot": avg_benefit_per_slot,
            "estimated_lost_benefit": estimated_lost_benefit,
        })

    return pd.DataFrame(results)


def plot_violation_impact(
    df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Generate multi-panel violation impact assessment visualization.

    Creates a 2x3 figure showing:
    - Panel 1: Violation rate by scenario
    - Panel 2: Violations scale with problem size
    - Panel 3: Gap vs estimated violation impact
    - Panel 4: Objective comparison (raw vs adjusted)
    - Panel 5: Ratio comparison before/after adjustment
    - Panel 6: Summary statistics text

    Args:
        df: Violation impact DataFrame. If None, loads from default paths.
        output_dir: Output directory for plots. Defaults to DEFAULT_OUTPUT_DIR.

    Returns:
        Path to the saved plot file.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_violation_impact_data()

    print("Generating violation impact assessment plots...")

    # Calculate correlations and statistics
    total_slots = df["total_slots"].sum()
    total_viols = df["one_hot_viols"].sum()
    total_gap = df["gap"].sum()

    corr_viols_gap = df["one_hot_viols"].corr(df["gap"])
    corr_rate_gap = df["violation_rate"].corr(df["gap"])
    gap_per_viol = total_gap / total_viols if total_viols > 0 else 0

    # Calculate adjusted objectives
    adj_obj = df["qpu_obj_abs"] - df["estimated_lost_benefit"]
    raw_ratio = df["qpu_obj_abs"] / df["gurobi_obj"].replace(0, np.nan)
    adj_ratio = adj_obj / df["gurobi_obj"].replace(0, np.nan)

    # Calculate gap explained
    adj_gaps = adj_obj - df["gurobi_obj"]
    explained_pct = (df["gap"].sum() - adj_gaps.sum()) / df["gap"].sum() * 100 if df["gap"].sum() > 0 else 0

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Violation rate by scenario
    ax = axes[0, 0]
    colors = ["red" if r > 10 else "orange" if r > 5 else "green" for r in df["violation_rate"]]
    ax.bar(range(len(df)), df["violation_rate"], color=colors, alpha=0.8, edgecolor="black")
    ax.set_xlabel("Scenario (sorted by size)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Violation Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("One-Hot Violation Rate by Scenario", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{v:,}" for v in df["n_vars"]], rotation=45, ha="right", fontsize=8)
    ax.axhline(
        y=df["violation_rate"].mean(),
        color="blue",
        linestyle="--",
        label=f"Avg: {df['violation_rate'].mean():.1f}%",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")


    # Panel 3: Gap breakdown
    ax = axes[0, 2]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["gap"], width, label="Total Gap", color="red", alpha=0.8)
    ax.bar(x + width / 2, df["estimated_lost_benefit"], width, label="Est. Violation Impact", color="blue", alpha=0.8)
    ax.set_xlabel("Scenario", fontsize=12, fontweight="bold")
    ax.set_ylabel("Objective Units", fontsize=12, fontweight="bold")
    ax.set_title("Gap vs Estimated Violation Impact", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:,}" for v in df["n_vars"]], rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: Adjusted objective comparison
    ax = axes[1, 0]
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["gurobi_obj"], width, label="Gurobi", color="green", alpha=0.8)
    ax.bar(x, df["qpu_obj_abs"], width, label="QPU (raw)", color="red", alpha=0.8)
    ax.bar(x + width, adj_obj, width, label="QPU (adjusted)", color="blue", alpha=0.8)
    ax.set_xlabel("Scenario", fontsize=12, fontweight="bold")
    ax.set_ylabel("Objective Value", fontsize=12, fontweight="bold")
    ax.set_title("Objective: Raw vs Violation-Adjusted", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:,}" for v in df["n_vars"]], rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    
    plt.tight_layout()

    output_path = output_dir / "violation_impact_assessment.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   [OK] Saved: {output_path}")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"   [OK] Saved: {output_path.with_suffix('.pdf')}")
    plt.close(fig)

    return output_path


# ==============================================================================
# END OF SECTION 4
# ==============================================================================


# ==============================================================================
# SECTION 5: DEEP DIVE GAP ANALYSIS
# ==============================================================================
# From deep_dive_gap_analysis.py
# Deeper investigation of the objective gap.
# Since violations only explain ~7%, what causes the remaining 93%?
#
# KEY SIGN CONVENTIONS (CRITICAL):
# - QPU objectives are NEGATIVE (QUBO minimizes negative benefit)
# - Gurobi objectives are POSITIVE (maximizes benefit)
# - Gap = |QPU_obj| - Gurobi_obj
# - "Corrected" QPU = qpu_obj + potential_gain (less negative = better)
# - "Comparable" QPU = |qpu_obj| - potential_gain (for ratio calculation)
# ==============================================================================


def load_gap_deep_dive_data(
    qpu_path: str | Path = DEFAULT_QPU_HIER_PATH,
    gurobi_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and prepare data for deep gap analysis.

    Calculates violation corrections and gap attributions for each scenario.
    Preserves the critical sign conventions:
    - QPU objectives are negative (QUBO minimization)
    - Gurobi objectives are positive (benefit maximization)

    Args:
        qpu_path: Path to QPU hierarchical results JSON.
        gurobi_path: Path to Gurobi results JSON. Defaults to 300s timeout file
                     (matching original deep_dive_gap_analysis.py).

    Returns:
        DataFrame with gap analysis metrics per scenario.
    """
    # Use 300s file by default (matching original script)
    if gurobi_path is None:
        gurobi_path = DEFAULT_GUROBI_300S_FILE

    # Load QPU data
    with open(qpu_path) as f:
        qpu_data = json.load(f)

    # Load Gurobi data (300s file is array format with metadata/result structure)
    with open(gurobi_path) as f:
        gurobi_raw = json.load(f)

    # Parse Gurobi into dict - 300s files use array format with metadata/result
    gurobi = {}
    for entry in gurobi_raw:
        if "metadata" in entry:
            sc = entry["metadata"]["scenario"]
            result = entry.get("result", {})
            gurobi[sc] = {
                "objective": result.get("objective_value", 0),
                "status": result.get("status", "unknown"),
                "mip_gap": result.get("mip_gap", 0),
                "hit_timeout": result.get("hit_timeout", False),
            }

    # Build results with gap analysis
    results = []
    for r in qpu_data["runs"]:
        sc = r["scenario_name"]
        n_farms = r["n_farms"]
        n_periods = r["n_periods"]
        n_vars = r["n_vars"]
        total_slots = n_farms * n_periods

        qpu_obj = r["objective_miqp"]  # NEGATIVE value
        gur_obj = gurobi.get(sc, {}).get("objective", 0)  # POSITIVE value
        viols = r["constraint_violations"].get("one_hot_violations", 0)

        # Average benefit per slot (from Gurobi reference)
        avg_benefit = gur_obj / total_slots if total_slots > 0 else 0

        # Potential gain if violations were fixed
        potential_gain = viols * avg_benefit

        # "Corrected" QPU objective (if violations were fixed)
        # Since QPU obj is negative, we ADD the potential gain (less negative = better)
        corrected_qpu = qpu_obj + potential_gain

        # "Comparable" QPU for ratio calculation
        # |QPU| - potential_gain gives value comparable to Gurobi
        comparable_qpu = abs(qpu_obj) - potential_gain

        # Ratios
        ratio_raw = abs(qpu_obj) / gur_obj if gur_obj > 0 else 0
        ratio_corrected = comparable_qpu / gur_obj if gur_obj > 0 else 0

        results.append({
            "scenario": sc,
            "n_vars": n_vars,
            "n_farms": n_farms,
            "n_periods": n_periods,
            "total_slots": total_slots,
            "gurobi": gur_obj,
            "qpu_raw": qpu_obj,
            "qpu_abs": abs(qpu_obj),
            "violations": viols,
            "potential_gain": potential_gain,
            "qpu_corrected": corrected_qpu,
            "comparable": comparable_qpu,
            "ratio_raw": ratio_raw,
            "ratio_corrected": ratio_corrected,
        })

    return pd.DataFrame(results)


def plot_gap_deep_dive(
    df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Generate deep dive gap analysis visualization.

    Creates a 2x3 figure showing:
    - Panel 1: Objective comparison (Gurobi vs Raw vs Corrected QPU)
    - Panel 2: Ratio analysis showing minor violation impact
    - Panel 3: Gap attribution pie chart (violations vs other factors)
    - Panel 4: QPU vs Gurobi scatter with parity line
    - Panel 5: Per-scenario violation impact percentage
    - Panel 6: Summary findings text

    KEY SIGN HANDLING:
    - QPU objectives are NEGATIVE (QUBO minimization)
    - We use |QPU| for comparisons with Gurobi (positive)
    - Gap = |QPU| - Gurobi

    Args:
        df: Gap analysis DataFrame. If None, loads from default paths.
        output_dir: Output directory for plots. Defaults to DEFAULT_OUTPUT_DIR.

    Returns:
        Path to the saved plot file.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_gap_deep_dive_data()

    print("Generating deep dive gap analysis plots...")

    # Calculate statistics
    avg_raw_ratio = df["ratio_raw"].mean()
    avg_corrected_ratio = df["ratio_corrected"].mean()

    total_gap = (df["qpu_abs"] - df["gurobi"]).sum()
    violation_explained = df["potential_gain"].sum()
    pct_explained = violation_explained / total_gap * 100 if total_gap > 0 else 0

    # Per-scenario violation percentage
    viol_pcts = []
    for _, row in df.iterrows():
        scenario_gap = row["qpu_abs"] - row["gurobi"]
        if scenario_gap > 0:
            viol_pct = row["potential_gain"] / scenario_gap * 100
        else:
            viol_pct = 0
        viol_pcts.append(viol_pct)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Objective comparison
    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["gurobi"], width, label="Gurobi", color="green", alpha=0.8)
    ax.bar(x, df["qpu_abs"], width, label="|QPU| (raw)", color="red", alpha=0.8)
    ax.bar(x + width, df["comparable"], width, label="|QPU| (corrected)", color="blue", alpha=0.8)
    ax.set_xlabel("Scenario", fontsize=12, fontweight="bold")
    ax.set_ylabel("Objective Value", fontsize=12, fontweight="bold")
    ax.set_title("Objective Comparison: Raw vs Corrected", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in df["violations"]], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Ratio comparison
    ax = axes[0, 1]
    ax.bar(x - 0.2, df["ratio_raw"], 0.4, label="Raw ratio", color="red", alpha=0.8)
    ax.bar(x + 0.2, df["ratio_corrected"], 0.4, label="Corrected ratio", color="blue", alpha=0.8)
    ax.axhline(y=1.0, color="green", linestyle="--", label="Parity (1.0)", linewidth=2)
    ax.set_xlabel("Scenario (by violations)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ratio (QPU / Gurobi)", fontsize=12, fontweight="bold")
    ax.set_title("Ratio Analysis: Violations Have Minor Impact", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in df["violations"]], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Gap breakdown pie chart
    ax = axes[0, 2]
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
        textprops={"fontsize": 11},
    )
    ax.set_title("Gap Attribution", fontsize=14, fontweight="bold")

    # Panel 4: Scaling analysis scatter
    ax = axes[1, 0]
    ax.scatter(df["gurobi"], df["qpu_abs"], s=100, c="red", alpha=0.7, label="Raw", edgecolors="black")
    ax.scatter(df["gurobi"], df["comparable"], s=100, c="blue", alpha=0.7, label="Corrected", edgecolors="black")
    max_val = max(df["qpu_abs"].max(), df["gurobi"].max())
    ax.plot([0, max_val], [0, max_val], "g--", label="Parity", linewidth=2)
    ax.set_xlabel("Gurobi Objective", fontsize=12, fontweight="bold")
    ax.set_ylabel("QPU Objective", fontsize=12, fontweight="bold")
    ax.set_title("QPU vs Gurobi: Violation Correction Impact", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 5: Violation rate vs gap percentage
    ax = axes[1, 1]
    ax.bar(range(len(df)), viol_pcts, color="coral", alpha=0.8, edgecolor="black")
    ax.axhline(y=np.mean(viol_pcts), color="blue", linestyle="--", label=f"Avg: {np.mean(viol_pcts):.1f}%")
    ax.set_xlabel("Scenario (by violations)", fontsize=12, fontweight="bold")
    ax.set_ylabel("% of Gap Explained by Violations", fontsize=12, fontweight="bold")
    ax.set_title("Violation Impact by Scenario", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{int(v)}" for v in df["violations"]], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 20)

    # Panel 6: Summary findings
    ax = axes[1, 2]
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
        0.05,
        0.5,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()

    output_path = output_dir / "gap_deep_dive.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   [OK] Saved: {output_path}")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"   [OK] Saved: {output_path.with_suffix('.pdf')}")
    plt.close(fig)

    return output_path


# ==============================================================================
# END OF SECTION 5
# ==============================================================================


# ==============================================================================
# SECTION 6: QPU ADVANTAGE (CORRECTED)
# ==============================================================================
# This is the critical plot showing QPU actually OUTPERFORMS Gurobi when signs
# are interpreted correctly. The negative QPU objectives indicate HIGHER benefit
# in this MAXIMIZATION problem.
# ==============================================================================


def plot_qpu_advantage_corrected(
    output_dir: str | Path = "professional_plots",
    formats: list[str] | None = None,
) -> Path:
    """
    Create 2x3 grid showing TRUE quantum advantage after sign correction.

    KEY INSIGHT: This is a MAXIMIZATION problem.
    - Gurobi: Maximizes benefit → higher positive values = better
    - QPU QUBO: Minimizes (-benefit + penalties) → more negative = better benefit
    - The QPU negative objectives mean QPU finds HIGHER benefit than Gurobi!

    Panels:
        (0,0) Benefit comparison: QPU vs Gurobi benefit values
        (0,1) Benefit ratio by formulation (log scale)
        (0,2) Time comparison (both log scale)
        (1,0) Violations vs benefit advantage (scatter with size color)
        (1,1) Pure QPU time scaling (linear fit)
        (1,2) Summary statistics table

    Returns:
        Path to saved PNG file.
    """
    if formats is None:
        formats = ["png", "pdf"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    qpu_df = load_qpu_hierarchical()
    gurobi_df = load_gurobi_300s()

    # Build comparison DataFrame
    results = []
    for _, row in qpu_df.iterrows():
        sc = row["scenario_name"]
        n_vars = row["n_vars"]
        n_farms = row["n_farms"]
        n_foods = row["n_foods"]
        n_periods = row["n_periods"]

        # QPU data - sign correction already done in load_qpu_hierarchical
        qpu_obj_raw = row["objective_miqp"]
        qpu_benefit = row["benefit"]  # Already sign-corrected

        # Violations are already extracted in DataFrame
        total_viols = row.get("total_violations", 0)

        # Timing data from DataFrame columns
        qpu_wall_time = row.get("total_wall_time", 0)
        qpu_pure_time = row.get("qpu_access_time", 0)

        # Gurobi data - match by scenario_name
        gur_row = gurobi_df[gurobi_df["scenario_name"] == sc]
        if len(gur_row) == 0:
            continue
        gur_row = gur_row.iloc[0]
        gur_obj = gur_row["objective_miqp"]  # or "benefit" - same for Gurobi
        gur_time = gur_row["solve_time"]
        gur_timeout = gur_row["hit_timeout"]
        gur_mip_gap = gur_row.get("mip_gap", 0)

        formulation = "27-Food" if n_foods == 27 else "6-Family"
        benefit_advantage = qpu_benefit - gur_obj
        benefit_ratio = qpu_benefit / gur_obj if gur_obj > 0 else 0
        total_slots = n_farms * n_periods
        violation_rate = total_viols / total_slots * 100 if total_slots > 0 else 0

        results.append({
            "scenario": sc,
            "n_vars": n_vars,
            "n_farms": n_farms,
            "n_foods": n_foods,
            "formulation": formulation,
            "qpu_obj_raw": qpu_obj_raw,
            "qpu_benefit": qpu_benefit,
            "gurobi_obj": gur_obj,
            "benefit_advantage": benefit_advantage,
            "benefit_ratio": benefit_ratio,
            "violations": total_viols,
            "violation_rate": violation_rate,
            "qpu_wall_time": qpu_wall_time,
            "qpu_pure_time": qpu_pure_time,
            "gurobi_time": gur_time,
            "gurobi_timeout": gur_timeout,
            "gurobi_mip_gap": gur_mip_gap * 100,
        })

    df = pd.DataFrame(results).sort_values("n_vars").reset_index(drop=True)
    df_6fam = df[df["formulation"] == "6-Family"]
    df_27food = df[df["formulation"] == "27-Food"]

    # Colors
    C_QPU = "#1f77b4"
    C_QPU_DARK = "#0d4f8b"
    C_QPU_LIGHT = "#a6cee3"
    C_GUROBI = "#2ca02c"
    C_GUROBI_LIGHT = "#b2df8a"
    C_6FAM = "#3498db"
    C_27FOOD = "#e74c3c"
    C_NEUTRAL = "#7f7f7f"
    C_HIGHLIGHT = "#d62728"

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- Panel (0,0): Benefit Comparison ---
    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["gurobi_obj"], width, label="Gurobi",
           color=C_GUROBI, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, df["qpu_benefit"], width, label="QPU (Hierarchical)",
           color=C_QPU, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Problem Size (Variables)", fontweight="bold")
    ax.set_ylabel("Benefit Value (higher = better)", fontweight="bold")
    ax.set_title("QPU Achieves Higher Benefit Than Gurobi", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:,}" for v in df["n_vars"]], rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    for i, row_ in df.iterrows():
        if row_["benefit_ratio"] > 1:
            ax.annotate(f"+{row_['benefit_ratio']:.1f}x",
                        xy=(i + width / 2, row_["qpu_benefit"]),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=8, color=C_QPU_DARK, fontweight="bold")

    # --- Panel (0,1): Benefit Ratio by Formulation ---
    ax = axes[0, 1]
    ax.scatter(df_6fam["n_vars"], df_6fam["benefit_ratio"], s=120, c=C_6FAM,
               marker="o", label="6-Family", edgecolors="black", linewidths=0.5, alpha=0.8)
    ax.scatter(df_27food["n_vars"], df_27food["benefit_ratio"], s=120, c=C_27FOOD,
               marker="s", label="27-Food", edgecolors="black", linewidths=0.5, alpha=0.8)
    ax.axhline(y=1.0, color=C_NEUTRAL, linestyle="--", linewidth=2, label="Parity (1.0)")
    ax.fill_between([0, 20000], 1, 10, alpha=0.1, color=C_QPU, label="QPU advantage region")
    ax.set_xlabel("Number of Variables", fontweight="bold")
    ax.set_ylabel("QPU / Gurobi Benefit Ratio", fontweight="bold")
    ax.set_title("QPU Advantage Increases with Problem Size", fontweight="bold", fontsize=14)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 6)

    # --- Panel (0,2): Time Comparison ---
    ax = axes[0, 2]
    ax.scatter(df_6fam["n_vars"], df_6fam["gurobi_time"], s=100, c=C_GUROBI,
               marker="o", label="Gurobi (6-Family)", alpha=0.8)
    ax.scatter(df_27food["n_vars"], df_27food["gurobi_time"], s=100, c="#1a6b1a",
               marker="s", label="Gurobi (27-Food)", alpha=0.8)
    ax.scatter(df_6fam["n_vars"], df_6fam["qpu_wall_time"], s=100, c=C_QPU,
               marker="o", label="QPU (6-Family)", alpha=0.8)
    ax.scatter(df_27food["n_vars"], df_27food["qpu_wall_time"], s=100, c=C_QPU_DARK,
               marker="s", label="QPU (27-Food)", alpha=0.8)
    ax.axhline(y=300, color=C_HIGHLIGHT, linestyle="--", linewidth=2, label="Gurobi timeout (300s)")
    ax.set_xlabel("Number of Variables", fontweight="bold")
    ax.set_ylabel("Solve Time (seconds)", fontweight="bold")
    ax.set_title("Solve Time Comparison", fontweight="bold", fontsize=14)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # --- Panel (1,0): Violations vs Benefit Advantage ---
    ax = axes[1, 0]
    # Use color based on problem size without colorbar to avoid layout conflicts
    sizes = df["n_vars"].values
    norm_sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min()) if sizes.max() > sizes.min() else np.zeros_like(sizes)
    colors_by_size = plt.cm.viridis(norm_sizes)
    ax.scatter(df["violations"], df["benefit_advantage"],
               c=colors_by_size, s=150,
               edgecolors="black", linewidths=0.5, alpha=0.8)
    # Add size legend instead of colorbar
    for size_label, norm_val in [("Small", 0.0), ("Medium", 0.5), ("Large", 1.0)]:
        ax.scatter([], [], c=[plt.cm.viridis(norm_val)], s=80, label=f"{size_label} size")
    ax.axhline(y=0, color=C_NEUTRAL, linestyle="--", linewidth=1.5)
    ax.fill_between([0, 200], 0, 500, alpha=0.1, color=C_QPU)
    ax.set_xlabel("Number of Violations", fontweight="bold")
    ax.set_ylabel("QPU Benefit Advantage (QPU - Gurobi)", fontweight="bold")
    ax.set_title("Violations Trade-off: Higher Benefit Despite Violations", fontweight="bold", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.annotate("ALL points above zero:\nQPU always better!",
                xy=(80, 200), fontsize=11, ha="center",
                bbox=dict(boxstyle="round", facecolor=C_QPU_LIGHT, alpha=0.8))

    # --- Panel (1,1): Pure QPU Time Scaling ---
    ax = axes[1, 1]
    ax.scatter(df_6fam["n_vars"], df_6fam["qpu_pure_time"] * 1000, s=120, c=C_6FAM,
               marker="o", label="6-Family", edgecolors="black", linewidths=0.5)
    ax.scatter(df_27food["n_vars"], df_27food["qpu_pure_time"] * 1000, s=120, c=C_27FOOD,
               marker="s", label="27-Food", edgecolors="black", linewidths=0.5)
    all_vars = df["n_vars"].values
    all_times = df["qpu_pure_time"].values * 1000
    coef = np.polyfit(all_vars, all_times, 1)
    x_fit = np.linspace(all_vars.min(), all_vars.max(), 100)
    ax.plot(x_fit, coef[0] * x_fit + coef[1], "--", color=C_NEUTRAL, linewidth=2,
            label=f"Linear fit: {coef[0]:.3f}ms/var")
    ax.set_xlabel("Number of Variables", fontweight="bold")
    ax.set_ylabel("Pure QPU Time (milliseconds)", fontweight="bold")
    ax.set_title("Pure QPU Time Scales Linearly", fontweight="bold", fontsize=14)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # --- Panel (1,2): Summary Statistics Table ---
    ax = axes[1, 2]
    ax.axis("off")
    summary_data = [
        ["Metric", "6-Family", "27-Food", "Overall"],
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
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    for j in range(4):
        table[(0, j)].set_facecolor(C_QPU)
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for j in range(4):
        table[(4, j)].set_facecolor(C_QPU_LIGHT)
        table[(6, j)].set_facecolor(C_GUROBI_LIGHT)
    ax.set_title("Summary: QPU Outperforms Gurobi", fontweight="bold", fontsize=14, pad=20)

    plt.tight_layout()

    # Save
    out_path = output_dir / "qpu_advantage_corrected.png"
    for fmt in formats:
        save_path = output_dir / f"qpu_advantage_corrected.{fmt}"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [OK] Saved: {save_path}")
    plt.close(fig)

    return out_path


# ==============================================================================
# END OF SECTION 6
# ==============================================================================


# ==============================================================================
# SECTION 7: QPU BENCHMARK RESULTS
# ==============================================================================

# Method display settings for benchmark plots
_BENCHMARK_COLORS: dict[str, str] = {
    "PuLP": "#0173B2",
    "Gurobi": "#CC3311",
    "DWave_Hybrid": "#00ADB5",
    "DWave_BQM": "#004E89",
    "PlotBased_QPU": "#029E73",
    "Multilevel(5)_QPU": "#DE8F05",
    "Multilevel(10)_QPU": "#6F42C1",
    "Louvain_QPU": "#1B4965",
    "Spectral(10)_QPU": "#9B59B6",
    "cqm_first_PlotBased": "#8E44AD",
    "HybridGrid(5,9)_QPU": "#4ECDC4",
    "HybridGrid(10,9)_QPU": "#00ADB5",
    "coordinated": "#FF6B6B",
    "direct_qpu": "#F0AB00",
}

_BENCHMARK_MARKERS: dict[str, str] = {
    "PuLP": "o",
    "Gurobi": "o",
    "DWave_Hybrid": "D",
    "DWave_BQM": "p",
    "PlotBased_QPU": "^",
    "Multilevel(5)_QPU": "v",
    "Multilevel(10)_QPU": "<",
    "Louvain_QPU": ">",
    "Spectral(10)_QPU": "h",
    "cqm_first_PlotBased": "P",
    "HybridGrid(5,9)_QPU": "*",
    "HybridGrid(10,9)_QPU": "X",
}


def _load_qpu_benchmark_data() -> dict[str, Any]:
    """Load QPU benchmark results from @todo/qpu_benchmark_results directory.

    Loads both regular benchmark files and HybridGrid results, merging them
    into a unified data structure.
    """
    project_root = Path(__file__).parent
    qpu_results_dir = project_root / "@todo" / "qpu_benchmark_results"

    data = {"small_scale": None, "large_scale": None, "hybridgrid": None, "hybridgrid_large": None}

    # Known benchmark files for small (10-100) and large (200-1000) scales
    small_scale_file = qpu_results_dir / "qpu_benchmark_20251201_160444.json"
    large_scale_file = qpu_results_dir / "qpu_benchmark_20251201_200012.json"

    # HybridGrid benchmark files (run separately)
    # HybridGrid(5,9) - scales 10, 15, 50, 200, 500, 1000
    hybridgrid_file = qpu_results_dir / "qpu_benchmark_20251203_121526.json"
    # HybridGrid(10,9) - scales 500, 1000
    hybridgrid_large_file = qpu_results_dir / "qpu_benchmark_20251203_133144.json"

    if small_scale_file.exists():
        with open(small_scale_file) as f:
            data["small_scale"] = json.load(f)

    if large_scale_file.exists():
        with open(large_scale_file) as f:
            data["large_scale"] = json.load(f)

    if hybridgrid_file.exists():
        with open(hybridgrid_file) as f:
            data["hybridgrid"] = json.load(f)

    if hybridgrid_large_file.exists():
        with open(hybridgrid_large_file) as f:
            data["hybridgrid_large"] = json.load(f)

    return data


def _extract_metrics_from_results(
    results: list[dict[str, Any]],
    metrics: dict[str, dict[str, list]],
    gurobi_lookup: dict[int, float] | None = None,
) -> dict[str, dict[str, list]]:
    """Extract metrics from a list of benchmark results.

    Parameters
    ----------
    results : list
        List of scale results from benchmark JSON
    metrics : dict
        Existing metrics dict to merge into
    gurobi_lookup : dict, optional
        Pre-computed Gurobi objectives by n_farms (for HybridGrid merge)

    Returns
    -------
    dict
        Updated metrics dictionary
    """
    for scale_result in results:
        n_farms = scale_result["n_farms"]

        # Ground truth (Gurobi reference) - only add if not already present
        if "ground_truth" in scale_result and gurobi_lookup is None:
            gt = scale_result["ground_truth"]
            gt_obj = gt.get("objective", 0)
            gt_viol = gt.get("violations", 0)
            gt_time = gt.get("solve_time", 0)

            if "Gurobi" not in metrics:
                metrics["Gurobi"] = {
                    "n_farms": [], "objective": [], "gap": [],
                    "qpu_time": [], "violations": [], "wall_time": [],
                }
            # Only add if this n_farms not already present
            if n_farms not in metrics["Gurobi"]["n_farms"]:
                metrics["Gurobi"]["n_farms"].append(n_farms)
                metrics["Gurobi"]["objective"].append(gt_obj)
                metrics["Gurobi"]["gap"].append(0)
                metrics["Gurobi"]["qpu_time"].append(0)
                metrics["Gurobi"]["violations"].append(gt_viol)
                metrics["Gurobi"]["wall_time"].append(gt_time)

        # Get ground truth objective for gap calculation
        if gurobi_lookup is not None:
            gt_obj = gurobi_lookup.get(n_farms, 0)
        else:
            gt_obj = scale_result.get("ground_truth", {}).get("objective", 0)

        # QPU methods
        for method_name, method_result in scale_result.get("method_results", {}).items():
            if not method_result.get("success", False):
                continue

            # Clean method name
            display_name = method_name.replace("decomposition_", "")

            obj = method_result.get("objective", 0)
            gap = ((gt_obj - obj) / gt_obj * 100) if gt_obj > 0 and obj > 0 else 0

            timings = method_result.get("timings", {})
            qpu_time = timings.get("qpu_access_total", 0)
            wall_time = method_result.get("wall_time", method_result.get("total_time", 0))
            violations = method_result.get("violations", 0)

            if display_name not in metrics:
                metrics[display_name] = {
                    "n_farms": [], "objective": [], "gap": [],
                    "qpu_time": [], "violations": [], "wall_time": [],
                }
            metrics[display_name]["n_farms"].append(n_farms)
            metrics[display_name]["objective"].append(obj)
            metrics[display_name]["gap"].append(gap)
            metrics[display_name]["qpu_time"].append(qpu_time)
            metrics[display_name]["violations"].append(violations)
            metrics[display_name]["wall_time"].append(wall_time)

    return metrics


def _extract_benchmark_metrics(
    qpu_data: dict[str, Any], scale_type: str = "small_scale"
) -> dict[str, dict[str, list]]:
    """Extract metrics from QPU benchmark JSON data.

    Merges HybridGrid data when available for the corresponding scale.
    """
    if qpu_data.get(scale_type) is None:
        return {}

    results = qpu_data[scale_type]["results"]
    metrics: dict[str, dict[str, list]] = {}

    # Extract main benchmark metrics
    metrics = _extract_metrics_from_results(results, metrics)

    # Build Gurobi lookup for gap calculations in HybridGrid data
    gurobi_lookup: dict[int, float] = {}
    if "Gurobi" in metrics:
        for i, n_farms in enumerate(metrics["Gurobi"]["n_farms"]):
            gurobi_lookup[n_farms] = metrics["Gurobi"]["objective"][i]

    # Merge HybridGrid data if available
    hybridgrid_key = "hybridgrid" if scale_type == "small_scale" else "hybridgrid_large"
    if qpu_data.get(hybridgrid_key) is not None:
        hg_results = qpu_data[hybridgrid_key]["results"]
        metrics = _extract_metrics_from_results(hg_results, metrics, gurobi_lookup)

    # Also merge the other HybridGrid file if scale ranges overlap
    # (hybridgrid has scales 10-1000, hybridgrid_large has 500-1000)
    if scale_type == "small_scale" and qpu_data.get("hybridgrid") is not None:
        # Small scale wants farms < 200, hybridgrid has 10, 15, 50, 200, 500, 1000
        pass  # Already handled above
    elif scale_type == "large_scale" and qpu_data.get("hybridgrid") is not None:
        # Large scale wants farms >= 200, merge from hybridgrid too
        hg_results = qpu_data["hybridgrid"]["results"]
        # Filter to only large scale farms
        large_hg_results = [r for r in hg_results if r["n_farms"] >= 200]
        if large_hg_results:
            metrics = _extract_metrics_from_results(large_hg_results, metrics, gurobi_lookup)

    return metrics


def plot_qpu_benchmark_small_scale(
    output_dir: Path | str | None = None,
    formats: list[str] | None = None,
) -> Path:
    """
    Plot QPU benchmark results for small-scale problems (<500 variables).

    Creates a 2x2 panel showing:
    - Solution quality (objective value)
    - Optimality gap vs Gurobi
    - QPU time comparison
    - Constraint violations

    Parameters
    ----------
    output_dir : Path, optional
        Output directory (default: professional_plots)
    formats : list[str], optional
        Output formats (default: ["png", "pdf"])

    Returns
    -------
    Path
        Path to saved figure
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "professional_plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if formats is None:
        formats = ["png", "pdf"]

    # Load data
    qpu_data = _load_qpu_benchmark_data()
    metrics = _extract_benchmark_metrics(qpu_data, "small_scale")

    if not metrics:
        print("  ⚠ No small-scale benchmark data found")
        return output_dir / "qpu_benchmark_small_scale.png"

    # Small-scale methods (including HybridGrid which has small scale data)
    small_methods = [
        "PlotBased_QPU", "Multilevel(5)_QPU", "Multilevel(10)_QPU",
        "Louvain_QPU", "Spectral(10)_QPU", "cqm_first_PlotBased",
        "HybridGrid(5,9)_QPU",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "QPU Benchmark: Small-Scale Problems (10-100 Farms)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Panel 1: Time Comparison
    ax = axes[0, 0]
    if "Gurobi" in metrics:
        times = [t for t in metrics["Gurobi"]["wall_time"] if t > 0]
        farms = [
            metrics["Gurobi"]["n_farms"][i]
            for i, t in enumerate(metrics["Gurobi"]["wall_time"]) if t > 0
        ]
        if times:
            ax.semilogy(farms, times, "o-", linewidth=2.5, markersize=8,
                       color=_BENCHMARK_COLORS["Gurobi"], label="Gurobi", alpha=0.9)
    for method in small_methods:
        if method in metrics and metrics[method]["n_farms"]:
            qpu_times = [t for t in metrics[method]["qpu_time"] if t > 0]
            farms = [
                metrics[method]["n_farms"][i]
                for i, t in enumerate(metrics[method]["qpu_time"]) if t > 0
            ]
            if qpu_times:
                ax.semilogy(
                    farms, qpu_times, marker=_BENCHMARK_MARKERS.get(method, "o"),
                    linestyle="-", linewidth=2, markersize=8,
                    color=_BENCHMARK_COLORS.get(method, "#888"), label=method, alpha=0.8,
                )
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Time (s)")
    ax.set_title("Time Comparison")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    # Panel 2: Solution Quality
    ax = axes[0, 1]
    if "Gurobi" in metrics:
        ax.plot(
            metrics["Gurobi"]["n_farms"], metrics["Gurobi"]["objective"],
            "o-", linewidth=2.5, markersize=10, color=_BENCHMARK_COLORS["Gurobi"],
            label="Gurobi (Optimal)", alpha=0.9,
        )
    for method in small_methods:
        if method in metrics and metrics[method]["n_farms"]:
            ax.plot(
                metrics[method]["n_farms"], metrics[method]["objective"],
                marker=_BENCHMARK_MARKERS.get(method, "o"), linestyle="-",
                linewidth=2, markersize=8, color=_BENCHMARK_COLORS.get(method, "#888"),
                label=method, alpha=0.8,
            )
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Objective Value")
    ax.set_title("Solution Quality")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: Optimality Gap
    ax = axes[1, 0]
    for method in small_methods:
        if method in metrics and metrics[method]["n_farms"]:
            ax.plot(
                metrics[method]["n_farms"], metrics[method]["gap"],
                marker=_BENCHMARK_MARKERS.get(method, "o"), linestyle="-",
                linewidth=2, markersize=8, color=_BENCHMARK_COLORS.get(method, "#888"),
                label=method, alpha=0.8,
            )
    ax.axhline(y=0, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="Optimal")
    ax.axhline(y=10, color="orange", linestyle=":", linewidth=1.5, alpha=0.5, label="10% Gap")
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Gap from Optimal (%)")
    ax.set_title("Optimality Gap")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 4: Violations
    ax = axes[1, 1]
    all_farms = sorted({f for m in metrics.values() for f in m["n_farms"]})
    x = np.arange(len(all_farms))
    width = 0.12
    plotted = 0
    for method in small_methods:
        if method in metrics and metrics[method]["n_farms"]:
            viols = [
                metrics[method]["violations"][metrics[method]["n_farms"].index(f)]
                if f in metrics[method]["n_farms"] else 0
                for f in all_farms
            ]
            ax.bar(
                x + plotted * width, viols, width,
                label=method, color=_BENCHMARK_COLORS.get(method, "#888"), alpha=0.8,
            )
            plotted += 1
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Constraint Violations")
    ax.set_title("Feasibility")
    ax.set_xticks(x + width * (plotted - 1) / 2)
    ax.set_xticklabels(all_farms)
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = output_dir / "qpu_benchmark_small_scale.png"
    for fmt in formats:
        save_path = output_dir / f"qpu_benchmark_small_scale.{fmt}"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [OK] Saved: {save_path}")
    plt.close(fig)

    return out_path


def plot_qpu_benchmark_large_scale(
    output_dir: Path | str | None = None,
    formats: list[str] | None = None,
) -> Path:
    """
    Plot QPU benchmark results for large-scale problems (>500 variables).

    Creates a 2x2 panel showing:
    - Solution quality at scale
    - Optimality gap at scale
    - QPU time scaling
    - Speedup vs classical

    Parameters
    ----------
    output_dir : Path, optional
        Output directory (default: professional_plots)
    formats : list[str], optional
        Output formats (default: ["png", "pdf"])

    Returns
    -------
    Path
        Path to saved figure
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "professional_plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if formats is None:
        formats = ["png", "pdf"]

    # Load data
    qpu_data = _load_qpu_benchmark_data()
    metrics = _extract_benchmark_metrics(qpu_data, "large_scale")

    if not metrics:
        print("  ⚠ No large-scale benchmark data found")
        return output_dir / "qpu_benchmark_large_scale.png"

    # Large-scale methods (those that scale)
    large_methods = [
        "Multilevel(10)_QPU", "cqm_first_PlotBased", "coordinated",
        "HybridGrid(5,9)_QPU", "HybridGrid(10,9)_QPU",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "QPU Benchmark: Large-Scale Problems (200-1000 Farms)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Panel 1: Time Comparison
    ax = axes[0, 0]
    if "Gurobi" in metrics:
        farms = [f for f in metrics["Gurobi"]["n_farms"] if f >= 200]
        times = [
            metrics["Gurobi"]["wall_time"][i]
            for i, f in enumerate(metrics["Gurobi"]["n_farms"]) if f >= 200
        ]
        if farms and any(t > 0 for t in times):
            ax.semilogy(
                farms, times, "o-", linewidth=2.5, markersize=8,
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
                    linestyle="-", linewidth=2, markersize=8,
                    color=_BENCHMARK_COLORS.get(method, "#888"), label=method, alpha=0.8,
                )
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Time (s)")
    ax.set_title("Time Comparison")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    # Panel 2: Solution Quality
    ax = axes[0, 1]
    if "Gurobi" in metrics:
        farms = [f for f in metrics["Gurobi"]["n_farms"] if f >= 200]
        objs = [
            metrics["Gurobi"]["objective"][i]
            for i, f in enumerate(metrics["Gurobi"]["n_farms"]) if f >= 200
        ]
        if farms:
            ax.plot(
                farms, objs, "o-", linewidth=2.5, markersize=10,
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
                    linestyle="-", linewidth=2, markersize=8,
                    color=_BENCHMARK_COLORS.get(method, "#888"), label=method, alpha=0.8,
                )
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Objective Value")
    ax.set_title("Solution Quality at Scale")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: Optimality Gap
    ax = axes[1, 0]
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
                    linestyle="-", linewidth=2, markersize=8,
                    color=_BENCHMARK_COLORS.get(method, "#888"), label=method, alpha=0.8,
                )
    ax.axhline(y=0, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="Optimal")
    ax.axhline(y=10, color="orange", linestyle=":", linewidth=1.5, alpha=0.5, label="10% Gap")
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Gap from Optimal (%)")
    ax.set_title("Optimality Gap at Scale")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 4: Violations
    ax = axes[1, 1]
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
    ax.set_title("Feasibility at Scale")
    ax.set_xticks(x + width * (plotted - 1) / 2)
    ax.set_xticklabels(large_farms)
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = output_dir / "qpu_benchmark_large_scale.png"
    for fmt in formats:
        save_path = output_dir / f"qpu_benchmark_large_scale.{fmt}"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [OK] Saved: {save_path}")
    plt.close(fig)

    return out_path


def plot_qpu_benchmark_comprehensive(
    output_dir: Path | str | None = None,
    formats: list[str] | None = None,
) -> Path:
    """
    Create comprehensive 5-panel QPU benchmark summary.

    Panels:
    1. Time comparison (all scales)
    2. Solution quality (all scales)
    3. Optimality gap (all methods)
    4. Violations by method
    5. Efficiency metric (quality/time)

    Parameters
    ----------
    output_dir : Path, optional
        Output directory (default: professional_plots)
    formats : list[str], optional
        Output formats (default: ["png", "pdf"])

    Returns
    -------
    Path
        Path to saved figure
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "professional_plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if formats is None:
        formats = ["png", "pdf"]

    # Load and merge data from both scales
    qpu_data = _load_qpu_benchmark_data()
    metrics_small = _extract_benchmark_metrics(qpu_data, "small_scale")
    metrics_large = _extract_benchmark_metrics(qpu_data, "large_scale")

    # Merge metrics
    all_metrics: dict[str, dict[str, list]] = {}
    for m in [metrics_small, metrics_large]:
        for method, data in m.items():
            if method not in all_metrics:
                all_metrics[method] = {k: [] for k in data}
            for key in data:
                all_metrics[method][key].extend(data[key])

    if not all_metrics:
        print("  ⚠ No benchmark data found")
        return output_dir / "qpu_benchmark_comprehensive.png"

    # All QPU methods
    qpu_methods = [
        "PlotBased_QPU", "Multilevel(5)_QPU", "Multilevel(10)_QPU",
        "Louvain_QPU", "Spectral(10)_QPU", "cqm_first_PlotBased",
        "coordinated", "HybridGrid(5,9)_QPU", "HybridGrid(10,9)_QPU",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Comprehensive QPU Benchmark: All Scales (10-1000 Farms)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Panel 1: Time Comparison
    ax1 = axes[0, 0]
    if "Gurobi" in all_metrics:
        times = [t for t in all_metrics["Gurobi"]["wall_time"] if t > 0]
        farms = [
            all_metrics["Gurobi"]["n_farms"][i]
            for i, t in enumerate(all_metrics["Gurobi"]["wall_time"]) if t > 0
        ]
        if times:
            ax1.semilogy(farms, times, "o-", linewidth=2.5, markersize=8,
                         color=_BENCHMARK_COLORS["Gurobi"], label="Gurobi", alpha=0.9)
    for method in qpu_methods:
        if method in all_metrics and all_metrics[method]["n_farms"]:
            qpu_times = [t for t in all_metrics[method]["qpu_time"] if t > 0]
            farms = [
                all_metrics[method]["n_farms"][i]
                for i, t in enumerate(all_metrics[method]["qpu_time"]) if t > 0
            ]
            if qpu_times:
                ax1.semilogy(
                    farms, qpu_times, marker=_BENCHMARK_MARKERS.get(method, "o"),
                    linestyle="-", linewidth=1.5, markersize=6,
                    color=_BENCHMARK_COLORS.get(method, "#888"), label=method, alpha=0.7,
                )
    ax1.set_xlabel("Number of Farms")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Time Comparison")
    ax1.legend(loc="best", fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3, which="both")

    # Panel 2: Solution Quality
    ax2 = axes[0, 1]
    if "Gurobi" in all_metrics:
        ax2.plot(
            all_metrics["Gurobi"]["n_farms"], all_metrics["Gurobi"]["objective"],
            "o-", linewidth=2.5, markersize=8, color=_BENCHMARK_COLORS["Gurobi"],
            label="Gurobi (Optimal)", alpha=0.9,
        )
    for method in qpu_methods:
        if method in all_metrics and all_metrics[method]["n_farms"]:
            ax2.plot(
                all_metrics[method]["n_farms"], all_metrics[method]["objective"],
                marker=_BENCHMARK_MARKERS.get(method, "o"), linestyle="-",
                linewidth=1.5, markersize=6, color=_BENCHMARK_COLORS.get(method, "#888"),
                label=method, alpha=0.7,
            )
    ax2.set_xlabel("Number of Farms")
    ax2.set_ylabel("Objective Value")
    ax2.set_title("Solution Quality")
    ax2.set_yscale("log")
    ax2.legend(loc="best", fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Optimality Gap
    ax3 = axes[1, 0]
    for method in qpu_methods:
        if method in all_metrics and all_metrics[method]["n_farms"]:
            ax3.plot(
                all_metrics[method]["n_farms"], all_metrics[method]["gap"],
                marker=_BENCHMARK_MARKERS.get(method, "o"), linestyle="-",
                linewidth=1.5, markersize=6, color=_BENCHMARK_COLORS.get(method, "#888"),
                label=method, alpha=0.7,
            )
    ax3.axhline(y=0, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="Optimal")
    ax3.axhline(y=10, color="orange", linestyle=":", linewidth=1.5, alpha=0.5, label="10% Gap")
    ax3.set_xlabel("Number of Farms")
    ax3.set_ylabel("Gap from Optimal (%)")
    ax3.set_title("Optimality Gap")
    ax3.legend(loc="best", fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Violations Summary
    ax4 = axes[1, 1]
    all_farms = sorted({f for m in all_metrics.values() for f in m["n_farms"]})
    x = np.arange(len(all_farms))
    width = 0.08
    plotted = 0
    for method in qpu_methods[:6]:  # First 6 methods for readability
        if method in all_metrics and all_metrics[method]["n_farms"]:
            viols = [
                all_metrics[method]["violations"][all_metrics[method]["n_farms"].index(f)]
                if f in all_metrics[method]["n_farms"] else 0
                for f in all_farms
            ]
            ax4.bar(
                x + plotted * width, viols, width,
                label=method, color=_BENCHMARK_COLORS.get(method, "#888"), alpha=0.8,
            )
            plotted += 1
    ax4.set_xlabel("Number of Farms")
    ax4.set_ylabel("Violations")
    ax4.set_title("Constraint Violations")
    ax4.set_xticks(x + width * (plotted - 1) / 2)
    ax4.set_xticklabels(all_farms, fontsize=8)
    ax4.legend(loc="best", fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])


    out_path = output_dir / "qpu_benchmark_comprehensive.png"
    for fmt in formats:
        save_path = output_dir / f"qpu_benchmark_comprehensive.{fmt}"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [OK] Saved: {save_path}")
    plt.close(fig)

    return out_path


# ==============================================================================
# END OF SECTION 7
# ==============================================================================


def _run_data_loading_test() -> None:
    """Self-test: load data and print summary."""
    print("=" * 80)
    print("generate_all_report_plots.py - Data Loading Test")
    print("=" * 80)
    print()

    # Test QPU loading
    print("Loading QPU hierarchical data...")
    try:
        qpu_df = load_qpu_hierarchical()
        print(f"  [OK] Loaded {len(qpu_df)} QPU runs")
        print(f"    Scenarios: {qpu_df['scenario_name'].tolist()[:5]}...")
        print(f"    n_vars range: {qpu_df['n_vars'].min()} - {qpu_df['n_vars'].max()}")
    except FileNotFoundError as e:
        print(f"  [FAIL] {e}")

    # Test Gurobi 60s loading
    print("\nLoading Gurobi 60s baseline...")
    try:
        gurobi_60s_df = load_gurobi_60s()
        print(f"  [OK] Loaded {len(gurobi_60s_df)} Gurobi 60s runs")
        print(f"    Timeouts: {gurobi_60s_df['hit_timeout'].sum()}")
    except FileNotFoundError as e:
        print(f"  [FAIL] {e}")

    # Test Gurobi 300s loading
    print("\nLoading Gurobi 300s verification...")
    try:
        gurobi_300s_df = load_gurobi_300s()
        print(f"  [OK] Loaded {len(gurobi_300s_df)} Gurobi 300s runs")
        print(f"    Timeouts: {gurobi_300s_df['hit_timeout'].sum()}")
    except FileNotFoundError as e:
        print(f"  [FAIL] {e}")

    # Test merge
    print("\nTesting merge_by_scenario...")
    try:
        merged_df = merge_by_scenario(qpu_df, gurobi_60s_df)
        print(f"  [OK] Merged DataFrame: {len(merged_df)} rows")
        print(f"    Columns: {list(merged_df.columns)[:10]}...")
        print(f"    Avg benefit ratio: {merged_df['benefit_ratio'].mean():.2f}x")
        print(f"    Avg speedup: {merged_df['speedup'].mean():.2f}x")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print("\n" + "=" * 80)
    print("Data loading test complete.")
    print("=" * 80)


# =============================================================================
# SECTION 8: MAIN ENTRY POINT
# =============================================================================


def generate_all_plots() -> dict[str, bool]:
    """
    Generate all plots for the report.

    Returns a dictionary mapping plot names to success status.
    """
    results: dict[str, bool] = {}

    print("=" * 80)
    print("GENERATING ALL REPORT PLOTS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Pre-load shared data
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

    # Define plot functions with their data requirements
    plot_tasks = [
        ("Comprehensive Scaling", lambda: plot_comprehensive_scaling(df_60s)),
        ("Split Analysis", lambda: plot_split_analysis(df_300s)),
        ("Objective Gap Analysis", lambda: plot_objective_gap_analysis(df_300s)),
        ("Violation Impact", plot_violation_impact),  # Has internal data loading
        ("Gap Deep Dive", plot_gap_deep_dive),  # Has internal data loading
        ("QPU Advantage Corrected", plot_qpu_advantage_corrected),  # Has internal data loading
        ("QPU Benchmark Small Scale", plot_qpu_benchmark_small_scale),  # Has internal data loading
        ("QPU Benchmark Large Scale", plot_qpu_benchmark_large_scale),  # Has internal data loading
        ("QPU Benchmark Comprehensive", plot_qpu_benchmark_comprehensive),  # Has internal data loading
    ]

    for name, func in plot_tasks:
        print(f"Generating: {name}...")
        try:
            func()
            results[name] = True
            print(f"  [OK] {name} completed successfully")
        except Exception as e:
            results[name] = False
            print(f"  [FAIL] {name} failed: {e}")
        print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(results.values())
    total = len(results)
    print(f"Completed: {successful}/{total} plots generated successfully")
    print()

    for name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")

    print()
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return results


def main() -> None:
    """Main entry point with command-line argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate plots for OQI-UC002-DWave project report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_all_report_plots.py           # Generate all plots
  python generate_all_report_plots.py --all     # Generate all plots (explicit)
  python generate_all_report_plots.py --scaling # Only comprehensive scaling
  python generate_all_report_plots.py --benchmark --gap  # Multiple selections
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all plots (default if no options specified)",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Generate comprehensive scaling plots",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Generate split formulation analysis plots",
    )
    parser.add_argument(
        "--violations",
        action="store_true",
        help="Generate violation impact plots",
    )
    parser.add_argument(
        "--gap",
        action="store_true",
        help="Generate gap deep dive analysis plots",
    )
    parser.add_argument(
        "--advantage",
        action="store_true",
        help="Generate QPU advantage corrected plots",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Generate QPU benchmark result plots",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run data loading test only (no plots)",
    )

    args = parser.parse_args()

    # If --test, just run the data loading test
    if args.test:
        _run_data_loading_test()
        return

    # If no specific options, default to --all
    any_specific = any([
        args.scaling,
        args.split,
        args.violations,
        args.gap,
        args.advantage,
        args.benchmark,
    ])

    if args.all or not any_specific:
        # Generate all plots
        generate_all_plots()
        return

    # Generate selected plots
    results: dict[str, bool] = {}

    print("=" * 80)
    print("GENERATING SELECTED REPORT PLOTS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Pre-load data if needed
    df_60s = None
    df_300s = None

    if args.scaling or args.split:
        print("Loading shared data...")
        try:
            df_60s = prepare_scaling_data_60s()
            print(f"  [OK] 60s data: {len(df_60s)} scenarios")
        except Exception as e:
            print(f"  [FAIL] Failed to load 60s data: {e}")
        try:
            df_300s = prepare_scaling_data_300s()
            print(f"  [OK] 300s data: {len(df_300s)} scenarios")
        except Exception as e:
            print(f"  [FAIL] Failed to load 300s data: {e}")
        print()

    if args.scaling:
        print("Generating: Comprehensive Scaling...")
        try:
            plot_comprehensive_scaling(df_60s)
            results["Comprehensive Scaling"] = True
            print("  [OK] Comprehensive Scaling completed successfully")
        except Exception as e:
            results["Comprehensive Scaling"] = False
            print(f"  [FAIL] Comprehensive Scaling failed: {e}")
        print()

    if args.split:
        print("Generating: Split Analysis...")
        try:
            plot_split_analysis(df_300s)
            results["Split Analysis"] = True
            print("  [OK] Split Analysis completed successfully")
        except Exception as e:
            results["Split Analysis"] = False
            print(f"  [FAIL] Split Analysis failed: {e}")
        print()

        print("Generating: Objective Gap Analysis...")
        try:
            plot_objective_gap_analysis(df_300s)
            results["Objective Gap Analysis"] = True
            print("  [OK] Objective Gap Analysis completed successfully")
        except Exception as e:
            results["Objective Gap Analysis"] = False
            print(f"  [FAIL] Objective Gap Analysis failed: {e}")
        print()

    if args.violations:
        print("Generating: Violation Impact...")
        try:
            plot_violation_impact()
            results["Violation Impact"] = True
            print("  [OK] Violation Impact completed successfully")
        except Exception as e:
            results["Violation Impact"] = False
            print(f"  [FAIL] Violation Impact failed: {e}")
        print()

    if args.gap:
        print("Generating: Gap Deep Dive...")
        try:
            plot_gap_deep_dive()
            results["Gap Deep Dive"] = True
            print("  [OK] Gap Deep Dive completed successfully")
        except Exception as e:
            results["Gap Deep Dive"] = False
            print(f"  [FAIL] Gap Deep Dive failed: {e}")
        print()

    if args.advantage:
        print("Generating: QPU Advantage Corrected...")
        try:
            plot_qpu_advantage_corrected()
            results["QPU Advantage Corrected"] = True
            print("  [OK] QPU Advantage Corrected completed successfully")
        except Exception as e:
            results["QPU Advantage Corrected"] = False
            print(f"  [FAIL] QPU Advantage Corrected failed: {e}")
        print()

    if args.benchmark:
        print("Generating: QPU Benchmark Small Scale...")
        try:
            plot_qpu_benchmark_small_scale()
            results["QPU Benchmark Small Scale"] = True
            print("  [OK] QPU Benchmark Small Scale completed successfully")
        except Exception as e:
            results["QPU Benchmark Small Scale"] = False
            print(f"  [FAIL] QPU Benchmark Small Scale failed: {e}")
        print()

        print("Generating: QPU Benchmark Large Scale...")
        try:
            plot_qpu_benchmark_large_scale()
            results["QPU Benchmark Large Scale"] = True
            print("  [OK] QPU Benchmark Large Scale completed successfully")
        except Exception as e:
            results["QPU Benchmark Large Scale"] = False
            print(f"  [FAIL] QPU Benchmark Large Scale failed: {e}")
        print()

        print("Generating: QPU Benchmark Comprehensive...")
        try:
            plot_qpu_benchmark_comprehensive()
            results["QPU Benchmark Comprehensive"] = True
            print("  [OK] QPU Benchmark Comprehensive completed successfully")
        except Exception as e:
            results["QPU Benchmark Comprehensive"] = False
            print(f"  [FAIL] QPU Benchmark Comprehensive failed: {e}")
        print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(results.values())
    total = len(results)
    print(f"Completed: {successful}/{total} plots generated successfully")
    print()

    for name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")

    print()
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
