"""
generate_report_visuals.py
Generate report figures and LaTeX tables for the agricultural optimization study.

Outputs:
  @todo/report/images/Plots/study1_hybrid_performance.pdf
  @todo/report/images/Plots/qpu_benchmark_small_scale.pdf
  @todo/report/images/Plots/qpu_benchmark_large_scale.pdf
  @todo/report/images/Plots/qpu_benchmark_comprehensive.pdf
  Benchmarks/decomposition_scaling/tables/report_tables.tex
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
QPU_DIR = HERE / "@todo" / "qpu_benchmark_results"
PLOT_DIR = HERE / "@todo" / "report" / "images" / "Plots"
TABLE_DIR = HERE / "Benchmarks" / "decomposition_scaling" / "tables"

# ── Study 2 source files and their expected scale coverage ────────────────────
QPU_FILES: list[tuple[str, list[int]]] = [
    ("qpu_benchmark_20251201_160444.json", [10, 15, 50, 100]),
    ("qpu_benchmark_20251201_142434.json", [25]),
    ("qpu_benchmark_20251201_200012.json", [200, 500, 1000]),
    ("qpu_benchmark_20251203_121526.json", [10, 15, 50, 200, 500, 1000]),  # HybridGrid
    ("qpu_benchmark_20251203_110358.json", [25]),  # HybridGrid @ 25
    ("qpu_benchmark_20251203_111656.json", [100]),  # HybridGrid @ 100
]

# ── 8 target decomposition method keys ────────────────────────────────────────
ALL_METHODS: list[str] = [
    "decomposition_PlotBased_QPU",
    "decomposition_Multilevel(5)_QPU",
    "decomposition_Multilevel(10)_QPU",
    "decomposition_Louvain_QPU",
    "decomposition_Spectral(10)_QPU",
    "cqm_first_PlotBased",
    "coordinated",
    "decomposition_HybridGrid(5,9)_QPU",
]

# ── Methods with data across all scales 10–1000 ───────────────────────────────
# PlotBased, Multilevel(5), Louvain, Spectral(10) only cover 10–100.
# These four span the full [10, 15, 25, 50, 100, 200, 500, 1000] range.
FULL_SPAN_METHODS: list[str] = [
    "decomposition_Multilevel(10)_QPU",
    "cqm_first_PlotBased",
    "coordinated",
    "decomposition_HybridGrid(5,9)_QPU",
]

# ── Gurobi-decomposed methods shown in all 4 panels for full-span results ─────
# Keys match GUROBI_DECOMP_COLORS and the solver_comparison_results decomposition field.
FULL_SPAN_GUROBI_DECOMP_METHODS: list[str] = [
    "HybridGrid(5,9)",
    "Multilevel(10)",
    "Multilevel(5)",
    "PlotBased",
]

METHOD_DISPLAY: dict[str, str] = {
    "decomposition_PlotBased_QPU": "PlotBased",
    "decomposition_Multilevel(5)_QPU": "Multilevel(5)",
    "decomposition_Multilevel(10)_QPU": "Multilevel(10)",
    "decomposition_Louvain_QPU": "Louvain",
    "decomposition_Spectral(10)_QPU": "Spectral(10)",
    "cqm_first_PlotBased": "CQM-First",
    "coordinated": "Coordinated",
    "decomposition_HybridGrid(5,9)_QPU": "HybridGrid(5,9)",
}

METHOD_COLORS: dict[str, str] = {
    "ground_truth": "#2ca02c",
    "PlotBased": "#9467bd",
    "Multilevel(5)": "#e377c2",
    "Multilevel(10)": "#8c564b",
    "Louvain": "#ff7f0e",
    "Spectral(10)": "#d62728",
    "CQM-First": "#17becf",
    "Coordinated": "#7f7f7f",
    "HybridGrid(5,9)": "#1f77b4",
}

METHOD_MARKERS: dict[str, str] = {
    "ground_truth": "D",
    "PlotBased": "o",
    "Multilevel(5)": "s",
    "Multilevel(10)": "^",
    "Louvain": "v",
    "Spectral(10)": "<",
    "CQM-First": ">",
    "Coordinated": "X",
    "HybridGrid(5,9)": "P",
}

# ── Matplotlib style ───────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.minor.size": 3.0,
    "ytick.minor.size": 3.0,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "grid.color": "#757575FF",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.55,
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.titlesize": 15,
    "legend.fontsize": 9,
})


# ── Generic helpers ────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def safe_float(val: Any) -> float | None:
    """Return float(val) or None if val is None or un-convertable."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def get_violations(validation: dict | None) -> int:
    if not validation:
        return 0
    return int(validation.get("n_violations", 0))


# ── Study 1 data loader ────────────────────────────────────────────────────────

def load_study1() -> list[dict[str, Any]]:
    """Load Study 1 patch-benchmark entries sorted ascending by n_units."""
    path = DATA_DIR / "comprehensive_benchmark_configs_dwave_20251130_212742.json"
    print(f"  Loading Study 1: {path.name}")
    data = load_json(path)
    return sorted(data["patch_results"], key=lambda e: e["n_units"])


# ── Study 2 data loader ────────────────────────────────────────────────────────

def load_study2() -> tuple[dict[tuple[int, str], dict], dict[int, dict]]:
    """
    Merge QPU benchmark data from all configured files.

    Returns:
        method_data:       {(n_farms, method_key): result_dict}
        ground_truth_data: {n_farms: ground_truth_dict}
    """
    method_data: dict[tuple[int, str], dict] = {}
    ground_truth_data: dict[int, dict] = {}

    for fname, _ in QPU_FILES:
        fpath = QPU_DIR / fname
        if not fpath.exists():
            print(f"  WARNING: {fname} not found — skipping")
            continue
        print(f"  Loading {fname}")
        data = load_json(fpath)
        for result in data.get("results", []):
            n: int = result["n_farms"]
            if n not in ground_truth_data and "ground_truth" in result:
                ground_truth_data[n] = result["ground_truth"]
            for mkey, mdata in result.get("method_results", {}).items():
                if mkey in ALL_METHODS:
                    method_data[(n, mkey)] = mdata  # last file wins per (n, method)

    return method_data, ground_truth_data


# ── Gurobi decomposed line colours (match plot_benchmark_results.py palette) ───────
GUROBI_DECOMP_COLORS: dict[str, str] = {
    "PlotBased":      "#377eb8",
    "Multilevel(5)":  "#4daf4a",
    "Multilevel(10)": "#a65628",
    "HybridGrid(5,9)": "#984ea3",
}

# ── Gurobi full baseline loader (Study 2 classical reference) ───────────────

def _load_solver_comparison() -> list[dict]:
    """Load solver comparison JSON (hard preferred)."""
    decomp_dir = HERE / "Benchmarks" / "decomposition_scaling"
    for fname in (
        "solver_comparison_results_hard.json",
        "solver_comparison_results_soft.json",
        "solver_comparison_results.json",
    ):
        path = decomp_dir / fname
        if path.exists():
            print(f"  Loading solver comparison from {fname}")
            return load_json(path)
    print("  WARNING: solver_comparison_results not found")
    return []


def load_gurobi_full_a(data: list[dict] | None = None) -> dict[int, dict]:
    """
    Return {n_farms: {"wall_time": float, "objective": float}} for Gurobi_full
    Variant-A entries (same source used by plot_benchmark_results.py).
    Pass pre-loaded data to avoid re-reading the file.
    """
    if data is None:
        data = _load_solver_comparison()
    result: dict[int, dict] = {}
    for r in data:
        if r.get("variant") != "A" or r.get("decomposition", "none") != "none":
            continue
        n = r["n_farms"]
        wt = safe_float(r.get("wall_time"))
        obj = safe_float(r.get("objective"))
        if wt is not None or obj is not None:
            result[n] = {"wall_time": wt, "objective": obj}
    print(f"  Gurobi full A scales: {sorted(result)}")
    return result


def load_gurobi_decompositions_a(data: list[dict] | None = None) -> dict[str, dict[int, dict]]:
    """
    Return {decomp_name: {n_farms: {"wall_time": float, "objective": float, "healed_objective": float|None}}}
    for all Gurobi_decomposed Variant-A entries.
    """
    if data is None:
        data = _load_solver_comparison()
    result: dict[str, dict[int, dict]] = {}
    for r in data:
        if r.get("variant") != "A" or r.get("decomposition", "none") == "none":
            continue
        dname = r["decomposition"]
        n = r["n_farms"]
        wt = safe_float(r.get("wall_time"))
        obj = safe_float(r.get("objective"))
        healed = safe_float(r.get("healed_objective"))
        result.setdefault(dname, {})[n] = {"wall_time": wt, "objective": obj, "healed_objective": healed}
    print(f"  Gurobi decomposed A methods: {sorted(result)}")
    return result


# ── Benefit matrix ─────────────────────────────────────────────────────────────

def load_benefit_matrix() -> dict[str, float]:
    """
    Return {crop_name: benefit_coefficient} from rotation_crop_matrix.csv.

    The CSV rows each have a constant value across all columns (the per-crop
    benefit coefficient b_c used in the objective: max Z = (1/T) Σ b_c A_{f,c}).
    We extract it as the first non-index column value of each row.
    """
    csv_path = HERE / "rotation_data" / "rotation_crop_matrix.csv"
    if not csv_path.exists():
        print("  WARNING: rotation_crop_matrix.csv not found — using hardcoded benefits")
        return _hardcoded_benefits()
    df = pd.read_csv(csv_path, index_col=0)
    # Each row has a constant benefit value; take the mean of the row
    benefits = df.mean(axis=1).to_dict()
    return benefits


def _hardcoded_benefits() -> dict[str, float]:
    """Fallback hardcoded benefit values from the Indonesia rotation scenario."""
    fruits = ["Mango", "Papaya", "Orange", "Banana", "Guava",
              "Watermelon", "Apple", "Avocado", "Durian"]
    plant_proteins = ["Tofu", "Tempeh", "Peanuts", "Chickpeas"]
    vegetables = ["Pumpkin", "Spinach", "Tomatoes", "Long bean",
                  "Cabbage", "Eggplant", "Cucumber"]
    starchy = ["Corn", "Potato"]
    animal = ["Egg", "Beef", "Lamb", "Pork", "Chicken"]
    b: dict[str, float] = {}
    for c in fruits:
        b[c] = 0.039
    for c in plant_proteins:
        b[c] = 0.192
    for c in vegetables:
        b[c] = 0.118
    for c in starchy:
        b[c] = 0.000
    for c in animal:
        b[c] = -0.073
    return b


# ── Study 1 healed objective computation ───────────────────────────────────────

def compute_healed_obj_study1(
    entries: list[dict],
    benefit_matrix: dict[str, float],
) -> dict[str, dict[int, float]]:
    """
    Compute healed objectives for Study 1 CQM and BQM solvers.

    CQM/BQM solutions massively over-assign crops: at n=1000, CQM assigns
    ~17,000 crops to 1,000 patches instead of 1 per patch.  The reported
    ``validation.n_violations`` only catches a small subset.

    Correct healing: recompute the objective from solution_plantations,
    keeping **only the highest-benefit crop per patch** (using actual patch
    areas from ``land_data``).

    Objective formula: Z = (1/T) * sum_f [ b_{best_f} * L_f ]
    where T = total area, L_f = area of patch f, b_{best_f} = benefit of
    the single best crop assigned to patch f.

    Returns {solver_label: {n_units: healed_objective}}.
    """
    import re

    result: dict[str, dict[int, float]] = {"cqm": {}, "bqm": {}}

    for e in entries:
        n: int = e["n_units"]
        s = e["solvers"]

        for sk, label in (("dwave_cqm", "cqm"), ("dwave_bqm", "bqm")):
            sol = s.get(sk) or {}
            obj = safe_float(sol.get("objective_value"))
            if obj is None:
                continue

            sp: dict[str, float] = sol.get("solution_plantations") or {}
            if not sp:
                result[label][n] = obj
                continue

            total_area = safe_float(sol.get("total_area")) or (n * 10.0)
            land_data: dict[str, float] = sol.get("land_data") or {}

            # Parse Y_{patch}_{crop} -> per-patch crop list
            patch_crops: dict[str, list[str]] = {}
            for key, val in sp.items():
                if float(val or 0) < 0.5:
                    continue
                m = re.match(r"Y_([^_]+)_(.+)", key)
                if m:
                    patch_crops.setdefault(m.group(1), []).append(m.group(2))

            # Recompute objective keeping only the best crop per patch
            healed = 0.0
            for patch, crops in patch_crops.items():
                patch_area = float(land_data.get(patch, total_area / n))
                best_benefit = max(benefit_matrix.get(c, 0.0) for c in crops)
                healed += best_benefit * patch_area / total_area

            result[label][n] = healed

    return result


# ── Analytical healed objective for QPU methods ────────────────────────────────

def compute_healed_obj_qpu(
    method_data: dict[tuple[int, str], dict],
    benefit_matrix: dict[str, float],
) -> dict[tuple[int, str], float]:
    """
    Compute analytical healed_obj for each QPU (n, method) pair with violations.

    healed_obj = recomputed from Y matrix keeping only the best crop per farm.

    QPU solutions use solution.Y = {farm: {crop: 0/1}}.  Violations are small
    (typically 1–6 farms with 2 crops vs the massive over-assignment seen in
    Study 1).  We recompute uniformly: keep best crop per farm, sum
    b_best * patch_area / total_area with uniform areas (total_area / n).
    total_area comes from entry.metadata.total_area (100.0 for all QPU runs).
    """
    healed: dict[tuple[int, str], float] = {}

    for (n, mkey), entry in method_data.items():
        obj = safe_float(entry.get("objective"))
        if obj is None:
            continue

        sol = entry.get("solution") or {}
        y_dict: dict[str, dict] = sol.get("Y") or {}

        if not y_dict:
            healed[(n, mkey)] = obj
            continue

        total_area = safe_float(
            (entry.get("metadata") or {}).get("total_area")
        )
        if total_area is None or total_area <= 0.0:
            total_area = n * 10.0
        patch_area = total_area / n

        # Recompute: keep best crop per farm
        healed_val = 0.0
        for farm, crop_vals in y_dict.items():
            assigned = [c for c, v in crop_vals.items() if int(v or 0) == 1]
            if not assigned:
                continue
            best_b = max(benefit_matrix.get(c, 0.0) for c in assigned)
            healed_val += best_b * patch_area / total_area

        healed[(n, mkey)] = healed_val

    return healed


# ── Gurobi decomposed violations ──────────────────────────────────────────────

def load_gurobi_decomp_violations(data: list[dict] | None = None) -> dict[str, dict[int, int]]:
    """
    Return {decomp_name: {n_farms: violation_count}} for Gurobi-decomposed Variant-A.
    Only covers PlotBased, Multilevel(5), Multilevel(10), HybridGrid(5,9).
    """
    if data is None:
        data = _load_solver_comparison()
    result: dict[str, dict[int, int]] = {}
    for r in data:
        if r.get("variant") != "A" or r.get("decomposition", "none") == "none":
            continue
        dname = r["decomposition"]
        if dname not in GUROBI_DECOMP_COLORS:
            continue
        n = r["n_farms"]
        v = r.get("violations")
        if v is not None:
            result.setdefault(dname, {})[n] = int(v)
    print(f"  Gurobi decomposed violations loaded: {sorted(result)}")
    return result


# ── Figure 1: Study 1 — Hybrid Solver Comparison ──────────────────────────────

def _collect_study1_series(
    entries: list[dict],
) -> dict[str, tuple[list[int], list[float | None]]]:
    """Return {series_name: (n_units_list, values_list)} for all 4 solvers × 2 metrics."""
    n_units: list[int] = []
    g_times: list[float | None] = []
    cqm_times: list[float | None] = []
    gqubo_times: list[float | None] = []
    bqm_times: list[float | None] = []
    g_objs: list[float | None] = []
    cqm_objs: list[float | None] = []
    gqubo_objs: list[float | None] = []
    bqm_objs: list[float | None] = []

    for e in entries:
        n_units.append(e["n_units"])
        s = e["solvers"]

        g = s.get("gurobi", {})
        g_times.append(safe_float(g.get("solve_time")))
        g_objs.append(safe_float(g.get("objective_value")))

        cqm = s.get("dwave_cqm", {})
        cqm_times.append(safe_float(cqm.get("hybrid_time")))
        cqm_objs.append(safe_float(cqm.get("objective_value")))

        gq = s.get("gurobi_qubo", {})
        gq_ok = gq.get("success", False) and gq.get("status", "").lower() != "error"
        gqubo_times.append(safe_float(gq.get("solve_time")) if gq_ok else None)
        gq_obj = safe_float(gq.get("objective_value")) if gq_ok else None
        # Keep obj=0 in the series; log-scale will simply not render 0, but we
        # annotate those points explicitly in the plot so they are visible.
        gqubo_objs.append(gq_obj)

        bqm = s.get("dwave_bqm", {})
        bqm_t = safe_float(bqm.get("hybrid_time") or bqm.get("solve_time"))
        bqm_times.append(bqm_t)
        bqm_objs.append(safe_float(bqm.get("objective_value")))

    return {
        "gurobi_time": (n_units, g_times),
        "cqm_time": (n_units, cqm_times),
        "gqubo_time": (n_units, gqubo_times),
        "bqm_time": (n_units, bqm_times),
        "gurobi_obj": (n_units, g_objs),
        "cqm_obj": (n_units, cqm_objs),
        "gqubo_obj": (n_units, gqubo_objs),
        "bqm_obj": (n_units, bqm_objs),
    }


def _add_line(
    ax: plt.Axes,
    xs: list[int],
    ys: list[float | None],
    label: str,
    color: str,
    marker: str,
    linestyle: str = "-",
) -> None:
    """Plot a series, skipping None values."""
    px = [x for x, y in zip(xs, ys) if y is not None]
    py = [y for y in ys if y is not None]
    if px:
        ax.plot(
            px, py,
            color=color, marker=marker, linestyle=linestyle,
            label=label, linewidth=1.8, markersize=7,
        )


def _print_series_table(name: str, xs: list[int], ys: list[float | None]) -> None:
    """Print a simple table of x and y values for debugging plotted lines."""
    try:
        import pandas as _pd
    except Exception:
        print(f"Series '{name}': (no pandas) {list(zip(xs, ys))}")
        return
    df = _pd.DataFrame({"x": xs, "y": ys})
    print(f"\n--- Series: {name} ---")
    print(df.to_string(index=False))


def plot_study1(
    entries: list[dict],
    gurobi_full_a: dict[int, dict],
    healed_study1: dict[str, dict[int, float]] | None = None,
) -> None:
    """
    Create Figure 1: 1×2 hybrid solver comparison.
    The Gurobi line uses gurobi_full_a (from solver_comparison_results_hard.json)
    so it matches fig_A_solver_time.pdf / fig_A_solver_quality.pdf.
    D-Wave results still come from the Study 1 comprehensive benchmark.
    X-axis clipped at 1000 to match QPU data range.
    If healed_study1 is provided, dotted lines are overlaid for CQM and BQM.
    """
    MAX_N = 1000
    print("\nPlotting Figure 1: Study 1 — Hybrid Solver Comparison")
    series = _collect_study1_series(entries)

    # Build Gurobi series from the solver comparison data (canonical source),
    # clipped to MAX_N so we don't extend beyond the D-Wave data range.
    gt_farms = sorted(n for n in gurobi_full_a if n <= MAX_N)
    gt_times = [gurobi_full_a[n]["wall_time"] for n in gt_farms]
    gt_objs  = [gurobi_full_a[n]["objective"] for n in gt_farms]
    gt_obj_map: dict[int, float] = dict(zip(gt_farms, gt_objs))

    fig, (ax_t, ax_g) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel 1: Runtime ─────────────────────────────────────────────────────
    _print_series_table("Gurobi_time", gt_farms, gt_times)
    for key in ("cqm_time", "gqubo_time", "bqm_time"):
        xs_s, ys_s = series.get(key, ([], []))
        _print_series_table(key, xs_s, ys_s)

    _add_line(ax_t, gt_farms, gt_times, "Gurobi", "#e41a1c", "o")
    _add_line(ax_t, *series["cqm_time"], "D-Wave CQM (Hybrid)", "#ff7f0e", "s")
    _add_line(ax_t, *series["gqubo_time"], "Gurobi QUBO", "#2ca02c", "^", "--")
    _add_line(ax_t, *series["bqm_time"], "D-Wave BQM (Hybrid)", "#d62728", "D")
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_xlabel("Number of Patches (n)")
    ax_t.set_ylabel("Solve Time (s)")
    ax_t.set_title("Solver Runtime Scaling")
    ax_t.legend()
    ax_t.grid(True, which="both", color="lightgray", linewidth=0.5)

    # ── Panel 2: Solution quality (COMMENTED OUT) ────────────────────────────────────
    # _print_series_table("Gurobi_obj", gt_farms, gt_objs)
    # for key in ("cqm_obj", "gqubo_obj", "bqm_obj"):
    #     xs_s, ys_s = series.get(key, ([], []))
    #     _print_series_table(key, xs_s, ys_s)
    #
    # _add_line(ax_q, gt_farms, gt_objs, "Gurobi", "#e41a1c", "o")
    # _add_line(ax_q, *series["cqm_obj"], "D-Wave CQM (Hybrid)*", "#ff7f0e", "s")
    # _add_line(ax_q, *series["bqm_obj"], "D-Wave BQM (Hybrid)*", "#d62728", "D")
    #
    # gqubo_ns, gqubo_os = series["gqubo_obj"]
    # gqubo_pos_ns = [x for x, y in zip(gqubo_ns, gqubo_os) if y is not None and y > 0.0]
    # gqubo_pos_ys = [y for y in gqubo_os if y is not None and y > 0.0]
    # gqubo_zero_ns = [x for x, y in zip(gqubo_ns, gqubo_os) if y is not None and y == 0.0]
    # if gqubo_pos_ns:
    #     _add_line(ax_q, gqubo_pos_ns, gqubo_pos_ys, "Gurobi QUBO", "#2ca02c", "^", "--")
    # for nz in gqubo_zero_ns:
    #     ax_q.axvline(nz, color="#2ca02c", linestyle=":", linewidth=0.8, alpha=0.6)
    #     ax_q.annotate(
    #         f"QUBO=0\n(n={nz})",
    #         xy=(nz, 0.04), xycoords=("data", "axes fraction"),
    #         color="#2ca02c", fontsize=7, ha="center", va="bottom", style="italic",
    #     )
    #
    # if healed_study1:
    #     n_units_list = series["cqm_obj"][0]
    #     for label, raw_series_key, color, marker, display in (
    #         ("cqm", "cqm_obj", "#ff7f0e", "s", "D-Wave CQM (healed)"),
    #         ("bqm", "bqm_obj", "#d62728", "D", "D-Wave BQM (healed)"),
    #     ):
    #         h_map = healed_study1.get(label, {})
    #         raw_ys_map = {n: y for n, y in zip(*series[raw_series_key]) if y is not None}
    #         xs_h = [
    #             n for n in n_units_list
    #             if n in h_map and n in raw_ys_map
    #             and abs(h_map[n] - raw_ys_map[n]) > 1e-9
    #         ]
    #         ys_h = [h_map[n] for n in xs_h]
    #         print(f"\n--- Healed {label} debug ---")
    #         for n in n_units_list:
    #             raw_v = raw_ys_map.get(n)
    #             heal_v = h_map.get(n)
    #             diff = (heal_v - raw_v) if (heal_v is not None and raw_v is not None) else None
    #             print(f"  n={n:5d}  raw={raw_v}  healed={heal_v}  diff={diff}  plotted={'YES' if n in xs_h else 'no'}")
    #         if xs_h:
    #             ax_q.plot(
    #                 xs_h, ys_h,
    #                 color=color, marker=marker, linestyle="--",
    #                 linewidth=2.2, markersize=9,
    #                 markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.8,
    #                 label=display, zorder=5,
    #             )
    # ax_q.annotate(
    #     "* CQM/BQM status = Infeasible at n=10--25",
    #     xy=(0.02, 0.02), xycoords="axes fraction",
    #     fontsize=8, color="#888", style="italic",
    # )
    # ax_q.set_xscale("log")
    # ax_q.set_yscale("log")
    # ax_q.set_xlabel("Number of Patches (n)")
    # ax_q.set_ylabel("Objective Value (Benefit)")
    # ax_q.set_title("Solution Quality (Benefit)")
    # ax_q.legend()
    # ax_q.grid(True, which="both", color="lightgray", linewidth=0.5)

    # ── Panel 3: Absolute gap |solver_obj - gurobi_obj|  (log-log) ────────────
    all_ns = series["cqm_obj"][0]
    gqubo_ns, gqubo_os = series["gqubo_obj"]  # full series including zero-obj points

    def _absgap_series(
        ns: list[int], ys: list[float | None], gt_map: dict[int, float],
    ) -> tuple[list[int], list[float]]:
        xs_g: list[int] = []
        ys_g: list[float] = []
        for n, y in zip(ns, ys):
            if y is None:
                continue
            gt = gt_map.get(n)
            if gt is None:
                continue
            xs_g.append(n)
            ys_g.append(abs(y - gt))
        return xs_g, ys_g

    cqm_gx, cqm_gy = _absgap_series(*series["cqm_obj"], gt_obj_map)
    bqm_gx, bqm_gy = _absgap_series(*series["bqm_obj"], gt_obj_map)
    gqubo_gx, gqubo_gy = _absgap_series(gqubo_ns, gqubo_os, gt_obj_map)

    _add_line(ax_g, cqm_gx, cqm_gy, "D-Wave CQM (Hybrid)*", "#ff7f0e", "s")
    _add_line(ax_g, bqm_gx, bqm_gy, "D-Wave BQM (Hybrid)*", "#d62728", "D")
    if gqubo_gx:
        _add_line(ax_g, gqubo_gx, gqubo_gy, "Gurobi QUBO", "#2ca02c", "^", "--")

    # Healed absolute gaps
    if healed_study1:
        for label, color, marker, display in (
            ("cqm", "#ff7f0e", "s", "D-Wave CQM (healed)"),
            ("bqm", "#d62728", "D", "D-Wave BQM (healed)"),
        ):
            h_map = healed_study1.get(label, {})
            xs_h2 = [n for n in all_ns if n in h_map and n in gt_obj_map]
            ys_h2 = [abs(h_map[n] - gt_obj_map[n]) for n in xs_h2]
            if xs_h2:
                ax_g.plot(
                    xs_h2, ys_h2,
                    color=color, marker=marker, linestyle="--",
                    linewidth=2.2, markersize=9,
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.8,
                    label=display, zorder=5,
                )

    ax_g.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6,
                  zorder=1, label="Gurobi optimum (gap=0)")
    ax_g.set_xscale("log")
    ax_g.set_yscale("symlog", linthresh=1e-4)
    ax_g.set_ylim(0, 100)
    ax_g.set_xlabel("Number of Patches (n)")
    ax_g.set_ylabel(r"$|$Objective $-$ Gurobi$|$")
    ax_g.set_title("Absolute Solution Gap vs Gurobi")
    ax_g.legend(fontsize=8)
    ax_g.grid(True, which="both", color="lightgray", linewidth=0.5)

    fig.tight_layout()
    out = PLOT_DIR / "study1_hybrid_performance.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: Study 2 — QPU Decomposition ────────────────────────────────────

def _method_xs_ys(
    method_key: str,
    scales: list[int],
    method_data: dict[tuple[int, str], dict],
    field: str,
    timing_subfield: str | None = None,
) -> tuple[list[int], list[float]]:
    """Extract (x, y) pairs for a method+field, skipping missing entries."""
    xs: list[int] = []
    ys: list[float] = []
    for n in sorted(scales):
        entry = method_data.get((n, method_key))
        if entry is None:
            continue
        if timing_subfield is not None:
            val = (entry.get("timings") or {}).get(timing_subfield)
        else:
            val = entry.get(field)
        fv = safe_float(val)
        if fv is not None:
            xs.append(n)
            ys.append(fv)
    return xs, ys


def _fill_1x2_panels(
    axes: np.ndarray,
    method_data: dict[tuple[int, str], dict],
    ground_truth_data: dict[int, dict],
    scales: list[int],
    methods: list[str],
    gurobi_decomp_data: dict[str, dict[int, dict]] | None = None,
    healed_obj: dict[tuple[int, str], float] | None = None,
) -> None:
    """
    Populate a (3,) axes array with:
      [0] Solve time (log-log)
      [1] Solution quality / objective value (log-log)
      [2] Absolute gap |objective - GT| (log-log)

    gurobi_decomp_data overlaid as dashed lines in time and quality panels.
    healed_obj plotted as dotted hollow-marker lines in quality and gap panels.
    """
    ax_time, ax_gap = axes[0], axes[1]
    sorted_scales = sorted(scales)

    # ── Ground-truth reference ────────────────────────────────────────────────
    gt_time_pts: list[tuple[int, float]] = []
    gt_obj_map: dict[int, float] = {}
    for n in sorted_scales:
        gt = ground_truth_data.get(n)
        if gt is None:
            continue
        t = safe_float(gt.get("solve_time") or gt.get("wall_time"))
        o = safe_float(gt.get("objective"))
        if t is not None:
            gt_time_pts.append((n, t))
        if o is not None:
            gt_obj_map[n] = o

    if gt_time_pts:
        ax_time.plot(
            [p[0] for p in gt_time_pts], [p[1] for p in gt_time_pts],
            color=METHOD_COLORS["ground_truth"], marker="D",
            linestyle="-", label="Gurobi (GT)", linewidth=2, markersize=7, zorder=10,
        )
    if gt_obj_map:
        pass  # Quality panel removed
        # ax_qual.plot(
        #     sorted(gt_obj_map), [gt_obj_map[n] for n in sorted(gt_obj_map)],
        #     color=METHOD_COLORS["ground_truth"], marker="D",
        #     linestyle="-", label="Gurobi (GT)", linewidth=2, markersize=7, zorder=10,
        # )

    # ── QPU method lines ──────────────────────────────────────────────────────
    for mkey in methods:
        disp = METHOD_DISPLAY.get(mkey, mkey)
        color = METHOD_COLORS.get(disp, "#333333")
        marker = METHOD_MARKERS.get(disp, "o")

        xs_t, ys_t = _method_xs_ys(mkey, scales, method_data, "wall_time")
        _print_series_table(f"{disp} time", xs_t, ys_t)
        if xs_t:
            ax_time.plot(xs_t, ys_t, color=color, marker=marker, label=disp,
                         linewidth=1.5, markersize=6)

        xs_o, ys_o = _method_xs_ys(mkey, scales, method_data, "objective")
        # _print_series_table(f"{disp} objective", xs_o, ys_o)
        # if xs_o:
        #     ax_qual.plot(xs_o, ys_o, color=color, marker=marker, label=disp,
        #                  linewidth=1.5, markersize=6)
    #
    # Healed quality overlay
    # if healed_obj:
    #     xs_h: list[int] = []
    #     ys_h: list[float] = []
    #     for n in sorted_scales:
    #         hv = healed_obj.get((n, mkey))
    #         raw = safe_float((method_data.get((n, mkey)) or {}).get("objective"))
    #         viols = int((method_data.get((n, mkey)) or {}).get("violations", 0))
    #         if hv is not None and viols > 0 and raw is not None and abs(hv - raw) > 1e-9:
    #             xs_h.append(n)
    #             ys_h.append(hv)
    #     if xs_h:
    #         ax_qual.plot(xs_h, ys_h, color=color, marker=marker,
    #                      linestyle=":", linewidth=1.2, markersize=5,
    #                      markerfacecolor="white", markeredgecolor=color,
    #                      label=f"{disp} (healed)")

        # Absolute gap (raw)
        xs_g: list[int] = []
        ys_g: list[float] = []
        for n, y in zip(xs_o, ys_o):
            gt_o = gt_obj_map.get(n)
            if gt_o is not None:
                xs_g.append(n)
                ys_g.append(abs(y - gt_o))
        _print_series_table(f"{disp} gap", xs_g, ys_g)
        if xs_g:
            ax_gap.plot(xs_g, ys_g, color=color, marker=marker, label=disp,
                        linewidth=1.5, markersize=6)

        # Absolute gap (healed)
        if healed_obj:
            xs_gh: list[int] = []
            ys_gh: list[float] = []
            for n in sorted_scales:
                hv = healed_obj.get((n, mkey))
                viols = int((method_data.get((n, mkey)) or {}).get("violations", 0))
                gt_o = gt_obj_map.get(n)
                if hv is not None and viols > 0 and gt_o is not None:
                    raw = safe_float((method_data.get((n, mkey)) or {}).get("objective"))
                    if raw is not None and abs(hv - raw) > 1e-9:
                        xs_gh.append(n)
                        ys_gh.append(abs(hv - gt_o))
            if xs_gh:
                ax_gap.plot(xs_gh, ys_gh, color=color, marker=marker,
                            linestyle=":", linewidth=1.2, markersize=5,
                            markerfacecolor="white", markeredgecolor=color,
                            label=f"{disp} (healed)")

    # ── Gurobi-decomposed overlay lines ───────────────────────────────────────
    if gurobi_decomp_data:
        max_scale = max(sorted_scales)
        for dname in FULL_SPAN_GUROBI_DECOMP_METHODS:
            farm_map = gurobi_decomp_data.get(dname)
            if not farm_map:
                continue
            color = GUROBI_DECOMP_COLORS.get(dname, "#555555")

            xs_t2 = sorted(n for n in farm_map
                           if n <= max_scale and farm_map[n].get("wall_time") is not None)
            ys_t2 = [farm_map[n]["wall_time"] for n in xs_t2]
            if xs_t2:
                ax_time.plot(xs_t2, ys_t2, color=color, marker="s",
                             linestyle="--", linewidth=1.5, markersize=5,
                             label=f"Gurobi [{dname}]*", alpha=0.85)

            xs_o2 = sorted(n for n in farm_map
                           if n <= max_scale and farm_map[n].get("objective") is not None)
            # ys_o2 = [farm_map[n]["objective"] for n in xs_o2]
            # if xs_o2:
            #     ax_qual.plot(xs_o2, ys_o2, color=color, marker="s",
            #                  linestyle="--", linewidth=1.5, markersize=5,
            #                  label=f"Gurobi [{dname}]*", alpha=0.85)
            #
            #     Healed quality
            xs_h2 = sorted(n for n in farm_map
                           if n <= max_scale
                           and farm_map[n].get("healed_objective") is not None
                           and farm_map[n].get("objective") is not None
                           and abs(farm_map[n]["healed_objective"] - farm_map[n]["objective"]) > 1e-9)
            # if xs_h2:
            #     ax_qual.plot(xs_h2, [farm_map[n]["healed_objective"] for n in xs_h2],
            #                  color=color, marker="s", linestyle=":", linewidth=1.2,
            #                  markersize=4, markerfacecolor="white", markeredgecolor=color,
            #                  label=f"Gurobi [{dname}] (healed)*", alpha=0.85)

            # Absolute gap (raw)
            xs_ag = [n for n in xs_o2 if gt_obj_map.get(n) is not None]
            ys_ag = [abs(farm_map[n]["objective"] - gt_obj_map[n]) for n in xs_ag]
            if xs_ag:
                ax_gap.plot(xs_ag, ys_ag, color=color, marker="s",
                            linestyle="--", linewidth=1.5, markersize=5,
                            label=f"Gurobi [{dname}]*", alpha=0.85)

            # Absolute gap (healed)
            xs_agh = [n for n in xs_h2 if gt_obj_map.get(n) is not None]
            ys_agh = [abs(farm_map[n]["healed_objective"] - gt_obj_map[n]) for n in xs_agh]
            if xs_agh:
                ax_gap.plot(xs_agh, ys_agh, color=color, marker="s",
                            linestyle=":", linewidth=1.2, markersize=4,
                            markerfacecolor="white", markeredgecolor=color,
                            label=f"Gurobi [{dname}] (healed)*", alpha=0.85)

    # ── Common axis formatting ─────────────────────────────────────────────────
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Wall Time (s)")
    ax_time.set_title("Solve Time")
    ax_time.set_ylim(0.00001, None)

    # ax_qual.set_yscale("log")
    # ax_qual.set_ylabel("Objective Value (Benefit)")
    # ax_qual.set_title("Solution Quality")
    # ax_time.set_ylim(0.0001, None)

    ax_gap.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6,
                   zorder=1, label="Gurobi optimum (gap=0)")
    ax_gap.set_yscale("symlog", linthresh=1e-4)
    ax_gap.set_ylim(0, 1)
    ax_gap.set_ylabel(r"$|$Objective $-$ Gurobi GT$|$")
    ax_gap.set_title("Absolute Solution Gap vs Gurobi")

    for ax in (ax_time,):
        ax.set_xscale("log")
        ax.set_xlabel("Number of Farms")
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ax.grid(True, which="both", color="lightgray", linewidth=0.5)
    ax_gap.set_xscale("log")
    ax_gap.set_xlabel("Number of Farms")
    ax_gap.legend(fontsize=7, ncol=2, loc="best")
    ax_gap.grid(True, which="both", color="lightgray", linewidth=0.5)


def plot_study2(
    method_data: dict[tuple[int, str], dict],
    ground_truth_data: dict[int, dict],
    gurobi_decomp_data: dict[str, dict[int, dict]] | None = None,
    healed_obj: dict[tuple[int, str], float] | None = None,
    gurobi_decomp_violations: dict[str, dict[int, int]] | None = None,
) -> None:
    """Create and save Figure 2: QPU comprehensive benchmark (full-span methods only)."""
    print("\nPlotting Figure 2: QPU Decomposition (comprehensive, full-span methods)")

    all_scales = [10, 15, 25, 50, 100, 200, 500, 1000]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("QPU Decomposition Benchmark --- Comprehensive (10--1000 Farms)")
    _fill_1x2_panels(
        axes, method_data, ground_truth_data,
        all_scales, FULL_SPAN_METHODS,
        gurobi_decomp_data=gurobi_decomp_data,
        healed_obj=healed_obj,
    )
    fig.tight_layout()
    out = PLOT_DIR / "qpu_benchmark_comprehensive.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── LaTeX table utilities ─────────────────────────────────────────────────────

def _row(*cells: str) -> str:
    return " & ".join(cells) + r" \\"


def _fmt(val: float | None, decimals: int = 2, scale: float = 1.0) -> str:
    if val is None:
        return r"---"
    return f"{val * scale:.{decimals}f}"


def _pct_str(part: float | None, total: float | None) -> str:
    if part is None or total is None or total == 0.0:
        return r"---"
    return f"{part / total * 100:.1f}\\%"


# ── LaTeX Table 1: Study 1 Timing ─────────────────────────────────────────────

def _study1_timing_table(entries: list[dict]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Hybrid Solver Timing Results (Study 1)}",
        r"\label{tab:study1_timing}",
        r"\adjustbox{max width=\textwidth}{",
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Patches} & \textbf{Variables} & \textbf{Gurobi (s)} & "
            r"\textbf{CQM Hybrid (s)} & \textbf{CQM QPU (ms)} & \textbf{CQM QPU\%} & "
            r"\textbf{QUBO Gurobi (s)} & \textbf{BQM Hybrid (s)} & \textbf{BQM QPU (ms)} \\"
        ),
        r"\midrule",
    ]
    for e in entries:
        n = e["n_units"]
        nvars = str(e.get("n_variables", "---"))
        s = e["solvers"]

        g = s.get("gurobi", {})
        g_t = _fmt(safe_float(g.get("solve_time")))

        cqm = s.get("dwave_cqm", {})
        cqm_h_s = safe_float(cqm.get("hybrid_time"))
        cqm_q_s = safe_float(cqm.get("qpu_time"))
        cqm_h = _fmt(cqm_h_s)
        cqm_q = _fmt(cqm_q_s, 1, 1000.0)
        cqm_pct = _pct_str(cqm_q_s, cqm_h_s)

        gq = s.get("gurobi_qubo", {})
        gq_ok = bool(gq.get("success")) and gq.get("status", "").lower() != "error"
        gq_t = _fmt(safe_float(gq.get("solve_time")) if gq_ok else None)

        bqm = s.get("dwave_bqm", {})
        bqm_h_s = safe_float(bqm.get("hybrid_time") or bqm.get("solve_time"))
        bqm_q_s = safe_float(bqm.get("qpu_time"))
        bqm_h = _fmt(bqm_h_s)
        bqm_q = _fmt(bqm_q_s, 1, 1000.0)

        lines.append(_row(str(n), nvars, g_t, cqm_h, cqm_q, cqm_pct, gq_t, bqm_h, bqm_q))

    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(lines)


# ── LaTeX Table 2: Study 1 Violations ────────────────────────────────────────

def _study1_violations_table(entries: list[dict]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Hybrid Solver Constraint Violations (Study 1)}",
        r"\label{tab:study1_violations}",
        r"\adjustbox{max width=\textwidth}{",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Patches} & \textbf{Gurobi Viols} & \textbf{CQM Viols} & "
            r"\textbf{CQM Feasible} & \textbf{QUBO Viols} & "
            r"\textbf{BQM Viols} & \textbf{BQM Feasible} \\"
        ),
        r"\midrule",
    ]
    for e in entries:
        n = e["n_units"]
        s = e["solvers"]

        # Gurobi is an exact solver — zero violations
        g_v = "0"

        cqm = s.get("dwave_cqm", {})
        cqm_val = cqm.get("validation")
        cqm_v = str(get_violations(cqm_val))
        cqm_feas = "Yes" if (cqm_val and cqm_val.get("is_feasible")) else "No"

        gq = s.get("gurobi_qubo", {})
        gq_ok = bool(gq.get("success")) and gq.get("status", "").lower() != "error"
        if gq_ok:
            gq_v = str(get_violations(gq.get("validation")))
        else:
            gq_v = "Error"

        bqm = s.get("dwave_bqm", {})
        bqm_val = bqm.get("validation")
        bqm_v = str(get_violations(bqm_val))
        bqm_feas = "Yes" if (bqm_val and bqm_val.get("is_feasible")) else "No"

        lines.append(_row(str(n), g_v, cqm_v, cqm_feas, gq_v, bqm_v, bqm_feas))

    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(lines)


# ── LaTeX Table 3: Study 2 Timing ────────────────────────────────────────────

def _study2_timing_table(
    method_data: dict[tuple[int, str], dict],
    ground_truth_data: dict[int, dict],
) -> str:
    scales = sorted({n for (n, _) in method_data})
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{QPU Decomposition Timing Breakdown (Study 2)}",
        r"\label{tab:study2_timing}",
        r"\adjustbox{max width=\textwidth}{",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (
            r"\textbf{Method} & \textbf{Farms} & \textbf{Total (s)} & "
            r"\textbf{QPU (s)} & \textbf{Embed (s)} & \textbf{QPU\%} \\"
        ),
        r"\midrule",
    ]
    for mkey in ALL_METHODS:
        disp = METHOD_DISPLAY[mkey]
        rows_added = 0
        for n in scales:
            entry = method_data.get((n, mkey))
            if entry is None:
                continue
            total_s = safe_float(entry.get("wall_time") or entry.get("total_time"))
            qpu_s = safe_float(entry.get("total_qpu_time"))
            embed_s = safe_float((entry.get("timings") or {}).get("embedding_total"))
            lines.append(_row(
                disp if rows_added == 0 else "",
                str(n),
                _fmt(total_s),
                _fmt(qpu_s),
                _fmt(embed_s),
                _pct_str(qpu_s, total_s),
            ))
            rows_added += 1
        if rows_added > 0:
            lines.append(r"\midrule")
    # Remove trailing \midrule before \bottomrule
    while lines and lines[-1].strip() == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(lines)


# ── LaTeX Table 4: Study 2 Violations ────────────────────────────────────────

def _study2_violations_table(
    method_data: dict[tuple[int, str], dict],
    healed_obj: dict[tuple[int, str], float] | None = None,
    gurobi_decomp_violations: dict[str, dict[int, int]] | None = None,
) -> str:
    scales = sorted({n for (n, _) in method_data})
    has_healed = healed_obj is not None
    col_spec = r"llrrrr" if has_healed else r"llrrr"
    header_extra = r" & \textbf{Healed Obj}" if has_healed else ""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{QPU Decomposition Constraint Violations (Study 2)}",
        r"\label{tab:study2_violations}",
        r"\adjustbox{max width=\textwidth}{",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        (
            r"\textbf{Method} & \textbf{Farms} & \textbf{Total Viols} & "
            r"\textbf{One-Crop Viols} & \textbf{Food-Group Viols}"
            + header_extra + r" \\"
        ),
        r"\midrule",
    ]
    for mkey in ALL_METHODS:
        disp = METHOD_DISPLAY[mkey]
        rows_added = 0
        for n in scales:
            entry = method_data.get((n, mkey))
            if entry is None:
                continue
            total_v = int(entry.get("violations", 0))
            vd = entry.get("violation_details") or {}
            one_crop = int((vd.get("one_crop_per_farm") or {}).get("violations", 0))
            food_grp = int((vd.get("food_group_constraints") or {}).get("violations", 0))
            cells = [
                disp if rows_added == 0 else "",
                str(n),
                str(total_v),
                str(one_crop),
                str(food_grp),
            ]
            if has_healed:
                hv = healed_obj.get((n, mkey))  # type: ignore[union-attr]
                cells.append(_fmt(hv, 4))
            lines.append(_row(*cells))
            rows_added += 1
        if rows_added > 0:
            lines.append(r"\midrule")
    # ── Gurobi decomposed section ──────────────────────────────────────────────
    if gurobi_decomp_violations:
        lines.append(r"\midrule")
        lines.append(
            r"\multicolumn{5}{ l }{\textit{Gurobi Decomposed (Variant-A, stricter formulation)*}} \\"
        )
        lines.append(r"\midrule")
        gd_scales = sorted(
            {n for vm in gurobi_decomp_violations.values() for n in vm}
        )
        for dname in sorted(gurobi_decomp_violations):
            farm_map = gurobi_decomp_violations[dname]
            rows_added = 0
            for n in gd_scales:
                v = farm_map.get(n)
                if v is None:
                    continue
                cells = [
                    dname if rows_added == 0 else "",
                    str(n),
                    str(v),
                    r"---",
                    r"---",
                ]
                if has_healed:
                    cells.append(r"---")
                lines.append(_row(*cells))
                rows_added += 1
            if rows_added > 0:
                lines.append(r"\midrule")
    while lines and lines[-1].strip() == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(lines)


# ── Write tables file ─────────────────────────────────────────────────────────

def write_tables(
    entries: list[dict],
    method_data: dict[tuple[int, str], dict],
    ground_truth_data: dict[int, dict],
    healed_obj: dict[tuple[int, str], float] | None = None,
    gurobi_decomp_violations: dict[str, dict[int, int]] | None = None,
) -> None:
    print("\nGenerating LaTeX tables")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / "report_tables.tex"
    sections = [
        "% AUTO-GENERATED by generate_report_visuals.py — do not edit manually",
        r"% Requires \usepackage{booktabs} and \usepackage{adjustbox}",
        "",
        _study1_timing_table(entries),
        "",
        _study1_violations_table(entries),
        "",
        _study2_timing_table(method_data, ground_truth_data),
        "",
        _study2_violations_table(method_data, healed_obj, gurobi_decomp_violations),
    ]
    out.write_text("\n".join(sections), encoding="utf-8")
    print(f"  Saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 64)
    print("generate_report_visuals.py")
    print("=" * 64)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading Study 1 data ...")
    study1_entries = load_study1()
    print(f"  {len(study1_entries)} scale entries: "
          f"{[e['n_units'] for e in study1_entries]}")

    print("\n[2/3] Loading Study 2 QPU data ...")
    method_data, qpu_ground_truth = load_study2()
    print(f"  {len(method_data)} (n_farms, method) data points loaded")
    print(f"  QPU benchmark GT at farms: {sorted(qpu_ground_truth)}")
    solver_comparison = _load_solver_comparison()
    gurobi_full_a = load_gurobi_full_a(solver_comparison)
    print(f"  Gurobi full A (solver comparison) at farms: {sorted(gurobi_full_a)}")
    gurobi_decomp_data = load_gurobi_decompositions_a(solver_comparison)
    print(f"  Gurobi decomposed methods loaded: {sorted(gurobi_decomp_data)}")
    gurobi_decomp_viols = load_gurobi_decomp_violations(solver_comparison)
    print(f"  Loading benefit matrix ...")
    benefit_matrix = load_benefit_matrix()
    print(f"  {len(benefit_matrix)} crops in benefit matrix")
    healed_obj = compute_healed_obj_qpu(method_data, benefit_matrix)
    print(f"  Healed objectives computed for {len(healed_obj)} (n, method) pairs")
    healed_study1 = compute_healed_obj_study1(study1_entries, benefit_matrix)
    print(f"  Study 1 healed objectives: "
          f"cqm={sorted(healed_study1['cqm'])} bqm={sorted(healed_study1['bqm'])}")

    print("\n[3/3] Generating figures and tables ...")
    # Study 1: Gurobi line from solver comparison (canonical, matches fig_A plots)
    plot_study1(study1_entries, gurobi_full_a, healed_study1=healed_study1)
    # Study 2: GT from QPU benchmark (same formulation as QPU methods);
    #          Gurobi decomposed overlay from solver comparison (stricter formulation,
    #          labelled with * to flag different constraint set).
    plot_study2(
        method_data, qpu_ground_truth,
        gurobi_decomp_data=gurobi_decomp_data,
        healed_obj=healed_obj,
        gurobi_decomp_violations=gurobi_decomp_viols,
    )
    write_tables(
        study1_entries, method_data, qpu_ground_truth,
        healed_obj=healed_obj,
        gurobi_decomp_violations=gurobi_decomp_viols,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    pdfs = [
        "study1_hybrid_performance.pdf",
        "qpu_benchmark_comprehensive.pdf",
    ]
    for name in pdfs:
        status = "OK" if (PLOT_DIR / name).exists() else "MISSING"
        print(f"  [{status}]  {PLOT_DIR / name}")
    tex_out = TABLE_DIR / "report_tables.tex"
    status = "OK" if tex_out.exists() else "MISSING"
    print(f"  [{status}]  {tex_out}")
    print("Done.")


if __name__ == "__main__":
    main()
