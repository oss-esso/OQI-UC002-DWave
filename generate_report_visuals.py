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
for _style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue


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
        gqubo_objs.append(safe_float(gq.get("objective_value")) if gq_ok else None)

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


def plot_study1(entries: list[dict]) -> None:
    """Create Figure 1: 1×2 hybrid solver comparison."""
    print("\nPlotting Figure 1: Study 1 — Hybrid Solver Comparison")
    series = _collect_study1_series(entries)

    fig, (ax_t, ax_q) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Runtime
    _add_line(ax_t, *series["gurobi_time"], "Gurobi", "#1f77b4", "o")
    _add_line(ax_t, *series["cqm_time"], "D-Wave CQM (Hybrid)", "#ff7f0e", "s")
    _add_line(ax_t, *series["gqubo_time"], "Gurobi QUBO", "#2ca02c", "^", "--")
    _add_line(ax_t, *series["bqm_time"], "D-Wave BQM (Hybrid)", "#d62728", "D")
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_xlabel("Number of Patches (n)", fontsize=11)
    ax_t.set_ylabel("Solve Time (s)", fontsize=11)
    ax_t.set_title("Solver Runtime Scaling", fontsize=12)
    ax_t.legend(fontsize=9)
    ax_t.grid(True, which="both", alpha=0.4)

    # Subplot 2: Solution quality
    _add_line(ax_q, *series["gurobi_obj"], "Gurobi", "#1f77b4", "o")
    _add_line(ax_q, *series["cqm_obj"], "D-Wave CQM (Hybrid)*", "#ff7f0e", "s")
    _add_line(ax_q, *series["gqubo_obj"], "Gurobi QUBO", "#2ca02c", "^", "--")
    _add_line(ax_q, *series["bqm_obj"], "D-Wave BQM (Hybrid)", "#d62728", "D")
    ax_q.set_xscale("log")
    ax_q.set_xlabel("Number of Patches (n)", fontsize=11)
    ax_q.set_ylabel("Objective Value (Benefit)", fontsize=11)
    ax_q.set_title("Solution Quality (Benefit)", fontsize=12)
    ax_q.legend(fontsize=9)
    ax_q.grid(True, which="both", alpha=0.4)
    ax_q.annotate(
        "* CQM status = Infeasible at all scales",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=8, color="#ff7f0e", style="italic",
    )

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


def _fill_2x2_panels(
    axes: np.ndarray,
    method_data: dict[tuple[int, str], dict],
    ground_truth_data: dict[int, dict],
    scales: list[int],
    methods: list[str],
) -> None:
    """Populate a (2,2) axes array with time, quality, gap, and violations panels."""
    ax_time = axes[0, 0]
    ax_qual = axes[0, 1]
    ax_gap = axes[1, 0]
    ax_viols = axes[1, 1]

    sorted_scales = sorted(scales)

    # ── Ground-truth reference line ────────────────────────────────────────────
    gt_time_pts: list[tuple[int, float]] = []
    gt_obj_pts: list[tuple[int, float]] = []
    for n in sorted_scales:
        gt = ground_truth_data.get(n)
        if gt is None:
            continue
        t = safe_float(gt.get("solve_time") or gt.get("wall_time"))
        o = safe_float(gt.get("objective"))
        if t is not None:
            gt_time_pts.append((n, t))
        if o is not None:
            gt_obj_pts.append((n, o))

    def _gt_line(ax: plt.Axes, pts: list[tuple[int, float]]) -> None:
        if pts:
            ax.plot(
                [p[0] for p in pts], [p[1] for p in pts],
                color=METHOD_COLORS["ground_truth"], marker="D",
                linestyle="-", label="Gurobi (GT)", linewidth=2, markersize=7, zorder=10,
            )

    _gt_line(ax_time, gt_time_pts)
    _gt_line(ax_qual, gt_obj_pts)

    # ── Method lines ───────────────────────────────────────────────────────────
    for mkey in methods:
        disp = METHOD_DISPLAY.get(mkey, mkey)
        color = METHOD_COLORS.get(disp, "#333333")
        marker = METHOD_MARKERS.get(disp, "o")

        # Time
        xs, ys = _method_xs_ys(mkey, scales, method_data, "wall_time")
        if xs:
            ax_time.plot(xs, ys, color=color, marker=marker, label=disp,
                         linewidth=1.5, markersize=6)

        # Quality
        xs_o, ys_o = _method_xs_ys(mkey, scales, method_data, "objective")
        if xs_o:
            ax_qual.plot(xs_o, ys_o, color=color, marker=marker, label=disp,
                         linewidth=1.5, markersize=6)

        # Optimality gap
        xs_g: list[int] = []
        ys_g: list[float] = []
        for n in sorted_scales:
            entry = method_data.get((n, mkey))
            gt = ground_truth_data.get(n)
            if entry is None or gt is None:
                continue
            m_obj = safe_float(entry.get("objective"))
            gt_obj = safe_float(gt.get("objective"))
            if m_obj is None or gt_obj is None or gt_obj == 0.0:
                continue
            xs_g.append(n)
            ys_g.append((gt_obj - m_obj) / gt_obj * 100.0)
        if xs_g:
            ax_gap.plot(xs_g, ys_g, color=color, marker=marker, label=disp,
                        linewidth=1.5, markersize=6)

    # ── Violations bar chart ───────────────────────────────────────────────────
    data_scales = [
        n for n in sorted_scales
        if any(method_data.get((n, m)) is not None for m in methods)
    ]
    if data_scales:
        n_m = len(methods)
        bar_w = 0.8 / max(n_m, 1)
        x_pos = np.arange(len(data_scales))
        for i, mkey in enumerate(methods):
            disp = METHOD_DISPLAY.get(mkey, mkey)
            color = METHOD_COLORS.get(disp, "#333333")
            bar_vals = [
                int((method_data.get((n, mkey)) or {}).get("violations", 0))
                for n in data_scales
            ]
            offset = (i - n_m / 2 + 0.5) * bar_w
            ax_viols.bar(
                x_pos + offset, bar_vals, bar_w * 0.9,
                color=color, label=disp, alpha=0.85,
            )
        ax_viols.set_xticks(x_pos)
        ax_viols.set_xticklabels([str(n) for n in data_scales], rotation=30, ha="right")
        ax_viols.set_xlabel("Number of Farms", fontsize=10)
        ax_viols.set_ylabel("Constraint Violations", fontsize=10)
        ax_viols.set_title("Constraint Violations by Method and Scale", fontsize=11)
        ax_viols.legend(fontsize=7, ncol=2)
        ax_viols.grid(True, axis="y", alpha=0.4)

    # ── Common axis formatting ─────────────────────────────────────────────────
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Wall Time (s)", fontsize=10)
    ax_time.set_title("Solve Time", fontsize=11)

    ax_qual.set_ylabel("Objective Value (Benefit)", fontsize=10)
    ax_qual.set_title("Solution Quality", fontsize=11)

    ax_gap.set_ylabel("Optimality Gap (%)", fontsize=10)
    ax_gap.set_title("Optimality Gap vs. Gurobi", fontsize=11)

    for ax in (ax_time, ax_qual, ax_gap):
        ax.set_xscale("log")
        ax.set_xlabel("Number of Farms", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which="both", alpha=0.4)


def plot_study2(
    method_data: dict[tuple[int, str], dict],
    ground_truth_data: dict[int, dict],
) -> None:
    """Create and save Figures 2a (small), 2b (large), 2c (comprehensive)."""
    print("\nPlotting Figure 2: QPU Decomposition")

    small_scales = [10, 15, 25, 50, 100]
    large_scales = [200, 500, 1000]
    all_scales = small_scales + large_scales

    large_methods = [
        "decomposition_Multilevel(10)_QPU",
        "cqm_first_PlotBased",
        "coordinated",
        "decomposition_HybridGrid(5,9)_QPU",
    ]

    configs: list[tuple[str, list[int], list[str], str]] = [
        ("qpu_benchmark_small_scale", small_scales, ALL_METHODS,
         "Small Scale (10–100 Farms)"),
        ("qpu_benchmark_large_scale", large_scales, large_methods,
         "Large Scale (200–1000 Farms)"),
        ("qpu_benchmark_comprehensive", all_scales, ALL_METHODS,
         "Comprehensive (10–1000 Farms)"),
    ]

    for stem, scales, methods, title in configs:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"QPU Decomposition Benchmark — {title}", fontsize=13, y=1.01)
        _fill_2x2_panels(axes, method_data, ground_truth_data, scales, methods)
        fig.tight_layout()
        out = PLOT_DIR / f"{stem}.pdf"
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

def _study2_violations_table(method_data: dict[tuple[int, str], dict]) -> str:
    scales = sorted({n for (n, _) in method_data})
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{QPU Decomposition Constraint Violations (Study 2)}",
        r"\label{tab:study2_violations}",
        r"\adjustbox{max width=\textwidth}{",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (
            r"\textbf{Method} & \textbf{Farms} & \textbf{Total Viols} & "
            r"\textbf{One-Crop Viols} & \textbf{Food-Group Viols} \\"
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
            lines.append(_row(
                disp if rows_added == 0 else "",
                str(n),
                str(total_v),
                str(one_crop),
                str(food_grp),
            ))
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
        _study2_violations_table(method_data),
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
    method_data, ground_truth_data = load_study2()
    print(f"  {len(method_data)} (n_farms, method) data points loaded")
    print(f"  Ground truth available at farms: {sorted(ground_truth_data)}")

    print("\n[3/3] Generating figures and tables ...")
    plot_study1(study1_entries)
    plot_study2(method_data, ground_truth_data)
    write_tables(study1_entries, method_data, ground_truth_data)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    pdfs = [
        "study1_hybrid_performance.pdf",
        "qpu_benchmark_small_scale.pdf",
        "qpu_benchmark_large_scale.pdf",
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
