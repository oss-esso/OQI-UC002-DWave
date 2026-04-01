#!/usr/bin/env python3
"""
Comprehensive plot script for all benchmark results.

Reads:
  - Benchmarks/decomposition_scaling/solver_comparison_results_{hard,soft}.json
    (falls back to solver_comparison_results.json if mode files absent)
  - Benchmarks/decomposition_scaling/decomposition_scaling_results.json
  - benchmark_20260329_105633.json  (unified_benchmark gurobi 200s run)
  - @todo/gurobi_timeout_verification/gurobi_timeout_test_20260329_163901.json  (1200s)
  - @todo/gurobi_timeout_verification/gurobi_timeout_test_20260331_141105.json  (200s)
  - gurobi_baseline_60s.json

Outputs PDF/PNG to Benchmarks/decomposition_scaling/
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DECOMP_DIR = HERE / "Benchmarks" / "decomposition_scaling"
TIMEOUT_DIR = HERE / "@todo" / "gurobi_timeout_verification"
OUT_DIR = DECOMP_DIR  # all plots land here

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

# ── colours & styles ──────────────────────────────────────────────────────────

SOLVER_COLORS = {
    "Gurobi_full":                      "#e41a1c",
    "Gurobi_decomposed|PlotBased":      "#377eb8",
    "Gurobi_decomposed|Multilevel(5)":  "#4daf4a",
    "Gurobi_decomposed|HybridGrid(5,9)":"#984ea3",
    "Gurobi_decomposed|Clique":         "#ff7f00",
    "Gurobi_decomposed|SpatialTemporal(5)": "#a65628",
}
SOLVER_MARKERS = {
    "Gurobi_full":                      ("o", "-",  2.0),
    "Gurobi_decomposed|PlotBased":      ("s", "--", 1.5),
    "Gurobi_decomposed|Multilevel(5)":  ("^", "--", 1.5),
    "Gurobi_decomposed|HybridGrid(5,9)":("D", "--", 1.5),
    "Gurobi_decomposed|Clique":         ("v", "--", 1.5),
    "Gurobi_decomposed|SpatialTemporal(5)": ("<", "-.", 1.5),
}
TIMEOUT_STYLES = {
    "200s":  {"color": "#e41a1c", "ls": "--", "marker": "o"},
    "1200s": {"color": "#377eb8", "ls": "-",  "marker": "s"},
    "60s":   {"color": "#4daf4a", "ls": "-.", "marker": "^"},
}

# ── data loaders ──────────────────────────────────────────────────────────────

def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_solver_comparison(mode: str = "soft"):
    """Load solver comparison results for the given mode."""
    preferred = DECOMP_DIR / f"solver_comparison_results_{mode}.json"
    fallback = DECOMP_DIR / "solver_comparison_results.json"
    path = preferred if preferred.exists() else fallback
    if not path.exists():
        print(f"  WARNING: solver_comparison_results not found (mode={mode})", file=sys.stderr)
        return []
    data = _load_json(path)
    print(f"  Loaded {len(data)} entries from {path.name} (mode={mode})")
    return data


def load_decomp_scaling():
    path = DECOMP_DIR / "decomposition_scaling_results.json"
    if not path.exists():
        return []
    data = _load_json(path)
    print(f"  Loaded {len(data)} entries from {path.name}")
    return data


def load_gurobi_unified(json_path: Path, label: str):
    """Load a unified_benchmark output JSON, return list of dicts with common keys."""
    if not json_path.exists():
        print(f"  WARNING: {json_path} not found", file=sys.stderr)
        return []
    raw = _load_json(json_path)
    runs = raw.get("runs", raw) if isinstance(raw, dict) else raw
    out = []
    for r in runs:
        mg = r.get("mip_gap")
        if mg is None or not np.isfinite(float(mg) if mg is not None else float("nan")):
            mg = None
        elif float(mg) > 1000:
            mg = None  # garbage values from Gurobi on infeasible sub-trees
        out.append({
            "scenario": r.get("scenario_name", r.get("scenario", "?")),
            "n_farms": r["n_farms"],
            "n_foods": r.get("n_foods", 6),
            "n_vars": r["n_vars"],
            "status": r["status"],
            "objective": r.get("objective_miqp", r.get("objective_value")),
            "mip_gap": float(mg) if mg is not None else None,
            "solve_time": r.get("timing", {}).get("total_wall_time",
                          r.get("timing", {}).get("solve_time",
                          r.get("solve_time", 0.0))),
            "label": label,
        })
    print(f"  Loaded {len(out)} runs from {json_path.name} ({label})")
    return out


def load_gurobi_timeout_test(json_path: Path, label: str):
    """Load a gurobi_timeout_test JSON, return list of dicts with common keys."""
    if not json_path.exists():
        print(f"  WARNING: {json_path} not found", file=sys.stderr)
        return []
    raw = _load_json(json_path)
    out = []
    for entry in raw:
        m = entry.get("metadata", {})
        r = entry.get("result", {})
        mg = r.get("mip_gap")
        if mg is not None and np.isfinite(float(mg)) and float(mg) < 1000:
            mg = float(mg)
        else:
            mg = None
        out.append({
            "scenario": m.get("scenario", r.get("scenario", "?")),
            "n_farms": m.get("n_farms", 0),
            "n_foods": m.get("n_foods", 6),
            "n_vars": r.get("n_vars", 0),
            "status": r.get("stopped_reason", "?"),
            "objective": r.get("objective_value"),
            "mip_gap": mg,
            "solve_time": r.get("solve_time", 0.0),
            "label": label,
        })
    print(f"  Loaded {len(out)} runs from {json_path.name} ({label})")
    return out


def load_gurobi_baseline_60s():
    path = HERE / "gurobi_baseline_60s.json"
    return load_gurobi_unified(path, "60s baseline")


# ── helper: solver key → display label ───────────────────────────────────────

def _solver_key(r):
    decomp = r.get("decomposition", "none")
    return r["solver"] if decomp == "none" else f"{r['solver']}|{decomp}"


def _display_label(key: str) -> str:
    if "|" not in key:
        return key
    solver, decomp = key.split("|", 1)
    return f"{solver} [{decomp}]"


# ── plotting helpers ──────────────────────────────────────────────────────────

def _style(key):
    m, ls, lw = SOLVER_MARKERS.get(key, ("o", "-", 1.2))
    c = SOLVER_COLORS.get(key, "#555555")
    return {"color": c, "marker": m, "linestyle": ls, "linewidth": lw, "markersize": 5}


def _plot_series(ax, xs, ys, key, **extra):
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pairs:
        return
    xs_s, ys_s = zip(*sorted(pairs))
    sty = _style(key)
    sty.update(extra)
    ax.plot(xs_s, ys_s, label=_display_label(key), **sty)


def _finalize(ax, xlabel, ylabel, title, log_x=True, log_y=True, loc="upper left"):
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=loc, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)


def _save(fig, fname):
    out = OUT_DIR / fname
    fig.savefig(out)
    # Also save PNG for easy viewing
    png = out.with_suffix(".png")
    fig.savefig(png, format="png")
    plt.close(fig)
    print(f"  -> {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Variant A – solve time
# ═══════════════════════════════════════════════════════════════════════════════

def fig_A_solve_time(data):
    series = defaultdict(lambda: ([], []))
    for r in data:
        if r["variant"] != "A":
            continue
        key = _solver_key(r)
        series[key][0].append(r["n_farms"])
        series[key][1].append(r["wall_time"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in sorted(series):
        _plot_series(ax, *series[key], key)
    _finalize(ax, "Number of farms", "Wall-clock time (s)",
              "Variant A — Solver Wall Time vs Problem Size")
    _save(fig, "fig_A_solver_time.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Variant A – objective quality
# ═══════════════════════════════════════════════════════════════════════════════

def fig_A_objective(data):
    series = defaultdict(lambda: ([], []))
    for r in data:
        if r["variant"] != "A" or r.get("objective") is None:
            continue
        key = _solver_key(r)
        series[key][0].append(r["n_farms"])
        series[key][1].append(abs(r["objective"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in sorted(series):
        _plot_series(ax, *series[key], key)
    _finalize(ax, "Number of farms", "Objective value (maximised area benefit)",
              "Variant A — Solution Quality vs Problem Size", log_y=False)
    _save(fig, "fig_A_solver_quality.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Variant A – quality ratio (decomposed / full)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_A_quality_ratio(data):
    full_obj = {}
    decomp_series = defaultdict(lambda: ([], []))
    for r in data:
        if r["variant"] != "A" or r.get("objective") is None:
            continue
        if r["decomposition"] == "none":
            full_obj[r["n_farms"]] = abs(r["objective"])
        else:
            key = _solver_key(r)
            decomp_series[key][0].append(r["n_farms"])
            decomp_series[key][1].append(abs(r["objective"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in sorted(decomp_series):
        xs, ys = decomp_series[key]
        ratios = [y / full_obj[x] if x in full_obj and full_obj[x] > 0 else None
                  for x, y in zip(xs, ys)]
        _plot_series(ax, xs, ratios, key)
    ax.axhline(1.0, color="red", ls=":", lw=1.5, label="Parity (1.0)")
    _finalize(ax, "Number of farms", "Objective ratio (decomposed / full MIQP)",
              "Variant A — Solution Quality Ratio vs Full Solver",
              log_x=True, log_y=False, loc="best")
    _save(fig, "fig_A_quality_ratio.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3b: Variant A – raw vs healed objective (decomposed methods)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_A_healed_objective(data):
    """Show raw (infeasible) vs healed (warm-started feasible) objective."""
    full_obj = {}
    raw_series = defaultdict(lambda: ([], []))
    healed_series = defaultdict(lambda: ([], []))

    for r in data:
        if r["variant"] != "A":
            continue
        if r["decomposition"] == "none":
            if r.get("objective") is not None:
                full_obj[r["n_farms"]] = abs(r["objective"])
            continue
        key = _solver_key(r)
        n = r["n_farms"]
        raw = r.get("objective")
        healed = r.get("healed_objective")
        if raw is not None:
            raw_series[key][0].append(n)
            raw_series[key][1].append(abs(raw))
        if healed is not None:
            healed_series[key][0].append(n)
            healed_series[key][1].append(abs(healed))

    if not healed_series:
        print("  SKIP fig_A_healed_objective (no healed data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute objective comparison
    ax = axes[0]
    # Plot full solver as reference
    if full_obj:
        xs = sorted(full_obj.keys())
        ys = [full_obj[x] for x in xs]
        ax.plot(xs, ys, color="#e41a1c", marker="o", ls="-", lw=2.0,
                markersize=5, label="Gurobi full", zorder=10)

    for key in sorted(raw_series):
        sty = _style(key)
        sty["alpha"] = 0.4
        sty["linestyle"] = ":"
        pairs = sorted(zip(*raw_series[key]))
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                label=f"{_display_label(key)} (raw)", **sty)

    for key in sorted(healed_series):
        sty = _style(key)
        pairs = sorted(zip(*healed_series[key]))
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                label=f"{_display_label(key)} (healed)", **sty)

    ax.set_xscale("log")
    ax.set_xlabel("Number of farms")
    ax.set_ylabel("Objective value")
    ax.set_title("Raw vs Healed Objective")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)

    # Right: healed / full ratio
    ax = axes[1]
    for key in sorted(healed_series):
        xs, ys = healed_series[key]
        ratios = [y / full_obj[x] if x in full_obj and full_obj[x] > 0 else None
                  for x, y in zip(xs, ys)]
        _plot_series(ax, xs, ratios, key)
    ax.axhline(1.0, color="red", ls=":", lw=1.5, label="Parity (1.0)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of farms")
    ax.set_ylabel("Healed / Full ratio")
    ax.set_title("Healed Objective as Fraction of Full Solver")
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig_A_healed_objective.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3c: Variant A – constraint violations (decomposed methods)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_A_violations(data):
    """Show constraint violation counts per decomposition method."""
    series_total = defaultdict(lambda: ([], []))
    series_by_type: dict[str, dict[str, list]] = {}  # key -> {viol_type -> (xs, ys)}

    for r in data:
        if r["variant"] != "A" or r["decomposition"] == "none":
            continue
        vc = r.get("violation_counts")
        if vc is None:
            continue
        key = _solver_key(r)
        n = r["n_farms"]
        total = r.get("violations", sum(vc.values()))
        series_total[key][0].append(n)
        series_total[key][1].append(total)

        if key not in series_by_type:
            series_by_type[key] = defaultdict(lambda: ([], []))
        for vtype, cnt in vc.items():
            series_by_type[key][vtype][0].append(n)
            series_by_type[key][vtype][1].append(cnt)

    if not series_total:
        print("  SKIP fig_A_violations (no violation data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total violations per method
    ax = axes[0]
    for key in sorted(series_total):
        _plot_series(ax, *series_total[key], key)
    ax.set_xscale("log")
    ax.set_xlabel("Number of farms")
    ax.set_ylabel("Total constraint violations")
    ax.set_title("Variant A — Constraint Violations vs Scale")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)

    # Right: violation breakdown for HybridGrid (or worst method)
    ax = axes[1]
    worst_key = max(series_total.keys(),
                    key=lambda k: max(series_total[k][1]) if series_total[k][1] else 0)
    vtype_colors = {
        "one_crop": "#e41a1c",
        "min_planting_area": "#377eb8",
        "max_percentage": "#4daf4a",
        "food_group_min": "#984ea3",
        "food_group_max": "#ff7f00",
    }
    if worst_key in series_by_type:
        for vtype in sorted(series_by_type[worst_key]):
            xs, ys = series_by_type[worst_key][vtype]
            if sum(ys) == 0:
                continue
            c = vtype_colors.get(vtype, "#555555")
            pairs = sorted(zip(xs, ys))
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                    marker="o", label=vtype, color=c, markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Number of farms")
    ax.set_ylabel("Violation count")
    ax.set_title(f"Violation Breakdown — {_display_label(worst_key)}")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig_A_violations.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Variant B – solve time
# ═══════════════════════════════════════════════════════════════════════════════

def fig_B_solve_time(data):
    series = defaultdict(lambda: ([], []))
    for r in data:
        if r["variant"] != "B" or r.get("wall_time", 0) == 0:
            continue
        if r.get("status") == "SKIPPED":
            continue
        key = _solver_key(r)
        series[key][0].append(r["n_farms"])
        series[key][1].append(r["wall_time"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in sorted(series):
        _plot_series(ax, *series[key], key)
    _finalize(ax, "Number of farms", "Wall-clock time (s)",
              "Variant B — Solver Wall Time vs Problem Size")
    _save(fig, "fig_B_solver_time.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Variant B – objective quality
# ═══════════════════════════════════════════════════════════════════════════════

def fig_B_objective(data):
    series = defaultdict(lambda: ([], []))
    for r in data:
        if r["variant"] != "B" or r.get("objective") is None:
            continue
        if r.get("status") == "SKIPPED":
            continue
        key = _solver_key(r)
        series[key][0].append(r["n_farms"])
        series[key][1].append(abs(r["objective"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in sorted(series):
        _plot_series(ax, *series[key], key)
    _finalize(ax, "Number of farms", "Objective value",
              "Variant B — Solution Quality vs Problem Size",
              log_x=True, log_y=True)
    _save(fig, "fig_B_solver_quality.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Decomposition overhead (Variant A & B, overlay)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_decomp_overhead(decomp_data):
    SHOW_A = {"PlotBased", "Multilevel(5)", "HybridGrid(5,9)"}
    SHOW_B = {"Clique(farm-by-farm)", "SpatialTemporal(5)"}
    colors_a = {"PlotBased": "#1f77b4", "Multilevel(5)": "#ff7f0e", "HybridGrid(5,9)": "#2ca02c"}
    colors_b = {"Clique(farm-by-farm)": "#d62728", "SpatialTemporal(5)": "#9467bd",
                "Clique": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for variant, ax, show_set, cmap in (
        ("A", axes[0], SHOW_A, colors_a),
        ("B", axes[1], SHOW_B, colors_b),
    ):
        series = defaultdict(lambda: ([], []))
        for r in decomp_data:
            if r["variant"] != variant:
                continue
            m = r["method"]
            if m not in show_set:
                continue
            series[m][0].append(r["n_farms"])
            series[m][1].append(r["decomposition_time_s"])
        for m, (xs, ys) in sorted(series.items()):
            c = cmap.get(m, "#555")
            pairs = sorted(zip(xs, ys))
            if pairs:
                xs_s, ys_s = zip(*pairs)
                ax.plot(xs_s, ys_s, marker="o", label=m, color=c, markersize=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of farms")
        ax.set_ylabel("Decomposition overhead (s)")
        ax.set_title(f"Decomposition Overhead — Variant {variant}")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig_decomp_overhead.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7: Rotation – 6-food families (objective + MIP gap + time)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_rotation_6food(runs_200s, runs_1200s, runs_60s):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for runs, label in ((runs_60s, "60s"), (runs_200s, "200s"), (runs_1200s, "1200s")):
        pts = [(r["n_vars"], r["objective"], r["mip_gap"], r["solve_time"])
               for r in runs if r["n_foods"] == 6 and r["objective"] is not None]
        if not pts:
            continue
        pts.sort()
        xs, objs, gaps, times = zip(*pts)
        sty = TIMEOUT_STYLES.get(label, {"color": "#555", "ls": "-", "marker": "o"})
        axes[0].plot(xs, objs, label=label, **sty, markersize=5, linewidth=1.5)
        valid_gaps = [(x, g) for x, g in zip(xs, gaps) if g is not None]
        if valid_gaps:
            gxs, gys = zip(*valid_gaps)
            axes[1].plot(gxs, gys, label=label, **sty, markersize=5, linewidth=1.5)
        axes[2].plot(xs, times, label=label, **sty, markersize=5, linewidth=1.5)

    for ax, (ylabel, title, logy) in zip(axes, [
        ("Objective value", "6-Food Families — Objective", False),
        ("MIP gap (%)",     "6-Food Families — MIP Gap",   True),
        ("Solve time (s)",  "6-Food Families — Solve Time", True),
    ]):
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Variables")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig_rotation_6food.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8: Rotation – 27-food crops (objective + MIP gap + time)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_rotation_27food(runs_200s, runs_1200s, runs_60s):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for runs, label in ((runs_60s, "60s"), (runs_200s, "200s"), (runs_1200s, "1200s")):
        pts = [(r["n_vars"], r["objective"], r["mip_gap"], r["solve_time"])
               for r in runs if r["n_foods"] == 27 and r["objective"] is not None]
        if not pts:
            continue
        pts.sort()
        xs, objs, gaps, times = zip(*pts)
        sty = TIMEOUT_STYLES.get(label, {"color": "#555", "ls": "-", "marker": "o"})
        axes[0].plot(xs, objs, label=label, **sty, markersize=5, linewidth=1.5)
        valid_gaps = [(x, g) for x, g in zip(xs, gaps) if g is not None]
        if valid_gaps:
            gxs, gys = zip(*valid_gaps)
            if gys:
                axes[1].plot(gxs, gys, label=label, **sty, markersize=5, linewidth=1.5)
        axes[2].plot(xs, times, label=label, **sty, markersize=5, linewidth=1.5)

    for ax, (ylabel, title, logy) in zip(axes, [
        ("Objective value",  "27-Crop Rotation — Objective",  False),
        ("MIP gap (%)",      "27-Crop Rotation — MIP Gap",    True),
        ("Solve time (s)",   "27-Crop Rotation — Solve Time", True),
    ]):
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Variables")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig_rotation_27food.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 9: Overview panel (A time, A quality, B time, rotation time)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_overview(solver_data, runs_200s, runs_1200s):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # ── top-left: Variant A wall time ──
    ax = axes[0, 0]
    series = defaultdict(lambda: ([], []))
    for r in solver_data:
        if r["variant"] != "A":
            continue
        key = _solver_key(r)
        series[key][0].append(r["n_farms"])
        series[key][1].append(r["wall_time"])
    for key in sorted(series):
        _plot_series(ax, *series[key], key)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Farms"); ax.set_ylabel("Time (s)")
    ax.set_title("Variant A — Wall Time")
    ax.legend(fontsize=7, framealpha=0.9); ax.grid(True, which="both", alpha=0.3)

    # ── top-centre: Variant A quality ──
    ax = axes[0, 1]
    series = defaultdict(lambda: ([], []))
    for r in solver_data:
        if r["variant"] != "A" or r.get("objective") is None:
            continue
        key = _solver_key(r)
        series[key][0].append(r["n_farms"])
        series[key][1].append(abs(r["objective"]))
    for key in sorted(series):
        _plot_series(ax, *series[key], key)
    ax.set_xscale("log")
    ax.set_xlabel("Farms"); ax.set_ylabel("Objective")
    ax.set_title("Variant A — Solution Quality")
    ax.legend(fontsize=7, framealpha=0.9); ax.grid(True, which="both", alpha=0.3)

    # ── top-right: Variant A quality ratio ──
    ax = axes[0, 2]
    full_obj = {r["n_farms"]: abs(r["objective"])
                for r in solver_data
                if r["variant"] == "A" and r["decomposition"] == "none"
                and r.get("objective") is not None}
    seen_keys = set()
    for r in solver_data:
        if r["variant"] != "A" or r["decomposition"] == "none" or r.get("objective") is None:
            continue
        key = _solver_key(r)
        nf = r["n_farms"]
        if nf in full_obj and full_obj[nf] > 0:
            ratio = abs(r["objective"]) / full_obj[nf]
            sty = _style(key)
            label = _display_label(key) if key not in seen_keys else None
            seen_keys.add(key)
            ax.scatter(nf, ratio, color=sty["color"], marker=sty["marker"], s=30, label=label)
    ax.axhline(1.0, color="red", ls=":", lw=1.2, label="Parity")
    ax.set_xscale("log")
    ax.set_xlabel("Farms"); ax.set_ylabel("Obj ratio (decomposed/full)")
    ax.set_title("Variant A — Quality Ratio")
    ax.legend(fontsize=7, framealpha=0.9); ax.grid(True, which="both", alpha=0.3)

    # ── bottom-left: Variant B wall time ──
    ax = axes[1, 0]
    series = defaultdict(lambda: ([], []))
    for r in solver_data:
        if r["variant"] != "B" or r.get("status") == "SKIPPED" or r.get("wall_time", 0) == 0:
            continue
        key = _solver_key(r)
        series[key][0].append(r["n_farms"])
        series[key][1].append(r["wall_time"])
    for key in sorted(series):
        _plot_series(ax, *series[key], key)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Farms"); ax.set_ylabel("Time (s)")
    ax.set_title("Variant B — Wall Time")
    ax.legend(fontsize=7, framealpha=0.9); ax.grid(True, which="both", alpha=0.3)

    # ── bottom-centre: Rotation 6-food MIP gap ──
    ax = axes[1, 1]
    for runs, label in ((runs_200s, "200s"), (runs_1200s, "1200s")):
        pts = [(r["n_vars"], r["mip_gap"]) for r in runs
               if r["n_foods"] == 6 and r.get("mip_gap") is not None]
        if pts:
            pts.sort()
            xs, ys = zip(*pts)
            sty = TIMEOUT_STYLES.get(label, {})
            ax.plot(xs, ys, label=label, markersize=5, linewidth=1.5, **sty)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Variables"); ax.set_ylabel("MIP gap (%)")
    ax.set_title("6-Food Rotation — MIP Gap")
    ax.legend(fontsize=8, framealpha=0.9); ax.grid(True, which="both", alpha=0.3)

    # ── bottom-right: Rotation 27-food objective ──
    ax = axes[1, 2]
    for runs, label in ((runs_200s, "200s"), (runs_1200s, "1200s")):
        pts = [(r["n_vars"], r["objective"]) for r in runs
               if r["n_foods"] == 27 and r.get("objective") is not None]
        if pts:
            pts.sort()
            xs, ys = zip(*pts)
            sty = TIMEOUT_STYLES.get(label, {})
            ax.plot(xs, ys, label=label, markersize=5, linewidth=1.5, **sty)
    ax.set_xscale("log")
    ax.set_xlabel("Variables"); ax.set_ylabel("Objective value")
    ax.set_title("27-Crop Rotation — Objective")
    ax.legend(fontsize=8, framealpha=0.9); ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Benchmark Results Overview", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "fig_overview.pdf")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data …")

    # Prefer hard mode if available, otherwise use soft/default
    hard_data = load_solver_comparison("hard")
    soft_data = load_solver_comparison("soft")
    solver_data = hard_data if hard_data else soft_data
    mode_label = "hard" if hard_data else "soft"
    print(f"  Using solver comparison mode: {mode_label}")

    decomp_data = load_decomp_scaling()

    # Rotation data – three timeout budgets
    runs_200s_unified  = load_gurobi_unified(
        HERE / "gurobi", "200s (unified)")
    runs_1200s_timeout = load_gurobi_timeout_test(
        TIMEOUT_DIR / "gurobi_timeout_test_20260329_163901.json", "1200s")
    runs_200s_timeout  = load_gurobi_timeout_test(
        TIMEOUT_DIR / "gurobi_timeout_test_20260401_154909.json", "200s")
    runs_60s           = load_gurobi_baseline_60s()

    # Merge 200s sources — prefer the timeout test (20 scenarios) over the unified (13)
    runs_200s = runs_200s_timeout if runs_200s_timeout else runs_200s_unified
    runs_1200s = runs_1200s_timeout

    print("\nGenerating plots …")
    fig_A_solve_time(solver_data)
    fig_A_objective(solver_data)
    fig_A_quality_ratio(solver_data)
    fig_A_healed_objective(solver_data)
    fig_A_violations(solver_data)
    fig_B_solve_time(solver_data)
    fig_B_objective(solver_data)
    if decomp_data:
        fig_decomp_overhead(decomp_data)
    else:
        print("  SKIP fig_decomp_overhead (no data)")
    fig_rotation_6food(runs_200s, runs_1200s, runs_60s)
    fig_rotation_27food(runs_200s, runs_1200s, runs_60s)
    fig_overview(solver_data, runs_200s, runs_1200s)

    print(f"\nAll plots written to {OUT_DIR}")


if __name__ == "__main__":
    main()
