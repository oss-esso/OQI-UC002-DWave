r"""
Create clear scientific plots from verification_report_*.json files.

Usage (PowerShell/CMD):
    python .\plots.py

Dependencies:
  pip install pandas matplotlib seaborn numpy
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

ROOT = Path(r"c:\Users\Dean\Programming\OQI2\OQI-UC002-DWave")
REPORT_GLOB = "verification_report_*.json"
OUT_DIR = ROOT / "Plots"
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def load_reports(root: Path):
    rows = []
    for p in sorted(root.glob(REPORT_GLOB)):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        m = j.get("manifest", {})
        timing = j.get("timing", {})
        pulp_ver = j.get("pulp_verification", {})
        dwave_ver = j.get("dwave_verification", {})
        dwave_t = timing.get("dwave", {}) if isinstance(timing, dict) else {}
        # Infer problem size from manifest lists
        farms = m.get("farms") or []
        foods = m.get("foods") or []
        n_farms = len(farms)
        n_foods = len(foods)
        problem_size = n_farms * n_foods if n_farms and n_foods else np.nan

        # Convert D-Wave sub-times to seconds where needed
        qpu_access_ms = (dwave_t.get("qpu_access_time_ms", np.nan)
                         if isinstance(dwave_t, dict) else np.nan)
        charge_ms = (dwave_t.get("charge_time_ms", np.nan)
                     if isinstance(dwave_t, dict) else np.nan)
        qpu_access_s = (
            qpu_access_ms / 1000.0) if pd.notna(qpu_access_ms) else np.nan
        charge_s = (charge_ms / 1000.0) if pd.notna(charge_ms) else np.nan
        run_s = dwave_t.get("run_time_seconds", np.nan)

        rows.append({
            "file": p.name,
            "scenario": m.get("scenario", p.stem),
            "timestamp": m.get("timestamp"),
            "constraints_path": m.get("constraints_path"),
            "pulp_path": m.get("pulp_path"),
            "pulp_status": m.get("pulp_status"),
            "pulp_objective": m.get("pulp_objective"),
            "dwave_objective": dwave_ver.get("objective") if dwave_ver else m.get("pulp_objective"),
            "pulp_time_s": timing.get("pulp_solve_time_seconds", np.nan),
            "dwave_run_s": run_s,
            "dwave_qpu_access_s": qpu_access_s,
            "dwave_charge_s": charge_s,
            "dwave_feasible_count": m.get("dwave_feasible_count", np.nan),
            "dwave_total_count": m.get("dwave_total_count", np.nan),
            "solutions_match": j.get("comparison", {}).get("solutions_match"),
            "objectives_match": j.get("comparison", {}).get("objectives_match"),
            "n_farms": n_farms,
            "n_foods": n_foods,
            "problem_size": problem_size,
        })
    return pd.DataFrame(rows)


def prep_df(df: pd.DataFrame):
    df = df.copy()
    df["objective_diff_abs"] = (
        df["pulp_objective"] - df["dwave_objective"]).abs()
    df["objective_diff_rel"] = df["objective_diff_abs"] / \
        (df["pulp_objective"].replace(0, np.nan)).abs()
    # Total wall-clock like measure for D-Wave components (optional)
    if {"dwave_run_s", "dwave_qpu_access_s", "dwave_charge_s"}.issubset(df.columns):
        df["dwave_total_wall_s"] = df[[
            "dwave_run_s", "dwave_qpu_access_s", "dwave_charge_s"
        ]].sum(axis=1, min_count=1)

    # Prefer ordering by problem size if present
    sort_cols = [c for c in ["problem_size",
                             "scenario", "timestamp"] if c in df.columns]
    df = df.sort_values(sort_cols)
    return df


def plot_objectives(df: pd.DataFrame):
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df["pulp_objective"],
            width, label="PuLP", color="#2b83ba")
    plt.bar(x + width/2, df["dwave_objective"],
            width, label="D-Wave", color="#abdda4")
    plt.xticks(x, list(df["scenario"].astype(str)), rotation=45, ha="right")
    plt.ylabel("Objective")
    plt.title("Objective: PuLP vs D-Wave")
    for j, r in enumerate(df.itertuples(index=False)):
        plt.annotate(f"{getattr(r, 'pulp_objective'):.6g}", (x[j]-width/2, getattr(r, "pulp_objective")),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        plt.annotate(f"{getattr(r, 'dwave_objective'):.6g}", (x[j]+width/2, getattr(r, "dwave_objective")),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "objectives_bar.png")
    plt.close()


def plot_obj_scatter(df: pd.DataFrame):
    plt.figure(figsize=(5.5, 5))
    plt.scatter(df["pulp_objective"], df["dwave_objective"],
                s=80, color="#fdae61", edgecolor="k")
    lims = [
        min(df["pulp_objective"].min(), df["dwave_objective"].min())*0.95,
        max(df["pulp_objective"].max(), df["dwave_objective"].max())*1.05
    ]
    plt.plot(lims, lims, ls="--", color="gray")
    for i, r in df.iterrows():
        plt.annotate(r["scenario"], (r["pulp_objective"], r["dwave_objective"]), xytext=(
            6, -4), textcoords="offset points", fontsize=9)
    plt.xlabel("PuLP objective")
    plt.ylabel("D-Wave objective")
    plt.title("Objective Agreement (1:1 line shown)")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR / "objectives_scatter.png")
    plt.close()


def plot_times(df: pd.DataFrame):
    plt.figure(figsize=(8, 4.2))
    x = np.arange(len(df))
    bar_width = 0.6
    pulp_times = df["pulp_time_s"].fillna(0)
    dwave_times = df["dwave_run_s"].fillna(0)
    plt.bar(x - 0.15, pulp_times, width=0.3,
            label="PuLP solve (s)", color="#5e4fa2")
    plt.bar(x + 0.15, dwave_times, width=0.3,
            label="D-Wave run (s)", color="#f46d43")
    plt.xticks(x, list(df["scenario"].astype(str)), rotation=45, ha="right")
    plt.ylabel("Time (s)")
    plt.title("Solver runtimes")
    for i, (pt, dt) in enumerate(zip(pulp_times, dwave_times)):
        plt.text(i-0.15, pt + max(0.001, pt*0.05),
                 f"{pt:.3g}", ha="center", fontsize=8)
        plt.text(i+0.15, dt + max(0.001, dt*0.05),
                 f"{dt:.3g}", ha="center", fontsize=8)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "runtimes_bar.png")
    plt.close()


def plot_dwave_counts(df: pd.DataFrame):
    plt.figure(figsize=(7, 4))
    x = np.arange(len(df))
    total = df["dwave_total_count"].fillna(0).astype(float)
    feasible = df["dwave_feasible_count"].fillna(0).astype(float)
    infeasible = (total - feasible).clip(lower=0)
    plt.bar(x, feasible, label="Feasible", color="#2ca25f")
    plt.bar(x, infeasible, bottom=feasible, label="Other", color="#fdae61")
    plt.xticks(x, list(df["scenario"].astype(str)), rotation=45, ha="right")
    plt.ylabel("Counts")
    plt.title("D-Wave samples: feasible vs total")
    for i, (f, t) in enumerate(zip(feasible, total)):
        pct = f / t * 100 if t > 0 else 0
        plt.text(i, t + max(0.5, t*0.02),
                 f"{int(f)}/{int(t)} ({pct:.0f}%)", ha="center", fontsize=8)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dwave_counts.png")
    plt.close()


def _invert_food_groups(groups_dict: dict) -> dict:
    """Invert {group: [foods...]} -> {food: group}."""
    food_to_group = {}
    for g, lst in (groups_dict or {}).items():
        for f in lst or []:
            food_to_group[f] = g
    return food_to_group


def load_group_areas(root: Path, df_reports: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total planted area per food group for each scenario.

    Uses the constraints file to map foods->groups and the PuLP results to sum
    planted areas across farms. Returns a long-form dataframe with columns:
    [scenario, problem_size, group, area, share].
    """
    rows = []
    for r in df_reports.itertuples(index=False):
        scenario = getattr(r, "scenario")
        problem_size = getattr(r, "problem_size", np.nan)
        constraints_path = getattr(r, "constraints_path", None)
        pulp_path = getattr(r, "pulp_path", None)
        if not constraints_path or not pulp_path:
            continue
        c_path = (root / constraints_path)
        p_path = (root / pulp_path)
        if not c_path.exists() or not p_path.exists():
            continue

        # Load food_groups mapping
        try:
            c_json = json.loads(c_path.read_text())
            food_groups = c_json.get("food_groups", {})
        except Exception:
            food_groups = {}
        food_to_group = _invert_food_groups(food_groups)

        # Load PuLP areas
        try:
            p_json = json.loads(p_path.read_text())
            areas = p_json.get("areas", {})
        except Exception:
            areas = {}

        # Sum areas per food then per group
        area_by_food = {}
        for k, v in areas.items():
            if v is None:
                continue
            try:
                # Keys are like "FarmX_Food Name"
                _, food = k.split("_", 1)
            except ValueError:
                # Unexpected key format; skip
                continue
            area_by_food[food] = area_by_food.get(food, 0.0) + float(v)

        total_area = sum(area_by_food.values())
        if total_area <= 0:
            continue

        area_by_group = {}
        for food, a in area_by_food.items():
            g = food_to_group.get(food, "Unknown")
            area_by_group[g] = area_by_group.get(g, 0.0) + a

        for g, a in sorted(area_by_group.items(), key=lambda x: -x[1]):
            share = a / total_area if total_area > 0 else np.nan
            rows.append({
                "scenario": scenario,
                "problem_size": problem_size,
                "group": g,
                "area": a,
                "share": share,
            })

    return pd.DataFrame(rows)


def plot_group_shares_stacked(df_groups: pd.DataFrame):
    """100% stacked bar chart: share of planted area by food group per scenario."""
    if df_groups.empty:
        return
    # Pivot to scenarios x groups shares
    pivot = df_groups.pivot_table(
        index="scenario", columns="group", values="share", aggfunc="mean").fillna(0)
    # Order scenarios by problem size if present
    # Create a scenario order from the average problem_size mapping
    size_map = (df_groups.groupby("scenario")[
                "problem_size"].mean().sort_values()).to_dict()
    scenario_order = sorted(pivot.index, key=lambda s: (
        size_map.get(s, np.inf), str(s)))
    pivot = pivot.loc[scenario_order]

    plt.figure(figsize=(9, 5))
    bottom = np.zeros(len(pivot), dtype=float)
    x = np.arange(len(pivot), dtype=float)
    for col in pivot.columns:
        vals = np.asarray(pivot[col].values, dtype=float)
        plt.bar(x, vals, bottom=bottom, label=col)
        bottom = bottom + vals
    plt.xticks(x, list(pivot.index), rotation=45, ha="right")
    plt.ylabel("Share of planted area")
    plt.title("Food group shares by scenario (PuLP areas)")
    plt.ylim(0, 1)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "group_shares_stacked.png", dpi=200)
    plt.close()


def plot_group_heatmap(df_groups: pd.DataFrame):
    """Heatmap of group area shares by scenario with sensible proportions.

    - Auto-scales figure size based on number of scenarios (rows) and groups (cols)
    - Uses smaller annotation/font sizes to avoid overlap when many scenarios exist
    - Fixes color range to [0,1] for comparable proportions across runs
    """
    if df_groups.empty:
        return
    pivot = df_groups.pivot_table(
        index="scenario", columns="group", values="share", aggfunc="mean").fillna(0)
    size_map = (df_groups.groupby("scenario")[
                "problem_size"].mean().sort_values()).to_dict()
    scenario_order = sorted(pivot.index, key=lambda s: (
        size_map.get(s, np.inf), str(s)))
    pivot = pivot.loc[scenario_order]

    # Compute figure size from grid size (cell_w x cell_h inches per cell)
    n_rows, n_cols = pivot.shape
    cell_w = 0.7
    cell_h = 0.42
    fig_w = max(7, cell_w * n_cols + 2.0)
    fig_h = max(4.5, cell_h * n_rows + 1.8)

    # Temporarily reduce font scaling for this plot to keep labels readable
    with sns.plotting_context("notebook", font_scale=0.9):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            annot_kws={"fontsize": 8},
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "share"},
        )
        ax.set_title("Food group share heatmap (PuLP areas)", fontsize=12)
        ax.set_xlabel("Food group", fontsize=11)
        ax.set_ylabel("Scenario", fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "group_shares_heatmap.png", dpi=200)
        plt.close(fig)


def plot_group_heatmap_log(df_groups: pd.DataFrame):
    """Heatmap of group shares with logarithmic color scaling to highlight
    small-but-nonzero differences. Zeros are mapped to a small epsilon so they
    remain visible rather than blank.
    """
    if df_groups.empty:
        return
    pivot = df_groups.pivot_table(
        index="scenario", columns="group", values="share", aggfunc="mean").fillna(0)
    size_map = (df_groups.groupby("scenario")[
                "problem_size"].mean().sort_values()).to_dict()
    scenario_order = sorted(pivot.index, key=lambda s: (
        size_map.get(s, np.inf), str(s)))
    pivot = pivot.loc[scenario_order]

    # Determine epsilon from data to keep zeros visible but very light
    flat = pivot.values.flatten()
    nonzero = flat[flat > 0]
    eps = float(max(1e-4, np.min(nonzero)/5.0)) if nonzero.size else 1e-4
    pivot_log = pivot.replace(0, eps)

    # Figure size proportional to grid size
    n_rows, n_cols = pivot_log.shape
    cell_w = 0.7
    cell_h = 0.42
    fig_w = max(7, cell_w * n_cols + 2.0)
    fig_h = max(4.5, cell_h * n_rows + 1.8)

    with sns.plotting_context("notebook", font_scale=0.9):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            pivot_log,
            ax=ax,
            annot=True,
            annot_kws={"fontsize": 8},
            fmt=".2f",
            cmap="YlGnBu",
            norm=mcolors.LogNorm(vmin=eps, vmax=1.0),
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "share (log scale)"},
        )
        ax.set_title(
            "Food group share heatmap (log scale, PuLP areas)", fontsize=12)
        ax.set_xlabel("Food group", fontsize=11)
        ax.set_ylabel("Scenario", fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "group_shares_heatmap_log.png", dpi=200)
        plt.close(fig)


def summary_csv(df: pd.DataFrame):
    out = OUT_DIR / "verification_summary.csv"
    df.to_csv(out, index=False)
    return out


def plot_times_vs_size_line(df: pd.DataFrame, log_scale: bool = False):
    """Line plot of solver times vs. problem size.

    Shows PuLP solve time, D-Wave run time, QPU access, and charge, each as a
    line across increasing problem sizes (farms × foods). Also saves a log-scale
    version to help when times differ by orders of magnitude.
    """
    # Build a long-form dataframe for plotting
    metric_map = {
        "pulp_time_s": "PuLP solve (s)",
        "dwave_run_s": "D-Wave run (s)",
        "dwave_qpu_access_s": "QPU access (s)",
        "dwave_charge_s": "Charge (s)",
        # Uncomment to include total:
        # "dwave_total_wall_s": "D-Wave total (s)",
    }

    cols_present = [c for c in metric_map.keys() if c in df.columns]
    if not cols_present:
        return

    plot_df = (
        df[["scenario", "problem_size", *cols_present]]
        .melt(id_vars=["scenario", "problem_size"], var_name="metric", value_name="seconds")
        .replace({"metric": metric_map})
        .dropna(subset=["seconds", "problem_size"])
        .sort_values(["problem_size", "metric"])
    )

    plt.figure(figsize=(9, 5.2))
    sns.set_theme(style="ticks", context="talk")
    palette = {
        "PuLP solve (s)": "#5e4fa2",
        "D-Wave run (s)": "#f46d43",
        "QPU access (s)": "#3288bd",
        "Charge (s)": "#66c2a5",
        "D-Wave total (s)": "#9e0142",
    }

    ax = sns.lineplot(
        data=plot_df,
        x="problem_size",
        y="seconds",
        hue="metric",
        style="metric",
        markers=True,
        dashes=False,
        palette=palette,
        linewidth=2.4,
        markersize=8,
    )

    ax.set_xlabel("Problem size (farms × foods)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Solve time vs problem size" +
                 (" (log scale)" if log_scale else ""))
    if log_scale:
        ax.set_yscale("log")

    # Integer x ticks
    sizes = sorted(df["problem_size"].dropna().unique())
    try:
        sizes = [int(s) for s in sizes]
    except Exception:
        pass
    if sizes:
        ax.set_xticks(sizes)
    ax.grid(True, which="both", axis="both", alpha=0.25)
    ax.legend(title=None, loc="center left", bbox_to_anchor=(1.02, 0.5))

    # Add scenario names on a secondary (top) x-axis to avoid overlapping point labels
    # If multiple scenarios share a size, join their names with " / ".
    if sizes:
        names_by_size = (
            df.dropna(subset=["problem_size"])
              .groupby("problem_size")["scenario"]
              .apply(lambda s: " / ".join(sorted({str(x) for x in s})))
              .to_dict()
        )
        top_labels = [names_by_size.get(s, "") for s in sizes]
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(sizes)
        ax_top.set_xticklabels(top_labels, rotation=0, fontsize=9)
        ax_top.set_xlabel("Scenario(s)")

    plt.tight_layout()
    fname = "times_vs_size_line_log.png" if log_scale else "times_vs_size_line.png"
    plt.savefig(OUT_DIR / fname, dpi=200)
    plt.close()


def main():
    df = load_reports(ROOT)
    if df.empty:
        print("No verification_report_*.json files found in", ROOT)
        return
    df = prep_df(df)
    plot_objectives(df)
    plot_obj_scatter(df)
    plot_times(df)
    plot_dwave_counts(df)
    # New line plots vs problem size (linear and log-scale)
    plot_times_vs_size_line(df, log_scale=False)
    plot_times_vs_size_line(df, log_scale=True)
    csv = summary_csv(df)
    # New: compute and plot food group shares from PuLP areas + constraints
    df_groups = load_group_areas(ROOT, df)
    if not df_groups.empty:
        plot_group_shares_stacked(df_groups)
        plot_group_heatmap(df_groups)
        plot_group_heatmap_log(df_groups)
    print("Plots written to:", OUT_DIR)
    print("Summary CSV:", csv)


if __name__ == "__main__":
    main()
