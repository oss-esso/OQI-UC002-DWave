#!/usr/bin/env python3
"""
Generate summary tables (terminal + LaTeX) for all benchmark results.

Reads the same data sources as plot_benchmark_results.py.
Writes .tex files to Benchmarks/decomposition_scaling/tables/
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

HERE = Path(__file__).parent
DECOMP_DIR = HERE / "Benchmarks" / "decomposition_scaling"
TIMEOUT_DIR = HERE / "@todo" / "gurobi_timeout_verification"
TABLES_DIR = DECOMP_DIR / "tables"


# ── data loaders (shared with plot_benchmark_results.py) ─────────────────────

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def load_solver_comparison(mode="soft"):
    preferred = DECOMP_DIR / f"solver_comparison_results_{mode}.json"
    fallback = DECOMP_DIR / "solver_comparison_results.json"
    path = preferred if preferred.exists() else fallback
    if not path.exists():
        return []
    return _load_json(path)


def load_decomp_scaling():
    p = DECOMP_DIR / "decomposition_scaling_results.json"
    return _load_json(p) if p.exists() else []


def _parse_rotation_runs(path, label, is_unified=False):
    if not path.exists():
        return []
    raw = _load_json(path)
    runs = raw.get("runs", raw) if isinstance(raw, dict) else raw
    out = []
    for entry in runs:
        if is_unified:
            r = entry
            scenario = r.get("scenario_name", "?")
            nf = r["n_farms"]
            nfoods = r.get("n_foods", 6)
            nv = r["n_vars"]
            status = r["status"]
            obj = r.get("objective_miqp")
            mg = r.get("mip_gap")
            t = r.get("timing", {}).get("total_wall_time", 0.0)
        else:
            m = entry.get("metadata", {})
            r = entry.get("result", {})
            scenario = m.get("scenario", "?")
            nf = m.get("n_farms", 0)
            nfoods = m.get("n_foods", 6)
            nv = r.get("n_vars", 0)
            status = r.get("stopped_reason", "?")
            obj = r.get("objective_value")
            mg = r.get("mip_gap")
            t = r.get("solve_time", 0.0)

        if mg is not None:
            try:
                mg = float(mg)
                if not np.isfinite(mg) or mg > 1000:
                    mg = None
            except (ValueError, TypeError):
                mg = None

        out.append({
            "scenario": scenario,
            "n_farms": nf,
            "n_foods": nfoods,
            "n_vars": nv,
            "status": status,
            "objective": obj,
            "mip_gap": mg,
            "solve_time": t,
            "label": label,
        })
    return out


# ── formatting helpers ────────────────────────────────────────────────────────

def _f(v, fmt=".4f", na="—"):
    if v is None:
        return na
    try:
        return format(float(v), fmt)
    except (ValueError, TypeError):
        return str(v)


def _pct(v, na="—"):
    return _f(v, ".1f") + "\\%" if v is not None else na


def _tex_row(*cells, sep=" & ", end=" \\\\"):
    return sep.join(str(c) for c in cells) + end


def _rule():
    return "\\midrule"


def _header_row(*cols):
    return " & ".join(f"\\textbf{{{c}}}" for c in cols) + " \\\\ \\midrule"


def _table_env(caption, label, cols, rows, note=""):
    body = "\n    ".join(rows)
    note_line = f"\n    \\smallskip\\noindent{{\\footnotesize {note}}}" if note else ""
    return (
        f"\\begin{{table}}[htbp]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{cols}}}\n"
        f"\\toprule\n"
        f"    {body}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"{note_line}\n"
        f"\\end{{table}}\n"
    )


# ── terminal printer ──────────────────────────────────────────────────────────

def _print_section(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def _print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in col_widths)
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


# ═══════════════════════════════════════════════════════════════════════════════
# Table 1: Variant A — full solver comparison summary
# ═══════════════════════════════════════════════════════════════════════════════

def table_variant_A(data, mode="soft"):
    _print_section("Table 1 — Variant A: Full Solver Comparison")

    headers = ["Solver", "Decomp", "Farms", "Vars", "Obj", "Time (s)"]
    rows = []
    for r in sorted(
        [x for x in data if x["variant"] == "A"],
        key=lambda x: (x["decomposition"], x["n_farms"])
    ):
        rows.append([
            r["solver"],
            r["decomposition"],
            r["n_farms"],
            r["n_vars"],
            _f(r.get("objective")),
            _f(r.get("wall_time"), ".3f"),
        ])
    _print_table(headers, rows)

    # LaTeX version
    tex_rows = [_header_row(*headers)]
    for r in sorted(
        [x for x in data if x["variant"] == "A"],
        key=lambda x: (x["decomposition"], x["n_farms"])
    ):
        tex_rows.append(_tex_row(
            r["solver"].replace("_", "\\_"),
            r["decomposition"].replace("(", "\\,(").replace("_", "\\_"),
            r["n_farms"], r["n_vars"],
            _f(r.get("objective")),
            _f(r.get("wall_time"), ".3f"),
        ))
    return _table_env(
        caption=f"Variant A solver comparison ({mode} mode, 200\\,s timeout).",
        label="tab:variant_A_full",
        cols="llrrrr",
        rows=tex_rows,
        note="Objectives are maximisation values; higher is better.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Table 2: Variant A — quality and speedup summary (pivot at each farm size)
# ═══════════════════════════════════════════════════════════════════════════════

def table_variant_A_quality(data):
    _print_section("Table 2 — Variant A: Quality Ratio at Selected Farm Sizes")

    # Build: n_farms -> {key -> (obj, time)}
    full_obj = {}
    full_time = {}
    decomp = defaultdict(dict)
    for r in data:
        if r["variant"] != "A":
            continue
        nf = r["n_farms"]
        if r["decomposition"] == "none":
            full_obj[nf] = r.get("objective")
            full_time[nf] = r.get("wall_time")
        else:
            k = r["decomposition"]
            decomp[nf][k] = (r.get("objective"), r.get("wall_time"))

    SHOW_FARMS = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    SHOW_DECOMPS = ["PlotBased", "Multilevel(5)", "HybridGrid(5,9)"]

    headers = ["Farms", "Full obj", "Full t (s)"] + [
        f"{d} obj" for d in SHOW_DECOMPS
    ] + [f"{d} t (s)" for d in SHOW_DECOMPS]

    rows = []
    for nf in SHOW_FARMS:
        if nf not in full_obj:
            continue
        fo = full_obj[nf]
        ft = full_time[nf]
        row = [nf, _f(fo), _f(ft, ".3f")]
        for d in SHOW_DECOMPS:
            row.append(_f(decomp[nf].get(d, (None, None))[0]))
        for d in SHOW_DECOMPS:
            row.append(_f(decomp[nf].get(d, (None, None))[1], ".3f"))
        rows.append(row)
    _print_table(headers, rows)

    # Concise LaTeX showing ratio and speedup
    tex_headers = ["Farms", "Full obj"] + [f"{d}\\newline ratio" for d in SHOW_DECOMPS]
    tex_rows = [_header_row(*tex_headers)]
    for nf in SHOW_FARMS:
        if nf not in full_obj:
            continue
        fo = full_obj.get(nf)
        row_vals = [nf, _f(fo)]
        for d in SHOW_DECOMPS:
            do = decomp[nf].get(d, (None,))[0]
            if fo and do is not None and fo > 0:
                ratio = float(do) / float(fo)
                row_vals.append(f"{ratio:.2f}×")
            else:
                row_vals.append("—")
        tex_rows.append(_tex_row(*row_vals))

    return _table_env(
        caption="Variant A: decomposed objective as fraction of full-MIQP objective.",
        label="tab:variant_A_quality",
        cols="r" + "r" * (1 + len(SHOW_DECOMPS)),
        rows=tex_rows,
        note="Ratio $>1$ means decomposed finds larger (better) objective than joint MIQP.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Table 3: Variant B — solver comparison summary
# ═══════════════════════════════════════════════════════════════════════════════

def table_variant_B(data, mode="soft"):
    _print_section("Table 3 — Variant B: Rotation Solver Comparison")

    headers = ["Solver", "Decomp", "Farms", "Vars", "Obj", "Time (s)", "Parts", "Status"]
    rows = []
    for r in sorted(
        [x for x in data if x["variant"] == "B"],
        key=lambda x: (x["decomposition"], x["n_farms"])
    ):
        status = r.get("status", "-")
        if status == "SKIPPED":
            continue
        rows.append([
            r["solver"],
            r["decomposition"],
            r["n_farms"],
            r["n_vars"],
            _f(r.get("objective")),
            _f(r.get("wall_time"), ".2f"),
            r.get("n_partitions", "—"),
            status,
        ])
    _print_table(headers, rows)

    tex_rows = [_header_row("Solver", "Decomp", "Farms", "Vars", "Obj", "Time (s)", "Parts")]
    for r in sorted(
        [x for x in data if x["variant"] == "B"],
        key=lambda x: (x["decomposition"], x["n_farms"])
    ):
        status = r.get("status", "-")
        if status == "SKIPPED":
            continue
        tex_rows.append(_tex_row(
            r["solver"].replace("_", "\\_"),
            r["decomposition"].replace("(", "\\,(").replace("_", "\\_"),
            r["n_farms"], r["n_vars"],
            _f(r.get("objective")),
            _f(r.get("wall_time"), ".2f"),
            r.get("n_partitions", "—"),
        ))
    return _table_env(
        caption=f"Variant B solver comparison ({mode} mode, 600\\,s timeout, farms~$\\le$2000).",
        label="tab:variant_B_full",
        cols="llrrrrr",
        rows=tex_rows,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Table 4: Rotation scenarios — Gurobi scaling summary
# ═══════════════════════════════════════════════════════════════════════════════

def table_rotation_scaling(all_runs_by_label):
    _print_section("Table 4 — Rotation Scenarios: Gurobi Scaling")

    all_scenarios = sorted(set(
        r["scenario"] for runs in all_runs_by_label.values() for r in runs
    ), key=lambda s: (next((r["n_vars"] for runs in all_runs_by_label.values()
                            for r in runs if r["scenario"] == s), 0), s))

    labels = list(all_runs_by_label)
    headers = ["Scenario", "Farms", "Foods", "Vars"] + [f"Obj ({l})" for l in labels] + [f"Gap ({l})" for l in labels]

    # Build lookup
    lookup = {}
    for label, runs in all_runs_by_label.items():
        for r in runs:
            lookup[(label, r["scenario"])] = r

    rows = []
    for s in all_scenarios:
        first = next((r for runs in all_runs_by_label.values() for r in runs if r["scenario"] == s), None)
        if first is None:
            continue
        row = [s, first["n_farms"], first["n_foods"], first["n_vars"]]
        for l in labels:
            r = lookup.get((l, s))
            row.append(_f(r["objective"] if r else None))
        for l in labels:
            r = lookup.get((l, s))
            row.append(_pct(r["mip_gap"] if r else None).replace("\\%", "%"))
        rows.append(row)

    _print_table(headers, rows, col_widths=[32, 6, 5, 6] + [10] * len(labels) + [8] * len(labels))

    # LaTeX (compact — objective only)
    n = len(labels)
    tex_rows = [_header_row("Scenario", "Farms", "Vars",
                             *[f"Obj ({l})" for l in labels],
                             *[f"Gap ({l})" for l in labels])]
    for s in all_scenarios:
        first = next((r for runs in all_runs_by_label.values() for r in runs if r["scenario"] == s), None)
        if first is None:
            continue
        cells = [s.replace("_", "\\_"), first["n_farms"], first["n_vars"]]
        for l in labels:
            r = lookup.get((l, s))
            cells.append(_f(r["objective"] if r else None, ".2f"))
        for l in labels:
            r = lookup.get((l, s))
            cells.append(_pct(r["mip_gap"] if r else None))
        tex_rows.append(_tex_row(*cells))

    return _table_env(
        caption="Rotation scenario Gurobi results at three timeout budgets.",
        label="tab:rotation_scaling",
        cols="l" + "r" * (2 + 2 * n),
        rows=tex_rows,
        note="Gaps $>100$\\,\\% indicate the solver found only a weak lower bound.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Table 5: Decomposition overhead summary (selected methods)
# ═══════════════════════════════════════════════════════════════════════════════

def table_decomp_overhead(decomp_data):
    _print_section("Table 5 — Decomposition Overhead (selected methods)")

    SHOW = [("A", "PlotBased"), ("A", "Multilevel(5)"), ("A", "HybridGrid(5,9)"),
            ("B", "Clique(farm-by-farm)"), ("B", "SpatialTemporal(5)")]

    lookup = defaultdict(dict)
    for r in decomp_data:
        lookup[(r["variant"], r["method"])][r["n_farms"]] = r

    SHOW_FARMS = [10, 50, 200, 1000, 5000, 10000]
    headers = ["Variant", "Method"] + [f"{n} farms" for n in SHOW_FARMS]

    rows = []
    seen = set()
    for vt, method in SHOW:
        d = lookup.get((vt, method), {})
        if not d:
            # Try alternate names (e.g. "Clique" vs "Clique(farm-by-farm)")
            for (v2, m2), dd in lookup.items():
                base = method.split("(")[0]
                if v2 == vt and m2.startswith(base) and (vt, m2) not in seen:
                    d = dd
                    method = m2
                    break
        if not d or (vt, method) in seen:
            continue
        seen.add((vt, method))
        row = [vt, method]
        for nf in SHOW_FARMS:
            entry = d.get(nf)
            row.append(_f(entry["decomposition_time_s"] if entry else None, ".4f"))
        rows.append(row)

    _print_table(headers, rows)

    tex_rows = [_header_row("Var.", "Method", *[f"{n}" for n in SHOW_FARMS])]
    for row in rows:
        tex_rows.append(_tex_row(*[str(c).replace("(", "\\,(").replace("_", "\\_") for c in row]))
    return _table_env(
        caption="Decomposition overhead in seconds for selected methods and farm sizes.",
        label="tab:decomp_overhead",
        cols="ll" + "r" * len(SHOW_FARMS),
        rows=tex_rows,
        note="Times are for the partitioning step only, excluding solver time.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Table 6: Study 2.B three-group comparison (27-food, Variant B)
# ═══════════════════════════════════════════════════════════════════════════════

def table_study2b_three_group(solver_data):
    _print_section("Table 6 — Study 2.B: Three-Group Comparison (27-crop, Variant B)")

    # Gurobi full: from solver_comparison
    gf_rows = {
        r["n_farms"]: r
        for r in solver_data
        if r.get("variant") == "B" and r.get("decomposition") == "none"
        and r.get("status") != "SKIPPED"
    }

    # Gurobi decomposed Clique / SpatialTemporal
    decomp_rows = {}
    for r in solver_data:
        if r.get("variant") != "B" or r.get("status") == "SKIPPED":
            continue
        if r.get("solver") == "Gurobi_decomposed":
            key = (r["n_farms"], r["decomposition"])
            decomp_rows[key] = r

    # QPU data
    qpu_path = HERE / "qpu_hier_repaired.json"
    qpu_27 = {}
    if qpu_path.exists():
        with open(qpu_path) as f:
            raw = json.load(f)
        runs = raw.get("runs", raw) if isinstance(raw, dict) else raw
        for r in runs:
            if r.get("n_foods") == 27:
                nf = r["n_farms"]
                timing = r.get("timing", {})
                qpu_27[nf] = {
                    "wall_time": timing.get("total_wall_time", 0.0),
                    "benefit": -(r.get("objective_miqp") or 0.0),
                }

    all_farms = sorted(set(list(gf_rows) + list(qpu_27)))

    headers = [
        "Farms", "Vars",
        "Gurobi full time(s)", "Gurobi full obj", "Status",
        "QPU time(s)", "QPU obj†",
        "Clique time(s)", "ST(5) time(s)",
    ]
    rows = []
    for nf in all_farms:
        gf = gf_rows.get(nf, {})
        qpu = qpu_27.get(nf, {})
        cl = decomp_rows.get((nf, "Clique"), {})
        st = decomp_rows.get((nf, "SpatialTemporal(5)"), {})
        rows.append([
            nf,
            gf.get("n_vars", "—"),
            _f(gf.get("wall_time"), ".1f"),
            _f(gf.get("objective")),
            gf.get("status", "—"),
            _f(qpu.get("wall_time"), ".1f"),
            _f(qpu.get("benefit")),
            _f(cl.get("wall_time"), ".1f"),
            _f(st.get("wall_time"), ".1f"),
        ])
    _print_table(headers, rows)

    # LaTeX
    tex_rows = [_header_row(
        "Farms", "Vars",
        "GF Time(s)", "GF Obj", "Status",
        "QPU Time(s)", "QPU Obj†",
        "Cl. Time(s)", "ST-5 Time(s)",
    )]
    for nf in all_farms:
        gf = gf_rows.get(nf, {})
        qpu = qpu_27.get(nf, {})
        cl = decomp_rows.get((nf, "Clique"), {})
        st = decomp_rows.get((nf, "SpatialTemporal(5)"), {})
        raw_status = gf.get("status", "—") if gf else "—"
        is_infeasible = not gf.get("feasible", True) if gf else False
        if raw_status == "timeout" and is_infeasible:
            status_str = "TO/IF"
        elif raw_status == "timeout":
            status_str = "TO"
        elif is_infeasible:
            status_str = "opt/IF"
        elif raw_status == "optimal":
            status_str = "optimal"
        else:
            status_str = raw_status or "—"
        tex_rows.append(_tex_row(
            nf, gf.get("n_vars", "—"),
            _f(gf.get("wall_time"), ".1f"),
            _f(gf.get("objective"), ".2f"),
            status_str,
            _f(qpu.get("wall_time"), ".1f"),
            _f(qpu.get("benefit"), ".2f"),
            _f(cl.get("wall_time"), ".1f"),
            _f(st.get("wall_time"), ".1f"),
        ))
    return _table_env(
        caption=(
            "Study 2.B three-group comparison: Gurobi full, QPU hierarchical, and "
            "Gurobi decomposed (Clique / SpatialTemporal-5) on 27-crop Variant~B.  "
            "All Gurobi runs use 600\\,s timeout.  "
            "GF=Gurobi full, TO=timeout, opt/IF=Gurobi optimal but rotation constraints violated.  "
            "$\\dagger$QPU benefit = $-$objective\\_miqp (QUBO sign-corrected); "
            "includes violation contributions, not directly comparable to Gurobi MIQP."
        ),
        label="tab:study2b_three_group",
        cols="rrrrrrrrr",
        rows=tex_rows,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    hard_data = load_solver_comparison("hard")
    soft_data = load_solver_comparison("soft")
    solver_data = hard_data if hard_data else soft_data
    mode_label = "hard" if hard_data else "soft"
    print(f"  Using solver comparison mode: {mode_label}")
    decomp_data = load_decomp_scaling()

    runs_200s = _parse_rotation_runs(
        TIMEOUT_DIR / "gurobi_timeout_test_20260331_141105.json", "200s")
    runs_1200s = _parse_rotation_runs(
        TIMEOUT_DIR / "gurobi_timeout_test_20260329_163901.json", "1200s")
    runs_60s_raw = HERE / "gurobi_baseline_60s.json"
    runs_60s = _parse_rotation_runs(runs_60s_raw, "60s", is_unified=True) if runs_60s_raw.exists() else []
    runs_200s_unified = _parse_rotation_runs(
        HERE / "benchmark_20260329_105633.json", "200s (unified)", is_unified=True)

    # Use the richer 200s
    if not runs_200s:
        runs_200s = runs_200s_unified

    print("\n--- GENERATING TABLES ---")
    tex_blocks = {}
    tex_blocks["variant_A_full"]    = table_variant_A(solver_data, mode_label)
    tex_blocks["variant_A_quality"] = table_variant_A_quality(solver_data)
    tex_blocks["variant_B_full"]    = table_variant_B(solver_data, mode_label)
    tex_blocks["rotation_scaling"]  = table_rotation_scaling({
        "200s": runs_200s, "1200s": runs_1200s, "60s": runs_60s,
    })
    if decomp_data:
        tex_blocks["decomp_overhead"] = table_decomp_overhead(decomp_data)
    tex_blocks["study2b_three_group"] = table_study2b_three_group(solver_data)

    # Write individual .tex files
    all_blocks = []
    for name, tex in tex_blocks.items():
        out = TABLES_DIR / f"table_{name}.tex"
        out.write_text(tex, encoding="utf-8")
        print(f"\n  Written {out.name}")
        all_blocks.append(tex)

    # Write combined all_tables.tex
    combined = (
        "% Auto-generated by generate_tables.py\n"
        "% Include with \\input{Benchmarks/decomposition_scaling/tables/all_tables}\n\n"
        + "\n\n".join(all_blocks)
    )
    combined_path = TABLES_DIR / "all_tables.tex"
    combined_path.write_text(combined, encoding="utf-8")
    print(f"\n  Written {combined_path.name} (all tables combined)")


if __name__ == "__main__":
    main()
