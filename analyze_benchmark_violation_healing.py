#!/usr/bin/env python3
"""
Analyze violation "healing" impact on benchmark data.

This script performs the same violation healing analysis done for the rotation
scenario on the benchmark data (small, large, and comprehensive scales).

Violation Healing Methodology:
1. Identify methods with constraint violations (infeasible solutions)
2. Estimate the "benefit lost" due to violations using average benefit per slot
3. Calculate healed (violation-adjusted) objective
4. Determine how much of the objective gap is explained by violations

Author: OQI-UC002-DWave Project
Date: 2025-01-21
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Project paths
PROJECT_ROOT = Path(__file__).parent
QPU_RESULTS_DIR = PROJECT_ROOT / "@todo" / "qpu_benchmark_results"
OUTPUT_DIR = PROJECT_ROOT / "phase3_results_plots"


@dataclass
class ViolationHealingResult:
    """Results from violation healing analysis for a single method-scale pair."""
    
    n_farms: int
    method: str
    gurobi_obj: float
    qpu_obj: float
    violations: int
    total_slots: int  # n_farms (since single-period problem)
    violation_rate: float  # violations / total_slots * 100
    avg_benefit_per_slot: float
    estimated_lost_benefit: float
    healed_obj: float  # qpu_obj - estimated_lost_benefit
    raw_gap: float  # qpu_obj - gurobi_obj
    healed_gap: float  # healed_obj - gurobi_obj
    gap_explained_pct: float  # % of gap explained by violations


def load_benchmark_files() -> dict[str, Any]:
    """Load all benchmark JSON files."""
    files = {
        "small_scale": QPU_RESULTS_DIR / "qpu_benchmark_20251201_160444.json",
        "large_scale": QPU_RESULTS_DIR / "qpu_benchmark_20251201_200012.json",
        "hybridgrid": QPU_RESULTS_DIR / "qpu_benchmark_20251203_121526.json",
        "hybridgrid_large": QPU_RESULTS_DIR / "qpu_benchmark_20251203_133144.json",
    }
    
    data = {}
    for key, path in files.items():
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data[key] = json.load(f)
            print(f"  ✓ Loaded {key}: {path.name}")
        else:
            print(f"  ✗ Missing {key}: {path}")
            data[key] = None
    
    return data


def extract_healing_data(benchmark_data: dict[str, Any]) -> list[ViolationHealingResult]:
    """Extract violation healing data from all benchmark results."""
    results: list[ViolationHealingResult] = []
    
    for scale_key in ["small_scale", "large_scale", "hybridgrid", "hybridgrid_large"]:
        if benchmark_data.get(scale_key) is None:
            continue
        
        for scale_result in benchmark_data[scale_key]["results"]:
            n_farms = scale_result["n_farms"]
            
            # Get ground truth (Gurobi)
            gt = scale_result.get("ground_truth", {})
            gurobi_obj = gt.get("objective", 0)
            
            if gurobi_obj <= 0:
                continue  # Skip invalid entries
            
            # Total slots = n_farms (single-period problem in benchmarks)
            total_slots = n_farms
            
            # Average benefit per slot (from Gurobi optimal)
            avg_benefit_per_slot = gurobi_obj / total_slots
            
            # Process each QPU method
            for method_name, method_result in scale_result.get("method_results", {}).items():
                if not method_result.get("success", False):
                    continue
                
                # Clean method name
                display_name = method_name.replace("decomposition_", "")
                
                qpu_obj = method_result.get("objective", 0)
                violations = method_result.get("violations", 0)
                
                if qpu_obj <= 0:
                    continue  # Skip invalid entries
                
                # Calculate violation rate
                violation_rate = (violations / total_slots) * 100 if total_slots > 0 else 0
                
                # Estimate lost benefit due to violations
                # Each violation = one farm slot with no crop assigned
                estimated_lost_benefit = violations * avg_benefit_per_slot
                
                # Healed objective (remove benefit from violated slots)
                healed_obj = qpu_obj - estimated_lost_benefit
                
                # Gap calculations
                raw_gap = qpu_obj - gurobi_obj
                healed_gap = healed_obj - gurobi_obj
                
                # Percentage of gap explained by violations
                if abs(raw_gap) > 1e-9:
                    gap_explained_pct = ((raw_gap - healed_gap) / abs(raw_gap)) * 100
                else:
                    gap_explained_pct = 0
                
                results.append(ViolationHealingResult(
                    n_farms=n_farms,
                    method=display_name,
                    gurobi_obj=gurobi_obj,
                    qpu_obj=qpu_obj,
                    violations=violations,
                    total_slots=total_slots,
                    violation_rate=violation_rate,
                    avg_benefit_per_slot=avg_benefit_per_slot,
                    estimated_lost_benefit=estimated_lost_benefit,
                    healed_obj=healed_obj,
                    raw_gap=raw_gap,
                    healed_gap=healed_gap,
                    gap_explained_pct=gap_explained_pct,
                ))
    
    return results


def results_to_dataframe(results: list[ViolationHealingResult]) -> pd.DataFrame:
    """Convert results to a DataFrame for analysis."""
    records = [
        {
            "n_farms": r.n_farms,
            "method": r.method,
            "gurobi_obj": r.gurobi_obj,
            "qpu_obj": r.qpu_obj,
            "violations": r.violations,
            "total_slots": r.total_slots,
            "violation_rate": r.violation_rate,
            "avg_benefit_per_slot": r.avg_benefit_per_slot,
            "estimated_lost_benefit": r.estimated_lost_benefit,
            "healed_obj": r.healed_obj,
            "raw_gap": r.raw_gap,
            "healed_gap": r.healed_gap,
            "gap_explained_pct": r.gap_explained_pct,
        }
        for r in results
    ]
    return pd.DataFrame(records)


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics for violation healing analysis."""
    print("\n" + "=" * 100)
    print("VIOLATION HEALING ANALYSIS - SUMMARY STATISTICS")
    print("=" * 100)
    
    # Overall stats
    total_entries = len(df)
    entries_with_violations = len(df[df["violations"] > 0])
    pct_with_violations = (entries_with_violations / total_entries) * 100 if total_entries > 0 else 0
    
    print(f"\nTotal method-scale entries: {total_entries}")
    print(f"Entries with violations: {entries_with_violations} ({pct_with_violations:.1f}%)")
    print(f"Entries without violations: {total_entries - entries_with_violations}")
    
    # Filter to entries with violations for healing analysis
    df_viol = df[df["violations"] > 0].copy()
    
    if len(df_viol) == 0:
        print("\n✓ No violations found in any method - all solutions are feasible!")
        return
    
    print("\n" + "-" * 100)
    print("VIOLATION STATISTICS (entries with violations)")
    print("-" * 100)
    
    print(f"\nViolation count range: {df_viol['violations'].min():.0f} - {df_viol['violations'].max():.0f}")
    print(f"Mean violations: {df_viol['violations'].mean():.2f}")
    print(f"Median violations: {df_viol['violations'].median():.2f}")
    print(f"Violation rate range: {df_viol['violation_rate'].min():.2f}% - {df_viol['violation_rate'].max():.2f}%")
    print(f"Mean violation rate: {df_viol['violation_rate'].mean():.2f}%")
    
    print("\n" + "-" * 100)
    print("GAP EXPLAINED BY VIOLATIONS")
    print("-" * 100)
    
    # Only consider entries where QPU > Gurobi (positive raw gap)
    df_positive_gap = df_viol[df_viol["raw_gap"] > 0].copy()
    
    if len(df_positive_gap) > 0:
        print(f"\nEntries where QPU obj > Gurobi obj (positive gap): {len(df_positive_gap)}")
        print(f"Mean gap explained by violations: {df_positive_gap['gap_explained_pct'].mean():.1f}%")
        print(f"Median gap explained by violations: {df_positive_gap['gap_explained_pct'].median():.1f}%")
        
        # Correlation analysis
        if len(df_positive_gap) >= 3:
            corr_viols_gap = df_positive_gap["violations"].corr(df_positive_gap["raw_gap"])
            print(f"\nCorrelation (violations vs raw_gap): {corr_viols_gap:.4f}")
        
    # Entries where QPU < Gurobi (negative gap - QPU underperforms)
    df_negative_gap = df_viol[df_viol["raw_gap"] < 0].copy()
    if len(df_negative_gap) > 0:
        print(f"\nEntries where QPU obj < Gurobi obj (negative gap): {len(df_negative_gap)}")
        print(f"  → Violations cannot explain gap when QPU is already underperforming")
    
    print("\n" + "-" * 100)
    print("BREAKDOWN BY METHOD")
    print("-" * 100)
    
    method_summary = df.groupby("method").agg({
        "n_farms": "count",
        "violations": ["sum", "mean"],
        "violation_rate": "mean",
        "raw_gap": "mean",
        "healed_gap": "mean",
        "gap_explained_pct": "mean",
    }).round(4)
    method_summary.columns = [
        "n_entries", "total_violations", "mean_violations",
        "mean_viol_rate%", "mean_raw_gap", "mean_healed_gap", "mean_gap_explained%"
    ]
    print(f"\n{method_summary.to_string()}")
    
    print("\n" + "-" * 100)
    print("BREAKDOWN BY SCALE (n_farms)")
    print("-" * 100)
    
    scale_summary = df.groupby("n_farms").agg({
        "method": "count",
        "violations": ["sum", "mean"],
        "violation_rate": "mean",
        "raw_gap": "mean",
        "gap_explained_pct": "mean",
    }).round(4)
    scale_summary.columns = [
        "n_methods", "total_violations", "mean_violations",
        "mean_viol_rate%", "mean_raw_gap", "mean_gap_explained%"
    ]
    print(f"\n{scale_summary.to_string()}")


def create_healing_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Create visualizations for violation healing analysis."""
    output_dir.mkdir(exist_ok=True)
    
    # Filter to entries with violations
    df_viol = df[df["violations"] > 0].copy()
    
    if len(df_viol) == 0:
        print("\nNo violations to visualize - all solutions are feasible!")
        return
    
    # Set up figure style
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Violation Healing Analysis: Benchmark Data",
        fontsize=14, fontweight="bold", y=0.98
    )
    
    # Color mapping for methods
    unique_methods = df_viol["method"].unique()
    cmap = plt.colormaps.get_cmap("tab10")
    method_colors = {m: cmap(i % 10) for i, m in enumerate(unique_methods)}
    
    # Panel 1: Violations vs Raw Gap
    ax = axes[0, 0]
    for method in unique_methods:
        mask = df_viol["method"] == method
        ax.scatter(
            df_viol.loc[mask, "violations"],
            df_viol.loc[mask, "raw_gap"],
            c=[method_colors[method]],
            label=method,
            alpha=0.7,
            s=60,
            edgecolors="black",
            linewidth=0.5,
        )
    
    # Add trend line if enough data
    if len(df_viol) >= 3:
        slope, intercept, r_value, _, _ = stats.linregress(
            df_viol["violations"], df_viol["raw_gap"]
        )
        x_fit = np.linspace(0, df_viol["violations"].max(), 100)
        ax.plot(
            x_fit, slope * x_fit + intercept,
            "--", color="gray", linewidth=2,
            label=f"Linear fit (r={r_value:.3f})"
        )
    
    ax.set_xlabel("Number of Violations")
    ax.set_ylabel("Raw Gap (QPU obj - Gurobi obj)")
    ax.set_title("Violations vs Objective Gap")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="red", linestyle="-", linewidth=1, alpha=0.5)
    
    # Panel 2: Violation Rate by Scale
    ax = axes[0, 1]
    scale_viols = df_viol.groupby("n_farms")["violation_rate"].mean()
    bars = ax.bar(
        range(len(scale_viols)),
        scale_viols.values,
        color="#F77F00",
        alpha=0.8,
        edgecolor="black",
    )
    ax.set_xticks(range(len(scale_viols)))
    ax.set_xticklabels([f"{n}" for n in scale_viols.index], rotation=45, ha="right")
    ax.set_xlabel("Number of Farms")
    ax.set_ylabel("Violation Rate (%)")
    ax.set_title("Average Violation Rate by Problem Scale")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels
    for bar, val in zip(bars, scale_viols.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=8
        )
    
    # Panel 3: Before/After Healing Comparison
    ax = axes[1, 0]
    
    # Group by method for comparison
    method_comparison = df_viol.groupby("method").agg({
        "raw_gap": "mean",
        "healed_gap": "mean",
    }).reset_index()
    
    x = np.arange(len(method_comparison))
    width = 0.35
    
    bars1 = ax.bar(
        x - width/2,
        method_comparison["raw_gap"],
        width,
        label="Raw Gap",
        color="#CC3311",
        alpha=0.8,
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width/2,
        method_comparison["healed_gap"],
        width,
        label="Healed Gap",
        color="#029E73",
        alpha=0.8,
        edgecolor="black",
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels(method_comparison["method"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Gap (QPU obj - Gurobi obj)")
    ax.set_title("Gap Before and After Violation Healing")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    
    # Panel 4: Gap Explained Summary
    ax = axes[1, 1]
    
    # Only positive gap entries
    df_positive = df_viol[df_viol["raw_gap"] > 0].copy()
    
    if len(df_positive) > 0:
        explained_by_method = df_positive.groupby("method")["gap_explained_pct"].mean()
        
        bars = ax.barh(
            range(len(explained_by_method)),
            explained_by_method.values,
            color="#6F42C1",
            alpha=0.8,
            edgecolor="black",
        )
        ax.set_yticks(range(len(explained_by_method)))
        ax.set_yticklabels(explained_by_method.index, fontsize=9)
        ax.set_xlabel("% of Gap Explained by Violations")
        ax.set_title("How Much Gap is Due to Violations?")
        ax.grid(True, alpha=0.3, axis="x")
        
        # Add value labels
        for bar, val in zip(bars, explained_by_method.values):
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                ha="left", va="center", fontsize=9
            )
        
        ax.set_xlim(0, min(max(explained_by_method.values) * 1.3, 110))
    else:
        ax.text(
            0.5, 0.5,
            "No entries with positive gap\n(QPU underperforms Gurobi)",
            ha="center", va="center", transform=ax.transAxes, fontsize=12
        )
        ax.set_axis_off()
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / "benchmark_violation_healing_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved: {output_path}")
    print(f"✓ Saved: {output_path.with_suffix('.pdf')}")
    plt.close()


def create_detailed_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a detailed table visualization."""
    output_dir.mkdir(exist_ok=True)
    
    # Filter to entries with violations
    df_viol = df[df["violations"] > 0].copy()
    
    if len(df_viol) == 0:
        print("\nNo violations to tabulate - all solutions are feasible!")
        return
    
    # Create summary by method
    summary = df_viol.groupby("method").agg({
        "n_farms": ["count", "min", "max"],
        "violations": ["sum", "mean"],
        "violation_rate": ["mean", "max"],
        "raw_gap": "mean",
        "healed_gap": "mean",
        "gap_explained_pct": "mean",
    })
    summary.columns = [
        "N Entries", "Min Farms", "Max Farms",
        "Total Viols", "Mean Viols",
        "Mean Viol%", "Max Viol%",
        "Mean Raw Gap", "Mean Healed Gap", "Gap Explained%"
    ]
    summary = summary.round(2)
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(16, len(summary) * 0.5 + 2))
    ax.axis("off")
    
    # Create table
    table = ax.table(
        cellText=summary.values,
        rowLabels=summary.index,
        colLabels=summary.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.08] * len(summary.columns),
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#E6E6E6")
        if col == -1:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#E6E6E6")
    
    ax.set_title(
        "Violation Healing Summary by Method",
        fontsize=14, fontweight="bold", pad=20
    )
    
    plt.tight_layout()
    
    output_path = output_dir / "benchmark_violation_healing_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_key_findings(df: pd.DataFrame) -> None:
    """Print key findings and interpretation."""
    print("\n" + "=" * 100)
    print("KEY FINDINGS AND INTERPRETATION")
    print("=" * 100)
    
    df_viol = df[df["violations"] > 0].copy()
    df_clean = df[df["violations"] == 0].copy()
    
    total = len(df)
    n_viol = len(df_viol)
    n_clean = len(df_clean)
    
    print(f"""
OVERALL FEASIBILITY:
• Total method-scale combinations: {total}
• Feasible (0 violations): {n_clean} ({n_clean/total*100:.1f}%)
• Infeasible (>0 violations): {n_viol} ({n_viol/total*100:.1f}%)
""")
    
    if n_viol == 0:
        print("""
✅ EXCELLENT NEWS: All benchmark solutions are feasible!
   No violation healing is needed - the QPU methods produce valid solutions.
""")
        return
    
    # Methods with/without violations
    methods_with_viols = df_viol["method"].unique()
    methods_without_viols = set(df["method"].unique()) - set(methods_with_viols)
    
    print(f"""
METHODS ANALYSIS:
• Methods with some violations: {', '.join(methods_with_viols)}
• Methods always feasible: {', '.join(methods_without_viols) if methods_without_viols else 'None'}
""")
    
    # Analyze gap direction
    df_positive = df_viol[df_viol["raw_gap"] > 0].copy()  # QPU > Gurobi
    df_negative = df_viol[df_viol["raw_gap"] < 0].copy()  # QPU < Gurobi
    
    print(f"""
GAP DIRECTION ANALYSIS (for infeasible entries):
• QPU obj > Gurobi obj (positive gap): {len(df_positive)} entries
• QPU obj < Gurobi obj (negative gap): {len(df_negative)} entries
• QPU obj ≈ Gurobi obj: {n_viol - len(df_positive) - len(df_negative)} entries
""")
    
    # Violation impact
    if len(df_viol) > 0:
        avg_viol_rate = df_viol["violation_rate"].mean()
        max_viol_rate = df_viol["violation_rate"].max()
        
        print(f"""
VIOLATION STATISTICS:
• Average violation rate (infeasible entries): {avg_viol_rate:.2f}%
• Maximum violation rate: {max_viol_rate:.2f}%
• Total violations across all entries: {df_viol['violations'].sum():.0f}
""")
    
    # Different interpretation based on gap direction
    if len(df_positive) > 0:
        avg_explained = df_positive["gap_explained_pct"].mean()
        print(f"""
HEALING ANALYSIS (for {len(df_positive)} entries where QPU > Gurobi):
• Average gap explained by violations: {avg_explained:.1f}%
""")
        
        if avg_explained > 50:
            print("""
🔴 SIGNIFICANT: Violations explain >50% of the positive gap!
   → QPU appears better only due to constraint violations
   → After healing, true performance is closer to Gurobi
""")
    
    if len(df_negative) > len(df_positive):
        print("""
📊 IMPORTANT OBSERVATION:
   Most infeasible entries have NEGATIVE gaps (QPU < Gurobi).
   This means:
   1. QPU is finding worse solutions than Gurobi
   2. AND these solutions have constraint violations
   
   Violations are NOT helping QPU "cheat" to higher objectives.
   Instead, violations indicate difficulty in the optimization:
   - Problems are harder for QPU to solve
   - Both solution quality AND feasibility suffer together
   
   Recommendation:
   - Focus on improving decomposition strategies
   - Consider hybrid classical-quantum approaches
   - The issue is optimization quality, not just constraint satisfaction
""")
    
    # Compare feasible vs infeasible performance
    print("\n" + "-" * 100)
    print("FEASIBLE vs INFEASIBLE COMPARISON")
    print("-" * 100)
    
    if len(df_clean) > 0 and len(df_viol) > 0:
        avg_gap_feasible = df_clean["raw_gap"].mean()
        avg_gap_infeasible = df_viol["raw_gap"].mean()
        
        print(f"""
• Mean gap (feasible solutions): {avg_gap_feasible:.4f}
• Mean gap (infeasible solutions): {avg_gap_infeasible:.4f}
• Difference: {avg_gap_feasible - avg_gap_infeasible:.4f}
""")
        
        if avg_gap_feasible > avg_gap_infeasible:
            print("""
✅ Feasible solutions have BETTER performance than infeasible ones!
   → Constraint satisfaction correlates with solution quality
   → Methods that maintain feasibility also find better solutions
""")
        else:
            print("""
⚠️  Infeasible solutions have similar/better performance
   → Feasibility is being traded for exploration
   → May indicate overly aggressive optimization
""")


def main() -> None:
    """Main entry point for violation healing analysis."""
    print("=" * 100)
    print("BENCHMARK VIOLATION HEALING ANALYSIS")
    print("=" * 100)
    print("\nLoading benchmark data...")
    
    # Load data
    benchmark_data = load_benchmark_files()
    
    # Extract healing data
    print("\nExtracting violation healing data...")
    results = extract_healing_data(benchmark_data)
    
    if not results:
        print("\n❌ No valid data extracted from benchmark files!")
        return
    
    print(f"  ✓ Extracted {len(results)} method-scale combinations")
    
    # Convert to DataFrame
    df = results_to_dataframe(results)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_healing_visualizations(df, OUTPUT_DIR)
    create_detailed_table(df, OUTPUT_DIR)
    
    # Print key findings
    print_key_findings(df)
    
    # Save DataFrame for further analysis
    csv_path = OUTPUT_DIR / "benchmark_violation_healing_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved data to: {csv_path}")
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
