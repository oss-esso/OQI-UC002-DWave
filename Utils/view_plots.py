"""
View all benchmark comparison plots
Opens the generated visualizations in your default image viewer
"""

import os
from pathlib import Path
import platform
import subprocess

def open_file(filepath):
    """Open a file with the default application."""
    if platform.system() == 'Windows':
        os.startfile(filepath)
    elif platform.system() == 'Darwin':  # macOS
        subprocess.run(['open', filepath])
    else:  # Linux
        subprocess.run(['xdg-open', filepath])

def main():
    plots_dir = Path(__file__).parent / "Plots"
    
    # List of main plots to view
    plots_to_view = [
        "comprehensive_speedup_comparison.png",
        "nln_speedup_comparison.png",
        "bqubo_speedup_comparison.png",
        "nln_solve_times_linear.png",
        "bqubo_solve_times_linear.png"
    ]
    
    print("="*80)
    print("Opening Benchmark Visualization Plots")
    print("="*80)
    
    for plot_name in plots_to_view:
        plot_path = plots_dir / plot_name
        if plot_path.exists():
            print(f"✓ Opening: {plot_name}")
            open_file(str(plot_path))
        else:
            print(f"✗ Not found: {plot_name}")
    
    print("\n" + "="*80)
    print("All available plots opened!")
    print("="*80)
    print("\nPlot Descriptions:")
    print("- comprehensive_speedup_comparison.png: Side-by-side NLN vs BQUBO comparison")
    print("- nln_speedup_comparison.png: Detailed NLN benchmarks (3 scales + speedup)")
    print("- bqubo_speedup_comparison.png: Detailed BQUBO benchmarks (3 scales + speedup)")
    print("- nln_solve_times_linear.png: High-res NLN linear scale")
    print("- bqubo_solve_times_linear.png: High-res BQUBO linear scale")
    print("\nSee BENCHMARK_VISUALIZATION_SUMMARY.md for detailed analysis.")
    print("="*80)

if __name__ == "__main__":
    main()
