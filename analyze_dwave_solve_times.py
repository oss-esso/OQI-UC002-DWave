import json
import glob
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple

# ============================================================
# DATA COLLECTION FUNCTIONS
# ============================================================

def extract_solve_time(data: Dict) -> Tuple[float, int]:
    """
    Extract solve time and number of variables from a result file.
    Handles both Legacy and Benchmarks formats.
    
    Returns:
        (solve_time, n_variables) tuple
    """
    solve_time = 0
    n_vars = 0
    
    # Legacy format - flat structure
    if 'solve_time' in data:
        solve_time = data.get('solve_time', 0)
        # Use hybrid_time if solve_time is 0
        if solve_time == 0:
            solve_time = data.get('hybrid_time', 0)
        n_vars = data.get('n_variables', 0)
    
    # Benchmarks format - nested in result
    elif 'result' in data:
        result = data['result']
        solve_time = result.get('hybrid_time', 0)
        if solve_time == 0:
            solve_time = result.get('dwave_time', 0)
        # Try to get variables from metadata or result
        n_vars = data.get('metadata', {}).get('n_variables', 0)
        if n_vars == 0:
            n_vars = result.get('n_variables', 0)
        
        # If still no variables, estimate from n_farms and n_foods (27 foods typical)
        # For Farm problem: n_vars ≈ n_farms * n_foods
        # For Patch problem: more complex, but roughly n_farms * n_foods * n_crops
        if n_vars == 0:
            n_farms = data.get('metadata', {}).get('n_farms', 0)
            if n_farms > 0:
                # Rough estimate: 27 foods per farm
                n_vars = n_farms * 27
    
    return solve_time, n_vars

def scan_dwave_results(base_dirs: List[str]) -> Dict:
    """
    Scan all DWave subdirectories in the given base directories.
    
    Args:
        base_dirs: List of base directory paths to scan
        
    Returns:
        Dictionary with structure:
        {
            'files': [list of file paths],
            'solve_times': [list of solve times],
            'n_variables': [list of variable counts],
            'by_scenario': {scenario_name: {'solve_times': [...], 'n_variables': [...]}},
            'by_problem_size': {n_vars: [solve_times]}
        }
    """
    results = {
        'files': [],
        'solve_times': [],
        'n_variables': [],
        'by_scenario': defaultdict(lambda: {'solve_times': [], 'n_variables': []}),
        'by_problem_size': defaultdict(list)
    }
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: Directory {base_dir} does not exist")
            continue
        
        # Pattern 1: Find subdirectories named "DWave" (Benchmarks format)
        dwave_dirs = glob.glob(f"{base_dir}/**/DWave", recursive=True)
        
        # Pattern 2: Find directories containing "DWave" in their name (Legacy format)
        # e.g., Farm_DWave, Patch_DWave, Patch_DWaveBQM
        legacy_dirs = glob.glob(f"{base_dir}/**/*DWave*", recursive=True)
        legacy_dirs = [d for d in legacy_dirs if os.path.isdir(d) and 'DWave' in os.path.basename(d)]
        
        # Combine both patterns, removing duplicates
        all_dirs = list(set(dwave_dirs + legacy_dirs))
        
        for dwave_dir in all_dirs:
            # Determine scenario name based on directory structure
            dir_name = os.path.basename(dwave_dir)
            parent_name = os.path.basename(os.path.dirname(dwave_dir))
            
            # For Legacy format (e.g., Farm_DWave, Patch_DWaveBQM)
            if 'DWave' in dir_name and dir_name != 'DWave':
                scenario_name = dir_name
            # For Benchmarks format (parent/DWave) - use grandparent if parent is COMPREHENSIVE
            elif dir_name == 'DWave':
                # Check if this is in Legacy/COMPREHENSIVE
                full_path = os.path.abspath(dwave_dir)
                if 'Legacy' in full_path or 'COMPREHENSIVE' in full_path:
                    scenario_name = parent_name
                else:
                    # Benchmarks format: use parent directory (BQUBO, LQ, NLN, etc.)
                    scenario_name = parent_name
            else:
                scenario_name = dir_name
            
            # Find all JSON files in this DWave directory
            json_files = glob.glob(f"{dwave_dir}/*.json")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    solve_time, n_vars = extract_solve_time(data)
                    
                    if solve_time > 0:  # Only include valid results
                        results['files'].append(json_file)
                        results['solve_times'].append(solve_time)
                        results['n_variables'].append(n_vars)
                        
                        # Group by scenario
                        results['by_scenario'][scenario_name]['solve_times'].append(solve_time)
                        results['by_scenario'][scenario_name]['n_variables'].append(n_vars)
                        
                        # Group by problem size
                        if n_vars > 0:
                            results['by_problem_size'][n_vars].append(solve_time)
                
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")
    
    return results

# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def compute_statistics(results: Dict) -> Dict:
    """
    Compute various statistics from the results.
    
    Returns:
        Dictionary with computed statistics
    """
    solve_times = np.array(results['solve_times'])
    n_variables = np.array(results['n_variables'])
    
    # Filter out zero variables for per-variable calculations
    valid_idx = n_variables > 0
    valid_times = solve_times[valid_idx]
    valid_vars = n_variables[valid_idx]
    
    time_per_var = valid_times / valid_vars if len(valid_vars) > 0 else np.array([])
    
    stats = {
        'total_runs': len(solve_times),
        'total_solve_time': float(np.sum(solve_times)),
        'mean_solve_time': float(np.mean(solve_times)),
        'median_solve_time': float(np.median(solve_times)),
        'std_solve_time': float(np.std(solve_times)),
        'min_solve_time': float(np.min(solve_times)),
        'max_solve_time': float(np.max(solve_times)),
        'mean_variables': float(np.mean(valid_vars)) if len(valid_vars) > 0 else 0,
        'median_variables': float(np.median(valid_vars)) if len(valid_vars) > 0 else 0,
        'min_variables': int(np.min(valid_vars)) if len(valid_vars) > 0 else 0,
        'max_variables': int(np.max(valid_vars)) if len(valid_vars) > 0 else 0,
        'mean_time_per_var': float(np.mean(time_per_var)) if len(time_per_var) > 0 else 0,
        'median_time_per_var': float(np.median(time_per_var)) if len(time_per_var) > 0 else 0,
        'std_time_per_var': float(np.std(time_per_var)) if len(time_per_var) > 0 else 0,
    }
    
    # Scenario-wise statistics
    stats['by_scenario'] = {}
    for scenario, data in results['by_scenario'].items():
        times = np.array(data['solve_times'])
        vars = np.array(data['n_variables'])
        valid_idx = vars > 0
        valid_times = times[valid_idx]
        valid_vars = vars[valid_idx]
        
        stats['by_scenario'][scenario] = {
            'n_runs': len(times),
            'total_time': float(np.sum(times)),
            'mean_time': float(np.mean(times)),
            'mean_variables': float(np.mean(valid_vars)) if len(valid_vars) > 0 else 0,
            'mean_time_per_var': float(np.mean(valid_times / valid_vars)) if len(valid_vars) > 0 else 0,
        }
    
    return stats

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_plots(results: Dict, stats: Dict, output_dir: str = "Latex"):
    """
    Create plots for the LaTeX document.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    solve_times = np.array(results['solve_times'])
    n_variables = np.array(results['n_variables'])
    
    # Filter valid data
    valid_idx = n_variables > 0
    valid_times = solve_times[valid_idx]
    valid_vars = n_variables[valid_idx]
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot 1: Solve Time vs Number of Variables (scatter) - only if we have variable data
    if len(valid_vars) > 0:
        plt.figure(figsize=(12, 7))
        
        # Define color palette for scenarios
        import matplotlib.cm as cm
        # Exclude the 'PATCH' scenario as requested
        scenarios = [s for s in sorted(results['by_scenario'].keys()) if s != 'PATCH']
        colors = cm.tab10(np.linspace(0, 1, len(scenarios)))
        
        # Plot scatter points colored by scenario
        for idx, scenario in enumerate(scenarios):
            scenario_data = results['by_scenario'][scenario]
            s_times = np.array(scenario_data['solve_times'])
            s_vars = np.array(scenario_data['n_variables'])
            valid_mask = (s_vars > 0) & (s_times > 0)
            if valid_mask.sum() > 0:
                plt.scatter(s_vars[valid_mask], s_times[valid_mask], 
                           alpha=0.6, s=50, color=colors[idx], label=f'{scenario} (data)')
        
        # Add power-law fits for each scenario
        if len(valid_vars) > 1:
            x_min, x_max = valid_vars.min(), valid_vars.max()
            x_pl = np.linspace(x_min, x_max, 200)
            
            for idx, scenario in enumerate(scenarios):
                scenario_data = results['by_scenario'][scenario]
                s_times = np.array(scenario_data['solve_times'])
                s_vars = np.array(scenario_data['n_variables'])
                positive_mask = (s_vars > 0) & (s_times > 0)
                
                if positive_mask.sum() > 1:
                    log_x = np.log(s_vars[positive_mask])
                    log_y = np.log(s_times[positive_mask])
                    b, log_a = np.polyfit(log_x, log_y, 1)
                    a = float(np.exp(log_a))
                    
                    # Plot scenario power-law fit
                    y_pl = a * (x_pl ** b)
                    plt.plot(x_pl, y_pl, "--", alpha=0.7, linewidth=1.5, color=colors[idx],
                            label=rf'{scenario}: $t={a:.1e}\,n^{{{b:.2f}}}$')
            
            # Overall "Average" power-law fit on all data
            positive_mask = (valid_vars > 0) & (valid_times > 0)
            if positive_mask.sum() > 1:
                log_x = np.log(valid_vars[positive_mask])
                log_y = np.log(valid_times[positive_mask])
                b, log_a = np.polyfit(log_x, log_y, 1)
                a = float(np.exp(log_a))

                # Create smooth curve for plotting
                y_pl = a * (x_pl ** b)
                plt.plot(x_pl, y_pl, "k-", alpha=0.9, linewidth=2.5,
                         label=rf'Average: $t={a:.2e}\,n^{{{b:.3f}}}$')

                # Compute R^2 in log space
                y_pred_log = log_a + b * log_x
                ss_res = np.sum((log_y - y_pred_log) ** 2)
                ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

                # Annotate average fit parameters on the plot
                ann_text = rf'Average: $t = {a:.2e} \, n^{{{b:.3f}}}$' + '\n' + rf'$R^2 = {r2:.3f}$'
                plt.gca().text(0.02, 0.98, ann_text, transform=plt.gca().transAxes,
                               fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

            plt.legend(fontsize=8, loc='best', ncol=2)
        
        # Use logarithmic scaling on the y-axis for better visualization
        plt.yscale('log')
        
        plt.xlabel('Number of Variables', fontsize=12)
        plt.ylabel('Solve Time (seconds)', fontsize=12)
        plt.title('DWave Solve Time vs Problem Size (Power-Law Fits by Scenario)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/solve_time_vs_variables.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Time per Variable Distribution - only if we have variable data
    if len(valid_vars) > 0:
        time_per_var = valid_times / valid_vars
        plt.figure(figsize=(10, 6))
        plt.hist(time_per_var, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(stats['mean_time_per_var'], color='red', linestyle='--', 
                    linewidth=2, label=f"Mean: {stats['mean_time_per_var']:.6f} s/var")
        plt.axvline(stats['median_time_per_var'], color='green', linestyle='--', 
                    linewidth=2, label=f"Median: {stats['median_time_per_var']:.6f} s/var")
        plt.xlabel('Time per Variable (seconds)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Solve Time per Variable', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_per_variable_dist.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Solve Time Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(solve_times, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(stats['mean_solve_time'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {stats['mean_solve_time']:.2f} s")
    plt.axvline(stats['median_solve_time'], color='green', linestyle='--', 
                linewidth=2, label=f"Median: {stats['median_solve_time']:.2f} s")
    plt.xlabel('Solve Time (seconds)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Solve Times', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/solve_time_dist.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Average time per variable by problem size - only if we have variable data
    if results['by_problem_size'] and len(results['by_problem_size']) > 1:
        sizes = sorted(results['by_problem_size'].keys())
        avg_times = [np.mean(results['by_problem_size'][s]) for s in sizes]
        std_times = [np.std(results['by_problem_size'][s]) for s in sizes]
        time_per_var_by_size = [np.mean(np.array(results['by_problem_size'][s]) / s) for s in sizes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Average solve time by problem size
        ax1.errorbar(sizes, avg_times, yerr=std_times, fmt='o-', capsize=5, capthick=2, markersize=8)
        ax1.set_xlabel('Number of Variables', fontsize=12)
        ax1.set_ylabel('Average Solve Time (seconds)', fontsize=12)
        ax1.set_title('Average Solve Time by Problem Size', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Time per variable by problem size
        ax2.plot(sizes, time_per_var_by_size, 'o-', markersize=8, linewidth=2)
        ax2.set_xlabel('Number of Variables', fontsize=12)
        ax2.set_ylabel('Time per Variable (seconds)', fontsize=12)
        ax2.set_title('Time per Variable by Problem Size', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/solve_time_by_size.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 5: Scenario comparison
    if len(results['by_scenario']) > 1:
        scenarios = sorted(results['by_scenario'].keys())
        scenario_stats = [stats['by_scenario'][s] for s in scenarios]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Total time by scenario
        total_times = [s['total_time'] for s in scenario_stats]
        ax1.bar(range(len(scenarios)), total_times, alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_xticklabels(scenarios, rotation=45, ha='right')
        ax1.set_ylabel('Total Solve Time (seconds)', fontsize=12)
        ax1.set_title('Total Solve Time by Scenario', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Mean time per variable by scenario - only if we have variable data
        if stats['mean_time_per_var'] > 0:
            mean_times_per_var = [s['mean_time_per_var'] for s in scenario_stats]
            ax2.bar(range(len(scenarios)), mean_times_per_var, alpha=0.7, edgecolor='black', color='orange')
            ax2.set_xticks(range(len(scenarios)))
            ax2.set_xticklabels(scenarios, rotation=45, ha='right')
            ax2.set_ylabel('Mean Time per Variable (seconds)', fontsize=12)
            ax2.set_title('Mean Time per Variable by Scenario', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            # Show mean solve time instead
            mean_times = [s['mean_time'] for s in scenario_stats]
            ax2.bar(range(len(scenarios)), mean_times, alpha=0.7, edgecolor='black', color='orange')
            ax2.set_xticks(range(len(scenarios)))
            ax2.set_xticklabels(scenarios, rotation=45, ha='right')
            ax2.set_ylabel('Mean Solve Time (seconds)', fontsize=12)
            ax2.set_title('Mean Solve Time by Scenario', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/scenario_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"All plots saved to {output_dir}/")

# ============================================================
# LATEX GENERATION FUNCTIONS
# ============================================================

def generate_latex_document(results: Dict, stats: Dict, output_file: str = "Latex/dwave_solve_time_analysis.tex"):
    """
    Generate a comprehensive LaTeX document with the analysis.
    """
    
    # Calculate some additional useful metrics
    total_hours = stats['total_solve_time'] / 3600
    total_days = total_hours / 24
    
    latex_content = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}

\title{DWave Quantum Annealer Solve Time Analysis}
\author{Computational Benchmark Report}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This document presents a comprehensive analysis of DWave quantum annealer solve times across all benchmark runs in the Legacy and Benchmarks directories. The analysis includes """ + str(stats['total_runs']) + r""" runs spanning various problem formulations and sizes, with a focus on understanding the relationship between problem size (number of variables) and computational time requirements.
\end{abstract}

\section{Executive Summary}

\subsection{Key Findings}

\begin{itemize}
    \item \textbf{Total Runs Analyzed:} """ + f"{stats['total_runs']:,}" + r"""
    \item \textbf{Total Solve Time:} """ + f"{stats['total_solve_time']:.2f}" + r""" seconds (""" + f"{total_hours:.2f}" + r""" hours, """ + f"{total_days:.2f}" + r""" days)
    \item \textbf{Average Solve Time:} """ + f"{stats['mean_solve_time']:.3f}" + r""" seconds
    \item \textbf{Median Solve Time:} """ + f"{stats['median_solve_time']:.3f}" + r""" seconds
    \item \textbf{Problem Size Range:} """ + f"{stats['min_variables']}" + r""" to """ + f"{stats['max_variables']}" + r""" variables
    \item \textbf{Average Time per Variable:} """ + f"{stats['mean_time_per_var']:.6f}" + r""" seconds/variable
    \item \textbf{Median Time per Variable:} """ + f"{stats['median_time_per_var']:.6f}" + r""" seconds/variable
\end{itemize}

\subsection{Computational Cost Estimate}

To reproduce all """ + str(stats['total_runs']) + r""" DWave runs would require approximately:

\begin{center}
\textbf{""" + f"{total_hours:.1f}" + r""" hours (""" + f"{total_days:.2f}" + r""" days)}
\end{center}

This assumes sequential execution. With parallel execution on multiple DWave systems, this time could be significantly reduced.

\section{Detailed Statistics}

\subsection{Solve Time Distribution}

\begin{table}[H]
\centering
\caption{Statistical Summary of Solve Times}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value (seconds)} \\
\midrule
Mean & """ + f"{stats['mean_solve_time']:.3f}" + r""" \\
Median & """ + f"{stats['median_solve_time']:.3f}" + r""" \\
Standard Deviation & """ + f"{stats['std_solve_time']:.3f}" + r""" \\
Minimum & """ + f"{stats['min_solve_time']:.3f}" + r""" \\
Maximum & """ + f"{stats['max_solve_time']:.3f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Problem Size Distribution}

\begin{table}[H]
\centering
\caption{Statistical Summary of Problem Sizes}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value (variables)} \\
\midrule
Mean & """ + f"{stats['mean_variables']:.1f}" + r""" \\
Median & """ + f"{stats['median_variables']:.0f}" + r""" \\
Minimum & """ + f"{stats['min_variables']}" + r""" \\
Maximum & """ + f"{stats['max_variables']}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Efficiency Metrics}

\begin{table}[H]
\centering
\caption{Time per Variable Statistics}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value (seconds/variable)} \\
\midrule
Mean & """ + f"{stats['mean_time_per_var']:.6f}" + r""" \\
Median & """ + f"{stats['median_time_per_var']:.6f}" + r""" \\
Standard Deviation & """ + f"{stats['std_time_per_var']:.6f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

\section{Scenario-Wise Analysis}

"""
    
    # Add scenario breakdown
    if stats['by_scenario']:
        latex_content += r"""\begin{table}[H]
\centering
\caption{Solve Time Analysis by Scenario}
\begin{tabular}{lrrrr}
\toprule
\textbf{Scenario} & \textbf{Runs} & \textbf{Total Time (s)} & \textbf{Mean Time (s)} & \textbf{Time/Var (s)} \\
\midrule
"""
        for scenario in sorted(stats['by_scenario'].keys()):
            s = stats['by_scenario'][scenario]
            latex_content += f"{scenario.replace('_', ' ')} & {s['n_runs']} & {s['total_time']:.2f} & {s['mean_time']:.3f} & {s['mean_time_per_var']:.6f} \\\\\n"
        
        latex_content += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    latex_content += r"""
\section{Visualizations}

\subsection{Solve Time vs Problem Size}

Figure \ref{fig:solve_time_vs_vars} shows the relationship between problem size (number of variables) and solve time. The polynomial fit line indicates the scaling behavior of the quantum annealer.

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{solve_time_vs_variables.pdf}
\caption{Solve time as a function of problem size with polynomial trend line}
\label{fig:solve_time_vs_vars}
\end{figure}

\subsection{Time per Variable Analysis}

Figure \ref{fig:time_per_var} shows the distribution of solve time per variable across all runs. This metric helps normalize for problem size and understand the efficiency of the solver.

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{time_per_variable_dist.pdf}
\caption{Distribution of solve time per variable}
\label{fig:time_per_var}
\end{figure}

\subsection{Solve Time Distribution}

Figure \ref{fig:solve_time_dist} shows the overall distribution of solve times across all benchmark runs.

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{solve_time_dist.pdf}
\caption{Distribution of solve times across all runs}
\label{fig:solve_time_dist}
\end{figure}

"""
    
    if results['by_problem_size'] and len(results['by_problem_size']) > 1:
        latex_content += r"""
\subsection{Analysis by Problem Size}

Figure \ref{fig:by_size} shows how solve time and efficiency vary with problem size.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{solve_time_by_size.pdf}
\caption{Left: Average solve time by problem size with error bars. Right: Time per variable showing efficiency scaling.}
\label{fig:by_size}
\end{figure}

"""
    
    if len(stats['by_scenario']) > 1:
        latex_content += r"""
\subsection{Scenario Comparison}

Figure \ref{fig:scenarios} compares the computational requirements across different problem scenarios.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{scenario_comparison.pdf}
\caption{Left: Total solve time by scenario. Right: Mean time per variable by scenario.}
\label{fig:scenarios}
\end{figure}

"""
    
    latex_content += r"""
\section{Time Estimation Formula}

Based on the analysis, we can estimate the solve time for a new problem using the following approaches:

\subsection{Linear Approximation}

For a problem with $n$ variables, the estimated solve time using the average time per variable is:

\begin{equation}
t_{\text{est}} = n \times """ + f"{stats['mean_time_per_var']:.6f}" + r""" \text{ seconds}
\end{equation}

\subsection{Conservative Estimate}

Using the mean solve time plus one standard deviation as a conservative estimate:

\begin{equation}
t_{\text{conservative}} = n \times \left(""" + f"{stats['mean_time_per_var']:.6f}" + r""" + """ + f"{stats['std_time_per_var']:.6f}" + r"""\right) = n \times """ + f"{stats['mean_time_per_var'] + stats['std_time_per_var']:.6f}" + r""" \text{ seconds}
\end{equation}

\subsection{Usage Examples}

\begin{table}[H]
\centering
\caption{Estimated Solve Times for Various Problem Sizes}
\begin{tabular}{rrr}
\toprule
\textbf{Variables} & \textbf{Expected Time (s)} & \textbf{Conservative Time (s)} \\
\midrule
"""
    
    # Add some example problem sizes
    example_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    for size in example_sizes:
        expected = size * stats['mean_time_per_var']
        conservative = size * (stats['mean_time_per_var'] + stats['std_time_per_var'])
        latex_content += f"{size} & {expected:.2f} & {conservative:.2f} \\\\\n"
    
    latex_content += r"""\bottomrule
\end{tabular}
\end{table}

\section{Conclusions}

\begin{enumerate}
    \item The total computational cost to reproduce all DWave runs is approximately """ + f"{total_hours:.1f}" + r""" hours.
    \item The average solve time per variable of """ + f"{stats['mean_time_per_var']:.6f}" + r""" seconds provides a useful metric for estimating computational requirements for new problems.
    \item There is """ + ("significant" if stats['std_time_per_var'] / stats['mean_time_per_var'] > 0.5 else "moderate") + r""" variability in solve times (CV = """ + f"{100 * stats['std_time_per_var'] / stats['mean_time_per_var']:.1f}" + r"""\%), suggesting that problem-specific characteristics significantly impact solver performance.
    \item For project planning, we recommend using the conservative estimate formula to ensure adequate computational resources.
\end{enumerate}

\section{Recommendations}

\begin{itemize}
    \item For large-scale benchmarking campaigns, consider using parallel execution across multiple DWave systems to reduce wall-clock time.
    \item Monitor time per variable as a key efficiency metric for detecting problematic problem formulations.
    \item Consider implementing early termination strategies for runs that exceed expected solve times by a large margin.
    \item Budget computational resources based on the conservative estimate to account for variability.
\end{itemize}

\end{document}
"""
    
    # Write the LaTeX file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX document generated: {output_file}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*70)
    print("DWave Solve Time Analysis")
    print("="*70)
    
    # Define directories to scan
    base_dirs = [
        "Legacy/COMPREHENSIVE",
        "Benchmarks"
    ]
    
    print("\nScanning directories for DWave results...")
    for dir in base_dirs:
        if os.path.exists(dir):
            print(f"  ✓ {dir}")
        else:
            print(f"  ✗ {dir} (not found)")
    
    # Collect all DWave results
    print("\nCollecting data from all DWave runs...")
    results = scan_dwave_results(base_dirs)
    
    print(f"\nFound {len(results['files'])} DWave result files")
    print(f"Valid results with solve times: {len(results['solve_times'])}")
    
    if len(results['solve_times']) == 0:
        print("\nNo valid results found. Exiting.")
        return
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(results)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nTotal DWave runs: {stats['total_runs']}")
    print(f"Total solve time: {stats['total_solve_time']:.2f} seconds")
    print(f"                  {stats['total_solve_time']/3600:.2f} hours")
    print(f"                  {stats['total_solve_time']/3600/24:.2f} days")
    print(f"\nAverage solve time: {stats['mean_solve_time']:.3f} seconds")
    print(f"Median solve time: {stats['median_solve_time']:.3f} seconds")
    print(f"Std dev solve time: {stats['std_solve_time']:.3f} seconds")
    print(f"\nProblem size range: {stats['min_variables']} to {stats['max_variables']} variables")
    print(f"Average problem size: {stats['mean_variables']:.1f} variables")
    print(f"\nTime per variable (mean): {stats['mean_time_per_var']:.6f} seconds/variable")
    print(f"Time per variable (median): {stats['median_time_per_var']:.6f} seconds/variable")
    
    print("\n" + "="*70)
    print("SCENARIO BREAKDOWN")
    print("="*70)
    for scenario in sorted(stats['by_scenario'].keys()):
        s = stats['by_scenario'][scenario]
        print(f"\n{scenario}:")
        print(f"  Runs: {s['n_runs']}")
        print(f"  Total time: {s['total_time']:.2f} seconds ({s['total_time']/3600:.2f} hours)")
        print(f"  Mean time: {s['mean_time']:.3f} seconds")
        print(f"  Mean variables: {s['mean_variables']:.1f}")
        print(f"  Time per variable: {s['mean_time_per_var']:.6f} seconds/variable")
    
    # Create visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    create_plots(results, stats)
    
    # Generate LaTeX document
    print("\n" + "="*70)
    print("GENERATING LATEX DOCUMENT")
    print("="*70)
    generate_latex_document(results, stats)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - Latex/dwave_solve_time_analysis.tex")
    print("  - Latex/solve_time_vs_variables.pdf")
    print("  - Latex/time_per_variable_dist.pdf")
    print("  - Latex/solve_time_dist.pdf")
    print("  - Latex/solve_time_by_size.pdf")
    print("  - Latex/scenario_comparison.pdf")
    print("\nTo compile the LaTeX document:")
    print("  cd Latex && pdflatex dwave_solve_time_analysis.tex")

if __name__ == "__main__":
    main()
