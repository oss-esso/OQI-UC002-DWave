"""
Visualization utilities for reserve design solutions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from reserve_design_instance import ReserveDesignInstance


def visualize_grid_solution(
    instance: ReserveDesignInstance,
    selected_sites: List[int],
    grid_rows: int,
    grid_cols: int,
    filename: Optional[str] = None
):
    """
    Visualize a solution on a spatial grid
    
    Args:
        instance: Problem instance
        selected_sites: List of selected site indices
        grid_rows: Number of rows in grid
        grid_cols: Number of columns in grid
        filename: Optional filename to save figure
    """
    if instance.num_sites != grid_rows * grid_cols:
        raise ValueError("Grid dimensions don't match number of sites")
    
    # Create grid representation
    grid = np.zeros((grid_rows, grid_cols))
    for site in selected_sites:
        row, col = site // grid_cols, site % grid_cols
        grid[row, col] = 1
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Site selection
    ax1 = axes[0]
    im1 = ax1.imshow(grid, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    ax1.set_title('Selected Sites (Green = Selected)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # Add grid lines
    for i in range(grid_rows + 1):
        ax1.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(grid_cols + 1):
        ax1.axvline(j - 0.5, color='gray', linewidth=0.5)
    
    # Add site labels
    for site in range(instance.num_sites):
        row, col = site // grid_cols, site % grid_cols
        color = 'white' if site in selected_sites else 'black'
        ax1.text(col, row, str(site), ha='center', va='center', 
                color=color, fontsize=10, fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, label='Selected')
    
    # Right plot: Species coverage heatmap
    ax2 = axes[1]
    coverage = np.zeros((grid_rows, grid_cols))
    for site in selected_sites:
        row, col = site // grid_cols, site % grid_cols
        # Count how many species are present at this site
        coverage[row, col] = np.sum(instance.presence[site, :] > 0)
    
    im2 = ax2.imshow(coverage, cmap='YlOrRd', interpolation='nearest')
    ax2.set_title('Species Richness at Selected Sites', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # Add grid lines
    for i in range(grid_rows + 1):
        ax2.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(grid_cols + 1):
        ax2.axvline(j - 0.5, color='gray', linewidth=0.5)
    
    # Add coverage labels
    for site in selected_sites:
        row, col = site // grid_cols, site % grid_cols
        count = int(coverage[row, col])
        if count > 0:
            ax2.text(col, row, str(count), ha='center', va='center', 
                    color='white', fontsize=10, fontweight='bold')
    
    plt.colorbar(im2, ax=ax2, label='Number of Species')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {filename}")
    else:
        plt.show()


def plot_species_coverage(
    instance: ReserveDesignInstance,
    selected_sites: List[int],
    filename: Optional[str] = None
):
    """
    Plot species coverage bar chart
    
    Args:
        instance: Problem instance
        selected_sites: Selected sites
        filename: Optional filename to save
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    species_names = instance.species_names if instance.species_names else \
                   [f"Species {j}" for j in range(instance.num_species)]
    
    coverage = []
    targets = []
    
    for j in range(instance.num_species):
        count = sum(instance.presence[i, j] for i in selected_sites 
                   if instance.presence[i, j] > 0)
        coverage.append(count)
        targets.append(instance.targets[j])
    
    x = np.arange(len(species_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, coverage, width, label='Achieved', color='steelblue')
    bars2 = ax.bar(x + width/2, targets, width, label='Target', color='coral')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Species', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Sites', fontsize=12, fontweight='bold')
    ax.set_title('Species Representation: Achieved vs Target', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(species_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {filename}")
    else:
        plt.show()


def plot_cost_breakdown(
    instance: ReserveDesignInstance,
    selected_sites: List[int],
    filename: Optional[str] = None
):
    """
    Plot cost breakdown
    
    Args:
        instance: Problem instance
        selected_sites: Selected sites
        filename: Optional filename to save
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Cost per site
    site_costs = [instance.costs[i] for i in selected_sites]
    site_labels = [instance.site_names[i] if instance.site_names 
                  else f"Site {i}" for i in selected_sites]
    
    bars = ax1.barh(range(len(selected_sites)), site_costs, color='teal')
    ax1.set_yticks(range(len(selected_sites)))
    ax1.set_yticklabels(site_labels)
    ax1.set_xlabel('Cost', fontsize=12, fontweight='bold')
    ax1.set_title('Individual Site Costs', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, cost) in enumerate(zip(bars, site_costs)):
        ax1.text(cost, i, f' {cost:.2f}', va='center', fontsize=9)
    
    # Right: Pie chart of budget allocation
    total_cost = sum(site_costs)
    remaining = instance.budget - total_cost
    
    if remaining >= 0:
        sizes = [total_cost, remaining]
        labels = [f'Used: ${total_cost:.2f}', f'Remaining: ${remaining:.2f}']
        colors = ['#ff6b6b', '#95e1d3']
        explode = (0.05, 0)
    else:
        sizes = [total_cost]
        labels = [f'Exceeded: ${total_cost:.2f}']
        colors = ['#ff6b6b']
        explode = (0,)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, 
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.set_title(f'Budget Utilization\n(Budget: ${instance.budget:.2f})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {filename}")
    else:
        plt.show()


def plot_optimization_progress(
    optimization_stats: dict,
    filename: Optional[str] = None
):
    """
    Plot optimization progress (binary search)
    
    Args:
        optimization_stats: Statistics from solve_with_optimization
        filename: Optional filename to save
    """
    if 'iteration_stats' not in optimization_stats:
        print("No iteration statistics available")
        return
    
    iteration_stats = optimization_stats['iteration_stats']
    
    iterations = []
    sat_results = []
    costs = []
    times = []
    
    for i, stats in enumerate(iteration_stats):
        iterations.append(i + 1)
        sat_results.append(1 if stats['is_sat'] else 0)
        costs.append(stats.get('cost', 0) if stats['is_sat'] else None)
        times.append(stats['solving_time'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: SAT/UNSAT results and costs
    ax1_twin = ax1.twinx()
    
    # Plot SAT results as scatter
    sat_iterations = [i for i, s in zip(iterations, sat_results) if s == 1]
    unsat_iterations = [i for i, s in zip(iterations, sat_results) if s == 0]
    
    ax1.scatter(sat_iterations, [1]*len(sat_iterations), 
               color='green', s=100, marker='o', label='SAT', zorder=3)
    ax1.scatter(unsat_iterations, [0]*len(unsat_iterations), 
               color='red', s=100, marker='x', label='UNSAT', zorder=3)
    
    # Plot costs
    valid_costs = [(i, c) for i, c in zip(iterations, costs) if c is not None]
    if valid_costs:
        cost_iterations, cost_values = zip(*valid_costs)
        ax1_twin.plot(cost_iterations, cost_values, 'b-o', 
                     linewidth=2, markersize=8, label='Cost')
        ax1_twin.set_ylabel('Cost', fontsize=12, fontweight='bold', color='b')
        ax1_twin.tick_params(axis='y', labelcolor='b')
    
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Result', fontsize=12, fontweight='bold')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['UNSAT', 'SAT'])
    ax1.set_title('Optimization Progress (Binary Search)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Solving time per iteration
    ax2.bar(iterations, times, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Solving Time (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Solving Time per Iteration', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add summary text
    best_cost = optimization_stats.get('best_cost', 'N/A')
    total_time = optimization_stats.get('total_time', 0)
    convergence = optimization_stats.get('convergence_gap', 0)
    
    summary_text = f"Best Cost: {best_cost:.2f}\n"
    summary_text += f"Total Time: {total_time:.3f}s\n"
    summary_text += f"Convergence Gap: {convergence:.3f}"
    
    ax2.text(0.98, 0.97, summary_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    print("Visualization examples\n")
    
    # Example 1: Grid visualization
    print("Example 1: Grid instance visualization")
    instance = ReserveDesignInstance.create_grid_instance(
        grid_rows=5,
        grid_cols=5,
        num_species=4,
        seed=42
    )
    
    # Simulate a solution (select some sites)
    selected = [0, 1, 5, 6, 10, 12, 18, 20, 24]
    
    print(f"  Visualizing {len(selected)} selected sites on 5x5 grid")
    visualize_grid_solution(instance, selected, 5, 5, 'grid_solution.png')
    
    # Example 2: Species coverage
    print("\nExample 2: Species coverage chart")
    plot_species_coverage(instance, selected, 'species_coverage.png')
    
    # Example 3: Cost breakdown
    print("\nExample 3: Cost breakdown")
    plot_cost_breakdown(instance, selected, 'cost_breakdown.png')
    
    print("\nVisualization examples complete!")
    print("Generated files: grid_solution.png, species_coverage.png, cost_breakdown.png")
