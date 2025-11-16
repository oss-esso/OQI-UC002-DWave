"""
Example: Solving Reserve Design Problem with K-SAT

This script demonstrates the complete workflow:
1. Create a reserve design problem instance
2. Encode it to K-SAT (CNF)
3. Solve with classical SAT solvers
4. Optimize to find minimum cost solution
"""

import numpy as np
import matplotlib.pyplot as plt
from reserve_design_instance import ReserveDesignInstance

# Check if solvers are available
try:
    from sat_solver import ReserveDesignSATSolver, PYSAT_AVAILABLE, Z3_AVAILABLE
    print("SAT solver module loaded successfully")
    print(f"  PySAT available: {PYSAT_AVAILABLE}")
    print(f"  Z3 available: {Z3_AVAILABLE}")
except ImportError as e:
    print(f"Warning: Could not import SAT solvers: {e}")
    print("Install dependencies:")
    print("  pip install python-sat")
    print("  pip install z3-solver")
    PYSAT_AVAILABLE = False
    Z3_AVAILABLE = False


def example_1_basic_solving():
    """Example 1: Basic SAT solving for feasibility"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Feasibility Check with SAT")
    print("="*70)
    
    # Create a small instance
    instance = ReserveDesignInstance.create_random_instance(
        num_sites=12,
        num_species=4,
        budget_fraction=0.55,
        target_coverage=2,
        connectivity_prob=0.3,
        seed=123
    )
    
    print(f"\nProblem instance:")
    print(f"  Sites: {instance.num_sites}")
    print(f"  Species: {instance.num_species}")
    print(f"  Budget: {instance.budget:.2f}")
    print(f"  Total available cost: {np.sum(instance.costs):.2f}")
    print(f"  Edges: {len(instance.adjacency)}")
    
    # Print species targets
    print(f"\nSpecies representation targets:")
    for j in range(instance.num_species):
        sites_with_j = np.sum(instance.presence[:, j] > 0)
        print(f"  Species {j}: need {int(instance.targets[j])} sites "
              f"(available: {sites_with_j})")
    
    if not PYSAT_AVAILABLE:
        print("\nSkipping (PySAT not installed)")
        return
    
    # Solve with Glucose4
    print(f"\nSolving with Glucose4 SAT solver...")
    solver = ReserveDesignSATSolver(instance, 'glucose4', verbose=True)
    is_sat, selected_sites, stats = solver.solve()
    
    if is_sat:
        print(f"\n✓ FEASIBLE SOLUTION FOUND")
        evaluation = instance.evaluate_solution(selected_sites)
        
        print(f"\nSolution details:")
        print(f"  Selected sites: {selected_sites}")
        print(f"  Number of sites: {len(selected_sites)}")
        print(f"  Total cost: {evaluation['total_cost']:.2f} / {instance.budget:.2f}")
        print(f"  Budget utilization: {evaluation['budget_utilization']*100:.1f}%")
        
        print(f"\nSpecies coverage:")
        for sc in evaluation['species_coverage']:
            status = "✓" if sc['satisfied'] else "✗"
            print(f"  {status} {sc['species']}: {sc['achieved']}/{sc['target']}")
        
        print(f"\nConstraint satisfaction:")
        print(f"  All constraints met: {evaluation['is_feasible']}")
        if not evaluation['is_feasible']:
            print(f"  Violations: {evaluation['violations']}")
    else:
        print(f"\n✗ NO FEASIBLE SOLUTION EXISTS")
        print("The problem is over-constrained given the budget")


def example_2_optimization():
    """Example 2: Optimization via binary search"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Cost Optimization with Binary Search")
    print("="*70)
    
    # Create instance
    instance = ReserveDesignInstance.create_random_instance(
        num_sites=15,
        num_species=5,
        budget_fraction=0.7,  # Generous budget
        target_coverage=2,
        connectivity_prob=0.25,
        seed=456
    )
    
    print(f"\nProblem instance:")
    print(f"  Sites: {instance.num_sites}")
    print(f"  Species: {instance.num_species}")
    print(f"  Budget: {instance.budget:.2f}")
    
    if not PYSAT_AVAILABLE:
        print("\nSkipping (PySAT not installed)")
        return
    
    # Optimize
    print(f"\nOptimizing with binary search...")
    solver = ReserveDesignSATSolver(instance, 'glucose4', verbose=False)
    is_opt, opt_sites, opt_cost, opt_stats = solver.solve_with_optimization(
        tolerance=0.05,
        max_iterations=15
    )
    
    if is_opt:
        print(f"\n✓ OPTIMAL SOLUTION FOUND")
        print(f"  Optimal cost: {opt_cost:.2f}")
        print(f"  Budget available: {instance.budget:.2f}")
        print(f"  Savings: {instance.budget - opt_cost:.2f} "
              f"({(1 - opt_cost/instance.budget)*100:.1f}%)")
        print(f"  Sites selected: {len(opt_sites)}")
        print(f"  Optimization iterations: {opt_stats['iterations']}")
        print(f"  Total time: {opt_stats['total_time']:.3f}s")
        
        evaluation = instance.evaluate_solution(opt_sites)
        print(f"\nSpecies coverage:")
        for sc in evaluation['species_coverage']:
            status = "✓" if sc['satisfied'] else "✗"
            print(f"  {status} {sc['species']}: {sc['achieved']}/{sc['target']}")
    else:
        print(f"\n✗ NO FEASIBLE SOLUTION")


def example_3_grid_instance():
    """Example 3: Spatial grid instance"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Spatial Grid Reserve Design")
    print("="*70)
    
    # Create grid instance
    instance = ReserveDesignInstance.create_grid_instance(
        grid_rows=4,
        grid_cols=4,
        num_species=3,
        seed=789
    )
    
    print(f"\nSpatial grid problem:")
    print(f"  Grid: 4x4 ({instance.num_sites} sites)")
    print(f"  Species: {instance.num_species}")
    print(f"  Budget: {instance.budget:.2f}")
    
    if not PYSAT_AVAILABLE:
        print("\nSkipping (PySAT not installed)")
        return
    
    # Solve
    print(f"\nSolving...")
    solver = ReserveDesignSATSolver(instance, 'glucose4', verbose=False)
    is_sat, selected_sites, stats = solver.solve()
    
    if is_sat:
        print(f"\n✓ Solution found in {stats['total_time']:.3f}s")
        
        # Visualize grid solution
        grid = np.zeros((4, 4))
        for site in selected_sites:
            row, col = site // 4, site % 4
            grid[row, col] = 1
        
        print(f"\nSelected sites on grid (1 = selected, 0 = not selected):")
        for row in grid:
            print("  ", " ".join(str(int(x)) for x in row))
        
        evaluation = instance.evaluate_solution(selected_sites)
        print(f"\nCost: {evaluation['total_cost']:.2f} / {instance.budget:.2f}")
        print(f"All species covered: {evaluation['all_species_satisfied']}")


def example_4_compare_solvers():
    """Example 4: Compare different SAT solvers"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Comparing SAT Solvers")
    print("="*70)
    
    # Create instance
    instance = ReserveDesignInstance.create_random_instance(
        num_sites=20,
        num_species=6,
        budget_fraction=0.6,
        target_coverage=3,
        connectivity_prob=0.2,
        seed=999
    )
    
    print(f"\nBenchmark instance:")
    print(f"  Sites: {instance.num_sites}")
    print(f"  Species: {instance.num_species}")
    
    if not PYSAT_AVAILABLE and not Z3_AVAILABLE:
        print("\nSkipping (No SAT solvers installed)")
        return
    
    # Test available solvers
    solvers_to_test = []
    if PYSAT_AVAILABLE:
        solvers_to_test.extend(['glucose4', 'minisat22'])
    if Z3_AVAILABLE:
        solvers_to_test.append('z3')
    
    results = {}
    for solver_name in solvers_to_test:
        print(f"\nTesting {solver_name}...")
        try:
            solver = ReserveDesignSATSolver(instance, solver_name, verbose=False)
            is_sat, selected_sites, stats = solver.solve()
            results[solver_name] = stats
            
            status = "SAT" if is_sat else "UNSAT"
            print(f"  Result: {status}")
            print(f"  Time: {stats['total_time']:.3f}s "
                  f"(encoding: {stats['encoding_time']:.3f}s, "
                  f"solving: {stats['solving_time']:.3f}s)")
            if is_sat:
                print(f"  Cost: {stats['cost']:.2f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary
    if results:
        print(f"\n{'Solver':<15} {'Time (s)':<12} {'Encoding':<12} {'Solving':<12}")
        print("-" * 51)
        for solver_name, stats in results.items():
            print(f"{solver_name:<15} "
                  f"{stats['total_time']:<12.3f} "
                  f"{stats['encoding_time']:<12.3f} "
                  f"{stats['solving_time']:<12.3f}")


def example_5_infeasible_instance():
    """Example 5: Detecting infeasibility"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Detecting Infeasible Instances")
    print("="*70)
    
    # Create an infeasible instance (very tight budget)
    instance = ReserveDesignInstance.create_random_instance(
        num_sites=10,
        num_species=4,
        budget_fraction=0.15,  # Very tight budget
        target_coverage=3,
        connectivity_prob=0.3,
        seed=111
    )
    
    print(f"\nTightly constrained instance:")
    print(f"  Sites: {instance.num_sites}")
    print(f"  Species: {instance.num_species}")
    print(f"  Budget: {instance.budget:.2f}")
    print(f"  Total cost: {np.sum(instance.costs):.2f}")
    print(f"  Budget covers: {instance.budget/np.sum(instance.costs)*100:.1f}% of sites")
    
    if not PYSAT_AVAILABLE:
        print("\nSkipping (PySAT not installed)")
        return
    
    # Try to solve
    print(f"\nAttempting to solve...")
    solver = ReserveDesignSATSolver(instance, 'glucose4', verbose=False)
    is_sat, selected_sites, stats = solver.solve()
    
    if not is_sat:
        print(f"\n✗ UNSATISFIABLE - No feasible solution exists")
        print(f"  The problem is over-constrained")
        print(f"  Possible reasons:")
        print(f"    - Budget too low")
        print(f"    - Species targets too high")
        print(f"    - Connectivity requirements too strict")
        print(f"\n  Solving time: {stats['solving_time']:.3f}s")
        
        # Suggest relaxations
        print(f"\n  Suggested relaxations:")
        print(f"    - Increase budget to at least {np.sum(instance.costs) * 0.3:.2f}")
        print(f"    - Reduce species targets")
        print(f"    - Allow more disconnected components")
    else:
        print(f"\n✓ Surprisingly, a solution was found!")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("RESERVE DESIGN K-SAT SOLVER - EXAMPLES")
    print("="*70)
    
    # Run examples
    example_1_basic_solving()
    example_2_optimization()
    example_3_grid_instance()
    example_4_compare_solvers()
    example_5_infeasible_instance()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
