"""
Classical SAT Solvers for Reserve Design Problem

This module provides interfaces to various classical SAT solvers
for solving the reserve design problem encoded as K-SAT.
"""

import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from reserve_design_instance import ReserveDesignInstance
from sat_encoder import ReserveDesignSATEncoder


# Try to import SAT solvers
try:
    from pysat.solvers import Glucose3, Glucose4, Minisat22, Lingeling
    # Try to import Cadical (version-specific)
    try:
        from pysat.solvers import Cadical195 as Cadical
    except ImportError:
        try:
            from pysat.solvers import Cadical153 as Cadical
        except ImportError:
            from pysat.solvers import Cadical103 as Cadical
    PYSAT_AVAILABLE = True
except ImportError as e:
    PYSAT_AVAILABLE = False
    print(f"Warning: PySAT not installed. Install with: pip install python-sat (Error: {e})")

try:
    from z3 import *
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("Warning: Z3 not installed. Install with: pip install z3-solver")


class ReserveDesignSATSolver:
    """
    Solve reserve design problem using SAT solvers
    """
    
    def __init__(
        self,
        instance: ReserveDesignInstance,
        solver_name: str = 'glucose4',
        verbose: bool = False
    ):
        """
        Initialize solver
        
        Args:
            instance: Problem instance
            solver_name: Name of SAT solver ('glucose3', 'glucose4', 'minisat22', 
                        'lingeling', 'cadical', 'z3')
            verbose: Print solving progress
        """
        self.instance = instance
        self.solver_name = solver_name.lower()
        self.verbose = verbose
        self.encoder = None
        self.solving_time = 0
        self.encoding_time = 0
    
    def solve(self, objective_bound: Optional[float] = None) -> Tuple[bool, List[int], Dict]:
        """
        Solve the problem (feasibility check)
        
        Args:
            objective_bound: Cost bound to check (uses budget if None)
        
        Returns:
            (is_sat, selected_sites, stats): Solution status, selected sites, and statistics
        """
        if not PYSAT_AVAILABLE and self.solver_name != 'z3':
            raise ImportError("PySAT not installed. Install with: pip install python-sat")
        
        if self.solver_name == 'z3':
            return self._solve_with_z3(objective_bound)
        else:
            return self._solve_with_pysat(objective_bound)
    
    def _solve_with_pysat(
        self,
        objective_bound: Optional[float] = None
    ) -> Tuple[bool, List[int], Dict]:
        """Solve using PySAT solvers"""
        # Encode problem
        start_encoding = time.time()
        self.encoder = ReserveDesignSATEncoder(self.instance, verbose=self.verbose)
        cnf = self.encoder.encode(objective_bound)
        self.encoding_time = time.time() - start_encoding
        
        # Choose solver
        solver_classes = {
            'glucose3': Glucose3,
            'glucose4': Glucose4,
            'minisat22': Minisat22,
            'lingeling': Lingeling,
            'cadical': Cadical,
        }
        
        if self.solver_name not in solver_classes:
            raise ValueError(f"Unknown solver: {self.solver_name}. "
                           f"Available: {list(solver_classes.keys())}")
        
        solver_class = solver_classes[self.solver_name]
        
        if self.verbose:
            print(f"\n=== Solving with {self.solver_name} ===")
            print(f"Encoding time: {self.encoding_time:.3f}s")
        
        # Solve
        start_solving = time.time()
        selected_sites = []
        is_sat = False
        
        try:
            with solver_class(bootstrap_with=cnf.clauses) as solver:
                is_sat = solver.solve()
                
                if is_sat:
                    model = solver.get_model()
                    selected_sites = self.encoder.decode_solution(model)
        except Exception as e:
            print(f"Error during solving: {e}")
            is_sat = False
        
        self.solving_time = time.time() - start_solving
        
        # Compile statistics
        stats = {
            'solver': self.solver_name,
            'encoding_time': self.encoding_time,
            'solving_time': self.solving_time,
            'total_time': self.encoding_time + self.solving_time,
            'is_sat': is_sat,
            'num_selected': len(selected_sites),
            **self.encoder.get_encoding_statistics()
        }
        
        if is_sat:
            evaluation = self.instance.evaluate_solution(selected_sites)
            stats['cost'] = evaluation['total_cost']
            stats['is_feasible'] = evaluation['is_feasible']
            stats['violations'] = evaluation['violations']
        
        if self.verbose:
            print(f"Solving time: {self.solving_time:.3f}s")
            print(f"Result: {'SAT' if is_sat else 'UNSAT'}")
            if is_sat:
                print(f"Selected {len(selected_sites)} sites")
                print(f"Total cost: {stats['cost']:.2f}")
        
        return is_sat, selected_sites, stats
    
    def _solve_with_z3(
        self,
        objective_bound: Optional[float] = None
    ) -> Tuple[bool, List[int], Dict]:
        """Solve using Z3 SMT solver"""
        if not Z3_AVAILABLE:
            raise ImportError("Z3 not installed. Install with: pip install z3-solver")
        
        if objective_bound is None:
            objective_bound = self.instance.budget
        
        if self.verbose:
            print(f"\n=== Solving with Z3 ===")
        
        start_encoding = time.time()
        
        # Create Boolean variables for each site
        x = [Bool(f'x_{i}') for i in range(self.instance.num_sites)]
        
        # Create solver
        solver = Solver()
        
        # Species representation constraints
        for j in range(self.instance.num_species):
            sites_with_j = []
            for i in range(self.instance.num_sites):
                if self.instance.presence[i, j] > 0:
                    sites_with_j.append(x[i])
            
            if sites_with_j:
                target = int(self.instance.targets[j])
                # Use PbGe (Pseudo-Boolean Greater or Equal)
                solver.add(PbGe([(s, 1) for s in sites_with_j], target))
        
        # Budget constraint
        # Scale costs to integers
        int_costs = (self.instance.costs * 100).astype(int)
        max_int_cost = int(objective_bound * 100)
        cost_terms = [(x[i], int(int_costs[i])) for i in range(self.instance.num_sites)]
        solver.add(PbLe(cost_terms, max_int_cost))
        
        self.encoding_time = time.time() - start_encoding
        
        # Solve
        start_solving = time.time()
        result = solver.check()
        self.solving_time = time.time() - start_solving
        
        is_sat = (result == sat)
        selected_sites = []
        
        if is_sat:
            model = solver.model()
            for i in range(self.instance.num_sites):
                if is_true(model.evaluate(x[i])):
                    selected_sites.append(i)
        
        # Statistics
        stats = {
            'solver': 'z3',
            'encoding_time': self.encoding_time,
            'solving_time': self.solving_time,
            'total_time': self.encoding_time + self.solving_time,
            'is_sat': is_sat,
            'num_selected': len(selected_sites),
        }
        
        if is_sat:
            evaluation = self.instance.evaluate_solution(selected_sites)
            stats['cost'] = evaluation['total_cost']
            stats['is_feasible'] = evaluation['is_feasible']
            stats['violations'] = evaluation['violations']
        
        if self.verbose:
            print(f"Encoding time: {self.encoding_time:.3f}s")
            print(f"Solving time: {self.solving_time:.3f}s")
            print(f"Result: {'SAT' if is_sat else 'UNSAT'}")
            if is_sat:
                print(f"Selected {len(selected_sites)} sites")
                print(f"Total cost: {stats['cost']:.2f}")
        
        return is_sat, selected_sites, stats
    
    def solve_with_optimization(
        self,
        tolerance: float = 0.01,
        max_iterations: int = 50
    ) -> Tuple[bool, List[int], float, Dict]:
        """
        Solve with optimization using binary search on objective
        
        Args:
            tolerance: Stop when cost range is within this tolerance
            max_iterations: Maximum number of binary search iterations
        
        Returns:
            (is_sat, selected_sites, best_cost, stats): Optimal solution
        """
        if self.verbose:
            print("\n=== Optimizing with Binary Search ===")
        
        # Initialize bounds
        lower = 0
        upper = self.instance.budget
        best_solution = None
        best_cost = float('inf')
        iteration = 0
        all_stats = []
        
        total_start = time.time()
        
        while iteration < max_iterations and (upper - lower) > tolerance:
            iteration += 1
            mid = (lower + upper) / 2.0
            
            if self.verbose:
                print(f"\nIteration {iteration}: trying cost bound {mid:.2f} "
                      f"(range: [{lower:.2f}, {upper:.2f}])")
            
            # Try to find solution with cost <= mid
            is_sat, selected_sites, stats = self.solve(objective_bound=mid)
            all_stats.append(stats)
            
            if is_sat:
                # Calculate actual cost
                cost = sum(self.instance.costs[i] for i in selected_sites)
                
                if cost < best_cost:
                    best_cost = cost
                    best_solution = selected_sites
                    
                    if self.verbose:
                        print(f"  ✓ Found solution with cost {cost:.2f}")
                
                # Try to find cheaper solution
                upper = mid
            else:
                # Need higher budget
                lower = mid
                if self.verbose:
                    print(f"  ✗ No solution at this cost bound")
        
        total_time = time.time() - total_start
        
        # Final statistics
        final_stats = {
            'optimization_method': 'binary_search',
            'iterations': iteration,
            'total_time': total_time,
            'best_cost': best_cost if best_solution else float('inf'),
            'found_solution': best_solution is not None,
            'convergence_gap': upper - lower,
            'iteration_stats': all_stats
        }
        
        if self.verbose:
            print(f"\n=== Optimization Complete ===")
            print(f"Iterations: {iteration}")
            print(f"Total time: {total_time:.3f}s")
            if best_solution:
                print(f"Best cost: {best_cost:.2f}")
                print(f"Sites selected: {len(best_solution)}")
            else:
                print("No feasible solution found")
        
        if best_solution is not None:
            return True, best_solution, best_cost, final_stats
        else:
            return False, [], float('inf'), final_stats


def compare_solvers(
    instance: ReserveDesignInstance,
    solvers: Optional[List[str]] = None,
    objective_bound: Optional[float] = None
) -> Dict:
    """
    Compare multiple SAT solvers on the same instance
    
    Args:
        instance: Problem instance
        solvers: List of solver names (default: all available)
        objective_bound: Cost bound (uses budget if None)
    
    Returns:
        Dictionary with comparison results
    """
    if solvers is None:
        solvers = ['glucose4', 'minisat22', 'cadical']
        if Z3_AVAILABLE:
            solvers.append('z3')
    
    results = {}
    
    for solver_name in solvers:
        print(f"\n{'='*60}")
        print(f"Testing solver: {solver_name}")
        print('='*60)
        
        try:
            solver = ReserveDesignSATSolver(instance, solver_name, verbose=True)
            is_sat, selected_sites, stats = solver.solve(objective_bound)
            
            results[solver_name] = {
                'success': True,
                'is_sat': is_sat,
                'selected_sites': selected_sites,
                'stats': stats
            }
        except Exception as e:
            print(f"Error with {solver_name}: {e}")
            results[solver_name] = {
                'success': False,
                'error': str(e)
            }
    
    return results


if __name__ == "__main__":
    from reserve_design_instance import ReserveDesignInstance
    
    # Create test instance
    print("Creating test instance...")
    instance = ReserveDesignInstance.create_random_instance(
        num_sites=15,
        num_species=4,
        budget_fraction=0.6,
        target_coverage=2,
        connectivity_prob=0.3,
        seed=42
    )
    
    print(f"\nInstance: {instance.num_sites} sites, {instance.num_species} species")
    print(f"Budget: {instance.budget:.2f}, Total cost: {np.sum(instance.costs):.2f}")
    
    # Test feasibility
    print("\n" + "="*60)
    print("FEASIBILITY TEST")
    print("="*60)
    
    if PYSAT_AVAILABLE:
        solver = ReserveDesignSATSolver(instance, 'glucose4', verbose=True)
        is_sat, selected_sites, stats = solver.solve()
        
        if is_sat:
            evaluation = instance.evaluate_solution(selected_sites)
            print(f"\nSolution evaluation:")
            print(f"  Selected sites: {selected_sites}")
            print(f"  Cost: {evaluation['total_cost']:.2f} / {instance.budget:.2f}")
            print(f"  Feasible: {evaluation['is_feasible']}")
            if not evaluation['is_feasible']:
                print(f"  Violations: {evaluation['violations']}")
    
    # Test optimization
    print("\n" + "="*60)
    print("OPTIMIZATION TEST")
    print("="*60)
    
    if PYSAT_AVAILABLE:
        solver = ReserveDesignSATSolver(instance, 'glucose4', verbose=True)
        is_opt, opt_sites, opt_cost, opt_stats = solver.solve_with_optimization(
            tolerance=0.1,
            max_iterations=20
        )
        
        if is_opt:
            evaluation = instance.evaluate_solution(opt_sites)
            print(f"\nOptimal solution:")
            print(f"  Selected sites: {opt_sites}")
            print(f"  Cost: {opt_cost:.2f}")
            print(f"  Feasible: {evaluation['is_feasible']}")
    
    # Test with Z3 if available
    if Z3_AVAILABLE:
        print("\n" + "="*60)
        print("Z3 SMT SOLVER TEST")
        print("="*60)
        
        solver = ReserveDesignSATSolver(instance, 'z3', verbose=True)
        is_sat, selected_sites, stats = solver.solve()
        
        if is_sat:
            evaluation = instance.evaluate_solution(selected_sites)
            print(f"\nZ3 Solution:")
            print(f"  Selected sites: {selected_sites}")
            print(f"  Cost: {evaluation['total_cost']:.2f}")
            print(f"  Feasible: {evaluation['is_feasible']}")
