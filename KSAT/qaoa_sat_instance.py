"""
QAOA SAT Paper Random Instance Generator

Reproduces the random k-SAT instance generation methodology from:
"Applying the quantum approximate optimization algorithm to general 
constraint satisfaction problems" by Boulebnane, Ciudad-Alañón, Mineh, 
Montanaro, and Vaishnav (2024), arXiv:2411.17442

This generator creates random k-SAT instances used for QAOA benchmarking,
following the planted and random instance generation procedures from the paper.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import random


@dataclass
class KSATInstance:
    """
    A k-SAT instance in CNF form
    
    Attributes:
        n: Number of variables
        m: Number of clauses
        k: Variables per clause
        clauses: List of clauses, each clause is a list of literals
                 Positive literal i represented as i (1-indexed)
                 Negative literal ¬i represented as -i
        alpha: Clause-to-variable ratio (m/n)
        is_planted: Whether instance has planted solution
        planted_solution: The planted assignment (if planted)
    """
    n: int  # Number of variables
    m: int  # Number of clauses
    k: int  # Variables per clause
    clauses: List[List[int]]  # CNF clauses
    alpha: float  # m/n ratio
    is_planted: bool = False
    planted_solution: Optional[List[bool]] = None
    
    def evaluate_assignment(self, assignment: List[bool]) -> Tuple[bool, int]:
        """
        Evaluate an assignment
        
        Args:
            assignment: Boolean assignment for variables (0-indexed)
        
        Returns:
            (is_sat, num_satisfied): Whether SAT and number of satisfied clauses
        """
        num_satisfied = 0
        for clause in self.clauses:
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1  # Convert to 0-indexed
                var_value = assignment[var_idx]
                # Positive literal: check if variable is True
                # Negative literal: check if variable is False
                if (lit > 0 and var_value) or (lit < 0 and not var_value):
                    satisfied = True
                    break
            if satisfied:
                num_satisfied += 1
        
        is_sat = (num_satisfied == self.m)
        return is_sat, num_satisfied
    
    def to_dimacs_cnf(self) -> str:
        """Export to DIMACS CNF format"""
        lines = []
        lines.append(f"p cnf {self.n} {self.m}")
        for clause in self.clauses:
            lines.append(" ".join(map(str, clause)) + " 0")
        return "\n".join(lines)


def generate_random_ksat(
    n: int,
    k: int,
    alpha: float,
    seed: Optional[int] = None
) -> KSATInstance:
    """
    Generate random k-SAT instance (Uniform Random k-SAT model)
    
    This follows the standard random k-SAT model used in QAOA benchmarks:
    - Each clause selects k distinct variables uniformly at random
    - Each variable appears negated with probability 0.5
    - Total of m = α·n clauses
    
    This model exhibits a phase transition around:
    - k=3: α_c ≈ 4.27 (satisfiable below, unsatisfiable above)
    - k=4: α_c ≈ 9.93
    - k=5: α_c ≈ 21.12
    
    Args:
        n: Number of variables
        k: Variables per clause
        alpha: Clause-to-variable ratio (m/n)
        seed: Random seed
    
    Returns:
        KSATInstance
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    m = int(alpha * n)
    clauses = []
    
    for _ in range(m):
        # Select k distinct variables
        variables = random.sample(range(1, n + 1), k)
        
        # Negate each with probability 0.5
        clause = []
        for var in variables:
            if random.random() < 0.5:
                clause.append(-var)
            else:
                clause.append(var)
        
        clauses.append(clause)
    
    return KSATInstance(
        n=n,
        m=m,
        k=k,
        clauses=clauses,
        alpha=alpha,
        is_planted=False
    )


def generate_planted_ksat(
    n: int,
    k: int,
    alpha: float,
    seed: Optional[int] = None
) -> KSATInstance:
    """
    Generate planted k-SAT instance with guaranteed satisfying assignment
    
    Methodology from QAOA SAT papers:
    1. Generate random satisfying assignment
    2. Generate m clauses that are satisfied by this assignment
    3. For each clause, select k distinct variables
    4. For each variable in clause, set literal polarity to satisfy assignment
    
    Planted instances are always satisfiable (by construction) but can still
    be hard for solvers if they don't know the planted solution.
    
    Args:
        n: Number of variables
        k: Variables per clause
        alpha: Clause-to-variable ratio
        seed: Random seed
    
    Returns:
        KSATInstance with planted solution
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    m = int(alpha * n)
    
    # Generate random planted solution
    planted_solution = [random.choice([True, False]) for _ in range(n)]
    
    clauses = []
    for _ in range(m):
        # Select k distinct variables
        variables = random.sample(range(1, n + 1), k)
        
        # Create clause satisfied by planted solution
        # At least one literal must be satisfied
        # We ensure this by making at least one literal agree with planted solution
        clause = []
        for i, var in enumerate(variables):
            var_idx = var - 1
            var_value = planted_solution[var_idx]
            
            if i == 0:
                # First literal always agrees with planted solution
                if var_value:
                    clause.append(var)  # x_i when True
                else:
                    clause.append(-var)  # ¬x_i when False
            else:
                # Other literals can be random
                if random.random() < 0.5:
                    clause.append(var)
                else:
                    clause.append(-var)
        
        clauses.append(clause)
    
    return KSATInstance(
        n=n,
        m=m,
        k=k,
        clauses=clauses,
        alpha=alpha,
        is_planted=True,
        planted_solution=planted_solution
    )


def generate_hard_random_ksat(
    n: int,
    k: int = 3,
    seed: Optional[int] = None
) -> KSATInstance:
    """
    Generate k-SAT instance at the phase transition (hardest instances)
    
    For k=3: Uses α ≈ 4.27 (threshold)
    For k=4: Uses α ≈ 9.93
    For k=5: Uses α ≈ 21.12
    
    These instances are in the "hard" region where SAT solvers struggle.
    
    Args:
        n: Number of variables
        k: Variables per clause
        seed: Random seed
    
    Returns:
        KSATInstance at phase transition
    """
    # Phase transition points (empirically determined)
    phase_transitions = {
        3: 4.27,
        4: 9.93,
        5: 21.12,
        6: 43.37
    }
    
    alpha = phase_transitions.get(k, 2.0 ** k)  # Approximate for k>6
    
    return generate_random_ksat(n, k, alpha, seed)


def generate_qaoa_benchmark_suite(
    n_values: List[int] = [20, 30, 40, 50],
    k: int = 3,
    alpha_values: List[float] = [3.0, 4.27, 5.0],
    instances_per_config: int = 10,
    seed: Optional[int] = None
) -> List[Tuple[str, KSATInstance]]:
    """
    Generate suite of instances for QAOA benchmarking
    
    Creates a comprehensive benchmark set following QAOA SAT paper methodology:
    - Various problem sizes (n)
    - Various clause densities (α)
    - Multiple instances per configuration for statistical analysis
    
    Args:
        n_values: List of variable counts
        k: Variables per clause
        alpha_values: List of clause-to-variable ratios
        instances_per_config: Number of instances per (n, α) pair
        seed: Base random seed
    
    Returns:
        List of (name, instance) pairs
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    benchmark_suite = []
    
    for n in n_values:
        for alpha in alpha_values:
            for trial in range(instances_per_config):
                # Generate random instance
                instance_seed = None if seed is None else seed + len(benchmark_suite)
                instance = generate_random_ksat(n, k, alpha, instance_seed)
                name = f"random_k{k}_n{n}_a{alpha:.2f}_t{trial}"
                benchmark_suite.append((name, instance))
                
                # Also generate planted version
                planted = generate_planted_ksat(n, k, alpha, instance_seed + 10000)
                planted_name = f"planted_k{k}_n{n}_a{alpha:.2f}_t{trial}"
                benchmark_suite.append((planted_name, planted))
    
    return benchmark_suite


if __name__ == "__main__":
    print("="*70)
    print("QAOA SAT PAPER INSTANCE GENERATOR")
    print("="*70)
    
    # Example 1: Random 3-SAT instance
    print("\n1. Random 3-SAT (n=20, α=4.27 - phase transition)")
    instance_random = generate_hard_random_ksat(n=20, k=3, seed=42)
    print(f"   Variables: {instance_random.n}")
    print(f"   Clauses: {instance_random.m}")
    print(f"   k: {instance_random.k}")
    print(f"   α (m/n): {instance_random.alpha:.3f}")
    print(f"   First 5 clauses: {instance_random.clauses[:5]}")
    
    # Test random assignment
    random_assignment = [random.choice([True, False]) for _ in range(instance_random.n)]
    is_sat, num_sat = instance_random.evaluate_assignment(random_assignment)
    print(f"   Random assignment satisfies {num_sat}/{instance_random.m} clauses")
    
    # Example 2: Planted 3-SAT
    print("\n2. Planted 3-SAT (n=20, α=4.27)")
    instance_planted = generate_planted_ksat(n=20, k=3, alpha=4.27, seed=123)
    print(f"   Variables: {instance_planted.n}")
    print(f"   Clauses: {instance_planted.m}")
    print(f"   Planted solution exists: {instance_planted.is_planted}")
    
    # Test planted solution
    is_sat, num_sat = instance_planted.evaluate_assignment(instance_planted.planted_solution)
    print(f"   Planted solution satisfies {num_sat}/{instance_planted.m} clauses: {is_sat}")
    
    # Test random assignment on planted instance
    random_assignment = [random.choice([True, False]) for _ in range(instance_planted.n)]
    is_sat_random, num_sat_random = instance_planted.evaluate_assignment(random_assignment)
    print(f"   Random assignment satisfies {num_sat_random}/{instance_planted.m} clauses: {is_sat_random}")
    
    # Example 3: DIMACS export
    print("\n3. DIMACS CNF Format")
    small_instance = generate_random_ksat(n=5, k=3, alpha=4.0, seed=999)
    print(small_instance.to_dimacs_cnf())
    
    # Example 4: Benchmark suite
    print("\n4. QAOA Benchmark Suite")
    suite = generate_qaoa_benchmark_suite(
        n_values=[20, 30],
        k=3,
        alpha_values=[4.0, 4.27],
        instances_per_config=2,
        seed=42
    )
    print(f"   Generated {len(suite)} instances:")
    for name, inst in suite[:6]:
        print(f"     {name}: n={inst.n}, m={inst.m}, planted={inst.is_planted}")
    
    print("\n✓ QAOA SAT instances generated successfully")
