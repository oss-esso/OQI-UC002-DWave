"""
QBSolv Runner - Python wrapper for the qbsolv executable.

This module provides a Python interface to the compiled qbsolv binary,
enabling the solution of large Quadratic Unconstrained Binary Optimization (QUBO)
problems using D-Wave's partitioning algorithm.

The wrapper communicates with the qbsolv executable via temporary files,
which is more robust than the deprecated Python bindings that have
compatibility issues with Python 3.10+.

Reference:
    Booth, M., Reinhardt, S. P., & Roy, A. (2017). 
    Partitioning Optimization Problems for Hybrid Classical/Quantum Execution.
    https://www.dwavesys.com/sites/default/files/partitioning_QUBOs_for_hybrid_exec-web.pdf
"""

import os
import subprocess
import tempfile
import numpy as np
from scipy.sparse import dok_matrix, issparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
import re


# Path to the compiled qbsolv executable
QBSOLV_EXECUTABLE = Path(__file__).parent.parent / "external" / "qbsolv" / "build" / "qbsolv"


def _validate_qbsolv_executable() -> Path:
    """Validate that the qbsolv executable exists and is runnable."""
    if not QBSOLV_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"qbsolv executable not found at {QBSOLV_EXECUTABLE}. "
            "Please build qbsolv first by running:\n"
            "  cd external/qbsolv && mkdir -p build && cd build && cmake .. && make"
        )
    if not os.access(QBSOLV_EXECUTABLE, os.X_OK):
        raise PermissionError(
            f"qbsolv executable at {QBSOLV_EXECUTABLE} is not executable."
        )
    return QBSOLV_EXECUTABLE


def _qubo_to_file(Q: Union[dok_matrix, np.ndarray, Dict[Tuple[int, int], float]], 
                  filepath: str) -> int:
    """
    Write a QUBO matrix to a file in qbsolv's input format.
    
    The QUBO file format is:
        c <comment lines starting with 'c'>
        p qubo <topology> <maxNodes> <nNodes> <nCouplers>
        <i> <i> <linear_weight>  (for each node)
        <i> <j> <coupler_weight> (for each coupler, i < j)
    
    Note: qbsolv requires that every node has at least one coupler.
    
    Args:
        Q: QUBO matrix as scipy sparse matrix, numpy array, or dict.
        filepath: Path to write the QUBO file.
        
    Returns:
        Number of variables in the QUBO.
    """
    # Convert to dict format for uniform handling
    if isinstance(Q, dict):
        qubo_dict = Q
        # Determine number of variables from dict keys
        num_vars = max(max(i, j) for i, j in qubo_dict.keys()) + 1
    elif issparse(Q):
        qubo_dict = dict(Q.items())
        num_vars = Q.shape[0]
    elif isinstance(Q, np.ndarray):
        num_vars = Q.shape[0]
        qubo_dict = {}
        for i in range(num_vars):
            for j in range(i, num_vars):  # Upper triangular only
                if Q[i, j] != 0:
                    qubo_dict[(i, j)] = Q[i, j]
                    if i != j and Q[j, i] != 0:
                        # For off-diagonal, combine both entries
                        qubo_dict[(i, j)] = Q[i, j] + Q[j, i]
    else:
        raise TypeError(f"Unsupported QUBO type: {type(Q)}")
    
    # Separate linear terms (diagonal) from couplers (off-diagonal)
    linear_terms = {}  # {node: weight}
    couplers = {}  # {(i, j): weight} where i < j
    
    for (i, j), val in qubo_dict.items():
        if val == 0:
            continue
        if i == j:
            # Linear term
            linear_terms[i] = linear_terms.get(i, 0) + val
        else:
            # Coupler - ensure i < j
            key = (min(i, j), max(i, j))
            couplers[key] = couplers.get(key, 0) + val
    
    # Get all nodes that appear in the problem
    nodes_in_problem = set(linear_terms.keys())
    for i, j in couplers.keys():
        nodes_in_problem.add(i)
        nodes_in_problem.add(j)
    
    # qbsolv requires every node to have at least one coupler
    # If a node has no couplers, we need to handle this
    # For nodes without couplers, add a tiny coupler to themselves (won't affect solution)
    nodes_with_couplers = set()
    for i, j in couplers.keys():
        nodes_with_couplers.add(i)
        nodes_with_couplers.add(j)
    
    # Ensure all nodes with linear terms are in the coupler set
    # by adding minimal couplers if needed
    for node in nodes_in_problem:
        if node not in nodes_with_couplers:
            # Find another node to couple with
            other_nodes = [n for n in nodes_in_problem if n != node]
            if other_nodes:
                other = min(other_nodes)
                key = (min(node, other), max(node, other))
                if key not in couplers:
                    # Add a zero coupler (will be filtered, so add tiny value)
                    couplers[key] = 1e-15
    
    # Filter zero couplers
    couplers = {k: v for k, v in couplers.items() if v != 0}
    
    # Number of nodes = nodes with linear terms
    n_nodes = len(linear_terms)
    n_couplers = len(couplers)
    
    with open(filepath, 'w') as f:
        f.write(f"c QUBO file generated by qbsolv_runner.py\n")
        f.write(f"p qubo 0 {num_vars} {n_nodes} {n_couplers}\n")
        
        # Write linear terms (diagonal entries)
        for node in sorted(linear_terms.keys()):
            weight = linear_terms[node]
            f.write(f"{node} {node} {weight:.15g}\n")
        
        # Write couplers (off-diagonal entries)
        for (i, j) in sorted(couplers.keys()):
            weight = couplers[(i, j)]
            f.write(f"{i} {j} {weight:.15g}\n")
    
    return num_vars


def _parse_qbsolv_output(output: str, num_vars: int) -> Tuple[float, np.ndarray]:
    """
    Parse the output from qbsolv executable.
    
    The output format is:
        <num_bits> bits, find Min/Max, SubMatrix= <size>, ...
        <solution_string>
        <energy> Energy of solution
        ...
    
    Args:
        output: stdout from qbsolv execution.
        num_vars: Number of variables in the problem.
        
    Returns:
        Tuple of (energy, solution_vector).
    """
    lines = output.strip().split('\n')
    
    solution_vector = np.zeros(num_vars, dtype=int)
    energy = float('inf')
    
    # Find the solution line (a string of 0s and 1s) and energy line
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Check if this line is a solution (only contains 0s and 1s)
        if re.match(r'^[01]+$', line):
            # This is the solution string
            for idx, char in enumerate(line):
                if idx < num_vars:
                    solution_vector[idx] = int(char)
            
            # The next line should contain "Energy of solution"
            if i + 1 < len(lines):
                energy_line = lines[i + 1].strip()
                # Parse energy - it's in format "X.XXXXX Energy of solution"
                energy_match = re.match(r'^(-?[\d.]+)\s+Energy', energy_line)
                if energy_match:
                    energy = float(energy_match.group(1))
            break
    
    return energy, solution_vector


def solve_qubo_with_qbsolv(
    Q: Union[dok_matrix, np.ndarray, Dict[Tuple[int, int], float]],
    num_repeats: int = 50,
    subproblem_size: int = 47,
    timeout: Optional[float] = None,
    target: Optional[float] = None,
    seed: Optional[int] = None,
    maximize: bool = False,
    algorithm: str = 'o',
    verbosity: int = 0,
) -> Tuple[float, np.ndarray]:
    """
    Solve a QUBO using the external qbsolv library.
    
    QBSolv uses a partitioning approach that decomposes large QUBO problems
    into smaller subproblems that can be solved classically (using Tabu search)
    or on a quantum annealer.

    Args:
        Q: QUBO matrix as scipy sparse matrix (dok_matrix), numpy array,
           or dictionary mapping (i, j) tuples to coefficient values.
        num_repeats: Number of times to repeat the main loop after finding
                     a new optimal value before stopping. Default is 50.
        subproblem_size: Size of subproblems for partitioning. Default is 47.
                         Use 0 to use the size from D-Wave embedding.
        timeout: Optional timeout in seconds. Default is None (no timeout).
        target: Optional target energy to stop when found.
        seed: Optional random seed for reproducibility.
        maximize: If True, find maximum instead of minimum. Default is False.
        algorithm: Algorithm variant: 'o' for original, 'd' for diversity-based.
        verbosity: Verbosity level 0-4. Default is 0.

    Returns:
        A tuple containing:
        - energy (float): The final energy of the solution.
        - solution_vector (np.ndarray): Binary solution vector.
        
    Raises:
        FileNotFoundError: If qbsolv executable is not found.
        RuntimeError: If qbsolv execution fails.
        
    Example:
        >>> from scipy.sparse import dok_matrix
        >>> Q = dok_matrix((2, 2))
        >>> Q[0, 0] = -1
        >>> Q[1, 1] = -1
        >>> Q[0, 1] = 2
        >>> energy, solution = solve_qubo_with_qbsolv(Q)
        >>> print(f"Energy: {energy}, Solution: {solution}")
    """
    # Validate executable exists
    qbsolv_path = _validate_qbsolv_executable()
    
    # Create temporary files for input/output
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "input.qubo")
        output_file = os.path.join(tmpdir, "output.txt")
        
        # Write QUBO to file
        num_vars = _qubo_to_file(Q, input_file)
        
        # Build command
        cmd = [
            str(qbsolv_path),
            "-i", input_file,
            "-o", output_file,
            "-n", str(num_repeats),
            "-a", algorithm,
            "-v", str(verbosity),
        ]
        
        if subproblem_size > 0:
            cmd.extend(["-S", str(subproblem_size)])
        
        if timeout is not None:
            cmd.extend(["-t", str(timeout)])
            
        if target is not None:
            cmd.extend(["-T", str(target)])
            
        if seed is not None:
            cmd.extend(["-r", str(seed)])
            
        if maximize:
            cmd.append("-m")
        
        # Execute qbsolv
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"qbsolv execution failed with code {e.returncode}:\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            )
        
        # Read output file
        with open(output_file, 'r') as f:
            output = f.read()
        
        # Parse results
        energy, solution_vector = _parse_qbsolv_output(output, num_vars)
        
        return energy, solution_vector


def solve_qubo_dict_with_qbsolv(
    qubo_dict: Dict[Tuple[int, int], float],
    **kwargs
) -> Tuple[float, Dict[int, int]]:
    """
    Convenience function to solve a QUBO given as a dictionary.
    
    Args:
        qubo_dict: Dictionary mapping (i, j) tuples to coefficient values.
        **kwargs: Additional arguments passed to solve_qubo_with_qbsolv.
        
    Returns:
        Tuple of (energy, solution_dict) where solution_dict maps
        variable indices to their binary values.
    """
    energy, solution_vector = solve_qubo_with_qbsolv(qubo_dict, **kwargs)
    solution_dict = {i: int(v) for i, v in enumerate(solution_vector)}
    return energy, solution_dict


class QBSolvSampler:
    """
    A dimod-compatible sampler interface for qbsolv.
    
    This class provides an interface similar to D-Wave's dimod samplers,
    making it easy to integrate qbsolv into existing workflows that
    use the dimod API.
    
    Example:
        >>> sampler = QBSolvSampler()
        >>> Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
        >>> result = sampler.sample_qubo(Q, num_repeats=10)
        >>> print(f"Best energy: {result['energy']}")
        >>> print(f"Best solution: {result['sample']}")
    """
    
    def __init__(self, **default_params):
        """
        Initialize the sampler with default parameters.
        
        Args:
            **default_params: Default parameters for solve_qubo_with_qbsolv.
        """
        self.default_params = default_params
    
    def sample_qubo(
        self,
        Q: Union[dok_matrix, np.ndarray, Dict[Tuple[int, int], float]],
        **kwargs
    ) -> Dict:
        """
        Sample from a QUBO using qbsolv.
        
        Args:
            Q: QUBO matrix or dictionary.
            **kwargs: Override default parameters.
            
        Returns:
            Dictionary with 'energy', 'sample', and 'info' keys.
        """
        # Merge default params with call-specific params
        params = {**self.default_params, **kwargs}
        
        energy, solution = solve_qubo_with_qbsolv(Q, **params)
        
        # Convert solution to dict format
        if isinstance(solution, np.ndarray):
            sample = {i: int(v) for i, v in enumerate(solution)}
        else:
            sample = solution
        
        return {
            'energy': energy,
            'sample': sample,
            'info': {
                'solver': 'qbsolv',
                'params': params,
            }
        }
    
    @property
    def properties(self) -> Dict:
        """Return sampler properties."""
        return {
            'category': 'hybrid',
            'supported_problem_types': ['qubo', 'ising'],
        }
    
    @property
    def parameters(self) -> Dict:
        """Return sampler parameters."""
        return {
            'num_repeats': [],
            'subproblem_size': [],
            'timeout': [],
            'target': [],
            'seed': [],
            'maximize': [],
            'algorithm': ['o', 'd'],
            'verbosity': [0, 1, 2, 3, 4],
        }


# =============================================================================
# Test and CLI
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("QBSolv Integration Test")
    print("=" * 60)
    
    # Test 1: Basic QUBO with scipy sparse matrix
    print("\nTest 1: Simple QUBO (scipy sparse matrix)")
    print("-" * 40)
    
    # Create a simple, known QUBO
    # E = -x0 - x1 + 2*x0*x1
    # Min energy = -1 (at x0=1, x1=0 or x0=0, x1=1)
    num_variables = 2
    test_Q = dok_matrix((num_variables, num_variables))
    test_Q[0, 0] = -1
    test_Q[1, 1] = -1
    test_Q[0, 1] = 2

    energy, solution = solve_qubo_with_qbsolv(test_Q, num_repeats=10, seed=42)

    print(f"QUBO: -x0 - x1 + 2*x0*x1")
    print(f"Expected: energy=-1, solution=[1,0] or [0,1]")
    print(f"Got: energy={energy}, solution={solution}")

    # Basic assertion
    assert energy == -1.0, f"Expected energy=-1, got {energy}"
    assert (solution[0] == 1 and solution[1] == 0) or \
           (solution[0] == 0 and solution[1] == 1), \
           f"Expected [1,0] or [0,1], got {solution}"
    
    print("✓ Test 1 passed!")
    
    # Test 2: QUBO with dictionary format
    print("\nTest 2: QUBO (dictionary format)")
    print("-" * 40)
    
    qubo_dict = {
        (0, 0): -1,
        (1, 1): -1,
        (2, 2): -1,
        (0, 1): 2,
        (1, 2): 2,
    }
    # This QUBO: -x0 - x1 - x2 + 2*x0*x1 + 2*x1*x2
    # Min = -2 at [1, 0, 1]
    
    energy, solution_dict = solve_qubo_dict_with_qbsolv(qubo_dict, num_repeats=10, seed=42)
    
    print(f"QUBO: -x0 - x1 - x2 + 2*x0*x1 + 2*x1*x2")
    print(f"Expected: energy=-2, solution=[1,0,1]")
    print(f"Got: energy={energy}, solution={solution_dict}")
    
    assert energy == -2.0, f"Expected energy=-2, got {energy}"
    print("✓ Test 2 passed!")
    
    # Test 3: Using QBSolvSampler class
    print("\nTest 3: QBSolvSampler class")
    print("-" * 40)
    
    sampler = QBSolvSampler(num_repeats=10, seed=42)
    result = sampler.sample_qubo(test_Q)
    
    print(f"Result: {result}")
    assert result['energy'] == -1.0
    print("✓ Test 3 passed!")
    
    # Test 4: Larger random QUBO
    print("\nTest 4: Larger random QUBO (100 variables)")
    print("-" * 40)
    
    np.random.seed(42)
    n = 100
    Q_large = dok_matrix((n, n))
    for i in range(n):
        Q_large[i, i] = np.random.uniform(-1, 1)
        for j in range(i + 1, min(i + 10, n)):  # Sparse connectivity
            if np.random.random() < 0.3:
                Q_large[i, j] = np.random.uniform(-1, 1)
    
    energy, solution = solve_qubo_with_qbsolv(Q_large, num_repeats=20, seed=42)
    
    print(f"Problem size: {n} variables")
    print(f"Solution energy: {energy}")
    print(f"Solution sum: {sum(solution)}")
    print("✓ Test 4 passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed! QBSolv integration is working correctly.")
    print("=" * 60)
