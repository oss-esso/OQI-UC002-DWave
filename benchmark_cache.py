"""
Benchmark Results Cache Manager

This module manages the storage and retrieval of benchmark results to enable:
1. Incremental saving of CQM models and solver solutions
2. Smart caching to avoid redundant runs
3. Organized folder hierarchy by benchmark type and solver
4. Tracking of completed runs per configuration

Folder Structure:
    Benchmarks/
        {BQUBO,NLD,NLN,LQ}/
            CQM/
                config_{n_farms}_run_{run_num}.json
            PuLP/
                config_{n_farms}_run_{run_num}.json
            Pyomo/
                config_{n_farms}_run_{run_num}.json
            DWave/
                config_{n_farms}_run_{run_num}.json
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


class BenchmarkCache:
    """Manages benchmark result caching and storage."""
    
    # Valid benchmark types
    BENCHMARK_TYPES = ['BQUBO', 'NLD', 'NLN', 'LQ']
    
    # Valid solver types per benchmark
    SOLVER_TYPES = {
        'BQUBO': ['CQM', 'PuLP', 'DWave'],
        'NLD': ['CQM', 'PuLP', 'Pyomo', 'DWave'],
        'NLN': ['CQM', 'PuLP', 'Pyomo', 'DWave'],
        'LQ': ['CQM', 'PuLP', 'Pyomo', 'DWave']
    }
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the benchmark cache manager.
        
        Args:
            base_dir: Base directory for benchmarks (default: ./Benchmarks)
        """
        if base_dir is None:
            # Get directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(script_dir, "Benchmarks")
        
        self.base_dir = Path(base_dir)
        self._ensure_folder_structure()
    
    def _ensure_folder_structure(self):
        """Create the complete folder structure for all benchmarks and solvers."""
        for benchmark_type in self.BENCHMARK_TYPES:
            benchmark_path = self.base_dir / benchmark_type
            
            # Create solver folders
            for solver in self.SOLVER_TYPES[benchmark_type]:
                solver_path = benchmark_path / solver
                solver_path.mkdir(parents=True, exist_ok=True)
    
    def _get_result_path(self, benchmark_type: str, solver: str, 
                        n_farms: int, run_num: int) -> Path:
        """
        Get the file path for a specific result.
        
        Args:
            benchmark_type: Type of benchmark (BQUBO, NLD, NLN, LQ)
            solver: Solver name (CQM, PuLP, Pyomo, DWave)
            n_farms: Number of farms in configuration
            run_num: Run number (1-based)
        
        Returns:
            Path to the result file
        """
        self._validate_benchmark_solver(benchmark_type, solver)
        
        filename = f"config_{n_farms}_run_{run_num}.json"
        return self.base_dir / benchmark_type / solver / filename
    
    def _validate_benchmark_solver(self, benchmark_type: str, solver: str):
        """Validate benchmark type and solver combination."""
        if benchmark_type not in self.BENCHMARK_TYPES:
            raise ValueError(f"Invalid benchmark type: {benchmark_type}. "
                           f"Must be one of {self.BENCHMARK_TYPES}")
        
        if solver not in self.SOLVER_TYPES[benchmark_type]:
            raise ValueError(f"Invalid solver '{solver}' for benchmark '{benchmark_type}'. "
                           f"Must be one of {self.SOLVER_TYPES[benchmark_type]}")
    
    def get_existing_runs(self, benchmark_type: str, solver: str, 
                         n_farms: int) -> List[int]:
        """
        Get list of existing run numbers for a configuration.
        
        Args:
            benchmark_type: Type of benchmark
            solver: Solver name
            n_farms: Number of farms
        
        Returns:
            Sorted list of run numbers that exist
        """
        self._validate_benchmark_solver(benchmark_type, solver)
        
        solver_path = self.base_dir / benchmark_type / solver
        pattern = f"config_{n_farms}_run_*.json"
        
        run_numbers = []
        for file_path in solver_path.glob(pattern):
            # Extract run number from filename
            filename = file_path.stem  # e.g., "config_5_run_3"
            parts = filename.split('_')
            if len(parts) >= 4 and parts[-2] == 'run':
                try:
                    run_num = int(parts[-1])
                    run_numbers.append(run_num)
                except ValueError:
                    continue
        
        return sorted(run_numbers)
    
    def get_runs_needed(self, benchmark_type: str, n_farms: int, 
                       target_runs: int) -> Dict[str, List[int]]:
        """
        Determine which runs are still needed for each solver.
        
        Args:
            benchmark_type: Type of benchmark
            n_farms: Number of farms
            target_runs: Target number of runs (NUM_RUNS)
        
        Returns:
            Dictionary mapping solver names to lists of run numbers still needed
        """
        if benchmark_type not in self.BENCHMARK_TYPES:
            raise ValueError(f"Invalid benchmark type: {benchmark_type}")
        
        runs_needed = {}
        
        for solver in self.SOLVER_TYPES[benchmark_type]:
            existing = set(self.get_existing_runs(benchmark_type, solver, n_farms))
            all_runs = set(range(1, target_runs + 1))
            needed = sorted(all_runs - existing)
            
            if needed:
                runs_needed[solver] = needed
        
        return runs_needed
    
    def save_result(self, benchmark_type: str, solver: str, n_farms: int, 
                   run_num: int, result_data: Dict[str, Any], 
                   cqm_data: Optional[Dict] = None):
        """
        Save a benchmark result to disk.
        
        Args:
            benchmark_type: Type of benchmark
            solver: Solver name
            n_farms: Number of farms
            run_num: Run number
            result_data: Dictionary containing solution and timing data
            cqm_data: Optional CQM model data (for CQM solver only)
        """
        self._validate_benchmark_solver(benchmark_type, solver)
        
        # Build complete result structure
        result = {
            'metadata': {
                'benchmark_type': benchmark_type,
                'solver': solver,
                'n_farms': n_farms,
                'run_number': run_num,
                'timestamp': datetime.now().isoformat(),
            },
            'result': result_data
        }
        
        # Add CQM data if provided
        if cqm_data is not None:
            result['cqm'] = cqm_data
        
        # Save to file
        file_path = self._get_result_path(benchmark_type, solver, n_farms, run_num)
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"✓ Saved {solver} result: config_{n_farms}_run_{run_num}")
    
    def load_result(self, benchmark_type: str, solver: str, 
                   n_farms: int, run_num: int) -> Optional[Dict]:
        """
        Load a benchmark result from disk.
        
        Args:
            benchmark_type: Type of benchmark
            solver: Solver name
            n_farms: Number of farms
            run_num: Run number
        
        Returns:
            Result dictionary or None if not found
        """
        file_path = self._get_result_path(benchmark_type, solver, n_farms, run_num)
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def get_all_results(self, benchmark_type: str, solver: str, 
                       n_farms: int) -> List[Dict]:
        """
        Load all results for a configuration.
        
        Args:
            benchmark_type: Type of benchmark
            solver: Solver name
            n_farms: Number of farms
        
        Returns:
            List of result dictionaries
        """
        run_numbers = self.get_existing_runs(benchmark_type, solver, n_farms)
        results = []
        
        for run_num in run_numbers:
            result = self.load_result(benchmark_type, solver, n_farms, run_num)
            if result:
                results.append(result)
        
        return results
    
    def get_cache_summary(self, benchmark_type: str, 
                         configurations: List[int]) -> Dict[str, Any]:
        """
        Get a summary of cached results for a benchmark.
        
        Args:
            benchmark_type: Type of benchmark
            configurations: List of configuration sizes (n_farms)
        
        Returns:
            Dictionary with summary statistics
        """
        if benchmark_type not in self.BENCHMARK_TYPES:
            raise ValueError(f"Invalid benchmark type: {benchmark_type}")
        
        summary = {
            'benchmark_type': benchmark_type,
            'configurations': {},
            'total_results': 0
        }
        
        for n_farms in configurations:
            config_summary = {
                'n_farms': n_farms,
                'solvers': {}
            }
            
            for solver in self.SOLVER_TYPES[benchmark_type]:
                run_numbers = self.get_existing_runs(benchmark_type, solver, n_farms)
                config_summary['solvers'][solver] = {
                    'num_runs': len(run_numbers),
                    'run_numbers': run_numbers
                }
                summary['total_results'] += len(run_numbers)
            
            summary['configurations'][n_farms] = config_summary
        
        return summary
    
    def print_cache_status(self, benchmark_type: str, 
                          configurations: List[int], 
                          target_runs: int):
        """
        Print a human-readable cache status report.
        
        Args:
            benchmark_type: Type of benchmark
            configurations: List of configuration sizes
            target_runs: Target number of runs per config
        """
        print(f"\n{'='*80}")
        print(f"Cache Status: {benchmark_type} Benchmark")
        print(f"Target runs per configuration: {target_runs}")
        print(f"{'='*80}\n")
        
        for n_farms in configurations:
            print(f"Configuration: {n_farms} farms")
            print(f"{'-'*60}")
            
            all_complete = True
            for solver in self.SOLVER_TYPES[benchmark_type]:
                existing = self.get_existing_runs(benchmark_type, solver, n_farms)
                num_existing = len(existing)
                needs_runs = target_runs - num_existing
                
                status = "✓ Complete" if needs_runs <= 0 else f"⚠ Need {needs_runs} more"
                print(f"  {solver:12s}: {num_existing}/{target_runs} runs {status}")
                
                if needs_runs > 0:
                    all_complete = False
            
            if all_complete:
                print(f"  Status: ✓ All solvers complete for this configuration")
            else:
                print(f"  Status: ⚠ Incomplete - more runs needed")
            print()


def serialize_cqm(cqm) -> Dict[str, Any]:
    """
    Serialize a D-Wave CQM model to a JSON-compatible dictionary.
    
    Args:
        cqm: D-Wave CQM model
    
    Returns:
        Dictionary representation of the CQM
    """
    try:
        # Try to extract basic CQM information
        cqm_dict = {
            'num_variables': len(cqm.variables),
            'num_constraints': len(cqm.constraints),
            'variables': list(cqm.variables),
            'objective_type': str(type(cqm.objective).__name__),
            'constraints': {}
        }
        
        # Add constraint information
        for label, constraint in cqm.constraints.items():
            cqm_dict['constraints'][str(label)] = {
                'sense': str(constraint.sense),
                'rhs': float(constraint.rhs) if hasattr(constraint, 'rhs') else None
            }
        
        return cqm_dict
    except Exception as e:
        # If serialization fails, return minimal info
        return {
            'serialization_error': str(e),
            'type': str(type(cqm).__name__)
        }


def deserialize_cqm(cqm_dict: Dict[str, Any]):
    """
    Deserialize a CQM from dictionary format.
    Note: Full deserialization of CQM is complex and may not be needed.
    This function is a placeholder for future implementation.
    
    Args:
        cqm_dict: Dictionary representation
    
    Returns:
        CQM model (or None if not implemented)
    """
    # Full CQM deserialization would require reconstructing the entire model
    # For now, we just store the metadata
    return None


# Example usage
if __name__ == "__main__":
    # Initialize cache manager
    cache = BenchmarkCache()
    
    # Example: Check what runs exist for NLN benchmark
    print("Example: Checking NLN benchmark cache status")
    
    configs = [5, 19, 72, 279, 1096, 1535]
    cache.print_cache_status('NLN', configs, target_runs=5)
    
    # Example: Get runs needed for a specific config
    runs_needed = cache.get_runs_needed('NLN', 72, target_runs=5)
    print(f"\nRuns needed for 72 farms: {runs_needed}")
