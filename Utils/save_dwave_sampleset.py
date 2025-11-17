"""
Utility to save D-Wave SampleSet to pandas DataFrame with standardized naming.

This utility ensures all D-Wave sampleset data is preserved for future analysis
by converting to pandas DataFrame format and saving to appropriate directories.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional
from pathlib import Path


def save_sampleset_to_dataframe(
    sampleset,
    benchmark_type: str,
    scenario_type: str,
    solver_type: str,
    config_id: Optional[int] = None,
    run_id: Optional[int] = None,
    output_base_dir: Optional[str] = None
) -> str:
    """
    Save D-Wave SampleSet to pandas DataFrame with standardized naming.
    
    Args:
        sampleset: D-Wave SampleSet object
        benchmark_type: Type of benchmark (e.g., 'COMPREHENSIVE', 'LQ', 'NLN', 'PATCH', 'ROTATION', 'BQUBO')
        scenario_type: Scenario name (e.g., 'Farm', 'Patch')
        solver_type: Solver used (e.g., 'DWave', 'DWaveBQM', 'DWave_CQM')
        config_id: Configuration ID (optional, e.g., number of units)
        run_id: Run ID (optional)
        output_base_dir: Base output directory (defaults to project_root/Benchmarks)
        
    Returns:
        str: Path to saved CSV file
        
    Example:
        >>> save_sampleset_to_dataframe(
        ...     sampleset=my_sampleset,
        ...     benchmark_type='COMPREHENSIVE',
        ...     scenario_type='Patch',
        ...     solver_type='DWave',
        ...     config_id=10,
        ...     run_id=1
        ... )
        '/path/to/Benchmarks/COMPREHENSIVE/Patch_DWave/samplesets/comprehensive_Patch_DWave_config10_run1_20251117_143052.csv'
    """
    
    # Determine project root
    if output_base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_base_dir = os.path.join(project_root, 'Benchmarks')
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct directory path: Benchmarks/{benchmark_type}/{scenario_type}_{solver_type}/samplesets/
    solver_dir_name = f"{scenario_type}_{solver_type}"
    sampleset_dir = os.path.join(
        output_base_dir,
        benchmark_type,
        solver_dir_name,
        'samplesets'
    )
    
    # Create directory if it doesn't exist
    os.makedirs(sampleset_dir, exist_ok=True)
    
    # Construct filename
    # Format: {benchmark_lower}_{scenario}_{solver}_[config{id}][_run{id}]_{timestamp}.csv
    benchmark_lower = benchmark_type.lower()
    filename_parts = [benchmark_lower, scenario_type, solver_type]
    
    if config_id is not None:
        filename_parts.append(f"config{config_id}")
    
    if run_id is not None:
        filename_parts.append(f"run{run_id}")
    
    filename_parts.append(timestamp)
    filename = '_'.join(filename_parts) + '.csv'
    
    # Full file path
    filepath = os.path.join(sampleset_dir, filename)
    
    # Convert SampleSet to DataFrame
    # Note: to_pandas_dataframe() can return constraint info which causes shape issues
    # We'll use a more robust approach by manually extracting the data
    try:
        df = sampleset.to_pandas_dataframe()
    except Exception as e:
        # Fallback: manually construct DataFrame from sampleset
        print(f"Warning: to_pandas_dataframe() failed ({e}), using manual extraction")
        
        # Extract samples manually
        records = []
        for datum in sampleset.data(['sample', 'energy', 'num_occurrences', 'is_feasible']):
            record = {'energy': datum.energy, 'num_occurrences': datum.num_occurrences}
            
            # Add feasibility if available
            if hasattr(datum, 'is_feasible'):
                record['is_feasible'] = datum.is_feasible
            
            # Add all sample variables
            record.update(datum.sample)
            records.append(record)
        
        df = pd.DataFrame(records)
    
    # Add metadata columns at the beginning
    df.insert(0, 'benchmark_type', benchmark_type)
    df.insert(1, 'scenario_type', scenario_type)
    df.insert(2, 'solver_type', solver_type)
    
    if config_id is not None:
        df.insert(3, 'config_id', config_id)
    
    if run_id is not None:
        insert_pos = 4 if config_id is not None else 3
        df.insert(insert_pos, 'run_id', run_id)
    
    # Add timestamp at the end of metadata columns
    metadata_cols = 3 + (1 if config_id is not None else 0) + (1 if run_id is not None else 0)
    df.insert(metadata_cols, 'timestamp', timestamp)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    return filepath


def save_sampleset_from_benchmark(
    sampleset,
    benchmark_category: str,
    solver_dir: str,
    config_id: Optional[int] = None,
    run_id: Optional[int] = None,
    output_base_dir: Optional[str] = None
) -> str:
    """
    Convenience function that infers benchmark_type, scenario_type, and solver_type
    from the solver directory name.
    
    Args:
        sampleset: D-Wave SampleSet object
        benchmark_category: Benchmark category (e.g., 'COMPREHENSIVE', 'LQ', 'ROTATION')
        solver_dir: Solver directory name (e.g., 'Patch_DWave', 'Farm_DWaveBQM')
        config_id: Configuration ID (optional)
        run_id: Run ID (optional)
        output_base_dir: Base output directory (defaults to project_root/Benchmarks)
        
    Returns:
        str: Path to saved CSV file
        
    Example:
        >>> save_sampleset_from_benchmark(
        ...     sampleset=my_sampleset,
        ...     benchmark_category='COMPREHENSIVE',
        ...     solver_dir='Patch_DWave',
        ...     config_id=10,
        ...     run_id=1
        ... )
    """
    
    # Parse solver_dir to extract scenario and solver type
    # Expected format: {Scenario}_{Solver} or {Scenario}_{Solver1}_{Solver2}
    parts = solver_dir.split('_')
    
    if len(parts) >= 2:
        scenario_type = parts[0]
        solver_type = '_'.join(parts[1:])
    else:
        # Fallback
        scenario_type = solver_dir
        solver_type = 'DWave'
    
    return save_sampleset_to_dataframe(
        sampleset=sampleset,
        benchmark_type=benchmark_category,
        scenario_type=scenario_type,
        solver_type=solver_type,
        config_id=config_id,
        run_id=run_id,
        output_base_dir=output_base_dir
    )


def load_sampleset_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load a previously saved sampleset DataFrame.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    return pd.read_csv(filepath)


def list_saved_samplesets(
    benchmark_type: Optional[str] = None,
    scenario_type: Optional[str] = None,
    solver_type: Optional[str] = None,
    output_base_dir: Optional[str] = None
) -> list:
    """
    List all saved sampleset CSV files, optionally filtered.
    
    Args:
        benchmark_type: Filter by benchmark type (optional)
        scenario_type: Filter by scenario type (optional)
        solver_type: Filter by solver type (optional)
        output_base_dir: Base output directory (defaults to project_root/Benchmarks)
        
    Returns:
        list: List of file paths matching the criteria
    """
    
    # Determine base directory
    if output_base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_base_dir = os.path.join(project_root, 'Benchmarks')
    
    # Build search pattern
    if benchmark_type and scenario_type and solver_type:
        search_pattern = os.path.join(
            output_base_dir,
            benchmark_type,
            f"{scenario_type}_{solver_type}",
            'samplesets',
            '*.csv'
        )
    elif benchmark_type:
        search_pattern = os.path.join(
            output_base_dir,
            benchmark_type,
            '**',
            'samplesets',
            '*.csv'
        )
    else:
        search_pattern = os.path.join(
            output_base_dir,
            '**',
            'samplesets',
            '*.csv'
        )
    
    # Find files
    import glob
    files = glob.glob(search_pattern, recursive=True)
    
    return sorted(files)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("D-WAVE SAMPLESET SAVER UTILITY")
    print("=" * 80)
    print()
    print("This utility saves D-Wave SampleSet objects to pandas DataFrames")
    print("with standardized naming and directory structure.")
    print()
    print("Usage example:")
    print()
    print("    from Utils.save_dwave_sampleset import save_sampleset_to_dataframe")
    print()
    print("    # After getting a sampleset from D-Wave")
    print("    filepath = save_sampleset_to_dataframe(")
    print("        sampleset=sampleset,")
    print("        benchmark_type='COMPREHENSIVE',")
    print("        scenario_type='Patch',")
    print("        solver_type='DWave',")
    print("        config_id=10,")
    print("        run_id=1")
    print("    )")
    print("    print(f'Saved to: {filepath}')")
    print()
    print("Directory structure created:")
    print("    Benchmarks/")
    print("    └── COMPREHENSIVE/")
    print("        └── Patch_DWave/")
    print("            └── samplesets/")
    print("                └── comprehensive_Patch_DWave_config10_run1_20251117_143052.csv")
    print()
    print("Filename format:")
    print("    {benchmark}_{scenario}_{solver}_[config{N}]_[run{M}]_{timestamp}.csv")
    print()
    print("=" * 80)
