"""
Utility Script to Check Benchmark Cache Status

This script provides a comprehensive overview of the benchmark results cache,
showing which configurations have been completed and which still need runs.
"""

import os
import sys
import json
from .benchmark_cache import BenchmarkCache

# Import configurations from each benchmark script
sys.path.insert(0, os.path.dirname(__file__))

# Default configurations (you can override these)
DEFAULT_CONFIGS = {
    'BQUBO': [5, 19, 40, 72, 109, 153],
    'NLD': [5, 19, 72, 279, 1096, 1535],
    'NLN': [5, 19, 72, 279, 1096, 1535],
    'LQ': [5, 19, 72, 279, 1096, 1535]
}

DEFAULT_NUM_RUNS = {
    'BQUBO': 5,
    'NLD': 5,
    'NLN': 5,
    'LQ': 5
}


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*100}")
    print(f"{title:^100}")
    print(f"{'='*100}")


def print_section(title):
    """Print a section divider."""
    print(f"\n{'-'*100}")
    print(f"{title}")
    print(f"{'-'*100}")


def print_cache_overview(cache):
    """Print an overview of all benchmark caches."""
    print_header("BENCHMARK CACHE STATUS OVERVIEW")
    
    total_results = 0
    total_needed = 0
    
    for benchmark_type in cache.BENCHMARK_TYPES:
        configs = DEFAULT_CONFIGS.get(benchmark_type, [])
        target_runs = DEFAULT_NUM_RUNS.get(benchmark_type, 5)
        
        print_section(f"{benchmark_type} Benchmark (Target: {target_runs} runs per config)")
        
        benchmark_total = 0
        benchmark_needed = 0
        
        for n_farms in configs:
            line_parts = [f"  Config {n_farms:5d} farms:"]
            
            solver_statuses = []
            for solver in cache.SOLVER_TYPES[benchmark_type]:
                existing = cache.get_existing_runs(benchmark_type, solver, n_farms)
                num_existing = len(existing)
                needs_runs = max(0, target_runs - num_existing)
                
                benchmark_total += num_existing
                benchmark_needed += needs_runs
                
                if needs_runs == 0:
                    status = f"✓"
                else:
                    status = f"⚠"
                
                solver_statuses.append(f"{solver:8s} {num_existing}/{target_runs} {status}")
            
            print(line_parts[0] + "  |  ".join(solver_statuses))
        
        total_results += benchmark_total
        total_needed += benchmark_needed
        
        print(f"\n  Summary: {benchmark_total} results cached, {benchmark_needed} more needed")
    
    print_header("OVERALL SUMMARY")
    print(f"  Total cached results: {total_results}")
    print(f"  Total runs still needed: {total_needed}")
    print(f"  Completion rate: {total_results / (total_results + total_needed) * 100:.1f}%")


def print_detailed_status(cache, benchmark_type):
    """Print detailed status for a specific benchmark."""
    if benchmark_type not in cache.BENCHMARK_TYPES:
        print(f"Error: Invalid benchmark type '{benchmark_type}'")
        return
    
    configs = DEFAULT_CONFIGS.get(benchmark_type, [])
    target_runs = DEFAULT_NUM_RUNS.get(benchmark_type, 5)
    
    print_header(f"DETAILED STATUS: {benchmark_type} Benchmark")
    
    for n_farms in configs:
        print_section(f"Configuration: {n_farms} farms (Target: {target_runs} runs)")
        
        all_complete = True
        
        for solver in cache.SOLVER_TYPES[benchmark_type]:
            existing_runs = cache.get_existing_runs(benchmark_type, solver, n_farms)
            num_existing = len(existing_runs)
            needs_runs = max(0, target_runs - num_existing)
            
            status_icon = "✓" if needs_runs == 0 else "⚠"
            status_text = "Complete" if needs_runs == 0 else f"Need {needs_runs} more"
            
            print(f"  {status_icon} {solver:12s}: {num_existing:2d}/{target_runs} runs  |  {status_text}")
            
            if needs_runs > 0:
                all_complete = False
                # Show which specific run numbers are missing
                needed_run_nums = sorted(set(range(1, target_runs + 1)) - set(existing_runs))
                print(f"       Missing runs: {needed_run_nums}")
        
        if all_complete:
            print(f"\n  ✅ All solvers complete for this configuration!")
        else:
            print(f"\n  ⚠️  Some runs still needed")


def export_cache_summary(cache, output_file='cache_summary.json'):
    """Export cache summary to JSON file."""
    summary = {
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'benchmarks': {}
    }
    
    for benchmark_type in cache.BENCHMARK_TYPES:
        configs = DEFAULT_CONFIGS.get(benchmark_type, [])
        target_runs = DEFAULT_NUM_RUNS.get(benchmark_type, 5)
        
        benchmark_summary = cache.get_cache_summary(benchmark_type, configs)
        benchmark_summary['target_runs'] = target_runs
        
        # Calculate completion statistics
        total_expected = len(configs) * len(cache.SOLVER_TYPES[benchmark_type]) * target_runs
        total_actual = benchmark_summary['total_results']
        
        benchmark_summary['completion_rate'] = (total_actual / total_expected * 100) if total_expected > 0 else 0
        benchmark_summary['runs_needed'] = total_expected - total_actual
        
        summary['benchmarks'][benchmark_type] = benchmark_summary
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Cache summary exported to: {output_file}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Check benchmark cache status',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_benchmark_cache.py                    # Show overview of all benchmarks
  python check_benchmark_cache.py --detailed NLN     # Show detailed status for NLN
  python check_benchmark_cache.py --export           # Export summary to JSON
  python check_benchmark_cache.py --detailed all     # Show detailed status for all benchmarks
        """
    )
    
    parser.add_argument('--detailed', '-d', 
                       help='Show detailed status for specific benchmark (BQUBO, NLD, NLN, LQ, or "all")')
    parser.add_argument('--export', '-e', action='store_true',
                       help='Export cache summary to JSON file')
    parser.add_argument('--output', '-o', default='cache_summary.json',
                       help='Output filename for export (default: cache_summary.json)')
    
    args = parser.parse_args()
    
    # Initialize cache
    cache = BenchmarkCache()
    
    if args.detailed:
        if args.detailed.upper() == 'ALL':
            for benchmark_type in cache.BENCHMARK_TYPES:
                print_detailed_status(cache, benchmark_type)
        else:
            print_detailed_status(cache, args.detailed.upper())
    else:
        print_cache_overview(cache)
    
    if args.export:
        export_cache_summary(cache, args.output)


if __name__ == "__main__":
    main()
