#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for Decomposed QPU (Strategic Problem Decomposition)

This script implements Alternative 2: Strategic Problem Decomposition
- Farm Scenario: Classical-only (Gurobi) for continuous optimization
- Patch Scenario: Quantum-only (low-level QPU) for binary optimization

The decomposition strategy:
1. Farm problems (continuous) → Solved with Gurobi (classical MINLP)
2. Patch problems (binary) → Converted to BQM → Solved with DWaveSampler (direct QPU)

This approach maximizes QPU utilization for problems that benefit from quantum annealing
while using classical solvers for problems better suited to traditional optimization.

IEEE Standard: Professional code with comprehensive documentation.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from benchmark_utils_decomposed import (
    run_single_benchmark,
    save_results,
    print_summary
)

# Default configurations
DEFAULT_CONFIGS = [25]  # 25 units for comprehensive testing with all 27 foods
DWAVE_TOKEN_PLACEHOLDER = 'YOUR_DWAVE_TOKEN_HERE'


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description='Run Decomposed QPU Benchmark (Strategic Problem Decomposition)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=int, default=25,
                       help='Number of units to test (default: 25)')
    parser.add_argument('--token', type=str, default=None,
                       help='D-Wave API token (or set DWAVE_API_TOKEN env var)')
    parser.add_argument('--output-dir', type=str, default='Benchmarks/DECOMPOSED',
                       help='Output directory for results')
    parser.add_argument('--num-reads', type=int, default=1000,
                       help='Number of QPU reads (default: 1000)')
    parser.add_argument('--annealing-time', type=int, default=20,
                       help='Annealing time in microseconds (default: 20)')
    
    args = parser.parse_args()
    
    # Get D-Wave token
    dwave_token = args.token or os.getenv('DWAVE_API_TOKEN', DWAVE_TOKEN_PLACEHOLDER)
    
    if dwave_token == DWAVE_TOKEN_PLACEHOLDER:
        print("\n⚠️  WARNING: No D-Wave token configured!")
        print("   Set DWAVE_API_TOKEN environment variable or use --token")
        print("   Running classical-only benchmarks.\n")
        dwave_token = None
    
    # Create output directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("DECOMPOSED QPU BENCHMARK (Strategic Problem Decomposition)")
    print("="*80)
    print(f"Configuration: {args.config} units")
    print(f"D-Wave QPU: {'Enabled' if dwave_token else 'Disabled (classical only)'}")
    print(f"QPU Reads: {args.num_reads}")
    print(f"Annealing Time: {args.annealing_time} μs")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Run benchmark
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        results = run_single_benchmark(
            n_units=args.config,
            dwave_token=dwave_token,
            total_land=100.0,
            num_reads=args.num_reads,
            annealing_time=args.annealing_time
        )
        
        # Add metadata
        results['metadata'] = {
            'timestamp': timestamp,
            'config': args.config,
            'dwave_enabled': dwave_token is not None,
            'num_reads': args.num_reads,
            'annealing_time': args.annealing_time
        }
        
        # Save results
        output_file = os.path.join(
            output_dir,
            f'results_config_{args.config}_{timestamp}.json'
        )
        save_results(results, output_file)
        
        # Print summary
        print_summary(results)
        
        print(f"\n✅ Benchmark complete!")
        print(f"   Results: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
