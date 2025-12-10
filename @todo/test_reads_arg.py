#!/usr/bin/env python3
"""
Test the --reads argument parsing for qpu_benchmark.py
"""

import argparse

parser = argparse.ArgumentParser(description='QPU Benchmark (Pure Quantum - No Hybrid)')
parser.add_argument('--all-small', action='store_true',
                    help='Test all small synthetic scenarios (micro_6 through medium_160)')
parser.add_argument('--reads', type=int, nargs='+',
                    help='Number of reads for QPU sampling (e.g., 100 250 500). '
                         'If multiple values, runs benchmark for each. Default: 500 for small scenarios, 1000 for others')
parser.add_argument('--methods', nargs='+',
                    help='Specific methods')

args = parser.parse_args()

print("=" * 80)
print("Testing --reads Argument Parsing")
print("=" * 80)

print(f"\nAll small: {args.all_small}")
print(f"Reads: {args.reads}")
print(f"Methods: {args.methods}")

if args.reads:
    print(f"\n✅ Reads configurations parsed: {args.reads}")
    print(f"   Number of configs: {len(args.reads)}")
    for i, num_reads in enumerate(args.reads, 1):
        print(f"   Config {i}: {num_reads} reads")
else:
    print(f"\n✅ No reads specified (will use defaults)")

print("\nExpected behavior:")
if args.reads:
    print(f"  - Will run benchmark {len(args.reads)} times per scenario/scale")
    print(f"  - Method names will have suffix: _r{args.reads[0]}, _r{args.reads[1]}, etc.")
else:
    print(f"  - Will run benchmark once per scenario/scale")
    print(f"  - 500 reads for small scenarios, 1000 for others")

print("\n" + "=" * 80)
