#!/usr/bin/env python
"""
Quick test: Compare all three methods on rotation_micro_25
- direct_qpu: Monolithic (90→120 BQM vars → 651 physical qubits)
- clique_qpu: Monolithic with clique sampler (fails for n>20)
- clique_decomp: Farm-by-farm decomposition (5 × 18 vars each)
"""

import subprocess
import sys

print("="*80)
print("CLIQUE DECOMPOSITION TEST: rotation_micro_25")
print("="*80)
print()
print("Methods:")
print("  1. ground_truth    - Gurobi optimal solution")
print("  2. direct_qpu      - Monolithic QPU (90 vars → 651 qubits, 75s embedding)")
print("  3. clique_qpu      - Monolithic clique (120 BQM vars, will struggle)")
print("  4. clique_decomp   - Farm-by-farm (5 × 18 vars, zero embedding!)")
print()
print("="*80)
print()

cmd = [
    sys.executable,
    "qpu_benchmark.py",
    "--scenario", "rotation_micro_25",
    "--methods", "ground_truth,direct_qpu,clique_qpu,clique_decomp",
    "--reads", "100"
]

subprocess.run(cmd)
