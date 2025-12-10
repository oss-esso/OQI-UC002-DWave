#!/bin/bash
# Quick test: Clique decomposition vs monolithic approaches

cd /Users/edoardospigarolo/Documents/OQI-UC002-DWave/@todo

echo "=========================================================================="
echo "Testing Clique Decomposition (Mohseni et al. style)"
echo "=========================================================================="
echo ""
echo "Scenario: rotation_micro_25 (5 farms × 6 families × 3 periods = 90 vars)"
echo ""
echo "Methods:"
echo "  - ground_truth:   Gurobi optimal (120s)"
echo "  - direct_qpu:     Monolithic QPU (90→120 BQM → 651 qubits, 75s embed)"
echo "  - clique_qpu:     Monolithic clique (120 vars, too large, fails)"
echo "  - clique_decomp:  Farm-by-farm (5 × 18 vars, FITS CLIQUES! <1s)"
echo ""
echo "Expected: clique_decomp should be 100-150× faster than direct_qpu!"
echo "=========================================================================="
echo ""

python qpu_benchmark.py \
  --scenario rotation_micro_25 \
  --methods ground_truth,direct_qpu,clique_qpu,clique_decomp \
  --reads 100
