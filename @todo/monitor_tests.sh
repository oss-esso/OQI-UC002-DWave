#!/bin/bash
# Monitor progress of running tests

echo "=============================================================================="
echo "TEST PROGRESS MONITOR"
echo "=============================================================================="
echo ""

echo "1. HYBRID TEST (20, 25, 50 farms with optimized Gurobi):"
echo "------------------------------------------------------------------------------"
if [ -f hybrid_rerun.log ]; then
    # Count completed tests
    completed=$(grep -c "✓ Gurobi:" hybrid_rerun.log 2>/dev/null || echo "0")
    total=6  # 2 runs × 3 sizes
    echo "  Progress: $completed/$total Gurobi runs completed"
    echo "  Last update:"
    tail -3 hybrid_rerun.log 2>/dev/null | sed 's/^/    /'
else
    echo "  Not started yet"
fi

echo ""
echo "2. COMPREHENSIVE SCALING TEST (25-1500 variables):"
echo "------------------------------------------------------------------------------"
if [ -f scaling_test.log ]; then
    # Count completed tests
    completed=$(grep -c "Running Gurobi..." scaling_test.log 2>/dev/null || echo "0")
    total=11  # 5 (6-family) + 6 (27-hybrid)
    echo "  Progress: $completed/$total problem sizes tested"
    echo "  Last update:"
    tail -3 scaling_test.log 2>/dev/null | sed 's/^/    /'
else
    echo "  Not started yet"
fi

echo ""
echo "=============================================================================="
echo "To monitor continuously: watch -n 5 ./monitor_tests.sh"
echo "To check full logs: tail -f hybrid_rerun.log  OR  tail -f scaling_test.log"
echo "=============================================================================="
