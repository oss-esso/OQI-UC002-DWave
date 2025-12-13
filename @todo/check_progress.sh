#!/bin/bash
# Quick progress checker for comprehensive test

LOG_FILE="comprehensive_final.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Test not started yet"
    exit 1
fi

echo "========================================================================"
echo "COMPREHENSIVE TEST PROGRESS"
echo "========================================================================"
echo ""

# Count completed Gurobi runs
gurobi_done=$(grep -c "✓ obj=" "$LOG_FILE" 2>/dev/null || echo "0")
total_tests=12  # 4 test points × 3 formulations

echo "Progress: $gurobi_done/$total_tests configurations completed"
echo ""

# Show current test
echo "Current activity:"
tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/  /'

echo ""
echo "========================================================================"
echo "To see full log: tail -f $LOG_FILE"
echo "To see violations: grep -A 20 'CONSTRAINT VIOLATION' $LOG_FILE"
echo "========================================================================"
