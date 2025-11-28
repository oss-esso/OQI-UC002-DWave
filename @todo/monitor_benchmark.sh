#!/bin/bash
# Monitor the comprehensive benchmark progress

LOG_FILE="/Users/edoardospigarolo/Documents/OQI-UC002-DWave/@todo/comprehensive_25farms.log"

echo "Monitoring benchmark progress..."
echo "================================"
echo ""

while true; do
    clear
    echo "BENCHMARK PROGRESS - 25 FARMS"
    echo "=============================="
    echo "Time: $(date '+%H:%M:%S')"
    echo ""
    
    # Show current experiment
    echo "Current Status:"
    tail -15 "$LOG_FILE" | grep -E "\[[0-9]+/[0-9]+\]|Building|Solving|Running|Embed|Result:" | tail -10
    echo ""
    
    # Count completed experiments
    TOTAL=$(grep -c "n_farms=25" "$LOG_FILE" || echo "0")
    COMPLETED=$(grep -c "Result: Embed=" "$LOG_FILE" || echo "0")
    echo "Progress: $COMPLETED / $TOTAL experiments completed"
    echo ""
    
    # Check if done
    if grep -q "BENCHMARK COMPLETE" "$LOG_FILE" 2>/dev/null; then
        echo "✓ BENCHMARK COMPLETED!"
        echo ""
        echo "Results:"
        tail -20 "$LOG_FILE"
        break
    fi
    
    # Check if process is still running
    if ! pgrep -f "comprehensive_embedding_and_solving_benchmark.py" > /dev/null; then
        echo "⚠ Process not running (may have finished or crashed)"
        break
    fi
    
    sleep 30
done
