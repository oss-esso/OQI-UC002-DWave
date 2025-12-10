#!/bin/bash
# Monitor QPU benchmark progress

echo "=== QPU BENCHMARK MONITOR ==="
echo "Started: $(date)"
echo

while true; do
    clear
    echo "=== Current Time: $(date) ==="
    echo
    
    # Check if process is running
    if ps aux | grep -E "qpu_benchmark.*rotation" | grep -v grep > /dev/null; then
        echo "✓ Benchmark RUNNING"
        ps aux | grep -E "qpu_benchmark.*rotation" | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "TIME:", $10}'
    else
        echo "✗ Benchmark STOPPED/COMPLETED"
    fi
    
    echo
    echo "=== Last 20 lines of log ==="
    tail -20 rotation_qpu_100reads.log 2>/dev/null || echo "Log file not found"
    
    echo
    echo "=== Log statistics ==="
    if [ -f rotation_qpu_100reads.log ]; then
        echo "  Total lines: $(wc -l < rotation_qpu_100reads.log)"
        echo "  File size: $(ls -lh rotation_qpu_100reads.log | awk '{print $5}')"
        echo "  Scenarios completed: $(grep -c "Scenario.*complete" rotation_qpu_100reads.log 2>/dev/null || echo 0)"
        echo "  QPU samples taken: $(grep -c "Sampling on QPU" rotation_qpu_100reads.log 2>/dev/null || echo 0)"
    fi
    
    echo
    echo "Press Ctrl+C to exit monitoring"
    sleep 10
done
