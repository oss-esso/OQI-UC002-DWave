# Quick Start Guide - Comprehensive Benchmark

## TL;DR - Run This Now

```bash
# Quick test (1 sample, ~5-10 minutes, no D-Wave needed)
python comprehensive_benchmark.py 1

# View results
ls Benchmarks/COMPREHENSIVE/
```

## What This Benchmark Does

Tests 6 different solver configurations:
1. **Farm + Gurobi** - Classical solver on large farms
2. **Farm + D-Wave CQM** - Quantum hybrid on large farms
3. **Patch + Gurobi** - Classical solver on patches  
4. **Patch + D-Wave CQM** - Quantum hybrid on patches
5. **Patch + Gurobi QUBO** - Classical struggling on QUBO
6. **Patch + D-Wave BQM** - Quantum solver on QUBO

## Why This Matters

Shows quantum advantage by comparing:
- ⚡ **Gurobi on CQM**: Fast (~0.05s) - good for constrained problems
- 🐌 **Gurobi on QUBO**: Slow (~300s) - struggles with QUBO
- 🚀 **D-Wave on QUBO**: Fast (~10s) - quantum advantage!

## Common Commands

```bash
# Without D-Wave (classical solvers only)
python comprehensive_benchmark.py 1              # 1 sample, ~10 min
python comprehensive_benchmark.py 5              # 5 samples, ~50 min
python comprehensive_benchmark.py --configs      # All configs, ~5 hours

# With D-Wave (faster, requires token)
export DWAVE_API_TOKEN="your-token"              # Set token first
python comprehensive_benchmark.py 5 --dwave      # 5 samples, ~5 min
python comprehensive_benchmark.py --configs --dwave  # All configs, ~30 min
```

## Understanding Results

Results saved to: `Benchmarks/COMPREHENSIVE/comprehensive_benchmark_*.json`

Key metrics to look at:
- **solve_time**: Total time to solve
- **objective_value**: Quality of solution
- **success**: Did solver find solution?
- **status**: Optimal, Time limit, Error, etc.

### Example Output

```json
{
  "solvers": {
    "gurobi": {
      "solve_time": 0.05,        // 50 milliseconds - FAST!
      "status": "Optimal",
      "success": true
    },
    "gurobi_qubo": {
      "solve_time": 300.45,      // 300 seconds - SLOW!
      "status": "Time limit reached",
      "success": true
    },
    "dwave_bqm": {
      "solve_time": 10.3,        // 10 seconds - QUANTUM ADVANTAGE!
      "status": "Optimal",
      "success": true
    }
  }
}
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'dimod'"
```bash
pip install dimod dwave-system dwave-ocean-sdk
```

### "Gurobi not found"
Make sure Gurobi is installed with valid license:
```bash
python -c "import gurobipy; print('Gurobi OK')"
```

### "D-Wave token not found"
Set environment variable:
```bash
# Linux/Mac
export DWAVE_API_TOKEN="your-token-here"

# Windows PowerShell
$env:DWAVE_API_TOKEN="your-token-here"

# Or pass directly
python comprehensive_benchmark.py 5 --dwave --token "your-token"
```

## Interpreting Performance

| Scenario | Expected Time | What It Shows |
|----------|--------------|---------------|
| Gurobi (CQM) | ~0.05s | Classical excels at constrained problems |
| Gurobi QUBO (BQM) | ~300s | Classical struggles with QUBO |
| D-Wave CQM | ~10-30s | Quantum hybrid handles constraints well |
| D-Wave BQM | ~5-15s | Quantum advantage for QUBO! |

## Configuration Options

Edit `BENCHMARK_CONFIGS` in `comprehensive_benchmark.py`:
```python
BENCHMARK_CONFIGS = [
    5,    # Small test
    10,   # Medium test
    15,   # Large test
    20,   # Larger test
    25    # Largest test
]
```

## Next Steps After Running

1. Check results: `cat Benchmarks/COMPREHENSIVE/comprehensive_benchmark_*.json`
2. Run plotting: `python plot_comprehensive_results.py <result-file>`
3. Analyze speedups and quantum advantage
4. Include in technical report

## Tips

💡 Start with 1 sample to make sure everything works
💡 Use --configs for systematic testing
💡 Run with D-Wave to see full comparison
💡 Check the README in Benchmarks/COMPREHENSIVE/ for details
💡 Gurobi QUBO will hit 300s time limit - this is expected!

## Need Help?

1. Read: `Benchmarks/COMPREHENSIVE/README.md` (detailed docs)
2. Check: `COMPREHENSIVE_BENCHMARK_IMPLEMENTATION.md` (technical details)
3. Test: `python test_comprehensive_quick.py` (interactive test)
