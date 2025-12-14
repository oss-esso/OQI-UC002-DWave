# Significant Scenarios Benchmark - Ready to Run! ✅

## Status: ALL SYSTEMS GO

All dependencies verified and benchmark is ready to execute.

## Quick Start

```bash
# 1. Activate environment
conda activate oqi

# 2. Navigate to directory
cd /Users/edoardospigarolo/Documents/OQI-UC002-DWave/@todo

# 3. Run benchmark (recommended: with logging)
python significant_scenarios_benchmark.py 2>&1 | tee benchmark_run_$(date +%Y%m%d_%H%M%S).log
```

## What Will Be Tested

### 6 Scenarios Spanning Different Problem Sizes

| # | Scenario | Size | Variables | Method | Expected Speedup |
|---|----------|------|-----------|--------|------------------|
| 1 | rotation_micro_25 | 5 farms × 6 foods | 90 vars | clique_decomp | 11.5× |
| 2 | rotation_small_50 | 10 farms × 6 foods | 180 vars | clique_decomp | 6.2× |
| 3 | rotation_medium_100 | 20 farms × 6 foods | 360 vars | clique_decomp | 5.2× |
| 4 | rotation_large_25farms_27foods | 25 farms × 27 foods | 2025 vars | hierarchical | 5.0× |
| 5 | rotation_xlarge_50farms_27foods | 50 farms × 27 foods | 4050 vars | hierarchical | 4.5× |
| 6 | rotation_xxlarge_100farms_27foods | 100 farms × 27 foods | 8100 vars | hierarchical | 2.5× |

### Metrics Tracked

**Performance:**
- ✅ Objective value (both Gurobi and QPU)
- ✅ Runtime (wall clock time)
- ✅ Speedup ratio (Gurobi time / QPU time)

**Quality:**
- ✅ Gap percentage: `(gurobi_obj - qpu_obj) / gurobi_obj × 100%`
  - Positive = QPU worse than Gurobi
  - Negative = QPU better than Gurobi (!)
- ✅ MIP gap (for Gurobi)

**Constraint Violations:**
- ✅ Rotation violations (same crop in consecutive periods)
- ✅ Diversity violations (no crops grown in a period)
- ✅ Area violations (allocation errors)
- ✅ Total violations

## Configuration

### Gurobi Settings (Optimized)
- **Timeout:** 300s (5 minutes)
- **MIP Gap:** 10% (find good solutions quickly)
- **MIP Focus:** 1 (prioritize feasible solutions)
- **Improve Start Time:** 30s (stop if no improvement)

These settings ensure Gurobi finds good solutions quickly rather than wasting time trying to close tiny optimality gaps.

### QPU Settings
- **Num Reads:** 100 (samples per QPU call)
- **Farms per Cluster:** 5 (for hierarchical decomposition)
- **Iterations:** 3 (boundary coordination rounds)

## Estimated Resource Usage

### Time
- **Small scenarios (5-20 farms):** ~5-10 minutes each
- **Large scenarios (25-100 farms):** ~10-20 minutes each
- **Total benchmark:** ~60-90 minutes

### D-Wave QPU Credits
- Approximately 600 QPU calls total
- ~100 calls per scenario × 6 scenarios
- Cost depends on your D-Wave plan

## Output Files

Results will be saved to `significant_scenarios_results/`:

```
significant_scenarios_results/
├── benchmark_results_YYYYMMDD_HHMMSS.json  # Detailed JSON format
└── benchmark_results_YYYYMMDD_HHMMSS.csv   # Tabular CSV format
```

### Example Output

```
================================================================================
FINAL RESULTS SUMMARY
================================================================================

Scenario                       Gurobi Obj      QPU Obj      Gap %    Speedup
--------------------------------------------------------------------------------
rotation_micro_25                    4.08         3.75      +8.2%     11.50×
rotation_small_50                    7.17         6.49      +9.6%      6.15×
rotation_medium_100                 14.89        12.98     +12.9%      5.26×
rotation_large_25farms_27foods      24.56        23.30      +5.1%      5.01×
rotation_xlarge_50farms_27foods     45.12        43.87      +2.8%      4.48×
rotation_xxlarge_100farms_27foods   87.34        85.60      +2.0%      2.19×
--------------------------------------------------------------------------------
Average Gap: +6.8%
Average Speedup: 5.7×
```

## What to Look For

### Good Results ✅
- **Gap < 20%:** QPU solution quality acceptable
- **Speedup > 3×:** Significant time savings
- **Violations = 0:** Feasible solution found
- **Consistent performance:** Similar speedup across scenarios

### Areas of Concern ⚠️
- **Gap > 30%:** QPU solution much worse than Gurobi
- **Speedup < 1×:** QPU slower than Gurobi (shouldn't happen!)
- **Violations > 0:** Constraint violations detected (investigate)
- **High variance:** Inconsistent performance across scenarios

## After Running

### 1. Analyze Results
```bash
# View CSV in Excel/Numbers or use pandas
conda activate oqi
python -c "import pandas as pd; df = pd.read_csv('significant_scenarios_results/benchmark_results_*.csv'); print(df)"
```

### 2. Generate Visualizations
Create plots showing:
- Speedup vs problem size
- Gap vs problem size
- Objective comparison (Gurobi vs QPU)
- Constraint violations

### 3. Document Findings
Use results for:
- Technical paper: "Quantum vs Classical for Crop Rotation"
- Determining which method to use for which problem size
- Identifying optimization opportunities

## Troubleshooting

If issues occur:

1. **Check log file:** `benchmark_run_YYYYMMDD_HHMMSS.log`
2. **Verify environment:** Run `python preflight_check.py`
3. **Check D-Wave status:** https://cloud.dwavesys.com/
4. **Test individual scenario:** Modify script to run just one scenario

## Files Created

All files are in `@todo/` directory:

1. **significant_scenarios_benchmark.py** - Main benchmark script
2. **BENCHMARK_QUICKSTART.md** - Quick start guide
3. **preflight_check.py** - Dependency verification
4. **data_loader_utils.py** - Data loading utilities
5. **clique_wrapper.py** - Clique decomposition wrapper
6. **THIS_FILE.md** - This ready-to-run summary

## Ready to Go!

Everything is set up and verified. When you're ready:

```bash
conda activate oqi
cd /Users/edoardospigarolo/Documents/OQI-UC002-DWave/@todo
python significant_scenarios_benchmark.py 2>&1 | tee benchmark_run_$(date +%Y%m%d_%H%M%S).log
```

⚠️ **Warning:** This will consume D-Wave QPU credits. Ensure you have sufficient credits before running.

---

**Questions?** Check:
- `BENCHMARK_QUICKSTART.md` - Detailed guide
- `SIGNIFICANT_SCENARIOS_ANALYSIS.md` - Previous analysis
- `complete_scenarios_inventory.json` - All available scenarios
