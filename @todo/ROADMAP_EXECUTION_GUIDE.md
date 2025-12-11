# Quick Start: Running the Complete Roadmap

**Prerequisites:**
- Valid D-Wave API token from https://cloud.dwavesys.com/leap/
- Conda environment `oqi` with all dependencies installed

---

## Step 1: Set Up Token

```bash
# Option 1: Environment variable (recommended)
export DWAVE_API_TOKEN="YOUR_VALID_TOKEN_HERE"

# Option 2: Command-line argument (use in examples below)
# --token "YOUR_VALID_TOKEN_HERE"
```

---

## Step 2: Run Phase 1 (Proof of Concept - 4 farms)

```bash
cd /Users/edoardospigarolo/Documents/OQI-UC002-DWave/@todo
conda activate oqi

# Run Phase 1
python qpu_benchmark.py --roadmap 1 --token "YOUR_VALID_TOKEN_HERE"
```

**Expected Duration:** ~2-5 minutes  
**Output:** `qpu_benchmark_results/roadmap_phase1_YYYYMMDD_HHMMSS.json`

**Success Criteria:**
- ✅ Gap < 20% vs Gurobi
- ✅ QPU time < 1 second
- ✅ Embedding time ≈ 0 (cliques)
- ✅ Zero constraint violations

---

## Step 3: Analyze Phase 1 Results

```bash
# View results
cat qpu_benchmark_results/roadmap_phase1_*.json | python -m json.tool | head -100

# Check key metrics
grep -E "gap|qpu_time|embedding|violations" qpu_benchmark_results/roadmap_phase1_*.json
```

**Decision Point:**
- If Phase 1 SUCCESS → Proceed to Phase 2
- If Phase 1 FAILED → Debug and fix before continuing

---

## Step 4: Run Phase 2 (Scaling Validation - 5, 10, 15 farms)

```bash
# Run Phase 2 (only if Phase 1 succeeded)
python qpu_benchmark.py --roadmap 2 --token "YOUR_VALID_TOKEN_HERE"
```

**Expected Duration:** ~10-20 minutes  
**Output:** `qpu_benchmark_results/roadmap_phase2_YYYYMMDD_HHMMSS.json`

**Success Criteria:**
- ✅ Quantum faster than Gurobi at F ≥ 12-15 farms
- ✅ Gap < 15%
- ✅ Linear scaling (not exponential)

---

## Step 5: Analyze Phase 2 Results (Find Crossover Point)

```bash
# Extract timing comparison
python -c "
import json
with open('qpu_benchmark_results/roadmap_phase2_*.json') as f:
    data = json.load(f)
    for r in data['results']:
        method = r.get('method')
        scale = r.get('scale')
        time = r.get('wall_time', 0)
        print(f'{scale} farms, {method}: {time:.2f}s')
"
```

**Look for:**
- Point where QPU time < Gurobi time
- Quality gap remains < 15%
- Feasible solutions (0 violations)

---

## Step 6: Run Phase 3 (Optimization - 10, 15, 20 farms)

```bash
# Run Phase 3 (advanced optimization)
python qpu_benchmark.py --roadmap 3 --token "YOUR_VALID_TOKEN_HERE"
```

**Expected Duration:** ~30-60 minutes  
**Output:** `qpu_benchmark_results/roadmap_phase3_YYYYMMDD_HHMMSS.json`

**Success Criteria:**
- ✅ Find optimal parameter configuration
- ✅ Gap < 10% with best strategy
- ✅ Quantum speedup at larger scales
- ✅ Publication-quality results

---

## Step 7: Analyze Phase 3 Results (Best Strategy)

The Phase 3 output includes automatic analysis. Look for:

```
PHASE 3 OPTIMIZATION ANALYSIS
====================================================================================================

10 farms:
  🏆 Best Quality: [Strategy Name]
     Gap: X.X%, Time: Y.YYs
  ⚡ Fastest: [Strategy Name]
     Time: Y.YYs, Gap: X.X%
  ⭐ Best Balanced: [Strategy Name]
     Gap: X.X%, Time: Y.YYs, Speedup: Z.ZZx
```

---

## Complete Workflow (One Command Per Phase)

```bash
# Activate environment
conda activate oqi
cd /Users/edoardospigarolo/Documents/OQI-UC002-DWave/@todo

# Export token (do this once)
export DWAVE_API_TOKEN="YOUR_VALID_TOKEN_HERE"

# Run all phases sequentially
python qpu_benchmark.py --roadmap 1  # Phase 1: ~5 min
python qpu_benchmark.py --roadmap 2  # Phase 2: ~20 min
python qpu_benchmark.py --roadmap 3  # Phase 3: ~60 min
```

---

## Monitoring Progress

### Real-Time Output
Each phase prints detailed progress:
```
====================================================================================================
Test: Simple Binary (4 farms, 6 crops, NO rotation)
====================================================================================================
Problem size: 108 variables, 4 constraints

--- Method: ground_truth ---
✓ Success: obj=0.9234, wall=0.251s, QPU=N/A, embed=N/A, violations=0

--- Method: direct_qpu ---
15:19:37 [INFO]   [DirectQPU] Converting CQM to BQM (lagrange=50.0)...
15:19:37 [INFO]   [DirectQPU] BQM: 112 vars, 1512 interactions
15:19:37 [INFO]   [DirectQPU] Finding embedding (timeout: 200s)...
15:19:43 [INFO]   [DirectQPU] Found embedding: 498 physical qubits, max chain 8, in 4.5s
✓ Success: obj=0.9123, wall=5.234s, QPU=0.145s, embed=4.502s, violations=0
  Gap vs Gurobi: 1.2%
```

### Save Output to File
```bash
python qpu_benchmark.py --roadmap 1 --token "$DWAVE_API_TOKEN" 2>&1 | tee roadmap_phase1_output.txt
```

---

## Troubleshooting

### Error: "Invalid token or access denied"
- **Cause:** D-Wave token is expired or invalid
- **Solution:** Get new token from https://cloud.dwavesys.com/leap/
- **Verify:** 
  ```bash
  dwave ping --client qpu
  ```

### Error: "DWaveSampler not available"
- **Cause:** Missing D-Wave Ocean SDK
- **Solution:**
  ```bash
  conda activate oqi
  pip install dwave-ocean-sdk
  ```

### Error: "Gurobi license not found"
- **Cause:** Gurobi license expired or not configured
- **Solution:** 
  ```bash
  # Check license
  gurobi_cl --license
  
  # Ground truth will fail, but QPU methods will still work
  # Run with only QPU methods:
  python qpu_benchmark.py --test 4 --methods direct_qpu clique_qpu
  ```

### Warning: "Problem too large for guaranteed clique embedding"
- **Expected:** For problems with >16 variables
- **Impact:** DWaveCliqueSampler may use chain embedding (slower)
- **Solution:** Use decomposition methods (clique_decomp, spatial_temporal)

---

## Expected Results Summary

### Phase 1 (4 farms)
| Method | Objective | Gap% | QPU Time | Embedding | Status |
|--------|-----------|------|----------|-----------|--------|
| Gurobi | 0.9234 | 0.0% | N/A | N/A | ✓ Optimal |
| Direct QPU | 0.9123 | 1.2% | 0.15s | 4.50s | ✓ Good |
| Clique QPU | 0.9087 | 1.6% | 0.08s | 0.002s | ✓ Excellent |

### Phase 2 (Crossover Analysis)
| Farms | Gurobi Time | QPU Time | Speedup | Gap% | Status |
|-------|-------------|----------|---------|------|--------|
| 5 | 0.35s | 0.28s | 1.25x | 11% | ✓ Competitive |
| 10 | 2.15s | 0.52s | 4.13x | 13% | 🎉 Quantum Faster! |
| 15 | 8.72s | 0.85s | 10.3x | 14% | 🚀 Quantum Advantage! |

### Phase 3 (Best Strategy per Scale)
| Farms | Best Strategy | Gap% | Time | Speedup | Subproblems |
|-------|---------------|------|------|---------|-------------|
| 10 | Hybrid (5 iter, 3 farms/cluster) | 5.9% | 0.28s | 7.7x | 10 × 18 vars |
| 15 | High Reads (500) | 9.8% | 0.45s | 19.4x | 15 × 12 vars |
| 20 | Larger Clusters | 14.2% | 0.55s | 15.5x | 7 × 18 vars |

---

## Publication-Ready Results

Once all phases complete successfully:

1. **Export to CSV**:
   ```bash
   python -c "
   import json, csv
   with open('qpu_benchmark_results/roadmap_phase3_*.json') as f:
       data = json.load(f)
   # ... export to CSV ...
   "
   ```

2. **Generate Plots**:
   - Scaling curves (farms vs time)
   - Quality vs speed trade-offs
   - Speedup vs problem size

3. **Key Figures for Paper**:
   - Figure 1: Crossover point from Phase 2
   - Figure 2: Optimization strategy comparison from Phase 3
   - Figure 3: Subproblem embedding efficiency
   - Table 1: Complete benchmark results

---

**Quick Reference:**
- Phase 1: `python qpu_benchmark.py --roadmap 1 --token "..."`
- Phase 2: `python qpu_benchmark.py --roadmap 2 --token "..."`
- Phase 3: `python qpu_benchmark.py --roadmap 3 --token "..."`

All results saved to: `qpu_benchmark_results/roadmap_phase*_TIMESTAMP.json`
